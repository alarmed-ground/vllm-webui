import express from 'express';
import fetch from 'node-fetch';

const app = express();
const IDLE_TIMEOUT_MS = parseInt(process.env.IDLE_TIMEOUT_SECONDS ?? '120') * 1000;

const MODEL_MAP = {
  'qwen2.5-0.5b': { host: 'vllm-qwen05', port: 8001 },
  'qwen3.5-0.8b': { host: 'vllm-qwen08', port: 8002 },
  'qwen2.5':      { host: 'vllm-qwen',   port: 8003 },
};

// 'loading' → 'asleep' → 'waking' → 'awake' → 'sleeping' → 'asleep'
const state = Object.fromEntries(Object.keys(MODEL_MAP).map(k => [k, 'loading']));

let activeModel   = null;   // model currently on the GPU
let switchMutex   = null;   // serializes all wake/sleep transitions
const idleTimers  = {};

// ── vLLM helpers ──────────────────────────────────────────────────────────────

async function vllmPost(modelName, path) {
  const { host, port } = MODEL_MAP[modelName];
  const res = await fetch(`http://${host}:${port}/${path}`, {
    method: 'POST',
    signal: AbortSignal.timeout(15_000),
  });
  if (!res.ok) throw new Error(`POST /${path} failed for ${modelName}: HTTP ${res.status}`);
}

async function probeLoaded(modelName) {
  const { host, port } = MODEL_MAP[modelName];
  try {
    const r = await fetch(`http://${host}:${port}/v1/models`, {
      signal: AbortSignal.timeout(3000),
    });
    return r.ok;
  } catch { return false; }
}

async function waitUntilAwake(modelName, timeoutMs = 30_000) {
  const { host, port } = MODEL_MAP[modelName];
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`http://${host}:${port}/is_sleeping`, {
        signal: AbortSignal.timeout(3000),
      });
      if (r.ok) {
        const body = await r.json();
        if (body.is_sleeping === false) return;
      }
    } catch {}
    await new Promise(r => setTimeout(r, 1000));
  }
  throw new Error(`${modelName} did not wake within ${timeoutMs / 1000}s`);
}

// ── Idle timer ────────────────────────────────────────────────────────────────

function clearIdleTimer(modelName) {
  clearTimeout(idleTimers[modelName]);
  delete idleTimers[modelName];
}

function resetIdleTimer(modelName) {
  clearIdleTimer(modelName);
  idleTimers[modelName] = setTimeout(async () => {
    console.log(`[router] ${modelName} idle timeout — sleeping…`);
    await switchTo(null);
  }, IDLE_TIMEOUT_MS);
}

// ── Core: serialized GPU switch ───────────────────────────────────────────────
//
// switchTo(targetModel) ensures:
//   1. Any currently active model is put to sleep first
//   2. The target model (if not null) is woken and confirmed awake
//   3. Only one switch runs at a time — concurrent callers queue behind the mutex

async function switchTo(targetModel) {
  // Chain onto the existing mutex so transitions never overlap
  switchMutex = (switchMutex ?? Promise.resolve()).then(async () => {
    // Nothing to do if already in the right state
    if (targetModel === activeModel) return;

    // Sleep the current active model
    if (activeModel !== null && state[activeModel] === 'awake') {
      const prev = activeModel;
      console.log(`[router] sleeping ${prev} to free GPU…`);
      state[prev] = 'sleeping';
      clearIdleTimer(prev);
      try {
        await vllmPost(prev, 'sleep');
        state[prev] = 'asleep';
        console.log(`[router] ${prev} asleep`);
      } catch (e) {
        console.error(`[router] sleep failed for ${prev}:`, e.message);
        state[prev] = 'awake'; // leave state consistent
      }
      activeModel = null;
    }

    // Wake the target model
    if (targetModel !== null) {
      console.log(`[router] waking ${targetModel}…`);
      state[targetModel] = 'waking';
      try {
        await vllmPost(targetModel, 'wake_up');
        await waitUntilAwake(targetModel);
        state[targetModel] = 'awake';
        activeModel = targetModel;
        console.log(`[router] ${targetModel} awake and active`);
      } catch (e) {
        state[targetModel] = 'asleep';
        throw e;
      }
    }
  });

  return switchMutex;
}

// ── Boot ──────────────────────────────────────────────────────────────────────

async function sleepOneModel(name) {
  try {
    await vllmPost(name, 'sleep');
    state[name] = 'asleep';
    console.log(`[router] ${name} loaded and asleep`);
  } catch (e) {
    // If sleep fails at boot just mark it awake — idle timer will sleep it
    state[name] = 'awake';
    activeModel  = name;
    resetIdleTimer(name);
    console.warn(`[router] ${name} loaded but sleep failed:`, e.message);
  }
}

async function boot() {
  console.log('[router] boot — polling each vLLM model until loaded…');
  app.listen(8000, () => console.log('[router] listening on :8000'));

  await Promise.all(
    Object.keys(MODEL_MAP).map(async name => {
      console.log(`[router] waiting for ${name} to finish loading…`);
      while (true) {
        if (await probeLoaded(name)) {
          console.log(`[router] ${name} finished loading — sleeping…`);
          await sleepOneModel(name);
          return;
        }
        await new Promise(r => setTimeout(r, 5000));
      }
    })
  );

  console.log('[router] all models loaded and asleep — ready');
}

// ── Routes ────────────────────────────────────────────────────────────────────

app.get('/v1/models', (_req, res) => {
  const data = Object.entries(state)
    .filter(([, s]) => s !== 'loading')
    .map(([id]) => ({ id, object: 'model', owned_by: 'vllm', created: 0 }));
  res.json({ object: 'list', data });
});

app.get('/status', (_req, res) => {
  res.json(Object.fromEntries(
    Object.entries(state).map(([k, v]) => [
      k, { state: v, active: activeModel === k, ...MODEL_MAP[k] }
    ])
  ));
});

// ── Main proxy ────────────────────────────────────────────────────────────────

app.use('/v1', async (req, res) => {
  // Buffer body
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const rawBody = Buffer.concat(chunks);

  // Extract model name
  let modelName;
  try { modelName = JSON.parse(rawBody.toString()).model; } catch {}
  if (!modelName) modelName = req.query.model;

  const entry = MODEL_MAP[modelName];
  if (!entry) {
    return res.status(400).json({
      error: `Unknown model "${modelName}". Available: ${Object.keys(MODEL_MAP).join(', ')}`,
    });
  }

  if (state[modelName] === 'loading') {
    return res.status(503).json({ error: `${modelName} is still loading, try again shortly` });
  }

  // Switch GPU to the requested model (sleeps any other active model first)
  try {
    await switchTo(modelName);
  } catch (e) {
    return res.status(503).json({ error: `Failed to switch to ${modelName}: ${e.message}` });
  }

  // Reset idle timer — 2 min of silence will sleep this model
  resetIdleTimer(modelName);

  // Forward request
  const { host, port } = entry;
  const url = `http://${host}:${port}${req.originalUrl}`;
  const headers = { ...req.headers, host: `${host}:${port}` };

  try {
    const upstream = await fetch(url, {
      method:  req.method,
      headers,
      body:    rawBody.length ? rawBody : undefined,
    });

    res.status(upstream.status);
    upstream.headers.forEach((v, k) => res.setHeader(k, v));
    upstream.body.pipe(res);
  } catch (e) {
    console.error(`[router] proxy error for ${modelName}:`, e.message);
    if (!res.headersSent) res.status(502).json({ error: `Upstream error: ${e.message}` });
  }
});

// ── Start ─────────────────────────────────────────────────────────────────────
boot().catch(e => console.error('[router] unexpected boot error:', e));