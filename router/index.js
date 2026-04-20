import express from 'express';
import fetch from 'node-fetch';
import { createProxyMiddleware } from 'http-proxy-middleware';

const app = express();
const IDLE_TIMEOUT_MS = parseInt(process.env.IDLE_TIMEOUT_SECONDS ?? '300') * 1000;

const MODEL_MAP = {
  'qwen2.5-0.5b': { host: 'vllm-qwen05',  port: 8001 },
  'qwen3.5-0.8b': { host: 'vllm-qwen08',  port: 8002 },
  'qwen2.5':      { host: 'vllm-qwen',     port: 8003 },
};

const state = Object.fromEntries(Object.keys(MODEL_MAP).map(k => [k, 'loading']));
const idleTimers = {};
const wakeQueue = {};

// ── vLLM HTTP helpers ─────────────────────────────────────────────────────────

async function vllmPost(modelName, path) {
  const { host, port } = MODEL_MAP[modelName];
  const res = await fetch(`http://${host}:${port}/${path}`, {
    method: 'POST',
    signal: AbortSignal.timeout(15_000),
  });
  if (!res.ok) throw new Error(`POST /${path} failed for ${modelName}: HTTP ${res.status}`);
}

// Single probe of /v1/models — returns true if model is loaded
async function probeLoaded(modelName) {
  const { host, port } = MODEL_MAP[modelName];
  try {
    const r = await fetch(`http://${host}:${port}/v1/models`, {
      signal: AbortSignal.timeout(3000),
    });
    return r.ok;
  } catch {
    return false;
  }
}

// Poll /is_sleeping after wake_up — returns true when awake
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

// ── Wake / sleep ──────────────────────────────────────────────────────────────

async function wakeModel(modelName) {
  if (state[modelName] === 'awake') return;
  if (state[modelName] === 'loading') throw new Error(`${modelName} is still loading`);
  if (wakeQueue[modelName]) return wakeQueue[modelName];

  wakeQueue[modelName] = (async () => {
    console.log(`[router] waking ${modelName}…`);
    state[modelName] = 'waking';
    try {
      await vllmPost(modelName, 'wake_up');
      await waitUntilAwake(modelName, 30_000);
      state[modelName] = 'awake';
      console.log(`[router] ${modelName} awake`);
    } catch (e) {
      state[modelName] = 'asleep';
      throw e;
    } finally {
      delete wakeQueue[modelName];
    }
  })();

  return wakeQueue[modelName];
}

async function sleepModel(modelName) {
  if (state[modelName] !== 'awake') return;
  console.log(`[router] sleeping ${modelName}…`);
  state[modelName] = 'sleeping';
  try {
    await vllmPost(modelName, 'sleep');
    state[modelName] = 'asleep';
    console.log(`[router] ${modelName} asleep`);
  } catch (e) {
    state[modelName] = 'awake';
    console.error(`[router] sleep failed for ${modelName}:`, e.message);
  }
}

function resetIdleTimer(modelName) {
  clearTimeout(idleTimers[modelName]);
  idleTimers[modelName] = setTimeout(() => sleepModel(modelName), IDLE_TIMEOUT_MS);
}

// ── Boot: poll each model independently, never reset progress on retry ────────
//
// Each model transitions: loading → asleep (once /v1/models returns 200 and
// we've called /sleep). The boot loop runs until all models are out of
// 'loading'. The router starts accepting requests immediately — models that
// finish early are available while others are still loading.

async function sleepOneModel(name) {
  try {
    await vllmPost(name, 'sleep');
    state[name] = 'asleep';
    console.log(`[router] ${name} loaded and asleep`);
  } catch (e) {
    // Sleep failed — leave awake, idle timer will retry
    state[name] = 'awake';
    resetIdleTimer(name);
    console.warn(`[router] ${name} loaded but sleep failed:`, e.message);
  }
}

async function boot() {
  console.log('[router] boot — polling each vLLM model until loaded (no timeout)…');

  // Start the HTTP server immediately so Open WebUI can connect
  // Models in 'loading' state return 503 until ready
  app.listen(8000, () => console.log('[router] listening on :8000'));

  // Poll all models in parallel, independently, forever until each is loaded
  await Promise.all(
    Object.keys(MODEL_MAP).map(async name => {
      let logged = false;
      while (true) {
        if (!logged) {
          console.log(`[router] waiting for ${name} to finish loading…`);
          logged = true;
        }
        const ready = await probeLoaded(name);
        if (ready) {
          console.log(`[router] ${name} finished loading — sleeping…`);
          await sleepOneModel(name);
          return;
        }
        await new Promise(r => setTimeout(r, 5000));
      }
    })
  );

  console.log('[router] all models loaded and asleep — fully ready');
}

// ── Routes ────────────────────────────────────────────────────────────────────

app.get('/v1/models', (_req, res) => {
  // Only advertise models that have finished loading
  const ready = Object.entries(state)
    .filter(([, s]) => s !== 'loading')
    .map(([id]) => ({ id, object: 'model', owned_by: 'vllm', created: 0 }));

  res.json({ object: 'list', data: ready });
});

app.get('/status', (_req, res) => {
  res.json(
    Object.fromEntries(
      Object.entries(state).map(([k, v]) => [k, { state: v, ...MODEL_MAP[k] }])
    )
  );
});

app.use('/v1', async (req, res, next) => {
  let rawBody = '';
  await new Promise(resolve => { req.on('data', c => rawBody += c); req.on('end', resolve); });
  req.rawBody = rawBody;

  let modelName;
  try { modelName = rawBody ? JSON.parse(rawBody).model : req.query.model; } catch {}

  const entry = MODEL_MAP[modelName];
  if (!entry) {
    return res.status(400).json({
      error: `Unknown model "${modelName}". Available: ${Object.keys(MODEL_MAP).join(', ')}`,
    });
  }

  if (state[modelName] === 'loading') {
    return res.status(503).json({
      error: `${modelName} is still loading, try again shortly`,
    });
  }

  try {
    await wakeModel(modelName);
  } catch (e) {
    return res.status(503).json({ error: `Failed to wake ${modelName}: ${e.message}` });
  }

  resetIdleTimer(modelName);

  const { host, port } = entry;
  createProxyMiddleware({
    target: `http://${host}:${port}`,
    changeOrigin: true,
    on: {
      proxyReq(proxyReq) {
        if (req.rawBody) {
          proxyReq.setHeader('content-length', Buffer.byteLength(req.rawBody));
          proxyReq.write(req.rawBody);
          proxyReq.end();
        }
      },
    },
  })(req, res, next);
});

// ── Start ─────────────────────────────────────────────────────────────────────
// boot() never throws — errors per model are logged and retried internally
boot().catch(e => console.error('[router] unexpected boot error:', e));