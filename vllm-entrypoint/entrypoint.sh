#!/bin/bash
set -e

MODEL="${MODEL_NAME}"
echo "[entrypoint] downloading ${MODEL}…"

hf download "${MODEL}" \
  --quiet

echo "[entrypoint] download complete, starting vLLM…"

# Hand off to the vLLM OpenAI server — all extra args passed through
exec vllm serve "$@"
