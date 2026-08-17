#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
source .env 2>/dev/null || true
MODEL_ID=${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}
MODEL_DIR=${MODEL_DIR:-/data/models/DeepSeek-V4-Flash-0731}
MODEL_MAX_WORKERS=${MODEL_MAX_WORKERS:-8}

[[ -x .venv/bin/modelscope ]] || {
  echo 'Run ./setup_env.sh first.' >&2
  exit 1
}

mkdir -p "$MODEL_DIR"
echo "Downloading $MODEL_ID to $MODEL_DIR"
echo 'The model weights are about 155.43 GiB; keep at least 180 GiB free for the download and temporary files.'
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  .venv/bin/modelscope download \
  --model "$MODEL_ID" \
  --local_dir "$MODEL_DIR" \
  --max-workers "$MODEL_MAX_WORKERS"

test -f "$MODEL_DIR/model.safetensors.index.json" || {
  echo "Model download appears incomplete: index file is missing." >&2
  exit 1
}
echo "Model is ready: $MODEL_DIR"
