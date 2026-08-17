#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT:-8000}}
MODEL=${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}

curl --fail --silent --show-error "$BASE_URL/health"
printf '\n'
curl --fail --silent --show-error "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(printf '%s' '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with exactly: SM89-OK"}],"temperature":0,"max_tokens":16,"stream":false}')"
printf '\n'
