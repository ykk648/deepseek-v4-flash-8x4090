#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
source .env 2>/dev/null || {
  echo 'Copy .env.example to .env and set MODEL_DIR first.' >&2
  exit 1
}

: "${MODEL_DIR:?MODEL_DIR is required}"
: "${PORT:=8000}"
: "${HOST:=127.0.0.1}"
: "${SERVED_MODEL_NAME:=DeepSeek-V4-Flash-0731}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${TENSOR_PARALLEL_SIZE:=8}"
: "${GPU_MEMORY_UTILIZATION:=0.895}"
: "${CPU_OFFLOAD_GB:=0}"
: "${MAX_MODEL_LEN:=1024}"
: "${MAX_NUM_SEQS:=1}"
: "${MAX_NUM_BATCHED_TOKENS:=512}"
: "${ENABLE_DSPARK:=0}"
: "${ENABLE_EXPERT_PARALLEL:=0}"
: "${ENFORCE_EAGER:=1}"
: "${NUM_SPECULATIVE_TOKENS:=7}"

[[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || {
  echo "Model index not found: $MODEL_DIR" >&2
  exit 1
}
[[ "$TENSOR_PARALLEL_SIZE" == "8" ]] || {
  echo 'This service file is intended for all eight GPUs; keep TENSOR_PARALLEL_SIZE=8.' >&2
  exit 1
}

export CUDA_VISIBLE_DEVICES
export FLASHINFER_DISABLE_VERSION_CHECK=1
if [[ -n "${VLLM_USE_V2_MODEL_RUNNER:-}" ]]; then
  export VLLM_USE_V2_MODEL_RUNNER
fi
# This is a single-node TP job. Disable any host-installed NCCL network
# plugin; the crashing plugin is not needed for intra-node P2P/SHM collectives.
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.3}
export PATH="$ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
if [[ -n "${PYNV_VIDEO_CODEC_LIB:-}" ]]; then
  export LD_LIBRARY_PATH="$PYNV_VIDEO_CODEC_LIB:${LD_LIBRARY_PATH:-}"
fi

args=(
  serve "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --kv-cache-dtype fp8_ds_mla
  --block-size 256
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --cpu-offload-gb "$CPU_OFFLOAD_GB"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4
  --reasoning-parser deepseek_v4
  --enable-auto-tool-choice
  --tool-call-parser deepseek_v4
  --trust-remote-code
  --host "$HOST"
  --port "$PORT"
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  args+=(--enforce-eager)
fi

if [[ "$ENABLE_DSPARK" == "1" ]]; then
  args+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS,\"draft_sample_method\":\"greedy\"}")
fi

if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
  args+=(--enable-expert-parallel)
fi

echo "Starting $SERVED_MODEL_NAME on GPUs $CUDA_VISIBLE_DEVICES"
echo "Model=$MODEL_DIR TP=$TENSOR_PARALLEL_SIZE CPU_OFFLOAD_GB=$CPU_OFFLOAD_GB MAX_MODEL_LEN=$MAX_MODEL_LEN DSpark=$ENABLE_DSPARK EP=$ENABLE_EXPERT_PARALLEL"
exec .venv/bin/vllm "${args[@]}"
