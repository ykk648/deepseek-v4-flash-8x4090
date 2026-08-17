#!/usr/bin/env bash
set -euo pipefail

echo '--- NVIDIA ---'
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader

count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$count" -ne 8 ]]; then
  echo "Expected 8 GPUs, found $count" >&2
  exit 1
fi

echo '--- RAM ---'
free -h
echo '--- Disk ---'
df -h "${MODEL_DIR:-/data/models/DeepSeek-V4-Flash-0731}"
echo '--- CUDA ---'
nvcc --version | tail -n 1
