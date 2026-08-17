#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
VENV=${VENV:-"$ROOT/.venv"}
RELEASE_TAG=${RELEASE_TAG:-v0.23.1rc1.dev904-g8e321cc4f-cu130-sm89}
VLLM_WHEEL=${VLLM_WHEEL:-vllm-0.23.1rc1.dev904%2Bg8e321cc4f.cu130-cp312-cp312-linux_x86_64.whl}
FLASHINFER_WHEEL=${FLASHINFER_WHEEL:-flashinfer_python-0.6.14%2Bsm89.1-py3-none-any.whl}
RELEASE_URL=https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/$RELEASE_TAG

command -v uv >/dev/null || { echo 'uv is required: https://docs.astral.sh/uv/'; exit 1; }
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}
export UV_CACHE_DIR=${UV_CACHE_DIR:-"$ROOT/.uv-cache"}
if [[ -n "${GITHUB_PROXY:-}" ]]; then
  export HTTP_PROXY=${HTTP_PROXY:-$GITHUB_PROXY}
  export HTTPS_PROXY=${HTTPS_PROXY:-$GITHUB_PROXY}
  export http_proxy=${http_proxy:-$GITHUB_PROXY}
  export https_proxy=${https_proxy:-$GITHUB_PROXY}
fi
uv venv --python "$PYTHON_VERSION" --seed "$VENV"

uv pip install \
  "$RELEASE_URL/$FLASHINFER_WHEEL" \
  "$RELEASE_URL/$VLLM_WHEEL" \
  'flashinfer-cubin==0.6.13' \
  --torch-backend=cu130 \
  --python "$VENV/bin/python"

uv pip install 'modelscope>=1.28' --python "$VENV/bin/python" \
  -i "${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

"$VENV/bin/python" - <<'PY'
import torch
import vllm
print(f"torch={torch.__version__}, torch_cuda={torch.version.cuda}")
print(f"vllm={vllm.__version__}")
print(f"cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY

echo "Environment ready: $VENV"
