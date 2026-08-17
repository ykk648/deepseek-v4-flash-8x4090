# vLLM SM89 baseline reproduction

This directory reproduces the historical official-safetensors baseline built
with `yhfgyyf/vllm-deepseek-v4-sm89`. On the tested 8 x RTX 4090 24 GB host,
the fastest stable all-GPU configuration was limited to a 1K context and
4.467 output token/s. It is retained for reproducibility and quality reference;
it is not the repository's recommended interactive deployment.

See [the complete vLLM report](../../docs/reports/vllm-deployment-2026-08-16.md)
for the tested environment, memory boundary, DSpark failures and benchmark
methodology. The current recommendation is ds4 with the 0731 Q2 imatrix GGUF,
documented in the repository [README](../../README.md).

## Local setup

```bash
cd deploy/vllm
cp .env.example .env
# Edit MODEL_DIR and CUDA_HOME before continuing.

./setup_env.sh
./download_model.sh
./serve.sh
```

The public template binds to `127.0.0.1` because this historical vLLM setup
does not validate an API key. Do not change it to `0.0.0.0` without adding an
authenticated reverse proxy and network access controls.

For a user-level systemd service, this example assumes the repository is at
`$HOME/deepseek-v4-flash-8x4090`:

```bash
mkdir -p ~/.config/systemd/user
cp deepseek-v4-flash.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start deepseek-v4-flash.service
```
