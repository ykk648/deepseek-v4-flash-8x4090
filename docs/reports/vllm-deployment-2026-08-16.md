# DeepSeek-V4-Flash-0731 部署与性能报告

日期：2026-08-16

## 1. 结论

官方 `DeepSeek-V4-Flash-0731` safetensors 权重已经在本机 8 张 RTX
4090 24 GB 上通过 `yhfgyyf/vllm-deepseek-v4-sm89` 跑通，并能提供
OpenAI-compatible HTTP 服务。

本机找到的最快稳定配置是全 GPU、TP=8、1K 上下文，稳定 decode 为
**4.467 token/s**。它证明这条技术路径可以运行，但 1K 上下文不适合
Codex。启用 CPU offload 可以扩大到 32K/65K 上下文，但 decode 约
2 token/s，长 prompt 的首 token 延迟也很高。

仓库公布的 82 token/s 非 DSpark、286-344 token/s DSpark 数据不能在
本机复现。核心差异不是一个遗漏的 vLLM 参数，而是每卡显存、GPU 拓扑和
硬件执行路径：上游使用 4 张每卡 48 GB 的改装 RTX 4090，本机是 8 张
每卡 24 GB 的标准 RTX 4090，并且没有 NVLink。

因此，本轮不再继续边际调参或为最新 `main` 源码重编。当前官方权重加
vLLM fork 的能力边界已经足够清楚。若继续追求实用速度，应开一个独立的
`ds4 + 兼容 GGUF Q2/Q3` 验证项目，而不是继续挤压当前配置。

## 2. 硬件和软件

| 项目 | 实际值 |
|---|---|
| GPU | 8 x NVIDIA GeForce RTX 4090 |
| 单卡显存 | 24,564 MiB（约 23.54 GiB 可用容量） |
| Compute capability | 8.9，即 SM89 / Ada Lovelace |
| GPU 互联 | 无 NVLink；0-3 和 4-7 各自为 PIX，跨组为 SYS |
| NUMA | GPU 0-3 在 NUMA 0，GPU 4-7 在 NUMA 1 |
| CUDA toolkit | `/usr/local/cuda-13.3` |
| Python | 3.12.10 |
| PyTorch | 2.11.0+cu130，`torch.version.cuda=13.0` |
| vLLM | 0.23.1rc1.dev904+g8e321cc4f.cu130 |
| FlashInfer | 0.6.14+sm89.1 |
| vLLM release commit | `8e321cc4f73a0e424b9c08621b50009d9d47c6c1` |

主机安装 CUDA 13.3 与 cu130 wheel 不冲突。wheel 自带/依赖 CUDA 13.0
运行库，关键条件是 NVIDIA 驱动能运行 CUDA 13.x，并且 PyTorch、vLLM、
FlashInfer 的版本保持配套。

### GPU 拓扑

```text
GPU0-3: 组内 PIX，NUMA 0
GPU4-7: 组内 PIX，NUMA 1
两组之间: SYS，需要经过 PCIe 和 CPU NUMA 互联
NVLink: 无
```

运行日志显示 TP collective 使用 `PYNCCL`。vLLM 的 custom all-reduce
不支持这组 8 卡 PCIe-only 拓扑。TP=8 时每个 token 都有跨卡同步，跨
NUMA 的 `SYS` 路径进一步放大通信开销。

## 3. 模型和运行路径

模型目录：

```text
<WORKDIR>/models/DeepSeek-V4-Flash-0731
```

模型包含 48 个 safetensors 分片，权重精确合计 155.43 GiB。配置包含：

- 43 层、256 个 routed experts、每 token 激活 6 个专家。
- 普通量化配置为 FP8，MoE `expert_dtype=fp4`。
- 架构最大位置为 1,048,576，原始 RoPE 长度为 65,536，并使用 YaRN。
- 模型内包含 DSpark 所需参数，不需要另下载一套 DSpark 权重。

RTX 4090 有 FP8 Tensor Core，但没有原生 FP4/microscaling MMA。日志确认
MoE 走：

```text
DeepSeek V4 expert_dtype resolved to 'fp4'
Using 'MARLIN' Mxfp4 MoE backend
```

Marlin 在 SM89 上读取 FP4 打包权重、应用 scale、反量化，然后使用硬件
支持的计算路径完成矩阵乘法。它解决兼容性，但不能得到 Blackwell 原生
FP4 指令的速度。

## 4. 最终稳定配置

当前 `.env`：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TENSOR_PARALLEL_SIZE=8
GPU_MEMORY_UTILIZATION=0.895
CPU_OFFLOAD_GB=0
MAX_MODEL_LEN=1024
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=512
ENABLE_DSPARK=0
ENABLE_EXPERT_PARALLEL=0
ENFORCE_EAGER=1
```

成功启动时的关键数据：

| 项目 | 实测 |
|---|---:|
| vLLM 报告模型占用 | 19.79 GiB/GPU |
| `nvidia-smi` 总占用 | 约 24,078 MiB/GPU |
| 可用 KV cache | 0.49 GiB |
| GPU KV cache | 2,479 tokens |
| 1,024-token 最大并发估算 | 2.42x |
| 启动结果 | `Application startup complete` |

`GPU_MEMORY_UTILIZATION=0.895` 不是性能最佳点，而是能通过 sparse MLA
warmup 的最高稳定边界附近。再提高会把 KV cache 或临时 workspace 挤到
无法完成 warmup。

## 5. 性能基准

基准脚本：`deploy/vllm/benchmark_stream.py`

方法：单并发、短固定 prompt、`temperature=0`、`ignore_eos=true`、流式
输出 128 tokens，连续运行三次。decode TPS 使用首个 token 之后的时间
计算，避免 TTFT 混入生成速度。

| 运行 | TTFT | 总耗时 | Decode TPS |
|---:|---:|---:|---:|
| 1 | 0.889 s | 29.67 s | 4.506 tok/s |
| 2 | 0.245 s | 28.77 s | 4.453 tok/s |
| 3 | 0.244 s | 28.24 s | 4.442 tok/s |
| 平均 | 0.459 s | 28.89 s | **4.467 tok/s** |

第一次 TTFT 较高来自首次请求/JIT 等预热成本；后两次约 0.245 秒。预热
可以改善首次 TTFT，但不会把稳定 decode 从 4.47 提升到几十或几百
token/s。

### Expert Parallel 对比

`ENABLE_EXPERT_PARALLEL=1` 启动成功，每卡持有 32/256 个专家，并使用
`AgRsAll2AllManager`。同样基准的平均 decode 为 **4.344 token/s**，比
普通 TP 的 4.467 token/s 慢约 2.8%。本机跨 NUMA PCIe 通信抵消了 EP 的
潜在收益，因此最终关闭。

### 历史 CPU offload 结果

历史配置使用每 worker 20 GiB offload 上限，实际约 18.98 GiB/卡的模型
数据通过 CPU/UVA 路径提供。它可以启动 32K/65K 上下文，短输出 decode
约 2 token/s，`/v1/chat/completions` 和 `/v1/responses` 均已跑通。

这一配置适合证明长上下文功能，不适合交互式 Codex：长 prompt prefill
本身很慢，decode 又持续经过 PCIe 数据路径，客户端容易在第一个 SSE
事件前触发 idle timeout。备份位于 `.env.cpu-offload-backup`。

## 6. 显存边界实验

### 普通 TP、无 DSpark、无 offload

| GMU | 上下文 | KV cache / 现象 | 结果 |
|---:|---:|---|---|
| 0.92 | 8192 | 0.73 GiB，不足以容纳 8K 请求 | 启动失败 |
| 0.93 | 8192 | 0.97 GiB | sparse MLA warmup OOM |
| 0.92 | 1024 | 1.08 GiB | warmup OOM |
| 0.90 | 1024 | 0.61 GiB；申请 512 MiB 时仅余 428.44 MiB | warmup OOM |
| **0.895** | **1024** | **0.49 GiB，2,479 tokens** | **成功** |

提高 GMU 会增加 KV cache 预算，却减少 warmup 可使用的非 KV 临时空间；
降低 GMU 则相反。这里不存在一个能同时稳定容纳 8K KV 和 warmup 的甜点
区间。

### DSpark

仓库对 0731 推荐 `method=dspark`，当前模型也确实包含 draft 参数。实测
DSpark 强制使用 V2 Model Runner，模型占用从 19.79 墠至 21.12 GiB/GPU。

| 配置 | 观察 | 结果 |
|---|---|---|
| GMU 0.895，1K | 可用 KV 为 -0.87 GiB | 失败 |
| GMU 0.953，1K | KV 0.50 GiB，2,408 tokens | warmup OOM |
| GMU 0.938，512 | KV 0.21 GiB，935 tokens | warmup OOM |
| GMU 0.90，offload 2 GiB | 可用 KV 为 -0.75 GiB | 失败 |

V2 runner 日志没有创建现有的 `UVAOffloader`，配置 2 GiB CPU offload 后
模型仍占 21.12 GiB/GPU。因此当前 wheel 中给 V1 runner 使用的 offload
路径不能用来救 DSpark。DSpark 在这台 24 GB/卡机器上没有可启动的显存
窗口。

## 7. 为什么上游快、本机慢

| 项目 | 上游公开基准 | 本机 |
|---|---|---|
| GPU | 4 x RTX 4090，**48 GB/卡** | 8 x RTX 4090，24 GB/卡 |
| TP | 4 | 8 |
| 单卡运行余量 | 约 48 GB 总容量 | 约 23.54 GiB 总容量 |
| 互联 | 未完整公开 | 两组 PIX、跨组 SYS、无 NVLink |
| GMU | 0.96-0.97 | 最高稳定 0.895 |
| CUDA Graph | 开启 | 为稳定性使用 eager |
| DSpark | 可启用 | 21.12 GiB 权重后 warmup OOM |
| 非 DSpark decode | 约 82 tok/s | 4.467 tok/s |
| DSpark decode | 286-344 tok/s | 无法启动 |

`4 x 48 GB` 和 `8 x 24 GB` 的总显存都约为 192 GB，但并不等价：

1. 权重和运行时内存必须在每张卡本地满足分片布局，不能把八张卡的空闲
   显存当成一块完全可交换的 192 GB 内存。
2. TP=8 比 TP=4 每层有更多 collective；本机还有跨 NUMA `SYS` 通信。
3. 每张 24 GB 卡装入约 19.79 GiB 模型后，只剩约 3.75 GiB 给 CUDA
   context、通信 buffer、KV、激活和 warmup workspace。
4. DSpark 把模型占用推到 21.12 GiB/GPU，进一步压缩运行空间。
5. 4090 没有原生 FP4 MMA，MoE 使用 Marlin FP4-to-FP16 fallback。

所以总显存相同不能推出性能或可启动配置相同。上游数字在其硬件条件下
可信，但不是对 8 x 24 GB 的性能承诺。

## 8. 上游更新判断

截至 2026-08-16，release commit 到 `main` 共前进 7 个 commit：

- `main` 最新 commit：`2b4f880455df5116745e737610b4b5395e6cde7d`
- commit 日期：2026-08-14
- 文档类改动：release 安装说明和 GSM8K 描述修正。
- 运行时修复一：TP=8 大型统一 KV pool 中 paged MQA block 地址的 int32
  溢出。超过约 2K blocks 时旧代码可能产生负 offset，并在长上下文或 prefix
  caching 中触发 illegal memory access。
- 运行时修复二：SM89 Triton per-shape kernel cache 持续增长。
- 最新 release：仍为本机已安装的
  `v0.23.1rc1.dev904-g8e321cc4f-cu130-sm89`。

远端 refs 还存在名称为 `dev1018-g8aba6ae7e` 的 tag，但该 tag 创建于
2026-07-10，是 2026-08-02 `dev904-g8e321cc4f` 的祖先；后者比前者多 22
个提交。版本号中的 dev 数字不能单独作为新旧依据，按 commit ancestry、日期和
GitHub release 页面判断，dev904 才是当前应使用的已验证 wheel。

地址修复在读取 paged MQA block index 后、参与 stride 乘法前转换为 int64。
缓存修复则把请求相关的 shape/stride 参数从 Triton `constexpr`
specialization 中移除，避免长时间接受不同形状请求时不断生成和保留新
kernel。两项修复对 TP=8 长上下文正确性和长期内存稳定性有价值，但没有
修改：

- Marlin MoE 计算路径。
- TP=8 / PYNCCL / PCIe collective。
- 24 GB 卡的模型、KV 和 warmup 空间。
- DSpark V2 runner 的显存占用和 offload 支持。

当前没有包含这些修复的新 wheel。当前稳定基线只有 1K 上下文，不会达到
大型 KV pool 的溢出条件；为不提升吞吐、也不解决启动边界的提交立即重编
整个 vLLM，收益不足，因此本轮不做源码升级。以后发布匹配 wheel 时应当
更新，尤其是在重新启用 CPU offload 长上下文或 prefix caching 之前；但不
应期待 4.47 token/s 因此变成 README 中的数百 token/s。

## 9. API 和运行方式

服务地址：

```text
http://<SERVER_IP>:8000/v1
```

模型名：`DeepSeek-V4-Flash-0731`

支持的已验证端点：

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

服务端没有配置 API key 校验。要求 key 非空的客户端可以填写
`<API_KEY>` 等任意占位值。

用户级 unit 没有 enable，且 `Restart=no`。测试阶段只手动启动：

```bash
cd <WORKDIR>/deepseek-v4-flash-8x4090/deploy/vllm
mkdir -p ~/.config/systemd/user
cp deepseek-v4-flash.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start deepseek-v4-flash.service
systemctl --user status deepseek-v4-flash.service
./smoke_test.sh
python3 benchmark_stream.py --max-tokens 128 --repeats 3
```

停止并检查显存：

```bash
systemctl --user stop deepseek-v4-flash.service
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

报告完成时服务为 `inactive`，8 张卡均为 `1 MiB / 24564 MiB`、利用率
0%，显存已释放。

## 10. 后续建议

### 不建议继续做的事

- 不再围绕 GMU 0.895 附近做更细的小数点扫描。当前限制是 warmup 临时
  空间，不是尚未找到的单一最优参数。
- 不再尝试当前 wheel 的 DSpark + 24 GB 卡组合。V2 runner 没有可用的
  offload 退路。
- 不把 Expert Parallel 当作解决方案；本机已实测略慢。
- 不投入 Eco-Tech W8A8：权重约 305.45 GB，超过总显存，且没有明确的
  NVIDIA vLLM/SGLang runtime 支持。
- 不为当前两个正确性/稳定性修复单独源码重编；等待正式 wheel。恢复长
  上下文前应优先升级。

### 真正不同的下一条路线

独立评估 `antirez/ds4` 和它明确兼容的 GGUF。优先从 Q2/Q3 开始：

| GGUF | 大致大小 | 对本机的意义 |
|---|---:|---|
| Q2_K_XL | 约 96.8 GB | 显存余量最大，先验证速度和质量 |
| Q3_K_XL | 约 128.2 GB | 质量/容量折中 |
| Q4_K_XL | 约 155.1 GB | 接近当前权重大小，运行余量较紧 |

这不是当前 vLLM 配置的微调，而是换模型格式和推理引擎。需要重新验收
多 GPU 切分、实际 VRAM、短/长 prompt、decode TPS、OpenAI API、工具
调用和回答质量。只有这类架构级变化，才有机会显著跨过当前 4.47
token/s 的上限。
