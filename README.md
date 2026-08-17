# 8 x RTX 4090 24GB 跑 DeepSeek-V4-Flash-0731：部署记录与实测

> 同一台服务器上对比官方 MXFP4 权重与 Q2 GGUF，记录 vLLM、ds4、SGLang
> 和 DSpark 的可用配置、失败条件、性能与 API 兼容性。

最后更新：2026-08-17

## 结论

在单机 `8 x RTX 4090 24GB`、无 NVLink、跨 NUMA PCIe 拓扑上，当前最实用的
DeepSeek-V4-Flash-0731 路线是：

```text
antirez/ds4
+ 80.76 GiB 的 0731 Q2 imatrix GGUF
+ 8 卡 CUDA tensor parallel
+ 131K context
+ 不启用 DSpark
```

本机实测稳定 decode 为 **46.75 token/s**，2K prefill 为 **807.54 token/s**。
38,717-token 冷请求在 **31.57 秒**完成，前缀缓存重放降到 **8.04 秒**。相比
官方 safetensors + vLLM fork 的 4.467 token/s，decode 提高约 **10.5 倍**，
上下文从 1K 提高到 131K。

这组性能变化来自权重格式和推理引擎的更换，不是 vLLM 参数微调：

```text
官方 FP8/MXFP4 safetensors + 通用 vLLM
                       -> 专用 Q2 GGUF + 专用 ds4 CUDA runtime
```

## 结果总览

下面只有标注“本机实测”的数据来自这台服务器。第三方仓库公开数据不能直接横向
比较，因为硬件、权重、上下文、并发和统计口径不同。

| 路线 | 权重 | Context | Decode / Generation | 状态 |
|---|---|---:|---:|---|
| vLLM fork，全 GPU，本机实测 | 官方 155.43 GiB | 1K | **4.467 tok/s** | 能跑，但不实用 |
| vLLM fork，CPU offload，本机实测 | 官方 155.43 GiB | 32K/65K | 约 **2 tok/s** | 长上下文验证 |
| ds4 Q2，本机实测 | 80.76 GiB Q2 GGUF | 131K | **46.75 tok/s** steady | **当前推荐** |
| ds4 Q2 + DSpark，本机实测 | Q2 + 5.58 GiB support | 8K 测试 | **46.43 tok/s** | 无有效 draft，关闭 |
| SGLang SM89 fork，仓库公开 C1 | 官方 MXFP4 | 8K | **54.75 tok/s** | 待本机复现 |
| SGLang SM89 fork，仓库公开 C8 | 官方 MXFP4 | 8K | **318.59 tok/s** 聚合端到端 | 非本机实测 |

`54.75` 和 `46.75` 不能直接比较。SGLang 数据使用 256 input、128 output、
CUDA Graph 和固定 8K 配置；ds4 的 steady decode 来自另一套 2K/128 基准。
引擎 A/B 需要统一请求、采样、并发和计时边界。

## 测试机器

| 项目 | 配置 |
|---|---|
| GPU | 8 x NVIDIA GeForce RTX 4090 |
| 单卡显存 | 24,564 MiB，约 23.54 GiB |
| 架构 | Ada Lovelace，Compute Capability 8.9，SM89 |
| GPU 互联 | 无 NVLink；0-3、4-7 各自 PIX，跨组 SYS |
| NUMA | GPU 0-3 位于 NUMA 0，GPU 4-7 位于 NUMA 1 |
| CUDA toolkit | `/usr/local/cuda-13.3` |
| PyTorch / wheel CUDA | PyTorch 2.11.0+cu130 / CUDA 13.0 ABI |

8 x 24GB 与 4 x 48GB 的总显存接近，但运行条件不同。每张卡都要独立容纳本地
权重分片、CUDA context、通信 buffer、KV、激活和 kernel workspace。TP=8 的
参与者更多，本机还有跨 NUMA 通信。

## 模型与权重体积

官方 `DeepSeek-V4-Flash-0731` 本机文件实测为 48 个 safetensors 分片，权重
合计 **155.43 GiB**。配置包含：

- 43 层 Transformer。
- 256 个 routed experts，每个 token 激活 6 个。
- 1 个 shared expert。
- routed expert 使用 FP4/MXFP4，其他模块混合 FP8/BF16/F32。
- 模型配置最大位置为 1,048,576，原始 RoPE 长度 65,536，并使用 YaRN。
- 官方 checkpoint 自带 DSpark 参数。

MoE 的“每 token 只计算 6 个专家”节省的是计算量，不是完整权重的存储需求。
router 在不同 token 上可能选择 256 个专家中的任意组合，运行时必须保证全部专家
权重可访问。若专家留在 CPU，每个 decode step 都可能发生 PCIe 搬运，所以 CPU
offload 会把生成速度压到很低。

### Q2 GGUF 的量化分布

本次使用的 GGUF：

```text
DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
大小：86,720,111,488 bytes / 80.76 GiB
SHA-256：ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0
```

它只对占空间最多的 routed experts 做激进量化：约 44.34 GiB 为 `IQ2_XXS`
gate/up，约 28.22 GiB 为 `Q2_K` down；投影、共享专家、输出等关键部分保留
Q8/F16。`imatrix` 代表量化时使用校准数据衡量权重重要性，通常比无校准的同位宽
量化更稳。

## SM89、FP4 和 Marlin

`SM89` 是 RTX 4090 的 CUDA 架构标识，不是显存大小或 CUDA 版本：

```text
RTX 4090 -> Ada Lovelace -> Compute Capability 8.9 -> SM89
A100      -> Ampere       -> SM80
H100      -> Hopper       -> SM90
```

4090 可以存储、读取 FP4 权重，但没有 Blackwell 那类原生 FP4 microscaling MMA
指令。vLLM/SGLang 的 SM89 fork 使用 Marlin 兼容 kernel：读取打包 FP4 权重，
应用 scale 并反量化，再走 Ada 支持的计算路径。它解决“能不能运行”，不会凭空让
4090 获得原生 FP4 Tensor Core。

## 三条推理路线

### 路线 A：yhfgyyf/vllm-deepseek-v4-sm89

[yhfgyyf/vllm-deepseek-v4-sm89](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89)
是 vLLM 的 SM89 专用 fork，使用官方权重。它补充了 sparse MLA、Indexer、
FP8 fallback、Marlin MoE 和 CUDA 13 wheel，并提供 OpenAI 风格 API。

本机使用版本：

```text
vLLM 0.23.1rc1.dev904+g8e321cc4f.cu130
FlashInfer 0.6.14+sm89.1
release commit 8e321cc4f73a0e424b9c08621b50009d9d47c6c1
```

远端 refs 中还能看到版本号更大的 `dev1018-g8aba6ae7e`，但该 tag 创建于
2026-07-10，是 2026-08-02 `dev904-g8e321cc4f` 的祖先；dev904 在它之后多
22 个提交。GitHub 最新正式 release 也是 dev904。判断版本新旧需要同时查看 tag
日期、commit ancestry 和 release 页面，不能只看 dev 后面的数字。

`main` 当前为 `2b4f880`，比 dev904 release 前进 7 个提交，包含 TP8 大型
paged-MQA KV pool 的 int32 地址溢出修复和 Triton per-shape kernel cache 增长
修复。它们对长上下文正确性和长期稳定性有价值，但没有改变 24GB 显存、Marlin
MoE 或 TP8 通信瓶颈，因此本轮没有为此重新编译和重测 vLLM。

最快稳定的全 GPU 配置：

```text
TP=8
GPU_MEMORY_UTILIZATION=0.895
CPU_OFFLOAD_GB=0
MAX_MODEL_LEN=1024
MAX_NUM_SEQS=1
ENABLE_DSPARK=0
ENABLE_EXPERT_PARALLEL=0
ENFORCE_EAGER=1
```

模型占用 19.79 GiB/GPU，`nvidia-smi` 总占用约 24,078 MiB/GPU，只剩
0.49 GiB KV。继续提高 memory utilization 会挤掉 warmup workspace，降低又会
让 KV 不够。最终平均 decode 为 4.467 tok/s。

CPU offload 能启动 32K/65K，但约 2 tok/s。Expert Parallel 为 4.344 tok/s，
比普通 TP 慢约 2.8%。这两项都没有成为解决方案。

上游公开的非 DSpark 约 82 tok/s、DSpark 286-344 tok/s 来自 **4 x 48GB
改装 RTX 4090**、TP=4、较高显存余量和 CUDA Graph 条件，不是对 8 x 24GB
机器的性能承诺。

### 路线 B：antirez/ds4

[antirez/ds4](https://github.com/antirez/ds4)，也叫 DwarfStar，是针对少数模型
做深度特化的 C/CUDA/Metal/ROCm 推理引擎，不是通用 GGUF runner。它对
DeepSeek-V4-Flash 的 GGUF layout、MoE、KV、HTTP server 和工具调用一体优化。

本机基于：

```text
commit 84cc882352757baf628a1776badf7cc54d584e28
CUDA 13.3
CUDA_ARCH=sm_89
```

8 卡逻辑顺序使用 `0,2,4,6,1,3,5,7`，让前四个 transformer tier 与后四个
expert owner 配对。由于本机 CUDA P2P 全部不可用，跨卡复制走 pinned-host
bounce。这个顺序只适用于本机拓扑，其他机器必须先看 `nvidia-smi topo -m`。

#### 下载模型

推荐从 `hf-mirror.com` 直连下载，不给大模型下载命令挂代理：

```bash
git clone https://github.com/antirez/ds4.git
cd ds4
mkdir -p gguf

MODEL=DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
aria2c -c -x16 -s16 -k1M \
  -d gguf -o "$MODEL" \
  "https://hf-mirror.com/antirez/deepseek-v4-gguf/resolve/main/$MODEL"
```

本机平均下载约 26 MiB/s。下载后务必核对文件大小和 SHA-256。

#### 编译

```bash
CUDA_HOME=/usr/local/cuda-13.3 \
  make -j16 cuda CUDA_ARCH=sm_89

cuobjdump --list-elf ./ds4 | grep sm_89
```

#### 无 P2P 机器的重要兼容点

该 commit 的底层跨卡复制已经支持 host bounce，但
`metal_graph_encode_mixed_routed_rows()` 的 owned-prefill 入口仍提前强制检查
P2P，导致首层 prefill 失败。本机删除了这一项前置拒绝，让已有 fallback 生效：

```diff
 if (total_rows < prefill_rows || total_rows > g->prefill_cap ||
     !g->cuda_tp_ep || partner_tier < 0 ||
-    !g_gpu_peer_ok[partner_tier][home_tier] ||
     !metal_graph_ensure_batch_ffn_out_on(g, home_tier)) {
     return false;
 }
```

这项改动没有开启 P2P，只允许 pinned host memory 做 D2H/H2D 中转。它修复了
当前拓扑下的执行路径，性能仍受 PCIe/NUMA 限制。支持 P2P 的机器不需要套用。

#### 启动服务

先用 loopback 验证 stock upstream：

```bash
./ds4-server \
  --cuda \
  --cuda-tensor-parallel \
  --gpu-vram auto \
  --gpu-devices 0,2,4,6,1,3,5,7 \
  -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --ctx 131072 \
  --tokens 4096 \
  --host 127.0.0.1 \
  --port 8000 \
  --kv-disk-dir runtime/kv-cache \
  --kv-disk-space-mb 8192
```

本机常驻使用 transient user unit，`Restart=no`、不 enable、不随开机启动。公开到
局域网或公网前必须增加认证；本机给 ds4 增加了 `--api-key-file` Bearer 校验，
公开复现更建议放在 Caddy/Nginx 后面做 TLS、认证、限流和访问日志。不要把没有认证
的 stock server 直接绑定公网地址。

#### OpenAI API

本机已验证：

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses` 和完整 SSE lifecycle
- Responses `tool_choice:auto` 和 `function_call_output` continuation

`tool_choice:"required"` 当前返回 HTTP 400。Codex/Codex++ 优先使用
`/v1/responses`；普通聊天客户端可使用 `/v1/chat/completions`。

### 路线 C：xltzsoft/deepseek-v4-sm89

[xltzsoft/deepseek-v4-sm89](https://github.com/xltzsoft/deepseek-v4-sm89) 是基于
SGLang 的 8 x RTX 4090 24GB / SM89 实验分支。核对时最新 commit 为：

```text
94a89e65736b45706603a78d3150b0151b135162
perf: publish SM89 paired tuning and lifecycle fixes
```

它使用官方 MXFP4 权重，重点补齐：

- SM89 MXFP4 Marlin MoE。
- sparse MLA decode/prefill Triton kernel。
- Ada 专用 sparse prefill geometry。
- TP8 原生 8 query heads，避免填充到 64 heads。
- FP8 Indexer、MHC prenorm fallback。
- decode full CUDA Graph 和 FP8 E4M3 KV。
- DSpark shared expert 加载、scheduler 和状态生命周期修复。

仓库公开配置为 context 8192、最大并发 8、关闭 custom all-reduce。公开结果：

| 场景 | 仓库公开结果 |
|---|---:|
| C1 decode | 54.75 tok/s |
| C1 端到端 | 50.36 tok/s |
| C1 TTFT | 0.222 s |
| C8 scheduler 输出 | 382.8-383.8 tok/s |
| C8 decode-window 聚合 | 374.58 tok/s |
| C8 端到端聚合 | 318.59 tok/s |
| C8 GPU 平均利用率 | 约 92.7% |

这些是仓库作者公开数据，**本项目尚未在本机复现**。SGLang 更接近通用高并发
服务框架，适合继续验证 8 并发聚合吞吐。

该仓库标注为性能研究快照：多请求正反顺序压力验证尚未完整重跑，DSpark 也没有
通过严格逐 token parity。仓库当前推荐 **target-only + decode full CUDA Graph**。

## DSpark 原理与本机 A/B

DSpark 是推测解码：轻量 draft 模块一次提出多个未来 token，主模型批量验证并只
提交被接受的前缀。接受率高时，一个 target step 可以推进多个 token；接受率低时，
draft 和验证成本可能让它更慢。DSpark 只加速 decode，不解决长 prompt prefill。

同一个 DSpark 在不同引擎中包装不同：

| 引擎 | DSpark 权重来源 |
|---|---|
| 官方 vLLM / SGLang 路线 | 官方 safetensors checkpoint 内已带 DSpark 参数 |
| ds4 GGUF 路线 | 单独下载约 5.58 GiB support GGUF，不是独立聊天模型 |

ds4 support 文件：

```text
DeepSeek-V4-Flash-DSpark-support-0731.gguf
大小：5,989,114,272 bytes / 5.58 GiB
SHA-256：7e319924541db3f7a163ed7e11d7532a70d48228ab59d36cb81e1d4511885360
```

本机 A/B 使用相同的 C 语言 LRU prompt、8 卡、8K context、256 greedy output、
`temperature=0`、no-thinking：

| 模式 | Generation | 总耗时 | 输出 |
|---|---:|---:|---|
| 普通 Q2 | **47.53 tok/s** | 32.82 s | 成功 |
| DSpark strict | 46.62 tok/s | 33.65 s | 与普通输出逐字一致 |
| DSpark enabled | 46.43 tok/s | 33.79 s | 成功，但没有有效 draft |

统计为：

```text
cycles=247
proposed=0
accepted_draft=0
scheduler_skips=188
tail_skips=8
net_saved=-44.539 ms
```

support 模型成功装入，测试峰值显存为 20,964 MiB，容量不是问题。但 support MoE
stage-chain 在无 P2P 多卡 placement 上发生 selective-cache miss，scheduler 全部
回退普通 target decode，最终比 baseline 慢约 2.3%。因此当前服务保持 DSpark
关闭。

SGLang fork 的两组 DSpark launch-meta 配对搜索分别为 `0.991327x` 和
`0.988452x`，没有产生 winner，逐 token parity 也尚未完成。DSpark 是否加速
取决于引擎实现、权重布局和 GPU 拓扑，需要逐机 A/B。

## 本机 ds4 性能明细

### 短基准

统一为 2,048 input + 128 greedy output：

```text
prefill_tps=807.54
gen_tps=46.02
gen_first_ms=52.867
gen_steady_tokens=127
gen_steady_tps=46.75
```

### 长上下文与 prefix cache

| 运行 | Prompt | Cached | Re-prefill | 总耗时 |
|---|---:|---:|---:|---:|
| 冷请求 | 38,717 | 0 | 38,717 | 31.57 s |
| 原样重放 | 38,717 | 30,720 | 7,997 | 8.04 s |

首次 SSE keepalive 在 5.10 秒到达，客户端不会在长 prefill 时一直无事件等待。磁盘
KV 以约 10K token 边界保存，因此重放命中 30,720，而不是完整 prompt。

### 显存

`--ctx 131072` 的 context buffers 为 2,426.72 MiB。服务静置为
13.35-15.02 GiB/GPU，38K 长 prefill 采样峰值约 18.02 GiB/GPU。131K 是当前
Codex 场景的保守值，不是硬件绝对上限。

## 产能与 API 等价价值

### 理论输出产能

按本机 steady decode `46.75 output token/s` 连续生成计算：

```text
每小时：46.75 x 3,600 = 168,300 output tokens
每天：  46.75 x 86,400 = 4,039,200 output tokens
每月：  4,039,200 x 30 = 121,176,000 output tokens
```

因此理论上限约为 **404 万 output tokens/天、1.212 亿/月**。这只计算输出，
没有把 input tokens 加进去，也假设 GPU 24 小时都处在 steady decode。

生产环境还要处理请求切换、prefill、长短不一、排队、缓存未命中和空闲时间。
若按理论值的 80%-90% 估算：

| 利用率 | 每天 output tokens | 每月 output tokens（30 天） |
|---:|---:|---:|
| 80% | 约 323 万 | 96,940,800，约 9,694 万 |
| 90% | 约 364 万 | 109,058,400，约 1.091 亿 |

这个折扣仍是较高利用率假设。输入越长、KV 命中率越低，GPU 花在 prefill 上的
时间越多，实际 output 日产量越低；若统计 input + output 总 token，则结果主要由
请求的输入输出比例决定，不能从 decode TPS 单独推出。

### DeepSeek 官方 API 等价价格

根据 [DeepSeek 官方价格页](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)
在 2026-08-17 显示的 `deepseek-v4-flash` 价格，每百万 tokens 单价为：

| 项目 | 空闲时段 | 高峰时段 |
|---|---:|---:|
| 输入，缓存命中 | ¥0.05 | ¥0.10 |
| 输入，缓存未命中 | ¥1.50 | ¥3.00 |
| 输出 | ¥4.50 | ¥9.00 |

官方定义北京时间 `09:00-12:00`、`14:00-18:00` 为高峰，共 7 小时；其余
17 小时为空闲。若假设请求在一天内均匀分布，对应 24 小时加权平均单价为：

```text
输出：¥5.8125 / 百万 tokens
缓存命中输入：约 ¥0.0646 / 百万 tokens
缓存未命中输入：¥1.9375 / 百万 tokens
```

理论满载月产量 1.21176 亿 output tokens 的官方 API 等价费用：

| 计费场景 | 全为空闲时段 | 全为高峰时段 | 24 小时均匀分布 |
|---|---:|---:|---:|
| 只计算输出 | ¥545.29 | ¥1,090.58 | **¥704.34** |
| 输入:输出=1:1，输入全命中缓存 | ¥551.35 | ¥1,102.70 | **¥712.16** |
| 输入:输出=1:1，输入全未命中 | ¥727.06 | ¥1,454.11 | **¥939.11** |

按更实际的 80%-90% output 利用率，并继续假设请求全天均匀分布：

| 场景 | 月度官方 API 等价费用 |
|---|---:|
| 只计算输出 | **¥563-¥634** |
| 输入:输出=1:1，输入全命中 | **¥570-¥641** |
| 输入:输出=1:1，输入全未命中 | **¥751-¥845** |

本文把“价值”定义为**按官方挂牌价计算的 API 替代成本**。它不代表收入，也不
假设本地 Q2 与官方服务在质量、1M context、可用性和并发上完全等价。官方价格
可能调整，引用这些数字时应保留价格日期。

只看 output API 替代成本，80%-90% 利用率对应每小时约 ¥0.78-¥0.88。电费的
盈亏阈值可按下面公式估算：

```text
可承受平均功耗（kW） = 每小时 API 等价费用 / 电价（元/kWh）
```

例如电价为 ¥0.8/kWh 时，output-only 的电费盈亏阈值仅约 0.98-1.10 kW；超过
这个整机平均功耗，仅电费就高于对应 API 替代成本，还没有计算硬件折旧、机房、
运维和故障成本。实际功耗应使用墙上功率计或 PDU 数据，不能用 GPU TDP 简单相加。

按这组价格，本地 8 卡方案很难仅靠节省 API 费用覆盖成本。适合它的需求是数据不出
内网、离线运行、避免外部限流与服务依赖，或者需要修改推理栈并控制缓存和日志。

## 长对话和代码能力实测

性能测试之外，又用三个端到端任务检查 Q2 的可用性。

### 27K 长文档记忆

首轮 27,755-token prompt 包含 520 条相似记录和三个分散审计条目，模型准确提取
全部 10 个字段；后续两轮正确完成跨记录计算，并拒绝用户故意提供的错误信息。

| 轮次 | Prompt | Cached | Output | 耗时 | 结果 |
|---|---:|---:|---:|---:|---|
| 提取 | 27,755 | 0 | 192 | 27.84 s | 全部字段正确 |
| 计算 | 28,001 | 27,947 | 24 | 0.87 s | 正确 |
| 抗错误纠正 | 28,079 | 28,025 | 41 | 1.32 s | 正确 |

### 代码修改

在约 10K-token C 项目中，no-thinking 首轮找全整数溢出、悬空指针、NUL 终止和
retry 边界问题，但生成最小 diff 时失败并重复无关代码。把完整历史提交给
`reasoning_effort=high` 后，模型给出三个可实施 patch hunk，并正确指出一个在
冻结 ABI/禁止分配/并发失效约束下无法安全满足的裸指针接口。

结论：短问答可 no-thinking，复杂 coding agent 默认应开启 reasoning，并给足输出
预算。当前测试没有官方权重同 prompt A/B，不能把 no-thinking 失败简单归因于 Q2。

### Responses 工具 continuation

模型先返回正确函数调用：

```text
lookup_incident({"incident_id":"INC-9042"})
```

提交 `function_call_output` 后没有重复调用工具，并准确保留工具结果中的级别、根因、
影响集群、缓解措施、解决时间和 owner。说明 Responses 多轮工具闭环可用。

## 基准口径

至少固定以下变量，否则 TPS 数字没有比较意义：

1. 同一模型版本和量化。
2. 同一 input/output token 数。
3. greedy 或相同采样参数，并设置 `ignore_eos` 避免提前停止。
4. 同一 context 上限、KV dtype 和 prefix cache 状态。
5. 同一并发数；单请求 TPS 和聚合吞吐必须分开。
6. 区分 TTFT、prefill TPS、steady decode TPS 和端到端吞吐。
7. 至少一次预热，重复多轮，同时记录功耗、温度和显存峰值。
8. 推测解码必须报告 proposed、accepted 和 acceptance rate，不能只给最终 TPS。
9. correctness 必须比较 token IDs；只看文本可能掩盖浮点路径差异。

ds4 复现命令示例：

```bash
./ds4-bench \
  --cuda \
  --cuda-tensor-parallel \
  --gpu-vram auto \
  --gpu-devices 0,2,4,6,1,3,5,7 \
  --prompt-file README.md \
  --ctx-start 2048 \
  --ctx-max 2048 \
  --ctx-alloc 4096 \
  --gen-tokens 128
```

## 常见问题

### 为什么 3 万 token 请求会报最大 32K？

上下文预算是 `input + output`，还包含 system prompt、工具定义和历史消息。客户端
看到 `requested 0 output tokens`，通常是输入已把 context 吃满，不是模型拒绝回答。

### 为什么 Codex 第一条请求十分钟没回复？

旧 vLLM + CPU offload 的长 prompt 仍在 prefill，服务端没有及时发 SSE 事件，客户端
触发 idle timeout。预热只能减少 JIT 和冷缓存成本，无法把持续经过 PCIe 的权重路径
变快一个数量级。

### Q2 位宽越低就一定越快吗？

不一定。更小通常意味着更少内存流量，但质量可能下降，kernel 和 layout 也可能让
低位宽反而出现额外开销。当前 Q2 能让权重和运行空间同时放进 8 x 24GB，但这组
结果不能外推为所有 Q2 都优于 Q3/Q4。

### W8A8 为什么比 FP4 大？

W8A8 表示权重和激活通常按 8 bit 处理，目的可能是适配特定 INT8/FP8 kernel，而
不是追求最小文件。本次查看的 Eco-Tech W8A8 权重约 305.45 GB，超过本机总显存，
且没有清晰的 NVIDIA vLLM/SGLang 支持声明，因此没有继续投入。

## 选择建议

| 目标 | 建议 |
|---|---|
| 8 x 4090 24GB 单用户交互、Codex | ds4 + 0731 Q2 imatrix，DSpark off |
| 需要官方权重做质量基线 | vLLM SM89 fork，但接受短 context/低速 |
| 追求多并发聚合吞吐 | 独立复现 SGLang SM89 fork 的 target-only 路线 |
| 追求更高 Q2 质量 | 下一步测约 98 GiB 的 ds4f-q2-q4 |
| 只想打开 DSpark | 先做严格 A/B；不要默认它一定加速 |
| 需要公网服务 | 反向代理 + TLS + 鉴权 + 限流，不直出实验 server |

## 已停止的实验

- 不再围绕 vLLM 的 `GPU_MEMORY_UTILIZATION=0.895` 扫更多小数点，限制来自单卡
  空间、warmup workspace 和 TP8 通信。
- 不在当前 24GB 卡配置上继续尝试 vLLM DSpark，V2 runner 没有可用显存窗口。
- 不把 Expert Parallel 当作默认优化，本机已实测略慢。
- 不在 ds4 上继续修当前 DSpark 多卡 placement，baseline 已经更快且稳定。
- 不下载 305GB 的 W8A8 期待它自动消除 offload。

## 下一步

1. 用完全相同的 256/128 请求在 ds4 Q2 和 SGLang MXFP4 上做本机 C1 A/B。
2. 在 SGLang 上复现 C8 聚合吞吐，并测试跨 NUMA 绑核和 GPU 顺序。
3. 建立 Q2 与官方权重的代码修复、中文、工具调用质量对照集。
4. Q2 质量不足时再下载 `ds4f-q2-q4`，不要直接跳到 153-156 GiB Q4/MXFP4。
5. 为长期服务补齐并发、限流、监控、故障恢复和磁盘 KV 清理策略。

## 文档与原始结果

仓库内容按“文档、实验结果、部署复现、评估脚本”分层：

```text
.
├── README.md
├── LICENSE
├── docs/
│   ├── concepts.md
│   └── reports/
│       ├── ds4-deployment-2026-08-16.md
│       └── vllm-deployment-2026-08-16.md
├── results/
│   └── quality/
│       ├── coding-recovery-nothink-2026-08-16.json
│       ├── coding-recovery-thinking-2026-08-16.json
│       └── long-conversation-2026-08-16.json
├── deploy/
│   └── vllm/
│       ├── .env.example
│       ├── deepseek-v4-flash.service
│       └── ...
└── scripts/
    └── eval/
        └── ds4_long_conversation_eval.py
```

`deploy/vllm/` 只用于复现官方权重的历史基线，不是当前推荐服务；它的范围和
安全边界见 [vLLM 复现说明](deploy/vllm/README.md)。当前 ds4 路线使用上游
`antirez/ds4` 源码，本仓库保留实测命令、兼容补丁说明和结果，不复制上游仓库。

三份 JSON 使用脚本生成的合成档案、合成 C 代码和虚构事件信息，不含真实业务数据。
以后重新运行测试时，仍应在提交结果前重新检查 prompt 和模型输出。质量测试脚本不
内置 key，可按服务是否启用鉴权选择下面任一方式：

```bash
DS4_BASE_URL=http://127.0.0.1:8000/v1 \
DS4_API_KEY_FILE=/path/to/api-key \
python3 scripts/eval/ds4_long_conversation_eval.py
```

key 文件建议设为 `0600`。脚本也支持由 CI secret 注入 `DS4_API_KEY`，但不要把
真实值写进命令、脚本、`.env.example` 或仓库设置以外的明文文件。

- [vLLM 完整部署报告](docs/reports/vllm-deployment-2026-08-16.md)
- [ds4 完整部署、质量和 DSpark A/B 报告](docs/reports/ds4-deployment-2026-08-16.md)
- [概念与路线说明](docs/concepts.md)
- [长对话测试结果](results/quality/long-conversation-2026-08-16.json)
- [no-thinking 代码恢复结果](results/quality/coding-recovery-nothink-2026-08-16.json)
- [thinking 代码恢复结果](results/quality/coding-recovery-thinking-2026-08-16.json)

本项目不分发模型权重、API key、服务器日志或本地 KV cache。

## 参考资料

- [DeepSeek-V4-Flash-0731 官方 ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash-0731/)
- [DeepSeek-V4-Flash-DSpark 官方 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)
- [yhfgyyf/vllm-deepseek-v4-sm89](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89)
- [antirez/ds4](https://github.com/antirez/ds4)
- [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)
- [xltzsoft/deepseek-v4-sm89，SGLang fork](https://github.com/xltzsoft/deepseek-v4-sm89)
- [SGLang 官方文档](https://docs.sglang.ai/)
- [Unsloth DeepSeek V4 文档](https://unsloth.ai/docs/models/deepseek-v4)
- [Unsloth DeepSeek-V4-Flash-0731 GGUF](https://www.modelscope.cn/models/unsloth/DeepSeek-V4-Flash-0731-GGUF)
- [Eco-Tech W8A8](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8)

## 开源说明

本文的数字只适用于上述硬件、软件版本和测试方法。模型、ds4、vLLM、SGLang
及其派生项目分别受各自许可证约束；本仓库不包含模型文件。原创文档与脚本采用
[MIT License](LICENSE)，上游项目继续遵循各自的许可证和 attribution 要求。
