# DeepSeek-V4-Flash-0731 部署与概念指南

> 本文解释本机 8 x RTX 4090 24 GB 部署所涉及的模型、引擎、量化和性能
> 概念。vLLM 历史实验见
> [vLLM 部署报告](reports/vllm-deployment-2026-08-16.md)，当前 ds4 部署见
> [ds4 部署报告](reports/ds4-deployment-2026-08-16.md)。
>
> 更新时间：2026-08-17

## 一、当前配置

当前实验服务使用以下配置：

```text
模型权重：antirez/deepseek-v4-gguf 的 Flash 0731 Q2 imatrix GGUF
推理引擎：antirez/ds4，commit 84cc882
GPU：8 x NVIDIA RTX 4090 24 GB，SM89，Tensor Parallel=8
CUDA：主机 CUDA 13.3，按 sm_89 原生编译
上下文：131,072 tokens
接口：Chat Completions、Responses、Completions、Anthropic Messages
```

ds4 的 2,048-token 专用基准为 prefill **807.54 token/s**、稳定 decode
**46.75 token/s**。真实 38,717-token Responses 请求从零 prefill 加极短
输出共 31.57 秒，重放时命中 30,720 个缓存 token 后为 8.04 秒。与历史
vLLM 全 GPU 1K 配置的 4.467 token/s 相比，ds4 decode 提升约 10.5 倍，
并解决了 Codex 需要数万 token 上下文的问题。

当前 ds4 服务监听 `0.0.0.0:8000`，LAN 地址在公开文档中记为 `<SERVER_IP>`。
本地补丁增加了 `--api-key-file` Bearer 鉴权；无 key 和错误 key 均返回
HTTP 401。服务由 `Restart=no` 的 transient user unit 托管，不开机启动、
退出后不自动重启。

## 二、权重、引擎、API 和客户端

这套服务可以分成四层：

```text
模型权重 -> 推理引擎 -> HTTP API -> Codex++ / Web UI / 自定义客户端
```

| 层 | 解决的问题 | 本次对应内容 |
|---|---|---|
| 模型权重 | 模型学到的参数、配置和 tokenizer | DeepSeek-V4-Flash-0731 Q2 imatrix GGUF |
| 推理引擎 | 如何切分、加载和计算权重 | antirez/ds4 |
| Serving API | 如何通过 HTTP 提交请求 | ds4 HTTP server |
| 客户端 | 如何组织消息、工具和显示结果 | Codex++、curl、Web UI |

换客户端不会改变模型参数，但会改变实际 prompt。用户只问一句“你是什么
模型”，Codex 仍可能附带数万 token 的系统提示、工具定义和项目上下文；
服务必须先完成全部 prefill 才能返回第一个 token。

## 三、模型为什么这么大

本机模型目录：

```text
<WORKDIR>/models/DeepSeek-V4-Flash-0731
```

模型有 48 个 safetensors 分片，权重合计 155.43 GiB。配置包含 256 个
routed experts，每个 token 由 router 选择其中 6 个，另有 shared expert。

### MoE 为什么只算少数专家，却仍要保存全部权重

MoE 的“稀疏”发生在每个 token 的计算选择上，不代表能提前删除未选专家：

```text
token A -> router -> 专家 3、18、42、...
token B -> router -> 专家 7、91、203、...
```

下一 token 会选择哪些专家，要等 router 算完才知道。为了保证任何输入都能
立即访问任意专家，完整推理必须能访问全部 256 个专家权重。它们可以分布在
多张 GPU，也可以部分放在 CPU，但不能只保留固定的 top-6。

因此 MoE 同时具备两个看似矛盾的特点：

- 每 token 实际计算量远小于把全部专家都计算一遍。
- 模型仍需要存储并能访问全部专家，内存容量压力依然很大。

若专家在 CPU，router 每步选中的专家数据要通过 PCIe 参与计算，decode
阶段每生成一个 token 都会重复这条路径，所以 CPU offload 会明显降速。

## 四、Tensor Parallel 和 Expert Parallel

### Tensor Parallel，TP=8

Tensor Parallelism 把同一个模型的张量切到 8 张卡。它不是 8 个独立模型，
也不是 8 个独立端口；一个请求会同时占用 8 张卡，并在每层通过 NCCL 同步。

本机 0-3 卡和 4-7 卡各自通过 `PIX` 连接，两组之间是跨 NUMA 的 `SYS`
路径，没有 NVLink。日志显示 TP collective 使用 `PYNCCL`。TP=8 虽然让
权重能分片装入，却比上游 TP=4 引入更多同步和较差的跨 NUMA 通信。

### Expert Parallel，EP

Expert Parallelism 按专家分配 GPU。本机开启后每卡持有 32/256 个专家，
服务可以启动，但平均 decode 为 4.344 token/s，比普通 TP 的 4.467
token/s 慢约 2.8%。原因是 token 到专家之间的 all-to-all 通信需要穿过
当前 PCIe/NUMA 拓扑，因此最终关闭 EP。

## 五、SM89 是什么意思

SM89 是 NVIDIA Compute Capability 8.9 的简称，描述 GPU 架构支持的
指令、Tensor Core 类型和可执行 kernel，不是显存大小，也不是 CUDA 版本。

```text
RTX 4090 -> Ada Lovelace -> Compute Capability 8.9 -> SM89
A100/A800 -> Ampere -> SM80
H100 -> Hopper -> SM90
```

一个为 SM90/SM100/SM120 编译的 sparse attention 或 FP4 kernel 不一定能
在 SM89 上执行。普通 vLLM 对 DeepSeek V4 的支持，也不自动等于这条模型
路径支持 RTX 4090。

本次 fork 的主要作用是给 SM89 补上 DeepSeek V4 所需的兼容路径：

```text
Sparse MLA attention -> FlashInfer SM89 sparse MLA JIT
Lightning Indexer   -> SM89 FP8/Triton/torch fallback
FP4 MoE experts     -> Marlin WNA16 fallback
mHC 等辅助算子       -> TileLang/Triton/torch 兼容路径
```

## 六、4090 支持 FP4 吗

准确说法是：RTX 4090 能存储和读取 FP4 量化权重，但没有 Blackwell 一类
更新 GPU 的原生 FP4/microscaling MMA 指令。

所以“能运行 FP4 模型”和“能用原生 FP4 Tensor Core 高速计算”是两件事。
当前 fork 让前者成立，但不能让 SM89 获得不存在的硬件指令。

模型日志中的两行给出了实际路径：

```text
DeepSeek V4 expert_dtype resolved to 'fp4'
Using 'MARLIN' Mxfp4 MoE backend
```

## 七、Marlin fallback 是什么

Marlin 是一套量化矩阵乘 CUDA kernel。这里的 fallback 表示当 GPU 没有
原生 FP4 MMA 时，使用兼容实现：

```text
FP4 打包权重 + block scale
        -> 读取和解包
        -> 按 scale 反量化
        -> 使用 SM89 支持的 FP16 等计算路径做矩阵乘
        -> 合并 router 选择的专家结果
```

它没有把磁盘上的模型永久展开成一份 FP16 权重；权重仍以紧凑 FP4 形式
存储，kernel 在计算中完成解包和 scale 应用。量化已经造成的信息损失也不
会因反量化而恢复。

Marlin 的价值是兼容性，代价是解包、反量化和额外数据处理，速度不能等同
于原生 FP4。仓库 README 也明确说明 Ada 上 MoE 使用 Marlin WNA16。

## 八、显存为何刚好卡住

8 张卡合计约 192 GB，模型约 155.43 GiB，看起来有余量，但“总显存”不能
当成一块完全共享的内存。每卡还必须同时容纳：

- 当前 TP 分片的权重。
- CUDA context 和 PyTorch allocator。
- NCCL/PYNCCL 通信 buffer。
- 激活值和临时 workspace。
- sparse MLA、Marlin、TileLang/Triton warmup 内存。
- KV cache。

全 GPU稳定配置中，vLLM 报告模型约 19.79 GiB/卡，而每卡实际容量约
23.54 GiB。其余空间要覆盖上面所有运行时开销。GMU 提高到 0.90-0.93
时，warmup 需要额外 512 MiB 或 Triton workspace，却只剩数百 MiB，因而
OOM；降低 GMU 又没有足够 KV cache。

### 为什么 4 x 48 GB 能跑，8 x 24 GB 不等价

两者总显存接近，但上游的 48 GB 改装 4090 每张卡有更大的本地连续余量，
TP 只有 4，通信参与者也更少。它能以 GMU 0.96-0.97 容纳模型、较大 KV、
warmup workspace、CUDA Graph 和 DSpark。本机 24 GB 卡在模型分片后已经
接近单卡边界，并且 TP=8 跨两个 NUMA 节点。

## 九、CPU offload 是什么

CPU offload 把部分权重留在主机内存，需要时通过 CPU/UVA/PCIe 路径提供给
GPU。它换来显存空间和长上下文能力，代价是带宽和延迟。

本机历史配置设置每 worker 20 GiB offload 上限，能启动 32K/65K 上下文，
但 decode 约 2 token/s。后续 vLLM 基线改为全 GPU 1K 配置，decode 提升到
4.467 token/s；当前 ds4 服务则不使用这套 offload 配置。

这两种配置代表不同取舍：

| 配置 | 上下文 | Decode | 用途 |
|---|---:|---:|---|
| 全 GPU | 1K | 4.467 tok/s | 短请求和功能实验 |
| CPU offload | 32K/65K | 约 2 tok/s | 验证长上下文，不适合交互 |

## 十、Prefill、Decode、TTFT 和上下文

一次生成分为两个阶段：

1. Prefill：处理全部输入 token，建立 KV cache。
2. Decode：逐 token 生成输出。

常用指标：

- TTFT：请求发出到第一个 token 的时间。
- Prefill TPS：输入处理速度。
- Decode TPS：稳定输出生成速度。
- TPOT：相邻输出 token 的平均间隔。

`MAX_MODEL_LEN` 是输入加输出的总预算，不是最大输出。例如 32K 配置中，
输入已经接近 32K 时几乎没有输出空间。客户端出现“requested 0 output
tokens”通常是 prompt 已经吃满服务端上下文窗口，不是模型拒绝回答。

Codex 的 idle timeout 主要发生在 TTFT 阶段：客户端在等待 SSE 首事件，
服务端仍在处理数万 token 的 prompt。首次请求的 JIT 预热完成后，后续短
请求会更快，但权重路径、长 prefill 和稳定 decode 不会因此发生数量级变化。

## 十一、DSpark 为什么没能加速本机

DSpark 属于推测解码：draft 模块一次提出多个候选 token，主模型批量验证，
接受率高时能显著加速 decode。它不减少长 prompt 的 prefill，而且 draft、
调度和 target verification 本身都有成本。

DSpark 在本次两套引擎中的包装不同：官方 safetensors 已包含 DSpark 参数，
vLLM/SGLang 不需要另一份模型；ds4 的 GGUF 路线需要单独下载 5.58 GiB 的
`DeepSeek-V4-Flash-DSpark-support-0731.gguf`。这个 support GGUF 不是可独立
对话的完整模型。

### vLLM 路线

本机实测 DSpark 强制使用 V2 Model Runner，使模型占用从 19.79 增加到
21.12 GiB/卡。多组 GMU 和 512/1K 上下文配置都在 KV 分配或 sparse MLA
warmup 阶段 OOM。

给 V2 runner 设置 2 GiB CPU offload 也没有生效：日志没有创建当前 V1
runner 使用的 `UVAOffloader`，模型仍占 21.12 GiB/卡。因此 vLLM DSpark
在这台 24 GB/卡机器上没有可用的显存窗口。

### ds4 路线

support GGUF 可以装入，本机峰值为 20,964 MiB。相同 8K context、256
greedy output 的 A/B 中，普通 Q2 为 47.53 tok/s，DSpark enabled 为
46.43 tok/s。统计为 `proposed=0`、`accepted_draft=0`，全部回退 target
decode，反而慢约 2.3%。

阻塞点是无 P2P 多卡环境中的 support MoE executor/peer placement，不是显存
容量。当前服务因此保持 DSpark 关闭。详细修复和原始统计见
[ds4 部署报告](reports/ds4-deployment-2026-08-16.md)第 11 节。

新的 SGLang SM89 fork 也暂不推荐 DSpark：两组 launch-meta 配对搜索均没有
找到快于生产配置的候选，而且逐 token parity 尚未完成。结论不是 DSpark
算法无效，而是“支持 DSpark”不等于在任意权重、引擎和 GPU 拓扑上打开就快。

## 十二、FP4、FP8、W8A8 和 GGUF

这些词不在同一层级，不能只按位数判断速度和大小。

### FP4 / FP8

官方 safetensors 是模型专用的混合结构：配置中的普通量化方法为 FP8，
MoE 专家为 FP4。具体体积还包含 scale、元数据、未量化张量和其他模块，
不能用“参数数 x 4 bit”简单推导整个仓库大小。

### W8A8

W8A8 通常表示 Weight 8 bit、Activation 8 bit。它可能适合具有原生 INT8/
FP8 kernel 的硬件，目的通常是提供规则的 8-bit 权重和激活计算、提升特定
推理栈吞吐，而不是保证文件比所有 FP4 模型小。

Eco-Tech 版本包含 74 个权重分片，权重约 305.45 GB，仓库存储约 314.72
GB，主要 tensor type 为 I8、I32、BF16、F32。8-bit 权重本来就可能比
4-bit 专家更大，而且还包含 scale 和非 8-bit 张量。它超过本机 8 卡总显存，
模型卡也没有声明 NVIDIA vLLM/SGLang/Transformers runtime 支持，组织信息
指向昇腾量化生态。因此不能把它直接替换到当前 `MODEL_DIR`，也不能用它
消除 CPU offload。

### GGUF

GGUF 是模型文件容器和元数据格式，不是单一量化算法，也不是推理引擎。
同一个模型可以有 Q2、Q3、Q4、Q8 等 GGUF。位数越低通常越小，但回答
质量、速度和 kernel 支持必须实测。

Unsloth GGUF 页面中的大致体积：

| 版本 | 大小 |
|---|---:|
| Q2_K_XL | 约 96.8 GB |
| Q3_K_XL | 约 128.2 GB |
| Q4_K_XL | 约 155.1 GB |
| Q8_K_XL | 约 161.9 GB |

Q2/Q3 理论上能在 192 GB 总显存中留下更多运行空间，但仍需目标引擎支持
准确的 tensor layout、多 GPU 分片和 DeepSeek V4 特殊结构。文件扩展名相同
不代表任意 GGUF runner 都能加载。

## 十三、参考仓库分别做什么

### 官方 ModelScope 模型

[deepseek-ai/DeepSeek-V4-Flash-0731](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash-0731/)

这是权重、配置、tokenizer/encoding 和许可证的来源，不是 HTTP 服务。下载
后仍需要推理引擎。

### vllm-deepseek-v4-sm89

[yhfgyyf/vllm-deepseek-v4-sm89](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89)

这是本次使用的 vLLM fork，不是另一个模型。它提供 SM89 sparse MLA、
FlashInfer 修复、FP8 fallback、Marlin MoE 路径、CUDA 13/Python 3.12 wheel
以及 OpenAI-compatible server。

选择它是因为目标硬件明确是 SM89、已有官方 safetensors，并且需要标准
OpenAI 风格 API。它用最少格式转换跑通了正确性，但性能受本机硬件边界限制。

### antirez/ds4

[antirez/ds4 / DwarfStar](https://github.com/antirez/ds4)

这是面向 DeepSeek V4 Flash 的另一套原生推理引擎，主要使用其明确兼容的
GGUF，支持 CUDA、多 GPU、专家缓存、micro-batching 和实验性 DSpark。
它不是 vLLM 插件。切换 ds4 需要重新验证 API、工具调用、模型质量和多 GPU
性能。它已成为当前实验服务采用的路线。

### xltzsoft/deepseek-v4-sm89，SGLang 分支

[xltzsoft/deepseek-v4-sm89](https://github.com/xltzsoft/deepseek-v4-sm89)

这是基于 SGLang 的 8 x RTX 4090 24 GB / SM89 实验分支，使用官方 MXFP4
权重，并补充 Marlin MoE、sparse MLA Triton kernel、TP8 原生 head layout、
FP8 Indexer fallback、CUDA Graph 和 DSpark 相关修复。

截至 2026-08-16，仓库公开的固定 8K 配置结果为 C1 decode 54.75 tok/s、
C1 端到端 50.36 tok/s、C8 端到端聚合 318.59 tok/s。这些不是本机实测，
不能和 ds4 的另一套 benchmark 直接比较。仓库明确标注为性能研究快照，当前
推荐 target-only + decode full CUDA Graph；DSpark 尚未通过严格逐 token
一致性验证。

### Unsloth

[DeepSeek V4 文档](https://unsloth.ai/docs/models/deepseek-v4)

[DeepSeek-V4-Flash-0731-GGUF](https://www.modelscope.cn/models/unsloth/DeepSeek-V4-Flash-0731-GGUF)

这些资料用于理解和获取不同 GGUF 量化。是否兼容 ds4 要以 ds4 当前 loader
支持和实际启动为准。

### Eco-Tech W8A8

[DeepSeek-V4-Flash-0731-w8a8](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8)

这是社区 W8A8 转换，体积和目标 runtime 都不适合当前 8 x 24 GB NVIDIA
路线，不建议投入。

## 十四、上游更新与性能判断

截至 2026-08-16，仓库 `main` 最新 commit 是 `2b4f880`，比当前 release
前进 7 个 commit。除文档外有两项运行时修复：大型 TP=8 统一 KV pool 中
paged MQA block 地址的 int32 溢出，以及 Triton per-shape kernel cache
持续增长。前者防止长上下文/prefix caching 在超过约 2K blocks 后产生
illegal memory access，后者改善多种请求形状长期运行时的缓存稳定性。

这两项都没有改变 Marlin、TP=8 通信、24 GB 显存或 DSpark V2 runner 的
性能边界。最新 release 仍是当前安装的
`v0.23.1rc1.dev904-g8e321cc4f-cu130-sm89`，没有包含新 commit 的 wheel。
当前 1K 基线不会达到大型 KV pool 溢出条件，因此暂不源码重编；等正式
wheel 后应升级，尤其是在恢复长上下文或 prefix caching 之前。

远端还存在名称为 `dev1018-g8aba6ae7e` 的 tag，但它创建于 2026-07-10，
是 2026-08-02 `dev904-g8e321cc4f` 的祖先；dev904 比它多 22 个提交。
判断版本新旧需要同时查看 commit ancestry、日期和 release 页面，不能只看
dev 后面的数字。

上游 82 token/s 和 DSpark 286-344 token/s 来自 4 x 48 GB 4090、TP=4、
GMU 0.96-0.97、CUDA Graph/DSpark 可用的环境。本机 8 x 24 GB、TP=8、跨
NUMA PCIe、eager、DSpark OOM，不是同一测试条件。

## 十五、ds4 路线的实测结论

原计划中的 `ds4 + 兼容 GGUF Q2` 已经完成，不再只是建议。使用文件为：

```text
DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
80.76 GiB，SHA-256 ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0
```

这份 Q2 不是“全模型统一 2 bit”。约 44.34 GiB 是 routed expert 的
`IQ2_XXS` gate/up，约 28.22 GiB 是 `Q2_K` down，其他投影、共享专家、
输出和控制张量保留 Q8/F16 等较高精度。它正是 ds4 明确支持和验证的布局。

本机所有 4090 之间都没有 CUDA P2P。ds4 能用 pinned-host bounce 完成跨卡
激活复制，但当前 commit 的 owned-expert prefill 入口错误地把 P2P 当成
硬条件，在调用已有 bounce 实现前直接返回失败。删除这个前置拒绝后，8 卡
Q2 prefill、decode 和 HTTP 服务全部通过。补丁和风险记录见 ds4 部署报告。

本机实测：

| 场景 | 结果 |
|---|---:|
| 2,048 input benchmark prefill | 807.54 tok/s |
| 128-token decode | 46.02 tok/s overall |
| 127-token steady decode | 46.75 tok/s |
| 38,717 input cold request | 31.57 s，约 1,230 tok/s |
| 同请求缓存重放 | 30,720 cached，8.04 s |
| 131K 静置显存 | 13.35-15.02 GiB/卡 |
| 长 prefill 采样峰值 | 约 18.02 GiB/卡 |

后续实验集中在质量和多会话：

1. 用代码、中文长文和工具调用集比较 Q2 与官方权重的质量。
2. 验证 Codex++ 的多轮工具结果 continuation，而不只是单次函数调用。
3. 评估 Q2 下 `--batched-session` 的吞吐与回退路径，再决定是否多会话。
4. 只有需要更高质量时，才下载约 98 GiB 的 Q2/Q4 混合量化；当前无需 Q4。

## 十六、服务操作

历史 vLLM 配置仍位于：

```text
<WORKDIR>/deepseek-v4-flash-8x4090/deploy/vllm/.env
```

它目前保持停止。当前 ds4 手动启动和停止：

```bash
cd <WORKDIR>/ds4
./run-server.sh
systemctl --user status ds4-server-test.service
./stop-server.sh
```

服务地址和模型名：

```text
Base URL: http://<SERVER_IP>:8000/v1
Model: deepseek-v4-flash
API key file: <WORKDIR>/ds4/runtime/api-key
```

已验证 `/v1/chat/completions`、`/v1/responses`、`/v1/models`、Responses
SSE 和 `tool_choice:auto` 函数调用。`tool_choice:required` 当前不支持，会
返回 HTTP 400。当前本地补丁要求 `Authorization: Bearer <key>`；无 key、
错误 key、正确 key 的状态码已分别验证为 401、401、200。

```bash
./ds4-bench --cuda --cuda-tensor-parallel \
  --gpu-vram auto --gpu-devices 0,2,4,6,1,3,5,7 \
  --prompt-file README.md --ctx-start 2048 --ctx-max 2048 \
  --ctx-alloc 4096 --gen-tokens 128
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

## 十七、当前结论

vLLM fork 证明官方 FP8/FP4 权重能在 SM89 上运行，但本机只有 4.47
token/s；切换到 ds4 专用 Q2 GGUF 后，8 x 24 GB 已达到约 46.75 token/s
稳定 decode、约 1,200 token/s 长 prefill 和 131K 上下文。后续需要继续验证
Q2 质量、多会话能力和局域网鉴权。
