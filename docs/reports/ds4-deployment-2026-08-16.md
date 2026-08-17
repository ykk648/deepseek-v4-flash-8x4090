# antirez/ds4 部署与性能报告

日期：2026-08-16

## 1. 结论

`antirez/ds4` 已在本机 8 x RTX 4090 24 GB 上跑通 DeepSeek-V4-Flash-0731
专用 Q2 imatrix GGUF。模型全部常驻 GPU，不使用 CPU weight offload，并已
提供 OpenAI Chat Completions、Responses SSE 和工具调用兼容接口。

核心结果：

| 指标 | 实测 |
|---|---:|
| 上下文 | 131,072 tokens |
| 2,048-token prefill | 807.54 tok/s |
| 128-token generation | 46.02 tok/s |
| steady decode | 46.75 tok/s |
| 38,717-token cold request | 31.57 s |
| 同请求缓存重放 | 8.04 s |

历史 vLLM 全 GPU 配置只有 1K 上下文和 4.467 tok/s。ds4 在本机将稳定
decode 提高约 10.5 倍，同时把上下文扩大到 131K，已经达到可交互使用水平。

## 2. 版本与文件

```text
源码：<WORKDIR>/ds4
仓库：https://github.com/antirez/ds4
commit：84cc882352757baf628a1776badf7cc54d584e28
CUDA：/usr/local/cuda-13.3
目标架构：sm_89
```

模型：

```text
<WORKDIR>/ds4/gguf/
DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
```

| 项目 | 值 |
|---|---:|
| 文件大小 | 86,720,111,488 bytes / 80.76 GiB |
| 逻辑参数 | 284.33B |
| IQ2_XXS | 44.34 GiB |
| Q2_K | 28.22 GiB |
| Q8_0 | 6.15 GiB |
| F16 | 2.04 GiB |
| SHA-256 | `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0` |

来源仓库为 `antirez/deepseek-v4-gguf`。实际下载使用
`https://hf-mirror.com` 直连、无代理、16 连接，平均约 26 MiB/s。以后从
Hugging Face 下载仍应设置 `HF_ENDPOINT=https://hf-mirror.com` 或直接使用
mirror URL，不给下载命令设置 `7890/7897` 代理。

## 3. 编译

```bash
cd <WORKDIR>/ds4
CUDA_HOME=/usr/local/cuda-13.3 CUDA_ARCH=sm_89 \
  make -j16 cuda CUDA_ARCH=sm_89
```

生成 `ds4`、`ds4-server`、`ds4-bench`、`ds4-eval` 和 `ds4-agent`。
链接结果使用 CUDA 13.3 的 `libcudart.so.13`、`libcublas.so.13`，并已用
`cuobjdump` 确认包含 `sm_89` cubin。基础参数、多 GPU placement、跨卡复制、
server 和 agent 单元测试均通过。

## 4. GPU 拓扑与兼容补丁

运行顺序使用：

```text
0,2,4,6,1,3,5,7
```

ds4 将前四个 logical tier 放 transformer layer，后四个 tier 作为配对的
expert owner。每卡驻留约 10.03-11.08 GiB 选择性权重，没有 CPU spill。

本机所有 4090 卡之间 CUDA P2P 都不可用，ds4 的矩阵显示全部 `BOUNCE`。
跨卡数据通过 pinned host memory 完成 D2H/H2D。这个路径功能测试通过，但
不应拿来与 README 的 8 x L40S P2P 环境直接比较。

原始 commit 在 `metal_graph_encode_mixed_routed_rows()` 中还有一处矛盾：
底层 `ds4_gpu_tensor_copy_xdev_default()` 支持 host bounce，但上层 owned
prefill 在进入复制前强制要求 `g_gpu_peer_ok`。因此第 0 层 router 后直接
失败：

```text
ds4: gpu layer 0 ffn batch encode failed
ds4: gpu whole-prefill layer 0 encode failed
```

本地补丁删除这个 P2P-only 前置拒绝，并在 bounce 时打印一次性能警告。补丁
仅放开已有 fallback，不伪造 P2P 能力。修复后 Q2 prefill、decode、长上下文
和 HTTP 服务全部通过。修改位于：

```text
<WORKDIR>/ds4/ds4.c
```

## 5. 性能

专用基准命令使用 2,048 input + 128 greedy decode：

```text
prefill_tps=807.54
gen_tps=46.02
gen_first_ms=52.867
gen_steady_tokens=127
gen_steady_tps=46.75
```

短中文 CLI 请求得到 44.09 tok/s prefill、44.61 tok/s generation，回答内容
正常。HTTP chat 请求生成 37 token，总耗时 1.31 秒，服务日志记录 decode
45.40 tok/s。

长 Responses SSE 请求实际输入 38,717 token：

| 运行 | Cached | Re-prefill | 总耗时 | 日志平均 prefill |
|---|---:|---:|---:|---:|
| 首次 | 0 | 38,717 | 31.57 s | 1,230.90 tok/s |
| 原样重放 | 30,720 | 7,997 | 8.04 s | 1,178.25 tok/s |

首次 SSE keepalive 在 5.10 秒到达，客户端不会像旧 vLLM 服务一样等待数十
分钟后触发 SSE idle timeout。磁盘 KV 每 10K token 保存边界，所以重放命中
30,720 而非完整 38,717；这个取舍减少存储频率，同时保留大部分前缀收益。

## 6. 上下文与显存

`--ctx 131072` 的 context buffers 为 2,426.72 MiB。服务静置显存：

```text
GPU0..7: 13,350-15,016 MiB / 24,564 MiB
```

38K 长 prefill 期间采样峰值约 18,018 MiB，仍有明显余量。131K 是当前
Codex 测试的保守配置，不是硬件绝对极限；README 认为 100K-300K 对 Q2
通常合理。继续增大 context 会增加 compressed indexer/KV 内存，应按实际
请求长度决定，不应只追求标称最大值。

## 7. API 验证

已验证：

- `GET /v1/models`，报告 `context_length=131072`。
- `POST /v1/chat/completions`，非流式 HTTP 200。
- `POST /v1/responses`，完整 SSE event lifecycle。
- Responses `tool_choice:auto`，返回标准 `function_call` 和 JSON arguments。
- 长输入期间 SSE keepalive、最终 `response.completed` 和 usage。

兼容边界：`tool_choice:"required"` 当前返回 HTTP 400
`tool_choice=required not supported`。Codex 常用的 `auto` 已通过，但完整的
多轮 function result continuation 仍应在真实 Codex++ 会话中验收。

## 8. 当前服务操作

服务使用 `Restart=no` 的 transient user unit，不 enable、不随开机启动、
退出后不自动重启：

```bash
cd <WORKDIR>/ds4
./run-server.sh
systemctl --user status ds4-server-test.service
tail -f runtime/ds4-server.log
./stop-server.sh
```

当前地址：

```text
Base URL: http://<SERVER_IP>:8000/v1
Model: deepseek-v4-flash
API key file: <WORKDIR>/ds4/runtime/api-key
```

ds4 原始 commit 不校验 Authorization。本地补丁增加 `--api-key-file`，按
`Authorization: Bearer <key>` 校验后才允许访问包括 `/v1/models` 在内的
API。随机 256-bit key 文件权限为 `0600`，不出现在进程命令行或仓库文档。
无 key、错误 key、正确 key 已分别验证为 HTTP 401、401、200。服务现在
可以安全地监听 `0.0.0.0:8000` 供局域网客户端使用。

日志和磁盘 KV：

```text
<WORKDIR>/ds4/runtime/ds4-server.log
<WORKDIR>/ds4/runtime/kv-cache
```

磁盘 KV 预算为 8 GiB。

## 9. 后续方向

1. 用真实 Codex++ 完成多轮读文件、写文件、shell 工具调用和 tool result 回传。
2. 建立 Q2 对官方权重的代码生成、中文和推理质量对照集。
3. 需要并发时再测 `--batched-session`；Q2 某些 grouped shape 可能走 exact
   fallback，单请求结果不能外推为多用户聚合吞吐。
4. 若 Q2 质量不足，优先尝试约 98 GiB 的 `ds4f-q2-q4`，而不是直接下载
   153-156 GiB 的 Q4/MXFP4。
5. DSpark support GGUF 已完成 A/B；当前多卡 CUDA stage-chain 没有生成
   有效 draft，不能加速，保持关闭。详见第 11 节。

## 10. 多轮长对话质量测试

使用 `temperature=0` 模拟了长文档审计、代码协作和 Responses 工具调用。
原始结果保存在：

- [长对话结果](../../results/quality/long-conversation-2026-08-16.json)
- [no-thinking 代码恢复结果](../../results/quality/coding-recovery-nothink-2026-08-16.json)
- [thinking 代码恢复结果](../../results/quality/coding-recovery-thinking-2026-08-16.json)

### 长文档记忆

首轮 prompt 为 27,755 tokens，包含 520 条高度相似记录和三个分散的特别
审计条目。模型准确提取全部 10 个检查字段。后续两轮分别完成跨记录算术，
并拒绝用户故意提供的错误协议、端口、保留天数和副本数。

| 轮次 | Prompt | Cached | Output | 耗时 | 结果 |
|---|---:|---:|---:|---:|---|
| 提取 | 27,755 | 0 | 192 | 27.84 s | 全部精确字段正确 |
| 计算 | 28,001 | 27,947 | 24 | 0.87 s | `28546/K9/9F2C` 正确 |
| 抗错误纠正 | 28,079 | 28,025 | 41 | 1.32 s | 三项错误均被拒绝 |

说明 KV 前缀复用在真实多轮对话中有效，第二、三轮只写入 54 个新 prompt
token，延迟降到约 1 秒。

### 代码协作

10,415-token C 项目中埋入整数溢出、悬空指针、NUL 终止和 retry 边界四个
问题。no-thinking 首轮准确找全四项，触发条件和后果基本正确。但第二轮要求
最小 unified diff 时，模型复制了大量未修改 `metric_*` 行，达到 900-token
上限仍未输出修复；追加明确纠错后仍重复同一错误。这是实际失败，不是评分
脚本误判。

对同一 12,563-token 完整历史改用 `reasoning_effort=high` 后，模型：

- 不再输出任何 `metric_*` 行。
- 给出三个可实施的最小 patch hunk。
- 正确指出 `cache_lookup` 在“ABI 冻结、hot path 不分配、解锁后可并发失效”
  同时成立时无法安全返回裸指针，必须放宽契约。
- 修正 `max_retries=0` 为完全不调用 `try_connect`。

high-reasoning 轮命中 12,402/12,563 prompt tokens，生成 1,102 tokens，耗时
30.15 秒，五项自动检查全部通过。因此复杂代码编辑不建议使用 no-thinking；
应让 Codex++ 请求 high reasoning，并设置足够输出上限。当前测试不能证明失败
一定由 Q2 量化造成，因为没有对官方权重做同 prompt A/B。

### Responses 工具 continuation

模型首先正确返回：

```text
lookup_incident({"incident_id":"INC-9042"})
```

随后以完整 Responses history 提交 `function_call_output`。第二轮命中 419
个缓存 token，模型没有重复调用工具，准确保留 `SEV-2`、epoch wraparound、
两个受影响集群、64-bit 缓解措施、解决时间和 owner。函数调用、参数、工具
结果 continuation、最终消息五项检查全部通过。

### 质量判断

当前 Q2 服务适合长文档提取、事实保持、普通对话和 Codex 风格工具调用。
复杂代码修改可以使用，但必须开启 reasoning；no-thinking 更适合短问答，不应
作为 coding agent 的默认模式。仍需用真实仓库任务对官方权重做质量 A/B，
才能量化 Q2 对代码正确率的影响。

## 11. DSpark support GGUF A/B

下载文件：

```text
<WORKDIR>/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf
大小：5,989,114,272 bytes / 5.58 GiB
SHA-256：7e319924541db3f7a163ed7e11d7532a70d48228ab59d36cb81e1d4511885360
```

文件从 `hf-mirror.com` 直连下载，未使用代理；16 路下载平均约 35 MiB/s。
主模型仍为 80.76 GiB 的 0731 Q2 imatrix GGUF。

### 多卡兼容修复

上游 commit `84cc8823` 的 DSpark CUDA 路径默认依赖跨卡可访问张量。本机全部
GPU pair 均无 P2P，原始代码在 prefill 第 41 层触发 CUDA illegal memory
access。为完成诊断，本地增加三个 DSpark 限定修复：

1. capture、draft 和 DSpark KV 张量按 `DS4_DSPARK_EXEC_TIER` 分配，不再
   固定到 logical tier 0。
2. support model 的 Q8 FP16/FP32 转换缓存查询应用与通用权重解析器相同的
   offset bias。
3. 长于 6 行的 stage-0 动态 packing buffer 分配到 executor tier。

测试固定 `DS4_DSPARK_EXEC_TIER=3`，使 40-42 层 hidden-state capture 与
DSpark executor 位于同一卡。未降低默认 4 GiB Q8 cache 安全余量，也未使用
`CUDA_LAUNCH_BLOCKING`。

### A/B 结果

参数为 8 卡 TP、8K context、256 greedy output、`temperature=0`、no-thinking，
使用同一个 C 语言 LRU cache 代码生成提示。

| 模式 | Generation | 进程总耗时 | 结果 |
|---|---:|---:|---|
| 普通 Q2 | 47.53 tok/s | 32.82 s | 256 token 完成 |
| DSpark strict | 46.62 tok/s | 33.65 s | 与普通输出逐字一致 |
| DSpark enabled | 46.43 tok/s | 33.79 s | 完成，但没有有效 draft |

三份输出 SHA-256 均为：

```text
a0a337cc2ab7f40cd634a2cc873315e090a7a0196d3984c0ea28c3e28d4d6a0a
```

DSpark 统计为 `cycles=247`、`proposed=0`、`accepted_draft=0`、
`scheduler_skips=188`、`tail_skips=8`、`net_saved=-44.539 ms`。support MoE
stage-chain 将带 bias 的 expert 权重请求路由到 logical tier 7，但权重只安装在
executor tier 3，产生重复 selective-cache miss；调度器随后回退普通 target
decode，所以结果正确但没有加速。

成功运行时最高显存为 physical GPU 6（logical tier 3）的 20,964 MiB，说明
5.58 GiB support GGUF 可以装入 24 GB 卡，当前阻塞点是无 P2P 多卡调度和
权重 placement，不是容量。

结论：当前服务不启用 DSpark。46.43 tok/s 比普通 47.53 tok/s 慢约 2.3%，
继续修复需要调整 support MoE 的 executor/peer placement 和无 P2P fallback，
已经超出一次低成本加速实验的合理范围。

## 12. 与 SGLang SM89 新分支的关系

本轮还核对了 [xltzsoft/deepseek-v4-sm89](https://github.com/xltzsoft/deepseek-v4-sm89)。
它不是 ds4 的后端，而是基于 SGLang 的独立 SM89 实验分支；使用官方 MXFP4
safetensors，目标同样是单机 8 x RTX 4090 24 GB、TP=8。

核对时版本：

```text
commit 94a89e65736b45706603a78d3150b0151b135162
perf: publish SM89 paired tuning and lifecycle fixes
```

仓库补充了 SM89 Marlin MoE、sparse MLA Triton kernel、Ada prefill geometry、
TP8 原生 query-head layout、FP8 Indexer/MHC fallback、decode full CUDA Graph，
并修复了部分 DSpark shared-expert、scheduler 和压缩状态生命周期问题。

其 README 公开数据为 context 8192、256 input、128 output、greedy、
`ignore_eos=true`：

| 场景 | 仓库公开结果 |
|---|---:|
| C1 decode | 54.75 tok/s |
| C1 端到端 | 50.36 tok/s |
| C1 TTFT | 0.222 s |
| C8 scheduler 输出 | 382.8-383.8 tok/s |
| C8 decode-window 聚合 | 374.58 tok/s |
| C8 端到端聚合 | 318.59 tok/s |

以上为第三方仓库公开结果，不是本机复现。它与本报告 46.75 tok/s 的输入长度、
权重精度、CUDA Graph 和计时口径不同，不能直接宣称快 17%。需要在本机用同一
256/128 请求做 C1 A/B 后才能比较。

该分支当前也不建议启用 DSpark：两组 schedule launch-meta 配对结果分别为
0.991327x 和 0.988452x，均未产生 winner；greedy 输出尚未完成 target reference
逐 token 一致性验证。其推荐模式仍是 target-only + decode full CUDA Graph。

SGLang 路线的潜在优势是成熟 scheduler、CUDA Graph 和多请求吞吐。若后续目标
是 8 个并发用户，而不是单会话 131K context，它比继续修 ds4 DSpark 更值得做
独立实验。

## 13. 最终部署判断

当前 8 x RTX 4090 24 GB 的最佳已验证配置仍是 ds4 Q2 target-only：

```text
80.76 GiB 0731 Q2 imatrix GGUF
8 卡 CUDA TP
context 131072
DSpark off
46.75 tok/s steady decode
```

vLLM 官方权重路线保留为质量参考，不再作为交互服务；SGLang SM89 分支列为下一
个独立候选，不把它的公开 benchmark 冒充本机成绩。任何后续优化都应先固定请求、
并发、输出 token、采样和计时边界，再比较速度与逐 token 正确性。

## 14. 产能和官方 API 等价价值

按 steady decode 46.75 output token/s 计算，理论产能为每小时 168,300、每天
4,039,200、30 天 121,176,000 output tokens。生产环境考虑 prefill、请求切换、
缓存未命中、排队和空闲后，按理论值 80%-90% 估算为每月 96,940,800-
109,058,400 output tokens，即约 9,694 万-1.091 亿。

DeepSeek 官方价格页在 2026-08-17 显示，`deepseek-v4-flash` 每百万 tokens 的
空闲/高峰价格分别为：缓存命中输入 ¥0.05/¥0.10，缓存未命中输入 ¥1.50/¥3.00，
输出 ¥4.50/¥9.00。高峰为北京时间每天 7 小时，若请求全天均匀分布，输出加权
平均价为 ¥5.8125/百万 tokens。

| 场景 | 理论满载 30 天 | 80%-90% 利用率 |
|---|---:|---:|
| 只计算输出 | ¥704.34 | ¥563-¥634 |
| 输入:输出=1:1，输入全命中 | ¥712.16 | ¥570-¥641 |
| 输入:输出=1:1，输入全未命中 | ¥939.11 | ¥751-¥845 |

以上使用 24 小时均匀分布的官方 API 加权价格。全为空闲时段时，理论满载输出价值
为 ¥545.29/月；全为高峰时段时为 ¥1,090.58/月。这里衡量的是官方 API 替代成本，
不是收入，也没有假设本地 Q2 与官方服务在质量、上下文、并发和 SLA 上完全等价。

output-only 的 80%-90% 利用率只相当于每小时 ¥0.78-¥0.88。以 ¥0.8/kWh 电价
举例，电费盈亏对应的整机平均功耗只有约 0.98-1.10 kW；超过该功耗时，仅电费就
高于 output-only API 替代成本，尚未计算硬件折旧和运维。因此本地部署的核心价值
应定位为隐私、内网和离线可用、自主控制、无外部限流，而不是低价 API 套利。

价格来源：[DeepSeek 官方模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)。
