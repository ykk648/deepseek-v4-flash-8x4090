import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = os.environ.get("DS4_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
RESULT_PATH = Path(
    os.environ.get("DS4_RESULT_PATH", "/tmp/ds4_long_conversation_results.json")
)


def load_api_key():
    key = os.environ.get("DS4_API_KEY", "").strip()
    key_file = os.environ.get("DS4_API_KEY_FILE", "").strip()
    if key or not key_file:
        return key
    return Path(key_file).expanduser().read_text(encoding="utf-8").strip()


API_KEY = load_api_key()


def post(endpoint, payload, timeout=300):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    request = urllib.request.Request(
        f"{BASE_URL}/{endpoint}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    return data, status, time.perf_counter() - started


def chat_turn(messages, prompt, max_tokens=512):
    messages.append({"role": "user", "content": prompt})
    data, status, elapsed = post(
        "chat/completions",
        {
            "model": "deepseek-v4-flash",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "thinking": {"type": "disabled"},
        },
    )
    answer = data["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": answer})
    return {
        "status": status,
        "elapsed_s": round(elapsed, 3),
        "usage": data.get("usage", {}),
        "answer": answer,
    }


def make_archive():
    records = []
    for i in range(520):
        code = f"C-{i:04d}"
        region = f"zone-{i % 19:02d}"
        owner = f"team-{(i * 7) % 31:02d}"
        note = (
            f"记录 {i:04d}：组件 {code} 位于 {region}，维护组 {owner}；"
            f"常规容量 {(i * 13) % 997 + 20}，巡检周期 {i % 29 + 1} 天。"
            "该条目的普通描述只用于容量规划，不得与异常处置字段混用。"
        )
        if i == 117:
            note += (
                " 特别审计字段：代号 Aster-117，监听端口 28473，传输协议 QUIC，"
                "校验串 AX-77-9F2C，负责人 顾岚。"
            )
        elif i == 392:
            note += (
                " 特别审计字段：代号 Ember-392，日志保留 73 天，负责人 林桥，"
                "故障切换目标 north-vault-6，审批号 AP-8841。"
            )
        elif i == 481:
            note += (
                " 特别审计字段：代号 Quartz-481，部署窗口 Tuesday 03:40 UTC，"
                "回滚代码 ROLLBACK-K9，最低副本数 11，密钥轮换周期 46 天。"
            )
        records.append(note)
    return "\n".join(records)


def document_conversation():
    archive = make_archive()
    messages = [
        {
            "role": "system",
            "content": (
                "你是审计助手。所有答案必须只依据用户提供的档案；"
                "精确保留数字、大小写和标点，不得用常识改写。"
            ),
        }
    ]
    turns = []
    turns.append(
        chat_turn(
            messages,
            "下面是完整档案。提取 Aster-117、Ember-392、Quartz-481 的特别审计字段，"
            "只输出一个 JSON 对象，不要解释。\n\n" + archive,
            420,
        )
    )
    turns.append(
        chat_turn(
            messages,
            "不要重新概述档案。计算 Aster 端口加 Ember 保留天数；再给出 Quartz 回滚代码，"
            "以及 Aster 校验串最后四个字符。格式严格为：sum=...; rollback=...; suffix=...",
            96,
        )
    )
    turns.append(
        chat_turn(
            messages,
            "同事声称 Aster 使用 TCP/28437，Ember 只保留 37 天，而且 Quartz 最低副本数是 7。"
            "逐项判断这些说法，并引用档案中的准确值。控制在 120 字内。",
            180,
        )
    )
    text = "\n".join(turn["answer"] for turn in turns)
    checks = {
        "turn1_exact_facts": all(
            token in turns[0]["answer"]
            for token in [
                "28473",
                "QUIC",
                "AX-77-9F2C",
                "73",
                "north-vault-6",
                "AP-8841",
                "03:40",
                "ROLLBACK-K9",
                "11",
                "46",
            ]
        ),
        "turn2_arithmetic": all(
            token in turns[1]["answer"]
            for token in ["28546", "ROLLBACK-K9", "9F2C"]
        ),
        "turn3_resists_false_correction": all(
            token in turns[2]["answer"] for token in ["QUIC", "28473", "73", "11"]
        ),
        "all_answers_nonempty": bool(text.strip()),
    }
    return {"archive_chars": len(archive), "turns": turns, "checks": checks}


def make_c_project():
    lines = [
        "/* Public function signatures are frozen. Hot-path functions may not allocate. */",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "#include <string.h>",
        "typedef struct cache cache;",
        "typedef struct entry { const char *value; } entry;",
        "void lock_cache(cache *c); void unlock_cache(cache *c);",
        "entry *find_entry(cache *c, const char *key);",
        "void consume_bytes(const uint8_t *p, uint32_t n);",
    ]
    for i in range(180):
        lines.append(
            f"static uint32_t metric_{i:03d}(uint32_t x) "
            f"{{ return (x ^ {i * 17 + 3}u) + {i % 11}u; }}"
        )
    lines.extend(
        [
            "int decode_packet(const uint8_t *p, size_t available, uint32_t payload_len) {",
            "    if (available < payload_len + 4u) return -1;",
            "    consume_bytes(p + 4, payload_len);",
            "    return 0;",
            "}",
            "const char *cache_lookup(cache *c, const char *key) {",
            "    lock_cache(c);",
            "    entry *e = find_entry(c, key);",
            "    unlock_cache(c);",
            "    return e ? e->value : NULL;",
            "}",
            "int copy_label(char dst[16], const char *src) {",
            "    size_t n = strlen(src);",
            "    if (n > 16) n = 16;",
            "    memcpy(dst, src, n);",
            "    return 0;",
            "}",
            "int connect_with_retry(int max_retries) {",
            "    for (int attempt = 0; attempt <= max_retries; attempt++) {",
            "        if (try_connect() == 0) return 0;",
            "    }",
            "    return -1;",
            "}",
        ]
    )
    for i in range(180, 360):
        lines.append(
            f"static uint32_t metric_{i:03d}(uint32_t x) "
            f"{{ return (x * {i % 23 + 1}u) ^ {i * 19 + 5}u; }}"
        )
    return "\n".join(lines)


def coding_conversation():
    project = make_c_project()
    messages = [
        {
            "role": "system",
            "content": (
                "你是资深 C/CUDA 代码审查者。优先发现可复现的正确性、安全和并发问题，"
                "不要把风格偏好冒充 bug。"
            ),
        }
    ]
    turns = []
    turns.append(
        chat_turn(
            messages,
            "审查下面单文件服务。项目约束：公开函数签名冻结；hot path 禁止堆分配；cache 可被"
            "其他线程并发失效。请只报告四个最确定的问题，每项写函数名、触发条件、后果和修复原则。"
            "\n\n```c\n" + project + "\n```",
            700,
        )
    )
    turns.append(
        chat_turn(
            messages,
            "保持最初两个项目约束不变。给这四个函数写最小 unified diff。"
            "如果某问题在不改 ABI、也不分配内存的条件下无法完全修复，明确说明并给出最安全的"
            "契约调整，不要假装返回的裸指针能自动获得生命周期。",
            900,
        )
    )
    turns.append(
        chat_turn(
            messages,
            "现在只列回归测试：每个函数两个测试，其中至少包含 uint32 边界、16 字节标签、"
            "并发失效和 max_retries=0。说明期望结果，控制在 500 字内。",
            600,
        )
    )
    first = turns[0]["answer"]
    second = turns[1]["answer"]
    third = turns[2]["answer"]
    checks = {
        "finds_all_four_functions": all(
            name in first
            for name in ["decode_packet", "cache_lookup", "copy_label", "connect_with_retry"]
        ),
        "recognizes_overflow": any(word in first for word in ["溢出", "overflow", "UINT32"]),
        "recognizes_lifetime": any(word in first for word in ["生命周期", "悬空", "use-after-free"]),
        "keeps_no_allocation_constraint": any(word in second for word in ["不能完全", "契约", "调用方", "生命周期"]),
        "tests_required_boundaries": all(
            token in third for token in ["max_retries=0", "16", "并发"]
        ),
    }
    return {"project_chars": len(project), "turns": turns, "checks": checks}


def responses_tool_conversation():
    tool = {
        "type": "function",
        "name": "lookup_incident",
        "description": "按事件编号查询权威故障记录",
        "parameters": {
            "type": "object",
            "properties": {"incident_id": {"type": "string"}},
            "required": ["incident_id"],
        },
    }
    user_item = {
        "role": "user",
        "content": "查询事件 INC-9042，并根据工具结果给值班经理写三点摘要。不要猜测记录内容。",
    }
    first_data, first_status, first_elapsed = post(
        "responses",
        {
            "model": "deepseek-v4-flash",
            "input": [user_item],
            "tools": [tool],
            "tool_choice": "auto",
            "max_output_tokens": 180,
            "temperature": 0,
            "reasoning": {"effort": "none"},
        },
    )
    calls = [item for item in first_data.get("output", []) if item.get("type") == "function_call"]
    if not calls:
        return {
            "first": first_data,
            "checks": {"function_call_emitted": False, "continuation_completed": False},
        }
    call = calls[0]
    tool_output = {
        "incident_id": "INC-9042",
        "severity": "SEV-2",
        "root_cause": "allocator epoch wraparound",
        "affected_clusters": ["cn-north-7", "cn-north-9"],
        "mitigation": "pin epoch to 64-bit and recycle workers",
        "resolved_at": "2026-08-14T22:17:00Z",
        "owner": "runtime-infra",
    }
    replay_call = {
        "type": "function_call",
        "id": call.get("id"),
        "call_id": call["call_id"],
        "name": call["name"],
        "arguments": call["arguments"],
        "status": "completed",
    }
    output_item = {
        "type": "function_call_output",
        "call_id": call["call_id"],
        "output": json.dumps(tool_output, ensure_ascii=False),
    }
    second_data, second_status, second_elapsed = post(
        "responses",
        {
            "model": "deepseek-v4-flash",
            "input": [user_item, replay_call, output_item],
            "tools": [tool],
            "tool_choice": "auto",
            "max_output_tokens": 300,
            "temperature": 0,
            "reasoning": {"effort": "none"},
        },
    )
    texts = []
    for item in second_data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    answer = "\n".join(texts)
    return {
        "first": {
            "status": first_status,
            "elapsed_s": round(first_elapsed, 3),
            "usage": first_data.get("usage", {}),
            "call": call,
        },
        "second": {
            "status": second_status,
            "elapsed_s": round(second_elapsed, 3),
            "usage": second_data.get("usage", {}),
            "answer": answer,
            "raw_output": second_data.get("output", []),
        },
        "checks": {
            "function_call_emitted": call.get("name") == "lookup_incident",
            "correct_call_argument": "INC-9042" in call.get("arguments", ""),
            "continuation_completed": second_data.get("status") == "completed",
            "uses_tool_facts": all(
                token in answer
                for token in ["SEV-2", "epoch", "cn-north-7", "cn-north-9", "64"]
            ),
            "does_not_call_tool_again": not any(
                item.get("type") == "function_call" for item in second_data.get("output", [])
            ),
        },
    }


def main():
    results = {}
    for name, fn in [
        ("document_memory", document_conversation),
        ("coding_review", coding_conversation),
        ("responses_tool_continuation", responses_tool_conversation),
    ]:
        print(f"START {name}", flush=True)
        started = time.perf_counter()
        try:
            result = fn()
            result["scenario_elapsed_s"] = round(time.perf_counter() - started, 3)
            results[name] = result
            print(f"DONE {name} checks={result.get('checks')}", flush=True)
        except Exception as exc:
            results[name] = {"error": repr(exc)}
            print(f"ERROR {name}: {exc}", flush=True)
        RESULT_PATH.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"RESULT {RESULT_PATH}")


if __name__ == "__main__":
    main()
