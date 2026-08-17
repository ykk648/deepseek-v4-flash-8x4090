#!/usr/bin/env python3
import argparse
import http.client
import json
import statistics
import time
from urllib.parse import urlparse


def run_once(base_url: str, model: str, max_tokens: int, prompt: str) -> dict:
    parsed = urlparse(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=900)
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        ensure_ascii=False,
    ).encode()

    started = time.perf_counter()
    connection.request(
        "POST",
        f"{parsed.path.rstrip('/')}/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read().decode(errors='replace')}")

    first_token_at = None
    token_arrivals = []
    usage = None
    output_parts = []
    while True:
        line = response.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            break
        chunk = json.loads(payload)
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content") or delta.get("reasoning_content") or ""
        if piece:
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            token_arrivals.append(now)
            output_parts.append(piece)

    finished = time.perf_counter()
    connection.close()
    if first_token_at is None or usage is None:
        raise RuntimeError("Stream ended without token data or usage")

    completion_tokens = usage["completion_tokens"]
    total_seconds = finished - started
    ttft = first_token_at - started
    decode_seconds = max(finished - first_token_at, 1e-9)
    intervals = [b - a for a, b in zip(token_arrivals, token_arrivals[1:])]
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": completion_tokens,
        "ttft_seconds": ttft,
        "total_seconds": total_seconds,
        "output_tokens_per_second": completion_tokens / total_seconds,
        "decode_tokens_per_second": max(completion_tokens - 1, 0) / decode_seconds,
        "median_chunk_interval_seconds": statistics.median(intervals) if intervals else None,
        "preview": "".join(output_parts)[:100].replace("\n", "\\n"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="DeepSeek-V4-Flash-0731")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--prompt",
        default="请从1开始逐个写出正整数，并用英文逗号分隔。不要解释。",
    )
    args = parser.parse_args()

    results = []
    for index in range(args.repeats):
        result = run_once(args.base_url, args.model, args.max_tokens, args.prompt)
        results.append(result)
        print(json.dumps({"run": index + 1, **result}, ensure_ascii=False), flush=True)

    if len(results) > 1:
        print(
            json.dumps(
                {
                    "summary": {
                        "repeats": len(results),
                        "mean_ttft_seconds": statistics.mean(r["ttft_seconds"] for r in results),
                        "mean_total_seconds": statistics.mean(r["total_seconds"] for r in results),
                        "mean_decode_tokens_per_second": statistics.mean(
                            r["decode_tokens_per_second"] for r in results
                        ),
                    }
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
