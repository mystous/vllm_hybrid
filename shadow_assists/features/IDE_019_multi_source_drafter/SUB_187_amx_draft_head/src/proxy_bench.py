#!/usr/bin/env python3
"""Tiny e2e proxy bench — N prompts × concurrency to a vllm OpenAI-compat completions
endpoint. Measures throughput (tokens / s wall) and per-request TTFT proxy.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request as urlreq

PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "Write a haiku about machine learning.",
    "What is the capital of France and why is it famous?",
    "Summarize the plot of Hamlet in three sentences.",
    "Describe the process of photosynthesis briefly.",
    "List five benefits of regular exercise.",
    "What are the main differences between Python and Rust?",
    "Provide a short recipe for chocolate chip cookies.",
    "Explain the theory of relativity in everyday language.",
    "What is the most populous country in the world today?",
] * 5  # 50 prompts


def call(url, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0, "n": 1,
        "stream": False,
    }).encode()
    req = urlreq.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urlreq.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    t1 = time.time()
    j = json.loads(data)
    text = j["choices"][0].get("text", "")
    usage = j.get("usage", {})
    out_tokens = usage.get("completion_tokens", 0) or len(text.split())
    return {
        "wall_ms": (t1 - t0) * 1000,
        "out_tokens": out_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = (PROMPTS * ((args.n_prompts // len(PROMPTS)) + 1))[: args.n_prompts]
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(call, args.url, args.model, p, args.max_tokens, 120)
                for p in prompts]
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"error": str(e)})
    t1 = time.time()
    wall_s = t1 - t0

    ok = [r for r in results if "out_tokens" in r]
    err = [r for r in results if "error" in r]
    total_tokens = sum(r["out_tokens"] for r in ok)
    tps = total_tokens / wall_s if wall_s > 0 else 0
    walls = sorted([r["wall_ms"] for r in ok])

    def pct(p):
        if not walls: return 0
        i = min(len(walls) - 1, int(len(walls) * p))
        return walls[i]

    out = {
        "wall_s": wall_s,
        "n_ok": len(ok),
        "n_err": len(err),
        "total_out_tokens": total_tokens,
        "tps": tps,
        "p50_ms": pct(0.5),
        "p90_ms": pct(0.9),
        "p99_ms": pct(0.99),
        "errors": err[:3],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
