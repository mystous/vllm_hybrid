#!/usr/bin/env python3
"""SUB_185 long-context benchmark for cold-KV CPU dequant overlap measurement.

Sends pre-built 8K-input prompts to a single vllm endpoint (direct
/v1/completions) at fixed concurrency. Records throughput, TTFT (best-effort
via stream first-token timestamp), latency p50/p99.

For OFF mode: just benchmark.
For ON mode: parent process (`launcher.sh`) starts a concurrent CPU dequant
firer thread BEFORE invoking this benchmark, simulating "CPU dequant during
GPU prefill" overlap. This benchmark file itself stays identical for both
modes — only the environment (concurrent CPU work) differs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen2.5-32B-Instruct")


async def send_one_stream(client: httpx.AsyncClient, url: str, prompt: str,
                          max_tokens: int) -> dict:
    """Send completion via stream to capture TTFT (time to first token)."""
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    total_tokens = 0
    async with client.stream("POST", url, json=body, timeout=600.0) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                d = json.loads(payload)
            except Exception:
                continue
            ch = d.get("choices") or []
            if not ch:
                continue
            text = ch[0].get("text") or ""
            if text:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                # approximate token count by whitespace split is crude; use
                # finish_reason / usage in last chunk if present
                total_tokens += 1
            usage = d.get("usage")
            if usage:
                # vllm provides this in final chunk for stream
                total_tokens = max(total_tokens, usage.get("completion_tokens", 0))
    wall = time.perf_counter() - t0
    return {"tokens": total_tokens, "latency_s": wall, "ttft_s": ttft}


async def run_bench(url: str, prompts: list[str], max_tokens: int,
                    concurrency: int) -> dict:
    limits = httpx.Limits(max_keepalive_connections=concurrency,
                          max_connections=concurrency * 2)
    async with httpx.AsyncClient(timeout=600.0, limits=limits) as client:
        # warmup (single short, ignore failure)
        try:
            await client.post(url, json={"model": MODEL, "prompt": "hello",
                                         "max_tokens": 4, "temperature": 0.0,
                                         "top_p": 1.0, "seed": 0}, timeout=30.0)
        except Exception:
            pass

        sem = asyncio.Semaphore(concurrency)
        results: list[dict] = []

        async def worker(p: str):
            async with sem:
                try:
                    return await send_one_stream(client, url, p, max_tokens)
                except Exception as e:
                    return {"tokens": 0, "latency_s": 0.0, "ttft_s": None,
                            "error": str(e)[:200]}

        t0 = time.perf_counter()
        results = await asyncio.gather(*[worker(p) for p in prompts])
        wall = time.perf_counter() - t0

    ok = [r for r in results if r["tokens"] > 0]
    err = [r for r in results if r["tokens"] == 0]
    total_tok = sum(r["tokens"] for r in ok)
    tps = total_tok / wall if wall > 0 else 0.0
    latencies = sorted(r["latency_s"] for r in ok)
    ttfts = sorted(r["ttft_s"] for r in ok if r.get("ttft_s") is not None)

    def pct(arr, q):
        if not arr:
            return None
        i = min(len(arr) - 1, int(len(arr) * q))
        return arr[i]

    return {
        "url": url,
        "n_prompts": len(prompts),
        "n_ok": len(ok),
        "n_err": len(err),
        "wall_s": wall,
        "total_out_tokens": total_tok,
        "tps": tps,
        "latency_p50_s": pct(latencies, 0.5),
        "latency_p99_s": pct(latencies, 0.99),
        "ttft_p50_s": pct(ttfts, 0.5),
        "ttft_p99_s": pct(ttfts, 0.99),
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "errors_sample": [r.get("error") for r in err[:3]],
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-file", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1/completions")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-prompts", type=int, default=None,
                    help="If set, limits the number of prompts used.")
    args = ap.parse_args()

    data = json.loads(args.prompts_file.read_text())
    prompts = [p["prompt"] for p in data["prompts"]]
    if args.num_prompts is not None:
        prompts = prompts[:args.num_prompts]

    summary = await run_bench(args.url, prompts, args.max_tokens, args.concurrency)
    summary["workload_meta"] = {
        "target_tokens": data.get("target_tokens"),
        "tokens_per_line": data.get("tokens_per_line"),
        "n_lines_per_prompt": data.get("n_lines_per_prompt"),
        "sampled_token_lens_p50": (
            sorted(data["sampled_token_lens"])[len(data["sampled_token_lens"]) // 2]
            if data.get("sampled_token_lens") else None),
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"[bench] saved -> {args.out}")
    print(f"[bench] tps={summary['tps']:.1f} "
          f"ttft_p50={(summary['ttft_p50_s'] or 0)*1000:.0f}ms "
          f"lat_p50={(summary['latency_p50_s'] or 0)*1000:.0f}ms "
          f"ok={summary['n_ok']}/{summary['n_prompts']}")


if __name__ == "__main__":
    asyncio.run(main())
