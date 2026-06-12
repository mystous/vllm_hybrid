#!/usr/bin/env python
"""Minimal concurrent tput driver — POST /v1/completions, measure
output_tokens/s + ttft + tpot from completion-tokens / wall time.

Usage:
  run_tput.py --port 8005 --model meta-llama/Llama-3.1-8B-Instruct \
      --num-prompts 10 --max-tokens 256 --concurrency 4 \
      --prompts sharegpt|fixed --out runs/tput_X.json
"""
import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import aiohttp


FIXED_PROMPTS = [
    "Explain why the sky appears blue in clear daytime conditions, in detail.",
    "Write a short story about a robot who discovers a hidden garden in the city.",
    "Summarize the main causes and outcomes of the French Revolution.",
    "Describe the process of photosynthesis at the molecular level.",
    "What are the key differences between supervised and reinforcement learning?",
    "Recommend five books for someone interested in moral philosophy and why.",
    "Translate the following English sentence into French and German: 'Knowledge is power, but wisdom is freedom.'",
    "Discuss the trade-offs between TCP and UDP in real-time communication.",
    "Outline a 7-day balanced vegetarian meal plan for an adult.",
    "Explain the role of the central bank during a financial crisis with examples.",
    "Compare the architectural style of Renaissance and Baroque cathedrals.",
    "Write Python code that finds the longest palindromic substring in a string.",
    "What advice would you give to a first-time public speaker on stage?",
    "Describe how a transformer attention mechanism works step by step.",
    "Explain quantum entanglement to a curious 12-year-old.",
    "List the major rivers of South America and the ecosystems they support.",
]


def load_prompts(name: str, n: int) -> list[str]:
    if name == "fixed":
        out = []
        i = 0
        while len(out) < n:
            out.append(FIXED_PROMPTS[i % len(FIXED_PROMPTS)])
            i += 1
        return out
    if name == "sharegpt":
        path = (
            "/workspace/host_vllm_hybrid/shadow_assists/features/"
            "IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/"
            "poc/b3_8gpu_full/sharegpt200.parquet"
        )
        if not Path(path).exists():
            return load_prompts("fixed", n)
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(path)
            df = tbl.to_pandas()
            col = "prompt" if "prompt" in df.columns else df.columns[0]
            prompts = [str(p) for p in df[col].tolist()[:n]]
            return prompts if len(prompts) >= n else load_prompts("fixed", n)
        except Exception:
            return load_prompts("fixed", n)
    raise ValueError(name)


async def one(session, url, model, prompt, max_tokens, temp, sem, results):
    async with sem:
        t0 = time.perf_counter()
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temp,
            "stream": False,
        }
        try:
            async with session.post(
                url, json=body, timeout=aiohttp.ClientTimeout(total=600)
            ) as r:
                txt = await r.text()
                if r.status != 200:
                    results.append({"ok": False, "err": f"{r.status}:{txt[:120]}"})
                    return
                data = json.loads(txt)
        except Exception as e:
            results.append({"ok": False, "err": f"{type(e).__name__}:{e}"})
            return
        wall = time.perf_counter() - t0
        usage = data.get("usage", {}) or {}
        n_out = int(usage.get("completion_tokens", 0))
        n_in = int(usage.get("prompt_tokens", 0))
        results.append({
            "ok": True,
            "wall": wall,
            "n_out": n_out,
            "n_in": n_in,
            "tps_per_req": (n_out / wall) if wall > 0 else 0,
        })


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8005)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompts", default="fixed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=2,
                    help="number of warmup requests not counted in tput")
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}/v1/completions"
    prompts = load_prompts(args.prompts, args.num_prompts + args.warmup)

    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        # warmup
        wr = []
        if args.warmup > 0:
            await asyncio.gather(*[
                one(s, url, args.model, p, 32, 0.0, sem, wr)
                for p in prompts[: args.warmup]
            ])
        # measured run
        results = []
        t_start = time.perf_counter()
        await asyncio.gather(*[
            one(s, url, args.model, p, args.max_tokens, args.temperature, sem, results)
            for p in prompts[args.warmup: args.warmup + args.num_prompts]
        ])
        wall = time.perf_counter() - t_start

    ok = [r for r in results if r.get("ok")]
    n_out_total = sum(r["n_out"] for r in ok)
    tps = n_out_total / wall if wall > 0 else 0
    tpot_per_req = [r["wall"] / max(r["n_out"], 1) * 1000 for r in ok]
    summary = {
        "model": args.model,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "wall_s": wall,
        "n_ok": len(ok),
        "n_fail": len(results) - len(ok),
        "tps_output_total": tps,
        "tps_per_req_mean": (statistics.mean([r["tps_per_req"] for r in ok])
                             if ok else 0),
        "tpot_p50_ms": (statistics.median(tpot_per_req) if tpot_per_req else 0),
        "tpot_mean_ms": (statistics.mean(tpot_per_req) if tpot_per_req else 0),
        "n_out_total": n_out_total,
        "fail_samples": [r["err"] for r in results if not r.get("ok")][:3],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
