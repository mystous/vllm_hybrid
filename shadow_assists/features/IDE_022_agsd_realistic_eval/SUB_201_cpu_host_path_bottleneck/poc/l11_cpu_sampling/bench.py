"""L11 throughput bench — sharegpt 100p × conc=16 × max-tok 256.

Sends `n_prompts` completions concurrently (conc) and reports tokens/s.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import pandas as pd


def load_prompts(parquet: Path, corpus: str, n: int) -> list[str]:
    df = pd.read_parquet(parquet)
    if corpus != "all":
        df = df[df["corpus"] == corpus]
    df = df.head(n).reset_index(drop=True)
    return df["raw_text"].astype(str).tolist()


async def _one(client, port, model, idx, prompt, max_tokens, temperature):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95 if temperature > 0 else 1.0,
        "top_k": 50 if temperature > 0 else -1,
        "stream": False,
    }
    t_send = time.perf_counter()
    r = await client.post(
        f"http://127.0.0.1:{port}/v1/completions",
        json=payload,
        timeout=httpx.Timeout(600.0, connect=10.0),
    )
    r.raise_for_status()
    t_done = time.perf_counter()
    data = r.json()
    usage = data.get("usage") or {}
    return {
        "i": idx,
        "n_input_tok": usage.get("prompt_tokens", 0),
        "n_output_tok": usage.get("completion_tokens", 0),
        "latency_s": t_done - t_send,
        "text_len": len(data["choices"][0]["text"]),
        "finish_reason": data["choices"][0]["finish_reason"],
    }


async def _bench_async(port, model, prompts, conc, max_tokens, temperature):
    sem = asyncio.Semaphore(conc)
    t_wall0 = time.perf_counter()

    async def _bounded(idx, prompt, cl):
        async with sem:
            return await _one(cl, port, model, idx, prompt, max_tokens,
                              temperature)

    async with httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=conc * 2,
            max_keepalive_connections=conc * 2,
        )
    ) as cl:
        rows = await asyncio.gather(
            *(_bounded(i, p, cl) for i, p in enumerate(prompts))
        )
    t_wall1 = time.perf_counter()
    rows.sort(key=lambda r: r["i"])
    total_in = sum(r["n_input_tok"] for r in rows)
    total_out = sum(r["n_output_tok"] for r in rows)
    wall = t_wall1 - t_wall0
    summary = {
        "n_prompts": len(prompts),
        "conc": conc,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "wall_s": wall,
        "total_input_tok": total_in,
        "total_output_tok": total_out,
        "throughput_in_tps": total_in / wall if wall > 0 else 0,
        "throughput_out_tps": total_out / wall if wall > 0 else 0,
        "throughput_total_tps": (total_in + total_out) / wall if wall > 0 else 0,
        "median_latency_s": sorted(r["latency_s"] for r in rows)[
            len(rows) // 2
        ],
        "p90_latency_s": sorted(r["latency_s"] for r in rows)[
            int(len(rows) * 0.9)
        ],
        "p99_latency_s": sorted(r["latency_s"] for r in rows)[
            min(int(len(rows) * 0.99), len(rows) - 1)
        ],
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument(
        "--parquet",
        default="/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet",
    )
    ap.add_argument("--corpus", default="sharegpt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = load_prompts(Path(args.parquet), args.corpus, args.n)
    print(f"[bench] loaded {len(prompts)} prompts from corpus={args.corpus}")
    summary, rows = asyncio.run(
        _bench_async(
            args.port, args.model, prompts, args.conc, args.max_tokens,
            args.temperature,
        )
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    (out.parent / (out.stem + ".raw.jsonl")).write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
