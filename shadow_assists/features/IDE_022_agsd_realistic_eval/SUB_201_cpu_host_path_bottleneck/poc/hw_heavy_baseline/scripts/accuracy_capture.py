#!/usr/bin/env python3
"""Capture greedy completions + logprobs from a single endpoint into a JSONL file.

Run this twice (once against baseline server, once against candidate server) and
then diff with `accuracy_diff.py`. This avoids running two TP=8 servers concurrently
on the same 8 GPUs (which would require gpu-memory-utilization 0.4 each).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = pq.read_table(args.parquet).to_pylist()
    import random
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n_prompts]

    out = []
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as c:
        for i, rec in enumerate(rows):
            payload = {
                "model": args.model,
                "prompt": rec["raw_text"],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "logprobs": 5,
                "stream": False,
            }
            t0 = time.time()
            r = c.post(f"{args.url}/v1/completions", json=payload)
            if r.status_code != 200:
                out.append({"i": i, "ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"})
                continue
            data = r.json()
            ch = data["choices"][0]
            lp = ch.get("logprobs") or {}
            out.append({
                "i": i,
                "ok": True,
                "prompt_id": rec.get("prompt_id"),
                "text": ch.get("text", ""),
                "tokens": lp.get("tokens") or [],
                "token_logprobs": lp.get("token_logprobs") or [],
                "elapsed_s": round(time.time() - t0, 2),
            })
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(rows)} captured", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    n_ok = sum(1 for r in out if r["ok"])
    print(f"[capture] wrote {n_ok}/{len(rows)} ok rows -> {args.out}")
    return 0 if n_ok == len(rows) else 2


if __name__ == "__main__":
    sys.exit(main())
