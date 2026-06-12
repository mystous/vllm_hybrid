#!/usr/bin/env python3
"""Capture per-token logprobs (top-1) for the first N prompts from a parquet.

Used by run_accuracy_gate.sh to compare baseline vs candidate.
"""
from __future__ import annotations
import argparse
import json
import sys

import httpx
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tbl = pq.read_table(args.parquet)
    # Find prompt column
    prompts_col = None
    for name in ["prompt", "text", "input"]:
        if name in tbl.column_names:
            prompts_col = name
            break
    if not prompts_col:
        print(f"No prompt col in {tbl.column_names}", file=sys.stderr)
        sys.exit(1)
    prompts = tbl[prompts_col].to_pylist()[: args.n]

    with httpx.Client(timeout=180) as c, open(args.out, "w") as f:
        for i, p in enumerate(prompts):
            body = {
                "model": args.model,
                "prompt": p,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "seed": args.seed,
                "logprobs": 1,
                "stream": False,
            }
            try:
                r = c.post(f"http://127.0.0.1:{args.port}/v1/completions", json=body)
                r.raise_for_status()
            except Exception as e:
                print(f"[capture] req {i} failed: {e}", file=sys.stderr)
                continue
            j = r.json()
            ch = j["choices"][0]
            row = {
                "idx": i,
                "tokens": ch["logprobs"]["tokens"],
                "token_logprobs": ch["logprobs"]["token_logprobs"],
                "text": ch["text"],
            }
            f.write(json.dumps(row) + "\n")
            if (i + 1) % 10 == 0:
                print(f"[capture] {i+1}/{args.n}", file=sys.stderr)


if __name__ == "__main__":
    main()
