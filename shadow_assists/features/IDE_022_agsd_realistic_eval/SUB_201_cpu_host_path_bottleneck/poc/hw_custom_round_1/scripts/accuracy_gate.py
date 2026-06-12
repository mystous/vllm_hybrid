#!/usr/bin/env python3
"""Accuracy gate for HWC1 candidates.

Per IDE_006/TST_003 operational interpretation:
  - greedy token-level identity is informational
  - distribution similarity (per-token logprob max-abs-diff, sequence PPL relative diff)
    is the binding gate.

Drives two vllm endpoints (baseline @ port_a, candidate @ port_b) with the same
prompts and seed, requests `logprobs=5` and compares.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import statistics
import sys
from collections.abc import Iterable

import httpx
import pyarrow.parquet as pq


async def gen_one(client: httpx.AsyncClient, port: int, model: str, prompt: str,
                  max_tokens: int = 64, logprobs: int = 5, seed: int = 0):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "logprobs": logprobs,
        "stream": False,
    }
    r = await client.post(f"http://127.0.0.1:{port}/v1/completions", json=body, timeout=120)
    r.raise_for_status()
    j = r.json()
    choice = j["choices"][0]
    tokens = choice["logprobs"]["tokens"]
    token_logprobs = choice["logprobs"]["token_logprobs"]
    return tokens, token_logprobs, choice["text"]


def seq_ppl(token_logprobs: Iterable[float]) -> float:
    vals = [v for v in token_logprobs if v is not None]
    if not vals:
        return float("nan")
    return statistics.fmean(vals)  # mean log-prob (neg → higher PPL)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-a", type=int, required=True, help="baseline port")
    ap.add_argument("--port-b", type=int, required=True, help="candidate port")
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tbl = pq.read_table(args.parquet)
    prompts_col = "prompt" if "prompt" in tbl.column_names else "text"
    prompts = tbl[prompts_col].to_pylist()[: args.n]

    diffs = []
    ppl_a, ppl_b = [], []
    tok_match = 0
    async with httpx.AsyncClient() as c:
        for i, p in enumerate(prompts):
            try:
                ta, la, txt_a = await gen_one(c, args.port_a, args.model, p, args.max_tokens)
                tb, lb, txt_b = await gen_one(c, args.port_b, args.model, p, args.max_tokens)
            except Exception as e:
                print(f"[acc] req {i} failed: {e}", file=sys.stderr)
                continue
            n_common = min(len(la), len(lb))
            if n_common == 0:
                continue
            max_diff = max(abs((la[j] or 0) - (lb[j] or 0)) for j in range(n_common))
            diffs.append(max_diff)
            ppl_a.append(seq_ppl(la))
            ppl_b.append(seq_ppl(lb))
            if ta[:n_common] == tb[:n_common]:
                tok_match += 1
            if (i + 1) % 10 == 0:
                print(f"[acc] {i+1}/{args.n} done; cur max_diff={max_diff:.4f}", file=sys.stderr)

    result = {
        "n": len(diffs),
        "tok_match_rate": tok_match / len(diffs) if diffs else 0.0,
        "logprob_maxabsdiff_mean": statistics.fmean(diffs) if diffs else float("nan"),
        "logprob_maxabsdiff_p99": sorted(diffs)[int(0.99 * (len(diffs)-1))] if diffs else float("nan"),
        "ppl_a_mean": statistics.fmean(ppl_a) if ppl_a else float("nan"),
        "ppl_b_mean": statistics.fmean(ppl_b) if ppl_b else float("nan"),
        "ppl_rel_diff_pct": ((statistics.fmean(ppl_b) - statistics.fmean(ppl_a)) / abs(statistics.fmean(ppl_a)) * 100) if ppl_a else float("nan"),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
