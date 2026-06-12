#!/usr/bin/env python3
"""Accuracy gate for hw_heavy_* levers.

Runs the same 50 sharegpt prompts (deterministic seed) against two endpoints:
    --baseline-url http://127.0.0.1:8091  (vanilla baseline)
    --candidate-url http://127.0.0.1:8092 (lever under test)

With temperature=0 it captures:
    - top-1 token at each step (via logprobs=5)
    - per-token logprob of chosen token
Then prints:
    - top-1 token agreement % over min(len_a, len_b) tokens, prompt-by-prompt and overall
    - per-token max-abs logprob diff (informational)
    - PPL relative diff (informational)
The "GATE OK" line appears iff top-1 agreement >= --threshold (default 0.95).

Designed to be invoked separately: bring up BOTH servers, run this script, then
tear down. Operational interpretation (project CONSTRAINT.md): distribution-level
similarity, not bit-exact equality.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import httpx
import pyarrow.parquet as pq


def call_completions(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "logprobs": 5,
        "stream": False,
    }
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as c:
        r = c.post(f"{url}/v1/completions", json=payload)
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
    data = r.json()
    ch = data["choices"][0]
    text = ch.get("text", "")
    lp = ch.get("logprobs") or {}
    tokens = lp.get("tokens") or []
    token_lp = lp.get("token_logprobs") or []
    return {"ok": True, "text": text, "tokens": tokens, "token_logprobs": token_lp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-url", required=True)
    ap.add_argument("--candidate-url", required=True)
    ap.add_argument("--baseline-model", required=True)
    ap.add_argument("--candidate-model", required=True)
    ap.add_argument("--parquet", required=True,
                    help="sharegpt500.parquet — same shuffle/seed as runner")
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=43,
                    help="match throughput_runner sweep1 seed (42+1=43)")
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="short, for tractable comparison (bit-perfect not goal)")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = pq.read_table(args.parquet).to_pylist()
    import random
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n_prompts]

    per_prompt = []
    tot_match = tot_compared = 0
    sum_ppl_b = sum_ppl_c = 0.0
    n_ppl = 0
    max_lp_diff = 0.0
    n_text_match = 0

    for i, rec in enumerate(rows):
        prompt = rec["raw_text"]
        t0 = time.time()
        a = call_completions(args.baseline_url, args.baseline_model, prompt, args.max_tokens)
        b = call_completions(args.candidate_url, args.candidate_model, prompt, args.max_tokens)
        if not (a["ok"] and b["ok"]):
            per_prompt.append({"i": i, "ok": False, "err_a": a.get("error"), "err_b": b.get("error")})
            continue
        ta = a["tokens"]; tb = b["tokens"]
        la = a["token_logprobs"]; lb = b["token_logprobs"]
        n = min(len(ta), len(tb))
        match = sum(1 for k in range(n) if ta[k] == tb[k])
        tot_match += match
        tot_compared += n
        # per-token max-abs logprob diff on shared tokens (only where token matches)
        for k in range(n):
            if ta[k] == tb[k] and la[k] is not None and lb[k] is not None:
                d = abs(la[k] - lb[k])
                if d > max_lp_diff:
                    max_lp_diff = d
        # PPL relative
        if la and lb:
            sa = sum(x for x in la if x is not None)
            sb = sum(x for x in lb if x is not None)
            ca = sum(1 for x in la if x is not None)
            cb = sum(1 for x in lb if x is not None)
            if ca > 0 and cb > 0:
                ppl_a = math.exp(-sa / ca)
                ppl_b = math.exp(-sb / cb)
                sum_ppl_b += ppl_a
                sum_ppl_c += ppl_b
                n_ppl += 1
        if a["text"] == b["text"]:
            n_text_match += 1
        per_prompt.append({
            "i": i,
            "ok": True,
            "n_compared": n,
            "n_match": match,
            "agreement": round(match / n, 4) if n > 0 else None,
            "len_a": len(ta), "len_b": len(tb),
            "text_eq": a["text"] == b["text"],
            "elapsed_s": round(time.time() - t0, 2),
        })

    agreement = tot_match / tot_compared if tot_compared > 0 else 0.0
    ppl_b_mean = sum_ppl_b / n_ppl if n_ppl else None
    ppl_c_mean = sum_ppl_c / n_ppl if n_ppl else None
    ppl_rel = abs(ppl_c_mean - ppl_b_mean) / ppl_b_mean if ppl_b_mean else None

    summary = {
        "baseline_url": args.baseline_url,
        "candidate_url": args.candidate_url,
        "baseline_model": args.baseline_model,
        "candidate_model": args.candidate_model,
        "n_prompts": args.n_prompts,
        "max_tokens": args.max_tokens,
        "tokens_compared": tot_compared,
        "tokens_matched": tot_match,
        "top1_agreement": round(agreement, 4),
        "text_equal_count": n_text_match,
        "max_abs_logprob_diff": round(max_lp_diff, 4),
        "ppl_baseline_mean": round(ppl_b_mean, 4) if ppl_b_mean else None,
        "ppl_candidate_mean": round(ppl_c_mean, 4) if ppl_c_mean else None,
        "ppl_relative_diff": round(ppl_rel, 4) if ppl_rel is not None else None,
        "threshold": args.threshold,
        "gate_ok": agreement >= args.threshold,
        "per_prompt": per_prompt,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[gate] agreement={agreement:.4f} text_eq={n_text_match}/{args.n_prompts} "
          f"max_lp_diff={max_lp_diff:.3f} ppl_rel={summary['ppl_relative_diff']} "
          f"-> {'GATE OK' if summary['gate_ok'] else 'GATE FAIL'} (thr={args.threshold})")
    return 0 if summary["gate_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
