#!/usr/bin/env python3
"""Diff two capture JSONL files (baseline vs candidate). Prints/writes gate result."""
import argparse
import json
import math
import sys
from pathlib import Path


def load(path):
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.95)
    args = ap.parse_args()

    a = load(args.baseline)
    b = load(args.candidate)
    n = min(len(a), len(b))

    tot_match = tot_compared = 0
    n_text_match = 0
    max_lp_diff = 0.0
    sum_ppl_b = sum_ppl_c = 0.0
    n_ppl = 0
    per_prompt = []
    for i in range(n):
        ra, rb = a[i], b[i]
        if not (ra.get("ok") and rb.get("ok")):
            per_prompt.append({"i": i, "ok": False, "err_a": ra.get("error"), "err_b": rb.get("error")})
            continue
        ta, tb = ra["tokens"], rb["tokens"]
        la, lb = ra["token_logprobs"], rb["token_logprobs"]
        m = min(len(ta), len(tb))
        match = sum(1 for k in range(m) if ta[k] == tb[k])
        tot_match += match
        tot_compared += m
        for k in range(m):
            if ta[k] == tb[k] and la[k] is not None and lb[k] is not None:
                d = abs(la[k] - lb[k])
                if d > max_lp_diff:
                    max_lp_diff = d
        if la and lb:
            sa = sum(x for x in la if x is not None)
            sb = sum(x for x in lb if x is not None)
            ca = sum(1 for x in la if x is not None)
            cb = sum(1 for x in lb if x is not None)
            if ca > 0 and cb > 0:
                sum_ppl_b += math.exp(-sa / ca)
                sum_ppl_c += math.exp(-sb / cb)
                n_ppl += 1
        if ra["text"] == rb["text"]:
            n_text_match += 1
        per_prompt.append({
            "i": i,
            "ok": True,
            "n_compared": m,
            "n_match": match,
            "agreement": round(match / m, 4) if m > 0 else None,
            "text_eq": ra["text"] == rb["text"],
        })

    agreement = tot_match / tot_compared if tot_compared > 0 else 0.0
    ppl_b_mean = sum_ppl_b / n_ppl if n_ppl else None
    ppl_c_mean = sum_ppl_c / n_ppl if n_ppl else None
    ppl_rel = abs(ppl_c_mean - ppl_b_mean) / ppl_b_mean if ppl_b_mean else None

    summary = {
        "baseline_file": args.baseline,
        "candidate_file": args.candidate,
        "n_prompts": n,
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
    print(f"[gate] agreement={agreement:.4f} text_eq={n_text_match}/{n} "
          f"max_lp_diff={max_lp_diff:.3f} ppl_rel={summary['ppl_relative_diff']} "
          f"-> {'GATE OK' if summary['gate_ok'] else 'GATE FAIL'} (thr={args.threshold})")
    return 0 if summary["gate_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
