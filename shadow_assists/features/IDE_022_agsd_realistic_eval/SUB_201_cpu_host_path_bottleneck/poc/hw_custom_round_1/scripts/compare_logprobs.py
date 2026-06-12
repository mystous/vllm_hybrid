#!/usr/bin/env python3
"""Compare two logprob captures.

Reports:
  - tok_match_rate: per-row exact token-id match rate
  - logp_max_abs_diff: per-row max abs (lp_a - lp_b) over common positions
  - ppl_a_mean / ppl_b_mean: per-row mean of token logprobs
  - ppl_rel_diff_pct: (mean_b - mean_a) / |mean_a| × 100

Operational gate per IDE_006/TST_003:
  - binding: distribution similarity → ppl_rel_diff_pct |Δ| ≤ 1% AND logp_max_abs_diff_mean ≤ 0.5
  - token match informational
"""
from __future__ import annotations
import argparse
import json
import statistics


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A = {r["idx"]: r for r in load_jsonl(args.a)}
    B = {r["idx"]: r for r in load_jsonl(args.b)}
    keys = sorted(A.keys() & B.keys())

    tok_match = 0
    diffs = []
    ppl_a, ppl_b = [], []
    for k in keys:
        ta = A[k]["tokens"]; tb = B[k]["tokens"]
        la = A[k]["token_logprobs"]; lb = B[k]["token_logprobs"]
        n = min(len(la), len(lb))
        if n == 0:
            continue
        if ta[:n] == tb[:n]:
            tok_match += 1
        max_diff = max(abs((la[j] or 0) - (lb[j] or 0)) for j in range(n))
        diffs.append(max_diff)
        ppl_a.append(statistics.fmean(v for v in la[:n] if v is not None) if any(v is not None for v in la[:n]) else 0)
        ppl_b.append(statistics.fmean(v for v in lb[:n] if v is not None) if any(v is not None for v in lb[:n]) else 0)

    result = {
        "n_common": len(keys),
        "tok_match_rate": tok_match / len(keys) if keys else 0.0,
        "logp_max_abs_diff_mean": statistics.fmean(diffs) if diffs else float("nan"),
        "logp_max_abs_diff_p99": sorted(diffs)[int(0.99 * (len(diffs) - 1))] if diffs else float("nan"),
        "ppl_a_mean": statistics.fmean(ppl_a) if ppl_a else float("nan"),
        "ppl_b_mean": statistics.fmean(ppl_b) if ppl_b else float("nan"),
        "ppl_rel_diff_pct": ((statistics.fmean(ppl_b) - statistics.fmean(ppl_a)) / abs(statistics.fmean(ppl_a)) * 100) if ppl_a else float("nan"),
    }
    # gate verdict
    gate_pass = (
        result["ppl_rel_diff_pct"] is not None
        and abs(result["ppl_rel_diff_pct"]) <= 1.0
        and result["logp_max_abs_diff_mean"] <= 0.5
    )
    result["gate_pass"] = bool(gate_pass)
    result["gate_thresholds"] = {"|ppl_rel_diff_pct|": "≤1.0", "logp_max_abs_diff_mean": "≤0.5"}

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
