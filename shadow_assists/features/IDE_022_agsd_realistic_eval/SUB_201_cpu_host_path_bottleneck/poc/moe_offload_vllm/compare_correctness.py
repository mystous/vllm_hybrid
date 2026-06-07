#!/usr/bin/env python3
"""Compare vanilla vs offload correctness probe results.

Computes:
  - text equality rate (exact greedy decoded text)
  - first-token logprob diff distribution (max abs diff per prompt)
  - PPL relative diff per prompt

Per CLAUDE.md §Constraint operating interpretation:
  - logprob max-abs-diff < 0.1 → PASS (informational metric)
  - text mismatch is informational only (BF16 non-associativity)
"""
import json
import math
import sys
from statistics import mean, median


def load(path):
    with open(path) as f:
        return json.load(f)


def per_token_logprob(rec):
    lp = rec.get("logprobs") or {}
    tlp = lp.get("token_logprobs") or []
    return [x for x in tlp if x is not None]


def main():
    a = load(sys.argv[1])
    b = load(sys.argv[2])

    A, B = a["results"], b["results"]
    assert len(A) == len(B), (len(A), len(B))

    text_match = 0
    max_abs_diffs = []
    ppl_rel_diffs = []
    text_pairs = []
    for ra, rb in zip(A, B):
        text_match += 1 if ra["text"] == rb["text"] else 0
        text_pairs.append((ra["text"][:80], rb["text"][:80]))

        lpa, lpb = per_token_logprob(ra), per_token_logprob(rb)
        n = min(len(lpa), len(lpb))
        if n == 0:
            continue
        diffs = [abs(lpa[i] - lpb[i]) for i in range(n)]
        max_abs_diffs.append(max(diffs))

        # PPL = exp(-mean(logprob))
        ppl_a = math.exp(-mean(lpa[:n]))
        ppl_b = math.exp(-mean(lpb[:n]))
        ppl_rel_diffs.append(abs(ppl_b - ppl_a) / max(ppl_a, 1e-9))

    print(f"prompts compared       : {len(A)}")
    print(f"exact text match       : {text_match}/{len(A)}")
    print(f"max-abs logprob diff   : "
          f"mean={mean(max_abs_diffs):.4f}  median={median(max_abs_diffs):.4f}  "
          f"max={max(max_abs_diffs):.4f}")
    print(f"PPL relative diff      : "
          f"mean={mean(ppl_rel_diffs):.4f}  median={median(ppl_rel_diffs):.4f}  "
          f"max={max(ppl_rel_diffs):.4f}")

    # Gate per CLAUDE.md
    LP_GATE = 0.1
    PPL_GATE = 0.05
    lp_pass = max(max_abs_diffs) < LP_GATE
    ppl_pass = max(ppl_rel_diffs) < PPL_GATE
    overall = lp_pass and ppl_pass

    print()
    print(f"GATE  logprob max diff < {LP_GATE} : {'PASS' if lp_pass else 'FAIL'}")
    print(f"GATE  PPL relative diff < {PPL_GATE} : {'PASS' if ppl_pass else 'FAIL'}")
    print(f"OVERALL                              : {'PASS' if overall else 'FAIL'}")

    print()
    print("Sample text snippets (vanilla / offload):")
    for i, (ta, tb) in enumerate(text_pairs[:3]):
        print(f"  [{i}] A: {ta!r}")
        print(f"       B: {tb!r}")


if __name__ == "__main__":
    main()
