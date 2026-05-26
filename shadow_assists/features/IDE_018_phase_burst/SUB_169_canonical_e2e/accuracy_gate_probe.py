#!/usr/bin/env python3
"""SUB_170 — IDE_018 accuracy gate probe.

Compares token-level logprobs between phase-burst OFF and ON modes,
using the canonical vLLM vanilla endpoint (port 8001). Same prompts,
same seed, greedy sampling (temperature=0). The gate is:

    per-token logprob max abs diff < 1e-3

Per CLAUDE.md operational interpretation: this is the distribution-
similarity binding metric, not bit-exact token match.

Usage:
    python accuracy_gate_probe.py off       # control run, capture logprobs
    python accuracy_gate_probe.py on        # treatment run, capture logprobs
    python accuracy_gate_probe.py compare   # compute max abs diff + verdict
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import requests

PROMPTS = [
    "The quick brown fox jumps over the lazy dog. Tell me about reflexes.",
    "Explain quantum entanglement in three short sentences.",
    "Write a haiku about machine learning.",
    "What is the capital of France, and why is it famous?",
    "Describe how a bubble sort works step by step.",
    "List five common HTTP status codes and their meanings.",
    "Translate to Korean: Hello, how are you today?",
    "Summarize the plot of Hamlet in two sentences.",
]

URL = os.environ.get("AGSD_VANILLA_URL", "http://127.0.0.1:8001/v1/completions")
MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen2.5-32B-Instruct")
MAX_TOKENS = 32   # short — focus on cumulative divergence
SEED = 42
TOP_LOGPROBS = 5  # capture top-5 logprobs per token

BASE = Path("/workspace/vllm_hybrid/shadow_assists/features/IDE_018_phase_burst/SUB_169_canonical_e2e")
ACC_DIR = BASE.parent / "SUB_170_accuracy_gate"
ACC_DIR.mkdir(parents=True, exist_ok=True)


def _completion(prompt: str) -> dict:
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": SEED,
        "logprobs": TOP_LOGPROBS,
    }
    r = requests.post(URL, json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def capture(mode: str) -> None:
    out_path = ACC_DIR / f"logprobs_{mode}.json"
    rows = []
    t0 = time.time()
    for i, p in enumerate(PROMPTS):
        try:
            j = _completion(p)
            ch = j["choices"][0]
            row = {
                "idx": i,
                "prompt": p,
                "text": ch.get("text", ""),
                "tokens": (ch.get("logprobs") or {}).get("tokens", []),
                "token_logprobs": (ch.get("logprobs") or {}).get("token_logprobs", []),
                "top_logprobs": (ch.get("logprobs") or {}).get("top_logprobs", []),
            }
            rows.append(row)
            print(f"  [{i}] tokens={len(row['tokens'])} text={row['text'][:60]!r}")
        except Exception as exc:
            print(f"  [{i}] FAIL: {exc}", file=sys.stderr)
            rows.append({"idx": i, "prompt": p, "error": str(exc)})
    with open(out_path, "w") as fh:
        json.dump(
            {
                "mode": mode,
                "url": URL,
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "seed": SEED,
                "elapsed_s": time.time() - t0,
                "prompts": rows,
            },
            fh,
            indent=2,
        )
    print(f"[capture {mode}] saved → {out_path}")


def compare() -> int:
    off_path = ACC_DIR / "logprobs_off.json"
    on_path = ACC_DIR / "logprobs_on.json"
    if not off_path.exists() or not on_path.exists():
        print(f"ERROR: missing capture files (off={off_path.exists()} on={on_path.exists()})")
        return 2
    off = json.loads(off_path.read_text())
    on = json.loads(on_path.read_text())

    print("=" * 78)
    print(f"SUB_170 accuracy gate — phase-burst OFF vs ON")
    print(f"  off: {off['model']} max_tokens={off['max_tokens']} seed={off['seed']}")
    print(f"  on : {on['model']} max_tokens={on['max_tokens']} seed={on['seed']}")
    print("=" * 78)

    overall_max_diff = 0.0
    overall_tok_match = 0
    overall_tok_total = 0
    rows_summary = []

    for i, (a, b) in enumerate(zip(off["prompts"], on["prompts"])):
        if "error" in a or "error" in b:
            print(f"  [{i}] SKIP (error in capture)")
            continue
        atoks = a.get("tokens") or []
        btoks = b.get("tokens") or []
        alogs = a.get("token_logprobs") or []
        blogs = b.get("token_logprobs") or []
        n = min(len(atoks), len(btoks), len(alogs), len(blogs))
        per_token_diffs = []
        tok_match = 0
        for k in range(n):
            la = alogs[k] if alogs[k] is not None else 0.0
            lb = blogs[k] if blogs[k] is not None else 0.0
            per_token_diffs.append(abs(la - lb))
            if atoks[k] == btoks[k]:
                tok_match += 1
        if per_token_diffs:
            max_diff = max(per_token_diffs)
            mean_diff = sum(per_token_diffs) / len(per_token_diffs)
        else:
            max_diff = mean_diff = 0.0
        overall_max_diff = max(overall_max_diff, max_diff)
        overall_tok_match += tok_match
        overall_tok_total += n
        text_match = "OK" if a.get("text", "")[:60] == b.get("text", "")[:60] else "diff"
        print(f"  [{i}] n={n} tok_match={tok_match}/{n} ({100.0*tok_match/max(1,n):.1f}%) "
              f"max_logprob_diff={max_diff:.6f} mean={mean_diff:.6f} text={text_match}")
        rows_summary.append({
            "idx": i,
            "n_tokens": n,
            "tok_match": tok_match,
            "max_logprob_diff": max_diff,
            "mean_logprob_diff": mean_diff,
        })

    print("-" * 78)
    print(f"  overall_max_abs_diff_logprob = {overall_max_diff:.6f}")
    print(f"  overall_token_match          = {overall_tok_match}/{overall_tok_total}"
          f" ({100.0*overall_tok_match/max(1,overall_tok_total):.2f}%)")
    gate = overall_max_diff < 1e-3
    print(f"  GATE (< 1e-3) = {'PASS' if gate else 'FAIL'}")
    print("=" * 78)

    out = ACC_DIR / "gate_summary.json"
    with open(out, "w") as fh:
        json.dump({
            "overall_max_abs_diff_logprob": overall_max_diff,
            "overall_token_match_count": overall_tok_match,
            "overall_token_total": overall_tok_total,
            "overall_token_match_pct": 100.0 * overall_tok_match / max(1, overall_tok_total),
            "gate_threshold": 1e-3,
            "gate_pass": gate,
            "per_prompt": rows_summary,
        }, fh, indent=2)
    print(f"summary → {out}")
    return 0 if gate else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["off", "on", "compare"])
    args = ap.parse_args()
    if args.mode in ("off", "on"):
        capture(args.mode)
        return 0
    return compare()


if __name__ == "__main__":
    sys.exit(main())
