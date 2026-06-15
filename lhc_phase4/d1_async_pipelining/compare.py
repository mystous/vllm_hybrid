#!/usr/bin/env python3
"""D-1 lever vs vanilla baseline 비교.

baseline: lhc_phase4/tsk042_10model_unified/Llama-3.1-70B-Instruct/vanilla_runs/
lever:    lhc_phase4/d1_async_pipelining/runs/{LEVER}/

각 corpus × sweep 의 output_tps 를 paired comparison.
ΔTPS = (lever - baseline) / baseline * 100.
3/7 corpus 이상에서 Δ ≥ +5% AND CI tight 면 STOP / 양수 확정.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys


CORPORA = ["sharegpt", "swebench", "humaneval", "mbpp", "wildchat", "lmsys", "mix"]
SWEEPS = [1, 2, 3]
BASE = "/workspace/host_vllm_hybrid/lhc_phase4/tsk042_10model_unified/Llama-3.1-70B-Instruct/vanilla_runs"
LEVERROOT = "/workspace/host_vllm_hybrid/lhc_phase4/d1_async_pipelining/runs"


def load_tps(d: str, corpus: str, sweep: int) -> float | None:
    f = os.path.join(d, f"{corpus}_s{sweep}.json")
    if not os.path.exists(f):
        return None
    try:
        data = json.load(open(f))
        return float(data.get("output_tps") or 0)
    except Exception:
        return None


def compare(lever: str) -> int:
    base_d = BASE
    lev_d = os.path.join(LEVERROOT, lever)
    if not os.path.isdir(lev_d):
        print(f"lever dir not found: {lev_d}", file=sys.stderr)
        return 2
    print(f"\n=== D-1 lever={lever} vs baseline ===")
    print(f"{'corpus':<10} {'sweep':<6} {'baseline':>9} {'lever':>9} {'Δtps':>8} {'Δ%':>7}")
    win_count = 0
    total_count = 0
    deltas: list[float] = []
    for c in CORPORA:
        for s in SWEEPS:
            b = load_tps(base_d, c, s)
            l = load_tps(lev_d, c, s)
            if b is None or l is None or b == 0:
                continue
            d = l - b
            dpct = d / b * 100
            print(f"{c:<10} s{s:<5} {b:>9.1f} {l:>9.1f} {d:>+8.1f} {dpct:>+6.2f}%")
            deltas.append(dpct)
            total_count += 1
            if dpct >= 5.0:
                win_count += 1
    if not deltas:
        print("\nNo paired data available.")
        return 1
    avg = sum(deltas) / len(deltas)
    # simple std for noise estimate
    var = sum((x - avg) ** 2 for x in deltas) / len(deltas) if len(deltas) > 1 else 0
    std = math.sqrt(var)
    print(f"\nSummary: n={total_count}, mean Δ%={avg:+.2f}, std={std:.2f}, wins(≥+5%)={win_count}")
    # STOP 조건: 3/7 corpus 이상에서 Δ ≥ +5%
    if win_count >= 3:
        print(f"\n** STOP — {win_count} corpus showed ≥+5% (양수 발견 후보). **")
        return 0
    print(f"\nNo STOP — wins={win_count} < 3.")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", required=True, help="lever name (e.g. L1_stream16)")
    args = ap.parse_args()
    sys.exit(compare(args.lever))
