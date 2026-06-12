#!/usr/bin/env python3
"""HWC2 measurement aggregator (uses Round 1 baseline as reference)."""
from __future__ import annotations
import glob
import json
import os
import re
import statistics

RUNS_R1 = "/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/hw_custom_round_1/runs"
RUNS_R2 = "/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/hw_custom_round_2/runs"
PAT = re.compile(r"^(.+)_s(\d+)\.json$")


def load(path: str) -> dict[str, list[dict]]:
    tags: dict[str, list[dict]] = {}
    for p in sorted(glob.glob(os.path.join(path, "*_s*.json"))):
        m = PAT.match(os.path.basename(p))
        if not m:
            continue
        tag = m.group(1)
        with open(p) as f:
            d = json.load(f)
        tags.setdefault(tag, []).append(d)
    return tags


def stat(vals):
    if not vals: return float("nan"), float("nan")
    if len(vals) == 1: return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def main():
    r1 = load(RUNS_R1)
    r2 = load(RUNS_R2)
    base_tps = [r.get("output_tps") for r in r1.get("baseline", []) if r.get("output_tps")]
    base_mean, base_std = stat(base_tps)

    rows = []
    for tag in sorted(r2.keys()):
        runs = r2[tag]
        tps = [r.get("output_tps") for r in runs if r.get("output_tps")]
        if not tps:
            rows.append((tag, "boot_fail", "-", "-", "-", "-", "-"))
            continue
        m, s = stat(tps)
        delta = (m - base_mean) / base_mean * 100
        gpu = statistics.mean([r.get("gpu_util", 0.0) for r in runs if r.get("gpu_util") is not None])
        cpu = statistics.mean([r.get("cpu_util", 0.0) for r in runs if r.get("cpu_util") is not None])
        rows.append((tag, len(tps), f"{m:.1f}", f"{s:.1f}", f"{delta:+.2f}%", f"{gpu:.1f}", f"{cpu:.1f}"))

    print(f"# HWC2 results — Round 1 baseline mean={base_mean:.1f} ± {base_std:.1f} (N={len(base_tps)})\n")
    print("| Tag | N | mean tps | std | Δ% vs baseline | GPU% | CPU% |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


if __name__ == "__main__":
    main()
