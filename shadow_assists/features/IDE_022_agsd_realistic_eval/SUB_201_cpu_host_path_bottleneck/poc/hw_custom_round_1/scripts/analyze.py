#!/usr/bin/env python3
"""HWC1 measurement aggregator.

Reads runs/*_s*.json, groups by tag, computes mean/std/sweep-count of output_tps,
and computes Δ% vs baseline. Prints a markdown table.
"""
from __future__ import annotations
import glob
import json
import os
import re
import statistics
import sys

RUNS = "/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/hw_custom_round_1/runs"

PAT = re.compile(r"^(.+)_s(\d+)\.json$")


def load() -> dict[str, list[dict]]:
    tags: dict[str, list[dict]] = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "*_s*.json"))):
        fn = os.path.basename(p)
        m = PAT.match(fn)
        if not m:
            continue
        tag, sweep = m.group(1), int(m.group(2))
        with open(p) as f:
            d = json.load(f)
        d["_sweep"] = sweep
        tags.setdefault(tag, []).append(d)
    return tags


def stat(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def main() -> None:
    tags = load()
    if not tags:
        print("no runs found")
        return
    # Compute baseline mean from 'baseline' tag (drop sweep 1 as warm-up? keep all for now)
    base_runs = tags.get("baseline", [])
    base_tps = [r.get("output_tps") for r in base_runs if r.get("output_tps")]
    base_mean, base_std = stat(base_tps)

    rows = []
    for tag in sorted(tags.keys()):
        runs = tags[tag]
        tps_vals = [r.get("output_tps") for r in runs if r.get("output_tps")]
        # boot fail entries with status
        if not tps_vals:
            rows.append((tag, "boot_fail", "-", "-", "-", "-", "-"))
            continue
        m, s = stat(tps_vals)
        delta = (m - base_mean) / base_mean * 100 if base_mean else float("nan")
        gpu = statistics.mean([r.get("gpu_util", 0.0) for r in runs if r.get("gpu_util") is not None])
        cpu = statistics.mean([r.get("cpu_util", 0.0) for r in runs if r.get("cpu_util") is not None])
        rows.append((tag, len(tps_vals), f"{m:.1f}", f"{s:.1f}", f"{delta:+.2f}%",
                     f"{gpu:.1f}", f"{cpu:.1f}"))

    print(f"# HWC1 Round-1 results — baseline mean={base_mean:.1f} std={base_std:.1f} (N={len(base_tps)})\n")
    print("| Tag | N | mean tps | std | Δ% vs baseline | GPU% | CPU% |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")

    # noise floor info
    if base_std and base_mean:
        rel_noise = base_std / base_mean * 100
        print(f"\n_baseline noise: relative std = {rel_noise:.2f}% of mean — accept gate ≥ +3% × √2 ≈ +{max(3, rel_noise*1.5):.1f}%_")


if __name__ == "__main__":
    main()
