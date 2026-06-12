#!/usr/bin/env python3
from __future__ import annotations
import glob, json, os, re, statistics

RUNS_R1 = "/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/hw_custom_round_1/runs"
RUNS = "/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/hw_custom_round_3/runs"
PAT = re.compile(r"^(.+)_s(\d+)\.json$")


def load(p):
    tags = {}
    for fp in sorted(glob.glob(os.path.join(p, "*_s*.json"))):
        m = PAT.match(os.path.basename(fp))
        if not m: continue
        with open(fp) as f:
            tags.setdefault(m.group(1), []).append(json.load(f))
    return tags


def stat(v):
    if not v: return float("nan"), float("nan")
    if len(v) == 1: return v[0], 0.0
    return statistics.mean(v), statistics.stdev(v)


r1 = load(RUNS_R1)
r3 = load(RUNS)
base_tps = [r.get("output_tps") for r in r1.get("baseline", []) if r.get("output_tps")]
fp8_tps = [r.get("output_tps") for r in r1.get("h8_kv_fp8", []) if r.get("output_tps")]
base_mean, _ = stat(base_tps)
fp8_mean, _ = stat(fp8_tps)
print(f"# HWC3 — vs baseline {base_mean:.0f}, vs fp8 {fp8_mean:.0f}")
print("")
print("| Tag | N | mean | std | Δ% base | Δ% fp8 | GPU% |")
print("|---|---|---|---|---|---|---|")
for tag in sorted(r3.keys()):
    runs = r3[tag]
    tps = [r.get("output_tps") for r in runs if r.get("output_tps")]
    if not tps:
        print(f"| {tag} | boot_fail | - | - | - | - | - |"); continue
    m, s = stat(tps)
    d1 = (m - base_mean) / base_mean * 100
    d2 = (m - fp8_mean) / fp8_mean * 100
    gpu = statistics.mean([r.get("gpu_util", 0.0) for r in runs if r.get("gpu_util") is not None])
    print(f"| {tag} | {len(tps)} | {m:.1f} | {s:.1f} | {d1:+.2f}% | {d2:+.2f}% | {gpu:.1f} |")
