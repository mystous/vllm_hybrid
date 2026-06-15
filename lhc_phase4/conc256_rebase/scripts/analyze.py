#!/usr/bin/env python3
"""Aggregate conc256 rebase sweep JSONs into a per-config × per-workload table.

For each config (vanilla, optA, path1, optC, stack), reads vanilla_runs/optionA_runs/
path1_runs/optionC_runs/stack_runs and emits:
  - mean ± std + CV per (config, workload, metric)
  - Δ% vs vanilla baseline per (config, workload) for output_throughput
  - 95% CI half-width (using Student t, df=n-1)
"""
from __future__ import annotations
import json, math, statistics, sys
from pathlib import Path

BASE = Path("/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase")
CONFIGS = [
    ("vanilla", "vanilla_runs", "vanilla_"),
    ("optA",    "optionA_runs", "optA_"),
    ("path1",   "path1_runs",   "path1_"),
    ("optC",    "optionC_runs", "optC_"),
    ("stack",   "stack_runs",   "stack_"),
]
WORKLOADS = ["sonnet", "chat", "code", "balanced", "sonnet-heavy", "code-heavy"]

# Student t 95% two-sided critical values for df=1..30 (good enough for our small n).
T95 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,
       9:2.262,10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,
       16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,29:2.045,30:2.042}

def load_config(dir_name: str, prefix: str) -> dict[str, list[dict]]:
    """workload -> list of run dicts."""
    out = {w: [] for w in WORKLOADS}
    d = BASE / dir_name
    if not d.is_dir():
        return out
    for p in sorted(d.glob(f"{prefix}*.json")):
        # filename like vanilla_sonnet-heavy_s1.json
        stem = p.stem[len(prefix):]
        # split off _s{n}
        if "_s" not in stem:
            continue
        workload = stem.rsplit("_s", 1)[0]
        if workload not in out:
            continue
        try:
            j = json.loads(p.read_text())
        except Exception as e:
            print(f"warn: bad json {p}: {e}", file=sys.stderr)
            continue
        out[workload].append(j)
    return out

def summarize(values: list[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "cv_pct": float("nan"), "ci95_half": float("nan")}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n >= 2 else 0.0
    cv = (std / mean * 100.0) if mean else float("nan")
    if n >= 2:
        df = n - 1
        # Find nearest tabled df
        keys = sorted(T95.keys())
        t = T95[min(keys, key=lambda k: abs(k - df))]
        ci = t * std / math.sqrt(n)
    else:
        ci = float("nan")
    return {"n": n, "mean": mean, "std": std, "cv_pct": cv, "ci95_half": ci}

def main():
    data = {name: load_config(d, p) for name, d, p in CONFIGS}

    # Build per-(config, workload) summary on output_throughput.
    rows = []
    base = data["vanilla"]
    base_summ = {w: summarize([r["output_throughput"] for r in base[w]]) for w in WORKLOADS}

    print("\n=== conc=256 rebase: output_throughput tok/s, per (config × workload) ===\n")
    header = f"{'config':<10} {'workload':<14} {'n':>3} {'mean':>10} {'std':>8} {'cv%':>6} {'±CI95':>8} {'Δ% vs van':>10}"
    print(header)
    print("-" * len(header))
    for name, _, _ in CONFIGS:
        for w in WORKLOADS:
            runs = data[name][w]
            s = summarize([r["output_throughput"] for r in runs])
            base_mean = base_summ[w]["mean"]
            if name == "vanilla" or base_mean != base_mean or s["mean"] != s["mean"]:
                delta = "—"
            else:
                d = (s["mean"] - base_mean) / base_mean * 100.0
                delta = f"{d:+.2f}"
            row = (f"{name:<10} {w:<14} {s['n']:>3} "
                   f"{s['mean']:>10.1f} {s['std']:>8.1f} "
                   f"{s['cv_pct']:>6.2f} {s['ci95_half']:>8.1f} "
                   f"{delta:>10}")
            print(row)
            rows.append({"config": name, "workload": w, **s, "delta_pct_vs_vanilla": delta})

    # Write JSON
    out = BASE / "AGGREGATED.json"
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\n[wrote] {out}")

    # Top-line summary across workloads (mean of Δ%)
    print("\n=== mean Δ% (avg over workloads with both vanilla+config data) ===")
    for name, _, _ in CONFIGS:
        if name == "vanilla":
            continue
        deltas = []
        for w in WORKLOADS:
            b = base_summ[w]["mean"]
            runs = data[name][w]
            s = summarize([r["output_throughput"] for r in runs])
            if b == b and s["mean"] == s["mean"]:
                deltas.append((s["mean"] - b) / b * 100.0)
        if deltas:
            print(f"  {name:<10} n_wl={len(deltas)}  mean Δ% = {statistics.mean(deltas):+.2f}  "
                  f"min={min(deltas):+.2f}  max={max(deltas):+.2f}")
        else:
            print(f"  {name:<10} no data")

if __name__ == "__main__":
    main()
