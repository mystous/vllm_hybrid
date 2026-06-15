#!/usr/bin/env python3
"""Aggregate Phase 4 conc256 rebase final tables (Step A/B/C).

Outputs:
  PHASE4_FINAL_aggregated.json
  printed Markdown-style tables.

Schema:
  - Step A baseline (5sw vs path1/optionC/stack 5sw on code workload)
  - Step B code-variant (python/rust/json, vanilla/path1/stack, 3sw)
  - Step C optionC_v2 (6 workloads, vbw vs optCv2, 3sw)

JSON loading robustly handles two output formats:
  - benchmark_workloads.py: {"summary": {"output_tps": X, ...}}
  - vllm bench serve:       {"output_throughput": X, ...}
"""
from __future__ import annotations
import json, math, statistics, sys
from pathlib import Path

BASE = Path("/workspace/host_vllm_hybrid/lhc_phase4/conc256_rebase")

# Student t two-sided 95% — small df.
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    20: 2.086, 29: 2.045, 30: 2.042,
}


def load_tput(path: Path) -> float | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    if "summary" in d:
        return d["summary"].get("output_tps")
    if "output_throughput" in d:
        return d["output_throughput"]
    return None


def summarize(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "ci95_half": float("nan"), "cv_pct": float("nan")}
    m = statistics.mean(values)
    s = statistics.stdev(values) if n >= 2 else 0.0
    cv = (s / m * 100.0) if m else float("nan")
    if n >= 2:
        df = n - 1
        keys = sorted(T95.keys())
        t = T95[min(keys, key=lambda k: abs(k - df))]
        ci = t * s / math.sqrt(n)
    else:
        ci = float("nan")
    return {"n": n, "mean": m, "std": s, "cv_pct": cv, "ci95_half": ci}


def delta_with_ci(test_vals: list[float], baseline_vals: list[float]) -> dict:
    """Return Δ% (mean ratio) and 95% CI (Welch-style approximation)."""
    a = summarize(test_vals)
    b = summarize(baseline_vals)
    if not a["n"] or not b["n"] or b["mean"] == 0:
        return {**a, "baseline_mean": b["mean"], "delta_pct": float("nan"),
                "delta_pct_ci_half": float("nan")}
    delta = (a["mean"] - b["mean"]) / b["mean"] * 100.0
    # half-width on delta using sqrt(var_a/n_a + var_b/n_b)
    var_a = (a["std"] ** 2) / max(a["n"], 1)
    var_b = (b["std"] ** 2) / max(b["n"], 1)
    se = math.sqrt(var_a + var_b)
    # use t (df=min-1)
    df = min(a["n"], b["n"]) - 1
    keys = sorted(T95.keys())
    t = T95[min(keys, key=lambda k: abs(k - df))] if df > 0 else float("nan")
    half = (t * se) / b["mean"] * 100.0 if df > 0 else float("nan")
    return {"n_test": a["n"], "n_base": b["n"],
            "test_mean": a["mean"], "base_mean": b["mean"],
            "test_std": a["std"], "base_std": b["std"],
            "delta_pct": delta, "delta_pct_ci_half": half}


def step_A_summary() -> dict:
    """Path 1 / optionC / stack code workload 5-sweep summary."""
    # baseline: 5-sweep vanilla_bw code
    baseline = []
    for s in range(1, 6):
        v = load_tput(BASE / "vanilla_bw_runs" / f"vbw_code_s{s}_bench.json")
        if v is not None:
            baseline.append(v)
    # path1 / optionC / stack: 3sw (existing) + 2sw (precision_runs)
    configs = {
        "path1":   ("path1_runs",   "path1_code",   "path1_code"),
        "optionC": ("optionC_runs", "optionC_code", "optionC_code"),
        "stack":   ("stack_runs",   "stack_code",   "stack_code"),
    }
    out = {"baseline_code_vanilla_bw": summarize(baseline),
           "results": {}}
    for name, (old_dir, old_stem, new_stem) in configs.items():
        vals = []
        for s in range(1, 4):
            v = load_tput(BASE / old_dir / f"{old_stem}_s{s}_bench.json")
            if v is not None:
                vals.append(v)
        for s in range(4, 6):
            v = load_tput(BASE / "precision_runs" / f"{new_stem}_s{s}_bench.json")
            if v is not None:
                vals.append(v)
        out["results"][name] = {
            "values_5sw": vals,
            "summary": summarize(vals),
            "delta_vs_baseline": delta_with_ci(vals, baseline),
        }
    return out


def step_B_summary() -> dict:
    """Code variant python/rust/json across vanilla/path1/stack 3sw."""
    out = {"results": {}}
    for variant in ["python", "rust", "json"]:
        out["results"][variant] = {}
        for cfg in ["vanilla", "path1", "stack"]:
            vals = []
            for s in range(1, 4):
                v = load_tput(
                    BASE / "code_variant_runs" /
                    f"{variant}_{cfg}_code_s{s}_bench.json"
                )
                if v is not None:
                    vals.append(v)
            out["results"][variant][cfg] = {
                "values": vals, "summary": summarize(vals),
            }
        # delta vs vanilla within variant
        van = out["results"][variant]["vanilla"]["values"]
        for cfg in ["path1", "stack"]:
            test = out["results"][variant][cfg]["values"]
            out["results"][variant][cfg]["delta_vs_vanilla"] = \
                delta_with_ci(test, van)
    return out


def step_C_summary() -> dict:
    """optionC_v2 across 6 workloads × {vbw, optCv2} × 3sw."""
    out = {"results": {}}
    workloads = ["sonnet", "chat", "code", "balanced",
                 "sonnet-heavy", "code-heavy"]
    for w in workloads:
        out["results"][w] = {}
        for cfg in ["vbw", "optCv2"]:
            vals = []
            for s in range(1, 4):
                v = load_tput(
                    BASE / "optionC_v2_runs" / f"{cfg}_{w}_s{s}_bench.json"
                )
                if v is not None:
                    vals.append(v)
            out["results"][w][cfg] = {
                "values": vals, "summary": summarize(vals),
            }
        van = out["results"][w]["vbw"]["values"]
        test = out["results"][w]["optCv2"]["values"]
        out["results"][w]["delta_vs_vbw"] = delta_with_ci(test, van)
    return out


def print_step_A(d: dict) -> None:
    print("\n=== Step A — 5-sweep code-workload precision (vs vanilla_bw 5sw) ===")
    b = d["baseline_code_vanilla_bw"]
    print(f"  vanilla_bw (n={b['n']}):  mean {b['mean']:.1f} ± {b['ci95_half']:.1f} tok/s")
    print(f"{'config':<8} {'n':>3} {'mean':>10} {'std':>8} {'CI95±':>8} "
          f"{'Δ%':>8} {'±CI%':>8}")
    print("-" * 60)
    for name, r in d["results"].items():
        s = r["summary"]
        dlt = r["delta_vs_baseline"]
        print(f"{name:<8} {s['n']:>3} {s['mean']:>10.1f} {s['std']:>8.1f} "
              f"{s['ci95_half']:>8.1f} {dlt['delta_pct']:>+7.2f}% "
              f"{dlt['delta_pct_ci_half']:>7.2f}%")


def print_step_B(d: dict) -> None:
    print("\n=== Step B — code variant generalization (3-sweep each) ===")
    print(f"{'variant':<8} {'config':<8} {'n':>3} {'mean':>10} {'std':>8} "
          f"{'Δ% vs van':>10} {'±CI%':>8}")
    print("-" * 70)
    for variant, cfgs in d["results"].items():
        for cfg in ["vanilla", "path1", "stack"]:
            s = cfgs[cfg]["summary"]
            if cfg == "vanilla":
                dlt_str, ci_str = "—", "—"
            else:
                dlt = cfgs[cfg]["delta_vs_vanilla"]
                dlt_str = f"{dlt['delta_pct']:+.2f}%"
                ci_str = f"{dlt['delta_pct_ci_half']:.2f}%"
            print(f"{variant:<8} {cfg:<8} {s['n']:>3} {s['mean']:>10.1f} "
                  f"{s['std']:>8.1f} {dlt_str:>10} {ci_str:>8}")


def print_step_C(d: dict) -> None:
    print("\n=== Step C — optionC_v2 (PREFIX_HOT adaptive) per workload (3sw) ===")
    print(f"{'workload':<13} {'vbw mean':>10} {'optCv2 mean':>12} "
          f"{'Δ%':>8} {'±CI%':>8}")
    print("-" * 60)
    for w, r in d["results"].items():
        v = r["vbw"]["summary"]
        o = r["optCv2"]["summary"]
        dlt = r["delta_vs_vbw"]
        print(f"{w:<13} {v['mean']:>10.1f} {o['mean']:>12.1f} "
              f"{dlt['delta_pct']:>+7.2f}% {dlt['delta_pct_ci_half']:>7.2f}%")


def main():
    a = step_A_summary()
    b = step_B_summary()
    c = step_C_summary()
    print_step_A(a)
    print_step_B(b)
    print_step_C(c)
    out = BASE / "PHASE4_FINAL_aggregated.json"
    out.write_text(json.dumps({"step_A": a, "step_B": b, "step_C": c}, indent=2))
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
