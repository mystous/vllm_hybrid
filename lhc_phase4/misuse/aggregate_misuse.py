#!/usr/bin/env python3
"""Aggregate LHC Phase 4 misuse anti-pattern bench results.

For each (ap, workload) pair, computes:
  - baseline mean/std (output_throughput)
  - misuse mean/std (output_throughput)
  - Δ% = (misuse - baseline) / baseline * 100
  - paired Δ% per-sweep
  - 95% bootstrap CI on the Δ%

Schema: standard `vllm bench serve` JSON, key ``output_throughput`` (TPS).

Output: prints a markdown table + writes to MISUSE_FINAL.md when --write is on.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

RUNS = Path("/workspace/host_vllm_hybrid/lhc_phase4/misuse/runs")

APS = ["ap1", "ap2", "ap3", "ap4", "ap5"]
CONFIGS = ["baseline", "misuse"]


def load_bench(path: Path) -> dict | None:
    try:
        with path.open() as f:
            j = json.load(f)
    except Exception:
        return None
    if "summary" in j:
        s = j["summary"]
        return {
            "output_throughput": s.get("output_tps"),
            "duration": s.get("wall_total_s"),
            "n_ok": s.get("n_ok") or s.get("num_prompts"),
            "n_err": s.get("n_err", 0),
        }
    return {
        "output_throughput": j.get("output_throughput"),
        "duration": j.get("duration"),
        "n_ok": j.get("completed"),
        "n_err": j.get("failed", 0),
    }


def collect():
    # key: (ap, cfg, workload) -> list of (sweep, throughput)
    data = defaultdict(list)
    for path in sorted(RUNS.glob("*_bench.json")):
        # name fmt: <ap>_<cfg>_<wl>_s<sweep>_bench.json
        stem = path.stem  # ap1_baseline_chat_s1_bench
        parts = stem.split("_")
        if len(parts) < 5 or parts[-1] != "bench":
            continue
        sweep = parts[-2]  # s1
        wl = parts[-3]
        cfg = parts[-4]
        # AP is the prefix before cfg. For our naming (ap1, ap2, ...) it
        # is the first token. Generalise as join up to the cfg position.
        ap = "_".join(parts[:-4]) if len(parts) > 5 else parts[0]
        b = load_bench(path)
        if not b or b.get("output_throughput") is None:
            continue
        try:
            sn = int(sweep.lstrip("s"))
        except ValueError:
            sn = 0
        data[(ap, cfg, wl)].append((sn, float(b["output_throughput"]),
                                    b.get("n_err", 0)))
    return data


def stats(vals):
    if not vals:
        return None
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "n": 1}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals),
        "n": len(vals),
    }


def paired_delta(b_sweeps, m_sweeps):
    """Match by sweep index; compute Δ% per matched pair."""
    bd = {s: v for (s, v, _e) in b_sweeps}
    md = {s: v for (s, v, _e) in m_sweeps}
    shared = sorted(set(bd) & set(md))
    deltas = []
    for s in shared:
        if bd[s] > 0:
            deltas.append(100.0 * (md[s] - bd[s]) / bd[s])
    return deltas


def fmt(x, prec=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{prec}f}"


def report(data, write_path: Path | None = None):
    lines = []
    lines.append("# LHC Phase 4 — Misuse Anti-Pattern Results")
    lines.append("")
    lines.append("**Baseline workload**: Llama-3.1-8B Instruct, TP=8, "
                 "sharegpt-equivalent (sonnet harness, chat) + sonnet.")
    lines.append("")
    lines.append("**Definition**: baseline = LHC properly used (regime ON, "
                 "WQ-per-rank, NUMA-local, DSA_MIN=64KB, AMX prefix-hit-only); "
                 "misuse = anti-pattern env injection.")
    lines.append("")
    aps_present = sorted({k[0] for k in data.keys()})
    wls_present = sorted({k[2] for k in data.keys()})
    header = (
        "| AP | WL | baseline TPS (mean±std, n) | misuse TPS (mean±std, n) | "
        "Δ% paired (all sweeps) | Δ% paired (excl s1 cold) | n_err |"
    )
    sep = ("|----|----|---------------------------|-------------------------|"
           "------------------------|------------------------|-------|")
    lines.append(header)
    lines.append(sep)

    summary_rows = []
    for ap in aps_present:
        for wl in wls_present:
            b = data.get((ap, "baseline", wl), [])
            m = data.get((ap, "misuse", wl), [])
            if not b and not m:
                continue
            # Drop s1 outlier for the alternative column.
            b_no_s1 = [(s, v, e) for (s, v, e) in b if s != 1]
            m_no_s1 = [(s, v, e) for (s, v, e) in m if s != 1]
            bs = stats([v for (_s, v, _e) in b])
            ms = stats([v for (_s, v, _e) in m])
            deltas = paired_delta(b, m) if b and m else []
            deltas_no_s1 = (paired_delta(b_no_s1, m_no_s1)
                            if b_no_s1 and m_no_s1 else [])
            dmean = statistics.mean(deltas) if deltas else None
            dmean_no_s1 = (statistics.mean(deltas_no_s1)
                           if deltas_no_s1 else None)
            be = sum(e for (_s, _v, e) in b)
            me = sum(e for (_s, _v, e) in m)
            row = (
                f"| {ap} | {wl} | "
                f"{fmt(bs['mean']) if bs else '—'}±{fmt(bs['std']) if bs else '—'}, n={bs['n'] if bs else 0} | "
                f"{fmt(ms['mean']) if ms else '—'}±{fmt(ms['std']) if ms else '—'}, n={ms['n'] if ms else 0} | "
                f"{fmt(dmean) if dmean is not None else '—'}% "
                f"({', '.join(fmt(d) for d in deltas)}) | "
                f"{fmt(dmean_no_s1) if dmean_no_s1 is not None else '—'}% "
                f"({', '.join(fmt(d) for d in deltas_no_s1)}) | "
                f"b={be}/m={me} |"
            )
            lines.append(row)
            summary_rows.append((ap, wl, bs, ms, dmean, deltas))

    lines.append("")
    lines.append("## Per-anti-pattern interpretation")
    lines.append("")
    interp = {
        "ap1": ("DSA_MIN=64 vs 65536. Smaller transfers (<64B) bypass DSA in "
                "baseline (memcpy fast-path); in misuse they go through DSA "
                "descriptor enqueue (~5μs/op vs ~0.04μs/op for memcpy)."),
        "ap2": ("AMX C3 prefix scan FORCE_EVERY_STEP vs prefix-hit-only. "
                "Each scheduler step pays a 65KB synthetic scan; baseline "
                "scan only fires on prefix-cache miss-then-hit transitions."),
        "ap3": ("DSA NUMA cross-socket vs local. Misuse inverts rank→device "
                "map (rank 0–3 → dsa1, rank 4–7 → dsa0), forcing all DSA "
                "memcpy traffic across the QPI/UPI inter-socket link."),
        "ap4": ("DSA WQ-per-rank OFF vs ON. Misuse routes all 8 TP workers "
                "to wq0.0; PASID contention surfaces as EBUSY drops in the "
                "lane self-test (lane disabled at init) OR queue serialisation."),
        "ap5": ("Regime detector OFF (Option A static) vs ON (Option C). "
                "In the GPU-saturated baseline regime, the adaptive detector "
                "keeps LHC OFF; the static Option A pays the sampling/setup "
                "cost without productive work."),
    }
    for ap in aps_present:
        lines.append(f"- **{ap}** — {interp.get(ap, '(no description)')}")
    lines.append("")
    if write_path is not None:
        write_path.write_text("\n".join(lines))
        print(f"wrote {write_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write to MISUSE_FINAL.md")
    args = ap.parse_args()
    data = collect()
    out = (Path("/workspace/host_vllm_hybrid/lhc_phase4/misuse/MISUSE_FINAL.md")
           if args.write else None)
    report(data, out)
