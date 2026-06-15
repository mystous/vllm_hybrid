#!/usr/bin/env python3
"""LHC Phase 4 — aggregate METRONOME-LHC sweep results.

Reads ``lhc_phase4/runs/<workload>_<config>_<sweep>_bench.json`` and emits:
  - lhc_phase4/metronome_lhc_throughput.md  (markdown table)
  - lhc_phase4/metronome_lhc_throughput.csv (raw)
  - paper/tables/tbl_metronome_lhc.tex      (LaTeX 9 × 7 Delta% table)

The bench json is vllm's standard ``vllm bench serve`` output.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"

WORKLOADS = [
    "sonnet", "chat", "code", "balanced", "sonnet-heavy", "code-heavy",
    "wd1", "wd2", "wd3",
]
CONFIGS = [
    "vanilla", "dsa", "amx_c3", "dsa_amx",
    "metronome", "metronome_sfx", "metronome_full",
]


def _load_one(path: Path) -> dict | None:
    try:
        with open(path) as f:
            j = json.load(f)
        # vllm bench serve: top-level key 'output_throughput' or similar.
        tput = (
            j.get("output_throughput")
            or j.get("throughput")
            or j.get("request_throughput")
        )
        if tput is None:
            return None
        return {
            "tput": float(tput),
            "duration": float(j.get("duration", 0)),
            "input_tokens": int(j.get("total_input_tokens", 0)),
            "output_tokens": int(j.get("total_output_tokens", 0)),
        }
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def collect():
    cells: dict[tuple[str, str], list[float]] = {}
    for w in WORKLOADS:
        for c in CONFIGS:
            tputs = []
            for sweep in (1, 2, 3, 4, 5):
                p = RUNS / f"{w}_{c}_{sweep}_bench.json"
                rec = _load_one(p)
                if rec is None:
                    continue
                tputs.append(rec["tput"])
            if tputs:
                cells[(w, c)] = tputs
    return cells


def fmt_delta(mean: float, base: float) -> tuple[float, str]:
    if base <= 0:
        return 0.0, "--"
    d = 100.0 * (mean - base) / base
    return d, f"{d:+.2f}\\%"


def write_csv(cells, out: Path):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "workload", "config", "n_sweeps", "mean_tput", "stdev_tput",
            "delta_pct_vs_vanilla",
        ])
        for wl in WORKLOADS:
            base_list = cells.get((wl, "vanilla"))
            if not base_list:
                continue
            base = statistics.mean(base_list)
            for cf in CONFIGS:
                t = cells.get((wl, cf))
                if not t:
                    continue
                mean = statistics.mean(t)
                sd = statistics.stdev(t) if len(t) > 1 else 0.0
                d = 100.0 * (mean - base) / base if base > 0 else 0.0
                w.writerow([
                    wl, cf, len(t), f"{mean:.2f}", f"{sd:.2f}", f"{d:+.2f}",
                ])


def write_md(cells, out: Path):
    lines = ["# METRONOME-LHC throughput sweep (LHC_P4_005)", ""]
    lines.append(
        "Mean ± std (tok/s, output throughput) and Δ% vs vanilla.\n"
    )
    head = "| workload | " + " | ".join(CONFIGS) + " |"
    sep = "|" + "---|" * (len(CONFIGS) + 1)
    lines += [head, sep]
    for wl in WORKLOADS:
        base = cells.get((wl, "vanilla"))
        if not base:
            continue
        bm = statistics.mean(base)
        row = [wl]
        for cf in CONFIGS:
            t = cells.get((wl, cf))
            if not t:
                row.append("--")
                continue
            m = statistics.mean(t)
            sd = statistics.stdev(t) if len(t) > 1 else 0.0
            d = 100.0 * (m - bm) / bm if bm > 0 else 0.0
            if cf == "vanilla":
                row.append(f"{m:.0f}±{sd:.0f}")
            else:
                row.append(f"{m:.0f}±{sd:.0f} ({d:+.1f}%)")
        lines.append("| " + " | ".join(row) + " |")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_tex(cells, out: Path):
    cols = "l" + "r" * len(CONFIGS)
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{METRONOME-LHC 9 워크로드 $\times$ 7 config "
        r"throughput. mean tok/s (3-sweep) 와 $\Delta\%$ vs vanilla. "
        r"LHC\_P4\_005 측정.}"
    )
    lines.append(r"\label{tbl:metronome-lhc}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + cols + r"}")
    lines.append(r"\toprule")
    headers = ["workload"] + [c.replace("_", r"\_") for c in CONFIGS]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for wl in WORKLOADS:
        base = cells.get((wl, "vanilla"))
        if not base:
            continue
        bm = statistics.mean(base)
        row = [wl.replace("_", r"\_")]
        for cf in CONFIGS:
            t = cells.get((wl, cf))
            if not t:
                row.append("--")
                continue
            m = statistics.mean(t)
            d = 100.0 * (m - bm) / bm if bm > 0 else 0.0
            if cf == "vanilla":
                row.append(f"{m:.0f}")
            else:
                row.append(f"{m:.0f} ({d:+.1f}\\%)")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    cells = collect()
    if not cells:
        print("no bench results found in", RUNS, file=sys.stderr)
        sys.exit(1)
    write_csv(cells, ROOT / "metronome_lhc_throughput.csv")
    write_md(cells, ROOT / "metronome_lhc_throughput.md")
    tex_out = ROOT.parent / "paper" / "tables" / "tbl_metronome_lhc.tex"
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    write_tex(cells, tex_out)
    print("wrote:", ROOT / "metronome_lhc_throughput.csv")
    print("wrote:", ROOT / "metronome_lhc_throughput.md")
    print("wrote:", tex_out)


if __name__ == "__main__":
    main()
