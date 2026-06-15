#!/usr/bin/env python3
"""Option C Step 3 aggregator — produces wd1_results.md from sweep JSONs.

Reads ``lhc_phase4/optionC/runs/wd1_{config}_s{n}_bench.json`` and outputs
mean ± std + paired Δ% vs vanilla for the W-D1 configs.

Reused for baseline_regression aggregation by setting --tag prefix.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

METRICS = [
    ("output_throughput", "output tok/s", "hi"),
    ("request_throughput", "req/s", "hi"),
    ("mean_ttft_ms", "TTFT mean (ms)", "lo"),
    ("p99_ttft_ms", "TTFT p99 (ms)", "lo"),
    ("mean_tpot_ms", "TPOT mean (ms)", "lo"),
    ("p99_tpot_ms", "TPOT p99 (ms)", "lo"),
    ("duration", "duration (s)", "lo"),
]


def load_runs(runs_dir: Path, prefix: str):
    """Load all <prefix>_<config>_s<n>_bench.json files and group by config."""
    by_config = defaultdict(list)
    for f in sorted(runs_dir.glob(f"{prefix}_*_bench.json")):
        stem = f.stem  # e.g. wd1_vanilla_s1_bench
        # stem is "<prefix>_<config>_s<n>_bench"
        rest = stem[len(prefix) + 1 : -len("_bench")]
        # rest = "<config>_s<n>"
        if "_s" not in rest:
            continue
        config, sweep = rest.rsplit("_s", 1)
        try:
            sweep_n = int(sweep)
        except ValueError:
            continue
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        by_config[config].append((sweep_n, data))
    return by_config


def stats(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(values) / n
    if n > 1:
        var = sum((v - m) ** 2 for v in values) / (n - 1)
        s = math.sqrt(var)
    else:
        s = 0.0
    return m, s, n


def delta_pct(test, base, direction):
    if base == 0:
        return float("nan")
    raw = (test - base) / base * 100.0
    return raw if direction == "hi" else -raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).parent / "runs"))
    ap.add_argument("--prefix", default="wd1")
    ap.add_argument("--baseline", default="vanilla")
    ap.add_argument("--out", default=None,
                    help="output markdown (default <runs>/../<prefix>_results.md)")
    args = ap.parse_args()
    runs_dir = Path(args.runs)
    by_config = load_runs(runs_dir, args.prefix)
    if not by_config:
        print(f"no bench JSONs found under {runs_dir} with prefix {args.prefix}")
        return

    out_path = (
        Path(args.out)
        if args.out
        else runs_dir.parent / f"{args.prefix}_results.md"
    )

    md = [f"# Option C — {args.prefix} sweep results", ""]
    md.append(f"Source: `{runs_dir}` ({sum(len(v) for v in by_config.values())} cells)")
    md.append("")

    # Header
    configs = sorted(by_config.keys())
    base_runs = by_config.get(args.baseline)
    if base_runs is None:
        md.append(f"> baseline `{args.baseline}` not found; reporting raw only")

    md.append("## Mean ± std (n sweeps)")
    md.append("")
    md.append(
        "| config | " + " | ".join(lbl for _, lbl, _ in METRICS) + " | n |"
    )
    md.append(
        "|" + "|".join(["---"] * (len(METRICS) + 2)) + "|"
    )
    base_means = {}
    for config in configs:
        rows = by_config[config]
        cells = []
        for key, _, _ in METRICS:
            vals = [r[1].get(key) for r in rows if r[1].get(key) is not None]
            m, s, n = stats(vals)
            cells.append(f"{m:.2f} ± {s:.2f}")
        n_sweeps = len(rows)
        if config == args.baseline:
            base_means = {
                key: stats([r[1].get(key) for r in rows
                            if r[1].get(key) is not None])[0]
                for key, _, _ in METRICS
            }
        md.append(f"| {config} | " + " | ".join(cells) + f" | {n_sweeps} |")
    md.append("")

    # Paired Δ%
    if base_means:
        md.append(f"## Δ% vs {args.baseline} (higher = better)")
        md.append("")
        md.append("| config | " + " | ".join(lbl for _, lbl, _ in METRICS) + " |")
        md.append("|" + "|".join(["---"] * (len(METRICS) + 1)) + "|")
        for config in configs:
            if config == args.baseline:
                continue
            rows = by_config[config]
            cells = []
            for key, _, direction in METRICS:
                vals = [r[1].get(key) for r in rows if r[1].get(key) is not None]
                m, _, _ = stats(vals)
                b = base_means.get(key, 0.0)
                d = delta_pct(m, b, direction) if b else float("nan")
                cells.append(f"{d:+.2f}%")
            md.append(f"| {config} | " + " | ".join(cells) + " |")
        md.append("")

    md.append("## Per-sweep raw")
    md.append("")
    for config in configs:
        md.append(f"### {config}")
        md.append("")
        md.append(
            "| sweep | output tok/s | req/s | TTFT mean (ms) | TPOT p99 (ms) | duration (s) |"
        )
        md.append("|---|---|---|---|---|---|")
        for sweep, data in sorted(by_config[config]):
            md.append(
                f"| {sweep} | "
                f"{data.get('output_throughput', float('nan')):.2f} | "
                f"{data.get('request_throughput', float('nan')):.4f} | "
                f"{data.get('mean_ttft_ms', float('nan')):.2f} | "
                f"{data.get('p99_tpot_ms', float('nan')):.2f} | "
                f"{data.get('duration', float('nan')):.2f} |"
            )
        md.append("")

    out_path.write_text("\n".join(md))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
