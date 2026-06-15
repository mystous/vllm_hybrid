#!/usr/bin/env python3
"""Extract regime classification stats from boot logs.

The regime detector logs `[LHC Regime] init: adaptive=...` once. We instead
infer regime decisions from the scheduler step end + KV cache usage reports
(`GPU KV cache usage: N.N%`) in boot logs, and classify the same way
``classify()`` does, producing a regime distribution per run.

Output: ``regime_accuracy.md`` with per-config GPU/KV/regime breakdown.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


KV_RE = re.compile(r"GPU KV cache usage: ([\d.]+)%")
# Engine logger line shows running + waiting too — useful for util proxy:
RUN_RE = re.compile(r"Running: (\d+) reqs, Waiting: (\d+) reqs")
# Avg generation throughput stays high under GPU-bound load.
GEN_RE = re.compile(r"Avg generation throughput: ([\d.]+)")


def classify(gpu_proxy: float, kv_pct: float, swap_proxy: float):
    """Mirror regime_detector.classify rule. gpu_proxy uses gen-tput
    saturation (1.0 = ≥2000 tok/s for 8B). swap_proxy is 1 if recent
    KV pressure observed else 0."""
    if kv_pct > 0.75 or swap_proxy > 10.0:
        return "KV_HEAVY"
    if gpu_proxy > 0.90 and kv_pct < 0.50 and swap_proxy < 1.0:
        return "GPU_SATURATED"
    return "BALANCED"


def parse_log(path: Path):
    out = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            kv_m = KV_RE.search(line)
            if not kv_m:
                continue
            kv_pct = float(kv_m.group(1)) / 100.0
            gen_m = GEN_RE.search(line)
            gen = float(gen_m.group(1)) if gen_m else 0.0
            run_m = RUN_RE.search(line)
            waiting = int(run_m.group(2)) if run_m else 0
            # gpu_proxy: ~2000 tok/s = saturated for 8B@TP8.
            gpu_proxy = min(gen / 2000.0, 1.0)
            # swap_proxy: KV > 70% used or waiting queue building.
            swap_proxy = 50.0 if (kv_pct > 0.70 or waiting > 4) else 0.0
            regime = classify(gpu_proxy, kv_pct, swap_proxy)
            out.append({
                "gen_tps": gen, "kv_pct": kv_pct, "waiting": waiting,
                "gpu_proxy": gpu_proxy, "regime": regime,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="lhc_phase4/optionC/runs")
    ap.add_argument("--out", default="lhc_phase4/optionC/regime_accuracy.md")
    args = ap.parse_args()
    runs = Path(args.runs)

    # group log files by (prefix, config)
    groups = defaultdict(list)
    for p in sorted(runs.glob("*_boot.log")):
        stem = p.stem  # tag_boot
        if not stem.endswith("_boot"):
            continue
        tag = stem[:-5]
        # Decide prefix vs config
        # wd1 sweep: wd1_<config>_s<n>
        # baseline regression: bl_<workload>_<config>_s<n>
        groups[tag].append(p)

    md = ["# Option C — Regime classification accuracy", ""]
    md.append("Per-run distribution of regime classifications inferred from")
    md.append("boot-log scheduler `Engine 000:` lines (KV%, gen tps, waiting).")
    md.append("")
    md.append("| run | samples | GPU_SAT | KV_HEAVY | BALANCED | mean KV% | mean gen tps |")
    md.append("|---|---|---|---|---|---|---|")
    for tag, files in sorted(groups.items()):
        all_samples = []
        for p in files:
            all_samples.extend(parse_log(p))
        if not all_samples:
            continue
        n = len(all_samples)
        c = Counter(s["regime"] for s in all_samples)
        mean_kv = sum(s["kv_pct"] for s in all_samples) / n
        mean_gen = sum(s["gen_tps"] for s in all_samples) / n
        md.append(
            f"| {tag} | {n} | "
            f"{c.get('GPU_SATURATED',0)/n*100:.1f}% | "
            f"{c.get('KV_HEAVY',0)/n*100:.1f}% | "
            f"{c.get('BALANCED',0)/n*100:.1f}% | "
            f"{mean_kv*100:.2f}% | "
            f"{mean_gen:.0f} |"
        )

    Path(args.out).write_text("\n".join(md))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
