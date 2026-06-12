#!/usr/bin/env python3
"""Aggregate hw_heavy_* sweep JSONs into a per-case summary row.

For each *_s{1..N}.json file with the same tag, computes:
    n_ok_sweeps, output_tps mean/std, ttft_p50 mean, tpot_p50 mean,
    accept_rate (if present), GPU util, CPU util.

Usage:
    python analyze.py --runs /path/to/runs --out summary.json [--out-csv summary.csv]
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict


def fmt(x, n=1):
    if x is None: return None
    if isinstance(x, float):
        if math.isnan(x): return None
        return round(x, n)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    by_tag = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(args.runs, "*_s*.json"))):
        name = os.path.basename(p)
        m = re.match(r"(.+)_s(\d+)\.json$", name)
        if not m: continue
        tag, _s = m.group(1), int(m.group(2))
        try:
            with open(p) as f: d = json.load(f)
        except Exception as e:
            print(f"[warn] skip {p}: {e}", file=sys.stderr)
            continue
        if d.get("status") == "boot_fail":
            by_tag[tag].append({"boot_fail": True})
            continue
        by_tag[tag].append(d)

    rows = []
    for tag, items in by_tag.items():
        ok = [x for x in items if not x.get("boot_fail")]
        if not ok:
            rows.append({"tag": tag, "n": 0, "boot_fail": True})
            continue
        tps = [x["output_tps"] for x in ok if x.get("output_tps")]
        ttft = [x["ttft_ms_p50"] for x in ok if x.get("ttft_ms_p50") is not None]
        tpot = [x["tpot_ms_p50"] for x in ok if x.get("tpot_ms_p50") is not None]
        acc = [x["accept_rate"] for x in ok if x.get("accept_rate") is not None]
        gpu = [x["gpu_util"] for x in ok if x.get("gpu_util") is not None]
        cpu = [x["cpu_util"] for x in ok if x.get("cpu_util") is not None]
        wall = [x["wall_total_s"] for x in ok if x.get("wall_total_s") is not None]
        def stats(xs):
            if not xs: return (None, None)
            m = sum(xs)/len(xs)
            v = sum((x-m)**2 for x in xs)/len(xs) if len(xs)>1 else 0.0
            return (m, math.sqrt(v))
        tps_m, tps_s = stats(tps)
        ttft_m, _ = stats(ttft)
        tpot_m, _ = stats(tpot)
        acc_m, _ = stats(acc)
        gpu_m, _ = stats(gpu)
        cpu_m, _ = stats(cpu)
        wall_m, _ = stats(wall)
        rows.append({
            "tag": tag,
            "n_sweeps": len(ok),
            "output_tps_mean": fmt(tps_m, 1),
            "output_tps_std": fmt(tps_s, 1),
            "wall_s_mean": fmt(wall_m, 1),
            "ttft_p50_ms_mean": fmt(ttft_m, 1),
            "tpot_p50_ms_mean": fmt(tpot_m, 2),
            "accept_rate_mean": fmt(acc_m, 4) if acc_m else None,
            "gpu_util_mean": fmt(gpu_m, 1),
            "cpu_util_mean": fmt(cpu_m, 1),
        })
    rows.sort(key=lambda r: r["tag"])

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    if args.out_csv:
        cols = ["tag", "n_sweeps", "output_tps_mean", "output_tps_std", "wall_s_mean",
                "ttft_p50_ms_mean", "tpot_p50_ms_mean", "accept_rate_mean",
                "gpu_util_mean", "cpu_util_mean"]
        with open(args.out_csv, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    for r in rows:
        print(f"{r['tag']:40s} n={r.get('n_sweeps',0):>2} "
              f"tps={r.get('output_tps_mean')} ±{r.get('output_tps_std')} "
              f"wall={r.get('wall_s_mean')} ttft={r.get('ttft_p50_ms_mean')} "
              f"tpot={r.get('tpot_p50_ms_mean')} acc={r.get('accept_rate_mean')}")


if __name__ == "__main__":
    main()
