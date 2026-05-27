#!/usr/bin/env python3
"""SUB_196 cellA aggregate."""
from __future__ import annotations
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
MIXES = ["balanced", "sonnet-heavy", "code-heavy"]
SCENARIOS = ["vanilla-only", "trident-only", "agsd-gated"]


def load_mix(mode_dir: Path, mix: str):
    p = mode_dir / mix / f"benchmark_{mix}.json"
    if not p.exists():
        return {}
    try:
        j = json.loads(p.read_text())
    except Exception:
        return {}
    return {s["scenario"]: s for s in j.get("scenarios", [])}


def aggregate_monitor(prefix: str):
    cpu_path = BASE / f"_monitor_{prefix}_cpu.csv"
    gpu_path = BASE / f"_monitor_{prefix}_gpu.csv"
    cpu_avg = 0.0
    gpu_avg = 0.0
    if cpu_path.exists():
        cpu_vals = []
        with open(cpu_path) as fh:
            for row in csv.DictReader(fh):
                try: cpu_vals.append(float(row["cpu_util_pct"]))
                except Exception: pass
        if cpu_vals: cpu_avg = statistics.mean(cpu_vals)
    if gpu_path.exists():
        per_gpu = defaultdict(list)
        with open(gpu_path) as fh:
            for row in csv.DictReader(fh):
                try:
                    per_gpu[int(row["gpu_index"])].append(float(row["gpu_util_pct"]))
                except Exception: pass
        if per_gpu:
            gpu_avg = sum(statistics.mean(v) for v in per_gpu.values()) / 8
    return cpu_avg, gpu_avg


def main():
    out = []
    def w(s=""): out.append(s); print(s)

    w("# SUB_196 cell A — regular SAXPY × 10ms cycle")
    w()
    w("| mix | scenario | OFF tps | ON tps | Δ% |")
    w("|---|---|---:|---:|---:|")
    summary = {}
    for mix in MIXES:
        off_sc = load_mix(BASE / "measurements" / "off", mix)
        on_sc = load_mix(BASE / "measurements" / "on", mix)
        for sc in SCENARIOS:
            off_v = off_sc.get(sc, {}).get("tps")
            on_v = on_sc.get(sc, {}).get("tps")
            delta = "—"
            if off_v and on_v:
                delta = f"{(on_v - off_v) / off_v * 100:+.2f}%"
            off_s = f"{off_v:.1f}" if off_v else "—"
            on_s = f"{on_v:.1f}" if on_v else "—"
            w(f"| {mix} | {sc} | {off_s} | {on_s} | {delta} |")
            summary[(mix, sc)] = (off_v, on_v)

    off_agsd = [summary[(m, "agsd-gated")][0] for m in MIXES if summary[(m, "agsd-gated")][0]]
    on_agsd  = [summary[(m, "agsd-gated")][1] for m in MIXES if summary[(m, "agsd-gated")][1]]
    w()
    if off_agsd and on_agsd:
        off_m = statistics.mean(off_agsd)
        on_m = statistics.mean(on_agsd)
        w("## AGSD 3-mix avg")
        w(f"- OFF: {off_m:.1f} tps  / ON: {on_m:.1f} tps  / Δ: {(on_m - off_m) / off_m * 100:+.2f}%")
    w()
    w("## monitor")
    for mode, pref in (("OFF", "off"), ("ON", "on")):
        cpu, gpu = aggregate_monitor(pref)
        w(f"- {mode}: CPU {cpu:.2f}%  GPU {gpu:.2f}%")

    (BASE / "AGGREGATE.md").write_text("\n".join(out))


if __name__ == "__main__":
    main()
