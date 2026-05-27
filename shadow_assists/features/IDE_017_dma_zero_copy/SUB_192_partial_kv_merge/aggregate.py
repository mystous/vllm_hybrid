#!/usr/bin/env python3
"""Aggregate SUB_192 benchmark JSONs and produce AGGREGATE.md core table."""
from __future__ import annotations
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
MIXES = ["balanced", "sonnet-heavy", "code-heavy"]
SCENARIOS = ["vanilla-only", "trident-only", "agsd-gated"]


def load_mix(mode_dir: Path, mix: str) -> dict[str, dict]:
    p = mode_dir / mix / f"benchmark_{mix}.json"
    if not p.exists():
        return {}
    try:
        j = json.loads(p.read_text())
    except Exception:
        return {}
    return {s["scenario"]: s for s in j.get("scenarios", [])}


def aggregate_monitor(prefix: str) -> tuple[float, float, list[float]]:
    cpu_path = BASE / f"_monitor_{prefix}_cpu.csv"
    gpu_path = BASE / f"_monitor_{prefix}_gpu.csv"
    cpu_avg = 0.0
    gpu_avg = 0.0
    per_gpu = [0.0] * 8

    if cpu_path.exists():
        cpu_vals = []
        with open(cpu_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    cpu_vals.append(float(row["cpu_util_pct"]))
                except Exception:
                    pass
        if cpu_vals:
            cpu_avg = statistics.mean(cpu_vals)

    if gpu_path.exists():
        per_gpu_vals: dict[int, list[float]] = defaultdict(list)
        with open(gpu_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    idx = int(row["gpu_index"])
                    val = float(row["gpu_util_pct"])
                    per_gpu_vals[idx].append(val)
                except Exception:
                    pass
        for i in range(8):
            if per_gpu_vals.get(i):
                per_gpu[i] = statistics.mean(per_gpu_vals[i])
        if any(per_gpu):
            gpu_avg = sum(per_gpu) / 8

    return cpu_avg, gpu_avg, per_gpu


def main() -> None:
    off_dir = BASE / "measurements" / "off"
    on_dir = BASE / "measurements" / "on"

    out_lines = []
    def w(s=""):
        out_lines.append(s)
        print(s)

    w("# SUB_192 — Aggregate (side-channel partial KV merge)")
    w()
    w("## tps table (9 cells × 2 modes)")
    w()
    w("| mix | scenario | OFF tps | ON tps | Δ% |")
    w("|---|---|---:|---:|---:|")
    summary = {}
    for mix in MIXES:
        off_sc = load_mix(off_dir, mix)
        on_sc = load_mix(on_dir, mix)
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
        off_mean = statistics.mean(off_agsd)
        on_mean = statistics.mean(on_agsd)
        w("## AGSD 3-mix avg")
        w()
        w(f"- OFF: {off_mean:.1f} tps")
        w(f"- ON : {on_mean:.1f} tps")
        if off_mean > 0:
            w(f"- Δ  : {(on_mean - off_mean) / off_mean * 100:+.2f}%")

    w()
    w("## utilization (monitor.py 0.5s)")
    w()
    w("| mode | CPU avg % | GPU avg (8 GPU) % | per-GPU avg % |")
    w("|---|---:|---:|---|")
    for mode, prefix in (("OFF", "off"), ("ON", "on")):
        cpu, gpu, per_gpu = aggregate_monitor(prefix)
        per_str = ", ".join(f"{v:.1f}" for v in per_gpu)
        w(f"| {mode} | {cpu:.2f} | {gpu:.2f} | {per_str} |")

    summary_path = BASE / "AGGREGATE.md"
    summary_path.write_text("\n".join(out_lines))
    print(f"\n[saved] {summary_path}")


if __name__ == "__main__":
    main()
