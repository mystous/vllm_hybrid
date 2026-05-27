"""SUB_185 aggregate: OFF vs ON long-context bench + monitor mean util."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

BASE = Path(__file__).parent


def load_bench(mode: str) -> dict | None:
    p = BASE / "measurements" / mode / "long_sonnet" / "bench.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_monitor_mean(mode: str) -> tuple[float | None, float | None]:
    """Return (cpu_pct_mean, gpu_pct_mean) from monitor csv files."""
    cpu_csv = BASE / f"_monitor_{mode}_cpu.csv"
    gpu_csv = BASE / f"_monitor_{mode}_gpu.csv"
    cpu_mean = gpu_mean = None
    if cpu_csv.exists():
        vals = []
        with cpu_csv.open() as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                v = row.get("cpu_pct") or row.get("cpu_percent") or row.get("cpu")
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        if vals:
            cpu_mean = statistics.fmean(vals)
    if gpu_csv.exists():
        vals = []
        with gpu_csv.open() as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                v = row.get("utilization_gpu") or row.get("util_gpu") or row.get("util")
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        if vals:
            gpu_mean = statistics.fmean(vals)
    return cpu_mean, gpu_mean


def main():
    off = load_bench("off")
    on = load_bench("on")
    cpu_off, gpu_off = load_monitor_mean("off")
    cpu_on, gpu_on = load_monitor_mean("on")

    rows = []
    for label, b in (("OFF", off), ("ON", on)):
        if b is None:
            rows.append({"mode": label, "tps": None, "ttft_p50_ms": None,
                         "ttft_p99_ms": None, "lat_p50_ms": None,
                         "lat_p99_ms": None, "n_ok": None, "wall_s": None})
            continue
        rows.append({
            "mode": label,
            "tps": b["tps"],
            "ttft_p50_ms": (b["ttft_p50_s"] or 0) * 1000,
            "ttft_p99_ms": (b["ttft_p99_s"] or 0) * 1000,
            "lat_p50_ms": (b["latency_p50_s"] or 0) * 1000,
            "lat_p99_ms": (b["latency_p99_s"] or 0) * 1000,
            "n_ok": b["n_ok"],
            "wall_s": b["wall_s"],
        })

    delta_tps = None
    delta_ttft = None
    if rows[0]["tps"] and rows[1]["tps"]:
        delta_tps = (rows[1]["tps"] - rows[0]["tps"]) / rows[0]["tps"] * 100.0
    if rows[0]["ttft_p50_ms"] and rows[1]["ttft_p50_ms"]:
        delta_ttft = ((rows[1]["ttft_p50_ms"] - rows[0]["ttft_p50_ms"])
                      / rows[0]["ttft_p50_ms"] * 100.0)

    print("| mode | tps | ttft_p50 ms | ttft_p99 ms | lat_p50 ms | lat_p99 ms | n_ok | wall s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        def f(x, fmt=",.1f"):
            return format(x, fmt) if x is not None else "—"
        print(f"| {r['mode']} | {f(r['tps'])} | {f(r['ttft_p50_ms'])} | "
              f"{f(r['ttft_p99_ms'])} | {f(r['lat_p50_ms'])} | {f(r['lat_p99_ms'])} | "
              f"{r['n_ok'] if r['n_ok'] is not None else '—'} | "
              f"{f(r['wall_s'])} |")

    print()
    print("| metric | OFF | ON | Δ% |")
    print("|---|---:|---:|---:|")
    def srow(label, off_v, on_v):
        if off_v is None or on_v is None:
            print(f"| {label} | — | — | — |")
            return
        d = (on_v - off_v) / off_v * 100.0
        print(f"| {label} | {off_v:,.2f} | {on_v:,.2f} | {d:+.2f}% |")
    if rows[0]["tps"] is not None and rows[1]["tps"] is not None:
        srow("tps", rows[0]["tps"], rows[1]["tps"])
    if rows[0]["ttft_p50_ms"] is not None and rows[1]["ttft_p50_ms"] is not None:
        srow("ttft_p50 ms", rows[0]["ttft_p50_ms"], rows[1]["ttft_p50_ms"])
    if rows[0]["lat_p50_ms"] is not None and rows[1]["lat_p50_ms"] is not None:
        srow("lat_p50 ms", rows[0]["lat_p50_ms"], rows[1]["lat_p50_ms"])
    if cpu_off is not None and cpu_on is not None:
        srow("cpu_pct mean", cpu_off, cpu_on)
    if gpu_off is not None and gpu_on is not None:
        srow("gpu_pct mean", gpu_off, gpu_on)

    # firer stats if present
    firer = BASE / "logs" / "firer_stats.json"
    if firer.exists():
        fd = json.loads(firer.read_text())
        print()
        print("**firer (ON only)**:", json.dumps(fd, indent=2))

    out = {
        "rows": rows,
        "delta_tps_pct": delta_tps,
        "delta_ttft_p50_pct": delta_ttft,
        "cpu_pct_mean": {"off": cpu_off, "on": cpu_on},
        "gpu_pct_mean": {"off": gpu_off, "on": gpu_on},
    }
    (BASE / "aggregate.json").write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {BASE / 'aggregate.json'}")


if __name__ == "__main__":
    main()
