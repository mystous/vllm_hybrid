#!/usr/bin/env python3
"""
monitor.py — Real-time GPU/CPU utilization monitoring and CSV export

Usage:
    python monitor.py <output_prefix> [--interval 1.0]

Output:
    <output_prefix>_gpu.csv  — Per-GPU card + overall average
    <output_prefix>_cpu.csv  — Per-physical-core + overall average

Exit:
    SIGTERM or SIGINT (Ctrl+C)
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
from pathlib import Path

import psutil

# ---------------------------------------------------------------------------
# GPU sampling (nvidia-smi based)
# ---------------------------------------------------------------------------

def _query_nvidia_smi() -> list[dict]:
    """Collect per-GPU metrics via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,"
                "memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        rows = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            rows.append({
                "gpu_index": int(parts[0]),
                "gpu_name": parts[1],
                "gpu_util_pct": float(parts[2]) if parts[2] not in ("N/A", "[N/A]") else 0.0,
                "gpu_mem_util_pct": float(parts[3]) if parts[3] not in ("N/A", "[N/A]") else 0.0,
                "gpu_mem_used_mb": float(parts[4]) if parts[4] not in ("N/A", "[N/A]") else 0.0,
                "gpu_mem_total_mb": float(parts[5]) if parts[5] not in ("N/A", "[N/A]") else 0.0,
                "gpu_power_w": float(parts[6]) if parts[6] not in ("N/A", "[N/A]") else 0.0,
                "gpu_temp_c": float(parts[7]) if parts[7] not in ("N/A", "[N/A]") else 0.0,
            })
        return rows
    except Exception:
        return []


def _get_gpu_count() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return len([l for l in r.stdout.strip().splitlines() if l.strip()])
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# CPU sampling (psutil based)
# ---------------------------------------------------------------------------

def _get_cpu_topology() -> tuple[int, int, int]:
    """Return (physical_cores, logical_cores, threads_per_core).

    Hybrid (P+E) architectures like i9-12900KF may not have logical/physical
    as an exact integer ratio, so we build headers based on logical count
    and attempt a physical->logical mapping via /proc/cpuinfo.
    """
    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False) or logical
    # Safe fallback: if logical is not a multiple of physical, use tpc=1
    if logical % physical == 0:
        tpc = logical // physical
    else:
        tpc = 1  # Hybrid architecture — treat logical cores as physical cores
    return physical, logical, tpc


def _sample_cpu_per_physical(logical_percents: list[float], tpc: int) -> list[float]:
    """Deprecated: positional (2i, 2i+1) pairing assumed HT siblings, which is
    wrong on Xeon (HT siblings are (N, N+cores_total)). Kept only for callers
    that still expect the legacy shape. New CSV schema emits raw per-logical-CPU
    columns (`cpu{N}_util_pct`) instead; analysis code does NUMA/HT grouping
    from `system_info.json` + sysfs topology.
    """
    if tpc == 1:
        return list(logical_percents)
    physical = []
    for i in range(0, len(logical_percents), tpc):
        chunk = logical_percents[i:i + tpc]
        physical.append(sum(chunk) / len(chunk))
    return physical


# ---------------------------------------------------------------------------
# CSV header construction
# ---------------------------------------------------------------------------

def _build_gpu_header(gpu_count: int) -> list[str]:
    base = ["timestamp", "elapsed_s"]
    for i in range(gpu_count):
        base += [
            f"gpu{i}_util_pct",
            f"gpu{i}_mem_util_pct",
            f"gpu{i}_mem_used_mb",
            f"gpu{i}_power_w",
            f"gpu{i}_temp_c",
        ]
    base += ["gpu_avg_util_pct", "gpu_avg_mem_util_pct", "gpu_avg_power_w"]
    return base


def _build_cpu_header(num_cpu_cols: int) -> list[str]:
    """Emit one column per logical CPU: `cpu{N}_util_pct` for N in [0, num_cpu_cols).
    Downstream analysis (hw_inspect) does NUMA/HT grouping from system_info.json
    + sysfs topology — schema-free of hardware layout assumptions.
    """
    base = ["timestamp", "elapsed_s"]
    for i in range(num_cpu_cols):
        base.append(f"cpu{i}_util_pct")
    base += ["cpu_avg_util_pct", "cpu_mem_used_mb", "cpu_mem_avail_mb"]
    return base


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------

def monitor(output_prefix: str, interval: float):
    physical_cores, logical_cores, tpc = _get_cpu_topology()
    gpu_count = _get_gpu_count()

    results_dir = Path(output_prefix).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    gpu_csv_path = f"{output_prefix}_gpu.csv"
    cpu_csv_path = f"{output_prefix}_cpu.csv"

    # One column per logical CPU — leaves HT/NUMA grouping to downstream analysis.
    _first_logical = psutil.cpu_percent(percpu=True)
    time.sleep(0.1)
    _first_logical = psutil.cpu_percent(percpu=True)
    num_cpu_cols = len(_first_logical)

    cpu_label = (f"{num_cpu_cols} logical CPUs "
                 f"(physical={physical_cores}, tpc={tpc})")

    print(f"[monitor] CPU: {cpu_label}")
    print(f"[monitor] GPU: {gpu_count} cards")
    print(f"[monitor] interval={interval}s")
    print(f"[monitor] GPU CSV → {gpu_csv_path}")
    print(f"[monitor] CPU CSV → {cpu_csv_path}")
    print("[monitor] Send SIGINT/SIGTERM to stop")

    start_time = time.monotonic()
    running = True

    def _stop(signum, frame):
        nonlocal running
        running = False
        print("\n[monitor] Stop signal received, closing files...")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    time.sleep(max(0, interval - 0.1))  # Initial sample already collected above

    with open(gpu_csv_path, "w", newline="") as gpu_f, \
         open(cpu_csv_path, "w", newline="") as cpu_f:

        gpu_writer = csv.DictWriter(gpu_f, fieldnames=_build_gpu_header(gpu_count))
        cpu_writer = csv.DictWriter(cpu_f, fieldnames=_build_cpu_header(num_cpu_cols))
        gpu_writer.writeheader()
        cpu_writer.writeheader()

        while running:
            ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            elapsed = round(time.monotonic() - start_time, 3)

            # --- GPU ---
            gpu_rows = _query_nvidia_smi()
            gpu_row: dict = {"timestamp": ts, "elapsed_s": elapsed}
            util_list, mem_util_list, power_list = [], [], []

            for i in range(gpu_count):
                data = gpu_rows[i] if i < len(gpu_rows) else {}
                gpu_row[f"gpu{i}_util_pct"] = data.get("gpu_util_pct", 0.0)
                gpu_row[f"gpu{i}_mem_util_pct"] = data.get("gpu_mem_util_pct", 0.0)
                gpu_row[f"gpu{i}_mem_used_mb"] = data.get("gpu_mem_used_mb", 0.0)
                gpu_row[f"gpu{i}_power_w"] = data.get("gpu_power_w", 0.0)
                gpu_row[f"gpu{i}_temp_c"] = data.get("gpu_temp_c", 0.0)
                util_list.append(gpu_row[f"gpu{i}_util_pct"])
                mem_util_list.append(gpu_row[f"gpu{i}_mem_util_pct"])
                power_list.append(gpu_row[f"gpu{i}_power_w"])

            gpu_row["gpu_avg_util_pct"] = round(sum(util_list) / len(util_list), 2) if util_list else 0.0
            gpu_row["gpu_avg_mem_util_pct"] = round(sum(mem_util_list) / len(mem_util_list), 2) if mem_util_list else 0.0
            gpu_row["gpu_avg_power_w"] = round(sum(power_list), 2)  # Total sum

            gpu_writer.writerow(gpu_row)
            gpu_f.flush()

            # --- CPU ---
            logical_pcts = psutil.cpu_percent(percpu=True)
            mem = psutil.virtual_memory()

            cpu_row: dict = {"timestamp": ts, "elapsed_s": elapsed}
            for i, pct in enumerate(logical_pcts):
                cpu_row[f"cpu{i}_util_pct"] = round(pct, 2)

            cpu_row["cpu_avg_util_pct"] = round(sum(logical_pcts) / len(logical_pcts), 2)
            cpu_row["cpu_mem_used_mb"] = round(mem.used / 1024 / 1024, 1)
            cpu_row["cpu_mem_avail_mb"] = round(mem.available / 1024 / 1024, 1)

            cpu_writer.writerow(cpu_row)
            cpu_f.flush()

            time.sleep(interval)

    print(f"[monitor] Saved: {gpu_csv_path}, {cpu_csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU/CPU utilization monitor")
    parser.add_argument("output_prefix", help="Output CSV prefix (e.g. results/20260327_120000/gpu_only)")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1.0)")
    args = parser.parse_args()

    # EVAL_RUN_DIR override: if set and prefix is relative, prepend it
    run_dir = os.environ.get("EVAL_RUN_DIR", "")
    prefix = args.output_prefix
    if run_dir and not os.path.isabs(prefix):
        prefix = os.path.join(run_dir, os.path.basename(prefix))

    monitor(prefix, args.interval)
