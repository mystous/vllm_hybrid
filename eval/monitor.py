#!/usr/bin/env python3
"""CPU/GPU utilization monitor — writes CSV rows at fixed intervals until killed."""

from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None


def gpu_samples() -> list[dict[str, str]]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,"
                "memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        rows.append({
            "gpu_index": parts[0],
            "gpu_util_pct": parts[1],
            "gpu_mem_util_pct": parts[2],
            "gpu_mem_used_mib": parts[3],
            "gpu_mem_total_mib": parts[4],
            "gpu_temp_c": parts[5],
            "gpu_power_w": parts[6],
        })
    return rows


def cpu_sample(interval: float) -> dict[str, str]:
    if psutil is None:
        return {"cpu_util_pct": "", "mem_used_gb": "", "mem_total_gb": ""}
    cpu = psutil.cpu_percent(interval=interval)
    vm = psutil.virtual_memory()
    return {
        "cpu_util_pct": f"{cpu:.1f}",
        "mem_used_gb": f"{vm.used / 1024 / 1024 / 1024:.2f}",
        "mem_total_gb": f"{vm.total / 1024 / 1024 / 1024:.2f}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="Output path prefix, e.g. results/run1/monitor")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cpu_path = prefix.with_name(prefix.name + "_cpu.csv")
    gpu_path = prefix.with_name(prefix.name + "_gpu.csv")

    cpu_fields = ["timestamp", "cpu_util_pct", "mem_used_gb", "mem_total_gb"]
    gpu_fields = [
        "timestamp", "gpu_index", "gpu_util_pct", "gpu_mem_util_pct",
        "gpu_mem_used_mib", "gpu_mem_total_mib", "gpu_temp_c", "gpu_power_w",
    ]

    cpu_f = cpu_path.open("w", newline="")
    gpu_f = gpu_path.open("w", newline="")
    cpu_w = csv.DictWriter(cpu_f, fieldnames=cpu_fields)
    gpu_w = csv.DictWriter(gpu_f, fieldnames=gpu_fields)
    cpu_w.writeheader()
    gpu_w.writeheader()

    stopping = False

    def handler(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    print(f"[monitor] cpu→{cpu_path} gpu→{gpu_path} interval={args.interval}s",
          flush=True)

    try:
        while not stopping:
            ts = datetime.now(timezone.utc).isoformat()
            # cpu_sample blocks for `interval` seconds internally; this doubles as pacing
            cpu_row = cpu_sample(args.interval)
            cpu_row["timestamp"] = ts
            cpu_w.writerow(cpu_row)
            cpu_f.flush()

            for row in gpu_samples():
                row["timestamp"] = ts
                gpu_w.writerow(row)
            gpu_f.flush()
    finally:
        cpu_f.close()
        gpu_f.close()
        print("[monitor] stopped", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
