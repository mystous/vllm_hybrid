#!/usr/bin/env python3
"""
monitor.py — GPU/CPU 활용률 실시간 모니터링 및 CSV 저장

사용법:
    python monitor.py <output_prefix> [--interval 1.0]

출력:
    <output_prefix>_gpu.csv  — GPU 카드별 + 종합 평균
    <output_prefix>_cpu.csv  — 물리 코어별 + 종합 평균

종료:
    SIGTERM 또는 SIGINT (Ctrl+C)
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

# ---------------------------------------------------------------------------
# GPU 샘플링 (nvidia-smi 기반)
# ---------------------------------------------------------------------------

def _query_nvidia_smi() -> list[dict]:
    """nvidia-smi로 GPU 카드별 메트릭 수집."""
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
# CPU 샘플링 (psutil 기반)
# ---------------------------------------------------------------------------

def _get_cpu_topology() -> tuple[int, int, int]:
    """(physical_cores, logical_cores, threads_per_core) 반환.

    i9-12900KF 같은 하이브리드(P+E) 아키텍처는 logical/physical이
    정수로 나누어지지 않을 수 있으므로, logical 기준으로 헤더를 만들고
    /proc/cpuinfo 기반으로 physical→logical 매핑을 시도한다.
    """
    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False) or logical
    # 안전하게: logical이 physical의 배수가 아니면 tpc=1로 처리
    if logical % physical == 0:
        tpc = logical // physical
    else:
        tpc = 1  # 하이브리드 아키텍처 — 논리 코어를 물리 코어로 취급
    return physical, logical, tpc


def _sample_cpu_per_physical(logical_percents: list[float], tpc: int) -> list[float]:
    """논리 코어 목록을 HT 쌍으로 묶어 물리 코어별 평균 반환.

    tpc=1 이면 논리 코어 = 물리 코어로 그대로 반환.
    하이브리드 아키텍처(P+E)에서는 실제 논리 코어 수를 그대로 사용.
    """
    if tpc == 1:
        return list(logical_percents)
    physical = []
    for i in range(0, len(logical_percents), tpc):
        chunk = logical_percents[i:i + tpc]
        physical.append(sum(chunk) / len(chunk))
    return physical


# ---------------------------------------------------------------------------
# CSV 헤더 생성
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
    """num_cpu_cols: _sample_cpu_per_physical 이 반환하는 실제 컬럼 수."""
    base = ["timestamp", "elapsed_s"]
    for i in range(num_cpu_cols):
        base.append(f"core{i}_util_pct")
    base += ["cpu_avg_util_pct", "cpu_mem_used_mb", "cpu_mem_avail_mb"]
    return base


# ---------------------------------------------------------------------------
# 메인 모니터 루프
# ---------------------------------------------------------------------------

def monitor(output_prefix: str, interval: float):
    physical_cores, logical_cores, tpc = _get_cpu_topology()
    gpu_count = _get_gpu_count()

    results_dir = Path(output_prefix).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    gpu_csv_path = f"{output_prefix}_gpu.csv"
    cpu_csv_path = f"{output_prefix}_cpu.csv"

    # 헤더 컬럼 수를 실제 샘플에서 결정 (하이브리드 아키텍처 대응)
    _first_logical = psutil.cpu_percent(percpu=True)
    time.sleep(0.1)
    _first_logical = psutil.cpu_percent(percpu=True)
    _first_physical = _sample_cpu_per_physical(_first_logical, tpc)
    num_cpu_cols = len(_first_physical)

    cpu_label = f"{num_cpu_cols} cores"
    if tpc > 1:
        cpu_label += f" (physical, {tpc}T/core)"
    else:
        cpu_label += f" (logical={logical_cores}, physical={physical_cores})"

    print(f"[monitor] CPU: {cpu_label}")
    print(f"[monitor] GPU: {gpu_count} cards")
    print(f"[monitor] interval={interval}s")
    print(f"[monitor] GPU CSV → {gpu_csv_path}")
    print(f"[monitor] CPU CSV → {cpu_csv_path}")
    print("[monitor] SIGINT/SIGTERM으로 종료")

    start_time = time.monotonic()
    running = True

    def _stop(signum, frame):
        nonlocal running
        running = False
        print("\n[monitor] 종료 신호 수신, 파일 닫는 중...")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    time.sleep(max(0, interval - 0.1))  # 위에서 이미 초기 샘플 수집함

    with open(gpu_csv_path, "w", newline="") as gpu_f, \
         open(cpu_csv_path, "w", newline="") as cpu_f:

        gpu_writer = csv.DictWriter(gpu_f, fieldnames=_build_gpu_header(gpu_count))
        cpu_writer = csv.DictWriter(cpu_f, fieldnames=_build_cpu_header(num_cpu_cols))
        gpu_writer.writeheader()
        cpu_writer.writeheader()

        while running:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
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
            gpu_row["gpu_avg_power_w"] = round(sum(power_list), 2)  # 전체 합계

            gpu_writer.writerow(gpu_row)
            gpu_f.flush()

            # --- CPU ---
            logical_pcts = psutil.cpu_percent(percpu=True)
            physical_pcts = _sample_cpu_per_physical(logical_pcts, tpc)
            mem = psutil.virtual_memory()

            cpu_row: dict = {"timestamp": ts, "elapsed_s": elapsed}
            for i, pct in enumerate(physical_pcts):
                cpu_row[f"core{i}_util_pct"] = round(pct, 2)

            cpu_row["cpu_avg_util_pct"] = round(sum(physical_pcts) / len(physical_pcts), 2)
            cpu_row["cpu_mem_used_mb"] = round(mem.used / 1024 / 1024, 1)
            cpu_row["cpu_mem_avail_mb"] = round(mem.available / 1024 / 1024, 1)

            cpu_writer.writerow(cpu_row)
            cpu_f.flush()

            time.sleep(interval)

    print(f"[monitor] 저장 완료: {gpu_csv_path}, {cpu_csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU/CPU utilization monitor")
    parser.add_argument("output_prefix", help="출력 CSV prefix (예: results/gpu_only)")
    parser.add_argument("--interval", type=float, default=1.0, help="샘플링 간격(초), 기본값 1.0")
    args = parser.parse_args()

    monitor(args.output_prefix, args.interval)
