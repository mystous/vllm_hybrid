#!/usr/bin/env python3
"""Collect host/CPU/GPU/software info and dump as JSON."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str], timeout: int = 10) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def parse_kv(raw: str) -> dict[str, str]:
    out = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def collect_cpu() -> dict:
    lscpu = parse_kv(run(["lscpu"]))
    flags_raw = lscpu.get("Flags", "")
    if not flags_raw and Path("/proc/cpuinfo").exists():
        m = re.search(r"^flags\s*:\s*(.+)$",
                      Path("/proc/cpuinfo").read_text(), re.MULTILINE)
        if m:
            flags_raw = m.group(1)
    important = [
        "avx", "avx2", "avx512f", "avx512bw", "avx512vl", "avx512_vnni",
        "amx_bf16", "amx_int8", "amx_tile", "sse4_1", "sse4_2", "fma",
    ]
    flags = [f for f in important if f in flags_raw.lower().split()]
    return {
        "model_name": lscpu.get("Model name", ""),
        "architecture": lscpu.get("Architecture", platform.machine()),
        "sockets": lscpu.get("Socket(s)", ""),
        "cores_per_socket": lscpu.get("Core(s) per socket", ""),
        "threads_per_core": lscpu.get("Thread(s) per core", ""),
        "total_cpus": lscpu.get("CPU(s)", ""),
        "cpu_max_mhz": lscpu.get("CPU max MHz", ""),
        "l1d_cache": lscpu.get("L1d cache", ""),
        "l2_cache": lscpu.get("L2 cache", ""),
        "l3_cache": lscpu.get("L3 cache", ""),
        "important_flags": flags,
    }


def collect_numa() -> dict:
    raw = run(["numactl", "--hardware"])
    if raw:
        nodes = re.findall(r"node (\d+) size: (\d+) MB", raw)
        return {
            "num_nodes": len(nodes),
            "nodes": [{"node": int(n), "size_mb": int(s)} for n, s in nodes],
        }
    return {"num_nodes": 1, "nodes": []}


def collect_memory() -> dict:
    p = Path("/proc/meminfo")
    if not p.exists():
        return {}
    mem = parse_kv(p.read_text())
    return {
        "total": mem.get("MemTotal", ""),
        "available": mem.get("MemAvailable", ""),
        "swap_total": mem.get("SwapTotal", ""),
    }


def collect_gpu() -> dict:
    if not shutil.which("nvidia-smi"):
        return {"count": 0, "devices": [], "note": "nvidia-smi not found"}
    raw = run([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id,"
        "power.limit,clocks.max.sm,clocks.max.mem,compute_mode",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 10:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "memory_total_mib": parts[3],
                "driver_version": parts[4],
                "pci_bus_id": parts[5],
                "power_limit_w": parts[6],
                "max_sm_clock_mhz": parts[7],
                "max_mem_clock_mhz": parts[8],
                "compute_mode": parts[9],
            })
    nvcc = run(["nvcc", "--version"])
    m = re.search(r"release (\S+),", nvcc) if nvcc else None
    return {
        "count": len(gpus),
        "devices": gpus,
        "driver_version": gpus[0]["driver_version"] if gpus else "",
        "cuda_toolkit_version": m.group(1) if m else "",
        "topology": run(["nvidia-smi", "topo", "-m"]),
    }


def collect_software() -> dict:
    sw = {"python_version": platform.python_version()}
    try:
        import torch  # type: ignore
        sw["torch_version"] = torch.__version__
        sw["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            sw["torch_cuda_version"] = torch.version.cuda or ""
    except ImportError:
        pass
    try:
        import vllm  # type: ignore
        sw["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except ImportError:
        pass
    return sw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outfile", help="Output JSON path")
    args = ap.parse_args()

    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "cpu": collect_cpu(),
        "numa": collect_numa(),
        "memory": collect_memory(),
        "gpu": collect_gpu(),
        "software": collect_software(),
    }

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"[sysinfo] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
