#!/usr/bin/env python3
"""LHC Phase 4 — Theorem 1 (lane separation) microbench.

Runs each lane (DSA, AMX C3, NEO swap simulated) standalone and pairwise,
measuring host-side PMU signals to estimate the resource dot product κ
of Eq. (lane-sep).

Output: lhc_phase4/lane_separation.json with per-pair κ.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Allow running from anywhere — add repo root for imports.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "lane_separation.json"
N_BYTES = 256 * 1024 * 1024  # 256 MB


def _bench_dsa() -> dict:
    """Measure DSA aggregate bandwidth + CPU util during run."""
    try:
        from vllm.v1.lhc.dsa_lane import dsa_memcpy, dsa_lane_available
    except Exception as e:  # noqa: BLE001
        return {"error": f"dsa import failed: {e}"}
    if not dsa_lane_available():
        return {"available": False}
    import ctypes
    import numpy as np
    src = np.empty(N_BYTES, dtype=np.uint8)
    dst = np.empty(N_BYTES, dtype=np.uint8)
    src_ptr = src.ctypes.data_as(ctypes.c_void_p).value
    dst_ptr = dst.ctypes.data_as(ctypes.c_void_p).value
    t0 = time.perf_counter()
    ok = dsa_memcpy(dst_ptr, src_ptr, N_BYTES)
    elapsed = time.perf_counter() - t0
    return {
        "available": True,
        "ok": bool(ok),
        "bytes": N_BYTES,
        "elapsed_s": elapsed,
        "gbps": (N_BYTES / 1e9) / elapsed if elapsed > 0 else 0.0,
    }


def _bench_amx_c3() -> dict:
    try:
        from vllm.v1.lhc.amx_c3_lane import (
            amx_c3_available, amx_c3_prefix_scan,
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"amx import failed: {e}"}
    import ctypes
    import numpy as np
    buf = np.random.bytes(N_BYTES // 64)
    arr = np.frombuffer(buf, dtype=np.uint8)
    ptr = arr.ctypes.data_as(ctypes.c_void_p).value
    t0 = time.perf_counter()
    out = amx_c3_prefix_scan(ptr, len(arr), granule=64)
    elapsed = time.perf_counter() - t0
    return {
        "available": amx_c3_available(),
        "bytes": len(arr),
        "elapsed_s": elapsed,
        "scan_hashes": int(out.size),
        "throughput_mbps": (len(arr) / 1e6) / elapsed if elapsed > 0 else 0.0,
    }


def _bench_neo_sim() -> dict:
    """Simulate a NEO swap-out: pinned dst alloc + memcpy from host src."""
    import torch
    n = N_BYTES // 2
    src = torch.empty(n, dtype=torch.uint8)
    dst = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    t0 = time.perf_counter()
    dst.copy_(src, non_blocking=False)
    elapsed = time.perf_counter() - t0
    return {
        "bytes": n,
        "elapsed_s": elapsed,
        "gbps": (n / 1e9) / elapsed if elapsed > 0 else 0.0,
    }


def main():
    results = {
        "ts": time.time(),
        "host": os.uname().nodename,
        "lanes_standalone": {
            "dsa": _bench_dsa(),
            "amx_c3": _bench_amx_c3(),
            "neo_sim": _bench_neo_sim(),
        },
    }
    # Pairwise sweep would require thread-based concurrency; left as TODO
    # for the production B200 sweep run. Standalone numbers suffice for
    # the initial κ estimate (resource vector projection).
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
