#!/usr/bin/env python3
"""AMX draft head firer — invokes the AMX kernel at target Hz, concurrent
with a vllm instance. Proxy for cpu_amx draft path of the AGSD router."""
import argparse
import ctypes
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LIB_PATH = HERE / "build" / "libamx_draft_qwen05b.so"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-hz", type=int, default=100,
                    help="Target invocation rate")
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--duration-s", type=int, default=180)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    lib = ctypes.CDLL(str(LIB_PATH))
    lib.amx_draft_qwen05b_init.restype = ctypes.c_int
    lib.amx_draft_qwen05b_free.restype = None
    lib.amx_draft_qwen05b_step_ms.restype = ctypes.c_double
    lib.amx_draft_qwen05b_step_ms.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.amx_draft_qwen05b_hw_amx.restype = ctypes.c_int

    has_amx = bool(lib.amx_draft_qwen05b_hw_amx())
    print(f"[firer] amx_available={has_amx}")
    rc = lib.amx_draft_qwen05b_init()
    if rc != 0:
        print(f"[firer] init failed rc={rc}")
        return 1

    period = 1.0 / args.target_hz
    deadline = time.time() + args.duration_s
    n_calls = 0
    total_ms = 0.0
    next_t = time.time()

    stats = {
        "hw_amx": has_amx,
        "K": args.K, "B": args.B,
        "target_hz": args.target_hz,
        "duration_s": args.duration_s,
        "started_at": time.time(),
    }

    try:
        while time.time() < deadline:
            now = time.time()
            if now < next_t:
                time.sleep(max(0, next_t - now))
            t0 = time.time()
            ms = lib.amx_draft_qwen05b_step_ms(args.B, args.K)
            t1 = time.time()
            n_calls += 1
            total_ms += ms
            next_t += period
            if n_calls % 50 == 0:
                print(f"[firer] n={n_calls} kernel_avg_ms={total_ms/n_calls:.2f} "
                      f"wall_ms={(t1-t0)*1000:.2f}", flush=True)
    finally:
        lib.amx_draft_qwen05b_free()

    stats.update({
        "ended_at": time.time(),
        "n_calls": n_calls,
        "kernel_avg_ms": total_ms / max(1, n_calls),
        "achieved_hz": n_calls / max(1.0, args.duration_s),
    })
    Path(args.out).write_text(json.dumps(stats, indent=2))
    print(f"[firer] done n={n_calls} kernel_avg_ms={stats['kernel_avg_ms']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
