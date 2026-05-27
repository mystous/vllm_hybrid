#!/usr/bin/env python3
"""SUB_198 — AMX draft head proxy proposer for vllm spec_decode.

Honest scope: this is a **proxy integration**, NOT a real Qwen 0.5B draft model.

Why proxy:
  The SUB_187 AMX kernel (libamx_draft_qwen05b.so) is a *latency microbench* —
  it allocates synthetic matrices of Qwen 0.5B LM-head shape (896 × 152064) and
  measures the AMX matmul chain timing per call. It does NOT load real Qwen
  weights, does NOT consume input context tokens, and does NOT produce valid
  draft token IDs that vllm's RejectionSampler can verify.

  Building a real AMX-accelerated draft model would require:
    1. Loading Qwen 0.5B BF16 weights into AMX tile layout
    2. Implementing the full forward chain (24 layers × attention + MLP)
       in AMX intrinsics
    3. KV cache management on CPU side
    4. Sampling (argmax + temperature) over the AMX-produced logits
    5. Tokenizer round-trip for the context

  That is a multi-week effort, out of scope for a single SUB measurement.

What this proxy does instead:
  Reuses the existing suffix_decoding proposer for ACTUAL draft token IDs
  (so vllm functional correctness is preserved), while concurrently firing the
  AMX kernel as a side-channel CPU consumer on cores 80-95 to demonstrate that
  the AMX compute capacity (10.24 TFLOPS available per SUB_117) can be drawn
  from in parallel WITHOUT disrupting suffix-decoding throughput.

  This is the same pattern as SUB_188 (softmax precompute side-channel), but
  uses the AMX kernel from SUB_187 as the work payload.

Limitations of this proxy (honest):
  - AMX kernel is NOT being consumed by spec_decode rejection sampler.
  - Theoretical 4.12× spec speedup from SUB_187 is NOT achievable here.
  - This measurement tells us: can AMX kernel fire concurrently with
    suffix-decoding vllm WITHOUT throughput regression?
  - If yes → AMX kernel passes co-residence test (paper §4 secondary lever).
  - If no → AMX kernel cannot co-reside even passively (paper §4 reject).

Activation: ENV `VLLM_USE_AMX_DRAFT=1`. Default OFF preserves stock behavior.
"""
import argparse
import ctypes
import json
import os
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB_PATH = (
    Path("/workspace/vllm_hybrid/shadow_assists/features/IDE_019_multi_source_drafter"
         "/SUB_187_amx_draft_head/build/libamx_draft_qwen05b.so")
)


def load_amx_lib():
    if not LIB_PATH.exists():
        return None
    try:
        lib = ctypes.CDLL(str(LIB_PATH))
    except OSError as e:
        print(f"[amx_proxy] failed to load lib: {e}", file=sys.stderr)
        return None
    lib.amx_draft_qwen05b_init.restype = ctypes.c_int
    lib.amx_draft_qwen05b_free.restype = None
    lib.amx_draft_qwen05b_step_ms.restype = ctypes.c_double
    lib.amx_draft_qwen05b_step_ms.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.amx_draft_qwen05b_hw_amx.restype = ctypes.c_int
    return lib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-hz", type=int, default=100,
                    help="Target invocation rate (default 100 Hz, matches SUB_187)")
    ap.add_argument("--K", type=int, default=7,
                    help="Draft K (default 7 — chat acceptance α=0.80 sweet spot)")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--duration-s", type=int, default=600,
                    help="Max duration (safety cap)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--core-base", type=int, default=80,
                    help="OMP base core (matches SUB_187 firer)")
    args = ap.parse_args()

    # Pin to cores 80-95 (away from vllm vanilla 0-49 + trident 56-105) ──
    # NOTE: ctypes OMP threads will be pinned by env var OMP_PROC_BIND if set.
    # We rely on the kernel's internal pin via syscall (SUB_187 kernel calls
    # sched_setaffinity for AMX permission).
    os.environ.setdefault("OMP_NUM_THREADS", "16")
    os.environ.setdefault("OMP_PROC_BIND", "close")

    lib = load_amx_lib()
    if lib is None:
        print("[amx_proxy] FATAL — kernel .so unavailable", file=sys.stderr)
        Path(args.out).write_text(json.dumps({"status": "lib_unavailable"}))
        return 1

    has_amx = bool(lib.amx_draft_qwen05b_hw_amx())
    print(f"[amx_proxy] amx_available={has_amx}", flush=True)
    rc = lib.amx_draft_qwen05b_init()
    if rc != 0:
        print(f"[amx_proxy] init rc={rc}", flush=True)
        Path(args.out).write_text(json.dumps({"status": "init_failed", "rc": rc}))
        return 1

    stop = {"v": False}

    def _sig(signum, frame):
        stop["v"] = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    period = 1.0 / args.target_hz
    deadline = time.time() + args.duration_s
    n_calls = 0
    total_ms = 0.0
    next_t = time.time()

    started = time.time()
    try:
        while not stop["v"] and time.time() < deadline:
            now = time.time()
            if now < next_t:
                time.sleep(max(0, next_t - now))
            ms = lib.amx_draft_qwen05b_step_ms(args.B, args.K)
            n_calls += 1
            total_ms += ms
            next_t += period
            if n_calls % 100 == 0:
                print(
                    f"[amx_proxy] n={n_calls} kernel_avg_ms={total_ms / n_calls:.3f}",
                    flush=True,
                )
    finally:
        try:
            lib.amx_draft_qwen05b_free()
        except Exception:  # noqa: BLE001
            pass

    ended = time.time()
    stats = {
        "status": "ok",
        "hw_amx": has_amx,
        "K": args.K,
        "B": args.B,
        "target_hz": args.target_hz,
        "duration_s": ended - started,
        "n_calls": n_calls,
        "kernel_avg_ms": total_ms / max(1, n_calls),
        "achieved_hz": n_calls / max(1.0, ended - started),
    }
    Path(args.out).write_text(json.dumps(stats, indent=2))
    print(
        f"[amx_proxy] done n={n_calls} avg_kernel={stats['kernel_avg_ms']:.3f} ms "
        f"hz={stats['achieved_hz']:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
