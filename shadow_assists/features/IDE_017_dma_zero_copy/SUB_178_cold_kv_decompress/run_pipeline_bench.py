#!/usr/bin/env python3
"""SUB_178 — Dequant + DMA-push pipeline microbench.

Pipeline:
  CPU INT8/INT4 cold-KV chunk
    → AVX-512 dequant (libcold_kv) → pinned BF16 buffer (libpinned_pool)
    → cudaMemcpyAsync H2D (libpinned_pool.push_async)
    → GPU consumes

Two timing modes:
  (a) sequential: dequant; then DMA; total = dequant + DMA
  (b) overlap: pipeline — chunk[i] DMA happens while chunk[i+1] is dequanting
      (in software with 2 stages)

Single-run per measurement.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLD_KV_LIB = HERE / "build" / "libcold_kv.so"
POOL_LIB = HERE.parent / "SUB_176_dma_pool_canonical" / "build" / "libpinned_pool.so"

assert COLD_KV_LIB.exists() and POOL_LIB.exists(), \
    f"libs not found: {COLD_KV_LIB} {POOL_LIB}"


def load_cold():
    lib = ctypes.CDLL(str(COLD_KV_LIB))
    lib.cold_kv_int8_to_bf16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.cold_kv_int8_to_bf16.restype = None
    lib.cold_kv_int4_to_bf16.argtypes = lib.cold_kv_int8_to_bf16.argtypes
    lib.cold_kv_int4_to_bf16.restype = None
    return lib


def load_pool():
    lib = ctypes.CDLL(str(POOL_LIB))
    lib.pinned_pool_create.argtypes = [ctypes.c_size_t, ctypes.c_int]
    lib.pinned_pool_create.restype = ctypes.c_void_p
    lib.pinned_pool_destroy.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_destroy.restype = None
    lib.pinned_pool_alloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.pinned_pool_alloc.restype = ctypes.c_void_p
    lib.pinned_pool_free.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.pinned_pool_free.restype = None
    lib.pinned_pool_push_async.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_void_p,
    ]
    lib.pinned_pool_push_async.restype = ctypes.c_void_p
    lib.pinned_pool_event_sync.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_event_sync.restype = ctypes.c_int
    lib.pinned_pool_event_destroy.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_event_destroy.restype = ctypes.c_int
    lib.pinned_pool_stream_create.argtypes = []
    lib.pinned_pool_stream_create.restype = ctypes.c_void_p
    lib.pinned_pool_stream_destroy.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_stream_destroy.restype = ctypes.c_int
    lib.pinned_pool_stream_sync.argtypes = [ctypes.c_void_p]
    lib.pinned_pool_stream_sync.restype = ctypes.c_int
    return lib


def stats(samples):
    n = len(samples)
    s = sorted(samples)
    return (sum(s) / n, s[n // 2], s[min(n - 1, int(n * 0.99))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=1,
                        help="GPU index (NUMA0 attached: 0-3)")
    parser.add_argument("--total-limit", type=int, default=2 * 1024 * 1024 * 1024)
    parser.add_argument("--numa", type=int, default=0)
    parser.add_argument("--out-json", type=str, default="pipeline_microbench.json")
    parser.add_argument("--mode", choices=["int8", "int4", "both"], default="both")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    print(f"[SUB_178] GPU = {torch.cuda.get_device_name(0)}  driver_idx={args.gpu}")

    cold = load_cold()
    pool = load_pool()
    pool_h = pool.pinned_pool_create(
        ctypes.c_size_t(args.total_limit), ctypes.c_int(args.numa))
    assert pool_h
    stream = pool.pinned_pool_stream_create()
    assert stream

    # Per-chunk shape: 16 * 2 * 128 = 4096 elems = 8 KB BF16, 4 KB INT8, 2 KB INT4
    # Cold-region multi-chunk batch sizes: 1, 8, 64, 256 chunks
    chunks_list = [1, 8, 64, 256, 1024]
    chunk_elems = 4096
    gs = 128
    rows = []

    modes = ["int8", "int4"] if args.mode == "both" else [args.mode]
    for mode in modes:
        for n_chunks in chunks_list:
            n_elems = n_chunks * chunk_elems
            out_bytes = n_elems * 2  # bf16
            in_bytes = n_elems if mode == "int8" else n_elems // 2

            # Allocate inputs (CPU side, not in pinned pool — represents the
            # cold-storage region which lives in plain CPU RAM).
            import numpy as np
            rng = np.random.default_rng(7)
            if mode == "int8":
                q_arr = rng.integers(-127, 128, size=n_elems, dtype=np.int8)
            else:
                # int4 nibbles packed
                nib = rng.integers(-8, 8, size=n_elems, dtype=np.int8)
                nib_u = (nib & 0x0F).astype(np.uint8)
                q_arr = np.zeros(n_elems // 2, dtype=np.uint8)
                q_arr[:] = nib_u[0::2] | (nib_u[1::2] << 4)
            n_groups = (n_elems + gs - 1) // gs
            scale_f32 = (rng.random(n_groups, dtype=np.float32) * 0.1 + 0.01).astype(np.float32)
            scale_bf16 = ((scale_f32.view(np.uint32) >> 16) & 0xFFFF).astype(np.uint16)
            q_p = q_arr.ctypes.data_as(ctypes.c_void_p)
            s_p = scale_bf16.ctypes.data_as(ctypes.c_void_p)

            # Pinned BF16 buffer from pool
            host_pinned = pool.pinned_pool_alloc(pool_h, ctypes.c_size_t(out_bytes))
            assert host_pinned, f"pool exhausted for {out_bytes} bytes"
            # GPU dst
            dev_buf = torch.empty(out_bytes, dtype=torch.uint8, device=dev)

            kernel_fn = (cold.cold_kv_int8_to_bf16 if mode == "int8"
                         else cold.cold_kv_int4_to_bf16)

            # (a) SEQUENTIAL — dequant then DMA
            # warmup
            for _ in range(5):
                kernel_fn(q_p, s_p, ctypes.c_void_p(host_pinned),
                          ctypes.c_int(n_elems), ctypes.c_int(gs))
                ev = pool.pinned_pool_push_async(
                    pool_h, ctypes.c_void_p(host_pinned),
                    ctypes.c_void_p(dev_buf.data_ptr()),
                    ctypes.c_size_t(out_bytes), ctypes.c_void_p(stream))
                pool.pinned_pool_event_sync(ctypes.c_void_p(ev))
                pool.pinned_pool_event_destroy(ctypes.c_void_p(ev))

            dequant_us = []
            dma_us = []
            total_us = []
            for _ in range(args.iters):
                t0 = time.perf_counter_ns()
                kernel_fn(q_p, s_p, ctypes.c_void_p(host_pinned),
                          ctypes.c_int(n_elems), ctypes.c_int(gs))
                t1 = time.perf_counter_ns()
                ev = pool.pinned_pool_push_async(
                    pool_h, ctypes.c_void_p(host_pinned),
                    ctypes.c_void_p(dev_buf.data_ptr()),
                    ctypes.c_size_t(out_bytes), ctypes.c_void_p(stream))
                pool.pinned_pool_event_sync(ctypes.c_void_p(ev))
                t2 = time.perf_counter_ns()
                pool.pinned_pool_event_destroy(ctypes.c_void_p(ev))
                dequant_us.append((t1 - t0) / 1000.0)
                dma_us.append((t2 - t1) / 1000.0)
                total_us.append((t2 - t0) / 1000.0)

            d_mean, d_p50, d_p99 = stats(dequant_us)
            m_mean, m_p50, m_p99 = stats(dma_us)
            t_mean, t_p50, t_p99 = stats(total_us)

            # (b) OVERLAP — issue DMA async, dequant next chunk in parallel
            # Here we simulate a 2-buffer pipeline. We allocate 2 pinned
            # buffers; on iter i we dequant into buf[i%2] then push prev buf.
            host2 = pool.pinned_pool_alloc(pool_h, ctypes.c_size_t(out_bytes))
            assert host2
            bufs = [host_pinned, host2]
            # warmup
            for _ in range(5):
                kernel_fn(q_p, s_p, ctypes.c_void_p(bufs[0]),
                          ctypes.c_int(n_elems), ctypes.c_int(gs))
                ev = pool.pinned_pool_push_async(
                    pool_h, ctypes.c_void_p(bufs[0]),
                    ctypes.c_void_p(dev_buf.data_ptr()),
                    ctypes.c_size_t(out_bytes), ctypes.c_void_p(stream))
                pool.pinned_pool_event_sync(ctypes.c_void_p(ev))
                pool.pinned_pool_event_destroy(ctypes.c_void_p(ev))

            overlap_us = []
            # We measure the time to process N=10 chunks back-to-back with overlap
            N_OL = 10
            for _ in range(max(1, args.iters // 5)):
                t0 = time.perf_counter_ns()
                evs = []
                for k in range(N_OL):
                    buf = bufs[k % 2]
                    kernel_fn(q_p, s_p, ctypes.c_void_p(buf),
                              ctypes.c_int(n_elems), ctypes.c_int(gs))
                    ev = pool.pinned_pool_push_async(
                        pool_h, ctypes.c_void_p(buf),
                        ctypes.c_void_p(dev_buf.data_ptr()),
                        ctypes.c_size_t(out_bytes), ctypes.c_void_p(stream))
                    evs.append(ev)
                # wait all
                for ev in evs:
                    pool.pinned_pool_event_sync(ctypes.c_void_p(ev))
                    pool.pinned_pool_event_destroy(ctypes.c_void_p(ev))
                t1 = time.perf_counter_ns()
                overlap_us.append((t1 - t0) / 1000.0 / N_OL)
            o_mean, o_p50, o_p99 = stats(overlap_us)

            row = {
                "mode": mode,
                "n_chunks": n_chunks,
                "n_elems": n_elems,
                "in_bytes": in_bytes,
                "out_bf16_bytes": out_bytes,
                "dequant_us_p50": d_p50,
                "dequant_us_p99": d_p99,
                "dma_us_p50": m_p50,
                "dma_us_p99": m_p99,
                "sequential_total_us_p50": t_p50,
                "sequential_total_us_p99": t_p99,
                "overlap_per_chunk_us_p50": o_p50,
                "overlap_per_chunk_us_p99": o_p99,
                "overlap_speedup": t_p50 / o_p50 if o_p50 > 0 else 0,
            }
            rows.append(row)
            print(f"[{mode}] chunks={n_chunks:>4} "
                  f"deq={d_p50:>8.2f}us  dma={m_p50:>8.2f}us  "
                  f"seq={t_p50:>8.2f}us  overlap={o_p50:>8.2f}us  "
                  f"speedup={row['overlap_speedup']:.2f}x")

            pool.pinned_pool_free(pool_h, ctypes.c_void_p(host_pinned))
            pool.pinned_pool_free(pool_h, ctypes.c_void_p(host2))
            del dev_buf
            torch.cuda.empty_cache()

    pool.pinned_pool_stream_destroy(ctypes.c_void_p(stream))
    pool.pinned_pool_destroy(ctypes.c_void_p(pool_h))

    out_path = HERE / args.out_json
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
