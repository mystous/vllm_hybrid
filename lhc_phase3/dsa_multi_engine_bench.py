"""LHC Phase 3 — Task E: DSA multi-engine integrated BW measurement.

Splits a 256 MB host→host copy into 8 chunks (32 MB each), submits one
chunk per WQ via 8 worker threads. Each WQ submit issues multiple
2 MB descriptors (== sysfs max_transfer_size cap for shared WQ on this
host) and polls completion records in user space.

Measurement modes (set via env LHC_BENCH_MODE):
  * ``dsa_local``     — only ranks 0–3 (dsa0, NUMA node 0)
  * ``dsa_remote``    — only ranks 4–7 (dsa1, NUMA node 1)
  * ``dsa_full``      — all 8 ranks
  * ``cuda_memcpy``   — torch cudaMemcpyAsync (D2D 0→0), single stream
  * ``cpu_memcpy``    — single-thread C memcpy baseline (sanity)

Outputs: aggregate BW (GB/s) and per-WQ throughput, optionally written
to JSON for the Task E report.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "/workspace/host_vllm_hybrid")

# Direct ctypes binding to libdsa_lane.so per WQ (one CDLL instance per
# thread would re-use the singleton; we need *per-thread* explicit dev
# paths, so we open() + mmap() the portal in pure Python ctypes here.)

# --- ENQCMD inline-asm via a tiny inline trampoline written to a tmpfile ---

LIB_PATH = Path("/workspace/host_vllm_hybrid/vllm/v1/lhc/libdsa_lane.so")

# Use a helper trampoline lib that exposes per-fd state.
HELPER_SRC = r"""
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <linux/idxd.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

typedef struct {
    int fd;
    volatile void *portal;
    int is_shared;
} dsa_ctx_t;

static int read_mode(const char *dev_path) {
    const char *p = strrchr(dev_path, '/');
    if (!p) return 0;
    p++;
    char sysfs[256];
    snprintf(sysfs, sizeof(sysfs), "/sys/bus/dsa/devices/%s/mode", p);
    FILE *f = fopen(sysfs, "r");
    if (!f) return 0;
    char buf[32] = {0};
    size_t n = fread(buf, 1, sizeof(buf)-1, f);
    fclose(f);
    return (n > 0 && strncmp(buf, "shared", 6) == 0) ? 1 : 0;
}

dsa_ctx_t *dsa_open(const char *dev_path) {
    int fd = open(dev_path, O_RDWR);
    if (fd < 0) return NULL;
    void *portal = mmap(NULL, 4096, PROT_WRITE,
                        MAP_SHARED | MAP_POPULATE, fd, 0);
    if (portal == MAP_FAILED) { close(fd); return NULL; }
    dsa_ctx_t *c = (dsa_ctx_t *)malloc(sizeof(*c));
    c->fd = fd; c->portal = portal; c->is_shared = read_mode(dev_path);
    return c;
}

static inline int enqcmd_submit(volatile void *p, const void *d) {
    uint8_t retry;
    asm volatile (
        ".byte 0xf2, 0x0f, 0x38, 0xf8, 0x02\n\t"
        "setz %0"
        : "=r"(retry)
        : "a"(p), "d"(d)
        : "memory", "cc");
    return retry ? -1 : 0;
}

/* one descriptor, n bytes, sync wait. Returns 0 on success. */
int dsa_one(dsa_ctx_t *c, void *dst, const void *src, size_t n) {
    struct dsa_hw_desc desc __attribute__((aligned(64)));
    struct dsa_completion_record comp __attribute__((aligned(32)));
    memset(&desc, 0, sizeof(desc));
    memset(&comp, 0, sizeof(comp));
    desc.flags = IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_BOF;
    desc.opcode = DSA_OPCODE_MEMMOVE;
    desc.completion_addr = (uint64_t)&comp;
    desc.xfer_size = (uint32_t)n;
    desc.src_addr = (uint64_t)src;
    desc.dst_addr = (uint64_t)dst;
    if (c->is_shared) {
        int r = 0;
        while (enqcmd_submit(c->portal, &desc) != 0) {
            _mm_pause();
            if (++r > 200000) return -4;
        }
    } else {
        _movdir64b((void *)c->portal, &desc);
    }
    /* poll */
    int spins = 0;
    while (comp.status == DSA_COMP_NONE) {
        _mm_pause();
        if (++spins > 50000000) return -3;
    }
    if ((comp.status & DSA_COMP_STATUS_MASK) != DSA_COMP_SUCCESS) {
        return -(int)comp.status;
    }
    return 0;
}

/* split n into chunks of max_xfer and submit sequentially. */
int dsa_copy(dsa_ctx_t *c, void *dst, const void *src, size_t n,
             size_t max_xfer) {
    size_t off = 0;
    while (off < n) {
        size_t chunk = (n - off > max_xfer) ? max_xfer : (n - off);
        int rc = dsa_one(c, (char *)dst + off, (const char *)src + off, chunk);
        if (rc != 0) return rc;
        off += chunk;
    }
    return 0;
}

void dsa_close(dsa_ctx_t *c) {
    if (!c) return;
    if (c->portal) munmap((void *)c->portal, 4096);
    if (c->fd >= 0) close(c->fd);
    free(c);
}
"""

HELPER_LIB_PATH = Path("/tmp/libdsa_helper.so")


def build_helper() -> ctypes.CDLL:
    src = Path("/tmp/libdsa_helper.c")
    src.write_text(HELPER_SRC)
    import subprocess
    rc = subprocess.run(
        [
            "gcc", "-O3", "-march=native", "-mmovdir64b", "-fPIC", "-shared",
            "-o", str(HELPER_LIB_PATH), str(src),
        ],
        check=False, capture_output=True, text=True,
    )
    if rc.returncode != 0:
        raise RuntimeError(f"helper build failed: {rc.stderr}")
    lib = ctypes.CDLL(str(HELPER_LIB_PATH))
    lib.dsa_open.argtypes = [ctypes.c_char_p]
    lib.dsa_open.restype = ctypes.c_void_p
    lib.dsa_copy.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t,
    ]
    lib.dsa_copy.restype = ctypes.c_int
    lib.dsa_close.argtypes = [ctypes.c_void_p]
    lib.dsa_close.restype = None
    return lib


# ------------- bench ----------------

def aligned_buf(nbytes: int):
    """Allocate page-aligned host buffer (4 KB aligned)."""
    pages = (nbytes + 4095) // 4096
    return (ctypes.c_uint8 * (pages * 4096))()


def dsa_bench(ranks: list[int], total_bytes: int, max_xfer: int,
              warmup: int, iters: int) -> dict:
    lib = build_helper()
    contexts: dict[int, int] = {}
    for r in ranks:
        dev = f"/dev/dsa/wq{r // 4}.{r % 4}"
        ctx = lib.dsa_open(dev.encode())
        if ctx is None:
            raise RuntimeError(f"open({dev}) failed")
        contexts[r] = ctx

    chunk = total_bytes // len(ranks)
    # Round chunk down to multiple of max_xfer for clean splits.
    chunk = (chunk // max_xfer) * max_xfer
    actual_total = chunk * len(ranks)

    # one src + one dst per rank, contiguous in same process address space.
    src_bufs = {r: aligned_buf(chunk) for r in ranks}
    dst_bufs = {r: aligned_buf(chunk) for r in ranks}
    for r in ranks:
        ctypes.memset(src_bufs[r], 0xA5, chunk)

    src_ptrs = {r: ctypes.cast(src_bufs[r], ctypes.c_void_p).value for r in ranks}
    dst_ptrs = {r: ctypes.cast(dst_bufs[r], ctypes.c_void_p).value for r in ranks}

    per_rank_times: dict[int, list[float]] = {r: [] for r in ranks}

    def run_one(rank: int, barrier: threading.Barrier,
                done_event: threading.Event,
                start_event: threading.Event) -> float:
        barrier.wait()
        start_event.wait()
        t0 = time.perf_counter_ns()
        rc = lib.dsa_copy(contexts[rank], dst_ptrs[rank], src_ptrs[rank],
                          chunk, max_xfer)
        dt = (time.perf_counter_ns() - t0) / 1e9
        per_rank_times[rank].append((dt, rc))
        done_event.set()
        return dt

    # warmup loop
    for it in range(warmup + iters):
        barrier = threading.Barrier(len(ranks) + 1)
        start_event = threading.Event()
        done_events = {r: threading.Event() for r in ranks}
        threads = []
        for r in ranks:
            t = threading.Thread(
                target=run_one, args=(r, barrier, done_events[r], start_event),
            )
            t.start()
            threads.append(t)
        barrier.wait()
        # all threads parked; release sim
        global_t0 = time.perf_counter_ns()
        start_event.set()
        for t in threads:
            t.join()
        global_dt = (time.perf_counter_ns() - global_t0) / 1e9
        if it < warmup:
            # discard warmup samples
            for r in ranks:
                per_rank_times[r].pop()

    # Reduce
    last_iter_global_bw = actual_total / global_dt / 1e9
    per_rank_bw_means = {}
    for r in ranks:
        ts = [t[0] for t in per_rank_times[r]]
        per_rank_bw_means[r] = (chunk / (sum(ts) / len(ts))) / 1e9

    # close contexts
    for r in ranks:
        lib.dsa_close(contexts[r])
    return {
        "ranks": ranks,
        "n_ranks": len(ranks),
        "per_rank_chunk_bytes": chunk,
        "max_xfer_bytes": max_xfer,
        "actual_total_bytes": actual_total,
        "iters": iters,
        "warmup": warmup,
        "global_bw_GBps": last_iter_global_bw,
        "per_rank_bw_GBps": per_rank_bw_means,
        "global_dt_sec": global_dt,
    }


def cpu_memcpy_bench(total_bytes: int, iters: int) -> dict:
    """Single-thread C memcpy baseline."""
    src = aligned_buf(total_bytes)
    dst = aligned_buf(total_bytes)
    ctypes.memset(src, 0xA5, total_bytes)
    libc = ctypes.CDLL("libc.so.6")
    libc.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    libc.memcpy.restype = ctypes.c_void_p
    sp = ctypes.cast(src, ctypes.c_void_p).value
    dp = ctypes.cast(dst, ctypes.c_void_p).value
    # warmup
    libc.memcpy(dp, sp, total_bytes)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        libc.memcpy(dp, sp, total_bytes)
        ts.append(time.perf_counter_ns() - t0)
    mean_s = (sum(ts) / len(ts)) / 1e9
    return {
        "mode": "cpu_memcpy",
        "total_bytes": total_bytes,
        "iters": iters,
        "global_bw_GBps": total_bytes / mean_s / 1e9,
        "mean_dt_sec": mean_s,
    }


def cuda_memcpy_bench(total_bytes: int, iters: int, direction: str = "h2d") -> dict:
    """torch cudaMemcpyAsync (h2d / d2h / d2d) using pinned host mem."""
    import torch
    if not torch.cuda.is_available():
        return {"mode": "cuda_memcpy", "error": "no CUDA"}
    if direction in ("h2d", "d2h"):
        host = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
        dev = torch.empty(total_bytes, dtype=torch.uint8, device="cuda:0")
        host.fill_(0xA5)
        src, dst = (host, dev) if direction == "h2d" else (dev, host)
    else:  # d2d
        src = torch.empty(total_bytes, dtype=torch.uint8, device="cuda:0")
        dst = torch.empty(total_bytes, dtype=torch.uint8, device="cuda:0")
        src.fill_(0xA5)
    torch.cuda.synchronize()
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        ts.append(time.perf_counter_ns() - t0)
    mean_s = (sum(ts) / len(ts)) / 1e9
    return {
        "mode": f"cuda_memcpy_{direction}",
        "total_bytes": total_bytes,
        "iters": iters,
        "global_bw_GBps": total_bytes / mean_s / 1e9,
        "mean_dt_sec": mean_s,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["dsa_local", "dsa_remote", "dsa_full",
                            "cuda_h2d", "cuda_d2h", "cuda_d2d", "cpu_memcpy"])
    p.add_argument("--total-bytes", type=int, default=256 * 1024 * 1024)
    p.add_argument("--max-xfer", type=int, default=2 * 1024 * 1024)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    if args.mode == "dsa_local":
        res = dsa_bench(list(range(0, 4)), args.total_bytes, args.max_xfer,
                        args.warmup, args.iters)
        res["mode"] = "dsa_local"
    elif args.mode == "dsa_remote":
        res = dsa_bench(list(range(4, 8)), args.total_bytes, args.max_xfer,
                        args.warmup, args.iters)
        res["mode"] = "dsa_remote"
    elif args.mode == "dsa_full":
        res = dsa_bench(list(range(0, 8)), args.total_bytes, args.max_xfer,
                        args.warmup, args.iters)
        res["mode"] = "dsa_full"
    elif args.mode == "cpu_memcpy":
        res = cpu_memcpy_bench(args.total_bytes, args.iters)
    elif args.mode in ("cuda_h2d", "cuda_d2h", "cuda_d2d"):
        direction = args.mode.split("_")[1]
        res = cuda_memcpy_bench(args.total_bytes, args.iters, direction)
    else:
        raise ValueError(args.mode)

    print(json.dumps(res, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
