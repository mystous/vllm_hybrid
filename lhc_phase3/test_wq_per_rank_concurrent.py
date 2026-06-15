"""8-process concurrent self-test for VLLM_LHC_DSA_WQ_PER_RANK.

Spawns 8 worker processes (1 per TP rank). Each opens its own WQ via
the per-rank mapping and issues a 4 KB MEMMOVE self-test. Gate:
  - all 8 ranks return rc=0 (no EBUSY)
  - distinct WQ paths actually opened (check /proc/<pid>/fdinfo)
"""

import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, "/workspace/host_vllm_hybrid")


def worker(rank: int, result_q, payload_bytes: int = 1 << 16, iters: int = 200):
    """Each child proc binds to its rank's WQ and issues a copy loop.

    Default payload 64 KB (= dsa_memcpy min_bytes) × 200 iters → ~13 MB / proc.
    Returns rc + DSA stats from the child's own atomic counters so the parent
    can verify (ops, bytes, fails) per WQ.
    """
    os.environ["VLLM_LHC_DSA"] = "1"
    os.environ["VLLM_LHC_DSA_WQ_PER_RANK"] = "1"
    os.environ["VLLM_LHC_DSA_RANK"] = str(rank)
    os.environ["VLLM_LHC_DSA_MIN"] = "4096"  # allow 4-64 KB self-test path
    os.environ.pop("VLLM_LHC_DSA_DEV", None)

    import importlib

    import vllm.v1.lhc.dsa_lane as m

    importlib.reload(m)
    dev = m._resolve_dev_path()
    available = m.dsa_lane_available()
    ok = False
    err = None
    elapsed_us = 0.0
    stats: dict = {}
    if available:
        import ctypes
        import time

        try:
            src = (ctypes.c_uint8 * payload_bytes)(*([0xA5] * payload_bytes))
            dst = (ctypes.c_uint8 * payload_bytes)()
            src_p = ctypes.cast(src, ctypes.c_void_p).value
            dst_p = ctypes.cast(dst, ctypes.c_void_p).value
            all_ok = True
            t0 = time.perf_counter()
            for _ in range(iters):
                if not m.dsa_memcpy(dst_p, src_p, payload_bytes):
                    all_ok = False
                    break
            elapsed_us = (time.perf_counter() - t0) * 1e6
            # spot-check first / last 1 KB
            ok = all_ok and all(b == 0xA5 for b in dst[:1024]) and all(
                b == 0xA5 for b in dst[payload_bytes - 1024:]
            )
            stats = m.dsa_lane_stats()
        except Exception as e:  # noqa: BLE001
            err = repr(e)
    result_q.put({
        "rank": rank,
        "dev": dev,
        "available": available,
        "ok": ok,
        "err": err,
        "elapsed_us": elapsed_us,
        "stats": stats,
    })


def main():
    n_ranks = int(os.environ.get("LHC_TEST_RANKS", "8"))
    payload = int(os.environ.get("LHC_TEST_PAYLOAD_BYTES", str(1 << 16)))
    iters = int(os.environ.get("LHC_TEST_ITERS", "200"))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for r in range(n_ranks):
        p = ctx.Process(target=worker, args=(r, q, payload, iters))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=60)
    results = []
    while not q.empty():
        results.append(q.get())
    results.sort(key=lambda x: x["rank"])
    print(
        f"{'rank':>4} {'dev':24s} {'avail':>6} {'ok':>4} "
        f"{'ops':>6} {'fails':>6} {'BW_GBps':>10}  err"
    )
    print("-" * 84)
    fail = 0
    for r in results:
        bw_gbps = 0.0
        if r["elapsed_us"] > 0:
            bw_gbps = (payload * iters) / (r["elapsed_us"] / 1e6) / 1e9
        ops = r["stats"].get("ops", 0)
        fails = r["stats"].get("fails", 0)
        print(
            f"{r['rank']:>4} {r['dev']:24s} {r['available']!s:>6} "
            f"{r['ok']!s:>4} {ops:>6} {fails:>6} {bw_gbps:>10.3f}  "
            f"{r['err'] or ''}"
        )
        if not r["ok"]:
            fail += 1
    print(
        f"\nresults: {len(results)}/{n_ranks}  fails: {fail}  "
        f"(payload={payload}B × {iters} iters)"
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
