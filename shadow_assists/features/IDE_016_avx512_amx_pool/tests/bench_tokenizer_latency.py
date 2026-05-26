"""SUB_171 microbench — AVX-512 batch detokenize vs python baseline.

Workload:
    32 sequences × ~64 token_ids × random ids from Qwen-scale vocab (152,064)
    -> measure p50/p99 latency of one batch decode.

Two baselines:
    - "scalar"  : C++ scalar path (kernel built without SIMD)
    - "python"  : pure python loop using the same flat vocab (representative
                  of `tokenizer.decode` cost without the HF library overhead)

Targets (per task.md TSK_024):
    p50 ≥ 1.4× speedup vs python baseline (kernel-only)
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from avx512_amx_pool import BatchDetokenizer, _core as C  # noqa: E402


def _random_vocab(V: int, max_piece: int = 8, seed: int = 0):
    rng = random.Random(seed)
    import string
    pool = string.ascii_letters + string.digits + " .,!?_-"
    return ["".join(rng.choice(pool) for _ in range(rng.randint(0, max_piece)))
            for _ in range(V)]


def _random_seqs(B, L, V, seed=0):
    rng = random.Random(seed)
    return [[rng.randint(0, V - 1) for _ in range(L)] for _ in range(B)]


def _bench_once(fn, *args, repeats=20):
    timings = []
    # warmup
    for _ in range(3):
        fn(*args)
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        timings.append((time.perf_counter() - t0) * 1e6)  # us
    timings.sort()
    return {
        "p50": timings[len(timings) // 2],
        "p99": timings[int(len(timings) * 0.99)],
        "mean": statistics.mean(timings),
        "min": timings[0],
        "max": timings[-1],
    }


def _python_decode(detok: BatchDetokenizer, seqs):
    """Pure python concat using the flat vocab (no HF tokenizer)."""
    return detok._scalar_decode(seqs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=152064,
                   help="Qwen 2.5 vocab size")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seqlen", type=int, default=64)
    p.add_argument("--max-piece", type=int, default=8)
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()

    print(f"=== SUB_171 microbench — V={args.vocab} B={args.batch} "
          f"L={args.seqlen} max_piece={args.max_piece} ===")
    print(f"cpu_has_avx512: {C.cpu_has_avx512()}")

    vocab = _random_vocab(args.vocab, max_piece=args.max_piece, seed=11)
    detok = BatchDetokenizer.from_vocab_strings(vocab)
    seqs = _random_seqs(args.batch, args.seqlen, args.vocab, seed=23)

    # AVX-512 path (kernel batch_decode_bytes use_avx512=True)
    avx = _bench_once(detok.batch_decode_bytes, seqs,
                      repeats=args.repeats)

    # scalar C++ path
    def _scalar_call(s):
        return detok.batch_decode_bytes(s, use_avx512=False)

    sca = _bench_once(_scalar_call, seqs, repeats=args.repeats)

    # python path
    py = _bench_once(_python_decode, detok, seqs, repeats=args.repeats)

    print("\n%-12s %10s %10s %10s %10s" %
          ("path", "p50_us", "p99_us", "mean_us", "min_us"))
    for name, r in [("avx512", avx), ("c++scalar", sca), ("python", py)]:
        print("%-12s %10.2f %10.2f %10.2f %10.2f" %
              (name, r["p50"], r["p99"], r["mean"], r["min"]))

    print("\nspeedups (vs python p50):")
    print(f"  avx512    : {py['p50'] / avx['p50']:.2f}×")
    print(f"  c++scalar : {py['p50'] / sca['p50']:.2f}×")
    print("speedups (vs python p99):")
    print(f"  avx512    : {py['p99'] / avx['p99']:.2f}×")
    print(f"  c++scalar : {py['p99'] / sca['p99']:.2f}×")

    # correctness gate
    a = detok.batch_decode_bytes(seqs, use_avx512=True)
    s = detok.batch_decode_bytes(seqs, use_avx512=False)
    assert a == s, "AVX-512 vs scalar diverged"
    print("\ncorrectness: PASS (avx-512 == scalar byte-exact)")


if __name__ == "__main__":
    main()
