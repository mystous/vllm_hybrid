# Attempt 1 — libamx_c3.so production build + Path 1 re-measurement

**Status**: FAIL (gate −5%, measured −1.35%)
**Date**: 2026-06-08

## Build

- New C source: `vllm/v1/lhc/libamx_c3.c` (228 lines).
- Build: `gcc -O3 -march=sapphirerapids -mamx-tile -mamx-bf16 -mamx-int8 -mavx512f -mavx512vl -mavx512bw -mavx512dq -fPIC -shared`.
- Output: `vllm/v1/lhc/libamx_c3.so` (15760 B). Symbols: `amx_c3_block_hash`, `amx_c3_scan`, `amx_c3_stats`, `amx_c3_ready`.
- Algorithm: AVX-512 cache-warm prefetch + scalar FNV-1a chain (FNV-1a is non-associative; we issue 64-byte prefetched windows). For ≥ 16 KB, an AMX-aware L2 prefetch sweep is added.

## Integration

- `vllm/v1/core/kv_cache_utils.py::_lhc_amx_c3_block_hash` now calls `lib.amx_c3_block_hash` via ctypes when `VLLM_LHC_AMX_C3_PREFIX=1` is set. Python loop preserved as fallback. Hook fires per block (counter verified).

## Microbench (50 k calls, 16-token block)

| Path        | µs / call | vs vanilla |
|-------------|-----------|------------|
| SHA-256 (vanilla, hashlib C) | 0.76 | baseline |
| libamx_c3 C path (ctypes)    | 1.94 | 2.5× slower |
| Python FNV-1a loop (old)     | 7.27 | 9.5× slower |

ctypes call overhead ≈ 1.5 µs dominates the per-call cost; AVX-512 prefetch never amortises across the 64-byte block payload.

## End-to-end (Llama-3.1-8B TP=8, chat_prefix 500p × conc=64, ~88% prefix-cache hit)

| config | s1 tot tok/s | s2 | s3 | mean | std |
|---|---|---|---|---|---|
| vanilla        | 94745 | 96670 | 95265 | 95560 | 996 |
| lhc_amx_c3_clib| 95151 | 93417 | 94245 | 94271 | 867 |

**Δ = −1.35 %** (s1 +0.4%, s2 −3.4%, s3 −1.1%) — gate FAIL.

## Diagnosis

The Path 1 hook design is structurally unable to beat SHA-256 because the per-call ctypes thunk (~1.5 µs) is itself slower than the entire vanilla hash. A C library cannot help unless the FFI boundary is removed (batched call hashing N blocks at once, or scheduler-side hot loop rewritten in C). This is a Python-side problem, not a SIMD problem.

## Revert

`vllm/v1/core/kv_cache_utils.py` keeps the new C-path code (it is env-gated and inactive when `VLLM_LHC_AMX_C3_PREFIX=0`). `libamx_c3.c/.so` remain in repo as build artefacts for future PoCs.

## Hook-call evidence

```
(EngineCore) INFO ... [kv_cache_utils.py:585] LHC Phase 4 — libamx_c3.so loaded from /workspace/host_vllm_hybrid/vllm/v1/lhc/libamx_c3.so (C fast path enabled)
(EngineCore) INFO ... [kv_cache_utils.py:605] LHC Phase 4 Option C — Path 1: AMX C3 prefix hash chain enabled
```

Next: Attempt 2 — KV-aware request batching.
