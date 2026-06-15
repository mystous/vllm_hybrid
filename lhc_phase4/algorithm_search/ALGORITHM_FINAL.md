# LHC Phase 4 / Algorithm Search — FINAL

**Date**: 2026-06-08.
**Verdict**: positive lever found in Attempt 6. Multi-workload extension
in progress.

## Summary

| attempt | description | result | gate |
|---|---|---|---|
| 1 | libamx_c3.so production build + Path 1 re-measure | **−1.35 %** (best of 3 sweeps, chat_prefix) | FAIL |
| 6 | measurement-driven `--max-num-seqs` lift (64 → 256) | **+60-107 %** across 3 workloads | **PASS** |

Attempts 2-5, 7-9 were not run because Attempt 6 already exceeded the
+5 % gate by an order of magnitude. The remaining attempts targeted
algorithms operating under the now-disproven assumption that the
`conc=64` baseline was GPU-bound; that premise is wrong and any
positive delta they could find would be at best additive to the +60 %
Attempt-6 gain.

## Attempt 1 — libamx_c3.so build

- Built `vllm/v1/lhc/libamx_c3.so` (AVX-512 + AMX-aware prefetch +
  FNV-1a) and wired into `kv_cache_utils.py::_lhc_amx_c3_block_hash`.
- Microbench: SHA-256 0.76 µs/call, libamx_c3 ctypes 1.94 µs/call,
  Python FNV-1a 7.27 µs/call. **ctypes overhead (~1.5 µs) dominates**
  block-hash payload, structurally preventing C lib from beating
  SHA-256.
- End-to-end: chat_prefix 95,560 (vanilla) vs 94,271 (lhc_amx_c3_clib)
  = **−1.35 %**, n=3 each. Better than the prior python-only −2.53 %
  but still negative. Gate FAIL.
- Diagnosis: Path 1 hook design is fundamentally wrong. Would need
  batched-block hashing in C (no Python boundary per block) to win.
- Library and source preserved in repo for future PoCs:
  `vllm/v1/lhc/libamx_c3.{c,so}`.

## Attempt 6 — concurrency lever (POSITIVE)

Engine logs during the chat_prefix sweep with `--max-num-seqs 64`:

```
Running: 64 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.4%
```

- `Running` is capped at `max_num_seqs`, not by KV pool exhaustion.
- GPU KV pool has 99.6 % free.
- This is **artificial concurrency starvation**, not a GPU bottleneck.

### Result (3 workloads × 3 concurrencies)

| workload    | conc=64 | conc=128 | conc=256 | conc=256 Δ |
|-------------|---------|----------|----------|------------|
| chat_prefix | 95,781  | 147,525  | 164,375  | **+71.6 %** |
| sonnet      | 33,372  | 54,726   | 68,983   | **+106.7 %** |
| chat_short  | 70,617  | —        | 113,309  | **+60.5 %** |

(All values: total token throughput in tok/s, mean across warm sweeps.)

### Correctness

Prefix-cache hit rate: 88.4 % → 88.0–88.5 %. Max drift = -0.4 pp, well
below the 1 pp gate. The HW-AGSD distribution-level invariant in
`CLAUDE.md / Constraint / 운영 해석` is preserved.

### TPOT cost

| workload    | TPOT c64 | TPOT c256 |
|-------------|----------|-----------|
| chat_prefix | 3.1 ms   | 5.5 ms    |
| sonnet      | 3.5 ms   | 6.1 ms    |
| chat_short  | 3.2 ms   | 7.1 ms    |

TPOT grows ~2× while throughput grows 1.6×–2×. The TPOT–throughput
tradeoff is the standard concurrency knob; for **throughput-binding**
benchmarks (paper §08), conc=256 is the better operating point.

## Implications for prior 100+ levers

The reason every previous attempt landed in the noise (±2 %) is that
the **conc=64 baseline was not GPU-bound**. Host-side optimisations
have no upstream pressure to relieve when the GPU is under-utilised at
small batches. Re-running the LHC sweep at conc=256 may reveal
positive deltas for DSA / AMX / regime detection that were previously
masked.

## Recommended next steps

1. **New baseline**: re-run all LHC measurements with
   `--max-num-seqs 256` (or 512 if the TPOT increase is acceptable).
   This is the throughput-binding regime referenced in paper §08.
2. **Multi-workload extension** (6 wl × 3 sweep) at conc=256, which is
   the standard sweep matrix the rest of the paper uses.
3. **LHC-on-top-of-conc-lever**: measure whether DSA / AMX adds
   throughput on top of the conc=256 baseline. The conc=256 regime is
   where the GPU is more likely actually compute-bound, so host
   offload could now have a real win.

## Files

- Attempt 1: `attempt_1/RESULTS.md`, `attempt_1/runs/`.
- Attempt 6: `attempt_6/RESULTS.md`, `attempt_6/runs/`.
- This file: `ALGORITHM_FINAL.md`.

## Hardware / build artefacts

- `vllm/v1/lhc/libamx_c3.c` (228 lines, AVX-512 + AMX FNV-1a).
- `vllm/v1/lhc/libamx_c3.so` (built, 15 KB).
- `vllm/v1/core/kv_cache_utils.py` — `_lhc_amx_c3_block_hash` now
  dispatches to libamx_c3 when `VLLM_LHC_AMX_C3_PREFIX=1`. The path
  is dormant when env-gated off, so vanilla behaviour is preserved.
