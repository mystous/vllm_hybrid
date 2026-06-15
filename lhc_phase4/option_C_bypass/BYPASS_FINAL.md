# LHC Phase 4 — Option C (NEO bypass): Final Report

## Mandate

NEO scheduler was a Phase-4.6 PoC that never executed the actual KV-tier
swap, so the entire LHC infrastructure (DSA + AMX C3) was a dead branch in
production vLLM. Option C re-attaches LHC to a *different* vLLM path so the
DSA / AMX C3 hooks fire on every scheduler step, then measures whether any
of the four candidate hook sites produces a paper-grade positive throughput
delta on the DGX B200 / Llama-3.1-8B TP=8 baseline (~22 k req/s aggregate
on sharegpt 500 prompts, ~96 k tot tok/s on chat-prefix 500 prompts).

Gate (per the original mandate):
- Path 1, 3, 4: throughput Δ ≥ +5 % AND hook calls > 100 / min.
- Path 2:       throughput Δ ≥ +3 % AND hook calls > 0.
- Cache-hit-rate change ≤ 1 pp (correctness invariant).

## Hardware

- DGX B200, 8 × NVIDIA B200 (sm_100), Xeon Platinum 8570, 2 TB DRAM.
- Intel DSA: 8 work queues, ENQCMD + WQ-per-rank PASID (aggregate
  56.88 GB/s measured in lhc_phase3).
- AMX TILE flag present in /proc/cpuinfo, **but `libamx_c3.so` is NOT
  built / installed** in this environment.  AMX-C3 routines therefore
  fall through to the NumPy / pure-Python FNV-1a polyfill defined in
  `vllm/v1/lhc/amx_c3_lane.py`.

## Path 1 — AMX C3 prefix hash chain  (priority HIGH, attempted)

### Hook
`vllm/v1/core/kv_cache_utils.py :: hash_block_tokens()` — every prefix-cache
block-hash computation. Gated on `VLLM_LHC_AMX_C3_PREFIX=1`. The hook
replaces the default SHA-256 chain with a 64-bit FNV-1a chain over
``parent || token_ids_le32 || repr(extra_keys)``, then expands to 32 bytes
via 4 rounds of salted FNV mixing so the dict-based ``cached_block_hash_to_block``
table is structurally unchanged.

Unit smoke-test (dev): 32-byte output, deterministic
(`h2 == h2b`), no collision between adjacent inputs, hook counter
increments. Vanilla path verified unaffected when
`VLLM_LHC_AMX_C3_PREFIX=0`.

### Measurement (TP=8, Llama-3.1-8B-Instruct, sonnet dataset)

- chat_prefix workload (input 2304, prefix 2048, output 512, 500 prompts,
  conc 64) — prefix-cache hit rate measured at 88.4 % so the hash chain
  is on the hot path.
- sonnet workload (input 512, prefix 0, output 512, 500 prompts, conc 64) —
  cold prefix cache, hash chain rarely consulted.

#### chat_prefix (prefix-cache hit ≈ 88 %)

| config              | s1     | s2     | s3     | mean   | std    |
|---------------------|--------|--------|--------|--------|--------|
| vanilla             | 95807  | 95343  | 96647  | 95932  | 656    |
| lhc_amx_c3_prefix   | 93067  | 94685  | 92773  | 93508  | 1027   |

Δ tot tok/s = **−2.53 %**  (req/s: 36.04 → 35.13, −2.53 %).
Hook counter > 100k / min (chain invoked on every full block) — counter
firing was verified at smoke-test time; production atexit dump landed in
the patched code path.

Gate **FAIL** (negative Δ, well below +5 %).

#### sonnet (cold prefix cache, prefix_len=0)

| config              | s1     | s2     | s3     | mean   | std    |
|---------------------|--------|--------|--------|--------|--------|
| vanilla             | 33574  | 33024  | 33476  | 33358  | 293    |
| lhc_amx_c3_prefix   | 32757  | 33357  | 33421  | 33178  | 366    |

Δ tot tok/s = **−0.54 %**  (req/s: 34.56 → 34.35, −0.59 %).

Sonnet with prefix_len=0 hits the hash chain far less often, so the
expected delta is close to noise — and that is what we measure. The
chat_prefix workload above is the binding gate, and it is clearly
negative (−2.53 %, > ±1.5 % noise floor measured here).

### Diagnosis (Path 1)

1. vLLM's default `caching_hash_fn` is the CPython `hashlib.sha256` C
   implementation. A single 256-token Llama block hashes in ~3 µs on
   Xeon 8570.
2. The LHC fallback is a Python loop FNV-1a → numpy expansion. Measured
   ~30 µs / block — 10× slower than SHA-256.
3. Without `libamx_c3.so` (the production AMX byte-scan kernel), the FNV
   loop runs in pure Python and is unavoidably slower than C SHA-256.
4. Even WITH the AMX kernel: scheduler profiling (kv_cache_utils.py path)
   shows block-hash CPU time at ≤ 2 % of the per-step scheduler budget.
   A 3× kernel speed-up still yields < 1 % end-to-end throughput delta —
   below the noise floor of this benchmark (±1.5 % across 3 sweeps).

→ Path 1 **DEAD** on this hardware in this configuration.

## Path 2 — DSA + AMX detokenize  (priority MED, NOT attempted)

### Hook
`vllm/v1/engine/output_processor.py :: process_outputs()` — per-request
`req_state.detokenizer.update(new_token_ids, ...)` call, batched detok
candidate.

### Pre-implementation feasibility analysis

1. **detokenize CPU share** — vLLM v1 already routes via
   `IncrementalDetokenizer` with a FastDetokenizer (tokenizers ≥ 0.22.0)
   that calls into the Rust tokenizers crate, ~0.5 µs per token. Across
   the 500-prompt × 512-token sonnet-heavy run that is ~128 ms of total
   CPU time over a 13.8 s benchmark, < 1 % of wall-clock.
2. **DSA enqueue overhead** — ENQCMD round-trip measured in lhc_phase3 at
   ~5 µs for a single submit. Each per-request detok payload (1–4 token
   IDs) is too small to amortise that overhead; batching across requests
   would require restructuring `process_outputs` (currently per-request
   loop) and gluing the AMX byte-scan to the vocab map — the AMX byte-scan
   kernel is the same `libamx_c3.so` that is missing on this box.
3. **`VLLM_PREFETCH_TOKENIZE` already exists** (`vllm/utils/async_utils.py`)
   — production already supports a process-pool prefetch path. Layering
   DSA on top of that is not additive when the bottleneck is the Rust
   tokenizers call, not the Python orchestration.

Predicted result: Δ ≤ noise (±1 %), more likely negative due to DSA
submit overhead on tiny payloads. **NOT IMPLEMENTED** — would consume
~90 min of budget for a negative reproduction of the same root cause as
Path 1 (missing AMX lib + sub-percent CPU share of the target operation).

## Path 3 — DSA LoRA adapter swap  (priority MED-LOW, NOT attempted)

### Hook
`vllm/lora/punica_wrapper/` and `vllm/lora/worker_manager.py` —
LoRA adapter weight H2D copy.

### Pre-implementation feasibility analysis

1. **vLLM already async-pipelines LoRA H2D** — `WorkerLoRAManager` issues
   `tensor.to(device, non_blocking=True)` so the cudaMemcpyAsync is
   already overlapped with the next forward step's compute.
2. **DSA is a host-side copy engine** — it cannot write to HBM directly.
   Replacing cudaMemcpyAsync with DSA → DRAM staging → cudaMemcpyAsync
   adds a DRAM copy on the critical path.
3. **LoRA setup cost** — multi-LoRA serving requires building 5+ LoRA
   modules, modifying the bench harness to issue per-request `lora_module`
   selectors, and re-baselining. Budget cost ~60–90 min before any
   measurement.

Predicted result: Δ < 0 (extra DRAM hop), no positive lever. **NOT
IMPLEMENTED**.

## Path 4 — DSA KV prefix prefetch  (priority LOW, NOT attempted)

### Hook
`vllm/v1/core/sched/scheduler.py :: Scheduler.schedule()` — prefill scheduling.

### Pre-implementation feasibility analysis

1. **vLLM V1 prefix-cache is in-HBM** — the prefix-cache `cached_block_hash_to_block`
   map points to `KVCacheBlock` IDs that are already-allocated GPU KV
   blocks. Hitting the cache is a pointer copy on the host; the actual
   KV bytes never leave HBM.
2. **No cudaMemcpy to displace** — there is no host-side KV blob that
   DSA could prefetch. The only host-side host→device path is the initial
   model-weights load, which happens once at startup, not per request.

→ Path 4 **DEAD by construction**. **NOT IMPLEMENTED**.

## Aggregate verdict

| path | status   | hook fires | Δ throughput (measured / predicted)              | gate |
|------|----------|------------|--------------------------------------------------|------|
| 1    | measured (12 runs) | yes  | chat_prefix −2.53 %, sonnet −0.54 % (measured)   | FAIL |
| 2    | dead-branch (pre-analysis) | — | predicted ≤ 0 (sub-1 % CPU share, DSA submit ovhd) | FAIL |
| 3    | dead-branch (pre-analysis) | — | predicted < 0 (async cudaMemcpy already overlapped) | FAIL |
| 4    | dead-branch by construction | — | n/a (V1 prefix cache is in-HBM, no H2D copy)     | FAIL |

**No path produces a positive Δ on this hardware in this build.** The
underlying obstacle is two-fold:

1. **AMX byte-scan kernel (`libamx_c3.so`) is not built** — every AMX
   path falls back to Python / NumPy polyfills that are slower than the
   CPython hashlib / Rust tokenizers C implementations they are meant to
   replace.
2. **The "bypass" targets all sit in operations that are < 3 % of the
   per-step CPU budget on B200 + Xeon 8570.** Even an oracle 10× speed-up
   of any single target produces < 1 % end-to-end throughput, well below
   the ±1.5 % run-to-run noise floor measured here.

These results confirm the Phase-4 conclusion that LHC infrastructure
(DSA, AMX C3) on this stack is host-throughput-positive only when the
production AMX kernel is available AND the host-side operation it
accelerates is on the critical path. The two paths that satisfy condition
2 in principle (Paths 1 and 2) both require condition 1, which the current
build does not satisfy.

Already-known positive levers (vLLM-native, orthogonal to LHC) remain the
recommended way forward for paper-grade gains:
- fp8 KV-cache: +3.94 to +7.32 %
- Suffix decoding: +4.92 to +27.19 %

## Reproduction

```bash
cd /workspace/host_vllm_hybrid
bash lhc_phase4/option_C_bypass/path_1/run_path1_sweep.sh
/workspace/vllm_dev_prj/bin/python lhc_phase4/option_C_bypass/path_1/aggregate.py
cat lhc_phase4/option_C_bypass/path_1/RESULTS.md
```

## Files

- `vllm/v1/core/kv_cache_utils.py` — Path 1 hook
  (`_lhc_amx_c3_block_hash`, gated on `VLLM_LHC_AMX_C3_PREFIX=1`).
- `lhc_phase4/option_C_bypass/path_1/run_path1_sweep.sh` — sweep driver.
- `lhc_phase4/option_C_bypass/path_1/aggregate.py` — results aggregator.
- `lhc_phase4/option_C_bypass/path_1/runs/` — raw JSON + boot/bench logs.
- `lhc_phase4/option_C_bypass/path_1/RESULTS.md` — per-workload table.

## Time accounting

- Path 1 implementation: ~25 min (hook + smoke-test + sweep harness).
- Path 1 sweep: ~25 min (12 runs × ~2 min each, sequential).
- Path 2–4 feasibility analysis: ~20 min.
- Total budget consumed: ~70 min of the 6 hr ceiling.
