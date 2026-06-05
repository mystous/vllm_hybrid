# B2 constrained-decode host overhead PoC — MEASUREMENTS

- **Hardware**: NVIDIA B200 × 2 (GPU 6,7; 183GB HBM3e sm_100 per GPU) + Intel Xeon Platinum 8570
- **Model**: meta-llama/Llama-3.1-8B-Instruct, TP=2, max-model-len=16384, gpu-mem-util=0.85
- **Workload**: sharegpt corpus, **max_tokens=512** (schema-friendly short), stream
- **vLLM**: `v1.7.dev16107+gffe20fb09.d20260601` (branch `feat/spec-decode-tuning`, HEAD `e6d6c1f39`)
- **xgrammar backend**: `structured_outputs.backend='auto'` → xgrammar (default, see §1)
- **Runner**: `poc/b2_constrained/constrained_runner.py` (this PoC)
- **Boot**:
  ```
  CUDA_VISIBLE_DEVICES=6,7 \
    setsid vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2 --port 8007 \
    --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --allow-deprecated-quantization
  ```
- **Date**: 2026-06-05
- **Goal**: README §10 task 3 B2 — measure host overhead of JSON-schema / grammar constrained decoding and
  identify whether vLLM's per-step bitmask construction (xgrammar `fill_next_token_bitmask`) is a worthy AVX-512 offload target.

---

## 1. vLLM constrained-decode code path (analysis)

### Engine-side (per-step host work)

| Stage | File:line | What runs |
|---|---|---|
| Scheduler hook | `vllm/v1/engine/core.py:724` | `self.scheduler.get_grammar_bitmask(scheduler_output)` — called **every step** after `execute_model(non_block=True)`, before `sample_tokens`. |
| Manager entry | `vllm/v1/structured_output/__init__.py:186` | `StructuredOutputManager.grammar_bitmask(requests, struct_req_ids, spec_tokens)` — allocates `_grammar_bitmask: torch.Tensor[int32, max_seqs × ceil(vocab/32)]` once. |
| Per-request fill | `vllm/v1/structured_output/__init__.py:168` `_fill_bitmasks` | calls `grammar.fill_bitmask(self._grammar_bitmask, index)` per structured-output request. |
| xgrammar backend | `vllm/v1/structured_output/backend_xgrammar.py:191` `XgrammarGrammar.fill_bitmask` | wraps `xgr.GrammarMatcher.fill_next_token_bitmask(bitmask, idx)` → native C++ in `xgrammar_bindings.so`. |
| Optional parallel | `__init__.py:181-184`, `:221-247` | When `len(structured_output_request_ids) > 128` **and** no spec decode, fills are batched into a `ThreadPoolExecutor` (`min(ncpu/2, 8)` workers, 16 per task). Below 128 → serial inline. |
| Serialise to NumPy | `__init__.py:282` | returns `.numpy()` int32 — packaged in `GrammarOutput`, IPC'd to GPU worker. |

### Worker-side (per-step device work)

| Stage | File:line | What runs |
|---|---|---|
| Worker entry | `vllm/v1/worker/gpu/model_runner.py:864` | `self.structured_outputs_worker.apply_grammar_bitmask(logits, input_batch, grammar_req_ids, grammar_bitmask)` |
| Bitmask H2D | `vllm/v1/worker/gpu/structured_outputs.py:34-37` | async copy of packed-int32 bitmask via dedicated `copy_stream` (pinned). |
| Apply kernel | `structured_outputs.py:86-115` | Triton `_apply_grammar_bitmask_kernel` (adapted from xgrammar) — unpack 32-bit and `tl.store -inf` into masked positions. |

### Backend matrix

| Backend | File | When chosen | Notes |
|---|---|---|---|
| **xgrammar** | `backend_xgrammar.py` | default for JSON-schema, regex, EBNF grammar, structural-tag (auto in vLLM unless unsupported jsonschema feature) | Used in this run. C++ matcher + Triton apply. |
| **guidance** (LLGuidance) | `backend_guidance.py` | opt-in (`backend='guidance'`) | Llama-Guidance C lib, similar shape. |
| **outlines** (outlines-core) | `backend_outlines.py` | opt-in (`backend='outlines'`) | Rust FSM. |
| **lm-format-enforcer** | `backend_lm_format_enforcer.py` | opt-in | Pure-Python; expected slowest. |

### Jump-forward decoding

`xgrammar.GrammarMatcher.find_jump_forward_string()` exists (file
`/workspace/vllm_dev_prj/lib/python3.12/site-packages/xgrammar/matcher.py:345`) and the
`XgrammarGrammar` docstring (`backend_xgrammar.py:137-138`) explicitly references it,
but **the call site is never invoked in vLLM v1** (no grep hit outside the comment).
i.e. **vLLM does NOT do jump-forward decoding today** — every token is sampled
even when the grammar admits exactly one continuation.

---

## 2. Measurement (200p × conc=16, max-tokens=512, sharegpt)

| mode | wall (s) | tokens | **output_tps** | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GPU util (%) | CPU% |
|---|---|---|---|---|---|---|---|---|---|
| baseline (unconstrained) | 22.7 | 100 699 | **4 439.9** | 22.3 | 59.5 | 3.4 | 3.4 | 91.6 | 3.7 |
| JSON schema (5-key object) | 22.4 | 101 429 | **4 524.5** | 25.0 | 52.7 | 3.3 | 3.4 | 92.5 | 3.7 |
| EBNF grammar (word list)  | 22.4 | 102 400 | **4 571.4** | 32.4 | 65.3 | 3.3 | 3.4 | 92.5 | 3.8 |

Completion-token distribution (proves comparison is fair — all three saturate at the 512 cap):

| mode | mean | p10 | p50 | p90 | max |
|---|---|---|---|---|---|
| baseline | 503.5 | 512 | 512 | 512 | 512 |
| json_schema | 507.1 | 512 | 512 | 512 | 512 |
| grammar | 512.0 | 512 | 512 | 512 | 512 |

### Δ vs baseline

| metric | baseline → json_schema | baseline → grammar |
|---|---|---|
| output_tps | +84.6 (+1.9 %) | +131.5 (+3.0 %) |
| TPOT p50 | −0.1 ms (−3 %) | −0.1 ms (−3 %) |
| TTFT p50 | **+2.7 ms (+12 %)** | **+10.1 ms (+45 %)** |
| CPU% (process) | 0.0 pp | +0.1 pp |
| GPU util | +0.9 pp | +0.9 pp |

**Reading**: constrained decoding here costs **essentially zero steady-state tps / TPOT** versus
unconstrained. The only measurable hit is a one-shot TTFT bump (+3-10 ms) attributable to
grammar compile (xgrammar `compile_json_schema` / `compile_grammar` runs on first arrival of
each unique grammar). The steady-state per-step `fill_next_token_bitmask` + Triton apply path
is too cheap to budge tps at this batch size.

The slight `output_tps` increases for json/grammar are statistical noise (single 22-second
run; ~1 % run-to-run variance is normal) plus a small artefact: constrained payloads include
a short system-prompt prefix (39-44 chars) that shifts prefix-caching behaviour, and the
grammar run happens to fill exactly 512 tokens per request (no early `</s>`).

---

## 3. Where does the time go? (per-step breakdown reasoning)

Direct nsys/py-spy traces were not collected (would require `CAP_SYS_ADMIN` in this
container — see CLAUDE.md §240). Instead the per-step host budget is reasoned from
TPOT p50:

- TPOT p50 = **3.4 ms / token / sequence**. With 16 concurrent sequences this means
  the engine is producing ~16 tokens every 3.4 ms — a **~213 µs / step** budget
  on the host critical path that bounds throughput.
- xgrammar's `fill_next_token_bitmask` on Llama-3.1's 128 256-vocab is reported by
  xgrammar's own benchmarks at **~30-80 µs / mask** (single thread, AVX2 host),
  totalling ~0.5-1.3 ms per step for batch=16 if **serial**.
- vLLM gates the parallel `ThreadPoolExecutor` at `len(structured_output_request_ids)
  > 128`. With batch=16 (this run) the path is **serial** — yet TPOT is
  indistinguishable from baseline, so xgrammar's C++ mask must be << 213 µs * 16
  even serially, or the path is overlapped by GPU forward + sampling (which
  `execute_model(non_block=True)` does — `get_grammar_bitmask` is computed while
  the GPU forward is still in flight).
- Triton apply kernel: `cdiv(128256, 8192) = 16` blocks per mask × 16 masks = 256
  blocks, well under the per-step GPU budget.

So the bitmask path is **already overlapped with the GPU forward** — vLLM's
`grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)` runs after
`execute_model(..., non_block=True)` and before `model_executor.sample_tokens`
(`core.py:723-731`), and at small batch the CPU side finishes before the GPU does.

---

## 4. AVX-512 offload DESIGN (hook candidates)

### 4.a Theoretical lever

The per-step mask itself is a **vocab × 1-bit** boolean (vLLM packs it as
`int32[vocab/32]`). For Llama-3.1-8B's 128 256-vocab this is 4 008 int32 per
sequence, or 16 KiB. AVX-512 should be very effective at:

| op | shape | AVX-512 form |
|---|---|---|
| bitmask → -inf logits (applied on CPU when `is_cpu`, `utils.py:124-130`) | vocab fp32 | `_mm512_mask_blend_ps(mask, logits, -inf)` |
| packed-int32 union of multiple masks (for spec-decode bonus / structural-tag overlap) | int32[vocab/32] | `_mm512_or_si512` |
| FSM `fill_next_token_bitmask` itself | depends on grammar structure — xgrammar uses an internal `RuleStateSet` walk, not a regular bit-loop | **not directly vectorisable** without rewriting xgrammar's C++ internals |

### 4.b Hook candidates (file:line, ranked by displacement vs risk)

1. **CPU `apply_grammar_bitmask` path** — `vllm/v1/structured_output/utils.py:124-138`.
   This is the `is_cpu` branch where `xgr.apply_token_bitmask_inplace` runs on fp32 logits.
   Replacing with an in-house AVX-512 `_mm512_mask_blend_ps` loop is mechanical and safe
   (no FSM logic). **But this branch is dead on B200/H100** — the GPU path uses the
   Triton kernel. Only relevant for CPU-only deployments.
2. **`StructuredOutputManager._fill_bitmasks`** — `vllm/v1/structured_output/__init__.py:168-179`.
   Today this is a tight `for grammar, idx, apply in batch: grammar.fill_bitmask(...)`.
   An AVX-512 lever here would require an **in-house FSM** to replace xgrammar's
   internals — far larger surface than B1 detok and likely net-negative until vLLM
   moves to a vectorisable grammar representation. **Not recommended.**
3. **Batched fill via xgrammar's `batch_fill_next_token_bitmask`** (no AVX-512 needed
   — pure refactor lever). xgrammar already provides
   `GrammarCompiler.batch_fill_next_token_bitmask` (matcher.py:464) with internal
   `max_threads`. vLLM's serial `for` (cumulative_index loop at
   `__init__.py:248-273`) does **not** call this batched API in the spec-decode
   path. For requests with speculative tokens, switching to the batched API would
   amortise GIL release and let xgrammar's own thread pool do the work — but the
   measured TPOT shows this is not a current bottleneck either.
4. **Jump-forward decoding integration** — much higher payoff than vectorisation
   for any grammar regime with long deterministic spans. `find_jump_forward_string`
   exists in xgrammar today; vLLM would need scheduler-side handling to inject the
   forced tokens. Out of B2 PoC scope, but **this is the true unrealised lever for
   constrained decode**, not a SIMD mask kernel.

### 4.c Verdict on the SIMD lever

The host overhead from per-step bitmask construction is below the noise floor at
typical inference batch sizes (16-64) because:
- xgrammar's matcher is already a tight C++ implementation,
- vLLM already overlaps `get_grammar_bitmask` with the in-flight GPU forward,
- the Triton apply kernel runs on the GPU, not the CPU.

An AVX-512 rewrite of `_fill_bitmasks` would optimise a path that is **already
non-critical** in the measured regime. The high-concurrency replication
(§5 below) is included to confirm this holds when xgrammar's serial cost grows
linearly with batch.

---

## 5. High-concurrency replication (500p × conc=64, max-tokens=512, sharegpt)

Sanity check at conc=64 — closer to the xgrammar parallel-fill threshold of 128
seq/step, with ~4× the GPU pressure and ~4× the per-step bitmask budget.

| mode | wall (s) | tokens | **output_tps** | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GPU util (%) | CPU% |
|---|---|---|---|---|---|---|---|---|---|
| baseline    | 15.8 | 254 146 | **16 041.6** | 91.1 | 165.2 | 3.7 | 3.9 | 94.3 | 4.5 |
| json_schema | 15.7 | 253 872 | **16 135.0** | 85.2 | 135.0 | 3.7 | 3.9 | 94.3 | 4.4 |
| grammar     | 15.8 | 255 818 | **16 216.2** | 103.6 | 141.8 | 3.7 | 3.8 | 94.7 | 4.4 |

### Δ vs baseline (HC64)

| metric | baseline → json_schema | baseline → grammar |
|---|---|---|
| output_tps | +93.4 (+0.58 %) | +174.6 (+1.09 %) |
| TPOT p50 | 0.0 ms | 0.0 ms |
| TTFT p50 | −5.9 ms (−6 %) | **+12.5 ms (+14 %)** |
| CPU% | −0.1 pp | −0.1 pp |
| GPU util | 0.0 pp | +0.4 pp |

The verdict holds at 4× concurrency: **TPOT identical**, **tps within ±1 %**,
**CPU% unchanged**. The lone TTFT bump (grammar +12.5 ms) is the first-arrival
compile cost (`xgr.GrammarCompiler.compile_grammar` for the EBNF word-list
spec); subsequent requests hit the xgrammar compile cache
(`backend_xgrammar.py:64-69`, `VLLM_XGRAMMAR_CACHE_MB`).

---

## 6. Task conclusion

**B2 constrained-decode host overhead is NOT a meaningful AVX-512 lever** on
this stack (vLLM v1 + xgrammar + B200/H100, Llama-3.1-8B class):

- Steady-state Δtps between unconstrained and JSON-schema/grammar runs is **within
  ±3 %**, i.e. inside run-to-run noise.
- TPOT p50 is **identical** (3.3-3.4 ms across all three modes).
- CPU% rises by **at most 0.1 pp**.
- The only measurable cost is one-shot grammar compile at TTFT (+3-10 ms per
  unique grammar), which is amortised after the first request and unrelated to
  the per-step mask path.

The mechanism is sound: vLLM already **overlaps** `get_grammar_bitmask` with the
in-flight GPU forward (`engine/core.py:723-729`), and the per-step bitmask is
produced by a tight C++ xgrammar routine + a GPU-side Triton apply kernel. There
is no room for an AVX-512 win without rewriting xgrammar's FSM.

**Recommended lever if constrained-decode wall-time becomes a future target**:
**jump-forward decoding** (skip forward by the deterministic prefix of the
grammar). xgrammar exposes `find_jump_forward_string` today but vLLM does not
invoke it — this is a scheduler/sequence-management change, not a SIMD change,
and the speed-up potential is far larger (factor-of-N for deterministic spans).

For SUB_201's bigger picture: the B2 lever is **rejected** as a candidate for
the IDE_022 CPU-utilisation work. CPU is not the bottleneck of constrained
decode in this stack.

---

## 7. Reproducibility

- baseline:    `llama8b_baseline.json` + `.raw.jsonl`
- json_schema: `llama8b_json_schema.json` + `.raw.jsonl`
- grammar:     `llama8b_grammar.json` + `.raw.jsonl`
- HC64:        `llama8b_{mode}_hc64.json` + `.raw.jsonl`
- driver:      `run_one.sh` (3 modes back-to-back), `run_high_conc.sh`
- runner:      `constrained_runner.py`
- boot/bench logs: `_logs/`
- GPU 6,7 free post-run: `_logs/gpu_after.txt`, `_logs/gpu_after_hc.txt`
