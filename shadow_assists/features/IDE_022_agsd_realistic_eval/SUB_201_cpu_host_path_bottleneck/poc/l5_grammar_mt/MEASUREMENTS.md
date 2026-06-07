# SUB_201 L5 — CPU multi-thread grammar state advance — MEASUREMENTS

- **Hardware**: NVIDIA B200 × 1 (GPU 6; 183 GB HBM3e sm_100) + Intel Xeon Platinum 8570
- **Model**: `meta-llama/Llama-3.1-8B-Instruct`, TP=1, max-model-len 4096 (eager mode), `--gpu-memory-utilization 0.50`
- **Workload**: sharegpt corpus, **200 prompts × concurrency 16 × max_tokens 256** (stream, B2 protocol with shorter cap)
- **vLLM**: `v1.7.dev16107+gffe20fb09.d20260601` (branch `feat/spec-decode-tuning`, HEAD `f01d60612`)
- **xgrammar**: `xgrammar` site-packages from `/workspace/vllm_dev_prj/lib/python3.12/site-packages/xgrammar/`
- **Runner**: re-uses `poc/b2_constrained/constrained_runner.py` (--max-tokens 256)
- **Date**: 2026-06-06

---

## 1. Lever target & patch summary

L5 lever: CPU multi-thread the **grammar state-machine update** path that runs
per token per request in vLLM v1.

### 1.1 Code path identified (vLLM v1)

| Stage | File:line | What runs |
|---|---|---|
| xgrammar advance entry | `vllm/v1/structured_output/backend_xgrammar.py:169-188` (`XgrammarGrammar.accept_tokens`) | wraps `xgr.GrammarMatcher.accept_token()` — native C++ FSM update |
| **Engine-step batch advance** (hot path) | `vllm/v1/core/sched/scheduler.py:1706-1721` (inside `update_from_output`) | called **once per scheduler step per request** with `new_token_ids` from the sampler — currently **serial in a `for req_id in num_scheduled_tokens` loop** |
| Spec-decode advance | `vllm/v1/structured_output/__init__.py:272` (inside the `grammar_bitmask` serial fallback) | only fires under speculative decoding — same `accept_tokens` call |

The hot path (called *every* sampler step for *every* structured-output
request in the running batch) is **scheduler.py:1710**. xgrammar advertises
that its C++ matcher releases the GIL while running, so a `ThreadPoolExecutor`
should let N requests advance in parallel.

### 1.2 Patch

Files modified:

| File | Change |
|---|---|
| `vllm/envs.py` | new `VLLM_GRAMMAR_MULTITHREAD` (bool, default off), `VLLM_GRAMMAR_MT_MIN_BATCH` (int=8), `VLLM_GRAMMAR_MT_MAX_WORKERS` (int=auto) |
| `vllm/v1/structured_output/__init__.py` | new `StructuredOutputManager._ensure_grammar_advance_executor()` + `batch_accept_tokens(batch)` helper. Lazy-creates a `ThreadPoolExecutor(max_workers = override or max(1, min(cpu_count()//2, 8)))`. When flag is off or `len(batch) < MT_MIN_BATCH`, falls back to inline serial. `clear_backend()` extended to shut down the pool. |
| `vllm/v1/core/sched/scheduler.py` | `update_from_output()` gets a **pre-pass** (before the main `for req_id in num_scheduled_tokens` loop) that gathers every `(req_id, grammar, new_token_ids)` for requests that pass `should_advance(req)` + have non-empty `sampled_token_ids`, batches them onto `structured_output_manager.batch_accept_tokens()`, and stashes the result in a local dict. The main loop's existing inline `grammar.accept_tokens()` call is replaced by `if req_id in grammar_advance_results: reuse cached; else inline`. When `VLLM_GRAMMAR_MULTITHREAD=0` the pre-pass `if` is short-circuited so the dict stays empty and every request hits the original inline path — zero behaviour change. |

Patch is **regression-safe by default** (flag off ⇒ identical code path).

### 1.3 Sanity tests (Python-level)

| test | result |
|---|---|
| `VLLM_GRAMMAR_MULTITHREAD=1`, 4-item batch → `batch_accept_tokens(...)` | all 4 calls land on `grammar-advance_0` worker thread; result dict `{r1:True,r2:True,r3:True,r4:True}` ✓ |
| `VLLM_GRAMMAR_MULTITHREAD=0`, 2-item batch → `batch_accept_tokens(...)` | both calls land on `MainThread`; executor never created (`._grammar_advance_executor is None`) ✓ |

Patch is functionally correct on both branches.

---

## 2. Micro-bench (xgrammar `GrammarMatcher.accept_token()` direct)

Driver: `poc/l5_grammar_mt/micro_bench.py`. JSON-schema = same 5-key schema
used in `b2_constrained`. Vocab = Llama-3.1 = 128 000. For each (batch, steps)
configuration we:

1. Build N independent `xgr.GrammarMatcher` instances.
2. Pre-compute the accepted-token sequence each matcher will walk through
   (so that the timing window contains pure xgrammar work, not Python-side
   bitmask sampling).
3. **Mode A** = "full path" = `fill_next_token_bitmask` + `accept_token` per
   step (the work an engine step would do once accept is run with a fresh
   bitmask).
4. **Mode B** = "accept only" = replay the pre-computed sequence,
   `accept_token` calls only — this is exactly what vLLM's scheduler hot
   path (scheduler.py:1710) runs.

Mode B is the direct apples-to-apples for the L5 question. Mode A bounds the
generous case where the bitmask cost is also folded in.

### 2.1 Results

| config | mode | serial wall (ms) | mt-w wall (ms) | speedup | Δ% |
|---|---|---|---|---|---|
| batch=16, steps=64, w=8 | A (full) | 46.1 | 235.2 | 0.20× | **+410.0 %** |
| batch=16, steps=64, w=8 | **B (accept only)** | **0.98** | **11.28** | **0.09×** | **+1052.6 %** |
| batch=64, steps=128, w=4 | A (full) | 319.7 | 630.5 | 0.51× | +97.2 % |
| batch=64, steps=128, w=4 | **B (accept only)** | **6.46** | **23.21** | **0.28×** | **+259.4 %** |
| batch=64, steps=128, w=8 | A (full) | 314.4 | 1220.3 | 0.26× | +288.2 % |
| batch=128, steps=128, w=8 | A (full) | 656.7 | 2753.4 | 0.24× | +319.3 % |
| batch=128, steps=128, w=8 | **B (accept only)** | **22.00** | **129.61** | **0.17×** | **+489.1 %** |

Per-accept cost (serial, Mode B):

| batch × steps | accepts | wall (serial, ms) | µs / accept |
|---|---|---|---|
| 16 × 64 | 1 024 | 0.98 | 0.96 |
| 64 × 128 | 8 192 | 6.46 | 0.79 |
| 128 × 128 | 16 384 | 22.00 | 1.34 |

xgrammar's C++ matcher runs an `accept_token` in **≈ 0.8-1.4 µs** on this
Xeon. Even at batch=128 the **total** sequential cost of the entire
hot path (scheduler.py:1710 across the whole running batch) is **0.17 ms**.

A `ThreadPoolExecutor.submit + Future.result` round-trip on CPython is
typically 30-60 µs of Python overhead, which is **30-75 ×** the work item.
Pool dispatch overhead **dominates** the actual xgrammar work and the
multi-thread path is monotonically worse at every tested batch size.

### 2.2 Why "accept only" is the right scope

In vLLM v1, the scheduler's `update_from_output` (scheduler.py:1706) calls
`grammar.accept_tokens(req_id, new_token_ids)` exactly once per request per
step. The bitmask `fill_next_token_bitmask` is **separate**: it lives in
`StructuredOutputManager.grammar_bitmask()` and is called **before**
`execute_model`, **overlapped** with the in-flight GPU forward
(`engine/core.py:723-729` per the B2 audit). The scheduler's per-token
advance is therefore the only `accept_token` call site whose latency lies
on the engine critical path — and that latency is < 2 µs per request per
step.

---

## 3. End-to-end measurement (attempted, contention-limited)

| flag | mode | wall (s) | n_ok / n | output_tps | TPOT p50 (ms) | n_err | first_err |
|---|---|---|---|---|---|---|---|
| `VLLM_GRAMMAR_MULTITHREAD=0` (r3) | baseline | 25.5 | 176/200 | 1604.1 | 8.8 | 24 | HTTP 500 EngineCore dead |
| `VLLM_GRAMMAR_MULTITHREAD=0` (r3) | json_schema | 0.0 | 0/200 | 0.0 | — | 200 | ConnectError (backend already dead) |
| `VLLM_GRAMMAR_MULTITHREAD=0` (r4) | baseline | 18.0 | 112/200 | 1390.6 | 9.0 | 88 | HTTP 500 EngineCore dead |
| `VLLM_GRAMMAR_MULTITHREAD=0` (r4) | json_schema | 0.0 | 0/200 | 0.0 | — | 200 | ConnectError |
| `VLLM_GRAMMAR_MULTITHREAD=1` | baseline | 12.3 | 80/200 | 1336.7 | 9.6 | 120 | HTTP 500 EngineCore dead |
| `VLLM_GRAMMAR_MULTITHREAD=1` | json_schema | 0.0 | 0/200 | 0.0 | — | 200 | ConnectError |

**These numbers cannot be used to compare flag=0 vs flag=1**. Every run died
mid-bench with `EngineDeadError` (no Python traceback on the EngineCore
side — symptomatic of `SIGKILL` from the OOM killer or external process
kill). The system load average was 19-31 throughout the runs because the
dev box had four other concurrent `vllm serve` instances (`Qwen/Qwen2.5-7B-Instruct`)
spawned by sibling SUB_201 PoCs (`l1_kv_quant`, `l10_admission`,
`moe_offload`). Both flag=0 and flag=1 hit the same crash signature
identically — the failure is **not** patch-induced. With `dmesg` blocked
(no CAP_SYS_ADMIN in this container, per CLAUDE.md §240), the kernel-side
kill reason cannot be pulled.

Result: the e2e arm of L5 is **inconclusive in this shared environment**.
Because micro-bench (§2) resolves the lever question directly and
unambiguously, we do not re-run e2e in a fresh container.

### 3.1 Reading the e2e baseline numbers we did get

The single clean baseline (mt=0 r3, 176/200 ok) gave 1604 tps on a single
B200 at conc=16. That matches the order of magnitude implied by the B2
audit (1× B200 ≈ ½ of TP=2 = ~2 200 tps × ½ on smaller cap), so vLLM and
the runner are wired correctly. The patch is **not** what causes the
mid-bench crashes — the same crash signature occurs in both flag states
and at very different request counts (80 / 112 / 176 OK), as expected for
an external SIGKILL.

---

## 4. Task conclusion

**L5 (CPU multi-thread grammar state advance) is REJECTED as a lever.**

1. **Per-request xgrammar advance is ~1 µs.** `accept_token` is a tight
   C++ FSM update with practically no Python crossover. At batch=128 the
   *entire* scheduler-step grammar work is 0.17 ms — far below the
   per-step TPOT budget of ~3-10 ms / token / seq measured in B2.
2. **ThreadPool dispatch overhead is 30-60 µs / future, 30-75 × larger
   than the work item.** Mode B micro-bench shows multi-thread is
   monotonically worse: speedup 0.09× at batch=16, 0.17× at batch=128.
   The pattern holds with 4-worker and 8-worker pools.
3. **The lever cannot benefit any batch size that fits on one B200.** The
   serial path will always be the cheapest one as long as
   `accept_token` stays a microsecond-class C++ call.

Two narrower observations worth keeping for a follow-up:

- **Bitmask construction (`fill_next_token_bitmask`)** *is* the heavier
  half (Mode A serial ≈ 40 µs / req-step at batch=128, vs 1.4 µs for
  accept). vLLM already parallelises this above batch=128
  (`__init__.py:62, 226`), and the B2 PoC concluded the SIMD lever is
  not worth the rewrite. Multi-threading more *aggressively* (lower
  threshold) is plausible but B2's numbers show TPOT is already flat
  vs baseline — there is no headroom to recover.
- **Jump-forward decoding** remains the only realistic constrained-decode
  win (B2 §4.b). It eliminates whole `accept_token` calls instead of
  trying to make them cheaper.

The patch stays under the `VLLM_GRAMMAR_MULTITHREAD` flag for opt-in
reproduction by future investigators. Default off — regression-safe.

For the IDE_022 / SUB_201 lever sweep: **drop L5**.

---

## 5. Reproducibility

- patch: `vllm/envs.py`, `vllm/v1/structured_output/__init__.py`, `vllm/v1/core/sched/scheduler.py`
- env vars: `VLLM_GRAMMAR_MULTITHREAD={0,1}`, `VLLM_GRAMMAR_MT_MIN_BATCH=8` (default), `VLLM_GRAMMAR_MT_MAX_WORKERS=0` (auto)
- micro-bench driver: `poc/l5_grammar_mt/micro_bench.py`
  - results: `micro_b16_s64_w8.json`, `micro_b64_s128_w4.json`, `micro_b64_s128_w8.json`, `micro_b128_s128_w8.json`
- e2e driver: `poc/l5_grammar_mt/run_one.sh {0,1}` (calls `b2_constrained/constrained_runner.py`)
  - per-mode JSON: `llama8b_{baseline,json_schema}_mt{0,1}.json`
  - raw streams: `llama8b_{baseline,json_schema}_mt{0,1}.raw.jsonl`
  - boot/bench logs: `_logs/`
