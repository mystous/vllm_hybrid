# ★★★ Best Configuration — Ngram Spec + ngram thread cap=8 (3-run avg 10,956.5 tps) (2026-05-23 KST)

> **갱신 (2026-05-23)**: SUB_047 t3 (cap=8 + div_tp=0) canonical 3-run avg **10,956.5 tps (+134.12% vs vanilla 4,679.8)** ⭐
> variance 0.454% (max-min = 49.7 tps).
> 이전 (SUB_044 t3 spec=7 only) = 10,778.6 tps 는 vLLM 의 ngram numba cap=1 (TODO 미적용) 의 한계 — env `VLLM_NGRAM_NUM_THREADS_CAP=8` + `VLLM_NGRAM_DIVIDE_BY_TP=0` 로 해소.

---

## 1. 측정 fact (3-run, 500p × 8192, 2026-05-23)

| run | output_tps | wall (s) | CPU% | GPU% | vs vanilla 4,679.8 |
|---|---:|---:|---:|---:|---:|
| 1 | 10,981.4 | 366.0 | 5.51 | 54.6 | +134.65% |
| 2 | 10,931.7 | 367.7 | 5.57 | 54.7 | +133.59% |
| 3 | 10,956.3 | 366.8 | 5.59 | 54.8 | +134.12% |
| **avg** | **10,956.5** | **366.83** | **5.557** | **54.70** | **+134.12%** |
| min | 10,931.7 | 366.0 | 5.51 | 54.6 | +133.59% |
| max | 10,981.4 | 367.7 | 5.59 | 54.8 | +134.65% |
| range / avg (variance) | **0.454%** | 0.46% | 1.43% | 0.37% | 0.06pp |

본 configuration 의 throughput = **10,956.5 ± 25 tps**.

**raw 위치**:
- run 1: `eval/results/20260523_100441_sub048_ngram_refinement/t1_baseline/result.json`
- run 2: `eval/results/20260523_162456_sub047_t3_verify/run2_cap8_div0/result.json`
- run 3: `eval/results/20260523_162456_sub047_t3_verify/run3_cap8_div0/result.json`

## 2. 설정 (production-ready)

### 2.1 vLLM LLM constructor

```python
LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=8,
    max_model_len=16384,
    max_num_seqs=256,
    gpu_memory_utilization=0.85,
    enforce_eager=False,           # CUDA graphs ON
    kv_cache_dtype="fp8",
    max_num_batched_tokens=8192,
    disable_log_stats=True,
    seed=0,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 7,    # ★ sweet spot (10 은 OOM crash)
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    },
)
```

### 2.2 env (CLI/shell)

```bash
export HF_HUB_OFFLINE=1
# SUB_047 ★ env-tunable ngram numba thread cap (vLLM 의 TODO 처리)
export VLLM_NGRAM_NUM_THREADS_CAP=8     # vLLM 기본 cap=1 → 8
export VLLM_NGRAM_DIVIDE_BY_TP=0         # tp_size 영역 나누지 않음 → 8 thread/rank
```

### 2.4 ★ SUB_047 code 변경 (vllm/v1/spec_decode/ngram_proposer.py:48)

```python
# BEFORE (vLLM 기본):
self.num_numba_thread_available = min(1, (cpu_count // 2))  # 사실상 1 thread
self.num_numba_thread_available //= tp_size  # 1//8 = 0 → fallback 1

# AFTER (SUB_047):
cap = int(os.environ.get("VLLM_NGRAM_NUM_THREADS_CAP", "1"))
divide_by_tp = int(os.environ.get("VLLM_NGRAM_DIVIDE_BY_TP", "1"))
self.num_numba_thread_available = max(1, min(cap, (cpu_count // 2)))
if divide_by_tp:
    self.num_numba_thread_available //= tp_size
self.num_numba_thread_available = max(1, self.num_numba_thread_available)
```

### 2.3 sampling

```python
SamplingParams(temperature=0.0, max_tokens=8192)
```

## 3. sweep history (SUB_044 v2 의 monotonic increase)

| num_spec | tps | wall (s) | vs vanilla | 비고 |
|---:|---:|---:|---:|---|
| 3 | 9,573.6 | 421.2 | +104.6% (2.05×) | conservative |
| 5 | 10,166.6 | 396.8 | +117.3% (2.17×) | vLLM ngram default |
| **★ 7** | **10,778.6** | **372.9** | **★ +130.3% (2.30×)** | ⭐ sweet spot |
| 10 | NO_RESULT (crash=3) | — | OOM | max 초과 |

→ num_spec=7 이 throughput 영역 최적, num_spec=10 은 KV memory 또는 spec verification slot 부족.

## 4. 동작 원리

### 4.1 Best (10,956.5 tps) 동작 pipeline

```
prompt
  ↓
[ngram lookup, CPU]  ← VLLM_NGRAM_NUM_THREADS_CAP=8 + DIVIDE_BY_TP=0
                        → 8 thread/rank 가 prompt 안에서 n-gram 매칭
                        → 다음 7 token 후보 추출 (sub-ms)
  ↓
[1 + 7 = 8 token batch]  → GPU H100×8 (TP=8) forward 1 회 → 8 position logit 동시 계산
  ↓
[rejection sampler]  → 매칭되는 prefix 까지 accept (평균 ~4-5 token/step)
  ↓
output (step 당 평균 ~5 token, vanilla 의 ~5× step 압축)
```

### 4.2 왜 빠른가 (mechanism)

- vanilla: 1 step = 1 forward = 1 token
- spec=7: 1 step = 1 forward = `min(7, accept_count) + 1` token (평균 4-5)
- workload (sonnet, Shakespeare) 의 어휘 반복 → ngram acceptance rate ~60%
- GPU 가 batch size 작은 forward 도 어차피 latency-bound → multiple-token 동시 verify 가 거의 공짜 (FLOPs 만 늘고 wall 거의 그대로)

### 4.3 왜 cap=8 이 필요한가 (SUB_047 의 본질)

vLLM upstream 의 `ngram_proposer.py` (SUB_047 패치 전):

```python
# 사실상 1 thread/rank 로 고정 — TODO 주석 그대로 둔 채 출시됨
self.num_numba_thread_available = min(1, (cpu_count // 2))  # min(1, 56) = 1
self.num_numba_thread_available //= tp_size                  # 1 // 8 = 0 → fallback 1
# TODO(ekagra-ranjan): bump up the cap from 1 to 8 when TP parallelization for ngram is implemented.
```

→ ngram lookup 가 1 thread 로 직렬 처리 → CPU stall 이 GPU forward 사이에 끼어 pipeline stall
→ SUB_047 patch 로 env-tunable 화 (cap=8, tp_size 로 나누지 않음) → 8 thread/rank 가 병렬 검색 → stall 제거 → +1.6% (10,778 → 10,956)

코드 변경 ≒ 6 줄 (`vllm/v1/spec_decode/ngram_proposer.py:36-58`).

### 4.4 본 lever 의 출처와 한계

| 측면 | 설명 |
|---|---|
| 출처 | vLLM **built-in feature** (`speculative_config={"method":"ngram",...}`) + SUB_047 한 줄짜리 thread cap 패치 |
| draft model | 불필요 — pure prompt-based |
| target workload 적합 | sonnet 의 어휘 반복 → acceptance ↑. **일반 chat / code 에서는 acceptance ↓ 가능** → 별도 검증 필요 |
| KV memory 제약 | `num_speculative_tokens=10` 은 OOM → 7 이 sweet spot |

## 5. vs vanilla baseline

| Approach | tps | wall (s) | vs vanilla |
|---|---:|---:|---:|
| vanilla baseline | 4,680 | 875 | — |
| vanilla + ngram spec=7 (SUB_044) | 10,778 | 373 | +130.3% (2.30×) |
| **★ vanilla + ngram spec=7 + cap=8 (SUB_047)** | **10,957** (3-run avg) | **367** | **★ +134.1% (2.341×)** ⭐ |

## 6. CLAUDE.md `# Objective` 평가

| 목표 | 평가 |
|---|---|
| "GPU 포함 서버 **전체 throughput 향상**" | **✓ 충족 — +134.1% throughput** |
| "CPU 활용률 **극도로** 끌어올리기" | ❌ — CPU 5.51% (vanilla 4.66% 거의 동일) |
| "CPU **Idle 허락 안 함**" | ❌ — CPU idle 94.5% |

→ **서버 throughput 목표 ✓ 충족**, CPU 활용 목표 미달 (spec=GPU lever). CPU 활용 추가 lever (Tier 1/3) 별도 시도 중.

## 7. 다음 path (CPU 활용 추가)

| Tier | Lever | SUB | 상태 / 비고 |
|---|---|---|---|
| Tier 1 A | CPU draft model (small LLM, 1B 정도 CPU 추론) | (plan) | vLLM 내부 코드 변경 필요 (draft device=cpu) — 미시작 |
| Tier 1 B | ngram numba thread cap env-tunable | **SUB_047 (★ 완료)** | **+134.1% 달성 (단 CPU idle)** |
| Tier 3 E | CPU LLM + GPU spec 동시 (별도 instance) | SUB_049 (완료) | 3-scenario: solo / +Qwen0.5B / +Qwen1.5B — CPU 활용 ↑ 26~28% |
| Tier 3 F | multi-stream (GPU spec + CPU BG workload) | SUB_045 (완료) | 3-scenario: spec solo / spec+BG / vanilla+BG — CPU 29% 달성 |
| Cat B (SUB_050~064) | CPU embedding preprocessor (BGE-large) | SUB_054 (완료 2026-05-23) | **batch=64 production config: main 10,848 (-1.0%) / CPU 21.21% / embedder 36.7 sps** |
| Cat B | CPU re-ranker (BGE-reranker) | SUB_055 (완료 2026-05-23) | main 10,556 (-3.7%) / CPU 21.23% / 44 pps |
| Cat D | NUMA + KMP/GOMP affinity | SUB_060 (회귀) | main 10,268 (-6.3%) — KMP_AFFINITY 영역 vLLM conflict 추정, 폐기 |
| Cat D | isolcpus + cgroup isolated partition | SUB_061 (infeasible) | container 영역 host cgroup partition root 필요 — 본 env 영역 불가 |
| (결합 시도) | SUB_054 + SUB_055 + SUB_049 동시 (NUMA1 56 core 분할) | Phase 1 combo (2026-05-23) | main 9,635 (**-12.1%**) / CPU 23.85% — contention 으로 단독 합 영역 영역, 영역 영역 영역 |
| (2-way combo A) | Qwen 1.5B + BGE emb b=64 (28+28 thread NUMA1) | Phase 3 combo A (2026-05-23) | main 10,268 (-6.28%) / CPU 23.68% — 영역 영역 단독 SUB_054 영역 영역 (-1.0%) 영역 |
| (2-way combo B) | BGE emb b=64 + BGE rerank (28+28 thread NUMA1) | Phase 3 combo B (2026-05-23) | main 9,598 (**-12.40%**) / CPU 24.01% — combo A 영역 영역 회귀, 영역 SUB 영역 단독 영역 사용 권장 |

## 8. raw 자료

| 항목 | 위치 |
|---|---|
| **run 1 result.json** | `eval/results/20260523_100441_sub048_ngram_refinement/t1_baseline/result.json` |
| **run 2 result.json** | `eval/results/20260523_162456_sub047_t3_verify/run2_cap8_div0/result.json` |
| **run 3 result.json** | `eval/results/20260523_162456_sub047_t3_verify/run3_cap8_div0/result.json` |
| RESULTS.md (SUB_044 base) | [`measurements/sub044_spec_decode_20260523/RESULTS.md`](measurements/sub044_spec_decode_20260523/RESULTS.md) |
| RESULTS.md (SUB_047 3-run) | [`measurements/sub047_t3_3run_verify_20260523/RESULTS.md`](measurements/sub047_t3_3run_verify_20260523/RESULTS.md) |
| launcher (5-way sweep) | `/tmp/run_sub047_ngram_threads.sh` |
| launcher (verify 2-run) | `/tmp/run_sub047_t3_verify_2runs.sh` |
| wrapper | `/tmp/run_spec_decode.py` |
| stdout log (verify new batch) | `/tmp/sub047_t3_verify_round2.log` |
