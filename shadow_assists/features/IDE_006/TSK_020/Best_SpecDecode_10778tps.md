# ★★★ Best Configuration — SUB_085 v2 (suffix + cudagraph PIECEWISE) — 2026-05-25 KST 갱신

> **★ 갱신 (2026-05-25 KST 09:58)**: SUB_085 v2 (suffix_spec32 + cudagraph_mode=PIECEWISE + gmu=0.80) 영역 **3 workload 모두 fair net positive 영역 새 best 영역 SUB_047 ngram 대신**:
>
> | workload | SUB_086 vanilla (gmu=0.80, fair baseline) | SUB_085 v2 suffix PIECEWISE (gmu=0.80) | **fair contribution** |
> |---|---:|---:|---:|
> | **sonnet** | 7,709.8 | **11,589.5** | **+50.3%** ⭐ |
> | **chat** | 2,186.9 | **3,582.4** | **+63.8%** ⭐ |
> | **code** | 6,717.8 | **7,990.0** | **+18.9%** ⭐ (ngram −23.2% 회귀 영역 완전 mitigation) |
>
> 상세: [`measurements/sub085_suffix_piecewise_20260525/RESULTS.md`](measurements/sub085_suffix_piecewise_20260525/RESULTS.md), [`measurements/sub086_vanilla_gmu080_20260525/RESULTS.md`](measurements/sub086_vanilla_gmu080_20260525/RESULTS.md)

---

## ★ historical (SUB_047 ngram cap=8) — 본 doc 영역 baseline 영역 historical (gmu=0.85 + wrapper-historical) 영역 caveat 영역 보존

> **이전 best (2026-05-23 KST)**: SUB_047 t3 (cap=8 + div_tp=0) canonical 3-run avg **10,956.5 tps (+134.12% vs vanilla 4,679.8)**
> variance 0.454% (max-min = 49.7 tps).
> 이전 (SUB_044 t3 spec=7 only) = 10,778.6 tps 는 vLLM 의 ngram numba cap=1 (TODO 미적용) 의 한계 — env `VLLM_NGRAM_NUM_THREADS_CAP=8` + `VLLM_NGRAM_DIVIDE_BY_TP=0` 로 해소.
>
> ⚠ **+134.12% 영역 historical baseline (4,679.8 영역 wrapper-historical) 영역 noise 포함**. SUB_086 영역 wrapper-consistent vanilla (gmu=0.80) = 7,709.8 영역 fair baseline 영역 SUB_047 (10,956.5 영역 gmu=0.85) vs SUB_086 (7,710 영역 gmu=0.80) 영역 직접 비교 영역 fair 아님 (gmu 차이) — 단 wrapper-consistent fair = vanilla 7,710 → ngram cap=8 10,957 = **+42%** / suffix PIECEWISE 11,590 = **+50%**.

> **★ Contribution breakdown** (SUB_073 / I001 정정, 2026-05-24): "+134.12% vs vanilla" 의 정확한 분해 —
>
> | 단계 | config | source | tps | vs 직전 | vs vanilla 누적 |
> |---|---|---|---:|---:|---:|
> | (1) vanilla | `speculative_config=None` | vLLM upstream (spec OFF) | 4,679.8 | — | — |
> | (2) **vLLM built-in spec ON (default cap=1)** | `num_spec=7, prompt_lookup=2/5` | **vLLM 영역 코드 변경 0** — feature 활성화만 | **10,778.6** (SUB_044 t3) | **+130.3%** | **+130.3%** |
> | (3) **SUB_047 fork patch** | `+ cap=8, div_tp=0` | **본 fork ~6 줄 patch** (env-tunable threading enable) | 10,956.5 (3-run avg) | **+1.65%** | **+134.12%** |
>
> → +134% 중 **130 pp 는 vLLM built-in 효과**, **본 fork patch 의 추가 기여는 1.65 pp**.

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

## 5. vs vanilla baseline (contribution breakdown)

| Approach | tps | wall (s) | vs vanilla | source / 추가 기여 |
|---|---:|---:|---:|---|
| (1) vanilla baseline | 4,680 | 875 | — | vLLM upstream (spec OFF) |
| (2) vLLM built-in spec ON (default cap=1, SUB_044) | 10,778 | 373 | +130.3% (2.30×) | **vLLM 영역 코드 변경 0** — `speculative_config={"method":"ngram",...}` 활성화만 |
| (3) **★ + SUB_047 fork patch (cap=8 + div_tp=0)** | **10,957** (3-run avg) | **367** | **★ +134.1% (2.341×)** ⭐ | **본 fork ~6 줄 patch** — vLLM upstream PR #24986 의 disabled threading enable. **추가 기여 = +1.65%** (over default) |

→ **본 작업 contribution = +1.65%** (fork patch) + workload generalization 분석 (SUB_071) + R/K framework + 외부 사례 통합. 나머지 130 pp 는 vLLM built-in ngram spec decoding 활성화 효과.

## 6. CLAUDE.md `# Objective` 평가

| 목표 | 평가 |
|---|---|
| "GPU 포함 서버 **전체 throughput 향상**" | **✓ 충족 — +134.1% throughput** |
| "CPU 활용률 **극도로** 끌어올리기" | ❌ — CPU 5.51% (vanilla 4.66% 거의 동일) |
| "CPU **Idle 허락 안 함**" | ❌ — CPU idle 94.5% |

→ **서버 throughput 목표 ✓ 충족**, CPU 활용 목표 미달 (spec=GPU lever). CPU 활용 추가 lever (Tier 1/3) 별도 시도 중.

## 7. 다음 path (CPU 활용 추가 + post-SUB_047 lever 시도 결과)

### 7.1 SUB_050~064 (Objective lever 탐색, 2026-05-23) — 단독 best dual-axis = SUB_054 b=64

| Tier | Lever | SUB | 상태 / 비고 |
|---|---|---|---|
| Tier 1 A | CPU draft model (small LLM, 1B 정도 CPU 추론) | (plan) | vLLM 내부 코드 변경 필요 (draft device=cpu) — 미시작 |
| Tier 1 B | ngram numba thread cap env-tunable | **SUB_047 (★ 완료)** | **+134.1% 달성 (단 CPU idle)** |
| Tier 3 E | CPU LLM + GPU spec 동시 (별도 instance) | SUB_049 (완료) | 3-scenario: solo / +Qwen0.5B / +Qwen1.5B — CPU 활용 ↑ 26~28% |
| Tier 3 F | multi-stream (GPU spec + CPU BG workload) | SUB_045 (완료) | 3-scenario: spec solo / spec+BG / vanilla+BG — CPU 29% 달성 |
| Cat B (SUB_050~064) | CPU embedding preprocessor (BGE-large) | SUB_054 (완료 2026-05-23) | **batch=64 production config: main 10,848 (-1.0%) / CPU 21.21% / embedder 36.7 sps** |
| Cat B | CPU re-ranker (BGE-reranker) | SUB_055 (완료 2026-05-23) | main 10,556 (-3.7%) / CPU 21.23% / 44 pps |
| Cat D | NUMA + KMP/GOMP affinity | SUB_060 (회귀) | main 10,268 (-6.3%) — KMP_AFFINITY 가 vLLM 과 conflict 추정, 폐기 |
| Cat D | isolcpus + cgroup isolated partition | SUB_061 (infeasible) | container 안에서 host cgroup partition root 필요 — 본 env 에서 불가 |
| (결합 시도) | SUB_054 + SUB_055 + SUB_049 동시 (NUMA1 56 core 분할) | Phase 1 combo (2026-05-23) | main 9,635 (**-12.1%**) / CPU 23.85% — contention 으로 단독 합 미달 |
| (2-way combo A) | Qwen 1.5B + BGE emb b=64 (28+28 thread NUMA1) | Phase 3 combo A (2026-05-23) | main 10,268 (-6.28%) / CPU 23.68% — 여전히 단독 SUB_054 (-1.0%) 보다 회귀 |
| (2-way combo B) | BGE emb b=64 + BGE rerank (28+28 thread NUMA1) | Phase 3 combo B (2026-05-23) | main 9,598 (**-12.40%**) / CPU 24.01% — combo A 보다 더 회귀, 단독 SUB 사용 권장 |

### 7.2 Bottleneck-driven SUB_065~069 (2026-05-24) — 5/5 모두 기각, plateau 신호

5 lever 가설을 ngram lookup 내부 + step pipeline 내부에서 탐색했으나 모두 noise 또는 회귀:

| SUB | bottleneck 가설 | 결과 (vs baseline ~10,980) | 판정 |
|---|---|---:|---|
| SUB_065 | B-4 small-batch threshold (`num_tokens_threshold` 5-way: 8192/4096/2048/1024/0) | -0.07 ~ -1.69% | **기각** |
| SUB_066 | B-2 ngram broadcast from rank 0 (`broadcast_object` cpu_group) | **-1.30%** (CPU 5.58→4.37 duplicate 절감, broadcast overhead 가 더 큼) | **기각** |
| SUB_067 | C1 speculative ngram precompute (background thread + per-request suffix cache + chain[0][0] 가정) | **-3.77%** (최대 회귀 — 16MB token_ids copy + low hit rate + numba single-thread overhead) | **기각** |
| SUB_068 | D2/D4 stop-string + tokenizer parallel (`RAYON_NUM_THREADS=8` + `TOKENIZERS_PARALLELISM=true`) | +0.03% noise | **기각** |
| SUB_069 | F1 prompt sorting by length (3-way + 3-run interleaved 재측정) | 1-run asc +1.10% (baseline noise) → 3-run -0.23% | **기각** |

**자체 비판**: 5 가설 모두 ngram lookup 자체 안에서만 lever 를 찾음. 그러나 측정값을 보면 ngram time ~1-2 ms / step time 70-90 ms = **1-2% only** — search space 자체가 잘못. best case 도 +1-2% 였고, 실제로는 step overhead 가 더 커서 회귀.

### 7.3 SUB_071 (2026-05-24) — chat/code large-scale workload generalization 검증

SUB_044/047 와 동일 scale (500p × 8192in × 8192max) 에서 chat / code workload 측정. Rec 1 medium (200p × 4096) 의 결론이 large 에서도 유지되는지 검증.

| workload | scale | vanilla | spec7+cap8 | speedup |
|---|---|---:|---:|---:|
| sonnet | 500p × 8192 | 4,679.8 | **10,956.5** | **+134.1%** ⭐ |
| **chat** | **500p × 8192** | **2,186.0** | **3,006.6** | **+37.5%** |
| **code** | **500p × 8192** | **6,964.5** | **5,346.8** | **−23.2% 회귀** |
| (ref) chat | 200p × 4096 | 2,113.6 | 2,577.1 | +22% |
| (ref) code | 200p × 4096 | 7,889.1 | 5,505.6 | −30% |

**핵심 fact**:
- chat 은 medium (+22%) → large (+37.5%) 로 **+15.5pp 개선** — sonnet excerpt 가 prompt 에 포함되어 ngram acceptance 일부 유지.
- code 는 medium (−30%) → large (−23.2%) 로 약간 완화되지만 여전히 **net regression**. code 응답은 EOS 없이 max 까지 생성 (out_tok 3.9M / 500 = ~7,830 tok/prompt) → spec overhead 가 wall 에 누적.

**production 함의**: SUB_047 best 는 **workload-shape 의존**. `workload-aware gating (code 검출 시 spec OFF)` 가 production 적용 시 필수 lever.

raw / 상세: [`measurements/sub071_workload_large_20260524/RESULTS.md`](measurements/sub071_workload_large_20260524/RESULTS.md)

**정량 mechanism 분석**: [`analysis/workload_acceptance_analysis_20260524.md`](analysis/workload_acceptance_analysis_20260524.md) — sonnet/chat/code 의 spec decode 효과를 K (평균 token 진척) + R (spec step overhead) 모델로 분해. 추정 K = sonnet ≈ 3.0~5.0 / chat ≈ 1.7~1.9 / code ≈ 1.0 (zero acceptance). 결정 인자 = **prompt ↔ generated 어휘 overlap**. workload-aware gating heuristic + Leviathan 2022 closed-form 정합 + 40 외부 reference (vLLM PR #24986 = SUB_047 patch 의 직접 origin, vLLM issue #16258/#19254 = code 회귀 외부 corroboration, PLD/EAGLE/REST/SuffixDecoding/Spec-Bench/Cascade/Nightjar 등) 포함.

### 7.4 SUB_070 (2026-05-24) — engine config sweep, 사용자 중단 (1/6 cell)

진짜 idle 영역은 **GPU SM 45.3%** (54.7% util). 5 SUB 실패 후 root lever 를 GPU concurrency ↑ (engine config) 로 재정의:

| cell | gmu | seqs | bt | tps | 상태 |
|---|---:|---:|---:|---:|---|
| baseline | 0.85 | 256 | 8,192 | **10,983.8** | ✓ SUB_047 재현 |
| A1 gmu+ | 0.90 | 256 | 8,192 | — | **fail** (KV init gloo 1,800,000ms timeout) |
| A2 gmu++ | 0.92 | 256 | 8,192 | — | 중단 (사용자 지시) |
| B1 seqs+ | 0.85 | 384 | 8,192 | — | 중단 |
| B2 seqs++ | 0.85 | 512 | 8,192 | — | 중단 |
| C1 bt+ | 0.85 | 256 | 16,384 | — | 중단 |

A1 의 KV init timeout 으로 환경 안정성 이슈 확인 (gmu 0.85 → 0.90 도 본 env 에서는 too aggressive). 후속 진행 사용자 지시 대기.

### 7.5 종합 결론 (2026-05-24 갱신)

| 영역 | 결론 |
|---|---|
| ngram-spec 내부 lever | **SUB_047 (10,956.5 tps, +134.1%) = plateau 확정** (SUB_065~069 모두 기각) |
| dual-axis (throughput + CPU) | **SUB_054 b=64 (10,848 tps / CPU 21.21% / -1.0%) = 현 best dual-axis production config** |
| engine config (GPU concurrency ↑) | SUB_070 — gmu 0.90 도 환경에서 timeout, 사용자 결정 대기 |
| **workload generalization (large)** | **SUB_071 — sonnet +134.1% / chat +37.5% / code −23.2% (회귀). workload-aware gating 필수성 fact 확정** |
| 추가 방향 | workload-aware gating PoC (code 검출 → spec OFF), Eagle CPU draft (model-matched ckpt 대기), 별도 host process 등 ngram-spec 외부 lever |

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
