# ★★★ Best Configuration — Ngram Spec + ngram thread cap=8 (10,949.8 tps) (2026-05-23 KST, updated)

> **갱신 (2026-05-23 turn)**: SUB_047 t3 (cap=8 + div_tp=0) = **10,949.8 tps (+134% vs vanilla, +1.6% vs SUB_044 spec=7)** ⭐.
> 이전 (SUB_044 t3 spec=7 only) = 10,778.6 tps 는 vLLM 의 ngram numba cap=1 (TODO 영역 미적용) 의 한계 — env `VLLM_NGRAM_NUM_THREADS_CAP=8` + `VLLM_NGRAM_DIVIDE_BY_TP=0` 로 해소.

# ★★★ 이전 Best — Ngram Speculative Decoding (10,778 tps) (2026-05-23 KST)

> **출처**: SUB_044 v2 t3_spec7 ([RESULTS](measurements/sub044_spec_decode_20260523/RESULTS.md))
> **본 환경 (HEAD `0d7dc0334`, H100×8 + SPR dual, Llama-70B FP8 KV, 500p × 8192) 의 throughput WINNER**.
> **vs vanilla baseline 4,680 tps: +130.3% (2.30× faster)**

---

## 1. 측정 fact

| 항목 | 값 |
|---|---|
| **output_tps** | **10,778.6** |
| wall (500p × 8192 = 4.1M token) | 372.9 s |
| crash | 0 ✓ |
| CPU busy avg | 2.46% |
| GPU util avg | 21.9% |
| GPU power avg | (측정 안 함, 추정 vanilla 와 비슷) |

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
    # NOTE: enable_neo_asymmetric NOT set (False) — 본 lever 는 vanilla path
)
```

### 2.2 env (CLI/shell)

```bash
export HF_HUB_OFFLINE=1
# 모든 VLLM_NEO_* env unset (vanilla path)
# SUB_047 ★ env-tunable ngram numba thread cap (TODO 처리)
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

## 4. 본 lever 의 본질

| 측면 | 설명 |
|---|---|
| **출처** | vLLM **built-in feature** (`speculative_config={"method":"ngram",...}`) |
| **본 NEO 코드 변경** | **없음** — `csrc/`, `vllm/v1/core/sched/neo_scheduler*`, `pacpu*` 모두 1줄도 안 만짐 |
| **mechanism** | prompt 내부 n-gram lookup → 다음 multiple tokens 예측 → 한 GPU forward 에 multiple tokens (1+7=8) 동시 verify (accept 시 효과적으로 step 당 multiple tokens 출력) |
| **draft model** | 불필요 — pure prompt-based n-gram |
| **target workload 적합** | sonnet (Shakespeare) 어휘 반복 → acceptance rate ↑ → 본 워크로드 최적 |

## 5. vs 본 NEO 시도

| Approach | tps | wall (s) | vs vanilla |
|---|---:|---:|---:|
| vanilla baseline | 4,680 | 875 | — |
| NEO env-ON 500p (SUB_036/040) | 1,779 | 2,293 | **-62%** ⚠️ |
| **★ vanilla + ngram spec=7** | **10,778** | **373** | **★ +130%** ⭐ |
| spec / NEO ratio | — | — | **spec 가 NEO 의 6.06×** |

## 6. CLAUDE.md `# Objective` 평가

| 목표 | 평가 |
|---|---|
| "GPU 포함 서버 **전체 throughput 향상**" | **✓ 충족 — +130% throughput** |
| "CPU 활용률 **극도로** 끌어올리기" | ❌ — CPU 2.46% (vanilla 4.66% 보다 더 idle) |
| "CPU **Idle 허락 안 함**" | ❌ — CPU idle 97.5% |

→ **서버 throughput 목표 ✓ 충족**, CPU 활용 목표 미달 (spec=GPU lever). CPU 활용 추가 lever 영역 (Tier 1/3) 별도 시도 중.

## 7. 다음 path (CPU 활용 추가)

| Tier | Lever | SUB |
|---|---|---|
| Tier 1 A | CPU draft model (small LLM CPU) | SUB_045 (진행 중) |
| Tier 1 B | ngram lookup CPU thread 분리 | SUB_046 (plan) |
| Tier 1 C | spec sampling CPU offload | SUB_047 (plan) |
| Tier 3 E | CPU draft + GPU verify + CPU sample | SUB_048 (plan) |
| Tier 3 F | multi-stream (GPU spec + CPU BG) | SUB_049 (진행 중) |

## 8. raw 자료

| 항목 | 위치 |
|---|---|
| result.json | `eval/results/20260523_005314_sub044_spec_decode/t3_spec7/result.json` |
| RESULTS.md | [`measurements/sub044_spec_decode_20260523/RESULTS.md`](measurements/sub044_spec_decode_20260523/RESULTS.md) |
| launcher | `/tmp/run_sub044_spec_decode.sh` |
| wrapper | `/tmp/run_spec_decode.py` |
| stdout log | `/tmp/sub044_spec_v2.log` |
