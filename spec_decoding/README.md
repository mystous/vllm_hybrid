# vllm_hybrid — Speculative Decoding (Trident core / AGSD framework)

> **last update**: 2026-05-25 KST (SUB_093 full matrix + util 반영)
> **branch**: `feat/spec-decode-tuning`
> **base**: vLLM 1.6.dev0+g858b6df7a (fork)
> **fork core change**: **14 줄** (`vllm/utils/__init__.py` +5, `vllm/engine/arg_utils.py` +9, backward-compat 100%)
> **상세 doc**: [`COMPREHENSIVE_REPORT_20260525.md`](../shadow_assists/features/IDE_006/TSK_020/COMPREHENSIVE_REPORT_20260525.md) (416 lines) + [`OUTSTANDING_CONTRIBUTIONS_20260525.md`](../shadow_assists/features/IDE_006/TSK_020/OUTSTANDING_CONTRIBUTIONS_20260525.md) (243 lines) + [`SUB_093 RESULTS`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub093_full_matrix_util_20260525/RESULTS.md) (57 cell + util)

---

## 0. 용어 정리 (Trident core vs AGSD)

| 용어 | 의미 | 활성화 |
|---|---|---|
| **Trident core** | **spec config 자체** = SuffixDecoding + cudagraph PIECEWISE + gmu=0.80, **always-on** (모든 request 영역 suffix 적용) | §1 코드 영역 그대로 사용 |
| **AGSD** (Auto Gating Spec Decoding) | **framework** = Trident core + workload/model-size **gating** (per-request method 선택) | classifier (SUB_076) + router (SUB_092) + per-request override (vLLM 영역 미지원 영역 후속 SUB) |

→ Llama 70B 단독 영역 모든 workload 영역 suffix 가 best 이므로 **AGSD = Trident core 결과 동일** (gating decision = 항상 suffix).
→ AGSD 영역 별도 가치 영역 **mixed-model deployment** (예: Llama 70B + Qwen 동시) + 일부 cell (Qwen 7B chat 영역 vanilla 선택) 영역 발현.

---

## Executive Summary (Trident core, SUB_093 측정)

| workload | vanilla | **Trident core** | **fair contribution** | CPU% | GPU% |
|---|---:|---:|---:|---:|---:|
| **sonnet** | 7,678.7 | **11,676.9** | **+52.1%** ⭐ | 5.3 (vs 5.6) | 73.3 (vs 93.8) |
| **chat** | 2,266.8 | **3,830.4** | **+68.9%** ⭐ | (config-wide) | (config-wide) |
| **code** | 6,717.7 | **7,981.4** | **+18.8%** ⭐ (ngram −20.2% 회귀 mitigation) | — | — |
| **mix-sh** (M1 60:20:20) | 6,325.9 | **10,297.7** | **+62.8%** ⭐ | — | — |
| **mix-bal** (M2 34:33:33) | 6,053.9 | **9,514.3** | **+57.2%** ⭐ | — | — |
| **mix-ch** (M3 10:20:70) | 6,494.9 | **9,457.3** | **+45.6%** ⭐ | — | — |

→ **6 workload 모두 net positive (+18.8% ~ +68.9%)**, mixed traffic 까지 포함. **wall 31% 단축** (config-wide).
→ GPU util 영역 73.3% (vanilla 93.8%) — spec decoding 영역 GPU 영역 idle 늘리며 throughput 영역 상승.

---

## 1. 활성화 (production-ready, copy-paste)

### 1.1 vLLM LLM constructor

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=8,
    max_model_len=16384,
    max_num_seqs=256,
    gpu_memory_utilization=0.80,                            # ★ cudagraph PIECEWISE + spec memory headroom
    enforce_eager=False,                                    # CUDA graph ON
    kv_cache_dtype="fp8",
    max_num_batched_tokens=8192,
    disable_log_stats=True,
    seed=0,
    compilation_config={"cudagraph_mode": "PIECEWISE"},     # ★ Trident 핵심 — FULL graph capture skip
    speculative_config={
        "method": "suffix",                                 # ★ SuffixDecoding (arctic_inference.SuffixDecodingCache lazy import)
        "num_speculative_tokens": 32,
    },
)

sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8192, seed=0)
outputs = llm.generate(prompts, sp)
```

### 1.2 필수 env

```bash
# arctic_inference plugin: vLLM 1.6과 binary incompat → disable, lazy SuffixDecodingCache 만 사용
export ARCTIC_INFERENCE_ENABLED=0
export VLLM_PLUGINS=""

# cudagraph PIECEWISE + spec 의 memory fragmentation 줄임
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 1.3 pip install (필수)

```bash
.venv/bin/pip install arctic-inference   # SuffixDecodingCache class 만 사용 (plugin은 disable)
```

### 1.4 small model (≤7B) 권장

→ **vanilla 만 (spec OFF)** — 모든 spec method universal regression (SUB_078/079/088/090/091 의 5 model cross-validated)

---

## 2. all-fair benchmark — vanilla / ngram / Trident core / AGSD (모두 gmu=0.80 + cudagraph PIECEWISE + same wrapper)

### 2.1 Large model 6-workload matrix (Llama-3.3-70B + TP=8 + H100×8 + 500p × 8192in × 8192max, SUB_093)

| workload | **vanilla** | **ngram cap=8** | **Trident core** (suffix+PIECEWISE) | **AGSD** (Trident + gating) | Trident vs vanilla |
|---|---:|---:|---:|---:|---:|
| sonnet | 7,678.7 | 10,758.8 (+40.1%) | **11,676.9** | **11,676.9** (→suffix) | **+52.1%** ⭐ |
| chat | 2,266.8 | 3,243.5 (+43.1%) | **3,830.4** | **3,830.4** (→suffix) | **+68.9%** ⭐ |
| code | 6,717.7 | 5,361.5 (−20.2%) ✗ | **7,981.4** | **7,981.4** (→suffix) | **+18.8%** ⭐ |
| **mix-sh** (M1 60:20:20) | 6,325.9 | 7,932.6 (+25.4%) | **10,297.7** | **10,297.7** | **+62.8%** ⭐ |
| **mix-bal** (M2 34:33:33) | 6,053.9 | 6,553.6 (+8.3%) | **9,514.3** | **9,514.3** | **+57.2%** ⭐ |
| **mix-ch** (M3 10:20:70) | 6,494.9 | 5,490.7 (−15.5%) ✗ | **9,457.3** | **9,457.3** | **+45.6%** ⭐ |

→ Llama 70B 영역 모든 6 workload 영역 **AGSD = Trident core** (gating decision 영역 항상 suffix).
→ ngram 영역 code-heavy mix-ch 까지 회귀 (−15.5%). Trident core 영역 mix-ch 까지 net positive (+45.6%).

### 2.1.1 (참조) prior SUB measurements

| 측정 | sonnet | chat | code | 비고 |
|---|---:|---:|---:|---|
| SUB_085 v2 (single run) | 11,589.5 | 3,582.4 | 7,990.0 | first Trident core measurement |
| SUB_089 (canonical 3-run avg) | **11,687.4** | — | — | variance 0.20% |
| **SUB_093 (latest)** | **11,676.9** | **3,830.4** | **7,981.4** | + mix 3 종 + util |

### 2.2 K (mean_accept_len) / α (per-position acceptance) — suffix vs ngram

| workload | ngram K | ngram α | suffix K | suffix α | K 비율 | α 비율 |
|---|---:|---:|---:|---:|---:|---:|
| sonnet | 1.66 | 9.5% | 5.11 | 77.0% | 3.08× | 8.1× |
| chat | 5.98 | 71.2% | 10.06 | 92.7% | 1.68× | 1.30× |
| **code** | **1.09** | **1.2%** | **4.08** | **40.1%** | **★ 3.74×** | **★ 33×** |

### 2.3 canonical 3-run variance (SUB_089, sonnet × suffix PIECEWISE)

| run | tps | wall (s) |
|---:|---:|---:|
| 1 | 11,695.3 | 345.5 |
| 2 | 11,694.7 | 345.5 |
| 3 | 11,672.1 | 346.2 |
| **avg** | **11,687.4** | **345.7** |
| variance | **0.20%** | 0.20% |

→ **canonical sonnet best = 11,687.4 tps (var 0.20%, fair +51.6%)**.

### 2.3a util matrix (SUB_093 config-wide avg, 6 workload 평균)

| config | wall sum (s) | CPU util (%) | GPU util (%) | 비고 |
|---|---:|---:|---:|---|
| vanilla | 2,750.5 | 5.6% | 93.8% | spec OFF — GPU 영역 fully bound |
| ngram | 2,635.8 | 7.6% | 84.2% | ngram drafter CPU 부담 (+2.0pp) / GPU −9.6pp (drafter wait) |
| **Trident core** | **1,892.4** | **5.3%** | **73.3%** | suffix 영역 ngram 보다 CPU 가벼움 / GPU −20.5pp |
| AGSD (Llama 70B) | =Trident core | =Trident core | =Trident core | gating decision 영역 항상 suffix → 동일 |

→ **Trident core wall 31% 단축 + GPU −20pp**. spec decoding 영역 GPU 활용률 영역 떨어지나 wall throughput 영역 늘림 (per-step K token output).

### 2.4 Small / medium model (Qwen 0.5B/1.5B/7B + TP=1 + 50p × 1024in × 512max, code workload, 이전 SUB_090 — default cudagraph)

| model | vanilla | ngram (PIECEWISE) | suffix (PIECEWISE) | best |
|---|---:|---:|---:|---|
| Qwen 0.5B | 11,220 | 7,794 (**−30.5%**) | 5,376 (−52.1%) | **vanilla** |
| Qwen 1.5B | 10,389 | 5,855 (**−43.6%**) | 4,064 (−60.9%) | **vanilla** |
| **Qwen 7B** | **5,556** | **4,594 (−17.3%)** ⭐ boundary 근접 | 3,516 (−36.7%) | **vanilla** |
| (참조) Llama 70B | 7,710 | 10,139 (+31.5%) | **11,590 (+50.3%)** ⭐ | **suffix PIECEWISE** |

→ **R/K boundary는 7B ↔ 70B 사이**. ≤ 7B model 영역 **AGSD gating decision = vanilla** / ≥ 70B model 영역 **AGSD gating decision = Trident core (suffix PIECEWISE)**.
→ (caveat) SUB_093 Phase 2 영역 PIECEWISE-only 재측정 영역 short-wall noise (1-8s wall) — prior SUB_090 영역 default cudagraph 영역 비교 영역 reliable.

### 2.5 issue #16258 cross-validation (5-model × hardware)

| model | hardware | spec regression | source |
|---|---|---:|---|
| opt-125m | 2× L4 | 2.12× | vLLM Issue #16258 (외부) |
| **opt-125m** | **H100×1** | **2.13× (정확 일치)** ⭐⭐ | 본 fork SUB_091 |
| starcoder2-3b | H100×1 | 2.30× | 본 fork SUB_091 |
| Qwen 0.5B | H100×1 | 2.46× | 본 fork SUB_078 |
| Qwen 1.5B | H100×1 | 2.63× | 본 fork SUB_078 |
| Qwen 0.5B + suffix | H100×1 | 2.06× (suffix 도 회귀) | 본 fork SUB_088 |

→ **small model + spec decoding = hardware-independent fundamental regression** (R≫K), 5-model cross-validated.

---

## 3. Trident core 기술 stack — 14 component

```
┌─────────────────────────────────────────────────────┐
│ ★ Trident core (SUB_085 v2 / SUB_089 / SUB_093)     │
│   spec config 자체 — always-on (no gating)          │
├─────────────────────────────────────────────────────┤
│ ★ SuffixDecoding (arctic, NeurIPS 2025)             │  ← drafter
│ ★ cudagraph_mode=PIECEWISE (vLLM v1 built-in)       │  ← capture mode
│ ★ vLLM v1 SpecDecode framework + RejectionSampler   │  ← scheduler/sampler
│ ★ CUDA Graph (PyTorch, replay-only piecewise)       │  ← forward replay
├─────────────────────────────────────────────────────┤
│ TP=8 / fp8 KV / chunked prefill / gmu=0.80          │  ← memory + parallelism
│ PYTORCH_CUDA_ALLOC_CONF=expandable_segments         │  ← fragmentation
├─────────────────────────────────────────────────────┤
│ 본 fork patch +14 줄 (utils + arg_utils stubs)       │  ← binary compat shim
│ arctic_inference plugin disable + lazy cache import  │  ← incompat 우회
└─────────────────────────────────────────────────────┘
```

### 3.1 핵심 lever (직접 성능 영향)

| # | 기술 | source | 본 fork 활용 |
|---|---|---|---|
| 1 ⭐ | **SuffixDecoding** | arctic_inference, [arXiv 2411.04975](https://arxiv.org/abs/2411.04975) (NeurIPS 2025 Spotlight) | prompt + 이전 generation 양쪽 suffix tree pattern match, frequency-weighted candidate, **adaptive num_spec (1~32 dynamic)** |
| 2 ⭐ | **cudagraph_mode=PIECEWISE** | vLLM v1 built-in `CUDAGraphMode` enum | FULL graph capture skip, per-op piecewise capture — suffix adaptive shape 호환 (FULL은 dynamic shape 차단) |
| 3 | **vLLM v1 SpecDecode framework** | vLLM `vllm/v1/spec_decode/suffix_decoding.py` (origin PR #12193) | SuffixDecodingProposer + rejection sampler (lossless) |
| 4 | **CUDA Graph (PyTorch)** | PyTorch built-in | decode-step forward replay → latency reduce |

### 3.2 지원 기술 (memory + parallelism)

| # | 기술 | 효과 |
|---|---|---|
| 5 | **Tensor Parallelism TP=8** | Llama-70B 8 GPU split (weights ~17.5 GB/GPU) |
| 6 | **fp8 KV cache** | KV memory 절반 → batch ↑ (max_num_seqs=256) |
| 7 | **chunked prefill** (vLLM v1 default) | prefill을 chunked 처리, batch step time stable |
| 8 | **PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True** | memory fragmentation 줄임 |
| 9 | **gpu_memory_utilization=0.80** | cudagraph PIECEWISE + spec memory headroom (gmu=0.85에서 OOM) |
| 10 | **enforce_eager=False** | CUDA graph ON (vLLM default) — PIECEWISE 활성화 의미 |

### 3.3 본 fork 14 줄 patch

| # | file | 라인 | 효과 |
|---|---|---:|---|
| 11 | [`vllm/utils/__init__.py`](../vllm/utils/__init__.py) (SUB_081) | +5 | `FlexibleArgumentParser` re-export (arctic_inference 0.1.2가 기대하는 vLLM 0.11.0 API path 호환) |
| 12 | [`vllm/engine/arg_utils.py`](../vllm/engine/arg_utils.py) (SUB_084) | +9 | `_is_v1_supported_oracle` stub (vLLM 1.6 deprecated method 항상 True 반환) |

→ 본 fork vLLM core 변경 **14 줄, backward-compat 100%**.

### 3.4 arctic_inference 부분 활용 (plugin disable)

| # | 기술 | 활용 방식 |
|---|---|---|
| 13 | **arctic_inference.suffix_decoding.SuffixDecodingCache** | **lazy import만** — plugin disable (vLLM 1.6과 binary incompat: architectural diff) |
| 14 | env: `ARCTIC_INFERENCE_ENABLED=0` + `VLLM_PLUGINS=""` | plugin auto-load 차단, 본 fork SuffixDecodingProposer는 cache class 만 사용 |

---

## 4. mechanism — 왜 Trident core가 모든 workload에서 net positive

### 4.1 ngram vs suffix drafter 차이

| 측면 | ngram (vLLM built-in) | suffix (arctic_inference, lazy import) |
|---|---|---|
| draft source | **prompt 내 n-gram 만** (KMP match) | **prompt + 이전 generations 양쪽** (suffix tree, frequency-weighted) |
| draft length | fixed γ (e.g., 7) | **adaptive (1~num_spec_max, e.g., 32 max)** |
| code workload fit | ✗ (prompt의 word salad ↔ generated Python disjoint, K≈1) | ✓ (self-generated keyword 활용, K≈4) |
| 본 fork의 cuda graph 호환 | ✓ (fixed batch shape) | ✓ (단 PIECEWISE mode 필요 — FULL은 dynamic shape 차단) |

### 4.2 R/K framework (정량 모델)

```
spec_wall / vanilla_wall ≈ R / K
  R = spec step overhead
  K = mean accepted tokens per draft

K > R → net positive (가속)
K < R → net regression
```

| 환경 | R (estimate) | K (suffix code) | 결과 |
|---|---:|---:|---|
| Llama 70B + TP=8 + H100×8 | ~1.3 | **4.08** (code, suffix) | **K > R → net positive (+18.9%)** |
| ngram (code, K=1.09) | ~1.3 | **1.09** | K < R → net regression (−20.7%) |
| Qwen 0.5B + TP=1 | ~5-10 | ~1-4 | K < R → universal regression (−30~−65%) |

→ **suffix의 K가 ngram의 3.7× (code)** → R을 능가 → net positive. small model은 R이 너무 커서 어떤 K도 못 따라감 → vanilla 권장.

---

## 5. AGSD production decision tree (workload + model-size gating)

```
prompt 입력
  ↓
[ AGSD gating: model size + workload 분류 ]
  ↓
  ├─ ≤ 7B (small/medium)
  │   → AGSD decision: vanilla (spec OFF)
  │   이유: 모든 spec method universal regression (-17~-73%, R≫K)
  │   증거: SUB_078/079/088/090/091 (5 model cross-validation, opt-125m 2.13× = issue #16258 정확 일치)
  │
  ├─ 7B ↔ 70B (boundary, Qwen 32B 후속 권장)
  │   → 현재: 7B ngram −17.3% (boundary 근접), 70B suffix +50.3% (net positive)
  │
  └─ ≥ 70B (large)
      → AGSD decision: Trident core (suffix + PIECEWISE + gmu=0.80)
      효과 (SUB_093): sonnet +52.1% / chat +68.9% / code +18.8% / mix-sh +62.8% / mix-bal +57.2% / mix-ch +45.6%
```

→ **AGSD = Trident core + workload/model-size gating**. Llama 70B 단독 영역 모든 workload 영역 Trident core 가 best 이므로 gating = always suffix. AGSD 영역 진짜 가치 영역 mixed-model deployment.

---

## 6. 독창성 (originality) — 본 fork만의 unique contribution

> 본 §6은 5 차별점 (D1~D5)을 외부 literature / OSS 와 정직하게 대조하여 **진짜 unique한 부분과 over-claim 영역을 분리**한 것입니다. 30+ source 외부 web research subagent 가 verify. 상세 reference 는 [`OUTSTANDING_CONTRIBUTIONS_20260525.md`](../shadow_assists/features/IDE_006/TSK_020/OUTSTANDING_CONTRIBUTIONS_20260525.md) (243 lines).

### 6.0 verdict 요약

| # | 차별점 | verdict | 외부 share 가치 |
|---|---|---|---|
| **D2** ⭐ | **vLLM Issue #16258 정확 reproduction (L4 2.12× ↔ H100 2.13× 일치)** + 4 추가 model cross-validation | **✓ unique** | **★ outstanding — vLLM issue closure update + doc caveat 후보** |
| **D3** | R/K boundary Qwen 0.5B/1.5B/7B + Llama 70B 4-size sweep | ◐ partial | Qwen 32B 후속 후 vLLM doc contribute 가능 |
| **D1** | fair contribution accounting (+134% wrapper noise 정정) | ◐ partial | self-correction 중심 — general framework 는 standard |
| D4 | workload × spec method 6-cell matrix (sonnet/chat/code × ngram/suffix) | ✗ not unique | 각 cell 정량값 first measurement, 단 main claim 은 Snowflake SuffixDecoding paper §4 에 published |
| D5 | cudagraph_mode=PIECEWISE 한 줄 우회 (SUB_084 dead-end 정정) | ✗ not unique | vLLM Issue #24943 error message 가 직접 suggest — rediscovery |

→ **진짜 outstanding = D2 (1 개)**, partial = D3 + D1 (2 개), self-correction / first-cell-measurement = D4 + D5.

---

### 6.1 D2 ⭐ — multi-model × multi-hardware × issue #16258 정확 reproduction (✓ unique, ★ outstanding)

**fact**: vLLM Issue #16258 (dtransposed, 2025-04-08, 2× L4) 의 opt-125m + ngram = **2.12× regression** 보고. 본 fork SUB_091 (H100×1) 의 같은 opt-125m + ngram = **2.13× regression** (정확 일치) + 4 추가 model cross-validation.

| model | hardware | spec regression | source |
|---|---|---:|---|
| opt-125m | 2× L4 | **2.12×** | vLLM Issue #16258 (외부) |
| **opt-125m** | **H100×1** | **2.13× (정확 일치)** ⭐⭐ | 본 fork SUB_091 |
| starcoder2-3b | H100×1 | 2.30× | 본 fork SUB_091 |
| Qwen 0.5B | H100×1 | 2.46× | 본 fork SUB_078 |
| Qwen 1.5B | H100×1 | 2.63× | 본 fork SUB_078 |
| Qwen 0.5B + suffix | H100×1 | 2.06× (suffix 도 회귀) | 본 fork SUB_088 |

**외부 verified — 발견 0**:
- [vLLM Issue #16258](https://github.com/vllm-project/vllm/issues/16258) (Closed, 1년 경과) — 다른 user 의 published reproduction 0.
- web search 결과: opt-125m / starcoder2-3b × spec decoding × H100/A100/L40 reproduction **발견 0**.
- [thc1006/qwen3.6-speculative-decoding-rtx3090](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) 은 llama.cpp engine, vLLM 아님.

**진짜 unique한 부분**:
1. **L4 → H100 hardware-cross 정확 일치** (2.12% ↔ 2.13%) — "small model + ngram regression 은 hardware-independent fundamental" 명제의 정량 입증.
2. **5-model 확장 cross-validation** (opt-125m / starcoder2-3b / Qwen 0.5B / Qwen 1.5B / Qwen 0.5B+suffix) — 외부 published 발견 0.
3. **suffix 도 small model 에서 회귀** (Qwen 0.5B+suffix 2.06×) — "spec 종류 무관 universal regression" 까지 확장.

**community contribution 가능 경로**:
- vLLM Issue #16258 의 closure 후 update post (reproduction 첨부)
- vLLM `docs/features/speculative_decoding/n_gram.md` 의 "model-size threshold" caveat
- "small model + spec decoding = universal regression" 명제 정량 confirmation post

---

### 6.2 D3 — R/K model-size boundary 7B↔70B sweep (◐ partial)

**fact**: SUB_090 의 Qwen 0.5B/1.5B/7B × code × {vanilla, ngram cap=8, suffix PIECEWISE} = 9 cell sweep 으로 net positive transition boundary 정량.

| model | vanilla | ngram | suffix |
|---|---:|---:|---:|
| Qwen 0.5B | 11,220 | −30.5% | −52.1% |
| Qwen 1.5B | 10,389 | −43.6% | −60.9% |
| **Qwen 7B** | **5,556** | **−17.3%** (boundary 근접) | −36.7% |
| (참조) Llama 70B | 7,710 | (+31.5%) | **(+50.3%)** ⭐ net positive |

→ **boundary 는 7B↔70B 사이**. ≤ 7B model = vanilla 권장.

**외부 evidence**:
- [Spec-Bench Leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md) — Vicuna-7B/13B/33B 3-size sweep, **code workload 0, 70B 0, 1B 0**.
- [arXiv 2505.07858 "Scaling Laws for Speculative Decoding"](https://arxiv.org/abs/2505.07858) (Liu et al, 2025-05) — Log-linear scaling laws (draft capacity × pretraining tokens × batch), Llama2/3 + Qwen2.5. **단 본 fork 의 정확 setup (Llama 70B + H100×8 + TP=8 + all-fair wrapper + ngram/suffix × Qwen 0.5B/1.5B/7B + Llama 70B) 직접 동등 발견 0**.

**진짜 unique한 부분**: "code workload + model-free spec (ngram/suffix) 의 7B↔70B 사이 net positive transition" 정량 — 본 fork first measurement candidate.

**partial 인 이유**: Scaling Laws paper 가 model-size axis 자체는 이미 cover. 본 fork 는 framework 의 specific instance 측정. Qwen 32B / 72B 추가 측정 (후속 SUB) 으로 boundary refinement 시 published 가능.

---

### 6.3 D1 — fair contribution accounting (◐ partial)

**fact**: 본 fork SUB_047 의 historical "+134.12% vs vanilla 4,679.8" 의 wrapper-historical noise 를 정량 분리.

| 단계 | 값 | contribution |
|---|---:|---:|
| historical vanilla (run_spec_decode.py, gmu=0.85) | 4,679.8 | — |
| fair vanilla (run_workload_gen.py, gmu=0.80) | **7,709.8** | +64.7% (wrapper noise) |
| fair ngram cap=8 (SUB_087) | 10,139.2 | +31.5% (vs fair vanilla) |
| **fair suffix PIECEWISE (SUB_085 v2)** | **11,589.5** | **+50.3% (vs fair vanilla)** |
| 본 fork patch 단독 (SUB_047 cap=8 vs vLLM default cap=1) | — | +1.65% sonnet only |

**외부 evidence**:
- [vLLM Issue #10318](https://github.com/vllm-project/vllm/issues/10318) (Closed "not planned") — vLLM blog 의 2.8× speedup 이 unreproducible. 단 "wrapper noise" 라는 직접 용어 미사용.
- [Red Hat Developer (2026-04-16)](https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss) — baseline 정의 = "default vLLM config + prefix caching". **gmu 동등 명시 없음, wrapper bias discussion 0**.
- [vLLM Discussion #13834](https://github.com/vllm-project/vllm/discussions/13834) — Llama-3.3-70B + 3B draft + A100 환경 spec −30~50% 회귀 보고, 단 baseline framing 분리 못 함.

**진짜 unique한 부분**: vLLM-specific historical claim 의 retroactive 정량 분해 (4,680 → 7,710 = +64.7% wrapper noise) — 외부 published 직접 동등 발견 0.

**partial 인 이유**: "같은 wrapper + 같은 gmu + 같은 cudagraph_mode" 의 fair baseline 은 일반 ML benchmarking standard (Spec-Bench 가 이미 follow). community 의 새 framework 가 아니라 self-correction.

---

### 6.4 D4 — workload × spec method 6-cell matrix (✗ not unique)

**fact**: 6-cell all-fair matrix (Llama 70B + TP=8 + H100×8 + gmu=0.80 + cudagraph PIECEWISE + same wrapper).

| workload | ngram cap=8 vs vanilla | suffix PIECEWISE vs vanilla | suffix vs ngram |
|---|---:|---:|---:|
| sonnet | +31.5% | +50.3% | +14.3% |
| chat | +30.2% | +63.8% | +25.9% |
| code | **−20.7%** (회귀) | +18.9% (mitigation) | **+50.0%** |

**외부 evidence**:
- [arXiv 2411.04975 SuffixDecoding NeurIPS 2025 Spotlight](https://arxiv.org/abs/2411.04975) — workload axis = AgenticSQL 5.3×, SWE-Bench code 1.8-4.5×, ShareGPT chat, MLPerf summ. **"1.4-3.9× faster than vLLM ngram on code-related tasks" 가 main claim** — **code workload 의 suffix > ngram 은 이미 published**.
- [prompt-lookup-decoding (apoorvumang)](https://github.com/apoorvumang/prompt-lookup-decoding) — "input-grounded (summ, doc QA, multi-turn chat, code editing) 2-4× speedup" published.
- [arXiv 2505.08600 Automatic Task Detection ...](https://arxiv.org/abs/2505.08600) + [Nightjar arXiv 2512.22420](https://arxiv.org/pdf/2512.22420) — workload-aware spec method selection / routing 의 prior art.

**verdict**: 각 cell 의 fact 가 모두 published. "code workload 에서 ngram 회귀 → suffix mitigation" 명제 자체가 SuffixDecoding paper §4 에 정확 published.

**본 fork 가치**: 각 cell 의 정확 정량값 (Llama 70B + TP=8 + 본 환경 + same wrapper + 정확 +50.3% / +63.8% / +18.9%) first measurement. mechanism / 결론은 unique 아님.

---

### 6.5 D5 — cudagraph PIECEWISE 한 줄 우회 (✗ not unique)

**fact**: SUB_081 / SUB_084 에서 arctic_inference v0.1.2 (vLLM 0.11.0 binary) + vLLM 1.6 의 binary incompat 를 "fundamental architectural dead-end" 결론. SUB_085 의 `compilation_config={"cudagraph_mode": "PIECEWISE"}` 한 줄 우회로 결론 정정.

**외부 evidence**:
- [vLLM Issue #24943](https://github.com/vllm-project/vllm/issues/24943) (Closed "not planned") — error message 자체가 `please try cudagraph_mode=PIECEWISE` 직접 suggest. **community 가 이미 알고 있던 workaround**.
- [vLLM Issue #33341](https://github.com/vllm-project/vllm/issues/33341) (Open, Stale) — "eagle.py drafter only supports piecewise cudagraphs, full graph forecast +5% TPOT". **spec decoding drafter 의 PIECEWISE only 가 current state**.
- [vLLM CUDA Graphs design doc](https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md) — PIECEWISE = "cudagraph-incompatible ops 를 keep outside, general flexibility" 의 design 명시.

**verdict**: community known workaround 의 rediscovery. 본 fork 의 자체 결론 정정만.

**본 fork 가치**: "arctic_inference v0.1.2 ↔ vLLM 1.6 + SuffixDecodingProposer + PIECEWISE 의 실제 작동 + sonnet +50.3% / chat +63.8% / code +18.9% 정량 측정" 의 specific 측정 contribution 은 D2 / D3 에 일부 포함.

---

### 6.6 외부 share priority (over-claim 회피)

| 우선순위 | action | 근거 |
|---|---|---|
| **1순위** ⭐ | **D2 — vLLM Issue #16258 의 comment update 또는 별도 reproduction post** | opt-125m 정확 일치 (2.12% ↔ 2.13%) + 4 추가 model 정량 |
| 2순위 | **D3 — Qwen 32B 후속 측정 후** vLLM doc 의 model-size threshold guidance | boundary refinement 필요 |
| 회피 | D1 / D4 / D5 의 외부 share 시 **over-claim 회피** — 단순 fact (cell value, self-correction) 만 reference | D4 main claim 은 Snowflake paper 가 이미 published, D5 는 vLLM issue 가 이미 suggest |

---

## 7. 본 fork code 변경 정리 (commit history)

| commit | 내용 | 영향 파일 |
|---|---|---|
| `ec886b240` ⭐⭐ | SUB_085 Phase 2 unblock + SUB_086 fair baseline | (measurement, no core code) |
| (in commit) | **SUB_081 vLLM core patch** | `vllm/utils/__init__.py` +5 줄 (FlexibleArgumentParser re-export) |
| `8cee979ef` | **SUB_084 vLLM core patch** | `vllm/engine/arg_utils.py` +9 줄 (`_is_v1_supported_oracle` stub) |

→ **본 fork vLLM core 변경 = 14 줄 (backward-compat 100%, default behavior에 영향 0)**.

본 session (2026-05-24~25) 총 commit: **18** (모두 `feat/spec-decode-tuning` branch에 push 완료).

---

## 8. raw 자료 link

| 항목 | 위치 |
|---|---|
| **종합 리포트** | [`shadow_assists/.../COMPREHENSIVE_REPORT_20260525.md`](../shadow_assists/features/IDE_006/TSK_020/COMPREHENSIVE_REPORT_20260525.md) (416 lines) |
| **outstanding contributions** | [`shadow_assists/.../OUTSTANDING_CONTRIBUTIONS_20260525.md`](../shadow_assists/features/IDE_006/TSK_020/OUTSTANDING_CONTRIBUTIONS_20260525.md) (243 lines) |
| Best Configuration | [`shadow_assists/.../Best_SpecDecode_10778tps.md`](../shadow_assists/features/IDE_006/TSK_020/Best_SpecDecode_10778tps.md) §0 (Trident production-ready) |
| 분석 doc (40 reference) | [`shadow_assists/.../analysis/workload_acceptance_analysis_20260524.md`](../shadow_assists/features/IDE_006/TSK_020/analysis/workload_acceptance_analysis_20260524.md) (680+ lines) |
| INDEX nav | [`shadow_assists/.../INDEX.md`](../shadow_assists/features/IDE_006/TSK_020/INDEX.md) |
| idea backlog | [`shadow_assists/.../idea/README.md`](../shadow_assists/features/IDE_006/TSK_020/idea/README.md) |

### 8.1 SUB measurement RESULTS

| SUB | RESULTS | 핵심 |
|---|---|---|
| **SUB_085 v2** ⭐⭐ | [`measurements/sub085_suffix_piecewise_20260525/`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub085_suffix_piecewise_20260525/RESULTS.md) | Trident best (suffix PIECEWISE) |
| **SUB_086** | [`measurements/sub086_vanilla_gmu080_20260525/`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub086_vanilla_gmu080_20260525/RESULTS.md) | fair vanilla baseline (gmu=0.80) |
| **SUB_087** | [`measurements/sub087_ngram_piecewise_20260525/`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub087_ngram_piecewise_20260525/RESULTS.md) | ngram cap=8 PIECEWISE fair baseline |
| **SUB_089** | [`measurements/sub089_sonnet_3run_20260525/`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub089_sonnet_3run_20260525/RESULTS.md) | sonnet canonical 3-run (var 0.20%) |
| **SUB_093** ⭐⭐ | [`measurements/sub093_full_matrix_util_20260525/`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub093_full_matrix_util_20260525/RESULTS.md) | **full 57-cell matrix + util** (Llama 70B 18 + 소형 27 + cross-val 12) |
| **SUB_090** | [`measurements/sub090_model_size_sweep_20260525/`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub090_model_size_sweep_20260525/RESULTS.md) | R/K boundary 7B↔70B |
| **SUB_091** ⭐⭐ | [`measurements/sub091_issue16258_precise_20260525/`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub091_issue16258_precise_20260525/RESULTS.md) | opt-125m 2.13× = issue #16258 정확 reproduction |
| SUB_092 | [`measurements/sub092_router_poc_20260525/`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub092_router_poc_20260525/RESULTS.md) | router HTTP server PoC |
| (이전 SUB_044~084) | `measurements/sub04X~08X_*/RESULTS.md` (15+ docs) | historical baseline + Phase 1~4 |

### 8.2 외부 reference (key)

- vLLM Issue #16258 — small model + ngram regression (정확 reproduction in SUB_091)
- arXiv 2411.04975 — SuffixDecoding (Snowflake AI Research, NeurIPS 2025)
- vLLM Issue #24943 — cudagraph_mode=PIECEWISE를 직접 suggest
- vLLM Issue #33341 — spec drafter PIECEWISE only (current state)
- Snowflake blog — SuffixDecoding at production scale
- 본 fork 분석 doc §10 — 40 reference 정리

---

## 9. 적용 권장 한 줄

```python
# AGSD gating decision = Trident core (Llama-70B + TP=8 + H100×8 → 6 workload 모두 net positive)
LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=8, gpu_memory_utilization=0.80,
    compilation_config={"cudagraph_mode": "PIECEWISE"},
    speculative_config={"method": "suffix", "num_speculative_tokens": 32})

# AGSD gating decision = vanilla (≤7B model → spec OFF)
```

```bash
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/pip install arctic-inference   # SuffixDecodingCache class 만 사용
```
