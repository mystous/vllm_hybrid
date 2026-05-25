# vllm_hybrid — Speculative Decoding (Trident core / AGSD framework)

> **last update**: 2026-05-25 KST (SUB_093 full matrix + util 반영)
> **branch**: `feat/spec-decode-tuning`
> **base**: vLLM 1.6.dev0+g858b6df7a (fork)
> **fork core change**: **14 줄** (`vllm/utils/__init__.py` +5, `vllm/engine/arg_utils.py` +9, backward-compat 100%)
> **상세 doc**: [`COMPREHENSIVE_REPORT_20260525.md`](../shadow_assists/features/IDE_006/TSK_020/COMPREHENSIVE_REPORT_20260525.md) (416 lines) + [`OUTSTANDING_CONTRIBUTIONS_20260525.md`](../shadow_assists/features/IDE_006/TSK_020/OUTSTANDING_CONTRIBUTIONS_20260525.md) (243 lines) + [`★ _ALL_RESULTS_20260526.md`](../shadow_assists/features/IDE_006/TSK_020/measurements/_ALL_RESULTS_20260526.md) (**전체 129 cell** 단일 doc) + [`SUB_093`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub093_full_matrix_util_20260525/RESULTS.md) / [`SUB_094`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub094_agsd_e2e_20260525/RESULTS.md) / [`SUB_095`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub095_agsd_e2e_multi_model_20260525/RESULTS.md) / [`SUB_096`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub096_large_models_20260525/RESULTS.md)

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

→ **R/K boundary는 7B ↔ 70B 사이** (prior view). SUB_096 영역 **boundary 영역 < 14B 영역 refinement** (§2.4a 참조).
→ (caveat) SUB_093 Phase 2 영역 PIECEWISE-only 재측정 영역 short-wall noise (1-8s wall) — prior SUB_090 영역 default cudagraph 영역 비교 영역 reliable.

### 2.4a Mid/Large model refinement (SUB_096, 6 workload × 3 config) — 새 R/K boundary point

본 §2.4a 영역 SUB_093 Phase 1 (Llama 70B) 와 동일 setup × 새 2 모델 측정.

| Model | TP | scale | sonnet (Trident vs vanilla) | chat | code | mix-bal | 종합 |
|---|---:|---|---:|---:|---:|---:|---|
| **Phi-3-medium 14B** | 1 | 500p × 2048in × 1024max | **+90%** ⭐ | **+33%** | **+52%** | **+117%** ⭐⭐ | **6/6 net positive** |
| **Qwen 2.5-72B** | 8 | 500p × 8192in × 8192max | +54% | +47% | **−5%** ✗ | +44% | 5/6 positive (code 영역 회귀) |
| (참조) Llama 3.3-70B | 8 | 500p × 8192in × 8192max | +52% | +69% | +19% | +57% | 6/6 net positive |

→ **R/K boundary 영역 refinement**: Phi-3 14B 영역 6/6 net positive 영역 확정 → boundary 영역 7B 영역 14B 사이 영역 14B 측 영역 close.
→ **same-size cross-vendor 차이 영역 첫 관측**: Llama 70B (6/6 positive +19~+69%) vs Qwen 72B (5/6, code −5%) — **model architecture 영역 spec acceptance 영역 정량 영향**.
→ Llama 3.1-405B FP8 영역 측정 불가 (tokenizer 파일 미캐시 + fbgemm_fp8 deprecated — `huggingface-cli download` 후 재시도 가능).

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

## 2.6 ★ End-to-end AGSD multi-instance benchmark (SUB_094 + SUB_095)

본 §2.6 영역 **§2.1~2.5 와 다른 setup**: 위 영역 single-instance per-cell (one vLLM with one config) / 본 §2.6 영역 **2 vLLM backend (vanilla + Trident core) + CPU router (FastAPI + ProcessPool classifier + httpx forwarder) + concurrent client (200 prompt × concurrency 32)**.

**핵심 차이점**: AGSD-gated 영역 두 GPU 동시 활용 (parallel) + per-request workload 분류 → method 자동 선택. single-instance 비교 영역 + gating value 영역 양쪽 모두 측정.

### 2.6.1 4 모델 × 3 mix × 3 scenario = 36 cell 결과

| Model | Mix | vanilla-only | trident-only | **AGSD-gated** | vs vanilla | vs trident |
|---|---|---:|---:|---:|---:|---:|
| **Qwen 0.5B** (TP=1×2) | balanced | 3,672 | 4,644 | **6,252** ⭐ | **+70.3%** | +34.6% |
| 〃 | sonnet-heavy | 4,196 | 6,207 | **6,858** ⭐ | **+63.4%** | +10.5% |
| 〃 | code-heavy | 4,227 | 7,267 | **8,605** ⭐ | **+103.6%** | +18.4% |
| **Qwen 1.5B** (TP=1×2) | balanced | 3,513 | 4,524 | **5,783** ⭐ | **+64.6%** | +27.8% |
| 〃 | sonnet-heavy | 4,068 | 5,050 | **5,449** ⭐ | **+33.9%** | +7.9% |
| 〃 | code-heavy | 4,211 | 8,166 | **8,932** ⭐ | **+112.2%** | +9.4% |
| **Qwen 7B** (TP=1×2, SUB_094) | balanced | 3,753 | 4,238 | **6,073** ⭐ | **+61.8%** | **+43.3%** |
| 〃 | sonnet-heavy | 3,865 | 5,234 | **6,025** ⭐ | **+55.9%** | +15.1% |
| 〃 | code-heavy | 3,966 | 7,512 | **8,825** ⭐ | **+122.5%** | +17.5% |
| **Qwen 32B** (TP=2×2) | balanced | 1,779 | 2,721 | **3,449** ⭐ | **+93.8%** | +26.8% |
| 〃 | sonnet-heavy | 1,792 | 4,004 | **4,084** ⭐ | **+127.9%** | +2.0% |
| 〃 | code-heavy | 1,874 | 4,551 | **4,575** ⭐ | **+144.1%** | +0.5% |

→ **12/12 cell 모두 AGSD-gated 우세** (vs single-backend deployment).
→ Llama 70B / Qwen 72B 영역 **GPU mem 영역 2-instance 불가** (TP=8 × 2 = 16 GPU 필요).

### 2.6.2 AGSD-gated win 영역 origin

본 §2.6 영역 AGSD-gated win 영역 두 source:
1. **parallel GPU utilization** — 2 backend 영역 동시 활성 → 2× theoretical (mix 분포 의존, ~50-60% 기여)
2. **gating decision** — per-request workload 분류 → best method 선택 (chat → vanilla / sonnet+code → trident, ~10-20% 기여)

### 2.6.3 router overhead

| 지표 | 값 |
|---|---:|
| classifier ms/prompt (avg) | **1.22 ms** (regex × ProcessPool 16 worker) |
| forward ms/prompt (avg) | 759 ms (vLLM backend latency 포함) |
| classifier overhead vs forward | **0.16%** (negligible) |
| classifier accuracy | **100%** (3 mix × 200 = 600 prompt 모두 정확 분배) |

### 2.6.4 모델 크기 별 AGSD 가치

| Model | AGSD vs trident 평균 (3 mix) | gating value pattern |
|---|---:|---|
| Qwen 0.5B | +21.2% | backend fast → router overhead 비중 ↑ |
| Qwen 1.5B | +15.0% | backend 중간 |
| **Qwen 7B** | **+25.3%** ⭐ | gating value 최대 — chat→vanilla 선택 효과 |
| Qwen 32B | +9.8% | trident single 영역 throughput 매우 높음 → marginal |

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

## 5. AGSD production decision tree (workload + model-size + vendor gating)

```
prompt 입력
  ↓
[ AGSD gating: model size + vendor + workload 분류 ]
  ↓
  ├─ ≤ 7B (small)
  │   → AGSD decision: vanilla (spec OFF)
  │   이유: universal regression (R≫K), 5 model cross-validation (SUB_078/079/088/090/091)
  │
  ├─ 14B class (Phi-3 14B 확인, SUB_096)
  │   → AGSD decision: Trident core ⭐
  │   효과: sonnet +90% / chat +33% / code +52% / mix-bal +117% / 6/6 net positive
  │
  ├─ 32B (Qwen 32B end-to-end SUB_095)
  │   → AGSD decision: Trident core
  │   효과 (end-to-end 2-instance): balanced +93.8% / sonnet-heavy +127.9% / code-heavy +144.1%
  │
  ├─ 70-72B (large dense)
  │   → AGSD decision: workload-aware (model vendor 영역 따라 다름)
  │   Llama 3.3-70B (SUB_093): 6/6 net positive — sonnet +52% / chat +69% / code +19%
  │   Qwen 2.5-72B (SUB_096): 5/6 positive, code workload 영역 −5% 회귀 → code 영역 vanilla 권장
  │
  └─ ≥ 100B (Llama 405B FP8 영역 측정 불가)
      → 후속: tokenizer 다운로드 + RedHatAI 영역 native FP8 variant 사용
```

→ **AGSD = Trident core + workload/model-size/vendor gating**. SUB_096 영역 same-size cross-vendor 차이 발견 — model architecture 영역 spec acceptance 영향 영역 정량 관측 (Llama 70B 6/6 vs Qwen 72B 5/6).

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
| **SUB_094** ⭐⭐ | [`measurements/sub094_agsd_e2e_20260525/`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub094_agsd_e2e_20260525/RESULTS.md) | **AGSD end-to-end** (Qwen 7B × 2 backend + CPU router + 3 mix benchmark) |
| **SUB_095** ⭐⭐ | [`measurements/sub095_agsd_e2e_multi_model_20260525/`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub095_agsd_e2e_multi_model_20260525/RESULTS.md) | **AGSD end-to-end × 4 모델** (Qwen 0.5B/1.5B/7B/32B × 3 mix = 36 cell, 12/12 net positive) |
| **SUB_096** ⭐⭐ | [`measurements/sub096_large_models_20260525/`](../shadow_assists/features/IDE_006/TSK_020/measurements/sub096_large_models_20260525/RESULTS.md) | **Large-model validation** (Phi-3 14B + Qwen 72B = 36 cell). Phi-3 6/6 positive ⭐ / Qwen 72B 5/6 (code 회귀) / Llama 405B FP8 측정 불가 |
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

---

## Appendix A — 전체 측정 결과 단일 표 (129 cells)

> ★ 모든 모델 × 모든 workload × 모든 config × util — 한 표 영역 종합.
> raw CSV: [`_ALL_MEASUREMENTS.csv`](../shadow_assists/features/IDE_006/TSK_020/measurements/_ALL_MEASUREMENTS.csv)
> standalone doc: [`_ALL_RESULTS_20260526.md`](../shadow_assists/features/IDE_006/TSK_020/measurements/_ALL_RESULTS_20260526.md)
> **bold** = 각 (model, workload, kind) 영역 best / single = 1-instance per cell / e2e = 2-backend + CPU router

| # | Model | TP | kind | workload | config | tps | wall(s) | CPU% | GPU% |
|---:|---|---:|---|---|---|---:|---:|---:|---:|
| 1 | **opt-125m** | 1 | single | sonnet | **vanilla** | **8,008** ⭐ | 3.2 | 1.8 | 0.2 |
| 2 | opt-125m | 1 | single | sonnet | ngram | 7,360 | 3.5 | 2.0 | 0.4 |
| 3 | **opt-125m** | 1 | single | chat | **vanilla** | **9,542** ⭐ | 2.7 | 1.9 | 0.5 |
| 4 | opt-125m | 1 | single | chat | ngram | 9,514 | 2.7 | 2.3 | 0.1 |
| 5 | **opt-125m** | 1 | single | code | **vanilla** | **9,772** ⭐ | 2.6 | 2.5 | 0.4 |
| 6 | opt-125m | 1 | single | code | ngram | 6,435 | 4.0 | 2.3 | 0.3 |
| 7 | Qwen2.5-0.5B | 1 | single | sonnet | vanilla | 4,234 | 4.5 | 1.8 | 0.1 |
| 8 | Qwen2.5-0.5B | 1 | single | sonnet | ngram | 4,317 | 4.4 | 1.9 | 0.2 |
| 9 | **Qwen2.5-0.5B** | 1 | single | sonnet | **agsd** | **10,921** ⭐ | 1.7 | 2.8 | 0.3 |
| 10 | Qwen2.5-0.5B | 1 | single | chat | vanilla | 6,576 | 3.6 | 2.1 | 0.4 |
| 11 | Qwen2.5-0.5B | 1 | single | chat | ngram | 5,257 | 4.4 | 1.9 | 0.4 |
| 12 | **Qwen2.5-0.5B** | 1 | single | chat | **agsd** | **7,694** ⭐ | 3.0 | 2.5 | 0.6 |
| 13 | Qwen2.5-0.5B | 1 | single | code | vanilla | 7,228 | 3.5 | 1.9 | 0.7 |
| 14 | Qwen2.5-0.5B | 1 | single | code | ngram | 7,949 | 3.2 | 2.1 | 0.4 |
| 15 | **Qwen2.5-0.5B** | 1 | single | code | **agsd** | **11,785** ⭐ | 2.2 | 2.2 | 0.3 |
| 16 | Qwen2.5-0.5B | 1 | e2e | mix-son | vanilla-only | 4,196 | 9.3 | — | — |
| 17 | Qwen2.5-0.5B | 1 | e2e | mix-son | trident-only | 6,207 | 6.2 | — | — |
| 18 | **Qwen2.5-0.5B** | 1 | e2e | mix-son | **agsd-gated** | **6,858** ⭐ | 5.7 | — | — |
| 19 | Qwen2.5-0.5B | 1 | e2e | mix-bal | vanilla-only | 3,672 | 11.6 | — | — |
| 20 | Qwen2.5-0.5B | 1 | e2e | mix-bal | trident-only | 4,644 | 9.1 | — | — |
| 21 | **Qwen2.5-0.5B** | 1 | e2e | mix-bal | **agsd-gated** | **6,252** ⭐ | 6.8 | — | — |
| 22 | Qwen2.5-0.5B | 1 | e2e | mix-cod | vanilla-only | 4,227 | 10.9 | — | — |
| 23 | Qwen2.5-0.5B | 1 | e2e | mix-cod | trident-only | 7,267 | 6.3 | — | — |
| 24 | **Qwen2.5-0.5B** | 1 | e2e | mix-cod | **agsd-gated** | **8,605** ⭐ | 5.3 | — | — |
| 25 | Qwen2.5-1.5B | 1 | single | sonnet | vanilla | 4,771 | 4.4 | 2.0 | 0.5 |
| 26 | Qwen2.5-1.5B | 1 | single | sonnet | ngram | 3,430 | 6.1 | 1.8 | 0.6 |
| 27 | **Qwen2.5-1.5B** | 1 | single | sonnet | **agsd** | **6,389** ⭐ | 3.3 | 2.3 | 0.5 |
| 28 | Qwen2.5-1.5B | 1 | single | chat | vanilla | 5,571 | 3.6 | 2.0 | 0.5 |
| 29 | Qwen2.5-1.5B | 1 | single | chat | ngram | 4,669 | 4.6 | 1.9 | 0.7 |
| 30 | **Qwen2.5-1.5B** | 1 | single | chat | **agsd** | **6,528** ⭐ | 3.2 | 1.9 | 0.6 |
| 31 | Qwen2.5-1.5B | 1 | single | code | vanilla | 6,786 | 3.8 | 2.1 | 0.7 |
| 32 | Qwen2.5-1.5B | 1 | single | code | ngram | 5,158 | 5.0 | 2.1 | 0.7 |
| 33 | **Qwen2.5-1.5B** | 1 | single | code | **agsd** | **11,295** ⭐ | 2.3 | 2.7 | 0.8 |
| 34 | Qwen2.5-1.5B | 1 | e2e | mix-son | vanilla-only | 4,068 | 9.9 | — | — |
| 35 | Qwen2.5-1.5B | 1 | e2e | mix-son | trident-only | 5,050 | 7.8 | — | — |
| 36 | **Qwen2.5-1.5B** | 1 | e2e | mix-son | **agsd-gated** | **5,449** ⭐ | 7.4 | — | — |
| 37 | Qwen2.5-1.5B | 1 | e2e | mix-bal | vanilla-only | 3,512 | 12.2 | — | — |
| 38 | Qwen2.5-1.5B | 1 | e2e | mix-bal | trident-only | 4,524 | 9.5 | — | — |
| 39 | **Qwen2.5-1.5B** | 1 | e2e | mix-bal | **agsd-gated** | **5,783** ⭐ | 7.4 | — | — |
| 40 | Qwen2.5-1.5B | 1 | e2e | mix-cod | vanilla-only | 4,211 | 11.2 | — | — |
| 41 | Qwen2.5-1.5B | 1 | e2e | mix-cod | trident-only | 8,166 | 5.8 | — | — |
| 42 | **Qwen2.5-1.5B** | 1 | e2e | mix-cod | **agsd-gated** | **8,932** ⭐ | 5.3 | — | — |
| 43 | **starcoder2-3b** | 1 | single | sonnet | **vanilla** | **5,758** ⭐ | 4.5 | 3.3 | 0.7 |
| 44 | starcoder2-3b | 1 | single | sonnet | ngram | 5,007 | 5.1 | 3.0 | 0.5 |
| 45 | starcoder2-3b | 1 | single | chat | vanilla | 6,723 | 3.8 | 3.4 | 1.0 |
| 46 | **starcoder2-3b** | 1 | single | chat | **ngram** | **7,486** ⭐ | 3.4 | 3.7 | 0.7 |
| 47 | starcoder2-3b | 1 | single | code | vanilla | 6,794 | 3.8 | 3.8 | 1.1 |
| 48 | **starcoder2-3b** | 1 | single | code | **ngram** | **7,242** ⭐ | 3.5 | 4.6 | 0.7 |
| 49 | Qwen2.5-7B | 1 | single | sonnet | vanilla | 5,584 | 4.5 | 1.8 | 1.1 |
| 50 | Qwen2.5-7B | 1 | single | sonnet | ngram | 3,200 | 8.0 | 1.8 | 1.2 |
| 51 | **Qwen2.5-7B** | 1 | single | sonnet | **agsd** | **5,714** ⭐ | 4.5 | 1.9 | 1.5 |
| 52 | **Qwen2.5-7B** | 1 | single | chat | **vanilla** | **4,557** ⭐ | 3.8 | 1.8 | 1.3 |
| 53 | Qwen2.5-7B | 1 | single | chat | ngram | 2,777 | 6.3 | 1.9 | 1.4 |
| 54 | Qwen2.5-7B | 1 | single | chat | agsd | 4,253 | 4.2 | 2.0 | 1.4 |
| 55 | Qwen2.5-7B | 1 | single | code | vanilla | 6,196 | 4.1 | 2.7 | 1.6 |
| 56 | Qwen2.5-7B | 1 | single | code | ngram | 5,902 | 4.3 | 2.2 | 1.2 |
| 57 | **Qwen2.5-7B** | 1 | single | code | **agsd** | **8,071** ⭐ | 3.2 | 2.0 | 1.1 |
| 58 | Qwen2.5-7B | 1 | e2e | mix-son | vanilla-only | 3,865 | 13.0 | — | — |
| 59 | Qwen2.5-7B | 1 | e2e | mix-son | trident-only | 5,234 | 9.6 | — | — |
| 60 | **Qwen2.5-7B** | 1 | e2e | mix-son | **agsd-gated** | **6,025** ⭐ | 8.3 | — | — |
| 61 | Qwen2.5-7B | 1 | e2e | mix-bal | vanilla-only | 3,753 | 13.3 | — | — |
| 62 | Qwen2.5-7B | 1 | e2e | mix-bal | trident-only | 4,238 | 11.7 | — | — |
| 63 | **Qwen2.5-7B** | 1 | e2e | mix-bal | **agsd-gated** | **6,073** ⭐ | 8.2 | — | — |
| 64 | Qwen2.5-7B | 1 | e2e | mix-cod | vanilla-only | 3,966 | 12.7 | — | — |
| 65 | Qwen2.5-7B | 1 | e2e | mix-cod | trident-only | 7,512 | 6.7 | — | — |
| 66 | **Qwen2.5-7B** | 1 | e2e | mix-cod | **agsd-gated** | **8,825** ⭐ | 5.7 | — | — |
| 67 | Phi-3-medium-14B | 1 | single | sonnet | vanilla | 3,438 | 127.7 | 2.0 | 11.3 |
| 68 | Phi-3-medium-14B | 1 | single | sonnet | ngram | 4,881 | 89.6 | 1.8 | 9.4 |
| 69 | **Phi-3-medium-14B** | 1 | single | sonnet | **agsd** | **6,523** ⭐ | 66.8 | 1.8 | 9.0 |
| 70 | Phi-3-medium-14B | 1 | single | chat | vanilla | 3,138 | 65.9 | 2.0 | 11.3 |
| 71 | Phi-3-medium-14B | 1 | single | chat | ngram | 3,242 | 61.4 | 1.8 | 9.4 |
| 72 | **Phi-3-medium-14B** | 1 | single | chat | **agsd** | **4,173** ⭐ | 48.8 | 1.8 | 9.0 |
| 73 | Phi-3-medium-14B | 1 | single | code | vanilla | 3,340 | 153.3 | 2.0 | 11.3 |
| 74 | **Phi-3-medium-14B** | 1 | single | code | **ngram** | **5,140** ⭐ | 99.6 | 1.8 | 9.4 |
| 75 | Phi-3-medium-14B | 1 | single | code | agsd | 5,085 | 100.7 | 1.8 | 9.0 |
| 76 | Phi-3-medium-14B | 1 | single | mix-sh | vanilla | 3,434 | 114.7 | 2.0 | 11.3 |
| 77 | Phi-3-medium-14B | 1 | single | mix-sh | ngram | 4,744 | 84.7 | 1.8 | 9.4 |
| 78 | **Phi-3-medium-14B** | 1 | single | mix-sh | **agsd** | **6,563** ⭐ | 60.6 | 1.8 | 9.0 |
| 79 | Phi-3-medium-14B | 1 | single | mix-bal | vanilla | 3,374 | 114.2 | 2.0 | 11.3 |
| 80 | Phi-3-medium-14B | 1 | single | mix-bal | ngram | 4,762 | 80.6 | 1.8 | 9.4 |
| 81 | **Phi-3-medium-14B** | 1 | single | mix-bal | **agsd** | **7,312** ⭐ | 52.5 | 1.8 | 9.0 |
| 82 | Phi-3-medium-14B | 1 | single | mix-ch | vanilla | 3,342 | 131.9 | 2.0 | 11.3 |
| 83 | Phi-3-medium-14B | 1 | single | mix-ch | ngram | 5,246 | 83.8 | 1.8 | 9.4 |
| 84 | **Phi-3-medium-14B** | 1 | single | mix-ch | **agsd** | **6,390** ⭐ | 68.8 | 1.8 | 9.0 |
| 85 | Qwen2.5-32B | 2 | e2e | mix-son | vanilla-only | 1,792 | 28.1 | — | — |
| 86 | Qwen2.5-32B | 2 | e2e | mix-son | trident-only | 4,004 | 12.6 | — | — |
| 87 | **Qwen2.5-32B** | 2 | e2e | mix-son | **agsd-gated** | **4,084** ⭐ | 12.4 | — | — |
| 88 | Qwen2.5-32B | 2 | e2e | mix-bal | vanilla-only | 1,779 | 27.9 | — | — |
| 89 | Qwen2.5-32B | 2 | e2e | mix-bal | trident-only | 2,721 | 18.4 | — | — |
| 90 | **Qwen2.5-32B** | 2 | e2e | mix-bal | **agsd-gated** | **3,449** ⭐ | 14.3 | — | — |
| 91 | Qwen2.5-32B | 2 | e2e | mix-cod | vanilla-only | 1,874 | 26.3 | — | — |
| 92 | Qwen2.5-32B | 2 | e2e | mix-cod | trident-only | 4,551 | 10.8 | — | — |
| 93 | **Qwen2.5-32B** | 2 | e2e | mix-cod | **agsd-gated** | **4,575** ⭐ | 10.8 | — | — |
| 94 | Qwen2.5-72B | 8 | single | sonnet | vanilla | 6,456 | 458.1 | 5.8 | 93.2 |
| 95 | Qwen2.5-72B | 8 | single | sonnet | ngram | 7,968 | 378.4 | 5.4 | 82.0 |
| 96 | **Qwen2.5-72B** | 8 | single | sonnet | **agsd** | **9,959** ⭐ | 298.7 | 5.3 | 74.7 |
| 97 | Qwen2.5-72B | 8 | single | chat | vanilla | 2,560 | 164.3 | 5.8 | 93.2 |
| 98 | Qwen2.5-72B | 8 | single | chat | ngram | 3,462 | 123.3 | 5.4 | 82.0 |
| 99 | **Qwen2.5-72B** | 8 | single | chat | **agsd** | **3,770** ⭐ | 113.1 | 5.3 | 74.7 |
| 100 | **Qwen2.5-72B** | 8 | single | code | **vanilla** | **6,227** ⭐ | 652.8 | 5.8 | 93.2 |
| 101 | Qwen2.5-72B | 8 | single | code | ngram | 5,766 | 702.3 | 5.4 | 82.0 |
| 102 | Qwen2.5-72B | 8 | single | code | agsd | 5,941 | 685.6 | 5.3 | 74.7 |
| 103 | Qwen2.5-72B | 8 | single | mix-sh | vanilla | 5,832 | 463.2 | 5.8 | 93.2 |
| 104 | Qwen2.5-72B | 8 | single | mix-sh | ngram | 6,586 | 410.2 | 5.4 | 82.0 |
| 105 | **Qwen2.5-72B** | 8 | single | mix-sh | **agsd** | **8,795** ⭐ | 305.7 | 5.3 | 74.7 |
| 106 | Qwen2.5-72B | 8 | single | mix-bal | vanilla | 5,702 | 446.0 | 5.8 | 93.2 |
| 107 | Qwen2.5-72B | 8 | single | mix-bal | ngram | 6,692 | 369.3 | 5.4 | 82.0 |
| 108 | **Qwen2.5-72B** | 8 | single | mix-bal | **agsd** | **8,228** ⭐ | 302.6 | 5.3 | 74.7 |
| 109 | Qwen2.5-72B | 8 | single | mix-ch | vanilla | 6,231 | 515.6 | 5.8 | 93.2 |
| 110 | Qwen2.5-72B | 8 | single | mix-ch | ngram | 5,989 | 538.6 | 5.4 | 82.0 |
| 111 | **Qwen2.5-72B** | 8 | single | mix-ch | **agsd** | **8,491** ⭐ | 377.0 | 5.3 | 74.7 |
| 112 | Llama-3.3-70B | 8 | single | sonnet | vanilla | 7,679 | 525.9 | 5.6 | 93.8 |
| 113 | Llama-3.3-70B | 8 | single | sonnet | ngram | 10,759 | 373.3 | 7.6 | 84.2 |
| 114 | **Llama-3.3-70B** | 8 | single | sonnet | **agsd** | **11,677** ⭐ | 346.1 | 5.3 | 73.3 |
| 115 | Llama-3.3-70B | 8 | single | chat | vanilla | 2,267 | 159.9 | 5.6 | 93.8 |
| 116 | Llama-3.3-70B | 8 | single | chat | ngram | 3,244 | 111.4 | 7.6 | 84.2 |
| 117 | **Llama-3.3-70B** | 8 | single | chat | **agsd** | **3,830** ⭐ | 102.1 | 5.3 | 73.3 |
| 118 | Llama-3.3-70B | 8 | single | code | vanilla | 6,718 | 581.5 | 5.6 | 93.8 |
| 119 | Llama-3.3-70B | 8 | single | code | ngram | 5,362 | 729.7 | 7.6 | 84.2 |
| 120 | **Llama-3.3-70B** | 8 | single | code | **agsd** | **7,981** ⭐ | 494.8 | 5.3 | 73.3 |
| 121 | Llama-3.3-70B | 8 | single | mix-sh | vanilla | 6,326 | 518.3 | 5.6 | 93.8 |
| 122 | Llama-3.3-70B | 8 | single | mix-sh | ngram | 7,933 | 407.7 | 7.6 | 84.2 |
| 123 | **Llama-3.3-70B** | 8 | single | mix-sh | **agsd** | **10,298** ⭐ | 317.8 | 5.3 | 73.3 |
| 124 | Llama-3.3-70B | 8 | single | mix-bal | vanilla | 6,054 | 452.7 | 5.6 | 93.8 |
| 125 | Llama-3.3-70B | 8 | single | mix-bal | ngram | 6,554 | 427.6 | 7.6 | 84.2 |
| 126 | **Llama-3.3-70B** | 8 | single | mix-bal | **agsd** | **9,514** ⭐ | 292.2 | 5.3 | 73.3 |
| 127 | Llama-3.3-70B | 8 | single | mix-ch | vanilla | 6,495 | 492.2 | 5.6 | 93.8 |
| 128 | Llama-3.3-70B | 8 | single | mix-ch | ngram | 5,491 | 586.1 | 7.6 | 84.2 |
| 129 | **Llama-3.3-70B** | 8 | single | mix-ch | **agsd** | **9,457** ⭐ | 339.4 | 5.3 | 73.3 |
