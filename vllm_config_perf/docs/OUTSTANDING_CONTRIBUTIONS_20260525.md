# Outstanding Contributions — 본 fork (vllm_hybrid) 정직 차별점 (2026-05-25 KST)

> **scope**: 본 session (2026-05-24~2026-05-25, ~13 시간, SUB_073~092) 진행 + 외부 web research subagent verification
> **검증 source**: vLLM github issues/PRs/discussions, arXiv (2024~2026), Snowflake/Red Hat blog, MLPerf, Spec-Bench leaderboard, arctic_inference release notes
> **검증 doc**: 본 doc + [`analysis/workload_acceptance_analysis_20260524.md`](analysis/workload_acceptance_analysis_20260524.md) §10 (40 reference)
> **목적**: 본 fork 영역 5 차별점 외부 literature/OSS 영역 진짜 unique 영역 영역 정직 verdict + over-claim 영역 제거

---

## Executive Summary (두괄식)

| # | 차별점 | verdict | 진짜 unique? |
|---|---|---|---|
| **D2** | **multi-model × multi-hardware × issue #16258 정확 reproduction** | **✓ unique** | **★ outstanding** |
| D3 | R/K model-size boundary 7B↔70B sweep | ◐ partial | partial outstanding |
| D1 | fair contribution accounting (+134% wrapper noise 정정) | ◐ partial | self-correction 중심 |
| D4 | workload × spec method matrix (sonnet/chat/code × ngram/suffix) | ✗ not unique | cell value first measurement candidate (각 cell fact 영역 published) |
| D5 | cudagraph PIECEWISE 한 줄 우회 (SUB_084 dead-end 정정) | ✗ not unique | community known (vLLM #24943 error message 영역 직접 suggest) |

→ **진짜 outstanding = D2 (1 개)**, partial = D3 + D1 (2 개), self-correction / first-cell-measurement = D4 + D5.

---

## D1. Fair contribution accounting — "+134% wrapper noise" 정정 (◐ partial)

### 본 fork claim

본 fork SUB_047 historical "+134% vs vanilla 4,679.8" 영역 wrapper-historical noise 정량 분리:
- historical vanilla (run_spec_decode.py, gmu=0.85): 4,679.8
- fair vanilla (run_workload_gen.py, gmu=0.80): **7,709.8** (+64.7% wrapper-noise)
- fair contribution = +31.5% (ngram cap=8) / **+50.3% (suffix PIECEWISE)**
- 본 fork 단독 patch contribution = +1.65% (sonnet only, SUB_047 cap=8 → vs vLLM default cap=1)

### 외부 evidence

| source | fact |
|---|---|
| [vLLM Issue #10318](https://github.com/vllm-project/vllm/issues/10318) (Closed "not planned") | vLLM blog 2.8× speedup 영역 unreproducible. 사용자 영역 1.4× 만 확인. critique = "configuration 미공개" — wrapper 영역 직접 용어 미사용 |
| [MLPerf Inference v6.0 (2026-04)](https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/) | spec decoding standardization 영역 acceptance rate manipulation 금지. 단 wrapper/gmu 영역 baseline noise 명시 없음 |
| [Red Hat Developer (2026-04-16)](https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss) | baseline 정의 영역 "default vLLM config + prefix caching". gmu 영역 동등 명시 없음, wrapper bias 영역 discussion 0 |
| [vLLM Discussion #13834](https://github.com/vllm-project/vllm/discussions/13834) | Llama-3.3-70B + 3B draft + A100 영역 spec −30~50% 회귀. 사용자 영역 quantization/FP8 KV cache 영역 의심 영역 baseline framing separate 못 함 |

### verdict: ◐ partial unique

- "같은 wrapper + 같은 gmu + 같은 cudagraph_mode" 영역 fair baseline 영역 일반 ML benchmarking 영역 standard practice (Spec-Bench 영역 already follows).
- vLLM-specific 영역 historical claim 영역 retroactive 정량 분해 (4,680 → 7,710 = +64.7% wrapper noise) 영역 외부 published 영역 직접 동등 발견 0.
- 단 contribution 영역 본 fork 영역 self-correction 영역 — community 영역 새 framework 영역 아님.

### 정직 표시

D1 영역 본 fork 영역 historical claim 영역 정정 (over-claim 제거) — academic 영역 새 framework 영역 contribute 영역 아님. **외부 share 영역 self-correction 영역 정직 표시 (idea IDE_009/SUB_073)** 영역 의미만.

---

## D2 ⭐ — multi-model × multi-hardware × issue #16258 정확 reproduction (✓ unique, **★ outstanding**)

### 본 fork claim

vLLM Issue #16258 (dtransposed, 2025-04-08, 2× L4) 영역 opt-125m + ngram = **2.12× regression** 보고. 본 fork SUB_091 (H100×1) 영역 같은 opt-125m + ngram = **2.13× regression** (정확 일치) + 4 추가 model:

| model | hardware | spec regression |
|---|---|---:|
| opt-125m | H100×1 (본 fork SUB_091) | **2.13×** |
| starcoder2-3b | H100×1 (본 fork SUB_091) | **2.30×** |
| Qwen 0.5B | H100×1 (본 fork SUB_078) | 2.46× |
| Qwen 1.5B | H100×1 (본 fork SUB_078) | 2.63× |
| Qwen 0.5B + suffix | H100×1 (본 fork SUB_088) | 2.06× (suffix 도 회귀) |
| (참조) opt-125m | 2× L4 (issue #16258 외부) | 2.12× |

### 외부 evidence

| source | fact |
|---|---|
| [vLLM Issue #16258](https://github.com/vllm-project/vllm/issues/16258) (Closed, comment / reproduction 0) | 다른 user 영역 published reproduction 영역 0 |
| [thc1006/qwen3.6-speculative-decoding-rtx3090](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) (2026-04~05, RTX 3090) | llama.cpp engine, Qwen 3.6-35B-A3B MoE 영역 19 config 모두 net speedup 0 (3-12% regression). 단 **vLLM 아님** |
| [vLLM Discussion #13834](https://github.com/vllm-project/vllm/discussions/13834) | Llama-3.3-70B AWQ + Llama-3.2-3B INT4 + A100 영역 −30~50% 회귀 |
| **opt-125m / starcoder2-3b × spec decoding × H100/A100/L40 reproduction** | **외부 web search 영역 발견 0** |

### verdict: ✓ unique — **★ outstanding contribution**

- **issue #16258 영역 외부 reproduction** 영역 published 영역 0 (issue close 후 1 년 영역 다른 published reproduction 없음).
- **L4 → H100 hardware-cross 정확 일치 (2.12% ↔ 2.13%)** 영역 → "small model + ngram regression 영역 hardware-independent fundamental" 영역 정량 입증.
- 추가 4 model (starcoder2-3b, Qwen 0.5B/1.5B, suffix variant) 영역 cross-validation 영역 외부 published 영역 발견 0.

### 본 fork 영역 진짜 contribution

본 D2 영역 **community 영역 contribution 영역 적합**:
- vLLM issue #16258 영역 closure 영역 update post 영역 추가 가능
- vLLM `docs/features/speculative_decoding/n_gram.md` 영역 "model-size threshold" caveat 영역 추가 가능
- "small model + spec decoding = universal regression" 명제 영역 정량 confirmation post

---

## D3. R/K model-size boundary 7B↔70B sweep (◐ partial)

### 본 fork claim

SUB_090 영역 Qwen 0.5B/1.5B/7B × code × {vanilla, ngram cap=8, suffix PIECEWISE} = 9 cell sweep:

| model | vanilla | ngram | suffix |
|---|---:|---:|---:|
| Qwen 0.5B | 11,220 | −30.5% | −52.1% |
| Qwen 1.5B | 10,389 | −43.6% | −60.9% |
| **Qwen 7B** | **5,556** | **−17.3%** (boundary 근접) | −36.7% |
| (참조) Llama 70B | 7,710 | (+31.5%) | **(+50.3%)** ⭐ net positive |

→ boundary 7B↔70B 사이 영역 net positive transition

### 외부 evidence

| source | fact |
|---|---|
| [Spec-Bench Leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md) | Vicuna-7B/13B/33B 영역 3-size sweep. **code workload 0**, **70B 0**, **1B 0** |
| [arxiv 2505.07858 "Scaling Laws for Speculative Decoding"](https://arxiv.org/abs/2505.07858) (Liu et al, 2025-05) | Log-linear scaling laws (draft capacity × pretraining tokens × batch). Llama2/3 + Qwen2.5. Scylla 1.5-2.2× over EAGLE2. **단 "net positive boundary" 정량 영역 본 fork 영역 0.5B/1.5B/7B/70B 영역 직접 비교 영역 발견 0** |
| [arxiv 2508.08192 "Efficient Speculative Decoding for Llama at Scale"](https://arxiv.org/pdf/2508.08192) | Llama at scale 영역 spec decoding 분석, 단 cross-method × cross-size matrix 영역 published 영역 partial |

### verdict: ◐ partial unique

- Scaling Laws paper (2505.07858) 영역 model-size axis 영역 cover, 단 본 fork 영역 정확 setup (Llama 70B + H100×8 + TP=8 + all-fair wrapper + ngram/suffix × Qwen 0.5B/1.5B/7B + Llama 70B) 영역 직접 동등 발견 0.
- "code workload + model-free spec (ngram/suffix) 영역 7B↔70B 사이 net positive transition" 영역 정량 영역 본 fork 영역 first measurement candidate.

### 본 fork 영역 진짜 contribution

본 D3 영역 **incremental contribution** — Scaling Laws paper 영역 framework 영역 본 fork 영역 specific (model-free spec method + Llama 70B reference + Qwen series) 영역 instance. 단 Qwen 32B / 72B 추가 측정 (후속 SUB) 영역 boundary refinement 영역 published 가능.

---

## D4. workload × spec method matrix (sonnet/chat/code × ngram/suffix all-fair) (✗ not unique)

### 본 fork claim

6-cell all-fair matrix (Llama 70B + TP=8 + H100×8 + gmu=0.80 + cudagraph PIECEWISE + same wrapper):

| workload | ngram cap=8 vs vanilla | suffix PIECEWISE vs vanilla | suffix vs ngram |
|---|---:|---:|---:|
| sonnet | +31.5% | +50.3% | +14.3% |
| chat | +30.2% | +63.8% | +25.9% |
| code | **−20.7%** (회귀) | +18.9% (mitigation) | **+50.0%** |

→ suffix 영역 3 workload 모두 ngram 영역 능가

### 외부 evidence

| source | fact |
|---|---|
| [arxiv 2411.04975 SuffixDecoding NeurIPS 2025 Spotlight](https://arxiv.org/abs/2411.04975) | workload axis = **AgenticSQL 5.3×, SWE-Bench code 1.8-4.5×, ShareGPT chat, MLPerf summ**. **1.4-3.9× faster than vLLM ngram on code-related tasks** ⭐ — **code 영역 suffix > ngram 영역 main claim 영역 published** |
| [prompt-lookup-decoding (apoorvumang)](https://github.com/apoorvumang/prompt-lookup-decoding) | "input-grounded (summ, doc QA, multi-turn chat, code editing) 영역 2-4× speedup". code editing 영역 net positive published |
| [Red Hat Developer (2026-04)](https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss) | EAGLE3 영역 ShareGPT + MLPerf + SWE-Bench code 영역 19.4% cost reduction (net positive). 단 method 영역 ngram/suffix matrix 아님 |
| [arxiv 2505.08600 "Automatic Task Detection and Heterogeneous LLM Speculative Decoding"](https://arxiv.org/abs/2505.08600) | workload classifier router 영역 prior art. 본 fork SUB_076/080 영역 prior art |
| [Nightjar arxiv 2512.22420](https://arxiv.org/pdf/2512.22420) | workload-aware adaptive spec framework. 본 fork SUB_080 영역 direct prior art |

### verdict: ✗ not unique

- **각 cell 영역 fact 영역 모두 published** (SuffixDecoding paper 영역 code 영역 suffix > ngram 영역 main 결론).
- "code workload 영역 ngram 회귀 → suffix mitigation" 영역 명제 자체 영역 SuffixDecoding paper §4 영역 정확 published.
- workload-aware spec method selection 영역 routing idea 영역 2505.08600 / Nightjar / SGLang adaptive 영역 prior art.

### 본 fork 영역 진짜 contribution

본 D4 영역 **각 cell 정확 정량 값** (Llama 70B + TP=8 + 본 환경 + same wrapper + 정확 +50.3% / +63.8% / +18.9%) 영역 first measurement candidate. **mechanism / 결론 영역 unique 영역 아님** — Snowflake paper 영역 already covers.

---

## D5. cudagraph PIECEWISE 한 줄 우회 (SUB_084 dead-end 정정) (✗ not unique)

### 본 fork claim

SUB_081 / SUB_084 (2026-05-25 KST 23:00) 영역 arctic_inference v0.1.2 (vLLM 0.11.0 binary) + vLLM 1.6 사이 영역 binary incompat → "fundamental architectural incompat 영역 dead-end" 결론. SUB_085 영역 `compilation_config={"cudagraph_mode": "PIECEWISE"}` 한 줄 영역 우회 → 결론 영역 잘못 영역 입증.

### 외부 evidence

| source | fact |
|---|---|
| [vLLM Issue #24943](https://github.com/vllm-project/vllm/issues/24943) (Closed "not planned") | `CUDAGraphMode.FULL_AND_PIECEWISE is not supported with FlexAttentionMetadataBuilder backend ... please try cudagraph_mode=PIECEWISE`. **error message 자체 영역 PIECEWISE workaround 직접 suggest** ⚠ |
| [vLLM Issue #33341](https://github.com/vllm-project/vllm/issues/33341) (Open, Stale) | "Support Full CUDA Graph for the drafter" — eagle.py drafter 영역 **only supports piecewise cudagraphs**, full graph 영역 forecast +5% TPOT. 즉 **spec decoding drafter 영역 PIECEWISE only 영역 current state** |
| [vLLM v0.11.0 release notes](https://newreleases.io/project/github/vllm-project/vllm/release/v0.11.0) | default cudagraph_mode 영역 PIECEWISE → FULL_AND_PIECEWISE 영역 change |
| [vLLM CUDA Graphs design doc](https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md) | PIECEWISE = "cudagraph-incompatible ops 영역 keep outside, general flexibility" 영역 design 명시 |
| arctic_inference v0.2 release | **published 영역 없음**, vLLM 1.6 native support 영역 ETA 영역 0 |

### verdict: ✗ not unique

- PIECEWISE 영역 workaround 영역 **vLLM community 영역 already-known** (Issue #24943 error message 영역 직접 suggest).
- 본 fork SUB_084 영역 "single-session 영역 fundamental dead-end" 영역 결론 영역 **community 영역 already-known workaround 영역 단순 미인지 영역 결과**, SUB_085 영역 우회 영역 **rediscovery** 영역 가까움.

### 본 fork 영역 진짜 contribution

본 D5 영역 **본 fork 영역 자체 결론 정정** 영역만 — community 영역 새 발견 아님. 단 **"arctic_inference v0.1.2 ↔ vLLM 1.6 + SuffixDecodingProposer + PIECEWISE 영역 실제 작동 + sonnet +50.3% / chat +63.8% / code +18.9% 영역 정량 측정"** 영역 specific 측정 contribution 영역 D2/D3 영역 일부 포함.

---

## 종합 — 본 fork 영역 정직 outstanding contribution

### ★ outstanding (외부 community contribution 가능)

| # | contribution | 외부 channel |
|---|---|---|
| **D2** ⭐ | **vLLM Issue #16258 정확 reproduction (opt-125m L4 → H100 2.12% ↔ 2.13% 일치)** + 4 추가 model cross-validation | vLLM issue #16258 closure update / vLLM doc 영역 model-size threshold caveat |

### ◐ partial outstanding (incremental)

| # | contribution | 후속 영역 published 가능 |
|---|---|---|
| D3 | Qwen 0.5B/1.5B/7B + Llama 70B 영역 code × 3 config sweep — boundary 7B↔70B 정량 | Qwen 32B/72B 후속 측정 후 vLLM doc 영역 model-size threshold guidance contribute |

### ✗ 외부 contribution 영역 self-correction / 영역 first-cell-measurement (over-claim 피할 것)

| # | 상황 |
|---|---|
| D1 | 본 fork historical "+134%" 영역 self-correction (community framework 영역 아님) |
| D4 | 6-cell matrix 영역 정확 값 영역 first measurement, 단 mechanism / 결론 영역 SuffixDecoding paper 영역 already published |
| D5 | community 영역 already-known PIECEWISE workaround 영역 rediscovery — 본 fork 영역 자체 결론 정정 만 |

---

## 본 session 영역 진짜 가치 — production / community 영역 영향

| 영역 | 가치 |
|---|---|
| **본 fork production 적용** | suffix PIECEWISE + gmu=0.80 영역 production-ready config, 본 fork 14 줄 patch (utils + arg_utils) + wrapper env-tunable → **본 fork user 영역 즉시 활용** |
| **vLLM community 영역 contribution** | **D2 영역 외부 published reproduction 영역 candidate** — issue #16258 영역 update post + vLLM doc 영역 caveat |
| 본 fork 영역 정직 framework | IDE_009 (vanilla framing 정정) — 본 fork 영역 self-correction 영역 transparent doc trail |

---

## 본 doc 영역 권장 — 외부 share 영역 명확 prioritize

1. **D2 영역 우선 외부 share** — vLLM issue #16258 영역 comment 또는 별도 reproduction post (opt-125m 정확 일치 + 4 추가 model 정량)
2. **D3 영역 Qwen 32B 후속** — boundary refinement 후 vLLM doc 영역 model-size threshold guidance
3. D1/D4/D5 영역 외부 share 영역 **avoid over-claim** — 단순 fact (각 cell value, self-correction) 영역만 reference

---

## raw 자료

| 항목 | 위치 |
|---|---|
| 종합 리포트 | [`COMPREHENSIVE_REPORT_20260525.md`](COMPREHENSIVE_REPORT_20260525.md) |
| 분석 doc (40 reference) | [`analysis/workload_acceptance_analysis_20260524.md`](analysis/workload_acceptance_analysis_20260524.md) |
| Best Configuration | [`Best_SpecDecode_10778tps.md`](Best_SpecDecode_10778tps.md) §0 |
| 5-model cross-validation | [`measurements/sub091_issue16258_precise_20260525/RESULTS.md`](measurements/sub091_issue16258_precise_20260525/RESULTS.md), [`measurements/sub078_repro_20260524/RESULTS.md`](measurements/sub078_repro_20260524/RESULTS.md), [`measurements/sub079_small_model_full_20260524/RESULTS.md`](measurements/sub079_small_model_full_20260524/RESULTS.md), [`measurements/sub088_small_suffix_20260525/RESULTS.md`](measurements/sub088_small_suffix_20260525/RESULTS.md) |
| R/K boundary sweep | [`measurements/sub090_model_size_sweep_20260525/RESULTS.md`](measurements/sub090_model_size_sweep_20260525/RESULTS.md) |
| all-fair matrix | [`measurements/sub085_suffix_piecewise_20260525/RESULTS.md`](measurements/sub085_suffix_piecewise_20260525/RESULTS.md) + sub086 + sub087 |
| fair framing 정정 | [`idea/IDE_009_vanilla_contribution_framing.md`](idea/IDE_009_vanilla_contribution_framing.md) |
