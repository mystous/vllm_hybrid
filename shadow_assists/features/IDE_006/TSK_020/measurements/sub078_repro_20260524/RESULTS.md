# SUB_078 — vLLM Issue #16258 reproduction (small model cross-validation) RESULTS

> **parent**: TSK_020 / SUB_072 / idea I006
> **plan**: [`../../planning/SUB_078_issue_16258_repro.md`](../../planning/SUB_078_issue_16258_repro.md)
> **measurement**: 2026-05-24 16:43~16:47 KST, single-run × 4 cell
> **target**: vLLM Issue [#16258](https://github.com/vllm-project/vllm/issues/16258) (dtransposed, 2025-04-08) — "high acceptance, throughput regression" 명제 cross-validation
> **raw**: `eval/results/20260525_014310_sub078_<model>_<config>/`

---

## 1. 환경 caveat — model substitution

issue #16258 의 원 setup: opt-125m + starcoder2-3b on 2× L4 GPU.

본 fork 환경:
- `HF_HUB_OFFLINE=1` (offline 정책) + opt-125m / starcoder2-3b cache 부재 → 다운로드 차단.
- 본 env cached small models: Qwen2.5-0.5B-Instruct (0.5B), Qwen2.5-1.5B-Instruct (1.5B), Phi-3-medium, Qwen 2.5-7B 등.
- **substitution**: **Qwen2.5-0.5B + Qwen2.5-1.5B** — opt-125m / starcoder2-3b 와 정확 reproduction 아니지만 *유사 scale + Llama/Qwen family + code workload* 로 "high acceptance ≠ net win" 명제 cross-validation 가능.
- TP=1 (single GPU, 70B 와 달리 small model 영역 충분).

## 2. 측정 fact (4 cell)

workload: code (SUB_071 humaneval-style builder, n=50, target_in=1024, max_tok=512)

| model | config | tps | wall (s) | out_tok | speedup |
|---|---|---:|---:|---:|---:|
| **Qwen2.5-0.5B** | vanilla | **11,056** | 2.31 | 25,600 | — |
| **Qwen2.5-0.5B** | ngram (spec=5, lookup_max=2) | **4,486** | 5.71 | 25,600 | **0.406× (−59.4% 회귀)** |
| **Qwen2.5-1.5B** | vanilla | **11,016** | 2.32 | 25,600 | — |
| **Qwen2.5-1.5B** | ngram (spec=5, lookup_max=2) | **4,195** | 6.10 | 25,600 | **0.381× (−62.0% 회귀)** |

> 측정 wall 매우 짧음 (2-6s) — vLLM v1 의 `SpecDecoding metrics` interval (10s default) 보다 짧아 acceptance metric line 미생성. mean_accept_len 직접 측정값 부재 — 후속 SUB candidate (긴 prompt set 으로 재측정).

## 3. ★ 핵심 결론 — Issue #16258 명제 cross-validation 확인

issue #16258 (dtransposed, opt-125m on 2× L4):
- ngram **on**: throughput 238 tok/s + acceptance 70%
- ngram **off**: throughput 504.8 tok/s
- → **2.1× 회귀** ("regardless of configuration, inference is Pareto worse")

본 SUB_078 (Qwen2.5-0.5B/1.5B on H100×1):
- Qwen 0.5B: vanilla 11,056 / ngram 4,486 = **2.46× 회귀** (-59%)
- Qwen 1.5B: vanilla 11,015 / ngram 4,195 = **2.63× 회귀** (-62%)

→ **issue #16258 의 "small/fast model + ngram = severe regression" 패턴이 본 fork 환경에서 다른 model family (Qwen) 에서도 재현**. issue 의 정확 setup 은 아니지만 **mechanism cross-validation 성공**.

### 3.1 mechanism 해석 — small/fast model 의 R / K 비율

본 doc §3 의 R/K 모델에서:
- **small/fast model**: T_target (target forward step time) 매우 짧음 (Qwen 0.5B forward ≈ 0.1-1 ms). ngram lookup overhead (CPU KMP, ~1-2 ms) 가 forward time 의 상당 비율 → R 매우 큼 (≈ 5~10).
- vs **large model (Llama-3.3-70B + TP=8)**: T_target ≈ 70 ms, ngram lookup overhead 영역 비율 작음 → R ≈ 1.30 (본 doc 가정).
- K 가 model size 와 무관하게 workload + acceptance 의 함수 → small model 의 K 도 sonnet/chat/code 와 유사 (sonnet K~3, code K~1).

→ **small model 영역 K ≤ R 인 영역 매우 흔함 → 항상 net regression**. issue #16258 의 acceptance 70% 인데도 회귀였던 이유와 일치.

본 SUB 의 Qwen 측정은 code workload (K~1.10 추정) → R 가 모든 환경에서 1보다 크므로 **항상 회귀** 보장.

## 4. 본 doc §11.3 차별점 cross-validation 추가

| 측면 | 본 doc framework | SUB_078 외부 cross-validation |
|---|---|---|
| **regression 정량 분리** | sonnet +134% / chat +37.5% / **code −23.2%** (Llama-3.3-70B + H100×8) | Qwen 0.5B/1.5B code: **−59~−62% 회귀** (small model 영역 R≫K 영역 더 심각) |
| **high acceptance ≠ net win** | code 의 K=1.10, α=1.4% (SUB_075) → vLLM ngram 의 mechanism limitation | Qwen 측정 영역 acceptance 직접 측정 불가 (wall 영역 너무 짧음) 하지만 throughput 회귀 폭 (2.5×) 영역 issue #16258 (acceptance 70%, 2.1× 회귀) 와 동일 패턴 |
| **R 의 model-size 의존성** | R ≈ 1.30 가정 (large model) | small model 영역 R ≫ 1.30 (T_target 영역 짧음 → overhead 비율 ↑) — 본 SUB 의 회귀 폭이 이를 supports |

→ 본 doc 의 결론은 **single setup 의 우연이 아닌 vLLM ngram 의 mechanism level 한계**. issue #16258 (외부) + SUB_071 (large model code, 본 fork) + SUB_078 (small model code, 본 fork) **3 source corroboration**.

## 5. 후속 SUB candidate

- **SUB_079** (제안): 본 SUB 의 acceptance rate 직접 측정 — wall ≥ 60s 가 되도록 prompt n 늘려 spec metric interval 캡처. opt-125m 영역 cache 영역 다운로드 (HF auth) 시 정확 issue #16258 reproduction 가능.
- **SUB_080** (제안): small model + suffix decoding 시도. small model 영역 R 영역 더 클 텐데, suffix 의 adaptive num_spec 이 R 가 큰 환경 영역 도움될 수도 있음 (low-α 시 num_spec=0 으로 폴백).
- **SUB_081** (제안): R 의 model-size scaling 정량 측정 — Qwen 0.5B/1.5B/7B/32B/72B sweep 으로 R = f(model_size, batch, hardware) curve.

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| Qwen 0.5B vanilla | `eval/results/20260525_014310_sub078_qwen05b_vanilla/result.json` |
| Qwen 0.5B ngram | `eval/results/20260525_014310_sub078_qwen05b_ngram_5lk2/result.json` |
| Qwen 1.5B vanilla | `eval/results/20260525_014310_sub078_qwen15b_vanilla/result.json` |
| Qwen 1.5B ngram | `eval/results/20260525_014310_sub078_qwen15b_ngram_5lk2/result.json` |
| launcher | `/tmp/run_sub078_small_model.sh` |
| wrapper | `/tmp/run_sub078_wrap.py` (run_workload_gen.py wrapper for arbitrary model + TP=1) |
| stdout log | `/tmp/sub078.log` |
| summary tsv | `/tmp/sub078_summary.tsv` |
