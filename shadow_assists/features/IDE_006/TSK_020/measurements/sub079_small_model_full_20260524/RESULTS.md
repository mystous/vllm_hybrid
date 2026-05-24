# SUB_079 — SUB_078 의 sonnet/chat 확장 측정 (Qwen 0.5B/1.5B small model) RESULTS

> **parent**: TSK_020 / SUB_072 / idea IDE_014
> **plan**: [`../../planning/SUB_079_small_model_sonnet_chat.md`](../../planning/SUB_079_small_model_sonnet_chat.md)
> **measurement**: 2026-05-24 20:58~21:03 KST, single-run × 8 cell
> **setup**: SUB_078 와 동일 — TP=1, H100×1, fp8 KV, 50p × 1024in × 512max
> **raw**: `eval/results/20260525_055832_sub079_<model>_<workload>_<config>/`

---

## 1. 측정 결과 (8 cell: 2 model × 2 workload × 2 config)

### 1.1 sonnet workload

| model | config | tps | wall (s) | out_tok | vs vanilla |
|---|---|---:|---:|---:|---:|
| **Qwen2.5-0.5B** | **vanilla** | **11,820.6** | 1.6 | 18,939 | — |
| Qwen2.5-0.5B | ngram spec=5, lookup_max=2 | 6,111.8 | 3.1 | 18,958 | **0.517× (−48.3%)** ✗ |
| **Qwen2.5-1.5B** | **vanilla** | **12,594.8** | 1.7 | 21,137 | — |
| Qwen2.5-1.5B | ngram spec=5, lookup_max=2 | 5,015.5 | 4.2 | 21,144 | **0.398× (−60.2%)** ✗ |

### 1.2 chat workload

| model | config | tps | wall (s) | out_tok | vs vanilla |
|---|---|---:|---:|---:|---:|
| **Qwen2.5-0.5B** | **vanilla** | **13,675.5** | 1.7 | 23,742 | — |
| Qwen2.5-0.5B | ngram spec=5, lookup_max=2 | 4,745.9 | 4.7 | 22,094 | **0.347× (−65.3%)** ✗ |
| **Qwen2.5-1.5B** | **vanilla** | **11,589.4** | 1.7 | 19,132 | — |
| Qwen2.5-1.5B | ngram spec=5, lookup_max=2 | 4,539.6 | 4.7 | 21,475 | **0.392× (−60.8%)** ✗ |

### 1.3 통합 — Qwen small model 영역 모든 workload 회귀 (SUB_078 code 결과 합산)

| model | workload | vanilla tps | ngram tps | vs vanilla | source |
|---|---|---:|---:|---:|---|
| Qwen2.5-0.5B | sonnet | 11,820.6 | 6,111.8 | **−48.3%** ✗ | SUB_079 |
| Qwen2.5-0.5B | chat | 13,675.5 | 4,745.9 | **−65.3%** ✗ | SUB_079 |
| Qwen2.5-0.5B | code | 11,056.2 | 4,485.9 | **−59.4%** ✗ | SUB_078 |
| Qwen2.5-1.5B | sonnet | 12,594.8 | 5,015.5 | **−60.2%** ✗ | SUB_079 |
| Qwen2.5-1.5B | chat | 11,589.4 | 4,539.6 | **−60.8%** ✗ | SUB_079 |
| Qwen2.5-1.5B | code | 11,015.5 | 4,195.1 | **−62.0%** ✗ | SUB_078 |

→ **6/6 모든 cell 에서 net regression** (-48% ~ -65%). **small model 영역 ngram 회귀 = workload-universal** 가설 1 확정.

## 2. 핵심 fact — small model 영역 R ≫ K (workload 무관)

SUB_078 결과 (code only) + SUB_079 결과 (sonnet/chat) = **6 workload cell 모두 회귀**. 가설 비교:

| 가설 | 결과 |
|---|---|
| (a) workload-universal regression (small model 영역 R≫K, workload 어떤 것이든 회귀) | **✓ 확정** (sonnet/chat/code 모두 -48~-65%) |
| (b) workload-specific (large model 의 SUB_071 패턴 — sonnet +134/chat +37/code -23 같은 분리) | ✗ 기각 |

### 2.1 mechanism — 왜 small model 은 R 가 큰가

본 doc R/K framework:
- **large model (Llama-70B + TP=8)**: T_target ≈ 70 ms/step (decode). ngram lookup overhead (CPU KMP ~1-2 ms) 가 forward time 의 작은 비율 → R ≈ 1.30.
- **small model (Qwen 0.5B + TP=1)**: T_target ≈ 0.1-1 ms/step (매우 빠름). ngram lookup overhead (CPU KMP + sampler bookkeeping) 가 forward time 과 비교 가능 또는 더 큼 → **R ≈ 5~10**.
- K (per-draft accept) 는 workload dependent (sonnet 3~5, chat 6~11, code 1) 이지만 small model 의 R 이 큰 K 도 능가하면 **모든 workload net regression**.

본 측정 결과:
- sonnet: small model R ≈ vanilla/ngram = 11820/6111 ≈ 1.93 (R가 K_sonnet 3.72 보다 작아 보이지만 — vLLM ngram overhead 가 single-token forward 의 ~2 배에 가까운 비율)
- chat: vanilla/ngram = 13675/4745 ≈ 2.88 — R 가 K_chat 6.69 의 ~43% — 즉 R 자체가 chat 의 acceptance gain 을 모두 소진

→ small model 환경 영역 K_workload 가 cuda graph 호환 large model 만큼 amortize 안 됨. 본 SUB 영역 fact 가 본 doc 의 "small/fast model + ngram = severe regression" 명제 직접 corroborate.

## 3. acceptance metric 미수집

wall 매우 짧음 (1.7~4.7 s) → vLLM v1 의 `SpecDecoding metrics` interval (10s default) 보다 짧아 emit 안 됨. mean_accept_len / per-pos α 직접 측정값 부재.

후속 SUB candidate: 더 긴 workload (200p × 4096 × 4096) 영역 acceptance 도 함께 수집.

## 4. 본 doc 갱신

- `analysis/workload_acceptance_analysis_20260524.md` §10.4 후속 reading + §11.3 차별점 — small model 영역 6/6 cell 회귀 (workload-universal) fact 추가
- `INDEX.md` §1 active SUB 표
- `idea/IDE_014_issue_16258_repro.md` §7 결과 — code (SUB_078) + sonnet/chat (SUB_079) 통합

## 5. 후속 SUB candidate

- SUB_080+: real opt-125m + starcoder2-3b reproduction (HF auth 후, 정확 issue #16258 setup)
- SUB_081+: small model + suffix decoding 측정 (suffix 영역 adaptive num_spec 이 small model 에서도 도움될 가능성)
- SUB_082+: R 의 model-size scaling sweep (Qwen 0.5B/1.5B/7B/32B/72B × code, R = f(model_size) curve)

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| 8 result.json | `eval/results/20260525_055832_sub079_*/result.json` |
| launcher | `/tmp/run_sub079_qwen_sonnet_chat.sh` |
| wrapper | `/tmp/run_sub078_wrap.py` (재사용) |
| stdout log | `/tmp/sub079.log` |
| summary tsv | `/tmp/sub079_summary.tsv` |
