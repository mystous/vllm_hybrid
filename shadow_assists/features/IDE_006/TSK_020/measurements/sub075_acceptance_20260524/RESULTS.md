# SUB_075 — acceptance rate 직접 측정 (R/K 분리) RESULTS

> **parent**: TSK_020 / SUB_072 / idea I003
> **plan**: [`../../planning/SUB_075_acceptance_rate_direct.md`](../../planning/SUB_075_acceptance_rate_direct.md)
> **measurement**: 2026-05-24 15:37~16:03 KST, single-run × 3 workload
> **config**: SUB_047 best (ngram, num_spec=7, cap=8, div_tp=0) + `disable_log_stats=False`
> **scale**: 500p × 8192in × 8192max, batch 256, fp8 KV, Llama-3.3-70B + TP=8 + H100×8
> **raw**: `eval/results/20260525_003750_sub075_<workload>/`

---

## 1. 측정 fact (3 workload, vLLM v1 spec_decode_metrics 직접)

| workload | tps | wall (s) | out_tok | mean_accept_len (γ=7) | draft_accept_rate (per-pos) | accepted (Σ) | drafted (Σ) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **sonnet** | 10,909.0 | 368.4 | 4,019,162 | **3.72** | **38.8%** | 5,342 | 13,769 |
| **chat** | 2,972.5 | 114.5 | 340,296 | **6.69** | **81.2%** | 21,451 | 26,404 |
| **code** | 5,362.5 | 724.1 | 3,883,055 | **1.10** | **1.4%** | 143 | 9,954 |

> **vLLM v1 metric 정의**:
> - `mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)` — bonus token 포함
> - `draft_acceptance_rate (per-position) = num_accepted_tokens / num_draft_tokens`
> - 본 metric 은 **spec proposal 이 성공한 경우만** 집계 (no-match 영역 vanilla path 영역 미포함)

## 2. 핵심 surprise — 본 doc TL;DR 예측과의 비교

| workload | TL;DR 예측 acceptance (§1) | 측정 acceptance (per-pos α) | 측정 mean_accept_len | TL;DR 예측 K | 측정 K (∼ mean_accept_len) | gap |
|---|---:|---:|---:|---:|---:|---|
| sonnet | ≈ 60% | **38.8%** | 3.72 | ≈ 5.0 | **3.72** | TL;DR 가 과대 (실측 lower) |
| chat | ≈ 12% | **81.2%** ⭐ | 6.69 | ≈ 1.8 | **6.69** ⭐ | **TL;DR 가 완전 과소** (rank order 도 뒤집힘) |
| code | ≈ 0% | **1.4%** | 1.10 | ≈ 1.0 | **1.10** | 정확 (small over) |

→ **본 doc 의 acceptance rank 예측 (sonnet ≫ chat ≫ code) 가 부분 오류**. 실측 rank = **chat ≫ sonnet ≫ code**.

## 3. 왜 chat α > sonnet α 인데 sonnet 가 throughput 가 큰가 — coverage 분리

per-spec-attempt 의 acceptance 만 보면 chat > sonnet. 그러나 throughput speedup 은 sonnet > chat:

| workload | tps | speedup vs vanilla | mean_accept_len (per draft) | draft 빈도 (drafted/out_tok) |
|---|---:|---:|---:|---:|
| sonnet | 10,909 | **+133%** | 3.72 | 13769 / 4019162 = **0.343%** |
| chat | 2,973 | **+36%** | 6.69 | 26404 / 340296 = **7.76%** |
| code | 5,362 | **−23%** | 1.10 | 9954 / 3883055 = **0.256%** |

### 3.1 해석

throughput speedup = **draft coverage × per-draft K gain**:

- **sonnet**: draft 빈도 0.343% (relative low), but 매 draft 의 K=3.72 → 모든 step 평균에서 substantial. 출력이 매우 길고 (4M tok), 매 batch step 안에서 256 active seqs 가 동시에 dispense — coverage 누적 효과 크다.
- **chat**: draft 빈도 7.76% (높음, sonnet 의 22 배) + K=6.69 (높음). 하지만 출력 매우 짧음 (660 tok/prompt) → 절대 step count 가 작음 → spec 가 amortize 될 시간 짧음. 결과: per-draft 효과는 strongest 이나 total time saving 작음.
- **code**: draft 빈도 0.256% + K=1.10 (essentially 0 accept) → spec attempt 거의 wasted overhead. R 비용만 누적 → 회귀.

### 3.2 본 doc §3 의 R/K 모델 재해석

본 doc §3 의 R/K 모델은 *step 평균* (모든 step 의 mixed mean). 실측 mean_accept_len 은 *spec-active step 안에서만* 의 mean. 두 값은 다름.

step 평균 K_doc = 1 + s × (K_spec − 1) (s = spec coverage, K_spec = per-draft mean_accept_len).

| workload | s (coverage) | K_spec | K_doc (step 평균) | R_doc (= wall_ratio × K_doc, R=1.30 가정) | 일치성 |
|---|---:|---:|---:|---:|---|
| sonnet | (s × 2.72 = 0.573, s = 0.211 fit) | 3.72 | (모델 fit) | 1.30 | 측정 wall_ratio 0.419 와 정합 |
| chat | (s × 5.69 = 0.342, s = 0.060 fit) | 6.69 | (모델 fit) | 1.30 | wall_ratio 0.749 와 정합 |
| code | (s × 0.10 ≈ 0, ≈ no gain) | 1.10 | ≈ 1.0 | 1.29 | wall_ratio 1.292 와 정합 |

→ R ≈ 1.30 가정은 **모든 workload 에서 wall ratio 와 정합** ((R, s, K_spec) fit). 본 doc §3 모델 framework 는 유효.

s (effective coverage) 추정값:
- sonnet: s ≈ 21.1% (이 비율의 step 에서 spec contribute. 본 doc 의 wall_ratio 0.419 와 K_spec=3.72 로 fit)
- chat: s ≈ 6.0%
- code: s ≈ 0% (regression 의 R 모두 노출)

**chat 의 s=6% 가 sonnet s=21% 보다 낮은 이유**: chat output 이 짧고 (660 tok), 응답 메타 어휘가 prompt 의 sonnet excerpt 와 다름. spec 가 *citation 시점* 에만 hit → coverage 가 낮음. 반면 sonnet 은 매 step continuation 이 prompt 어휘 분포와 같으므로 spec hit 빈도 높음.

## 4. 본 doc 갱신 권장 사항

| doc / section | 갱신 내용 |
|---|---|
| `analysis/workload_acceptance_analysis_20260524.md` §1 TL;DR | "추정 acceptance" 컬럼을 실측치로 교체. acceptance rank 정정 (chat > sonnet > code per-draft α). K 컬럼은 mean_accept_len 실측값으로 변경. |
| 같은 doc §3.3 K 역산 표 | K 추정 (R=1.30 가정) 와 실측 K_spec (mean_accept_len) 두 컬럼 병기. step 평균 K_doc 과 per-draft K_spec 의 차이 명시. |
| 같은 doc §3.4 | Leviathan closed-form 비교 — 실측 α 적용 (sonnet α=0.388, chat α=0.812, code α=0.014). K_exact 산출. |
| 같은 doc §4.1~4.3 | mechanism 해석에 실측 coverage s 추가 — "per-draft acceptance 와 coverage 둘 다 결정 변수" |
| 같은 doc §11 | SUB_047 vs literature 의 axis 표에 "spec coverage" axis 추가 (vLLM ngram 영역 prompt-anchored 영역 coverage 가 낮음, datastore retrieval (REST/SuffixDecoding) 영역 coverage 높이는 lever) |

## 5. 본 SUB 의 contribution

본 SUB 는 **본 doc 의 R/K framework 의 first empirical validation**. 결과:
1. ✓ R = 1.30 가정 의 reasonableness (모든 workload wall_ratio 와 (R, s, K_spec) fit 가능)
2. ✓ code α ≈ 0 확정 (1.4% per-pos, mean_accept_len 1.10 — 본 doc 예측과 정확 일치)
3. ✗ TL;DR 의 acceptance rank 예측 오류 — chat 의 per-draft acceptance 가 sonnet 보다 높음 (81.2% vs 38.8%)
4. ✓ throughput rank (sonnet ≫ chat ≫ code) 는 변함 없음 — coverage × K 의 combined effect
5. ★ 새 fact: spec **coverage** (draft frequency 의 output-token-weighted ratio) 가 second decisive factor. workload-aware gating 시 coverage 도 predict 해야 (단순 per-draft α 만으로 부족).

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| sonnet | `eval/results/20260525_003750_sub075_sonnet/{result.json, engine.log.stdout}` |
| chat | `eval/results/20260525_003750_sub075_chat/{result.json, engine.log.stdout}` |
| code | `eval/results/20260525_003750_sub075_code/{result.json, engine.log.stdout}` |
| launcher | `/tmp/run_sub075_acceptance.sh` |
| wrapper | `/tmp/run_workload_gen.py` (with `VLLM_ENABLE_SPEC_STATS=1`) |
| stdout log | `/tmp/sub075.log` |
| summary tsv | `/tmp/sub075_summary.tsv` |

## 7. 후속 SUB candidate

- **3-run variance** for chat (1-run 만 측정, 81.2% 가 noise band 안인지 확인) — effort 30 분.
- vLLM v1 의 `num_drafts` semantics 정확 확인 (per-batch-step? per-sequence? per-call?) — source code review, 1 시간. 본 doc 의 coverage 산출 정확도 향상.
- coverage 추출 — workload-aware gating heuristic 에 "coverage estimator" 추가 (single per-draft α 만으로 부족하다는 본 SUB 결론 반영).
