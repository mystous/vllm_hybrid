# IDE_011 — acceptance rate 직접 측정 (R/K 분리)

> **parent backlog**: [`README.md`](README.md) (TSK_020 / SUB_072)
> **자식 SUB**: [`SUB_075`](../planning/SUB_075_acceptance_rate_direct.md)
> **발견**: 2026-05-24, analysis doc §7 (검증 방안) + §10.4 follow-up
> **priority**: ★★ (1 시간 내, framework accuracy)
> **status**: ✅ **완료 ⭐** (2026-05-24) — R/K framework first empirical validation, chat α=81% surprise

## 1. fact

본 분석 doc §3 의 R/K 모델은 **wall_ratio 만으로 K 역산** (R = 1.30 가정). 정확한 K 와 R 의 분리는 vLLM v1 의 spec decode metric 으로 직접 측정 가능:

- vLLM v1 PR #15151 (markmc, 2025-04-01) 가 도입한 metric:
  - `vllm:spec_decode_num_drafts`
  - `vllm:spec_decode_num_draft_tokens`
  - `vllm:spec_decode_num_accepted_tokens`
  - `vllm:spec_decode_num_accepted_tokens_per_pos`
- enable: `LLM(disable_log_stats=False, ...)` (현 wrapper 는 `True` 로 설정)
- vLLM 영역 정의: `mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)`

→ α = `num_accepted_tokens / num_draft_tokens` 직접 측정 → K_exact (Leviathan closed-form 또는 linear) 산출 → 본 doc R 정확값 분리 (R = wall_ratio × K).

## 2. 측정 계획

### 2.1 변경

- wrapper `/tmp/run_workload_gen.py` 의 LLM kwargs:
  - `disable_log_stats=True` → `False`
  - 또는 별도 wrapper `/tmp/run_workload_gen_with_metrics.py` 신설

### 2.2 cell (3-workload × spec ON only, vanilla 는 spec metric 없음)

| workload | config | 측정값 |
|---|---|---|
| sonnet | spec7+cap8 (SUB_047 best) | acceptance_rate, mean_acceptance_length |
| chat | spec7+cap8 | 같음 |
| code | spec7+cap8 | 같음 |

→ 3 cell 만. 기존 SUB_047 / SUB_071 결과의 wall 은 그대로, **metric 만 추가 수집**.

### 2.3 effort

- wrapper 변경: 5 분
- 3 cell 재측정 (각 5-10 min): 15-30 min
- log 파싱 (mean_acceptance_length 추출) + 본 doc §3.3 표 갱신: 15 분
- **총 effort: 30-50 분**

## 3. 진행 시 신설 SUB (candidate)

- **SUB_074** (제안 번호): acceptance rate 직접 측정 → R/K 분리 → 분석 doc framework accuracy 보강.
- 측정 자체는 가벼우므로 SUB 신설 없이 idea md 안에서 직접 처리하는 옵션도 가능.

## 4. 확인 / 업데이트 필요 doc

| 파일 | 갱신 위치 |
|---|---|
| `analysis/workload_acceptance_analysis_20260524.md` | §3.3 표 (추정 K → 측정 α 기반 K_exact) · §3.4 Leviathan closed-form 영역 직접값 적용 · §1 TL;DR (acceptance 컬럼 실측치로) |
| `INDEX.md` | §0 best 표에 sonnet acceptance rate (실측값) 한 줄 |
| 새 또는 기존 measurements/sub074_<TS>/RESULTS.md | 측정 결과 |

## 5. 가설 검증

I001 의 contribution framing 정정 + 본 I003 의 R/K 직접 측정 두 가지가 끝나면, 분석 doc 의 §3 정량 모델이 **fully empirical** 화 됨 — R 추정 (1.30) 가 실제와 얼마나 다른지 확정, K_linear vs K_exact 의 framework 차이도 실측값으로 닫힘.

예측:
- sonnet acceptance ~50-60% (vLLM literature 의 ngram sonnet 영역 추정값) → α = 0.5~0.6 → linear K = 4.5~5.2 / Leviathan K_exact = 1.99~2.46. wall_ratio 0.419 와 정합 → R_actual = 0.83~1.08. **본 doc R = 1.30 추정이 실제보다 약간 과대평가** 일 가능성.
- chat acceptance ~10-15% → α = 0.10~0.15 → K_exact ≈ 1.10~1.16 / wall_ratio 0.749 → R_actual ≈ 0.82~0.87. **chat 의 R 가 sonnet 보다 작을 수도** (응답 짧아 step overhead 가 amortize 안 되므로).
- code acceptance ~0-3% → α ≈ 0~0.03 → K_exact ≈ 1.00~1.03 / wall_ratio 1.292 → R_actual ≈ 1.29~1.33. **code 의 R 가 본 doc 추정과 거의 일치** (acceptance 가 0 에 가까우면 step overhead 가 그대로 노출).

## 6. risk / caveat

- vLLM v1 의 spec metric 출력 format 이 본 fork repo 의 vLLM 영역 정확히 어떻게 emit 되는지 — `vllm/v1/spec_decode/metrics.py` 의 `SpecDecodingLogging.log()` 가 어떤 trigger 로 stdout/log 에 print 되는지 확인 필요.
- batch 영역 aggregation — 본 metric 이 매 step 누적인지, 끝에 한 번 emit 인지 확인.

## 7. 결과 (SUB_075, 2026-05-24)

### 7.1 측정 결과 (3 workload × ngram_spec7_cap8, vanilla 대비 + spec metric)

| workload | tps | vs vanilla | mean_accept_len (per-draft K) | per-pos α | accepted (Σ) | drafted (Σ) |
|---|---:|---:|---:|---:|---:|---:|
| **sonnet** | 10,909.0 | **+133.1%** | **3.72** | **38.8%** | 5,342 | 13,769 |
| **chat** | 2,972.5 | **+36.0%** | **6.69** ⭐ | **81.2%** ⭐ | 21,451 | 26,404 |
| **code** | 5,362.5 | **−23.0%** ✗ | **1.10** | **1.4%** | 143 | 9,954 |

### 7.2 본 doc 예측 vs 실측

| workload | TL;DR 예측 K | 실측 K | TL;DR 예측 α | 실측 α | gap |
|---|---:|---:|---:|---:|---|
| sonnet | ≈ 5.0 | 3.72 | ≈ 60% | 38.8% | 예측 과대 (linear K @ literature α 차용) |
| **chat** | ≈ 1.8 | **6.69** | ≈ 12% | **81.2%** | **예측 완전 과소 (rank reversal)** |
| code | ≈ 1.0 | 1.10 | ≈ 0% | 1.4% | 정확 일치 |

### 7.3 R/K framework fit (R=1.30 가정)

| workload | wall_ratio | K_doc (model 평균) | K_spec (per-draft 실측) | s (coverage fit, % of step) |
|---|---:|---:|---:|---:|
| sonnet | 0.419 | 3.10 | 3.72 | **21.1%** (높음) |
| chat | 0.749 | 1.74 | 6.69 | **6.0%** (낮음) |
| code | 1.292 | 1.01 | 1.10 | ~0% |

→ chat 의 per-draft α 가 sonnet 보다 훨씬 높은데도 throughput 향상은 sonnet 가 큼 — **spec coverage s (draft 시도 빈도)** 가 second decisive factor.

### 7.4 Leviathan closed-form vs linear vs 실측 (γ=7)

| workload | 실측 α | linear K=1+7α | Leviathan K_exact | 실측 mean_accept_len | 결론 |
|---|---:|---:|---:|---:|---|
| sonnet | 0.388 | 3.72 | 1.63 | 3.72 | linear 일치 ✓ (Leviathan 2.3× 과소) |
| chat | 0.812 | 6.68 | 4.31 | 6.69 | linear 일치 ✓ (Leviathan 1.55× 과소) |
| code | 0.014 | 1.10 | 1.01 | 1.10 | 둘 다 일치 |

→ **ngram drafter 의 mean_accept_len 가 linear approximation 과 정합 (Leviathan i.i.d. 가정 위반)**. ngram match 가 발견된 위치 = conformant position → position 간 strong positive correlation → first-token accept 이후 chain 끝까지 accept 될 확률 ↑.

### 7.5 새 발견

1. **chat acceptance > sonnet (rank reversal)** — sonnet excerpt 가 chat prompt 에 포함되어 spec match 시 acceptance 매우 높음. 단 응답 짧고 coverage 낮아 net throughput 낮음.
2. **coverage s 가 두 번째 결정 변수** — 본 doc §3 모델 (R/K) 에 s 추가 필요. workload-aware gating 시 per-draft α 만으로 부족.
3. **linear `1+7α` 가 vLLM ngram 의 K 와 정합** — Leviathan closed-form 보다 우월 (position 간 correlation 영역).

### 7.6 본 doc 갱신 (완료)

- `analysis/workload_acceptance_analysis_20260524.md` §1.1 TL;DR (실측 K/α 컬럼) · §3.3 K 역산 표 (실측 + coverage s) · §3.4 Leviathan closed-form 비교
