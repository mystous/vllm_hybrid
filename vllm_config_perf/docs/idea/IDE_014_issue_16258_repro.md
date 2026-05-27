# IDE_014 — vLLM Issue #16258 reproduction (small model cross-validation)

> **parent backlog**: [`README.md`](README.md) (TSK_020 / SUB_072)
> **자식 SUB**: [`SUB_078`](../planning/SUB_078_issue_16258_repro.md) (code only), [`SUB_079`](../planning/SUB_079_small_model_sonnet_chat.md) (sonnet/chat 확장)
> **발견**: 2026-05-24, analysis doc §10.4 후속 reading + §10 R35
> **priority**: ◐ (exploratory) → ★ (small model R≫K universal regression 발견 후 priority 승급)
> **status**: ✅ **완료 ⭐** (2026-05-24) — sonnet/chat/code 모든 workload 영역 small model + ngram = -48~-65% universal regression 확정

## 1. fact

vLLM Issue #16258 (dtransposed, 2025-04-08, [vllm-project/vllm#16258](https://github.com/vllm-project/vllm/issues/16258)):
- Hardware: 2× L4
- Models: opt-125m, starcoder2-3b
- Config: `num_speculative_tokens=5, ngram_prompt_lookup_max=2`
- 결과: ngram **on** throughput 238 tok/s + acceptance 70% / ngram **off** 504.8 tok/s → **2.1× 회귀**
- 결론: "regardless of the configuration, the inference is Pareto worse than the inference without the n-gram model"
- 상태: stale close (resolution 없이)

본 fork repo 의 sonnet/chat/code 측정 (SUB_047 / SUB_071) 와 더불어:
- **본 환경에서 reproduction** = 2-model + 다른 hardware 영역 본 doc 의 "high acceptance ≠ net win" 명제 cross-validation.
- 본 doc §11.3 의 "regression 정량 분리" contribution 을 두 번째 setup 에서 corroborate.

## 2. reproduction 계획

### 2.1 cell

| model | hardware | config | 측정 |
|---|---|---|---|
| opt-125m | H100×1 (single GPU, vLLM TP=1) | vanilla (spec OFF) | tps |
| opt-125m | 같음 | ngram (num_spec=5, lookup_max=2) | tps + acceptance |
| starcoder2-3b | H100×1 | vanilla | tps |
| starcoder2-3b | 같음 | ngram (num_spec=5, lookup_max=2) | tps + acceptance |

→ 4 cell, ~5 min × 4 = 20 min.

### 2.2 workload

issue 본문에서 explicit workload 명시 없음. starcoder2-3b 영역 가장 자연스러운 = HumanEval-like code prompts. opt-125m 영역 일반 chat / wikitext.

본 측정 의 workload = **본 SUB_071 의 code prompt** + **ShareGPT subset** 또는 issue 본문의 link 확인 후 결정.

### 2.3 expected (issue 본문 기반)

- 본 fork 환경 (H100 단일, batch 작음) 영역 throughput 절대치는 다를 것.
- 단 **acceptance high (70%) 인데 net regression** 패턴이 본 환경에서도 재현되면 본 doc 의 "high acceptance ≠ net win, R 가 결정" 명제 확정.

## 3. effort

- vLLM 영역 small model (opt-125m / starcoder2-3b) 로드 시간: 5-10 min × 2
- 4 cell 측정: 20 min
- 결과 분석 + 분석 doc §11.3 갱신: 30 min
- **총 effort: ~1 시간**

## 4. 진행 시 신설 SUB (candidate)

- **SUB_076** (제안 번호): vLLM Issue #16258 reproduction + 본 doc §11.3 차별점 cross-validation.
- 또는 SUB 신설 없이 idea md 안에서 처리 (가벼움).

## 5. 확인 / 업데이트 필요 doc

| 파일 | 갱신 위치 |
|---|---|
| `analysis/workload_acceptance_analysis_20260524.md` | §11.3 차별점 — 본 reproduction 결과 두 번째 setup corroboration 으로 인용 추가 |
| 본 idea md | 측정 결과 |
| 새 measurements/ 또는 idea md 안 §결과 | tps + acceptance 표 |

## 6. risk

- starcoder2-3b 영역 vLLM 영역 정상 로드 여부 — 일부 small model 영역 vLLM 영역 호환성 issue 있을 수 있음.
- single GPU + batch 작음 영역 본 환경 (TP=8 + batch 256) 와 결과 비교 시 caveat 필요 (regime 차이).

## 7. 결과 (SUB_078 + SUB_079, 2026-05-24)

### 7.1 환경 caveat — model substitution

issue #16258 의 원 setup: opt-125m + starcoder2-3b on 2× L4. 본 env 영역 `HF_HUB_OFFLINE=1` + opt-125m/starcoder2-3b cache 부재 → 다운로드 차단.

**substitution**: Qwen2.5-0.5B + Qwen2.5-1.5B (본 env cached, 유사 scale). 정확 issue 영역 reproduction 아니지만 *small model regime cross-validation* 영역 valid.

### 7.2 측정 결과 (vanilla 대비, 모든 workload)

| model | workload | vanilla tps | ngram tps (spec=5, lookup=2) | vs vanilla | source |
|---|---|---:|---:|---:|---|
| Qwen2.5-0.5B | sonnet | 11,820.6 | 6,111.8 | **−48.3%** ✗ | SUB_079 |
| Qwen2.5-0.5B | chat | 13,675.5 | 4,745.9 | **−65.3%** ✗ | SUB_079 |
| Qwen2.5-0.5B | **code** | **11,056.2** | **4,485.9** | **−59.4%** ✗ | SUB_078 |
| Qwen2.5-1.5B | sonnet | 12,594.8 | 5,015.5 | **−60.2%** ✗ | SUB_079 |
| Qwen2.5-1.5B | chat | 11,589.4 | 4,539.6 | **−60.8%** ✗ | SUB_079 |
| Qwen2.5-1.5B | **code** | **11,015.5** | **4,195.1** | **−62.0%** ✗ | SUB_078 |

→ **6/6 cell 모두 net regression** (−48% ~ −65%). **small model 영역 ngram 회귀 = workload-universal** 확정 (가설 1).

### 7.3 issue #16258 (외부) vs 본 SUB_078/SUB_079 (내부) cross-validation

| source | hardware | model | workload | regression |
|---|---|---|---|---:|
| issue #16258 (dtransposed) | 2× L4 | opt-125m | (code-like) | **2.1×** |
| 본 SUB_078 | H100×1 | Qwen 0.5B | code | **2.46×** |
| 본 SUB_079 | H100×1 | Qwen 0.5B | sonnet | **1.93×** |
| 본 SUB_079 | H100×1 | Qwen 0.5B | chat | **2.88×** (worst) |
| 본 SUB_078 | H100×1 | Qwen 1.5B | code | **2.63×** |
| 본 SUB_079 | H100×1 | Qwen 1.5B | sonnet | **2.51×** |
| 본 SUB_079 | H100×1 | Qwen 1.5B | chat | **2.55×** |

→ **issue #16258 의 small model + ngram = severe regression 패턴이 본 fork 환경에서 다른 model family (Qwen) + 다른 workload (sonnet/chat) 에서도 재현**. **3 source corroboration** (외부 issue + 본 fork large model code SUB_071 + 본 fork small model 3 workload SUB_078/079).

### 7.4 새 가설 검증 결과

| 가설 | 결과 | 근거 |
|---|---|---|
| (a) workload-universal regression (small model 영역 R≫K, workload 무관) | **✓ 확정** | 6/6 cell 모두 회귀 (-48~-65%) |
| (b) workload-specific (large model 의 SUB_071 패턴) | ✗ 기각 | sonnet 도 -48~-60% (large model 의 +134% 와 정반대) |

### 7.5 mechanism — small model 의 R 가 큰 이유

본 doc R/K framework:
- **large model**: T_target ≈ 70 ms/step, ngram lookup overhead ~1-2 ms → R ≈ 1.30
- **small model**: T_target ≈ 0.1-1 ms/step, ngram lookup overhead 가 forward time 과 비교 가능 → **R ≈ 5~10**
- K (per-draft) 는 workload dependent (sonnet 3-5, chat 6-11, code 1) 이지만 small model 의 R 이 큰 K 도 능가

→ 본 SUB 영역 fact 가 본 doc §3 의 "R 의 model-size 의존성" 추가 framework axis 필요성 입증.

### 7.6 본 doc 갱신 (완료)

- `analysis/workload_acceptance_analysis_20260524.md` §10.4 후속 reading + §11.3 차별점 — small model 6/6 cell 회귀 fact 추가
- `INDEX.md` §1 active SUB 표 — SUB_079 entry 추가
- `id_registry.md` SUB_079 entry 신설, IDE_014 status `완료 ⭐`

### 7.7 후속 SUB candidate

- **SUB_080+** (제안): 정확 issue #16258 reproduction — opt-125m / starcoder2-3b HF auth 후 cache 다운로드
- **SUB_081+** (제안): small model + suffix decoding 측정 — suffix 의 adaptive num_spec 이 small model 의 R 큰 환경에서도 도움될 가능성 (low-α 시 num_spec=0 폴백)
- **SUB_082+** (제안): R 의 model-size scaling sweep — Qwen 0.5B/1.5B/7B/32B/72B × code, R = f(model_size) curve
