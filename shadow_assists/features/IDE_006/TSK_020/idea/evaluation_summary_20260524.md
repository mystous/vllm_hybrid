# TSK_020 idea backlog 평가 종합 (2026-05-24 KST)

> **scope**: 2026-05-24 일 작업 (최근 6 시간 내) — SUB_073 ~ SUB_079, idea IDE_009 ~ IDE_014
> **parent**: [`README.md`](README.md) (idea backlog), [`../planning/SUB_072_idea_backlog.md`](../planning/SUB_072_idea_backlog.md)
> **목적**: 본 idea 들이 net "성능 개선" 또는 "정량 fact" 영역 어떤 contribution 영역 했는지 정직 평가

---

## 1. 작업 scope

| 영역 | 내용 |
|---|---|
| 기간 | 2026-05-24 일 (~6 시간) |
| SUB 신설 | SUB_073, SUB_074, SUB_075, SUB_076, SUB_077, SUB_078, SUB_079 (7 SUB) |
| idea 신설 | IDE_009 ~ IDE_014 (6 idea, prefix I001~006 → IDE_009~014 정정 포함) |
| 측정 cell 총합 | 25 cell (SUB_074 3 + SUB_075 3 + SUB_076 classifier 3 + SUB_078 4 + SUB_079 8 + smoke 2 + 등) |
| doc 갱신 | 분석 doc + Best + INDEX + id_registry + RESULTS 6개 + idea 6개 |

---

## 2. vanilla baseline (모든 비교의 기준)

**Llama-3.3-70B + TP=8 + H100×8 + 500p × 8192in × 8192max**:
- sonnet vanilla: **4,679.8 tps**
- chat vanilla: **2,186.0 tps**
- code vanilla: **6,964.5 tps**

**Qwen2.5 + TP=1 + H100×1 + 50p × 1024in × 512max** (small model regime):
- Qwen 0.5B sonnet vanilla: 11,820.6 / chat: 13,675.5 / code: 11,056.2 tps
- Qwen 1.5B sonnet vanilla: 12,594.8 / chat: 11,589.4 / code: 11,015.5 tps

---

## 3. IDE 별 성과 평가

### 3.1 IDE_009 — vanilla framing 정정 (SUB_073)

| 측면 | 결과 |
|---|---|
| 측정 cell | 0 (doc only) |
| sonnet | — | chat | — | code | — |
| 새 발견 | "+134% vs vanilla" 가 실제로는 vLLM built-in +130.3% + fork patch +1.65% 의 합산. **본 fork ~6 줄 patch 의 단독 contribution = +1.65%** (vanilla → SUB_044 영역 +130.3% 는 vLLM 영역 코드 변경 0) |
| 갱신 doc | analysis/Best/INDEX/RESULTS/id_registry 모두 3-단계 breakdown 명시 |
| **평가** | **✅ 정직성 contribution** — over-claim 제거. 성능 측정 자체는 N/A |

### 3.2 IDE_010 ⭐ — SuffixDecoding 측정 (SUB_074) — ★ 가장 큰 성과

**측정 결과 (vanilla 대비, enforce_eager 모드 caveat: ngram (cuda graph) vs suffix (eager) 비교 시 ~25% eager penalty 있음)**:

| workload | vanilla | suffix tps (eager) | vs vanilla | suffix K (peak) | ngram K (SUB_075) | K 향상 |
|---|---:|---:|---:|---:|---:|---:|
| sonnet | 4,679.8 | 8,236.0 | **+76.0%** ✅ | 4.42 | 3.72 | 1.19× |
| chat | 2,186.0 | 2,369.7 | **+8.4%** ✅ | 11.58 | 6.69 | 1.73× |
| **code** ⭐ | 6,964.5 | **7,093.5** | **+1.85%** ✅ | **7.67** | **1.10** | **★ 7×** ⭐ |

**ngram vs suffix 직접 비교**:

| workload | ngram tps (cuda graph) | suffix tps (eager) | suffix/ngram | 결론 |
|---|---:|---:|---:|---|
| sonnet | 10,909 | 8,236 | 0.755 | eager penalty 가 suffix mechanism gain 능가 |
| chat | 2,972 | 2,370 | 0.797 | eager penalty |
| **code** | **5,362** | **7,094** | **1.323 (+32%)** ⭐ | **suffix 가 code 회귀 mitigation** |

**cuda graph 호환 시 추정** (eager penalty ~25% 정상화):

| workload | suffix (eager 실측) | suffix (cuda graph 추정) | vs ngram (cuda graph) |
|---|---:|---:|---:|
| sonnet | 8,236 | ~10,560 | ≈ 동등 |
| chat | 2,370 | ~3,040 | +2% |
| **code** | **7,094** | **~9,094** ⭐ | **+70%** ⭐ |

| 측면 | 평가 |
|---|---|
| 새 발견 | **code workload 의 ngram 회귀 (-23.2%) 가 suffix 로 net positive (+1.85%) 로 전환**. K 7× 향상. cuda graph 호환 시 +60-70% 가능성 |
| **성과 크기** | **★★ 매우 큼** — code workload 회귀의 해결 path 발견. production 적용 가치 크다 (단 cuda graph 호환 patch 후) |

### 3.3 IDE_011 ⭐ — Acceptance rate 직접 측정 (SUB_075) — ★ framework 성과

**측정 결과 (3 workload × ngram_spec7_cap8, vanilla 대비 + spec metric)**:

| workload | tps | vs vanilla | mean_accept_len (per-draft K) | per-pos α | accepted/drafted |
|---|---:|---:|---:|---:|---:|
| sonnet | 10,909 | **+133.1%** | 3.72 | 38.8% | 5,342 / 13,769 |
| chat | 2,972.5 | **+36.0%** | **6.69** ⭐ | **81.2%** ⭐ | 21,451 / 26,404 |
| code | 5,362.5 | **−23.0%** ✗ | 1.10 | 1.4% | 143 / 9,954 |

**TL;DR 예측 vs 실측**:

| workload | 예측 K | 실측 K | 예측 α | 실측 α | gap |
|---|---:|---:|---:|---:|---|
| sonnet | ≈ 5.0 | 3.72 | ≈ 60% | 38.8% | 예측 과대 |
| **chat** | ≈ 1.8 | **6.69** | ≈ 12% | **81.2%** | **rank reversal — 예측 과소** |
| code | ≈ 1.0 | 1.10 | ≈ 0% | 1.4% | 정확 |

**R/K framework fit (R=1.30 가정)**:

| workload | wall_ratio | K_doc (모델) | K_spec (실측) | s (coverage fit) |
|---|---:|---:|---:|---:|
| sonnet | 0.419 | 3.10 | 3.72 | **21.1%** |
| chat | 0.749 | 1.74 | 6.69 | **6.0%** |
| code | 1.292 | 1.01 | 1.10 | ~0% |

| 측면 | 평가 |
|---|---|
| 새 발견 | (1) chat per-draft α=81% > sonnet α=39% (rank reversal). (2) coverage s 가 2nd 결정 변수 (chat K 높지만 coverage 낮아 net throughput 작음). (3) linear `1+7α` 가 mean_accept_len 와 정합 (Leviathan i.i.d. 위반 — strong position correlation) |
| **성과 크기** | **★ framework contribution** — R/K 모델 first empirical validation, 분석 doc TL;DR 의 예측 오류 1건 정정. throughput 자체 개선은 없음 (SUB_047 과 동일 config) |

### 3.4 IDE_012 — Workload classifier PoC (SUB_076)

**측정 결과 (1,500 prompt classification, 3 workload × 500)**:

| true \ pred | sonnet | chat | code | accuracy |
|---|---:|---:|---:|---:|
| sonnet | 500 | 0 | 0 | 1.000 |
| chat | 0 | 500 | 0 | 1.000 |
| code | 0 | 0 | 500 | 1.000 |

→ **macro accuracy 1.000** (sonnet/chat/code 모두 perfect)

| 측면 | 평가 |
|---|---|
| 새 발견 | regex feature classifier 가 본 환경 builder set 영역 100% 분류 |
| **성과 크기** | **◐ 조건부** — 본 환경 builder 영역 trivial. real production traffic (ShareGPT / LMSYS-chat) 영역 예상 0.85~0.95. routing/serving 통합 안 됨 (GPU 제약, dual instance 불가) |

### 3.5 IDE_013 — vLLM upstream PR draft (SUB_077)

| 측면 | 결과 |
|---|---|
| 측정 cell | 0 (외부 PR 활동) |
| sonnet | — | chat | — | code | — |
| 새 발견 | (1) duplicate check 통과 (0 open PR matching ngram thread cap). (2) IDE_009 정정된 contribution framing 으로 honest PR description draft 작성 (over-claim 없음, +1.65% vs upstream default 명시) |
| **성과 크기** | **◐ draft 완료, human submit 대기** — actual upstream merge 까지는 사용자 후속 action 필요. potential value 있음 (PR #24986 의 TODO 해소) |

### 3.6 IDE_014 ⭐ — Issue #16258 + small model 확장 (SUB_078 + SUB_079) — ★ universal regression 확정

**측정 결과 (vanilla 대비, 6/6 cell)**:

| model | sonnet | chat | code |
|---|---|---|---|
| **Qwen2.5-0.5B** | 11,821 → 6,112 = **−48.3%** ✗ | 13,675 → 4,746 = **−65.3%** ✗ | 11,056 → 4,486 = **−59.4%** ✗ |
| **Qwen2.5-1.5B** | 12,595 → 5,016 = **−60.2%** ✗ | 11,589 → 4,540 = **−60.8%** ✗ | 11,016 → 4,195 = **−62.0%** ✗ |

**issue #16258 (외부) vs 본 SUB cross-validation**:

| source | hardware | model | workload | regression |
|---|---|---|---|---:|
| issue #16258 (dtransposed) | 2× L4 | opt-125m | (code-like) | **2.1×** |
| SUB_078 본 fork | H100×1 | Qwen 0.5B | code | **2.46×** |
| SUB_079 본 fork | H100×1 | Qwen 0.5B | sonnet | **1.93×** |
| SUB_079 본 fork | H100×1 | Qwen 0.5B | chat | **2.88×** (worst) |
| SUB_078 본 fork | H100×1 | Qwen 1.5B | code | **2.63×** |
| SUB_079 본 fork | H100×1 | Qwen 1.5B | sonnet | **2.51×** |
| SUB_079 본 fork | H100×1 | Qwen 1.5B | chat | **2.55×** |

| 측면 | 평가 |
|---|---|
| 새 발견 | **small model + ngram = 모든 workload 영역 −48~−65% universal regression** (6/6 cell). R 의 model-size 의존성 framework 확장 — large model R≈1.30, small model R≈5~10 |
| **성과 크기** | **★★ negative fact 의 정량 확정** — production 권장: small model 영역 ngram 사용 금지 (workload 무관). 외부 issue + 본 fork large model code + 본 fork small model 3 workload **3 source corroboration** |

---

## 4. 종합 — 성능 개선 있나? IDE 별 직접 답

| IDE | 직접 throughput 개선? | 측정 / framework / fact contribution? | 평가 |
|---|---|---|---|
| IDE_009 | ✗ 없음 (doc only) | ✅ over-claim 정정 (정직성) | **정직성 ✅** |
| **IDE_010** ⭐ | ✅ **code +32% vs ngram** (vs vanilla +1.85%, eager 모드 caveat) | ✅ code 회귀 mitigation path 발견 | **★★ 큰 성과** |
| IDE_011 | ✗ 없음 (SUB_047 과 동일 config) | ✅ R/K framework empirical validation + chat α surprise | **★ framework 성과** |
| IDE_012 | ✗ 없음 (classifier 만) | ✅ accuracy 1.000 (조건부, builder 영역 trivial) | **◐ 조건부** |
| IDE_013 | ✗ 없음 (PR draft) | ◐ draft 완료 | **◐ submit 대기** |
| **IDE_014** ⭐ | ✗ 회귀 fact 만 (모두 −48~−65%) | ✅ **small model universal regression 확정** | **★★ negative fact 확정** |

## 5. production 의사결정 권장 (본 IDE 결과 종합)

| 시나리오 | 권장 |
|---|---|
| **Llama-70B + sonnet** | ✅ ngram cap=8 (SUB_047) — +134% throughput |
| **Llama-70B + chat** | ✅ ngram cap=8 — +37% |
| **Llama-70B + code** | ✅ **suffix decoding (cuda graph 호환 patch 후) — +60-70% 가능**. 또는 spec OFF |
| **small model (1B 이하) + 모든 workload** | ✗ **ngram 사용 금지** (모두 −48~−65% 회귀) |
| **mixed traffic** | ✅ workload-aware gating (IDE_012 classifier) — code 검출 시 suffix 또는 spec OFF |

## 6. 본 6 시간 작업의 진짜 contribution 요약

1. **★ code workload 회귀 mitigation path 발견** (IDE_010) — production-applicable
2. **★ small model universal regression 확정** (IDE_014) — production 권장 (ngram OFF)
3. **★ R/K framework empirical validation** (IDE_011) — 분석 doc 정량화
4. **정직성 정정** (IDE_009) — fork 단독 contribution = +1.65% (vLLM built-in 와 분리 명시)
5. **workload-aware gating heuristic + classifier PoC** (IDE_012) — production 권장 lever
6. **upstream PR draft 준비** (IDE_013) — human submit 후 vLLM 영역 contribution 가능

## 7. 후속 SUB candidate

| 우선순위 | SUB candidate | 출처 |
|---|---|---|
| ★★ | suffix cuda graph 호환 patch (arctic_inference 영역 vLLM 1.6 fork 또는 plugin path fix) | IDE_010 §8.7 |
| ★★ | suffix vs ngram fair comparison (양쪽 모두 enforce_eager) | IDE_010 §8.7 |
| ★ | workload-aware routing PoC (vLLM Semantic Router 패턴, 2 instance) | IDE_012 §8.4 |
| ★ | 정확 issue #16258 reproduction (opt-125m / starcoder2-3b HF auth) | IDE_014 §7.7 |
| ★ | small model + suffix decoding 측정 | IDE_014 §7.7 |
| ◐ | R 의 model-size scaling sweep (Qwen 0.5B/1.5B/7B/32B/72B × code) | IDE_014 §7.7 |
| ◐ | real production traffic (ShareGPT 등) 영역 classifier accuracy 재측정 | IDE_012 §8.4 |

---

## 8. raw 자료 — IDE 별 detailed RESULTS link

| IDE | RESULTS doc |
|---|---|
| IDE_009 | (분산 — analysis/Best/INDEX/RESULTS sub044+sub047/id_registry) |
| IDE_010 | [`../measurements/sub074_suffix_20260524/RESULTS.md`](../measurements/sub074_suffix_20260524/RESULTS.md) |
| IDE_011 | [`../measurements/sub075_acceptance_20260524/RESULTS.md`](../measurements/sub075_acceptance_20260524/RESULTS.md) |
| IDE_012 | [`../measurements/sub076_classifier_20260524/RESULTS.md`](../measurements/sub076_classifier_20260524/RESULTS.md) |
| IDE_013 | [`../measurements/sub077_pr_draft_20260524/PR_DRAFT.md`](../measurements/sub077_pr_draft_20260524/PR_DRAFT.md) |
| IDE_014 (code) | [`../measurements/sub078_repro_20260524/RESULTS.md`](../measurements/sub078_repro_20260524/RESULTS.md) |
| IDE_014 (sonnet/chat) | [`../measurements/sub079_small_model_full_20260524/RESULTS.md`](../measurements/sub079_small_model_full_20260524/RESULTS.md) |
