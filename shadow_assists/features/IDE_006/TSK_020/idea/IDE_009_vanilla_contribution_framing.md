# IDE_009 — vanilla 대비 +134% framing 의 fork-patch / vLLM built-in 분리

> **parent backlog**: [`README.md`](README.md) (TSK_020 / SUB_072)
> **자식 SUB**: [`SUB_073`](../planning/SUB_073_vanilla_framing_correction.md)
> **발견**: 2026-05-24, 사용자 지적 "vanilla 대비 향상이 아니잖아"
> **priority**: ★★★ (정직성 / accuracy)
> **status**: ✅ **완료** (2026-05-24, doc only 정정 — 측정 변경 없음)

## 1. fact

vLLM v1 영역 ngram speculative decoding 은 **`vllm/v1/spec_decode/ngram_proposer.py` 에 built-in** 으로 포함. 본 fork 의 SUB_047 patch (`VLLM_NGRAM_NUM_THREADS_CAP=8` + `VLLM_NGRAM_DIVIDE_BY_TP=0`) 는 vLLM upstream PR #24986 의 disabled threading 만 env 로 enable 한 ~6 줄 변경. 따라서 "+134% vs vanilla" 표현은 vLLM built-in 효과를 본 작업의 contribution 으로 합산해 표시하는 형태.

### 정확한 contribution breakdown (3 단계)

| 단계 | config | source | tps | vs 직전 단계 | vs vanilla 누적 |
|---|---|---|---:|---:|---:|
| (1) **vanilla** | `speculative_config=None` | vLLM upstream (spec OFF) | 4,679.8 | — | — |
| (2) **vLLM built-in spec ON (default cap=1)** | `num_spec=7, prompt_lookup=2/5` | vLLM 영역 코드 변경 0 — feature 활성화만 | **10,778.6** (SUB_044 t3) | **+130.3%** | **+130.3%** |
| (3) **SUB_047 fork patch** | `+ cap=8, div_tp=0` | 본 fork ~6 줄 patch | 10,956.5 (canonical 3-run avg) | **+1.65%** | **+134.12%** |

→ **+134% 중 130 pp 는 vLLM built-in 효과, 1.65 pp 만 본 fork 의 추가 기여**.

raw 근거:
- SUB_044 t3 (cap=1, vLLM default): `eval/results/20260523_005314_sub044_spec_decode/t3_spec7/result.json` — 10,778.6 tps
- SUB_047 t3 canonical 3-run: `eval/results/20260523_100441_sub048_ngram_refinement/t1_baseline/result.json` + `20260523_162456_sub047_t3_verify/run{2,3}_cap8_div0/result.json` — avg 10,956.5

## 2. 본 작업의 정확한 contribution (재정리)

1. **측정·검증**: vLLM ngram spec decoding 이 본 환경 (Llama-3.3-70B, H100×8, TP=8) 에서 어떤 num_spec / workload 조합에서 net positive 인지 systematic sweep + 3-run canonical 확정.
2. **fork patch (+1.65%)**: vLLM upstream PR #24986 의 disabled threading 을 env 로 enable (6 줄, `vllm/v1/spec_decode/ngram_proposer.py:48`).
3. **workload generalization 측정**: chat/code 에서 회귀 폭 정량 분리 (SUB_071: sonnet ≫ chat ≫ code rank, +37.5% / −23.2%).
4. **분석 framework**: R/K 분해 + Leviathan closed-form alignment + workload-aware predictive gating heuristic (analysis doc §3·§6·§11).
5. **외부 사례 통합**: vLLM Issue #16258 / #19254 의 code 회귀 fact 와 본 측정 정합 확인 (analysis doc §10 R35/R36).

→ "+134% throughput 향상" 이 본 작업의 contribution 이 **아님**. 정확한 표현 = "vLLM built-in 의 ngram spec decoding 을 본 환경에서 +130% 까지 활용 가능함을 확정하고, 추가 fork patch 로 +1.65% 를 더했으며, workload 별 회귀 mechanism 을 분석".

## 3. 확인 / 업데이트 필요 doc

`grep -l "134%\|130%\|vs vanilla\|+134\|+130" shadow_assists/features/IDE_006/TSK_020/ shadow_assists/id_registry.md` 로 영향 범위 확인.

| 파일 | 갱신 필요 위치 | 권장 변경 |
|---|---|---|
| `analysis/workload_acceptance_analysis_20260524.md` | §1 TL;DR · §2.2 ratio 표 · §11.4 contribution 표 | 3-단계 breakdown 표 추가, "+134%" 옆에 "(vLLM built-in +130.3% + fork patch +1.65%)" 명시 |
| `Best_SpecDecode_10778tps.md` | 헤더 · §1 측정 fact · §3 sweep history · §5 vs vanilla baseline · §6 Objective 평가 | 3-단계 breakdown 표 추가. "본 fork patch contribution = +1.65%" 명시 |
| `INDEX.md` | §0 현 absolute best · §1 active SUB 표 | breakdown 한 줄 추가 |
| `measurements/sub044_spec_decode_20260523/RESULTS.md` | "+130.3% vs vanilla" → "+130.3% = vLLM built-in 활성화 효과 (코드 변경 0)" 로 framing |
| `measurements/sub047_t3_3run_verify_20260523/RESULTS.md` | "+134.1% vs vanilla" → "+1.65% over SUB_044 default cap=1 (fork patch contribution) / +134.1% 누적 vs vanilla" |
| `measurements/sub071_workload_large_20260524/RESULTS.md` | §3 비교 표의 sonnet baseline framing 도 동기화 |
| `id_registry.md` | SUB_044 / SUB_047 / TSK_020 entry 의 "+134%" / "+130%" 옆에 breakdown 표기 |

## 4. 업데이트 procedure (doc only, 코드 변경 없음)

1. analysis doc §1 TL;DR 위 또는 §11.4 안에 위 3-단계 breakdown 표 1개 추가 — 본 idea md 의 §1 표 copy.
2. Best doc §1 측정 fact 표 위에 같은 breakdown 표 추가 (또는 §5 vs vanilla baseline 영역 확장).
3. INDEX §0 "현 absolute best" 표 옆에 한 줄 "본 fork patch contribution = +1.65% (10,778.6 → 10,956.5)" 추가.
4. SUB_044 / SUB_047 RESULTS 의 "vs vanilla" 표현 옆에 framing 보강 (위 표 인용).
5. id_registry SUB_044 / SUB_047 / TSK_020 entry 의 "+134%" / "+130%" 옆에 breakdown 명시 (한 줄로 충분).
6. git commit: "docs(IDE_006/TSK_020): I001 contribution framing 정정 — vLLM built-in (+130.3%) vs fork patch (+1.65%) 분리 표기".
7. 본 idea md 의 status → `완료`, README 표 동기화.

## 5. 비교 framing 의 추가 정직 표시 (선택)

본 fork patch (+1.65%) 가 vLLM upstream PR #24986 의 review (benchislett: "threading plateaus after 4-8") 와 정합 → patch contribution 의 ceiling 도 명시 가능. 즉 본 fork patch 는 *이론적으로* +1-3% 영역의 lever 였고 실제로도 그렇게 나왔음 (over-claim 없음).

## 6. trade-off — "vs vanilla" framing 의 정당성도 부분 존재

본 fork patch 의 "단독 contribution" 은 +1.65% 이지만, **vLLM 영역 ngram path 가 default 로 disabled** (사용자가 명시적으로 `speculative_config={...}` 를 지정해야 ON) 이므로, 본 측정·검증 작업이 없었다면 본 환경의 production 도 "vanilla" 상태였을 가능성. 즉 본 작업의 contribution 은:

- patch contribution: +1.65%
- **"vLLM built-in feature 가 본 환경에서 net positive 임을 검증한 measurement contribution"**: +130.3%

→ 두 contribution 을 모두 인정하되 명확히 분리 표기하는 것이 가장 정직.

## 7. 결과 (SUB_073, 2026-05-24)

**측정 변경 없음** (기존 SUB_044/SUB_047 measurement 그대로 사용, framing 정정만 진행).

### 7.1 갱신된 doc 영역 (모두 본 idea §1 의 3-단계 breakdown 표 인용)

| doc | 갱신 위치 |
|---|---|
| `analysis/workload_acceptance_analysis_20260524.md` | §1.1 (TL;DR 다음에 breakdown 표) |
| `Best_SpecDecode_10778tps.md` | 헤더 callout box + §5 vs vanilla baseline 표 |
| `INDEX.md` | §0 현 absolute best — breakdown 한 줄 |
| `measurements/sub044_spec_decode_20260523/RESULTS.md` | 헤더 callout |
| `measurements/sub047_t3_3run_verify_20260523/RESULTS.md` | 헤더 callout |
| `id_registry.md` | SUB_044 + SUB_047 entries breakdown 명시 |

### 7.2 정량 fact (idem)

| 단계 | tps | vs 직전 | vs vanilla 누적 | source |
|---|---:|---:|---:|---|
| (1) vanilla (spec OFF) | 4,679.8 | — | — | vLLM upstream, `speculative_config=None` |
| (2) vLLM built-in spec ON (cap=1) | **10,778.6** (SUB_044 t3) | **+130.3%** | +130.3% | **vLLM 영역 코드 변경 0** — feature 활성화만 |
| (3) SUB_047 fork patch (cap=8, div_tp=0) | 10,956.5 (3-run avg) | **+1.65%** | +134.12% | 본 fork ~6 줄 patch (PR #24986 TODO 해소) |

### 7.3 결론

- **본 fork 단독 contribution = +1.65% (10,778.6 → 10,956.5)**
- **+130.3% 는 vLLM built-in feature 활성화 효과** (코드 변경 0)
- 본 idea 영역 fact 기반 정정 완료, fork 의 자랑된 over-claim 영역 제거됨
