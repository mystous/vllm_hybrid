# SUB_073 — I001 vanilla +134% framing 정정 (doc only)

> **parent**: TSK_020 / idea I001
> **status**: 활성 (2026-05-24 신설)
> **effort**: 30-60 분 (doc 갱신만)
> **idea**: [`../idea/IDE_009_vanilla_contribution_framing.md`](../idea/IDE_009_vanilla_contribution_framing.md)

## 1. 진행 절차

idea I001 §4 procedure 그대로:

1. analysis doc §1 TL;DR + §11.4 contribution 표 — 3-단계 breakdown 추가
2. Best doc 헤더 + §1·§3·§5·§6 — breakdown 표 + "+1.65% fork patch contribution" 명시
3. INDEX §0 현 absolute best 표 — breakdown 한 줄
4. SUB_044 / SUB_047 / SUB_071 RESULTS 의 "vs vanilla" framing 보강
5. id_registry SUB_044/SUB_047/TSK_020 entry — breakdown 표기
6. 본 SUB plan + idea I001 status → 완료

## 2. 정확한 breakdown 표 (모든 doc 에 동일 표 인용)

| 단계 | config | source | tps | vs 직전 단계 | vs vanilla 누적 |
|---|---|---|---:|---:|---:|
| (1) vanilla | `speculative_config=None` | vLLM upstream (spec OFF) | 4,679.8 | — | — |
| (2) vLLM built-in spec ON (default cap=1) | `num_spec=7, prompt_lookup=2/5` | vLLM 영역 코드 변경 0 | **10,778.6** (SUB_044 t3) | **+130.3%** | **+130.3%** |
| (3) SUB_047 fork patch | `+ cap=8, div_tp=0` | 본 fork ~6 줄 patch | 10,956.5 (3-run avg) | **+1.65%** | **+134.12%** |

## 3. 측정 없음 (doc only)

본 SUB 는 새 측정 없이 기존 fact 의 framing 재정리. 측정 결과 자체는 SUB_044 / SUB_047 의 RESULTS 그대로 사용.

## 4. 산출물

- doc 갱신 (위 §1 list)
- commit message: `docs(IDE_006/TSK_020): SUB_073 — I001 vanilla framing 정정 (vLLM built-in +130.3% / fork patch +1.65% 분리 표기)`
