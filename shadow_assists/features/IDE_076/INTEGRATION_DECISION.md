# INTEGRATION_DECISION — 최종 조합 (2026-09-18 14:50 KST 확정)

## 1. 개별 판정 요약 (수정 후 모집단, 같은 블록 R0 대비 짝 이득, MAIN_SHORT OFF)
| 변형 | 유효 블록 | 평균 이득 | 판정 | 문서 |
|---|---|---|---|---|
| A1 (GFNI 언팩) | 4 | +1.6 % | 불채택 (< 2 %) | A1_DECISION |
| A1e (PF=1) | 4 | +3.1 % | 채택 | A1_DECISION |
| A1ae (GFNI+PF) | 3 | +3.7 % | 채택 | A1_DECISION |
| A2 (묶음 dispatch) | 4 | +5.0 % | 채택 | A2_DECISION |
| A12 (GFNI+A2) | 5 | +5.9 % | 채택 | A2_DECISION |
| **A12e (GFNI+A2+PF)** | 5 | **+7.7 %** | **채택 (최고)** | A2_DECISION |
| B (HB_MIX1) | 1 (c3) | MAIN −6.1 %, IND +7.2 % | 불채택 | B_DECISION |
| A2B | 3 | MAIN −6.2 % (−10.6/−7.2/−0.8), IND +9.5 % | 불채택 | B_DECISION |
| A12B | 0 (3 부팅 사망) | — | 판정 불가·불채택 | B_DECISION |

## 2. 통합 조건
- **최종 후보: A12e = v8f 바이너리 + `KT_OPT_A1_ENABLE=1` + `KT_OPT_A2_ENABLE=1` + `KT_AVX_PF=1`, Hot 배치 H0(hotmap_v2 + layer_budget_5952).**
- 회귀: C1 63.3 (R0 63.4), LONG 165.1 (R0 164.2), TTFT p95 비 0.99, TPOT p95 비 0.89 — 모두 한도 내. 수치: OFF 와 bitwise 동일(CPU 경로), FP8 native 회귀 PASS, smoke greedy 4/4 동일.
- 단독 기준 미달인 A1-a 는 A2·PF 와 결합할 때 +1~2 %p 를 더하므로(A12 vs A2, A12e vs A1e) 통합 조건에서는 켠다(§9 "조합 효과").
- B 는 고정 평가 입력에서 손실이라 통합에서 제외. B 의 이득은 calibration 분포 전용(B_DECISION §4).

## 3. 확인 결과
- CONFIRMATION c1~c3 으로 B 불채택 확정(고정 입력 전부 음수, 수치 기준 미달). A12e 는 H0 배치 5 블록으로 판정. 원복 시험은 ROLLBACK.md §4.

## 4. 사고 상태
- callback-free ∧ overlap ∧ 빈 imm 생략 조합에서 GPU fused_moe 출력 NaN 으로 서버가 사망하는 잠재 결함(원인 미해결, IDE_076 이전부터 존재, 새 바이너리·A1/A2 무관). 사망 부팅은 실패로 기록·제외했으며 판정은 유효 블록만 사용. 운영 시 같은 발생률(부팅당 30~40 %)이 예상되므로 별도 과제.
