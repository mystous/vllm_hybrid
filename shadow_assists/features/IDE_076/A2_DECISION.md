# A2_DECISION — 같은 Cold job 내부 descriptor 묶음 dispatch (잠정 2026-09-18 12:50 KST)

## 1. 적용성 (C08)
- v7 probe(왜곡 제거): 풀 장벽 tail 3.7 %(A1=0)/4.4 %(A1=1), 워커 종료 폭 5.2/6.0 % of dispatch wall, S29 비-GEMM 스테이지 합 10 % → 줄일 내부 비용 ≥3 % 확인 → 구현.

## 2. 구현·수치 계약 (C09, C04)
- `forward_prefill_a2_`: S1~S5(QA→GU→ACT→QD→DN) 를 expert 단위 atomic readiness 로 단일 dispatch. `KT_OPT_A2_ENABLE=1`, OFF 는 기존 경로 그대로(`else`).
- replay(v8f, n=40): A2 ON = OFF bitwise 동일(6 장면), A1+A2 도 동일. 시간 p50: hot_r3 −2.0 %, seq_mixed −1.9 %, hot_r1/r2 −0.3~0.5 %, e18 +0.4~2.1 %. proof: 적격 job 142/142 묶음, fallback 0, 장벽 tail 2.8 % of dispatch wall.
- FP8 native 회귀 PASS.

## 3. E2E (MAIN_SHORT OFF, 같은 블록 R0 대비 짝 이득)
| 변형 | 유효 블록 | 짝 이득 % | 평균 | 전부 양수 | TTFT p95 비 | TPOT p95 비 | C1 | LONG | 판정 |
|---|---|---|---|---|---|---|---|---|---|
| A2 | 4 | +4.3, +6.0, +4.5, +5.0 | +5.0 % | 예 | 1.00 | 0.94 | 63.4 | 164.9 | **채택** |
| A12 (GFNI+A2) | 5 | +10.5, +4.5, +5.8, +4.2, +4.2 | +5.9 % | 예 | 0.99 | 0.92 | 63.3 | 166.2 | **채택** |
| A12e (GFNI+A2+PF) | 5 | +6.7, +8.1, +9.4, +7.0, +7.3 | +7.7 % | 예 | 0.99 | 0.89 | 63.3 | 165.1 | **채택 (최고)** |
- 사망 제외: A2 1, A12 1, A12e 1 부팅.

## 4. 해석
- replay 의 −2 % 보다 E2E 이득(+5 %)이 크다: 서비스 시간 단축이 GPU 대기(late_share 큰 층 4~23)에 그대로 노출되는 구조(B_COST_MODEL §7)와 정합.
- A2 위에 GFNI 는 +1 %p, PF 는 +2~3 %p 를 더한다. 세 가지 결합(A12e)이 가장 높고 회귀 항목은 모두 한도 내(TPOT p95 −11 %).

## 5. 결정
- **A2 채택.** 통합 조건은 INTEGRATION_DECISION 에서 B(HB_MIX1) 와의 조합(A12B/A2B, CONFIRMATION) 결과와 함께 확정.
- 원복: `KT_OPT_A2_ENABLE` unset (ROLLBACK.md §1, §3).
