# SUB_243 — CANARY (카나리 기반 간섭 센싱), 2026-06-15

> **판정: positive ⭐ — 메모리-지연 canary 가 간섭 +400% 감지.** SUB_230(runq blind)의
> gap 을 메우는 SLO 신호. CSMA/MERCATO/CLOSPACK 의 입력 신호로 활용.

## 결과 (canary = serving 코어(0) 동거 경량 pointer-chase probe)
| canary ws | 무harvest ns/load | harvest ns/load | 감지 |
|---|---:|---:|---:|
| 1MB (L2 상주) | 5.4 | 5.3 | −2% (blind) |
| 8MB (DRAM) | 35.3 | 178.0 | **+404%** |
| 64MB (DRAM) | 39.2 | 193.5 | +394% |

## 판정
1. **메모리-지연 canary(ws≥8MB)가 harvest BW 간섭을 +400% latency 로 감지** — 강력한
   SLO 신호. SUB_230 의 runq latency 는 동일 간섭에 **blind(0%)** 였음 → CANARY 가 정확히
   그 차원(메모리 지연)을 측정.
2. **설계 파라미터**: canary ws ≥ DRAM-touching(≈8MB) 필수. 1MB(캐시상주)는 blind
   (runq 처럼 메모리 BW 간섭 못 봄). probe 오버헤드 작음(단일 코어 일부).
3. **활용**: CSMA(238)/MERCATO(242)/CLOSPACK(241)이 총 BW 대신 canary 지연을 신호로
   쓰면 victim 체감 직접 반영 → 더 정확한 보호. governor 입력 = CANARY.

## 비고
- 실제 serving 에선 serving CLOS *안에* 카나리 스레드 동거 (serving 과 같은 간섭 경험).
- 1ms 윈도우 p99 집계로 조기경보 (SUB_230 게이트 ≤20ms 충족 — 즉시 감지).

산출물: `runs/results.csv`.
