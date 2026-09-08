# IDE_033-b — 빈 immediate 생략 (KT_CF_SKIP_EMPTY_IMM) + 비-워커 스레드 코어 재배치 (pin_nonkt.sh) A/B (2026-09-09 00:36~)
구성: callback-free (IDE_033) 위에 (a) N ≥ k 이면 immediate 작업 생략 → 패킷 = [done 신호, deferred] (b) KTransformers 워커 코어 (물리 0-47/56-103) 의 HT 형제에 올라간 스핀 스레드 (TP 랭크 메인·폴러·TaskQueue 워커 등 1338 스레드) 를 빈 코어 48-55/104-111(+HT) 로 이동.

## hot-96 N=8, KV 49152, sonnet 512/128
| 구성 | C32 tok/s / TPOT | C64 tok/s / TPOT | GSM40 |
|---|---|---|---|
| 기존 최종 (hot-96+def4, host 콜백) | 490.9 / 51.5 | 408.2 / 86.7 | 97.0% (100) |
| IDE_032 N=8 (host 콜백) | 505.4 / 48.3 | — | 97.5% |
| IDE_033 CF | 531.0 / 45.5 | — | 97.5% |
| CF + 빈 immediate 생략 (미고정) | 575.1 / 42.9 | — | — |
| CF + 생략 + 코어 재배치 | **599.7 / 43.5**, 반복 **601.3 / 43.4** | **760.1 / 69.8** | **97.5%** |

- greedy 2문 기준과 동일, 32버스트 정상. kill 기준 (C32 ≥580 AND GSM40 ≥95) **통과** (재배치 포함 시).
- 분해: host 노드 제거 +5.1%, 빈 immediate 생략 +8.3%, 코어 재배치 +4.3% (TPOT 불변 → decode 스텝이 아니라 prefill/스케줄러 쪽 이득). 합계 기존 최종 대비 C32 **+22.5%**, C64 **+86%**.
- TaskQueue 계측 (생략 후): 층당 [신호, deferred] 2개, 워커 busy 0.99~1.00, deferred 평균 ≈0.8ms/층. 남은 병목 = CPU 층 작업 시간 자체 (모델 항 0.3ms 대비 2.7배) → phase 분해 (`…_ide033_phase_prof/`).
- C64 760 은 별도 부팅에서 재현 예정 (phase prof 체인).
