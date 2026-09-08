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

## hot-80 N=8, KV 131072
| 구성 | C32 tok/s / TPOT | C64 tok/s / TPOT |
|---|---|---|
| hot-80 N=4 (host 콜백, IDE_032) | 329.0 / 80.1 | 487.0 / 117.7 |
| hot-80 N=8 (host 콜백) | 343.4 / 76.8 | 479.7 / 120.2 |
| CF + 생략 (미고정) | 394.3 / 69.0 | — |
| CF + 생략 + 재배치 | **430.1 / 70.9** | **559.6 / 106.6** |

- hot-80 도 +25% (C32) / +15% (C64). 그러나 새 메커니즘 아래에서는 **hot-96 이 C64 에서도 hot-80 을 이김** (760 vs 560) — IDE_032 에서 관찰한 "C64 → hot-80" 전이가 사라짐. 정책 모델 (M2) 의 KV/thrash 항과 CPU 항을 새 메커니즘 파라미터로 재교정해야 함 (paper 의 전이점 주장은 "메커니즘이 전이점을 옮긴다" 로 보강).
- 두 체제 모두 재배치 이득은 TPOT 에 없고 처리량에만 있음 → prefill (eager 경로, 호스트 스레드 밀집) 구간 이득으로 추정. 별도 TTFT 비교 필요 (로그의 Mean TTFT 로 확인 가능).
h96_skip_unpinned.log: TTFT 1673.77 ms, P99 TPOT 
h96_skip_pinned.log: TTFT 1304.27 ms, P99 TPOT 
h80_skip_unpinned.log: TTFT 1627.35 ms, P99 TPOT 
h80_skip_pinned.log: TTFT 520.38 ms, P99 TPOT 
