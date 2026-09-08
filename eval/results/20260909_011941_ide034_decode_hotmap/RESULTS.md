# IDE_034 1차 — decode 라우팅 (입력 64/출력 512 워크로드) 기반 hotmap, hot-96 N=8 (CF+skip+pin) (2026-09-09 01:19~01:45)
트레이스: recorder per_pass, 553 패스 (decode 패스 547 = 14,824 토큰, prefill 6 = 2,214 토큰). 현행(prompt 기반) hotmap 의 decode 커버리지 H=96 **95.8%** (prompt 트레이스 주장 98.6%), E[D_cold](B=32) **8.1** (in-situ 실측 중앙값 6 과 부합). decode 기반 재구성: 커버리지 97.8%, E[D_cold] 5.0, 층당 hot-96 교체 ≈10개.

| 항목 | prompt hotmap (기준) | decode(in64/out512) hotmap |
|---|---|---|
| C32 sonnet 512/128 | 580~600 / TPOT 43.5 | **555.9 / 45.4** |
| C64 | 760~786 / 66.7~69.8 | 745.7 / 70.0 |
| GSM40 | 97.5% | 95.0% (38/40) |
| decode 층당 활성 cold expert 중앙값 (phase 분해) | 6 (평균 6.5) | **7 (평균 7.5)** |
| greedy | 기준 | 1문 분기 (Paris 이후), 1문 동일 |

**판정: 사전 등록 기준 (D_c ≤4.5 AND C32 ≥640) 미달 — 기각.** 원인 = 트레이스 워크로드 불일치: 입력 64 프롬프트의 생성 텍스트 라우팅으로 고른 hot set 은 sonnet 512/128 의 decode 에서 오히려 cold 를 늘림 (6→7). "decode 라우팅이 prompt 와 다르다" 는 사실 (커버리지 98.6→95.8) 은 확인됐으나, hot set 은 **워크로드-일치 decode 트레이스**로 골라야 함 → 2차 (측정 워크로드 자체의 decode 패스 ≤64 tok 만 집계).

**wrapper 계측 ([kt-wrap])**: do_numa_job 중앙값 622µs vs 커널 total 중앙값 600 → 디스패치/합류 ≈20µs, merge 21µs. **wrapper 오버헤드 ≈40µs/층** — 앞서 "0.2~0.25ms" 라 한 것은 TaskQueue 작업 평균(prefill 청크·긴 꼬리 포함) 과 커널 중앙값을 비교한 오류. 정정: hot-96 N=8 CPU 층 작업 ≈ 40 + 90 + 71·D_c µs. D_c=6.5 → 0.59ms × 62 = 37ms ≈ 스텝 43ms 의 대부분. expert 당 71µs = 23.6MB / 71µs = **332 GB/s** → DDR 실측 천장 (390~430) 의 ~80%. 즉 host 노드·큐 오버헤드 제거 후 하이브리드는 **DDR 대역폭 한계에 근접**했고, 남은 손잡이는 D_c (hot set 품질·H) 뿐.
