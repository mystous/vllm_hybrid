# BOTTLENECK_EVIDENCE — IDE_075 (2026-09-17T20:53:53+09:00)

지시서 §0 의 5 질문에 대한 직접 근거. 원값은 세션/v2. 확인 / 원인 미분리 / 미측정 구분. E2E 환산·개선율 없음.

## S10_CORR — ['step[DECODE bs=63]'] (cold_present_nonempty n=15493)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 848.2 / 2215.5, late share 0.949 (thr5 0.948) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 1178.8 = 선행 실행 점유 1177.8 (deferred 1177.5) + 미분리 0.8 / p95 1.5; go→enqueue 3.5 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 1427.2; end→pub 0.5; pub→h2d_start 13.2; h2d 30.2; post-h2d→combine 7.2; pre-h2d gap 861.5 중 GPU idle 861.5 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 62, clock {'OK': 23808}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S10_CORR — ['step[DECODE bs=2]'] (cold_present_nonempty n=3061)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 -85.0 / 59.0, late share 0.098 (thr5 0.092) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 4.5 = 선행 실행 점유 0.2 (deferred 0.0) + 미분리 3.8 / p95 6.2; go→enqueue 2.5 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 148.5; end→pub 93.2; pub→h2d_start 95.5; h2d 4.0; post-h2d→combine 5.5; pre-h2d gap 11.2 중 GPU idle 11.2 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 62, clock {'OK': 23808}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S12_C1_CORR — ['step[DECODE bs=1]'] (cold_present_nonempty n=46635)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 -59.2 / 116.2, late share 0.186 (thr5 0.178) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 4.0 = 선행 실행 점유 0.2 (deferred 0.0) + 미분리 3.2 / p95 5.8; go→enqueue 2.5 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 148.8; end→pub 56.0; pub→h2d_start 69.0; h2d 3.5; post-h2d→combine 5.8; pre-h2d gap 11.5 중 GPU idle 11.5 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 62, clock {'OK': 126976}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S13_LONG_CORR — ['step[DECODE bs=8]'] (cold_present_nonempty n=28213)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 23.8 / 708.0, late share 0.528 (thr5 0.521) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 196.2 = 선행 실행 점유 195.2 (deferred 195.0) + 미분리 1.2 / p95 5.5; go→enqueue 2.2 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 472.8; end→pub 1.0; pub→h2d_start 18.2; h2d 9.0; post-h2d→combine 6.0; pre-h2d gap 37.0 중 GPU idle 37.0 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 62, clock {'OK': 31744}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S9_CORR — ['step[DECODE bs=63]'] (cold_present_nonempty n=15541)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 971.8 / 2295.5, late share 0.963 (thr5 0.962) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 1309.2 = 선행 실행 점유 1308.5 (deferred 1308.2) + 미분리 0.8 / p95 1.5; go→enqueue 4.2 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 1556.2; end→pub 0.5; pub→h2d_start 13.2; h2d 30.0; post-h2d→combine 7.2; pre-h2d gap 985.8 중 GPU idle 985.8 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 24552, clock {'OK': 23808}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S9_CORR — ['step[DECODE bs=2]'] (cold_present_nonempty n=3061)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 -84.8 / 61.5, late share 0.097 (thr5 0.094) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 4.0 = 선행 실행 점유 0.2 (deferred 0.0) + 미분리 3.5 / p95 6.0; go→enqueue 2.8 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 149.2; end→pub 91.0; pub→h2d_start 95.5; h2d 4.2; post-h2d→combine 5.5; pre-h2d gap 11.2 중 GPU idle 11.2 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 24552, clock {'OK': 23808}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S17_CORR_pinned — ['step[DECODE bs=63]'] (cold_present_nonempty n=15493)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 869.5 / 2235.2, late share 0.953 (thr5 0.952) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 1199.2 = 선행 실행 점유 1198.2 (deferred 1198.0) + 미분리 1.0 / p95 1.8; go→enqueue 4.2 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 1447.2; end→pub 0.8; pub→h2d_start 13.0; h2d 29.8; post-h2d→combine 7.2; pre-h2d gap 882.2 중 GPU idle 881.8 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 62, clock {'OK': 23808}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S17_CORR_pinned — ['step[DECODE bs=2]'] (cold_present_nonempty n=3061)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 -84.8 / 62.8, late share 0.097 (thr5 0.094) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 4.0 = 선행 실행 점유 0.2 (deferred 0.0) + 미분리 3.2 / p95 6.8; go→enqueue 2.5 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 151.2; end→pub 88.8; pub→h2d_start 95.2; h2d 4.0; post-h2d→combine 5.5; pre-h2d gap 11.2 중 GPU idle 11.2 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 62, clock {'OK': 23808}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S18_CORR_pinned — ['step[DECODE bs=63]'] (cold_present_nonempty n=15493)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 882.0 / 2262.0, late share 0.952 (thr5 0.952) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 1210.0 = 선행 실행 점유 1209.0 (deferred 1209.0) + 미분리 1.0 / p95 1.8; go→enqueue 4.0 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 1459.0; end→pub 0.5; pub→h2d_start 13.2; h2d 30.0; post-h2d→combine 7.2; pre-h2d gap 895.0 중 GPU idle 895.0 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 62, clock {'OK': 23808}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S18_CORR_pinned — ['step[DECODE bs=2]'] (cold_present_nonempty n=3061)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 -84.2 / 69.0, late share 0.104 (thr5 0.100) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 4.0 = 선행 실행 점유 0.2 (deferred 0.0) + 미분리 3.5 / p95 6.8; go→enqueue 3.0 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 151.0; end→pub 88.8; pub→h2d_start 95.0; h2d 4.2; post-h2d→combine 5.8; pre-h2d gap 11.5 중 GPU idle 11.5 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 62, clock {'OK': 23808}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S22_CORR_vllmw — ['step[DECODE bs=63]'] (cold_present_nonempty n=15410)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 392.0 / 59969.0, late share 0.916 (thr5 0.914) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 670.5 = 선행 실행 점유 669.8 (deferred 669.2) + 미분리 0.8 / p95 1.2; go→enqueue 3.0 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 921.8; end→pub 0.5; pub→h2d_start 12.0; h2d 30.2; post-h2d→combine 7.2; pre-h2d gap 337.2 중 GPU idle 337.2 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 86, clock {'OK': 22668, 'CLOCK_INDETERMINATE': 966, 'CLOCK_ORDER_VIOLATION': 19}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S22_CORR_vllmw — ['step[DECODE bs=2]'] (cold_present_nonempty n=2463)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 -84.5 / -50.2, late share 0.020 (thr5 0.019) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 3.2 = 선행 실행 점유 0.0 (deferred 0.0) + 미분리 3.2 / p95 5.8; go→enqueue 3.5 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 171.0; end→pub 73.8; pub→h2d_start 95.0; h2d 4.0; post-h2d→combine 5.5; pre-h2d gap 10.8 중 GPU idle 10.8 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 86, clock {'OK': 22668, 'CLOCK_INDETERMINATE': 966, 'CLOCK_ORDER_VIOLATION': 19}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S23_CORR_vllmw — ['step[DECODE bs=63]'] (cold_present_nonempty n=15541)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 328.2 / 826.2, late share 0.911 (thr5 0.907) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 667.5 = 선행 실행 점유 666.0 (deferred 666.0) + 미분리 1.2 / p95 2.0; go→enqueue 4.5 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 915.8; end→pub 0.8; pub→h2d_start 13.5; h2d 29.2; post-h2d→combine 7.2; pre-h2d gap 341.2 중 GPU idle 341.0 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 78, clock {'OK': 23754}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S23_CORR_vllmw — ['step[DECODE bs=2]'] (cold_present_nonempty n=3006)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 -84.8 / -22.5, late share 0.039 (thr5 0.036) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 3.8 = 선행 실행 점유 0.0 (deferred 0.0) + 미분리 3.5 / p95 6.5; go→enqueue 3.0 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 146.2; end→pub 94.8; pub→h2d_start 95.8; h2d 4.2; post-h2d→combine 5.5; pre-h2d gap 11.0 중 GPU idle 11.0 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 78, clock {'OK': 23754}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S26_C1_CORR_vllmw — ['step[DECODE bs=1]'] (cold_present_nonempty n=20646)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 -62.8 / -51.8, late share 0.018 (thr5 0.016) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 3.8 = 선행 실행 점유 0.0 (deferred 0.0) + 미분리 3.5 / p95 6.2; go→enqueue 2.8 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 134.0; end→pub 93.5; pub→h2d_start 73.2; h2d 3.0; post-h2d→combine 6.0; pre-h2d gap 10.8 중 GPU idle 10.8 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 0, clock {'OK': 126976}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S27_LONG_CORR_vllmw — ['step[DECODE bs=8]'] (cold_present_nonempty n=23657)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 -180.8 / -49.5, late share 0.030 (thr5 0.029) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 3.8 = 선행 실행 점유 0.0 (deferred 0.0) + 미분리 3.2 / p95 6.8; go→enqueue 3.2 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 245.8; end→pub 132.8; pub→h2d_start 191.0; h2d 8.8; post-h2d→combine 6.0; pre-h2d gap 10.5 중 GPU idle 10.5 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 12, clock {'OK': 30618, 'CLOCK_INDETERMINATE': 1070}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S3_CORR — ['step[DECODE bs=63]'] (cold_present_nonempty n=15598)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 321.0 / 791.0, late share 0.900 (thr5 0.897) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 663.5 = 선행 실행 점유 662.8 (deferred 661.8) + 미분리 0.8 / p95 1.2; go→enqueue 3.2 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 911.5; end→pub 0.5; pub→h2d_start 13.2; h2d 30.8; post-h2d→combine 7.2; pre-h2d gap 334.2 중 GPU idle 334.0 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 0, clock {'OK': 23808}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S3_CORR — ['step[DECODE bs=2]'] (cold_present_nonempty n=2220)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 -85.5 / -66.8, late share 0.015 (thr5 0.014) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 4.0 = 선행 실행 점유 0.0 (deferred 0.0) + 미분리 3.8 / p95 7.2; go→enqueue 2.8 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 144.0; end→pub 99.0; pub→h2d_start 96.2; h2d 4.0; post-h2d→combine 5.5; pre-h2d gap 10.8 중 GPU idle 10.8 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 0, clock {'OK': 23808}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S4_CORR — ['step[DECODE bs=63]'] (cold_present_nonempty n=15596)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 315.0 / 823.0, late share 0.899 (thr5 0.896) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 649.8 = 선행 실행 점유 648.8 (deferred 648.2) + 미분리 0.8 / p95 1.5; go→enqueue 3.5 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 902.2; end→pub 0.5; pub→h2d_start 13.5; h2d 30.5; post-h2d→combine 7.2; pre-h2d gap 328.0 중 GPU idle 327.8 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 0, clock {'OK': 23808}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S4_CORR — ['step[DECODE bs=2]'] (cold_present_nonempty n=2297)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 -83.5 / -61.5, late share 0.017 (thr5 0.017) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 3.5 = 선행 실행 점유 0.0 (deferred 0.0) + 미분리 3.5 / p95 6.0; go→enqueue 3.0 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 143.0; end→pub 97.2; pub→h2d_start 93.8; h2d 4.0; post-h2d→combine 5.5; pre-h2d gap 10.8 중 GPU idle 10.8 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 0, clock {'OK': 23808}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S29_CORR_vllmw_fp8fixso — ['step[DECODE bs=63]'] (cold_present_nonempty n=15600)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 335.0 / 822.0, late share 0.911 (thr5 0.908) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 673.0 = 선행 실행 점유 672.0 (deferred 671.8) + 미분리 0.8 / p95 1.5; go→enqueue 3.8 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 921.0; end→pub 0.5; pub→h2d_start 13.2; h2d 26.8; post-h2d→combine 7.2; pre-h2d gap 348.5 중 GPU idle 348.5 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 62, clock {'OK': 23808}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## S29_CORR_vllmw_fp8fixso — ['step[DECODE bs=2]'] (cold_present_nonempty n=2273)

| 질문 | 관측 (p50 / p95, µs) | 구분 |
|---|---|---|
| Q1 어떤 생산층·작업이 지연을 만드나 | producer_layer_costs.csv 층별 표 (FULL_REPORT §7); cold_pub_delta p50 -83.8 / -61.2, late share 0.017 (thr5 0.016) | 확인 (층별 분포 제공) |
| Q2 go→start 중 선행 FIFO vs 전달 지연 | enqueue_to_start p50 3.5 = 선행 실행 점유 0.0 (deferred 0.0) + 미분리 3.5 / p95 7.0; go→enqueue 3.2 | 확인 (같은 사건 단위; 미분리 잔여는 OS/입력 준비/bracket 오차 포함) |
| Q3 CPU 완료 이전 / 신호 / HtoD / 결합 중 어디가 소비를 늦추나 | service 142.5; end→pub 98.8; pub→h2d_start 94.2; h2d 4.0; post-h2d→combine 5.5; pre-h2d gap 10.8 중 GPU idle 10.8 | 확인 |
| Q4 계측·clock·창·매핑 산물인가 | 기록 전용 task 0, 매핑 불일치 62, clock {'OK': 23808}, residual max 0.0, 간섭 status: FULL_REPORT §3 | 검증 (offset 불변 가능성·부팅 간 비교 한계 남음) |
| Q5 근거가 생긴 변경 / 부족한 변경 | READINESS.md | 판정 문서 |

## 미측정·한계

- expert 별 rows·분기는 작업 단위 합계(unique/max/sum rows, AMX/AVX 호출 수)로만 수집. expert 단위 표본 없음.
- prefill(EXTEND) 층 분절 없음. TP collective 위치 귀속 미확정 (조건부 미실행).
- OFF 두 부팅이 CORE 앞에 위치, CORE/CORR/RESOURCE 는 같은 부팅. 단발 CORE/RESOURCE 통계 한계.
- clock: 일정 offset 은 순서검사로 검출되지 않음.
- DRAM: 부하 창 PCM 값은 관측값이며 포화 판정 없음 (비교 기준 부재).
