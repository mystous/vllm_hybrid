# BOTTLENECK_EVIDENCE — IDE_075 (2026-09-17T15:58:37+09:00)

지시서 §0 의 5 질문에 대한 직접 근거. 원값은 세션/v2. 확인 / 원인 미분리 / 미측정 구분. E2E 환산·개선율 없음.

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

## 미측정·한계

- expert 별 rows·분기는 작업 단위 합계(unique/max/sum rows, AMX/AVX 호출 수)로만 수집. expert 단위 표본 없음.
- prefill(EXTEND) 층 분절 없음. TP collective 위치 귀속 미확정 (조건부 미실행).
- OFF 두 부팅이 CORE 앞에 위치, CORE/CORR/RESOURCE 는 같은 부팅. 단발 CORE/RESOURCE 통계 한계.
- clock: 일정 offset 은 순서검사로 검출되지 않음.
- DRAM: 부하 창 PCM 값은 관측값이며 포화 판정 없음 (비교 기준 부재).
