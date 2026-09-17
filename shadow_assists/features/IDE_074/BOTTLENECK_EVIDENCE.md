# BOTTLENECK_EVIDENCE — IDE_074 (2026-09-17T14:05:54+09:00)

지시서 §1.1 질문별 직접 근거. 각 값은 세션별 dependency_summary.json / validation_results.json 원값. 확인 / 원인 미분리 / 미측정 을 구분. 성능 개선 상한·E2E 환산 없음.

## 세션 P1_r1 (boot B2_132633) — 표본 coverage {'gpu_layer_rows_total': 23808, 'no_prev_deferred': 384, 'matched_rows': 23808}, 순서 위반 {}

### graph 2 ['step[DECODE bs=63]'] — 유효 행 15872 (prev 연결 15616)

| 질문 | 관측 (µs, p50 / p95 / p99) | 표본 | 구분 |
|---|---|---|---|
| Q1 결합 시 Hot/Cold 중 늦은 입력 | cold_pub − hot_ready p50 302.2 / 800.2 / 1028.5; cold 가 늦은 비율 0.885; other(잔차) 는 hot 이전에 준비 | 15872 | 확인 (CPU done 시각 vs GPU hot 종료, 같은 host 시계; 순서검사 통과) |
| Q1' GPU 에 노출된 대기 | hot 종료→HtoD 시작 p50 315.2 / 812.0 / 1041.0 (하한≈bs=2 graph 의 p50) | 15872 | 확인 (GPU 시계 단일) |
| Q2 Cold 경로 어디서 늦나 | 제출→시작 650.5 / 1172.8; 계산 898.2 / 1407.5; 게시→해제 13.2 / 145.0; 게시→HtoD 완료 43.8 / 166.0; cold 예산(go(L−1)→hot_ready(L)) 1247.5 / 1781.8 | 15616 | 확인 (제출→시작은 단일 FIFO 대기+입력 준비 포함; 순수 큐 대기로 명명하지 않음) |
| Q3 Hot/다른 GPU 작업 | hot fused_moe 커널 합 p50 323.8 / 435.0 | 15872 | 확인 (커널 이름 기준; attention/router 분해는 미집계=미측정) |
| Q4 NUMA 서브풀 | 완료 시차 p50 30.8 / 80.2 / 168.5; numa0 870.0 numa1 835.5 | 15616 | 확인 (원격 메모리 접근 등 원인 미분리) |
| Q4' layer/expert 집중 | dependency_metrics.csv.gz 의 layer 별 분포 (본 문서 미집계) | | 원자료 제공 |
| Q5 무계측 적용 가능성 | observer_overhead.csv 의 OFF/ON 비율, 반복 3회 | | 부팅 간 비교 한계 |

### graph 32 ['step[DECODE bs=2]'] — 유효 행 7936 (prev 연결 7808)

| 질문 | 관측 (µs, p50 / p95 / p99) | 표본 | 구분 |
|---|---|---|---|
| Q1 결합 시 Hot/Cold 중 늦은 입력 | cold_pub − hot_ready p50 -85.0 / -65.8 / -31.0; cold 가 늦은 비율 0.006; other(잔차) 는 hot 이전에 준비 | 7936 | 확인 (CPU done 시각 vs GPU hot 종료, 같은 host 시계; 순서검사 통과) |
| Q1' GPU 에 노출된 대기 | hot 종료→HtoD 시작 p50 10.8 / 14.0 / 15.0 (하한≈bs=2 graph 의 p50) | 7936 | 확인 (GPU 시계 단일) |
| Q2 Cold 경로 어디서 늦나 | 제출→시작 6.0 / 14.5; 계산 32.8 / 194.0; 게시→해제 95.5 / 106.0; 게시→HtoD 완료 99.8 / 110.2; cold 예산(go(L−1)→hot_ready(L)) 339.5 / 358.0 | 7808 | 확인 (제출→시작은 단일 FIFO 대기+입력 준비 포함; 순수 큐 대기로 명명하지 않음) |
| Q3 Hot/다른 GPU 작업 | hot fused_moe 커널 합 p50 74.2 / 82.0 | 7936 | 확인 (커널 이름 기준; attention/router 분해는 미집계=미측정) |
| Q4 NUMA 서브풀 | 완료 시차 p50 2.8 / 10.0 / 24.8; numa0 12.5 numa1 11.0 | 7808 | 확인 (원격 메모리 접근 등 원인 미분리) |
| Q4' layer/expert 집중 | dependency_metrics.csv.gz 의 layer 별 분포 (본 문서 미집계) | | 원자료 제공 |
| Q5 무계측 적용 가능성 | observer_overhead.csv 의 OFF/ON 비율, 반복 3회 | | 부팅 간 비교 한계 |

## 세션 P1_r2 (boot B2_132633) — 표본 coverage {'gpu_layer_rows_total': 23808, 'no_prev_deferred': 384, 'matched_rows': 23808}, 순서 위반 {}

### graph 2 ['step[DECODE bs=63]'] — 유효 행 15872 (prev 연결 15616)

| 질문 | 관측 (µs, p50 / p95 / p99) | 표본 | 구분 |
|---|---|---|---|
| Q1 결합 시 Hot/Cold 중 늦은 입력 | cold_pub − hot_ready p50 307.0 / 801.8 / 1022.5; cold 가 늦은 비율 0.884; other(잔차) 는 hot 이전에 준비 | 15872 | 확인 (CPU done 시각 vs GPU hot 종료, 같은 host 시계; 순서검사 통과) |
| Q1' GPU 에 노출된 대기 | hot 종료→HtoD 시작 p50 319.8 / 813.5 / 1034.0 (하한≈bs=2 graph 의 p50) | 15872 | 확인 (GPU 시계 단일) |
| Q2 Cold 경로 어디서 늦나 | 제출→시작 655.2 / 1177.5; 계산 902.2 / 1410.2; 게시→해제 13.5 / 144.8; 게시→HtoD 완료 42.5 / 166.5; cold 예산(go(L−1)→hot_ready(L)) 1251.8 / 1779.5 | 15616 | 확인 (제출→시작은 단일 FIFO 대기+입력 준비 포함; 순수 큐 대기로 명명하지 않음) |
| Q3 Hot/다른 GPU 작업 | hot fused_moe 커널 합 p50 326.0 / 437.2 | 15872 | 확인 (커널 이름 기준; attention/router 분해는 미집계=미측정) |
| Q4 NUMA 서브풀 | 완료 시차 p50 30.2 / 78.5 / 153.5; numa0 870.2 numa1 838.0 | 15616 | 확인 (원격 메모리 접근 등 원인 미분리) |
| Q4' layer/expert 집중 | dependency_metrics.csv.gz 의 layer 별 분포 (본 문서 미집계) | | 원자료 제공 |
| Q5 무계측 적용 가능성 | observer_overhead.csv 의 OFF/ON 비율, 반복 3회 | | 부팅 간 비교 한계 |

### graph 32 ['step[DECODE bs=2]'] — 유효 행 7936 (prev 연결 7808)

| 질문 | 관측 (µs, p50 / p95 / p99) | 표본 | 구분 |
|---|---|---|---|
| Q1 결합 시 Hot/Cold 중 늦은 입력 | cold_pub − hot_ready p50 -83.0 / -63.5 / -16.5; cold 가 늦은 비율 0.008; other(잔차) 는 hot 이전에 준비 | 7936 | 확인 (CPU done 시각 vs GPU hot 종료, 같은 host 시계; 순서검사 통과) |
| Q1' GPU 에 노출된 대기 | hot 종료→HtoD 시작 p50 10.8 / 14.0 / 15.2 (하한≈bs=2 graph 의 p50) | 7936 | 확인 (GPU 시계 단일) |
| Q2 Cold 경로 어디서 늦나 | 제출→시작 5.2 / 13.2; 계산 34.0 / 196.2; 게시→해제 93.2 / 105.8; 게시→HtoD 완료 97.5 / 110.0; cold 예산(go(L−1)→hot_ready(L)) 334.8 / 356.2 | 7808 | 확인 (제출→시작은 단일 FIFO 대기+입력 준비 포함; 순수 큐 대기로 명명하지 않음) |
| Q3 Hot/다른 GPU 작업 | hot fused_moe 커널 합 p50 71.5 / 81.8 | 7936 | 확인 (커널 이름 기준; attention/router 분해는 미집계=미측정) |
| Q4 NUMA 서브풀 | 완료 시차 p50 2.5 / 10.2 / 21.8; numa0 12.8 numa1 10.2 | 7808 | 확인 (원격 메모리 접근 등 원인 미분리) |
| Q4' layer/expert 집중 | dependency_metrics.csv.gz 의 layer 별 분포 (본 문서 미집계) | | 원자료 제공 |
| Q5 무계측 적용 가능성 | observer_overhead.csv 의 OFF/ON 비율, 반복 3회 | | 부팅 간 비교 한계 |

## 세션 P1_r3 (boot B2_132633) — 표본 coverage {'gpu_layer_rows_total': 23808, 'no_prev_deferred': 384, 'matched_rows': 23808}, 순서 위반 {'t_go_before_dtoh_end_within_5us(jitter)': 1659}

### graph 2 ['step[DECODE bs=63]'] — 유효 행 15872 (prev 연결 15616)

| 질문 | 관측 (µs, p50 / p95 / p99) | 표본 | 구분 |
|---|---|---|---|
| Q1 결합 시 Hot/Cold 중 늦은 입력 | cold_pub − hot_ready p50 328.5 / 837.5 / 1073.5; cold 가 늦은 비율 0.896; other(잔차) 는 hot 이전에 준비 | 15872 | 확인 (CPU done 시각 vs GPU hot 종료, 같은 host 시계; 순서검사 통과) |
| Q1' GPU 에 노출된 대기 | hot 종료→HtoD 시작 p50 341.2 / 849.5 / 1084.0 (하한≈bs=2 graph 의 p50) | 15872 | 확인 (GPU 시계 단일) |
| Q2 Cold 경로 어디서 늦나 | 제출→시작 677.2 / 1214.8; 계산 923.0 / 1448.8; 게시→해제 13.8 / 134.5; 게시→HtoD 완료 43.5 / 156.5; cold 예산(go(L−1)→hot_ready(L)) 1269.5 / 1814.0 | 15616 | 확인 (제출→시작은 단일 FIFO 대기+입력 준비 포함; 순수 큐 대기로 명명하지 않음) |
| Q3 Hot/다른 GPU 작업 | hot fused_moe 커널 합 p50 323.5 / 436.2 | 15872 | 확인 (커널 이름 기준; attention/router 분해는 미집계=미측정) |
| Q4 NUMA 서브풀 | 완료 시차 p50 32.5 / 84.2 / 161.2; numa0 892.0 numa1 856.8 | 15616 | 확인 (원격 메모리 접근 등 원인 미분리) |
| Q4' layer/expert 집중 | dependency_metrics.csv.gz 의 layer 별 분포 (본 문서 미집계) | | 원자료 제공 |
| Q5 무계측 적용 가능성 | observer_overhead.csv 의 OFF/ON 비율, 반복 3회 | | 부팅 간 비교 한계 |

### graph 32 ['step[DECODE bs=2]'] — 유효 행 7936 (prev 연결 7808)

| 질문 | 관측 (µs, p50 / p95 / p99) | 표본 | 구분 |
|---|---|---|---|
| Q1 결합 시 Hot/Cold 중 늦은 입력 | cold_pub − hot_ready p50 -85.0 / -69.5 / -45.0; cold 가 늦은 비율 0.004; other(잔차) 는 hot 이전에 준비 | 7936 | 확인 (CPU done 시각 vs GPU hot 종료, 같은 host 시계; 순서검사 통과) |
| Q1' GPU 에 노출된 대기 | hot 종료→HtoD 시작 p50 10.5 / 14.0 / 14.8 (하한≈bs=2 graph 의 p50) | 7936 | 확인 (GPU 시계 단일) |
| Q2 Cold 경로 어디서 늦나 | 제출→시작 5.5 / 9.0; 계산 33.8 / 189.0; 게시→해제 95.5 / 105.8; 게시→HtoD 완료 99.8 / 110.0; cold 예산(go(L−1)→hot_ready(L)) 338.0 / 355.8 | 7808 | 확인 (제출→시작은 단일 FIFO 대기+입력 준비 포함; 순수 큐 대기로 명명하지 않음) |
| Q3 Hot/다른 GPU 작업 | hot fused_moe 커널 합 p50 74.0 / 81.8 | 7936 | 확인 (커널 이름 기준; attention/router 분해는 미집계=미측정) |
| Q4 NUMA 서브풀 | 완료 시차 p50 4.2 / 8.8 / 18.2; numa0 10.2 numa1 13.5 | 7808 | 확인 (원격 메모리 접근 등 원인 미분리) |
| Q4' layer/expert 집중 | dependency_metrics.csv.gz 의 layer 별 분포 (본 문서 미집계) | | 원자료 제공 |
| Q5 무계측 적용 가능성 | observer_overhead.csv 의 OFF/ON 비율, 반복 3회 | | 부팅 간 비교 한계 |

## 세션 F1_C1_P1 (boot F1_ON_134513) — 표본 coverage {'gpu_layer_rows_total': 126976, 'no_prev_deferred': 2048, 'matched_rows': 126976}, 순서 위반 {}

### graph 35 ['step[DECODE bs=1]'] — 유효 행 126976 (prev 연결 124928)

| 질문 | 관측 (µs, p50 / p95 / p99) | 표본 | 구분 |
|---|---|---|---|
| Q1 결합 시 Hot/Cold 중 늦은 입력 | cold_pub − hot_ready p50 -63.2 / -55.8 / -40.0; cold 가 늦은 비율 0.004; other(잔차) 는 hot 이전에 준비 | 126976 | 확인 (CPU done 시각 vs GPU hot 종료, 같은 host 시계; 순서검사 통과) |
| Q1' GPU 에 노출된 대기 | hot 종료→HtoD 시작 p50 10.8 / 14.0 / 14.8 (하한≈bs=2 graph 의 p50) | 126976 | 확인 (GPU 시계 단일) |
| Q2 Cold 경로 어디서 늦나 | 제출→시작 5.5 / 11.2; 계산 20.5 / 143.0; 게시→해제 74.0 / 80.0; 게시→HtoD 완료 77.0 / 83.5; cold 예산(go(L−1)→hot_ready(L)) 301.2 / 312.0 | 124928 | 확인 (제출→시작은 단일 FIFO 대기+입력 준비 포함; 순수 큐 대기로 명명하지 않음) |
| Q3 Hot/다른 GPU 작업 | hot fused_moe 커널 합 p50 51.8 / 52.5 | 126976 | 확인 (커널 이름 기준; attention/router 분해는 미집계=미측정) |
| Q4 NUMA 서브풀 | 완료 시차 p50 1.0 / 7.0 / 19.0; numa0 7.8 numa1 6.2 | 124928 | 확인 (원격 메모리 접근 등 원인 미분리) |
| Q4' layer/expert 집중 | dependency_metrics.csv.gz 의 layer 별 분포 (본 문서 미집계) | | 원자료 제공 |
| Q5 무계측 적용 가능성 | observer_overhead.csv 의 OFF/ON 비율, 반복 3회 | | 부팅 간 비교 한계 |

## 세션 F1_LONG_P1 (boot F1_ON_134513) — 표본 coverage {'gpu_layer_rows_total': 31744, 'no_prev_deferred': 512, 'matched_rows': 31744}, 순서 위반 {}

### graph 26 ['step[DECODE bs=8]'] — 유효 행 31744 (prev 연결 31232)

| 질문 | 관측 (µs, p50 / p95 / p99) | 표본 | 구분 |
|---|---|---|---|
| Q1 결합 시 Hot/Cold 중 늦은 입력 | cold_pub − hot_ready p50 -184.8 / -96.5 / 26.2; cold 가 늦은 비율 0.013; other(잔차) 는 hot 이전에 준비 | 31744 | 확인 (CPU done 시각 vs GPU hot 종료, 같은 host 시계; 순서검사 통과) |
| Q1' GPU 에 노출된 대기 | hot 종료→HtoD 시작 p50 10.2 / 14.0 / 39.5 (하한≈bs=2 graph 의 p50) | 31744 | 확인 (GPU 시계 단일) |
| Q2 Cold 경로 어디서 늦나 | 제출→시작 6.0 / 86.5; 계산 204.8 / 413.5; 게시→해제 194.8 / 228.2; 게시→HtoD 완료 203.5 / 236.5; cold 예산(go(L−1)→hot_ready(L)) 587.0 / 645.8 | 31232 | 확인 (제출→시작은 단일 FIFO 대기+입력 준비 포함; 순수 큐 대기로 명명하지 않음) |
| Q3 Hot/다른 GPU 작업 | hot fused_moe 커널 합 p50 170.0 / 203.5 | 31744 | 확인 (커널 이름 기준; attention/router 분해는 미집계=미측정) |
| Q4 NUMA 서브풀 | 완료 시차 p50 2.2 / 18.8 / 46.0; numa0 150.0 numa1 149.5 | 31232 | 확인 (원격 메모리 접근 등 원인 미분리) |
| Q4' layer/expert 집중 | dependency_metrics.csv.gz 의 layer 별 분포 (본 문서 미집계) | | 원자료 제공 |
| Q5 무계측 적용 가능성 | observer_overhead.csv 의 OFF/ON 비율, 반복 3회 | | 부팅 간 비교 한계 |

## 세션 S1_P1_PROBE128 (boot SMOKE_131024) — 표본 coverage {'gpu_layer_rows_total': 23808, 'no_prev_deferred': 384, 'matched_rows': 23808}, 순서 위반 {}

### graph 2 ['step[DECODE bs=63]'] — 유효 행 15872 (prev 연결 15616)

| 질문 | 관측 (µs, p50 / p95 / p99) | 표본 | 구분 |
|---|---|---|---|
| Q1 결합 시 Hot/Cold 중 늦은 입력 | cold_pub − hot_ready p50 324.5 / 819.5 / 1050.5; cold 가 늦은 비율 0.896; other(잔차) 는 hot 이전에 준비 | 15872 | 확인 (CPU done 시각 vs GPU hot 종료, 같은 host 시계; 순서검사 통과) |
| Q1' GPU 에 노출된 대기 | hot 종료→HtoD 시작 p50 337.0 / 831.8 / 1061.0 (하한≈bs=2 graph 의 p50) | 15872 | 확인 (GPU 시계 단일) |
| Q2 Cold 경로 어디서 늦나 | 제출→시작 671.8 / 1190.0; 계산 918.0 / 1426.8; 게시→해제 13.2 / 134.2; 게시→HtoD 완료 44.2 / 156.0; cold 예산(go(L−1)→hot_ready(L)) 1269.8 / 1793.8 | 15616 | 확인 (제출→시작은 단일 FIFO 대기+입력 준비 포함; 순수 큐 대기로 명명하지 않음) |
| Q3 Hot/다른 GPU 작업 | hot fused_moe 커널 합 p50 323.8 / 435.2 | 15872 | 확인 (커널 이름 기준; attention/router 분해는 미집계=미측정) |
| Q4 NUMA 서브풀 | 완료 시차 p50 18.0 / 68.5 / 142.0; numa0 887.0 numa1 871.5 | 15616 | 확인 (원격 메모리 접근 등 원인 미분리) |
| Q4' layer/expert 집중 | dependency_metrics.csv.gz 의 layer 별 분포 (본 문서 미집계) | | 원자료 제공 |
| Q5 무계측 적용 가능성 | observer_overhead.csv 의 OFF/ON 비율, 반복 3회 | | 부팅 간 비교 한계 |

### graph 32 ['step[DECODE bs=2]'] — 유효 행 7936 (prev 연결 7808)

| 질문 | 관측 (µs, p50 / p95 / p99) | 표본 | 구분 |
|---|---|---|---|
| Q1 결합 시 Hot/Cold 중 늦은 입력 | cold_pub − hot_ready p50 -83.0 / -64.0 / 36.0; cold 가 늦은 비율 0.014; other(잔차) 는 hot 이전에 준비 | 7936 | 확인 (CPU done 시각 vs GPU hot 종료, 같은 host 시계; 순서검사 통과) |
| Q1' GPU 에 노출된 대기 | hot 종료→HtoD 시작 p50 10.5 / 14.0 / 49.5 (하한≈bs=2 graph 의 p50) | 7936 | 확인 (GPU 시계 단일) |
| Q2 Cold 경로 어디서 늦나 | 제출→시작 7.2 / 22.2; 계산 37.2 / 220.5; 게시→해제 93.2 / 104.2; 게시→HtoD 완료 97.2 / 108.5; cold 예산(go(L−1)→hot_ready(L)) 337.2 / 357.2 | 7808 | 확인 (제출→시작은 단일 FIFO 대기+입력 준비 포함; 순수 큐 대기로 명명하지 않음) |
| Q3 Hot/다른 GPU 작업 | hot fused_moe 커널 합 p50 72.5 / 81.8 | 7936 | 확인 (커널 이름 기준; attention/router 분해는 미집계=미측정) |
| Q4 NUMA 서브풀 | 완료 시차 p50 6.2 / 16.8 / 64.8; numa0 10.2 numa1 15.5 | 7808 | 확인 (원격 메모리 접근 등 원인 미분리) |
| Q4' layer/expert 집중 | dependency_metrics.csv.gz 의 layer 별 분포 (본 문서 미집계) | | 원자료 제공 |
| Q5 무계측 적용 가능성 | observer_overhead.csv 의 OFF/ON 비율, 반복 3회 | | 부팅 간 비교 한계 |

## 미측정·한계

- prefill(eager, graph id 0) 층은 층 타임라인 분절 대상에서 제외 (graph node id 없음). F1 LONG 세션의 prefill 구간 Hot/Cold 지연은 미집계. kt_evt.csv 의 eager 슬롯 레코드(qlen>1) 는 원자료로 보존.
- attention / router / TP collective 의 층별 분해: 커널 이름 수준으로만 트레이스에 존재, 본 보고서 집계 없음 (원 트레이스 서버 보존).
- 층 번호는 memcpy 순번 패턴 (HEURISTIC_LAYER_ID). rank 1~3 의 대기 위치(all-reduce 순번↔attention/MoE) 귀속 미확정.
- OFF/ON 은 부팅 간 비교. clock: CPU CLOCK_REALTIME ↔ kineto host 시계 (CUPTI 정렬 오차 미검증; go ≥ dtoh_end 순서검사로만 확인).
- P2 의 perf/PCM 은 host 전체 값이며 KT 만의 대역폭으로 귀속하지 않음. DRAM 포화 판정 보류.
