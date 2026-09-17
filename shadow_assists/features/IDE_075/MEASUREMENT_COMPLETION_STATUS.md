# MEASUREMENT_COMPLETION_STATUS — IDE_075 (2026-09-17T15:58:37+09:00)

| 작업 | 상태 | 근거 |
|---|---|---|
| A00 근거·코드·예산 | 완료 | PLAN_RESOLVED §1, budget_reconciliation.json |
| A01 PCM·시간창·GPU union·분모 정정 | 완료 (perf 구간값 NOT_RECOVERABLE, client_request_window ESTIMATED) | WINDOW_RECONSTRUCTION.md, legacy_vs_corrected.csv |
| A02 v2 지표 | 완료 | METRIC_DEFINITIONS_V2.yaml, 세션/v2 |
| A03 저간섭 기록기 | 완료 (task 0 추가, 합성 시험 PASS) | kt_evt_patch_v2.py, parser_test_results.json |
| A04 barrier·clock·TID | 부분: 클라이언트 barrier 미지원(vllm bench serve) → 서버측 이벤트로 사후 절단; clock 은 순서검사·오차구간 | window_events.jsonl, clock_alignment_summary.csv |
| A05 producer/consumer·FIFO | 완료 | queue_wait_breakdown, fifo_casebook |
| A06 expert rows·분기·단계 | 완료 (작업 단위; expert 별 rows 는 미수집) | producer_layer_costs.csv, expert_cost_samples.csv.gz |
| A07 부하 창 CPU·DRAM | 완료 (perf -I 1000 interval, PCM v2) | resource_summary.json |
| A08 GPU 전송·결합 | 완료 (HtoD/DtoH·combine·idle-in-gap); TP collective 심화 미실행(조건부) | gpu_layer_timeline, v2 |
| A09 prefill·C1·tail | 미실행 (조건부; 예산 없음) | — |
| A10 정상성 | smoke 4 요청 부팅당, lifecycle/task_count 검증 | validation_results_v2 |
| A11 OFF/CORE/CORR/RESOURCE | 세션 6/6 (+예비 0) | FULL_REPORT §2·3 |
| A12 판정·인계 | READINESS.md, OPTIMIZATION_HANDOFF.md | — |
| G00 GLM | GLM_NORMAL_OUTPUT_BLOCKED (실행 없음) | IDE_074 gate_result.json |

예산: 세션 23/24 · 부팅 12/12 (IDE_075 시도 5, 중단 1) · 품질 4/80
