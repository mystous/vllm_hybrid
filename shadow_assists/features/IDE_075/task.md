# IDE_075 task

| 작업 | 내용 | 산출 | 상태 |
|---|---|---|---|
| A00 | 예산·환경·원자료 고정 | PLAN_RESOLVED §1, evidence/, budget_reconciliation.json | 완료 |
| A01 | PCM 시각·시간창·GPU union·분모 정정 | tsparse.py, reconstruct_windows.py, WINDOW_RECONSTRUCTION.md, legacy_vs_corrected.csv | 완료 (perf 구간값 NOT_RECOVERABLE) |
| A02 | v2 지표·cohort·민감도·항등식 | METRIC_DEFINITIONS_V2.yaml, dependency_v2.py | 완료 |
| A03 | 기록 전용 task 없는 기록기 | kt_evt_patch_v2.py, tests/test_recorder.cpp, PROBE_MAP_V2.md | 완료 |
| A04 | 창·clock·TID | harness.py(window_events, thread_role_map), clock_alignment_summary.csv | 부분 (클라이언트 barrier 미지원) |
| A05 | producer/consumer·FIFO 분해 | analyze_costs_v2.py (queue_wait_breakdown, edges, casebook) | 완료 |
| A06 | expert rows·분기·단계 | producer_layer_costs.csv, expert_cost_samples.csv.gz | 완료 (작업 단위) |
| A07 | 부하 창 perf/PCM | resource_v2.py | 완료 |
| A08 | GPU 전송·결합·idle | dependency_v2 (gpu_idle_inside_gap 등) | 완료 (TP 심화 미실행) |
| A09 | prefill·C1·tail | — | 미실행 (조건부·예산 없음) |
| A10 | 정상성·유효성 | validate_v2.py, VALIDATION_REPORT.md | 완료 |
| A11 | OFF/CORE/CORR/RESOURCE | S1~S6 | 완료 (예비 S7 없음) |
| A12 | 판정·인계 | READINESS.md, OPTIMIZATION_HANDOFF.md, MEASUREMENT_COMPLETION_STATUS.md | 완료 |
| G00 | GLM | — | GLM_NORMAL_OUTPUT_BLOCKED |
