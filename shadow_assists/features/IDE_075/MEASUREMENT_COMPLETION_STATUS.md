# MEASUREMENT_COMPLETION_STATUS — IDE_075 (2026-09-17T20:53:53+09:00)

| 작업 | 상태 | 근거 |
|---|---|---|
| A00 근거·코드·예산 | 완료 | PLAN_RESOLVED §1, budget_reconciliation.json |
| A01 PCM·시간창·GPU union·분모 정정 | 완료 (perf 구간값 NOT_RECOVERABLE, client_request_window ESTIMATED) | WINDOW_RECONSTRUCTION.md, legacy_vs_corrected.csv |
| A02 v2 지표 | 완료 | METRIC_DEFINITIONS_V2.yaml, 세션/v2 |
| A03 저간섭 기록기 | 완료 (task 0 추가, 합성 시험 PASS) | kt_evt_patch_v2.py, parser_test_results.json |
| A04 barrier·clock·TID | 완료 (확장): vllm bench 래퍼(vllmw) 로 ready/start barrier + perf_counter↔REALTIME anchor + 요청별 start/ttft/itl (S21b, CORE4); clock anchor 실측 (clock_anchors.jsonl, GPU→host offset ≤ +3.5 µs); TID 역할 = 스레드 이름(v3) + affinity | window_events.jsonl, *.anchor.json, clock_alignment_summary.csv, cpu_time_by_role.json |
| A05 producer/consumer·FIFO | 완료 | queue_wait_breakdown, fifo_casebook |
| A06 expert rows·분기·단계 | 완료 (작업 단위 + v3 expert 단위 rows 표본 1/16) | producer_layer_costs.csv, expert_cost_samples.csv.gz, expert_rows_summary.json |
| A07 부하 창 CPU·DRAM | 완료 (perf -I 1000 interval, PCM v2) | resource_summary.json |
| A08 GPU 전송·결합 | 완료 (HtoD/DtoH·combine·idle-in-gap) + TP collective 위치·rank·도착 시차 (확장) | gpu_layer_timeline, v2, tp_collective_summary.json |
| A09 prefill·C1·tail | 완료 (확장): prefill/EXTEND 층 분절(eager_summary), C1(bs=1) 의존성(S12·S26, anchor 재동기화), tail off-CPU(FOCUS S14/S19/S25, perf sched) | eager_summary.json, S12/S26 v2, tail_offcpu_summary.json |
| A10 정상성 | smoke 4 요청 부팅당, lifecycle/task_count 검증 | validation_results_v2 |
| A11 OFF/CORE/CORR/RESOURCE | 세션 6/6 (+예비 0) | FULL_REPORT §2·3 |
| A12 판정·인계 | READINESS.md, OPTIMIZATION_HANDOFF.md | — |
| G00 GLM | 확장: 진단 D1~D5(전부 비정상) → 커널 bisect → 근본 원인(moe_base.hpp if-constexpr dangling else, TSK_060) → 수정·재빌드 → D6/D7 게이트(전 GPU 기준 대조) → D8 세션(G_OFF2/G_CORR2/G_OFF3) → D9 callback-free 세션(G_CORR3: CPU 레코드 + 트레이스, G_OFF4) (§8c) | G00_code_review.md §8, glm/BASIC4/GLM_D*/ |

예산: 세션 23/24 · 부팅 12/12 (IDE_075 시도 5, 중단 2) · 품질 4/80
