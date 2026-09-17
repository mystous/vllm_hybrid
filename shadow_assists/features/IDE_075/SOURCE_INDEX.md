# SOURCE_INDEX — IDE_075 (어떤 주장·식·수치가 어떤 자료에서 왔는가)

| 항목 | 출처 |
|---|---|
| 잔여 예산 (세션 7·부팅 5·품질 76) | `eval/results/IDE_074_20260917/state/execution_events.jsonl` 재계산 (`state/budget_reconciliation.json`) |
| PCM 정정값 (237.0/251.0/247.8 GB/s) | `offline/pcm_window_summary_v2.csv` ← IDE_074 `B3_133528/P2_r*/pcm_memory.csv` + `~/.cache/huggingface/kt/ide074/B3_133528/kt_evt.csv` (forward_envelope) |
| 시간창 7종 | `offline/legacy_windows.csv` ← IDE_074 metrics.json, 트레이스 step annotation, kt_evt |
| v2 지표 정의 | `METRIC_DEFINITIONS_V2.yaml`; 구현 `eval/ide075/dependency_v2.py` |
| IDE_074 P1 재집계 (cohort·민감도) | `eval/results/IDE_074_20260917/qwen/OPT4/*/*/v2/dependency_summary_v2.json` |
| 기록기 v2 코드 | `eval/ide075/kt_evt_patch_v2.py`, 적용 diff `evidence/kt_evt_patch_v2.diff`, .so SHA a4e9045a… (`evidence/environment.json`) |
| 합성 시험 | `eval/ide075/tests/test_recorder.cpp`, 결과 `evidence/parser_test_results.json`; `eval/ide075/tsparse.py` selftest |
| 세션 원값 (S1~S6) | `eval/results/IDE_075_20260917/qwen/OPT4/{OFF_A_152830,OFF_B_153208,CORE_153555}/*/metrics.json` |
| FIFO 분해·casebook·층별 비용표 | `CORE_153555/S3_CORR/v2/`, `S4_CORR/v2/` (`queue_wait_breakdown.csv.gz`, `fifo_casebook.md`, `producer_layer_costs.csv`, `expert_cost_samples.csv.gz`) |
| 자원 (perf -I, PCM) | `CORE_153555/S5_RESOURCE/v2/resource_summary.json`, `cpu_counter_windows.csv`, `pcm_samples_v2.csv` |
| 유효성 | `*/v2/validation_results_v2.json`, `VALIDATION_REPORT.md`, `profiles/clock_alignment_summary.csv` |
| 계측 간섭 | `profiles/observer_overhead_v2.csv` (기준 `PLAN_RESOLVED.md §6`) |
| 서비스↔expert 수 상관 (r=0.925, 65 µs/expert) | `S3_CORR/v2/producer_consumer_metrics_v2.csv.gz` (cold_present_nonempty, bs=63) 단순 선형회귀 (READINESS 작성 시 계산) |
| 역사값 (IDE_074) | `shadow_assists/features/IDE_074/FULL_REPORT.md` (게시 커밋 87dee8e55) |
| 코드 위치 (probe) | `PROBE_MAP_V2.md`, IDE_074 `profiles/code_map_survey.md` |
| 서버 보존 대형 자료 | `~/.cache/huggingface/kt/ide075/<boot>/{kt_evt.csv, .tasks, .map, profiles/*.trace.json.gz}` — `ARTIFACT_INDEX.csv` SHA256 |
