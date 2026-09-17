# IDE_074 task — 단계별 구현

| 작업 | 내용 | 산출 | 상태 |
|---|---|---|---|
| M0 | 브랜치 `feat/cpu-moe-bottleneck-20260917`, ID 등록, 환경·원자료·고정 구성 수집 (`eval/ide074/m0_collect.py`) | SOURCE_INDEX.md, evidence/environment.json, evidence/hotmap_budget.json, evidence/effective_config_reference.json | 진행 |
| M1 | `eval/ide074/gpu_union.py` (§4.2 정의 + §4.3 합성 단위시험) / `recompute_d2.py` (IDE_073 D2 retry2 TP0~3 재집계, trace 전체 창·요청 창) | profiles/corrected_metrics.csv, inter_moe_gaps, meta | 진행 |
| M2 | 하네스: 수집기 arm → 준비 확인 → 시작 barrier → 요청 → drain 확인 → 수집기 종료(생성 PID 만, wait) / observer_windows.csv | `eval/ide074/harness.py`, observer_windows.csv | 대기 |
| M3 | 코드 지도(probe_map.md: file::function::line, timestamp 위치, 완료 의미, clock domain, capture/replay) + 이벤트 스키마 + correlation 검증 | profiles/probe_map.md, event_schema.md | 대기 |
| M4 | CPU job 이벤트(submit/start/done/publish, NUMA subjob), 전송 이벤트, GPU hot/attention/router/combine/collective 구분, 자원 시계열 | stage_events.jsonl 등 | 대기 |
| M5 | smoke 내 계측 정상성 검증표 (§8) | validation_results.json | 대기 |
| M6 | B×3, P0×3, P1×3, P2×3 (+F1/F2/G/R 조건부), 간섭 비율 | observer_overhead.csv, 반복 원값 | 대기 |
| M7 | lateness·Cold 내부·NUMA skew·전송 집계 (§10) | dependency_metrics.csv, missing_coverage.csv, BOTTLENECK_EVIDENCE.md | 대기 |
| M8 | GLM native FP8 정상성 게이트(4문항) → G OFF/ON 또는 BLOCKED_NORMAL_OUTPUT | glm/ | 대기 |
| M9 | FULL_REPORT/FULL_RAW_DATA/COMPLETION_STATUS/WORK_LOG, 색인·해시, 게시·전달 | 전부 | 대기 |
