# COMPLETION_STATUS — IDE_074 (2026-09-17T14:06:01+09:00)

- 캠페인 상태: COMPLETED_WITH_FAILURES
- 실행량: 세션 17/24 · 부팅 7/12 · 품질 문항 4/80

| 작업 | 최종 상태 |
|---|---|
| M0 | 완료 (SOURCE_INDEX.md, evidence/) |
| M1 | 완료 (gpu_union.py selftest PASS, corrected_metrics.csv) |
| M2 | 완료 (harness.py: arm→요청→drain→종료, observer_windows.csv) |
| M3 | 완료 (probe_map.md, kt_evt 스키마, correlation 검증) |
| M4 | 완료 (CPU 이벤트·GPU 트레이스·perf/pcm) |
| M5 | PASS (validation_results) |
| M6 | 세션 17/24 (OFF 8, P1 6, P2 3) |
| M7 | dependency_summary 6개 세션 |
| M8 | BLOCKED_NORMAL_OUTPUT |
| M9 | 본 문서 생성 시점 기준 |

## 최종 체크리스트 (§13)

| 항목 | 상태 |
|---|---|
| Qwen OPT4 고정 구성·모델/manifest/실행 코드 해시 | evidence/environment.json, hotmap_budget.json, requested_config.json |
| GPU duration sum/union/gap 분석기 수정·합성 단위시험 | PASS (gpu_union.py) |
| 요청·수집기 시간창 정렬·장치별 유효 구간 | observer_windows.csv, validation window |
| layer·step·rank·NUMA·job·ring epoch·graph replay 상관관계 | (slot,epoch)↔(layer,replay) 매칭·순서검사 (validation correlation_clock) |
| Hot/Cold/other ready·combine 연결 | dependency_metrics (other=잔차, hot 이전 준비) |
| CPU 큐/연산/완료 전달·전송·GPU 작업 구분 | dependency_metrics / gpu_layer_timeline |
| 같은 PROBE128 OFF/ON 비교·반복성·간섭·누락률 | observer_overhead.csv, missing_coverage.csv |
| 실제 scheduler phase 사용 | step[...] annotation |
| GLM 조건부 | BLOCKED_NORMAL_OUTPUT |
| 최종 상태·원시 자료·해시·게시·전달 | ARTIFACT_INDEX.csv, SHA256SUMS.txt, PUBLISH_RECEIPT.md |
