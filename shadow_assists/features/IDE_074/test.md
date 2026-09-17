# IDE_074 test — 검증 방법과 기대 결과

| 검증 | 방법 | 기대 | 실패 시 상태 |
|---|---|---|---|
| M1 합성 단위시험 | `python3 eval/ide074/gpu_union.py` | `SELFTEST PASS` (§4.3 8건) | INVALID_AGGREGATION |
| M1 재집계 일관성 | corrected_metrics.csv 의 `gpu_busy_union_us ≤ window_len_us`, busy+idle = window | 전 행 통과 | INVALID_AGGREGATION |
| M2 시간창 | observer_windows.csv 의 `request_coverage_fraction` 와 수집기 exit_code | 주 수집기 coverage 1.0, exit 0 | INVALID_WINDOW / 부분 측정 |
| M3 correlation | 표본 job 마다 layer/step/rank/epoch/결합 단위 연결 수 | 누락 개수 보고 | job 제외 |
| M3 clock | CPU/GPU 이벤트 domain·변환 검증 (kineto base + host epoch 대조, 오차 기록) | 오차 범위 저장 | UNMEASURED_CLOCK_ALIGNMENT |
| M5 GPU activity | smoke 트레이스에 kernel 이벤트·copy 경로 존재 | 존재 | INVALID_NO_GPU_ACTIVITY |
| M5 실행 정상성 | smoke/warmup 성공, OFF/ON 동일 smoke 출력 token IDs 비교 | 동일 또는 차이 원인 조사 | 본 측정 중단 |
| M6 간섭 | throughput/elapsed/latency ratio (P1,P2 vs 같은 PROBE128 P0), 반복 3회 원값·SD | 비율·한계 보고 (허용 기준 사전 명시) | 샘플링 하향·분리 재시도 |
| M7 순서 검사 | `combine_start < required_input_ready` 건수 | 0 또는 원인(clock/graph mapping) 기록 | max(0) 로 숨기지 않음 |
