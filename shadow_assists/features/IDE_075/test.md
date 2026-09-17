# IDE_075 test

| 검증 | 방법 | 결과 |
|---|---|---|
| T01~T07, W01~W06 (부록 B.1) | `python3 eval/ide075/tsparse.py` | PASS |
| G01~G05 (부록 B.2 GPU union) | IDE_074 `gpu_union.py` selftest 재사용 | PASS |
| Q01/Q02/Q11/Q12, SPSC stale, 순서 (부록 B.3) | `tests/test_recorder.cpp` (g++ -std=c++20, 컨테이너) | PASS (evidence/parser_test_results.json) |
| 세션 유효성 (§13.4 항목별) | `validate_v2.py` | S1/S6 all pass; S2·S4·S5 lifecycle fwd 미기록 5/2/1 (≤0.1 % 기준 통과, 건수 공개); S3 all pass |
| 매핑·clock | dependency_v2 coverage/clock_validity | 불일치 0, 위반 0, CLOCK_INDETERMINATE 0.7~0.8 % |
| 항등식 residual | dependency_v2 | max 0 |
| 계측 간섭 기준 (§15.3) | observer_overhead_v2.csv | CORE/CORR DESCRIPTIVE_LOW_DISTORTION, RESOURCE DIAGNOSTIC_ONLY, OFF 자체 0.9 % |
