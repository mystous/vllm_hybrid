# COMPLETION_STATUS — IDE_071 (2026-09-16T07:52:02.303263+09:00)

## compact (cpu_offload_no_02_compact) — 등록 항목별 종료 상태

| 항목 | 셀 | 종료 상태 | 반복(valid/total) | 비고 |
|---|---|---|---|---|
| 대조군 B3 (재사용) | P2_cf1_skip1_pin1_def8 | REUSED | 3/3 | 12단계 P2 셀, 같은 워크로드 |
| 대조군 B4 (재사용) | R04_d4_epoch | REUSED | 3/3 (SHORT_COLD) | 12단계 P1 셀 |
| S1 | S1_b4_pin | COMPLETED | 1/1 |  |
| S2 | S2_b3_v2_nu5952 | COMPLETED | 1/1 |  |
| S3 | S3_b4_graph64 | COMPLETED | 1/1 |  |
| S4 | S4_b4_kv40960 | COMPLETED | 1/1 |  |
| S5 | S5_b4_chunk2048 | COMPLETED | 1/1 |  |
| B3 | B3_recheck | COMPLETED | 1/1 |  |
| 확인 대상 | S2_b3_v2_nu5952__confirm | COMPLETED | 4/4 |  |
| 연속 부하 | S2_b3_v2_nu5952__load1024 | COMPLETED (attempt a2) | 1/1 |  |
| S6 | S6_b4_combo | NOT_TRIGGERED | — | B4 계열 조건 충족 변경 0개 (selection_trace) |
| 확인 대상 B4 계열 | — | NOT_TRIGGERED | — | 확인 후보 없음 → B4 재확인·확인 3회 생략 (규칙 §4.1 '새 후보 없는 계열은 생략 가능') |
| 선택적 진단 1회 | — | NOT_RUN | — | 선택 항목, 미실행 (예산 잔여) |

## 실행량 (compact)

- 일반 벤치 10/20 (1차 5 + 부모 재확인 1 + 확인 3 + 별도 입력 1) · 연속 부하 1 (시도 2) · 진단 0 · 재시도 1/2 · GSM40 1/2
- 부팅 9/14 · 워밍업 = 성공 부팅당 1회 (C16·32요청) · greedy4 smoke = 성공 부팅당 1세트
- 종료 상태: **BOUNDED_SEARCH_COMPLETE** (탐색 범위 내 등록 항목 처리 완료; 전역 최적·통계적 동등성 판단 아님)

## 12단계 계획 (cpu_offload_no_02) — 07:05 에 compact 로 대체

- 실행 셀 60: COMPLETED 59, FAILED 1, 기타 0 (P1 8, P2 20, P3 20/35, 앵커 재측정 포함)
- 등록 후 미실행 (OUT_OF_SCOPE): 88 셀 (P3 잔여, P4~P7); P8~P11 은 manifest 미생성
- 상세: FULL_REPORT.md §3.2~§3.3

