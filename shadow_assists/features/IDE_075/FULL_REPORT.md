# FULL_REPORT — IDE_075 후속 추가 측정 (2026-09-17T15:58:37+09:00)

지시서 `PLAN.md` (SHA 8a3225f4ad0eff8f…). 실행 확정 `PLAN_RESOLVED.md`, 조사 기록 `WORK_LOG.md`, 창 재구성 `WINDOW_RECONSTRUCTION.md`, 지표 정의 `METRIC_DEFINITIONS_V2.yaml`, 검증 `VALIDATION_REPORT.md`, 판정 `READINESS.md`, 인계 `OPTIMIZATION_HANDOFF.md`. 원인 추정·개선 권고는 본 문서에 없음.

## 1. 환경·구성

```
host violet-h100-016 kernel 5.14.0-427.124.1.el9_4.x86_64 no_turbo=1 governor=performance
kt_kernel_ext.so v2 (IDE_075): a4e9045a038bc7af2ec81b1cf251b007399d7e998a5d58c86f579c1bd1bd6ed3 8319008 B  (v1 adf49e48… / 원본 e29357f7… 은 /sgl-workspace/ide074_backup)
launch argv / env / hotmap / layer budget: IDE_074 와 동일 (CONFIG_DIFF.md)
```

부팅 시도 (원장):

| boot_id | plan/mode | verdict | 비고 |
|---|---|---|---|
| OFF_OPEN_152426 | OFF_OPEN/OFF | HEALTH_OK |  |
| CORE_152703 | CORE/CORE | ABORTED_HARNESS_BUG | ATTACH 분기 전 PLANS 조회 KeyError → 체인의 다음 plan(OFF_A) 의 stop_server 가 로딩 중이던 서버를 종료. 부팅 시도로 계수 |
| OFF_A_152830 | OFF_A/OFF | HEALTH_OK |  |
| OFF_B_153208 | OFF_B/OFF | HEALTH_OK |  |
| CORE_153555 | CORE/CORE | HEALTH_OK |  |

실행량: IDE_074 세션 17/부팅 7 + IDE_075 세션 6/부팅 시도 5 (중단 1) → 합계 세션 23/24 · 부팅 12/12 · smoke 요청 16 · warmup 요청 128 (`state/budget_reconciliation.json`)

## 2. 모든 세션 (원값)

| boot | session | mode | workload | C | n | 완료/실패 | dur s | in | out | out tok/s | TTFT p50/p95 | TPOT p50/p95 | E2EL p50/p95 | valid | .so | start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CORE_153555 | S2_CORE | CORE | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.1 | 65886 | 16384 | 708.22 | 2236/3261 | 64.5/70.5 | 10492/10515 | True | a4e9045a038b | 2026-09-17T15:38:42.908276+09:00 |
| CORE_153555 | S3_CORR | CORR | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.4 | 65886 | 16384 | 699.09 | 2283/3223 | 64.1/72.1 | 10547/10675 | True | a4e9045a038b | 2026-09-17T15:39:26.427098+09:00 |
| CORE_153555 | S4_CORR | CORR | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.4 | 65886 | 16384 | 701.50 | 2258/3205 | 65.0/71.0 | 10576/10591 | True | a4e9045a038b | 2026-09-17T15:41:30.323881+09:00 |
| CORE_153555 | S5_RESOURCE | RESOURCE | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 21.0 | 65886 | 16384 | 781.56 | 2174/3144 | 57.2/68.6 | 9450/10149 | True | a4e9045a038b | 2026-09-17T15:43:38.926657+09:00 |
| OFF_A_152830 | S1_OFF_A | OFF | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.4 | 65886 | 16384 | 700.86 | 2339/3213 | 63.0/73.9 | 10788/10802 | True | a4e9045a038b | 2026-09-17T15:31:11.799733+09:00 |
| OFF_B_153208 | S6_OFF_B | OFF | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.2 | 65886 | 16384 | 706.89 | 2271/3176 | 64.8/73.8 | 10468/10524 | True | a4e9045a038b | 2026-09-17T15:34:48.726118+09:00 |

역사적 기준(IDE_074, 재측정 없이 계승; 새 분모로 쓰지 않음): MAIN_SHORT OFF 743.02/747.53/768.93, PROBE128 OFF 720.80/780.72/732.61, P1 696.39/705.62/690.76, P2 706.46/720.67/705.25 tok/s (v1 바이너리).

## 3. 계측 간섭 (profiles/observer_overhead_v2.csv; OFF 평균 및 OFF 원값 각각 대비; 기준: |처리량|≤3 %, p95 ≤5 %)

| session | mode | ref | throughput_ratio | duration_ratio | e2el_p50 | ttft_p95 | tpot_p95 | status |
|---|---|---|---|---|---|---|---|---|
| S2_CORE | CORE | OFF_mean | 1.006 | 0.994 | 0.987 | 1.021 | 0.955 | DESCRIPTIVE_LOW_DISTORTION |
| S2_CORE | CORE | S1_OFF_A | 1.011 | 0.990 | 0.973 | 1.015 | 0.954 | DESCRIPTIVE_LOW_DISTORTION |
| S2_CORE | CORE | S6_OFF_B | 1.002 | 0.998 | 1.002 | 1.027 | 0.955 | DESCRIPTIVE_LOW_DISTORTION |
| S3_CORR | CORR | OFF_mean | 0.993 | 1.007 | 0.992 | 1.009 | 0.976 | DESCRIPTIVE_LOW_DISTORTION |
| S3_CORR | CORR | S1_OFF_A | 0.997 | 1.003 | 0.978 | 1.003 | 0.975 | DESCRIPTIVE_LOW_DISTORTION |
| S3_CORR | CORR | S6_OFF_B | 0.989 | 1.011 | 1.008 | 1.015 | 0.977 | DESCRIPTIVE_LOW_DISTORTION |
| S4_CORR | CORR | OFF_mean | 0.997 | 1.003 | 0.995 | 1.003 | 0.961 | DESCRIPTIVE_LOW_DISTORTION |
| S4_CORR | CORR | S1_OFF_A | 1.001 | 0.999 | 0.980 | 0.997 | 0.960 | DESCRIPTIVE_LOW_DISTORTION |
| S4_CORR | CORR | S6_OFF_B | 0.992 | 1.008 | 1.010 | 1.009 | 0.961 | DESCRIPTIVE_LOW_DISTORTION |
| S5_RESOURCE | RESOURCE | OFF_mean | 1.110 | 0.901 | 0.889 | 0.984 | 0.929 | DIAGNOSTIC_ONLY |
| S5_RESOURCE | RESOURCE | S1_OFF_A | 1.115 | 0.897 | 0.876 | 0.979 | 0.928 | DIAGNOSTIC_ONLY |
| S5_RESOURCE | RESOURCE | S6_OFF_B | 1.106 | 0.904 | 0.903 | 0.990 | 0.929 | DIAGNOSTIC_ONLY |

OFF 자체 변동 (max/min−1): 0.009 → 3 % 이내. 단발 CORE/RESOURCE 에 SD/CI 없음. OFF 두 부팅은 시간순으로 CORE 부팅 앞에 위치 (OFF_OPEN/OFF_CLOSE 배치 미충족, WORK_LOG).

## 4. 유효성 (세션별 v2/validation_results_v2.json, 항목 분리)

| session | mode | execution_valid | workload_valid | config_valid | window_valid | lifecycle_valid | task_count_valid | sampling_valid | gpu_activity_valid | mapping_valid | clock_valid_by_pair | producer_consumer_valid | counter_running_valid | counter_scope_valid | artifact_valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2_CORE | CORE | True | True | True | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | True |
| S3_CORR | CORR | True | True | True | True | True | True | True | True | True | True | True | n/a | n/a | True |
| S4_CORR | CORR | True | True | True | True | True | True | True | True | True | True | True | n/a | n/a | True |
| S5_RESOURCE | RESOURCE | True | True | True | True | True | True | True | n/a | n/a | n/a | n/a | True | True | True |
| S1_OFF_A | OFF | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | True |
| S6_OFF_B | OFF | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | True |

## 5. v2 의존 지표 (CORR 세션; cohort cold_present_nonempty; µs)

### S3_CORR — coverage {'gpu_layer_rows_total': 23808} · clock {'OK': 23808}

| graph:cohort | n | 지표 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | cold_pub_delta_us | 321.0 | 791.0 | 1013.5 | 5509.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | cold_pub_lateness_us | 321.0 | 791.0 | 1013.5 | 5509.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | gpu_pre_h2d_gap_us | 334.2 | 804.5 | 1027.2 | 5521.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | gpu_idle_inside_gap_us | 334.0 | 804.5 | 1027.2 | 5521.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | h2d_duration_us | 30.8 | 33.8 | 35.2 | 44.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | gpu_post_h2d_gap_us | 7.2 | 9.5 | 10.0 | 15.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | combine_schedule_gap_us | 7.2 | 9.5 | 10.0 | 15.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | go_to_enqueue_us | 3.2 | 5.2 | 7.2 | 511.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | enqueue_to_start_us | 663.5 | 1158.0 | 1389.8 | 5845.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | predecessor_exec_union_us | 662.8 | 1157.5 | 1389.2 | 5844.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | predecessor_deferred_exec_us | 661.8 | 1157.0 | 1389.0 | 5844.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | queue_gap_unattributed_us | 0.8 | 1.2 | 3.2 | 33.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | deferred_service_span_us | 911.5 | 1398.0 | 1628.0 | 6084.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | task_exec_span_us | 911.8 | 1398.5 | 1628.5 | 6085.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | numa0_span_us | 884.2 | 1365.0 | 1575.8 | 6046.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | numa1_span_us | 858.2 | 1321.8 | 1524.0 | 5322.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | numa_completion_skew_us | 24.0 | 72.2 | 143.2 | 4907.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | producer_end_to_pub_us | 0.5 | 0.8 | 7.8 | 902.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | pub_to_h2d_start_us | 13.2 | 98.5 | 320.0 | 533.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | overlap_budget_us | 1261.5 | 1769.5 | 2003.8 | 6494.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | n_cold_assignments | 16.0 | 31.0 | 40.0 | 99.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15598 | go_minus_dtoh_end_us | 5.2 | 6.5 | 9.5 | 1484.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | cold_pub_delta_us | -325.5 | -140.2 | -135.5 | -135.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | gpu_pre_h2d_gap_us | 10.2 | 13.8 | 14.2 | 14.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | gpu_idle_inside_gap_us | 10.2 | 13.8 | 14.2 | 14.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | h2d_duration_us | 21.0 | 27.2 | 29.8 | 29.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | gpu_post_h2d_gap_us | 4.0 | 8.8 | 9.5 | 9.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | combine_schedule_gap_us | 4.0 | 8.8 | 9.5 | 9.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | go_to_enqueue_us | 5.8 | 7.8 | 10.2 | 10.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | enqueue_to_start_us | 70.2 | 512.2 | 732.0 | 732.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | predecessor_exec_union_us | 66.0 | 511.5 | 730.5 | 730.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | predecessor_deferred_exec_us | 0.0 | 445.0 | 660.0 | 660.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | queue_gap_unattributed_us | 3.8 | 5.0 | 6.0 | 6.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | deferred_service_span_us | 63.5 | 68.2 | 82.0 | 82.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | task_exec_span_us | 63.8 | 68.8 | 82.2 | 82.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | numa0_span_us | 34.2 | 40.8 | 50.8 | 50.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | numa1_span_us | 37.5 | 40.0 | 55.2 | 55.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | numa_completion_skew_us | 4.5 | 6.0 | 6.2 | 6.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | producer_end_to_pub_us | 450.5 | 548.8 | 554.2 | 554.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | pub_to_h2d_start_us | 328.8 | 414.0 | 480.5 | 480.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | overlap_budget_us | 952.0 | 1121.0 | 1260.2 | 1260.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 18 | go_minus_dtoh_end_us | 5.2 | 5.8 | 5.8 | 5.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_delta_us | -184.2 | 266.2 | 580.2 | 1454.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_lateness_us | 0.0 | 266.2 | 580.2 | 1454.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_pre_h2d_gap_us | 11.5 | 286.8 | 592.0 | 1474.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_idle_inside_gap_us | 11.5 | 286.8 | 587.5 | 1474.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | h2d_duration_us | 22.2 | 34.0 | 35.0 | 39.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_post_h2d_gap_us | 5.5 | 9.2 | 10.0 | 10.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | combine_schedule_gap_us | 5.5 | 9.2 | 10.0 | 10.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | pub_to_h2d_start_us | 192.5 | 483.8 | 504.8 | 529.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | go_minus_dtoh_end_us | 5.0 | 6.0 | 6.8 | 15.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | cold_pub_delta_us | -85.5 | -66.8 | 32.5 | 744.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | cold_pub_lateness_us | 0.0 | 0.0 | 32.5 | 744.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | gpu_pre_h2d_gap_us | 10.8 | 14.0 | 49.2 | 755.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | gpu_idle_inside_gap_us | 10.8 | 14.0 | 49.2 | 755.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | h2d_duration_us | 4.0 | 5.0 | 5.8 | 7.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | gpu_post_h2d_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | combine_schedule_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | go_to_enqueue_us | 2.8 | 4.5 | 5.5 | 25.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | enqueue_to_start_us | 4.0 | 10.5 | 71.5 | 1119.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | predecessor_exec_union_us | 0.0 | 0.2 | 69.2 | 385.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | predecessor_deferred_exec_us | 0.0 | 0.0 | 69.0 | 385.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | queue_gap_unattributed_us | 3.8 | 7.2 | 13.2 | 1118.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | deferred_service_span_us | 144.0 | 218.8 | 334.8 | 1065.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | task_exec_span_us | 144.5 | 219.0 | 334.8 | 1066.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | numa0_span_us | 122.0 | 195.8 | 313.8 | 529.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | numa1_span_us | 125.2 | 197.0 | 313.2 | 1049.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | numa_completion_skew_us | 5.0 | 16.5 | 33.2 | 938.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | producer_end_to_pub_us | 99.0 | 119.0 | 126.0 | 139.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | pub_to_h2d_start_us | 96.2 | 106.0 | 109.5 | 111.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | overlap_budget_us | 339.0 | 356.2 | 363.5 | 1389.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | n_cold_assignments | 1.0 | 2.0 | 4.0 | 6.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2220 | go_minus_dtoh_end_us | 4.5 | 5.5 | 6.5 | 282.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | cold_pub_delta_us | -85.2 | -70.8 | -64.5 | 1048.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 1048.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | gpu_pre_h2d_gap_us | 10.8 | 13.8 | 14.5 | 1062.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | gpu_idle_inside_gap_us | 10.8 | 13.8 | 14.5 | 1062.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | h2d_duration_us | 4.0 | 4.8 | 5.5 | 6.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | gpu_post_h2d_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | combine_schedule_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | go_to_enqueue_us | 2.8 | 4.2 | 5.8 | 286.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | enqueue_to_start_us | 4.0 | 8.5 | 38.8 | 821.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | predecessor_exec_union_us | 0.0 | 0.2 | 34.2 | 820.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | predecessor_deferred_exec_us | 0.0 | 0.0 | 0.0 | 820.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | queue_gap_unattributed_us | 3.8 | 6.8 | 11.0 | 99.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | deferred_service_span_us | 29.5 | 37.8 | 47.8 | 569.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | task_exec_span_us | 30.0 | 38.0 | 48.2 | 569.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | numa0_span_us | 9.2 | 12.2 | 18.0 | 306.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | numa1_span_us | 12.8 | 16.8 | 22.0 | 71.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | numa_completion_skew_us | 4.8 | 9.0 | 14.8 | 541.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | producer_end_to_pub_us | 217.8 | 232.8 | 239.2 | 1198.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | pub_to_h2d_start_us | 95.8 | 106.0 | 109.0 | 114.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | overlap_budget_us | 341.8 | 359.2 | 366.0 | 1069.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5588 | go_minus_dtoh_end_us | 4.5 | 5.5 | 6.8 | 26.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_delta_us | -61.0 | -34.8 | 62.0 | 118.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_lateness_us | 0.0 | 0.0 | 62.0 | 118.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_pre_h2d_gap_us | 11.2 | 14.0 | 75.5 | 129.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_idle_inside_gap_us | 11.2 | 14.0 | 75.5 | 129.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | h2d_duration_us | 3.8 | 5.0 | 5.2 | 5.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_post_h2d_gap_us | 5.0 | 9.8 | 10.0 | 10.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | combine_schedule_gap_us | 5.0 | 9.8 | 10.0 | 10.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | pub_to_h2d_start_us | 71.2 | 78.8 | 80.8 | 81.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | go_minus_dtoh_end_us | 4.5 | 5.5 | 6.8 | 10.2 |

민감도 (late share, cold_present_nonempty): {"2:cold_present_nonempty": {"late_share_thr_0us": 0.9, "late_share_thr_1us": 0.899, "late_share_thr_2us": 0.899, "late_share_thr_5us": 0.897, "clock_indeterminate_share_5us": 0.008, "n": 15598, "cold_gpu_ready_late_share": 1.0}, "32:cold_present_nonempty": {"late_share_thr_0us": 0.015, "late_share_thr_1us": 0.015, "late_share_thr_2us": 0.015, "late_share_thr_5us": 0.014, "clock_indeterminate_share_5us": 0.001, "n": 2220, "cold_gpu_ready_late_share": 1.0}}

### S4_CORR — coverage {'gpu_layer_rows_total': 23808} · clock {'OK': 23808}

| graph:cohort | n | 지표 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | cold_pub_delta_us | 315.0 | 823.0 | 1055.8 | 2988.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | cold_pub_lateness_us | 315.0 | 823.0 | 1055.8 | 2988.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | gpu_pre_h2d_gap_us | 328.0 | 837.2 | 1068.0 | 2999.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | gpu_idle_inside_gap_us | 327.8 | 837.2 | 1068.0 | 2999.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | h2d_duration_us | 30.5 | 33.5 | 35.0 | 42.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | gpu_post_h2d_gap_us | 7.2 | 9.5 | 10.0 | 11.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | combine_schedule_gap_us | 7.2 | 9.5 | 10.0 | 11.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | go_to_enqueue_us | 3.5 | 6.2 | 8.5 | 536.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | enqueue_to_start_us | 649.8 | 1189.0 | 1424.0 | 3420.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | predecessor_exec_union_us | 648.8 | 1188.2 | 1423.2 | 3419.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | predecessor_deferred_exec_us | 648.2 | 1188.0 | 1423.2 | 3419.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | queue_gap_unattributed_us | 0.8 | 1.5 | 4.5 | 22.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | deferred_service_span_us | 902.2 | 1426.8 | 1659.5 | 3654.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | task_exec_span_us | 902.5 | 1427.2 | 1660.0 | 3655.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | numa0_span_us | 874.2 | 1397.5 | 1621.0 | 3626.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | numa1_span_us | 844.8 | 1344.5 | 1557.8 | 3289.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | numa_completion_skew_us | 24.2 | 77.2 | 169.2 | 1756.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | producer_end_to_pub_us | 0.5 | 1.0 | 13.2 | 745.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | pub_to_h2d_start_us | 13.5 | 107.0 | 368.8 | 538.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | overlap_budget_us | 1245.5 | 1791.8 | 2037.8 | 4023.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | n_cold_assignments | 17.0 | 32.0 | 41.0 | 77.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15596 | go_minus_dtoh_end_us | 5.2 | 6.5 | 7.5 | 1398.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | cold_pub_delta_us | -304.5 | -139.5 | -133.2 | -133.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | gpu_pre_h2d_gap_us | 10.2 | 13.2 | 13.5 | 13.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | gpu_idle_inside_gap_us | 10.2 | 13.2 | 13.5 | 13.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | h2d_duration_us | 20.8 | 27.0 | 42.0 | 42.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | gpu_post_h2d_gap_us | 4.8 | 8.5 | 9.5 | 9.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | combine_schedule_gap_us | 4.8 | 8.5 | 9.5 | 9.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | go_to_enqueue_us | 5.8 | 8.8 | 8.8 | 8.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | enqueue_to_start_us | 83.0 | 880.0 | 1230.8 | 1230.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | predecessor_exec_union_us | 72.5 | 878.2 | 1228.5 | 1228.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | predecessor_deferred_exec_us | 0.0 | 804.5 | 1141.2 | 1141.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | queue_gap_unattributed_us | 4.8 | 11.2 | 12.0 | 12.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | deferred_service_span_us | 68.8 | 83.0 | 86.8 | 86.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | task_exec_span_us | 69.0 | 83.2 | 87.2 | 87.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | numa0_span_us | 33.8 | 46.0 | 53.2 | 53.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | numa1_span_us | 38.2 | 49.2 | 53.8 | 53.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | numa_completion_skew_us | 5.2 | 10.2 | 13.0 | 13.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | producer_end_to_pub_us | 413.0 | 509.8 | 528.8 | 528.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | pub_to_h2d_start_us | 319.0 | 428.5 | 469.2 | 469.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | overlap_budget_us | 912.0 | 1279.2 | 1618.5 | 1618.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 20 | go_minus_dtoh_end_us | 4.8 | 5.8 | 8.2 | 8.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_delta_us | -276.8 | 86.8 | 339.2 | 1075.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_lateness_us | 0.0 | 86.8 | 339.2 | 1075.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_pre_h2d_gap_us | 10.8 | 99.8 | 351.0 | 1084.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_idle_inside_gap_us | 10.8 | 99.8 | 351.0 | 1073.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | h2d_duration_us | 21.8 | 33.2 | 34.5 | 36.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_post_h2d_gap_us | 5.8 | 9.2 | 9.8 | 10.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | combine_schedule_gap_us | 5.8 | 9.2 | 9.8 | 10.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | pub_to_h2d_start_us | 292.2 | 493.8 | 525.2 | 537.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | go_minus_dtoh_end_us | 5.2 | 6.0 | 8.8 | 350.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | cold_pub_delta_us | -83.5 | -61.5 | 42.5 | 267.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | cold_pub_lateness_us | 0.0 | 0.0 | 42.5 | 267.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | gpu_pre_h2d_gap_us | 10.8 | 14.0 | 58.5 | 281.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | gpu_idle_inside_gap_us | 10.8 | 14.0 | 58.5 | 276.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | h2d_duration_us | 4.0 | 5.2 | 5.8 | 6.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | gpu_post_h2d_gap_us | 5.5 | 9.5 | 10.0 | 11.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | combine_schedule_gap_us | 5.5 | 9.5 | 10.0 | 11.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | go_to_enqueue_us | 3.0 | 4.2 | 5.8 | 21.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | enqueue_to_start_us | 3.5 | 12.5 | 117.0 | 338.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | predecessor_exec_union_us | 0.0 | 0.5 | 116.0 | 336.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | predecessor_deferred_exec_us | 0.0 | 0.0 | 116.0 | 336.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | queue_gap_unattributed_us | 3.5 | 6.0 | 8.2 | 49.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | deferred_service_span_us | 143.0 | 222.8 | 330.5 | 512.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | task_exec_span_us | 143.5 | 223.2 | 330.8 | 512.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | numa0_span_us | 121.2 | 198.5 | 307.2 | 489.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | numa1_span_us | 123.8 | 195.0 | 307.0 | 496.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | numa_completion_skew_us | 4.5 | 15.0 | 33.5 | 354.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | producer_end_to_pub_us | 97.2 | 118.5 | 125.2 | 149.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | pub_to_h2d_start_us | 93.8 | 105.2 | 108.2 | 115.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | overlap_budget_us | 335.0 | 355.0 | 370.2 | 599.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | n_cold_assignments | 1.0 | 3.0 | 4.0 | 7.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2297 | go_minus_dtoh_end_us | 4.5 | 5.5 | 6.5 | 278.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | cold_pub_delta_us | -83.2 | -67.0 | -62.2 | 229.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 229.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | gpu_pre_h2d_gap_us | 10.8 | 14.0 | 14.8 | 238.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | gpu_idle_inside_gap_us | 10.8 | 13.8 | 14.8 | 238.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | h2d_duration_us | 4.0 | 4.8 | 5.2 | 6.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | gpu_post_h2d_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | combine_schedule_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | go_to_enqueue_us | 2.8 | 3.8 | 5.2 | 24.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | enqueue_to_start_us | 3.5 | 7.0 | 38.0 | 308.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | predecessor_exec_union_us | 0.0 | 0.2 | 34.0 | 307.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | predecessor_deferred_exec_us | 0.0 | 0.0 | 0.0 | 306.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | queue_gap_unattributed_us | 3.2 | 5.8 | 10.2 | 46.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | deferred_service_span_us | 30.2 | 38.0 | 45.2 | 552.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | task_exec_span_us | 30.8 | 38.5 | 45.5 | 553.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | numa0_span_us | 9.5 | 12.0 | 16.2 | 328.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | numa1_span_us | 13.0 | 17.2 | 21.5 | 33.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | numa_completion_skew_us | 5.0 | 9.0 | 11.5 | 522.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | producer_end_to_pub_us | 213.8 | 230.2 | 235.8 | 274.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | pub_to_h2d_start_us | 93.5 | 105.2 | 108.2 | 112.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | overlap_budget_us | 336.2 | 356.2 | 363.2 | 553.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5511 | go_minus_dtoh_end_us | 4.5 | 5.2 | 6.5 | 273.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_delta_us | -59.2 | -37.2 | -28.0 | -23.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_pre_h2d_gap_us | 11.0 | 14.2 | 15.2 | 15.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_idle_inside_gap_us | 11.0 | 14.2 | 15.2 | 15.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | h2d_duration_us | 3.8 | 5.0 | 6.0 | 6.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_post_h2d_gap_us | 5.2 | 9.2 | 10.0 | 10.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | combine_schedule_gap_us | 5.2 | 9.2 | 10.0 | 10.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | pub_to_h2d_start_us | 69.8 | 77.8 | 79.2 | 79.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | go_minus_dtoh_end_us | 4.5 | 5.5 | 5.8 | 10.2 |

민감도 (late share, cold_present_nonempty): {"2:cold_present_nonempty": {"late_share_thr_0us": 0.899, "late_share_thr_1us": 0.899, "late_share_thr_2us": 0.898, "late_share_thr_5us": 0.896, "clock_indeterminate_share_5us": 0.007, "n": 15596, "cold_gpu_ready_late_share": 1.0}, "32:cold_present_nonempty": {"late_share_thr_0us": 0.017, "late_share_thr_1us": 0.017, "late_share_thr_2us": 0.017, "late_share_thr_5us": 0.017, "clock_indeterminate_share_5us": 0.002, "n": 2297, "cold_gpu_ready_late_share": 1.0}}

## 6. FIFO 대기 분해 (queue_wait_breakdown; 같은 사건 단위, µs)

| session | n | queue_wait_wall p50/p95 | predecessor_exec_union p50/p95 | pred_deferred p50 | pred_done_setter p50 | unattributed p50/p95/max | dequeue→exec p50 | enq bracket p50 |
|---|---|---|---|---|---|---|---|---|
| S3_CORR | 17818 | 607.0/1139.5 | 606.2/1138.5 | 605.8 | 0.0 | 0.8/4.2/1118.5 | 0.0 | 0.6 |
| S4_CORR | 17893 | 593.2/1162.2 | 592.5/1160.8 | 592.0 | 0.0 | 0.8/4.2/50.0 | 0.0 | 0.6 |

## 7. 생산층별 비용표 (producer_layer_costs.csv; 관측값, 한계 이득 아님)

### S3_CORR (bs=63 graph, 층별 p50, µs)

| producer L | n | cold assign | unique n0/n1 | max rows n0 | AMX/AVX n0 | enq→start | pred union | unattrib | service | numa0/1 | skew | pub delta | late share | pre_h2d gap | budget | stages n0 (prep,cpy,q_in,up_gate,act,q_down,down,weight,total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 238 | 13 | 7.0/7.0 | 3.0 | 0.0/21.0 | 386.5 | 386.0 | 1.2 | 599.5 | 573.8/563.8 | 15.2 | -306.5 | 0.101 | 10.5 | 1301.8 | 3,13,22,298,13,5,196,16,573 |
| 1 | 256 | 18 | 13.0/13.0 | 3.0 | 0.0/39.0 | 157.2 | 156.2 | 0.8 | 995.8 | 971.0/932.2 | 31.8 | 14.2 | 0.527 | 33.0 | 1168.5 | 7,15,22,540,21,6,333,17,970 |
| 10 | 256 | 22 | 16.0/16.0 | 3.0 | 0.0/48.0 | 646.2 | 645.5 | 0.8 | 1197.8 | 1172.8/1122.5 | 46.8 | 573.2 | 0.988 | 585.5 | 1287.8 | 7,20,26,660,24,7,404,18,1172 |
| 11 | 256 | 18 | 11.0/11.0 | 3.0 | 0.0/33.0 | 968.0 | 967.5 | 0.8 | 915.5 | 887.5/864.2 | 33.2 | 329.2 | 0.980 | 341.5 | 1570.5 | 6,17,26,486,19,6,299,16,887 |
| 12 | 256 | 16 | 12.0/12.0 | 3.0 | 0.0/36.0 | 679.8 | 678.8 | 0.8 | 921.2 | 894.5/854.2 | 30.8 | 292.0 | 0.949 | 304.2 | 1311.2 | 8,18,24,497,19,6,300,17,893 |
| 13 | 256 | 24 | 16.0/16.0 | 4.0 | 0.0/48.0 | 689.8 | 688.8 | 0.8 | 1227.0 | 1198.8/1151.0 | 38.5 | 622.2 | 1.000 | 638.8 | 1300.0 | 8,20,28,662,25,7,414,18,1198 |
| 14 | 256 | 17 | 12.0/12.0 | 3.0 | 0.0/36.0 | 994.8 | 994.2 | 0.8 | 956.5 | 925.0/893.8 | 27.2 | 378.8 | 1.000 | 390.5 | 1562.5 | 7,20,25,503,20,6,316,18,924 |
| 15 | 256 | 20 | 14.0/14.0 | 3.0 | 0.0/42.0 | 718.8 | 718.2 | 0.8 | 1093.0 | 1064.2/1036.0 | 27.8 | 521.2 | 1.000 | 539.0 | 1290.5 | 7,19,26,592,22,6,360,18,1063 |
| 16 | 256 | 21 | 15.0/15.0 | 3.0 | 0.0/45.0 | 859.5 | 859.0 | 0.8 | 1134.5 | 1109.2/1074.0 | 29.0 | 541.8 | 1.000 | 557.0 | 1439.5 | 7,21,26,621,23,6,373,19,1108 |
| 17 | 256 | 15 | 12.0/12.0 | 2.0 | 0.0/36.0 | 898.5 | 897.8 | 0.8 | 905.5 | 872.2/852.8 | 23.8 | 334.2 | 1.000 | 344.5 | 1495.2 | 7,19,22,477,19,6,295,18,871 |
| 18 | 256 | 20 | 15.0/15.0 | 3.0 | 0.0/45.0 | 671.0 | 670.0 | 0.8 | 1126.2 | 1095.0/1075.5 | 25.2 | 459.8 | 1.000 | 473.5 | 1330.2 | 7,21,26,612,24,6,376,19,1094 |
| 19 | 256 | 15 | 10.0/10.0 | 4.0 | 0.0/30.0 | 890.2 | 889.5 | 0.8 | 819.5 | 792.5/768.2 | 21.8 | 255.5 | 0.977 | 268.0 | 1470.8 | 7,17,26,426,17,5,270,17,791 |
| 2 | 256 | 23 | 14.0/14.0 | 4.0 | 0.0/42.0 | 453.2 | 452.8 | 0.8 | 1088.5 | 1063.0/1019.0 | 33.0 | 352.8 | 0.777 | 363.2 | 1175.0 | 7,16,28,592,23,6,365,17,1062 |
| 20 | 256 | 29 | 18.0/18.0 | 4.0 | 0.0/54.0 | 584.5 | 583.8 | 0.8 | 1376.8 | 1350.0/1314.0 | 37.2 | 783.8 | 1.000 | 795.5 | 1196.8 | 8,22,30,751,28,8,477,19,1349 |
| 21 | 256 | 26 | 16.0/16.0 | 4.0 | 0.0/48.0 | 1139.0 | 1138.2 | 0.8 | 1211.8 | 1188.0/1153.0 | 27.2 | 663.2 | 1.000 | 672.5 | 1711.5 | 7,20,30,657,25,7,410,18,1187 |
| 22 | 256 | 14 | 11.0/11.0 | 2.0 | 0.0/33.0 | 980.5 | 979.8 | 0.8 | 858.5 | 832.2/808.5 | 21.2 | 296.0 | 0.953 | 306.0 | 1532.8 | 7,18,23,444,19,5,284,17,831 |
| 23 | 256 | 19 | 14.0/14.0 | 3.0 | 0.0/42.0 | 618.2 | 617.5 | 0.8 | 1032.5 | 1008.0/981.2 | 25.0 | 459.5 | 0.996 | 469.2 | 1200.0 | 7,19,26,562,22,6,346,18,1007 |
| 24 | 256 | 20 | 13.0/13.0 | 4.0 | 0.0/39.0 | 800.8 | 800.0 | 0.8 | 1031.8 | 1006.0/971.5 | 29.5 | 455.8 | 0.984 | 469.8 | 1378.2 | 7,18,29,544,21,6,348,17,1005 |
| 25 | 256 | 20 | 13.0/13.0 | 4.0 | 0.0/39.0 | 794.0 | 793.0 | 0.8 | 1018.8 | 993.0/973.8 | 26.0 | 463.5 | 1.000 | 473.5 | 1369.5 | 7,21,28,546,21,6,339,19,992 |
| 26 | 256 | 15 | 11.0/11.0 | 3.0 | 0.0/33.0 | 784.0 | 783.2 | 0.8 | 827.5 | 799.0/785.2 | 17.0 | 261.8 | 0.941 | 275.2 | 1350.0 | 7,18,24,433,18,5,270,17,798 |
| 27 | 256 | 17 | 11.0/11.0 | 4.0 | 0.0/33.0 | 595.8 | 595.0 | 0.8 | 892.5 | 867.2/842.5 | 22.8 | 349.2 | 1.000 | 360.5 | 1145.5 | 7,19,27,467,19,6,295,18,866 |
| 28 | 256 | 21 | 14.0/14.0 | 4.0 | 0.0/42.0 | 661.0 | 660.0 | 0.8 | 1093.0 | 1062.2/1039.2 | 24.5 | 541.2 | 1.000 | 554.5 | 1225.5 | 7,21,29,586,23,6,364,19,1061 |
| 29 | 256 | 12 | 9.0/9.0 | 2.0 | 0.0/27.0 | 857.2 | 856.8 | 0.8 | 730.8 | 703.5/681.5 | 18.5 | 174.5 | 0.934 | 186.5 | 1418.8 | 6,17,23,374,16,5,236,17,702 |
| 3 | 256 | 17 | 12.0/12.0 | 3.0 | 0.0/36.0 | 826.8 | 826.0 | 0.8 | 928.0 | 900.2/863.8 | 33.2 | 261.2 | 0.809 | 272.5 | 1499.2 | 6,15,25,493,20,6,310,16,899 |
| 30 | 256 | 21 | 12.0/12.0 | 5.0 | 0.0/36.0 | 496.0 | 495.5 | 0.8 | 984.0 | 953.8/923.2 | 26.0 | 413.0 | 0.996 | 425.5 | 1062.0 | 7,19,31,514,21,6,328,17,953 |
| 31 | 256 | 7 | 5.0/5.0 | 2.0 | 0.0/15.0 | 742.0 | 741.5 | 0.8 | 490.2 | 464.8/445.0 | 13.2 | -76.0 | 0.223 | 11.0 | 1306.0 | 5,14,22,244,11,4,146,15,464 |
| 32 | 256 | 9 | 8.0/8.0 | 2.0 | 0.0/24.0 | 258.2 | 257.5 | 0.8 | 640.2 | 613.5/600.2 | 11.0 | 37.2 | 0.602 | 50.5 | 855.5 | 6,15,17,333,14,5,200,16,612 |
| 33 | 256 | 11 | 9.0/9.0 | 2.0 | 0.0/27.0 | 326.8 | 326.2 | 0.8 | 711.5 | 684.8/668.0 | 13.8 | 159.0 | 0.844 | 171.5 | 875.5 | 6,17,19,366,15,5,229,17,684 |
| 34 | 256 | 10 | 8.0/8.0 | 2.0 | 0.0/24.0 | 464.0 | 463.5 | 0.8 | 684.0 | 658.0/637.5 | 14.0 | 145.8 | 0.941 | 161.0 | 1001.2 | 6,18,20,350,15,5,220,18,657 |
| 35 | 256 | 9 | 8.0/8.0 | 1.0 | 0.0/24.0 | 442.0 | 441.0 | 0.8 | 656.5 | 629.0/617.5 | 13.2 | 114.5 | 0.910 | 129.8 | 990.5 | 6,19,20,342,14,5,200,18,628 |
| 36 | 256 | 16 | 9.0/9.0 | 5.0 | 0.0/27.0 | 422.5 | 421.8 | 0.5 | 789.8 | 761.2/743.8 | 18.8 | 230.2 | 0.992 | 242.0 | 995.5 | 6,18,30,404,17,6,255,17,761 |
| 37 | 256 | 16 | 10.0/10.0 | 4.0 | 0.0/30.0 | 553.0 | 552.5 | 0.8 | 832.2 | 807.2/787.2 | 20.0 | 271.0 | 1.000 | 285.8 | 1108.2 | 7,20,27,429,17,6,275,18,806 |
| 38 | 256 | 24 | 13.0/13.0 | 6.0 | 0.0/39.0 | 599.0 | 598.2 | 0.8 | 1031.8 | 1006.2/976.0 | 25.5 | 413.5 | 1.000 | 424.8 | 1224.5 | 7,20,35,540,21,7,346,17,1005 |
| 39 | 256 | 12 | 8.0/8.0 | 3.0 | 0.0/24.0 | 793.5 | 793.0 | 0.8 | 672.0 | 644.8/623.8 | 18.2 | 95.0 | 0.828 | 105.8 | 1383.0 | 6,17,24,343,15,5,214,16,644 |
| 4 | 256 | 25 | 15.0/15.0 | 5.0 | 0.0/45.0 | 693.2 | 692.2 | 0.8 | 1168.8 | 1145.2/1095.0 | 44.5 | 590.0 | 0.879 | 604.8 | 1279.5 | 7,17,31,637,24,7,399,16,1144 |
| 40 | 256 | 14 | 10.0/10.0 | 4.0 | 0.0/30.0 | 437.0 | 436.5 | 0.8 | 793.8 | 767.8/746.0 | 19.8 | 203.8 | 0.938 | 216.0 | 1031.0 | 7,18,25,415,17,5,255,17,767 |
| 41 | 256 | 11 | 8.0/8.0 | 2.0 | 0.0/24.0 | 554.5 | 554.0 | 0.8 | 698.8 | 669.5/652.0 | 18.5 | 163.2 | 0.898 | 175.5 | 1100.0 | 6,16,23,356,15,5,222,16,668 |
| 42 | 256 | 28 | 16.0/16.0 | 5.0 | 0.0/48.0 | 460.2 | 459.5 | 0.8 | 1255.0 | 1229.2/1196.2 | 31.5 | 676.5 | 1.000 | 689.2 | 1036.0 | 7,20,34,677,26,7,430,17,1228 |
| 43 | 256 | 14 | 10.0/10.0 | 3.0 | 0.0/30.0 | 1019.2 | 1018.8 | 0.8 | 793.5 | 767.5/742.2 | 18.0 | 250.0 | 0.961 | 258.5 | 1572.0 | 7,17,24,420,17,5,250,17,766 |
| 44 | 256 | 18 | 12.0/12.0 | 4.0 | 0.0/36.0 | 562.2 | 561.8 | 0.8 | 936.5 | 911.0/881.8 | 21.8 | 373.2 | 0.910 | 385.5 | 1127.5 | 8,18,26,499,20,6,314,16,910 |
| 45 | 256 | 19 | 12.0/12.0 | 4.0 | 0.0/36.0 | 702.2 | 701.5 | 0.8 | 971.2 | 942.5/917.5 | 21.8 | 377.5 | 0.973 | 389.8 | 1284.2 | 7,17,29,515,20,6,317,17,942 |
| 46 | 256 | 8 | 6.0/6.0 | 2.0 | 0.0/18.0 | 734.8 | 734.2 | 0.5 | 546.5 | 509.0/499.2 | 13.0 | -2.5 | 0.488 | 13.8 | 1289.2 | 6,15,22,265,12,4,163,15,508 |
| 47 | 256 | 17 | 13.0/13.0 | 2.0 | 0.0/39.0 | 308.2 | 307.8 | 0.8 | 992.5 | 964.8/942.2 | 23.0 | 369.0 | 0.945 | 378.5 | 936.2 | 7,18,23,529,21,6,323,18,964 |
| 48 | 256 | 12 | 9.0/9.0 | 2.0 | 0.0/27.0 | 729.2 | 728.5 | 0.8 | 726.8 | 691.0/673.5 | 19.2 | 140.8 | 0.801 | 154.8 | 1339.2 | 6,17,23,366,15,5,234,16,690 |
| 49 | 256 | 27 | 8.0/8.0 | 13.0 | 0.0/24.0 | 485.5 | 485.2 | 0.8 | 851.2 | 823.8/807.2 | 18.0 | 273.2 | 0.957 | 286.2 | 1058.8 | 7,18,52,425,17,9,269,16,823 |
| 5 | 256 | 22 | 16.0/16.0 | 3.0 | 0.0/48.0 | 933.5 | 932.2 | 0.8 | 1205.8 | 1181.2/1112.2 | 64.0 | 527.8 | 0.887 | 536.0 | 1636.0 | 8,18,26,668,24,6,409,17,1180 |
| 50 | 256 | 17 | 14.0/14.0 | 3.0 | 0.0/42.0 | 616.5 | 615.8 | 0.8 | 1035.2 | 1010.2/984.5 | 25.5 | 458.5 | 1.000 | 471.2 | 1201.0 | 7,21,23,562,22,6,350,19,1009 |
| 51 | 256 | 16 | 12.0/12.0 | 3.0 | 0.0/36.0 | 799.2 | 798.8 | 0.8 | 893.0 | 868.2/841.5 | 22.0 | 291.8 | 0.918 | 309.5 | 1415.2 | 6,17,24,470,19,6,294,17,867 |
| 52 | 256 | 13 | 10.0/10.0 | 2.0 | 0.0/30.0 | 659.0 | 658.2 | 0.8 | 794.5 | 770.2/750.8 | 19.0 | 206.2 | 0.949 | 216.0 | 1262.0 | 7,19,22,429,17,5,254,17,769 |
| 53 | 256 | 24 | 14.0/14.0 | 5.0 | 0.0/42.0 | 559.5 | 559.0 | 0.8 | 1108.8 | 1084.2/1051.2 | 26.5 | 495.5 | 1.000 | 511.2 | 1168.8 | 7,20,32,593,23,7,375,18,1083 |
| 54 | 256 | 16 | 12.0/12.0 | 3.0 | 0.0/36.0 | 871.8 | 871.2 | 0.8 | 908.0 | 879.2/859.0 | 21.2 | 333.8 | 0.984 | 344.0 | 1466.2 | 7,18,24,479,19,6,298,17,878 |
| 55 | 256 | 17 | 15.0/15.0 | 2.0 | 0.0/45.0 | 671.0 | 670.2 | 0.8 | 1113.0 | 1088.0/1050.2 | 25.5 | 508.2 | 1.000 | 521.5 | 1275.0 | 7,19,24,614,23,6,372,18,1087 |
| 56 | 256 | 20 | 15.0/15.0 | 3.0 | 0.0/45.0 | 879.8 | 879.0 | 0.5 | 1121.8 | 1093.2/1068.0 | 28.2 | 519.2 | 0.988 | 532.2 | 1494.8 | 7,19,26,609,24,6,376,18,1092 |
| 57 | 256 | 12 | 11.0/11.0 | 2.0 | 0.0/33.0 | 878.8 | 878.0 | 0.8 | 828.2 | 801.5/779.5 | 20.5 | 214.8 | 0.934 | 227.5 | 1495.0 | 6,18,22,436,18,5,269,17,800 |
| 58 | 256 | 12 | 11.0/11.0 | 2.0 | 0.0/33.0 | 591.2 | 590.5 | 0.8 | 832.2 | 802.5/778.0 | 17.0 | 208.8 | 0.918 | 222.0 | 1218.5 | 6,18,22,435,18,5,266,18,801 |
| 59 | 256 | 12 | 10.0/10.0 | 2.0 | 0.0/30.0 | 591.0 | 590.5 | 0.8 | 784.2 | 758.0/737.2 | 20.5 | 141.2 | 0.680 | 156.0 | 1246.0 | 7,16,22,417,17,5,251,16,757 |
| 6 | 256 | 18 | 12.0/12.0 | 3.0 | 0.0/36.0 | 972.0 | 971.2 | 0.5 | 978.0 | 949.0/893.8 | 46.8 | 351.5 | 0.895 | 364.8 | 1614.2 | 7,16,26,526,20,6,320,16,948 |
| 60 | 256 | 16 | 13.0/13.0 | 2.0 | 0.0/39.0 | 552.2 | 551.0 | 0.8 | 960.5 | 931.2/900.5 | 21.2 | 311.0 | 0.918 | 325.8 | 1200.2 | 7,18,22,518,20,6,313,18,930 |
| 7 | 256 | 20 | 13.0/13.0 | 3.0 | 0.0/39.0 | 744.0 | 743.0 | 0.8 | 1049.0 | 1018.0/960.2 | 56.5 | 462.2 | 0.910 | 473.0 | 1319.5 | 7,17,26,568,21,6,346,16,1017 |
| 8 | 256 | 18 | 12.0/12.0 | 3.0 | 0.0/36.0 | 814.5 | 813.8 | 0.8 | 999.2 | 974.0/924.2 | 43.5 | 372.8 | 0.938 | 384.8 | 1439.0 | 7,17,27,540,20,6,328,16,973 |
| 9 | 256 | 15 | 11.0/11.0 | 2.0 | 0.0/33.0 | 763.0 | 762.2 | 0.8 | 883.0 | 855.2/815.0 | 42.2 | 322.8 | 0.938 | 335.2 | 1344.8 | 7,18,23,486,19,5,291,17,854 |

### S4_CORR (bs=63 graph, 층별 p50, µs)

| producer L | n | cold assign | unique n0/n1 | max rows n0 | AMX/AVX n0 | enq→start | pred union | unattrib | service | numa0/1 | skew | pub delta | late share | pre_h2d gap | budget | stages n0 (prep,cpy,q_in,up_gate,act,q_down,down,weight,total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 236 | 12 | 6.0/6.0 | 3.0 | 0.0/18.0 | 270.8 | 269.8 | 1.2 | 603.0 | 578.8/558.2 | 13.8 | -389.2 | 0.072 | 11.0 | 1295.8 | 3,13,20,298,13,5,197,15,578 |
| 1 | 256 | 18 | 13.0/13.0 | 3.0 | 0.0/39.0 | 91.0 | 90.2 | 1.0 | 990.8 | 966.5/930.8 | 29.5 | -44.0 | 0.449 | 13.2 | 1168.8 | 7,15,22,538,21,6,333,17,965 |
| 10 | 256 | 22 | 15.0/15.0 | 3.0 | 0.0/45.0 | 675.0 | 674.5 | 0.8 | 1188.5 | 1158.8/1107.0 | 50.5 | 559.8 | 0.988 | 572.2 | 1296.8 | 8,21,26,650,24,7,398,18,1157 |
| 11 | 256 | 19 | 12.0/12.0 | 3.0 | 0.0/36.0 | 951.0 | 950.0 | 0.8 | 936.8 | 910.5/861.8 | 39.5 | 357.2 | 0.977 | 366.2 | 1555.2 | 6,17,26,502,19,6,312,16,909 |
| 12 | 256 | 16 | 11.0/11.0 | 3.0 | 0.0/33.0 | 701.2 | 699.8 | 0.8 | 907.8 | 875.0/837.8 | 34.0 | 293.0 | 0.969 | 305.5 | 1332.2 | 8,18,24,486,19,6,295,17,874 |
| 13 | 256 | 24 | 16.0/16.0 | 4.0 | 0.0/48.0 | 673.0 | 672.5 | 0.8 | 1241.8 | 1217.8/1168.0 | 47.5 | 636.0 | 1.000 | 650.2 | 1278.2 | 8,20,28,681,25,7,418,18,1216 |
| 14 | 256 | 17 | 12.0/12.0 | 3.0 | 0.0/36.0 | 1010.5 | 1009.8 | 0.8 | 970.5 | 939.5/907.2 | 30.2 | 392.0 | 1.000 | 405.8 | 1590.5 | 7,20,24,520,20,6,319,18,938 |
| 15 | 256 | 21 | 15.0/15.0 | 3.0 | 0.0/45.0 | 732.5 | 732.0 | 0.8 | 1130.5 | 1105.0/1060.0 | 32.2 | 563.8 | 1.000 | 572.0 | 1303.0 | 7,19,26,614,23,7,376,18,1104 |
| 16 | 256 | 23 | 16.0/16.0 | 3.0 | 0.0/48.0 | 895.5 | 894.0 | 0.8 | 1178.2 | 1152.0/1119.0 | 31.5 | 603.8 | 1.000 | 612.8 | 1479.0 | 7,20,27,644,24,7,396,19,1151 |
| 17 | 256 | 16 | 12.0/12.0 | 3.0 | 0.0/36.0 | 942.2 | 941.8 | 0.8 | 924.0 | 897.5/858.2 | 26.8 | 348.8 | 1.000 | 362.8 | 1535.2 | 7,19,23,488,20,6,303,18,897 |
| 18 | 256 | 22 | 15.0/15.0 | 3.0 | 0.0/45.0 | 692.5 | 691.5 | 0.8 | 1153.2 | 1127.5/1106.8 | 28.2 | 491.8 | 1.000 | 509.0 | 1362.2 | 7,21,26,625,24,7,390,19,1127 |
| 19 | 256 | 16 | 10.0/10.0 | 4.0 | 0.0/30.0 | 916.5 | 915.8 | 0.8 | 833.2 | 808.5/779.8 | 20.5 | 282.2 | 0.980 | 297.8 | 1498.0 | 7,18,26,436,18,6,272,17,807 |
| 2 | 256 | 22 | 14.0/14.0 | 4.0 | 0.0/42.0 | 377.0 | 376.0 | 1.0 | 1078.8 | 1053.5/1018.0 | 31.8 | 286.5 | 0.750 | 300.5 | 1148.8 | 7,16,27,583,23,6,365,17,1052 |
| 20 | 256 | 32 | 19.0/19.0 | 4.0 | 0.0/57.0 | 601.2 | 600.8 | 0.8 | 1438.2 | 1412.2/1359.5 | 41.5 | 851.5 | 1.000 | 863.5 | 1200.2 | 8,22,30,785,29,8,501,19,1411 |
| 21 | 256 | 28 | 17.0/17.0 | 4.0 | 0.0/51.0 | 1201.0 | 1200.0 | 0.8 | 1263.5 | 1234.0/1206.8 | 27.2 | 697.5 | 1.000 | 713.0 | 1772.0 | 7,20,30,689,26,7,428,18,1233 |
| 22 | 256 | 15 | 12.0/12.0 | 3.0 | 0.0/36.0 | 1031.2 | 1030.5 | 0.8 | 891.0 | 861.5/833.0 | 22.8 | 341.0 | 0.961 | 353.5 | 1572.8 | 7,18,23,469,19,5,297,17,860 |
| 23 | 256 | 20 | 14.0/14.0 | 4.0 | 0.0/42.0 | 655.2 | 654.2 | 0.8 | 1055.2 | 1030.2/1002.2 | 27.2 | 476.2 | 1.000 | 488.8 | 1224.8 | 7,19,27,572,22,6,351,18,1029 |
| 24 | 256 | 22 | 14.0/14.0 | 4.0 | 0.0/42.0 | 814.8 | 814.0 | 0.8 | 1102.5 | 1077.5/1041.2 | 30.5 | 524.8 | 0.980 | 536.0 | 1411.0 | 7,18,29,596,23,7,370,17,1076 |
| 25 | 256 | 21 | 14.0/14.0 | 4.0 | 0.0/42.0 | 863.0 | 862.0 | 0.8 | 1095.5 | 1068.5/1034.5 | 27.2 | 537.2 | 1.000 | 544.2 | 1428.8 | 7,21,28,583,22,6,367,19,1067 |
| 26 | 256 | 14 | 11.0/11.0 | 3.0 | 0.0/33.0 | 860.0 | 859.2 | 0.8 | 828.5 | 802.2/781.8 | 15.8 | 263.8 | 0.949 | 277.8 | 1430.5 | 7,18,24,435,18,5,270,17,801 |
| 27 | 256 | 18 | 11.0/11.0 | 4.0 | 0.0/33.0 | 592.8 | 592.0 | 0.8 | 907.2 | 879.5/858.0 | 22.8 | 356.8 | 1.000 | 371.2 | 1153.8 | 7,19,29,473,19,6,300,18,879 |
| 28 | 256 | 22 | 15.0/15.0 | 4.0 | 0.0/45.0 | 672.5 | 671.8 | 0.8 | 1118.2 | 1092.2/1062.5 | 27.5 | 561.5 | 1.000 | 573.5 | 1241.5 | 7,21,29,604,23,7,378,19,1091 |
| 29 | 256 | 13 | 10.0/10.0 | 3.0 | 0.0/30.0 | 883.2 | 882.2 | 1.0 | 773.5 | 747.8/724.8 | 17.8 | 211.2 | 0.945 | 222.8 | 1452.2 | 6,17,23,404,17,5,248,17,747 |
| 3 | 256 | 17 | 12.0/12.0 | 3.0 | 0.0/36.0 | 743.5 | 742.8 | 0.8 | 951.5 | 921.5/877.5 | 33.8 | 263.2 | 0.828 | 275.2 | 1419.5 | 6,16,25,510,20,6,314,16,920 |
| 30 | 256 | 21 | 12.0/12.0 | 5.0 | 0.0/36.0 | 538.5 | 537.5 | 0.8 | 990.0 | 965.2/936.0 | 27.0 | 433.5 | 1.000 | 446.8 | 1098.8 | 7,19,30,523,21,6,335,17,964 |
| 31 | 256 | 6 | 5.0/5.0 | 2.0 | 0.0/15.0 | 752.2 | 751.8 | 0.8 | 479.5 | 454.0/434.5 | 12.2 | -86.2 | 0.195 | 11.5 | 1323.8 | 5,14,22,233,11,4,137,15,453 |
| 32 | 256 | 10 | 8.0/8.0 | 2.0 | 0.0/24.0 | 243.8 | 242.8 | 0.8 | 646.8 | 621.5/608.2 | 10.2 | 39.2 | 0.598 | 52.5 | 842.5 | 6,15,16,334,14,5,203,16,620 |
| 33 | 256 | 11 | 9.0/9.0 | 2.0 | 0.0/27.0 | 320.8 | 319.8 | 0.8 | 704.5 | 676.8/653.5 | 14.8 | 150.5 | 0.809 | 164.5 | 861.8 | 6,17,19,357,15,5,226,17,676 |
| 34 | 256 | 10 | 9.0/9.0 | 2.0 | 0.0/27.0 | 446.8 | 446.0 | 0.8 | 698.2 | 672.8/653.8 | 16.0 | 157.2 | 0.930 | 171.0 | 987.5 | 6,19,20,354,15,5,225,18,672 |
| 35 | 256 | 10 | 8.0/8.0 | 2.0 | 0.0/24.0 | 456.5 | 455.8 | 0.8 | 662.2 | 636.5/620.2 | 14.0 | 122.8 | 0.938 | 136.8 | 997.0 | 6,19,19,347,15,5,204,18,636 |
| 36 | 256 | 16 | 9.0/9.0 | 5.0 | 0.0/27.0 | 425.0 | 424.2 | 1.0 | 789.2 | 764.2/741.0 | 18.2 | 232.5 | 0.992 | 242.5 | 988.5 | 6,19,29,401,17,6,255,17,763 |
| 37 | 256 | 17 | 9.0/9.0 | 4.0 | 0.0/27.0 | 552.8 | 551.8 | 0.8 | 804.8 | 779.5/757.5 | 19.5 | 244.8 | 1.000 | 256.8 | 1116.5 | 7,20,27,413,17,5,263,18,778 |
| 38 | 256 | 24 | 13.0/13.0 | 6.0 | 0.0/39.0 | 573.0 | 572.0 | 0.8 | 1011.5 | 986.2/959.0 | 26.2 | 401.2 | 1.000 | 418.0 | 1197.2 | 7,20,33,538,21,7,340,17,985 |
| 39 | 256 | 11 | 8.0/8.0 | 3.0 | 0.0/24.0 | 781.5 | 781.0 | 1.0 | 656.2 | 629.2/607.0 | 17.2 | 80.5 | 0.801 | 92.8 | 1371.0 | 6,17,24,331,14,5,205,17,628 |
| 4 | 256 | 28 | 15.0/15.0 | 5.0 | 0.0/45.0 | 698.2 | 697.0 | 0.8 | 1190.2 | 1163.2/1103.0 | 45.8 | 600.2 | 0.910 | 609.2 | 1278.2 | 7,18,32,642,24,7,401,16,1162 |
| 40 | 256 | 15 | 10.0/10.0 | 3.0 | 0.0/30.0 | 418.8 | 418.0 | 0.8 | 801.5 | 776.2/751.5 | 20.2 | 210.8 | 0.918 | 221.5 | 1005.0 | 7,17,24,419,17,5,261,17,775 |
| 41 | 256 | 12 | 8.0/8.0 | 2.0 | 0.0/24.0 | 558.5 | 557.8 | 0.8 | 702.0 | 676.2/654.0 | 17.5 | 175.5 | 0.914 | 186.0 | 1112.5 | 6,17,22,360,14,5,226,16,675 |
| 42 | 256 | 29 | 16.0/16.0 | 6.0 | 0.0/48.0 | 463.8 | 463.5 | 0.8 | 1251.8 | 1225.2/1179.5 | 31.0 | 676.8 | 1.000 | 688.2 | 1038.2 | 7,20,34,668,25,7,426,17,1224 |
| 43 | 256 | 14 | 10.0/10.0 | 3.0 | 0.0/30.0 | 1007.5 | 1006.8 | 0.8 | 796.5 | 769.8/743.5 | 16.0 | 255.0 | 0.977 | 269.0 | 1554.2 | 7,17,24,420,17,5,252,17,769 |
| 44 | 256 | 18 | 11.0/11.0 | 4.0 | 0.0/33.0 | 562.5 | 562.2 | 0.8 | 906.0 | 876.8/860.2 | 22.2 | 341.2 | 0.934 | 351.8 | 1119.2 | 8,17,26,472,19,6,301,16,876 |
| 45 | 256 | 19 | 12.0/12.0 | 5.0 | 0.0/36.0 | 664.2 | 663.5 | 0.8 | 934.0 | 900.2/875.0 | 21.8 | 346.8 | 0.977 | 358.8 | 1252.8 | 7,18,30,492,19,6,303,17,899 |
| 46 | 256 | 8 | 6.0/6.0 | 2.0 | 0.0/18.0 | 702.0 | 701.2 | 0.8 | 530.5 | 505.0/489.0 | 13.5 | -6.5 | 0.473 | 13.5 | 1259.8 | 6,15,21,266,12,4,159,16,504 |
| 47 | 256 | 16 | 13.0/13.0 | 3.0 | 0.0/39.0 | 296.5 | 295.8 | 0.8 | 974.0 | 943.2/916.2 | 22.5 | 361.8 | 0.973 | 375.2 | 918.8 | 7,18,23,516,20,6,320,18,942 |
| 48 | 256 | 12 | 8.0/8.0 | 2.0 | 0.0/24.0 | 711.8 | 711.0 | 0.8 | 690.5 | 664.5/643.8 | 16.0 | 88.8 | 0.781 | 101.2 | 1298.0 | 6,16,23,355,15,5,221,16,664 |
| 49 | 256 | 27 | 8.0/8.0 | 13.0 | 0.0/24.0 | 449.2 | 448.5 | 0.8 | 824.0 | 794.8/776.8 | 20.2 | 258.5 | 0.957 | 273.0 | 1030.5 | 7,18,54,408,16,9,261,16,794 |
| 5 | 256 | 23 | 17.0/17.0 | 3.0 | 0.0/51.0 | 952.8 | 952.0 | 0.8 | 1282.0 | 1256.2/1186.5 | 63.2 | 592.8 | 0.902 | 602.2 | 1646.5 | 8,18,26,714,26,7,433,17,1255 |
| 50 | 256 | 17 | 14.0/14.0 | 3.0 | 0.0/42.0 | 585.0 | 583.8 | 0.8 | 1062.0 | 1037.2/1000.2 | 25.8 | 482.0 | 1.000 | 499.2 | 1165.5 | 7,20,23,572,22,6,358,19,1036 |
| 51 | 256 | 15 | 11.0/11.0 | 3.0 | 0.0/33.0 | 824.0 | 823.2 | 0.8 | 854.0 | 830.0/803.0 | 21.5 | 253.8 | 0.953 | 270.2 | 1432.8 | 6,17,24,451,18,5,281,17,829 |
| 52 | 256 | 13 | 10.0/10.0 | 2.0 | 0.0/30.0 | 616.2 | 615.8 | 0.8 | 798.2 | 767.5/744.2 | 17.5 | 204.8 | 0.977 | 218.0 | 1230.8 | 7,19,22,417,17,5,255,17,767 |
| 53 | 256 | 26 | 15.0/15.0 | 5.0 | 0.0/45.0 | 559.2 | 558.8 | 0.8 | 1144.8 | 1119.8/1089.0 | 28.0 | 544.8 | 1.000 | 558.2 | 1157.8 | 7,20,33,615,24,7,386,18,1119 |
| 54 | 256 | 16 | 11.0/11.0 | 3.0 | 0.0/33.0 | 911.2 | 910.5 | 0.8 | 866.5 | 837.0/809.2 | 20.5 | 295.8 | 0.996 | 307.2 | 1505.0 | 6,18,24,456,18,5,288,17,835 |
| 55 | 256 | 18 | 15.0/15.0 | 2.0 | 0.0/45.0 | 633.8 | 633.2 | 0.8 | 1114.8 | 1087.0/1066.2 | 21.8 | 507.2 | 1.000 | 522.0 | 1234.2 | 7,19,24,605,23,6,372,18,1086 |
| 56 | 256 | 18 | 14.0/14.0 | 3.0 | 0.0/42.0 | 881.0 | 879.8 | 0.8 | 1018.8 | 994.5/967.5 | 25.8 | 418.2 | 0.996 | 429.8 | 1477.2 | 7,19,25,543,22,6,345,18,993 |
| 57 | 256 | 11 | 10.0/10.0 | 2.0 | 0.0/30.0 | 783.5 | 783.0 | 0.8 | 783.8 | 758.0/742.5 | 17.8 | 187.5 | 0.918 | 198.8 | 1387.8 | 6,18,21,418,17,5,253,17,757 |
| 58 | 256 | 11 | 10.0/10.0 | 2.0 | 0.0/30.0 | 550.0 | 549.5 | 0.8 | 771.8 | 740.0/728.8 | 16.2 | 151.5 | 0.895 | 167.8 | 1180.8 | 5,19,20,412,16,5,242,18,739 |
| 59 | 256 | 10 | 8.0/8.0 | 2.0 | 0.0/24.0 | 532.2 | 530.8 | 1.0 | 691.5 | 657.8/635.5 | 17.2 | 55.0 | 0.605 | 65.8 | 1159.5 | 6,16,21,352,15,5,223,16,657 |
| 6 | 256 | 19 | 13.0/13.0 | 3.0 | 0.0/39.0 | 1047.5 | 1046.8 | 1.0 | 999.2 | 972.5/916.8 | 53.0 | 373.0 | 0.883 | 384.0 | 1685.2 | 7,17,26,544,21,6,326,16,971 |
| 60 | 256 | 14 | 12.0/12.0 | 2.0 | 0.0/36.0 | 449.8 | 449.0 | 0.8 | 904.2 | 873.8/849.2 | 18.2 | 244.8 | 0.918 | 256.5 | 1098.5 | 7,18,21,485,19,5,298,18,872 |
| 7 | 256 | 20 | 13.0/13.0 | 4.0 | 0.0/39.0 | 764.5 | 763.5 | 0.8 | 1065.5 | 1041.0/979.8 | 58.0 | 481.0 | 0.918 | 492.8 | 1350.2 | 8,17,27,580,22,6,355,16,1040 |
| 8 | 256 | 19 | 13.0/13.0 | 3.0 | 0.0/39.0 | 832.2 | 831.0 | 0.8 | 1009.2 | 977.2/926.5 | 46.8 | 385.8 | 0.945 | 397.2 | 1452.2 | 7,18,26,548,21,6,333,16,976 |
| 9 | 256 | 16 | 11.0/11.0 | 3.0 | 0.0/33.0 | 772.2 | 770.8 | 1.0 | 908.0 | 882.2/832.0 | 43.5 | 340.2 | 0.945 | 351.2 | 1351.8 | 7,18,22,488,19,5,296,17,881 |

## 8. 자원 (RESOURCE 세션; 부하 창 = forward_envelope(kt_evt))

### S5_RESOURCE

```
{
 "session": "S5_RESOURCE",
 "window": {
  "name": "forward_envelope(kt_evt)",
  "start_ns": 1789627431709006665,
  "end_ns": 1789627452631345168,
  "len_s": 20.922338503,
  "status": "OK"
 },
 "client_process_window_s": 34.130108001991175,
 "benchmark_duration_s": 20.963225270970725,
 "perf": {
  "intervals_total": 38,
  "full": 20,
  "boundary": 2,
  "outside": 16,
  "bad_rows": 0,
  "full_window_s": 20.0,
  "task_clock_msec_sum_full": 2054151.2599999998,
  "cpu_equivalents_full": 102.70756299999998,
  "cycles_sum": 4086917481498.0,
  "instructions_sum": 5315574416061.0,
  "ipc_full": 1.3006316961683928,
  "context_switches_sum": 75232.0,
  "cpu_migrations_sum": 7230.0,
  "cache_misses_sum": 69534456890.0,
  "scope": "스케줄러 4 PID 프로세스 전체(스레드 상속)",
  "target_pids": "3757104,3757200,3757361,3757517",
  "interval_contract": "perf -I 1000: 각 행 = 직전 1 s 구간 (상대 초, 시작 = collector actual_start)",
  "note": "task-clock 은 논리 CPU 시간 합 (busy-poll 포함); 물리코어 수 아님. 역할별 분리는 thread_role_map 대조 필요 (프로세스 집계)"
 },
 "pcm": {
  "invalid_rows": 0,
  "interval_contract": "ASSUMED_END_TIMESTAMP (1 s)",
  "unit": "MB/s",
  "by_metric": {
   "System|Read": {
    "n_full": 19,
    "n_boundary": 2,
    "n_outside": 15,
    "n_duplicate": 0,
    "full_sample_coverage": 0.9080246931898136,
    "mean_full": 232896.0457894737,
    "mean_overlap_est": 218911.5671575807,
    "mean_overlap_assumption": "each value represents a constant rate over its sample interval",
    "overlap_coverage": 0.9995698377598322
   },
   "System|Write": {
    "n_full": 19,
    "n_boundary": 2,
    "n_outside": 15,
    "n_duplicate": 0,
    "full_sample_coverage": 0.9080246931898136,
    "mean_full": 28314.602631578946,
    "mean_overlap_est": 28921.051527156924,
    "mean_overlap_assumption": "each value represents a constant rate over its sample interval",
    "overlap_coverage": 0.9995698377598322
   },
   "System|Memory": {
    "n_full": 19,
    "n_boundary": 2,
    "n_outside": 15,
    "n_duplicate": 0,
    "full_sample_coverage": 0.9080246931898136,
    "mean_full": 261210.64789473685,
    "mean_overlap_est": 247832.61776754228,
    "mean_overlap_assumption": "each value represents a constant rate over its sample interval",
    "overlap_coverage": 0.9995698377598322
   }
  }
 }
}
```

## 9. 기존 자료 정정 (A01; profiles/legacy_vs_corrected.csv, WINDOW_RECONSTRUCTION.md)

| item | session | legacy | corrected | 사유 |
|---|---|---|---|---|
| PCM System|Read MB/s | P2_r1 | 144783.7 | 236992.9 | 창(클라이언트 준비 12.6~14.4 s 포함) + 파싱 실패 표본 포함 |
| PCM System|Write MB/s | P2_r1 | 16728.6 | 26535.2 | 창(클라이언트 준비 12.6~14.4 s 포함) + 파싱 실패 표본 포함 |
| PCM System|Read MB/s | P2_r2 | 143391.1 | 250972.7 | 창(클라이언트 준비 12.6~14.4 s 포함) + 파싱 실패 표본 포함 |
| PCM System|Write MB/s | P2_r2 | 16680.8 | 26994.0 | 창(클라이언트 준비 12.6~14.4 s 포함) + 파싱 실패 표본 포함 |
| PCM System|Read MB/s | P2_r3 | 139422.6 | 247825.6 | 창(클라이언트 준비 12.6~14.4 s 포함) + 파싱 실패 표본 포함 |
| PCM System|Write MB/s | P2_r3 | 16056.1 | 27151.1 | 창(클라이언트 준비 12.6~14.4 s 포함) + 파싱 실패 표본 포함 |
| GPU busy fraction (IDE_073 D2 retry2) | TP-0. trace_full | 0.401 | 0.392 | stream 중첩·annotation |
| GPU busy fraction (IDE_073 D2 retry2) | TP-0. request_window | 0.401 | 0.421 | stream 중첩·annotation |
| GPU busy fraction (IDE_073 D2 retry2) | TP-1. trace_full | 0.775 | 0.572 | stream 중첩·annotation |
| GPU busy fraction (IDE_073 D2 retry2) | TP-1. request_window | 0.775 | 0.624 | stream 중첩·annotation |
| GPU busy fraction (IDE_073 D2 retry2) | TP-2. trace_full | 0.785 | 0.579 | stream 중첩·annotation |
| GPU busy fraction (IDE_073 D2 retry2) | TP-2. request_window | 0.785 | 0.624 | stream 중첩·annotation |
| GPU busy fraction (IDE_073 D2 retry2) | TP-3. trace_full | 0.785 | 0.578 | stream 중첩·annotation |
| GPU busy fraction (IDE_073 D2 retry2) | TP-3. request_window | 0.785 | 0.624 | stream 중첩·annotation |
| late share (cold later than hot) | F1_C1_P1 graph 35 | 0.0036148563508064517 | 0.02 | cohort 분리 |
| cold_ready_lateness_us p50 → cold_pub_lateness_us p50 | F1_C1_P1 graph 35 | 0.0 | 0.0 | cohort |
| late share (cold later than hot) | F1_LONG_P1 graph 26 | 0.012821320564516129 | 0.017 | cohort 분리 |
| cold_ready_lateness_us p50 → cold_pub_lateness_us p50 | F1_LONG_P1 graph 26 | 0.0 | 0.0 | cohort |
| late share (cold later than hot) | S1_P1_PROBE128 graph 2 | 0.8958543346774194 | 0.91 | cohort 분리 |
| cold_ready_lateness_us p50 → cold_pub_lateness_us p50 | S1_P1_PROBE128 graph 2 | 324.5 | 330.5 | cohort |
| late share (cold later than hot) | S1_P1_PROBE128 graph 32 | 0.013860887096774193 | 0.042 | cohort 분리 |
| cold_ready_lateness_us p50 → cold_pub_lateness_us p50 | S1_P1_PROBE128 graph 32 | 0.0 | 0.0 | cohort |
| late share (cold later than hot) | P1_r1 graph 2 | 0.8850176411290323 | 0.898 | cohort 분리 |
| cold_ready_lateness_us p50 → cold_pub_lateness_us p50 | P1_r1 graph 2 | 302.25 | 307.75 | cohort |
| late share (cold later than hot) | P1_r1 graph 32 | 0.006048387096774193 | 0.018 | cohort 분리 |
| cold_ready_lateness_us p50 → cold_pub_lateness_us p50 | P1_r1 graph 32 | 0.0 | 0.0 | cohort |
| late share (cold later than hot) | P1_r2 graph 2 | 0.8843876008064516 | 0.897 | cohort 분리 |
| cold_ready_lateness_us p50 → cold_pub_lateness_us p50 | P1_r2 graph 2 | 307.0 | 312.5 | cohort |
| late share (cold later than hot) | P1_r2 graph 32 | 0.007686491935483871 | 0.02 | cohort 분리 |
| cold_ready_lateness_us p50 → cold_pub_lateness_us p50 | P1_r2 graph 32 | 0.0 | 0.0 | cohort |
| late share (cold later than hot) | P1_r3 graph 2 | 0.8956653225806451 | 0.909 | cohort 분리 |
| cold_ready_lateness_us p50 → cold_pub_lateness_us p50 | P1_r3 graph 2 | 328.5 | 334.0 | cohort |
| late share (cold later than hot) | P1_r3 graph 32 | 0.004158266129032258 | 0.014 | cohort 분리 |
| cold_ready_lateness_us p50 → cold_pub_lateness_us p50 | P1_r3 graph 32 | 0.0 | 0.0 | cohort |

## 10. 오류·중단·미실행

| 항목 | 내용 |
|---|---|
| boot_end CORE_152703 | ABORTED_HARNESS_BUG ATTACH 분기 전 PLANS 조회 KeyError → 체인의 다음 plan(OFF_A) 의 stop_server 가 로딩 중이던 서버를 종료. 부팅 시도로 계수 |
| GLM (G00) | 실행 없음. IDE_074 게이트 BLOCKED_NORMAL_OUTPUT 계승 (GLM_NORMAL_OUTPUT_BLOCKED) |
| S7 예비 | 하네스 결함으로 부팅 2회 소진 → 예비 없음 (BLOCKED_BY_BUDGET) |

## 11. 파일

- offline/: legacy_windows.csv, pcm_samples_v2.csv, pcm_window_summary_v2.csv, timestamp_parse_errors.jsonl, parser_test_results.json
- 세션/v2/: producer_consumer_metrics_v2.csv.gz, dependency_summary_v2.json, missing_coverage_v2.csv, queue_wait_breakdown.csv.gz, fifo_dependency_edges.csv.gz, producer_layer_costs.csv, expert_cost_samples.csv.gz, fifo_casebook.md, validation_results_v2.json, resource_summary.json
- 세션/: window_events.jsonl, observer_config.json, cpu_counter_intervals.csv, pcm_memory.raw.csv, requests.jsonl, 시계열
- 서버 보존: ~/.cache/huggingface/kt/ide075/<boot>/{kt_evt.csv, kt_evt.csv.tasks, kt_evt.csv.map, profiles/*.trace.json.gz} (ARTIFACT_INDEX.csv SHA256)
