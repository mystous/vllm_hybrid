# FULL_REPORT — IDE_074 CPU MoE Hot/Cold 병목 측정 (2026-09-17T14:05:54+09:00)

지시서 `PLAN.md` (SHA 374c8f6f60bb2071…). 확정 실행안 `PLAN_RESOLVED.md`, 조사 기록 `WORK_LOG.md`, 설정 차이 `CONFIG_DIFF.md`, 원자료 색인 `SOURCE_INDEX.md`, 상태 `COMPLETION_STATUS.md`. 평가·원인 추정·권고 없음.

## 1. 환경·고정 구성

```
host violet-h100-016 kernel 5.14.0-427.124.1.el9_4.x86_64 no_turbo=1 governor=performance smt=1
git HEAD(수집 시) 82ad7bca44c2a46b9ac1e4bb2f90483b9743069e branch feat/cpu-moe-bottleneck-20260917 base_is_ancestor=True
container python/torch/sglang: 3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]
2.13.0+cu130 13.0
/sgl-workspace/sglang/python/sglang/__init__.py
/usr/local/lib/python3.12/dist-packages/kt_kernel/__init__.py
0.5.18
kt_kernel_ext.so (IDE_073 기준): e29357f786a53030296f75beaf3db2745baa730835ddc7d98aca88dd014aa46d 8290336 /usr/lo
kt_kernel_ext.so (IDE_074 재빌드): adf49e48ec0510fafb55f14e72ee348938c7410653af18b77529d9124d1b52be 8306720 B (CONFIG_DIFF.md)
```

부팅별 실효 증빙 (runtime_proofs.json):

| model/profile/boot | mode | verdict | per_layer 행 수 / 합 | cf_ready | kt_evt | smoke_greedy4 | kt .so SHA(부팅 시) |
|---|---|---|---|---|---|---|---|
| glm/BASIC4/G_135723 | GLM_GATE | HEALTH_OK | 0 / None | None | 0 | 없음 |  |
| qwen/OPT4/B1_131834 | OFF | HEALTH_OK | 248 / 23808 | 1 | 0 | 있음 | adf49e48ec05 |
| qwen/OPT4/B2_132633 | P1 | HEALTH_OK | 248 / 23808 | 1 | 1 | 있음 | adf49e48ec05 |
| qwen/OPT4/B3_133528 | P2 | HEALTH_OK | 248 / 23808 | 1 | 1 | 있음 | adf49e48ec05 |
| qwen/OPT4/F1_OFF_134055 | OFF | HEALTH_OK | 248 / 23808 | 1 | 0 | 있음 | adf49e48ec05 |
| qwen/OPT4/F1_ON_134513 | P1 | HEALTH_OK | 248 / 23808 | 1 | 1 | 있음 | adf49e48ec05 |
| qwen/OPT4/SMOKE_131024 | P1 | HEALTH_OK | 248 / 23808 | 1 | 1 | 있음 | adf49e48ec05 |

부팅별 smoke_greedy4 출력 동일성 (OFF/ON 실행 의미 비교; 텍스트 SHA):

| boot | 항목별 출력 SHA256 앞 12자 |
|---|---|
| B1_131834 | 63337bc82326, fcb16240af17, 42f1f589bf67, d43d1f961375 |
| B2_132633 | 63337bc82326, fcb16240af17, 42f1f589bf67, d43d1f961375 |
| B3_133528 | 63337bc82326, fcb16240af17, 42f1f589bf67, d43d1f961375 |
| F1_OFF_134055 | 63337bc82326, fcb16240af17, 42f1f589bf67, d43d1f961375 |
| F1_ON_134513 | 63337bc82326, fcb16240af17, 42f1f589bf67, d43d1f961375 |
| SMOKE_131024 | 63337bc82326, fcb16240af17, 42f1f589bf67, d43d1f961375 |

## 2. 모든 세션 (원값·유효성)

| model | profile | boot | session | mode | kind | workload | C | n | 완료/실패 | dur s | in tok | out tok | out tok/s | TTFT p50/p95/p99 | TPOT p50/p95/p99 | ITL p50/p95 | E2EL p50/p95 | drain s (GPU idle 확인) | valid | start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen | OPT4 | B1_131834 | B_r1 | OFF | PERF | M073_QWEN_MAIN_SHORT | 64 | 256 | 256/0 | 44.1 | 131786 | 32768 | 743.02 | 2199/3184/10870 | 63.7/72.5/74.6 | 59.6/69.1 | 10304/10674 | 1.3 (True) | True | 2026-09-17T13:21:08.337716+09:00 |
| qwen | OPT4 | B1_131834 | B_r2 | OFF | PERF | M073_QWEN_MAIN_SHORT | 64 | 256 | 256/0 | 43.8 | 131786 | 32768 | 747.53 | 2177/3056/10757 | 64.0/73.2/74.8 | 59.9/68.6 | 10402/10467 | 1.3 (True) | True | 2026-09-17T13:22:53.458175+09:00 |
| qwen | OPT4 | B1_131834 | B_r3 | OFF | PERF | M073_QWEN_MAIN_SHORT | 64 | 256 | 256/0 | 42.6 | 131786 | 32768 | 768.93 | 2134/3068/10442 | 61.5/68.4/72.7 | 56.5/64.5 | 10006/10160 | 1.3 (True) | True | 2026-09-17T13:24:36.798547+09:00 |
| qwen | OPT4 | B1_131834 | P0_r1 | OFF | DIAG | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 22.7 | 65886 | 16384 | 720.80 | 2179/3114/8428 | 62.5/73.3/73.3 | 58.8/67.9 | 10422/10444 | 1.3 (True) | True | 2026-09-17T13:22:11.816317+09:00 |
| qwen | OPT4 | B1_131834 | P0_r2 | OFF | DIAG | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 21.0 | 65886 | 16384 | 780.72 | 2089/3043/3045 | 57.3/69.9/70.0 | 49.1/66.2 | 9477/10291 | 1.3 (True) | True | 2026-09-17T13:23:56.547215+09:00 |
| qwen | OPT4 | B1_131834 | P0_r3 | OFF | DIAG | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 22.4 | 65886 | 16384 | 732.61 | 2229/3162/8260 | 60.2/71.0/71.0 | 56.6/66.6 | 10299/10317 | 1.3 (True) | True | 2026-09-17T13:25:38.767034+09:00 |
| qwen | OPT4 | B2_132633 | P1_r1 | P1 | DIAG | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.5 | 65886 | 16384 | 696.39 | 2382/3398/8799 | 65.3/71.0/74.9 | 59.5/69.1 | 10684/10711 | 1.3 (True) | True | 2026-09-17T13:29:06.941324+09:00 |
| qwen | OPT4 | B2_132633 | P1_r2 | P1 | DIAG | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.2 | 65886 | 16384 | 705.62 | 2226/3126/8626 | 63.9/71.2/74.9 | 60.2/68.6 | 10523/10546 | 1.3 (True) | True | 2026-09-17T13:31:09.118873+09:00 |
| qwen | OPT4 | B2_132633 | P1_r3 | P1 | DIAG | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.7 | 65886 | 16384 | 690.76 | 2266/3217/8814 | 65.8/72.8/75.6 | 61.1/69.8 | 10809/10831 | 1.3 (True) | True | 2026-09-17T13:33:10.323523+09:00 |
| qwen | OPT4 | B3_133528 | P2_r1 | P2 | DIAG | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.2 | 65886 | 16384 | 706.46 | 2255/3245/8837 | 64.0/71.7/77.1 | 59.4/67.4 | 10493/10614 | 1.3 (True) | True | 2026-09-17T13:37:59.818596+09:00 |
| qwen | OPT4 | B3_133528 | P2_r2 | P2 | DIAG | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 22.7 | 65886 | 16384 | 720.67 | 2129/3062/8441 | 63.3/70.3/74.3 | 58.9/65.8 | 10365/10383 | 1.3 (True) | True | 2026-09-17T13:38:45.768950+09:00 |
| qwen | OPT4 | B3_133528 | P2_r3 | P2 | DIAG | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.2 | 65886 | 16384 | 705.25 | 2236/3083/8739 | 65.5/71.8/77.4 | 60.7/69.9 | 10526/10566 | 1.3 (True) | True | 2026-09-17T13:39:31.625757+09:00 |
| qwen | OPT4 | F1_OFF_134055 | F1_C1_OFF | OFF | DIAG | M073_QWEN_LOW_CONCURRENCY | 1 | 16 | 16/0 | 32.4 | 8234 | 2048 | 63.27 | 235/318/330 | 13.9/14.0/14.0 | 13.9/14.2 | 2006/2088 | 1.3 (True) | True | 2026-09-17T13:43:24.939354+09:00 |
| qwen | OPT4 | F1_OFF_134055 | F1_LONG_OFF | OFF | DIAG | M073_QWEN_LONGER_PREFILL | 8 | 32 | 32/0 | 24.8 | 131147 | 4096 | 164.89 | 2482/3285/3299 | 29.7/41.7/42.1 | 23.9/25.6 | 6203/6318 | 1.3 (True) | True | 2026-09-17T13:44:16.283669+09:00 |
| qwen | OPT4 | F1_ON_134513 | F1_C1_P1 | P1 | DIAG | M073_QWEN_LOW_CONCURRENCY | 1 | 16 | 16/0 | 35.6 | 8234 | 2048 | 57.57 | 327/456/498 | 14.7/14.8/14.9 | 14.8/15.6 | 2199/2340 | 1.3 (True) | True | 2026-09-17T13:47:40.171714+09:00 |
| qwen | OPT4 | F1_ON_134513 | F1_LONG_P1 | P1 | DIAG | M073_QWEN_LONGER_PREFILL | 8 | 32 | 32/0 | 25.7 | 131147 | 4096 | 159.34 | 2528/3373/3427 | 30.8/42.9/43.4 | 25.0/26.3 | 6407/6578 | 1.2 (True) | True | 2026-09-17T13:54:31.768189+09:00 |
| qwen | OPT4 | SMOKE_131024 | S1_P1_PROBE128 | P1 | DIAG | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 24.1 | 65886 | 16384 | 679.64 | 2465/3534/9223 | 65.4/73.2/78.7 | 60.4/71.6 | 10850/10934 | 1.3 (True) | True | 2026-09-17T13:12:52.144657+09:00 |

통계 (유효 rep, 같은 mode·workload):

| model | mode | workload | n | out tok/s 원값 | 평균 | 중앙값 | sd | boots |
|---|---|---|---|---|---|---|---|---|
| qwen | OFF | M073_QWEN_MAIN_SHORT | 3 | 743.02, 747.53, 768.93 | 753.16 | 747.53 | 13.84 | ['B1_131834'] |
| qwen | OFF | M073_QWEN_PROBE128 | 3 | 720.80, 780.72, 732.61 | 744.71 | 732.61 | 31.74 | ['B1_131834'] |
| qwen | P1 | M073_QWEN_PROBE128 | 4 | 696.39, 705.62, 690.76, 679.64 | 693.10 | 693.58 | 10.87 | ['B2_132633', 'SMOKE_131024'] |
| qwen | P2 | M073_QWEN_PROBE128 | 3 | 706.46, 720.67, 705.25 | 710.79 | 706.46 | 8.58 | ['B3_133528'] |
| qwen | OFF | M073_QWEN_LOW_CONCURRENCY | 1 | 63.27 | 63.27 | 63.27 | null | ['F1_OFF_134055'] |
| qwen | OFF | M073_QWEN_LONGER_PREFILL | 1 | 164.89 | 164.89 | 164.89 | null | ['F1_OFF_134055'] |
| qwen | P1 | M073_QWEN_LOW_CONCURRENCY | 1 | 57.57 | 57.57 | 57.57 | null | ['F1_ON_134513'] |
| qwen | P1 | M073_QWEN_LONGER_PREFILL | 1 | 159.34 | 159.34 | 159.34 | null | ['F1_ON_134513'] |

## 3. 계측 간섭 (profiles/observer_overhead.csv; 같은 PROBE128 의 OFF(P0) 평균 대비)

| boot | mode | session | out tok/s | throughput_ratio | elapsed_ratio | e2el_p50 ratio | ttft_p95 ratio | tpot_p95 ratio |
|---|---|---|---|---|---|---|---|---|
| B1_131834 | OFF | P0_r1 | 720.80 | 0.968 | 1.032 | 1.035 | 1.002 | 1.026 |
| B1_131834 | OFF | P0_r2 | 780.72 | 1.048 | 0.953 | 0.941 | 0.980 | 0.979 |
| B1_131834 | OFF | P0_r3 | 732.61 | 0.984 | 1.015 | 1.023 | 1.018 | 0.994 |
| B2_132633 | P1 | P1_r1 | 696.39 | 0.935 | 1.068 | 1.061 | 1.094 | 0.994 |
| B2_132633 | P1 | P1_r2 | 705.62 | 0.948 | 1.054 | 1.045 | 1.006 | 0.997 |
| B2_132633 | P1 | P1_r3 | 690.76 | 0.928 | 1.077 | 1.074 | 1.036 | 1.019 |
| B3_133528 | P2 | P2_r1 | 706.46 | 0.949 | 1.053 | 1.042 | 1.045 | 1.004 |
| B3_133528 | P2 | P2_r2 | 720.67 | 0.968 | 1.032 | 1.030 | 0.986 | 0.985 |
| B3_133528 | P2 | P2_r3 | 705.25 | 0.947 | 1.055 | 1.046 | 0.992 | 1.006 |
| SMOKE_131024 | P1 | S1_P1_PROBE128 | 679.64 | 0.913 | 1.094 | 1.078 | 1.138 | 1.025 |

## 4. 계측 정상성 검증 (M5; 세션별 validation_results.json)

| session | execution | gpu_activity | window | cpu_events | correlation_clock | matched | clock 위반 | lead s(첫 forward − bench 시작) |
|---|---|---|---|---|---|---|---|---|
| P1_r1 | True | True | True | True | True | 23808 | 0 | 13.11 |
| P1_r2 | True | True | True | True | True | 23808 | 0 | 12.81 |
| P1_r3 | True | True | True | True | True | 23808 | 1659 | 12.99 |
| F1_C1_P1 | True | True | True | True | True | 126976 | 0 | 12.93 |
| F1_LONG_P1 | True | True | True | True | True | 31744 | 0 | 13.20 |
| S1_P1_PROBE128 | True | True | True | True | True | 23808 | 0 | 13.09 |

## 5. 의존 관계 지표 요약 (P1; 세션별 dependency_summary.json, µs, 유효 행만; 정의는 PLAN_RESOLVED §3 / dependency_metrics.py docstring)

### P1_r1 (qwen, boot B2_132633) — coverage {'gpu_layer_rows_total': 23808, 'no_prev_deferred': 384, 'matched_rows': 23808} · violations {}

| graph(step) | rows | 지표 | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 2 ['step[DECODE bs=63]'] | 15872 | cold_ready_lateness_us | 15872 | 302.2 | 800.2 | 1028.5 | 4309.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | cold_pub_minus_hot_ready_us | 15872 | 302.2 | 800.2 | 1028.5 | 4309.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | exposed_wait_us | 15872 | 315.2 | 812.0 | 1041.0 | 4327.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | publish_to_release_us | 15872 | 13.2 | 145.0 | 396.0 | 538.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | submit_to_start_us | 15616 | 650.5 | 1172.8 | 1397.0 | 4664.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | cpu_compute_span_us | 15616 | 898.2 | 1407.5 | 1635.0 | 4883.2 |
| 2 ['step[DECODE bs=63]'] | 15872 | cold_budget_us | 15616 | 1247.5 | 1781.8 | 2014.5 | 5235.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | numa0_span_us | 15616 | 870.0 | 1376.5 | 1593.2 | 4708.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | numa1_span_us | 15616 | 835.5 | 1323.5 | 1524.0 | 3160.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | numa_completion_skew_us | 15616 | 30.8 | 80.2 | 168.5 | 2871.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | publish_delay_us | 15616 | 0.2 | 0.5 | 20.8 | 534.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | cpu_to_gpu_ready_us | 15872 | 43.8 | 166.0 | 416.8 | 563.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | go_minus_dtoh_end_us | 15872 | 5.5 | 6.5 | 8.2 | 26.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | hot_kernel_sum_us | 15872 | 323.8 | 435.0 | 525.2 | 603.0 |
| 2 ['step[DECODE bs=63]'] | 15872 | prev_n_cold_ids | 15616 | 17.0 | 32.0 | 41.0 | 124.0 |
| 2 | | share_cold_later_than_hot | 15872 | 0.885 | | | |
| 32 ['step[DECODE bs=2]'] | 7936 | cold_ready_lateness_us | 7936 | 0.0 | 0.0 | 0.0 | 1283.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | cold_pub_minus_hot_ready_us | 7936 | -85.0 | -65.8 | -31.0 | 1283.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | exposed_wait_us | 7936 | 10.8 | 14.0 | 15.0 | 1289.0 |
| 32 ['step[DECODE bs=2]'] | 7936 | publish_to_release_us | 7936 | 95.5 | 106.0 | 109.0 | 114.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | submit_to_start_us | 7808 | 6.0 | 14.5 | 58.5 | 1360.8 |
| 32 ['step[DECODE bs=2]'] | 7936 | cpu_compute_span_us | 7808 | 32.8 | 194.0 | 288.0 | 1542.5 |
| 32 ['step[DECODE bs=2]'] | 7936 | cold_budget_us | 7808 | 339.5 | 358.0 | 365.0 | 1593.0 |
| 32 ['step[DECODE bs=2]'] | 7936 | numa0_span_us | 7808 | 12.5 | 173.0 | 266.8 | 1115.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | numa1_span_us | 7808 | 11.0 | 171.2 | 264.5 | 1414.8 |
| 32 ['step[DECODE bs=2]'] | 7936 | numa_completion_skew_us | 7808 | 2.8 | 10.0 | 24.8 | 1250.0 |
| 32 ['step[DECODE bs=2]'] | 7936 | publish_delay_us | 7808 | 211.0 | 231.5 | 237.8 | 530.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | cpu_to_gpu_ready_us | 7936 | 99.8 | 110.2 | 113.0 | 118.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | go_minus_dtoh_end_us | 7936 | 4.8 | 5.8 | 7.5 | 35.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | hot_kernel_sum_us | 7936 | 74.2 | 82.0 | 82.8 | 84.0 |
| 32 ['step[DECODE bs=2]'] | 7936 | prev_n_cold_ids | 7808 | 0.0 | 2.0 | 3.0 | 6.0 |
| 32 | | share_cold_later_than_hot | 7936 | 0.006 | | | |

### P1_r2 (qwen, boot B2_132633) — coverage {'gpu_layer_rows_total': 23808, 'no_prev_deferred': 384, 'matched_rows': 23808} · violations {}

| graph(step) | rows | 지표 | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 2 ['step[DECODE bs=63]'] | 15872 | cold_ready_lateness_us | 15872 | 307.0 | 801.8 | 1022.5 | 2858.2 |
| 2 ['step[DECODE bs=63]'] | 15872 | cold_pub_minus_hot_ready_us | 15872 | 307.0 | 801.8 | 1022.5 | 2858.2 |
| 2 ['step[DECODE bs=63]'] | 15872 | exposed_wait_us | 15872 | 319.8 | 813.5 | 1034.0 | 2876.2 |
| 2 ['step[DECODE bs=63]'] | 15872 | publish_to_release_us | 15872 | 13.5 | 144.8 | 410.8 | 554.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | submit_to_start_us | 15616 | 655.2 | 1177.5 | 1401.5 | 3318.0 |
| 2 ['step[DECODE bs=63]'] | 15872 | cpu_compute_span_us | 15616 | 902.2 | 1410.2 | 1633.8 | 3536.2 |
| 2 ['step[DECODE bs=63]'] | 15872 | cold_budget_us | 15616 | 1251.8 | 1779.5 | 2018.2 | 3948.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | numa0_span_us | 15616 | 870.2 | 1378.0 | 1597.2 | 3046.2 |
| 2 ['step[DECODE bs=63]'] | 15872 | numa1_span_us | 15616 | 838.0 | 1314.8 | 1515.2 | 3472.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | numa_completion_skew_us | 15616 | 30.2 | 78.5 | 153.5 | 1659.0 |
| 2 ['step[DECODE bs=63]'] | 15872 | publish_delay_us | 15616 | 0.2 | 0.5 | 39.0 | 891.2 |
| 2 ['step[DECODE bs=63]'] | 15872 | cpu_to_gpu_ready_us | 15872 | 42.5 | 166.5 | 431.8 | 574.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | go_minus_dtoh_end_us | 15872 | 5.5 | 6.8 | 11.2 | 1556.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | hot_kernel_sum_us | 15872 | 326.0 | 437.2 | 527.0 | 611.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | prev_n_cold_ids | 15616 | 16.0 | 31.0 | 39.0 | 79.0 |
| 2 | | share_cold_later_than_hot | 15872 | 0.884 | | | |
| 32 ['step[DECODE bs=2]'] | 7936 | cold_ready_lateness_us | 7936 | 0.0 | 0.0 | 0.0 | 8842.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | cold_pub_minus_hot_ready_us | 7936 | -83.0 | -63.5 | -16.5 | 8842.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | exposed_wait_us | 7936 | 10.8 | 14.0 | 15.2 | 8856.5 |
| 32 ['step[DECODE bs=2]'] | 7936 | publish_to_release_us | 7936 | 93.2 | 105.8 | 109.2 | 115.8 |
| 32 ['step[DECODE bs=2]'] | 7936 | submit_to_start_us | 7808 | 5.2 | 13.2 | 72.5 | 8930.5 |
| 32 ['step[DECODE bs=2]'] | 7936 | cpu_compute_span_us | 7808 | 34.0 | 196.2 | 293.0 | 9135.5 |
| 32 ['step[DECODE bs=2]'] | 7936 | cold_budget_us | 7808 | 334.8 | 356.2 | 364.8 | 9180.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | numa0_span_us | 7808 | 12.8 | 174.5 | 267.8 | 9102.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | numa1_span_us | 7808 | 10.2 | 171.8 | 262.2 | 824.0 |
| 32 ['step[DECODE bs=2]'] | 7936 | numa_completion_skew_us | 7808 | 2.5 | 10.2 | 21.8 | 8739.5 |
| 32 ['step[DECODE bs=2]'] | 7936 | publish_delay_us | 7808 | 205.0 | 227.8 | 235.5 | 2960.5 |
| 32 ['step[DECODE bs=2]'] | 7936 | cpu_to_gpu_ready_us | 7936 | 97.5 | 110.0 | 113.2 | 119.8 |
| 32 ['step[DECODE bs=2]'] | 7936 | go_minus_dtoh_end_us | 7936 | 4.8 | 5.8 | 7.2 | 248.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | hot_kernel_sum_us | 7936 | 71.5 | 81.8 | 82.5 | 83.8 |
| 32 ['step[DECODE bs=2]'] | 7936 | prev_n_cold_ids | 7808 | 0.0 | 2.0 | 3.0 | 10.0 |
| 32 | | share_cold_later_than_hot | 7936 | 0.008 | | | |

### P1_r3 (qwen, boot B2_132633) — coverage {'gpu_layer_rows_total': 23808, 'no_prev_deferred': 384, 'matched_rows': 23808} · violations {'t_go_before_dtoh_end_within_5us(jitter)': 1659}

| graph(step) | rows | 지표 | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 2 ['step[DECODE bs=63]'] | 15872 | cold_ready_lateness_us | 15872 | 328.5 | 837.5 | 1073.5 | 3981.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | cold_pub_minus_hot_ready_us | 15872 | 328.5 | 837.5 | 1073.5 | 3981.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | exposed_wait_us | 15872 | 341.2 | 849.5 | 1084.0 | 3992.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | publish_to_release_us | 15872 | 13.8 | 134.5 | 433.5 | 551.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | submit_to_start_us | 15616 | 677.2 | 1214.8 | 1442.2 | 4333.2 |
| 2 ['step[DECODE bs=63]'] | 15872 | cpu_compute_span_us | 15616 | 923.0 | 1448.8 | 1675.8 | 4565.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | cold_budget_us | 15616 | 1269.5 | 1814.0 | 2056.2 | 4923.2 |
| 2 ['step[DECODE bs=63]'] | 15872 | numa0_span_us | 15616 | 892.0 | 1416.0 | 1639.8 | 4540.2 |
| 2 ['step[DECODE bs=63]'] | 15872 | numa1_span_us | 15616 | 856.8 | 1358.0 | 1556.5 | 2665.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | numa_completion_skew_us | 15616 | 32.5 | 84.2 | 161.2 | 2432.2 |
| 2 ['step[DECODE bs=63]'] | 15872 | publish_delay_us | 15616 | 0.2 | 0.5 | 30.8 | 644.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | cpu_to_gpu_ready_us | 15872 | 43.5 | 156.5 | 452.8 | 570.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | go_minus_dtoh_end_us | 15872 | 5.2 | 8.5 | 10.0 | 1622.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | hot_kernel_sum_us | 15872 | 323.5 | 436.2 | 524.8 | 607.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | prev_n_cold_ids | 15616 | 17.0 | 32.0 | 42.0 | 77.0 |
| 2 | | share_cold_later_than_hot | 15872 | 0.896 | | | |
| 32 ['step[DECODE bs=2]'] | 7936 | cold_ready_lateness_us | 7936 | 0.0 | 0.0 | 0.0 | 546.8 |
| 32 ['step[DECODE bs=2]'] | 7936 | cold_pub_minus_hot_ready_us | 7936 | -85.0 | -69.5 | -45.0 | 546.8 |
| 32 ['step[DECODE bs=2]'] | 7936 | exposed_wait_us | 7936 | 10.5 | 14.0 | 14.8 | 561.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | publish_to_release_us | 7936 | 95.5 | 105.8 | 109.0 | 114.0 |
| 32 ['step[DECODE bs=2]'] | 7936 | submit_to_start_us | 7808 | 5.5 | 9.0 | 46.8 | 650.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | cpu_compute_span_us | 7808 | 33.8 | 189.0 | 280.0 | 900.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | cold_budget_us | 7808 | 338.0 | 355.8 | 363.5 | 904.5 |
| 32 ['step[DECODE bs=2]'] | 7936 | numa0_span_us | 7808 | 10.2 | 169.2 | 260.5 | 882.0 |
| 32 ['step[DECODE bs=2]'] | 7936 | numa1_span_us | 7808 | 13.5 | 169.8 | 259.2 | 413.5 |
| 32 ['step[DECODE bs=2]'] | 7936 | numa_completion_skew_us | 7808 | 4.2 | 8.8 | 18.2 | 745.0 |
| 32 ['step[DECODE bs=2]'] | 7936 | publish_delay_us | 7808 | 209.5 | 228.5 | 234.8 | 490.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | cpu_to_gpu_ready_us | 7936 | 99.8 | 110.0 | 113.0 | 117.8 |
| 32 ['step[DECODE bs=2]'] | 7936 | go_minus_dtoh_end_us | 7936 | 4.5 | 5.5 | 7.0 | 27.0 |
| 32 ['step[DECODE bs=2]'] | 7936 | hot_kernel_sum_us | 7936 | 74.0 | 81.8 | 82.8 | 83.8 |
| 32 ['step[DECODE bs=2]'] | 7936 | prev_n_cold_ids | 7808 | 0.0 | 2.0 | 3.0 | 5.0 |
| 32 | | share_cold_later_than_hot | 7936 | 0.004 | | | |

### F1_C1_P1 (qwen, boot F1_ON_134513) — coverage {'gpu_layer_rows_total': 126976, 'no_prev_deferred': 2048, 'matched_rows': 126976} · violations {}

| graph(step) | rows | 지표 | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 35 ['step[DECODE bs=1]'] | 126976 | cold_ready_lateness_us | 126976 | 0.0 | 0.0 | 0.0 | 6488.0 |
| 35 ['step[DECODE bs=1]'] | 126976 | cold_pub_minus_hot_ready_us | 126976 | -63.2 | -55.8 | -40.0 | 6488.0 |
| 35 ['step[DECODE bs=1]'] | 126976 | exposed_wait_us | 126976 | 10.8 | 14.0 | 14.8 | 6499.5 |
| 35 ['step[DECODE bs=1]'] | 126976 | publish_to_release_us | 126976 | 74.0 | 80.0 | 81.8 | 97.0 |
| 35 ['step[DECODE bs=1]'] | 126976 | submit_to_start_us | 124928 | 5.5 | 11.2 | 27.8 | 6553.0 |
| 35 ['step[DECODE bs=1]'] | 126976 | cpu_compute_span_us | 124928 | 20.5 | 143.0 | 200.8 | 6754.5 |
| 35 ['step[DECODE bs=1]'] | 126976 | cold_budget_us | 124928 | 301.2 | 312.0 | 315.8 | 6790.8 |
| 35 ['step[DECODE bs=1]'] | 126976 | numa0_span_us | 124928 | 7.8 | 128.5 | 182.8 | 6730.0 |
| 35 ['step[DECODE bs=1]'] | 126976 | numa1_span_us | 124928 | 6.2 | 125.2 | 179.0 | 1501.8 |
| 35 ['step[DECODE bs=1]'] | 126976 | numa_completion_skew_us | 124928 | 1.0 | 7.0 | 19.0 | 6531.0 |
| 35 ['step[DECODE bs=1]'] | 126976 | publish_delay_us | 124928 | 210.5 | 222.5 | 227.2 | 5752.5 |
| 35 ['step[DECODE bs=1]'] | 126976 | cpu_to_gpu_ready_us | 126976 | 77.0 | 83.5 | 85.2 | 100.2 |
| 35 ['step[DECODE bs=1]'] | 126976 | go_minus_dtoh_end_us | 126976 | 4.5 | 5.8 | 6.8 | 1358.2 |
| 35 ['step[DECODE bs=1]'] | 126976 | hot_kernel_sum_us | 126976 | 51.8 | 52.5 | 53.0 | 56.8 |
| 35 ['step[DECODE bs=1]'] | 126976 | prev_n_cold_ids | 124928 | 0.0 | 1.0 | 2.0 | 6.0 |
| 35 | | share_cold_later_than_hot | 126976 | 0.004 | | | |

### F1_LONG_P1 (qwen, boot F1_ON_134513) — coverage {'gpu_layer_rows_total': 31744, 'no_prev_deferred': 512, 'matched_rows': 31744} · violations {}

| graph(step) | rows | 지표 | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 26 ['step[DECODE bs=8]'] | 31744 | cold_ready_lateness_us | 31744 | 0.0 | 0.0 | 26.2 | 5262.5 |
| 26 ['step[DECODE bs=8]'] | 31744 | cold_pub_minus_hot_ready_us | 31744 | -184.8 | -96.5 | 26.2 | 5262.5 |
| 26 ['step[DECODE bs=8]'] | 31744 | exposed_wait_us | 31744 | 10.2 | 14.0 | 39.5 | 5276.5 |
| 26 ['step[DECODE bs=8]'] | 31744 | publish_to_release_us | 31744 | 194.8 | 228.2 | 243.2 | 296.0 |
| 26 ['step[DECODE bs=8]'] | 31744 | submit_to_start_us | 31232 | 6.0 | 86.5 | 214.5 | 5465.8 |
| 26 ['step[DECODE bs=8]'] | 31744 | cpu_compute_span_us | 31232 | 204.8 | 413.5 | 524.5 | 5670.5 |
| 26 ['step[DECODE bs=8]'] | 31744 | cold_budget_us | 31232 | 587.0 | 645.8 | 690.5 | 5855.8 |
| 26 ['step[DECODE bs=8]'] | 31744 | numa0_span_us | 31232 | 150.0 | 355.0 | 471.8 | 1398.2 |
| 26 ['step[DECODE bs=8]'] | 31744 | numa1_span_us | 31232 | 149.5 | 350.2 | 466.0 | 5605.0 |
| 26 ['step[DECODE bs=8]'] | 31744 | numa_completion_skew_us | 31232 | 2.2 | 18.8 | 46.0 | 5241.0 |
| 26 ['step[DECODE bs=8]'] | 31744 | publish_delay_us | 31232 | 168.2 | 324.8 | 346.8 | 1354.8 |
| 26 ['step[DECODE bs=8]'] | 31744 | cpu_to_gpu_ready_us | 31744 | 203.5 | 236.5 | 251.5 | 303.8 |
| 26 ['step[DECODE bs=8]'] | 31744 | go_minus_dtoh_end_us | 31744 | 4.5 | 5.5 | 7.0 | 1138.2 |
| 26 ['step[DECODE bs=8]'] | 31744 | hot_kernel_sum_us | 31744 | 170.0 | 203.5 | 223.8 | 278.5 |
| 26 ['step[DECODE bs=8]'] | 31744 | prev_n_cold_ids | 31232 | 2.0 | 5.0 | 7.0 | 15.0 |
| 26 | | share_cold_later_than_hot | 31744 | 0.013 | | | |

### S1_P1_PROBE128 (qwen, boot SMOKE_131024) — coverage {'gpu_layer_rows_total': 23808, 'no_prev_deferred': 384, 'matched_rows': 23808} · violations {}

| graph(step) | rows | 지표 | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 2 ['step[DECODE bs=63]'] | 15872 | cold_ready_lateness_us | 15872 | 324.5 | 819.5 | 1050.5 | 3797.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | cold_pub_minus_hot_ready_us | 15872 | 324.5 | 819.5 | 1050.5 | 3797.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | exposed_wait_us | 15872 | 337.0 | 831.8 | 1061.0 | 3815.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | publish_to_release_us | 15872 | 13.2 | 134.2 | 389.8 | 537.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | submit_to_start_us | 15616 | 671.8 | 1190.0 | 1432.8 | 4028.0 |
| 2 ['step[DECODE bs=63]'] | 15872 | cpu_compute_span_us | 15616 | 918.0 | 1426.8 | 1665.5 | 4241.8 |
| 2 ['step[DECODE bs=63]'] | 15872 | cold_budget_us | 15616 | 1269.8 | 1793.8 | 2034.2 | 4480.0 |
| 2 ['step[DECODE bs=63]'] | 15872 | numa0_span_us | 15616 | 887.0 | 1390.8 | 1624.8 | 4216.0 |
| 2 ['step[DECODE bs=63]'] | 15872 | numa1_span_us | 15616 | 871.5 | 1375.2 | 1587.0 | 3985.0 |
| 2 ['step[DECODE bs=63]'] | 15872 | numa_completion_skew_us | 15616 | 18.0 | 68.5 | 142.0 | 3476.0 |
| 2 ['step[DECODE bs=63]'] | 15872 | publish_delay_us | 15616 | 0.2 | 0.5 | 20.2 | 969.0 |
| 2 ['step[DECODE bs=63]'] | 15872 | cpu_to_gpu_ready_us | 15872 | 44.2 | 156.0 | 410.8 | 557.2 |
| 2 ['step[DECODE bs=63]'] | 15872 | go_minus_dtoh_end_us | 15872 | 5.2 | 6.8 | 8.2 | 1512.2 |
| 2 ['step[DECODE bs=63]'] | 15872 | hot_kernel_sum_us | 15872 | 323.8 | 435.2 | 524.2 | 599.5 |
| 2 ['step[DECODE bs=63]'] | 15872 | prev_n_cold_ids | 15616 | 17.0 | 31.0 | 40.0 | 99.0 |
| 2 | | share_cold_later_than_hot | 15872 | 0.896 | | | |
| 32 ['step[DECODE bs=2]'] | 7936 | cold_ready_lateness_us | 7936 | 0.0 | 0.0 | 36.0 | 1114.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | cold_pub_minus_hot_ready_us | 7936 | -83.0 | -64.0 | 36.0 | 1114.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | exposed_wait_us | 7936 | 10.5 | 14.0 | 49.5 | 1126.5 |
| 32 ['step[DECODE bs=2]'] | 7936 | publish_to_release_us | 7936 | 93.2 | 104.2 | 107.5 | 111.5 |
| 32 ['step[DECODE bs=2]'] | 7936 | submit_to_start_us | 7808 | 7.2 | 22.2 | 123.8 | 1209.0 |
| 32 ['step[DECODE bs=2]'] | 7936 | cpu_compute_span_us | 7808 | 37.2 | 220.5 | 323.5 | 1432.5 |
| 32 ['step[DECODE bs=2]'] | 7936 | cold_budget_us | 7808 | 337.2 | 357.2 | 374.0 | 1456.0 |
| 32 ['step[DECODE bs=2]'] | 7936 | numa0_span_us | 7808 | 10.2 | 189.5 | 289.0 | 1414.5 |
| 32 ['step[DECODE bs=2]'] | 7936 | numa1_span_us | 7808 | 15.5 | 198.5 | 300.0 | 717.5 |
| 32 ['step[DECODE bs=2]'] | 7936 | numa_completion_skew_us | 7808 | 6.2 | 16.8 | 64.8 | 1266.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | publish_delay_us | 7808 | 204.8 | 225.5 | 232.8 | 1289.8 |
| 32 ['step[DECODE bs=2]'] | 7936 | cpu_to_gpu_ready_us | 7936 | 97.2 | 108.5 | 111.5 | 116.0 |
| 32 ['step[DECODE bs=2]'] | 7936 | go_minus_dtoh_end_us | 7936 | 4.8 | 6.0 | 8.0 | 1193.0 |
| 32 ['step[DECODE bs=2]'] | 7936 | hot_kernel_sum_us | 7936 | 72.5 | 81.8 | 82.8 | 84.2 |
| 32 ['step[DECODE bs=2]'] | 7936 | prev_n_cold_ids | 7808 | 0.0 | 2.0 | 3.0 | 5.0 |
| 32 | | share_cold_later_than_hot | 7936 | 0.014 | | | |

## 6. CPU 단독 지표·자원 (P2; kt_evt 세션 창, perf stat, pcm-memory)

### P2_r1

kt_evt 레코드 24490

| rows | 지표 | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 1 | submit_to_start_us | 682 | 118.3 | 2552.2 | 5337.1 | 28007.4 |
| 1 | cpu_compute_span_us | 682 | 8737.3 | 11017.5 | 12320.1 | 38994.6 |
| 1 | numa_skew_us | 682 | 155.1 | 389.5 | 1265.5 | 29064.7 |
| 1 | numa0_span_us | 682 | 5954.4 | 7878.8 | 9208.8 | 35779.1 |
| 1 | numa1_span_us | 682 | 6039.9 | 7943.1 | 8987.2 | 12783.9 |
| 1 | n_cold_ids | 682 | 371.0 | 792.0 | 960.0 | 1289.0 |
| 1 | go_to_done_set_us | 682 | 117.7 | 2551.8 | 5336.5 | 28006.4 |
| 64 | submit_to_start_us | 15872 | 677.1 | 1176.4 | 1399.1 | 7407.7 |
| 64 | cpu_compute_span_us | 15872 | 907.9 | 1395.8 | 1624.2 | 7614.5 |
| 64 | numa_skew_us | 15872 | 24.3 | 63.3 | 132.9 | 2861.0 |
| 64 | numa0_span_us | 15872 | 872.5 | 1355.8 | 1570.8 | 3320.1 |
| 64 | numa1_span_us | 15872 | 844.3 | 1319.5 | 1533.3 | 5094.7 |
| 64 | n_cold_ids | 15872 | 17.0 | 31.0 | 40.0 | 99.0 |
| 64 | go_to_done_set_us | 15872 | 676.5 | 1176.1 | 1398.7 | 7406.8 |
| 2 | submit_to_start_us | 7936 | 9.0 | 22.1 | 80.1 | 1207.8 |
| 2 | cpu_compute_span_us | 7936 | 32.7 | 195.5 | 288.3 | 1433.1 |
| 2 | numa_skew_us | 7936 | 2.4 | 11.1 | 25.6 | 1279.4 |
| 2 | numa0_span_us | 7936 | 13.2 | 173.1 | 264.3 | 722.7 |
| 2 | numa1_span_us | 7936 | 10.2 | 173.2 | 265.7 | 1415.4 |
| 2 | n_cold_ids | 7936 | 0.0 | 2.0 | 3.0 | 6.0 |
| 2 | go_to_done_set_us | 7936 | 8.6 | 20.4 | 76.4 | 1207.5 |

perf stat (스케줄러 4 프로세스, 세션 전체 창): `{"msec task-clock": 2412718.6, "cpus_utilized": 59.876, "cycles": 4797138188289.0, "instructions": 6192352881516.0, "ipc": 1.29, "context-switches": 1217853.0, "cpu-migrations": 27095.0, "cache-misses": 83849815902.0, "elapsed_s": 40.295073}`
pcm-memory (요청 창 내 표본 평균, 경계 표본 boundary_partial 계수, host 전체 값): `{"samples_total": 38, "samples_in_window": 38, "boundary_partial": 0, "system_read_mean": 144783.69210526315, "system_write_mean": 16728.566578947368, "system_memory_mean": 161512.25789473683, "unit": "MB/s (pcm-memory 기본)", "header_sample": ["System|DRAMWrite", "System|PMMREAD", "System|PMMWrite", "System|Read", "System|Write", "System|Memory"]}`

### P2_r2

kt_evt 레코드 24490

| rows | 지표 | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 1 | submit_to_start_us | 682 | 114.9 | 144.2 | 4741.5 | 6230.4 |
| 1 | cpu_compute_span_us | 682 | 8795.1 | 10865.2 | 11642.4 | 16120.0 |
| 1 | numa_skew_us | 682 | 129.6 | 366.3 | 494.4 | 3053.2 |
| 1 | numa0_span_us | 682 | 6034.6 | 7767.4 | 8952.6 | 13311.4 |
| 1 | numa1_span_us | 682 | 6182.1 | 7976.9 | 8992.7 | 10256.7 |
| 1 | n_cold_ids | 682 | 382.0 | 798.0 | 959.0 | 1284.0 |
| 1 | go_to_done_set_us | 682 | 114.3 | 142.4 | 4740.9 | 6230.1 |
| 64 | submit_to_start_us | 15872 | 656.4 | 1160.0 | 1381.5 | 4958.9 |
| 64 | cpu_compute_span_us | 15872 | 893.1 | 1379.7 | 1607.4 | 5170.4 |
| 64 | numa_skew_us | 15872 | 25.8 | 63.6 | 140.4 | 3790.3 |
| 64 | numa0_span_us | 15872 | 863.8 | 1347.0 | 1567.1 | 5141.0 |
| 64 | numa1_span_us | 15872 | 836.0 | 1308.0 | 1512.4 | 4073.2 |
| 64 | n_cold_ids | 15872 | 16.0 | 31.0 | 40.0 | 84.0 |
| 64 | go_to_done_set_us | 15872 | 656.0 | 1159.6 | 1381.2 | 4958.3 |
| 2 | submit_to_start_us | 7936 | 9.5 | 22.7 | 86.1 | 873.4 |
| 2 | cpu_compute_span_us | 7936 | 33.5 | 193.5 | 291.5 | 1087.5 |
| 2 | numa_skew_us | 7936 | 2.5 | 9.4 | 20.9 | 813.6 |
| 2 | numa0_span_us | 7936 | 12.7 | 170.8 | 266.7 | 1073.3 |
| 2 | numa1_span_us | 7936 | 11.0 | 171.8 | 264.2 | 929.5 |
| 2 | n_cold_ids | 7936 | 0.0 | 2.0 | 3.0 | 5.0 |
| 2 | go_to_done_set_us | 7936 | 9.1 | 20.7 | 79.7 | 873.0 |

perf stat (스케줄러 4 프로세스, 세션 전체 창): `{"msec task-clock": 2377963.66, "cpus_utilized": 59.058, "cycles": 4727496065236.0, "instructions": 6086937099045.0, "ipc": 1.29, "context-switches": 1231892.0, "cpu-migrations": 27588.0, "cache-misses": 83419115153.0, "elapsed_s": 40.26475298}`
pcm-memory (요청 창 내 표본 평균, 경계 표본 boundary_partial 계수, host 전체 값): `{"samples_total": 38, "samples_in_window": 38, "boundary_partial": 0, "system_read_mean": 143391.06236842106, "system_write_mean": 16680.848157894736, "system_memory_mean": 160071.90921052633, "unit": "MB/s (pcm-memory 기본)", "header_sample": ["System|DRAMWrite", "System|PMMREAD", "System|PMMWrite", "System|Read", "System|Write", "System|Memory"]}`

### P2_r3

kt_evt 레코드 24490

| rows | 지표 | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 1 | submit_to_start_us | 682 | 119.0 | 1832.8 | 5394.8 | 108397.8 |
| 1 | cpu_compute_span_us | 682 | 8779.0 | 10935.0 | 12218.2 | 118037.7 |
| 1 | numa_skew_us | 682 | 128.6 | 319.6 | 640.2 | 103131.2 |
| 1 | numa0_span_us | 682 | 5998.1 | 7861.2 | 9226.8 | 115183.6 |
| 1 | numa1_span_us | 682 | 6120.5 | 7957.4 | 8965.7 | 12050.6 |
| 1 | n_cold_ids | 682 | 381.0 | 799.0 | 960.0 | 1280.0 |
| 1 | go_to_done_set_us | 682 | 118.4 | 1832.4 | 5394.2 | 108397.0 |
| 64 | submit_to_start_us | 15872 | 691.7 | 1187.2 | 1412.1 | 3035.6 |
| 64 | cpu_compute_span_us | 15872 | 922.1 | 1408.4 | 1631.2 | 3233.2 |
| 64 | numa_skew_us | 15872 | 26.0 | 65.2 | 138.5 | 1680.6 |
| 64 | numa0_span_us | 15872 | 885.4 | 1371.8 | 1576.2 | 3194.1 |
| 64 | numa1_span_us | 15872 | 856.4 | 1329.8 | 1528.5 | 3024.1 |
| 64 | n_cold_ids | 15872 | 17.0 | 31.0 | 40.0 | 80.0 |
| 64 | go_to_done_set_us | 15872 | 691.3 | 1186.6 | 1411.5 | 3035.3 |
| 2 | submit_to_start_us | 7936 | 8.8 | 18.7 | 67.8 | 2088.2 |
| 2 | cpu_compute_span_us | 7936 | 32.7 | 189.2 | 283.3 | 2298.6 |
| 2 | numa_skew_us | 7936 | 2.6 | 9.6 | 23.6 | 2084.5 |
| 2 | numa0_span_us | 7936 | 13.1 | 167.4 | 259.3 | 1070.2 |
| 2 | numa1_span_us | 7936 | 10.0 | 168.3 | 258.4 | 1272.5 |
| 2 | n_cold_ids | 7936 | 0.0 | 2.0 | 3.0 | 6.0 |
| 2 | go_to_done_set_us | 7936 | 8.4 | 17.7 | 66.7 | 2087.9 |

perf stat (스케줄러 4 프로세스, 세션 전체 창): `{"msec task-clock": 2430384.94, "cpus_utilized": 57.581, "cycles": 4832109480727.0, "instructions": 6206380581892.0, "ipc": 1.28, "context-switches": 1332121.0, "cpu-migrations": 30012.0, "cache-misses": 85328286429.0, "elapsed_s": 42.207974011}`
pcm-memory (요청 창 내 표본 평균, 경계 표본 boundary_partial 계수, host 전체 값): `{"samples_total": 40, "samples_in_window": 40, "boundary_partial": 0, "system_read_mean": 139422.6445, "system_write_mean": 16056.13825, "system_memory_mean": 155478.78475, "unit": "MB/s (pcm-memory 기본)", "header_sample": ["System|DRAMWrite", "System|PMMREAD", "System|PMMWrite", "System|Read", "System|Write", "System|Memory"]}`

부팅 B3_133528 stdout: [kt-tq] 48행, [kt-wrap] 1593행 (원문은 raw_md)

## 7. 기존 자료 재집계 (M1; IDE_073 D2 retry2, profiles/corrected_metrics.csv, HEURISTIC_MOE_BOUNDARY)

| trace | window | window s | op sum s | busy union s | busy frac | compute union s | memcpy union s | overlap s | 원 busy s / 원 frac |
|---|---|---|---|---|---|---|---|---|---|
| TP-0. | trace_full | 40.961 | 16.439 | 16.073 | 0.392 | 12.419 | 3.654 | 0.001 | 16.438 / 0.401 |
| TP-0. | request_window | 37.521 | 16.030 | 15.815 | 0.421 | 12.162 | 3.653 | 0.001 | 16.438 / 0.401 |
| TP-1. | trace_full | 40.962 | 31.757 | 23.433 | 0.572 | 23.423 | 0.010 | 0.001 | 31.758 / 0.775 |
| TP-1. | request_window | 37.521 | 31.711 | 23.401 | 0.624 | 23.391 | 0.010 | 0.001 | 31.758 / 0.775 |
| TP-2. | trace_full | 40.961 | 32.150 | 23.699 | 0.579 | 23.689 | 0.010 | 0.002 | 32.151 / 0.785 |
| TP-2. | request_window | 37.521 | 31.677 | 23.410 | 0.624 | 23.401 | 0.010 | 0.002 | 32.151 / 0.785 |
| TP-3. | trace_full | 40.962 | 32.147 | 23.671 | 0.578 | 23.661 | 0.010 | 0.002 | 32.149 / 0.785 |
| TP-3. | request_window | 37.521 | 31.706 | 23.400 | 0.624 | 23.390 | 0.010 | 0.002 | 32.149 / 0.785 |

IDE_073 D2 retry2 층 타임라인(패턴 휴리스틱, profiles/gpu_layer_timeline_d2_retry2_TP*.csv.gz): TP-0 graph 2(bs=63) hot 종료→HtoD 시작 gap n 15872 p50 343.2 p90 747.0 p99 1121.5 max 3751.5 µs 합 5.988 s; graph 32(bs=2) p50 10.8 µs. TP-1~3 all-reduce 커널(짝수 순번) duration p50 445 µs 합 7.69 s (순번↔위치 귀속 미확정).

## 8. 오류·재시도·미실행·차단

| 항목 | 상태 | 근거 |
|---|---|---|
| GLM 게이트 G_135723 | BLOCKED_NORMAL_OUTPUT | eval/results/IDE_074_20260917/glm/BASIC4/G_135723/gate_result.json |

## 9. 실행량 (원장 재계산): 세션 17/24 · 부팅 7/12 · 품질 문항 4/80

## 10. 파일

- profiles/: corrected_metrics.csv, observer_windows.csv, observer_overhead.csv, validation_results_*.json, gpu_layer_timeline_*.csv.gz, code_map_survey.md, probe_map.md
- 세션 디렉터리: dependency_metrics.csv.gz, dependency_summary.json, missing_coverage.csv, gpu_layer_timeline_TP0.csv.gz, requests.jsonl, *_timeseries.csv, perf_stat.txt, pcm_memory.csv, metrics.json
- 서버 보존(대형): ~/.cache/huggingface/kt/ide074/<boot>/{kt_evt.csv, kt_evt.csv.map, profiles/*.trace.json.gz} — ARTIFACT_INDEX.csv 에 SHA256·크기
- FULL_RAW_DATA.md + raw_md/part-*.md, ARTIFACT_INDEX.csv, SHA256SUMS.txt, PUBLISH_RECEIPT.md(게시 후)
