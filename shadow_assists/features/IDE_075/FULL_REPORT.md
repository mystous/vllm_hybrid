# FULL_REPORT — IDE_075 후속 추가 측정 (2026-09-17T20:53:52+09:00)

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
| OFF_OPEN2_161158 | OFF_OPEN2/OFF | HEALTH_OK |  |
| CORE2_162053 | CORE2/CORE | HEALTH_OK |  |
| OFF_CLOSE2_165803 | OFF_CLOSE2/OFF | HEALTH_OK |  |
| GLM_D1_170439 | None/GLM_DIAG | DIED |  |
| GLM_D2_170535 | None/GLM_DIAG | HEALTH_OK |  |
| OFF_X_171100 | OFF_X/OFF | HEALTH_OK |  |
| CORE3_171618 | CORE3/CORE | HEALTH_OK |  |
| GLM_D3_172545 | None/GLM_DIAG | HEALTH_OK |  |
| GLM_D4_173122 | None/GLM_DIAG | HEALTH_OK |  |
| GLM_D5_173737 | None/GLM_DIAG | HEALTH_OK |  |
| OFF_Y_174421 | OFF_Y/OFF | HEALTH_OK |  |
| CORE4_174930 | CORE4/CORE | HEALTH_OK |  |
| GLM_D6_181159 | None/GLM_DIAG | HEALTH_OK |  |
| GLM_D7_183516 | None/GLM_DIAG | HEALTH_OK |  |
| FOCUS2_183951 | FOCUS2/CORE | HEALTH_OK |  |
| GLM_D8_185927 | None/GLM_DIAG | HEALTH_OK |  |
| GLM_D9_194029 | None/GLM_DIAG | HEALTH_OK |  |

실행량: IDE_074 세션 17/부팅 7 + IDE_075 기본 세션 6/부팅 시도 5 (중단 2) → 합계 세션 23/24 · 부팅 12/12 · smoke 요청 64 · warmup 요청 512. **확장 단계(사용자 승인, 상한 밖)**: 세션 35 · 부팅 17 (`state/budget_reconciliation.json`)

## 2. 모든 세션 (원값)

| boot | session | mode | client | workload | C | n | 완료/실패 | dur s | in | out | out tok/s | TTFT p50/p95 | TPOT p50/p95 | E2EL p50/p95 | valid | .so | start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CORE2_162053 | S10_CORR | CORR | probe | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 32.3 | null | 16384 | 506.81 | 2608/3169 | 97.9/108.0 | 15218/15305 | True | d659ca0b274c | 2026-09-17T16:44:49.314498+09:00 |
| CORE2_162053 | S11_RESOURCE | RESOURCE | probe | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 32.4 | null | 16384 | 505.79 | 2600/3123 | 99.3/109.0 | 15290/15432 | True | d659ca0b274c | 2026-09-17T16:46:55.653443+09:00 |
| CORE2_162053 | S12_C1_CORR | CORR | probe | M073_QWEN_LOW_CONCURRENCY | 1 | 16 | 16/0 | 35.4 | null | 2048 | 57.79 | 315/343 | 15.0/15.1 | 2216/2256 | True | d659ca0b274c | 2026-09-17T16:47:39.221350+09:00 |
| CORE2_162053 | S13_LONG_CORR | CORR | probe | M073_QWEN_LONGER_PREFILL | 8 | 32 | 32/0 | 30.3 | null | 4096 | 135.15 | 2641/3418 | 39.6/50.2 | 7675/8036 | True | d659ca0b274c | 2026-09-17T16:54:04.323899+09:00 |
| CORE2_162053 | S14_FOCUS_sched | FOCUS | probe | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 33.0 | null | 16384 | 496.47 | 2639/3199 | 99.4/110.3 | 15439/15519 | True | d659ca0b274c | 2026-09-17T16:56:34.751515+09:00 |
| CORE2_162053 | S9_CORR | CORR | probe | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 34.7 | null | 16384 | 472.36 | 2808/3798 | 106.1/125.1 | 17448/17556 | True | d659ca0b274c | 2026-09-17T16:28:27.611504+09:00 |
| CORE3_171618 | S17_CORR_pinned | CORR | probe | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 32.7 | null | 16384 | 500.60 | 2630/3218 | 98.2/109.1 | 15355/15437 | True | d659ca0b274c | 2026-09-17T17:19:07.378737+09:00 |
| CORE3_171618 | S18_CORR_pinned | CORR | probe | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 32.9 | null | 16384 | 498.58 | 2611/3161 | 99.5/109.6 | 15411/15492 | True | d659ca0b274c | 2026-09-17T17:21:10.207836+09:00 |
| CORE3_171618 | S19_FOCUS_pinned | FOCUS | probe | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 33.6 | null | 16384 | 487.90 | 2651/3161 | 101.4/112.3 | 15510/15561 | True | d659ca0b274c | 2026-09-17T17:23:17.264368+09:00 |
| CORE3_171618 | S20_RESOURCE_pinned | RESOURCE | probe | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 32.7 | null | 16384 | 500.77 | 2648/3141 | 99.8/110.0 | 15431/15514 | True | d659ca0b274c | 2026-09-17T17:24:31.674875+09:00 |
| CORE4_174930 | S22_CORR_vllmw | CORR | vllmw | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.8 | 65886 | 16384 | 689.38 | 2259/3319 | 65.8/73.2 | 10837/10859 | True | d659ca0b274c | 2026-09-17T17:52:26.325606+09:00 |
| CORE4_174930 | S23_CORR_vllmw | CORR | vllmw | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.5 | 65886 | 16384 | 696.23 | 2274/3227 | 64.4/73.0 | 10832/10849 | True | d659ca0b274c | 2026-09-17T17:54:29.911834+09:00 |
| CORE4_174930 | S24_RESOURCE_vllmw | RESOURCE | vllmw | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.4 | 65886 | 16384 | 699.76 | 2226/3103 | 65.6/72.4 | 10638/10662 | True | d659ca0b274c | 2026-09-17T17:56:40.295584+09:00 |
| CORE4_174930 | S25_FOCUS_vllmw | FOCUS | vllmw | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 24.2 | 65886 | 16384 | 677.33 | 2242/3321 | 66.9/77.3 | 11252/11270 | True | d659ca0b274c | 2026-09-17T17:57:26.711449+09:00 |
| CORE4_174930 | S26_C1_CORR_vllmw | CORR | vllmw | M073_QWEN_LOW_CONCURRENCY | 1 | 16 | 16/0 | 35.5 | 8234 | 2048 | 57.62 | 326/439 | 14.7/14.8 | 2197/2313 | True | d659ca0b274c | 2026-09-17T17:58:38.184999+09:00 |
| CORE4_174930 | S27_LONG_CORR_vllmw | CORR | vllmw | M073_QWEN_LONGER_PREFILL | 8 | 32 | 32/0 | 25.7 | 131147 | 4096 | 159.37 | 2584/3390 | 30.6/43.0 | 6417/6565 | True | d659ca0b274c | 2026-09-17T18:05:22.600213+09:00 |
| CORE_153555 | S2_CORE | CORE | vllm | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.1 | 65886 | 16384 | 708.22 | 2236/3261 | 64.5/70.5 | 10492/10515 | True | a4e9045a038b | 2026-09-17T15:38:42.908276+09:00 |
| CORE_153555 | S3_CORR | CORR | vllm | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.4 | 65886 | 16384 | 699.09 | 2283/3223 | 64.1/72.1 | 10547/10675 | True | a4e9045a038b | 2026-09-17T15:39:26.427098+09:00 |
| CORE_153555 | S4_CORR | CORR | vllm | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.4 | 65886 | 16384 | 701.50 | 2258/3205 | 65.0/71.0 | 10576/10591 | True | a4e9045a038b | 2026-09-17T15:41:30.323881+09:00 |
| CORE_153555 | S5_RESOURCE | RESOURCE | vllm | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 21.0 | 65886 | 16384 | 781.56 | 2174/3144 | 57.2/68.6 | 9450/10149 | True | a4e9045a038b | 2026-09-17T15:43:38.926657+09:00 |
| FOCUS2_183951 | S28_FOCUS2_vllmw_syswide | FOCUS | vllmw | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.3 | 65886 | 16384 | 703.47 | 2215/3189 | 64.1/72.7 | 10699/10718 | True | 12926df2c30b | 2026-09-17T18:42:48.951658+09:00 |
| FOCUS2_183951 | S29_CORR_vllmw_fp8fixso | CORR | vllmw | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.8 | 65886 | 16384 | 689.69 | 2313/3274 | 64.9/72.8 | 10843/10863 | True | 12926df2c30b | 2026-09-17T18:44:31.456662+09:00 |
| OFF_A_152830 | S1_OFF_A | OFF | vllm | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.4 | 65886 | 16384 | 700.86 | 2339/3213 | 63.0/73.9 | 10788/10802 | True | a4e9045a038b | 2026-09-17T15:31:11.799733+09:00 |
| OFF_B_153208 | S6_OFF_B | OFF | vllm | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.2 | 65886 | 16384 | 706.89 | 2271/3176 | 64.8/73.8 | 10468/10524 | True | a4e9045a038b | 2026-09-17T15:34:48.726118+09:00 |
| OFF_CLOSE2_165803 | S15_OFF_CLOSE2 | OFF | vllm | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 22.9 | 65886 | 16384 | 716.43 | 2363/3212 | 61.4/72.0 | 10524/10539 | True | d659ca0b274c | 2026-09-17T17:00:43.642826+09:00 |
| OFF_CLOSE2_165803 | S15b_OFF_CLOSE2_probe | OFF | probe | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 32.1 | null | 16384 | 509.90 | 2579/3123 | 97.6/107.6 | 15141/15226 | True | d659ca0b274c | 2026-09-17T17:01:27.832560+09:00 |
| OFF_OPEN2_161158 | S8_OFF_OPEN2 | OFF | vllm | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.1 | 65886 | 16384 | 707.96 | 2257/3327 | 63.4/71.8 | 10663/10685 | True | d659ca0b274c | 2026-09-17T16:14:46.899883+09:00 |
| OFF_OPEN2_161158 | S8b_OFF_OPEN2_probe | OFF | probe | M073_QWEN_PROBE128 | 64 | 128 | null/null | null | null | null | null | null/null | null/null | null/null | False | d659ca0b274c | 2026-09-17T16:20:30.185996+09:00 |
| OFF_X_171100 | S16_OFF_X_vllm | OFF | vllm | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.6 | 65886 | 16384 | 693.45 | 2458/3502 | 65.6/72.1 | 10909/10929 | True | d659ca0b274c | 2026-09-17T17:13:45.267288+09:00 |
| OFF_X_171100 | S16b_OFF_X_probe_pinned | OFF | probe | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 39.9 | null | 16384 | 410.62 | 2610/3084 | 115.0/160.0 | 21500/21580 | True | d659ca0b274c | 2026-09-17T17:14:33.033041+09:00 |
| OFF_X_171100 | S16c_OFF_X_vllm2 | OFF | vllm | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 20.8 | 65886 | 16384 | 788.58 | 2126/3105 | 56.4/68.8 | 9368/10138 | True | d659ca0b274c | 2026-09-17T17:15:20.595974+09:00 |
| OFF_Y_174421 | S21_OFF_Y_vllm | OFF | vllm | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.2 | 65886 | 16384 | 705.15 | 2283/3191 | 63.0/73.5 | 10740/10759 | True | d659ca0b274c | 2026-09-17T17:47:05.116950+09:00 |
| OFF_Y_174421 | S21b_OFF_Y_vllmw | OFF | vllmw | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.3 | 65886 | 16384 | 703.11 | 2193/3084 | 63.9/74.9 | 10791/10809 | True | d659ca0b274c | 2026-09-17T17:47:59.649714+09:00 |
| OFF_Y_174421 | S21c_OFF_Y_vllm2 | OFF | vllm | M073_QWEN_PROBE128 | 64 | 128 | 128/0 | 23.2 | 65886 | 16384 | 705.75 | 2220/3054 | 65.3/71.6 | 10519/10525 | True | d659ca0b274c | 2026-09-17T17:48:29.860348+09:00 |

역사적 기준(IDE_074, 재측정 없이 계승; 새 분모로 쓰지 않음): MAIN_SHORT OFF 743.02/747.53/768.93, PROBE128 OFF 720.80/780.72/732.61, P1 696.39/705.62/690.76, P2 706.46/720.67/705.25 tok/s (v1 바이너리).

## 3. 계측 간섭 (profiles/observer_overhead_v2.csv; OFF 평균 및 OFF 원값 각각 대비; 기준: |처리량|≤3 %, p95 ≤5 %)

| session | mode | ref | throughput_ratio | duration_ratio | e2el_p50 | ttft_p95 | tpot_p95 | status |
|---|---|---|---|---|---|---|---|---|
| S10_CORR | CORR | OFF_mean(probe) | 1.101 | 0.898 | 0.831 | 1.021 | 0.807 | DIAGNOSTIC_ONLY |
| S10_CORR | CORR | S15b_OFF_CLOSE2_probe | 0.994 | 1.006 | 1.005 | 1.015 | 1.004 | DESCRIPTIVE_LOW_DISTORTION |
| S10_CORR | CORR | S16b_OFF_X_probe_pinned | 1.234 | 0.810 | 0.708 | 1.028 | 0.675 | DIAGNOSTIC_ONLY |
| S11_RESOURCE | RESOURCE | OFF_mean(probe) | 1.099 | 0.899 | 0.835 | 1.006 | 0.815 | DIAGNOSTIC_ONLY |
| S11_RESOURCE | RESOURCE | S15b_OFF_CLOSE2_probe | 0.992 | 1.008 | 1.010 | 1.000 | 1.013 | DESCRIPTIVE_LOW_DISTORTION |
| S11_RESOURCE | RESOURCE | S16b_OFF_X_probe_pinned | 1.232 | 0.812 | 0.711 | 1.013 | 0.681 | DIAGNOSTIC_ONLY |
| S14_FOCUS_sched | FOCUS | OFF_mean(probe) | 1.079 | 0.916 | 0.843 | 1.031 | 0.824 | DIAGNOSTIC_ONLY |
| S14_FOCUS_sched | FOCUS | S15b_OFF_CLOSE2_probe | 0.974 | 1.027 | 1.020 | 1.024 | 1.025 | DESCRIPTIVE_LOW_DISTORTION |
| S14_FOCUS_sched | FOCUS | S16b_OFF_X_probe_pinned | 1.209 | 0.827 | 0.718 | 1.037 | 0.689 | DIAGNOSTIC_ONLY |
| S9_CORR | CORR | OFF_mean(probe) | 1.026 | 0.963 | 0.952 | 1.224 | 0.934 | DIAGNOSTIC_ONLY |
| S9_CORR | CORR | S15b_OFF_CLOSE2_probe | 0.926 | 1.079 | 1.152 | 1.216 | 1.162 | DIAGNOSTIC_ONLY |
| S9_CORR | CORR | S16b_OFF_X_probe_pinned | 1.150 | 0.869 | 0.812 | 1.232 | 0.781 | DIAGNOSTIC_ONLY |
| S17_CORR_pinned | CORR | OFF_mean(probe) | 1.088 | 0.909 | 0.838 | 1.037 | 0.815 | DIAGNOSTIC_ONLY |
| S17_CORR_pinned | CORR | S15b_OFF_CLOSE2_probe | 0.982 | 1.019 | 1.014 | 1.031 | 1.014 | DESCRIPTIVE_LOW_DISTORTION |
| S17_CORR_pinned | CORR | S16b_OFF_X_probe_pinned | 1.219 | 0.820 | 0.714 | 1.043 | 0.682 | DIAGNOSTIC_ONLY |
| S18_CORR_pinned | CORR | OFF_mean(probe) | 1.083 | 0.912 | 0.841 | 1.019 | 0.819 | DIAGNOSTIC_ONLY |
| S18_CORR_pinned | CORR | S15b_OFF_CLOSE2_probe | 0.978 | 1.023 | 1.018 | 1.012 | 1.018 | DESCRIPTIVE_LOW_DISTORTION |
| S18_CORR_pinned | CORR | S16b_OFF_X_probe_pinned | 1.214 | 0.824 | 0.717 | 1.025 | 0.685 | DIAGNOSTIC_ONLY |
| S19_FOCUS_pinned | FOCUS | OFF_mean(probe) | 1.060 | 0.932 | 0.847 | 1.018 | 0.839 | DIAGNOSTIC_ONLY |
| S19_FOCUS_pinned | FOCUS | S15b_OFF_CLOSE2_probe | 0.957 | 1.045 | 1.024 | 1.012 | 1.043 | DIAGNOSTIC_ONLY |
| S19_FOCUS_pinned | FOCUS | S16b_OFF_X_probe_pinned | 1.188 | 0.842 | 0.721 | 1.025 | 0.702 | DIAGNOSTIC_ONLY |
| S20_RESOURCE_pinned | RESOURCE | OFF_mean(probe) | 1.088 | 0.908 | 0.842 | 1.012 | 0.822 | DIAGNOSTIC_ONLY |
| S20_RESOURCE_pinned | RESOURCE | S15b_OFF_CLOSE2_probe | 0.982 | 1.018 | 1.019 | 1.006 | 1.022 | DESCRIPTIVE_LOW_DISTORTION |
| S20_RESOURCE_pinned | RESOURCE | S16b_OFF_X_probe_pinned | 1.220 | 0.820 | 0.718 | 1.019 | 0.687 | DIAGNOSTIC_ONLY |
| S22_CORR_vllmw | CORR | OFF_mean(vllm) | 0.963 | 1.037 | 1.032 | 1.030 | 1.014 | DIAGNOSTIC_ONLY |
| S22_CORR_vllmw | CORR | S1_OFF_A | 0.984 | 1.017 | 1.005 | 1.033 | 0.990 | DESCRIPTIVE_LOW_DISTORTION |
| S22_CORR_vllmw | CORR | S6_OFF_B | 0.975 | 1.025 | 1.035 | 1.045 | 0.991 | DESCRIPTIVE_LOW_DISTORTION |
| S22_CORR_vllmw | CORR | S15_OFF_CLOSE2 | 0.962 | 1.039 | 1.030 | 1.033 | 1.016 | DIAGNOSTIC_ONLY |
| S22_CORR_vllmw | CORR | S8_OFF_OPEN2 | 0.974 | 1.027 | 1.016 | 0.997 | 1.019 | DESCRIPTIVE_LOW_DISTORTION |
| S22_CORR_vllmw | CORR | S16_OFF_X_vllm | 0.994 | 1.006 | 0.993 | 0.948 | 1.016 | DIAGNOSTIC_ONLY |
| S22_CORR_vllmw | CORR | S16c_OFF_X_vllm2 | 0.874 | 1.144 | 1.157 | 1.069 | 1.064 | DIAGNOSTIC_ONLY |
| S22_CORR_vllmw | CORR | S21_OFF_Y_vllm | 0.978 | 1.023 | 1.009 | 1.040 | 0.995 | DESCRIPTIVE_LOW_DISTORTION |
| S22_CORR_vllmw | CORR | S21c_OFF_Y_vllm2 | 0.977 | 1.024 | 1.030 | 1.087 | 1.022 | DIAGNOSTIC_ONLY |
| S23_CORR_vllmw | CORR | OFF_mean(vllm) | 0.973 | 1.026 | 1.032 | 1.001 | 1.011 | DESCRIPTIVE_LOW_DISTORTION |
| S23_CORR_vllmw | CORR | S1_OFF_A | 0.993 | 1.007 | 1.004 | 1.004 | 0.987 | DESCRIPTIVE_LOW_DISTORTION |
| S23_CORR_vllmw | CORR | S6_OFF_B | 0.985 | 1.015 | 1.035 | 1.016 | 0.989 | DESCRIPTIVE_LOW_DISTORTION |
| S23_CORR_vllmw | CORR | S15_OFF_CLOSE2 | 0.972 | 1.029 | 1.029 | 1.004 | 1.013 | DESCRIPTIVE_LOW_DISTORTION |
| S23_CORR_vllmw | CORR | S8_OFF_OPEN2 | 0.983 | 1.017 | 1.016 | 0.970 | 1.017 | DESCRIPTIVE_LOW_DISTORTION |
| S23_CORR_vllmw | CORR | S16_OFF_X_vllm | 1.004 | 0.996 | 0.993 | 0.921 | 1.013 | DIAGNOSTIC_ONLY |
| S23_CORR_vllmw | CORR | S16c_OFF_X_vllm2 | 0.883 | 1.133 | 1.156 | 1.039 | 1.062 | DIAGNOSTIC_ONLY |
| S23_CORR_vllmw | CORR | S21_OFF_Y_vllm | 0.987 | 1.013 | 1.009 | 1.011 | 0.993 | DESCRIPTIVE_LOW_DISTORTION |
| S23_CORR_vllmw | CORR | S21c_OFF_Y_vllm2 | 0.987 | 1.014 | 1.030 | 1.057 | 1.020 | DIAGNOSTIC_ONLY |
| S24_RESOURCE_vllmw | RESOURCE | OFF_mean(vllm) | 0.978 | 1.021 | 1.013 | 0.963 | 1.003 | DESCRIPTIVE_LOW_DISTORTION |
| S24_RESOURCE_vllmw | RESOURCE | S1_OFF_A | 0.998 | 1.002 | 0.986 | 0.966 | 0.980 | DESCRIPTIVE_LOW_DISTORTION |
| S24_RESOURCE_vllmw | RESOURCE | S6_OFF_B | 0.990 | 1.010 | 1.016 | 0.977 | 0.981 | DESCRIPTIVE_LOW_DISTORTION |
| S24_RESOURCE_vllmw | RESOURCE | S15_OFF_CLOSE2 | 0.977 | 1.024 | 1.011 | 0.966 | 1.006 | DESCRIPTIVE_LOW_DISTORTION |
| S24_RESOURCE_vllmw | RESOURCE | S8_OFF_OPEN2 | 0.988 | 1.012 | 0.998 | 0.933 | 1.009 | DIAGNOSTIC_ONLY |
| S24_RESOURCE_vllmw | RESOURCE | S16_OFF_X_vllm | 1.009 | 0.991 | 0.975 | 0.886 | 1.006 | DIAGNOSTIC_ONLY |
| S24_RESOURCE_vllmw | RESOURCE | S16c_OFF_X_vllm2 | 0.887 | 1.127 | 1.136 | 1.000 | 1.054 | DIAGNOSTIC_ONLY |
| S24_RESOURCE_vllmw | RESOURCE | S21_OFF_Y_vllm | 0.992 | 1.008 | 0.991 | 0.972 | 0.985 | DESCRIPTIVE_LOW_DISTORTION |
| S24_RESOURCE_vllmw | RESOURCE | S21c_OFF_Y_vllm2 | 0.992 | 1.009 | 1.011 | 1.016 | 1.012 | DESCRIPTIVE_LOW_DISTORTION |
| S25_FOCUS_vllmw | FOCUS | OFF_mean(vllm) | 0.946 | 1.055 | 1.072 | 1.031 | 1.070 | DIAGNOSTIC_ONLY |
| S25_FOCUS_vllmw | FOCUS | S1_OFF_A | 0.966 | 1.035 | 1.043 | 1.034 | 1.045 | DIAGNOSTIC_ONLY |
| S25_FOCUS_vllmw | FOCUS | S6_OFF_B | 0.958 | 1.044 | 1.075 | 1.046 | 1.046 | DIAGNOSTIC_ONLY |
| S25_FOCUS_vllmw | FOCUS | S15_OFF_CLOSE2 | 0.945 | 1.058 | 1.069 | 1.034 | 1.073 | DIAGNOSTIC_ONLY |
| S25_FOCUS_vllmw | FOCUS | S8_OFF_OPEN2 | 0.957 | 1.045 | 1.055 | 0.998 | 1.076 | DIAGNOSTIC_ONLY |
| S25_FOCUS_vllmw | FOCUS | S16_OFF_X_vllm | 0.977 | 1.024 | 1.031 | 0.948 | 1.072 | DIAGNOSTIC_ONLY |
| S25_FOCUS_vllmw | FOCUS | S16c_OFF_X_vllm2 | 0.859 | 1.164 | 1.201 | 1.070 | 1.124 | DIAGNOSTIC_ONLY |
| S25_FOCUS_vllmw | FOCUS | S21_OFF_Y_vllm | 0.961 | 1.041 | 1.048 | 1.041 | 1.050 | DIAGNOSTIC_ONLY |
| S25_FOCUS_vllmw | FOCUS | S21c_OFF_Y_vllm2 | 0.960 | 1.042 | 1.070 | 1.087 | 1.079 | DIAGNOSTIC_ONLY |
| S2_CORE | CORE | OFF_mean(vllm) | 0.990 | 1.009 | 0.999 | 1.012 | 0.977 | DESCRIPTIVE_LOW_DISTORTION |
| S2_CORE | CORE | S1_OFF_A | 1.011 | 0.990 | 0.973 | 1.015 | 0.954 | DESCRIPTIVE_LOW_DISTORTION |
| S2_CORE | CORE | S6_OFF_B | 1.002 | 0.998 | 1.002 | 1.027 | 0.955 | DESCRIPTIVE_LOW_DISTORTION |
| S2_CORE | CORE | S15_OFF_CLOSE2 | 0.989 | 1.012 | 0.997 | 1.015 | 0.979 | DESCRIPTIVE_LOW_DISTORTION |
| S2_CORE | CORE | S8_OFF_OPEN2 | 1.000 | 1.000 | 0.984 | 0.980 | 0.982 | DESCRIPTIVE_LOW_DISTORTION |
| S2_CORE | CORE | S16_OFF_X_vllm | 1.021 | 0.979 | 0.962 | 0.931 | 0.979 | DIAGNOSTIC_ONLY |
| S2_CORE | CORE | S16c_OFF_X_vllm2 | 0.898 | 1.113 | 1.120 | 1.050 | 1.026 | DIAGNOSTIC_ONLY |
| S2_CORE | CORE | S21_OFF_Y_vllm | 1.004 | 0.996 | 0.977 | 1.022 | 0.959 | DESCRIPTIVE_LOW_DISTORTION |
| S2_CORE | CORE | S21c_OFF_Y_vllm2 | 1.004 | 0.997 | 0.997 | 1.068 | 0.985 | DIAGNOSTIC_ONLY |
| S3_CORR | CORR | OFF_mean(vllm) | 0.977 | 1.022 | 1.005 | 1.000 | 0.999 | DESCRIPTIVE_LOW_DISTORTION |
| S3_CORR | CORR | S1_OFF_A | 0.997 | 1.003 | 0.978 | 1.003 | 0.975 | DESCRIPTIVE_LOW_DISTORTION |
| S3_CORR | CORR | S6_OFF_B | 0.989 | 1.011 | 1.008 | 1.015 | 0.977 | DESCRIPTIVE_LOW_DISTORTION |
| S3_CORR | CORR | S15_OFF_CLOSE2 | 0.976 | 1.025 | 1.002 | 1.003 | 1.001 | DESCRIPTIVE_LOW_DISTORTION |
| S3_CORR | CORR | S8_OFF_OPEN2 | 0.987 | 1.013 | 0.989 | 0.969 | 1.004 | DESCRIPTIVE_LOW_DISTORTION |
| S3_CORR | CORR | S16_OFF_X_vllm | 1.008 | 0.992 | 0.967 | 0.920 | 1.001 | DIAGNOSTIC_ONLY |
| S3_CORR | CORR | S16c_OFF_X_vllm2 | 0.887 | 1.128 | 1.126 | 1.038 | 1.049 | DIAGNOSTIC_ONLY |
| S3_CORR | CORR | S21_OFF_Y_vllm | 0.991 | 1.009 | 0.982 | 1.010 | 0.980 | DESCRIPTIVE_LOW_DISTORTION |
| S3_CORR | CORR | S21c_OFF_Y_vllm2 | 0.991 | 1.010 | 1.003 | 1.055 | 1.007 | DIAGNOSTIC_ONLY |
| S4_CORR | CORR | OFF_mean(vllm) | 0.980 | 1.019 | 1.007 | 0.994 | 0.983 | DESCRIPTIVE_LOW_DISTORTION |
| S4_CORR | CORR | S1_OFF_A | 1.001 | 0.999 | 0.980 | 0.997 | 0.960 | DESCRIPTIVE_LOW_DISTORTION |
| S4_CORR | CORR | S6_OFF_B | 0.992 | 1.008 | 1.010 | 1.009 | 0.961 | DESCRIPTIVE_LOW_DISTORTION |
| S4_CORR | CORR | S15_OFF_CLOSE2 | 0.979 | 1.021 | 1.005 | 0.998 | 0.985 | DESCRIPTIVE_LOW_DISTORTION |
| S4_CORR | CORR | S8_OFF_OPEN2 | 0.991 | 1.009 | 0.992 | 0.963 | 0.989 | DESCRIPTIVE_LOW_DISTORTION |
| S4_CORR | CORR | S16_OFF_X_vllm | 1.012 | 0.989 | 0.969 | 0.915 | 0.985 | DIAGNOSTIC_ONLY |
| S4_CORR | CORR | S16c_OFF_X_vllm2 | 0.890 | 1.124 | 1.129 | 1.032 | 1.032 | DIAGNOSTIC_ONLY |
| S4_CORR | CORR | S21_OFF_Y_vllm | 0.995 | 1.005 | 0.985 | 1.004 | 0.965 | DESCRIPTIVE_LOW_DISTORTION |
| S4_CORR | CORR | S21c_OFF_Y_vllm2 | 0.994 | 1.006 | 1.005 | 1.049 | 0.991 | DESCRIPTIVE_LOW_DISTORTION |
| S5_RESOURCE | RESOURCE | OFF_mean(vllm) | 1.092 | 0.914 | 0.900 | 0.976 | 0.950 | DIAGNOSTIC_ONLY |
| S5_RESOURCE | RESOURCE | S1_OFF_A | 1.115 | 0.897 | 0.876 | 0.979 | 0.928 | DIAGNOSTIC_ONLY |
| S5_RESOURCE | RESOURCE | S6_OFF_B | 1.106 | 0.904 | 0.903 | 0.990 | 0.929 | DIAGNOSTIC_ONLY |
| S5_RESOURCE | RESOURCE | S15_OFF_CLOSE2 | 1.091 | 0.917 | 0.898 | 0.979 | 0.953 | DIAGNOSTIC_ONLY |
| S5_RESOURCE | RESOURCE | S8_OFF_OPEN2 | 1.104 | 0.906 | 0.886 | 0.945 | 0.956 | DIAGNOSTIC_ONLY |
| S5_RESOURCE | RESOURCE | S16_OFF_X_vllm | 1.127 | 0.887 | 0.866 | 0.898 | 0.952 | DIAGNOSTIC_ONLY |
| S5_RESOURCE | RESOURCE | S16c_OFF_X_vllm2 | 0.991 | 1.009 | 1.009 | 1.013 | 0.998 | DESCRIPTIVE_LOW_DISTORTION |
| S5_RESOURCE | RESOURCE | S21_OFF_Y_vllm | 1.108 | 0.902 | 0.880 | 0.985 | 0.933 | DIAGNOSTIC_ONLY |
| S5_RESOURCE | RESOURCE | S21c_OFF_Y_vllm2 | 1.107 | 0.903 | 0.898 | 1.029 | 0.958 | DIAGNOSTIC_ONLY |
| S28_FOCUS2_vllmw_syswide | FOCUS | OFF_mean(vllm) | 0.983 | 1.016 | 1.019 | 0.990 | 1.007 | DESCRIPTIVE_LOW_DISTORTION |
| S28_FOCUS2_vllmw_syswide | FOCUS | S1_OFF_A | 1.004 | 0.996 | 0.992 | 0.993 | 0.983 | DESCRIPTIVE_LOW_DISTORTION |
| S28_FOCUS2_vllmw_syswide | FOCUS | S6_OFF_B | 0.995 | 1.005 | 1.022 | 1.004 | 0.984 | DESCRIPTIVE_LOW_DISTORTION |
| S28_FOCUS2_vllmw_syswide | FOCUS | S15_OFF_CLOSE2 | 0.982 | 1.018 | 1.017 | 0.993 | 1.009 | DESCRIPTIVE_LOW_DISTORTION |
| S28_FOCUS2_vllmw_syswide | FOCUS | S8_OFF_OPEN2 | 0.994 | 1.006 | 1.003 | 0.959 | 1.012 | DESCRIPTIVE_LOW_DISTORTION |
| S28_FOCUS2_vllmw_syswide | FOCUS | S16_OFF_X_vllm | 1.014 | 0.986 | 0.981 | 0.911 | 1.009 | DIAGNOSTIC_ONLY |
| S28_FOCUS2_vllmw_syswide | FOCUS | S16c_OFF_X_vllm2 | 0.892 | 1.121 | 1.142 | 1.027 | 1.057 | DIAGNOSTIC_ONLY |
| S28_FOCUS2_vllmw_syswide | FOCUS | S21_OFF_Y_vllm | 0.998 | 1.002 | 0.996 | 0.999 | 0.988 | DESCRIPTIVE_LOW_DISTORTION |
| S28_FOCUS2_vllmw_syswide | FOCUS | S21c_OFF_Y_vllm2 | 0.997 | 1.003 | 1.017 | 1.044 | 1.015 | DESCRIPTIVE_LOW_DISTORTION |
| S29_CORR_vllmw_fp8fixso | CORR | OFF_mean(vllm) | 0.964 | 1.036 | 1.033 | 1.016 | 1.008 | DIAGNOSTIC_ONLY |
| S29_CORR_vllmw_fp8fixso | CORR | S1_OFF_A | 0.984 | 1.016 | 1.005 | 1.019 | 0.984 | DESCRIPTIVE_LOW_DISTORTION |
| S29_CORR_vllmw_fp8fixso | CORR | S6_OFF_B | 0.976 | 1.025 | 1.036 | 1.031 | 0.985 | DESCRIPTIVE_LOW_DISTORTION |
| S29_CORR_vllmw_fp8fixso | CORR | S15_OFF_CLOSE2 | 0.963 | 1.039 | 1.030 | 1.019 | 1.010 | DIAGNOSTIC_ONLY |
| S29_CORR_vllmw_fp8fixso | CORR | S8_OFF_OPEN2 | 0.974 | 1.026 | 1.017 | 0.984 | 1.013 | DESCRIPTIVE_LOW_DISTORTION |
| S29_CORR_vllmw_fp8fixso | CORR | S16_OFF_X_vllm | 0.995 | 1.005 | 0.994 | 0.935 | 1.010 | DIAGNOSTIC_ONLY |
| S29_CORR_vllmw_fp8fixso | CORR | S16c_OFF_X_vllm2 | 0.875 | 1.143 | 1.157 | 1.055 | 1.058 | DIAGNOSTIC_ONLY |
| S29_CORR_vllmw_fp8fixso | CORR | S21_OFF_Y_vllm | 0.978 | 1.022 | 1.010 | 1.026 | 0.989 | DESCRIPTIVE_LOW_DISTORTION |
| S29_CORR_vllmw_fp8fixso | CORR | S21c_OFF_Y_vllm2 | 0.977 | 1.023 | 1.031 | 1.072 | 1.016 | DIAGNOSTIC_ONLY |

OFF 자체 변동 (max/min−1): 0.137 → INCONCLUSIVE_BOOT_VARIANCE. 단발 CORE/RESOURCE 에 SD/CI 없음. OFF 두 부팅은 시간순으로 CORE 부팅 앞에 위치 (OFF_OPEN/OFF_CLOSE 배치 미충족, WORK_LOG).

## 4. 유효성 (세션별 v2/validation_results_v2.json, 항목 분리)

| session | mode | execution_valid | workload_valid | config_valid | window_valid | lifecycle_valid | task_count_valid | sampling_valid | gpu_activity_valid | mapping_valid | clock_valid_by_pair | producer_consumer_valid | counter_running_valid | counter_scope_valid | artifact_valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S10_CORR | CORR | True | None | True | True | True | True | True | True | True | True | True | n/a | n/a | False |
| S11_RESOURCE | RESOURCE | True | None | True | True | True | True | True | n/a | n/a | n/a | n/a | True | True | False |
| S12_C1_CORR | CORR | True | None | True | True | True | True | True | True | True | True | True | n/a | n/a | False |
| S13_LONG_CORR | CORR | True | None | True | True | True | True | True | True | True | True | True | n/a | n/a | False |
| S14_FOCUS_sched | FOCUS | True | None | True | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | False |
| S9_CORR | CORR | True | None | True | True | True | True | True | True | False | True | True | n/a | n/a | False |
| S17_CORR_pinned | CORR | True | None | True | True | True | True | True | True | True | True | True | n/a | n/a | False |
| S18_CORR_pinned | CORR | True | None | True | True | True | True | True | True | True | True | True | n/a | n/a | False |
| S19_FOCUS_pinned | FOCUS | True | None | True | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | False |
| S20_RESOURCE_pinned | RESOURCE | True | None | True | True | True | True | True | n/a | n/a | n/a | n/a | True | True | False |
| S22_CORR_vllmw | CORR | True | True | True | True | True | True | True | True | True | False | True | n/a | n/a | True |
| S23_CORR_vllmw | CORR | True | True | True | True | True | True | True | True | True | True | True | n/a | n/a | True |
| S24_RESOURCE_vllmw | RESOURCE | True | True | True | True | True | True | True | n/a | n/a | n/a | n/a | True | True | True |
| S25_FOCUS_vllmw | FOCUS | True | True | True | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | True |
| S26_C1_CORR_vllmw | CORR | True | None | True | True | True | True | True | True | True | True | True | n/a | n/a | True |
| S27_LONG_CORR_vllmw | CORR | True | None | True | True | True | True | True | True | True | True | True | n/a | n/a | True |
| S2_CORE | CORE | True | True | True | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | True |
| S3_CORR | CORR | True | True | True | True | True | True | True | True | True | True | True | n/a | n/a | True |
| S4_CORR | CORR | True | True | True | True | True | True | True | True | True | True | True | n/a | n/a | True |
| S5_RESOURCE | RESOURCE | True | True | True | True | True | True | True | n/a | n/a | n/a | n/a | True | True | True |
| S28_FOCUS2_vllmw_syswide | FOCUS | True | True | True | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | True |
| S29_CORR_vllmw_fp8fixso | CORR | True | True | True | True | True | True | True | True | True | True | True | n/a | n/a | True |
| S1_OFF_A | OFF | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | True |
| S6_OFF_B | OFF | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | True |
| S15_OFF_CLOSE2 | OFF | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | True |
| S15b_OFF_CLOSE2_probe | OFF | True | None | True | True | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | False |
| S8_OFF_OPEN2 | OFF | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | True |
| S8b_OFF_OPEN2_probe | OFF | False | None | True | True | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | False |
| S16_OFF_X_vllm | OFF | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | True |
| S16b_OFF_X_probe_pinned | OFF | True | None | True | True | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | False |
| S16c_OFF_X_vllm2 | OFF | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | True |
| S21_OFF_Y_vllm | OFF | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | True |
| S21b_OFF_Y_vllmw | OFF | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | True |
| S21c_OFF_Y_vllm2 | OFF | True | True | True | True | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | True |

## 5. v2 의존 지표 (CORR 세션; cohort cold_present_nonempty; µs)

### S10_CORR — coverage {'gpu_layer_rows_total': 23870, 'method:count_match_zip': 23808, 'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} · clock {'OK': 23808}

| graph:cohort | n | 지표 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | cold_pub_delta_us | 848.2 | 2215.5 | 2827.5 | 6561.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | cold_pub_lateness_us | 848.2 | 2215.5 | 2827.5 | 6561.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | gpu_pre_h2d_gap_us | 861.5 | 2228.2 | 2845.2 | 6570.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | gpu_idle_inside_gap_us | 861.5 | 2228.2 | 2845.2 | 6570.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | h2d_duration_us | 30.2 | 32.8 | 34.5 | 44.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | gpu_post_h2d_gap_us | 7.2 | 9.5 | 9.8 | 11.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | combine_schedule_gap_us | 7.2 | 9.5 | 9.8 | 11.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | go_to_enqueue_us | 3.5 | 5.5 | 7.0 | 28.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | enqueue_to_start_us | 1178.8 | 2512.5 | 3105.8 | 6810.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | predecessor_exec_union_us | 1177.8 | 2511.8 | 3104.5 | 6809.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | predecessor_deferred_exec_us | 1177.5 | 2511.5 | 3104.5 | 6808.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | queue_gap_unattributed_us | 0.8 | 1.5 | 5.5 | 130.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | deferred_service_span_us | 1427.2 | 2752.0 | 3339.5 | 7048.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | task_exec_span_us | 1427.8 | 2752.2 | 3340.0 | 7048.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | numa0_span_us | 1400.5 | 2718.5 | 3306.0 | 6964.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | numa1_span_us | 1352.5 | 2645.5 | 3226.0 | 6620.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | numa_completion_skew_us | 40.8 | 105.5 | 225.2 | 4132.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | producer_end_to_pub_us | 0.5 | 1.0 | 40.0 | 936.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | pub_to_h2d_start_us | 13.2 | 21.8 | 367.5 | 455.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | overlap_budget_us | 1752.0 | 3061.0 | 3646.2 | 7287.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | n_cold_assignments | 40.0 | 128.0 | 165.0 | 240.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | go_minus_dtoh_end_us | 5.0 | 6.2 | 8.8 | 1790.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | cold_pub_delta_us | -402.5 | -315.2 | -230.8 | -210.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | gpu_pre_h2d_gap_us | 10.5 | 13.2 | 14.2 | 14.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | gpu_idle_inside_gap_us | 10.5 | 13.2 | 14.2 | 14.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | h2d_duration_us | 20.0 | 21.8 | 22.5 | 23.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | gpu_post_h2d_gap_us | 5.5 | 8.8 | 9.8 | 9.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | combine_schedule_gap_us | 5.5 | 8.8 | 9.8 | 9.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | go_to_enqueue_us | 4.0 | 7.2 | 8.8 | 8.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | enqueue_to_start_us | 521.0 | 838.5 | 973.8 | 1011.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | predecessor_exec_union_us | 519.8 | 837.5 | 972.2 | 1010.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | predecessor_deferred_exec_us | 448.0 | 767.8 | 896.8 | 943.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | queue_gap_unattributed_us | 1.2 | 2.0 | 6.0 | 6.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | deferred_service_span_us | 74.0 | 83.5 | 93.5 | 95.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | task_exec_span_us | 74.2 | 83.5 | 93.8 | 95.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | numa0_span_us | 45.8 | 52.8 | 57.5 | 58.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | numa1_span_us | 45.2 | 53.0 | 57.0 | 58.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | numa_completion_skew_us | 2.2 | 7.8 | 10.0 | 12.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | producer_end_to_pub_us | 175.0 | 445.5 | 512.5 | 549.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | pub_to_h2d_start_us | 412.8 | 430.8 | 445.5 | 452.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | overlap_budget_us | 1166.0 | 1500.8 | 1629.0 | 1686.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | go_minus_dtoh_end_us | 5.0 | 5.8 | 6.0 | 10.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_delta_us | 42.0 | 367.5 | 500.0 | 1456.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_lateness_us | 42.0 | 367.5 | 500.0 | 1456.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_pre_h2d_gap_us | 52.5 | 379.2 | 515.5 | 1468.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_idle_inside_gap_us | 52.5 | 379.2 | 515.5 | 1453.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | h2d_duration_us | 31.5 | 35.5 | 36.8 | 38.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_post_h2d_gap_us | 7.5 | 9.5 | 9.8 | 10.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | combine_schedule_gap_us | 7.5 | 9.5 | 9.8 | 10.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | pub_to_h2d_start_us | 16.2 | 286.5 | 337.0 | 404.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | go_minus_dtoh_end_us | 5.2 | 6.5 | 8.2 | 866.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | cold_pub_delta_us | -85.0 | 59.0 | 189.2 | 1191.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | cold_pub_lateness_us | 0.0 | 59.0 | 189.2 | 1191.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | gpu_pre_h2d_gap_us | 11.2 | 70.5 | 206.8 | 1202.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | gpu_idle_inside_gap_us | 11.2 | 70.5 | 206.8 | 1202.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | h2d_duration_us | 4.0 | 5.2 | 5.8 | 7.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | gpu_post_h2d_gap_us | 5.5 | 9.5 | 10.0 | 11.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | combine_schedule_gap_us | 5.5 | 9.5 | 10.0 | 11.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | go_to_enqueue_us | 2.5 | 3.2 | 4.2 | 20.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | enqueue_to_start_us | 4.5 | 134.8 | 269.2 | 1284.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | predecessor_exec_union_us | 0.2 | 133.0 | 268.2 | 1269.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | predecessor_deferred_exec_us | 0.0 | 133.0 | 268.2 | 1269.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | queue_gap_unattributed_us | 3.8 | 6.2 | 10.8 | 255.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | deferred_service_span_us | 148.5 | 334.2 | 441.5 | 1434.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | task_exec_span_us | 148.8 | 334.5 | 441.5 | 1435.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | numa0_span_us | 125.5 | 310.8 | 416.2 | 1310.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | numa1_span_us | 127.5 | 310.8 | 411.2 | 931.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | numa_completion_skew_us | 5.0 | 16.2 | 41.8 | 550.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | producer_end_to_pub_us | 93.2 | 120.2 | 127.2 | 139.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | pub_to_h2d_start_us | 95.5 | 106.0 | 108.8 | 119.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | overlap_budget_us | 341.5 | 386.0 | 517.0 | 1544.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | n_cold_assignments | 1.0 | 4.0 | 5.0 | 7.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | go_minus_dtoh_end_us | 4.5 | 5.5 | 9.0 | 18.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | cold_pub_delta_us | -86.8 | -75.5 | -68.5 | 263.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 263.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | gpu_pre_h2d_gap_us | 10.8 | 14.0 | 14.5 | 270.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | gpu_idle_inside_gap_us | 10.8 | 14.0 | 14.5 | 270.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | h2d_duration_us | 4.0 | 4.8 | 5.2 | 6.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | gpu_post_h2d_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | combine_schedule_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | go_to_enqueue_us | 2.5 | 3.2 | 4.8 | 291.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | enqueue_to_start_us | 4.0 | 14.5 | 56.8 | 449.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | predecessor_exec_union_us | 0.2 | 0.5 | 51.5 | 448.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | predecessor_deferred_exec_us | 0.0 | 0.0 | 50.5 | 448.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | queue_gap_unattributed_us | 3.8 | 6.5 | 11.2 | 358.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | deferred_service_span_us | 30.5 | 37.8 | 45.8 | 73.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | task_exec_span_us | 30.8 | 38.0 | 45.8 | 74.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | numa0_span_us | 9.5 | 11.8 | 15.2 | 32.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | numa1_span_us | 13.2 | 17.0 | 22.0 | 33.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | numa_completion_skew_us | 5.0 | 8.8 | 11.0 | 51.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | producer_end_to_pub_us | 218.2 | 233.0 | 238.5 | 566.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | pub_to_h2d_start_us | 97.2 | 106.8 | 109.2 | 112.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | overlap_budget_us | 344.0 | 360.2 | 366.5 | 724.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | go_minus_dtoh_end_us | 4.5 | 5.2 | 7.2 | 35.5 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_delta_us | -60.2 | -44.5 | -33.8 | -33.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_pre_h2d_gap_us | 11.0 | 13.5 | 14.0 | 15.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_idle_inside_gap_us | 11.0 | 13.5 | 14.0 | 15.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | h2d_duration_us | 3.8 | 4.8 | 5.0 | 5.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_post_h2d_gap_us | 5.8 | 9.8 | 10.0 | 10.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | combine_schedule_gap_us | 5.8 | 9.8 | 10.0 | 10.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | pub_to_h2d_start_us | 71.0 | 79.2 | 81.2 | 81.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | go_minus_dtoh_end_us | 4.5 | 5.8 | 11.0 | 13.8 |

민감도 (late share, cold_present_nonempty): {"2:cold_present_nonempty": {"late_share_thr_0us": 0.949, "late_share_thr_1us": 0.949, "late_share_thr_2us": 0.949, "late_share_thr_5us": 0.948, "clock_indeterminate_share_5us": 0.002, "n": 15493, "cold_gpu_ready_late_share": 1.0}, "32:cold_present_nonempty": {"late_share_thr_0us": 0.098, "late_share_thr_1us": 0.096, "late_share_thr_2us": 0.095, "late_share_thr_5us": 0.092, "clock_indeterminate_share_5us": 0.009, "n": 3061, "cold_gpu_ready_late_share": 1.0}}

### S12_C1_CORR — coverage {'gpu_layer_rows_total': 127038, 'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 126976, 'method:ANCHOR_RESYNC(go>=dtoh_end<next)': 126976} · clock {'OK': 126976}

| graph:cohort | n | 지표 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | cold_pub_delta_us | -59.2 | 116.2 | 226.2 | 5033.8 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | cold_pub_lateness_us | 0.0 | 116.2 | 226.2 | 5033.8 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | gpu_pre_h2d_gap_us | 11.5 | 130.8 | 240.5 | 5042.5 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | gpu_idle_inside_gap_us | 11.5 | 130.8 | 240.5 | 5042.5 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | h2d_duration_us | 3.5 | 4.8 | 5.5 | 7.2 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | gpu_post_h2d_gap_us | 5.8 | 9.5 | 10.0 | 14.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | combine_schedule_gap_us | 5.8 | 9.5 | 10.0 | 14.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | go_to_enqueue_us | 2.5 | 3.5 | 5.0 | 375.2 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | enqueue_to_start_us | 4.0 | 166.0 | 280.8 | 1893.8 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | predecessor_exec_union_us | 0.2 | 164.5 | 279.2 | 1893.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | predecessor_deferred_exec_us | 0.0 | 164.2 | 278.8 | 1892.8 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | queue_gap_unattributed_us | 3.2 | 5.8 | 9.8 | 884.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | deferred_service_span_us | 148.8 | 348.8 | 453.8 | 2056.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | task_exec_span_us | 149.2 | 349.0 | 454.2 | 2056.2 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | numa0_span_us | 134.0 | 326.8 | 437.2 | 1731.5 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | numa1_span_us | 133.0 | 318.8 | 432.8 | 1987.5 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | numa_completion_skew_us | 3.5 | 17.2 | 69.5 | 1846.2 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | producer_end_to_pub_us | 56.0 | 108.8 | 115.2 | 3862.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | pub_to_h2d_start_us | 69.0 | 78.5 | 80.5 | 99.2 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | overlap_budget_us | 296.8 | 396.0 | 507.2 | 2138.8 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | n_cold_assignments | 1.0 | 4.0 | 6.0 | 8.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 46635 | go_minus_dtoh_end_us | 4.5 | 5.5 | 6.8 | 1333.0 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | cold_pub_delta_us | -62.8 | -55.8 | -48.8 | 1265.2 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 1265.2 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | gpu_pre_h2d_gap_us | 10.5 | 14.0 | 14.8 | 1275.2 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | gpu_idle_inside_gap_us | 10.5 | 14.0 | 14.8 | 1275.2 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | h2d_duration_us | 3.0 | 4.0 | 4.8 | 7.2 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | gpu_post_h2d_gap_us | 6.0 | 9.8 | 10.0 | 12.5 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | combine_schedule_gap_us | 6.0 | 9.8 | 10.0 | 12.5 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | go_to_enqueue_us | 2.5 | 3.5 | 5.0 | 387.8 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | enqueue_to_start_us | 3.8 | 13.8 | 57.2 | 3836.2 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | predecessor_exec_union_us | 0.0 | 0.5 | 54.8 | 1217.0 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | predecessor_deferred_exec_us | 0.0 | 0.0 | 54.2 | 1217.0 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | queue_gap_unattributed_us | 3.8 | 6.2 | 11.2 | 3836.0 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | deferred_service_span_us | 20.5 | 24.0 | 34.0 | 1261.0 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | task_exec_span_us | 20.8 | 24.2 | 34.5 | 1261.5 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | numa0_span_us | 5.8 | 8.0 | 11.8 | 634.5 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | numa1_span_us | 7.0 | 9.0 | 13.0 | 316.5 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | numa_completion_skew_us | 2.5 | 5.2 | 8.5 | 1244.2 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | producer_end_to_pub_us | 210.5 | 222.0 | 226.8 | 1543.8 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | pub_to_h2d_start_us | 73.2 | 79.8 | 81.5 | 100.2 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | overlap_budget_us | 301.2 | 312.5 | 317.2 | 4074.8 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 78293 | go_minus_dtoh_end_us | 4.2 | 5.2 | 6.5 | 1320.2 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | cold_pub_delta_us | -43.0 | -35.5 | -21.0 | 844.8 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 844.8 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | gpu_pre_h2d_gap_us | 11.2 | 14.2 | 15.0 | 861.0 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | gpu_idle_inside_gap_us | 11.2 | 14.2 | 15.0 | 861.0 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | h2d_duration_us | 3.0 | 3.8 | 4.2 | 4.5 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | gpu_post_h2d_gap_us | 6.2 | 9.8 | 9.8 | 10.2 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | combine_schedule_gap_us | 6.2 | 9.8 | 9.8 | 10.2 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | pub_to_h2d_start_us | 53.8 | 61.0 | 62.8 | 64.8 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | go_minus_dtoh_end_us | 4.2 | 5.2 | 6.0 | 22.2 |

민감도 (late share, cold_present_nonempty): {"35:cold_present_nonempty": {"late_share_thr_0us": 0.186, "late_share_thr_1us": 0.184, "late_share_thr_2us": 0.182, "late_share_thr_5us": 0.178, "clock_indeterminate_share_5us": 0.018, "n": 46635, "cold_gpu_ready_late_share": 1.0}}

### S13_LONG_CORR — coverage {'gpu_layer_rows_total': 31806, 'method:count_match_zip': 31744, 'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} · clock {'OK': 31744}

| graph:cohort | n | 지표 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | cold_pub_delta_us | 23.8 | 708.0 | 1033.0 | 4291.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | cold_pub_lateness_us | 23.8 | 708.0 | 1033.0 | 4291.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | gpu_pre_h2d_gap_us | 37.0 | 721.0 | 1045.0 | 4308.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | gpu_idle_inside_gap_us | 37.0 | 721.0 | 1045.0 | 4308.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | h2d_duration_us | 9.0 | 10.2 | 15.0 | 27.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | gpu_post_h2d_gap_us | 6.0 | 9.2 | 9.8 | 12.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | combine_schedule_gap_us | 6.0 | 9.2 | 9.8 | 12.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | go_to_enqueue_us | 2.2 | 4.5 | 6.0 | 440.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | enqueue_to_start_us | 196.2 | 856.2 | 1157.5 | 4471.8 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | predecessor_exec_union_us | 195.2 | 855.2 | 1156.2 | 4469.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | predecessor_deferred_exec_us | 195.0 | 855.0 | 1155.8 | 4469.2 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | queue_gap_unattributed_us | 1.2 | 5.5 | 8.2 | 210.2 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | deferred_service_span_us | 472.8 | 1063.2 | 1363.2 | 4671.2 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | task_exec_span_us | 473.2 | 1064.0 | 1363.5 | 4673.2 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | numa0_span_us | 415.8 | 1000.5 | 1296.8 | 3818.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | numa1_span_us | 412.5 | 987.8 | 1276.0 | 4531.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | numa_completion_skew_us | 5.2 | 33.0 | 77.2 | 3677.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | producer_end_to_pub_us | 1.0 | 189.5 | 221.5 | 1181.2 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | pub_to_h2d_start_us | 18.2 | 208.5 | 225.2 | 260.8 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | overlap_budget_us | 620.5 | 1221.0 | 1503.2 | 4838.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | n_cold_assignments | 6.0 | 21.0 | 29.0 | 47.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 28213 | go_minus_dtoh_end_us | 5.0 | 6.2 | 7.2 | 1309.8 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | cold_pub_delta_us | -180.0 | -116.5 | -91.2 | 1004.2 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 1004.2 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | gpu_pre_h2d_gap_us | 10.2 | 14.0 | 14.8 | 1019.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | gpu_idle_inside_gap_us | 10.2 | 14.0 | 14.8 | 1019.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | h2d_duration_us | 8.2 | 12.0 | 17.5 | 25.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | gpu_post_h2d_gap_us | 6.0 | 9.5 | 10.0 | 10.8 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | combine_schedule_gap_us | 6.0 | 9.5 | 10.0 | 10.8 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | go_to_enqueue_us | 3.0 | 5.2 | 7.0 | 23.8 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | enqueue_to_start_us | 4.8 | 93.2 | 267.2 | 3191.2 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | predecessor_exec_union_us | 0.2 | 84.8 | 265.5 | 3190.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | predecessor_deferred_exec_us | 0.0 | 38.2 | 262.0 | 3111.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | queue_gap_unattributed_us | 4.0 | 8.5 | 15.0 | 100.8 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | deferred_service_span_us | 77.5 | 88.5 | 99.8 | 416.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | task_exec_span_us | 77.8 | 88.8 | 100.5 | 419.2 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | numa0_span_us | 27.2 | 31.0 | 39.0 | 310.5 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | numa1_span_us | 28.0 | 31.5 | 39.2 | 297.2 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | numa_completion_skew_us | 2.0 | 5.2 | 9.2 | 280.2 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | producer_end_to_pub_us | 301.2 | 343.2 | 361.8 | 1461.5 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | pub_to_h2d_start_us | 190.5 | 226.2 | 238.8 | 256.2 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | overlap_budget_us | 582.2 | 653.8 | 695.2 | 3531.2 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 3019 | go_minus_dtoh_end_us | 4.5 | 5.8 | 6.8 | 1151.0 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | cold_pub_delta_us | -127.2 | -88.2 | 211.2 | 3260.0 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | cold_pub_lateness_us | 0.0 | 0.0 | 211.2 | 3260.0 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | gpu_pre_h2d_gap_us | 10.2 | 14.0 | 229.2 | 3272.8 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | gpu_idle_inside_gap_us | 10.2 | 14.0 | 229.2 | 3257.0 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | h2d_duration_us | 8.2 | 9.2 | 12.2 | 19.5 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | gpu_post_h2d_gap_us | 6.5 | 9.5 | 10.0 | 10.5 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | combine_schedule_gap_us | 6.5 | 9.5 | 10.0 | 10.5 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | pub_to_h2d_start_us | 138.0 | 173.5 | 193.2 | 210.8 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | go_minus_dtoh_end_us | 4.8 | 5.8 | 7.2 | 21.5 |

민감도 (late share, cold_present_nonempty): {"26:cold_present_nonempty": {"late_share_thr_0us": 0.528, "late_share_thr_1us": 0.526, "late_share_thr_2us": 0.525, "late_share_thr_5us": 0.521, "clock_indeterminate_share_5us": 0.013, "n": 28213, "cold_gpu_ready_late_share": 1.0}}

### S9_CORR — coverage {'gpu_layer_rows_total': 48360, 'count_mismatch_pairs': 124, 'count_mismatch_rows': 24552, 'resync_unmatched_gpu': 24552, 'resync_unmatched_cpu': 0, 'resync_pairs': 0, 'method:count_match_zip': 23808} · clock {'OK': 23808}

| graph:cohort | n | 지표 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | cold_pub_delta_us | 971.8 | 2295.5 | 2918.8 | 7281.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | cold_pub_lateness_us | 971.8 | 2295.5 | 2918.8 | 7281.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | gpu_pre_h2d_gap_us | 985.8 | 2308.5 | 2933.2 | 7293.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | gpu_idle_inside_gap_us | 985.8 | 2308.5 | 2933.2 | 7293.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | h2d_duration_us | 30.0 | 33.0 | 34.5 | 41.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | gpu_post_h2d_gap_us | 7.2 | 9.5 | 10.0 | 11.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | combine_schedule_gap_us | 7.2 | 9.5 | 10.0 | 11.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | go_to_enqueue_us | 4.2 | 6.5 | 8.8 | 471.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | enqueue_to_start_us | 1309.2 | 2597.2 | 3219.0 | 7575.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | predecessor_exec_union_us | 1308.5 | 2596.2 | 3218.2 | 7574.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | predecessor_deferred_exec_us | 1308.2 | 2595.8 | 3218.2 | 7574.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | queue_gap_unattributed_us | 0.8 | 1.5 | 4.5 | 21.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | deferred_service_span_us | 1556.2 | 2832.0 | 3455.8 | 7814.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | task_exec_span_us | 1557.0 | 2832.5 | 3456.2 | 7814.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | numa0_span_us | 1526.5 | 2796.8 | 3381.8 | 7603.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | numa1_span_us | 1468.0 | 2715.2 | 3287.5 | 7784.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | numa_completion_skew_us | 50.2 | 119.2 | 282.8 | 3733.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | producer_end_to_pub_us | 0.5 | 1.0 | 1.5 | 948.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | pub_to_h2d_start_us | 13.2 | 20.2 | 351.5 | 467.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | overlap_budget_us | 1888.2 | 3152.5 | 3768.5 | 8119.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | n_cold_assignments | 43.0 | 127.0 | 164.0 | 234.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | go_minus_dtoh_end_us | 5.2 | 6.2 | 8.2 | 1596.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | cold_pub_delta_us | -403.2 | -323.5 | -233.5 | -214.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | gpu_pre_h2d_gap_us | 9.8 | 12.8 | 13.2 | 13.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | gpu_idle_inside_gap_us | 9.8 | 12.8 | 13.2 | 13.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | h2d_duration_us | 20.0 | 21.5 | 22.0 | 23.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | gpu_post_h2d_gap_us | 5.8 | 9.0 | 9.5 | 10.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | combine_schedule_gap_us | 5.8 | 9.0 | 9.5 | 10.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | go_to_enqueue_us | 4.8 | 7.2 | 8.0 | 8.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | enqueue_to_start_us | 589.0 | 863.2 | 942.2 | 1012.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | predecessor_exec_union_us | 588.0 | 862.2 | 941.5 | 1011.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | predecessor_deferred_exec_us | 518.5 | 791.8 | 873.2 | 934.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | queue_gap_unattributed_us | 1.0 | 1.5 | 1.5 | 2.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | deferred_service_span_us | 74.8 | 85.2 | 90.0 | 172.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | task_exec_span_us | 74.8 | 85.5 | 90.0 | 172.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | numa0_span_us | 47.8 | 55.0 | 63.0 | 65.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | numa1_span_us | 46.8 | 54.5 | 63.0 | 64.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | numa_completion_skew_us | 2.5 | 6.0 | 9.8 | 10.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | producer_end_to_pub_us | 172.2 | 259.5 | 282.8 | 350.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | pub_to_h2d_start_us | 413.2 | 432.8 | 438.2 | 446.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | overlap_budget_us | 1250.5 | 1513.5 | 1600.8 | 1681.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 75 | go_minus_dtoh_end_us | 5.2 | 6.2 | 6.8 | 7.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_delta_us | 65.2 | 375.0 | 691.5 | 1713.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_lateness_us | 65.2 | 375.0 | 691.5 | 1713.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_pre_h2d_gap_us | 78.8 | 389.8 | 698.0 | 1720.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_idle_inside_gap_us | 78.8 | 389.8 | 698.0 | 1695.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | h2d_duration_us | 31.2 | 35.0 | 37.2 | 37.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_post_h2d_gap_us | 7.8 | 9.5 | 10.0 | 10.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | combine_schedule_gap_us | 7.8 | 9.5 | 10.0 | 10.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | pub_to_h2d_start_us | 15.5 | 168.2 | 319.8 | 356.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | go_minus_dtoh_end_us | 5.2 | 6.5 | 7.5 | 857.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | cold_pub_delta_us | -84.8 | 61.5 | 191.8 | 1117.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | cold_pub_lateness_us | 0.0 | 61.5 | 191.8 | 1117.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | gpu_pre_h2d_gap_us | 11.2 | 74.5 | 204.8 | 1126.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | gpu_idle_inside_gap_us | 11.2 | 74.5 | 204.2 | 1126.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | h2d_duration_us | 4.2 | 5.2 | 5.8 | 7.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | gpu_post_h2d_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | combine_schedule_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | go_to_enqueue_us | 2.8 | 4.0 | 5.5 | 276.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | enqueue_to_start_us | 4.0 | 135.0 | 267.2 | 510.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | predecessor_exec_union_us | 0.2 | 134.2 | 266.2 | 494.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | predecessor_deferred_exec_us | 0.0 | 134.0 | 266.0 | 493.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | queue_gap_unattributed_us | 3.5 | 6.0 | 9.8 | 29.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | deferred_service_span_us | 149.2 | 332.2 | 442.8 | 1385.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | task_exec_span_us | 149.5 | 332.5 | 443.0 | 1385.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | numa0_span_us | 128.2 | 311.8 | 417.5 | 1369.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | numa1_span_us | 129.2 | 310.2 | 409.0 | 520.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | numa_completion_skew_us | 4.8 | 17.0 | 36.5 | 1182.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | producer_end_to_pub_us | 91.0 | 120.0 | 127.2 | 1311.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | pub_to_h2d_start_us | 95.5 | 105.5 | 108.2 | 113.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | overlap_budget_us | 341.2 | 385.8 | 512.8 | 745.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | n_cold_assignments | 1.0 | 4.0 | 5.0 | 7.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | go_minus_dtoh_end_us | 4.5 | 5.8 | 11.0 | 1208.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | cold_pub_delta_us | -86.2 | -75.5 | -67.8 | 192.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 192.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | gpu_pre_h2d_gap_us | 10.8 | 13.8 | 14.5 | 208.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | gpu_idle_inside_gap_us | 10.8 | 13.8 | 14.5 | 208.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | h2d_duration_us | 4.0 | 4.8 | 5.2 | 6.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | gpu_post_h2d_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | combine_schedule_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | go_to_enqueue_us | 3.0 | 4.0 | 5.8 | 56.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | enqueue_to_start_us | 4.0 | 17.2 | 61.8 | 1166.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | predecessor_exec_union_us | 0.2 | 0.5 | 50.0 | 1165.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | predecessor_deferred_exec_us | 0.0 | 0.0 | 49.5 | 1164.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | queue_gap_unattributed_us | 4.0 | 6.8 | 13.5 | 275.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | deferred_service_span_us | 30.5 | 37.8 | 47.2 | 324.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | task_exec_span_us | 30.8 | 37.8 | 47.5 | 324.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | numa0_span_us | 9.5 | 12.0 | 18.8 | 291.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | numa1_span_us | 13.2 | 16.8 | 23.2 | 311.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | numa_completion_skew_us | 5.2 | 8.5 | 12.5 | 273.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | producer_end_to_pub_us | 218.0 | 232.8 | 238.5 | 470.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | pub_to_h2d_start_us | 96.8 | 106.0 | 108.8 | 113.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | overlap_budget_us | 344.0 | 360.0 | 366.2 | 1420.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | go_minus_dtoh_end_us | 4.5 | 5.5 | 7.5 | 281.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_delta_us | -58.2 | -44.0 | 9.5 | 35.5 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_lateness_us | 0.0 | 0.0 | 9.5 | 35.5 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_pre_h2d_gap_us | 11.2 | 13.8 | 17.8 | 48.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_idle_inside_gap_us | 11.2 | 13.8 | 17.8 | 48.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | h2d_duration_us | 3.8 | 4.5 | 5.5 | 5.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_post_h2d_gap_us | 5.8 | 9.5 | 9.8 | 10.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | combine_schedule_gap_us | 5.8 | 9.5 | 9.8 | 10.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | pub_to_h2d_start_us | 69.0 | 78.2 | 79.2 | 82.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | go_minus_dtoh_end_us | 4.5 | 5.5 | 10.0 | 15.8 |

민감도 (late share, cold_present_nonempty): {"2:cold_present_nonempty": {"late_share_thr_0us": 0.963, "late_share_thr_1us": 0.963, "late_share_thr_2us": 0.963, "late_share_thr_5us": 0.962, "clock_indeterminate_share_5us": 0.001, "n": 15541, "cold_gpu_ready_late_share": 1.0}, "32:cold_present_nonempty": {"late_share_thr_0us": 0.097, "late_share_thr_1us": 0.096, "late_share_thr_2us": 0.096, "late_share_thr_5us": 0.094, "clock_indeterminate_share_5us": 0.006, "n": 3061, "cold_gpu_ready_late_share": 1.0}}

### S17_CORR_pinned — coverage {'gpu_layer_rows_total': 23870, 'method:count_match_zip': 23808, 'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} · clock {'OK': 23808}

| graph:cohort | n | 지표 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | cold_pub_delta_us | 869.5 | 2235.2 | 2866.8 | 10633.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | cold_pub_lateness_us | 869.5 | 2235.2 | 2866.8 | 10633.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | gpu_pre_h2d_gap_us | 882.2 | 2249.2 | 2879.8 | 10652.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | gpu_idle_inside_gap_us | 881.8 | 2249.2 | 2879.8 | 10652.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | h2d_duration_us | 29.8 | 33.0 | 35.0 | 43.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | gpu_post_h2d_gap_us | 7.2 | 9.5 | 10.0 | 14.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | combine_schedule_gap_us | 7.2 | 9.5 | 10.0 | 14.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | go_to_enqueue_us | 4.2 | 7.0 | 9.5 | 689.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | enqueue_to_start_us | 1199.2 | 2532.8 | 3155.2 | 10927.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | predecessor_exec_union_us | 1198.2 | 2531.8 | 3154.5 | 10926.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | predecessor_deferred_exec_us | 1198.0 | 2531.8 | 3154.2 | 10926.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | queue_gap_unattributed_us | 1.0 | 1.8 | 6.2 | 50.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | deferred_service_span_us | 1447.2 | 2768.0 | 3393.8 | 11159.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | task_exec_span_us | 1447.8 | 2768.5 | 3394.2 | 11160.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | numa0_span_us | 1409.5 | 2731.5 | 3315.2 | 11131.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | numa1_span_us | 1372.0 | 2676.0 | 3282.8 | 11005.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | numa_completion_skew_us | 39.5 | 108.2 | 265.0 | 2996.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | producer_end_to_pub_us | 0.8 | 1.2 | 33.8 | 373.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | pub_to_h2d_start_us | 13.0 | 20.8 | 365.2 | 447.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | overlap_budget_us | 1771.8 | 3085.8 | 3694.5 | 11478.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | n_cold_assignments | 40.0 | 128.0 | 165.0 | 240.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | go_minus_dtoh_end_us | 5.2 | 6.8 | 8.0 | 1385.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | cold_pub_delta_us | -400.8 | -321.0 | -235.5 | -217.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | gpu_pre_h2d_gap_us | 10.0 | 13.8 | 14.8 | 16.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | gpu_idle_inside_gap_us | 10.0 | 13.8 | 14.8 | 16.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | h2d_duration_us | 20.0 | 21.8 | 22.8 | 23.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | gpu_post_h2d_gap_us | 6.0 | 9.2 | 10.2 | 11.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | combine_schedule_gap_us | 6.0 | 9.2 | 10.2 | 11.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | go_to_enqueue_us | 5.0 | 7.2 | 10.2 | 10.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | enqueue_to_start_us | 531.8 | 896.0 | 1345.8 | 1570.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | predecessor_exec_union_us | 530.2 | 895.0 | 1335.0 | 1569.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | predecessor_deferred_exec_us | 448.2 | 819.0 | 957.2 | 1500.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | queue_gap_unattributed_us | 1.2 | 2.2 | 6.2 | 10.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | deferred_service_span_us | 76.8 | 93.2 | 141.8 | 144.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | task_exec_span_us | 77.0 | 93.5 | 142.0 | 144.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | numa0_span_us | 45.2 | 59.0 | 117.5 | 118.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | numa1_span_us | 45.8 | 54.8 | 61.8 | 68.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | numa_completion_skew_us | 3.5 | 9.5 | 67.8 | 71.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | producer_end_to_pub_us | 175.0 | 417.2 | 489.0 | 555.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | pub_to_h2d_start_us | 410.2 | 427.8 | 442.2 | 448.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | overlap_budget_us | 1179.0 | 1567.0 | 1992.0 | 2230.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | go_minus_dtoh_end_us | 5.2 | 6.5 | 8.8 | 15.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_delta_us | 52.8 | 397.0 | 835.0 | 1622.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_lateness_us | 52.8 | 397.0 | 835.0 | 1622.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_pre_h2d_gap_us | 67.8 | 411.2 | 844.0 | 1634.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_idle_inside_gap_us | 67.8 | 411.2 | 844.0 | 1616.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | h2d_duration_us | 28.8 | 37.0 | 38.2 | 39.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_post_h2d_gap_us | 8.0 | 9.2 | 9.5 | 10.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | combine_schedule_gap_us | 8.0 | 9.2 | 9.5 | 10.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | pub_to_h2d_start_us | 16.5 | 275.8 | 337.8 | 397.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | go_minus_dtoh_end_us | 5.2 | 7.0 | 8.2 | 13.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | cold_pub_delta_us | -84.8 | 62.8 | 197.8 | 1083.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | cold_pub_lateness_us | 0.0 | 62.8 | 197.8 | 1083.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | gpu_pre_h2d_gap_us | 11.2 | 76.2 | 211.5 | 1097.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | gpu_idle_inside_gap_us | 11.2 | 76.2 | 211.5 | 1097.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | h2d_duration_us | 4.0 | 5.0 | 5.8 | 6.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | gpu_post_h2d_gap_us | 5.5 | 9.5 | 10.0 | 11.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | combine_schedule_gap_us | 5.5 | 9.5 | 10.0 | 11.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | go_to_enqueue_us | 2.5 | 4.2 | 5.5 | 27.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | enqueue_to_start_us | 4.0 | 139.2 | 271.2 | 1172.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | predecessor_exec_union_us | 0.2 | 137.2 | 270.8 | 1171.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | predecessor_deferred_exec_us | 0.0 | 136.8 | 270.5 | 1171.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | queue_gap_unattributed_us | 3.2 | 6.8 | 11.0 | 270.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | deferred_service_span_us | 151.2 | 335.0 | 445.5 | 1426.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | task_exec_span_us | 151.5 | 335.5 | 446.0 | 1427.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | numa0_span_us | 127.8 | 314.5 | 422.5 | 1358.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | numa1_span_us | 130.5 | 310.8 | 414.8 | 1411.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | numa_completion_skew_us | 4.5 | 19.0 | 63.5 | 1295.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | producer_end_to_pub_us | 88.8 | 118.0 | 126.2 | 378.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | pub_to_h2d_start_us | 95.2 | 106.2 | 109.2 | 112.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | overlap_budget_us | 340.8 | 388.8 | 517.5 | 1428.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | n_cold_assignments | 1.0 | 4.0 | 5.0 | 7.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | go_minus_dtoh_end_us | 4.5 | 5.8 | 6.8 | 288.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | cold_pub_delta_us | -86.5 | -75.5 | -67.8 | 317.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 317.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | gpu_pre_h2d_gap_us | 10.8 | 13.8 | 14.5 | 337.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | gpu_idle_inside_gap_us | 10.8 | 13.8 | 14.5 | 337.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | h2d_duration_us | 3.8 | 4.8 | 5.2 | 6.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | gpu_post_h2d_gap_us | 5.5 | 9.5 | 10.0 | 10.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | combine_schedule_gap_us | 5.5 | 9.5 | 10.0 | 10.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | go_to_enqueue_us | 2.8 | 4.8 | 6.2 | 330.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | enqueue_to_start_us | 3.8 | 18.2 | 61.8 | 1134.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | predecessor_exec_union_us | 0.0 | 0.5 | 55.2 | 1134.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | predecessor_deferred_exec_us | 0.0 | 0.0 | 54.5 | 1133.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | queue_gap_unattributed_us | 3.8 | 7.5 | 12.8 | 371.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | deferred_service_span_us | 31.0 | 38.2 | 47.5 | 552.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | task_exec_span_us | 31.2 | 38.5 | 47.5 | 552.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | numa0_span_us | 9.2 | 12.2 | 16.0 | 89.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | numa1_span_us | 13.0 | 16.8 | 21.5 | 318.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | numa_completion_skew_us | 5.0 | 9.0 | 11.5 | 527.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | producer_end_to_pub_us | 217.8 | 232.8 | 239.2 | 624.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | pub_to_h2d_start_us | 97.2 | 106.8 | 109.2 | 113.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | overlap_budget_us | 344.0 | 359.8 | 366.8 | 1394.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | go_minus_dtoh_end_us | 4.2 | 5.2 | 6.0 | 275.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_delta_us | -60.2 | -44.2 | -38.0 | -35.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_pre_h2d_gap_us | 11.0 | 13.8 | 14.5 | 15.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_idle_inside_gap_us | 11.0 | 13.8 | 14.5 | 15.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | h2d_duration_us | 3.8 | 4.5 | 5.0 | 5.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_post_h2d_gap_us | 5.5 | 9.8 | 9.8 | 10.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | combine_schedule_gap_us | 5.5 | 9.8 | 9.8 | 10.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | pub_to_h2d_start_us | 70.8 | 78.8 | 81.0 | 82.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | go_minus_dtoh_end_us | 4.2 | 5.0 | 5.5 | 9.2 |

민감도 (late share, cold_present_nonempty): {"2:cold_present_nonempty": {"late_share_thr_0us": 0.953, "late_share_thr_1us": 0.953, "late_share_thr_2us": 0.952, "late_share_thr_5us": 0.952, "clock_indeterminate_share_5us": 0.001, "n": 15493, "cold_gpu_ready_late_share": 1.0}, "32:cold_present_nonempty": {"late_share_thr_0us": 0.097, "late_share_thr_1us": 0.096, "late_share_thr_2us": 0.095, "late_share_thr_5us": 0.094, "clock_indeterminate_share_5us": 0.006, "n": 3061, "cold_gpu_ready_late_share": 1.0}}

### S18_CORR_pinned — coverage {'gpu_layer_rows_total': 23870, 'method:count_match_zip': 23808, 'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} · clock {'OK': 23808}

| graph:cohort | n | 지표 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | cold_pub_delta_us | 882.0 | 2262.0 | 2898.0 | 6531.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | cold_pub_lateness_us | 882.0 | 2262.0 | 2898.0 | 6531.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | gpu_pre_h2d_gap_us | 895.0 | 2274.8 | 2914.0 | 6544.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | gpu_idle_inside_gap_us | 895.0 | 2274.8 | 2914.0 | 6544.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | h2d_duration_us | 30.0 | 33.2 | 35.0 | 41.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | gpu_post_h2d_gap_us | 7.2 | 9.5 | 10.0 | 11.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | combine_schedule_gap_us | 7.2 | 9.5 | 10.0 | 11.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | go_to_enqueue_us | 4.0 | 7.0 | 9.2 | 488.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | enqueue_to_start_us | 1210.0 | 2551.0 | 3179.2 | 6854.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | predecessor_exec_union_us | 1209.0 | 2549.5 | 3178.2 | 6853.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | predecessor_deferred_exec_us | 1209.0 | 2549.5 | 3178.2 | 6853.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | queue_gap_unattributed_us | 1.0 | 1.8 | 6.2 | 28.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | deferred_service_span_us | 1459.0 | 2790.2 | 3418.5 | 7078.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | task_exec_span_us | 1459.2 | 2790.5 | 3418.8 | 7078.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | numa0_span_us | 1422.8 | 2746.5 | 3359.8 | 7046.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | numa1_span_us | 1374.2 | 2681.0 | 3284.8 | 5356.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | numa_completion_skew_us | 46.2 | 138.2 | 318.0 | 3704.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | producer_end_to_pub_us | 0.5 | 1.2 | 30.2 | 409.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | pub_to_h2d_start_us | 13.2 | 21.0 | 364.5 | 456.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | overlap_budget_us | 1782.5 | 3103.5 | 3716.2 | 7437.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | n_cold_assignments | 40.0 | 128.0 | 165.0 | 240.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15493 | go_minus_dtoh_end_us | 5.2 | 6.2 | 7.8 | 615.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | cold_pub_delta_us | -401.0 | -319.8 | -217.8 | -214.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | gpu_pre_h2d_gap_us | 10.2 | 13.5 | 14.8 | 14.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | gpu_idle_inside_gap_us | 10.2 | 13.5 | 14.8 | 14.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | h2d_duration_us | 20.2 | 22.2 | 23.5 | 26.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | gpu_post_h2d_gap_us | 5.8 | 9.0 | 9.5 | 9.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | combine_schedule_gap_us | 5.8 | 9.0 | 9.5 | 9.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | go_to_enqueue_us | 4.8 | 7.5 | 8.0 | 8.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | enqueue_to_start_us | 542.5 | 890.0 | 1174.0 | 1951.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | predecessor_exec_union_us | 540.2 | 888.0 | 1173.0 | 1948.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | predecessor_deferred_exec_us | 468.8 | 814.5 | 1103.5 | 1860.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | queue_gap_unattributed_us | 1.2 | 2.5 | 4.0 | 5.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | deferred_service_span_us | 74.8 | 90.2 | 137.5 | 164.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | task_exec_span_us | 75.0 | 90.2 | 137.5 | 164.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | numa0_span_us | 43.8 | 51.5 | 65.8 | 102.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | numa1_span_us | 45.0 | 53.5 | 57.5 | 62.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | numa_completion_skew_us | 3.5 | 10.2 | 23.5 | 58.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | producer_end_to_pub_us | 175.2 | 431.8 | 510.8 | 548.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | pub_to_h2d_start_us | 412.0 | 429.0 | 438.2 | 441.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | overlap_budget_us | 1192.0 | 1547.2 | 1832.0 | 2607.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 123 | go_minus_dtoh_end_us | 5.0 | 6.0 | 14.0 | 17.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_delta_us | 61.5 | 406.8 | 1413.2 | 1621.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_lateness_us | 61.5 | 406.8 | 1413.2 | 1621.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_pre_h2d_gap_us | 77.5 | 417.5 | 1432.8 | 1641.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_idle_inside_gap_us | 77.5 | 417.5 | 1414.8 | 1641.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | h2d_duration_us | 30.0 | 36.0 | 37.5 | 38.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_post_h2d_gap_us | 8.0 | 9.5 | 9.8 | 10.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | combine_schedule_gap_us | 8.0 | 9.5 | 9.8 | 10.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | pub_to_h2d_start_us | 17.0 | 282.0 | 352.5 | 398.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | go_minus_dtoh_end_us | 5.2 | 6.2 | 7.0 | 7.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | cold_pub_delta_us | -84.2 | 69.0 | 197.5 | 9705.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | cold_pub_lateness_us | 0.0 | 69.0 | 197.5 | 9705.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | gpu_pre_h2d_gap_us | 11.5 | 84.2 | 215.5 | 9716.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | gpu_idle_inside_gap_us | 11.5 | 84.2 | 215.5 | 9716.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | h2d_duration_us | 4.2 | 5.5 | 6.2 | 9.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | gpu_post_h2d_gap_us | 5.8 | 9.5 | 9.8 | 11.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | combine_schedule_gap_us | 5.8 | 9.5 | 9.8 | 11.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | go_to_enqueue_us | 3.0 | 4.0 | 10.0 | 350.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | enqueue_to_start_us | 4.0 | 141.5 | 271.5 | 5446.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | predecessor_exec_union_us | 0.2 | 134.0 | 268.2 | 490.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | predecessor_deferred_exec_us | 0.0 | 134.0 | 268.2 | 490.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | queue_gap_unattributed_us | 3.5 | 6.8 | 13.0 | 5446.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | deferred_service_span_us | 151.0 | 337.2 | 443.5 | 1577.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | task_exec_span_us | 151.2 | 337.8 | 444.2 | 1578.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | numa0_span_us | 130.5 | 315.0 | 420.0 | 878.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | numa1_span_us | 129.0 | 309.2 | 415.5 | 1560.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | numa_completion_skew_us | 4.5 | 20.5 | 50.5 | 1247.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | producer_end_to_pub_us | 88.8 | 119.8 | 128.2 | 9858.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | pub_to_h2d_start_us | 95.0 | 105.8 | 108.8 | 112.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | overlap_budget_us | 341.8 | 389.8 | 522.5 | 5700.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | n_cold_assignments | 1.0 | 4.0 | 5.0 | 7.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3061 | go_minus_dtoh_end_us | 4.8 | 5.8 | 7.5 | 835.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | cold_pub_delta_us | -86.0 | -75.0 | -66.8 | 5838.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 5838.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | gpu_pre_h2d_gap_us | 10.8 | 14.0 | 15.0 | 5852.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | gpu_idle_inside_gap_us | 10.8 | 14.0 | 15.0 | 5844.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | h2d_duration_us | 4.0 | 5.0 | 5.5 | 8.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | gpu_post_h2d_gap_us | 5.5 | 9.8 | 10.0 | 11.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | combine_schedule_gap_us | 5.5 | 9.8 | 10.0 | 11.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | go_to_enqueue_us | 3.0 | 4.0 | 7.0 | 385.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | enqueue_to_start_us | 3.8 | 26.5 | 69.0 | 9798.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | predecessor_exec_union_us | 0.2 | 0.8 | 55.0 | 1330.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | predecessor_deferred_exec_us | 0.0 | 0.0 | 54.5 | 1330.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | queue_gap_unattributed_us | 3.8 | 7.5 | 14.0 | 9798.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | deferred_service_span_us | 28.2 | 35.8 | 47.8 | 319.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | task_exec_span_us | 28.8 | 36.0 | 48.0 | 320.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | numa0_span_us | 11.8 | 15.0 | 22.8 | 305.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | numa1_span_us | 9.8 | 12.2 | 18.0 | 299.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | numa_completion_skew_us | 1.5 | 4.8 | 7.8 | 285.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | producer_end_to_pub_us | 220.5 | 235.8 | 241.5 | 6133.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | pub_to_h2d_start_us | 97.0 | 106.2 | 108.8 | 112.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | overlap_budget_us | 344.2 | 360.5 | 368.0 | 10044.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4747 | go_minus_dtoh_end_us | 4.5 | 5.5 | 8.5 | 392.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_delta_us | -61.2 | -46.2 | -40.8 | 8.5 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 8.5 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_pre_h2d_gap_us | 11.0 | 13.5 | 14.2 | 21.5 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_idle_inside_gap_us | 11.0 | 13.5 | 14.2 | 21.5 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | h2d_duration_us | 4.0 | 4.8 | 5.2 | 6.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_post_h2d_gap_us | 5.2 | 9.8 | 9.8 | 10.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | combine_schedule_gap_us | 5.2 | 9.8 | 9.8 | 10.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | pub_to_h2d_start_us | 72.0 | 82.0 | 83.2 | 83.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | go_minus_dtoh_end_us | 4.5 | 5.8 | 6.0 | 6.8 |

민감도 (late share, cold_present_nonempty): {"2:cold_present_nonempty": {"late_share_thr_0us": 0.952, "late_share_thr_1us": 0.952, "late_share_thr_2us": 0.952, "late_share_thr_5us": 0.952, "clock_indeterminate_share_5us": 0.002, "n": 15493, "cold_gpu_ready_late_share": 1.0}, "32:cold_present_nonempty": {"late_share_thr_0us": 0.104, "late_share_thr_1us": 0.103, "late_share_thr_2us": 0.102, "late_share_thr_5us": 0.1, "clock_indeterminate_share_5us": 0.008, "n": 3061, "cold_gpu_ready_late_share": 1.0}}

### S22_CORR_vllmw — coverage {'gpu_layer_rows_total': 23870, 'count_mismatch_pairs': 86, 'count_mismatch_rows': 86, 'resync_unmatched_gpu': 216, 'resync_gpu_rows_without_dtoh': 1, 'resync_unmatched_cpu': 155, 'resync_pairs': 15717, 'method:ANCHOR_RESYNC(go>=dtoh_end<next)': 6084, 'zip_order_violation_groups': 38, 'method:ANCHOR_RESYNC(order_violation)': 9633, 'method:count_match_zip': 7936} · clock {'OK': 22668, 'CLOCK_INDETERMINATE': 966, 'CLOCK_ORDER_VIOLATION': 19}

| graph:cohort | n | 지표 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | cold_pub_delta_us | 392.0 | 59969.0 | 63555.5 | 111427.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | cold_pub_lateness_us | 392.0 | 59969.0 | 63555.5 | 111427.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | gpu_pre_h2d_gap_us | 337.2 | 832.0 | 1083.5 | 111432.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | gpu_idle_inside_gap_us | 337.2 | 832.0 | 1083.5 | 111432.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | h2d_duration_us | 30.2 | 32.8 | 34.0 | 43.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | gpu_post_h2d_gap_us | 7.2 | 9.5 | 10.0 | 10.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | combine_schedule_gap_us | 7.2 | 9.5 | 10.0 | 10.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | go_to_enqueue_us | 3.0 | 5.8 | 8.2 | 478.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | enqueue_to_start_us | 670.5 | 1186.0 | 1434.2 | 111828.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | predecessor_exec_union_us | 669.8 | 1185.2 | 1433.8 | 111826.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | predecessor_deferred_exec_us | 669.2 | 1185.0 | 1433.5 | 111826.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | queue_gap_unattributed_us | 0.8 | 1.2 | 4.2 | 28.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | deferred_service_span_us | 921.8 | 1422.8 | 1669.0 | 112062.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | task_exec_span_us | 922.2 | 1423.0 | 1669.2 | 112063.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | numa0_span_us | 895.2 | 1391.0 | 1611.5 | 53252.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | numa1_span_us | 855.2 | 1329.8 | 1545.0 | 3349.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | numa_completion_skew_us | 35.2 | 96.5 | 189.8 | 51910.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | producer_end_to_pub_us | 0.5 | 169479.5 | 196611.5 | 3279518.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | pub_to_h2d_start_us | 12.0 | 81.5 | 326.8 | 540.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | overlap_budget_us | 1049.5 | 152901.2 | 198086.5 | 3281104.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | n_cold_assignments | 16.0 | 31.0 | 41.0 | 100.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15410 | go_minus_dtoh_end_us | 5.5 | 59600.8 | 63156.2 | 65531.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | cold_pub_delta_us | -265.5 | -185.0 | -141.0 | -141.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | gpu_pre_h2d_gap_us | 12.2 | 13.0 | 13.8 | 13.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | gpu_idle_inside_gap_us | 12.2 | 13.0 | 13.8 | 13.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | h2d_duration_us | 20.0 | 21.5 | 25.8 | 25.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | gpu_post_h2d_gap_us | 6.5 | 9.0 | 10.5 | 10.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | combine_schedule_gap_us | 6.5 | 9.0 | 10.5 | 10.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | go_to_enqueue_us | 5.2 | 7.0 | 8.2 | 8.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | enqueue_to_start_us | 78.2 | 592.8 | 879.8 | 879.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | predecessor_exec_union_us | 70.5 | 591.2 | 878.5 | 878.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | predecessor_deferred_exec_us | 0.0 | 512.8 | 802.2 | 802.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | queue_gap_unattributed_us | 5.0 | 8.5 | 9.5 | 9.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | deferred_service_span_us | 63.5 | 76.8 | 79.8 | 79.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | task_exec_span_us | 63.8 | 77.0 | 80.2 | 80.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | numa0_span_us | 35.8 | 46.8 | 49.2 | 49.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | numa1_span_us | 37.8 | 49.5 | 51.5 | 51.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | numa_completion_skew_us | 3.2 | 5.2 | 15.2 | 15.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | producer_end_to_pub_us | 33759.5 | 40743.2 | 40762.0 | 40762.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | pub_to_h2d_start_us | 297.5 | 360.2 | 362.2 | 362.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | overlap_budget_us | 34109.8 | 41234.8 | 41240.5 | 41240.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | go_minus_dtoh_end_us | 4.5 | 5.0 | 5.2 | 5.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 255 | cold_pub_delta_us | -200.5 | 59115.0 | 62321.2 | 64800.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 255 | cold_pub_lateness_us | 0.0 | 59115.0 | 62321.2 | 64800.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 255 | gpu_pre_h2d_gap_us | 10.5 | 122.0 | 308.2 | 737.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 255 | gpu_idle_inside_gap_us | 10.5 | 122.0 | 308.2 | 732.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 255 | h2d_duration_us | 21.8 | 33.8 | 34.5 | 35.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 255 | gpu_post_h2d_gap_us | 6.0 | 9.2 | 9.5 | 10.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 255 | combine_schedule_gap_us | 6.0 | 9.2 | 9.5 | 10.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 255 | pub_to_h2d_start_us | 212.2 | 502.8 | 526.5 | 555.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 255 | go_minus_dtoh_end_us | 5.2 | 59319.0 | 62670.0 | 64807.0 |
| 2:unknown_path_or_mapping ['step[DECODE bs=63]'] | 36 | cold_pub_delta_us | 443.8 | 781.8 | 852.2 | 852.2 |
| 2:unknown_path_or_mapping ['step[DECODE bs=63]'] | 36 | cold_pub_lateness_us | 443.8 | 781.8 | 852.2 | 852.2 |
| 2:unknown_path_or_mapping ['step[DECODE bs=63]'] | 36 | gpu_pre_h2d_gap_us | 456.2 | 790.8 | 871.5 | 871.5 |
| 2:unknown_path_or_mapping ['step[DECODE bs=63]'] | 36 | gpu_idle_inside_gap_us | 456.2 | 788.8 | 871.5 | 871.5 |
| 2:unknown_path_or_mapping ['step[DECODE bs=63]'] | 36 | h2d_duration_us | 30.2 | 32.2 | 33.5 | 33.5 |
| 2:unknown_path_or_mapping ['step[DECODE bs=63]'] | 36 | gpu_post_h2d_gap_us | 8.2 | 9.0 | 9.8 | 9.8 |
| 2:unknown_path_or_mapping ['step[DECODE bs=63]'] | 36 | combine_schedule_gap_us | 8.2 | 9.0 | 9.8 | 9.8 |
| 2:unknown_path_or_mapping ['step[DECODE bs=63]'] | 36 | pub_to_h2d_start_us | 12.2 | 18.8 | 61.2 | 61.2 |
| 2:unknown_path_or_mapping ['step[DECODE bs=63]'] | 36 | go_minus_dtoh_end_us | 5.2 | 6.5 | 25.2 | 25.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | cold_pub_delta_us | -84.5 | -50.2 | 48.0 | 4761.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | cold_pub_lateness_us | 0.0 | 0.0 | 48.0 | 4761.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | gpu_pre_h2d_gap_us | 10.8 | 14.0 | 58.2 | 4779.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | gpu_idle_inside_gap_us | 10.8 | 14.0 | 58.2 | 4779.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | h2d_duration_us | 4.0 | 5.0 | 5.5 | 6.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | gpu_post_h2d_gap_us | 5.5 | 9.2 | 10.0 | 10.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | combine_schedule_gap_us | 5.5 | 9.2 | 10.0 | 10.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | go_to_enqueue_us | 3.5 | 4.8 | 6.2 | 270.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | enqueue_to_start_us | 3.2 | 19.2 | 100.8 | 418.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | predecessor_exec_union_us | 0.0 | 9.2 | 96.0 | 417.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | predecessor_deferred_exec_us | 0.0 | 0.5 | 95.8 | 417.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | queue_gap_unattributed_us | 3.2 | 5.8 | 11.0 | 276.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | deferred_service_span_us | 171.0 | 276.8 | 338.8 | 5100.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | task_exec_span_us | 171.2 | 276.8 | 339.0 | 5101.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | numa0_span_us | 154.8 | 258.8 | 314.8 | 722.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | numa1_span_us | 124.0 | 255.5 | 307.5 | 5074.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | numa_completion_skew_us | 24.0 | 59.2 | 78.5 | 4899.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | producer_end_to_pub_us | 73.8 | 108.8 | 118.5 | 352.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | pub_to_h2d_start_us | 95.0 | 105.5 | 108.5 | 112.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | overlap_budget_us | 337.5 | 355.2 | 365.2 | 675.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | n_cold_assignments | 1.0 | 3.0 | 4.0 | 6.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2463 | go_minus_dtoh_end_us | 4.5 | 5.8 | 6.5 | 21.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | cold_pub_delta_us | -85.0 | -71.5 | -66.0 | 288.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 288.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | gpu_pre_h2d_gap_us | 10.8 | 13.8 | 14.5 | 301.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | gpu_idle_inside_gap_us | 10.5 | 13.8 | 14.5 | 301.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | h2d_duration_us | 3.8 | 4.8 | 5.2 | 6.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | gpu_post_h2d_gap_us | 5.5 | 9.2 | 10.0 | 11.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | combine_schedule_gap_us | 5.5 | 9.2 | 10.0 | 11.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | go_to_enqueue_us | 3.5 | 4.8 | 6.8 | 26.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | enqueue_to_start_us | 3.2 | 9.2 | 40.8 | 4836.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | predecessor_exec_union_us | 0.0 | 0.5 | 37.2 | 4834.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | predecessor_deferred_exec_us | 0.0 | 0.0 | 33.8 | 4834.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | queue_gap_unattributed_us | 3.2 | 5.8 | 9.8 | 153.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | deferred_service_span_us | 30.2 | 37.2 | 47.5 | 332.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | task_exec_span_us | 30.5 | 37.5 | 47.8 | 333.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | numa0_span_us | 9.8 | 12.2 | 16.5 | 286.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | numa1_span_us | 13.0 | 16.2 | 22.5 | 314.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | numa_completion_skew_us | 4.5 | 8.0 | 11.8 | 306.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | producer_end_to_pub_us | 217.0 | 231.8 | 237.5 | 494.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | pub_to_h2d_start_us | 95.8 | 105.5 | 108.5 | 112.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | overlap_budget_us | 341.0 | 358.0 | 364.5 | 5125.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5345 | go_minus_dtoh_end_us | 4.5 | 5.8 | 6.5 | 382.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_delta_us | -61.2 | -42.0 | -13.5 | 183.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 183.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_pre_h2d_gap_us | 11.0 | 14.2 | 16.5 | 198.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_idle_inside_gap_us | 11.0 | 14.2 | 16.5 | 198.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | h2d_duration_us | 3.8 | 5.0 | 5.0 | 5.5 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_post_h2d_gap_us | 5.8 | 9.5 | 10.0 | 10.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | combine_schedule_gap_us | 5.8 | 9.5 | 10.0 | 10.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | pub_to_h2d_start_us | 72.0 | 79.2 | 80.5 | 80.5 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | go_minus_dtoh_end_us | 4.8 | 6.0 | 11.5 | 12.0 |

민감도 (late share, cold_present_nonempty): {"2:cold_present_nonempty": {"late_share_thr_0us": 0.916, "late_share_thr_1us": 0.916, "late_share_thr_2us": 0.915, "late_share_thr_5us": 0.914, "clock_indeterminate_share_5us": 0.007, "n": 15410, "cold_gpu_ready_late_share": 1.0}, "32:cold_present_nonempty": {"late_share_thr_0us": 0.02, "late_share_thr_1us": 0.02, "late_share_thr_2us": 0.02, "late_share_thr_5us": 0.019, "clock_indeterminate_share_5us": 0.002, "n": 2463, "cold_gpu_ready_late_share": 1.0}}

### S23_CORR_vllmw — coverage {'gpu_layer_rows_total': 23870, 'count_mismatch_pairs': 78, 'count_mismatch_rows': 78, 'resync_gpu_rows_without_dtoh': 1, 'resync_unmatched_cpu': 54, 'resync_pairs': 15818, 'method:ANCHOR_RESYNC(go>=dtoh_end<next)': 4088, 'resync_unmatched_gpu': 115, 'zip_order_violation_groups': 46, 'method:ANCHOR_RESYNC(order_violation)': 11730, 'method:count_match_zip': 7936} · clock {'OK': 23754}

| graph:cohort | n | 지표 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | cold_pub_delta_us | 328.2 | 826.2 | 1054.8 | 7230.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | cold_pub_lateness_us | 328.2 | 826.2 | 1054.8 | 7230.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | gpu_pre_h2d_gap_us | 341.2 | 839.2 | 1071.5 | 7241.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | gpu_idle_inside_gap_us | 341.0 | 839.2 | 1071.5 | 7232.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | h2d_duration_us | 29.2 | 32.2 | 34.0 | 40.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | gpu_post_h2d_gap_us | 7.2 | 9.5 | 10.0 | 11.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | combine_schedule_gap_us | 7.2 | 9.5 | 10.0 | 11.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | go_to_enqueue_us | 4.5 | 7.0 | 9.2 | 733.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | enqueue_to_start_us | 667.5 | 1185.8 | 1414.0 | 7615.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | predecessor_exec_union_us | 666.0 | 1184.2 | 1412.2 | 7614.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | predecessor_deferred_exec_us | 666.0 | 1184.2 | 1412.2 | 7614.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | queue_gap_unattributed_us | 1.2 | 2.0 | 6.5 | 21.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | deferred_service_span_us | 915.8 | 1422.5 | 1648.8 | 7882.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | task_exec_span_us | 916.2 | 1423.0 | 1649.0 | 7883.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | numa0_span_us | 884.5 | 1385.2 | 1598.2 | 3482.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | numa1_span_us | 843.0 | 1323.2 | 1529.2 | 7842.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | numa_completion_skew_us | 37.8 | 96.2 | 181.8 | 6769.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | producer_end_to_pub_us | 0.8 | 1.2 | 90.2 | 68410.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | pub_to_h2d_start_us | 13.5 | 89.2 | 347.2 | 537.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | overlap_budget_us | 1267.5 | 1810.5 | 2197.5 | 69743.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | n_cold_assignments | 17.0 | 32.0 | 41.0 | 83.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15541 | go_minus_dtoh_end_us | 5.5 | 7.5 | 9.5 | 709.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | cold_pub_delta_us | -307.2 | -137.5 | -136.8 | -136.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | gpu_pre_h2d_gap_us | 10.0 | 12.2 | 14.0 | 14.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | gpu_idle_inside_gap_us | 10.0 | 12.2 | 14.0 | 14.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | h2d_duration_us | 20.0 | 24.2 | 25.8 | 25.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | gpu_post_h2d_gap_us | 6.0 | 9.2 | 9.5 | 9.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | combine_schedule_gap_us | 6.0 | 9.2 | 9.5 | 9.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | go_to_enqueue_us | 7.0 | 8.8 | 10.5 | 10.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | enqueue_to_start_us | 82.5 | 1103.5 | 1117.0 | 1117.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | predecessor_exec_union_us | 76.8 | 1100.2 | 1115.5 | 1115.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | predecessor_deferred_exec_us | 0.0 | 1019.8 | 1021.5 | 1021.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | queue_gap_unattributed_us | 5.8 | 12.0 | 17.5 | 17.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | deferred_service_span_us | 70.2 | 88.5 | 94.0 | 94.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | task_exec_span_us | 70.5 | 88.5 | 94.2 | 94.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | numa0_span_us | 35.8 | 55.0 | 55.2 | 55.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | numa1_span_us | 37.0 | 50.5 | 55.8 | 55.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | numa_completion_skew_us | 2.5 | 4.8 | 8.0 | 8.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | producer_end_to_pub_us | 421.8 | 529.2 | 533.5 | 533.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | pub_to_h2d_start_us | 319.0 | 431.2 | 457.0 | 457.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | overlap_budget_us | 909.0 | 1498.0 | 1515.2 | 1515.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 21 | go_minus_dtoh_end_us | 2.5 | 5.2 | 5.8 | 5.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_delta_us | -242.2 | 74.0 | 335.8 | 968.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_lateness_us | 0.0 | 74.0 | 335.8 | 968.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_pre_h2d_gap_us | 10.8 | 84.2 | 352.2 | 988.5 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_idle_inside_gap_us | 10.8 | 84.2 | 352.2 | 982.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | h2d_duration_us | 21.2 | 32.8 | 33.5 | 35.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_post_h2d_gap_us | 6.2 | 9.2 | 9.8 | 10.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | combine_schedule_gap_us | 6.2 | 9.2 | 9.8 | 10.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | pub_to_h2d_start_us | 253.8 | 487.5 | 508.5 | 512.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | go_minus_dtoh_end_us | 5.2 | 7.2 | 11.2 | 17.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | cold_pub_delta_us | -84.8 | -22.5 | 98.5 | 419.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | cold_pub_lateness_us | 0.0 | 0.0 | 98.5 | 419.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | gpu_pre_h2d_gap_us | 11.0 | 14.8 | 113.8 | 426.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | gpu_idle_inside_gap_us | 11.0 | 14.8 | 113.8 | 426.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | h2d_duration_us | 4.2 | 5.5 | 6.0 | 7.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | gpu_post_h2d_gap_us | 5.5 | 9.8 | 10.0 | 11.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | combine_schedule_gap_us | 5.5 | 9.8 | 10.0 | 11.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | go_to_enqueue_us | 3.0 | 3.8 | 5.0 | 25.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | enqueue_to_start_us | 3.8 | 50.8 | 172.0 | 433.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | predecessor_exec_union_us | 0.0 | 49.0 | 170.2 | 432.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | predecessor_deferred_exec_us | 0.0 | 48.8 | 170.2 | 432.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | queue_gap_unattributed_us | 3.5 | 6.5 | 11.0 | 68.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | deferred_service_span_us | 146.2 | 291.2 | 403.5 | 686.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | task_exec_span_us | 146.8 | 291.5 | 404.0 | 686.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | numa0_span_us | 124.2 | 269.5 | 344.5 | 524.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | numa1_span_us | 124.8 | 264.5 | 343.0 | 567.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | numa_completion_skew_us | 3.8 | 16.5 | 42.5 | 324.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | producer_end_to_pub_us | 94.8 | 120.5 | 127.0 | 212.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | pub_to_h2d_start_us | 95.8 | 106.2 | 109.0 | 111.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | overlap_budget_us | 340.0 | 357.5 | 417.2 | 677.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | n_cold_assignments | 1.0 | 3.0 | 4.0 | 10.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 3006 | go_minus_dtoh_end_us | 4.5 | 5.5 | 7.5 | 384.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | cold_pub_delta_us | -85.2 | -62.5 | -58.5 | 136.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 136.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | gpu_pre_h2d_gap_us | 10.8 | 14.0 | 14.8 | 148.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | gpu_idle_inside_gap_us | 10.8 | 14.0 | 14.8 | 148.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | h2d_duration_us | 4.0 | 5.0 | 5.8 | 8.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | gpu_post_h2d_gap_us | 5.5 | 9.8 | 10.0 | 11.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | combine_schedule_gap_us | 5.5 | 9.8 | 10.0 | 11.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | go_to_enqueue_us | 3.0 | 4.2 | 5.2 | 360.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | enqueue_to_start_us | 3.8 | 11.5 | 58.8 | 519.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | predecessor_exec_union_us | 0.0 | 0.5 | 56.8 | 518.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | predecessor_deferred_exec_us | 0.0 | 0.0 | 56.5 | 518.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | queue_gap_unattributed_us | 3.5 | 6.5 | 10.8 | 58.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | deferred_service_span_us | 29.2 | 36.0 | 45.5 | 476.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | task_exec_span_us | 29.8 | 36.2 | 45.8 | 477.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | numa0_span_us | 12.5 | 15.0 | 19.0 | 33.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | numa1_span_us | 9.5 | 13.0 | 16.8 | 149.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | numa_completion_skew_us | 2.2 | 5.2 | 8.0 | 451.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | producer_end_to_pub_us | 217.8 | 233.8 | 239.5 | 270.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | pub_to_h2d_start_us | 96.0 | 106.5 | 109.5 | 118.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | overlap_budget_us | 341.8 | 360.2 | 367.2 | 747.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 4802 | go_minus_dtoh_end_us | 4.5 | 5.5 | 6.0 | 105.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_delta_us | -60.0 | -32.8 | -18.8 | -18.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_pre_h2d_gap_us | 11.5 | 14.2 | 14.8 | 15.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_idle_inside_gap_us | 11.5 | 14.2 | 14.8 | 15.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | h2d_duration_us | 4.0 | 5.2 | 5.5 | 5.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_post_h2d_gap_us | 6.5 | 9.5 | 9.8 | 9.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | combine_schedule_gap_us | 6.5 | 9.5 | 9.8 | 9.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | pub_to_h2d_start_us | 71.8 | 80.5 | 83.5 | 84.5 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | go_minus_dtoh_end_us | 4.5 | 5.5 | 6.2 | 7.2 |

민감도 (late share, cold_present_nonempty): {"2:cold_present_nonempty": {"late_share_thr_0us": 0.911, "late_share_thr_1us": 0.91, "late_share_thr_2us": 0.909, "late_share_thr_5us": 0.907, "clock_indeterminate_share_5us": 0.007, "n": 15541, "cold_gpu_ready_late_share": 1.0}, "32:cold_present_nonempty": {"late_share_thr_0us": 0.039, "late_share_thr_1us": 0.038, "late_share_thr_2us": 0.037, "late_share_thr_5us": 0.036, "clock_indeterminate_share_5us": 0.006, "n": 3006, "cold_gpu_ready_late_share": 1.0}}

### S26_C1_CORR_vllmw — coverage {'gpu_layer_rows_total': 126976, 'method:count_match_zip': 126976} · clock {'OK': 126976}

| graph:cohort | n | 지표 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | cold_pub_delta_us | -62.8 | -51.8 | 32.0 | 1279.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | cold_pub_lateness_us | 0.0 | 0.0 | 32.0 | 1279.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | gpu_pre_h2d_gap_us | 10.8 | 14.0 | 46.2 | 1294.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | gpu_idle_inside_gap_us | 10.8 | 14.0 | 46.2 | 1294.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | h2d_duration_us | 3.0 | 4.5 | 5.0 | 8.5 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | gpu_post_h2d_gap_us | 6.0 | 9.8 | 9.8 | 11.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | combine_schedule_gap_us | 6.0 | 9.8 | 9.8 | 11.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | go_to_enqueue_us | 2.8 | 3.8 | 5.0 | 224.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | enqueue_to_start_us | 3.8 | 8.2 | 68.0 | 1550.8 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | predecessor_exec_union_us | 0.0 | 0.2 | 62.8 | 1262.8 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | predecessor_deferred_exec_us | 0.0 | 0.0 | 62.5 | 1262.8 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | queue_gap_unattributed_us | 3.5 | 6.2 | 11.5 | 1527.5 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | deferred_service_span_us | 134.0 | 201.2 | 295.5 | 1526.8 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | task_exec_span_us | 134.2 | 201.8 | 296.0 | 1527.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | numa0_span_us | 122.2 | 188.8 | 279.5 | 1432.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | numa1_span_us | 118.5 | 179.8 | 272.5 | 1516.8 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | numa_completion_skew_us | 4.5 | 17.8 | 52.8 | 1381.5 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | producer_end_to_pub_us | 93.5 | 110.5 | 119.2 | 1386.5 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | pub_to_h2d_start_us | 73.2 | 79.5 | 81.2 | 85.2 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | overlap_budget_us | 298.5 | 309.8 | 315.0 | 1798.8 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | n_cold_assignments | 1.0 | 2.0 | 3.0 | 5.0 |
| 35:cold_present_nonempty ['step[DECODE bs=1]'] | 20646 | go_minus_dtoh_end_us | 4.2 | 5.2 | 6.5 | 1342.5 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | cold_pub_delta_us | -63.2 | -57.5 | -50.0 | 2559.5 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 2559.5 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | gpu_pre_h2d_gap_us | 10.8 | 13.8 | 14.8 | 2573.5 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | gpu_idle_inside_gap_us | 10.5 | 13.8 | 14.8 | 2573.5 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | h2d_duration_us | 3.0 | 4.0 | 4.5 | 9.8 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | gpu_post_h2d_gap_us | 6.0 | 9.8 | 10.0 | 13.0 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | combine_schedule_gap_us | 6.0 | 9.8 | 10.0 | 13.0 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | go_to_enqueue_us | 2.8 | 3.8 | 5.2 | 409.8 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | enqueue_to_start_us | 3.8 | 7.5 | 25.2 | 2607.2 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | predecessor_exec_union_us | 0.0 | 0.2 | 20.0 | 1303.8 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | predecessor_deferred_exec_us | 0.0 | 0.0 | 0.0 | 1303.5 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | queue_gap_unattributed_us | 3.5 | 6.2 | 11.8 | 2606.8 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | deferred_service_span_us | 20.2 | 24.0 | 35.5 | 604.8 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | task_exec_span_us | 20.8 | 24.2 | 36.0 | 605.0 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | numa0_span_us | 5.8 | 8.0 | 11.0 | 577.2 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | numa1_span_us | 7.5 | 9.2 | 14.0 | 326.5 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | numa_completion_skew_us | 3.0 | 5.2 | 9.0 | 589.2 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | producer_end_to_pub_us | 210.8 | 222.0 | 227.2 | 2679.8 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | pub_to_h2d_start_us | 73.5 | 79.8 | 81.5 | 86.2 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | overlap_budget_us | 301.8 | 312.5 | 316.0 | 2861.2 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 35:cold_path_empty ['step[DECODE bs=1]'] | 104282 | go_minus_dtoh_end_us | 4.2 | 5.2 | 6.5 | 1223.8 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | cold_pub_delta_us | -43.0 | -30.5 | -19.2 | 1513.8 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 1513.8 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | gpu_pre_h2d_gap_us | 11.2 | 14.2 | 15.5 | 1524.0 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | gpu_idle_inside_gap_us | 11.2 | 14.2 | 15.5 | 1524.0 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | h2d_duration_us | 3.0 | 4.2 | 5.0 | 6.0 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | gpu_post_h2d_gap_us | 6.0 | 9.5 | 9.8 | 10.8 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | combine_schedule_gap_us | 6.0 | 9.5 | 9.8 | 10.8 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | pub_to_h2d_start_us | 53.5 | 61.0 | 63.2 | 65.0 |
| 35:no_cold_consumed ['step[DECODE bs=1]'] | 2048 | go_minus_dtoh_end_us | 4.2 | 5.2 | 6.5 | 23.5 |

민감도 (late share, cold_present_nonempty): {"35:cold_present_nonempty": {"late_share_thr_0us": 0.018, "late_share_thr_1us": 0.017, "late_share_thr_2us": 0.017, "late_share_thr_5us": 0.016, "clock_indeterminate_share_5us": 0.006, "n": 20646, "cold_gpu_ready_late_share": 1.0}}

### S27_LONG_CORR_vllmw — coverage {'gpu_layer_rows_total': 31744, 'count_mismatch_pairs': 12, 'count_mismatch_rows': 12, 'resync_gpu_rows_without_dtoh': 1, 'resync_unmatched_cpu': 56, 'resync_pairs': 31688, 'method:ANCHOR_RESYNC(go>=dtoh_end<next)': 6138, 'resync_unmatched_gpu': 55, 'zip_order_violation_groups': 50, 'method:ANCHOR_RESYNC(order_violation)': 25550} · clock {'OK': 30618, 'CLOCK_INDETERMINATE': 1070}

| graph:cohort | n | 지표 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | cold_pub_delta_us | -180.8 | -49.5 | 122.2 | 1939.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | cold_pub_lateness_us | 0.0 | 0.0 | 122.2 | 1939.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | gpu_pre_h2d_gap_us | 10.5 | 14.5 | 136.8 | 1947.8 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | gpu_idle_inside_gap_us | 10.5 | 14.5 | 136.8 | 1947.8 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | h2d_duration_us | 8.8 | 11.5 | 15.8 | 23.2 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | gpu_post_h2d_gap_us | 6.0 | 9.5 | 10.0 | 11.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | combine_schedule_gap_us | 6.0 | 9.5 | 10.0 | 11.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | go_to_enqueue_us | 3.2 | 4.8 | 6.5 | 650.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | enqueue_to_start_us | 3.8 | 132.8 | 311.8 | 1982.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | predecessor_exec_union_us | 0.0 | 131.5 | 309.8 | 1978.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | predecessor_deferred_exec_us | 0.0 | 131.2 | 307.5 | 1978.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | queue_gap_unattributed_us | 3.2 | 6.8 | 10.2 | 660.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | deferred_service_span_us | 245.8 | 482.5 | 594.8 | 2326.8 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | task_exec_span_us | 246.0 | 483.0 | 595.0 | 2327.2 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | numa0_span_us | 189.2 | 423.2 | 528.8 | 1819.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | numa1_span_us | 182.5 | 414.2 | 515.5 | 2221.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | numa_completion_skew_us | 3.2 | 31.8 | 78.5 | 1708.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | producer_end_to_pub_us | 132.8 | 222.8 | 250.8 | 3223849.8 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | pub_to_h2d_start_us | 191.0 | 226.8 | 242.8 | 299.5 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | overlap_budget_us | 585.8 | 654.8 | 821.0 | 3224379.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | n_cold_assignments | 2.0 | 6.0 | 8.0 | 20.0 |
| 26:cold_present_nonempty ['step[DECODE bs=8]'] | 23657 | go_minus_dtoh_end_us | 4.5 | 6.0 | 8.0 | 523.5 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | cold_pub_delta_us | -175.8 | -106.2 | -87.0 | -73.8 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | gpu_pre_h2d_gap_us | 10.2 | 14.0 | 14.8 | 22.5 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | gpu_idle_inside_gap_us | 10.2 | 14.0 | 14.8 | 22.5 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | h2d_duration_us | 8.5 | 11.2 | 16.0 | 23.8 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | gpu_post_h2d_gap_us | 6.0 | 9.5 | 10.0 | 11.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | combine_schedule_gap_us | 6.0 | 9.5 | 10.0 | 11.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | go_to_enqueue_us | 3.2 | 4.8 | 6.5 | 29.2 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | enqueue_to_start_us | 3.8 | 74.8 | 157.0 | 2475.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | predecessor_exec_union_us | 0.0 | 71.5 | 156.0 | 2473.2 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | predecessor_deferred_exec_us | 0.0 | 13.8 | 154.8 | 2391.5 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | queue_gap_unattributed_us | 3.5 | 7.0 | 11.0 | 55.8 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | deferred_service_span_us | 74.2 | 84.8 | 95.2 | 368.2 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | task_exec_span_us | 74.5 | 85.0 | 95.8 | 368.2 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | numa0_span_us | 24.0 | 27.8 | 36.5 | 311.8 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | numa1_span_us | 22.5 | 24.8 | 32.2 | 318.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | numa_completion_skew_us | 1.0 | 4.0 | 7.0 | 292.2 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | producer_end_to_pub_us | 305.0 | 343.8 | 364.0 | 25308.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | pub_to_h2d_start_us | 186.2 | 221.8 | 238.8 | 290.2 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | overlap_budget_us | 568.5 | 640.0 | 689.2 | 25573.8 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 26:cold_path_empty ['step[DECODE bs=8]'] | 7519 | go_minus_dtoh_end_us | 4.5 | 5.8 | 8.8 | 26.8 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | cold_pub_delta_us | -151.0 | -103.2 | 6.8 | 2376.2 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | cold_pub_lateness_us | 0.0 | 0.0 | 6.8 | 2376.2 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | gpu_pre_h2d_gap_us | 10.5 | 14.0 | 24.0 | 2396.8 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | gpu_idle_inside_gap_us | 10.5 | 14.0 | 24.0 | 2388.2 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | h2d_duration_us | 8.5 | 11.2 | 17.2 | 19.2 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | gpu_post_h2d_gap_us | 6.2 | 9.5 | 9.8 | 10.2 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | combine_schedule_gap_us | 6.2 | 9.5 | 9.8 | 10.2 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | pub_to_h2d_start_us | 161.0 | 217.0 | 224.2 | 230.2 |
| 26:no_cold_consumed ['step[DECODE bs=8]'] | 512 | go_minus_dtoh_end_us | 4.5 | 5.8 | 8.8 | 17.0 |

민감도 (late share, cold_present_nonempty): {"26:cold_present_nonempty": {"late_share_thr_0us": 0.03, "late_share_thr_1us": 0.03, "late_share_thr_2us": 0.03, "late_share_thr_5us": 0.029, "clock_indeterminate_share_5us": 0.003, "n": 23657, "cold_gpu_ready_late_share": 1.0}}

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

### S29_CORR_vllmw_fp8fixso — coverage {'gpu_layer_rows_total': 23870, 'method:count_match_zip': 23808, 'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} · clock {'OK': 23808}

| graph:cohort | n | 지표 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | cold_pub_delta_us | 335.0 | 822.0 | 1062.0 | 7087.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | cold_pub_lateness_us | 335.0 | 822.0 | 1062.0 | 7087.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | gpu_pre_h2d_gap_us | 348.5 | 833.0 | 1073.0 | 7101.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | gpu_idle_inside_gap_us | 348.5 | 832.0 | 1073.0 | 7101.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | h2d_duration_us | 26.8 | 32.2 | 33.8 | 39.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | gpu_post_h2d_gap_us | 7.2 | 9.5 | 9.8 | 11.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | combine_schedule_gap_us | 7.2 | 9.5 | 9.8 | 11.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | go_to_enqueue_us | 3.8 | 6.8 | 9.0 | 705.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | enqueue_to_start_us | 673.0 | 1186.0 | 1413.8 | 7436.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | predecessor_exec_union_us | 672.0 | 1185.2 | 1413.0 | 7435.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | predecessor_deferred_exec_us | 671.8 | 1185.2 | 1413.0 | 7435.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | queue_gap_unattributed_us | 0.8 | 1.5 | 4.5 | 30.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | deferred_service_span_us | 921.0 | 1420.8 | 1653.0 | 7665.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | task_exec_span_us | 921.5 | 1421.0 | 1653.2 | 7665.8 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | numa0_span_us | 887.0 | 1382.5 | 1613.2 | 7624.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | numa1_span_us | 847.0 | 1323.8 | 1530.0 | 2646.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | numa_completion_skew_us | 34.8 | 87.5 | 165.8 | 5890.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | producer_end_to_pub_us | 0.5 | 1.0 | 9.0 | 1141.5 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | pub_to_h2d_start_us | 13.2 | 91.0 | 349.5 | 533.2 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | overlap_budget_us | 1268.2 | 1783.2 | 2017.2 | 8025.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | n_cold_assignments | 17.0 | 32.0 | 41.0 | 77.0 |
| 2:cold_present_nonempty ['step[DECODE bs=63]'] | 15600 | go_minus_dtoh_end_us | 5.5 | 6.8 | 8.2 | 1672.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | cold_pub_delta_us | -304.2 | -151.8 | -132.5 | -132.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | gpu_pre_h2d_gap_us | 10.0 | 12.0 | 13.0 | 13.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | gpu_idle_inside_gap_us | 10.0 | 12.0 | 13.0 | 13.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | h2d_duration_us | 19.8 | 26.0 | 30.0 | 30.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | gpu_post_h2d_gap_us | 6.0 | 8.8 | 9.0 | 9.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | combine_schedule_gap_us | 6.0 | 8.8 | 9.0 | 9.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | go_to_enqueue_us | 7.8 | 14.0 | 14.2 | 14.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | enqueue_to_start_us | 84.0 | 730.2 | 1400.8 | 1400.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | predecessor_exec_union_us | 77.5 | 210.8 | 1398.5 | 1398.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | predecessor_deferred_exec_us | 0.0 | 147.0 | 1304.2 | 1304.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | queue_gap_unattributed_us | 5.8 | 13.0 | 625.8 | 625.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | deferred_service_span_us | 73.5 | 86.8 | 87.0 | 87.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | task_exec_span_us | 73.5 | 87.2 | 87.2 | 87.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | numa0_span_us | 37.0 | 46.0 | 47.5 | 47.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | numa1_span_us | 37.2 | 45.5 | 47.5 | 47.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | numa_completion_skew_us | 2.2 | 4.5 | 5.8 | 5.8 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | producer_end_to_pub_us | 470.0 | 506.5 | 510.5 | 510.5 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | pub_to_h2d_start_us | 314.2 | 408.5 | 417.2 | 417.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | overlap_budget_us | 995.0 | 1128.8 | 1802.2 | 1802.2 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cold_path_empty ['step[DECODE bs=63]'] | 16 | go_minus_dtoh_end_us | 5.2 | 6.0 | 6.0 | 6.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_delta_us | -275.5 | 131.2 | 300.0 | 1249.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | cold_pub_lateness_us | 0.0 | 131.2 | 300.0 | 1249.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_pre_h2d_gap_us | 11.2 | 143.5 | 316.5 | 1266.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_idle_inside_gap_us | 11.2 | 143.5 | 316.5 | 1258.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | h2d_duration_us | 21.2 | 30.0 | 33.5 | 35.8 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | gpu_post_h2d_gap_us | 6.0 | 9.5 | 9.8 | 10.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | combine_schedule_gap_us | 6.0 | 9.5 | 9.8 | 10.0 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | pub_to_h2d_start_us | 288.2 | 483.2 | 511.0 | 561.2 |
| 2:no_cold_consumed ['step[DECODE bs=63]'] | 256 | go_minus_dtoh_end_us | 5.5 | 6.8 | 7.2 | 10.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | cold_pub_delta_us | -83.8 | -61.2 | 18.5 | 1146.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | cold_pub_lateness_us | 0.0 | 0.0 | 18.5 | 1146.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | gpu_pre_h2d_gap_us | 10.8 | 14.2 | 32.8 | 1159.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | gpu_idle_inside_gap_us | 10.8 | 14.2 | 32.8 | 1159.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | h2d_duration_us | 4.0 | 5.0 | 5.8 | 6.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | gpu_post_h2d_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | combine_schedule_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | go_to_enqueue_us | 3.2 | 4.2 | 5.2 | 390.5 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | enqueue_to_start_us | 3.5 | 12.2 | 95.5 | 373.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | predecessor_exec_union_us | 0.0 | 0.5 | 85.8 | 372.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | predecessor_deferred_exec_us | 0.0 | 0.0 | 85.2 | 372.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | queue_gap_unattributed_us | 3.5 | 7.0 | 12.8 | 155.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | deferred_service_span_us | 142.5 | 220.0 | 337.5 | 1466.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | task_exec_span_us | 142.8 | 220.2 | 337.8 | 1466.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | numa0_span_us | 122.5 | 197.0 | 315.8 | 441.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | numa1_span_us | 122.2 | 191.5 | 310.0 | 1449.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | numa_completion_skew_us | 3.5 | 15.0 | 37.0 | 1328.2 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | producer_end_to_pub_us | 98.8 | 120.0 | 127.8 | 136.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | pub_to_h2d_start_us | 94.2 | 105.0 | 108.0 | 111.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | overlap_budget_us | 336.0 | 354.2 | 362.8 | 633.8 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | n_cold_assignments | 1.0 | 2.0 | 4.0 | 6.0 |
| 32:cold_present_nonempty ['step[DECODE bs=2]'] | 2273 | go_minus_dtoh_end_us | 4.5 | 5.8 | 6.5 | 128.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | cold_pub_delta_us | -83.5 | -69.5 | -64.0 | 1180.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 1180.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | gpu_pre_h2d_gap_us | 10.8 | 14.0 | 14.8 | 1197.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | gpu_idle_inside_gap_us | 10.8 | 14.0 | 14.8 | 1197.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | h2d_duration_us | 4.0 | 4.8 | 5.2 | 6.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | gpu_post_h2d_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | combine_schedule_gap_us | 5.5 | 9.5 | 10.0 | 11.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | go_to_enqueue_us | 3.0 | 4.0 | 5.2 | 22.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | enqueue_to_start_us | 3.8 | 8.5 | 41.2 | 1238.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | predecessor_exec_union_us | 0.0 | 0.2 | 34.8 | 1236.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | predecessor_deferred_exec_us | 0.0 | 0.0 | 3.8 | 1236.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | queue_gap_unattributed_us | 3.5 | 6.5 | 11.2 | 111.2 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | deferred_service_span_us | 29.2 | 35.5 | 44.5 | 610.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | task_exec_span_us | 29.5 | 36.0 | 45.0 | 610.5 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | numa0_span_us | 11.5 | 14.5 | 19.2 | 597.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | numa1_span_us | 10.2 | 13.8 | 18.5 | 301.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | numa_completion_skew_us | 3.0 | 6.0 | 8.0 | 552.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | producer_end_to_pub_us | 216.5 | 232.0 | 238.5 | 1434.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | pub_to_h2d_start_us | 94.2 | 105.0 | 108.2 | 111.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | overlap_budget_us | 338.5 | 356.8 | 365.2 | 1501.8 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | identity_residual_us | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | n_cold_assignments | 0.0 | 0.0 | 0.0 | 0.0 |
| 32:cold_path_empty ['step[DECODE bs=2]'] | 5535 | go_minus_dtoh_end_us | 4.5 | 5.8 | 7.0 | 1255.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_delta_us | -59.0 | -37.2 | -18.8 | 102.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | cold_pub_lateness_us | 0.0 | 0.0 | 0.0 | 102.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_pre_h2d_gap_us | 11.2 | 13.8 | 14.5 | 117.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_idle_inside_gap_us | 11.2 | 13.8 | 14.5 | 117.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | h2d_duration_us | 4.0 | 5.0 | 5.8 | 6.2 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | gpu_post_h2d_gap_us | 5.8 | 9.5 | 9.8 | 9.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | combine_schedule_gap_us | 5.8 | 9.5 | 9.8 | 9.8 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | pub_to_h2d_start_us | 70.5 | 78.8 | 81.0 | 81.0 |
| 32:no_cold_consumed ['step[DECODE bs=2]'] | 128 | go_minus_dtoh_end_us | 4.5 | 6.0 | 7.2 | 17.2 |

민감도 (late share, cold_present_nonempty): {"2:cold_present_nonempty": {"late_share_thr_0us": 0.911, "late_share_thr_1us": 0.91, "late_share_thr_2us": 0.91, "late_share_thr_5us": 0.908, "clock_indeterminate_share_5us": 0.006, "n": 15600, "cold_gpu_ready_late_share": 1.0}, "32:cold_present_nonempty": {"late_share_thr_0us": 0.017, "late_share_thr_1us": 0.017, "late_share_thr_2us": 0.017, "late_share_thr_5us": 0.016, "clock_indeterminate_share_5us": 0.002, "n": 2273, "cold_gpu_ready_late_share": 1.0}}

## 6. FIFO 대기 분해 (queue_wait_breakdown; 같은 사건 단위, µs)

| session | n | queue_wait_wall p50/p95 | predecessor_exec_union p50/p95 | pred_deferred p50 | pred_done_setter p50 | unattributed p50/p95/max | dequeue→exec p50 | enq bracket p50 |
|---|---|---|---|---|---|---|---|---|
| S10_CORR | 18554 | 1011.2/2441.8 | 1010.8/2440.2 | 1010.5 | 0.0 | 1.0/5.0/255.2 | 0.0 | 0.5 |
| S12_C1_CORR | 46635 | 4.0/166.0 | 0.2/164.5 | 0.0 | 0.0 | 3.5/5.8/884.2 | 0.0 | 0.3 |
| S13_LONG_CORR | 28213 | 196.5/856.2 | 195.2/855.2 | 195.2 | 0.0 | 1.2/5.5/210.2 | 0.0 | 0.5 |
| S9_CORR | 18602 | 1132.0/2521.5 | 1131.2/2520.0 | 1131.2 | 0.0 | 1.0/4.5/29.8 | 0.0 | 0.5 |
| S17_CORR_pinned | 18554 | 1027.2/2467.0 | 1026.0/2466.2 | 1026.0 | 0.0 | 1.0/4.8/270.2 | 0.0 | 0.6 |
| S18_CORR_pinned | 18554 | 1036.0/2490.5 | 1035.0/2488.0 | 1035.0 | 0.0 | 1.0/4.8/5446.8 | 0.0 | 0.6 |
| S22_CORR_vllmw | 17873 | 605.2/1164.2 | 604.2/1163.5 | 604.0 | 0.0 | 0.8/4.0/277.0 | 0.0 | 0.5 |
| S23_CORR_vllmw | 18547 | 598.5/1157.0 | 597.2/1155.5 | 596.8 | 0.0 | 1.2/4.5/68.0 | 0.0 | 0.8 |
| S26_C1_CORR_vllmw | 20646 | 3.8/8.5 | 0.0/0.2 | 0.0 | 0.0 | 3.8/6.5/1527.8 | 0.1 | 0.3 |
| S27_LONG_CORR_vllmw | 23657 | 4.0/133.0 | 0.0/131.8 | 0.0 | 0.0 | 3.5/6.8/660.2 | 0.0 | 0.4 |
| S3_CORR | 17818 | 607.0/1139.5 | 606.2/1138.5 | 605.8 | 0.0 | 0.8/4.2/1118.5 | 0.0 | 0.6 |
| S4_CORR | 17893 | 593.2/1162.2 | 592.5/1160.8 | 592.0 | 0.0 | 0.8/4.2/50.0 | 0.0 | 0.6 |
| S29_CORR_vllmw_fp8fixso | 17873 | 617.2/1164.0 | 616.5/1163.5 | 616.0 | 0.0 | 1.0/4.2/155.2 | 0.1 | 0.6 |

## 7. 생산층별 비용표 (producer_layer_costs.csv; 관측값, 한계 이득 아님)

### S10_CORR (bs=63 graph, 층별 p50, µs)

| producer L | n | cold assign | unique n0/n1 | max rows n0 | AMX/AVX n0 | enq→start | pred union | unattrib | service | numa0/1 | skew | pub delta | late share | pre_h2d gap | budget | stages n0 (prep,cpy,q_in,up_gate,act,q_down,down,weight,total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 133 | 1 | 1.0/1.0 | 1.0 | 0.0/3.0 | 513.2 | 512.2 | 1.2 | 197.0 | 170.2/160.8 | 9.2 | -407.8 | 0.000 | 10.0 | 1180.0 | 3,12,12,75,3,1,41,15,170 |
| 1 | 256 | 12 | 8.0/8.0 | 3.0 | 0.0/24.0 | 6.0 | 0.2 | 5.5 | 707.8 | 681.0/661.0 | 19.2 | -264.5 | 0.055 | 10.8 | 1000.8 | 7,16,20,366,16,5,229,17,680 |
| 10 | 256 | 25 | 14.0/14.0 | 6.0 | 0.0/42.0 | 782.5 | 781.5 | 0.8 | 1143.0 | 1117.0/1066.8 | 46.5 | 543.5 | 1.000 | 557.2 | 1380.2 | 7,22,35,610,23,7,378,19,1116 |
| 11 | 256 | 15 | 11.0/11.0 | 3.0 | 0.0/33.0 | 915.0 | 914.0 | 0.8 | 862.5 | 834.0/799.2 | 36.8 | 297.2 | 0.938 | 310.5 | 1485.5 | 6,17,25,457,19,6,279,16,833 |
| 12 | 256 | 34 | 13.0/13.0 | 15.0 | 0.0/39.0 | 626.2 | 625.0 | 0.8 | 1198.5 | 1171.2/1124.5 | 48.0 | 636.8 | 0.996 | 644.2 | 1209.0 | 8,22,60,618,24,10,402,17,1170 |
| 13 | 256 | 35 | 17.0/17.0 | 8.0 | 0.0/51.0 | 962.0 | 961.5 | 1.0 | 1395.0 | 1368.2/1308.5 | 48.5 | 816.8 | 1.000 | 832.8 | 1543.2 | 8,23,43,759,27,9,468,19,1367 |
| 14 | 256 | 32 | 16.0/16.0 | 10.0 | 0.0/48.0 | 1162.2 | 1161.2 | 0.8 | 1301.8 | 1269.0/1227.8 | 38.8 | 751.0 | 1.000 | 766.5 | 1712.2 | 7,23,45,691,26,9,444,18,1268 |
| 15 | 256 | 22 | 13.0/13.0 | 4.0 | 0.0/39.0 | 1062.0 | 1061.0 | 1.0 | 1045.8 | 1016.2/988.5 | 25.5 | 486.5 | 1.000 | 499.2 | 1628.8 | 7,20,31,551,22,7,345,18,1015 |
| 16 | 256 | 46 | 18.0/18.0 | 9.0 | 0.0/54.0 | 813.5 | 812.5 | 0.8 | 1470.5 | 1444.5/1416.5 | 34.0 | 925.0 | 1.000 | 934.0 | 1351.5 | 8,24,46,791,29,10,510,19,1443 |
| 17 | 256 | 23 | 14.0/14.0 | 5.0 | 0.0/42.0 | 1233.8 | 1232.8 | 0.8 | 1117.2 | 1092.8/1059.2 | 32.0 | 578.8 | 1.000 | 592.5 | 1785.0 | 8,22,31,603,24,7,378,19,1091 |
| 18 | 256 | 33 | 16.0/16.0 | 8.0 | 0.0/48.0 | 885.8 | 885.0 | 0.8 | 1282.5 | 1253.5/1218.5 | 27.8 | 661.5 | 1.000 | 672.0 | 1512.5 | 8,23,40,683,26,8,434,19,1252 |
| 19 | 256 | 23 | 11.0/11.0 | 7.0 | 0.0/33.0 | 1045.8 | 1045.0 | 0.8 | 987.2 | 958.0/931.2 | 24.0 | 436.2 | 0.980 | 448.8 | 1602.0 | 7,20,36,514,21,7,324,17,957 |
| 2 | 256 | 12 | 9.0/9.0 | 2.0 | 0.0/27.0 | 97.8 | 96.8 | 1.0 | 787.0 | 756.5/736.8 | 21.8 | -90.8 | 0.285 | 11.8 | 995.0 | 7,16,18,419,17,5,250,16,756 |
| 20 | 256 | 78 | 20.0/20.0 | 25.0 | 0.0/60.0 | 751.8 | 750.8 | 0.8 | 1928.5 | 1903.8/1864.8 | 33.5 | 1362.2 | 1.000 | 1378.2 | 1314.8 | 9,28,84,1027,33,15,676,20,1903 |
| 21 | 256 | 47 | 17.0/17.0 | 11.0 | 0.0/51.0 | 1694.2 | 1693.8 | 0.8 | 1499.5 | 1473.5/1431.8 | 31.2 | 946.2 | 1.000 | 958.8 | 2251.8 | 8,23,50,804,28,11,511,18,1473 |
| 22 | 256 | 56 | 17.0/17.0 | 10.0 | 0.0/51.0 | 1266.2 | 1265.5 | 0.8 | 1556.0 | 1532.0/1479.0 | 40.2 | 1065.0 | 1.000 | 1077.5 | 1772.5 | 9,23,50,838,29,11,544,18,1531 |
| 23 | 256 | 64 | 19.0/19.0 | 14.0 | 0.0/57.0 | 1325.2 | 1324.2 | 0.8 | 1711.0 | 1680.2/1640.0 | 41.0 | 1200.8 | 1.000 | 1210.2 | 1834.2 | 9,24,61,918,31,13,599,18,1679 |
| 24 | 256 | 58 | 18.0/18.0 | 11.0 | 0.0/54.0 | 1486.5 | 1485.5 | 0.8 | 1562.2 | 1538.2/1510.2 | 38.0 | 1066.2 | 1.000 | 1080.5 | 2018.5 | 8,23,52,830,30,12,554,18,1537 |
| 25 | 256 | 55 | 18.0/18.0 | 13.0 | 0.0/54.0 | 1324.0 | 1323.2 | 0.8 | 1589.8 | 1565.0/1506.0 | 39.8 | 1060.5 | 1.000 | 1069.5 | 1862.2 | 8,24,56,843,30,12,551,20,1558 |
| 26 | 256 | 76 | 20.0/20.0 | 15.0 | 0.0/60.0 | 1352.8 | 1352.0 | 0.8 | 1909.8 | 1884.5/1836.2 | 45.2 | 1368.5 | 0.996 | 1378.0 | 1909.8 | 9,25,65,1044,32,14,662,18,1883 |
| 27 | 256 | 94 | 26.0/26.0 | 15.0 | 0.0/78.0 | 1684.0 | 1683.5 | 0.8 | 2317.8 | 2292.2/2198.0 | 50.2 | 1783.0 | 1.000 | 1791.8 | 2201.0 | 10,26,70,1277,40,16,821,20,2291 |
| 28 | 256 | 123 | 29.0/29.0 | 18.0 | 0.0/87.0 | 2092.0 | 2091.0 | 1.0 | 2725.0 | 2700.2/2600.0 | 52.5 | 2177.5 | 1.000 | 2186.2 | 2648.0 | 10,27,76,1494,45,18,970,20,2699 |
| 29 | 256 | 78 | 21.0/21.0 | 12.0 | 0.0/63.0 | 2488.2 | 2487.0 | 0.8 | 1903.2 | 1877.2/1831.2 | 44.5 | 1371.0 | 0.996 | 1383.2 | 3009.0 | 9,24,61,1032,34,14,680,18,1876 |
| 3 | 256 | 12 | 9.0/9.0 | 3.0 | 0.0/27.0 | 309.8 | 308.8 | 1.0 | 707.8 | 682.0/651.5 | 21.8 | 14.0 | 0.527 | 28.5 | 1009.5 | 6,15,22,364,16,5,228,16,681 |
| 30 | 256 | 113 | 27.0/27.0 | 15.0 | 0.0/81.0 | 1669.0 | 1668.2 | 0.8 | 2492.8 | 2467.0/2402.0 | 51.8 | 1943.2 | 1.000 | 1957.0 | 2233.8 | 10,26,75,1381,41,17,877,19,2466 |
| 31 | 256 | 85 | 22.0/22.0 | 12.0 | 0.0/66.0 | 2263.2 | 2262.2 | 0.8 | 1982.0 | 1954.5/1906.5 | 49.2 | 1437.2 | 0.992 | 1452.2 | 2800.5 | 9,24,62,1085,35,14,711,18,1953 |
| 32 | 256 | 111 | 27.0/27.0 | 14.0 | 0.0/81.0 | 1751.0 | 1750.2 | 0.8 | 2450.2 | 2425.0/2335.2 | 54.2 | 1974.2 | 1.000 | 1988.5 | 2250.2 | 10,24,72,1341,42,15,889,19,2424 |
| 33 | 256 | 136 | 33.0/33.0 | 14.0 | 0.0/99.0 | 2220.5 | 2219.5 | 0.8 | 3033.2 | 3007.0/2934.5 | 72.8 | 2520.5 | 1.000 | 2537.8 | 2742.0 | 11,26,79,1708,50,18,1095,20,3006 |
| 34 | 256 | 99 | 29.0/29.0 | 15.0 | 0.0/87.0 | 2794.8 | 2793.5 | 1.0 | 2500.2 | 2473.5/2403.5 | 54.8 | 1955.2 | 1.000 | 1967.2 | 3339.5 | 10,26,70,1381,42,16,891,20,2472 |
| 35 | 256 | 100 | 26.0/26.0 | 14.0 | 0.0/78.0 | 2257.0 | 2256.5 | 0.8 | 2358.2 | 2329.8/2280.0 | 50.8 | 1830.5 | 0.996 | 1847.2 | 2801.2 | 10,25,69,1295,40,14,847,20,2329 |
| 36 | 256 | 102 | 27.0/27.0 | 13.0 | 0.0/81.0 | 2125.5 | 2123.8 | 1.0 | 2437.5 | 2412.0/2349.8 | 60.5 | 1898.0 | 0.996 | 1911.8 | 2670.2 | 10,25,68,1334,41,15,890,19,2411 |
| 37 | 256 | 82 | 27.0/27.0 | 12.0 | 0.0/81.0 | 2204.2 | 2203.5 | 1.0 | 2263.0 | 2237.0/2148.8 | 58.0 | 1718.5 | 1.000 | 1735.5 | 2752.2 | 9,24,65,1226,40,15,807,20,2236 |
| 38 | 256 | 91 | 26.0/26.0 | 13.0 | 0.0/78.0 | 2029.5 | 2028.8 | 0.8 | 2227.5 | 2202.2/2145.2 | 51.8 | 1636.2 | 1.000 | 1649.2 | 2629.5 | 10,25,64,1221,39,15,802,19,2201 |
| 39 | 256 | 62 | 21.0/21.0 | 9.0 | 0.0/63.0 | 1997.5 | 1996.8 | 0.8 | 1756.8 | 1731.5/1672.0 | 44.5 | 1183.2 | 0.988 | 1192.2 | 2593.0 | 8,22,51,950,33,12,615,18,1730 |
| 4 | 256 | 14 | 11.0/11.0 | 3.0 | 0.0/33.0 | 378.8 | 377.8 | 1.0 | 845.2 | 819.2/791.0 | 27.0 | 282.2 | 0.883 | 295.2 | 947.2 | 7,16,22,453,19,5,274,15,818 |
| 40 | 256 | 63 | 21.0/21.0 | 10.0 | 0.0/63.0 | 1523.8 | 1523.0 | 0.8 | 1735.5 | 1709.2/1657.0 | 44.0 | 1176.8 | 0.992 | 1188.2 | 2091.0 | 9,23,52,936,34,11,621,18,1708 |
| 41 | 256 | 94 | 22.0/22.0 | 15.0 | 0.0/66.0 | 1500.5 | 1499.2 | 0.8 | 2097.0 | 2072.8/2022.0 | 44.2 | 1567.0 | 0.996 | 1575.0 | 2058.8 | 9,24,69,1141,35,15,753,18,2071 |
| 42 | 256 | 52 | 18.0/18.0 | 8.0 | 0.0/54.0 | 1865.0 | 1864.0 | 0.8 | 1520.5 | 1497.0/1452.2 | 42.0 | 940.8 | 1.000 | 953.5 | 2456.2 | 8,23,46,821,29,10,531,19,1496 |
| 43 | 256 | 31 | 15.0/15.0 | 6.0 | 0.0/45.0 | 1286.5 | 1285.5 | 0.8 | 1199.8 | 1170.8/1124.5 | 31.2 | 652.8 | 0.996 | 667.8 | 1833.0 | 8,20,34,637,25,8,412,18,1170 |
| 44 | 256 | 84 | 23.0/23.0 | 13.0 | 0.0/69.0 | 970.0 | 969.2 | 0.8 | 2068.2 | 2040.8/1993.2 | 50.2 | 1483.2 | 1.000 | 1500.5 | 1568.2 | 9,24,65,1123,36,14,743,17,2039 |
| 45 | 256 | 64 | 21.0/21.0 | 10.0 | 0.0/63.0 | 1829.8 | 1829.0 | 1.0 | 1802.0 | 1770.0/1726.8 | 40.8 | 1175.0 | 0.996 | 1190.2 | 2465.8 | 8,24,53,984,33,12,634,18,1769 |
| 46 | 256 | 23 | 11.0/11.0 | 6.0 | 0.0/33.0 | 1568.2 | 1567.5 | 0.8 | 910.2 | 884.2/858.0 | 26.2 | 361.5 | 0.992 | 375.2 | 2127.2 | 7,20,32,479,20,7,302,16,883 |
| 47 | 256 | 62 | 25.0/25.0 | 8.0 | 0.0/75.0 | 671.0 | 670.0 | 0.8 | 1939.0 | 1911.8/1834.5 | 51.5 | 1308.5 | 1.000 | 1319.8 | 1312.0 | 9,24,49,1051,38,11,691,19,1910 |
| 48 | 256 | 37 | 17.0/17.0 | 6.0 | 0.0/51.0 | 1702.5 | 1702.0 | 0.8 | 1338.8 | 1314.0/1266.5 | 36.8 | 697.2 | 0.988 | 711.0 | 2353.0 | 8,21,36,717,27,8,464,17,1313 |
| 49 | 256 | 43 | 16.0/16.0 | 10.0 | 0.0/48.0 | 1103.2 | 1102.0 | 0.8 | 1395.0 | 1370.5/1314.8 | 36.8 | 799.5 | 0.977 | 812.8 | 1681.0 | 8,22,47,743,28,9,476,17,1369 |
| 5 | 256 | 29 | 15.0/15.0 | 8.0 | 0.0/45.0 | 577.8 | 577.0 | 0.8 | 1241.8 | 1208.0/1146.0 | 52.5 | 619.0 | 1.000 | 631.5 | 1206.8 | 8,21,40,664,26,8,424,17,1207 |
| 50 | 256 | 61 | 25.0/25.0 | 8.0 | 0.0/75.0 | 1161.8 | 1161.0 | 0.8 | 2025.2 | 1998.0/1934.0 | 51.0 | 1408.8 | 1.000 | 1420.0 | 1767.2 | 9,25,49,1108,38,11,708,20,1997 |
| 51 | 256 | 56 | 22.0/22.0 | 7.0 | 0.0/66.0 | 1783.0 | 1782.0 | 1.0 | 1737.2 | 1711.0/1667.0 | 47.5 | 1095.2 | 0.996 | 1107.2 | 2453.8 | 8,22,46,947,34,10,615,18,1710 |
| 52 | 256 | 31 | 17.0/17.0 | 5.0 | 0.0/51.0 | 1507.5 | 1506.8 | 0.8 | 1345.0 | 1318.5/1269.5 | 39.2 | 731.8 | 0.996 | 740.5 | 2134.2 | 8,23,33,719,28,8,466,18,1317 |
| 53 | 256 | 54 | 22.0/22.0 | 8.0 | 0.0/66.0 | 1109.0 | 1108.0 | 0.8 | 1705.0 | 1678.5/1637.5 | 45.2 | 1080.5 | 1.000 | 1093.2 | 1715.2 | 8,24,45,921,34,10,612,19,1677 |
| 54 | 256 | 32 | 17.0/17.0 | 5.0 | 0.0/51.0 | 1472.0 | 1471.2 | 0.8 | 1356.8 | 1332.0/1274.5 | 40.0 | 750.8 | 1.000 | 770.2 | 2083.5 | 8,21,35,724,28,8,472,18,1331 |
| 55 | 256 | 55 | 25.0/25.0 | 6.0 | 0.0/75.0 | 1119.2 | 1118.0 | 0.8 | 1913.2 | 1886.5/1833.5 | 50.5 | 1326.5 | 1.000 | 1336.8 | 1712.0 | 8,24,43,1046,38,10,682,19,1885 |
| 56 | 256 | 48 | 22.0/22.0 | 6.0 | 0.0/66.0 | 1680.5 | 1679.2 | 1.0 | 1682.0 | 1655.8/1604.0 | 43.5 | 1070.2 | 1.000 | 1081.2 | 2296.0 | 8,23,42,920,34,10,595,18,1654 |
| 57 | 256 | 38 | 20.0/20.0 | 4.0 | 0.0/60.0 | 1446.2 | 1444.8 | 1.0 | 1548.2 | 1522.8/1466.5 | 42.8 | 930.0 | 1.000 | 939.8 | 2051.8 | 8,21,36,838,31,9,544,18,1521 |
| 58 | 256 | 40 | 21.0/21.0 | 5.0 | 0.0/63.0 | 1316.5 | 1315.5 | 0.8 | 1563.5 | 1538.2/1476.5 | 44.8 | 958.8 | 1.000 | 970.8 | 1932.0 | 7,22,38,857,33,9,542,19,1537 |
| 59 | 256 | 38 | 19.0/19.0 | 5.0 | 0.0/57.0 | 1325.5 | 1325.0 | 0.8 | 1472.5 | 1444.8/1398.2 | 37.5 | 827.0 | 0.996 | 844.0 | 1966.2 | 7,21,37,798,30,9,513,17,1443 |
| 6 | 256 | 13 | 8.0/8.0 | 4.0 | 0.0/24.0 | 992.8 | 991.8 | 0.8 | 722.8 | 694.0/663.0 | 28.8 | 168.5 | 0.914 | 181.8 | 1559.5 | 7,18,25,377,16,5,224,16,693 |
| 60 | 256 | 40 | 18.0/18.0 | 6.0 | 0.0/54.0 | 1239.5 | 1238.2 | 0.8 | 1428.2 | 1398.2/1355.8 | 40.0 | 835.2 | 1.000 | 852.2 | 1824.2 | 8,23,42,773,29,9,494,19,1397 |
| 7 | 256 | 47 | 10.0/10.0 | 22.0 | 0.0/30.0 | 487.5 | 486.8 | 0.8 | 1212.0 | 1186.5/1137.2 | 45.0 | 653.2 | 0.992 | 663.5 | 1048.0 | 8,23,78,622,20,13,388,17,1185 |
| 8 | 256 | 56 | 16.0/16.0 | 18.0 | 0.0/48.0 | 978.2 | 977.8 | 0.8 | 1515.8 | 1490.0/1406.0 | 70.0 | 922.2 | 1.000 | 935.2 | 1556.0 | 9,22,67,814,27,13,517,17,1489 |
| 9 | 256 | 28 | 11.0/11.0 | 13.0 | 0.0/33.0 | 1281.2 | 1280.2 | 1.0 | 1012.8 | 981.8/938.8 | 46.8 | 510.5 | 0.996 | 521.5 | 1823.0 | 7,20,53,524,20,9,332,17,981 |

### S9_CORR (bs=63 graph, 층별 p50, µs)

| producer L | n | cold assign | unique n0/n1 | max rows n0 | AMX/AVX n0 | enq→start | pred union | unattrib | service | numa0/1 | skew | pub delta | late share | pre_h2d gap | budget | stages n0 (prep,cpy,q_in,up_gate,act,q_down,down,weight,total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 181 | 3 | 2.0/2.0 | 1.0 | 0.0/6.0 | 539.8 | 538.5 | 1.2 | 286.2 | 255.2/228.5 | 15.0 | -396.8 | 0.006 | 10.2 | 1207.0 | 3,13,18,123,6,3,82,15,254 |
| 1 | 256 | 16 | 10.0/10.0 | 3.0 | 0.0/30.0 | 6.8 | 0.2 | 4.2 | 810.5 | 784.8/750.2 | 28.8 | -161.8 | 0.262 | 11.5 | 1012.5 | 7,16,21,423,18,6,263,17,784 |
| 10 | 256 | 29 | 16.0/16.0 | 6.0 | 0.0/48.0 | 860.5 | 860.0 | 0.8 | 1290.2 | 1264.5/1201.0 | 54.0 | 694.0 | 1.000 | 706.2 | 1457.8 | 8,23,36,698,26,8,431,19,1263 |
| 11 | 256 | 18 | 12.0/12.0 | 3.0 | 0.0/36.0 | 1058.5 | 1057.8 | 0.8 | 982.0 | 955.8/907.8 | 40.2 | 420.5 | 0.965 | 434.0 | 1630.8 | 7,18,26,530,21,6,326,16,955 |
| 12 | 256 | 40 | 16.0/16.0 | 15.0 | 0.0/48.0 | 749.0 | 748.0 | 0.8 | 1384.5 | 1360.5/1287.2 | 57.0 | 815.0 | 0.996 | 828.2 | 1338.8 | 8,22,59,725,28,10,479,17,1358 |
| 13 | 256 | 37 | 18.0/18.0 | 8.0 | 0.0/54.0 | 1154.0 | 1153.5 | 0.8 | 1442.0 | 1415.5/1361.8 | 52.0 | 852.8 | 1.000 | 865.0 | 1729.5 | 8,22,43,787,29,10,492,19,1414 |
| 14 | 256 | 35 | 17.0/17.0 | 9.0 | 0.0/51.0 | 1211.8 | 1211.0 | 0.8 | 1360.5 | 1336.2/1283.0 | 49.5 | 808.8 | 1.000 | 822.2 | 1766.2 | 7,23,46,722,28,9,473,19,1335 |
| 15 | 256 | 27 | 15.0/15.0 | 4.0 | 0.0/45.0 | 1129.5 | 1128.2 | 0.8 | 1201.0 | 1175.2/1146.5 | 33.5 | 644.2 | 0.996 | 659.5 | 1691.8 | 8,21,32,648,25,8,411,18,1174 |
| 16 | 256 | 51 | 19.0/19.0 | 9.0 | 0.0/57.0 | 969.2 | 967.8 | 0.8 | 1614.8 | 1590.0/1531.2 | 43.8 | 1059.0 | 1.000 | 1071.5 | 1528.2 | 8,23,48,863,31,11,566,20,1589 |
| 17 | 256 | 28 | 16.0/16.0 | 5.0 | 0.0/48.0 | 1378.8 | 1378.2 | 0.8 | 1249.8 | 1223.8/1172.5 | 42.0 | 695.8 | 1.000 | 706.8 | 1929.8 | 8,22,32,664,26,8,427,19,1222 |
| 18 | 256 | 35 | 17.0/17.0 | 7.0 | 0.0/51.0 | 1009.2 | 1008.8 | 1.0 | 1339.8 | 1315.2/1264.5 | 38.2 | 715.0 | 1.000 | 728.0 | 1633.5 | 8,23,40,712,28,9,465,20,1314 |
| 19 | 256 | 26 | 13.0/13.0 | 7.0 | 0.0/39.0 | 1101.8 | 1101.0 | 0.8 | 1103.8 | 1072.2/1044.8 | 33.2 | 546.0 | 0.984 | 559.5 | 1663.8 | 7,20,37,589,22,8,360,17,1071 |
| 2 | 256 | 20 | 12.0/12.0 | 4.0 | 0.0/36.0 | 209.8 | 209.0 | 0.8 | 950.0 | 925.5/888.2 | 32.5 | 168.0 | 0.695 | 180.0 | 1015.2 | 7,16,24,505,21,6,323,17,924 |
| 20 | 256 | 79 | 21.0/21.0 | 24.0 | 0.0/63.0 | 872.5 | 871.5 | 0.8 | 2031.5 | 2000.2/1953.8 | 53.2 | 1472.0 | 1.000 | 1487.5 | 1439.8 | 9,27,82,1089,35,16,713,20,1999 |
| 21 | 256 | 51 | 18.0/18.0 | 10.0 | 0.0/54.0 | 1797.2 | 1796.8 | 0.8 | 1571.0 | 1543.5/1481.2 | 39.5 | 1012.5 | 0.996 | 1027.5 | 2354.0 | 8,22,50,848,30,11,542,18,1542 |
| 22 | 256 | 58 | 18.0/18.0 | 10.0 | 0.0/54.0 | 1332.8 | 1331.8 | 0.8 | 1641.2 | 1614.2/1565.5 | 50.8 | 1137.2 | 1.000 | 1152.5 | 1844.0 | 9,22,50,885,30,11,578,18,1613 |
| 23 | 256 | 71 | 21.0/21.0 | 14.0 | 0.0/63.0 | 1407.8 | 1406.8 | 0.8 | 1831.2 | 1807.0/1743.0 | 49.5 | 1299.2 | 1.000 | 1312.0 | 1937.8 | 9,25,64,988,34,13,647,19,1805 |
| 24 | 256 | 61 | 20.0/20.0 | 11.0 | 0.0/60.0 | 1598.8 | 1598.2 | 0.8 | 1707.5 | 1683.2/1614.0 | 46.5 | 1178.8 | 0.996 | 1193.2 | 2137.5 | 8,22,53,917,32,12,610,18,1682 |
| 25 | 256 | 57 | 19.0/19.0 | 12.0 | 0.0/57.0 | 1463.8 | 1463.2 | 0.8 | 1682.2 | 1653.5/1609.8 | 48.2 | 1144.0 | 1.000 | 1154.2 | 2042.5 | 8,24,55,908,32,12,585,20,1652 |
| 26 | 256 | 76 | 20.0/20.0 | 14.0 | 0.0/60.0 | 1454.5 | 1453.2 | 0.8 | 1953.0 | 1927.5/1871.5 | 51.5 | 1409.8 | 1.000 | 1421.5 | 2018.5 | 9,24,63,1064,34,14,682,18,1926 |
| 27 | 256 | 94 | 26.0/26.0 | 15.0 | 0.0/78.0 | 1725.8 | 1724.8 | 0.8 | 2377.8 | 2295.0/2242.8 | 58.5 | 1827.0 | 1.000 | 1840.5 | 2259.0 | 10,25,70,1273,41,16,843,20,2294 |
| 28 | 256 | 123 | 30.0/30.0 | 17.0 | 0.0/90.0 | 2139.8 | 2138.8 | 1.0 | 2755.5 | 2729.8/2650.2 | 67.0 | 2216.0 | 1.000 | 2227.8 | 2699.2 | 10,26,75,1507,46,17,1007,21,2728 |
| 29 | 256 | 79 | 21.0/21.0 | 12.0 | 0.0/63.0 | 2516.5 | 2516.0 | 0.8 | 1927.2 | 1901.0/1846.0 | 49.5 | 1393.8 | 0.996 | 1402.8 | 3066.8 | 9,23,61,1040,34,13,685,18,1900 |
| 3 | 256 | 18 | 11.0/11.0 | 3.0 | 0.0/33.0 | 573.8 | 573.0 | 0.8 | 865.5 | 841.5/806.5 | 35.5 | 293.0 | 0.691 | 305.0 | 1180.2 | 6,17,26,460,19,6,286,16,840 |
| 30 | 256 | 111 | 28.0/28.0 | 14.0 | 0.0/84.0 | 1693.5 | 1693.0 | 0.8 | 2537.0 | 2510.0/2441.5 | 63.2 | 1963.5 | 1.000 | 1974.2 | 2269.8 | 10,25,73,1396,42,16,905,19,2509 |
| 31 | 256 | 87 | 22.0/22.0 | 12.0 | 0.0/66.0 | 2293.0 | 2292.0 | 0.8 | 2068.8 | 2026.8/1966.8 | 55.0 | 1511.2 | 0.996 | 1526.0 | 2857.5 | 9,23,63,1119,36,13,749,17,2026 |
| 32 | 256 | 112 | 28.0/28.0 | 13.0 | 0.0/84.0 | 1840.8 | 1838.8 | 1.0 | 2503.5 | 2478.0/2401.5 | 60.2 | 2006.5 | 1.000 | 2019.8 | 2343.5 | 10,24,71,1376,43,15,912,19,2477 |
| 33 | 256 | 136 | 34.0/34.0 | 14.0 | 0.0/102.0 | 2266.0 | 2265.0 | 0.8 | 3126.2 | 3101.5/3009.0 | 76.2 | 2603.8 | 1.000 | 2618.5 | 2791.5 | 11,26,79,1752,51,17,1131,20,3100 |
| 34 | 256 | 100 | 29.0/29.0 | 14.0 | 0.0/87.0 | 2881.0 | 2880.2 | 0.8 | 2500.8 | 2474.8/2400.0 | 63.5 | 1972.5 | 1.000 | 1982.8 | 3443.5 | 10,25,68,1382,43,15,898,20,2473 |
| 35 | 256 | 101 | 27.0/27.0 | 13.0 | 0.0/81.0 | 2268.8 | 2268.2 | 0.8 | 2418.2 | 2392.2/2312.8 | 58.8 | 1879.5 | 0.996 | 1895.5 | 2810.5 | 10,24,68,1338,41,14,868,20,2391 |
| 36 | 256 | 102 | 28.0/28.0 | 13.0 | 0.0/84.0 | 2188.2 | 2187.5 | 0.8 | 2480.2 | 2455.5/2362.0 | 70.8 | 1943.2 | 0.996 | 1952.8 | 2728.2 | 10,24,68,1347,42,15,909,19,2454 |
| 37 | 256 | 82 | 28.0/28.0 | 12.0 | 0.0/84.0 | 2249.2 | 2248.8 | 0.8 | 2357.0 | 2324.5/2266.0 | 64.2 | 1812.8 | 1.000 | 1826.8 | 2804.2 | 10,23,61,1290,42,14,859,20,2323 |
| 38 | 256 | 93 | 27.0/27.0 | 12.0 | 0.0/81.0 | 2123.0 | 2122.0 | 0.8 | 2322.5 | 2297.5/2227.8 | 59.2 | 1718.0 | 1.000 | 1734.8 | 2736.5 | 10,24,63,1274,40,14,832,19,2296 |
| 39 | 256 | 65 | 22.0/22.0 | 9.0 | 0.0/66.0 | 2089.5 | 2088.5 | 0.8 | 1851.0 | 1818.0/1762.5 | 47.8 | 1255.2 | 1.000 | 1267.5 | 2680.8 | 8,21,51,1009,34,12,657,18,1817 |
| 4 | 256 | 23 | 14.0/14.0 | 3.0 | 0.0/42.0 | 624.0 | 623.5 | 0.8 | 1134.2 | 1110.8/1050.8 | 45.5 | 548.8 | 0.969 | 561.0 | 1165.8 | 8,17,28,618,24,7,382,16,1109 |
| 40 | 256 | 63 | 22.0/22.0 | 9.0 | 0.0/66.0 | 1610.2 | 1609.5 | 0.8 | 1795.8 | 1764.0/1712.0 | 55.8 | 1227.2 | 0.996 | 1237.8 | 2193.0 | 9,23,50,971,35,12,643,18,1763 |
| 41 | 256 | 96 | 22.0/22.0 | 14.0 | 0.0/66.0 | 1563.0 | 1562.2 | 0.8 | 2141.8 | 2117.0/2051.5 | 55.5 | 1587.8 | 1.000 | 1600.5 | 2115.2 | 9,23,69,1171,36,14,767,18,2116 |
| 42 | 256 | 53 | 19.0/19.0 | 8.0 | 0.0/57.0 | 1908.8 | 1907.8 | 0.8 | 1578.8 | 1548.8/1503.8 | 46.8 | 983.5 | 1.000 | 1001.0 | 2501.8 | 9,23,44,848,30,11,558,19,1548 |
| 43 | 256 | 32 | 16.0/16.0 | 6.0 | 0.0/48.0 | 1347.8 | 1347.0 | 0.8 | 1270.2 | 1238.8/1196.0 | 37.2 | 713.8 | 1.000 | 727.2 | 1904.2 | 8,20,36,679,26,8,435,18,1237 |
| 44 | 256 | 85 | 24.0/24.0 | 13.0 | 0.0/72.0 | 1033.5 | 1031.5 | 0.8 | 2103.2 | 2075.8/2024.2 | 55.8 | 1519.5 | 1.000 | 1538.2 | 1624.0 | 9,23,65,1145,37,14,768,17,2074 |
| 45 | 256 | 66 | 23.0/23.0 | 10.0 | 0.0/69.0 | 1871.2 | 1870.5 | 0.8 | 1876.0 | 1847.5/1796.2 | 51.8 | 1264.8 | 1.000 | 1271.5 | 2514.8 | 8,23,52,1025,35,12,666,18,1846 |
| 46 | 256 | 24 | 12.0/12.0 | 5.0 | 0.0/36.0 | 1646.5 | 1646.0 | 0.8 | 1011.0 | 981.5/945.8 | 33.2 | 453.8 | 0.992 | 462.8 | 2209.8 | 7,20,32,524,21,7,338,16,980 |
| 47 | 256 | 66 | 26.0/26.0 | 8.0 | 0.0/78.0 | 774.0 | 773.0 | 0.8 | 2043.5 | 2015.2/1952.8 | 61.8 | 1397.5 | 1.000 | 1408.5 | 1420.2 | 9,24,49,1119,40,11,737,19,2014 |
| 48 | 256 | 40 | 18.0/18.0 | 6.0 | 0.0/54.0 | 1805.8 | 1805.0 | 0.8 | 1425.2 | 1400.8/1349.5 | 45.5 | 780.2 | 1.000 | 796.2 | 2461.8 | 8,21,38,767,29,9,501,17,1400 |
| 49 | 256 | 44 | 18.0/18.0 | 9.0 | 0.0/54.0 | 1189.5 | 1188.2 | 0.8 | 1511.8 | 1485.5/1431.8 | 44.8 | 921.2 | 1.000 | 934.8 | 1775.8 | 9,21,46,815,29,10,519,17,1484 |
| 5 | 256 | 36 | 19.0/19.0 | 8.0 | 0.0/57.0 | 859.8 | 859.0 | 0.8 | 1458.8 | 1435.5/1370.0 | 83.8 | 849.0 | 1.000 | 862.0 | 1492.0 | 8,21,42,804,30,9,507,17,1433 |
| 50 | 256 | 66 | 27.0/27.0 | 8.0 | 0.0/81.0 | 1281.2 | 1279.8 | 0.8 | 2101.8 | 2073.8/2002.0 | 60.2 | 1477.8 | 1.000 | 1491.0 | 1896.0 | 9,24,49,1151,40,11,751,20,2071 |
| 51 | 256 | 56 | 23.0/23.0 | 7.0 | 0.0/69.0 | 1860.8 | 1860.0 | 0.8 | 1796.2 | 1766.5/1707.5 | 53.5 | 1137.5 | 1.000 | 1151.2 | 2519.5 | 8,22,44,979,36,10,636,18,1765 |
| 52 | 256 | 32 | 18.0/18.0 | 5.0 | 0.0/54.0 | 1569.0 | 1567.8 | 0.8 | 1365.8 | 1332.8/1274.0 | 45.5 | 748.2 | 1.000 | 761.5 | 2185.8 | 8,22,32,735,28,8,471,18,1331 |
| 53 | 256 | 55 | 23.0/23.0 | 8.0 | 0.0/69.0 | 1128.5 | 1128.0 | 0.8 | 1768.8 | 1741.5/1684.8 | 51.8 | 1165.5 | 1.000 | 1180.0 | 1741.5 | 8,24,45,961,35,10,635,19,1739 |
| 54 | 256 | 36 | 19.0/19.0 | 5.0 | 0.0/57.0 | 1539.8 | 1539.0 | 0.8 | 1453.5 | 1426.8/1367.0 | 49.0 | 863.8 | 1.000 | 877.8 | 2140.2 | 8,21,35,786,30,9,514,18,1425 |
| 55 | 256 | 57 | 28.0/28.0 | 6.0 | 0.0/84.0 | 1217.0 | 1216.5 | 0.8 | 2070.0 | 2032.2/1969.0 | 64.0 | 1480.0 | 1.000 | 1493.2 | 1814.2 | 9,23,43,1135,41,11,734,19,2031 |
| 56 | 256 | 52 | 24.0/24.0 | 6.0 | 0.0/72.0 | 1841.5 | 1840.0 | 1.0 | 1822.5 | 1791.0/1732.5 | 57.5 | 1206.8 | 1.000 | 1221.2 | 2453.8 | 9,23,43,991,37,10,654,18,1788 |
| 57 | 256 | 43 | 23.0/23.0 | 5.0 | 0.0/69.0 | 1583.5 | 1583.0 | 0.8 | 1709.8 | 1677.5/1618.8 | 54.5 | 1075.0 | 1.000 | 1089.5 | 2215.5 | 8,21,39,932,34,9,606,18,1676 |
| 58 | 256 | 43 | 23.0/23.0 | 5.0 | 0.0/69.0 | 1478.5 | 1477.0 | 0.8 | 1697.2 | 1671.0/1610.0 | 50.8 | 1089.0 | 1.000 | 1102.2 | 2098.8 | 8,23,38,933,35,10,601,19,1670 |
| 59 | 256 | 41 | 21.0/21.0 | 5.0 | 0.0/63.0 | 1461.5 | 1460.8 | 0.8 | 1581.2 | 1550.5/1489.8 | 51.0 | 936.8 | 1.000 | 949.5 | 2114.8 | 8,21,37,860,32,9,553,17,1549 |
| 6 | 256 | 16 | 10.0/10.0 | 4.0 | 0.0/30.0 | 1227.2 | 1226.8 | 0.8 | 869.5 | 842.2/797.2 | 48.0 | 302.5 | 0.926 | 318.8 | 1795.8 | 7,18,25,472,19,6,277,16,841 |
| 60 | 256 | 43 | 19.0/19.0 | 6.0 | 0.0/57.0 | 1348.0 | 1347.2 | 1.0 | 1548.2 | 1521.8/1455.8 | 47.0 | 945.5 | 1.000 | 960.8 | 1950.0 | 8,23,41,830,30,9,537,19,1520 |
| 7 | 256 | 48 | 12.0/12.0 | 22.0 | 0.0/36.0 | 637.2 | 636.0 | 0.8 | 1277.8 | 1253.8/1168.5 | 61.8 | 707.8 | 0.996 | 722.0 | 1198.0 | 8,23,77,660,22,14,417,17,1253 |
| 8 | 256 | 61 | 18.0/18.0 | 18.0 | 0.0/54.0 | 1047.5 | 1046.8 | 0.8 | 1650.8 | 1618.5/1511.0 | 81.0 | 1058.2 | 1.000 | 1068.5 | 1634.2 | 9,23,64,885,29,13,563,17,1617 |
| 9 | 256 | 28 | 12.0/12.0 | 13.0 | 0.0/36.0 | 1415.5 | 1414.5 | 0.8 | 1092.2 | 1068.5/983.5 | 51.5 | 556.5 | 0.996 | 570.5 | 1952.2 | 7,21,51,580,22,9,355,18,1067 |

### S17_CORR_pinned (bs=63 graph, 층별 p50, µs)

| producer L | n | cold assign | unique n0/n1 | max rows n0 | AMX/AVX n0 | enq→start | pred union | unattrib | service | numa0/1 | skew | pub delta | late share | pre_h2d gap | budget | stages n0 (prep,cpy,q_in,up_gate,act,q_down,down,weight,total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 133 | 1 | 1.0/1.0 | 1.0 | 0.0/3.0 | 524.5 | 523.2 | 1.2 | 200.2 | 169.8/162.5 | 7.2 | -404.5 | 0.000 | 9.8 | 1184.0 | 3,11,11,76,3,1,41,15,169 |
| 1 | 256 | 12 | 8.0/8.0 | 3.0 | 0.0/24.0 | 6.8 | 0.2 | 6.2 | 713.0 | 682.0/663.0 | 22.5 | -250.8 | 0.059 | 10.5 | 998.8 | 7,15,19,371,15,5,230,17,681 |
| 10 | 256 | 25 | 14.0/14.0 | 6.0 | 0.0/42.0 | 808.0 | 807.5 | 1.0 | 1162.5 | 1133.0/1090.2 | 46.2 | 582.2 | 1.000 | 594.5 | 1403.8 | 8,21,35,626,22,7,382,18,1132 |
| 11 | 256 | 15 | 11.0/11.0 | 3.0 | 0.0/33.0 | 926.8 | 926.0 | 1.0 | 863.5 | 837.2/809.5 | 32.8 | 316.2 | 0.938 | 327.0 | 1503.5 | 6,16,25,461,18,6,289,16,836 |
| 12 | 256 | 34 | 13.0/13.0 | 15.0 | 0.0/39.0 | 632.0 | 630.5 | 1.0 | 1216.0 | 1161.8/1163.2 | 43.2 | 642.8 | 0.996 | 657.8 | 1214.0 | 8,20,58,620,22,10,402,17,1161 |
| 13 | 256 | 35 | 17.0/17.0 | 8.0 | 0.0/51.0 | 983.5 | 982.5 | 1.0 | 1419.0 | 1386.5/1347.0 | 41.2 | 843.0 | 1.000 | 853.8 | 1565.5 | 8,21,43,769,26,9,473,18,1385 |
| 14 | 256 | 32 | 16.0/16.0 | 10.0 | 0.0/48.0 | 1188.8 | 1187.2 | 1.0 | 1302.8 | 1275.2/1227.2 | 39.2 | 767.2 | 1.000 | 778.8 | 1734.5 | 7,21,45,691,25,9,448,18,1274 |
| 15 | 256 | 22 | 13.0/13.0 | 4.0 | 0.0/39.0 | 1068.5 | 1067.2 | 1.0 | 1052.8 | 1020.8/998.5 | 25.2 | 490.0 | 0.992 | 503.8 | 1633.8 | 7,19,30,562,21,7,351,17,1020 |
| 16 | 256 | 46 | 18.0/18.0 | 9.0 | 0.0/54.0 | 815.8 | 814.2 | 1.0 | 1519.2 | 1470.0/1433.2 | 37.8 | 959.5 | 1.000 | 969.8 | 1365.2 | 7,22,46,808,28,10,516,19,1467 |
| 17 | 256 | 23 | 14.0/14.0 | 5.0 | 0.0/42.0 | 1283.2 | 1282.5 | 1.0 | 1146.8 | 1116.0/1078.2 | 33.5 | 594.8 | 1.000 | 606.0 | 1842.8 | 8,20,32,608,23,7,381,18,1115 |
| 18 | 256 | 33 | 16.0/16.0 | 8.0 | 0.0/48.0 | 908.8 | 907.5 | 1.0 | 1282.5 | 1246.8/1216.5 | 31.0 | 671.8 | 1.000 | 684.5 | 1534.8 | 8,22,41,690,25,8,438,19,1246 |
| 19 | 256 | 23 | 11.0/11.0 | 7.0 | 0.0/33.0 | 1053.8 | 1053.0 | 1.0 | 1000.8 | 974.8/947.0 | 27.8 | 452.0 | 0.984 | 462.2 | 1604.5 | 7,19,36,528,19,7,323,17,973 |
| 2 | 256 | 12 | 9.0/9.0 | 2.0 | 0.0/27.0 | 109.2 | 107.0 | 1.2 | 790.0 | 757.2/737.0 | 20.2 | -88.0 | 0.355 | 12.8 | 993.0 | 7,15,17,421,16,5,252,16,756 |
| 20 | 256 | 78 | 20.0/20.0 | 25.0 | 0.0/60.0 | 771.0 | 770.2 | 1.0 | 1976.0 | 1926.0/1902.0 | 39.2 | 1415.0 | 1.000 | 1428.2 | 1326.0 | 9,25,85,1047,32,16,686,20,1925 |
| 21 | 256 | 47 | 17.0/17.0 | 11.0 | 0.0/51.0 | 1741.8 | 1740.5 | 1.0 | 1525.2 | 1487.0/1448.0 | 30.5 | 980.0 | 1.000 | 995.8 | 2302.5 | 9,21,50,818,27,10,520,18,1486 |
| 22 | 256 | 56 | 17.0/17.0 | 10.0 | 0.0/51.0 | 1288.8 | 1287.5 | 1.0 | 1566.5 | 1542.2/1497.8 | 33.8 | 1079.2 | 1.000 | 1090.0 | 1792.8 | 9,21,51,848,28,11,553,17,1541 |
| 23 | 256 | 64 | 19.0/19.0 | 14.0 | 0.0/57.0 | 1335.2 | 1334.5 | 1.0 | 1724.8 | 1694.0/1668.8 | 34.8 | 1203.5 | 1.000 | 1220.8 | 1853.8 | 9,23,62,920,30,12,603,18,1693 |
| 24 | 256 | 58 | 18.0/18.0 | 11.0 | 0.0/54.0 | 1493.2 | 1492.5 | 1.0 | 1632.0 | 1576.0/1555.5 | 34.0 | 1096.8 | 1.000 | 1105.8 | 2027.0 | 8,21,53,872,28,11,553,18,1575 |
| 25 | 256 | 55 | 18.0/18.0 | 13.0 | 0.0/54.0 | 1370.5 | 1369.2 | 1.0 | 1627.8 | 1559.8/1547.8 | 38.0 | 1092.0 | 1.000 | 1105.2 | 1915.2 | 8,22,56,856,28,12,545,20,1558 |
| 26 | 256 | 76 | 20.0/20.0 | 15.0 | 0.0/60.0 | 1394.0 | 1393.0 | 1.0 | 1923.5 | 1882.8/1876.2 | 34.0 | 1373.5 | 0.992 | 1389.0 | 1945.2 | 9,22,65,1051,31,14,665,18,1882 |
| 27 | 256 | 94 | 26.0/26.0 | 15.0 | 0.0/78.0 | 1687.5 | 1686.2 | 1.0 | 2354.0 | 2322.5/2285.2 | 46.2 | 1809.8 | 1.000 | 1827.0 | 2222.0 | 10,24,70,1291,39,16,821,20,2321 |
| 28 | 256 | 123 | 29.0/29.0 | 18.0 | 0.0/87.0 | 2109.0 | 2108.2 | 1.0 | 2745.5 | 2706.5/2643.8 | 45.8 | 2204.0 | 1.000 | 2216.2 | 2665.8 | 10,25,75,1503,43,18,989,20,2706 |
| 29 | 256 | 78 | 21.0/21.0 | 12.0 | 0.0/63.0 | 2511.2 | 2510.5 | 1.0 | 1929.0 | 1900.8/1864.0 | 40.5 | 1420.2 | 0.996 | 1429.2 | 3031.0 | 9,21,61,1059,33,13,688,18,1899 |
| 3 | 256 | 12 | 9.0/9.0 | 3.0 | 0.0/27.0 | 320.8 | 320.0 | 1.0 | 724.8 | 689.2/664.2 | 23.2 | 36.5 | 0.582 | 46.2 | 1010.2 | 6,14,22,377,15,5,231,15,688 |
| 30 | 256 | 113 | 27.0/27.0 | 15.0 | 0.0/81.0 | 1709.5 | 1708.5 | 1.0 | 2510.0 | 2481.0/2428.8 | 43.5 | 1954.0 | 1.000 | 1965.5 | 2263.5 | 11,24,75,1388,40,17,893,19,2480 |
| 31 | 256 | 85 | 22.0/22.0 | 12.0 | 0.0/66.0 | 2271.8 | 2270.2 | 1.0 | 2014.8 | 1973.2/1937.8 | 40.0 | 1464.0 | 0.996 | 1474.0 | 2820.0 | 9,22,61,1093,34,13,720,17,1972 |
| 32 | 256 | 111 | 27.0/27.0 | 14.0 | 0.0/81.0 | 1786.0 | 1784.8 | 1.0 | 2480.2 | 2448.2/2386.2 | 46.8 | 1986.0 | 1.000 | 1999.8 | 2286.2 | 10,22,71,1358,41,15,899,18,2447 |
| 33 | 256 | 136 | 33.0/33.0 | 14.0 | 0.0/99.0 | 2235.5 | 2234.5 | 1.0 | 3083.0 | 3051.2/2993.2 | 54.5 | 2581.2 | 1.000 | 2597.2 | 2763.8 | 11,25,78,1732,48,17,1106,20,3050 |
| 34 | 256 | 99 | 29.0/29.0 | 15.0 | 0.0/87.0 | 2847.2 | 2846.5 | 1.0 | 2516.8 | 2482.0/2419.0 | 51.5 | 1982.8 | 1.000 | 1994.8 | 3389.2 | 10,23,69,1383,41,16,903,19,2480 |
| 35 | 256 | 100 | 26.0/26.0 | 14.0 | 0.0/78.0 | 2272.8 | 2270.8 | 1.0 | 2387.5 | 2359.5/2302.0 | 48.8 | 1855.8 | 1.000 | 1870.8 | 2823.5 | 10,23,67,1305,39,15,861,19,2358 |
| 36 | 256 | 102 | 27.0/27.0 | 13.0 | 0.0/81.0 | 2157.0 | 2156.2 | 1.0 | 2466.8 | 2441.0/2365.0 | 52.2 | 1933.0 | 0.996 | 1945.5 | 2706.5 | 10,22,69,1358,40,15,898,19,2440 |
| 37 | 256 | 82 | 27.0/27.0 | 12.0 | 0.0/81.0 | 2227.8 | 2227.2 | 1.0 | 2275.5 | 2243.5/2189.2 | 51.5 | 1743.5 | 1.000 | 1752.8 | 2770.5 | 9,22,63,1253,39,14,821,19,2242 |
| 38 | 256 | 91 | 26.0/26.0 | 13.0 | 0.0/78.0 | 2039.8 | 2038.8 | 1.0 | 2256.0 | 2223.5/2170.2 | 51.2 | 1653.5 | 1.000 | 1662.0 | 2646.0 | 10,22,65,1233,38,15,813,19,2222 |
| 39 | 256 | 62 | 21.0/21.0 | 9.0 | 0.0/63.0 | 2012.5 | 2011.0 | 1.0 | 1769.0 | 1739.8/1707.8 | 36.0 | 1185.2 | 0.988 | 1197.2 | 2600.5 | 9,20,51,961,31,11,623,18,1738 |
| 4 | 256 | 14 | 11.0/11.0 | 3.0 | 0.0/33.0 | 401.2 | 400.5 | 1.0 | 850.8 | 816.0/791.5 | 25.5 | 291.2 | 0.914 | 303.0 | 955.0 | 7,15,22,452,17,5,276,15,815 |
| 40 | 256 | 63 | 21.0/21.0 | 10.0 | 0.0/63.0 | 1532.0 | 1531.0 | 1.0 | 1766.8 | 1733.2/1690.0 | 45.2 | 1204.8 | 0.996 | 1216.0 | 2093.2 | 9,21,51,955,32,12,631,18,1732 |
| 41 | 256 | 94 | 22.0/22.0 | 15.0 | 0.0/66.0 | 1534.5 | 1533.8 | 1.0 | 2112.2 | 2075.0/2020.8 | 44.2 | 1564.5 | 1.000 | 1579.8 | 2087.0 | 9,22,70,1140,34,14,769,18,2073 |
| 42 | 256 | 52 | 18.0/18.0 | 8.0 | 0.0/54.0 | 1875.0 | 1872.8 | 1.0 | 1529.8 | 1497.8/1452.5 | 40.8 | 950.8 | 1.000 | 959.2 | 2467.2 | 8,21,45,826,28,10,527,18,1496 |
| 43 | 256 | 31 | 15.0/15.0 | 6.0 | 0.0/45.0 | 1292.2 | 1291.8 | 1.0 | 1217.0 | 1184.8/1144.5 | 32.0 | 674.8 | 1.000 | 687.2 | 1826.5 | 8,19,34,651,24,8,420,17,1183 |
| 44 | 256 | 84 | 23.0/23.0 | 13.0 | 0.0/69.0 | 984.2 | 983.2 | 1.0 | 2095.5 | 2051.5/2016.0 | 49.8 | 1508.0 | 1.000 | 1522.8 | 1587.2 | 9,22,63,1133,35,14,740,17,2050 |
| 45 | 256 | 64 | 21.0/21.0 | 10.0 | 0.0/63.0 | 1864.0 | 1863.0 | 1.0 | 1815.0 | 1789.2/1746.2 | 40.5 | 1193.0 | 1.000 | 1205.5 | 2495.0 | 8,21,52,989,31,12,647,18,1788 |
| 46 | 256 | 23 | 11.0/11.0 | 6.0 | 0.0/33.0 | 1587.0 | 1586.5 | 1.0 | 932.8 | 901.5/870.0 | 27.8 | 383.2 | 0.992 | 400.5 | 2140.5 | 8,18,33,489,19,7,307,16,900 |
| 47 | 256 | 62 | 25.0/25.0 | 8.0 | 0.0/75.0 | 700.8 | 699.8 | 1.0 | 1945.8 | 1907.0/1857.5 | 58.8 | 1313.2 | 1.000 | 1325.2 | 1333.0 | 9,23,47,1058,36,11,697,19,1906 |
| 48 | 256 | 37 | 17.0/17.0 | 6.0 | 0.0/51.0 | 1717.5 | 1716.0 | 1.0 | 1361.2 | 1331.5/1283.8 | 41.8 | 713.0 | 0.988 | 725.0 | 2369.2 | 8,19,37,724,26,9,473,16,1331 |
| 49 | 256 | 43 | 16.0/16.0 | 10.0 | 0.0/48.0 | 1117.2 | 1116.5 | 1.0 | 1397.2 | 1359.8/1316.0 | 33.5 | 816.5 | 0.984 | 830.2 | 1708.5 | 8,19,47,748,27,10,482,17,1358 |
| 5 | 256 | 29 | 15.0/15.0 | 8.0 | 0.0/45.0 | 596.2 | 595.8 | 1.0 | 1247.8 | 1214.8/1159.2 | 50.2 | 637.8 | 1.000 | 649.8 | 1212.5 | 8,19,40,667,24,8,423,17,1214 |
| 50 | 256 | 61 | 25.0/25.0 | 8.0 | 0.0/75.0 | 1172.5 | 1171.8 | 1.0 | 2036.8 | 2010.5/1946.2 | 53.5 | 1420.0 | 1.000 | 1434.2 | 1763.2 | 9,23,48,1128,37,11,721,20,2009 |
| 51 | 256 | 56 | 22.0/22.0 | 7.0 | 0.0/66.0 | 1792.2 | 1791.2 | 1.0 | 1748.8 | 1721.0/1669.0 | 48.2 | 1103.5 | 0.996 | 1116.5 | 2446.0 | 8,20,46,955,33,10,616,18,1720 |
| 52 | 256 | 31 | 17.0/17.0 | 5.0 | 0.0/51.0 | 1514.8 | 1513.8 | 1.0 | 1359.8 | 1329.5/1282.2 | 40.8 | 744.2 | 1.000 | 760.0 | 2128.5 | 8,20,33,738,26,8,464,18,1328 |
| 53 | 256 | 54 | 22.0/22.0 | 8.0 | 0.0/66.0 | 1130.8 | 1129.5 | 1.0 | 1717.5 | 1689.5/1640.8 | 50.5 | 1119.2 | 1.000 | 1132.8 | 1733.2 | 8,22,45,935,32,10,609,19,1688 |
| 54 | 256 | 32 | 17.0/17.0 | 5.0 | 0.0/51.0 | 1484.5 | 1482.8 | 1.0 | 1361.2 | 1327.8/1295.0 | 40.5 | 778.2 | 1.000 | 793.5 | 2092.0 | 8,19,35,743,27,8,467,17,1326 |
| 55 | 256 | 55 | 25.0/25.0 | 6.0 | 0.0/75.0 | 1128.0 | 1127.0 | 1.0 | 1946.8 | 1918.0/1839.2 | 54.5 | 1358.2 | 1.000 | 1378.0 | 1727.0 | 9,22,42,1076,36,10,691,19,1912 |
| 56 | 256 | 48 | 22.0/22.0 | 6.0 | 0.0/66.0 | 1718.5 | 1716.8 | 1.0 | 1725.2 | 1688.2/1628.5 | 52.0 | 1109.2 | 1.000 | 1130.2 | 2326.5 | 8,21,43,934,33,10,605,18,1687 |
| 57 | 256 | 38 | 20.0/20.0 | 4.0 | 0.0/60.0 | 1489.0 | 1488.2 | 1.0 | 1560.0 | 1535.8/1479.8 | 44.2 | 941.5 | 1.000 | 953.2 | 2102.8 | 7,20,37,851,30,9,550,18,1534 |
| 58 | 256 | 40 | 21.0/21.0 | 5.0 | 0.0/63.0 | 1328.2 | 1327.2 | 1.0 | 1580.5 | 1544.0/1477.0 | 50.8 | 976.8 | 1.000 | 989.0 | 1942.8 | 7,21,38,868,31,9,550,19,1543 |
| 59 | 256 | 38 | 19.0/19.0 | 5.0 | 0.0/57.0 | 1335.8 | 1334.2 | 1.0 | 1484.8 | 1460.0/1397.0 | 44.8 | 854.5 | 0.996 | 872.0 | 1985.5 | 8,19,37,816,29,9,518,17,1459 |
| 6 | 256 | 13 | 8.0/8.0 | 4.0 | 0.0/24.0 | 1016.0 | 1014.2 | 1.0 | 723.5 | 698.0/664.5 | 25.8 | 168.8 | 0.922 | 182.5 | 1574.8 | 6,16,25,380,15,5,225,15,697 |
| 60 | 256 | 40 | 18.0/18.0 | 6.0 | 0.0/54.0 | 1250.5 | 1249.5 | 1.0 | 1439.8 | 1408.0/1370.8 | 42.2 | 851.5 | 1.000 | 866.0 | 1841.2 | 8,21,41,777,27,9,498,19,1407 |
| 7 | 256 | 47 | 10.0/10.0 | 22.0 | 0.0/30.0 | 491.2 | 490.8 | 1.0 | 1256.2 | 1184.0/1170.5 | 40.5 | 684.5 | 0.992 | 700.0 | 1052.2 | 8,20,78,614,19,14,391,16,1183 |
| 8 | 256 | 56 | 16.0/16.0 | 18.0 | 0.0/48.0 | 1016.5 | 1015.5 | 1.0 | 1525.5 | 1480.0/1442.0 | 56.5 | 952.8 | 1.000 | 967.2 | 1595.5 | 9,20,66,811,26,12,520,16,1479 |
| 9 | 256 | 28 | 11.0/11.0 | 13.0 | 0.0/33.0 | 1291.2 | 1290.2 | 1.0 | 1037.2 | 995.5/948.0 | 44.2 | 524.8 | 1.000 | 538.5 | 1826.8 | 7,18,53,537,19,10,330,17,994 |

### S18_CORR_pinned (bs=63 graph, 층별 p50, µs)

| producer L | n | cold assign | unique n0/n1 | max rows n0 | AMX/AVX n0 | enq→start | pred union | unattrib | service | numa0/1 | skew | pub delta | late share | pre_h2d gap | budget | stages n0 (prep,cpy,q_in,up_gate,act,q_down,down,weight,total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 133 | 1 | 1.0/1.0 | 1.0 | 0.0/3.0 | 549.5 | 548.2 | 1.2 | 201.0 | 171.2/163.2 | 8.8 | -403.8 | 0.000 | 9.8 | 1205.5 | 3,11,11,76,3,1,42,15,170 |
| 1 | 256 | 12 | 8.0/8.0 | 3.0 | 0.0/24.0 | 6.5 | 0.2 | 6.0 | 724.5 | 697.8/662.8 | 25.2 | -246.2 | 0.051 | 10.8 | 999.5 | 7,15,20,376,15,5,232,17,697 |
| 10 | 256 | 25 | 14.0/14.0 | 6.0 | 0.0/42.0 | 798.8 | 798.0 | 1.0 | 1158.2 | 1127.5/1082.0 | 46.8 | 574.8 | 1.000 | 587.0 | 1388.5 | 8,20,35,628,22,7,384,18,1127 |
| 11 | 256 | 15 | 11.0/11.0 | 3.0 | 0.0/33.0 | 930.0 | 928.5 | 1.0 | 872.2 | 843.5/809.0 | 34.8 | 312.0 | 0.938 | 324.5 | 1506.0 | 6,16,26,468,18,6,284,16,842 |
| 12 | 256 | 34 | 13.0/13.0 | 15.0 | 0.0/39.0 | 639.5 | 639.0 | 1.0 | 1228.2 | 1164.8/1160.0 | 46.8 | 658.0 | 1.000 | 672.8 | 1216.0 | 8,19,59,630,23,10,404,17,1164 |
| 13 | 256 | 35 | 17.0/17.0 | 8.0 | 0.0/51.0 | 992.0 | 990.5 | 1.0 | 1434.5 | 1397.2/1350.2 | 52.0 | 854.0 | 1.000 | 868.5 | 1577.2 | 8,21,44,782,26,9,473,19,1396 |
| 14 | 256 | 32 | 16.0/16.0 | 10.0 | 0.0/48.0 | 1204.0 | 1203.0 | 1.0 | 1320.8 | 1293.0/1234.2 | 44.0 | 769.5 | 1.000 | 781.8 | 1750.2 | 7,21,46,700,25,9,451,18,1292 |
| 15 | 256 | 22 | 13.0/13.0 | 4.0 | 0.0/39.0 | 1089.5 | 1088.5 | 1.0 | 1056.5 | 1023.2/999.5 | 30.8 | 496.2 | 0.996 | 508.5 | 1651.8 | 7,19,30,558,21,7,349,17,1022 |
| 16 | 256 | 46 | 18.0/18.0 | 9.0 | 0.0/54.0 | 822.0 | 820.5 | 1.0 | 1507.2 | 1473.5/1434.8 | 44.0 | 967.2 | 1.000 | 977.8 | 1366.2 | 7,22,47,817,28,10,520,19,1472 |
| 17 | 256 | 23 | 14.0/14.0 | 5.0 | 0.0/42.0 | 1270.5 | 1269.5 | 1.0 | 1152.2 | 1118.2/1069.5 | 37.0 | 610.0 | 1.000 | 623.8 | 1827.5 | 8,20,30,616,23,7,385,19,1117 |
| 18 | 256 | 33 | 16.0/16.0 | 8.0 | 0.0/48.0 | 917.5 | 916.5 | 1.0 | 1296.2 | 1268.0/1223.2 | 37.2 | 688.2 | 1.000 | 699.2 | 1546.5 | 8,22,41,692,25,8,443,19,1267 |
| 19 | 256 | 23 | 11.0/11.0 | 7.0 | 0.0/33.0 | 1068.0 | 1067.5 | 0.8 | 1014.5 | 980.0/945.2 | 30.8 | 463.8 | 0.984 | 477.5 | 1613.8 | 7,18,37,537,20,7,323,17,979 |
| 2 | 256 | 12 | 9.0/9.0 | 2.0 | 0.0/27.0 | 114.8 | 113.8 | 1.2 | 794.0 | 759.0/733.0 | 22.8 | -76.8 | 0.348 | 12.0 | 993.8 | 7,15,17,421,16,5,253,16,758 |
| 20 | 256 | 78 | 20.0/20.0 | 25.0 | 0.0/60.0 | 779.2 | 778.5 | 1.0 | 2008.0 | 1961.0/1924.8 | 49.5 | 1455.8 | 1.000 | 1468.2 | 1346.5 | 9,25,84,1074,32,16,690,20,1960 |
| 21 | 256 | 47 | 17.0/17.0 | 11.0 | 0.0/51.0 | 1776.5 | 1774.8 | 1.0 | 1539.0 | 1508.8/1456.2 | 37.5 | 983.5 | 0.996 | 997.2 | 2329.8 | 8,20,50,826,28,10,524,18,1507 |
| 22 | 256 | 56 | 17.0/17.0 | 10.0 | 0.0/51.0 | 1305.0 | 1304.0 | 1.0 | 1578.0 | 1536.5/1504.2 | 43.5 | 1081.8 | 0.996 | 1099.8 | 1819.0 | 9,21,50,845,28,11,551,18,1535 |
| 23 | 256 | 64 | 19.0/19.0 | 14.0 | 0.0/57.0 | 1345.2 | 1344.2 | 1.0 | 1747.2 | 1709.5/1657.8 | 45.0 | 1237.5 | 1.000 | 1245.5 | 1855.5 | 9,23,62,927,30,13,611,18,1708 |
| 24 | 256 | 58 | 18.0/18.0 | 11.0 | 0.0/54.0 | 1522.2 | 1521.5 | 1.0 | 1630.0 | 1598.2/1536.5 | 42.0 | 1091.2 | 0.996 | 1105.2 | 2054.5 | 8,20,53,895,29,11,554,18,1597 |
| 25 | 256 | 55 | 18.0/18.0 | 13.0 | 0.0/54.0 | 1401.8 | 1400.5 | 1.0 | 1674.8 | 1613.5/1601.2 | 49.5 | 1145.2 | 1.000 | 1160.2 | 1943.0 | 8,22,57,904,28,12,552,20,1612 |
| 26 | 256 | 76 | 20.0/20.0 | 15.0 | 0.0/60.0 | 1438.5 | 1437.8 | 1.0 | 1967.2 | 1932.2/1874.0 | 53.2 | 1410.2 | 0.992 | 1421.0 | 1986.0 | 9,22,65,1079,31,14,683,18,1931 |
| 27 | 256 | 94 | 26.0/26.0 | 15.0 | 0.0/78.0 | 1725.2 | 1724.5 | 1.0 | 2435.2 | 2350.2/2292.0 | 61.5 | 1884.2 | 1.000 | 1900.8 | 2272.5 | 10,23,71,1318,38,16,827,20,2349 |
| 28 | 256 | 123 | 29.0/29.0 | 18.0 | 0.0/87.0 | 2200.0 | 2199.2 | 1.0 | 2756.5 | 2730.0/2656.2 | 55.8 | 2214.5 | 1.000 | 2227.2 | 2737.2 | 10,25,76,1525,44,18,998,20,2729 |
| 29 | 256 | 78 | 21.0/21.0 | 12.0 | 0.0/63.0 | 2519.0 | 2517.5 | 1.0 | 1961.2 | 1923.5/1864.0 | 48.2 | 1430.2 | 0.996 | 1443.5 | 3035.8 | 9,21,62,1071,34,13,689,18,1922 |
| 3 | 256 | 12 | 9.0/9.0 | 3.0 | 0.0/27.0 | 323.5 | 322.8 | 1.0 | 715.2 | 687.0/661.0 | 21.8 | 41.8 | 0.598 | 54.5 | 1009.8 | 6,13,22,372,15,5,233,15,686 |
| 30 | 256 | 113 | 27.0/27.0 | 15.0 | 0.0/81.0 | 1727.2 | 1726.0 | 1.0 | 2532.0 | 2481.5/2435.8 | 53.0 | 1964.5 | 1.000 | 1976.0 | 2296.5 | 10,23,75,1412,40,17,889,19,2480 |
| 31 | 256 | 85 | 22.0/22.0 | 12.0 | 0.0/66.0 | 2293.8 | 2293.0 | 1.0 | 2017.2 | 1989.8/1938.5 | 56.5 | 1484.2 | 0.992 | 1496.2 | 2840.2 | 9,22,61,1105,34,14,724,18,1989 |
| 32 | 256 | 111 | 27.0/27.0 | 14.0 | 0.0/81.0 | 1789.2 | 1788.0 | 1.0 | 2469.0 | 2439.0/2368.0 | 56.5 | 1976.8 | 1.000 | 1994.0 | 2287.8 | 10,22,71,1366,41,16,897,19,2438 |
| 33 | 256 | 136 | 33.0/33.0 | 14.0 | 0.0/99.0 | 2232.2 | 2231.2 | 1.0 | 3130.2 | 3099.0/3004.8 | 70.8 | 2625.2 | 1.000 | 2639.8 | 2760.5 | 11,24,79,1752,48,17,1110,20,3098 |
| 34 | 256 | 99 | 29.0/29.0 | 15.0 | 0.0/87.0 | 2899.2 | 2898.5 | 1.0 | 2533.5 | 2500.0/2409.8 | 58.2 | 2000.5 | 1.000 | 2013.5 | 3425.8 | 10,24,70,1402,41,16,895,20,2499 |
| 35 | 256 | 100 | 26.0/26.0 | 14.0 | 0.0/78.0 | 2299.0 | 2298.0 | 1.0 | 2416.5 | 2374.5/2299.5 | 55.5 | 1890.8 | 0.996 | 1900.0 | 2837.8 | 10,23,69,1329,39,15,858,20,2373 |
| 36 | 256 | 102 | 27.0/27.0 | 13.0 | 0.0/81.0 | 2188.0 | 2187.0 | 1.0 | 2445.5 | 2416.5/2349.5 | 62.8 | 1921.0 | 0.996 | 1936.2 | 2718.0 | 10,22,69,1345,40,15,893,19,2415 |
| 37 | 256 | 82 | 27.0/27.0 | 12.0 | 0.0/81.0 | 2216.8 | 2215.5 | 1.0 | 2278.0 | 2241.8/2182.5 | 57.5 | 1741.0 | 1.000 | 1752.0 | 2752.5 | 9,22,62,1259,39,14,817,19,2241 |
| 38 | 256 | 91 | 26.0/26.0 | 13.0 | 0.0/78.0 | 2045.5 | 2044.8 | 1.0 | 2258.8 | 2224.0/2162.2 | 57.0 | 1660.8 | 1.000 | 1672.5 | 2664.8 | 9,22,64,1240,38,15,814,19,2223 |
| 39 | 256 | 62 | 21.0/21.0 | 9.0 | 0.0/63.0 | 2017.0 | 2016.2 | 1.0 | 1800.2 | 1771.5/1711.0 | 51.2 | 1225.5 | 0.988 | 1236.0 | 2609.0 | 9,20,50,976,32,12,627,18,1770 |
| 4 | 256 | 14 | 11.0/11.0 | 3.0 | 0.0/33.0 | 402.0 | 401.0 | 1.0 | 850.0 | 820.8/799.5 | 30.2 | 295.8 | 0.926 | 307.2 | 957.0 | 7,15,22,459,17,5,277,15,819 |
| 40 | 256 | 63 | 21.0/21.0 | 10.0 | 0.0/63.0 | 1567.2 | 1566.2 | 1.0 | 1769.2 | 1735.8/1687.8 | 55.8 | 1200.2 | 0.992 | 1215.2 | 2134.2 | 9,21,53,956,33,12,625,18,1735 |
| 41 | 256 | 94 | 22.0/22.0 | 15.0 | 0.0/66.0 | 1532.5 | 1532.0 | 1.0 | 2111.0 | 2080.8/2021.5 | 58.0 | 1574.5 | 0.996 | 1585.2 | 2070.5 | 9,22,71,1146,34,15,762,19,2079 |
| 42 | 256 | 52 | 18.0/18.0 | 8.0 | 0.0/54.0 | 1874.8 | 1873.5 | 1.0 | 1550.5 | 1515.2/1459.0 | 50.0 | 976.8 | 1.000 | 986.8 | 2468.5 | 8,21,45,837,28,11,533,18,1514 |
| 43 | 256 | 31 | 15.0/15.0 | 6.0 | 0.0/45.0 | 1316.2 | 1315.5 | 1.0 | 1208.2 | 1175.5/1145.8 | 39.2 | 674.8 | 1.000 | 685.2 | 1858.2 | 8,19,34,652,24,8,422,17,1174 |
| 44 | 256 | 84 | 23.0/23.0 | 13.0 | 0.0/69.0 | 979.2 | 978.0 | 1.0 | 2110.0 | 2078.8/2012.8 | 56.2 | 1532.5 | 1.000 | 1544.0 | 1572.8 | 10,21,65,1152,35,14,744,18,2077 |
| 45 | 256 | 64 | 21.0/21.0 | 10.0 | 0.0/63.0 | 1873.2 | 1872.5 | 1.0 | 1825.5 | 1798.0/1747.0 | 52.5 | 1218.0 | 0.996 | 1227.8 | 2518.2 | 8,21,53,995,32,12,652,18,1797 |
| 46 | 256 | 23 | 11.0/11.0 | 6.0 | 0.0/33.0 | 1597.0 | 1596.0 | 1.0 | 943.2 | 914.8/871.8 | 32.0 | 386.5 | 0.992 | 399.0 | 2159.2 | 8,18,33,492,19,7,310,16,913 |
| 47 | 256 | 62 | 25.0/25.0 | 8.0 | 0.0/75.0 | 707.0 | 705.8 | 0.8 | 1931.8 | 1906.2/1853.2 | 63.0 | 1308.0 | 1.000 | 1323.8 | 1337.0 | 9,23,48,1063,36,11,697,19,1905 |
| 48 | 256 | 37 | 17.0/17.0 | 6.0 | 0.0/51.0 | 1701.2 | 1700.5 | 1.0 | 1349.2 | 1323.5/1278.2 | 41.0 | 717.8 | 0.988 | 728.0 | 2347.5 | 8,19,37,728,26,9,473,16,1322 |
| 49 | 256 | 43 | 16.0/16.0 | 10.0 | 0.0/48.0 | 1109.8 | 1109.0 | 1.0 | 1395.8 | 1362.8/1328.8 | 38.0 | 809.2 | 0.980 | 818.5 | 1702.8 | 8,19,48,755,27,10,478,17,1361 |
| 5 | 256 | 29 | 15.0/15.0 | 8.0 | 0.0/45.0 | 599.8 | 599.0 | 1.0 | 1258.8 | 1223.8/1163.2 | 54.5 | 635.8 | 1.000 | 649.8 | 1218.2 | 8,19,40,667,24,8,430,17,1222 |
| 50 | 256 | 61 | 25.0/25.0 | 8.0 | 0.0/75.0 | 1171.8 | 1170.5 | 1.0 | 2040.2 | 2009.5/1947.0 | 61.8 | 1416.5 | 1.000 | 1432.8 | 1770.5 | 9,23,48,1133,37,11,735,20,2008 |
| 51 | 256 | 56 | 22.0/22.0 | 7.0 | 0.0/66.0 | 1804.2 | 1803.5 | 1.0 | 1763.0 | 1736.2/1672.5 | 52.2 | 1106.5 | 0.996 | 1119.0 | 2473.0 | 8,20,45,964,33,10,619,18,1736 |
| 52 | 256 | 31 | 17.0/17.0 | 5.0 | 0.0/51.0 | 1528.8 | 1528.2 | 1.0 | 1357.2 | 1331.8/1285.5 | 42.5 | 739.5 | 0.996 | 753.0 | 2148.0 | 8,21,33,736,27,8,469,18,1330 |
| 53 | 256 | 54 | 22.0/22.0 | 8.0 | 0.0/66.0 | 1123.8 | 1123.2 | 1.0 | 1724.2 | 1698.2/1639.8 | 53.0 | 1115.8 | 1.000 | 1124.5 | 1728.5 | 8,22,47,943,33,10,612,19,1697 |
| 54 | 256 | 32 | 17.0/17.0 | 5.0 | 0.0/51.0 | 1489.8 | 1489.0 | 1.0 | 1369.0 | 1342.0/1302.8 | 44.0 | 781.2 | 1.000 | 795.2 | 2094.2 | 8,19,34,742,27,8,474,17,1341 |
| 55 | 256 | 55 | 25.0/25.0 | 6.0 | 0.0/75.0 | 1136.2 | 1134.8 | 1.0 | 1948.8 | 1923.2/1849.2 | 60.0 | 1345.2 | 1.000 | 1357.0 | 1721.8 | 9,22,42,1078,37,10,692,19,1922 |
| 56 | 256 | 48 | 22.0/22.0 | 6.0 | 0.0/66.0 | 1717.2 | 1716.8 | 1.0 | 1718.2 | 1682.2/1622.0 | 56.5 | 1095.0 | 1.000 | 1111.2 | 2330.2 | 8,20,43,929,32,10,612,18,1681 |
| 57 | 256 | 38 | 20.0/20.0 | 4.0 | 0.0/60.0 | 1472.2 | 1471.5 | 1.0 | 1558.2 | 1534.0/1481.0 | 50.5 | 940.2 | 1.000 | 952.2 | 2084.0 | 7,20,37,854,30,9,548,18,1533 |
| 58 | 256 | 40 | 21.0/21.0 | 5.0 | 0.0/63.0 | 1319.8 | 1318.8 | 1.0 | 1587.5 | 1554.2/1501.2 | 52.0 | 985.8 | 1.000 | 998.0 | 1936.2 | 7,21,39,871,31,9,549,19,1553 |
| 59 | 256 | 38 | 19.0/19.0 | 5.0 | 0.0/57.0 | 1355.0 | 1354.0 | 1.0 | 1487.0 | 1462.2/1410.2 | 47.2 | 841.8 | 0.996 | 851.8 | 2001.5 | 8,19,38,813,29,9,524,17,1461 |
| 6 | 256 | 13 | 8.0/8.0 | 4.0 | 0.0/24.0 | 1019.5 | 1019.0 | 1.0 | 727.8 | 699.8/668.5 | 29.5 | 177.0 | 0.930 | 189.8 | 1583.5 | 7,16,26,382,15,5,229,15,698 |
| 60 | 256 | 40 | 18.0/18.0 | 6.0 | 0.0/54.0 | 1251.8 | 1251.0 | 1.0 | 1445.8 | 1413.2/1366.5 | 49.5 | 855.8 | 1.000 | 872.0 | 1839.5 | 8,21,41,786,28,9,503,19,1412 |
| 7 | 256 | 47 | 10.0/10.0 | 22.0 | 0.0/30.0 | 491.0 | 490.2 | 1.0 | 1300.8 | 1256.0/1189.2 | 68.8 | 740.8 | 0.992 | 749.0 | 1053.8 | 8,20,79,670,19,13,391,16,1255 |
| 8 | 256 | 56 | 16.0/16.0 | 18.0 | 0.0/48.0 | 1069.8 | 1068.2 | 1.0 | 1548.8 | 1500.8/1435.0 | 67.5 | 969.5 | 1.000 | 983.8 | 1648.5 | 9,20,66,821,26,12,518,17,1500 |
| 9 | 256 | 28 | 11.0/11.0 | 13.0 | 0.0/33.0 | 1321.0 | 1320.2 | 1.0 | 1026.0 | 999.2/948.2 | 49.0 | 518.5 | 1.000 | 530.2 | 1850.8 | 7,19,55,540,19,9,333,17,998 |

### S22_CORR_vllmw (bs=63 graph, 층별 p50, µs)

| producer L | n | cold assign | unique n0/n1 | max rows n0 | AMX/AVX n0 | enq→start | pred union | unattrib | service | numa0/1 | skew | pub delta | late share | pre_h2d gap | budget | stages n0 (prep,cpy,q_in,up_gate,act,q_down,down,weight,total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 238 | 12 | 6.0/6.0 | 3.0 | 0.0/18.0 | 280.8 | 279.5 | 1.2 | 597.2 | 574.5/549.0 | 22.0 | -308.8 | 0.235 | 10.5 | 61128.8 | 3,14,20,296,15,5,193,15,574 |
| 1 | 252 | 18 | 12.0/12.0 | 3.0 | 0.0/36.0 | 92.2 | 91.8 | 1.0 | 986.5 | 960.0/917.5 | 43.0 | 24.5 | 0.512 | 12.8 | 120981.2 | 7,16,22,538,22,6,326,17,959 |
| 10 | 253 | 22 | 15.0/15.0 | 3.0 | 0.0/45.0 | 707.8 | 707.0 | 0.8 | 1205.2 | 1181.5/1110.5 | 65.0 | 640.5 | 0.980 | 588.8 | 121029.0 | 7,23,27,667,25,7,402,18,1180 |
| 11 | 253 | 19 | 12.0/12.0 | 3.0 | 0.0/36.0 | 974.0 | 973.2 | 0.8 | 954.8 | 924.2/874.5 | 45.8 | 407.8 | 0.964 | 380.2 | 1558.5 | 6,20,27,513,21,6,308,16,923 |
| 12 | 251 | 16 | 11.0/11.0 | 3.0 | 0.0/33.0 | 724.0 | 723.2 | 0.8 | 906.5 | 881.5/822.2 | 43.5 | 367.0 | 0.964 | 293.2 | 121024.0 | 7,20,24,485,21,6,288,17,880 |
| 13 | 251 | 24 | 16.0/16.0 | 4.0 | 0.0/48.0 | 673.2 | 672.2 | 0.8 | 1247.2 | 1221.0/1147.5 | 52.2 | 702.0 | 1.000 | 640.8 | -183534.5 | 8,23,28,680,26,7,416,18,1219 |
| 14 | 252 | 18 | 13.0/13.0 | 3.0 | 0.0/39.0 | 1016.8 | 1016.0 | 0.8 | 996.5 | 970.2/932.5 | 37.5 | 461.0 | 0.996 | 430.2 | 121470.0 | 7,22,25,532,22,6,328,18,969 |
| 15 | 252 | 20 | 15.0/15.0 | 3.0 | 0.0/45.0 | 754.8 | 753.8 | 0.8 | 1128.8 | 1102.8/1057.5 | 41.0 | 598.2 | 1.000 | 564.5 | -186055.0 | 7,21,25,614,24,7,371,17,1102 |
| 16 | 252 | 21 | 15.0/15.0 | 3.0 | 0.0/45.0 | 895.8 | 894.8 | 0.8 | 1138.5 | 1112.5/1076.5 | 39.5 | 617.5 | 1.000 | 589.0 | 182192.5 | 6,22,26,625,25,7,376,19,1111 |
| 17 | 252 | 16 | 12.0/12.0 | 2.0 | 0.0/36.0 | 907.8 | 907.0 | 0.8 | 938.5 | 912.0/869.5 | 37.8 | 404.2 | 0.996 | 373.0 | -60596.0 | 7,21,22,504,21,6,300,18,911 |
| 18 | 253 | 21 | 15.0/15.0 | 3.0 | 0.0/45.0 | 707.0 | 706.2 | 0.8 | 1129.2 | 1099.8/1065.5 | 37.2 | 490.5 | 1.000 | 475.8 | 1359.0 | 8,23,25,616,25,7,374,19,1099 |
| 19 | 253 | 16 | 10.0/10.0 | 4.0 | 0.0/30.0 | 893.5 | 892.8 | 0.8 | 844.0 | 816.8/775.2 | 30.5 | 317.8 | 0.980 | 289.0 | 1430.8 | 6,20,26,438,19,6,272,17,816 |
| 2 | 252 | 23 | 14.0/14.0 | 4.0 | 0.0/42.0 | 379.8 | 379.0 | 0.8 | 1098.0 | 1071.0/1017.5 | 42.5 | 443.8 | 0.810 | 297.2 | -60316.2 | 7,18,27,594,24,7,368,17,1070 |
| 20 | 253 | 30 | 19.0/19.0 | 4.0 | 0.0/57.0 | 607.0 | 606.2 | 0.8 | 1428.2 | 1402.8/1338.0 | 52.0 | 865.0 | 1.000 | 848.5 | -123372.0 | 7,24,30,781,30,8,490,19,1401 |
| 21 | 253 | 26 | 16.0/16.0 | 4.0 | 0.0/48.0 | 1189.2 | 1188.5 | 0.8 | 1249.5 | 1222.0/1169.0 | 41.5 | 733.2 | 1.000 | 687.0 | 121230.0 | 7,23,30,680,26,8,419,18,1221 |
| 22 | 253 | 13 | 11.0/11.0 | 2.0 | 0.0/33.0 | 1012.5 | 1011.5 | 0.8 | 841.0 | 812.2/784.8 | 29.2 | 341.8 | 0.976 | 281.5 | 1534.2 | 7,19,21,439,19,6,270,17,811 |
| 23 | 253 | 19 | 14.0/14.0 | 3.0 | 0.0/42.0 | 602.8 | 602.0 | 0.8 | 1067.0 | 1032.0/987.8 | 41.8 | 547.0 | 1.000 | 512.0 | 1136.8 | 7,22,26,571,24,7,350,18,1031 |
| 24 | 252 | 21 | 14.0/14.0 | 4.0 | 0.0/42.0 | 838.2 | 837.2 | 0.8 | 1106.0 | 1078.5/1040.0 | 43.5 | 624.0 | 0.992 | 541.0 | 61225.5 | 7,21,29,594,24,7,375,17,1077 |
| 25 | 252 | 19 | 13.0/13.0 | 4.0 | 0.0/39.0 | 870.0 | 869.2 | 0.8 | 1042.8 | 1013.8/976.0 | 34.2 | 537.8 | 1.000 | 497.0 | -122203.0 | 7,23,27,562,23,7,342,19,1012 |
| 26 | 254 | 13 | 10.0/10.0 | 3.0 | 0.0/30.0 | 799.2 | 798.0 | 0.8 | 812.0 | 786.5/759.5 | 26.0 | 274.0 | 0.972 | 250.2 | 1332.0 | 6,20,23,425,18,6,258,17,785 |
| 27 | 254 | 18 | 11.0/11.0 | 5.0 | 0.0/33.0 | 578.2 | 577.2 | 0.8 | 919.8 | 891.2/850.5 | 36.2 | 389.8 | 1.000 | 369.5 | -60027.0 | 6,22,29,478,20,7,293,18,890 |
| 28 | 253 | 22 | 15.0/15.0 | 4.0 | 0.0/45.0 | 682.0 | 681.5 | 0.8 | 1139.8 | 1112.5/1080.5 | 38.2 | 616.0 | 1.000 | 600.5 | 121279.8 | 6,23,28,617,25,7,376,19,1111 |
| 29 | 253 | 12 | 9.0/9.0 | 2.0 | 0.0/27.0 | 906.2 | 905.8 | 0.8 | 768.0 | 744.5/710.5 | 28.5 | 238.0 | 0.949 | 224.8 | 1418.8 | 6,19,22,410,18,5,240,17,743 |
| 3 | 253 | 17 | 12.0/12.0 | 3.0 | 0.0/36.0 | 753.8 | 752.8 | 0.8 | 977.2 | 953.0/896.8 | 40.2 | 351.0 | 0.846 | 307.2 | -60561.2 | 6,18,24,531,21,6,316,16,951 |
| 30 | 252 | 21 | 13.0/13.0 | 5.0 | 0.0/39.0 | 533.0 | 532.0 | 0.8 | 1014.5 | 986.8/945.5 | 37.0 | 475.8 | 1.000 | 469.2 | 60942.8 | 7,22,29,533,22,7,338,17,986 |
| 31 | 252 | 7 | 5.0/5.0 | 2.0 | 0.0/15.0 | 775.5 | 775.0 | 0.8 | 487.8 | 461.2/435.5 | 20.8 | -41.2 | 0.393 | 12.0 | -186062.8 | 5,16,21,238,12,4,138,15,459 |
| 32 | 255 | 9 | 8.0/8.0 | 2.0 | 0.0/24.0 | 249.0 | 248.0 | 0.8 | 649.8 | 625.0/610.0 | 17.0 | 112.8 | 0.702 | 76.0 | 844.0 | 6,16,16,337,15,5,203,16,624 |
| 33 | 252 | 11 | 9.0/9.0 | 2.0 | 0.0/27.0 | 346.0 | 345.0 | 0.8 | 708.5 | 682.8/660.8 | 23.2 | 205.8 | 0.845 | 175.8 | 182149.8 | 6,19,20,368,16,5,228,17,682 |
| 34 | 252 | 10 | 9.0/9.0 | 2.0 | 0.0/27.0 | 466.0 | 465.5 | 0.8 | 699.2 | 674.8/648.0 | 23.2 | 181.2 | 0.964 | 177.5 | -123973.5 | 6,20,19,359,16,5,223,18,673 |
| 35 | 254 | 9 | 8.0/8.0 | 2.0 | 0.0/24.0 | 458.5 | 457.5 | 0.8 | 672.8 | 645.8/619.5 | 23.8 | 160.0 | 0.965 | 137.0 | 980.0 | 6,21,19,347,15,5,205,18,645 |
| 36 | 254 | 16 | 9.0/9.0 | 5.0 | 0.0/27.0 | 435.8 | 435.0 | 0.8 | 800.2 | 773.8/747.0 | 30.2 | 272.5 | 0.996 | 253.5 | 998.0 | 6,21,28,413,18,6,256,17,773 |
| 37 | 254 | 17 | 10.0/10.0 | 4.0 | 0.0/30.0 | 564.0 | 563.2 | 0.8 | 835.2 | 808.8/779.2 | 30.5 | 288.2 | 1.000 | 287.0 | 1102.5 | 7,22,27,429,18,6,273,18,808 |
| 38 | 254 | 23 | 12.0/12.0 | 6.0 | 0.0/36.0 | 604.2 | 603.5 | 0.8 | 997.0 | 972.5/935.2 | 31.8 | 401.5 | 0.996 | 380.8 | -61148.0 | 7,23,34,522,22,7,328,17,971 |
| 39 | 253 | 12 | 8.0/8.0 | 3.0 | 0.0/24.0 | 766.0 | 765.0 | 0.8 | 680.0 | 649.5/627.5 | 25.5 | 129.0 | 0.866 | 124.5 | 121307.0 | 6,19,24,342,16,5,213,17,648 |
| 4 | 254 | 26 | 15.0/15.0 | 5.0 | 0.0/45.0 | 715.8 | 715.0 | 0.8 | 1179.0 | 1152.5/1090.2 | 50.8 | 716.5 | 0.906 | 611.5 | -60879.2 | 7,21,31,638,25,7,394,16,1151 |
| 40 | 253 | 15 | 10.0/10.0 | 4.0 | 0.0/30.0 | 446.0 | 445.5 | 0.8 | 808.8 | 780.5/755.5 | 29.5 | 256.8 | 0.960 | 249.5 | -60941.8 | 7,19,26,425,18,6,258,16,779 |
| 41 | 253 | 11 | 8.0/8.0 | 2.0 | 0.0/24.0 | 574.0 | 573.0 | 0.8 | 707.8 | 683.2/656.5 | 25.0 | 220.8 | 0.960 | 180.0 | 60555.0 | 6,19,21,368,16,5,223,16,682 |
| 42 | 253 | 27 | 15.0/15.0 | 6.0 | 0.0/45.0 | 474.2 | 473.5 | 0.8 | 1221.0 | 1196.0/1153.5 | 44.2 | 708.8 | 1.000 | 666.5 | -60920.0 | 7,23,34,658,26,8,411,17,1195 |
| 43 | 254 | 14 | 10.0/10.0 | 3.0 | 0.0/30.0 | 985.8 | 985.0 | 0.8 | 791.0 | 765.0/732.0 | 29.2 | 259.0 | 0.984 | 263.8 | -60413.0 | 7,20,23,418,18,6,249,17,764 |
| 44 | 255 | 18 | 11.0/11.0 | 3.0 | 0.0/33.0 | 559.5 | 558.5 | 0.8 | 916.5 | 892.5/851.8 | 37.5 | 374.2 | 0.918 | 355.5 | 1092.0 | 7,20,26,482,20,6,306,16,891 |
| 45 | 253 | 19 | 12.0/12.0 | 4.0 | 0.0/36.0 | 678.2 | 677.5 | 0.8 | 941.0 | 917.0/880.8 | 36.5 | 398.5 | 0.980 | 367.2 | 121281.5 | 6,20,28,497,21,7,311,17,916 |
| 46 | 253 | 7 | 6.0/6.0 | 2.0 | 0.0/18.0 | 714.8 | 714.0 | 0.8 | 517.5 | 491.8/467.8 | 21.8 | -6.2 | 0.478 | 12.8 | 1246.0 | 6,17,20,255,13,5,154,15,491 |
| 47 | 253 | 17 | 13.0/13.0 | 3.0 | 0.0/39.0 | 279.8 | 279.0 | 0.8 | 1016.5 | 992.2/957.0 | 34.0 | 438.0 | 0.960 | 397.0 | -61184.8 | 7,19,22,550,22,6,332,18,991 |
| 48 | 253 | 12 | 8.0/8.0 | 2.0 | 0.0/24.0 | 734.5 | 733.5 | 0.8 | 693.2 | 667.8/643.2 | 24.0 | 134.0 | 0.791 | 113.5 | 60604.5 | 6,18,22,357,16,5,222,16,666 |
| 49 | 253 | 27 | 8.0/8.0 | 13.0 | 0.0/24.0 | 459.0 | 458.0 | 0.8 | 879.2 | 845.5/814.8 | 32.5 | 332.2 | 0.968 | 324.5 | -60949.5 | 7,21,51,439,18,9,271,16,845 |
| 5 | 253 | 24 | 17.0/17.0 | 3.0 | 0.0/51.0 | 939.2 | 938.2 | 0.8 | 1299.5 | 1275.5/1188.0 | 74.5 | 702.0 | 0.933 | 606.8 | 121421.5 | 8,21,27,726,27,7,430,17,1274 |
| 50 | 253 | 17 | 14.0/14.0 | 3.0 | 0.0/42.0 | 643.8 | 642.8 | 0.8 | 1062.0 | 1031.8/990.0 | 38.0 | 528.8 | 0.996 | 500.5 | 60441.5 | 7,22,23,575,23,7,353,19,1031 |
| 51 | 253 | 15 | 11.0/11.0 | 2.0 | 0.0/33.0 | 822.8 | 822.0 | 0.8 | 879.8 | 853.8/818.2 | 30.5 | 302.2 | 0.957 | 282.2 | -59451.2 | 6,20,23,466,20,6,289,17,853 |
| 52 | 254 | 13 | 10.0/10.0 | 3.0 | 0.0/30.0 | 648.2 | 647.2 | 0.8 | 810.2 | 785.0/754.2 | 29.8 | 249.8 | 0.969 | 229.5 | 1219.2 | 6,20,22,431,18,6,262,17,784 |
| 53 | 254 | 26 | 15.0/15.0 | 5.0 | 0.0/45.0 | 576.2 | 575.8 | 0.8 | 1182.5 | 1147.0/1108.5 | 40.5 | 616.2 | 1.000 | 586.8 | 1162.0 | 7,23,32,641,25,8,391,18,1145 |
| 54 | 251 | 15 | 11.0/11.0 | 3.0 | 0.0/33.0 | 939.8 | 938.0 | 0.8 | 879.2 | 847.8/810.0 | 35.2 | 342.8 | 0.992 | 317.5 | 182765.2 | 6,20,22,462,19,6,284,17,847 |
| 55 | 251 | 17 | 14.0/14.0 | 2.0 | 0.0/42.0 | 644.8 | 644.0 | 0.8 | 1069.2 | 1040.0/999.0 | 35.8 | 544.2 | 1.000 | 478.5 | -121803.0 | 7,21,23,582,24,6,358,18,1039 |
| 56 | 253 | 18 | 13.0/13.0 | 3.0 | 0.0/39.0 | 834.8 | 833.8 | 0.8 | 1019.8 | 995.5/949.5 | 35.5 | 466.5 | 0.992 | 427.5 | -122853.2 | 6,21,25,549,22,6,337,17,994 |
| 57 | 252 | 11 | 10.0/10.0 | 2.0 | 0.0/30.0 | 782.0 | 781.2 | 0.8 | 793.8 | 768.5/743.8 | 28.2 | 229.2 | 0.956 | 195.0 | 182017.2 | 6,19,21,425,18,5,250,17,768 |
| 58 | 252 | 11 | 10.0/10.0 | 2.0 | 0.0/30.0 | 557.5 | 557.0 | 0.8 | 794.0 | 763.8/740.2 | 28.8 | 212.0 | 0.933 | 191.0 | 1139.0 | 5,20,20,422,18,5,248,18,757 |
| 59 | 252 | 10 | 9.0/9.0 | 2.0 | 0.0/27.0 | 553.5 | 553.0 | 0.8 | 699.0 | 668.5/642.2 | 27.2 | 87.0 | 0.671 | 59.5 | -122384.0 | 6,18,22,354,16,5,226,16,667 |
| 6 | 253 | 18 | 12.0/12.0 | 3.0 | 0.0/36.0 | 1067.2 | 1066.5 | 0.8 | 1010.2 | 984.2/919.2 | 64.0 | 455.5 | 0.917 | 398.8 | -60195.2 | 7,19,25,556,22,6,326,16,982 |
| 60 | 253 | 14 | 12.0/12.0 | 2.0 | 0.0/36.0 | 455.5 | 455.0 | 0.8 | 929.0 | 903.5/866.5 | 28.8 | 329.5 | 0.937 | 282.5 | 60448.2 | 6,20,21,503,21,6,303,18,902 |
| 7 | 254 | 19 | 13.0/13.0 | 3.0 | 0.0/39.0 | 773.5 | 773.0 | 0.8 | 1071.5 | 1046.0/976.2 | 69.5 | 554.0 | 0.909 | 493.5 | 1331.5 | 7,20,26,589,23,7,353,16,1045 |
| 8 | 253 | 18 | 13.0/13.0 | 3.0 | 0.0/39.0 | 844.5 | 843.5 | 0.8 | 1020.2 | 992.8/926.8 | 58.0 | 434.8 | 0.945 | 410.0 | 60972.8 | 7,19,26,554,22,6,331,16,992 |
| 9 | 253 | 16 | 12.0/12.0 | 3.0 | 0.0/36.0 | 783.5 | 782.8 | 0.8 | 947.0 | 923.5/852.8 | 56.8 | 416.5 | 0.941 | 374.5 | -123072.2 | 7,20,23,510,20,6,304,17,922 |

### S23_CORR_vllmw (bs=63 graph, 층별 p50, µs)

| producer L | n | cold assign | unique n0/n1 | max rows n0 | AMX/AVX n0 | enq→start | pred union | unattrib | service | numa0/1 | skew | pub delta | late share | pre_h2d gap | budget | stages n0 (prep,cpy,q_in,up_gate,act,q_down,down,weight,total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 235 | 13 | 6.0/6.0 | 3.0 | 0.0/18.0 | 308.5 | 305.8 | 2.0 | 611.0 | 581.5/555.2 | 23.8 | -367.8 | 0.085 | 10.8 | 1303.8 | 4,15,21,299,15,5,198,15,581 |
| 1 | 256 | 18 | 12.0/12.0 | 3.0 | 0.0/36.0 | 107.0 | 106.2 | 1.5 | 987.5 | 955.2/917.8 | 47.0 | -3.2 | 0.488 | 13.8 | 1177.8 | 7,16,22,534,21,6,325,17,954 |
| 10 | 255 | 21 | 16.0/16.0 | 3.0 | 0.0/48.0 | 698.0 | 697.0 | 1.2 | 1200.2 | 1168.5/1107.8 | 62.0 | 592.8 | 0.988 | 606.8 | 1327.2 | 7,22,25,662,26,7,399,18,1167 |
| 11 | 255 | 18 | 12.0/12.0 | 3.0 | 0.0/36.0 | 969.8 | 968.8 | 1.2 | 949.2 | 923.5/857.2 | 46.5 | 374.2 | 0.980 | 388.0 | 1562.8 | 6,19,25,509,21,6,312,16,922 |
| 12 | 255 | 16 | 12.0/12.0 | 3.0 | 0.0/36.0 | 719.2 | 717.8 | 1.2 | 926.0 | 894.0/846.8 | 42.8 | 306.8 | 0.949 | 320.5 | 1347.2 | 8,20,23,499,20,6,302,17,893 |
| 13 | 255 | 24 | 16.0/16.0 | 4.0 | 0.0/48.0 | 690.8 | 689.2 | 1.2 | 1266.5 | 1239.0/1177.8 | 52.2 | 651.8 | 1.000 | 664.2 | 1308.8 | 8,23,27,686,27,7,425,18,1238 |
| 14 | 255 | 17 | 13.0/13.0 | 3.0 | 0.0/39.0 | 1034.2 | 1033.0 | 1.2 | 999.0 | 970.0/925.5 | 38.8 | 418.5 | 1.000 | 428.8 | 1612.2 | 7,22,24,528,22,6,327,18,969 |
| 15 | 255 | 20 | 15.0/15.0 | 3.0 | 0.0/45.0 | 757.0 | 756.0 | 1.2 | 1123.2 | 1096.5/1054.8 | 40.8 | 559.2 | 1.000 | 571.5 | 1336.5 | 8,21,24,612,24,7,374,18,1096 |
| 16 | 255 | 22 | 15.0/15.0 | 3.0 | 0.0/45.0 | 892.0 | 891.0 | 1.2 | 1173.2 | 1144.0/1111.0 | 44.0 | 601.0 | 1.000 | 611.5 | 1466.8 | 7,23,26,638,25,7,392,19,1143 |
| 17 | 255 | 16 | 12.0/12.0 | 3.0 | 0.0/36.0 | 934.5 | 932.5 | 1.2 | 954.0 | 916.0/863.5 | 40.8 | 384.0 | 0.996 | 394.0 | 1534.0 | 7,21,22,506,21,6,308,18,915 |
| 18 | 255 | 21 | 15.0/15.0 | 3.0 | 0.0/45.0 | 720.2 | 719.0 | 1.2 | 1131.8 | 1106.5/1066.2 | 39.0 | 480.0 | 1.000 | 492.0 | 1378.0 | 8,23,24,611,25,7,377,19,1105 |
| 19 | 255 | 16 | 10.0/10.0 | 4.0 | 0.0/30.0 | 896.2 | 895.0 | 1.2 | 842.8 | 813.8/784.0 | 33.5 | 285.0 | 1.000 | 297.5 | 1473.0 | 7,20,25,440,19,6,277,17,812 |
| 2 | 256 | 23 | 14.0/14.0 | 4.0 | 0.0/42.0 | 433.8 | 431.8 | 1.5 | 1100.5 | 1070.5/1014.5 | 47.2 | 339.0 | 0.785 | 360.2 | 1166.8 | 7,18,26,594,24,7,370,17,1069 |
| 20 | 255 | 30 | 19.0/19.0 | 4.0 | 0.0/57.0 | 614.2 | 613.0 | 1.2 | 1431.0 | 1396.8/1341.8 | 56.2 | 832.8 | 1.000 | 851.5 | 1214.0 | 7,24,29,772,30,8,494,19,1395 |
| 21 | 255 | 27 | 17.0/17.0 | 4.0 | 0.0/51.0 | 1193.5 | 1192.0 | 1.2 | 1285.5 | 1253.0/1214.0 | 47.8 | 720.8 | 1.000 | 734.5 | 1756.8 | 7,22,30,703,27,8,433,18,1252 |
| 22 | 255 | 14 | 11.0/11.0 | 2.0 | 0.0/33.0 | 1050.2 | 1048.5 | 1.2 | 847.8 | 820.0/790.8 | 34.0 | 300.2 | 0.969 | 314.2 | 1605.2 | 7,20,21,450,20,6,278,17,819 |
| 23 | 255 | 19 | 14.0/14.0 | 3.0 | 0.0/42.0 | 614.2 | 613.5 | 1.2 | 1052.8 | 1024.0/983.2 | 42.8 | 479.5 | 1.000 | 489.8 | 1201.8 | 7,21,25,561,23,7,352,18,1023 |
| 24 | 255 | 22 | 14.0/14.0 | 4.0 | 0.0/42.0 | 820.2 | 818.5 | 1.2 | 1089.2 | 1058.5/1008.5 | 47.2 | 511.0 | 0.984 | 526.0 | 1406.2 | 7,21,28,575,24,7,369,17,1058 |
| 25 | 255 | 20 | 13.0/13.0 | 4.0 | 0.0/39.0 | 850.2 | 848.2 | 1.2 | 1054.2 | 1029.0/986.2 | 41.0 | 509.0 | 1.000 | 519.5 | 1408.2 | 7,23,27,566,23,7,350,19,1027 |
| 26 | 255 | 14 | 10.0/10.0 | 3.0 | 0.0/30.0 | 822.0 | 820.8 | 1.2 | 820.0 | 786.0/757.5 | 31.0 | 258.2 | 0.976 | 270.0 | 1389.8 | 6,20,23,429,19,6,262,17,785 |
| 27 | 255 | 18 | 11.0/11.0 | 5.0 | 0.0/33.0 | 580.5 | 579.5 | 1.2 | 913.5 | 881.0/847.5 | 37.5 | 361.0 | 1.000 | 376.8 | 1154.2 | 6,22,31,474,20,7,295,18,880 |
| 28 | 255 | 21 | 14.0/14.0 | 4.0 | 0.0/42.0 | 679.5 | 678.8 | 1.2 | 1121.8 | 1092.5/1044.2 | 43.2 | 572.2 | 1.000 | 585.5 | 1239.2 | 7,23,27,601,25,7,368,19,1091 |
| 29 | 255 | 12 | 9.0/9.0 | 3.0 | 0.0/27.0 | 886.8 | 885.8 | 1.2 | 729.2 | 695.5/664.8 | 29.2 | 184.5 | 0.957 | 195.5 | 1438.5 | 6,19,21,371,17,5,232,17,694 |
| 3 | 256 | 17 | 12.0/12.0 | 3.0 | 0.0/36.0 | 815.2 | 814.0 | 1.2 | 930.2 | 897.2/849.8 | 45.2 | 262.8 | 0.820 | 275.0 | 1480.5 | 6,18,24,494,21,6,307,16,896 |
| 30 | 255 | 21 | 12.0/12.0 | 5.0 | 0.0/36.0 | 489.2 | 488.5 | 1.2 | 1000.5 | 973.5/932.0 | 38.5 | 440.8 | 1.000 | 450.8 | 1048.0 | 7,21,29,528,22,7,331,17,972 |
| 31 | 255 | 7 | 5.0/5.0 | 2.0 | 0.0/15.0 | 763.2 | 761.5 | 1.2 | 493.2 | 465.8/439.2 | 21.2 | -70.0 | 0.259 | 11.8 | 1314.5 | 5,17,20,244,12,4,138,15,465 |
| 32 | 255 | 9 | 8.0/8.0 | 2.0 | 0.0/24.0 | 262.0 | 260.5 | 1.2 | 665.0 | 635.8/606.2 | 21.2 | 74.0 | 0.667 | 89.8 | 854.8 | 6,16,17,339,15,5,204,16,634 |
| 33 | 255 | 11 | 9.0/9.0 | 2.0 | 0.0/27.0 | 346.2 | 345.2 | 1.2 | 743.2 | 712.2/677.0 | 28.0 | 200.0 | 0.886 | 212.8 | 881.8 | 6,18,19,380,17,5,237,17,711 |
| 34 | 255 | 10 | 9.0/9.0 | 2.0 | 0.0/27.0 | 495.0 | 494.0 | 1.2 | 699.0 | 669.0/641.2 | 26.0 | 171.5 | 0.953 | 184.2 | 1031.0 | 6,20,18,354,16,5,224,18,668 |
| 35 | 255 | 9 | 8.0/8.0 | 2.0 | 0.0/24.0 | 464.0 | 462.8 | 1.2 | 672.0 | 639.2/617.2 | 26.2 | 130.8 | 0.953 | 142.5 | 1011.2 | 6,21,19,346,15,5,205,18,638 |
| 36 | 255 | 16 | 9.0/9.0 | 5.0 | 0.0/27.0 | 439.0 | 437.8 | 1.2 | 802.2 | 774.2/740.5 | 32.8 | 245.0 | 0.996 | 257.8 | 1001.0 | 6,21,27,411,18,6,255,17,773 |
| 37 | 255 | 17 | 10.0/10.0 | 4.0 | 0.0/30.0 | 568.0 | 567.0 | 1.2 | 845.8 | 814.5/775.0 | 29.8 | 287.8 | 1.000 | 301.2 | 1134.5 | 7,22,26,431,18,6,274,18,813 |
| 38 | 255 | 24 | 12.0/12.0 | 6.0 | 0.0/36.0 | 611.2 | 610.5 | 1.2 | 998.2 | 969.8/928.2 | 35.2 | 378.8 | 0.988 | 392.8 | 1238.2 | 7,22,33,527,21,7,327,18,968 |
| 39 | 255 | 12 | 8.0/8.0 | 3.0 | 0.0/24.0 | 766.2 | 764.8 | 1.2 | 660.8 | 632.5/607.5 | 24.5 | 81.5 | 0.839 | 94.8 | 1355.0 | 6,19,22,334,15,5,205,17,631 |
| 4 | 256 | 26 | 15.0/15.0 | 4.0 | 0.0/45.0 | 690.0 | 688.5 | 1.2 | 1180.8 | 1145.5/1088.0 | 57.0 | 600.2 | 0.914 | 615.0 | 1273.2 | 7,20,29,642,25,7,393,16,1144 |
| 40 | 255 | 15 | 10.0/10.0 | 4.0 | 0.0/30.0 | 425.5 | 424.0 | 1.2 | 801.2 | 774.8/740.2 | 32.0 | 225.8 | 0.953 | 242.5 | 1015.2 | 7,19,26,414,18,6,256,17,773 |
| 41 | 255 | 12 | 9.0/9.0 | 2.0 | 0.0/27.0 | 564.5 | 563.0 | 1.2 | 720.0 | 690.5/657.0 | 28.8 | 188.8 | 0.941 | 201.2 | 1113.5 | 6,19,21,368,16,5,228,17,689 |
| 42 | 255 | 28 | 16.0/16.0 | 6.0 | 0.0/48.0 | 488.2 | 487.0 | 1.2 | 1234.8 | 1205.5/1153.8 | 51.2 | 675.5 | 1.000 | 685.8 | 1062.8 | 7,23,34,662,26,8,420,18,1204 |
| 43 | 255 | 14 | 9.0/9.0 | 3.0 | 0.0/27.0 | 1004.5 | 1003.5 | 1.2 | 785.2 | 756.5/729.8 | 32.8 | 251.2 | 0.992 | 260.5 | 1552.5 | 7,19,22,414,17,6,249,17,755 |
| 44 | 255 | 18 | 11.0/11.0 | 3.0 | 0.0/33.0 | 555.2 | 553.5 | 1.2 | 895.2 | 868.2/843.0 | 36.0 | 338.0 | 0.937 | 354.8 | 1112.5 | 7,20,25,470,20,6,300,16,867 |
| 45 | 255 | 19 | 11.0/11.0 | 4.0 | 0.0/33.0 | 662.8 | 661.5 | 1.2 | 944.0 | 911.2/865.5 | 38.5 | 358.0 | 0.988 | 369.8 | 1268.2 | 6,19,28,493,20,6,312,17,909 |
| 46 | 255 | 8 | 6.0/6.0 | 2.0 | 0.0/18.0 | 705.2 | 704.0 | 1.2 | 540.8 | 510.0/478.5 | 23.8 | 7.5 | 0.522 | 19.8 | 1264.0 | 6,18,19,262,13,5,161,15,509 |
| 47 | 255 | 17 | 13.0/13.0 | 2.0 | 0.0/39.0 | 305.0 | 303.8 | 1.2 | 1016.2 | 982.5/938.2 | 33.8 | 394.0 | 0.973 | 407.8 | 938.8 | 7,19,21,543,22,7,332,18,981 |
| 48 | 255 | 12 | 8.0/8.0 | 2.0 | 0.0/24.0 | 746.2 | 745.8 | 1.2 | 694.0 | 664.5/638.2 | 26.2 | 109.5 | 0.780 | 119.2 | 1341.2 | 6,18,21,357,16,5,224,16,663 |
| 49 | 255 | 27 | 8.0/8.0 | 14.0 | 0.0/24.0 | 457.5 | 456.2 | 1.2 | 847.8 | 811.5/773.2 | 35.8 | 264.8 | 0.976 | 280.8 | 1041.8 | 7,20,54,422,18,9,261,16,810 |
| 5 | 256 | 23 | 16.0/16.0 | 3.0 | 0.0/48.0 | 947.8 | 946.0 | 1.2 | 1279.8 | 1246.0/1171.2 | 74.5 | 586.5 | 0.914 | 598.5 | 1641.8 | 8,21,25,711,27,7,423,17,1244 |
| 50 | 255 | 18 | 14.0/14.0 | 3.0 | 0.0/42.0 | 608.8 | 607.5 | 1.2 | 1093.8 | 1051.8/1004.8 | 43.8 | 509.2 | 1.000 | 522.0 | 1204.0 | 7,23,23,588,23,7,357,19,1050 |
| 51 | 255 | 15 | 11.0/11.0 | 3.0 | 0.0/33.0 | 855.2 | 854.0 | 1.2 | 896.0 | 869.0/829.8 | 32.5 | 281.2 | 0.961 | 292.0 | 1462.2 | 6,20,23,467,20,6,291,17,868 |
| 52 | 255 | 13 | 10.0/10.0 | 2.0 | 0.0/30.0 | 661.0 | 659.8 | 1.2 | 827.0 | 797.2/762.5 | 34.5 | 232.2 | 0.980 | 244.5 | 1262.5 | 7,21,20,436,18,6,268,17,796 |
| 53 | 255 | 27 | 15.0/15.0 | 5.0 | 0.0/45.0 | 590.2 | 589.2 | 1.2 | 1183.2 | 1154.2/1096.8 | 44.8 | 580.2 | 1.000 | 591.8 | 1210.8 | 7,22,33,633,25,8,397,18,1153 |
| 54 | 255 | 16 | 12.0/12.0 | 3.0 | 0.0/36.0 | 953.8 | 952.2 | 1.2 | 918.8 | 892.0/852.5 | 38.5 | 348.2 | 0.996 | 361.2 | 1548.5 | 7,20,22,489,20,6,303,17,891 |
| 55 | 255 | 18 | 15.0/15.0 | 2.0 | 0.0/45.0 | 689.0 | 683.2 | 1.2 | 1160.0 | 1123.2/1084.8 | 40.0 | 545.0 | 1.000 | 552.5 | 1289.0 | 7,21,22,632,25,7,384,18,1122 |
| 56 | 255 | 18 | 14.0/14.0 | 3.0 | 0.0/42.0 | 919.8 | 918.5 | 1.2 | 1035.5 | 1006.2/962.8 | 39.2 | 436.0 | 1.000 | 448.0 | 1544.0 | 7,21,23,558,23,7,343,17,1005 |
| 57 | 255 | 11 | 10.0/10.0 | 2.0 | 0.0/30.0 | 793.8 | 792.5 | 1.2 | 789.8 | 760.5/727.5 | 29.8 | 178.5 | 0.933 | 192.0 | 1409.8 | 6,20,19,418,18,5,250,17,759 |
| 58 | 255 | 11 | 9.0/9.0 | 2.0 | 0.0/27.0 | 558.0 | 556.5 | 1.2 | 752.0 | 720.2/682.8 | 31.0 | 156.0 | 0.910 | 166.2 | 1179.2 | 6,20,19,386,17,5,241,18,719 |
| 59 | 255 | 10 | 8.0/8.0 | 2.0 | 0.0/24.0 | 517.2 | 516.2 | 1.2 | 682.0 | 656.5/617.8 | 28.8 | 43.5 | 0.651 | 56.0 | 1156.5 | 6,17,20,351,16,5,217,16,655 |
| 6 | 256 | 18 | 12.0/12.0 | 3.0 | 0.0/36.0 | 1042.2 | 1041.5 | 1.2 | 1006.2 | 978.8/893.2 | 61.8 | 378.8 | 0.906 | 390.2 | 1678.0 | 7,19,25,550,21,6,327,16,977 |
| 60 | 255 | 14 | 12.0/12.0 | 2.0 | 0.0/36.0 | 451.8 | 450.5 | 1.2 | 912.5 | 879.0/846.5 | 31.5 | 261.2 | 0.941 | 281.2 | 1103.8 | 6,20,21,480,20,6,297,18,878 |
| 7 | 255 | 19 | 13.0/13.0 | 3.0 | 0.0/39.0 | 770.0 | 769.2 | 1.2 | 1034.8 | 1007.0/944.0 | 67.5 | 457.8 | 0.933 | 470.5 | 1553.2 | 7,20,25,569,22,6,342,16,1006 |
| 8 | 255 | 18 | 12.0/12.0 | 3.0 | 0.0/36.0 | 806.0 | 804.2 | 1.2 | 979.2 | 949.8/881.0 | 54.2 | 358.5 | 0.933 | 370.0 | 1440.5 | 7,20,25,529,21,6,321,16,948 |
| 9 | 255 | 16 | 11.0/11.0 | 3.0 | 0.0/33.0 | 741.8 | 741.2 | 1.2 | 929.2 | 895.2/830.5 | 59.0 | 363.5 | 0.941 | 379.0 | 1322.8 | 7,20,22,502,20,6,294,17,894 |

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

### S29_CORR_vllmw_fp8fixso (bs=63 graph, 층별 p50, µs)

| producer L | n | cold assign | unique n0/n1 | max rows n0 | AMX/AVX n0 | enq→start | pred union | unattrib | service | numa0/1 | skew | pub delta | late share | pre_h2d gap | budget | stages n0 (prep,cpy,q_in,up_gate,act,q_down,down,weight,total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 240 | 13 | 6.0/6.0 | 3.0 | 0.0/18.0 | 284.8 | 283.8 | 1.2 | 605.0 | 571.2/538.2 | 25.2 | -366.5 | 0.087 | 10.8 | 1294.5 | 4,14,20,293,14,5,192,16,570 |
| 1 | 256 | 18 | 12.0/12.0 | 3.0 | 0.0/36.0 | 92.2 | 91.2 | 1.0 | 1000.0 | 970.0/920.2 | 45.5 | -36.8 | 0.465 | 13.2 | 1170.8 | 7,16,22,540,21,6,331,17,969 |
| 10 | 256 | 22 | 16.0/16.0 | 3.0 | 0.0/48.0 | 709.2 | 708.2 | 0.8 | 1232.2 | 1190.5/1117.5 | 57.2 | 599.0 | 0.988 | 611.0 | 1343.0 | 7,21,25,674,26,7,407,18,1190 |
| 11 | 256 | 18 | 12.0/12.0 | 3.0 | 0.0/36.0 | 1002.2 | 1001.2 | 0.8 | 942.5 | 906.5/859.8 | 40.0 | 358.2 | 0.984 | 372.5 | 1599.0 | 6,18,25,497,21,6,305,16,905 |
| 12 | 256 | 16 | 12.0/12.0 | 3.0 | 0.0/36.0 | 708.8 | 707.8 | 0.8 | 921.8 | 885.2/841.8 | 41.2 | 327.8 | 0.992 | 341.2 | 1346.2 | 8,19,23,494,21,6,295,17,884 |
| 13 | 256 | 25 | 16.0/16.0 | 4.0 | 0.0/48.0 | 691.5 | 690.8 | 1.0 | 1229.2 | 1202.5/1141.8 | 47.8 | 626.5 | 1.000 | 640.0 | 1297.2 | 7,21,28,662,26,7,410,19,1201 |
| 14 | 256 | 17 | 12.0/12.0 | 3.0 | 0.0/36.0 | 997.2 | 996.5 | 1.0 | 979.2 | 946.5/904.8 | 42.0 | 399.8 | 1.000 | 413.8 | 1573.2 | 6,20,24,519,22,6,323,18,945 |
| 15 | 256 | 21 | 15.0/15.0 | 3.0 | 0.0/45.0 | 745.8 | 745.0 | 0.8 | 1143.2 | 1117.5/1069.2 | 40.0 | 590.5 | 1.000 | 609.0 | 1318.5 | 7,20,26,623,25,7,383,18,1116 |
| 16 | 256 | 22 | 15.0/15.0 | 3.0 | 0.0/45.0 | 913.8 | 912.8 | 1.0 | 1175.5 | 1143.5/1101.0 | 41.5 | 596.0 | 1.000 | 608.8 | 1488.2 | 7,21,26,635,25,7,390,19,1142 |
| 17 | 256 | 16 | 12.0/12.0 | 3.0 | 0.0/36.0 | 939.2 | 938.0 | 1.0 | 950.0 | 916.2/868.2 | 39.5 | 382.8 | 1.000 | 397.5 | 1520.5 | 7,20,23,502,21,6,311,19,915 |
| 18 | 256 | 21 | 15.0/15.0 | 3.0 | 0.0/45.0 | 722.0 | 721.0 | 0.8 | 1155.8 | 1116.2/1071.5 | 36.0 | 494.8 | 1.000 | 507.2 | 1379.8 | 7,22,25,622,25,7,385,19,1115 |
| 19 | 256 | 15 | 10.0/10.0 | 4.0 | 0.0/30.0 | 918.0 | 917.0 | 0.8 | 837.8 | 807.8/769.2 | 29.2 | 282.8 | 0.996 | 295.5 | 1491.0 | 7,19,26,437,19,6,273,17,807 |
| 2 | 256 | 23 | 13.0/13.0 | 4.0 | 0.0/39.0 | 402.5 | 401.2 | 1.0 | 1061.2 | 1014.5/982.8 | 37.0 | 286.0 | 0.801 | 296.0 | 1154.2 | 7,17,27,572,23,6,347,17,1013 |
| 20 | 256 | 31 | 19.0/19.0 | 4.0 | 0.0/57.0 | 605.5 | 604.5 | 0.8 | 1466.5 | 1434.0/1377.0 | 54.0 | 884.5 | 1.000 | 895.5 | 1205.2 | 8,23,30,796,31,8,504,19,1433 |
| 21 | 256 | 27 | 17.0/17.0 | 4.0 | 0.0/51.0 | 1230.5 | 1229.0 | 0.8 | 1293.8 | 1254.0/1206.2 | 41.5 | 728.0 | 1.000 | 737.5 | 1791.8 | 7,21,30,698,27,7,429,18,1253 |
| 22 | 256 | 14 | 11.0/11.0 | 2.0 | 0.0/33.0 | 1055.0 | 1054.0 | 0.8 | 859.5 | 823.2/792.0 | 33.0 | 298.0 | 0.969 | 314.8 | 1608.8 | 7,19,21,446,19,5,281,17,822 |
| 23 | 256 | 19 | 14.0/14.0 | 3.0 | 0.0/42.0 | 617.2 | 616.8 | 0.8 | 1042.5 | 1012.8/971.5 | 39.0 | 477.0 | 1.000 | 488.5 | 1197.8 | 7,19,25,562,23,6,344,18,1012 |
| 24 | 256 | 22 | 15.0/15.0 | 4.0 | 0.0/45.0 | 811.2 | 810.5 | 0.8 | 1113.2 | 1079.0/1047.8 | 40.2 | 535.0 | 0.984 | 547.2 | 1389.0 | 6,19,28,597,24,7,374,17,1077 |
| 25 | 256 | 20 | 14.0/14.0 | 4.0 | 0.0/42.0 | 882.2 | 881.5 | 0.8 | 1053.5 | 1015.8/977.5 | 35.2 | 509.5 | 1.000 | 521.8 | 1446.5 | 7,21,27,562,23,6,347,19,1015 |
| 26 | 256 | 14 | 10.0/10.0 | 3.0 | 0.0/30.0 | 822.8 | 822.0 | 0.8 | 822.8 | 784.5/753.5 | 29.2 | 264.8 | 0.953 | 280.8 | 1383.5 | 7,18,23,430,18,5,262,17,784 |
| 27 | 256 | 18 | 11.0/11.0 | 5.0 | 0.0/33.0 | 592.8 | 592.2 | 0.8 | 922.8 | 888.0/845.0 | 35.2 | 375.2 | 1.000 | 388.8 | 1143.5 | 7,20,29,474,20,6,297,18,887 |
| 28 | 256 | 22 | 15.0/15.0 | 4.0 | 0.0/45.0 | 687.5 | 686.8 | 0.8 | 1133.0 | 1096.2/1055.2 | 39.8 | 575.2 | 1.000 | 589.2 | 1248.5 | 7,21,28,608,25,7,376,19,1095 |
| 29 | 256 | 13 | 9.0/9.0 | 2.0 | 0.0/27.0 | 899.5 | 898.8 | 0.8 | 774.2 | 745.5/707.8 | 29.0 | 217.5 | 0.965 | 231.8 | 1452.8 | 6,18,21,408,18,5,243,17,745 |
| 3 | 256 | 17 | 12.0/12.0 | 3.0 | 0.0/36.0 | 760.5 | 759.8 | 0.8 | 934.2 | 894.2/849.2 | 39.5 | 253.5 | 0.809 | 264.2 | 1431.0 | 6,17,23,497,20,6,299,16,893 |
| 30 | 256 | 21 | 12.0/12.0 | 5.0 | 0.0/36.0 | 535.2 | 534.8 | 0.8 | 986.2 | 959.8/921.0 | 34.5 | 438.2 | 1.000 | 450.5 | 1089.2 | 7,20,30,516,22,6,327,17,959 |
| 31 | 256 | 6 | 5.0/5.0 | 2.0 | 0.0/15.0 | 755.8 | 754.8 | 0.8 | 484.2 | 453.5/429.0 | 21.2 | -78.2 | 0.285 | 12.0 | 1312.2 | 5,15,20,231,11,4,136,16,452 |
| 32 | 256 | 10 | 8.0/8.0 | 2.0 | 0.0/24.0 | 249.0 | 248.0 | 1.0 | 664.5 | 627.8/604.2 | 24.0 | 76.2 | 0.664 | 88.2 | 848.8 | 6,15,16,337,15,5,203,17,626 |
| 33 | 256 | 12 | 9.0/9.0 | 2.0 | 0.0/27.0 | 348.0 | 347.2 | 0.8 | 741.8 | 709.2/682.5 | 25.5 | 210.8 | 0.852 | 223.8 | 885.2 | 6,18,19,381,17,5,237,18,708 |
| 34 | 256 | 10 | 9.0/9.0 | 2.0 | 0.0/27.0 | 500.5 | 499.8 | 0.8 | 697.5 | 667.2/640.0 | 27.0 | 173.8 | 0.949 | 184.5 | 1037.5 | 7,19,19,354,16,5,220,18,666 |
| 35 | 256 | 10 | 8.0/8.0 | 2.0 | 0.0/24.0 | 458.5 | 457.5 | 0.8 | 679.0 | 650.2/621.8 | 24.2 | 140.8 | 0.941 | 151.2 | 1001.8 | 6,20,19,351,16,5,215,18,649 |
| 36 | 256 | 17 | 10.0/10.0 | 5.0 | 0.0/30.0 | 443.8 | 443.0 | 0.8 | 813.8 | 778.5/749.0 | 28.2 | 257.0 | 1.000 | 270.2 | 1010.8 | 6,19,28,416,18,6,258,17,777 |
| 37 | 256 | 17 | 10.0/10.0 | 4.0 | 0.0/30.0 | 576.8 | 576.0 | 0.8 | 840.5 | 804.8/774.5 | 28.8 | 288.2 | 1.000 | 302.0 | 1138.2 | 7,20,26,428,18,6,271,18,804 |
| 38 | 256 | 24 | 12.0/12.0 | 6.0 | 0.0/36.0 | 609.5 | 608.5 | 0.8 | 1018.8 | 986.0/945.8 | 33.0 | 406.0 | 0.996 | 418.5 | 1228.5 | 7,21,33,529,22,7,334,18,985 |
| 39 | 256 | 12 | 8.0/8.0 | 3.0 | 0.0/24.0 | 783.5 | 782.2 | 0.8 | 671.2 | 638.5/610.8 | 24.8 | 96.8 | 0.863 | 109.8 | 1366.2 | 6,18,22,335,15,5,209,17,638 |
| 4 | 256 | 27 | 15.0/15.0 | 5.0 | 0.0/45.0 | 692.0 | 691.2 | 1.0 | 1181.2 | 1151.0/1093.2 | 51.5 | 613.8 | 0.926 | 626.0 | 1267.8 | 7,18,31,639,25,7,398,16,1150 |
| 40 | 256 | 15 | 10.0/10.0 | 4.0 | 0.0/30.0 | 435.2 | 434.2 | 1.0 | 822.0 | 789.2/749.2 | 30.5 | 238.8 | 0.957 | 249.5 | 1022.8 | 7,18,24,422,18,6,264,17,788 |
| 41 | 256 | 12 | 9.0/9.0 | 2.0 | 0.0/27.0 | 588.8 | 588.0 | 0.8 | 726.5 | 694.5/660.0 | 28.8 | 212.8 | 0.973 | 223.5 | 1123.5 | 6,18,21,374,16,5,231,17,693 |
| 42 | 256 | 28 | 16.0/16.0 | 6.0 | 0.0/48.0 | 493.0 | 492.2 | 0.8 | 1242.5 | 1207.8/1164.8 | 38.8 | 676.5 | 1.000 | 692.2 | 1056.0 | 7,21,33,665,27,8,423,17,1206 |
| 43 | 256 | 14 | 10.0/10.0 | 3.0 | 0.0/30.0 | 1004.8 | 1004.2 | 0.8 | 814.8 | 783.5/749.0 | 30.0 | 278.2 | 0.984 | 288.8 | 1551.5 | 7,18,22,430,18,5,258,17,782 |
| 44 | 256 | 18 | 11.0/11.0 | 3.0 | 0.0/33.0 | 585.2 | 584.8 | 1.0 | 891.2 | 853.2/818.5 | 33.5 | 330.0 | 0.934 | 345.2 | 1140.5 | 7,19,24,460,19,6,292,16,852 |
| 45 | 256 | 18 | 12.0/12.0 | 5.0 | 0.0/36.0 | 663.2 | 662.5 | 0.8 | 944.8 | 911.8/868.2 | 34.5 | 356.8 | 0.992 | 370.2 | 1243.8 | 6,18,28,495,20,6,307,17,911 |
| 46 | 256 | 7 | 6.0/6.0 | 2.0 | 0.0/18.0 | 707.2 | 706.8 | 0.8 | 518.8 | 485.2/459.5 | 24.0 | -20.5 | 0.426 | 12.8 | 1267.5 | 6,17,19,252,12,4,152,16,484 |
| 47 | 256 | 17 | 13.0/13.0 | 2.0 | 0.0/39.0 | 281.8 | 280.8 | 0.8 | 991.0 | 962.8/918.8 | 32.0 | 358.0 | 0.957 | 370.8 | 927.5 | 7,18,22,531,22,6,326,18,962 |
| 48 | 256 | 12 | 9.0/9.0 | 3.0 | 0.0/27.0 | 723.5 | 722.5 | 1.0 | 710.8 | 679.8/649.5 | 27.5 | 123.2 | 0.781 | 138.0 | 1304.0 | 6,17,21,362,16,5,230,16,679 |
| 49 | 256 | 28 | 8.0/8.0 | 12.0 | 0.0/24.0 | 475.8 | 475.2 | 0.8 | 879.8 | 842.2/800.8 | 33.2 | 303.8 | 0.977 | 321.8 | 1060.8 | 7,19,52,440,18,9,269,16,841 |
| 5 | 256 | 23 | 16.0/16.0 | 3.0 | 0.0/48.0 | 948.0 | 947.0 | 1.0 | 1268.2 | 1201.8/1137.2 | 71.2 | 577.2 | 0.934 | 589.5 | 1643.8 | 8,19,26,683,26,7,419,17,1201 |
| 50 | 256 | 17 | 14.0/14.0 | 2.0 | 0.0/42.0 | 640.0 | 639.0 | 0.8 | 1071.8 | 1031.8/992.8 | 36.0 | 479.0 | 1.000 | 490.0 | 1224.8 | 7,21,23,573,23,6,347,19,1030 |
| 51 | 256 | 16 | 11.0/11.0 | 3.0 | 0.0/33.0 | 830.8 | 830.0 | 0.8 | 868.8 | 839.2/804.2 | 31.8 | 281.0 | 0.949 | 296.8 | 1442.2 | 6,18,23,453,19,6,286,17,838 |
| 52 | 256 | 13 | 10.0/10.0 | 3.0 | 0.0/30.0 | 643.2 | 642.2 | 1.0 | 822.2 | 782.8/747.2 | 32.5 | 223.8 | 0.977 | 236.5 | 1244.0 | 7,19,21,431,18,5,257,18,782 |
| 53 | 256 | 26 | 15.0/15.0 | 5.0 | 0.0/45.0 | 584.8 | 584.2 | 0.8 | 1149.2 | 1118.2/1070.2 | 39.2 | 547.5 | 1.000 | 565.5 | 1201.2 | 7,21,31,612,25,7,381,18,1117 |
| 54 | 256 | 16 | 12.0/12.0 | 3.0 | 0.0/36.0 | 924.2 | 923.5 | 0.8 | 927.0 | 891.5/853.0 | 30.8 | 345.8 | 0.992 | 360.2 | 1505.8 | 6,19,23,493,21,6,300,17,891 |
| 55 | 256 | 17 | 14.0/14.0 | 2.0 | 0.0/42.0 | 697.2 | 696.8 | 0.8 | 1059.5 | 1024.0/985.2 | 37.5 | 476.8 | 1.000 | 487.5 | 1285.0 | 7,20,22,567,23,6,356,18,1022 |
| 56 | 256 | 19 | 14.0/14.0 | 3.0 | 0.0/42.0 | 831.5 | 830.5 | 0.8 | 1048.2 | 1014.8/976.0 | 38.8 | 460.5 | 1.000 | 470.8 | 1429.2 | 7,20,24,562,23,6,351,18,1014 |
| 57 | 256 | 11 | 10.0/10.0 | 2.0 | 0.0/30.0 | 814.5 | 813.8 | 0.8 | 799.8 | 763.0/731.5 | 28.8 | 198.5 | 0.934 | 211.5 | 1424.2 | 6,18,20,419,18,5,254,18,762 |
| 58 | 256 | 11 | 10.0/10.0 | 2.0 | 0.0/30.0 | 567.5 | 567.0 | 0.8 | 791.2 | 753.0/722.8 | 27.8 | 166.2 | 0.918 | 180.0 | 1167.2 | 6,19,20,413,17,5,242,18,752 |
| 59 | 256 | 10 | 8.0/8.0 | 2.0 | 0.0/24.0 | 556.2 | 555.2 | 0.8 | 679.2 | 645.2/615.5 | 26.5 | 41.5 | 0.664 | 57.2 | 1173.2 | 7,16,20,350,16,5,217,16,645 |
| 6 | 256 | 19 | 12.0/12.0 | 4.0 | 0.0/36.0 | 1044.2 | 1039.5 | 1.0 | 1013.2 | 975.5/915.5 | 57.8 | 385.8 | 0.918 | 396.8 | 1663.8 | 7,19,26,542,22,6,328,16,974 |
| 60 | 256 | 14 | 12.0/12.0 | 2.0 | 0.0/36.0 | 444.0 | 443.0 | 1.0 | 905.2 | 868.2/831.2 | 31.0 | 239.8 | 0.953 | 254.5 | 1095.0 | 7,19,21,474,20,6,292,18,867 |
| 7 | 256 | 21 | 14.0/14.0 | 4.0 | 0.0/42.0 | 777.2 | 776.8 | 1.0 | 1090.0 | 1051.5/985.2 | 65.8 | 517.8 | 0.938 | 530.0 | 1371.8 | 7,19,26,595,23,6,355,16,1050 |
| 8 | 256 | 19 | 13.0/13.0 | 3.0 | 0.0/39.0 | 859.5 | 858.5 | 1.0 | 1041.5 | 1008.2/942.2 | 53.0 | 422.5 | 0.945 | 434.5 | 1473.8 | 7,19,25,560,22,6,339,16,1006 |
| 9 | 256 | 16 | 12.0/12.0 | 3.0 | 0.0/36.0 | 806.8 | 805.8 | 1.0 | 942.2 | 913.8/865.5 | 48.2 | 376.2 | 0.949 | 389.2 | 1387.0 | 7,18,23,507,21,6,304,17,912 |

## 8. 자원 (RESOURCE 세션; 부하 창 = forward_envelope(kt_evt))

### S11_RESOURCE

```
{
 "session": "S11_RESOURCE",
 "window": {
  "name": "forward_envelope(kt_evt)",
  "start_ns": 1789631215670818489,
  "end_ns": 1789631248059315249,
  "len_s": 32.38849676,
  "status": "OK"
 },
 "client_process_window_s": 32.51436090504285,
 "benchmark_duration_s": 32.392713914,
 "perf": {
  "intervals_total": 39,
  "full": 31,
  "boundary": 2,
  "outside": 6,
  "bad_rows": 0,
  "full_window_s": 31.0,
  "task_clock_msec_sum_full": 3187644.17,
  "cpu_equivalents_full": 102.82723129032259,
  "cycles_sum": 6351395371708.0,
  "instructions_sum": 10284801316459.0,
  "ipc_full": 1.6192979203077449,
  "context_switches_sum": 105059.0,
  "cpu_migrations_sum": 12042.0,
  "cache_misses_sum": 119926377696.0,
  "scope": "스케줄러 4 PID 프로세스 전체(스레드 상속)",
  "target_pids": "4013165,4013235,4013320,4013424",
  "interval_contract": "perf -I 1000: 각 행 = 직전 1 s 구간 (상대 초, 시작 = collector actual_start)",
  "note": "task-clock 은 논리 CPU 시간 합 (busy-poll 포함); 물리코어 수 아님. 역할별 분리는 thread_role_map 대조 필요 (프로세스 집계)"
 },
 "pcm": {
  "invalid_rows": 0,
  "interval_contract": "ASSUMED_END_TIMESTAMP (1 s)",
  "unit": "MB/s",
  "by_metric": {
   "System|Read": {
    "n_full": 32,
    "n_boundary": 2,
    "n_outside": 3,
    "n_duplicate": 0,
    "full_sample_coverage": 0.9878507248139416,
    "mean_full": 246117.3409375,
    "mean_overlap_est": 243520.06103216903,
    "mean_overlap_assumption": "each value represents a constant rate over its sample interval",
    "overlap_coverage": 0.999691248406059
   },
   "System|Write": {
    "n_full": 32,
    "n_boundary": 2,
    "n_outside": 3,
    "n_duplicate": 0,
    "full_sample_coverage": 0.9878507248139416,
    "mean_full": 21186.0603125,
    "mean_overlap_est": 20985.02722008366,
    "mean_overlap_assumption": "each value represents a constant rate over its sample interval",
    "overlap_coverage": 0.999691248406059
   },
   "System|Memory": {
    "n_full": 32,
    "n_boundary": 2,
    "n_outside": 3,
    "n_duplicate": 0,
    "full_sample_coverage": 0.9878507248139416,
    "mean_full": 267303.4025,
    "mean_overlap_est": 264505.0896058733,
    "mean_overlap_assumption": "each value represents a constant rate over its sample interval",
    "overlap_coverage": 0.999691248406059
   }
  }
 }
}
```

### S20_RESOURCE_pinned

```
{
 "session": "S20_RESOURCE_pinned",
 "window": {
  "name": "forward_envelope(kt_evt)",
  "start_ns": 1789633471691072422,
  "end_ns": 1789633504405422984,
  "len_s": 32.714350562,
  "status": "OK"
 },
 "client_process_window_s": 32.80924088298343,
 "benchmark_duration_s": 32.717810269,
 "perf": {
  "intervals_total": 40,
  "full": 31,
  "boundary": 2,
  "outside": 7,
  "bad_rows": 0,
  "full_window_s": 31.0,
  "task_clock_msec_sum_full": 3191946.4100000006,
  "cpu_equivalents_full": 102.96601322580646,
  "cycles_sum": 6358614518172.0,
  "instructions_sum": 10252438886674.0,
  "ipc_full": 1.612369936465564,
  "context_switches_sum": 103182.0,
  "cpu_migrations_sum": 13264.0,
  "cache_misses_sum": 119710433133.0,
  "scope": "스케줄러 4 PID 프로세스 전체(스레드 상속)",
  "target_pids": "171924,172101,172168,172332",
  "interval_contract": "perf -I 1000: 각 행 = 직전 1 s 구간 (상대 초, 시작 = collector actual_start)",
  "note": "task-clock 은 논리 CPU 시간 합 (busy-poll 포함); 물리코어 수 아님. 역할별 분리는 thread_role_map 대조 필요 (프로세스 집계)"
 },
 "pcm": {
  "invalid_rows": 0,
  "interval_contract": "ASSUMED_END_TIMESTAMP (1 s)",
  "unit": "MB/s",
  "by_metric": {
   "System|Read": {
    "n_full": 32,
    "n_boundary": 2,
    "n_outside": 3,
    "n_duplicate": 0,
    "full_sample_coverage": 0.9779805941543973,
    "mean_full": 246626.3359375,
    "mean_overlap_est": 242102.65912386714,
    "mean_overlap_assumption": "each value represents a constant rate over its sample interval",
    "overlap_coverage": 0.9996331885000359
   },
   "System|Write": {
    "n_full": 32,
    "n_boundary": 2,
    "n_outside": 3,
    "n_duplicate": 0,
    "full_sample_coverage": 0.9779805941543973,
    "mean_full": 21604.0821875,
    "mean_overlap_est": 21210.051861066833,
    "mean_overlap_assumption": "each value represents a constant rate over its sample interval",
    "overlap_coverage": 0.9996331885000359
   },
   "System|Memory": {
    "n_full": 32,
    "n_boundary": 2,
    "n_outside": 3,
    "n_duplicate": 0,
    "full_sample_coverage": 0.9779805941543973,
    "mean_full": 268230.4209375,
    "mean_overlap_est": 263312.7138540554,
    "mean_overlap_assumption": "each value represents a constant rate over its sample interval",
    "overlap_coverage": 0.9996331885000359
   }
  }
 }
}
```

### S24_RESOURCE_vllmw

```
{
 "session": "S24_RESOURCE_vllmw",
 "window": {
  "name": "forward_envelope(kt_evt)",
  "start_ns": 1789635400351205301,
  "end_ns": 1789635423729823076,
  "len_s": 23.378617775,
  "status": "OK"
 },
 "client_process_window_s": 23.830283408984542,
 "benchmark_duration_s": 23.41383522003889,
 "perf": {
  "intervals_total": 39,
  "full": 22,
  "boundary": 2,
  "outside": 15,
  "bad_rows": 0,
  "full_window_s": 22.0,
  "task_clock_msec_sum_full": 2261570.77,
  "cpu_equivalents_full": 102.79867136363636,
  "cycles_sum": 4502612338534.0,
  "instructions_sum": 5707569965608.0,
  "ipc_full": 1.267613006956384,
  "context_switches_sum": 79175.0,
  "cpu_migrations_sum": 8948.0,
  "cache_misses_sum": 81875970003.0,
  "scope": "스케줄러 4 PID 프로세스 전체(스레드 상속)",
  "target_pids": "415714,415784,415871,416077",
  "interval_contract": "perf -I 1000: 각 행 = 직전 1 s 구간 (상대 초, 시작 = collector actual_start)",
  "note": "task-clock 은 논리 CPU 시간 합 (busy-poll 포함); 물리코어 수 아님. 역할별 분리는 thread_role_map 대조 필요 (프로세스 집계)"
 },
 "pcm": {
  "invalid_rows": 0,
  "interval_contract": "ASSUMED_END_TIMESTAMP (1 s)",
  "unit": "MB/s",
  "by_metric": {
   "System|Read": {
    "n_full": 22,
    "n_boundary": 2,
    "n_outside": 13,
    "n_duplicate": 0,
    "full_sample_coverage": 0.94098805206203,
    "mean_full": 244905.05727272728,
    "mean_overlap_est": 233584.8124047965,
    "mean_overlap_assumption": "each value represents a constant rate over its sample interval",
    "overlap_coverage": 0.9997433552292208
   },
   "System|Write": {
    "n_full": 22,
    "n_boundary": 2,
    "n_outside": 13,
    "n_duplicate": 0,
    "full_sample_coverage": 0.94098805206203,
    "mean_full": 28870.12318181818,
    "mean_overlap_est": 27530.38797220124,
    "mean_overlap_assumption": "each value represents a constant rate over its sample interval",
    "overlap_coverage": 0.9997433552292208
   },
   "System|Memory": {
    "n_full": 22,
    "n_boundary": 2,
    "n_outside": 13,
    "n_duplicate": 0,
    "full_sample_coverage": 0.94098805206203,
    "mean_full": 273775.18272727274,
    "mean_overlap_est": 261115.20251616172,
    "mean_overlap_assumption": "each value represents a constant rate over its sample interval",
    "overlap_coverage": 0.9997433552292208
   }
  }
 }
}
```

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

## 8b. 확장 단계 (사용자 승인; 원장 phase=extended) — 클라이언트 동등성·TP collective·prefill·TID 역할·tail off-CPU·expert 표본

### 클라이언트 동등성 (OFF 세션, 같은 부팅 안 연속 실행; PROBE128 C=64)

| boot | session | client | tps | TTFT p50/p95 | TPOT p50/p95 | 같은 부팅 vllm 대비 tps 비율 |
|---|---|---|---|---|---|---|
| OFF_CLOSE2_165803 | S15_OFF_CLOSE2 | vllm | 716.4 | 2363/3212 | 61.4/72.0 | 1.000 |
| OFF_CLOSE2_165803 | S15b_OFF_CLOSE2_probe | probe | 509.9 | 2579/3123 | 97.6/107.6 | 0.712 |
| OFF_X_171100 | S16_OFF_X_vllm | vllm | 693.5 | 2458/3502 | 65.6/72.1 | 1.000 |
| OFF_X_171100 | S16b_OFF_X_probe_pinned | probe | 410.6 | 2610/3084 | 115.0/160.0 | 0.592 |
| OFF_X_171100 | S16c_OFF_X_vllm2 | vllm | 788.6 | 2126/3105 | 56.4/68.8 | 1.137 |
| OFF_Y_174421 | S21_OFF_Y_vllm | vllm | 705.1 | 2283/3191 | 63.0/73.5 | 1.000 |
| OFF_Y_174421 | S21b_OFF_Y_vllmw | vllmw | 703.1 | 2193/3084 | 63.9/74.9 | 0.997 |
| OFF_Y_174421 | S21c_OFF_Y_vllm2 | vllm | 705.8 | 2220/3054 | 65.3/71.6 | 1.001 |

판정: `vllmw`(= vllm bench serve 본체를 그대로 쓰고 benchmark() 진입 시 ready/start barrier + perf_counter↔REALTIME anchor 만 추가한 래퍼, `eval/ide075/vllm_bench_wrapper.py`) 는 vllm bench 와 tps 0.3% 이내 → **동등**. `probe`(aiohttp 자체 클라이언트) 는 핀 여부와 무관하게 서버 스텝을 늦춤 (tps 0.58~0.72 배, TPOT 1.5~1.8 배) → **비동등**; probe 로 얻은 CORE2/CORE3 의존성 값은 '클라이언트 간섭 하 관측' 으로 표기하고, CORE4(vllmw) 값을 대표값으로 쓴다.

### 클라이언트별 의존성 핵심값 비교 (bs=63 decode, cold_present_nonempty; p50 µs)

| session | client | n | cold_pub_delta | late share | enq→start | service | pre_h2d gap | mapping |
|---|---|---|---|---|---|---|---|---|
| S10_CORR | probe | 15493 ['step[DECODE bs=63]'] | 848.2 | 0.949 | 1178.8 | 1427.2 | 861.5 |  cov={'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} |
| S10_CORR | probe | 3061 ['step[DECODE bs=2]'] | -85.0 | 0.098 | 4.5 | 148.5 | 11.2 |  cov={'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} |
| S12_C1_CORR | probe | 46635 ['step[DECODE bs=1]'] | -59.2 | 0.186 | 4.0 | 148.8 | 11.5 |  cov={'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 126976} |
| S13_LONG_CORR | probe | 28213 ['step[DECODE bs=8]'] | 23.8 | 0.528 | 196.2 | 472.8 | 37.0 |  cov={'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} |
| S9_CORR | probe | 15541 ['step[DECODE bs=63]'] | 971.8 | 0.963 | 1309.2 | 1556.2 | 985.8 |  cov={'count_mismatch_pairs': 124, 'count_mismatch_rows': 24552, 'resync_unmatched_gpu': 24552, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} |
| S9_CORR | probe | 3061 ['step[DECODE bs=2]'] | -84.8 | 0.097 | 4.0 | 149.2 | 11.2 |  cov={'count_mismatch_pairs': 124, 'count_mismatch_rows': 24552, 'resync_unmatched_gpu': 24552, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} |
| S17_CORR_pinned | probe | 15493 ['step[DECODE bs=63]'] | 869.5 | 0.953 | 1199.2 | 1447.2 | 882.2 |  cov={'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} |
| S17_CORR_pinned | probe | 3061 ['step[DECODE bs=2]'] | -84.8 | 0.097 | 4.0 | 151.2 | 11.2 |  cov={'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} |
| S18_CORR_pinned | probe | 15493 ['step[DECODE bs=63]'] | 882.0 | 0.952 | 1210.0 | 1459.0 | 895.0 |  cov={'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} |
| S18_CORR_pinned | probe | 3061 ['step[DECODE bs=2]'] | -84.2 | 0.104 | 4.0 | 151.0 | 11.5 |  cov={'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} |
| S22_CORR_vllmw | vllmw | 15410 ['step[DECODE bs=63]'] | 392.0 | 0.916 | 670.5 | 921.8 | 337.2 |  cov={'count_mismatch_pairs': 86, 'count_mismatch_rows': 86, 'resync_unmatched_gpu': 216, 'resync_gpu_rows_without_dtoh': 1, 'resync_unmatched_cpu': 155, 'resync_pairs': 15717} |
| S22_CORR_vllmw | vllmw | 2463 ['step[DECODE bs=2]'] | -84.5 | 0.020 | 3.2 | 171.0 | 10.8 |  cov={'count_mismatch_pairs': 86, 'count_mismatch_rows': 86, 'resync_unmatched_gpu': 216, 'resync_gpu_rows_without_dtoh': 1, 'resync_unmatched_cpu': 155, 'resync_pairs': 15717} |
| S23_CORR_vllmw | vllmw | 15541 ['step[DECODE bs=63]'] | 328.2 | 0.911 | 667.5 | 915.8 | 341.2 |  cov={'count_mismatch_pairs': 78, 'count_mismatch_rows': 78, 'resync_gpu_rows_without_dtoh': 1, 'resync_unmatched_cpu': 54, 'resync_pairs': 15818, 'resync_unmatched_gpu': 115} |
| S23_CORR_vllmw | vllmw | 3006 ['step[DECODE bs=2]'] | -84.8 | 0.039 | 3.8 | 146.2 | 11.0 |  cov={'count_mismatch_pairs': 78, 'count_mismatch_rows': 78, 'resync_gpu_rows_without_dtoh': 1, 'resync_unmatched_cpu': 54, 'resync_pairs': 15818, 'resync_unmatched_gpu': 115} |
| S26_C1_CORR_vllmw | vllmw | 20646 ['step[DECODE bs=1]'] | -62.8 | 0.018 | 3.8 | 134.0 | 10.8 |  cov={} |
| S27_LONG_CORR_vllmw | vllmw | 23657 ['step[DECODE bs=8]'] | -180.8 | 0.030 | 3.8 | 245.8 | 10.5 |  cov={'count_mismatch_pairs': 12, 'count_mismatch_rows': 12, 'resync_gpu_rows_without_dtoh': 1, 'resync_unmatched_cpu': 56, 'resync_pairs': 31688, 'resync_unmatched_gpu': 55} |
| S3_CORR | vllm | 15598 ['step[DECODE bs=63]'] | 321.0 | 0.900 | 663.5 | 911.5 | 334.2 |  cov={} |
| S3_CORR | vllm | 2220 ['step[DECODE bs=2]'] | -85.5 | 0.015 | 4.0 | 144.0 | 10.8 |  cov={} |
| S4_CORR | vllm | 15596 ['step[DECODE bs=63]'] | 315.0 | 0.899 | 649.8 | 902.2 | 328.0 |  cov={} |
| S4_CORR | vllm | 2297 ['step[DECODE bs=2]'] | -83.5 | 0.017 | 3.5 | 143.0 | 10.8 |  cov={} |
| S29_CORR_vllmw_fp8fixso | vllmw | 15600 ['step[DECODE bs=63]'] | 335.0 | 0.911 | 673.0 | 921.0 | 348.5 |  cov={'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} |
| S29_CORR_vllmw_fp8fixso | vllmw | 2273 ['step[DECODE bs=2]'] | -83.8 | 0.017 | 3.5 | 142.5 | 10.8 |  cov={'count_mismatch_pairs': 62, 'count_mismatch_rows': 62, 'resync_unmatched_gpu': 62, 'resync_unmatched_cpu': 0, 'resync_pairs': 0} |

매핑: 계수 일치 (graph,layer) 는 순서 zip; 불일치(층당 1건, 세션 창 밖 replay) 는 anchor 재동기화(GPU dtoh_last_end 이후 첫 CPU t_go, 다음 replay 전; 5 µs 지터) 로 대응하고 남는 레코드는 `resync_unmatched_*` 로 계수. zip 이동 없음.

### TP collective 위치·rank (tp_collective_summary.json; 위치 = 직전 fused_moe 유무 휴리스틱)

| session | graph:position | step | n | dur p50 r0/r1/r2/r3 (µs) | 도착 시차 p50/p95 | 마지막 도착 rank |
|---|---|---|---|---|---|---|
| S10_CORR | 2:post_attention | step[DECODE bs=63] | 16128 | 16.2/13.8/13.5/13.5 | 3.0/4.2 | {'3': 7878, '0': 165, '2': 3254, '1': 4831} |
| S10_CORR | 2:post_moe | step[DECODE bs=63] | 15872 | 12.5/949.5/949.5/949.0 | 936.8/2316.0 | {'0': 15872} |
| S10_CORR | 32:post_attention | step[DECODE bs=2] | 8064 | 9.5/7.5/5.8/7.8 | 2.5/4.0 | {'2': 4654, '0': 37, '1': 2372, '3': 1001} |
| S10_CORR | 32:post_moe | step[DECODE bs=2] | 7936 | 6.0/64.5/58.2/64.0 | 58.0/71.8 | {'0': 7936} |
| S10_CORR | 35:post_attention | step[DECODE bs=1] | 63 | 9.2/8.0/5.8/8.2 | 2.8/4.8 | {'3': 4, '2': 50, '1': 7, '0': 2} |
| S10_CORR | 35:post_moe | step[DECODE bs=1] | 62 | 5.8/253.8/254.2/250.2 | 247.5/359.0 | {'0': 62} |
| S12_C1_CORR | 35:post_attention | step[DECODE bs=1] | 129087 | 8.5/8.2/5.5/8.2 | 3.0/4.8 | {'3': 6049, '2': 108603, '0': 7340, '1': 7095} |
| S12_C1_CORR | 35:post_moe | step[DECODE bs=1] | 127038 | 6.0/64.5/64.5/61.0 | 58.5/103.0 | {'0': 127038} |
| S13_LONG_CORR | 26:post_attention | step[DECODE bs=8] | 32256 | 10.2/7.2/7.5/7.5 | 3.2/4.8 | {'2': 5957, '1': 17916, '3': 8275, '0': 108} |
| S13_LONG_CORR | 26:post_moe | step[DECODE bs=8] | 31744 | 6.0/80.2/54.5/80.0 | 73.2/754.2 | {'0': 31744} |
| S13_LONG_CORR | 35:post_attention | step[DECODE bs=1] | 63 | 9.5/8.2/5.8/8.8 | 3.5/5.2 | {'1': 7, '2': 55, '0': 1} |
| S13_LONG_CORR | 35:post_moe | step[DECODE bs=1] | 62 | 5.8/244.5/243.8/241.0 | 238.0/369.0 | {'0': 62} |
| S9_CORR | 2:post_attention | step[DECODE bs=63] | 16128 | 16.2/13.8/13.5/13.5 | 2.8/4.2 | {'0': 142, '3': 5694, '2': 4928, '1': 5364} |
| S9_CORR | 2:post_moe | step[DECODE bs=63] | 15872 | 12.5/1076.5/1075.8/1076.0 | 1063.8/2394.5 | {'0': 15872} |
| S9_CORR | 20:post_attention | step[DECODE bs=16] | 16128 | 10.5/8.2/8.2/8.2 | 2.5/3.8 | {'0': 137, '1': 6396, '2': 4030, '3': 5565} |
| S9_CORR | 20:post_moe | step[DECODE bs=16] | 15872 | 7.0/75.2/23.5/75.2 | 67.5/79.8 | {'0': 15856, '2': 16} |
| S9_CORR | 32:post_attention | step[DECODE bs=2] | 8064 | 9.8/7.5/5.8/7.8 | 3.0/4.5 | {'3': 506, '0': 47, '2': 5495, '1': 2016} |
| S9_CORR | 32:post_moe | step[DECODE bs=2] | 7936 | 6.0/64.2/58.0/63.8 | 57.8/71.5 | {'0': 7936} |
| S9_CORR | 35:post_attention | step[DECODE bs=1] | 8820 | 9.0/8.0/5.5/8.2 | 3.2/5.2 | {'2': 7391, '1': 935, '0': 159, '3': 335} |
| S9_CORR | 35:post_moe | step[DECODE bs=1] | 8680 | 5.8/97.0/96.5/93.5 | 90.5/312.2 | {'0': 8680} |
| S17_CORR_pinned | 2:post_attention | step[DECODE bs=63] | 16128 | 16.2/13.8/13.5/13.5 | 3.2/4.8 | {'1': 4188, '0': 140, '2': 5722, '3': 6078} |
| S17_CORR_pinned | 2:post_moe | step[DECODE bs=63] | 15872 | 12.5/968.5/968.5/967.8 | 955.5/2339.0 | {'0': 15872} |
| S17_CORR_pinned | 32:post_attention | step[DECODE bs=2] | 8064 | 9.5/7.2/5.8/7.5 | 3.0/4.2 | {'2': 5048, '0': 57, '3': 540, '1': 2419} |
| S17_CORR_pinned | 32:post_moe | step[DECODE bs=2] | 7936 | 6.0/64.2/58.0/63.8 | 57.5/71.5 | {'0': 7936} |
| S17_CORR_pinned | 35:post_attention | step[DECODE bs=1] | 63 | 9.2/8.5/5.8/8.5 | 3.2/5.2 | {'3': 1, '2': 53, '1': 8, '0': 1} |
| S17_CORR_pinned | 35:post_moe | step[DECODE bs=1] | 62 | 5.8/262.0/263.0/259.8 | 256.2/372.0 | {'0': 62} |
| S18_CORR_pinned | 2:post_attention | step[DECODE bs=63] | 16066 | 16.2/13.8/13.5/13.5 | 103162.8/123079.2 | {'2': 7045, '1': 893, '0': 238, '3': 7890} |
| S18_CORR_pinned | 2:post_moe | step[DECODE bs=63] | 15809 | 12.5/980.0/985.2/984.8 | 104259.5/124546.2 | {'0': 15809} |
| S18_CORR_pinned | 32:post_attention | step[DECODE bs=2] | 8064 | 9.5/7.2/5.8/7.5 | 2.8/4.2 | {'1': 1901, '0': 95, '3': 886, '2': 5182} |
| S18_CORR_pinned | 32:post_moe | step[DECODE bs=2] | 7936 | 6.0/65.5/59.2/65.0 | 59.0/72.2 | {'0': 7936} |
| S18_CORR_pinned | 35:post_attention | step[DECODE bs=1] | 63 | 9.2/8.5/5.8/8.5 | 3.5/5.2 | {'2': 55, '1': 6, '3': 2} |
| S18_CORR_pinned | 35:post_moe | step[DECODE bs=1] | 62 | 5.8/261.0/262.0/259.5 | 257.5/369.0 | {'0': 62} |
| S22_CORR_vllmw | 2:post_attention | step[DECODE bs=63] | 16003 | 16.5/13.8/13.5/13.8 | 59274.0/68483.0 | {'0': 3602, '1': 316, '3': 5393, '2': 6692} |
| S22_CORR_vllmw | 2:post_moe | step[DECODE bs=63] | 15747 | 12.5/438.0/438.2/437.8 | 59246.2/68473.8 | {'0': 3546, '1': 308, '2': 5864, '3': 6029} |
| S22_CORR_vllmw | 32:post_attention | step[DECODE bs=2] | 8064 | 9.8/7.5/5.8/7.8 | 2.8/4.5 | {'1': 1854, '0': 42, '2': 5550, '3': 618} |
| S22_CORR_vllmw | 32:post_moe | step[DECODE bs=2] | 7936 | 6.0/63.8/57.8/63.5 | 57.5/67.5 | {'0': 7936} |
| S22_CORR_vllmw | 35:post_attention | step[DECODE bs=1] | 63 | 9.8/8.5/5.8/8.8 | 3.5/5.2 | {'3': 2, '2': 56, '1': 5} |
| S22_CORR_vllmw | 35:post_moe | step[DECODE bs=1] | 62 | 5.8/282.8/281.8/277.8 | 276.0/393.8 | {'0': 62} |
| S23_CORR_vllmw | 2:post_attention | step[DECODE bs=63] | 16065 | 16.5/13.5/13.5/13.8 | 3.8/65098.0 | {'3': 5394, '0': 1639, '1': 4074, '2': 4958} |
| S23_CORR_vllmw | 2:post_moe | step[DECODE bs=63] | 15810 | 12.5/442.2/441.8/442.0 | 706.5/64535.5 | {'0': 9618, '1': 1810, '2': 2134, '3': 2248} |
| S23_CORR_vllmw | 32:post_attention | step[DECODE bs=2] | 8064 | 9.8/7.5/5.8/7.8 | 2.8/4.2 | {'2': 5367, '0': 39, '3': 1087, '1': 1571} |
| S23_CORR_vllmw | 32:post_moe | step[DECODE bs=2] | 7936 | 6.0/64.8/58.8/64.5 | 58.5/69.8 | {'0': 7936} |
| S23_CORR_vllmw | 35:post_attention | step[DECODE bs=1] | 63 | 9.2/8.2/5.8/8.5 | 3.2/5.2 | {'2': 58, '1': 3, '3': 2} |
| S23_CORR_vllmw | 35:post_moe | step[DECODE bs=1] | 62 | 5.8/254.2/252.8/250.2 | 247.5/379.2 | {'0': 62} |
| S26_C1_CORR_vllmw | 35:post_attention | step[DECODE bs=1] | 129024 | 8.2/8.2/5.5/8.2 | 2.8/4.8 | {'2': 100872, '0': 11854, '3': 9899, '1': 6399} |
| S26_C1_CORR_vllmw | 35:post_moe | step[DECODE bs=1] | 126976 | 6.0/64.0/64.0/60.5 | 58.0/67.5 | {'0': 126976} |
| S27_LONG_CORR_vllmw | 26:post_attention | step[DECODE bs=8] | 32193 | 10.2/7.2/7.5/7.5 | 3.5/25728.5 | {'0': 5605, '1': 10935, '2': 6339, '3': 9314} |
| S27_LONG_CORR_vllmw | 26:post_moe | step[DECODE bs=8] | 31682 | 6.0/73.2/47.2/73.2 | 70.2/25679.2 | {'0': 21520, '1': 1, '2': 10161} |
| S3_CORR | 2:post_attention | step[DECODE bs=63] | 16066 | 16.5/13.8/13.5/13.8 | 4.2/68302.2 | {'2': 5498, '0': 213, '1': 5718, '3': 4637} |
| S3_CORR | 2:post_moe | step[DECODE bs=63] | 15809 | 12.5/437.0/436.8/437.0 | 577.0/68898.5 | {'0': 15809} |
| S3_CORR | 32:post_attention | step[DECODE bs=2] | 8064 | 9.8/7.5/5.8/7.8 | 3.0/4.5 | {'3': 764, '0': 75, '2': 5466, '1': 1759} |
| S3_CORR | 32:post_moe | step[DECODE bs=2] | 7936 | 6.0/64.2/58.2/64.0 | 57.8/68.0 | {'0': 7936} |
| S4_CORR | 2:post_attention | step[DECODE bs=63] | 16128 | 16.5/13.5/13.5/13.8 | 3.2/4.5 | {'3': 5138, '0': 207, '2': 5103, '1': 5680} |
| S4_CORR | 2:post_moe | step[DECODE bs=63] | 15872 | 12.5/429.8/429.5/429.8 | 417.0/928.2 | {'0': 15872} |
| S4_CORR | 32:post_attention | step[DECODE bs=2] | 8064 | 9.8/7.2/5.8/7.8 | 2.8/4.2 | {'3': 797, '0': 112, '2': 5161, '1': 1994} |
| S4_CORR | 32:post_moe | step[DECODE bs=2] | 7936 | 6.0/64.2/58.0/63.8 | 57.8/67.8 | {'0': 7936} |
| S29_CORR_vllmw_fp8fixso | 2:post_attention | step[DECODE bs=63] | 16128 | 16.5/13.5/13.5/13.8 | 3.2/4.5 | {'2': 5114, '0': 133, '1': 6700, '3': 4181} |
| S29_CORR_vllmw_fp8fixso | 2:post_moe | step[DECODE bs=63] | 15872 | 12.5/448.2/448.2/448.2 | 435.8/923.5 | {'0': 15872} |
| S29_CORR_vllmw_fp8fixso | 32:post_attention | step[DECODE bs=2] | 8064 | 9.8/7.5/5.8/7.8 | 3.0/4.5 | {'3': 513, '0': 13, '2': 5977, '1': 1561} |
| S29_CORR_vllmw_fp8fixso | 32:post_moe | step[DECODE bs=2] | 7936 | 6.0/64.2/58.2/63.8 | 57.8/67.8 | {'0': 7936} |
| S29_CORR_vllmw_fp8fixso | 35:post_attention | step[DECODE bs=1] | 63 | 9.8/8.2/5.8/8.5 | 3.5/6.8 | {'1': 4, '2': 59} |
| S29_CORR_vllmw_fp8fixso | 35:post_moe | step[DECODE bs=1] | 62 | 5.8/257.2/257.0/253.0 | 251.5/387.8 | {'0': 62} |

### prefill/EXTEND 층 (eager_summary.json; 층 = step 내 cold HtoD 순번, CPU 는 같은 step 창 eager 레코드 순)

| session | toks | n | cold_pub_delta p50 | pre_h2d gap p50 | service p50 | budget p50 | late share | AMX/AVX(numa0) |
|---|---|---|---|---|---|---|---|---|
| S10_CORR | 510 | 61 | -1533.0 | 12.5 | 1993.8 | 6638.8 | 0.000 | 54/0 |
| S10_CORR | 1018 | 61 | -1538.0 | 11.2 | 2778.8 | 6077.8 | 0.000 | 72/0 |
| S10_CORR | 6839 | 61 | -2592.2 | 11.8 | 9048.2 | 12553.2 | 0.016 | 132/0 |
| S10_CORR | 7218 | 61 | -2725.0 | 11.2 | 9285.5 | 13187.8 | 0.000 | 126/0 |
| S10_CORR | 8192 | 366 | -3070.5 | 12.0 | 10426.8 | 14827.5 | 0.041 | 135/0 |
| S10_CORR coverage | {"steps": 11, "gpu_layers": 620, "cpu_recs": 620, "step_count_mismatch": 1} | | | | | | | |
| S12_C1_CORR | 501 | 61 | -1385.5 | 11.8 | 2038.0 | 5843.0 | 0.000 | 54/0 |
| S12_C1_CORR | 502 | 61 | -1379.8 | 12.8 | 2142.8 | 5784.2 | 0.000 | 60/0 |
| S12_C1_CORR | 503 | 61 | -1460.8 | 13.0 | 2173.0 | 6242.0 | 0.000 | 60/0 |
| S12_C1_CORR | 504 | 183 | -1363.0 | 11.8 | 2114.5 | 5757.0 | 0.000 | 57/0 |
| S12_C1_CORR | 505 | 61 | -1355.5 | 11.5 | 1993.2 | 5703.0 | 0.000 | 54/0 |
| S12_C1_CORR | 506 | 61 | -1358.5 | 12.2 | 1606.8 | 5723.5 | 0.000 | 42/0 |
| S12_C1_CORR | 507 | 61 | -1369.8 | 12.2 | 2218.5 | 5744.0 | 0.000 | 63/0 |
| S12_C1_CORR | 508 | 244 | -1378.2 | 12.5 | 1973.8 | 5830.0 | 0.000 | 54/0 |
| S12_C1_CORR | 511 | 122 | -1463.2 | 12.8 | 2164.0 | 6206.0 | 0.000 | 60/0 |
| S12_C1_CORR | 512 | 61 | -1650.2 | 13.2 | 2332.0 | 6456.8 | 0.000 | 66/0 |
| S12_C1_CORR coverage | {"steps": 17, "gpu_layers": 992, "cpu_recs": 992, "step_count_mismatch": 1} | | | | | | | |
| S13_LONG_CORR | 4050 | 61 | 898.2 | 910.5 | 6817.2 | 8558.8 | 0.770 | 114/0 |
| S13_LONG_CORR | 4056 | 61 | 789.8 | 802.8 | 6668.5 | 8430.2 | 0.754 | 114/0 |
| S13_LONG_CORR | 4067 | 61 | 1072.2 | 1087.0 | 6979.5 | 8680.0 | 0.754 | 117/0 |
| S13_LONG_CORR | 4086 | 61 | -8.8 | 15.5 | 6997.2 | 9894.0 | 0.492 | 123/0 |
| S13_LONG_CORR | 4088 | 122 | 838.2 | 847.2 | 6781.0 | 8463.5 | 0.713 | 117/0 |
| S13_LONG_CORR | 8152 | 61 | -3015.5 | 12.0 | 11041.0 | 15379.2 | 0.098 | 132/0 |
| S13_LONG_CORR | 8180 | 61 | -3019.0 | 12.2 | 11012.2 | 15532.8 | 0.115 | 129/0 |
| S13_LONG_CORR | 8192 | 671 | -3018.2 | 12.0 | 11239.2 | 15562.0 | 0.130 | 132/0 |
| S13_LONG_CORR coverage | {"steps": 20, "gpu_layers": 1178, "cpu_recs": 1178, "step_count_mismatch": 1} | | | | | | | |
| S9_CORR | 510 | 61 | -1806.8 | 48.8 | 1996.8 | 8079.0 | 0.000 | 54/0 |
| S9_CORR | 1018 | 61 | -640.2 | 11.8 | 2787.5 | 6507.0 | 0.000 | 72/0 |
| S9_CORR | 6839 | 61 | -2552.0 | 12.8 | 9113.5 | 12514.2 | 0.295 | 132/0 |
| S9_CORR | 7218 | 61 | -2705.8 | 11.8 | 9332.2 | 13003.8 | 0.066 | 126/0 |
| S9_CORR | 8192 | 366 | -3053.8 | 12.0 | 10686.8 | 15101.5 | 0.115 | 138/0 |
| S9_CORR coverage | {"steps": 29, "gpu_layers": 1116, "cpu_recs": 620, "step_count_mismatch": 19} | | | | | | | |
| S17_CORR_pinned | 510 | 61 | -1360.5 | 12.8 | 2009.2 | 5764.8 | 0.000 | 54/0 |
| S17_CORR_pinned | 1018 | 61 | -1544.0 | 11.8 | 2751.2 | 6188.8 | 0.000 | 72/0 |
| S17_CORR_pinned | 6839 | 61 | -2538.5 | 14.0 | 9174.8 | 12655.0 | 0.328 | 132/0 |
| S17_CORR_pinned | 7218 | 61 | -2719.8 | 12.0 | 9314.0 | 12897.0 | 0.066 | 126/0 |
| S17_CORR_pinned | 8192 | 366 | -3056.2 | 11.8 | 10552.5 | 14990.0 | 0.016 | 135/0 |
| S17_CORR_pinned coverage | {"steps": 11, "gpu_layers": 620, "cpu_recs": 620, "step_count_mismatch": 1} | | | | | | | |
| S18_CORR_pinned | 510 | 61 | -1426.0 | 12.2 | 2013.8 | 6077.0 | 0.000 | 54/0 |
| S18_CORR_pinned | 1018 | 61 | -1530.8 | 11.8 | 2843.5 | 6153.5 | 0.000 | 72/0 |
| S18_CORR_pinned | 6839 | 61 | -2575.2 | 11.8 | 9045.2 | 12335.5 | 0.164 | 132/0 |
| S18_CORR_pinned | 7218 | 61 | -2724.8 | 11.8 | 9245.8 | 13086.8 | 0.000 | 126/0 |
| S18_CORR_pinned | 8192 | 366 | -3054.8 | 11.2 | 10524.5 | 14851.5 | 0.027 | 135/0 |
| S18_CORR_pinned coverage | {"steps": 11, "gpu_layers": 620, "cpu_recs": 620, "step_count_mismatch": 1} | | | | | | | |
| S22_CORR_vllmw | 510 | 61 | -1386.8 | 12.5 | 2191.0 | 5902.5 | 0.000 | 60/0 |
| S22_CORR_vllmw | 512 | 61 | -1498.5 | 12.8 | 1894.2 | 6417.5 | 0.000 | 51/0 |
| S22_CORR_vllmw | 1021 | 61 | -1515.0 | 11.8 | 2457.0 | 6094.2 | 0.000 | 63/0 |
| S22_CORR_vllmw | 7111 | 61 | -2704.8 | 12.0 | 8802.0 | 12587.2 | 0.000 | 117/0 |
| S22_CORR_vllmw | 7173 | 61 | -2702.8 | 11.5 | 8829.8 | 12792.5 | 0.016 | 114/0 |
| S22_CORR_vllmw | 8192 | 366 | -3055.0 | 12.0 | 9881.5 | 14609.5 | 0.003 | 123/0 |
| S22_CORR_vllmw coverage | {"steps": 12, "gpu_layers": 682, "cpu_recs": 682, "step_count_mismatch": 1} | | | | | | | |
| S23_CORR_vllmw | 510 | 61 | -495.2 | 12.2 | 2161.2 | 6168.0 | 0.000 | 60/0 |
| S23_CORR_vllmw | 1013 | 61 | -1517.8 | 11.8 | 2447.5 | 6079.0 | 0.000 | 63/0 |
| S23_CORR_vllmw | 1020 | 61 | -1556.8 | 11.8 | 2791.8 | 6175.2 | 0.000 | 72/0 |
| S23_CORR_vllmw | 6612 | 61 | -2516.5 | 12.0 | 8502.0 | 11908.8 | 0.000 | 120/0 |
| S23_CORR_vllmw | 7172 | 61 | -2716.2 | 11.8 | 8874.5 | 12839.8 | 0.000 | 114/0 |
| S23_CORR_vllmw | 8192 | 366 | -3074.2 | 11.8 | 9883.5 | 14499.2 | 0.000 | 123/0 |
| S23_CORR_vllmw coverage | {"steps": 12, "gpu_layers": 682, "cpu_recs": 682, "step_count_mismatch": 1} | | | | | | | |
| S26_C1_CORR_vllmw | 506 | 61 | -1401.2 | 12.0 | 1681.8 | 5975.0 | 0.000 | 45/0 |
| S26_C1_CORR_vllmw | 507 | 61 | -1420.0 | 12.8 | 1786.5 | 6008.0 | 0.000 | 48/0 |
| S26_C1_CORR_vllmw | 509 | 183 | -1399.5 | 12.0 | 1812.8 | 5923.5 | 0.000 | 48/0 |
| S26_C1_CORR_vllmw | 510 | 61 | -1395.2 | 11.8 | 1592.5 | 5926.5 | 0.000 | 42/0 |
| S26_C1_CORR_vllmw | 511 | 61 | -503.5 | 12.2 | 2285.0 | 6208.8 | 0.000 | 63/0 |
| S26_C1_CORR_vllmw | 512 | 122 | -1938.8 | 82.2 | 1842.5 | 8582.0 | 0.000 | 48/0 |
| S26_C1_CORR_vllmw | 513 | 183 | -1417.0 | 13.8 | 1701.0 | 6006.8 | 0.000 | 45/0 |
| S26_C1_CORR_vllmw | 514 | 122 | -1302.2 | 12.2 | 1787.8 | 5827.2 | 0.000 | 45/0 |
| S26_C1_CORR_vllmw | 516 | 61 | -1417.5 | 12.8 | 1696.8 | 6044.8 | 0.000 | 45/0 |
| S26_C1_CORR_vllmw | 517 | 61 | -1425.5 | 11.8 | 2037.8 | 5978.0 | 0.000 | 57/0 |
| S26_C1_CORR_vllmw coverage | {"steps": 16, "gpu_layers": 992, "cpu_recs": 992} | | | | | | | |
| S27_LONG_CORR_vllmw | 4077 | 61 | 509.0 | 524.2 | 6664.2 | 8228.5 | 0.754 | 114/0 |
| S27_LONG_CORR_vllmw | 4083 | 61 | 492.0 | 508.0 | 6613.2 | 8317.0 | 0.754 | 114/0 |
| S27_LONG_CORR_vllmw | 4084 | 61 | 671.5 | 678.8 | 6755.2 | 8441.5 | 0.705 | 117/0 |
| S27_LONG_CORR_vllmw | 4096 | 61 | 142.2 | 153.2 | 6596.8 | 8511.0 | 0.557 | 117/0 |
| S27_LONG_CORR_vllmw | 4098 | 183 | 509.2 | 525.5 | 6651.8 | 8303.2 | 0.689 | 120/0 |
| S27_LONG_CORR_vllmw | 4105 | 61 | -55.0 | 14.5 | 6620.8 | 8247.0 | 0.492 | 117/0 |
| S27_LONG_CORR_vllmw | 8192 | 732 | -3027.8 | 11.5 | 10700.5 | 15371.2 | 0.010 | 132/0 |
| S27_LONG_CORR_vllmw coverage | {"steps": 20, "gpu_layers": 1240, "cpu_recs": 1240} | | | | | | | |
| S3_CORR | 510 | 61 | -1346.5 | 12.2 | 2134.5 | 5652.5 | 0.000 | 60/0 |
| S3_CORR | 1013 | 61 | -1487.5 | 11.5 | 2475.5 | 5891.8 | 0.000 | 63/0 |
| S3_CORR | 2553 | 61 | -1763.2 | 10.0 | 4248.2 | 7087.5 | 0.000 | 93/0 |
| S3_CORR | 5065 | 61 | -1986.5 | 12.5 | 6821.0 | 9286.5 | 0.279 | 111/0 |
| S3_CORR | 7170 | 61 | -2726.2 | 11.5 | 8706.0 | 12565.0 | 0.000 | 117/0 |
| S3_CORR | 8192 | 366 | -3075.0 | 11.8 | 9732.0 | 14368.2 | 0.000 | 123/0 |
| S3_CORR coverage | {"steps": 11, "gpu_layers": 682, "cpu_recs": 682} | | | | | | | |
| S4_CORR | 510 | 61 | -495.0 | 11.8 | 2191.8 | 6241.0 | 0.000 | 60/0 |
| S4_CORR | 512 | 61 | -1545.0 | 12.8 | 1830.5 | 6665.8 | 0.000 | 51/0 |
| S4_CORR | 1021 | 61 | -1560.8 | 12.0 | 2584.5 | 6217.8 | 0.000 | 66/0 |
| S4_CORR | 7109 | 61 | -2695.5 | 12.2 | 8719.5 | 12532.8 | 0.000 | 120/0 |
| S4_CORR | 7172 | 61 | -2671.5 | 11.8 | 8803.5 | 12588.0 | 0.049 | 114/0 |
| S4_CORR | 8192 | 366 | -3066.5 | 11.5 | 9745.8 | 14340.8 | 0.000 | 123/0 |
| S4_CORR coverage | {"steps": 11, "gpu_layers": 682, "cpu_recs": 682} | | | | | | | |
| S29_CORR_vllmw_fp8fixso | 510 | 61 | -1946.8 | 80.8 | 2230.5 | 8623.0 | 0.000 | 60/0 |
| S29_CORR_vllmw_fp8fixso | 1018 | 61 | -1928.2 | 11.5 | 2442.0 | 7822.5 | 0.000 | 63/0 |
| S29_CORR_vllmw_fp8fixso | 2554 | 61 | -1773.2 | 11.0 | 4634.5 | 7239.5 | 0.000 | 99/0 |
| S29_CORR_vllmw_fp8fixso | 5069 | 61 | -1962.5 | 12.8 | 7032.5 | 9322.5 | 0.311 | 114/0 |
| S29_CORR_vllmw_fp8fixso | 7173 | 61 | -2726.2 | 11.2 | 8913.0 | 12861.5 | 0.000 | 117/0 |
| S29_CORR_vllmw_fp8fixso | 8192 | 366 | -3074.2 | 11.8 | 9964.8 | 14548.0 | 0.011 | 123/0 |
| S29_CORR_vllmw_fp8fixso coverage | {"steps": 12, "gpu_layers": 682, "cpu_recs": 682, "step_count_mismatch": 1} | | | | | | | |

### TID 역할별 CPU 시간 (cpu_time_by_role.json; 부하 창, /proc stat 1 s)

| session | role | tids | cpu_equivalents | comms |
|---|---|---|---|---|
| S11_RESOURCE | unknown | 162 | 0.07 | {"pt_gloo_runloop": 48, "pt_nccl_watchdg": 28, "pt_nccl_heartbt": 28, "gloo_tcp_ |
| S11_RESOURCE | python/tokenizer | 995 | 0.07 | {"python3": 995} |
| S11_RESOURCE | scheduler_main_or_child | 60 | 3.88 | {"sglang::schedul": 60} |
| S11_RESOURCE | KT_NUMA_worker | 96 | 94.74 | {"numa_0_t_1": 1, "numa_0_t_2": 1, "numa_0_t_3": 1, "numa_0_t_4": 1, "numa_0_t_5 |
| S11_RESOURCE | task_worker | 1 | 0.94 | {"kt-task-worker": 1} |
| S11_RESOURCE | observer_flusher | 1 | 0.01 | {"kt-evt-flush": 1} |
| S11_RESOURCE | poller | 1 | 0.99 | {"kt-cf-poll": 1} |
| S14_FOCUS_sched | unknown | 162 | 0.06 | {"pt_gloo_runloop": 48, "pt_nccl_watchdg": 28, "pt_nccl_heartbt": 28, "gloo_tcp_ |
| S14_FOCUS_sched | python/tokenizer | 995 | 0.06 | {"python3": 995} |
| S14_FOCUS_sched | scheduler_main_or_child | 60 | 3.88 | {"sglang::schedul": 60} |
| S14_FOCUS_sched | KT_NUMA_worker | 96 | 94.90 | {"numa_0_t_1": 1, "numa_0_t_2": 1, "numa_0_t_3": 1, "numa_0_t_4": 1, "numa_0_t_5 |
| S14_FOCUS_sched | task_worker | 1 | 0.94 | {"kt-task-worker": 1} |
| S14_FOCUS_sched | observer_flusher | 1 | 0.01 | {"kt-evt-flush": 1} |
| S14_FOCUS_sched | poller | 1 | 0.99 | {"kt-cf-poll": 1} |
| S19_FOCUS_pinned | unknown | 162 | 0.06 | {"pt_gloo_runloop": 48, "pt_nccl_watchdg": 28, "pt_nccl_heartbt": 28, "gloo_tcp_ |
| S19_FOCUS_pinned | python/tokenizer | 995 | 0.07 | {"python3": 995} |
| S19_FOCUS_pinned | scheduler_main_or_child | 60 | 3.87 | {"sglang::schedul": 60} |
| S19_FOCUS_pinned | KT_NUMA_worker | 96 | 94.79 | {"numa_0_t_1": 1, "numa_0_t_2": 1, "numa_0_t_3": 1, "numa_0_t_4": 1, "numa_0_t_5 |
| S19_FOCUS_pinned | task_worker | 1 | 0.94 | {"kt-task-worker": 1} |
| S19_FOCUS_pinned | observer_flusher | 1 | 0.01 | {"kt-evt-flush": 1} |
| S19_FOCUS_pinned | poller | 1 | 0.99 | {"kt-cf-poll": 1} |
| S20_RESOURCE_pinned | unknown | 162 | 0.07 | {"pt_gloo_runloop": 48, "pt_nccl_watchdg": 28, "pt_nccl_heartbt": 28, "gloo_tcp_ |
| S20_RESOURCE_pinned | python/tokenizer | 995 | 0.07 | {"python3": 995} |
| S20_RESOURCE_pinned | scheduler_main_or_child | 60 | 3.87 | {"sglang::schedul": 60} |
| S20_RESOURCE_pinned | KT_NUMA_worker | 96 | 94.90 | {"numa_0_t_1": 1, "numa_0_t_2": 1, "numa_0_t_3": 1, "numa_0_t_4": 1, "numa_0_t_5 |
| S20_RESOURCE_pinned | task_worker | 1 | 0.94 | {"kt-task-worker": 1} |
| S20_RESOURCE_pinned | observer_flusher | 1 | 0.01 | {"kt-evt-flush": 1} |
| S20_RESOURCE_pinned | poller | 1 | 0.99 | {"kt-cf-poll": 1} |
| S24_RESOURCE_vllmw | unknown | 162 | 0.09 | {"pt_gloo_runloop": 48, "pt_nccl_watchdg": 28, "pt_nccl_heartbt": 28, "gloo_tcp_ |
| S24_RESOURCE_vllmw | python/tokenizer | 995 | 0.11 | {"python3": 995} |
| S24_RESOURCE_vllmw | scheduler_main_or_child | 60 | 3.81 | {"sglang::schedul": 60} |
| S24_RESOURCE_vllmw | KT_NUMA_worker | 96 | 94.56 | {"numa_0_t_1": 1, "numa_0_t_2": 1, "numa_0_t_3": 1, "numa_0_t_4": 1, "numa_0_t_5 |
| S24_RESOURCE_vllmw | task_worker | 1 | 0.91 | {"kt-task-worker": 1} |
| S24_RESOURCE_vllmw | observer_flusher | 1 | 0.01 | {"kt-evt-flush": 1} |
| S24_RESOURCE_vllmw | poller | 1 | 0.99 | {"kt-cf-poll": 1} |
| S25_FOCUS_vllmw | unknown | 162 | 0.08 | {"pt_gloo_runloop": 48, "pt_nccl_watchdg": 28, "pt_nccl_heartbt": 28, "gloo_tcp_ |
| S25_FOCUS_vllmw | python/tokenizer | 995 | 0.10 | {"python3": 995} |
| S25_FOCUS_vllmw | scheduler_main_or_child | 60 | 3.81 | {"sglang::schedul": 60} |
| S25_FOCUS_vllmw | KT_NUMA_worker | 96 | 94.65 | {"numa_0_t_1": 1, "numa_0_t_2": 1, "numa_0_t_3": 1, "numa_0_t_4": 1, "numa_0_t_5 |
| S25_FOCUS_vllmw | task_worker | 1 | 0.91 | {"kt-task-worker": 1} |
| S25_FOCUS_vllmw | observer_flusher | 1 | 0.01 | {"kt-evt-flush": 1} |
| S25_FOCUS_vllmw | poller | 1 | 0.99 | {"kt-cf-poll": 1} |
| S28_FOCUS2_vllmw_syswide | unknown | 162 | 0.07 | {"pt_gloo_runloop": 48, "pt_nccl_watchdg": 28, "pt_nccl_heartbt": 28, "gloo_tcp_ |
| S28_FOCUS2_vllmw_syswide | python/tokenizer | 995 | 0.10 | {"python3": 995} |
| S28_FOCUS2_vllmw_syswide | scheduler_main_or_child | 48 | 3.80 | {"sglang::schedul": 48} |
| S28_FOCUS2_vllmw_syswide | KT_NUMA_worker | 96 | 94.40 | {"numa_0_t_1": 1, "numa_0_t_2": 1, "numa_0_t_3": 1, "numa_0_t_4": 1, "numa_0_t_5 |
| S28_FOCUS2_vllmw_syswide | task_worker | 1 | 0.91 | {"kt-task-worker": 1} |
| S28_FOCUS2_vllmw_syswide | observer_flusher | 1 | 0.01 | {"kt-evt-flush": 1} |
| S28_FOCUS2_vllmw_syswide | poller | 1 | 0.99 | {"kt-cf-poll": 1} |

### tail off-CPU (FOCUS; tail_offcpu_summary.json)

S14_FOCUS_sched: `{"decode_records": 23808, "prefill_records": 620, "tails": 1485, "tails_prefill(service>12ms|skew>1ms)": 31, "rows": 3032, "decode_tail_offcpu_sum_us_p50_p90_max": [0, 0, 0], "decode_tail_offcpu_max_us_p50_p90_max": [0, 0, 0], "decode_tail_rows_with_offcpu": 0, "decode_tail_rows_offcpu_gt_300us": 0, "prefill_tail_offcpu_max_us_max": 0, "rows_inside_perf_coverage": 96, "workers_identified": 96, "sched_switch_events": 200813, "perf_switch_window_realtime_ns": [1789631825116582559, 1789631833387721559], "perf_lost_samples_pct": ["9.86"], "perf_warnings": "Warning:\nProcessed 491689 events and lost 13 chunks!\n\nCheck IO/CPU overload!\n\nWarning:\nProcessed 539598 samples and lost 9.86%!", "work`

S19_FOCUS_pinned: `{"decode_records": 23808, "prefill_records": 682, "tails": 1564, "tails_prefill(service>12ms|skew>1ms)": 24, "rows": 3176, "decode_tail_offcpu_sum_us_p50_p90_max": [0, 0, 0], "decode_tail_offcpu_max_us_p50_p90_max": [0, 0, 0], "decode_tail_rows_with_offcpu": 0, "decode_tail_rows_offcpu_gt_300us": 0, "prefill_tail_offcpu_max_us_max": 0, "rows_inside_perf_coverage": 210, "workers_identified": 96, "sched_switch_events": 204094, "perf_switch_window_realtime_ns": [1789633427337087572, 1789633436448009572], "perf_lost_samples_pct": ["6.17"], "perf_warnings": "Warning:\nProcessed 504576 events and lost 14 chunks!\n\nCheck IO/CPU overload!\n\nWarning:\nProcessed 532073 samples and lost 6.17%!", "wor`

S25_FOCUS_vllmw: `{"decode_records": 23808, "prefill_records": 682, "tails": 44, "tails_prefill(service>12ms|skew>1ms)": 20, "rows": 128, "decode_tail_offcpu_sum_us_p50_p90_max": [0, 0, 0], "decode_tail_offcpu_max_us_p50_p90_max": [0, 0, 0], "decode_tail_rows_with_offcpu": 0, "decode_tail_rows_offcpu_gt_300us": 0, "prefill_tail_offcpu_max_us_max": 0, "rows_inside_perf_coverage": 42, "workers_identified": 96, "sched_switch_events": 203491, "perf_switch_window_realtime_ns": [1789635451664208477, 1789635475739075477], "perf_lost_samples_pct": ["40.54"], "perf_warnings": "Warning:\nProcessed 576574 events and lost 36 chunks!\n\nCheck IO/CPU overload!\n\nWarning:\nProcessed 959647 samples and lost 40.54%!", "worke`

S28_FOCUS2_vllmw_syswide: `{"decode_records": 23808, "prefill_records": 682, "tails": 31, "tails_prefill(service>12ms|skew>1ms)": 6, "rows": 74, "decode_tail_offcpu_sum_us_p50_p90_max": [0, 0, 284.0], "decode_tail_offcpu_max_us_p50_p90_max": [0, 0, 284.0], "decode_tail_rows_with_offcpu": 6, "decode_tail_rows_offcpu_gt_300us": 0, "prefill_tail_offcpu_max_us_max": 38.0, "rows_inside_perf_coverage": 62, "workers_identified": 96, "sched_switch_events": 10009122, "perf_switch_window_realtime_ns": [1789638157908375486, 1789638193980007486], "perf_lost_samples_pct": [], "perf_warnings": "", "worker_switch_outs_total": 4379, "worker_switch_ins_total": 4283, "worker_switch_out_states": {"S": 672, "R": 3707}, "worker_switch_out`

### expert 단위 rows 표본 (expert_rows_summary.json; v3, slot 별 1/16)

CORE2_162053: 표본 20716 (decode bs63 7948, prefill 636); decode n_unique p50 18, rows_max p50 7, expert rows 분포 {"64": 52, "1": 73476, "2": 23276, "3": 13838, "12": 974, "4": 9756, "5": 5338, "22": 38, "7": 2994, "10": 1190, "6": 3838, "8": 2030, "9": 1660, "14": 512, "11": 1158, "17": 294, "13": 622, "15": 434, rows≤2 비율 0.677; prefill n_unique p50 37, rows_max p50 55

CORE3_171618: 표본 14442 (decode bs63 7948, prefill 346); decode n_unique p50 17, rows_max p50 7, expert rows 분포 {"64": 52, "1": 72552, "2": 22160, "13": 632, "5": 5272, "3": 13184, "22": 40, "8": 2080, "7": 2992, "4": 9376, "9": 1648, "6": 3840, "16": 424, "11": 1144, "10": 1184, "14": 528, "12": 984, "19": 192, rows≤2 비율 0.677; prefill n_unique p50 41, rows_max p50 58

CORE4_174930: 표본 18714 (decode bs63 7948, prefill 650); decode n_unique p50 11, rows_max p50 3, expert rows 분포 {"64": 52, "1": 69374, "51": 2, "4": 1966, "8": 142, "3": 4866, "2": 13844, "5": 962, "6": 490, "11": 68, "10": 68, "9": 134, "26": 2, "19": 10, "21": 16, "22": 4, "35": 2, "7": 288, "20": 4, "15": 28, rows≤2 비율 0.899; prefill n_unique p50 34, rows_max p50 44

FOCUS2_183951: 표본 8344 (decode bs63 3980, prefill 200); decode n_unique p50 11, rows_max p50 3, expert rows 분포 {"64": 52, "1": 34966, "45": 4, "2": 6998, "6": 212, "3": 2468, "5": 514, "4": 934, "12": 24, "7": 170, "15": 14, "8": 114, "10": 46, "17": 2, "26": 4, "9": 66, "11": 32, "18": 8, "16": 4, "21": 6, "1, rows≤2 비율 0.899; prefill n_unique p50 36, rows_max p50 47

## 8c. G00 — GLM-4.7-FP8 (BASIC4) 정상 출력 진단·근본 원인·수정 (확장 단계)

| boot | 진단 | 구성 | 게이트 | 정답 | 출력 앞부분 |
|---|---|---|---|---|---|
| GLM_D1_170439 | — | gpu_experts=160 deferred=2 rsf 원본 | NOT_RUN (부팅 실패/OOM) | | |
| GLM_D2_170535 | D2_fixed | gpu_experts=80 deferred=2 rsf 수정 | BLOCKED_NORMAL_OUTPUT | 0/4 | We need for's eggs do, many to', which whiching:, with, A th |
| GLM_D3_172545 | D3_fixed_deferred0 | gpu_experts=80 deferred=0 rsf 수정 | BLOCKED_NORMAL_OUTPUT | 0/4 | Letable puzzle step the of or problem the Janint,ations and  |
| GLM_D4_173122 | D4_fixed_allcpu | gpu_experts=0 deferred=2 rsf 수정 | BLOCKED_NORMAL_OUTPUT | 0/4 | : the3 the for.,;y:{1/ 4 the, the(s or for for;4; modules fo |
| GLM_D5_173737 | D5_unfixed_allcpu_deferred0 | gpu_experts=0 deferred=0 rsf 원본 | BLOCKED_NORMAL_OUTPUT | 0/4 | :, case  module modules as44-basedunit for matrix and presen |
| GLM_D6_181159 | D6_dispatchfix_rsf | gpu_experts=80 deferred=2 rsf 수정 | GATE_PASS | 4/4 | Janet's ducks lay 16 eggs per day. She eats 3 eggs for break |
| GLM_D7_183516 | D7_dispatchfix_norsf | gpu_experts=80 deferred=2 rsf 원본 | GATE_PASS | 4/4 | First, we determine the total number of eggs laid by the duc |
| GLM_D8_185927 | D8_dispatchfix_rsf | gpu_experts=80 deferred=2 rsf 수정 | GATE_PASS | 4/4 | Janet's ducks lay 16 eggs per day. She eats 3 eggs for break |
| GLM_D9_194029 | D9_dispatchfix_rsf_cbfree | gpu_experts=80 deferred=2 rsf 수정 | GATE_PASS | 4/4 | Janet's ducks lay 16 eggs per day. She eats 3 eggs for break |

kt-kernel FP8 단독 테스트 (`examples/test_fp8_perchannel_moe.py`, `test_fp8_moe.py`): 수정 전 설치 .so(v1~v3)·원본 .so 모두 출력 0 (Mean relative L1 100%); upstream 6d460cc 클린 빌드 PASS 0.5829%. 파일 이식 bisect → `operators/amx/moe_base.hpp` 단독으로 재현.

근본 원인 (G00_code_review.md §8): IDE_046-b/IDE_051 hunk 3곳의 `if constexpr (requires(... from_mat_row/from_mat_block ...)) if (env) {...} else ORIGINAL;` — `else` 가 안쪽 `if` 에 결합해 ORIGINAL 도 `if constexpr` 본문에 들어감. FP8/FP8PerChannel/BF16 커널의 BufferA(`BufferABF16Impl`) 에는 해당 메서드가 없어 조건이 거짓 → 입력 gather·A 양자화·down A 양자화가 전부 컴파일에서 제거 → 출력 0. INT4(BufferAImpl) 는 조건 참 → Qwen 무영향. 앞서 확인한 `routed_scaling_factor` 결함(§4, GPU 기여에만 2.5 배 + decode 이중 적용) 은 독립 결함.

수정 후 (`eval/ide075/glm_fp8_dispatch_fix.py`, .so 재빌드):

```
== test_fp8_perchannel_moe.py (installed .so, fp8fix)
Mean relative L1 diff: 0.5829%
PASS: Mean error 0.5829% < 15.0% threshold
== test_fp8_moe.py (installed .so, fp8fix)
Mean relative L1 diff: 0.5829%
PASS: Mean error 0.5829% < 15.0% threshold

```

전 GPU 기준(IDE_073 G-GPU8, TP8, kt 없음) greedy 텍스트와의 공통 접두 길이 (문자; 완전 일치 = ref 길이):

| boot/tag | Q0 | Q1 | Q2 | Q3 | 완전 일치 수 |
|---|---|---|---|---|---|
| ref 길이 | 325 | 421 | 885 | 296 | 4 |
| GLM_D2_170535/D2_fixed | 0 | 0 | 1 | 6 | 0 |
| GLM_D3_172545/D3_fixed_deferred0 | 0 | 0 | 0 | 0 | 0 |
| GLM_D4_173122/D4_fixed_allcpu | 0 | 0 | 0 | 0 | 0 |
| GLM_D5_173737/D5_unfixed_allcpu_deferred0 | 0 | 0 | 0 | 0 | 0 |
| GLM_D6_181159/D6_dispatchfix_rsf | 129 | 421 | 885 | 296 | 3 |
| GLM_D7_183516/D7_dispatchfix_norsf | 0 | 1 | 0 | 21 | 0 |
| GLM_D8_185927/D8_dispatchfix_rsf | 129 | 421 | 885 | 296 | 3 |
| GLM_D9_194029/D9_dispatchfix_rsf_cbfree | 270 | 421 | 120 | 21 | 1 |

GLM 세션 (수정 .so v4 + rsf 패치; PROBE128 C=64 n=128). D6 은 `--model qwen480`(Qwen 토크나이저) 로 실행된 참고값, D8 은 glm47 토크나이저·KT_EVT 설정(BASIC4 는 callback-free 가 아니라 CPU 레코드 없음), D9 는 BASIC4 인자 + KT_CALLBACK_FREE/KT_CF_SKIP_EMPTY_IMM (CPU 기록기 동작; 전 GPU 기준 greedy 일치 1/4 로 D6/D8 의 3/4 과 다름):

| session | mode | client | tps | TTFT p50/p95 | TPOT p50/p95 | 의존성 (cold_present_nonempty p50: pub_delta / late share / enq→start / service) |
|---|---|---|---|---|---|---|
| GLM_D6_181159/G_CORR | CORR | vllmw | 43.3 | 49202/130282 | 1071.2/1215.8 | v2 없음 |
| GLM_D6_181159/G_OFF | OFF | vllm | 42.8 | 49023/131907 | 1086.0/1224.6 | v2 없음 |
| GLM_D8_185927/G_CORR2 | CORR | vllmw | 44.0 | 47470/129916 | 1061.0/1188.0 | v2 없음 |
| GLM_D8_185927/G_OFF2 | OFF | vllm | 44.1 | 47568/129471 | 1051.4/1182.4 | v2 없음 |
| GLM_D8_185927/G_OFF3 | OFF | vllm | 44.2 | 47605/129551 | 1054.0/1179.6 | v2 없음 |
| GLM_D9_194029/G_CORR3_cbfree | CORR | vllmw | 44.0 | 47622/129678 | 1055.2/1181.6 | v2 없음 |
| GLM_D9_194029/G_OFF4_cbfree | OFF | vllm | 43.9 | 47801/130207 | 1060.3/1186.2 | v2 없음 |

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
| boot_end GLM_D1_170439 | DIED  |
| GLM (G00) 기본 단계 | 실행 없음 (IDE_074 게이트 BLOCKED_NORMAL_OUTPUT 계승). 확장 단계에서 진단 D1~D7·근본 원인·수정 → §8c |
| S7 예비 | 하네스 결함으로 부팅 2회 소진 → 예비 없음 (기본 단계 BLOCKED_BY_BUDGET; 확장 단계에서 OFF_OPEN2/OFF_CLOSE2/CORE2~4 로 보완) |
| 기록기 v2 → v3 | v2 는 SPSC ring 경쟁으로 fwd 레코드 일부 누락 → v3 (Node 가 rec 포인터 보유, 스레드 이름 부여, expert 표본 ring) 로 교체. CORE2 이후 세션은 v3 |
| probe 클라이언트 | 비동등 (서버 스텝 지연) → CORE2/CORE3 값은 간섭 하 관측, CORE4(vllmw) 가 대표 |

## 11. 파일

- offline/: legacy_windows.csv, pcm_samples_v2.csv, pcm_window_summary_v2.csv, timestamp_parse_errors.jsonl, parser_test_results.json
- 세션/v2/: producer_consumer_metrics_v2.csv.gz, dependency_summary_v2.json, missing_coverage_v2.csv, queue_wait_breakdown.csv.gz, fifo_dependency_edges.csv.gz, producer_layer_costs.csv, expert_cost_samples.csv.gz, fifo_casebook.md, validation_results_v2.json, resource_summary.json
- 세션/: window_events.jsonl, observer_config.json, cpu_counter_intervals.csv, pcm_memory.raw.csv, requests.jsonl, 시계열
- 서버 보존: ~/.cache/huggingface/kt/ide075/<boot>/{kt_evt.csv, kt_evt.csv.tasks, kt_evt.csv.map, profiles/*.trace.json.gz} (ARTIFACT_INDEX.csv SHA256)
