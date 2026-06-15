# SUB_220 — 전력/주파수 간섭 채널 (RAPL·uncore), 채널 ⑤ 정량 (2026-06-15)

> **판정: uncore-주파수 채널 negligible / uncore-pin 레버 무효.** BW(MBA) 가드는 유효.
> (positive 아님. mix 단일점 이상치에 주의 — 7-corpus 기하평균이 binding.)

## 1. 설계 (70B suffix K32, taskset 0-47,56-103, per-cell boot, turbostat 병행)
| 셀 | harvest aggressor | MBA | uncore min |
|---|---|---|---|
| P0_base | 없음 | — | (800MHz) |
| P1_guard | 16T @ sibling 112-119,168-175 | 20% | 800MHz(비고정) |
| P2_guard_pin | 16T @ sibling | 20% | **2.4GHz 고정** |
| P3_unguard_pin | 16T @ sibling | 100% | 2.4GHz 고정 |

## 2. 결과 — 7-corpus 기하평균 (binding)
| 셀 | gm tps | vs P0 | turbostat UncMHz | PkgWatt |
|---|---:|---:|---:|---:|
| P0_base | 4,651 | — | — | — |
| P1_guard | 4,378 | −5.9% | ~2400 | ~335 |
| P2_guard_pin | 4,376 | −5.9% | ~2400 | ~335 |
| P3_unguard_pin | 3,915 | −15.8% | ~2400 | ~357 |

per-corpus tps:
```
corpus      P0    P1    P2    P3
sharegpt   4682  4557  4488  4108
swebench   5338  5368  5043  4937
humaneval  4664  4250  4372  3722
mbpp       2661  2968  2510  2330
wildchat   5098  5172  4728  4399
lmsys      4309  4044  3814  3358
mix        6914  4780* 6867  5426   (*P1 mix = 이상치, mix/gm=1.09 vs P0 1.49)
```

## 3. 판정
1. **uncore 주파수 채널(⑤) = negligible**: P1(uncore 비고정)에서도 turbostat
   UncMHz~2400(최대) → harvest 16T(array 32MB, basic) 가 uncore freq 를 안 떨어뜨림.
   따라서 **uncore-pin(P2)이 P1 대비 회복 0** (둘 다 −5.9%). pin 레버 무효.
2. **BW(MBA) 가드는 유효**: MBA20(P1 −5.9%) vs MBA100(P3 −15.8%) → BW 격리가
   harvest 페널티를 −16%→−6% 로 절감 (잔여 −6% = LLC/코어 채널).
3. **mix 단일점 함정**: P1 mix(4780) 단일 측정이 이상치였음. 초기 "uncore-pin
   +44% 회복" 은 P1 mix outlier 의 산물 — 7-corpus 기하평균으로 반증.

## 4. 함의
- "RDT 가 못 막는 하한 = 전력/주파수" 가설은 **이 HW/harvest 강도에선 미성립**
  (uncore 가 알아서 max 유지). 더 강한 AMX/AVX-512 heavy harvest 라면 다를 수 있으나
  (aggr-mode=amx 변형 미시험), 현 basic harvest 에선 freq 채널 무의미.
- 실효 harvest 가드 = **MBA(BW)** 가 핵심. uncore-pin 은 불필요.

산출물: `run_sub220.sh`, `runs/` (4셀×7corpus + turbostat_*.log).
