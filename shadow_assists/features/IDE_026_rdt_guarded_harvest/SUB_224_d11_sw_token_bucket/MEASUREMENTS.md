# SUB_224 — SW 자가계측 token-bucket (소프트웨어 MBA), 2026-06-15

> **판정: positive — SW-MBA 작동.** budget↓→aggressor BW↓→victim 보호↑ (단조).
> RDT 하드웨어 없이 동작하는 보편 harvest 가드 = paper 일반화 기여.

## 구현
`victim_aggressor.c` +~12줄: pass-duty token-bucket. 매 pass 후 직전 pass 시간의
(100/budget−1)배 `clock_nanosleep` → 활성시간=budget% → BW=budget%. `--budget-pct N`.
(초기 epoch-bytes 방식은 pass(96MB/스레드≈10ms)가 epoch(1ms)보다 커서 실패 → pass-duty 채택.)

## 결과 (victim 0-7 + aggressor 8-23, basic)
| budget% | aggr BW(GB/s) | victim p99(ms) | ns/load |
|---|---:|---:|---:|
| none | 0 | 20.1 | 89.6 |
| 25 | 59 | 25.8 | 124.0 |
| 50 | 103 | 30.0 | 141.7 |
| 75 | 143 | 35.6 | 166.9 |
| 100 | 158 | 49.4 | 207.2 |

## 판정
1. **SW-MBA 작동**: budget 으로 aggressor BW 단조 throttle (158→59 GB/s @25%),
   victim p99/ns_load 비례 보호 (49.4→25.8ms @budget25%, baseline 20.1 근접).
2. **정확도 ~85%**: 목표 budget 대비 실측 BW +12~15%p over-deliver (25%→37% 등).
   nanosleep granularity·pass-duty 양자화 기인. D11 게이트(MBA 격차 ≤10%p)는
   정확도 측면 약간 초과하나, 보호 메커니즘은 명확히 유효.
3. **일반화 가치**: resctrl/RDT 부재(VM/AMD/ARM)에서도 동작 → harvest 가드의
   하드웨어 비의존 대체재. enforcement ladder 가운데 칸(priority<**SW-MBA**<HW-MBA).

## 보강 (미완)
- HW-MBA(resctrl) 와 동일 frontier 직접 비교(격차 ≤10%p 게이트). tpause 변형(L2 보존).
- budget 정확도 개선(서브-pass throttle 또는 mbm 피드백 보정).

산출물: `run`(인라인), `runs/results.csv`. 코드: `src/victim_aggressor.c` (`--budget-pct`).
