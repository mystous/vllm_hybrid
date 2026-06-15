# SUB_229 — 건설적 간섭 (constructive interference), 2026-06-15

> **판정: 유의 개선 없음 (탐색 실패, 비용 소).** cross-core prefetch helper 가 victim
> 포인터체이스 latency 를 낮추지 못함.

## 마이크로벤치 (공유 128MB 체인, victim cpu0 + helper cpu8, helper ON/OFF)
| helper | p50(ns) | p99(ns) | mean(ns) |
|---|---:|---:|---:|
| OFF | 40.7 / 40.3 | 132.5 / 123.7 | 51.5 / 48.7 |
| ON | 40.7 / 40.8 | 123.3 / 124.0 | 48.5 / 48.8 |

## 판정
- p50 동일(40.5ns), p99/mean 차이는 OFF 자체 run 변동(132.5 vs 123.7) 안의 노이즈.
  → **helper ON 의 유의한 개선 없음**.
- **구조적 이유**: 별 코어 helper 의 `__builtin_prefetch(p,0,1)` 는 helper 의 private
  L1/L2 로 prefetch → 다른 코어의 victim 은 그걸 못 씀. shared LLC 로 가는 prefetch 만
  도움 될 수 있으나, 데이터-의존 포인터체이스에서 helper 가 victim 보다 안정적으로
  앞서기 어려움. cross-core constructive prefetch 는 본질적 불리.
- 참고: SUB_228 에서 연산-bound 코러너가 victim 을 −3.3% (≈0, 약간 빠름) → "간섭이
  음수일 수 있다"의 약한 신호는 거기서 이미 관찰(코러너 중립). 능동 constructive
  prefetch 의 추가 이득은 없음.

## 비고
- 디버그: chase 가 -O3 에서 DCE 됨 → `asm volatile("":"+r"(p))` 배리어로 강제 후 측정.
- 탐색적 항목(README "실패해도 비용 소"). 결론: 간섭 스펙트럼을 음수로 확장하는
  능동 레버는 미발견.

산출물: `constructive_bench.c`.
