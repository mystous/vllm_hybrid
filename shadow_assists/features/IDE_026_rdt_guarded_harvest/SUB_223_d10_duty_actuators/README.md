# SUB_223 — [D10] duty-cycle actuator 3종 비교

> **상태**: 대기 | **parent**: `TSK_048` (`IDE_026`) | **수준**: 코어/BW
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

SCHED_DEADLINE(ms, CBS 코어 양보)/SIGSTOP·CONT(이식성 최고)/tpause(µs, 코어 점유 유지+L2 보존, umwait max 100k TSC 실측).

## 가설 / 메커니즘

tpause 는 context switch 없이 BW 만 끊어 재개 시 cold miss 가 없다 — 같은 duty 라도 actuator 별 보호력·유효 처리량이 다를 것.

## 실험 설계

동일 duty 50% 를 3 actuator 로 구현 × 주기 {100µs, 1ms, 10ms} sweep — victim p99 + harvest 처리량 + mbm.

## 게이트

duty 비율 → BW 비율 선형성 ±10% (T4 governor 제어성 전제).

## 의존 / 비고

GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3b D10`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
