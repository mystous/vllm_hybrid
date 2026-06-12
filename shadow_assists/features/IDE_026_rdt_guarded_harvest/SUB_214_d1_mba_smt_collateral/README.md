# SUB_214 — [D1] `thread_throttle_mode=max` SMT 연좌 throttle 정량화

> **상태**: 대기 — 즉시 묶음 | **parent**: `TSK_048` (`IDE_026`) | **수준**: 코어/SMT
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

MBA 는 코어 단위 적용 (실측 `info/MB/thread_throttle_mode=max`) — SMT sibling 이 다른 CLOS 면 가장 강한 throttle 이 물리 코어 전체에 걸린다.

## 가설 / 메커니즘

serving 과 MBA 20% harvest 가 같은 물리 코어의 HT 짝이면 serving 이 20% 로 연좌 throttle → p99 폭증.

## 실험 설계

victim cpu0 고정 + aggressor {cpu16(타 코어), cpu112(sibling)} × MBA {100,20}% = 4셀. 연좌 배율 곡선.

## 게이트

연좌 악화 ≥ +20% → 'vLLM 스레드 배치는 코어-배타 필수' 설계규칙 승격 (논문 §9 design rule).

## 의존 / 비고

T1 인프라 재사용 (+30분). GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3 D1`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
