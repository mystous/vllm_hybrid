# SUB_229 — [D16] 건설적 간섭 (constructive interference)

> **상태**: 대기 — CPU-only 재범위화·탐색 | **parent**: `TSK_048` (`IDE_026`) | **수준**: LLC
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

간섭은 부호가 있다 — harvest 슬롯에 victim 을 *돕는* 작업 (working set 선행 prefetch) 을 넣으면 p99 가 개선될 수 있다.

## 가설 / 메커니즘

victim 의 pointer-chase 노드·memcpy 버퍼를 serving CLOS way 로 선행 prefetch 하는 harvest 변형은 음수가 아닌 양수 기여.

## 실험 설계

prefetch-harvest {ON,OFF} × CAT {off,on} — victim p99 의 통계 유의 개선 검정.

## 게이트

유의 개선 → 간섭 분류학을 부호 있는 스펙트럼으로 확장. 실패해도 비용 소.

## 의존 / 비고

합성판 GPU 불요 (vLLM 자료구조 판은 범위 외 후속).

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3b D16`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
