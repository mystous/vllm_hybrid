# SUB_234 — [D21] TLB/hugepage — page walk 의 2차 트래픽

> **상태**: 대기 — 페이지 계층 | **parent**: `TSK_048` (`IDE_026`) | **수준**: 페이지/TLB
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

4KB 로 60MB+ 스트리밍 시 dTLB miss → page walk 자체가 메모리 트래픽. SMT sibling 은 STLB 공유 (SUB_215 와 상호작용). THP=madvise 실측.

## 가설 / 메커니즘

harvest 버퍼의 2MB THP 적용만으로 victim 간섭이 줄어든다 (walk 트래픽 절감).

## 실험 설계

{4KB, THP madvise 2MB, 1GB hugetlb} 3변형 + SUB_215 sibling 셀에서 STLB 오염 분리.

## 게이트

2MB 가 4KB 대비 victim p99 ≥3% 개선 또는 harvest +10%.

## 의존 / 비고

GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3c D21`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
