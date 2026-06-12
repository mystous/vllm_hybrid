# SUB_221 — [D8] CDP code/data 분리

> **상태**: 대기 — 후순위 | **parent**: `TSK_048` (`IDE_026`) | **수준**: LLC
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

`cdp_l3` 실측 지원 (`mount -o cdp` 재마운트 필요 — CLOS 15→7 반감).

## 가설 / 메커니즘

detok/Python 경로는 코드 footprint 가 큼 — harvest 스트리밍의 코드 라인 evict 가 frontend stall 로 p99 악화. L3CODE 보호가 data 분할보다 효율적일 수 있음.

## 실험 설계

CDP 재마운트 후 {CODE 보호, DATA 보호, 둘 다} × aggressor — victim p99 비교.

## 게이트

CODE-보호 단독이 동등 way 예산의 data 분할보다 p99 우수 → 채택.

## 의존 / 비고

재마운트 필요 (기존 그룹 소실) — T1 완료 후 별도 세션. GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3 D8`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
