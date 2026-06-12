# SUB_227 — [D14] partition-aware harvest 커널 + 간섭 효율 (IE) 지표

> **상태**: 대기 | **parent**: `TSK_048` (`IDE_026`) | **수준**: 캐시 전 계층
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

cache blocking ≤ CAT way 용량(4-way=60MB) / NT-store / THP madvise 선택 적용 / SW prefetch 거리 — 커널 작성 규칙별 효과 정량.

## 가설 / 메커니즘

같은 유용작업량에서 간섭을 덜 만드는 작성법이 존재하며 IE = 유용작업 ÷ victim p99 악화 % 로 서열화 가능.

## 실험 설계

규칙별 A/B (blocking 크기 sweep, NT on/off, THP on/off, PD sweep) — IE 지표 산출.

## 게이트

IE 가 규칙 간 ≥2× 차이 → portfolio 선택 기준 (SUB_228) 의 전제 성립.

## 의존 / 비고

GPU 불요. SUB_228 의 선행.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3c D14`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
