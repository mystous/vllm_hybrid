# SUB_218 — [D5] AMX-tile aggressor 추가

> **상태**: 대기 — T1 대표성 보강 | **parent**: `TSK_048` (`IDE_026`) | **수준**: 레지스터/명령
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

victim_aggressor.c 는 AVX-512 triad 만 — 실 harvest 후보 (AMX draft) 는 tile load 패턴으로 LLC 점유·prefetch 행태가 다름. R4 위험 대응.

## 가설 / 메커니즘

같은 GB/s 라도 AMX tile 스트리밍 GEMM 의 간섭 프로파일은 AVX-512 와 다를 것.

## 실험 설계

aggressor `--mode amx` (tile_loadd 스트리밍 GEMM) 추가 → B0~B2 재실행.

## 게이트

AVX-512 와 간섭 프로파일 차이 >10% 면 이후 모든 셀에 2패턴 유지.

## 의존 / 비고

GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3 D5`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
