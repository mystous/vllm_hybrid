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

## ✅ 결과 (2026-06-12 — 70B × 7 corpus × 4셀)

| 셀 | serving (X0 대비) | harvest BW | IE |
|---|---:|---:|---:|
| X1 AVX-512 무가드 | 0.838 | 77.5 GB/s | 4.8 |
| X2 AMX 무가드 | 0.874 | 74.2 GB/s | 5.9 |
| X3 AMX+MBA20 | 0.958 | 11.7 GB/s | 2.8 |

**판정**: ① AMX↔AVX-512 차 +3.6pp = 게이트(>10%) 미달 → 이후 AVX-512 단일 패턴.
② MBA 는 AMX 에도 유효 (실효 15.8% — 패턴별 차이: AVX store 21.4%) → T1.5 LUT 에
패턴 축 필요. ③ 무가드 sibling 적자 −16% 수렴 (SUB_214 재현). 상세: `MEASUREMENTS.md`
