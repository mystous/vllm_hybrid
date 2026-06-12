# SUB_220 — [D7] 전력/주파수 간섭 채널 (RAPL·uncore)

> **상태**: 대기 — CPU-only 재범위화 | **parent**: `TSK_048` (`IDE_026`) | **수준**: 패키지 전력
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

uncore 0.8–2.4 GHz sysfs 제어 실측. AMX/AVX-512 heavy 부하는 RAPL 예산 소모 + uncore/mesh 주파수 요동.

## 가설 / 메커니즘

코어·캐시·BW 완전 격리 (B2) 후에도 전력 채널만으로 serving p99 악화 — RDT 로 차단 불가능한 간섭의 하한.

## 실험 설계

T1 B2 셀 + turbostat (PkgWatt/Bzy_MHz/UncMHz) 병행, uncore min 2.4GHz 고정 변형 셀로 회복분 분리. vLLM/GPU 불요.

## 게이트

격리 잔여 간섭 중 uncore-고정으로 회복되는 비율 정량 → 간섭 분류학 채널 ⑤ 실측.

## 의존 / 비고

GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3 D7`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
