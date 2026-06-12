# SUB_232 — [D19] 크기-적응 memcpy 디스패처 (FSRM/AVX-512/NT)

> **상태**: ✅ 완료 — negative + 원인 확정 (2026-06-12) | **parent**: `TSK_048` (`IDE_026`) | **수준**: 레지스터/명령
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

erms+fsrm 실측 ✓. SUB_201: 70B 류 host path 의 80% 가 memcpy-bound — memcpy 명령 선택이 곧 LLC 오염량.

## 가설 / 메커니즘

<4KB rep movsb / 중간 AVX-512 / >θ_NT vmovntdq+sfence (LLC 비오염). **θ_NT = CAT 파티션 크기의 함수** 가 신규성.

## 실험 설계

크기 sweep 1KB~256MB × 3경로 → crossover 곡선 + MBM occupancy 로 NT 의 LLC-비오염 직접 검증.

## 게이트

`memcpy_dispatch(size, cat_alloc)` 단일 함수 산출 — detok/KV-copy 경로에 삽입 가능 형태.

## 의존 / 비고

GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3c D19`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)

## ✅ 결과 (2026-06-12): negative — LD_PRELOAD 디스패처 무효 (M1 0.972 / M2 0.961)

**원인 실측**: steady-state libc memcpy = 13.6M 호출 × 평균 34 B, NT-급은 25회뿐 —
"host path 80% memcpy" 는 torch 내부 copy 커널 (libc 미경유). **개입 지점을 vLLM/torch
코드 내부로 옮겨야 함** → py-spy 재프로파일 후 재조준. 상세: `MEASUREMENTS.md`
