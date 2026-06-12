# SUB_233 — [D20] SW prefetch 거리 오토튜닝 (pointer-chase)

> **상태**: 대기 — 명령 마이크로 | **parent**: `TSK_048` (`IDE_026`) | **수준**: 레지스터/명령
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

의존 load 체인 (suffix tree walk) 은 HW prefetcher 사각. PD=⌈latency×BW/64B⌉≈28 초기값.

## 가설 / 메커니즘

깊이 k 소프트웨어 파이프라이닝 + PD 8~64 이분 탐색 (채택 기준 = IE). prefetchnta 변형 = 들어올 때 LLC 비충전 (cldemote 와 상보).

## 실험 설계

PD sweep × {t0, nta} — 탐색 처리량 + llc_occupancy.

## 게이트

오토튠 PD 가 기본 대비 처리량 +15% 또는 동일 처리량 LLC 점유 −30%.

## 의존 / 비고

vLLM 적용점: suffix tree walk C 확장 후보. GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3c D20`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
