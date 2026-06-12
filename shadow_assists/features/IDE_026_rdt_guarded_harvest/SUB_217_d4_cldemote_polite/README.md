# SUB_217 — [D4] `cldemote` 협조적 aggressor (polite harvest)

> **상태**: ✅ 완료 — positive (2026-06-12) | **parent**: `TSK_048` (`IDE_026`) | **수준**: 캐시라인
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

cldemote 실측 지원 — 사용 완료 캐시라인을 L3→메모리 방향으로 자발 강등.

## 가설 / 메커니즘

harvest 가 소비 완료 라인을 즉시 cldemote 하면 CAT way 추가 없이 LLC 점유·간섭이 줄어든다 (NT-store 와 달리 read 스트림에도 적용).

## 실험 설계

aggressor 3변형 {기본, +cldemote, +NT-store} × CAT {off, 4-way} — llc_occupancy 시계열로 자발 강등 직접 관측.

## 게이트

CAT-off 에서 cldemote 변형의 victim p99 영향이 기본 대비 유의 감소 (간섭 α% 제거 정량).

## 의존 / 비고

dev 머신 (Alder Lake) cldemote 미지원 — prod 전용. GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3 D4`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)

## ✅ 결과 (2026-06-12 — 70B × 7 corpus × 4셀, 무가드 sibling, 셀별 fresh boot)

| 변형 | serving (A0 대비) | harvest BW | LLC 점유 | IE (GB/s÷손실pp) |
|---|---:|---:|---:|---:|
| A1 일반 store | 0.918 | 77.0 GB/s | 276 MB | 9.4 |
| **A2 cldemote** | **0.966** | 67.8 GB/s | 218 MB | **19.9 (2.1×)** |
| A3 NT-store | 0.797 | 109.9 GB/s | 279 MB | 5.4 |

**판정: D4 게이트 통과** — cldemote 가 무가드 적자의 59% 를 회복 (HW 노브 0).
**반전 발견**: NT-store 는 RFO 제거로 버스 압력 +43% → BW-지배 환경의 anti-pattern
(단독 사용 금지, BW 상한과 병행 필수). IE 지표 첫 실측 (D14/SUB_227 가동).
상세: `MEASUREMENTS.md`
