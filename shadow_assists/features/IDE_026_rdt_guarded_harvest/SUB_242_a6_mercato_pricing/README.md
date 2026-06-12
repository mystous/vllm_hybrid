# SUB_242 — [A6] MERCATO: 혼잡 가격 기반 BW 시장

> **상태**: 대기 — 2차 알고리즘·경제학 ⭐ | **parent**: `TSK_048` (`IDE_026`) | **수준**: 메모리 BW 제어
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

BW 에 혼잡 가격 — governor 가 p←p·exp(γ·err) 지수 갱신 (movdiri 게시), 워커는 IE_i ≥ p 일 때만 소비. 혼잡 시 가치 낮은 작업부터 스스로 물러남.

## 가설 / 메커니즘

CSMA(공평 backoff) 와 같은 신호로 다른 목적함수 (utility 극대) — harvest 작업이 이질적일 때 (SUB_228) 총 유용가치 극대. 지수 갱신 = tatonnement 수렴.

## 실험 설계

이질 IE 3종 portfolio × 동일 victim p99 — 총 유용가치 vs CSMA.

## 게이트

동일 p99 에서 총 유용가치 ≥ CSMA +20%.

## 의존 / 비고

선행: SUB_227 (IE), SUB_238 (워커 기반). GPU 불요.

## 참조

- 상세: `../ALGORITHMS.md A6`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
