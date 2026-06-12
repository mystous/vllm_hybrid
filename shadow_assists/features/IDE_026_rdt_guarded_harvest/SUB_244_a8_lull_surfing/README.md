# SUB_244 — [A8] LULL-SURF: MBM-위상 안티-페이즈 TDM

> **상태**: 대기 — D13 의 신호-준수 부활 | **parent**: `TSK_048` (`IDE_026`) | **수준**: 시간 위상
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

serving CLOS mbm_total 1ms 샘플로 트래픽 골 (lull) 감지 → harvest burst 를 골에만. 간섭 비용 = 순간 합산 BW 의 볼록함수 (iMC 큐잉) 이용.

## 가설 / 메커니즘

같은 평균 BW 라도 serving 피크와 겹치지 않게 재배치하면 p99 기여 감소. 신호 = MBM (비-GPU) — 기각된 SUB_226 의 준수 부활.

## 실험 설계

T1 (버스트형 victim) — 상시-균등 vs LULL-SURF, 동일 harvest 평균 BW.

## 게이트

victim p99 영향 ≤ 상시-균등의 60%.

## 의존 / 비고

선행: MBM 1ms 샘플링 오버헤드 검증. GPU 불요.

## 참조

- 상세: `../ALGORITHMS.md A8`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
