# SUB_237 — [A1] CC-CAT: AIMD elastic LLC way 할당

> **상태**: 대기 — 신규 알고리즘 ⭐⭐ | **parent**: `TSK_048` (`IDE_026`) | **수준**: LLC 제어
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

TCP 혼잡제어 (AIMD) 를 CAT way 윈도우에 이식 — 연속-mask 제약 (sparse_masks=0) 이 cwnd 동형성을 준다.

## 가설 / 메커니즘

무혼잡 K=5 epoch → +1 way / SLO 위반 → ½ + **cldemote 스윕으로 점유 감쇠 τ 단축** (CAT 축소가 즉효 아님을 보정 — 신규). τ 측정 자체가 신규 데이터.

## 실험 설계

T1 합성판 — 외란 (aggressor 급증) step 응답, 정상상태 진동, 정적 best-way oracle 대비 추종률.

## 게이트

정상 진동 ≤±1 way, 외란 후 3 epoch SLO 복귀, harvest ≥ oracle 의 90%.

## 의존 / 비고

선행: T1 정적 CAT 곡선, SUB_230/243 (신호). GPU 불요.

## 참조

- 상세: `../ALGORITHMS.md A1`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
