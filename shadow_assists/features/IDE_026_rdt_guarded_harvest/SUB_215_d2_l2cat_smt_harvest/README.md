# SUB_215 — [D2] L2 CAT 기반 SMT co-location harvest

> **상태**: ✅ 완료 — negative (범위 한정) (2026-06-12) | **parent**: `TSK_048` (`IDE_026`) | **수준**: L2 (코어/SMT)
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

L2 CAT 16-way/8 CLOS 실측 발견 (way=128KB, 도메인=물리코어, sibling 공유). 유휴 HT 112 thread = 최대 미개척 자원.

## 가설 / 메커니즘

harvest 를 serving 의 sibling HT 에 넣되 L2 를 12-4 분할하면 오염이 상한된다. MBA 는 D1 함정 때문에 끄고 CAT(L2+L3)만.

## 실험 설계

victim cpu0 + aggressor cpu112(sibling), {L2 CAT off / 12-4 / 14-2} × pointer-chase victim. p99 vs aggressor 처리량 frontier.

## 게이트

L2 12-4 에서 victim p99 악화 ≤10% AND aggressor ≥ 단독 HT 의 50%. 실패해도 negative 기록 가치.

## 의존 / 비고

선행 SUB_214 (연좌 규칙). 반나절. GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3 D2`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)

## ✅ 결과 (2026-06-12 — 70B × 7 corpus × 4셀, 셀별 fresh boot)

| 셀 | 기하평균 (S0 대비) | harvest BW |
|---|---:|---:|
| S1 MBA20 단독 | 0.972 | 16.2 GB/s |
| S2 +L2 CAT 12-4 | 0.981 | 15.8 GB/s |
| S3 +L2 CAT 14-2 | 0.947 | 15.9 GB/s |

**판정: 스트리밍형 harvest 에 L2 CAT 추가 효과 없음 (S1↔S2↔S3 차이가 noise band
±4~6pp 내) — MBA 단독 충분.** SUB_214 의 BW-지배 간섭 결론을 L2 계층에서 확증.
규칙: 스트리밍형 클러스터에 L2 way 배분 금지 (CLOSPACK 매핑 반영). 유보: L2-상주형
aggressor (branchy/tile) 는 미검증 — SUB_218 에서 후속. 상세: `MEASUREMENTS.md`
