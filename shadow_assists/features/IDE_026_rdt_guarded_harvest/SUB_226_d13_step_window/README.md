# SUB_226 — [D13] GPU-step-window 정렬 harvest

> **상태**: ❌ 기각 (2026-06-12 범위 재정의) | **parent**: `TSK_048` (`IDE_026`) | **수준**: 시간 위상
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

GPU verify kernel 창 (SUB_162: vLLM threads 96-100% sleep) 안에서만 harvest quantum 발행하는 시간적 회피.

## 가설 / 메커니즘

신호원이 *GPU step 경계* — GPU 에 결박된 설계라 범위 밖, 기각.

## 실험 설계

(기각 — 실험 없음)

## 게이트

(기각)

## 의존 / 비고

대체 1: governor 시간 신호 = SUB_230 (eBPF runq). 대체 2: 위상 회피 아이디어 자체는 SUB_244 (LULL-SURF, MBM 신호) 로 부활.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3b D13`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
