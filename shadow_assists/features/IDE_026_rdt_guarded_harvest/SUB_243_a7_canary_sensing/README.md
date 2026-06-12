# SUB_243 — [A7] CANARY: 카나리 기반 간섭 센싱

> **상태**: 대기 — 2차 알고리즘·센싱 ⭐⭐ | **parent**: `TSK_048` (`IDE_026`) | **수준**: 센싱/신호
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

serving CLOS *안에* 동거하는 초경량 pointer-chase 카나리 (duty 1%) — 파티션-관점 메모리-지연 p99 를 SLO 신호로. vLLM 무수정.

## 가설 / 메커니즘

eBPF runq 는 CPU-시간 차원만 — 카나리는 간섭의 자원 차원 (loaded latency) 을 직접 감지. CLOS-내 동거가 대표성의 핵심.

## 실험 설계

T1 — victim p99 와 카나리 신호의 상관, 간섭 감지 지연, duty 비용.

## 게이트

상관 ≥0.9 AND 감지 지연 ≤20ms AND 비용 <1% 코어.

## 의존 / 비고

A1/A2/A6 의 공용 신호원 후보. GPU 불요.

## 참조

- 상세: `../ALGORITHMS.md A7`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
