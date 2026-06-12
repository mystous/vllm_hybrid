# SUB_231 — [D18] false-sharing/캐시라인 레이아웃 감사

> **상태**: 대기 — 64B 마이크로 | **parent**: `TSK_048` (`IDE_026`) | **수준**: 캐시라인
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

코어 간 공유 구조체에서 다른 스레드가 쓰는 필드가 같은 64B 라인 (adjacent-prefetch 128B 쌍) 에 있으면 RFO ping-pong — CAT 으로 못 막는 라인 *소유권* 경합.

## 가설 / 메커니즘

per-core shard 카운터 + alignas(128) 분리로 ping-pong 제거 가능. vLLM 적용점: ngram 큐 인덱스, tempo 통계 카운터.

## 실험 설계

합성 재현 (같은/다른 라인 인접 필드 store 2스레드) → ping-pong 배율 p99 정량 → vLLM 구조체 감사.

## 게이트

합성 ping-pong ≥2× 악화 재현 → 감사 1회 가치 확정. 구성적 해법은 SUB_240 (RELAY-Q).

## 의존 / 비고

GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3c D18`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
