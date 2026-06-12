# SUB_240 — [A4] RELAY-Q: RFO-free 핸드오프 큐

> **상태**: 대기 — 신규 알고리즘 ⭐ | **parent**: `TSK_048` (`IDE_026`) | **수준**: 캐시라인 프로토콜
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

SPSC 에서 라인 소유권 강탈 0회 불변식 — 소형 store+cldemote (소비자 L3 히트 ~50ns) / 대형 NT-store 분기 (D19 와 동일 θ), movdiri 도어벨, umonitor/umwait 대기.

## 가설 / 메커니즘

전통 SPSC 는 라인당 소유권 이전 2회 — RELAY-Q 는 '소유권 없는 쓰기'(movdiri/NT) 와 '자발 양도'(cldemote) 만 사용 → 0회. SUB_231 의 구성적 해법.

## 실험 설계

마이크로벤치 — atomic head/tail SPSC 대비 메시지 지연 p99, 동일 처리량 L2 miss, 소비자 idle BW.

## 게이트

지연 p99 ≤70% AND L2 miss ≤50% AND idle BW = 0 (umwait 실측).

## 의존 / 비고

FERRY/MIRROR 의 내부 프리미티브. vLLM 적용점: detok 결과·ngram 전달. GPU 불요.

## 참조

- 상세: `../ALGORITHMS.md A4`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
