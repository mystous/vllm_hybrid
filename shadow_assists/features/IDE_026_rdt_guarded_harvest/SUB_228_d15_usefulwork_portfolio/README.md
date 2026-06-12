# SUB_228 — [D15] useful-work portfolio (무엇을 harvest 하나)

> **상태**: 대기 — ⭐ | **parent**: `TSK_048` (`IDE_026`) | **수준**: 작업 설계
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

prefix index build / KV 압축·스캔 / IAA 오프로드 / AMX 보조모델 / xgrammar prefetch — 후보별 연산 특성·IE 가 전혀 다름.

## 가설 / 메커니즘

IE-순 portfolio + RDT 파티션 조합 권장표가 'CPU 를 뭘로 채우나' (C4 의 실용화) 에 답한다.

## 실험 설계

각 후보를 SUB_227 IE 지표로 측정 → IE 순위 + 파티션 권장 조합.

## 게이트

후보 ≥3 개가 IE 측정 완료 + 권장표 1매 (reviewer 'so what do you run?' 선제 답변).

## 의존 / 비고

선행 SUB_227. GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3b D15`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
