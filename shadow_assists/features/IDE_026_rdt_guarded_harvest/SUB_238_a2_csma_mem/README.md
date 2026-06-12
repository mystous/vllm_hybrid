# SUB_238 — [A2] CSMA-MEM: 분산 carrier-sense 메모리 BW 중재

> **상태**: 대기 — 신규 알고리즘 ⭐⭐ | **parent**: `TSK_048` (`IDE_026`) | **수준**: 메모리 BW 제어
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

메모리 버스 = 공유 매체. 각 워커가 rdtscp probe load 지연으로 혼잡 감지 (carrier sense) → randomized exponential backoff 를 tpause(C0.2, 트래픽·전력 0) 로.

## 가설 / 메커니즘

중앙 제어자·HW 노브 불요, UPI 에도 동작, MBA(10%) 보다 곱다. 'probe 는 누가 만든 혼잡이든 오른다' = 보수적 무해 우선. probe 비용 <0.1%.

## 실험 설계

T1 — 보호력 vs MBA {20,50}%, harvest 합산 처리량, 워커 수 {4,16,64} 공평성 (Jain).

## 게이트

보호력 MBA 동급 (±10%) AND harvest ≥ MBA 대비 +15% AND Jain ≥0.9.

## 의존 / 비고

선행: T1 IDLE 캘리브레이션. GPU 불요.

## 참조

- 상세: `../ALGORITHMS.md A2`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
