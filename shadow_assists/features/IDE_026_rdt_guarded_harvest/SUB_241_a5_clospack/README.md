# SUB_241 — [A5] CLOSPACK: 480-RMID 측정 주도 15-CLOS 패킹

> **상태**: 대기 — 신규 알고리즘 ⭐⭐ | **parent**: `TSK_048` (`IDE_026`) | **수준**: 제어 평면
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

모니터링 (480 RMID) ≫ 제어 (15 CLOS) 비대칭의 명시적 알고리즘화 — per-thread mon_group 상시 측정, '측정은 per-thread, 제어는 per-cluster'.

## 가설 / 메커니즘

(bw_l, bw_t−bw_l, occ, 변동성) k-means → occ/bw 점수 (재사용형=넓은 way / 스트리밍형=좁은 way+MBA / 원격형=RDT 무용→FERRY·DUTY 회부). 히스테리시스 2회.

## 실험 설계

합성판 (T1, 이질 워커) → vLLM 판 (T2, 스레드 ~220). 수동 2-CLOS 대비 harvest 처리량 + p99.

## 게이트

수동 2-CLOS 대비 harvest +10% (p99 동등), 재배정 ≤6회/h, 측정 오버헤드 <0.5%.

## 의존 / 비고

간섭 채널 분류학의 실행기. GPU 불요 (vLLM 판만 T2 무대).

## 참조

- 상세: `../ALGORITHMS.md A5`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
