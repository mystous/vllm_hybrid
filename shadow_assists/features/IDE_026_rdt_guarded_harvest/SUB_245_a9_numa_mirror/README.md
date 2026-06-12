# SUB_245 — [A9] NUMA-MIRROR: 소켓별 복제 + RCU-식 epoch 발행

> **상태**: 대기 — 2차 알고리즘·복제 | **parent**: `TSK_048` (`IDE_026`) | **수준**: NUMA/일관성
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

읽기-다수 핫 구조 (룩업 테이블, suffix tree 상위) 를 노드별 미러 (MPOL_BIND) — 읽기 UPI 0·로컬 지연, 쓰기는 RELAY-Q 푸시 후 movdiri version bump.

## 가설 / 메커니즘

단조 버전 + 양 미러 발행 완료 후 bump → torn read 없음 (RCU-식 grace). 메모리 2배 비용은 DRAM 2TB 여유로 흡수. CLOSPACK 원격형 클러스터의 자동 회부처.

## 실험 설계

T1+libnuma — 원격 읽기 지배 워크로드에서 단일 사본 vs MIRROR.

## 게이트

처리량 ≥+40% AND mbm remote 분 ≥−80%.

## 의존 / 비고

선행: SUB_240 (RELAY-Q), SUB_241 (선별). GPU 불요.

## 참조

- 상세: `../ALGORITHMS.md A9`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
