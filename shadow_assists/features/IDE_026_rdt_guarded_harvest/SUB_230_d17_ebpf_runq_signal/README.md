# SUB_230 — [D17] eBPF run-queue latency 조기경보 (governor 입력)

> **상태**: 대기 | **parent**: `TSK_048` (`IDE_026`) | **수준**: OS/런타임
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

bpftrace + BTF 실측 가용 — serving TID 의 sched_wakeup→switch 지연 p99 를 100ms 창 집계 (오버헤드 ~1%).

## 가설 / 메커니즘

runq 지연은 GPU step-time 보다 10-100× 빠른 SLO 신호이며 vLLM 무수정. (메모리-지연 차원은 SUB_243 CANARY 가 보완)

## 실험 설계

T1 합성판에서 신호 민감도 (victim p99 악화를 몇 ms 만에 감지) 측정 → T4 governor 입력 채택 판정.

## 게이트

감지 지연 ≤ 2 epoch (20ms) AND 오탐율 <5%.

## 의존 / 비고

bpftrace 스크립트 ~30줄. 합성판 GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3b D17`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
