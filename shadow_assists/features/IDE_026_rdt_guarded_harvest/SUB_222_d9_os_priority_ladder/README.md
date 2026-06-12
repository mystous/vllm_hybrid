# SUB_222 — [D9] OS 우선순위 사다리 (RDT-無 baseline)

> **상태**: 대기 — 즉시 묶음 | **parent**: `TSK_048` (`IDE_026`) | **수준**: OS/런타임
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

SCHED_IDLE/SCHED_DEADLINE/cgroup v2 (cpu.weight, cpu.max) 전부 실측 가용.

## 가설 / 메커니즘

스케줄러는 CPU *시간* 만 나누고 메모리 BW 를 모름 — 코어 분리 설계에선 거의 무력할 것. 그 무력함의 실측이 RDT 필요성의 직접 논거.

## 실험 설계

T1 victim/aggressor 에서 aggressor 를 {기본, nice19, SCHED_IDLE, cpu.weight=1, cpu.max=50%} 5변형 → B2(RDT) 와 같은 frontier 플롯.

## 게이트

enforcement ladder 1칸 확정 (보호력 서열 데이터).

## 의존 / 비고

T1 셀 추가 (+30분). GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3b D9`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
