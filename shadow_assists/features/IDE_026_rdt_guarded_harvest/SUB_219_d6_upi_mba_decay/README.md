# SUB_219 — [D6] UPI-aware 부하 분리 (MBA 무력화 곡선)

> **상태**: 대기 | **parent**: `TSK_048` (`IDE_026`) | **수준**: NUMA/UPI
> **등록**: 2026-06-12, `shadow_assists/id_registry.md` | **GPU**: 불요 (§0 범위 준수)

## 정의 / HW 근거 (실측)

mbm_total − mbm_local = 원격(UPI) 트래픽 실측 가능. MBA 는 socket-로컬만 throttle.

## 가설 / 메커니즘

aggressor 가 remote NUMA 메모리를 치면 victim 간섭은 유지되는데 B2(MBA) 가 무력화 — MBA 회복률은 remote 비율의 함수.

## 실험 설계

AGGR_CPUS=16-55 + numactl --membind=1 로 remote 비율 sweep → MBA 회복률 곡선.

## 게이트

곡선 단조성 확인 → T4 governor 의 모드 전환 신호 (remote_ratio>θ → DUTY) 근거 확정.

## 의존 / 비고

GPU 불요.

## 참조

- 상세: `../RESEARCH_DIRECTIONS.md §3 D6`
- 범위 선언: `../RESEARCH_DIRECTIONS.md` §0 (비-GPU 하드웨어, 마이크로~매크로)
- 공용 인프라: `../src/` (rdt_ctl.py, victim_aggressor.c, run_t1_ab.sh)
