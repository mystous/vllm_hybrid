# CLAUDE.md — IDE_020 구현 시 알아야 할 것

## 0. 핵심 규칙
- 본 IDE 의 모든 config 는 **SUB_165 에 이미 작성됨** ([deliverable list](../IDE_015_cpu_extreme_util/SUB_165_cgroup_isolcpus_doc/RESULTS.md))
- 본 IDE README 는 **dispatch / 적용 protocol / measurement plan** 만
- **host root 접근 + container 외부에서 boot params 변경 가능** 환경에서만 적용 가능 (Phase A container 내부 적용 불가, SUB_114 finding)
- 사용자 constraint: **물리 100 core max** (12 core kernel 여유 보존)
- 측정 KST 표시, 1-run default

## 1. Phase A measurement input

| Phase A finding | IDE_020 의 의미 |
|---|---|
| SUB_113 GPU PCIe affinity (GPU 0-3↔NUMA0, GPU 4-7↔NUMA1) | cgroup vllm vanilla/trident NUMA split 의 기준 |
| SUB_114 container env cgroup partition invalid | host 적용 의무 (container 내부 안 됨) |
| SUB_148 VLLM thread default full-mask | isolcpus + cgroup 으로만 격리 가능 |
| SUB_116 N=16 valley −14.35% | cgroup vllm/fill split 으로 해소 검증 대상 |
| SUB_165 deliverable (9 config files) | 본 IDE 의 적용 artifact |

## 2. risk + fallback

| risk | severity | fallback |
|---|---|---|
| host root 접근 부재 (container env) | high | dev/staging 에서 검증, prod 별도 deploy turn |
| isolcpus + irqbalance off 가 다른 service 에 영향 | medium | cpu 0-5 + 56-61 (12 core) 의 kernel 여유 |
| hugepages reserve 시 host RAM 압박 (8 GB) | low | host 의 free RAM 충분 (2 TB 환경) |
| cgroup vllm trident split (cpu 62-79) 이 너무 좁음 (18 core only) | medium | vllm trident TP=4 → 5-10 thread per worker → 충분 |

## 3. 적용 후 ablation 측정 계획 (paper §4)

- baseline (no IDE_020): SUB_160 Phase 1 의 500p baseline
- +sched_setaffinity (Phase A SUB_112 protocol): 비교 baseline
- +isolcpus alone: incremental gain 측정
- +cgroup: N=16 valley 해소 검증
- +hugepages: TLB miss 감소 검증
- +IRQ reroute: IRQ hot zone 제거 검증
- **+full IDE_020 stack**: paper main "production lift"

## 4. 검증 게이트
- CLAUDE.md 운영 해석: per-token logprob 분포 동등성 (host 적용 후 재측정)
- 사용자 1-run rule 준수
