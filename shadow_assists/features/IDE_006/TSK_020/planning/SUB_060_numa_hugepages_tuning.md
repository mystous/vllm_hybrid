# SUB_060 — NUMA + hugepages + cache prefetch 튜닝

> **parent**: TSK_020 / 카테고리 D (HPC classic)
> **status**: 대기 (plan only)
> **effort**: small (1-2 일)
> **CPU% target**: 30-40% (SUB_049 의 26% 에서 +5-10%)
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §4

---

## 1. Mechanism

3 개 classic HPC 기법 결합:

1. **numactl 명시 NUMA 분리** — main vLLM (NUMA0 0-55) ↔ CPU LLM (NUMA1 56-111). SUB_049 영역 영역 thread binding 만 사용했음 (영역 NUMA 영역 implicit). 본 SUB 영역 영역 numactl 영역 `--cpubind` + `--membind` 영역 명시.
2. **Transparent Huge Pages (THP) + explicit 1GB hugepages** — KV cache + main weight 영역 large mem 영역 영역 TLB miss 영역 ↓ + memory bandwidth ↑.
3. **KMP_AFFINITY + GOMP affinity + Intel TBB pinning 일관화** — 영역 thread library 영역 affinity 정책 영역 conflict 영역 영역 정리.

```bash
# main vLLM (NUMA0)
numactl --cpubind=0 --membind=0 \
  python -u vllm_main.py ...

# CPU LLM (NUMA1, isolcpus 영역 영역 SUB_061)
numactl --cpubind=1 --membind=1 \
  python -u cpu_llm_loop.py ...

# env (양 process 공통)
export KMP_AFFINITY=verbose,scatter,granularity=core
export GOMP_CPU_AFFINITY="0-111"
export TBB_THREAD_AFFINITY=1

# hugepages
echo "always" | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
sudo bash -c 'echo 32 > /proc/sys/vm/nr_hugepages_1G'  # 32 GB hugepage pool
```

## 2. 출처

| 자료 | 위치 |
|---|---|
| numactl(8) | Linux man page |
| THP docs | `Documentation/admin-guide/mm/transhuge.rst` (Linux kernel) |
| Intel TBB | https://oneapi-src.github.io/oneTBB/ |
| Intel performance tuning guide | Intel SDM Vol. 3 §17 (Performance Monitoring) |
| perf cache-miss 측정 | `perf stat -e cache-misses,cache-references` |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `/tmp/run_sub060_numa_tuning.sh` (신규) | numactl + hugepages + env 설정 launcher |
| **vLLM 변경 없음** | launcher 영역 환경 변수 변경만 |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | 현 NUMA 영역 상태 측정 (numastat, `/proc/cpuinfo`, hugepages 영역 현황) | 0.25 일 |
| Phase 1 | numactl 영역 launcher 작성 (main + CPU LLM 각각) | 0.25 일 |
| Phase 2 | hugepages 영역 enable (THP + explicit 1GB) | 0.25 일 |
| Phase 3 | KMP/GOMP/TBB affinity 일관화 + verify (KMP_AFFINITY=verbose 영역 로그 확인) | 0.25 일 |
| Phase 4 | SUB_047 best + SUB_049 + 본 lever 영역 결합 측정 (3-run) | 0.5 일 |
| 총 | | **1.5 일** |

## 5. CPU% target / throughput 가설

- numactl 영역 strict NUMA → memory bandwidth ↑ (cross-socket 영역 absent)
- hugepages → TLB miss ↓ (~5% perf gain 일반적)
- affinity 일관화 → thread migration ↓, cache hit ↑
- SUB_049 baseline 26% → **30-40%** 가능 (CPU LLM 영역 더 효율)
- main vLLM throughput 영역 영향 ±1% (NUMA 효과 영역 SUB_049 영역 이미 영역 사용했으므로 marginal)

## 6. Risk

| 위험 | 완화 |
|---|---|
| 1GB hugepages 영역 reserve 시 OOM (시스템 영역 다른 작업 영역 mem 영역 부족할 수 있음) | nr_hugepages_1G=8 부터 시작 |
| KMP_AFFINITY 영역 strict 영역 영역 vLLM main 영역 single-thread 영역 영역 영향 줄 수 있음 | env 영역 main vs CPU LLM 영역 분리 |
| numactl 영역 spawn 자식 영역 inherit 영역 시 problem | nested 영역 numactl 영역 보장 |

## 7. Dependencies

- SUB_049 baseline (NUMA1 binding pattern)
- numactl 영역 설치 (양 numa control)
- sudo 권한 (hugepages enable)

## 8. Acceptance criteria

- [ ] numastat 영역 NUMA0/1 영역 traffic 영역 strict 영역 (cross-socket = 0%)
- [ ] AnonHugePages 사용 영역 ≥ 50% (proc/meminfo)
- [ ] SUB_049 baseline CPU 26% → 영역 ≥ 30%
- [ ] SUB_047 best throughput 영역 영향 ≤ -1%
