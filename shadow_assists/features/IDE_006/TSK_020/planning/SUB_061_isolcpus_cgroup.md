# SUB_061 — Isolcpus + cgroup v2 cpuset 분리

> **parent**: TSK_020 / 카테고리 D (HPC classic)
> **status**: 대기 (plan only)
> **effort**: small (1 일)
> **CPU% target**: **70-90% saturate** (★ highest target)
> **위험**: boot param 변경 또는 cgroup runtime — OS process drift
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §4

---

## 1. Mechanism

Linux kernel `isolcpus` 또는 cgroup v2 `cpuset.cpus.partition=isolated` 영역 NUMA1 영역 56 cores (56-111) 영역 OS scheduler 영역 격리. 영역 CPU LLM (SUB_049) 만 영역 cores 영역 점유 → 다른 process 영역 들어 못 와 CPU LLM 영역 saturate 가능.

```bash
# Option A: kernel boot param (영구, reboot 필요)
isolcpus=56-111 nohz_full=56-111 rcu_nocbs=56-111

# Option B: cgroup v2 runtime (no reboot)
mkdir /sys/fs/cgroup/cpu_llm
echo "+cpuset" > /sys/fs/cgroup/cgroup.subtree_control
echo "56-111" > /sys/fs/cgroup/cpu_llm/cpuset.cpus
echo "1" > /sys/fs/cgroup/cpu_llm/cpuset.cpus.partition  # isolated
echo $$ > /sys/fs/cgroup/cpu_llm/cgroup.procs  # 본 process 영역 영역 cgroup 영역 join
```

영역 OS process 영역 cores 56-111 영역 도달 안 됨 → CPU LLM 영역 사실상 dedicated.

## 2. 출처

| 자료 | 위치 |
|---|---|
| isolcpus | Linux kernel `Documentation/admin-guide/kernel-parameters.txt` |
| cgroup v2 cpuset | Linux kernel `Documentation/admin-guide/cgroup-v2.rst` (cpuset 영역) |
| nohz_full / rcu_nocbs | Linux kernel `Documentation/admin-guide/kernel-parameters.txt` |
| 실용 가이드 | https://www.kernel.org/doc/html/latest/admin-guide/cputopology.html |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `/tmp/run_sub061_isolcpus.sh` (신규) | cgroup v2 cpuset 설정 + 영역 process join + launcher |
| `/tmp/run_sub061_kernel_boot.md` (신규) | kernel boot param 설정 가이드 (reboot 영역 영역 영역) |
| **vLLM 변경 없음** | runtime 영역 cgroup 영역 영역 활용 |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | 현 cgroup v2 영역 active 확인 (`mount | grep cgroup2`) + isolcpus 영역 boot 영역 가능성 검토 | 0.25 일 |
| Phase 1 | cgroup v2 cpuset 영역 isolated partition 설정 (runtime 영역, reboot 영역) | 0.25 일 |
| Phase 2 | CPU LLM process 영역 영역 cgroup 영역 join + binding 확인 | 0.25 일 |
| Phase 3 | SUB_047 best + 본 lever 영역 결합 측정 | 0.25 일 |
| 총 | | **1 일** |

## 5. CPU% target / throughput 가설

- 56 cores dedicated → CPU LLM 영역 native saturate 가능 (영역 다른 process 영역 압박 영역 없음)
- SUB_049 t3 (Qwen 1.5B + 56 thread) 영역 26.41% → **70-90%** 가능
- 영역 throughput: main vLLM 영역 NUMA0 0-55 cores 영역 fully isolated → +0~+2% (cleaner allocation)

## 6. Risk

| 위험 | 완화 |
|---|---|
| OS process 영역 56-111 영역 도달 못 함 → systemd / kernel daemon 영역 NUMA0 영역 몰림 | NUMA0 영역 영역 main vLLM (single-thread Python 영역) + OS process 영역 충분 |
| isolcpus 영역 kernel boot param 영역 reboot 영역 — runtime cgroup 영역 alternative 영역 충분할 가능 | cgroup v2 영역 시도, fail 시 boot param |
| sudo 권한 영역 영역 (cgroup mount + write 영역 영역) | 권한 영역 사용자 영역 사용자 영역 confirm |

## 7. Dependencies

- cgroup v2 영역 active (Linux ≥ 5.0)
- 본 컨테이너 영역 영역 cgroup 영역 mount 영역 (호스트 영역 따라 다름)
- SUB_049 baseline + Qwen 1.5B model

## 8. Acceptance criteria

- [ ] cgroup v2 cpuset isolated partition 영역 적용 성공
- [ ] CPU LLM (Qwen 1.5B 영역 또는 더 큰) 영역 cores 56-111 영역 sustained 70%+ active
- [ ] OS process 영역 56-111 영역 actively scheduled 영역 없음 (top -p 영역 확인)
- [ ] main vLLM throughput ≥ 10,800 tps (SUB_047 의 -1.5% 안)
- [ ] CPU 영역 system-level avg ≥ 50%
