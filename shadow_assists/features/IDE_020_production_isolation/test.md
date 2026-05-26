# test.md — IDE_020 테스트 계획

## 1. config 검증

```bash
# boot params
cat /proc/cmdline | grep isolcpus
# expected: isolcpus=80-99 nohz_full=80-99 rcu_nocbs=80-99 irqaffinity=0-23

# isolated CPUs (kernel exposed)
cat /sys/devices/system/cpu/isolated
# expected: 80-99

# hugepages
cat /proc/meminfo | grep -E "HugePages_Total|HugePages_Free"
# expected: 4096 / 4096 (use 후 Free 감소)

# cgroup
systemctl status spec_decode.slice
cat /sys/fs/cgroup/spec_decode.slice/cpu_fill.slice/cpuset.cpus
# expected: 80-99
cat /sys/fs/cgroup/spec_decode.slice/cpu_fill.slice/cpuset.mems
# expected: 1

# IRQ reroute
for i in $(grep nvidia /proc/interrupts | awk -F: '{print $1}'); do
    cat /proc/irq/$i/smp_affinity_list
done | sort -u
# expected: 모두 6-23 범위
```

## 2. correctness (post-deploy)

### 2.1 vllm 시작 후 thread placement 확인

```bash
# vllm vanilla 가 cpu 6-30 에 있어야
ps -eLo psr,comm | grep VLLM | awk '$1 >= 6 && $1 <= 30' | wc -l
# expected: 많은 thread (TP=4 × ~10 thread)

# vllm trident 가 cpu 62-79 에 있어야
ps -eLo psr,comm | grep VLLM | awk '$1 >= 62 && $1 <= 79' | wc -l
# expected: 많은 thread

# cpu_fill 이 cpu 80-99 에 있어야
ps -eLo psr,comm | grep sub112_cpu_fill | awk '$1 >= 80 && $1 <= 99' | wc -l
# expected: 20 (N=20 worker)

# 다른 process 가 cpu 80-99 침범 안 함 (isolcpus 효과)
ps -eLo psr,comm | awk '$1 >= 80 && $1 <= 99 && $2 !~ /sub112|kworker|migration|cpuhp|ksoftirqd|rcu/' | wc -l
# expected: 0
```

### 2.2 N=16 valley 해소 검증

```python
# tests/test_n16_valley_resolved.py
def test_n16_with_cgroup_split():
    """Phase A SUB_116 의 N=16 −14.35% valley 가 IDE_020 cgroup split 으로 해소되는지."""
    # baseline: no cgroup
    tps_no_cgroup = run_canonical(use_cgroup=False, fill_workers=16)
    # with cgroup vllm trident (62-79) + fill (80-99)
    tps_with_cgroup = run_canonical(use_cgroup=True, fill_workers=16)
    delta = (tps_with_cgroup - tps_no_cgroup) / tps_no_cgroup * 100
    print(f"N=16 with cgroup: {delta:+.1f}% vs no-cgroup")
    # target: delta > 0 (cgroup 이 valley 해소)
    assert delta > 0, "cgroup split did not resolve N=16 valley"
```

## 3. e2e ablation (paper §4)

```bash
# ablation order
for CONFIG in baseline schedaffinity isolcpus cgroup hugepages irqreroute full; do
    bash /tmp/run_canonical_agsd_500p_${CONFIG}.sh
    # capture: balanced AGSD tps, CPU util, GPU util
done

# expected ablation table (paper §4):
#   baseline           5,474 (SUB_160 Phase 1)
#   +sched_setaffinity 5,743 (SUB_160 Phase 2 last-3)
#   +isolcpus          5,800 (~+1%)
#   +cgroup            5,900 (~+1.7%) — N=16 valley resolution
#   +hugepages         5,950 (~+0.8%)
#   +IRQ reroute       6,000 (~+0.8%)
#   full IDE_020       6,050-6,200 (target +10-13% vs baseline)
```

## 4. 1-hour stability (single run, 사용자 1-run rule)

```bash
# single continuous benchmark (loop 아님)
bash /tmp/run_canonical_agsd_500p_full_ide020.sh --duration 3600

# monitor.py 0.5s interval background
# expected: no throughput drift > 5% over 1 hour
#           no thermal regression (CPU/GPU temperature stable)
#           CPU util sustained 25-30%+ (IDE_020 alone, IDE_018 미적용)
```

## 5. accuracy gate
- CLAUDE.md 운영 해석: per-token logprob max abs diff < 1e-3
- 1-hour stability 내내 accuracy 유지
- baseline (no IDE_020) 와 token-level 일치 99%+ (분포 동일성)

## 6. util capture
- monitor.py 0.5s interval (paper Figure 6 후보)
- per-CPU activity heatmap (cpu 0-111 × time)
- nvidia-smi sampling (per-GPU SM util)
- IRQ rate per-CPU (cat /proc/interrupts before/after)
