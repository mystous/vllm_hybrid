# IDE_020 — CPU Isolation + NUMA + Hugepages (production deploy)

> **scope**: production-ready OS-level isolation — isolcpus / cgroup cpuset / hugepages / IRQ smp_affinity.
> **paper angle**: production-readiness — CPU util 5% → 70-90% guarantee. engineering contribution.
> **status**: ✅ design + config 작성 완료 (SUB_165) / ⚠ host 적용 + 재측정 별도 turn.

---

## 1. 이론적 배경

### 1.1 Phase A measurement 의 제약 (SUB_114)

| 항목 | 현재 (container) | production host (본 IDE) |
|---|---|---|
| `sched_setaffinity` task-pin | ✅ | ✅ |
| `isolcpus=80-99` boot param | ❌ | ✅ |
| cgroup v2 cpuset systemd slice | ❌ (partition invalid) | ✅ |
| 2 MB hugepages × 4096 | ❌ (THP only) | ✅ |
| IRQ smp_affinity reroute | ⚠ (root only) | ✅ |

### 1.2 expected lift (paper §4 ablation)

| lever | Phase A measurement | Production 추가 가능 |
|---|---:|---:|
| physical-core pinning alone | +3.9% (SUB_112) | base |
| isolcpus | — | +1-2% |
| cgroup vllm/fill split | — | +1-3% (N=16 valley 해소 가설) |
| hugepages 명시 | — | +0.5-1.5% |
| IRQ reroute | — | +0.5-1% |
| **합산 예상** | +3.9% | **+6-11%** |

### 1.3 2 sub-task

| TSK | 영역 | scope | deliverable |
|---|---|---|---|
| TSK_038 | NUMA topology audit + IRQ affinity reroute | host 측 IRQ smp_affinity | SUB_165 의 irq_affinity_setup.sh |
| TSK_039 | cgroup + isolcpus + hugepages | systemd slice + boot config | SUB_165 의 cgroup_yaml/ + boot_config/ |

---

## 2. 구현 deliverable (이미 작성됨 — SUB_165)

본 IDE 의 design + config 는 [SUB_165](../IDE_015_cpu_extreme_util/SUB_165_cgroup_isolcpus_doc/RESULTS.md) 의 deliverable 로 작성 완료.

| 파일 | 위치 | 용도 |
|---|---|---|
| GRUB cmdline | `SUB_165/boot_config/grub_cmdline.txt` | isolcpus + hugepages + irqaffinity boot param |
| Hugepages sysctl | `SUB_165/boot_config/hugepages_sysctl.conf` | nr_hugepages = 4096 (8 GB reserved) |
| IRQ setup script | `SUB_165/boot_config/irq_affinity_setup.sh` | nvidia IRQ reroute + irqbalance disable |
| systemd slice root | `SUB_165/cgroup_yaml/spec_decode.slice` | AllowedCPUs=6-99 / AllowedMemoryNodes=0,1 |
| vllm vanilla slice | `SUB_165/cgroup_yaml/vllm_vanilla.slice` | cpuset 6-30 (NUMA 0) |
| vllm trident slice | `SUB_165/cgroup_yaml/vllm_trident.slice` | cpuset 62-79 (NUMA 1) |
| CPU fill slice | `SUB_165/cgroup_yaml/cpu_fill.slice` | cpuset 80-99 (NUMA 1, isolcpus) |
| CPU fill service | `SUB_165/cgroup_yaml/cpu_fill_n20.service` | N=20 sub112 fill worker — kernel-여유 12 core 보존 |

→ 본 IDE 의 README 는 **dispatch 메타 / 적용 protocol / measurement 계획** 만 다룸. 실제 config 는 SUB_165 참조.

---

## 3. Production deploy 검증 protocol

### 3.1 prerequisites
- host root 접근
- container 외부에서 boot params 변경 가능
- nvidia driver host-level mount

### 3.2 deploy 순서

```bash
# step 1: GRUB cmdline 추가 + reboot
sudo cp SUB_165/boot_config/grub_cmdline.txt /etc/default/grub.d/spec_decode.conf
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
sudo reboot

# step 2: hugepages sysctl
sudo cp SUB_165/boot_config/hugepages_sysctl.conf /etc/sysctl.d/99-spec-decode.conf
sudo sysctl -p /etc/sysctl.d/99-spec-decode.conf

# step 3: systemd cgroup slice
sudo cp SUB_165/cgroup_yaml/*.slice /etc/systemd/system/
sudo cp SUB_165/cgroup_yaml/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# step 4: IRQ reroute
sudo bash SUB_165/boot_config/irq_affinity_setup.sh

# step 5: 활성
sudo systemctl start spec_decode.slice
sudo systemctl start vllm_vanilla.service
sudo systemctl start vllm_trident.service
sudo systemctl start cpu_fill_n20.service
```

### 3.3 검증 명령 (SUB_165 §6)

```bash
cat /proc/cmdline | grep isolcpus       # → isolcpus=80-99
cat /proc/meminfo | grep HugePages_Total # → 4096
systemctl status spec_decode.slice       # → running
cat /sys/fs/cgroup/spec_decode.slice/cpu_fill.slice/cpuset.cpus  # → 80-99
```

---

## 4. 측정 계획

### 4.1 baseline 재측정 (with full IDE_020 stack)

- SUB_098 protocol (canonical AGSD-gated 500p × 3 mix)
- isolcpus + cgroup + hugepages + IRQ reroute 활성 상태
- expected: +6-11% vs Phase A SUB_160 baseline (5,474 tps balanced AGSD)

### 4.2 ablation (paper §4)

| config | tps balanced AGSD | source |
|---|---:|---|
| baseline (no isolation) | 5,474 | SUB_160 Phase 1 |
| + sched_setaffinity (SUB_112) | 5,743 (last-3) | SUB_160 Phase 2 |
| + isolcpus alone | (target +1-2%) | TBD |
| + cgroup vllm/fill split | (target +1-3%) | TBD — N=16 valley 해소 검증 |
| + hugepages | (target +0.5-1.5%) | TBD |
| + IRQ reroute | (target +0.5-1%) | TBD |
| **+ all IDE_020** | **6,050-6,200** (+10-13%) | full stack |

### 4.3 stability 1-hour 단일 측정 (사용자 1-run rule)

- single continuous 1-hour benchmark (not loop)
- monitor.py 캡처 0.5s interval
- CPU util / GPU util drift check

---

## 5. dependencies
- IDE_015 의 Phase A 측정 (SUB_098~168) — baseline
- IDE_018 (phase-burst) 가 IDE_020 의 isolation 위에서 동작 — TSK_034 의 task pool 도 cpu 80-99 cgroup 안
