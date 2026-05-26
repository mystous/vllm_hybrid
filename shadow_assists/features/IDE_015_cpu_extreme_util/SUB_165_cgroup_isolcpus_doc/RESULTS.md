# SUB_165 — Production cgroup + isolcpus + hugepages config (host-level)

> **parent**: IDE_020 / TSK_039 (원 SUB_151/152) — production deploy config doc
> **scope**: 2026-05-26 KST (doc-only, container env 적용 불가)
> **status**: ✅ doc 작성 완료 — host 적용은 별도 단계
> **dependency**: SUB_113 NUMA audit + SUB_114 container constraint + SUB_148 trident thread placement
> **constraint**: 사용자 지시 — 물리 코어 100 core max (12 core kernel 여유 보존)

---

## 0. 두괄식 — Production deploy 의 4 단계 OS-level config

| 단계 | 항목 | 목적 | 본 SUB deliverable |
|---:|---|---|---|
| 1 | **isolcpus boot param** | OS scheduler 를 CPU 80-99 (20 core) 격리 — vllm 의 default full-mask affinity 가 침범 불가 | `boot_config/grub_cmdline.txt` |
| 2 | **2 MB hugepages** (×4096 = 8 GB) | AMX worker 의 5120×27648 BF16 matmul (~142 MB / shape) TLB miss 감소 | `boot_config/hugepages_sysctl.conf` |
| 3 | **cgroup v2 cpuset** (systemd slice) | 호스트 단 vllm trident → cpu 56-79 / CPU fill → cpu 80-99 분리 | `cgroup_yaml/spec_decode.slice` + child unit |
| 4 | **IRQ smp_affinity** 재배치 | nvidia GPU IRQ (CPU 97/98/101/103/106/111 등) 를 cpu 0-23 (NUMA 0 free area) 으로 reroute | `boot_config/irq_affinity_setup.sh` |

→ **본 fork SUB_112 의 +3.9% lever** = task-level pinning (`sched_setaffinity`) → IDE_020 step 1-4 추가 시 **추가 lift** 가능 (가설: +2~5% 추가).

---

## 1. CPU layout 설계 (사용자 100 core max constraint 반영)

### 1.1 물리 코어 100 core 할당 (112 중 12 free)

| 구획 | CPU 범위 | NUMA | 코어 수 | 용도 |
|---|---|---:|---:|---|
| **KERNEL FREE** | 0-5 + 56-61 | 0/1 | **12** | kernel kthread / IRQ / system service 여유 (사용자 지시) |
| **VLLM vanilla (GPU 0-3)** | 6-30 | 0 | 25 | vanilla backend Python + TP worker thread |
| **VLLM HT-free buffer** | 31-55 | 0 | 25 | vanilla overhead + sched_setaffinity fallback |
| **VLLM trident (GPU 4-7)** | 62-79 | 1 | 18 | trident backend Python + TP worker thread |
| **CPU fill (AMX worker)** | 80-99 | 1 | **20** | SUB_112 pinned fill (N=20, 사용자 100 core max 준수) |
| (free 잔여) | 100-111 | 1 | 12 | trident GPU IRQ + future expansion |
| **HT siblings 전체** | 112-223 | 0/1 | 112 | **사용 금지** (사용자 지시 + SUB_113 확인) |

→ 총 active core = 0-99 + 100-111 미사용 + HT 미사용 = **100 core 활용 / 112 free buffer 12 core**.
→ SUB_112 의 N=32 (80-111) protocol 은 본 layout 에서 **N=20 (80-99) 으로 축소** 권장 (kernel 여유 + IRQ 영역 보존).

### 1.2 이전 PoC 와의 차이

| 항목 | Phase A PoC (SUB_112) | 본 SUB_165 production design |
|---|---|---|
| 사용 코어 수 | 32 (80-111) | **20 (80-99)** — kernel 여유 + IRQ 보존 |
| vllm placement | OS scheduler 자유 | **isolcpus + cgroup 명시 격리** |
| IRQ | default (cpu 97/98/106/111 등 in pinned range) | NUMA 0 으로 reroute |
| hugepages | THP only | **2 MB × 4096 명시 reserve** |
| cgroup | container env 제약 | host slice + systemd unit |

---

## 2. Deliverable 1: GRUB boot cmdline (`boot_config/grub_cmdline.txt`)

```bash
# /etc/default/grub 의 GRUB_CMDLINE_LINUX 에 추가:
GRUB_CMDLINE_LINUX="... \
  isolcpus=80-99 \
  nohz_full=80-99 \
  rcu_nocbs=80-99 \
  irqaffinity=0-23 \
  default_hugepagesz=2M \
  hugepagesz=2M \
  hugepages=4096 \
  transparent_hugepage=madvise \
"

# 적용:
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
sudo reboot
```

### 옵션 설명
- `isolcpus=80-99`: OS scheduler 가 user task 를 cpu 80-99 에 자동 배치 안 함. **sched_setaffinity 로 명시 할당해야만 사용 가능**.
- `nohz_full=80-99`: tick-less mode — kernel timer interrupt 도 회피 (마이크로초 단위 noise 감소).
- `rcu_nocbs=80-99`: RCU callback 도 다른 코어로 우회.
- `irqaffinity=0-23`: 모든 device IRQ default 를 NUMA 0 의 cpu 0-23 으로 (nvidia IRQ 도 영향).
- `default_hugepagesz=2M / hugepages=4096`: boot 시점 4096 × 2 MB = 8 GB hugepage 예약.
- `transparent_hugepage=madvise`: app 이 명시 madvise(MADV_HUGEPAGE) 시에만 THP 사용.

---

## 3. Deliverable 2: Hugepages sysctl + verification (`boot_config/hugepages_sysctl.conf`)

```bash
# /etc/sysctl.d/99-spec-decode-hugepages.conf
vm.nr_hugepages = 4096
vm.hugetlb_shm_group = 0
kernel.numa_balancing = 0

# 적용:
sudo sysctl -p /etc/sysctl.d/99-spec-decode-hugepages.conf

# 확인:
cat /proc/meminfo | grep -E "HugePages_Total|HugePages_Free|Hugepagesize"
# → HugePages_Total: 4096 / Hugepagesize: 2048 kB
```

### Phase A 대비 변경
- SUB_113 결과: 현재 container env 의 `HugePages_Total: 0` (THP anon 23 GB 만)
- Production target: 명시 4096 × 2 MB → 가용 8 GB

---

## 4. Deliverable 3: cgroup v2 slice + service unit (`cgroup_yaml/`)

### 4.1 systemd slice — `/etc/systemd/system/spec_decode.slice`

```ini
[Unit]
Description=Spec Decode CPU Co-Inference slice
Before=slices.target

[Slice]
# cgroup v2 properties applied via systemd
CPUAffinity=6-30,62-79,80-99
AllowedCPUs=6-99
AllowedMemoryNodes=0,1
MemoryHigh=1.5T
TasksMax=infinity
```

### 4.2 child slice — vllm vanilla (`vllm_vanilla.slice`)

```ini
[Unit]
Description=vLLM vanilla backend (GPU 0-3, NUMA 0)
PartOf=spec_decode.slice

[Slice]
AllowedCPUs=6-30
AllowedMemoryNodes=0
CPUWeight=200
```

### 4.3 child slice — vllm trident (`vllm_trident.slice`)

```ini
[Unit]
Description=vLLM trident backend (GPU 4-7, NUMA 1)
PartOf=spec_decode.slice

[Slice]
AllowedCPUs=62-79
AllowedMemoryNodes=1
CPUWeight=200
```

### 4.4 child slice — CPU fill workers (`cpu_fill.slice`)

```ini
[Unit]
Description=CPU AMX fill workers (NUMA 1, 80-99)
PartOf=spec_decode.slice

[Slice]
AllowedCPUs=80-99
AllowedMemoryNodes=1
CPUWeight=100
```

### 4.5 service unit 예시 — `cpu_fill_n20.service`

```ini
[Unit]
Description=CPU AMX fill workers N=20 pinned 80-99
PartOf=cpu_fill.slice

[Service]
Type=simple
Slice=cpu_fill.slice
User=vllm
WorkingDirectory=/opt/vllm_hybrid
ExecStart=/opt/venv/bin/python /opt/vllm_hybrid/scripts/sub112_cpu_fill_pinned.py \
    --workers 20 --shape qwen32b --batch 128 --dtype bf16 \
    --duration-s 86400 --cpu-base 80 \
    --out-dir /var/log/vllm_hybrid/cpu_workers

# Environment
Environment=OMP_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1

# Restart policy
Restart=on-failure
RestartSec=10

[Install]
WantedBy=spec_decode.slice
```

### 4.6 적용 + 확인 명령

```bash
sudo systemctl daemon-reload
sudo systemctl start spec_decode.slice
sudo systemctl start cpu_fill_n20.service

# 확인:
systemctl status cpu_fill_n20.service
cat /sys/fs/cgroup/spec_decode.slice/cpu_fill.slice/cpuset.cpus
# → 80-99

# verify pin:
for tid in $(cat /sys/fs/cgroup/spec_decode.slice/cpu_fill.slice/cpu_fill_n20.service/cgroup.procs); do
    taskset -p $tid
done
```

---

## 5. Deliverable 4: IRQ smp_affinity reroute (`boot_config/irq_affinity_setup.sh`)

### Phase A finding (SUB_114) 기반 — nvidia high-traffic IRQ 위치 정정

```bash
#!/usr/bin/env bash
# IRQ smp_affinity reroute — nvidia GPU IRQs to NUMA 0 free area (cpu 0-5 reserved kernel + 6-23)
set -euo pipefail

# nvidia IRQs 식별 (current snapshot 기준 high-traffic)
NVIDIA_IRQS=$(grep -i nvidia /proc/interrupts | awk -F: '{print $1}' | tr -d ' ')

# kernel-reserved buffer cpu 6-23 (NUMA 0) 로 reroute
# 18 cpu × bitmask = 0x0000_0000_00ff_ffc0 (cpus 6-23)
TARGET_MASK="ffffc0"

for IRQ in $NVIDIA_IRQS; do
    echo "$TARGET_MASK" > /proc/irq/${IRQ}/smp_affinity 2>/dev/null || \
        echo "WARN: cannot set IRQ ${IRQ} affinity (check root + irqbalance disabled)"
done

# irqbalance 도 비활성 (자동 재배치 방지)
sudo systemctl stop irqbalance 2>/dev/null || true
sudo systemctl disable irqbalance 2>/dev/null || true

echo "Done. Verify: cat /proc/irq/<IRQ>/smp_affinity_list"
```

### Phase A 측정 대비 변화

| 항목 | Phase A (현재) | Production |
|---|---|---|
| nvidia IRQ on cpu 97 | 132 hits (max) | **0** (reroute → cpu 6-23) |
| nvidia IRQ on cpu 111 | 103 hits | **0** |
| nvidia IRQ on cpu 80-99 | ~700 cumulative | **0** ← CPU fill 영역 완전 격리 |
| irqbalance | active default | **stopped + disabled** |

---

## 6. Production deploy 검증 protocol

1. **boot params 확인**: `cat /proc/cmdline | grep isolcpus`
2. **hugepages 확인**: `cat /proc/meminfo | grep HugePages_Total` → 4096
3. **cgroup 활성**: `systemctl status spec_decode.slice` → running
4. **IRQ reroute**: `for i in $(grep nvidia /proc/interrupts | awk -F: '{print $1}'); do cat /proc/irq/$i/smp_affinity_list; done` → 모두 6-23 범위
5. **isolcpus 확인**: `ps -eLo psr,comm | awk '$1>=80 && $1<=99' | sort -k1 -n | uniq -c | head` → CPU 80-99 에 일반 process 없음 (cpu_fill 만)
6. **N=20 fill 작동**: `pgrep -f sub112_cpu_fill_pinned | wc -l` → 20
7. **canonical benchmark**: SUB_098 protocol 으로 3-mix × 3-config 측정 → +3.9% (Phase A) + α 추가 lift 확인

---

## 7. Phase A 측정 결과 → Production 예상 lift

| lever | Phase A 측정 | Production 추가 가능 | 가설 |
|---|---:|---:|---|
| physical-core pinning (sched_setaffinity) | +3.9% (SUB_112 N=32) | 동일 | base |
| isolcpus | (없음) | **+1-2%** | OS sched contention 제거 |
| cgroup vllm/fill 분리 | (container 제약으로 미적용) | **+1-3%** | N=16 valley 해소 (SUB_116) |
| hugepages 명시 | (THP only) | **+0.5-1.5%** | AMX matmul TLB miss 감소 |
| IRQ reroute | (default) | **+0.5-1%** | pinned range IRQ 0 |
| **합산 예상** | +3.9% | **+6-11%** | 추가 +2-7pp |

→ **paper §4 evaluation** 의 "OS-level isolation 추가 lift" 행 입력.

---

## 8. 다음 step

- **(host root 필요)** boot config 적용 → reboot → SUB_112 protocol 재측정 — 본 SUB 의 가설 검증
- **SUB_166 (분석)** — 본 design 의 production 실제 적용 결과 doc 화

## 9. Container env (현재) vs Production host 차이 요약

| 항목 | Container (현재) | Production host |
|---|---|---|
| cgroup v2 cpuset write | ❌ partition invalid (SUB_114) | ✅ |
| systemd-run | ❌ systemd not init | ✅ |
| boot params 변경 | ❌ container 외부 | ✅ |
| irqbalance | ❌ | ✅ stop/disable |
| hugepages reserve | ❌ host config 영향 | ✅ |
| sched_setaffinity | ✅ (Phase A 가 사용) | ✅ |

→ Phase A 의 **sched_setaffinity 단독으로 +3.9%** 가 본 SUB 의 추가 OS-level isolation 없이 달성됨. Production 추가 lift 는 **incremental gain (+2-7pp)** 으로 paper §4 의 추가 cell 이 될 전망.
