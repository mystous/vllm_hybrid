# task.md — IDE_020 단계별 적용

## TSK_038 — NUMA topology audit + IRQ affinity

### Step 1 (✅ 완료 — SUB_113 + SUB_114)
- NUMA topology + GPU PCIe affinity 매핑 확정
- IRQ smp_affinity 현재 상태 측정 (high-traffic IRQ 97, 106 in pinned range)

### Step 2 — IRQ reroute (host root 필요)
- SUB_165/boot_config/irq_affinity_setup.sh 실행
- 모든 nvidia IRQ → cpu 6-23 (NUMA 0 free area)
- irqbalance disable

### Step 3 — 측정
- Phase A SUB_112 protocol 재실행 (canonical AGSD-gated balanced 500p, 1-run)
- target: +0.5-1% (IRQ hot zone 제거 효과)

## TSK_039 — cgroup + isolcpus + hugepages

### Step 1 — boot params (host root + reboot 필요)
- SUB_165/boot_config/grub_cmdline.txt → /etc/default/grub.d/
- grub2-mkconfig + reboot
- 검증: cat /proc/cmdline | grep isolcpus

### Step 2 — hugepages
- SUB_165/boot_config/hugepages_sysctl.conf → /etc/sysctl.d/
- sysctl -p
- 검증: /proc/meminfo 의 HugePages_Total 4096

### Step 3 — systemd cgroup slice 활성
- SUB_165/cgroup_yaml/*.slice + *.service → /etc/systemd/system/
- systemctl daemon-reload
- systemctl start spec_decode.slice
- 검증: /sys/fs/cgroup/spec_decode.slice/cpu_fill.slice/cpuset.cpus → 80-99

### Step 4 — full-stack 측정 (paper §4 main)
- canonical AGSD-gated 500p × 3 mix × 1-run
- with vllm vanilla (cpu 6-30, NUMA 0) + vllm trident (cpu 62-79, NUMA 1) + cpu_fill (cpu 80-99, NUMA 1)
- target: 베이스라인 5,474 → **6,050-6,200 (+10-13%)**

### Step 5 — N=16 valley 해소 검증
- SUB_116 finding: N=16 unpinned 시 −14.35%
- cgroup vllm trident (62-79) 와 cpu_fill (80-99) 명시 분리 시 valley 해소 가설 검증
- 측정: N=16 cgroup-separated vs N=16 sched_setaffinity-only

### Step 6 — 1-hour stability single run (사용자 1-run rule)
- continuous benchmark 1 hour (loop 아닌 single benchmark, 또는 streaming)
- monitor.py background
- drift / thermal regression check

### Step 7 — ablation (paper §4)
- 6 cell measurement (baseline + each lever)
- 결과 → paper §4 의 production lift breakdown
