# SUB_114 — IRQ / cgroup feasibility + N=16 valley mechanism 추적

> **parent**: IDE_015 (Phase A 완료 위해) / IDE_020 TSK_038 preview
> **scope**: 2026-05-26 12:58 ~ 13:05 KST (~7 min shell only)
> **status**: ✅ 완료 — IRQ 분포 + cgroup feasibility + container env 제약 정리
> **목표**: SUB_116 N=16 valley (−14.35%) 의 mechanism 검증 (IRQ contention 가설) + IDE_020 cgroup 설계 입력

---

## 0. 두괄식 — IRQ 가설 부분 부정 + 환경 제약 정리

| 가설 / 발견 | 결과 |
|---|---|
| (A) nvidia IRQ 가 N=16 pinned range (80-95) 에 집중 → valley 원인 | ❌ **부분 부정** — IRQ hot zone 은 96-111 (N=16 미포함). N=16 valley 의 직접 원인 아님 |
| (B) cgroup v2 cpuset 으로 trident / fill worker 분리 가능 | ❌ **제한** — Podman container 내부에서 root cpuset partition "domain invalid" 상태. 호스트 레벨 설정 필요 |
| (C) container 안에서 sched_setaffinity 는 작동 | ✅ **확인** — SUB_112 가 이미 사용 중인 방식 |
| (D) trident worker thread placement 정확한 매핑 | ⏳ **미확인** — vllm boot 필요 (다음 SUB 후보) |

→ N=16 valley 의 mechanism 은 **IRQ 단독** 아님 — vllm trident worker thread placement (가설 D) 확인 필요.
→ 현 container 환경에서는 **task-level pinning (`sched_setaffinity`)** 만 사용 가능. **production deploy (IDE_020) 시 host-level cgroup** 필요.

---

## 1. nvidia IRQ 분포

### 1.1 high-traffic nvidia IRQ → CPU 매핑 (cumulative since boot, 18 day uptime)

| IRQ | total | 1st CPU (count) | 2nd CPU (count) | smp_affinity_list |
|---:|---:|---|---|---:|
| 2763 | 527 | **CPU 111 (103)** | CPU 101 (83) | 97 |
| 2761 | 346 | CPU 220 (50) | CPU 209 (49) | 178 |
| 2784 | 297 | CPU 118 (68) | CPU 131 (37) | 163 |
| 2749 | 269 | CPU 35 (42) | CPU 51 (23) | 23 |
| 2778 | 264 | **CPU 98 (132)** | CPU 178 (48) | **106** |
| 2754 | 255 | **CPU 103 (75)** | CPU 188 (50) | 223 |
| 2760 | 253 | CPU 41 (82) | CPU 118 (28) | 41 |

→ nvidia driver 는 `smp_affinity` 로 각 IRQ 를 **단일 CPU 에 pin** (NUMA 0/1 + HT siblings 다양).
→ Pinned range (80-111) 안에 떨어지는 nvidia IRQ smp_affinity = **97, 106** (2 개 IRQ).
→ 그 외 IRQs 는 NUMA 0 (23, 41, 51, 35), HT siblings (118, 163, 178, 188, 220, 223) 분산.

### 1.2 nvidia IRQ 누적 횟수 — pinned range (80-111) per-CPU 분포

| CPU | total IRQs | 비고 |
|---:|---:|---|
| 82 | 11 | (N=8 안) |
| 84 | 5 | (N=8 안) |
| 87 | 2 | (N=8 boundary) |
| 88 | 32 | (N=16 안, N=8 밖) |
| 89 | 6 | (N=16) |
| 90 | 22 | (N=16) |
| 96 | 14 | (N=32 안, N=16 밖) |
| **97** | **61** | (N=32) |
| **98** | **132** | (N=32) — max single IRQ landed |
| **101** | **83** | (N=32) |
| **103** | **75** | (N=32) |
| 106 | 26 | (N=32) |
| **111** | **103** | (N=32) |

→ **IRQ hot zone** = CPU 96-111 (특히 97/98/101/103/111).
→ N=8 (80-87): light IRQ traffic (총 ~18) — 50% throttle 의 직접 원인 아님.
→ N=16 (80-95): light IRQ traffic (총 ~104, 대부분 88-90) — 그러나 **−14.35% 회귀**.
→ N=32 (80-111): **heaviest IRQ traffic 흡수** (~700+) — 그럼에도 **+3.9% positive**, per-worker 99.4% saturated.

### 1.3 가설 (A) 부분 부정

| N | IRQ overlap | 측정 결과 (SUB_112/116/117) | IRQ 가설 |
|---:|---|---:|---|
| 8 | 매우 light (~18) | +3.6%, per-worker 50% | IRQ 가설로는 50% 설명 불가 |
| 16 | light (~104) | **−14.35%** ⚠ | IRQ 가설로는 valley 설명 불가 |
| 32 | heavy (~700+) | +3.9%, per-worker 99.4% | IRQ 가설이면 N=32 가 worst 되어야 |

→ **IRQ 단독 가설은 valley 의 mechanism 으로 부족**. 다른 vector 필요 — 가설 (D) trident worker thread placement.

---

## 2. cgroup v2 feasibility (container 환경)

### 2.1 환경 식별

| 항목 | 값 |
|---|---|
| /.dockerenv | 없음 |
| /run/.containerenv | 있음 → **Podman container** |
| PID 1 comm | `systemd` (container 내부에 boot) |
| cgroup ns id | `cgroup:[4026557800]` (호스트와 분리됨) |
| cgroup mount | `cgroup2 on /sys/fs/cgroup` (v2 unified) |
| root cgroup.type | `domain threaded` |
| root cpuset.cpus.partition | **`root invalid (Parent is not a partition root)`** ⚠ |

→ Podman container 안 root cgroup 이 **"invalid partition" 상태**. Container 가 cgroup v2 namespace 를 받았지만 partition root 권한이 없음.

### 2.2 cgroup 생성 + process 이동 PoC

```bash
# 1) cgroup 생성: 성공
mkdir /sys/fs/cgroup/sub114_test
echo "80-95" > sub114_test/cpuset.cpus     # ✓ written
echo "1" > sub114_test/cpuset.mems          # ✓ written
cat sub114_test/cpuset.cpus.effective       # → "80-95" ✓
cat sub114_test/cgroup.type                 # → "domain invalid" ⚠

# 2) process 이동: 실패
echo $PID > sub114_test/cgroup.procs
# → "Operation not supported"

# 3) systemd-run 시도: 실패
systemd-run --slice=test.slice -p AllowedCPUs=80-95 sleep 30
# → "System has not been booted with systemd as init system... Host is down"
```

→ container 내부에서 process 를 cgroup 에 이동 불가. cgroup 생성은 가능하나 **사용 불가 상태**.

### 2.3 결론 — 본 환경에서 IDE_020 cgroup 접근 제한

| 접근 방식 | 본 env 가능 | 비고 |
|---|---|---|
| `sched_setaffinity` (SUB_112) | ✅ 작동 | task-level pinning. 충분 |
| `taskset` CLI | ✅ 작동 | 동일 mechanism |
| cgroup v2 cpuset (container 내부) | ❌ partition invalid | host root 권한 필요 |
| systemd-run --scope --slice | ❌ | systemd 호스트 init 필요 |
| Podman/Docker `--cpus`, `--cpuset-cpus` | ⏳ | container start 시 orchestrator 가 설정해야 |

→ **production deploy (IDE_020/TSK_038) 시**: 호스트 또는 orchestrator (Podman/K8s) 단에서 cgroup 설정 필요. SUB_112 의 `sched_setaffinity` 만으로도 +3.9% 가능하지만, **isolcpus + IRQ smp_affinity 재배치** 가 추가 lever.

---

## 3. kthread 분포 (NUMA 1 pinned range 80-111)

| CPU 80-111 32 개 코어 | kthread 수 (snapshot) |
|---|---|
| min | 7 |
| max | 9 |
| avg | ~8 |
| 합계 | **258** kthread |

→ 각 CPU 에 7-9 개의 kthread (kworker/ksoftirqd/rcu/migration/cpuhp). 분포 **uniform** — 특정 CPU 가 contention vector 아님.
→ N=8 의 per-worker 50% throttle 의 직접 원인은 kthread 아님.

---

## 4. 핵심 finding + N=16 valley mechanism 잔존

| finding | 의미 |
|---|---|
| nvidia IRQ smp_affinity: 본 env 의 7 high-traffic IRQs 중 **2 개** (97, 106) 만 pinned range 안 | IRQ 가설 단독으론 N=16 valley 설명 부족 |
| N=32 가 IRQ hot zone 흡수해도 +3.9% — IRQ overhead 미미 | IRQ 절대량은 valley 의 1차 원인 아님 |
| container env 에서 cgroup v2 cpuset 사용 제한 | IDE_020 의 cgroup 접근은 **host-level orchestrator** 통해야 |
| `sched_setaffinity` 기반 SUB_112 protocol 은 본 container 에서 fully 작동 | 본 env 의 main lever 그대로 유효 |
| kthread 분포 uniform | 분포 자체 contention 아님 |
| **남은 가설 (D)**: trident vllm worker thread (TP=4) 가 N=16 (80-95) 와 overlap | **vllm boot 후 검증 필요 다음 SUB** |

---

## 5. 추천 follow-up SUB

| SUB | 영역 | 의존 | 예상 ETA |
|---|---|---|---|
| (미할당) | vllm trident worker thread placement: `ps -eLo psr,comm` for VLLM::Worker_TP* | vllm boot 필요 (~5 min) | 15 min |
| SUB_148 (IDE_020/TSK_038) | host orchestrator 단 cgroup cpuset.cpus=80-111 split | host root + orchestrator 변경 | production deploy 시 |
| (미할당) | IRQ smp_affinity 재배치 후 SUB_112 재측정 — IRQ hot zone (97/98/101/103/111) 을 cores 56-79 로 이동 | root 권한 필요. 본 container 가능여부 미정 | 30 min |

---

## 6. raw data

- `raw/proc_interrupts.txt` — `/proc/interrupts` 전체 (6.7 MB)
- `raw/nvidia_irq_top.txt` — high-traffic nvidia IRQ + top 3 CPU
- `raw/nvidia_irq_cpu_distribution.txt` — IRQ landing CPU 분포
- `raw/nvidia_irq_smp_affinity.txt` — smp_affinity 정책
- `raw/nvidia_irqs_pinned_range_per_cpu.txt` — pinned range 80-111 per-CPU cumulative
- `raw/kthread_per_cpu.txt` — kthread 분포
