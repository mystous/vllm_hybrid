# SUB_113 — NUMA topology audit + GPU-PCIe affinity

> **parent**: IDE_015 / IDE_020 (TSK_038) preview
> **scope**: 2026-05-26 12:25 ~ 12:27 KST (~2 min — shell only)
> **status**: ✅ 완료 — 8 raw audit + 1 summary doc
> **목표**: SUB_112 의 pinning 효과 정확한 mechanism 정량 + IDE_020 baseline 캡처

---

## 0. 두괄식 — SUB_112 의 "cross-NUMA isolation" 표현 정정

| 이전 가정 (SUB_112 RESULTS) | 실제 (본 SUB_113 audit) |
|---|---|
| "CPU 80-111 = NUMA1 후반부, vLLM 은 NUMA0 0-55 사용" | **CPU 80-111 = NUMA1 physical** ↔ **trident backend (GPU 4-7) = NUMA1** → 같은 NUMA |
| "cross-NUMA isolation 이 lever" | **실제 lever = physical-core pinning + HT 회피 + worker 가 vllm 코어 침범 방지** |

→ SUB_112 의 +3.5~3.9% gain 은 **NUMA 격리 효과 아님**. 같은 NUMA1 위에서 physical-core 32개를 worker 가 점유 → trident backend 가 NUMA1 의 나머지 (56-79) + HT siblings 를 사용하도록 강제한 효과.
→ 본 사실은 IDE_020 cgroup 설계에 직접 영향 — `cpuset.cpus` 를 단순히 80-111 로 잡으면 trident backend NUMA-aware allocation 과 경합 가능. **NUMA-aware split 필요** (e.g. trident → CPU 56-79, fill → CPU 80-111).

---

## 1. NUMA 토폴로지

| NUMA | Physical cores | HT siblings | Mem size | Mem free |
|---:|---|---|---:|---:|
| 0 | 0-55 | 112-167 | 1031 GB | **431 GB** |
| 1 | 56-111 | 168-223 | 1032 GB | **97 GB** |

- Distance matrix: local 10 / remote 21 (**2.1× 페널티**)
- **메모리 불균형**: NUMA1 가 NUMA0 대비 334 GB 더 사용됨 → 현재 vllm/cache 가 NUMA1 에 편중 (load average 5.05 1-min)

---

## 2. GPU ↔ NUMA 매핑 (`nvidia-smi topo -m`)

| GPU | PCIe BDF | NUMA | CPU Affinity |
|---:|---|---:|---|
| 0 | 0a:00.0 | **0** | 0-55, 112-167 |
| 1 | 18:00.0 | 0 | 0-55, 112-167 |
| 2 | 23:00.0 | 0 | 0-55, 112-167 |
| 3 | 2c:00.0 | 0 | 0-55, 112-167 |
| 4 | 87:00.0 | **1** | 56-111, 168-223 |
| 5 | 90:00.0 | 1 | 56-111, 168-223 |
| 6 | b8:00.0 | 1 | 56-111, 168-223 |
| 7 | c1:00.0 | 1 | 56-111, 168-223 |

→ canonical AGSD (Qwen 32B TP=4×2) 배치:
- **vanilla (port 8001) GPU 0-3 ↔ NUMA 0**
- **trident (port 8002) GPU 4-7 ↔ NUMA 1**

→ GPU 간 NVLink: 전 쌍 NV18 (= 18-link bonded) **full mesh** — cross-NUMA NVLink 전송도 호스트 RAM 우회.

---

## 3. HT (Hyper-Threading) 매핑 확인

| Physical CPU | HT Sibling | Package (NUMA) |
|---:|---:|---:|
| 0 | 112 | 0 |
| 1 | 113 | 0 |
| 56 | 168 | 1 |
| 80 | 192 | 1 |
| 100 | 212 | 1 |
| 111 | 223 | 1 |

→ HT 규칙: physical N (0-111) 의 sibling 은 N+112 (112-223).
→ SUB_112 CPU 80-111 → HT siblings 192-223 회피 ✓ (memory rule 준수).
→ NUMA1 physical core 56-79 (24 개) 는 SUB_112 에서 사용되지 않음 — trident backend 가 활용 가능 영역.

---

## 4. OS-level isolation 현황 (IDE_020 baseline)

| 항목 | 현재 | IDE_020 목표 |
|---|---|---|
| `isolcpus` (boot cmdline) | ❌ 미설정 | `isolcpus=80-111` (CPU fill 코어 OS scheduler 분리) |
| Hugepages (1G/2M reserved) | ❌ 0 페이지 (THP 23 GB anon 만) | 2 MB × N 페이지 사전 예약 (or 1 GB 페이지) |
| cgroup `cpuset.cpus` | ❌ 미사용 | cpu fill worker → 80-111 / vllm trident → 56-79 NUMA-aware split |
| IRQ `smp_affinity` | ❌ default (모든 NUMA) | network IRQ → NUMA 0 만 (CPU fill NUMA 1 보호) |
| Transparent Huge Pages | ✅ on (anonymous 23 GB 사용) | madvise + 명시적 hugepages 병행 |

→ 4 항목 중 **THP only ON, 나머지 모두 OFF**. SUB_112 의 +3.9% 는 **OS-level isolation 없이도** 달성된 값 → IDE_020 도입 후 추가 gain 여지 큼.

---

## 5. 핵심 finding (IDE_020 / paper 입력)

| finding | 의미 |
|---|---|
| **GPU 0-3 (vanilla) ↔ NUMA 0, GPU 4-7 (trident) ↔ NUMA 1** | tensor parallel split 이 이미 NUMA 라인 따라 정렬됨 — 좋은 baseline |
| **SUB_112 fill (80-111) 가 trident 와 같은 NUMA 1** | "cross-NUMA isolation" 가설 부정 — pinning 이 본질 lever |
| NUMA distance 21 vs 10 (2.1× penalty) | cross-NUMA host RAM 접근 시 비용 정량 |
| NUMA 1 mem 97 GB free vs NUMA 0 431 GB | vllm cache + model weight 가 NUMA1 편중 |
| Hugepages 0 reserved | IDE_020 의 first-order 최적화 — TLB miss 절감 잠재력 큼 |
| isolcpus 미사용 → OS scheduler 자유 | SUB_109/110 unpinned 회귀 패턴의 근본 원인 |
| NV18 full mesh (모든 GPU 쌍) | TP=4×2 split 의 cross-backend overhead 미미 |

---

## 6. SUB_112 RESULTS.md 정정 사항

SUB_112 RESULTS.md §0 "cross-NUMA isolation" 문구 → **"physical-core pinning + HT 회피"** 로 정정 필요.

원문 발췌:
> Physical-core pinning + cross-NUMA isolation 영역 +3.9% net positive

정정 제안:
> **Physical-core pinning + HT 회피 (intra-NUMA1 lever)** → +3.9% net positive

→ 본 SUB_113 의 별도 finding 으로 처리 가능 — SUB_112 의 numeric result 자체는 정확.

---

## 7. 다음 step

- **SUB_116**: N=16 outlier 재측정 (variance check, ~15 min)
- **SUB_117**: per-worker actual CPU util — pinned 워커 active% 정량 (1 h)
- **SUB_148 (IDE_020/TSK_038)**: cgroup `cpuset.cpus=80-111` + isolcpus boot config + hugepages 도입 후 SUB_112 protocol 재측정 → 추가 gain 정량 (production target)
- **SUB_138 (IDE_018)**: trident backend NUMA 1 동일 RAM bandwidth contention 정량 → phase-burst scheduler 의 mem-bound window 회피 입력

---

## 8. raw data

- `raw/numactl_hardware.txt` — `numactl --hardware`
- `raw/lscpu.txt` — `lscpu` (CPU flags 포함: amx_bf16 / amx_tile / amx_int8 / avx512_bf16)
- `raw/nvidia_smi_topo.txt` — `nvidia-smi topo -m` 전체
- `raw/gpu_numa_affinity.txt` — GPU BDF list (직접 /sys 접근 불가)
- `raw/ht_siblings.txt` — HT sibling 매핑 sample
- `raw/hugepages.txt` — `/proc/meminfo` huge page 항목
- `raw/cmdline.txt` — `/proc/cmdline` (boot params)
- `raw/numa_mem.txt` / `raw/free.txt` — NUMA memory 점유
- `raw/_summary_topology.md` — 본 RESULTS 요약 입력
