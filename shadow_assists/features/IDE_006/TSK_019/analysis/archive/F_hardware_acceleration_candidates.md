# Phase F — Hardware 가속 후보 (AMX/AVX 외)

> 분석 시각: 2026-05-15 KST
> 산출 목적: AMX/AVX-512 외에 본 prod 환경 (H100×8 + Xeon SPR 2S + ConnectX-7 × 12) 에서 활용 가능한 hardware 특성 전 영역 인벤토리. 각 후보의 위치 / 현재 사용 여부 / 예상 효과 / 적용 cost.
> 본 문서는 **결정 문서 아님** — fact + 적용 가능성만 제공.

---

## F.1 prod 환경 hardware spec 재확인

| 영역 | spec | 현재 활용도 |
|---|---|---|
| GPU | NVIDIA H100 80GB HBM3 × 8 | TP=8 |
| GPU 간 통신 | NVLink Gen4, 18 link/pair, NV18 full-mesh (≈900 GB/s/pair) | NCCL all_reduce 활용 |
| GPU↔CPU | PCIe Gen5 (추정, NV switch + Xeon SPR root) | cudaMemcpyAsync (pinned) |
| CPU | Intel Xeon Platinum 8480+ × 2 socket | OMP 12 thread/worker, AVX-512+AMX native |
| CPU memory | DDR5-4800 8ch × 2 socket = 2 TB total | 1031 GB/NUMA0, 1032 GB/NUMA1 |
| NUMA | 2 node (node 0: GPU 0-3 + NIC 0-5, node 1: GPU 4-7 + NIC 6-11) | VLLM_NEO_NUMA_BIND=1 |
| NIC | ConnectX-7 (MT4129) × 12 (NDR 400Gb/s 가능) | (단일 노드 운영, 외부 통신 없음) |
| Hugepages | THP only (AnonHugePages 13.4 GB 활성, 1GB hugepages 0개) | 미사용 (default THP) |
| GPUDirect RDMA | `nvidia_peermem` 미로드 | **미활용** (single-node 라 의미 작음) |
| GPUDirect Storage | `/etc/cufile.json` 존재, nvidia-fs 모듈 상태 미상 | (확인 필요) |
| CXL | 없음 | N/A |
| CUDA | 13.0 + driver 580.126.20 | 최신 |

---

## F.2 후보 long-list — AMX/AVX 외 가속 영역

각 후보의 적용 영역 / 효과 / 위험 / 본 환경 가용성.

### Category G — GPU 간 / GPU 내 (NVLink, CUDA Graphs, Tensor Core)

| # | 후보 | 영역 | 현재 | 효과 추정 | 적용 cost | 위험 |
|---|---|---|---|---|---|---|
| G01 | **CUDA Stream Priority** | gdec attention forward stream 을 high-priority | default priority | latency 단축 (μs 영역) | 작음 (cudaStreamCreateWithPriority) | 다른 stream starvation 위험 |
| G02 | **CUDA Graphs 확대** | NEO 의 sub_batch_executor 영역까지 graph capture | enforce-eager=False (forward 영역 graph) | overhead 10-20% 절감 | 중간 (NEO dynamic dispatch 호환성 필요) | dynamic shape 충돌 |
| G03 | **NVLink P2P direct copy** | KV cache GPU 간 migration (TP 가 아닌 다른 dim) | NCCL via NVLink | inter-GPU KV 이동 시 CPU 우회 | 중간 (cudaMemcpyPeerAsync) | TP 영역 외 작업 발생 시만 의미 |
| G04 | **Hopper TMA (Tensor Memory Accelerator)** | attention kernel 내 async copy | FlashAttention 3 가 이미 사용 추정 | (이미 활용 중) | - | 미적용 시 - |
| G05 | **NVSHMEM** | symmetric memory 통한 inter-GPU low-latency | NCCL only | small-message all_reduce 의 latency 단축 | 큼 (nvshmem 통합) | 본 workload 의 collective 크기 큼 — 효과 작음 |
| G06 | **FP8 Transformer Engine 확장** | KV cache 외 attention 의 score / matmul 도 FP8 | KV cache fp8 (이미) | compute speedup 1.5-2× | 큼 (정확도 검증 필수) | precision drop |
| G07 | **CUDA Multi-Stream sub-batch** | NEO 의 sub-batch[0] / sub-batch[1] 별도 stream | 이미 분리됨 (`sub_batch_executor`) | (이미 사용) | - | - |

### Category C — CPU 측 (AMX/AVX 외)

| # | 후보 | 영역 | 현재 | 효과 추정 | 적용 cost | 위험 |
|---|---|---|---|---|---|---|
| C01 | **1GB Hugepages** | KV cache + pinned staging buffer 영역의 TLB miss 절감 | THP 13.4 GB (4KB → 2MB) | TLB miss 30-50% 절감, BW 5-10% 향상 (큰 영역) | 중간 (boot param 또는 boot 후 alloc, 권한 필요) | hugepage alloc 실패 가능 |
| C02 | **NUMA local pinned staging** | per-worker staging buffer 의 NUMA 정합 (worker N → NUMA M) | NEO_NUMA_BIND=1 (general) | inter-NUMA traffic 30-50% 절감 | 작음 (mbind 또는 set_mempolicy) | 이미 부분 적용 |
| C03 | **CPU PREFETCHT0 + MOVNTI** | swap copy_layer_out 의 직접 memcpy (ATen 우회) | ATen index_kernel + OMP | swap-in copy BW 1.5-2× (NT-store + prefetch) | 중간 (cpp kernel 신규 작성) | cache pollution, alignment |
| C04 | **Persistent OMP team** | pacpu attn_one_seq 의 매 layer OMP team launch | libgomp default | OMP launch overhead 30-50% 절감 (영역 작음) | 작음 (omp_set_dynamic(0)) | - |
| C05 | **KMP_BLOCKTIME=0** | OMP team idle wait 의 spin-vs-sleep tunable | default 200ms | thread wake latency 단축 | 작음 (env var) | CPU idle 시 spin 증가 |
| C06 | **CXL memory tier** | NEO 의 CPU KV pool 을 CXL 으로 확장 | 본 환경 CXL 없음 | 추가 KV capacity (DDR5 의 ~80% BW) | **N/A** (hw 없음) | 본 환경 미적용 |
| C07 | **Direct memcpy with AVX-512 NT-store** | BM07 swap copy_layer_out 의 ATen index_kernel 우회 | ATen path (8.26% OMP) | swap-in copy BW 1.5-2× | 중간 (cpp neo_cpu_kv_buffer 의 copy_layer_out 재작성) | alignment 요구 |

### Category I — CPU ↔ GPU 통신 (PCIe)

| # | 후보 | 영역 | 현재 | 효과 추정 | 적용 cost | 위험 |
|---|---|---|---|---|---|---|
| I01 | **PCIe Gen5 BW 활용도 측정 + 확대** | swap-out gather + DMA 의 PCIe BW 활용 측정 | 추정 50-70 GB/s 영역 | (측정 후 결정) | 측정 필요 | 측정 미달 |
| I02 | **cudaMemAdvise / UVM prefetch** | NEO swap path 의 명시적 prefetch | 미사용 | swap-in latency 단축 | 중간 (UVM 활용) | UVM 의 page fault overhead |
| I03 | **GDR (GPUDirect RDMA) for memory copy** | NIC ↔ GPU 직접 DMA (single-node 의미 작음) | nvidia_peermem 미설치 | N/A in single-node | - | 본 환경 단일 노드, 의미 작음 |
| I04 | **Multi-stream DMA 채널 활용** | H100 의 dedicated copy engine 4개 활용 (현재 1개 stream 사용?) | ASYNC_SWAP_BUFFERS=3 (3 stream 동시 활용) | DMA throughput 1.5-2× | 작음 (이미 부분 적용) | 이미 SUB_026 에서 도입 |
| I05 | **Pinned staging buffer의 NUMA 정합** | worker N (NUMA M) 의 pinned buffer 도 NUMA M 에 위치 | NEO_NUMA_BIND=1 (확인 필요) | DMA latency 단축 (NUMA crossing 회피) | 작음 (numa_alloc_onnode) | 검증 필요 |

### Category R — RPC / 동기화 (engine ↔ worker)

| # | 후보 | 영역 | 현재 | 효과 추정 | 적용 cost | 위험 |
|---|---|---|---|---|---|---|
| R01 | **Lock-free SPSC queue** | shm_broadcast 의 dequeue (16.23%) + acquire_read (13.52%) 의 lock 제거 | shared memory + sched_yield | RPC latency 단축 (μs 영역) | 큼 (shm_broadcast 재작성) | 정확성 검증 부담 |
| R02 | **futex_wait 기반 wake** | sched_yield 영역 의 명시적 thread wake | sched_yield 9.37% + 10.32% | latency 단축 + CPU duty 절감 | 중간 (utils.py:48 의 yield → futex) | wake 정확성 |
| R03 | **Busy-spin only (no yield)** | RPC wait 의 yield 제거, 1 step 안에 처리되도록 spin | sched_yield 가 일부 영역 | latency 매우 작음, 단 CPU 100% | 작음 (config) | 다른 작업 starvation, power 증가 |
| R04 | **CPU isolation (cgroup + isolcpus)** | worker thread 가 OS scheduling 의 영향 없이 동작 | taskset only | jitter 감소, tail latency 단축 | 큼 (kernel boot param) | 시스템 영역 |
| R05 | **mDIE / posted IO** | engine ↔ worker 의 dispatch 를 message queue 가 아닌 ring buffer 로 | shm_broadcast | latency 단축 | 큼 (재설계) | shm_broadcast 가 이미 ring-like |

### Category O — OS / kernel tuning

| # | 후보 | 영역 | 현재 | 효과 추정 | 적용 cost | 위험 |
|---|---|---|---|---|---|---|
| O01 | **IRQ affinity** | NIC IRQ 를 NUMA local core 로 pin | default | 작음 (다른 작업 영역) | 작음 (irqbalance off + manual) | 본 환경 NIC 활용도 작음 |
| O02 | **Disable TurboBoost variability** | CPU freq 고정 (jitter 절감) | TurboBoost default ON | tail latency 단축 (5-10%) | 작음 (cpupower) | 평균 throughput 감소 가능 |
| O03 | **mlock pinned memory** | swap staging buffer 의 page fault 방지 | pin_memory=True (cuda pinned) | 이미 안전 (CUDA pinned 자체가 mlock) | - | - |
| O04 | **kernel.sched_migration_cost_ns** | OS scheduler 의 thread migration 빈도 감소 | default | jitter 감소 | 작음 (sysctl) | 다른 process 영향 |
| O05 | **Disable Hyper-Threading on hot cores** | OMP team 의 inter-thread cache contention 절감 | HT 활성 (CPU 224개) | OMP team 의 BW 효율 5-10% | 큼 (BIOS reboot) | dev 환경 영향 |

### Category S — NVMe / 저장 영역

| # | 후보 | 영역 | 현재 | 효과 추정 | 적용 cost | 위험 |
|---|---|---|---|---|---|---|
| S01 | **GPUDirect Storage (cuFile)** | model weight load NVMe → GPU 직접 | mmap or torch.load | model load wall (init phase) 절감 | 중간 (cuFile API 통합) | 본 환경 nvidia-fs 모듈 상태 미상 |
| S02 | **NVMe → CPU KV tier** | NEO 의 CPU KV pool 이 RAM 초과 시 NVMe extend | RAM only | KV capacity 확장 (long-context) | 큼 (NEO scheduler 확장) | NVMe BW 가 RAM 의 1/10 — latency 큼 |

---

## F.3 우선순위 (효과 × 적용 가능성)

### Tier 1 — 즉시 적용 가능, 효과 분명

| Tier 1 | 후보 | 효과 | cost | 영역 |
|---|---|---|---|---|
| ◎ | **C01 1GB Hugepages** | TLB miss 30-50% 절감, KV cache 영역 BW 5-10% | 중간 | 시스템 영역 (boot param) |
| ◎ | **C04 Persistent OMP team** | OMP launch overhead 절감 | 작음 | pacpu kernel + ATen 영역 |
| ◎ | **C05 KMP_BLOCKTIME=0** | thread wake latency | 작음 | env var |
| ◎ | **C03/C07 Direct memcpy (NT-store + prefetch)** | swap copy_layer_out 의 BW 1.5-2× | 중간 | BM07 영역 |
| ◎ | **I05 NUMA local pinned staging** | DMA latency 단축 (NUMA crossing 회피) | 작음 | swap path 영역 |
| ◎ | **G01 CUDA Stream Priority** | gdec attention forward 우선 | 작음 | sub_batch_executor 영역 |

### Tier 2 — 중간 cost, 중간 효과

| Tier 2 | 후보 | 효과 | cost | 영역 |
|---|---|---|---|---|
| ○ | **G02 CUDA Graphs 확대** | sub_batch dispatch overhead 절감 | 중간 | sub_batch_executor 영역 |
| ○ | **R02 futex_wait 기반 wake** | RPC latency 단축 | 중간 | shm_broadcast 영역 |
| ○ | **I02 cudaMemAdvise prefetch** | swap-in latency 단축 | 중간 | swap-in path |
| ○ | **O02 Disable TurboBoost variability** | tail latency 단축 | 작음 | OS 영역 |

### Tier 3 — 큰 cost 또는 효과 작음

| Tier 3 | 후보 | 효과 | cost | 영역 |
|---|---|---|---|---|
| △ | **G06 FP8 attention 확장** | matmul 1.5-2× | 큼 (정확도) | TE 통합 |
| △ | **R01 Lock-free SPSC queue** | RPC 영역 재작성 | 큼 | shm_broadcast |
| △ | **S01 GPUDirect Storage** | init phase 단축 (1회 영향) | 중간 | model load |
| △ | **O05 Disable HT** | OMP BW 효율 | 큼 (reboot) | BIOS 영역 |

### Tier X — 본 환경 적용 불가

| Tier X | 후보 | 이유 |
|---|---|---|
| ✗ | C06 CXL memory tier | hw 없음 |
| ✗ | I03 GPUDirect RDMA | single-node 의미 작음 |
| ✗ | G05 NVSHMEM | NCCL 와 합산 효과 작음 |
| ✗ | R04 CPU isolation | 시스템 boot param 영역 |
| ✗ | S02 NVMe KV tier | RAM 충분 (2 TB) |

---

## F.4 영역별 정량 추정 (Tier 1 후보)

### C01 — 1GB Hugepages

- KV cache 영역 ≈ 174,684 blocks × 16 token × FP8 = ~22 GB / GPU × 8 = 176 GB GPU memory + CPU pinned staging (3.8 GB/worker × 8 = 30 GB) + CPU KV pool (TBD)
- 4KB page → 1GB page: TLB entries 4 ~ million → 30. TLB miss 영역 30-50% 절감
- 추정 wall 절감 = 2-5% (KV-bound workload)

### C03/C07 — Direct memcpy (NT-store + prefetch)

- BM07 (swap copy_layer_out): ATen index_kernel + OMP 8.26% of worker
- 직접 memcpy (alignment + NT-store + PREFETCHT0): BW 1.5-2× → 영역 4-6% 절감
- ASYNC=1 시 이 영역 이미 부분 hidden (별도 OMP thread) — 그러나 swap-in 의 BM09 영역에는 wall critical

### C04 — Persistent OMP team

- pacpu 의 attn_one_seq launch overhead 영역 작음 (cdec dispatch 4.5% 영역 안의 5% = 전체 0.2%)
- 그러나 chain firing 99% 영역 도달 시 OMP launch overhead 가 큼 — 미래 영역 대비 도입 가치

### G01 — CUDA Stream Priority

- gdec attention (BM11) 의 latency 우선 → sub_batch[0] wall 단축
- 영역 작음 (μs 단위), tail 영역 영향 가능

### I05 — NUMA local pinned staging

- staging buffer 가 NUMA crossing 발생 시 DMA latency 1.5× 증가
- NEO_NUMA_BIND=1 이 정확히 어디 적용되는지 검증 필요
- 영역 작음 (전체 swap 영역의 일부)

---

## F.5 사용자 명시 영역 검토 — "GPUDirect RDMA, CUDA bypass low-level"

### GPUDirect RDMA (I03)

- **본 환경 (single-node, TP=8)** 에서는 의미 작음 — NVLink 가 같은 host 의 GPU 간 더 빠름
- **multi-node disaggregated prefill** 또는 **cross-node KV migration** 시 큰 효과
- 본 plan 의 vllm-hybrid 영역에서는 **future work** (multi-node 확장 시)

### CUDA bypass low-level (Driver API, IOMMU bypass)

- vllm 은 CUDA Runtime API 사용
- Driver API (cuMemcpyAsync 등) 는 lightweight 이지만 Runtime API 와 functional 차이 작음
- **진짜 bypass**: `cuMemAddressReserve` + `cuMemMap` (VMM API) — virtual memory mapping 으로 KV cache 의 dynamic alloc/free 효율
- **NVSHMEM** = same-node symmetric memory, CUDA stream-aware put/get
- **현재 활용 가능**: VMM API 도입 시 KV cache fragmentation 개선 + alloc/free overhead 단축. 그러나 vllm 의 PagedAttention 이 이미 block-based 라 효과 작음
- IOMMU bypass — 본 환경 (root container) 에서는 의미 작음

→ 두 영역 모두 본 single-node 환경에서 **효과 제한적**. multi-node 확장 시 GPUDirect RDMA + NVSHMEM 이 핵심 도구.

---

## F.6 결론 — Hardware 가속 후보 종합

### AMX/AVX-512 외에 본 환경 적용 가능한 영역 = **20+ 후보 식별**

- **Tier 1 (즉시 적용 가능)**: 6 후보 — C01, C03/C07, C04, C05, G01, I05
- **Tier 2 (중간)**: 4 후보 — G02, R02, I02, O02
- **Tier 3 (큰 cost)**: 4 후보 — G06, R01, S01, O05
- **Tier X (적용 불가)**: 5 후보

### Tier 1 합산 예상 효과 (정량 추정)

**※ 2026-05-15 KST Phase 1 측정 정정 — `E_open_questions.md` E.13.6 fact 반영**

| 후보 | 이전 추정 | 실측 후 정정 | 근거 |
|---|---|---|---|
| C01 1GB Hugepages | wall 2-5% | **~0%** | TLB miss 실측 0.01% (이미 매우 낮음) |
| C03/C07 NT-store memcpy | wall 4-6% | **0.5-2%** (영역 작음) | DDR5 BW 활용 11.2% (saturated 아님) |
| C04 Persistent OMP team | wall 1-2% | **0.5-1%** | cdec b1_avg=0 영역 |
| G01 CUDA Stream Priority | wall 1-2% | **0.5-1%** (변경 없음) | gdec wall critical 영역 변화 |
| I05 NUMA local pinned staging | wall 1-2% | **~0%** | NUMA 정합 이미 99.6-99.9% |
| **합 wall 절감 (정정)** | 10-17% | **~2-5%** | 본 환경의 hw 영역 가속 효과 limited |

→ **본 환경의 진짜 wall 절감 영역 = chain firing 영역 도달 + workload 영역 조정** (Phase F 의 hw 가속 단독으로는 큰 영향 어려움). 본 fact 는 Phase 1 (`E_open_questions.md` E.13.6) 측정 결과의 정합.

### Phase E AMX/AVX 적용 가능성 표 (`E_amx_avx_applicability.md`) 와의 합산

- AMX 적용 시 wall **4-12% 추가 절감** (조건: chain firing 80-99% 영역)
- Tier 1 hw 가속 + AMX 합산 시 wall **15-25% 절감** 가능 (이론, 영역 합산 시 overlap 영역 차감)

→ 본 plan 의 v1.6 baseline (1,638 tps @ 200p) 에서 **+15-25% throughput 향상 영역 가능** (paper claim H100 14% 의 1.5-2× 도달 영역).

### 본 분석의 범위 명시

- 본 문서는 **결정 문서 아님**
- 각 후보의 fact + 적용 가능성 + 예상 효과만 제공
- 실제 적용 결정 + 우선순위 + 구현 plan 은 후속 작업

---

## F.7 측정 미달 영역 (open questions for Phase F)

| ID | 측정 항목 | 방법 |
|---|---|---|
| OQ12 | PCIe Gen5 의 실효 BW (현재 swap-out gather + DMA 의 활용도) | `nvidia-smi dmon` + `pcm-pcie` |
| OQ13 | NUMA crossing 정량 (pinned staging buffer 의 NUMA placement 검증) | `numastat` per-worker |
| OQ14 | 1GB hugepage 활성 가능 여부 (boot param vs runtime alloc) | `cat /sys/devices/system/node/node*/hugepages/hugepages-1048576kB/` |
| OQ15 | TLB miss 측정 (perf event `dTLB-load-misses`) | `perf stat -e dTLB-load-misses ./run_neo_*` |
| OQ16 | nvidia_peermem 모듈 로드 가능 여부 | `modprobe nvidia_peermem` (root 권한) |
| OQ17 | CUDA stream priority 의 latency 영향 | 직접 측정 |
| OQ18 | shm_broadcast 의 dispatch latency 분포 | profile log + percentile |
