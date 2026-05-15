# After-NEO 성능 개선 구현 plan

> 작성 시각: 2026-05-15 KST
> base branch: `feat/neo-amx-apply` (commit `1636a6692`)
> 본 plan 의 입력: `analysis/` 13 .md (Phase A-F + 측정 정정)
> 본 plan 은 **구현 plan**. 작업 진입 시 sub-branch 분기 + Phase 별 PR / commit gate.

---

## 0. plan 의 위치 + 출발점

### 현재 baseline (commit `64f9e0c48` 정합)
- 측정 영역: 200p × 8192 in/out, max_num_seqs=256, fp8 KV, gmu=0.92
- v1.6 fix 영역 (shape mismatch + 22 strict 19/19)
- output_tps = **1,638** (vanilla 4,886 의 33.5% — 200p 단위 영역)
- async_swap = 100% async path (sync fallback 0)
- chain firing **4.5%** 영역 (mirror=10/running=222)

### 본 plan 의 목표

H100 paper claim +14% (vanilla 대비) 영역 도달 시도. 또는 throughput 적어도 vanilla 수준 도달.

**진짜 bottleneck (실측 정정 후)**:
- cdec sub-batch query `b1_avg = 0` — cdec 가 actual work 영역 안 도달 (chain firing 4.5% 영역 의 결과)
- 즉 **CPU 가속 (AMX/AVX) 의 효과 영역은 chain firing 80-99% 영역 도달 후** 의미.

---

## 1. 정정된 fact (Phase 1 측정 결과)

| 항목 | 이전 추정 | 실측 (2026-05-15) | 의미 |
|---|---:|---:|---|
| cdec_wait_avg | 8.75 ms/layer | **2.55 ms/layer** | CPU 가 더 빠름 |
| GPU/CPU ratio | 89-94× | **32×** | (AMX 5-10× 가속 시 3-6× 격차 잔존) |
| b1_avg (cdec query) | (미상) | **0** | cdec sub-batch 비어 있음, 본 work 영역 안 도달 |
| skip_gpu / cdec_count | (미상) | 90.7% | 실제 cdec work 영역 9.3% |
| TLB miss | (-) | **0.01%** | 1GB hugepage 효과 1-3% (limited) |
| DDR5 BW 활용 | (-) | **11.2%** | BW 가속 영역 효과 작음 |
| NUMA placement | (검증) | **99.6-99.9% 정합** | NUMA local 영역 가속 효과 ~0 |
| async swap fallback | (잘못 분석) | **0** (100% async) | swap path 대부분 wall hidden |
| memcpy BW (page size 비교) | (-) | 4KB 9.8 / 2MB 10.1 / 1GB 9.9 GB/s | hugepage 영역 +1-3% |

→ **본 환경의 dominant bottleneck = chain firing 영역 도달**. hw 가속 단독은 limited.

---

## 2. 구현 phase 개관 (Phase 1 완료, Phase 2-7 진행)

```
Phase 1 측정 (완료) ─────────────────────────┐
                                              ▼
Phase 2 chain firing 영역 도달 (★ 핵심) ──→ Phase 7 (통합)
  │                                           ▲
  ▼                                           │
Phase 3 hw quick wins (NUMA/OMP/stream) ────┤
  │                                           │
  ▼                                           │
Phase 4 1GB hugepage 도입 ─────────────────┤
  │                                           │
  ▼                                           │
Phase 5 AVX-512 fast_exp + BF16 dot ──────┤
  │                                           │
  ▼                                           │
Phase 6 AMX qk_product (chain 99% 영역 후) ─┘
```

총 영역: **18-26 주** (Phase 2-7 합산, parallel 진행 시 영역 단축).

---

## Phase 2 — chain firing 영역 도달 (★ 본 plan 의 dominant 영역)

**기존 분석에서 도출된 핵심 사실**: AMX/AVX 가속 의 효과는 chain firing 영역 (80-99%) 에서만 발화. 현재 4.5% 영역에서는 가속 효과 사실상 0.

### 목표
- chain firing 영역 80-99% 도달
- 그러면서 throughput 회귀 영역 최소화 (이전 try102 의 627 tps 영역 회피)
- 22 strict 19/19 유지

### 작업 영역
1. **Option C/L/M2 sweep** — 이전 시도 영역 (try77~try105) 의 fact 재검토. 어떤 환경 영역 조합이 chain firing 영역 + throughput 균형 영역
2. **workload 조정** — paper sweet spot 영역 (작은 batch + 긴 seq + larger mirror cap) 도달 시도. 본 plan workload (200p × 8192) 변경 검토
3. **mirror cap sweep** — 80 → 120 → 200 영역
4. **cdec_executor max_workers cap** 영역의 layer 의존성 정량 (SUB_023 settled 영역 재검증)

### Gate
- chain firing % ≥ 50% + throughput ≥ vanilla baseline 의 50% 도달 시 → Phase 3 진입
- chain firing 영역 안 도달 시 → 본 plan 의 hw 가속 영역 (Phase 3-6) 의 효과 영역 0 인 fact 가 확정 → workload 영역 변경 또는 plan 영역 재정의

### 위험
- chain firing 99% 영역 도달 시 try102 의 627 tps 영역 회귀 가능 (이전 fact)
- workload 영역 변경 시 baseline 영역 재정의 필요

---

## Phase 3 — hw quick wins (sequence: 영향 작음, cost 작음)

각 항목 측정 영역 분리.

### Phase 3.1 — Persistent OMP team
- 작업: `csrc/cpu/pacpu/core.h` 의 OMP team 영역에 `omp_set_dynamic(0)` + KMP_AFFINITY tuning
- 측정: cdec dispatch 영역 시 OMP launch overhead 의 변화
- 예상 효과: wall **0.5-1%** (Phase 2 도달 후 영역)

### Phase 3.2 — KMP_BLOCKTIME=0
- 작업: env var 만 도입
- 측정: thread wake latency 영역
- 예상 효과: wall **0.5-1%**

### Phase 3.3 — CUDA Stream Priority
- 작업: `cudaStreamCreateWithPriority` 도입, gdec attention stream 을 high-priority
- 측정: gdec sub-batch[0] 영역 의 latency tail
- 예상 효과: wall **0.5-1%** (tail 영역)

### Phase 3.4 — NUMA local 영역 검증 (이미 정합)
- 작업: 측정 영역만 (Phase 1 fact 보존)
- pinned staging buffer 의 NUMA 정합 확인
- 예상 효과: **0%** (이미 정합)

### Gate
- 각 항목 wall 차이 ≥ 0.5% → 도입
- 누적 wall 절감 ≤ 2% 영역에서 다음 Phase 진입

---

## Phase 4 — 1GB Hugepage 도입 (★ 추가 영역, Phase 1 검증 후)

**Phase 1 측정 영역의 검증 결과**:
- container 안에서 1GB hugepage 활용 **가능** (privileged + cap_sys_admin)
- pool reserve + hugetlbfs mount + anonymous `mmap(MAP_HUGETLB | MAP_HUGE_1GB)` 모두 동작
- 그러나 **vllm 영역의 활용 = 0** (코드 영역에서 도입 안 됨)
- 효과 영역: memcpy bench +1-3% (TLB miss 이미 0.01% 영역 정합)

### 작업 영역

#### Phase 4.1 — pool reserve + mount setup
- Init script 영역 (vllm 시작 전):
  - `echo N > /sys/devices/system/node/nodeX/hugepages/hugepages-1048576kB/nr_hugepages` (N = 16-32 per NUMA)
  - `mkdir -p /mnt/huge1g && mount -t hugetlbfs -o pagesize=1G none /mnt/huge1g`
- run script (`eval/run_neo_standard.sh`) 영역에 통합
- container privileged 영역 의존 — 운영 영역 검토 필요

#### Phase 4.2 — vllm 측 pinned staging buffer alloc 변경
- 대상 코드: `vllm/v1/worker/gpu_model_runner.py:6390-6394` 의 `_neo_init_swap_staging`
  ```python
  k = torch.empty(shape, dtype=spec.dtype, pin_memory=True)  # 현재
  v = torch.empty(shape, dtype=spec.dtype, pin_memory=True)
  ```
- 변경 패턴:
  1. `mmap(MAP_HUGETLB | MAP_HUGE_1GB)` 또는 hugetlbfs file 영역에서 alloc
  2. `torch.from_blob` 또는 ctypes 영역으로 torch tensor 영역 래핑
  3. `cudaHostRegister` 영역으로 CUDA pinned 영역 등록
- pool 부족 시 graceful fallback (default `pin_memory=True`)

#### Phase 4.3 — KV cache (CPU 영역) alloc 변경 (만약 NeoCpuKvBuffer 영역 사용 시)
- 대상 코드: `vllm/v1/core/sched/neo_cpu_kv_buffer.py` 의 alloc 영역
- 동일 패턴 (hugetlbfs file 또는 anonymous MAP_HUGETLB)

### Gate
- 측정: vllm process 의 `/proc/<pid>/smaps` 의 Private_Hugetlb 영역 > 0 도달
- 측정: 1GB pool 의 free count 감소 (실제 사용 영역 확인)
- 예상 효과: wall **1-3%** (Phase 1 의 memcpy bench 영역 정합)

### 위험
- `cudaHostRegister` 의 hugetlbfs memory 호환성 — CUDA driver 영역 검증 필요
- pool 부족 시 alloc 실패 → graceful fallback 영역 안전 path 필수
- container privileged 영역 없으면 pool reserve 불가 — 운영 영역 의존

---

## Phase 5 — AVX-512 (fast_exp + BF16 dot product)

dev (i9-12900KF, AVX-512 native) + prod (Xeon SPR, AVX-512 + AMX native) 모두 가능.

### Phase 5.1 — `_mm512_fast_exp_ps` ports
- 대상: pacpu softmax (`csrc/cpu/pacpu/pacpu.ispc:109-140`)
- 재사용: `csrc/cpu/cpu_arch_macros.h::_mm512_fast_exp_ps`
- ISPC kernel 의 exp 영역을 명시적 AVX-512 intrinsic 통해 교체
- 예상 효과: softmax 영역 2-3× (영역 작음 = 전체의 5-10%)

### Phase 5.2 — AVX-512 BF16 `vdpbf16ps`
- 대상: pacpu qk_product / av_product (BM01 / BM02)
- dtype: FP16 → BF16 변환 + FP32 accumulator
- ISPC kernel 의 GEMM 영역 명시 intrinsic 교체
- 예상 효과: qk_product 영역 1.5-2× (chain firing 영역 도달 시)

### Gate
- 분포 유사성 게이트: per-token logprob max abs diff < 1e-3
- v1.1 SUB_006 v42 의 −3.16% 회귀 영역 회피
- chain firing 영역 도달 후 측정

### 위험
- BF16 변환의 정확도 — careful 검증 영역 필수

---

## Phase 6 — AMX qk_product (prod only)

본 plan 의 최종 가속 영역. **Phase 2 chain firing 영역 도달 + Phase 5 AVX-512 영역 안정화 후 진입**.

### 사전 조건
- chain firing 영역 80% 이상 (Phase 2 통과)
- AVX-512 BF16 영역 정확도 검증 (Phase 5 통과)
- AMX dev 검증 path 영역 (Intel SDE simulator 또는 prod 직접 측정)

### 작업
1. `csrc/cpu/pacpu/pacpu_amx.cpp` 신규 — AMX-TMUL `tdpbf16ps` 기반 qk_product
2. 재사용: `csrc/cpu/micro_gemm/cpu_micro_gemm_amx.hpp::TileGemm224` template
3. layout 변환: NEO 의 `[BLOCK_SIZE, HEAD_DIM]` → AMX tile expected layout
4. build flag: `-mamx-tile -mamx-bf16 -mavx512bf16`
5. runtime gate: `__has_amx()` cpuid + fallback (Phase 5 의 AVX-512)

### Gate
- 분포 유사성 (logprob max abs diff < 1e-3)
- BM01 qk_product 의 prod 측정 speedup ≥ 4× (이론 4-7×)
- chain firing 99% 영역 의 cdec_wait 영역 wall 단축

### 위험
- dev (i9-12900KF) 의 AMX 미지원 — prod 만 검증 가능
- SDE simulator 의 검증 영역 cost

---

## Phase 6.5 — GPU 측 가속 (NVLink P2P + PTX + NVSHMEM)

본 plan 의 GPU 측 영역. CPU 가속 (Phase 5/6) 과 parallel 진행 가능.

### 환경 fact (Phase 1 측정 + 라이브러리 설치 후)

- **NVLink 4 full-mesh** (NV18 영역 — H100×8 의 900 GB/s/pair 영역)
- **CUDA 12.8 + nvcc** 가용 (PTX 영역 작성 가능)
- **gdrcopy v2.4.4** install 완료 (`/usr/local/lib/libgdrapi.so.2.4`) — GPU memory ↔ user-space direct mapping
- **NVSHMEM** pip package `nvidia_nvshmem_cu12-3.4.5` 설치됨 — same-node symmetric memory
- **GPU P2P 동작 확인**: GPU0 ↔ GPU1 cudaDeviceCanAccessPeer = OK
- **CUTLASS** 영역 vllm csrc 영역에 이미 포함

### Phase 6.5.1 — NVLink P2P direct copy (single-node 영역)

**대상**: NEO 의 KV cache migration 영역 (CPU swap 외의 inter-GPU 영역). 현재 NCCL all_reduce 영역만 NVLink 활용 — KV migration 영역은 host 경유 (PCIe round-trip)

**작업**:
- `cudaMemcpyPeerAsync` 도입 — GPU 간 direct DMA (host 우회)
- 대상 코드: 해당 영역의 inter-GPU KV copy (NEO 의 prefix sharing 또는 TP 영역 외 KV migration 영역)
- 본 plan workload (TP=8) 에서는 영역 작음 — 다른 workload (PP, 또는 disaggregated prefill) 에서 큰 효과

**효과 영역**: wall **0-2%** (본 workload 영역 작음), multi-node disaggregated prefill 영역 도입 시 **10-30%**

### Phase 6.5.2 — PTX inline kernel (gdec attention hot path)

**대상**: NEO 의 GPU sub-batch[0] (gdec) 의 attention forward 영역. flash-attention 의 inner loop 영역 의 추가 가속 가능.

**작업**:
1. flash-attention 의 kernel 영역 disassembly (`cuobjdump --dump-sass`) → 현재 PTX 영역의 inefficiency 영역 식별
2. inline PTX 영역 도입:
   - `asm volatile ("..." : ... :);` 영역 — CUDA C 안의 PTX directive
   - 대상 영역: BF16 의 matmul (Tensor Core `mma.sync.aligned.m16n8k16` 영역)
   - 또는 softmax 의 `exp.approx.f32` (PTX fast-math)
3. 또는 CUTLASS template 활용 — vllm csrc 영역에 이미 포함

**효과 영역**: wall **2-5%** (gdec attention 의 inner loop 영역의 효율 영역 향상)

**위험**:
- PTX intrinsic 영역의 정확도 영향 (예: `exp.approx.f32` 의 정밀도)
- SM 아키텍처 의존 (sm_90 영역 H100 specific) → portable 영역 작음
- flash-attention 영역의 upstream 영역 fork 영역 cost

### Phase 6.5.3 — NVSHMEM symmetric memory (multi-GPU collective)

**대상**: TP=8 all_reduce 영역의 low-latency 영역 (NCCL 의 small-message 영역 대안).

**작업**:
- `import nvshmem` → 8 GPU symmetric memory window 영역 alloc
- TP collective (`tensor_model_parallel_all_reduce`) 영역의 small-message path 영역 NVSHMEM 으로 대체
- 대상 코드: `vllm/distributed/parallel_state.py:519` `_all_reduce_out_place`

**효과 영역**: wall **0.5-2%** (NCCL all_reduce 2.86% 영역의 latency 영역 일부 단축)

**위험**:
- NVSHMEM 의 vllm 통합 영역 cost
- NCCL 와의 hybrid 영역 검증 필요

### Phase 6.5.4 — gdrcopy 활용 (CPU ↔ GPU low-latency direct mapping)

**대상**: NEO 의 swap-in path 의 sync H2D 영역 (BM09 영역, swap-in 4.3 GB/s).

**작업**:
- `gdr_map()` + `gdr_get_info()` → GPU memory ↔ user-space direct mapping
- CPU side 의 memcpy 영역으로 GPU memory 직접 write (cudaMemcpy 우회)
- 작은 영역 (KB-MB 영역) 의 sync transfer 영역에 효과 큼

**효과 영역**: wall **1-3%** (swap-in path 영역의 latency 영역 절감)

**위험**:
- gdrcopy 의 kernel module (`nvidia_peermem` 또는 `gdrdrv`) 영역 — container 영역 에서 modprobe 불가
- 본 환경 `nvidia_peermem` 미로드 영역 — host side modprobe 영역 필요
- userspace 영역 lib 만 build 됨 (kernel module 미설치) — full functional 영역의 검증 필요

### Phase 6.5 Gate

- NVLink P2P bench (이미 동작 확인) → wall 측정
- PTX inline 의 정확도 검증 (logprob diff)
- NVSHMEM all_reduce 의 NCCL 대비 latency 측정
- gdrcopy 의 kernel module 영역 의존 — 호스트 운영 영역 검토

---

## Phase 7 — 통합 + paper claim 영역 도달 시도

### 작업
1. Phase 2-6 의 영역 전체 통합
2. workload sweep (paper sweet spot 영역 매칭 시도)
3. vanilla baseline 대비 throughput 측정
4. paper claim H100 +14% 영역 도달 시도

### Gate
- 통합 throughput ≥ vanilla baseline 도달 (이론 NEO net-win 영역)
- 22 strict 19/19 유지
- 분포 유사성 게이트 통과

### 위험
- chain firing 영역 도달 시 throughput collapse 패턴 (try102 reference)
- workload 가 paper sweet spot 영역 매칭 안 됨 → H100 영역 net-win 불가 영역의 결정

---

## 합계 + 예상 누적 wall 절감

| Phase | 항목 | 단독 절감 (정정 후) | 누적 |
|---|---|---:|---:|
| 1 | 측정 완료 | (fact) | — |
| 2 | **chain firing 영역 도달** | **결정적 영역** | + 영역 의존 |
| 3 | hw quick wins (OMP + stream + KMP) | 2-3% | +2-3% |
| 4 | 1GB hugepage | 1-3% | +3-6% |
| 5 | AVX-512 fast_exp + BF16 | 3-6% (Phase 2 도달 후) | +6-12% |
| 6 | AMX qk_product (chain 99% 영역) | 4-12% (Phase 2 도달 후) | +10-24% |
| 6.5 | GPU 측 가속 (P2P + PTX + NVSHMEM + gdrcopy) | 3-12% | +13-36% |
| 7 | 통합 + workload 영역 | 0-5% (workload 영향) | +13-41% |

### 핵심 finding 재정리

- **Phase 2 (chain firing 영역 도달)** 없이는 Phase 5/6 (AVX-512 + AMX) 영역 효과 작음 (cdec 자체가 work 영역 안 도달)
- **Phase 4 (1GB hugepage)** 는 측정 영역 효과 1-3% — limited 그러나 추가 영역. cost 작음 (코드 영역 + setup)
- **Phase 3 hw quick wins** 의 효과 영역 작음 (이미 정합) — 그러나 cost 작음 = 진행 가치 있음

---

## 진행 순서 + branch 전략

- **base branch**: `feat/neo-amx-apply` (현재)
- **Phase 별 sub-branch**: `feat/neo-amx-apply-phase{2..7}` 또는 통합 branch 안 단계 commit
- Phase 진입 전 사용자 승인 영역
- Phase 결과 PR / commit gate 영역 후 다음 Phase 진입

### 본 plan 의 hint 영역

본 plan 의 핵심: **chain firing 영역 도달이 dominant**. hw 가속 영역 도입 전에 Phase 2 영역 본격 진행 필요. Phase 3 (hw quick wins) 는 Phase 2 와 parallel 진행 가능 (cost 작음).

---

## Open questions (Phase 진입 전 답 영역)

| OQ | 내용 | Phase |
|---|---|---|
| OQ-A | chain firing 영역 80-99% 도달 시 throughput 영역 회귀 영역 정도 (try102 의 627 tps 영역 vs 새 영역) | Phase 2 |
| OQ-B | workload (200p × 8192) 의 paper sweet spot 영역 매칭 여부 (작은 batch / 긴 seq 영역 변경 시 NEO 효과 영역 변화) | Phase 2 |
| OQ-C | cdec_executor max_workers cap (현 2) 의 AMX 적용 후 의존성 변화 | Phase 6 |
| OQ-D | cudaHostRegister 의 hugetlbfs memory 호환성 | Phase 4 |
| OQ-E | BF16 변환의 분포 유사성 영역 (v1.1 SUB_006 v42 의 −3.16% 회귀 영역 재발 회피 영역) | Phase 5/6 |
| OQ-F | 1GB hugepage pool reserve 의 운영 영역 자동화 (container restart 시 영역) | Phase 4 |
| OQ-G | NVLink P2P 의 본 workload (TP=8) 의 효과 영역 (이미 NCCL via NVLink 활용 중) | Phase 6.5.1 |
| OQ-H | PTX inline 의 정확도 영향 + SM 아키텍처 의존 영역 | Phase 6.5.2 |
| OQ-I | NVSHMEM 의 vllm 통합 영역 cost | Phase 6.5.3 |
| OQ-J | gdrcopy kernel module (nvidia_peermem/gdrdrv) 의 호스트 영역 활성화 가능 여부 | Phase 6.5.4 |

---

## 도구 + 라이브러리 setup 영역 (Phase 6.5 진입 전)

본 plan 의 Phase 6.5 영역의 의존 영역. **이미 install 완료** 영역:

| 라이브러리 / 도구 | 영역 | 위치 |
|---|---|---|
| `nvcc` (CUDA 12.8) | PTX 영역 build | `/usr/local/cuda/bin/nvcc` |
| `gdrcopy` v2.4.4 (userspace) | GPU memory ↔ user-space direct mapping | `/usr/local/lib/libgdrapi.so.2.4` + `/usr/local/include/gdrapi.h` |
| NVSHMEM (pip) | same-node symmetric memory | `nvidia_nvshmem_cu12-3.4.5` (vllm_dev_prj venv) |
| CUTLASS | header-only template | `csrc/quantization/w8a8/cutlass`, `.deps/qutlass-src/third_party/cutlass` |
| PCM | Intel performance counter | `/usr/local/bin/pcm*` (build 영역 from `opcm/pcm`) |
| `perf` | Linux performance counter | `/usr/local/bin/perf` (5.15 from linux-tools-generic) |
| `numastat`, `pmap`, `ipcs` | NUMA + memory | apt 기본 영역 |
| `libhugetlbfs` | hugepage helper | apt 영역 설치 — Python mmap 영역 intercept 안 됨 (Phase 4 영역에서 코드 영역 도입 필요) |

**미설치 / 활성 영역**:
- `nvidia_peermem` kernel module (gdrcopy 의 GDR 영역 의존) — container modprobe 불가, **host 영역 설치 필요**
- `gdrdrv` kernel module (gdrcopy 의 fallback path) — 동상
- AMX BF16 토큰 정확도 검증 영역 (Phase 5/6 영역 의 사전 단계)
