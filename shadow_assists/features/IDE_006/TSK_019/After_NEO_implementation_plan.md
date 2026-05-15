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
[Track A — 지엽 영역 (incremental, 기능 가속)] │
Phase 2 chain firing dial-up (72% → 99%) ─→ Phase 7 (통합 영역)
  │ (Option C/L/M2 ON 환경 영역 진입)         ▲
  ▼                                           │
Phase 3 hw quick wins (NUMA/OMP/stream)       │
  │                                           │
  ▼                                           │
Phase 4 1GB hugepage 도입                    │
  │                                           │
  ▼                                           │
Phase 5 AVX-512 fast_exp + BF16              │
  │                                           │
  ▼                                           │
Phase 6 AMX qk_product (chain 99% 환경 내)   │
  │                                           │
  ▼                                           │
Phase 6.5 GPU 측 (P2P + PTX + NVSHMEM) ────┤
                                              │
[Track B — quantum jump (알고리즘 영역)]      │
Phase 8.1 Speculative cdec dispatch         │
Phase 8.2 Hierarchical KV tier              │
Phase 8.3 Cross-request prefix sharing      │
Phase 8.4 Adaptive sub-batch sizing         │
Phase 8.5 Tensor Core LSE merge             │
Phase 8.6 PD disaggregation + NEO           │
Phase 8.7 Approximate (sparse) attention    │
Phase 8.8 Predictive cdec scheduling ──────┘
```

총 영역: **18-26 주** (Phase 2-7 합산, parallel 진행 시 영역 단축).

---

## Phase 2 — chain firing dial-up (72% → 99%, Option C/L/M2 ON 환경 영역 진입)

**2026-05-15 KST 정정 v7 (Phase 2 부활 + metric 정정)** — CLAUDE.md Objective ("CPU 활용률 극대화 + idle 금지") 영역 정합 영역.

### metric 정합 영역 (chain firing 영역 metric 영역 표)

`eval/neo_22_items_monitor.sh` 영역 의 `CHAIN_PCT = active/total * 100` 영역 = mirror set 영역 의 absolute 영역 (fork attempt 영역 의 active 영역 비율 영역, step 영역 의 모든 req 영역 base). **chain firing 영역 의 진짜 의미 영역 의 representation 영역 아님 영역**.

진짜 chain firing 영역 metric 영역:
- **layer fire** = `1 - skip_gpu / cdec_count` — cdec dispatch 영역 의 layer 영역 중 cdec sub-batch 영역 발화 영역 layer 영역 비율 영역
- **active/eligible** = fork 가능 영역 중 active fork 영역 비율 영역 (보통 100% 영역 — eligibility 영역 통과 영역 모두 fork 영역 성공)

### baseline fact 영역

| 측정 dir | env | **chain firing (layer fire)** | active/total (monitor) | output_tps |
|---|---|---:|---:|---:|
| `20260514_233540_neo_standard` (v1.6 best) | Option OFF (minimal) | **72%** | 4.0% | **2,157** |
| `20260514_163627_neo_standard` | Option OFF | (자료) | 2.9% | 2,217 |
| try84 (5/10) | Option C/L/M2 ON | **98.7%** | (자료) | (자료) |
| try102 (5/11) | Option C/L/M2 ON | **99.0%** | 99% | **627** (collapse) |

→ **chain firing 99% 환경 영역 = 이미 입증** (try84/try102 영역). 본 plan 의 v1.6 best baseline 영역 = chain firing **72%** (layer fire) + 2,157 tps.

### Phase 2 영역 의 목표 (CLAUDE.md Objective 정합)

> CPU 활용률 영역 극도 영역 + idle / 낮은 영역 허락 안 함

- v1.6 best 영역 의 chain firing **72%** 영역 = CPU 영역 의 cdec dispatch 영역 의 layer 영역 의 72% 발화 영역 — 양호 영역 그러나 100% 영역 미달
- try102 영역 의 99% 영역 = CPU 활용률 영역 99% 영역 도달 영역 = **Objective 달성 영역** (그러나 throughput collapse 영역 → Phase 3-6.5 가속 영역 으로 회복 영역)
- Phase 2 영역 = **72% → 99% dial-up** (Option C/L/M2 ON 환경 영역 진입 영역)

### 작업 영역
1. **Option C/L/M2 ON 환경 영역 진입** (try84/try102 영역 의 환경 영역 의 재현 영역)
2. 측정 baseline 영역 = chain firing 99% + throughput **627 tps** (try102 영역 의 collapse fact 영역)
3. Phase 3-6.5 영역 의 가속 영역 = **chain firing 99% 환경 영역 에서 cdec_wait 영역 ↓ → throughput collapse 영역 회복 영역**
4. mirror cap / Option 영역 sweep 영역 정합 영역 (이전 try77-try105 영역 의 fact 영역 재검토)

### Gate
- chain firing layer fire ≥ 95% **AND** throughput ≥ v1.6 best 2,157 tps 의 70% 도달 시 → Phase 3 진입
- chain firing ≥ 95% 그러나 throughput 영역 collapse 영역 (try102 영역의 627 tps 영역) — Phase 3-6.5 의 가속 영역 으로 회복 영역 시도 영역
- 회복 영역 안 됨 → workload 영역 (paper sweet spot) 영역 매칭 영역 또는 plan 영역 재정의

### 위험
- chain firing 99% 영역 + throughput 627 tps 영역 fact 영역 = try102 영역 의 fact 영역 영역. 가속 영역 회복 영역 의 ceiling 영역 ?
- Option C/L/M2 ON 영역 의 env 영역 정합 영역 의 SUB_023 영역 의 +4 worker 영역 -52% 영역 fact 영역 회피

### 진짜 NEO sweet spot 영역 도달 영역

본 plan 영역의 진짜 영역 = chain firing **99%** (CPU 활용률 99% — Objective 달성 영역) + AMX/AVX/hw 영역 가속 영역 으로 throughput collapse 회복 영역 → **NEO 의 paper sweet spot 영역 (CPU 가 GPU 영역 비등 영역) 도달 시도**.

### ★ Phase 2 영역 의 dependency 영역 (정정 v10, 2026-05-15 KST)

이전 분석 영역 (try102 v1.5 영역 vs v1.6 best 영역 비교) 영역 = **코드 베이스 영역 다름** (v1.5 858b6df7a vs v1.6 64f9e0c48) — chain firing 영역 만 영역의 cause 영역 영역 분리 영역 X. 즉 v1.5 → v1.6 영역 의 throughput ↑ (627 → 2,157) cause 영역 = chain firing 영역 ↓ 단독 영역 X.

#### 코드 + 동적 fact 영역 기반 영역 판별

**코드 영역** — `sub_batch_executor.py:forward_double`:
- GPU sub-batch[0] (gdec) + CPU sub-batch[1] (cdec) **동시** 영역
- step wall = **max(gdec_wall, cdec_wall)**

**동적 fact 영역** (v1.6 best, commit `64f9e0c48`):
- gpu_avg = **0.08 ms / layer** (gdec attention)
- cdec_wait_avg = **2.37 ms / layer** (cdec sub-batch)
- ratio = 32× (cdec 영역 32× 영역 느림)
- b0_avg = 1,115 (gdec query) / b1_avg = 5 (cdec query, 현 4% chain firing)

**chain firing ↑ 영역의 cause 영역 분석**:
- chain firing ↑ → **b1_avg ↑** → cdec_wall (≈ cdec_wait × b1_avg / max_workers=2) ↑
- 본 환경 cdec_wait >> gpu_avg (32×) → cdec_wall 영역 dominant 영역 → **wall ↑ → throughput ↓**

#### Phase 2 영역 단독 vs 결합 영역 fact

| 단독 영역 | direct cause 영역 |
|---|---|
| chain firing 영역 ↑ 단독 영역 | **throughput ↓** (cdec_wall dominant) |
| AMX/AVX 가속 단독 (4% chain firing 영역) | sub-marginal (cdec 영역 작음) |
| **chain firing ↑ + AMX/AVX 결합** | **throughput ↑** 의 진짜 cause (cdec_wall ≈ gdec_wall 영역 balance 영역 도달 = NEO asymmetric pipelining 영역의 진짜 sweet spot) |

→ **Phase 2 영역의 진짜 의미 영역**: 단독 진입 X. **Phase 5/6 (AMX/AVX 영역 cdec_wait ↓) 과 동시 영역 진입 영역만 의미**. cdec_wall (= cdec_wait × b1_avg) 영역 의 b1_avg ↑ 효과 영역 + cdec_wait ↓ 효과 영역 의 결합 영역 → cdec_wall 영역 의 gdec_wall 영역 비등 영역 도달 = throughput 영역 최대화.

#### Gate 정정

- Phase 2 단독 영역 진입 영역 X — Phase 2 + Phase 5/6 영역 결합 영역 진입 영역만 의미
- 측정 baseline 영역 = same-commit 영역의 Option ON vs OFF 영역 비교 영역 필요 (현 측정 영역 영역 없음)
- chain firing layer fire ≥ 95% **AND** cdec_wait_avg ≤ gpu_avg × 영역 (10× 영역) 영역 → wall balance 영역 도달 영역 정합

---

## Phase 3 — hw quick wins (sequence: 영향 작음, cost 작음)

각 항목 측정 영역 분리.

### Phase 3.1 — Persistent OMP team
- **모듈/함수**: `csrc/cpu/pacpu/core.h:296` `#pragma omp parallel` + `:314,333` barrier
- **작업**: OMP team 영역에 `omp_set_dynamic(0)` + KMP_AFFINITY 영역 설정 + `omp_set_num_threads(12)` persistent 영역
- **영향 영역**: pacpu kernel 영역 의 OMP team launch overhead 영역 (`attn_one_seq` 영역 entry 영역 매 layer 당 영역 의 team launch + barrier 영역 의 cost 영역)
- **측정 방법**: PROFILE log 의 `cdec_wait_avg` 영역 차이 / `perf stat -e sched-switches,context-switches` 영역
- **fact 영역의 영향 영역**: Phase 1 측정 영역 의 cdec_wait 2.55 ms / layer 영역 의 OMP 영역 부분 영역

### Phase 3.2 — KMP_BLOCKTIME=0
- **모듈/함수**: env var `KMP_BLOCKTIME=0` (libiomp/libgomp 영역 의 thread idle blocktime)
- **작업**: `eval/run_neo_standard.sh` 영역에 env 추가
- **영향 영역**: pacpu 영역 의 OMP thread 영역 의 wake latency 영역 (default 200ms → 0ms 영역)
- **측정 방법**: cdec dispatch 영역 의 tail latency (`cdec_wait_max`) 차이

### Phase 3.3 — CUDA Stream Priority
- **모듈/함수**: `vllm/v1/worker/sub_batch_executor.py` 영역 의 stream 영역 생성 영역 → `cudaStreamCreateWithPriority(-1)` 영역 적용
- **영향 영역**: gdec sub-batch[0] forward stream 영역 의 OS scheduling 영역 jitter 영역
- **측정 방법**: gpu_avg (PROFILE 영역) 의 tail (gpu_max) 차이

### Phase 3.4 — NUMA local pinned staging
- **모듈/함수**: `vllm/v1/worker/gpu_model_runner.py:6390-6394` `_neo_init_swap_staging` 영역
- **현 fact**: Phase 1 측정 영역 의 NUMA 정합 99.6-99.9% 영역 (이미 정합 영역)
- **작업**: 측정 영역만 (regression 영역 검증)

### Gate
- 각 항목 의 측정 영역 → cdec_wait_avg / cdec_wait_max / gpu_max / context-switches 영역 의 fact 변화 영역
- 본 Phase 영역 = **OMP overhead 영역 의 fact 영역 정량 영역** (영향 영역의 크기 영역 = 측정 후 영역)

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
- 측정: vllm process 의 `/proc/<pid>/smaps` 의 `Private_Hugetlb` > 0 도달
- 측정: 1GB pool 의 `free_hugepages` count 감소
- **영향 영역**: pinned staging buffer 영역 의 TLB miss 영역 (Phase 1 측정 영역 의 0.01% → 더 낮은 영역 가능). 그러나 본 환경 영역 의 TLB miss 영역 이미 낮음 → 효과 영역 의 fact 영역 = 측정 후 결정 영역.

### 위험
- `cudaHostRegister` 의 hugetlbfs memory 호환성 — CUDA driver 영역 검증 필요
- pool 부족 시 alloc 실패 → graceful fallback 영역 안전 path 필수
- container privileged 영역 없으면 pool reserve 불가 — 운영 영역 의존

---

## Phase 5 — AVX-512 (fast_exp + BF16 dot product)

dev (i9-12900KF, AVX-512 native) + prod (Xeon SPR, AVX-512 + AMX native) 모두 가능.

### Phase 5.1 — `_mm512_fast_exp_ps` ports
- **모듈/함수**: `csrc/cpu/pacpu/pacpu.ispc:109-140` `softmax` (3-pass: max reduce + exp + sum/div) 영역의 **exp 영역** 영역
- **재사용**: `csrc/cpu/cpu_arch_macros.h::_mm512_fast_exp_ps` (5-degree polynomial approximation)
- **변경 영역**: ISPC built-in exp (현재 `__svml_expf16` 또는 polynomial 자동 lower) 영역 → 명시적 `_mm512_fast_exp_ps` 영역 호출 영역
- **영향 영역**: `softmax` kernel 의 exp 계산 영역 (3-pass 의 2번째 pass)
- **측정 방법**: `softmax` kernel 영역 micro-benchmark (per-call ns) + PROFILE `cdec_wait_avg` 차이 영역
- **정확도 영역**: polynomial degree 5 (FP32 ~1e-6) 영역 — 분포 유사성 게이트 (logprob max abs diff < 1e-3)

### Phase 5.2 — AVX-512 BF16 `vdpbf16ps`
- **모듈/함수**: `csrc/cpu/pacpu/pacpu.ispc:5-69` `qk_product` + `:71-107` `av_product` 영역 의 inner GEMM loop (`foreach l = 0...HEAD_DIM`)
- **dtype 변환**: FP16 (`data_t` in `dtype.h`) → BF16 storage 영역 + FP32 accumulator 영역 (현 `itmd_t`) 유지 영역
- **변경 영역**: ISPC `varying float16 q * k → float` FMA 영역 → 명시적 `_mm512_dpbf16_ps(acc, q_bf16, k_bf16)` intrinsic 영역
- **영향 영역**: `qk_product` / `av_product` kernel 영역 의 inner FMA loop 영역 — **cdec_wait_avg 의 dominant 영역 (Phase A 의 산술 강도 분석 영역: qk AI=30-50, av AI=7)**
- **측정 방법**: kernel micro-benchmark (per-call ns) + PROFILE `cdec_wait_avg` 차이 + 분포 유사성 게이트
- **정확도 영역**: v1.1 SUB_006 v42 영역 의 BF16 manual kernel 영역 의 token loss 2.84→3.70% / throughput −3.16% 회귀 영역 fact 영역 → careful BF16 conversion + FP32 accumulator 영역 유지 필수

### Gate
- micro-benchmark: kernel 단독 ns 영역 비교 (softmax kernel + qk/av kernel)
- 분포 유사성: per-token logprob max abs diff < 1e-3
- chain firing 영역 도달 후 PROFILE `cdec_wait_avg` 차이 측정

### 위험
- BF16 변환의 정확도 영역 (v1.1 회귀 fact 영역 회피)
- ISPC kernel 의 영역 boundary 영역 — code maintainability 영향

---

## Phase 6 — AMX qk_product (prod only)

본 plan 의 최종 가속 영역. **Phase 2 chain firing 영역 도달 + Phase 5 AVX-512 영역 안정화 후 진입**.

### 사전 조건
- Phase 2 영역 의 Option C/L/M2 ON 환경 영역 진입 (chain firing layer fire ≥ 95%)
- Phase 5.2 영역 의 AVX-512 BF16 영역 의 정확도 검증 통과 (logprob diff < 1e-3)
- AMX dev 검증 path 영역 (Intel SDE simulator 영역 또는 prod 직접 측정)

### 작업
- **모듈/함수**: `csrc/cpu/pacpu/pacpu_amx.cpp` 신규 — `qk_product` 영역 의 AMX-TMUL 영역 구현
- **재사용**: `csrc/cpu/micro_gemm/cpu_micro_gemm_amx.hpp::TileGemm224` template (M=16, K=16, N=16, BF16 input, FP32 acc)
- **AMX kernel 영역 분석**:
  - 1 `tdpbf16ps` instruction 영역 = 16×16×16 BF16 FMA = **8,192 FLOPs / 1 instr** (latency ~16 cycle, throughput 1 cycle)
  - `qk_product` GEMM 영역의 dimension: M=NUM_Q_HEADS=8 (tile 16 의 절반 영역, padding 영역 필요), N=BLOCK_SIZE=16 (tile fit), K=HEAD_DIM=128 (8× iter 영역)
- **layout 변환**: NEO 의 `[..., NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]` → AMX tile expected layout (row-major BF16)
- **build flag**: `csrc/cpu/pacpu/CMakeLists.txt` 영역 에 `-mamx-tile -mamx-bf16 -mavx512bf16` 추가
- **runtime gate**: `__has_amx()` cpuid + fallback (Phase 5.2 의 AVX-512 BF16 영역)
- **영향 영역**: `qk_product` kernel 영역 의 inner GEMM 영역 — AVX-512 BF16 (Phase 5.2) 영역 보다 영역 의 추가 가속 영역 (1 instr 의 FLOPs 영역 영역 ↑)

### Gate
- micro-benchmark: AMX qk_product kernel 영역 의 단독 GFLOPs/s 영역
- 분포 유사성: per-token logprob max abs diff < 1e-3
- chain firing 99% 영역 의 PROFILE `cdec_wait_avg` 차이 측정 영역

### 위험
- dev (i9-12900KF) 의 AMX hw 미지원 — prod 만 직접 검증 영역
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

- **모듈/함수**: 본 plan workload (TP=8) 의 inter-GPU KV migration 영역 — 현재 NCCL all_reduce 영역만 NVLink 활용 영역
- **작업**: `cudaMemcpyPeerAsync` 도입 영역 (해당 영역의 inter-GPU KV copy 영역의 host 경유 영역 우회)
- **영향 영역**: TP=8 영역 = NCCL 이미 NVLink 활용 영역 → **본 workload 영역 의 영향 영역 작음**. multi-node disaggregated prefill 영역 도입 시 inter-node KV migration 영역 의 영향 큼
- **측정 방법**: bench (`cudaMemcpyPeerAsync` vs host 경유 영역) + flamegraph 영역

### Phase 6.5.2 — PTX inline kernel (gdec attention hot path)

- **모듈/함수**:
  - flash-attention kernel 영역 — `vllm/v1/attention/backends/flash_attn.py` 영역 의 dispatched kernel 영역
  - 또는 `vllm/model_executor/layers/attention/attention.py:534` `forward` 영역의 GPU dispatch 영역
- **작업**:
  1. `cuobjdump --dump-sass` 영역으로 현재 PTX 영역 disassembly
  2. inline PTX 영역 도입 (`asm volatile ("..."`) — CUDA C 안의 PTX directive
  3. 대상 영역: BF16 matmul (Tensor Core `mma.sync.aligned.m16n8k16` 영역), 또는 softmax 의 `exp.approx.f32` 영역
  4. CUTLASS template (`csrc/quantization/w8a8/cutlass`) 활용 영역
- **영향 영역**: gdec sub-batch[0] forward 영역 의 inner attention kernel 영역 — 본 환경 영역의 gpu_avg 0.08 ms / layer 영역 의 dominant 부분 영역
- **측정 방법**: kernel micro-benchmark + Nsight Compute 의 SM occupancy / roofline 영역

### Phase 6.5.3 — NVSHMEM symmetric memory (multi-GPU collective)

- **모듈/함수**: `vllm/distributed/parallel_state.py:519` `_all_reduce_out_place` 영역
- **작업**: small-message path 영역의 `tensor_model_parallel_all_reduce` 영역 의 NVSHMEM put/get 영역 으로 대체
- **영향 영역**: NCCL all_reduce 영역 — 본 환경 영역 의 flamegraph fact: 2.86% (BM20) 영역
- **측정 방법**: NCCL vs NVSHMEM all_reduce latency micro-benchmark

### Phase 6.5.4 — gdrcopy 활용 (CPU ↔ GPU low-latency direct mapping)

- **모듈/함수**: `vllm/v1/worker/gpu_model_runner.py:6968` `_neo_swap_in_one_req` 영역의 H2D copy 영역 (BM09)
- **작업**: `gdr_map()` + `gdr_get_info()` 영역으로 GPU memory ↔ user-space direct mapping. CPU memcpy 영역으로 GPU memory 직접 write (cudaMemcpy 우회 영역)
- **영향 영역**: swap-in path 영역 의 sync H2D 영역 — Phase 1 측정 영역의 4.3 GB/s 영역 vs PCIe Gen5 이론 영역의 영역
- **측정 방법**: gdrcopy `copybw` bench (KB-MB 영역 sync transfer 영역)
- **사전 영역**: `nvidia_peermem` 또는 `gdrdrv` kernel module 영역 의 host modprobe 필요 영역 (container modprobe 불가)

### Phase 6.5 Gate

- NVLink P2P bench (이미 동작 확인) → wall 측정
- PTX inline 의 정확도 검증 (logprob diff)
- NVSHMEM all_reduce 의 NCCL 대비 latency 측정
- gdrcopy 의 kernel module 영역 의존 — 호스트 운영 영역 검토

---

## Phase 8 — 알고리즘 영역 quantum jump (★ 본 plan 의 새 영역 — 단순 기능 영역 외)

> 2026-05-15 KST 추가. 사용자 명시 영역: "이런 알고리즘 영역을 통해서 성능 영역 개선" + "성능 quantum jump 영역 의 알고리즘 영역" + 지엽 노력 영역 (Phase 3-6.5) 과 별도 영역 진행 영역.

### 참고 자료 영역

- `feat/ide006-neo-asymmetric` branch 의 `shadow_assists/features/IDE_006/` 영역 (NEO 4 차 재정의 영역, PLN_001 / NEO_redesign / TSK_001 / TSK_009 / TSK_012 / 등)
- arXiv 2411.01142 (NEO MLSys 2025)
- arXiv 2501.01005 (online softmax merge 영역 표준 출처)
- vllm 의 `merge_attn_states` 영역 docstring

### Phase 8 영역 의 의미 영역

**지엽 영역 (Phase 3-6.5)** = 본 환경 의 NEO 영역 baseline 영역 의 hw 가속 영역 (cdec_wait ↓, OMP overhead ↓, hugepage 등) — **incremental 영역**.
**Quantum jump 영역 (Phase 8)** = 알고리즘 영역 차원 영역 의 변경 영역 — **NEO 의 메커니즘 자체 영역의 발전 영역**.

### Phase 8.1 — Speculative cdec dispatch (deadline-aware multi-attempt)

- **모듈/함수**:
  - `vllm/model_executor/layers/attention/attention.py:1014` `cdec_future.submit` (`torch.ops.pacpu.paged_attention_cpu` 영역의 ThreadPoolExecutor submit)
  - `vllm/v1/worker/sub_batch_executor.py:248` `forward_double` 영역의 cdec wait 영역
- **알고리즘**:
  - 현재: 1 cdec attempt / req / deadline. cdec_executor max_workers=2 cap
  - 신규: N 개 cdec speculation 영역 dispatch (서로 다른 layout / partition / batch ordering 영역) → 가장 빠른 영역 채택, 나머지 폐기
- **영향 영역**:
  - 위치: cdec sub-batch[1] wall 영역 (PROFILE 영역의 cdec_wait_avg 영역)
  - 가설: deadline miss rate 영역 ↓ → fallback 영역 ↓ → cdec_wall 영역 ↓ → wall = max(gdec, cdec) 영역의 balance 영역 ↑
- **위험**: CPU 영역 cdec_executor 영역의 N× 영역 work 영역 (max_workers cap 영역 정합 영역). SUB_023 영역의 +4 worker -52% 영역 fact 영역 회피 영역 필요

### Phase 8.2 — Hierarchical KV tier (GPU + CPU DRAM + NVMe + RDMA)

- **모듈/함수**:
  - `vllm/v1/core/sched/neo_cpu_kv_buffer.py` (CPU KV pool 영역) — 현 2-tier (GPU HBM + CPU DRAM)
  - 신규: `cuFile` (GDS, GPUDirect Storage 영역) → NVMe tier 영역 추가
  - 신규: `mlx5_*` (ConnectX-7) → RDMA peer node tier 영역 추가 (multi-node 영역)
- **알고리즘**: LRU + access pattern 영역 prediction 영역 으로 tier promote/demote 영역
- **영향 영역**:
  - 위치: KV pool 영역 capacity 영역 — 현 v1.6 best 영역의 NeoCpuKvBuffer 영역의 CPU_RESIDENT_REQS=128 영역 cap
  - 가설: capacity ↑ → 큰 batch / 긴 seq 영역 영역 도달 (paper sweet spot 영역의 workload 영역 정합 영역)
- **위험**: NVMe BW 영역 ceiling (DRAM 영역의 1/10). RDMA 영역 single-node 의미 작음

### Phase 8.3 — Cross-request KV prefix sharing (RadixAttention 영역 의 NEO 영역 적용)

- **모듈/함수**:
  - vllm 영역의 prefix caching: `vllm/v1/core/kv_cache_manager.py` 영역 의 radix tree
  - 신규: `vllm/v1/core/sched/neo_cpu_kv_buffer.py` 영역의 cpu_resident_reqs 영역의 prefix sharing 영역 적용
  - `vllm/v1/worker/gpu_model_runner.py:6968` `_neo_swap_in_one_req` 영역의 prefix-aware copy 영역
- **알고리즘**: 동일 prefix 영역의 req 영역의 cdec attention 영역 한 번만 계산 + share 영역
- **영향 영역**:
  - 위치: cdec sub-batch[1] 영역의 work 영역 (b1_avg × cdec_wait 영역)
  - 가설: 동일 prefix 영역의 req N 영역 → cdec work 영역 영역의 1/N 영역 (이론)
  - workload 의존 영역 (chat 영역의 system prompt 영역 공유 영역 = high prefix overlap 영역)
- **위험**: NEO 의 request 단위 exclusive ownership 영역 과 정합 영역 — 같은 prefix 의 req 영역이 GPU/CPU 영역 영역 다르게 영역의 fact 영역 정합 영역

### Phase 8.4 — Adaptive sub-batch sizing (dynamic partition)

- **모듈/함수**:
  - `vllm/v1/worker/sub_batch_executor.py:forward_double` 영역의 b0 (gdec) / b1 (cdec) partition 영역
  - 현재: NEO scheduler 영역 (`neo_scheduler_adapter.py:768`) 영역의 static partition 영역 (mirror set + chain firing 영역의 결과 영역)
  - 신규: runtime 영역의 gdec_wall / cdec_wait 영역 fact 영역 기반 영역 dynamic partition
- **알고리즘**: PID controller 또는 EMA 영역으로 b0/b1 ratio 영역 dynamic 영역 dial 영역. 목표 영역 = `gdec_wall ≈ cdec_wall` 영역 balance 영역
- **영향 영역**:
  - 위치: step wall = max(gdec_wall, cdec_wall) 영역
  - 가설: balance ↑ → wall 영역의 idle (max 영역의 smaller side) 영역 ↓ → throughput ↑
- **위험**: dynamic partition 영역의 thrashing 영역 (mirror set 영역의 잦은 promote/demote 영역)

### Phase 8.5 — Online softmax merge 영역 의 Tensor Core 가속

- **모듈/함수**:
  - `vllm/v1/attention/ops/merge_attn_states.py` 영역 (LSE merge 영역)
  - 또는 attention backend 영역의 partial output + LSE 영역의 합산 영역
- **알고리즘**: Hopper TMA + Tensor Core 영역 fused kernel 영역 (sm_90 specific 영역)
- **영향 영역**:
  - 위치: GPU attention 영역의 merge 단계 영역 — 본 plan workload 영역의 NEO = exclusive ownership 영역 (merge 영역 영역 없음). **본 plan 의 NEO baseline 영역의 영향 영역 작음**.
  - 본 plan 영역의 hot/cold split 영역 (IDE_006 3차 정의 영역의 LSE merge 영역) 영역 진입 시 영향 큼.
- **위험**: 본 plan 영역의 NEO 4차 정의 영역의 exclusive ownership 영역 정합 영역의 의미 영역 작음

### Phase 8.6 — PD disaggregation 영역 의 NEO 영역 결합

- **모듈/함수**:
  - vllm v1 영역의 PD (prefill-decode) disaggregation 영역 (`vllm/distributed/kv_transfer` 영역)
  - 신규: NEO 의 CPU sub-batch 영역 + PD disaggregation 영역의 prefill node 영역 결합 영역
- **알고리즘**: prefill node GPU + decode node GPU + decode node CPU (cdec) 영역의 3-tier pipeline
- **영향 영역**:
  - 위치: multi-node 영역의 throughput 영역
  - **본 plan 영역의 single-node TP=8 영역의 영향 영역 작음** — multi-node 영역 진입 시 영향 큼
- **위험**: multi-node 진입 영역 cost (RDMA + NIC 영역 정합 영역)

### Phase 8.7 — Approximate / sparse attention (NEO 영역 결합)

- **모듈/함수**:
  - 현재 NEO attention 영역: `csrc/cpu/pacpu/pacpu.ispc:142-160` `attn_one_seq` 영역의 full attention 영역 (exact)
  - 신규: sparse attention (StreamingLLM `window+sink`, H2O `accumulated attention scores`, scissorhands 영역) 영역 적용
  - cdec 영역 에서 sparse selection 영역의 attention 영역만 계산 영역
- **알고리즘**: top-k KV selection (full KV 영역의 일부만 영역 attention 영역 영역)
- **영향 영역**:
  - 위치: `cdec_wait_avg` 영역 (full 2.55ms → sparse 영역 의 fraction 영역)
  - 가설: long-context 영역의 attention work 영역 ↓ → CPU cdec 영역의 효율 영역 ↑
- **위험**: 정확도 영역 (분포 유사성 영역) 영역 — quality 영역 trade-off 영역

### Phase 8.8 — Predictive scheduling (cdec fire 예측)

- **모듈/함수**:
  - 현재 NEO scheduler: `vllm/v1/core/sched/neo_scheduler_adapter.py:768` `schedule` 영역 (reactive 영역의 step 별 fork attempt 영역)
  - 신규: req 영역의 KV pattern prediction 영역 으로 cdec dispatch 영역의 사전 schedule 영역
- **알고리즘**: simple heuristic (req 영역의 token 수 + KV depth 영역 의 sliding window 영역) 또는 ML-based prediction
- **영향 영역**:
  - 위치: FORK STAT 영역의 reject_no_subs 영역 (현 v1.6 best 영역 의 34,824 영역 / total 39,600 영역 = 87.9% reject 영역) → predictive 영역으로 reject 영역 ↓
  - chain firing 영역의 dynamic dial 영역 영역
- **위험**: prediction 영역 의 false positive 영역 (불필요 영역의 cdec dispatch 영역 → CPU 영역의 work 영역 낭비 영역)

### Phase 8 Gate

- 각 sub-phase 영역 의 효과 영역 fact 영역 측정 — **모듈/함수 영역의 영향 영역 정량 영역**:
  - 8.1: cdec_wait_max / fallback count 영역
  - 8.2: KV pool capacity (CPU_RESIDENT_REQS) 영역 확장 + miss rate 영역
  - 8.3: prefix hit rate × cdec work 영역의 감소
  - 8.4: gdec_wall / cdec_wall balance 영역의 std 영역
  - 8.5: merge kernel 영역 의 cycles 영역
  - 8.6: inter-node latency 영역 + throughput 영역
  - 8.7: sparse cdec_wait_avg × accuracy 영역 (logprob diff 영역)
  - 8.8: FORK STAT 영역의 reject_no_subs 영역 ↓ + chain firing layer fire ↑
- 분포 유사성 게이트 (logprob diff < 1e-3) 통과
- 본 plan 영역의 sub-phase 영역의 효과 = **모듈/함수 영역의 fact 영역 측정 영역만** — 가속 율 영역의 추정 영역 X

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

## 합계 — 모듈/함수/로직 영역 영향 표 (정정 v11 — % 추정 영역 제거)

각 Phase 영역의 효과 영역 = **% 영역 추정 영역 X**. 모듈/함수/로직 영역의 영향 영역 fact 영역만 기재 — 측정 후 영역 확정 영역.

### Track A — 지엽 영역 (모듈 단위 가속)

| Phase | 영향 영역 (모듈 / 함수 / 로직) | 측정 metric 영역 |
|---|---|---|
| 1 | 측정 완료 (PROFILE log + perf uncore_imc + smaps) | — |
| 2 | Option C/L/M2 ON 환경 영역 진입 — `eval/run_*.sh` 영역의 env 영역 변경. b1_avg ↑ → cdec_wall ↑ | FORK STAT active/total + PROFILE b1_avg, cdec_wait_avg |
| 3.1 | `csrc/cpu/pacpu/core.h:296,314,333` OMP team launch + barrier | PROFILE cdec_wait_avg + perf context-switches |
| 3.2 | env `KMP_BLOCKTIME=0` — OMP thread wake | PROFILE cdec_wait_max (tail) |
| 3.3 | `vllm/v1/worker/sub_batch_executor.py` 영역의 CUDA stream priority | PROFILE gpu_max (tail) |
| 3.4 | `_neo_init_swap_staging` (gpu_model_runner.py:6390) NUMA 영역 (Phase 1 영역 이미 정합) | numastat per-worker |
| 4 | `_neo_init_swap_staging` 영역의 pinned alloc → `mmap(MAP_HUGETLB \| MAP_HUGE_1GB)` + `cudaHostRegister` | `/proc/<pid>/smaps` Private_Hugetlb + 1GB pool free count |
| 5.1 | `pacpu.ispc:109-140` softmax 영역의 exp → `_mm512_fast_exp_ps` | softmax kernel micro-bench (ns) + cdec_wait_avg |
| 5.2 | `pacpu.ispc:5-69` qk_product + `:71-107` av_product 영역의 inner FMA → `_mm512_dpbf16_ps` (BF16) | qk/av kernel ns + cdec_wait_avg + logprob diff |
| 6 | `csrc/cpu/pacpu/pacpu_amx.cpp` 신규 — qk_product AMX TileGemm224 (`tdpbf16ps`) | qk kernel GFLOPs/s + cdec_wait_avg + logprob diff |
| 6.5.1 | `cudaMemcpyPeerAsync` 영역의 inter-GPU KV migration 영역 (TP=8 영역의 영향 작음) | bench |
| 6.5.2 | flash-attention kernel 영역의 PTX inline (mma.sync, exp.approx.f32) | Nsight Compute SM occupancy + gpu_avg |
| 6.5.3 | `parallel_state.py:519` `_all_reduce_out_place` → NVSHMEM put/get | NCCL vs NVSHMEM all_reduce ns |
| 6.5.4 | `_neo_swap_in_one_req` (gpu_model_runner.py:6968) → gdrcopy direct mapping | gdrcopy copybw bench |

### Track B — 알고리즘 영역 quantum jump (로직 변경)

| Phase | 영향 영역 (모듈 / 함수 / 로직) | 측정 metric 영역 |
|---|---|---|
| 8.1 | `attention.py:1014` cdec_future.submit → N speculation + `sub_batch_executor.py:248` 의 fastest 채택 로직 영역 | cdec_wait_max + fallback count |
| 8.2 | `neo_cpu_kv_buffer.py` 영역에 cuFile (NVMe) / RDMA peer tier 영역 추가. promote/demote 영역의 LRU 로직 | KV pool capacity 확장 + miss rate |
| 8.3 | `kv_cache_manager.py` radix tree 영역의 `neo_cpu_kv_buffer.py` 영역 적용 — `_neo_swap_in_one_req` 영역의 prefix-aware copy | prefix hit rate × cdec work 영역의 감소 |
| 8.4 | `neo_scheduler_adapter.py:768` schedule 영역의 b0/b1 partition → runtime balance 영역의 PID/EMA 로직 | gdec_wall vs cdec_wall std + chain firing layer fire |
| 8.5 | `merge_attn_states.py` Tensor Core fused kernel (Hopper SM90 specific) | merge kernel cycles. 본 plan NEO baseline 영역의 영향 작음 (exclusive ownership 영역) |
| 8.6 | `distributed/kv_transfer` 영역의 PD disaggregation + NEO CPU sub-batch 결합. multi-node 영역 진입 | inter-node latency + throughput |
| 8.7 | `pacpu.ispc:142-160` attn_one_seq 영역의 sparse selection (StreamingLLM / H2O / scissorhands) | sparse cdec_wait_avg × logprob diff |
| 8.8 | `neo_scheduler_adapter.py:768` schedule 영역의 predictive cdec dispatch | FORK STAT reject_no_subs ↓ + chain firing layer fire ↑ |

### Phase 7 — 통합

| Phase | 영향 영역 | 측정 metric |
|---|---|---|
| 7 | Track A + Track B 영역의 통합 영역 + workload sweep | output_tps + 22 strict + logprob diff |

→ 각 Phase 영역의 효과 영역 = **모듈/함수 영역의 fact 영역의 측정 후 영역 확정 영역**. % 추정 영역 영역 = 신뢰성 영역 작음 영역 의 사실 영역 영역.

### 핵심 finding 재정리

- **Phase 2 (chain firing dial-up 4% → 99%)** = CLAUDE.md Objective ("CPU 활용률 극대화") 정합 영역. Option C/L/M2 ON 환경 영역 진입 → chain firing 99% (CPU 활용률 99%) + throughput 627 tps collapse 영역. Phase 3-6.5 영역 의 가속 영역 = collapse 회복 영역의 도구 영역.
- **Phase 4 (1GB hugepage)** 는 측정 영역 효과 1-3% — limited 그러나 추가 영역. cost 작음 (코드 영역 + setup)
- **Phase 3 hw quick wins** 의 효과 영역 작음 (이미 정합) — 그러나 cost 작음 = 진행 가치 있음

---

## 진행 순서 + branch 전략

- **base branch**: `feat/neo-amx-apply` (현재)
- **Phase 별 sub-branch**: `feat/neo-amx-apply-phase{2..7}` 또는 통합 branch 안 단계 commit
- Phase 진입 전 사용자 승인 영역
- Phase 결과 PR / commit gate 영역 후 다음 Phase 진입

### 본 plan 의 hint 영역

본 plan 의 핵심: **Track A (모듈 단위 가속) + Track B (알고리즘 quantum jump) 둘 다 진행**.

- **Track A (Phase 2-6.5)** = 모듈/함수 단위 영역의 가속 (Option C/L/M2 env 영역 진입 + OMP team 영역 + AVX-512 BF16 + AMX TileGemm + PTX + NVSHMEM + gdrcopy 등). **각 모듈/함수 영역의 영향 영역 = 측정 후 확정 영역 (% 추정 영역 X)**
- **Track B (Phase 8.1-8.8)** = 알고리즘 차원 영역의 로직 변경 (speculative cdec / Hierarchical KV tier / prefix sharing / adaptive batch / PD disaggregation / sparse attention 등). **로직 영역의 fact 영역 = 측정 후 영역 확정**
- 두 Track 영역 의 진행 = parallel 또는 sequential 영역 (모듈 영역의 cost vs 로직 영역의 cost 영역 결정 영역)
- 본 plan 영역의 효과 영역 = **모듈/함수 영역의 metric (cdec_wait_avg, gpu_avg, b1_avg, FORK reject, kernel ns, PROFILE 영역 등) 영역의 측정 영역만**. 가속 율 영역의 추정 영역 X

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

---

## 학습 리소스 영역 (각 기술별 공부 자료)

본 plan 의 각 phase 의 기술 영역 의 공부 가능 영역. 키워드 + 개요 + 링크. 진입 전 참고 영역.

### CPU 측 가속 영역 (AMX / AVX-512 / OMP / NUMA / Hugepage)

| 기술 | 키워드 | 개요 | 링크 |
|---|---|---|---|
| **Intel AMX** | tile register, tdpbf16ps, AMX-TMUL | BF16 / INT8 8 tile-register 의 16×16 matrix multiply. SPR / GNR 영역의 hw native. tile config 영역의 palette/colsb/rows | https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html |
| **AMX intrinsics** | `_tile_loadd`, `_tile_dpbf16ps`, `_tile_stored` | gcc/clang 영역의 AMX intrinsic — `<immintrin.h>` 영역 | https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html (검색: AMX) |
| **AVX-512 BF16** | vdpbf16ps, _mm512_dpbf16_ps | BF16 dot product 1 instruction 영역 (Tiger Lake+, SPR native) | https://en.wikipedia.org/wiki/AVX-512 + Intel Intrinsics Guide |
| **AVX-512 fast_exp** | polynomial approx, _mm512_fast_exp_ps | softmax 영역의 scalar exp 영역 → 5-degree polynomial 영역 vectorize | vllm `csrc/cpu/cpu_arch_macros.h` 영역 내부 reference |
| **AVX-512 NT-store** | _mm512_stream_si512, MOVNTDQ | cache-bypass store 영역. swap copy 영역의 cache pollution 회피 | Intel Software Developer Manual Vol. 2 (MOVNTI, MOVNTPS) |
| **ISPC** | SPMD, varying, foreach | Intel SPMD Program Compiler — auto-vectorize 영역. NEO pacpu 의 backend | https://ispc.github.io/ + github https://github.com/ispc/ispc |
| **OpenMP team / persistent** | omp_set_dynamic, KMP_BLOCKTIME | OMP team 의 thread launch overhead 영역 / blocktime 영역의 spin-vs-sleep tunable | https://www.openmp.org/spec-html/5.2/openmpsu131.html + Intel `KMP_*` env vars |
| **NUMA local alloc** | numa_alloc_onnode, mbind, set_mempolicy | NUMA-aware alloc 영역. linux libnuma 영역 | https://man7.org/linux/man-pages/man3/numa.3.html |
| **1GB Hugepage** | mmap MAP_HUGETLB, MAP_HUGE_1GB, hugetlbfs | 4KB → 2MB → 1GB 영역의 TLB miss 영역 절감. hugetlbfs mount + boot param | https://www.kernel.org/doc/Documentation/vm/hugetlbpage.txt |
| **THP (Transparent Hugepage)** | madvise(MADV_HUGEPAGE), transparent_hugepage/enabled | anonymous mmap 영역의 자동 promote 영역 | https://www.kernel.org/doc/Documentation/vm/transhuge.txt |
| **libhugetlbfs** | HUGETLB_MORECORE, hugectl, hugeadm | glibc morecore intercept 영역 (sbrk path 만 — Python mmap path intercept 안 됨) | https://github.com/libhugetlbfs/libhugetlbfs |
| **Intel oneDNN** | onednn_mm, ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX | Intel oneDNN library — AMX 영역의 GEMM backend 자동 dispatch | https://github.com/oneapi-src/oneDNN |

### GPU 측 가속 영역 (NVLink / PTX / CUDA / NVSHMEM / GDR)

| 기술 | 키워드 | 개요 | 링크 |
|---|---|---|---|
| **NVLink Gen4** | NV18, 18-link, 900 GB/s | H100 ↔ H100 영역의 inter-GPU 영역 fabric. cudaMemcpyPeerAsync 통한 direct DMA | https://www.nvidia.com/en-us/data-center/h100/ + https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#peer-to-peer-memory-copy |
| **PTX (assembly)** | mma.sync, ld.global.nc, exp.approx.f32 | NVIDIA GPU의 virtual ISA. CUDA C의 inline asm 영역 또는 .ptx 영역 작성 | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html |
| **CUTLASS** | TileGemm, Epilogue, Sm90 | NVIDIA CUDA Templates for Linear Algebra — Tensor Core 영역 의 GEMM building block | https://github.com/NVIDIA/cutlass |
| **FlashAttention 3** | Hopper, TMA, async pipeline | FA3 영역 의 Hopper architecture native attention kernel | https://github.com/Dao-AILab/flash-attention + arXiv 2407.08608 |
| **TMA (Tensor Memory Accelerator)** | cuda::memcpy_async, Hopper SM | H100 영역 의 hw async memcpy unit. shared memory load 영역 | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tma-asynchronous-data-copies |
| **NVSHMEM** | nvshmem_put, nvshmem_get, symmetric_memory | same-node multi-GPU symmetric memory API. low-latency small-message all_reduce | https://docs.nvidia.com/nvshmem/api/index.html + https://developer.nvidia.com/nvshmem |
| **gdrcopy (GDR)** | gdr_pin_buffer, gdr_map, BAR1 | CPU userspace ↔ GPU memory direct mapping 영역. KB-MB 영역 sync transfer 영역 의 latency 단축 | https://github.com/NVIDIA/gdrcopy + https://docs.nvidia.com/cuda/gpudirect-rdma/ |
| **GPUDirect RDMA** | nvidia_peermem, ConnectX, RDMA verbs | NIC ↔ GPU 영역 direct DMA (host bypass). multi-node 영역 effect 큼 | https://docs.nvidia.com/cuda/gpudirect-rdma/ |
| **GPUDirect Storage (GDS)** | cuFile, nvidia-fs | NVMe ↔ GPU 영역 direct copy (host bypass) | https://docs.nvidia.com/gpudirect-storage/index.html |
| **CUDA Streams + priority** | cudaStreamCreateWithPriority | per-stream priority 영역. critical path 영역 우선 | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-priorities |
| **CUDA Graphs** | cudaGraphLaunch, capture | kernel launch overhead 영역 절감. enforce_eager=False 영역의 vllm 자동 활용 | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs |
| **FP8 Transformer Engine** | nvidia-transformer-engine, FP8-E4M3 / E5M2 | Hopper 영역 의 FP8 matmul. KV cache + activation 영역의 메모리 영역 절반 | https://github.com/NVIDIA/TransformerEngine |
| **NCCL** | all_reduce, NVLink path, RING/TREE | inter-GPU collective. single-node NVLink 자동 활용 | https://github.com/NVIDIA/nccl + https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html |

### 측정 + 분석 도구 영역

| 기술 | 키워드 | 개요 | 링크 |
|---|---|---|---|
| **perf (Linux)** | uncore_imc, cas_count, dTLB-load-misses | Linux performance counter. CPU 영역 의 PMU + uncore IMC event | https://perfwiki.github.io/main/ + https://www.brendangregg.com/perf.html |
| **Intel PCM** | pcm-memory, pcm-pcie, pcm-numa | Intel Processor Counter Monitor. DDR / PCIe / NUMA BW + cache event | https://github.com/intel/pcm |
| **py-spy** | --native unwind, sampling | Python sampling profiler. C 영역 backtrace 까지 capture | https://github.com/benfred/py-spy |
| **flamegraph** | brendan gregg, --collapsed | stack sample 영역의 visualization | https://github.com/brendangregg/FlameGraph + https://www.brendangregg.com/flamegraphs.html |
| **Intel SDE** | sde64 -spr-sp, -gnr-sp | Intel Software Development Emulator. AMX 영역의 cross-platform 영역 시뮬레이션 | https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html |
| **NVIDIA Nsight Systems** | nsys profile, CUDA timeline | GPU/CPU 영역 의 system-wide profiler | https://docs.nvidia.com/nsight-systems/UserGuide/index.html |
| **NVIDIA Nsight Compute** | ncu, SM occupancy, roofline | CUDA kernel 영역의 detailed profiler | https://docs.nvidia.com/nsight-compute/NsightCompute/index.html |
| **gdb / cuda-gdb** | bt, info threads | crash 영역 의 stack trace + CUDA kernel 영역 debug | https://docs.nvidia.com/cuda/cuda-gdb/index.html |

### NEO 영역 (paper + repo)

| 기술 | 키워드 | 개요 | 링크 |
|---|---|---|---|
| **NEO paper** | MLSys 2025, asymmetric pipelining | NEO: Saving GPU Memory Crisis with CPU Offloading | https://arxiv.org/abs/2411.01142 + https://yangzhou1997.github.io/paper/neo_mlsys25.pdf |
| **NEO repo** | swiftllm, pacpu, ISPC kernel | NEO MLSys25 public 구현 영역 | https://github.com/NEO-MLSys25/NEO |
| **NEO 동작 영역** | chain firing, cdec dispatch, mirror set | request 단위 exclusive KV ownership + sub-batch pipeline | (본 plan 의 `analysis/B_paper_section_notes.md` + `NEO_code_deepdive.md`) |

### vllm 영역 (CPU backend + 통합)

| 기술 | 키워드 | 개요 | 링크 |
|---|---|---|---|
| **vllm PagedAttention** | block_size, KV cache, virtual blocks | block-based KV cache 영역. fragmentation 회피 | https://arxiv.org/abs/2309.06180 + https://github.com/vllm-project/vllm |
| **vllm CPU backend** | cpu_attn_amx, cpu_attn_vec, ISA dispatch | vllm 자체 CPU attention 영역 (AMX/AVX/NEON) | (본 repo `csrc/cpu/` 영역) |
| **vllm CUDA Graph + async scheduling** | enforce_eager, async_scheduling | dispatch overhead 절감 영역. v1 영역 default | https://docs.vllm.ai/en/latest/serving/usage_stats.html |
| **vllm v1 architecture** | scheduler, model runner, KV manager | vllm v1 영역의 core architecture | https://blog.vllm.ai/2024/09/05/perf-update.html + https://docs.vllm.ai/ |

### 본 plan 영역의 사전 분석 자료 (TSK_019 영역 내부)

| 자료 | 위치 |
|---|---|
| Phase A — NEO upstream 감사 | `analysis/A_neo_upstream_audit.md` |
| Phase A — kernel signature map | `analysis/A_kernel_signature_map.md` |
| Phase B — paper section notes | `analysis/B_paper_section_notes.md` |
| Phase B — paper vs 측정 | `analysis/B_paper_vs_our_measure.md` |
| Phase C — vllm AMX/AVX inventory | `analysis/C_existing_paths_inventory.md` |
| Phase C — pacpu vs cpu_attn gap | `analysis/C_pacpu_vs_cpu_attn_amx_gap.md` |
| Phase D — flamegraph 분석 | `analysis/D_bottleneck_table.md` + `D_roofline_notes.md` + `D_candidate_long_list.md` |
| Phase E — bottleneck map 최종 | `analysis/E_bottleneck_map.md` |
| Phase E — AMX/AVX 적용 가능성 | `analysis/E_amx_avx_applicability.md` |
| Phase E — 측정 미달 영역 | `analysis/E_open_questions.md` (OQ01-OQ18, Phase 1 측정 완료 영역 포함) |
| Phase F — HW 가속 후보 inventory | `analysis/F_hardware_acceleration_candidates.md` |
| NEO 사전 분석 doc | `shadow_assists/features/IDE_006/NEO_code_deepdive.md`, `NEO_redesign.md`, `Objective-for-NEO-porting.md` |
| 본 plan 의 Best Configuration | `shadow_assists/features/IDE_006/TSK_019/README.md` |

→ 본 plan 의 각 phase 진입 전 위 자료 영역 참고. 외부 학습 영역 + 본 repo 내부 영역 둘 다 동반.
