# M — SUB_015-Phase 3 HPC 측면 최적화 분석

> 작성 일자: 2026-05-20 KST.
> 대상: NEO (MLSys 2025) pacpu CPU paged attention kernel.
> 측정 환경: Intel Sapphire Rapids (Xeon Platinum 8480+ × 2) + H100 80GB × 8.
> 현재 상태: Phase 3 의 모든 AMX variant 가 S1-S9 baseline (2,238.6 tps) 초과 불가. 본 문서 = HPC 관점 root cause 재정리 + 외부 1차 출처 backing + 다음 lever roadmap.
> 자매 분석: [`reference/I_amx_proper_design.md`](reference/I_amx_proper_design.md), [`reference/J_sub015_root_cause_analysis.md`](reference/J_sub015_root_cause_analysis.md), [`reference/K_sub015_improvement_roadmap.md`](reference/K_sub015_improvement_roadmap.md), [`reference/L_sub015_evidence_based_priority.md`](reference/L_sub015_evidence_based_priority.md).

---

## 0. Executive Summary

| 항목 | 현재 상태 | 외부 문헌 backing | 정합 |
|---|---|---|:-:|
| **AMX qk_product 3-run avg vs S1-S9** | **-4.3%** (Phase 3 A) | 50% partial-M (M=8/16) 의 effective FLOPS 정확히 절반 (`TDPBF16PS` 16-cycle throughput 은 tile shape 무관) | ✓ |
| **AMX Step 5 (B+A+vec K) 3-run avg** | **-2.35%** (cheap variant 다 적용) | libxsmm small-matmul cutoff `(MNK)^(1/3) ≤ 64`, NEO qk 의 `(8×16×128)^(1/3) ≈ 23` → small 영역 내부 | ✓ |
| **AMX setup overhead** | 650-900 cyc / block (cycle counting) | `LDTILECFG` high-latency, "함수 진입당 1회 amortize" 가이드 | ✓ |
| **libgomp barrier** | perf record 의 43.75% | EPCC barrier latency 1.3-2.6 μs / 2-4 thread = ms 단위 누적 | ✓ |
| **K cache FP16→BF16 cast** | per-block 200-400 cyc | `VCVTNE2PS2BF16` single-instruction, BW saving 대비 negligible | ✓ (단 cycle GAP) |

**핵심 진단**: SUB_015-Phase 3 의 회귀 root cause = (1) AMX tile underutilization (M=8 vs 16 = 50%), (2) setup overhead amortize 실패 (small matmul 영역), (3) 분산 차원 Amdahl 한계 (cdec_executor cap=2, async depth=0). HPC 영역의 단순 AMX dropin **은 정량적으로 net loss 가 예측 가능** — 외부 1차 출처 (OpenBLAS Discussion #5205, AWS Compute Blog, FlashDecoding++) 모두 같은 메커니즘으로 보고.

**HPC 관점 진정한 lever 영역** (priority 순):
1. **F3 (K cache BF16 host store)** — cast 영역 영구 제거 + storage 2× (BW 이득), 이론 +1-5%, 측정 미시도
2. **F4 (TP=8 → TP=4)** — M=8 → M=16 (AMX tile 100% occupancy, AVX-512 BF16 대비 4× 우위 회복), 이론 +5-10%, effort 2-4 주
3. **F5 (BLOCK_SIZE 16 → 32)** — N=16 → N=32 (per-block work 2×, setup amortize 비율 ↑), 이론 +3-7%, effort 2-4 주
4. **F1 (async cdec pipeline depth)** — 분산 차원 Amdahl 완화, cdec wall 50-75% 단축 시 +5-15%, 단 sequential KV dep 으로 infra 미구현 (P4 측정 -1.9%)
5. **F2 / F6** — 측정 결과 sweet spot 이미 도달 또는 atomic overhead dominant (회귀)

---

## 1. SUB_015 의 정의 변천 (frame history)

### 1.1 1차 frame — "AMX 로 4-7× speedup"

**source**: [`reference/H_dynamic_analysis.md:31-43`](reference/H_dynamic_analysis.md) — perf record 60s, 413K cycles sample:

| 영역 | 점유율 |
|---|---:|
| libpacpu (ISPC kernel) | 26.38% (qk 8.75% + softmax 9.73% + av 7.90%) |
| **libgomp (OMP runtime)** | **43.75%** ★ 최대 영역 (barrier spinning) |
| libtorch_cpu (swap path) | 10.24% |

원래 가설: `_tile_dpbf16ps` 의 1024 BF16 ops/cycle/core 가 ISPC AVX-512 의 128 BF16 ops/cycle 대비 8× peak → roofline AI=7.11 (L2-bound) 영역에서 실효 **3-5× speedup** 가능.

### 1.2 2차 frame — "Step 1~6 variant sweep"

**source**: [`reference/I_amx_proper_design.md`](reference/I_amx_proper_design.md) — Strategy A~H ranking + Step 1~6 build/measure plan. cheap variant 부터 cumulative apply.

| Step | 변경 (누적) | 1-run tps | vs S1-S9 |
|---|---|---:|---:|
| (base) | Phase 3 A dropin | 2,142.5 (3-run avg) | -4.3% |
| **1** | + B thread-Q-cache | 2,237.4 | -0.05% |
| **2** | + A K^T outer pre-pack | 2,275.6 | +1.7% |
| 3 | + G SW prefetch | 2,184.7 | -2.4% ✗ |
| 4 | + C' 2-block fused | 2,184.0 | -2.4% ✗ |
| **5** | + vec K conv (AVX-512) | 2,284.0 | +2.0% |
| 6 | + G+C' 통합 | 2,199.7 | -1.7% ✗ |

→ Step 5 가 1-run 에서 best 로 보였으나 **3-run avg = 2,186.1, -2.35%** (cold-start cache + thermal drift 의 round 단조 감소 -7% 흡수).

### 1.3 3차 frame (현재) — "분산 차원의 Amdahl 한계"

**source**: [`reference/J_sub015_root_cause_analysis.md:15-30`](reference/J_sub015_root_cause_analysis.md):

- cdec_executor max_workers cap: **2** (cap=4 시 -8% 회귀, SUB_030)
- cdec_wait: **8.75 ms/layer** (Phase 1 측정)
- async pipeline depth: **0** (80 layer sequential)
- cdec wall = step wall 의 **~70%** → Amdahl speedup ceiling = 1/(1-0.7) = **3.33×** *최대*
- HPC 이득 (+15% 이론) **<** 분산 overhead (-20% 실제) = **net loss 구조적 보장**

---

## 2. pacpu kernel 의 HPC 영역 정량 (소스 직접 분석)

### 2.1 ISPC qk_product — vectorization fact

**source**: [`csrc/cpu/pacpu/pacpu.ispc:5-69`](../../../../../csrc/cpu/pacpu/pacpu.ispc) (`qk_product`)

```c
foreach (l = 0 ... HEAD_DIM) {              // 128-lane gang (avx512spr-x16, 16 wide × 8 inst)
  for (uniform int g = 0; g < K_TILE_WIDTH; g++) {   // K_TILE_WIDTH=2
    itmd_t k_val = k[k_off + g * HEAD_DIM + l];
    for (uniform int h = 0; h < QH_PER_KVH; h++) {   // QH_PER_KVH=8 (TP=8)
      sum[h][g] += q[q_off + h * HEAD_DIM + l] * k_val;
    }
  }
}
```

**측정**: qk_product 점유 **8.75% of 6.86 T cycle = 600.3 G cycle ≈ 2.0 ms/layer** ([`reference/H_dynamic_analysis.md:107-112`](reference/H_dynamic_analysis.md))
**Arithmetic Intensity**: AI = 4.19 M FLOP / (4,608+2,048) byte = **7.11** ([`reference/H_dynamic_analysis.md:121`](reference/H_dynamic_analysis.md))
**Roofline ceiling**: L2 BW 50 GB/s/core × 7.11 = **355 GFLOP/s** ([`archive/D_roofline_notes.md:56`](archive/D_roofline_notes.md))
**실효**: 8-10 GFLOP/s/core (56 core × ≈ 500 GFLOP/s aggregate) → **L2 ceiling 대비 ~1.5%** 도달 — peak 미달

### 2.2 AMX qk kernel — tile occupancy

**source**: [`csrc/cpu/pacpu/amx_kernel.cpp:62-68`](../../../../../csrc/cpu/pacpu/amx_kernel.cpp) (`ensure_amx_init`)

```cpp
TileConfig cfg = {};
cfg.palette = 1;
for (int i = 0; i < 3; ++i) {
    cfg.rows[i] = 16;
    cfg.colsb[i] = 64;  // 64 byte = 32 BF16 pair
}
_tile_loadconfig(&cfg);
```

**occupancy 분해** ([`reference/I_amx_proper_design.md:13-20`](reference/I_amx_proper_design.md)):

| dimension | hardware peak | NEO 사용 | 사용률 |
|---|---:|---:|---:|
| M (rows of A) | 16 | **8** (NUM_Q_HEADS, TP=8) | **50%** |
| N (cols of B) | 16 (BLOCK_SIZE) | 16 | 100% |
| K (cols of A / rows of B) | 32 (BF16 pair) × 4 round = 128 | 128 | 100% |
| Tile pool | 8 tiles | 3 tiles used (A/B/C) | **37.5%** |
| **Net effective** | | | **50% × 37.5% = 19%** |

→ AMX 이론 peak 1024 BF16 FLO/cycle/core 의 **19% = 195 BF16 FLO/cycle/core** 실효. AVX-512 BF16 peak 128 대비 **여전히 1.52× 우위** *이론상*.

### 2.3 AMX cycle 분해 (per-block)

**source**: [`reference/I_amx_proper_design.md:34-46`](reference/I_amx_proper_design.md)

| Operation | Cycle | 빈도 | 총 (per block) |
|---|---:|---:|---:|
| Q FP16→BF16 (AVX-512) | 50-100 | 1 (Step 1 thread-local cache hoist) | 50-100 |
| **K^T pre-pack + FP16→BF16** (2048 elem × 4 round) | 200-400 | per block | **200-400** |
| `_tile_loadd` × 2 × 4 round | 240 | per block | 240 |
| **`_tile_dpbf16ps` × 4 round** (work) | 16 | per block | **64** |
| `_tile_zero` + `_tile_stored` | 60 | per block | 60 |
| C copy (16×16 → 8×16 valid extract) | 30 | per block | 30 |
| **AMX total / block** | | | **~650-900** |
| ISPC AVX-512 baseline / block | | | ~400-500 |

**setup / work ratio**: (650-900 - 64) / 64 ≈ **10-13×** — work 대비 setup overhead 10배 이상.

→ 외부 fact 정합: **`TDPBF16PS` 의 throughput 16 cycle 은 tile shape 무관** ([felixcloutier `TDPBF16PS`](https://www.felixcloutier.com/x86/tdpbf16ps), Intel Optimization Reference Manual #355308). 즉 M=8 의 경우 instruction time 동일 → effective FLOPS 정확히 (M/16) = 0.5 → 절반 손실 메커니즘 외부 1차 출처 confirm.

### 2.4 softmax — exp chain bottleneck

**source**: [`csrc/cpu/pacpu/pacpu.ispc:111-142`](../../../../../csrc/cpu/pacpu/pacpu.ispc)

```c
// Pass 1: max
foreach (h = 0 ... NUM_Q_HEADS) {
  for (uniform int i = 0; i < seq_len; i++) {
    a[i*NUM_Q_HEADS+h] *= softmax_scale;
    amb[h] = max(amb[h], a[i*NUM_Q_HEADS+h]);
  }
}
// Pass 2: exp + sum  ← exp chain dep
foreach (h = 0 ... NUM_Q_HEADS) {
  for (uniform int i = 0; i < seq_len; i++) {
    a[i*NUM_Q_HEADS+h] = exp(a[i*NUM_Q_HEADS+h] - amb[h]);
    asb[h] += a[i*NUM_Q_HEADS+h];  // sum dependency
  }
}
```

**점유**: softmax 9.73% (= qk 보다 큼). **exp 가 ILP 차단** (i 의 sum dependency).

**HPC 개선 여지** (분석 문서 미언급):
- `expf` polynomial 5-degree fast approx → 2-3× speedup (analysis 문서 안 측정)
- 2-pass merge (`max + exp + sum` 한 pass 로) — single-pass online softmax (FlashAttention 식 streaming): **softmax 영역 2× 가능**
- 외부 fact: **FlashAttention 의 online softmax** 가 CPU 영역에도 적용 가능 ([Princeton NLP FlashDecoding](https://princeton-nlp.github.io/flash-decoding/))

---

## 3. 외부 1차 출처 backing 정리

### 3.1 AMX 의 small-matmul 영역 회귀 — 외부 confirm

| 출처 | 핵심 fact |
|---|---|
| [OpenBLAS Discussion #5205](https://github.com/OpenMathLib/OpenBLAS/discussions/5205) | "small batch / 비-square / K 가 짧은 경우, AVX-512 BF16 경로가 AMX 보다 빠를 수 있음" — 사용자 보고로 OpenBLAS 가 break-even threshold 두는 이유 |
| [AWS Compute Blog — Accelerate CPU AI with Intel AMX](https://aws.amazon.com/blogs/compute/accelerate-cpu-based-ai-inference-workloads-using-intel-amx-on-amazon-ec2/) | "AMX 의 setup/data-movement overhead 가 작은 model/single-batch 의 matmul 양에 의해 amortize 되지 못하면 AMX 의 이득이 사라진다" (정성) |
| [libxsmm — readthedocs](https://libxsmm.readthedocs.io/en/latest/) | small matmul 정의 `(M*N*K)^(1/3) ≤ 64`. dispatch overhead negligible 한 영역 = **20×20 이상**. NEO qk 의 `(8*16*128)^(1/3) ≈ 22.6` → small 영역 |
| [Intel AMX brief](https://cdrdv2-public.intel.com/785250/Intel-AMXBrief-Final-3.17.pdf) | AMX = 1024 BF16 ops/cycle/core peak (M=N=16, K=32 full tile 가정) |
| [FlashDecoding++ MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/file/5321b1dabcd2be188d796c21b733e8c7-Paper-Conference.pdf) | small batch (예: batch=8) zero-padding 해서 64×64 GEMM 으로 키우는 결과 **>50% compute under-utilization**. NEO M=8 의 동일 패턴 |
| [Sequence-Aware Split Heuristic — arxiv 2604.00028](https://arxiv.org/pdf/2604.00028) | FA-3 의 default 가 short-decode 에서 SM under-utilization. 동일 메커니즘이 AMX tile partial-M 에 적용 |

**Key formula** (외부 출처 정합):
> Effective AMX FLOPS = `peak (1024) × (M/16) × (tiles_used / 8)` = NEO 의 경우 `1024 × 0.5 × 0.375 = 192` BF16 FLO/cycle/core

→ **이론 ceiling 자체가 AVX-512 BF16 peak (128) 의 1.5× 에 불과**, setup overhead 까지 더하면 net loss 일관 — 외부 다수 fact 로 backing.

### 3.2 AMX setup overhead — 외부 1차 출처

| 출처 | 핵심 fact |
|---|---|
| [felixcloutier `LDTILECFG`](https://www.felixcloutier.com/x86/ldtilecfg) | tile palette 변경 high-latency, "**함수 진입당 1회 amortize**" 가이드 |
| [Fixstars Tech Blog — Intel AMX Explained](https://blog.us.fixstars.com/intel-amx-advanced-matrix-extension-explained-introduction/) | "같은 configuration 을 최대한 reuse 하라" |
| [LLVM-dev — Intel AMX programming model](https://groups.google.com/g/llvm-dev/c/caHJWyUNWNk) | tile shape 변화가 필요한 모든 경우는 하나의 `ldtilecfg` 로 통합 |
| [felixcloutier `TILELOADDT1`](https://www.felixcloutier.com/x86/tileloaddt1) | throughput 23 cycle / latency 48 cycle (Intel Opt Ref Manual #355308 인용) |
| [Intel Opt Ref Manual #355308 v049](https://cdrdv2-public.intel.com/814201/355308-Optimization-Reference-Manual-049-Changes-Doc.pdf) | `TDP*` family: throughput 16 cycle / latency 52 cycle (BF16/INT8 dot product 모두) |

**GAP**: `LDTILECFG` 의 *정확한* cycle 수치는 본 검색 범위 미확보. **uops.info / Agner Fog instruction_tables.pdf 직접 조회** 필요. NEO 의 경우 `ensure_amx_init` 가 thread 마다 1회 호출되어 OMP team lifecycle 영역에서 amortize 됨 — 이미 best practice.

### 3.3 OpenMP barrier + thread pinning — 외부 1차 출처

| 출처 | 핵심 fact |
|---|---|
| [Effective Barrier Sync on Xeon Phi (Springer)](https://link.springer.com/chapter/10.1007/978-3-662-48096-0_45) | EPCC OpenMP barrier: 2 thread → 1.20 μs (1300 cyc), 3 → 1.76 μs, 4 → 2.39 μs |
| [Hager — Intel vs GCC OpenMP barrier shootout](https://blogs.fau.de/hager/archives/6883) | Intel libomp barrier < GCC libgomp barrier (16+ thread 에서 tournament 우위) |
| [Intel Extension for PyTorch tuning guide](https://intel.github.io/intel-extension-for-pytorch/cpu/2.1.0+cpu/tutorials/performance_tuning/tuning_guide.html) | LLM inference 표준: `KMP_BLOCKTIME=INF`, `KMP_AFFINITY=granularity=fine,compact,1,0`, jemalloc/tcmalloc 권장. **default KMP_BLOCKTIME=200 ms** |
| [IPEX LLM example](https://github.com/intel/intel-extension-for-pytorch/tree/v2.1.0+cpu/examples/cpu/inference/python/llm) | `KMP_BLOCKTIME=INF` 시 **2~3× speedup** 보고 (cache-miss check disable) |

**NEO 현재 환경** ([`README.md`의 표준 launch script 인용](../../../../../eval/run_neo_standard.sh)):
- `OMP_NUM_THREADS=10`, `OMP_PROC_BIND=false` (이미 unset), `VLLM_NEO_CPU_PIN_PER_WORKER=1`, `VLLM_NEO_CPU_PIN_CORES=12`, `VLLM_NEO_NUMA_BIND=1`
- **`KMP_BLOCKTIME` env 변경 미확인** — default 200 ms 가능성. **IPEX 권장 INF 적용 시 LLM throughput 2-3× 의 일부 흡수 여지** (현재 측정 미시도)
- `KMP_AFFINITY` 가 explicit 영역 안 보임 — `granularity=fine,compact,1,0` 또는 `verbose,scatter` (Phase 1 cpu112 분석 용도) 적용 sweep 가치 있음

### 3.4 ISPC vs hand-tuned intrinsic — 외부 fact

| 출처 | 핵심 fact |
|---|---|
| [Intel — SIMD Made Easy with ISPC](https://www.intel.com/content/www/us/en/developer/articles/technical/simd-made-easy-with-intel-ispc.html) | ISPC code ≈ hand-written intrinsic 성능. 둘 다 naive auto-vectorize 보다 빠름 |
| [ISPC Performance Guide](https://ispc.github.io/perfguide.html) | gather/scatter 가 vector load/store 보다 **느림**. branchy varying-test code 는 양쪽 branch 다 실행 후 mask |
| [AVX-512 First Impressions — Shihab Khan](https://shihab-shahriar.github.io/blog/2026/AVX-512-First-Impressions-on-Performance-and-Programmability/) | intrinsic win 영역: (a) very small inner-loop 의 register pressure, (b) cross-lane shuffle/broadcast 빈번 시, (c) BF16/VNNI 의 finer 제어 |
| [Phoronix — ISPC 1.13](https://www.phoronix.com/news/Intel-ISPC-1.13) | AVX-512 target 의 shuffle/shift/rotate/reduce 최적화로 ~5% 전체 speedup, 일부 ops 90% 가까이 |

**NEO 의 결론**: ISPC 가 이미 hand-tuned intrinsic 수준. AVX-512 intrinsic 으로 변환의 win 영역은 BF16 cast 또는 cross-lane reduce 영역에 한정 — Step 5 의 `_mm512_cvtneps_pbh` 직접 호출이 이 영역. 측정 결과 +0.4% 1-run, 3-run noise — fundamental win 아님.

### 3.5 BF16 cast cost — 외부 1차 출처 + GAP

| 출처 | 핵심 fact |
|---|---|
| [Wikichip `avx512_bf16`](https://en.wikichip.org/wiki/x86/avx512_bf16) | BF16 ↔ FP32 cast 가 single instruction. `VCVTNE2PS2BF16` = 두 zmm FP32 → 한 zmm BF16 (32 lane) |
| [felixcloutier `VCVTNEPS2BF16`](https://www.felixcloutier.com/x86/vcvtneps2bf16) | 512-bit packed FP32 (16 lane) → 256-bit packed BF16 (16 lane). round-to-nearest-even |
| [Intel — bfloat16 instruction](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-deep-learning-boost-new-instruction-bfloat16.html) | BF16 → FP32 reverse 는 별도 instruction 불필요 (상위 16-bit BF16 + 하위 16-bit 0 fill = zero-extension shuffle) |
| **GAP** | `VCVTNEPS2BF16` 의 SPR 정확한 latency/throughput/port = 본 검색 범위 미확보. uops.info 또는 Agner Fog 직접 조회 필요. 정성: latency ~5 cyc / throughput 0.5-1 cyc per issue 수준 (정성 인용 가능, 단일 1차 값 미확보) |

**F3 (K cache BF16 host store) 의 정량적 이득** (가설):
- 현재 per-block cast cost: 200-400 cycle (FP16→FP32→BF16, 16 lane batch × 128 lane = 8 iter)
- F3 적용 후: cast 영역 **영구 제거** → 80 layer × 200-400 = 16,000-32,000 cycle/step 절약
- step wall ~110 ms = 4.4 G cycle (40 GHz aggregate) 의 ~0.0007% → **단독 effect 작음 (-1%)**
- 단 **storage 50%** 영역 (BW 2× effective) — DRAM-bound 영역에서 더 큰 효과 가능 (NEO 의 cdec 영역이 L2-fit 이라 효과 제한)

### 3.6 KV cache CPU swap — 외부 1차 출처

| 출처 | 핵심 fact |
|---|---|
| [NVIDIA Forum — PCIe asymmetric bandwidth](https://forums.developer.nvidia.com/t/asymmetric-pcie-bandwidth-in-bidirectional-transfers-h2d-drops-56-while-d2h-maintains-performance/352186) | A100 PCIe Gen4 x16 pinned H2D 또는 D2H 단방향 ≈ **24 GB/s** (이론 32 GB/s 의 75%) |
| [Lenovo Press — H100 PCIe Gen5](https://lenovopress.lenovo.com/lp1732-thinksystem-nvidia-h100-pcie-gen5-gpu) | PCIe Gen5 x16 이론 unidir ≈ **64 GB/s** |
| [vLLM Blog — KV Offloading Connector](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html) | vLLM 0.11.0: pinned memory + MemcpyAsync + separate stream → async transfer latency hiding |
| [NEO MLSys 2025 PDF](http://minlanyu.seas.harvard.edu/writeup/mlsys25.pdf) | KV cache + attention compute 의 일부 CPU offload. asymmetric pipelining. T4 7.5×, A10G 26%, **H100 14% throughput up** |

**NEO 의 swap path 영역**: SUB_025/SUB_026 에서 async swap_out + staging buffer N=3 으로 +187% 달성. 추가 가속 영역은 **H100 PCIe Gen5 의 64 GB/s 가까이 도달 여부** + **bidir 동시 활용** 영역.

---

## 4. 분석 문서 ↔ 외부 1차 출처 cross-validation

### 4.1 정합 영역 (분석 진단이 외부 fact 와 일치)

| 분석 문서 진단 | 외부 1차 fact | 정합 |
|---|---|:-:|
| AMX M=8 vs tile 16 의 50% occupancy ([`reference/I_amx_proper_design.md:13-20`](reference/I_amx_proper_design.md)) | `TDPBF16PS` 16-cycle throughput tile shape 무관 (Intel Opt Ref Manual #355308) | ✓ |
| setup overhead 650-900 cyc > work 64 cyc × 10배 ([`reference/I_amx_proper_design.md:34-46`](reference/I_amx_proper_design.md)) | `LDTILECFG` high-latency, "함수 진입당 1회 amortize" (felixcloutier) | ✓ |
| `(MNK)^(1/3) ≈ 22.6` → small matmul 영역 (정량 미언급) | libxsmm cutoff `≤ 64`, dispatch negligible 영역 `≥ 20×20` (libxsmm readthedocs) | ✓ (외부 fact 가 정량 backing) |
| libgomp 43.75% (barrier spinning) ([`reference/H_dynamic_analysis.md`](reference/H_dynamic_analysis.md)) | EPCC barrier 1.3-2.6 μs / 2-4 thread × 80 layer × 256 batch = ms 단위 누적 | ✓ |
| cdec wall 70% → Amdahl ceiling 3.33× ([`reference/J_sub015_root_cause_analysis.md:15-30`](reference/J_sub015_root_cause_analysis.md)) | Amdahl law 표준 | ✓ |
| K cache BF16 store 의 이론 +1-5% (F3 가설) ([`reference/K_sub015_improvement_roadmap.md`](reference/K_sub015_improvement_roadmap.md)) | BF16 cast = single instruction, BW saving 대비 cast cost negligible (Wikichip) | ✓ |

### 4.2 분석 문서 누락 / 외부 fact 가 추가하는 영역

| 영역 | 분석 문서 상태 | 외부 fact 추가 |
|---|---|---|
| `KMP_BLOCKTIME` sweep | 미언급 | **default 200 ms → INF 시 2-3× speedup** (IPEX 권장). NEO 의 표준 launch script 에 explicit 설정 없음 — 측정 가치 매우 높음 |
| `KMP_AFFINITY` explicit | 미언급 (`VLLM_NEO_CPU_PIN_*` 가 affinity 영역) | `granularity=fine,compact,1,0` IPEX 표준. 현재 PROC_BIND=false 가 변경 영역 |
| online softmax (FlashAttention 식) | 미언급 | softmax 점유 9.73% (qk 보다 큼) — single-pass online 변환 시 2× 가능 |
| `VCVTNE2PS2BF16` paired conversion | 미언급 | `_mm512_cvtne2ps2bf16`: 두 zmm FP32 → 한 zmm BF16. F3 적용 시 K cache store 의 paired cast 로 throughput 2× |
| Sapphire Rapids SNC (Sub-NUMA Cluster) | 미언급 | SNC-4 적용 시 per-socket >1 TB/s HBM (Xeon Max 모델) — 단 NEO 의 DDR5-only Xeon Platinum 8480+ 는 ~300-400 GB/s 영역 |
| FlashDecoding++ flat-shape GEMM | 미언급 | small batch 의 padding GEMM 의 >50% under-utilization → NEO 의 M=8 동일 영역. **dedicated small-matmul kernel** 영역 (libxsmm JIT, TPP backend) |
| libxsmm / TPP backend dispatch | E_amx_avx_applicability.md 에서 일부 언급 | `(MNK)^(1/3) ≤ 64` 영역 = JIT byte-sequence emission. NEO qk 영역에 직접 적용 가능 lever |

### 4.3 분석 문서 정량 ≠ 외부 fact (재검토 필요)

| 영역 | 분석 문서 | 외부 fact | 검토 사항 |
|---|---|---|---|
| AMX peak 1024 BF16 ops/cycle | I_amx_proper_design.md 가 1024/19% = 195 도출 | AWS/Wikichip "1024 BF16 ops/cycle" 는 peak instantaneous. throughput 기반 sustained 는 16-cycle 의 throughput interpretation 가능 (= 1024/16 = 64 ops/cycle sustained) 의 해석 차이 | 두 해석 모두 외부 출처에 있음. NEO 의 effective FLOPS 측정 시 어느 정의를 쓰는지 명시 필요 |
| `_tile_dpbf16ps` 16 cycle / round | I_amx_proper_design.md:36 | Intel Opt Ref Manual #355308: throughput 16 / latency 52 cycle | ✓ |
| L2 BW 50 GB/s/core | D_roofline_notes.md:56 | McCalpin IXPUG SPR 측정: 약 200 GB/s L2 BW per socket (≈3.5 GB/s/core, 56 core) — 분석 문서가 STREAM 측정 또는 다른 정의 사용 가능 | 측정 정의 명시 필요 (per-core peak vs all-core sustained) |

---

## 5. HPC 영역 prioritized lever roadmap

### 5.1 측정 결과 backing 강도별 우선순위

[`reference/L_sub015_evidence_based_priority.md`](reference/L_sub015_evidence_based_priority.md) 의 evidence-based ranking + 외부 fact backing 으로 재책정:

| 순위 | Lever | Internal evidence | 외부 fact backing | 예상 win | Effort |
|---|---|---|---|---:|:-:|
| **★ M0** | **KMP_BLOCKTIME=INF sweep** (env-only) | 미시도 | IPEX 권장, LLM 2-3× speedup 보고 | **+1-5%** (가설 기반 정성) | **1 시간** |
| **★ M1** | KMP_AFFINITY explicit `granularity=fine,compact,1,0` | 미시도 (현재 `PROC_BIND=false`) | IPEX 표준 | **+0-3%** | 1 시간 |
| **★ M2** | **F3 K cache BF16 host store** | 가설 (시도 안 함) | BF16 cast single-instruction, storage 2× | **+1-5%** | 중 (2-3 일) |
| **★ M3** | **online softmax (single-pass)** | 분석 문서 미언급 | softmax 9.73% 점유, FlashAttention 식 streaming 표준 | **+2-5%** | 중 (3-5 일) |
| M4 | F1 async cdec pipeline depth>0 | P4 측정 -1.9% (회귀, infra 미완성) | Amdahl 완화 가능, 단 sequential KV dep 으로 implementation hard | +5-15% (이론) | 고 (1-2 주, infra 재설계) |
| M5 | F4 TP=8 → TP=4 | 가설 | M=8 → M=16 (AMX tile 100% occupancy) | +5-10% (이론) | 매우 고 (2-4 주, architecture) |
| M6 | F5 BLOCK_SIZE 16 → 32 | 가설 | per-block work 2×, AMX setup amortize | +3-7% (이론) | 매우 고 (2-4 주, scheduler) |
| ✗ | F6 OMP dynamic schedule | P2 측정 -1.4% (회귀) | atomic counter overhead | net loss | — |
| ✗ | F2 MIRROR_MAX sweep | P5 측정 80 = best, 축소·확대 모두 회귀 | 이미 sweet spot | — | — |

### 5.2 Phase 단계별 plan

#### Phase α (Quick wins, effort 1-2 일)

**M0 + M1**: env-only sweep — 코드 변경 0, 빌드 0.

```bash
# Sweep 매트릭스 (각 100p × 8192 short, 1-run, ~7 min/sample = ~30 min total)
KMP_BLOCKTIME 값       : 0, 1, 50, 200 (default), INF
KMP_AFFINITY 값        : verbose,scatter (현재) vs granularity=fine,compact,1,0
```

**측정 metric**:
- output_tps
- vmstat 의 system + iowait %
- perf record: libgomp / libpacpu / libtorch_cpu % 변화

**선택 기준**: best combo 의 3-run avg 가 S1-S9 baseline (2,238.6) 의 ±2% 안에 들면 **fundamental tuning 적합도 confirm**. baseline 초과 시 신규 best.

#### Phase β (HPC 영역 surgical change, effort 3-5 일)

**M3 online softmax** — pacpu.ispc 영역 single-pass 변환:

```c
// 현재: 3-pass (max → exp + sum → normalize)
// 변경: single-pass online (FlashAttention 식)
//   m_i = max(m_{i-1}, x_i)
//   l_i = l_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)
//   sum = l_n
```

기존 ISPC 영역 `qk_softmax_av` 의 3 영역 통합 시도. ILP 회복 + cache locality 개선.

**M2 F3 K cache BF16 host store** — Python + C++ 변경:
- `vllm/v1/core/sched/neo_cpu_kv_buffer.py`: dtype FP16 → BF16
- `vllm/v1/worker/gpu_model_runner.py`: GPU FP16 → host BF16 swap-out (`_mm512_cvtne2ps2bf16` paired)
- `csrc/cpu/pacpu/core.h:Step 0 store_kv`: BF16 store
- `csrc/cpu/pacpu/amx_kernel.cpp`: K 변환 영역 제거

**검증**:
- 정확도 (numerical precision 영역): per-token logprob max abs diff, sequence PPL relative diff
- 정량: 3-run avg, S1-S9 baseline 대비

#### Phase γ (architectural change, effort 2-4 주)

**F4 (TP=8→4)** + **F5 (BLOCK_SIZE→32)** 영역. 본 phase 는 NEO scheduler / block manager / GPU 측 parallelism 영향 측정 후 결정. **현재 Phase 3 영역 밖** — 별도 SUB 발급 영역.

### 5.3 시너지 분석

[`reference/K_sub015_improvement_roadmap.md:30-57`](reference/K_sub015_improvement_roadmap.md) 의 interaction 표 + 외부 fact:

| Pair | Interaction | Mechanism | 외부 fact backing |
|---|:-:|---|---|
| (M0, M1) | + | KMP_BLOCKTIME × affinity 의 nested effect | IPEX LLM example 공통 |
| (M0, M2) | + | barrier spin 절약 + cast 영역 제거 누적 | — |
| (M3, M2) | + | softmax single-pass × K BF16 (cast 제거) 양쪽 모두 cycle saving | — |
| (M4, F4/F5) | + | async pipeline 안에서 큰 matmul 의 cdec wall 단축 | Amdahl law |
| (M4, M0) | + | KMP_BLOCKTIME=INF 가 async future 영역의 wait 영역 단축 | IPEX |

**Cumulative 보수 추정**: M0+M1 (+1-3%) + M2 (+1-3%) + M3 (+2-5%) = **+4-11%**. M4 이상은 측정 후 결정.

---

## 6. NEO pacpu 의 미해결 HPC 영역 (분석 문서 미언급 + 외부 fact 기반 신규 제안)

### 6.1 libxsmm / TPP backend dispatch 도입

**source**: [TPP arxiv 2304.12576](https://arxiv.org/pdf/2304.12576), [libxsmm readthedocs](https://libxsmm.readthedocs.io/en/latest/)

NEO 의 qk_product 는 `(MNK)^(1/3) ≈ 22.6` 의 small matmul 영역. libxsmm 의 JIT byte-sequence emission 으로 **problem-shape-specialized kernel** 생성 가능. AMX dispatch 영역 자동 결정 (M ≥ X 시 AMX, 미만 시 AVX-512 BF16 FMA).

**Effort**: libxsmm 의존성 추가 + pacpu 영역 의 qk/av kernel 영역에 libxsmm wrapping. **2-3 일** 영역.
**예상 effect**: AMX의 setup overhead 영역을 libxsmm 의 dispatch heuristic 이 자동 회피. NEO 의 M=8 영역에서는 AVX-512 BF16 fallback → ISPC 영역과 동등 또는 win 가능.

### 6.2 online softmax (Phase β 의 M3)

**source**: [Princeton NLP FlashDecoding](https://princeton-nlp.github.io/flash-decoding/)

이미 5.2 Phase β 에서 다룸. **softmax 점유 9.73% 가 분석 문서에서 별도 lever 영역으로 분리 안 됨** — 본 문서가 추가 제안 영역.

### 6.3 KMP_BLOCKTIME=INF 적용 (Phase α 의 M0)

**source**: [IPEX tuning guide](https://intel.github.io/intel-extension-for-pytorch/cpu/2.1.0+cpu/tutorials/performance_tuning/tuning_guide.html)

분석 문서에 미언급. NEO 의 표준 launch script ([`eval/run_neo_standard.sh`](../../../../../eval/run_neo_standard.sh)) 에서 explicit 설정 없음 → default 200 ms 가능성. **frequent parallel region (80 layer × decode-step) 에서 cache-miss check spin 영역 disable 로 2-3× speedup 보고**. 시도 가치 매우 높음 (env-only, 1 시간).

### 6.4 PCIe Gen5 bidir 활용

**source**: [vLLM Blog](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html), [arxiv 2512.16056](https://arxiv.org/html/2512.16056)

SUB_025/026 의 async swap_out 가 H2D direction 만 최적화. PCIe Gen5 의 bidir 64 GB/s × 2 = 128 GB/s 영역 중 D2H async 도 동시 활용 영역 가능. **multipath memory access (NVLink + PCIe + CXL 동시)** 가 최신 연구 방향.

**현재 NEO** = single PCIe lane. **이 영역은 SUB_015 영역 밖** (별도 SUB 발급 가치 있음).

### 6.5 BF16 paired conversion `_mm512_cvtne2ps2bf16`

**source**: [felixcloutier](https://www.felixcloutier.com/x86/vcvtne2ps2bf16)

현재 amx_kernel.cpp:89-100 의 vectorized FP16→BF16 는 `_mm512_cvtph_ps` (FP16→FP32, 256 bit input) + `_mm512_cvtneps_pbh` (FP32→BF16, 32 lane output) 의 2-step. **`_mm512_cvtne2ps2bf16` 를 쓰면 두 zmm FP32 → 한 zmm BF16 (32 lane)** 으로 throughput 2×.

**적용 영역**: F3 (M2) 적용 시 K cache store 영역 의 paired cast. effort 작음 (1 일).

---

## 7. 측정 plan (Phase α 적용 후)

### 7.1 Phase α 측정 매트릭스

| Run | KMP_BLOCKTIME | KMP_AFFINITY | 예상 시간 | 측정 metric |
|---|---|---|---:|---|
| baseline | 0 (현재) | (현재) | 7 min | output_tps, libgomp %, libpacpu % |
| α1 | INF | (현재 unset) | 7 min | output_tps Δ |
| α2 | 50 | (현재 unset) | 7 min | output_tps Δ |
| α3 | 0 | granularity=fine,compact,1,0 | 7 min | output_tps Δ |
| **α4** | **INF** | **granularity=fine,compact,1,0** | **7 min** | output_tps Δ |
| α5 | 50 | granularity=fine,compact,1,0 | 7 min | output_tps Δ |

→ 30-40 min total. workload = 100p × 8192 short (variance 우려시 best combo 3-run avg).

### 7.2 Phase β 측정

| Run | 변경 | 검증 |
|---|---|---|
| β1 | M3 online softmax (ISPC kernel 변경 + 빌드) | TST 정확도 (lp diff, ppl diff) + 3-run avg |
| β2 | M2 K BF16 store (Python + C++) | 정확도 + 3-run avg |
| β3 | β1 + β2 cumulative | 정확도 + 3-run avg |

### 7.3 성공 기준

| 단계 | 성공 기준 |
|---|---|
| Phase α | 3-run avg ≥ S1-S9 baseline (2,238.6 tps) 의 ±1% (정합 보장) — best combo 가 신규 best 발급 가능 |
| Phase β | 3-run avg ≥ S1-S9 + 2% (= 2,283 tps) — **신규 best 자격** |
| Phase γ | 별도 SUB 발급 (architectural 영역) |

---

## 8. 결론

### 8.1 SUB_015-Phase 3 의 진정한 진단

1. **HPC dropin 한계 입증**: AMX-only / cheap variant cumulative 시도 모두 -2.35% ~ -4.3% 회귀. 외부 1차 fact (OpenBLAS, AWS, libxsmm, FlashDecoding++) 가 동일 메커니즘 backing.
2. **Amdahl ceiling 정합**: cdec wall 70% × Amdahl law = ceiling 3.33× — HPC 영역 win 이 분산 차원에서 상쇄.
3. **현재 best (S1-S9, 2,238.6 tps) = NEO 원본 100% 정합** — 이미 알고리즘 best.

### 8.2 다음 turn 추천 (실행 plan)

**즉시 (1-2 일)**: Phase α (M0+M1) sweep → baseline tuning 확정.

**다음 1-2 주**: Phase β (M2+M3) 의 정합 적용 + 정확도 검증 + 3-run avg.

**중장기 (2-4 주)**: Phase γ (F4/F5) 의 별도 SUB 발급 결정 (필요 시).

### 8.3 분석 문서 ↔ 외부 fact 정합도 평가

| 영역 | 정합 |
|---|:-:|
| AMX small-matmul 회귀 root cause | ✓ (외부 다수 fact backing) |
| setup overhead vs work amortize | ✓ (felixcloutier 가이드 정합) |
| libgomp barrier overhead | ✓ (EPCC 측정 정합) |
| Amdahl 한계 | ✓ (표준 law) |

**분석 문서 누락 영역** (본 문서 추가):
- KMP_BLOCKTIME / KMP_AFFINITY explicit tuning
- online softmax (FlashAttention 식 streaming)
- `_mm512_cvtne2ps2bf16` paired conversion
- libxsmm / TPP backend dispatch 도입

---

## 9. References

### 분석 문서 (내부)
- [`reference/I_amx_proper_design.md`](reference/I_amx_proper_design.md) — AMX Strategy ranking (Step 1~6 설계)
- [`reference/J_sub015_root_cause_analysis.md`](reference/J_sub015_root_cause_analysis.md) — Amdahl 한계 정밀 분석
- [`reference/K_sub015_improvement_roadmap.md`](reference/K_sub015_improvement_roadmap.md) — F1~F6 lever roadmap
- [`reference/L_sub015_evidence_based_priority.md`](reference/L_sub015_evidence_based_priority.md) — evidence-based 재책정
- [`reference/H_dynamic_analysis.md`](reference/H_dynamic_analysis.md) — perf record 결과
- [`archive/E_amx_avx_applicability.md`](archive/E_amx_avx_applicability.md), [`archive/C_pacpu_vs_cpu_attn_amx_gap.md`](archive/C_pacpu_vs_cpu_attn_amx_gap.md), [`archive/D_roofline_notes.md`](archive/D_roofline_notes.md)

### 측정 (내부)
- [`../measurements/sub015_p3_measurement_timeline_20260518.md`](../measurements/sub015_p3_measurement_timeline_20260518.md)
- [`../measurements/sub015_p3_amx_500p_3run_20260518/README.md`](../measurements/sub015_p3_amx_500p_3run_20260518/README.md) — Phase 3 A 3-run
- [`../measurements/sub015_p3_amx_steps_500p_1run_20260518/README.md`](../measurements/sub015_p3_amx_steps_500p_1run_20260518/README.md) — Step 1~6 sweep
- [`../measurements/sub015_p3_step5_amx_bav_500p_3run_20260518/README.md`](../measurements/sub015_p3_step5_amx_bav_500p_3run_20260518/README.md) — Step 5 3-run 정식
- [`../measurements/p3_compare_3run_085_20260520/README.md`](../measurements/p3_compare_3run_085_20260520/README.md) — gmu=0.85 5-case 3-run
- [`../measurements/p4_p5_lever_20260520/README.md`](../measurements/p4_p5_lever_20260520/README.md) — F1/F2 검증

### 소스 코드 (내부)
- [`csrc/cpu/pacpu/pacpu.ispc`](../../../../../csrc/cpu/pacpu/pacpu.ispc) — ISPC qk/softmax/av/gather kernel
- [`csrc/cpu/pacpu/amx_kernel.cpp`](../../../../../csrc/cpu/pacpu/amx_kernel.cpp) — AMX BF16 host C++ (env-gated)
- [`csrc/cpu/pacpu/core.h`](../../../../../csrc/cpu/pacpu/core.h) — OMP partition + dispatch
- [`csrc/cpu/pacpu/dtype.h`](../../../../../csrc/cpu/pacpu/dtype.h) — config macros (BLOCK_SIZE, HEAD_DIM, NUM_Q_HEADS)

### AMX / Intel ISA (외부 1차)
- [felixcloutier `TDPBF16PS`](https://www.felixcloutier.com/x86/tdpbf16ps) — throughput 16 cyc / latency 52 cyc
- [felixcloutier `LDTILECFG`](https://www.felixcloutier.com/x86/ldtilecfg) — high-latency, amortize 가이드
- [felixcloutier `TILELOADDT1`](https://www.felixcloutier.com/x86/tileloaddt1) — throughput 23 / latency 48 cyc
- [felixcloutier `VCVTNEPS2BF16`](https://www.felixcloutier.com/x86/vcvtneps2bf16)
- [felixcloutier `VCVTNE2PS2BF16`](https://www.felixcloutier.com/x86/vcvtne2ps2bf16)
- [Wikipedia AMX](https://en.wikipedia.org/wiki/Advanced_Matrix_Extensions)
- [Wikichip x86 AMX](https://en.wikichip.org/wiki/x86/amx)
- [Wikichip `avx512_bf16`](https://en.wikichip.org/wiki/x86/avx512_bf16)
- [Wikichip Fuse — AMX brings matrix operations](https://fuse.wikichip.org/news/3600/the-x86-advanced-matrix-extension-amx-brings-matrix-operations-to-debut-with-sapphire-rapids/)
- [Intel — AMX product brief PDF](https://cdrdv2-public.intel.com/785250/Intel-AMXBrief-Final-3.17.pdf)
- [Intel Optimization Reference Manual #355308 v049](https://cdrdv2-public.intel.com/814201/355308-Optimization-Reference-Manual-049-Changes-Doc.pdf)
- [Intel Optimization Reference Manual #355308 v050](https://cdrdv2-public.intel.com/821613/355308-Optimization-Reference-Manual-050-Changes-Doc.pdf)
- [Intel — Deep Learning Boost bfloat16](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-deep-learning-boost-new-instruction-bfloat16.html)
- [Fixstars Tech Blog — Intel AMX Explained](https://blog.us.fixstars.com/intel-amx-advanced-matrix-extension-explained-introduction/)
- [llvm-dev Intel AMX programming model](https://groups.google.com/g/llvm-dev/c/caHJWyUNWNk)

### AMX 적용 평가 / break-even (외부)
- [OpenBLAS Discussion #5205 — small batch AVX-512 BF16 faster than AMX](https://github.com/OpenMathLib/OpenBLAS/discussions/5205)
- [AWS Compute Blog — Accelerate CPU AI with Intel AMX on EC2](https://aws.amazon.com/blogs/compute/accelerate-cpu-based-ai-inference-workloads-using-intel-amx-on-amazon-ec2/)
- [Microsoft Open Source — ONNX Runtime Intel AMX](https://opensource.microsoft.com/blog/2023/09/07/boosting-performance-in-onnx-runtime-with-intel-amx-for-4th-gen-intel-xeon-processors/)
- [phoenixNAP KB — Intel AMX Explained](https://phoenixnap.com/kb/intel-amx-advanced-matrix-extensions)
- [Intel — Tuning Guide for AI on 4th Gen Xeon](https://www.intel.com/content/www/us/en/developer/articles/technical/tuning-guide-for-ai-on-the-4th-generation.html)

### attention kernel HPC (외부)
- [FlashDecoding++ MLSys 2024 PDF](https://proceedings.mlsys.org/paper_files/paper/2024/file/5321b1dabcd2be188d796c21b733e8c7-Paper-Conference.pdf)
- [Flash-Decoding — PyTorch Blog](https://pytorch.org/blog/flash-decoding/)
- [Flash-Decoding — Princeton NLP](https://princeton-nlp.github.io/flash-decoding/)
- [Sequence-Aware Split Heuristic FA-3 — arxiv 2604.00028](https://arxiv.org/pdf/2604.00028)
- [Mind the Memory Gap — arxiv 2503.08311](https://arxiv.org/html/2503.08311v2)
- [Roofline model — Williams et al. PDF](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- [NEO MLSys 2025 PDF](http://minlanyu.seas.harvard.edu/writeup/mlsys25.pdf)
- [NEO arxiv 2411.01142](https://arxiv.org/abs/2411.01142)

### ISPC / hand-tuned intrinsic (외부)
- [Pharr & Mark — ISPC SPMD compiler PDF](https://pharr.org/matt/assets/ispc.pdf)
- [Intel — SIMD Made Easy with ISPC](https://www.intel.com/content/www/us/en/developer/articles/technical/simd-made-easy-with-intel-ispc.html)
- [ISPC Performance Guide](https://ispc.github.io/perfguide.html)
- [Phoronix — Intel ISPC 1.13 brings AVX-512 perf](https://www.phoronix.com/news/Intel-ISPC-1.13)
- [AVX-512 First Impressions — Shihab Khan blog](https://shihab-shahriar.github.io/blog/2026/AVX-512-First-Impressions-on-Performance-and-Programmability/)

### OpenMP / NUMA / thread pinning (외부)
- [Intel — Thread Affinity Interface](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/thread-affinity-interface.html)
- [GCC libgomp — GOMP_CPU_AFFINITY](https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fCPU_005fAFFINITY.html)
- [NAS HECC — Intel OpenMP Thread Affinity](https://www.nas.nasa.gov/hecc/support/kb/using-intel-openmp-thread-affinity-for-pinning_285.html)
- [LLNL HPC — Process/Thread Affinity tutorial PDF](https://hpc-tutorials.llnl.gov/openmp/ProcessThreadAffinity.pdf)
- [Intel Extension for PyTorch tuning guide](https://intel.github.io/intel-extension-for-pytorch/cpu/2.1.0+cpu/tutorials/performance_tuning/tuning_guide.html)
- [IPEX LLM example](https://github.com/intel/intel-extension-for-pytorch/tree/v2.1.0+cpu/examples/cpu/inference/python/llm)
- [Effective Barrier Sync on Xeon Phi — Springer](https://link.springer.com/chapter/10.1007/978-3-662-48096-0_45)
- [Hager — Intel vs GCC OpenMP barrier shootout](https://blogs.fau.de/hager/archives/6883)
- [Microsoft Learn — OpenMP directives](https://learn.microsoft.com/en-us/cpp/parallel/openmp/reference/openmp-directives?view=msvc-170)
- [Intel — 4th Gen Xeon Scalable Family overview](https://www.intel.com/content/www/us/en/developer/articles/technical/fourth-generation-xeon-scalable-family-overview.html)
- [Next Platform — HBM Gives Xeon SPs A Big Boost](https://www.nextplatform.com/2022/11/15/sapphire-rapids-xeon-sps-plus-hbm-offer-big-performance-boost/)
- [McCalpin — Bandwidth Limits Xeon Max (IXPUG ISC23 PDF)](https://www.ixpug.org/images/docs/ISC23/McCalpin_SPR_BW_limits_2023-05-24_final.pdf)
- [Chips and Cheese — Sapphire Rapids: Golden Cove Hits Servers](https://chipsandcheese.com/p/a-peek-at-sapphire-rapids)
- [Understanding LLM Inference on CPUs (NSF) PDF](https://par.nsf.gov/servlets/purl/10576248)
- [Improving Throughput-oriented LLM Inference with CPU (PACT)](https://dl.acm.org/doi/fullHtml/10.1145/3656019.3676949)

### KV cache offload / PCIe (외부)
- [NVIDIA Forum — PCIe asymmetric bandwidth](https://forums.developer.nvidia.com/t/asymmetric-pcie-bandwidth-in-bidirectional-transfers-h2d-drops-56-while-d2h-maintains-performance/352186)
- [Lenovo Press — ThinkSystem H100 PCIe Gen5](https://lenovopress.lenovo.com/lp1732-thinksystem-nvidia-h100-pcie-gen5-gpu)
- [Telesens — Pipelining H2D data processing](https://www.telesens.co/2019/02/16/efficient-data-transfer-from-paged-memory-to-gpu-using-multi-threading/)
- [Multipath Memory Access — arxiv 2512.16056](https://arxiv.org/html/2512.16056)
- [vLLM Blog — KV Offloading Connector](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Issue #16144 — Offload KV cache to CPU in V1](https://github.com/vllm-project/vllm/issues/16144)

### Small matmul / micro-kernel (외부)
- [libxsmm — README](https://github.com/libxsmm/libxsmm)
- [libxsmm — readthedocs](https://libxsmm.readthedocs.io/en/latest/)
- [LIBXSMM SC15 poster PDF](https://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/poster_files/post137s2-file3.pdf)
- [BLIS — KernelsHowTo](https://github.com/flame/blis/blob/master/docs/KernelsHowTo.md)
- [BLISlab — arxiv 1609.00076](https://arxiv.org/pdf/1609.00076)
- [GEMMFIP — arxiv 2302.08417](https://arxiv.org/pdf/2302.08417)
- [LibShalom — small/irregular GEMM PDF](https://jianbinfang.github.io/files/2021-06-22-sc.pdf)
- [Exo — Matrix multiplication micro-kernel arxiv 2310.17408](https://arxiv.org/pdf/2310.17408)
- [TVM — Automatic generators for GEMM arxiv 2310.20347](https://arxiv.org/pdf/2310.20347)
- [TPP — High-Level Loop/Tensor Abstractions arxiv 2304.12576](https://arxiv.org/pdf/2304.12576)

### llama.cpp / CPU LLM (외부)
- [llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [llama.cpp Flash Attention DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/8.2-flash-attention-and-optimizations)
- [llama.cpp CPU Backend DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/4.2-cpu-backend-and-optimization)
- [llama.cpp Issue #2555 — AMX](https://github.com/ggml-org/llama.cpp/issues/2555)
- [ik_llama.cpp fork](https://github.com/ikawrakow/ik_llama.cpp)

### 본 문서가 GAP 으로 표시한 영역 (정확한 cycle / 정량 필요 시 1차 source)
- [Agner Fog — Instruction tables PDF](https://www.agner.org/optimize/instruction_tables.pdf)
- [Agner Fog — main page](https://www.agner.org/optimize/)
- [uops.info — instruction cycle DB](https://uops.info/links.html)
- Intel Architecture Instruction Set Extensions Programming Reference (#319433)
- oneDNN source: `src/cpu/x64/matmul/brgemm_matmul_utils.cpp`, `src/cpu/x64/amx_tile_configure.cpp`

---

## 10. Change Log

| 일자 (KST) | 변경 |
|---|---|
| **2026-05-20** | 신설. HPC 측면 분석 + 외부 1차 출처 backing + 다음 lever roadmap. branch `feat/neo-amx-apply` HEAD `dd80747a6` 기준. |
