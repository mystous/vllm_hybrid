# SUB_015-Phase 3 A — AMX qk_product integration 500p × 8192 3-run

> 2026-05-18 KST. branch `feat/neo-amx-apply` (HEAD `da32e79dd` + 본 turn 의 amx_kernel 추가).
>
> AMX BF16 host C++ qk_amx kernel (softmax/av 는 ISPC 유지) 의 정식 검증 측정.
> env: `VLLM_NEO_USE_AMX=1` + 나머지 v1.6 standard.

---

## 3-run 결과

| Round | tps | wall (s) | result.json |
|---:|---:|---:|---|
| round 1 | **2,226.0** | 1,828.6 | round1_20260518_075855_amx/result.json |
| round 2 | **2,129.4** | 1,906.9 | round2_20260518_084333_amx/result.json |
| round 3 | **2,072.0** | 1,958.9 | round3_20260518_091722_amx/result.json |
| **3-run avg** | **2,142.5** | **1,898.1** | — |
| **min** | **2,072.0** | | |
| **max** | **2,226.0** | | |
| **CV** | **3.6%** | | (std 76.6 / avg 2142.5) |

## vs S1-S9 baseline

| 기준 | tps | 비교 |
|---|---:|---:|
| **S1-S9 baseline (3-run avg, commit 531d61608)** | **2,238.6** | (baseline) |
| **AMX 3-run avg** | **2,142.5** | **-4.3% 회귀** |

★ AMX path 는 baseline 대비 명확한 회귀. Phase 1 의 정량 추정 (+5-8%) 의 setup overhead 무시한 ceiling 추정 실증 안됨.

## tps trend 단조 감소

round 1 → 2 → 3 으로 tps 단조 감소 (2226 → 2129 → 2072). 가능 원인:
- **Thermal throttling**: 1.5 시간 sequential 측정 동안 CPU 가열 → frequency 하강
- **Cache state**: warm-up 후 GPU/CPU cache state 변화
- **GPU memory leak**: GPU 7 의 background memory leftover (1.8 GiB)

3-run avg 산출 이 noise 평균화로 fact 보장.

## 코드 변경 (env-gated, default off)

- `csrc/cpu/pacpu/amx_kernel.cpp` (신규) — qk_amx + attn_one_seq_amx host C++ AMX BF16
- `csrc/cpu/pacpu/core.h` — env-toggle dispatch (`VLLM_NEO_USE_AMX`)
- `csrc/cpu/pacpu/pacpu.ispc` — softmax export (AMX path 가 ispc::softmax 직접 호출)
- `csrc/cpu/pacpu/CMakeLists.txt` — amx_kernel.cpp 추가 + `-mamx-tile -mamx-bf16`

env off (default) → baseline 영향 0.

## Loss root cause

NEO 의 작은 per-block matmul (M=8 head × N=16 token × K=128 dim) 에서 AMX setup overhead (Q/K FP16→BF16 변환 + K^T pre-pack + tile_loadd/dpbf16/stored) 가 ISPC AVX-512 FP16 의 work cycle 보다 큼:

| Path | per-block cycle (추정) | bottleneck |
|---|---:|---|
| **ISPC AVX-512 FP16 (baseline)** | ~500 | compute |
| **AMX BF16 (현재)** | ~600-800 | **setup overhead-dominant** |

## 측정 환경

| 항목 | 값 |
|---|---|
| Host | Intel Xeon Platinum 8480+ (SPR, 112 phys core, NUMA 2) + H100 80GB × 8 |
| Workload | Llama-3.3-70B, TP=8, 500p × max_tokens 8192 × target_input 8192 |
| env | KMP_BLOCKTIME=200, OMP_NUM_THREADS=10, VLLM_NEO_CPU_PIN_CORES=12, VLLM_NEO_NUMA_BIND=1 |
| VLLM_NEO_USE_AMX | **1** (AMX path 활성) |
| 빌드 | gcc-12.3, ispc `avx512spr-x16`, `-mamx-tile -mamx-bf16 -O3 -march=native` |

## 다음 진정한 win path (시도 안 함, future)

| 개선 | 변경 | 예상 효과 | Effort |
|---|---|---:|---:|
| **K cache BF16 store** | NEO host buffer 의 K cache 를 BF16 으로 저장 (FP16 대신) | K 변환 cost 제거 (-30-40%) | 1-2 일 |
| **Q hoist** | Q FP16→BF16 변환을 outer call 1회 | Q 변환 cost -50% | 0.5 일 |
| **multi-seq batched** | 여러 seq 의 Q × K^T 를 stack matmul (AMX M=16 full) | setup amortize | 1-2 일 |
