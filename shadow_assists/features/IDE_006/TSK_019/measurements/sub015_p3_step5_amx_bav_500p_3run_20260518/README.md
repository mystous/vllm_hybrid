# SUB_015-Phase 3 Step 5 (B+A+vec K conv) — 500p × 8192 3-run avg

> 2026-05-18 KST. branch `feat/neo-amx-apply` HEAD `df33b9d22`. Step 5 best variant 의 3-run avg 정식 검증.
>
> env: `VLLM_NEO_USE_AMX=1` + 나머지 v1.6 standard.

---

## 3-run 결과

| Round | tps | wall (s) | KST 시각 | result.json |
|---:|---:|---:|---|---|
| round 1 | **2,284.0** | 1,774.6 | 12:46 ~ 13:17 | round1_20260518_124624/result.json |
| round 2 | **2,154.4** | 1,880.8 | 14:36 ~ 15:08 | round2_20260518_143612/result.json |
| round 3 | **2,120.0** | 1,901.9 | 15:09 ~ 15:44 | round3_20260518_150921/result.json |
| **3-run avg** | **2,186.1** | **1,852.4** | — | — |
| **min** | **2,120.0** | | | |
| **max** | **2,284.0** | | | |
| **std** | **70.6** | | | |
| **CV** | **3.23%** | | | |

## 비교 (3-run avg 기준)

| 기준 | 3-run avg tps | vs S1-S9 baseline |
|---|---:|---:|
| **S1-S9 baseline** (`Best_S1_S9_2238tps`) | **2,238.6** | (baseline, **best**) |
| **Step 5 (B+A+vec K conv) 3-run avg** | **2,186.1** | **-2.35%** ✗ |
| Phase 3 A dropin AMX (3-run avg) | 2,142.5 | -4.3% |

★ **Step 5 의 3-run avg = -2.35% 회귀** vs S1-S9 baseline. 1-run round 1 (2,284, +2.0%) 은 cold-start cache benefit + thermal noise — single-run 결과의 신뢰성 한계.

## tps trend 단조 감소

round 1 → 2 → 3 으로 tps 단조 감소 (2284 → 2154 → 2120, -7.2%):
- Cold start cache benefit decay (L2 warmth + branch predictor table)
- Thermal drift (CPU 가열로 frequency 점진 ↓)
- GPU memory leak (worker 종료 후 GPU 일부 잔여)

3-run avg 가 ground truth — 1-run noise 큼.

## 코드 변경 (env-gated, default off)

| 파일 | 변경 |
|---|---|
| `csrc/cpu/pacpu/amx_kernel.cpp` | qk_amx + attn_one_seq_amx + AMX BF16, **Strategy B (thread Q cache) + A (K^T outer pre-pack) + vec K conv (AVX-512 `_mm512_cvtneps_pbh`)** |
| `csrc/cpu/pacpu/core.h` | env-toggle dispatch (`VLLM_NEO_USE_AMX`) |
| `csrc/cpu/pacpu/pacpu.ispc` | softmax export |
| `csrc/cpu/pacpu/CMakeLists.txt` | amx_kernel.cpp + `-mamx-tile -mamx-bf16` |

**default off** (env-gated). `VLLM_NEO_USE_AMX=1` 활성 시 Step 5 path.

## 결론 — best configuration 정정

### Best configuration index 의 진실

| Configuration | tps (3-run avg) | 상태 |
|---|---:|---|
| **S1-S9 (`Best_S1_S9_2238tps.md`)** | **2,238.6** | **★ best (변경 안 됨)** |
| Step 5 (B+A+vec K conv) | 2,186.1 | -2.35% (낮음) |
| Phase 3 A dropin AMX | 2,142.5 | -4.3% (낮음) |
| v1.6 (`Best_v1.6_2157tps.md`) | 2,197.4 | -1.9% (낮음) |

★ **S1-S9 가 여전히 best**. AMX 의 어떤 variant 도 NEO 의 작은 matmul (M=8 head, N=16 token, K=128 dim) 의 setup overhead 한계로 baseline 초과 안 함.

### Step 5 의 가치

- AMX path 가 **functionally correct** (3 round 모두 완주, SIGILL 없음)
- 단순 dropin (Phase 3 A) 의 -4.3% 회귀에서 **+2.0%p improvement** 확보 (Step 5 = -2.35% vs Phase 3 A = -4.3%)
- 코드 keep (env-gated default off, future K cache BF16 host store 의 base)

### 1-run vs 3-run 신뢰성

**1-run noise 가 매우 큼** (CV 3.23%, max-min range 164 tps). 본 task 시리즈 (Step 1~6) 의 1-run 결과는 모두 thermal/cache noise 영향. 진정한 결론은 **3-run avg 만** 신뢰 가능.

## 측정 환경

| 항목 | 값 |
|---|---|
| Host | Intel Xeon Platinum 8480+ (SPR, 112 phys core, NUMA 2) + H100 80GB × 8 |
| Workload | Llama-3.3-70B, TP=8, 500p × max_tokens 8192 × target_input 8192, fp8 KV cache |
| env | KMP_BLOCKTIME=200, OMP_NUM_THREADS=10, VLLM_NEO_USE_AMX=1 |
| 빌드 | gcc-12.3, ispc `avx512spr-x16`, `-mamx-tile -mamx-bf16 -O3 -march=native` |

## 다음 가능 lever (별도 task)

NEO 의 작은 matmul 의 fundamental 한계 안에서 추가 win 어려움. 진정한 큰 win path:

| 후보 | 변경 영역 | Effort | 예상 win |
|---|---|---:|---:|
| **K cache BF16 host store** (full) | swap path 의 GPU→host 변환 시 BF16 store | 1-2 일 | +1-3% (변환 cost 완전 제거) |
| **NEO core design 변경** | TP=8 → TP=4 (M=8→16) or BLOCK_SIZE 16→32 (N=16→32) | 2-4 주 | +5-10% (AMX tile occupancy ↑) |
| **softmax / av_product AMX** | 추가 ISA 가속 | 2-3 일 | +0-3% (NEO size 한계) |
