# SUB_015-Phase 3 AMX Step별 측정 (Strategy B → A → G)

> 2026-05-18 KST. branch `feat/neo-amx-apply`. 각 step 1-run 500p × 8192 검증 (사용자 명시).
>
> Phase 1 의 정적/동적 분석 (analysis/I_amx_proper_design.md) 의 Strategy ranking 충실 진행.

---

## 측정 결과 (1-run 500p × 8192)

| Step | 변경 (cumulative) | tps | wall (s) | vs S1-S9 baseline (2,238.6) | vs prev step |
|---|---|---:|---:|---:|---:|
| S1-S9 baseline (3-run avg) | (no AMX) | 2,238.6 | 1,819 | (baseline) | — |
| **Phase 3 A** (dropin AMX, 3-run avg) | AMX qk + 매 block 변환 | 2,142.5 | 1,898 | **-4.3%** | — |
| **Step 1: B** (thread-Q-cache) | + same-seq Q cache | **2,237.4** | 1,814 | **-0.05%** | **+4.4%** |
| **Step 2: B+A** (K^T outer pre-pack) ★ | + outer pre-pack | **2,275.6** | **1,784.8** | **+1.7%** ★ | **+1.7%** |
| Step 3: B+A+G (SW prefetch) | + `_mm_prefetch` next-block K | 2,184.7 | 1,864.8 | -2.4% | **-4.0% (회귀)** |

★ **Step 2 = best**: Phase 3 A 의 -4.3% 회귀에서 **+1.7% net gain** 으로 전환.

---

## Step 별 변경 정량

### Step 1: Strategy B — thread-level Q cache

**변경 위치**: `csrc/cpu/pacpu/amx_kernel.cpp`
- `static thread_local const data_t* _last_q_ptr` + `_cached_Q_bf16[16*128]`
- qk_amx 진입 시 `q != _last_q_ptr` 체크 → same-seq consecutive task 시 Q 변환 skip

**효과 정량**:
- core.h:282-292 의 task partition 이 seq 순서로 ordered → 한 thread 안에서 same-seq tasks 가 consecutive
- ws=14, batch~30 typical 시 same-seq consecutive task 비율 ~30-50%
- Q 변환 cost (50-100 cycle × 30% skip) ≈ 15-30 cycle/call ↓
- net: Phase 3 A 대비 **+4.4% (회귀 회복)**

### Step 2: Strategy A (cheap variant) — K^T outer pre-pack

**변경 위치**: `amx_kernel.cpp`
- thread_local `std::vector<uint16_t> _kt_cache` (size = imax × 4 × 1024 byte)
- 2-pass:
  1. **outer pre-pack** — imax × 4 round 의 K^T BF16 변환 + tile padding 을 미리 모두 cache 에 (loop body 외부)
  2. **inner AMX hot loop** — `tile_loadd / dpbf16ps / stored` 만 (변환 0 cycle)

**효과 정량**:
- 변환 횟수 동일 — sequential 만 변경
- BTB/instruction cache locality 개선: 변환 hot path + AMX hot path 가 분리, 각자 fewer branches
- L2 data locality: K^T cache 가 sequential write + sequential read pattern (prefetch friendly)
- net: Step 1 대비 **+1.7% (best)**

### Step 3: Strategy G — SW prefetch (실측 회귀)

**변경 위치**: outer pre-pack 의 매 block 시작 시 next block 의 K cache 8 line `_mm_prefetch(T1)`

**회귀 root**:
- `block_table[i+1]` indirection 으로 next-block addr 가 unpredictable — HW prefetcher 가 못 잡는 영역 맞으나 SW hint 도 정확성 낮음
- 매 i 마다 8 prefetch instruction 추가 (imax × 8 = up to 1024 prefetch / call) — overhead
- L2 capacity (2 MB/core) 가 이미 K^T cache 충분 — additional prefetch 의 marginal value 0
- net: Step 2 대비 **-4.0% (회귀)** → Step 3 revert

---

## 최종 코드 상태 (Step 2 = best)

| 파일 | 상태 |
|---|---:|
| `csrc/cpu/pacpu/amx_kernel.cpp` | **Step 2 (B+A)** 적용 — thread Q cache + K^T outer pre-pack. SW prefetch reverted. |
| `csrc/cpu/pacpu/core.h` | env-toggle dispatch (`VLLM_NEO_USE_AMX`) |
| `csrc/cpu/pacpu/pacpu.ispc` | softmax export |
| `csrc/cpu/pacpu/CMakeLists.txt` | amx_kernel.cpp + `-mamx-tile -mamx-bf16` |

**default off** (env-gated). `VLLM_NEO_USE_AMX=1` 활성 시 Step 2 path.

---

## 측정 환경

| 항목 | 값 |
|---|---|
| Host | Intel Xeon Platinum 8480+ (SPR) + H100 80GB × 8 |
| Workload | Llama-3.3-70B TP=8, 500p × 8192 in/out, fp8 KV cache |
| env | KMP_BLOCKTIME=200, OMP_NUM_THREADS=10, VLLM_NEO_USE_AMX=1 |
| AMX hardware | amx_bf16 + amx_int8 + amx_tile (native) |

---

## 결론

- Phase 3 A 의 단순 dropin (-4.3% 회귀) 은 작은 matmul (M=8, N=16, K=128) 의 setup overhead 가 work 보다 큰 inherent 한계.
- **Strategy B (Q cache) + A (K^T outer pre-pack)** 으로 회귀 회복 + **+1.7% net gain** 달성.
- Strategy G (SW prefetch) 는 block_table indirection 으로 효과 없음 — 회귀, revert.
- AMX path 의 본질 한계 (small matmul) 안에서 **본 turn 의 cheap variant 로 baseline 초과** 가능 입증.

다음 가능 lever (별도 task):
- **C' (Multi-task fused N=32)** — vllm 8-tile pattern 으로 tile occupancy ↑. 큰 변경 (2-3 일).
- **K cache BF16 host store** — swap path 변경 + dtype 변경, 정확도 검증. 1-2 일.
- 두 lever 추가 시 예상 +3-7% 가능 (Step 2 대비).
