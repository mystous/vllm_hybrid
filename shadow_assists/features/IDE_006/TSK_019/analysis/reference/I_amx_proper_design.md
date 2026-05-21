# I — AMX 의 제대로 된 구현 분석 (Phase 3 A 의 -4.3% 회귀 의 진정 해결)

> 2026-05-18 KST. Phase 3 A 측정 결과 (3-run avg 2,142.5 tps, S1-S9 baseline 대비 -4.3% 회귀) 의 root + "제대로 된" AMX design 의 진정한 win path 분석.
>
> 본 분석 산출 = (1) 단순 dropin 의 inherent 한계 정량 (2) AMX hardware sweet spot vs NEO size mismatch (3) 제대로 된 design 의 7 가지 전략 + effort/win/risk

---

## 1. 단순 dropin 의 inherent 한계 (Phase 3 A 회귀 root)

### 1.1 NEO paged attention 의 size vs AMX hardware sweet spot

| 매개 | NEO (Llama-3.3-70B TP=8) | AMX hardware sweet spot | mismatch ratio |
|---|---:|---:|---:|
| **M** (heads) | **8** | 32 (4 tile × 16 row × 2 group) | **4× 낭비** |
| **N** (tokens / 1 block) | **16** | 32 (2 tile × 16 col) | **2× 낭비** |
| **K** (head_dim) | **128** | up to 64 BF16 per round (4 round for K=128) | round overhead |
| **tile occupancy** | M=8/16 = **50%** + 8 tile 中 3 만 사용 (37.5%) | M=32 + 8 tile 모두 사용 | **~19% effective** |

★ NEO 의 per-block matmul 이 AMX 의 **1 tile** 안에 들어가지만, tile 의 **절반만 valid** (M=8 < 16) + **8 tile 중 3 만 사용** (vllm AMX 의 standard 는 8 tile 모두 사용).

### 1.2 vllm 자체 AMX path 의 standard pattern (`csrc/cpu/cpu_attn_amx.hpp`)

```
TILE 0, 1: load A matrix      → M = up to 32 (2 group × 16 row)
TILE 2, 3: load B matrix      → N = up to 32 (2 group × 16 col)
TILE 4, 5, 6, 7: store C      → M × N = 32 × 32 quadrant per cycle
```

★ vllm 의 standard = **8 tile fully utilized**. 4 dpbf16ps 명령 1 cycle 으로 32×32 FP32 output 생성. NEO 의 dropin (3 tile, 1 dpbf16ps per round × 4 round) 와 6-8× 효율 차이.

### 1.3 Setup overhead breakdown (Phase 3 A 측정 backing)

NEO 의 per-block matmul (M=8, N=16, K=128) 의 AMX 실측 cost 분해 추정:

| Operation | Cycle 추정 | 빈도 (per block) | 총 cycle |
|---|---:|---:|---:|
| Q FP16→BF16 변환 (1024 elem AVX-512) | 50-100 | 1 | 50-100 |
| K^T pre-pack + FP16→BF16 (2048 elem × 4 round) | 200-400 | 1 | 200-400 |
| `_tile_loadd` × 2 (A, B) × 4 round | 30 × 8 = 240 | — | 240 |
| `_tile_dpbf16ps` × 4 round | 16 × 4 = 64 | — | 64 |
| `_tile_zero` + `_tile_stored` × 1 | 30 + 30 = 60 | — | 60 |
| C copy (16×16 FP32 → 8×16 actual) | 30 | — | 30 |
| **AMX total / block** | **~650-900** | | |
| **ISPC AVX-512 FP16 (baseline)** | **~400-500** | | |

→ AMX 의 setup overhead (Q+K 변환 + tile load) 가 **AMX work cycle (64) 의 4-8× 더 큼**. 작은 matmul 에서 setup-dominant 의 전형.

---

## 2. 제대로 된 AMX 구현 의 7 가지 전략

각 전략은 setup overhead 감소 / tile occupancy 향상 / data layout 통합 의 한 가지 axis 공략.

### Strategy A — K cache BF16 store (★ highest ROI)

**개념**: NEO host buffer 의 K/V cache 를 **FP16 대신 BF16 으로 store**. swap path 도 BF16 으로.

**현재**:
- GPU KV cache = FP16 (data_t = `_Float16`)
- swap-out 시 GPU FP16 → host FP16
- AMX dropin 매 block 마다 FP16 → BF16 변환 × 2048 elem × 4 round

**변경**:
- swap-out 시 **GPU FP16 → host BF16** (1 회 변환)
- AMX 호출 시 **변환 없이 host BF16 직접 read**

**Win 정량**:
- Per-block 의 K 변환 cost 200-400 cycle → **완전 제거**
- AMX total / block ~650-900 → **~450-700** (-30-50%)
- 새 ISPC AVX-512 baseline (~500 cycle) 와 동등 또는 약간 우수

**Effort**: 1-2 일
- `csrc/cpu/pacpu/dtype.h` 의 host-side data_t 만 BF16 으로 분기 (kvcache 만)
- `gpu_model_runner.py` 의 swap-out path 에 FP16→BF16 변환 추가 (또는 host buffer dtype change)
- AMX kernel 의 FP16→BF16 변환 부분 제거
- 정확도 검증 (TST_003 verdict)

**Risk**:
- BF16 mantissa 7 bit < FP16 mantissa 10 bit → 3 bit precision drop
- swap-out 변환 cost 가 NEO 의 async swap pipeline 안에서 amortize 되는지 검증 필요

**예상 net throughput win**: **+1~5%** (단순 변환 제거 — 다른 overhead 도 amortize 가능 시 +5-10%)

---

### Strategy B — Q hoist (per-attn_one_seq 1회)

**개념**: 현재 매 block 마다 Q FP16→BF16 변환 (NUM_Q_HEADS × HEAD_DIM × 2 = 8×128×2 = 2048 byte). 한 seq 의 imax block (8-128) 동안 같은 Q 사용 → outer 1 회로 hoist.

**Win 정량**:
- Per-call (per attn_one_seq) Q 변환 1 회 = 50-100 cycle (vs imax × 50-100)
- imax = 8 (seq_len 128) 시: 50 → 50 (효과 0)
- imax = 64 (seq_len 1024) 시: 3,200 → 50 (-98%)
- imax = 128 (seq_len 2048) 시: 6,400 → 50 (-99%)

**Effort**: 0.5 일
- `amx_kernel.cpp` 의 qk_amx 시그니처에 pre-converted Q_bf16 buffer 추가
- `core.h` 의 caller 에서 attn_one_seq 호출 전 Q 변환 (per-task once)
- thread_local Q_bf16 buffer (NUM_Q_HEADS × HEAD_DIM × 2 byte = 2 KB, L1 fit)

**Risk**: 0 (data 변환만, 순서 동일)

**예상 net throughput win**: **+0.5~3%** (long seq workload 에서 더 큼)

---

### Strategy C — Multi-seq batched (M expand)

**개념**: NEO 의 task partition 안에서 한 thread 가 multiple seq 의 Q × K^T 를 처리. 같은 K^T 가 아니므로 stack 못함 — 단 같은 cur_layer 의 다른 seq 의 Q 들을 **stack** 후 batched matmul.

**현재**: M=8 (한 seq 의 8 head), 1 tile 의 절반.
**변경**: M=8 × 2 seq = 16 (1 tile full), 또는 M=8 × 4 seq = 32 (vllm standard).

**문제**: K^T 가 seq 별 다름 → batched matmul 불가. **Q 는 stack 가능, K^T 는 seq 별 별개 호출 필요**. M=16 으로 두 seq 의 Q stack + 같은 K^T 로 matmul 불가.

**대안 변형**: 한 K block 을 multiple seq 의 Q 와 dot product — 그러나 NEO 의 K cache 가 seq 별 block_table 으로 indexed 라 batch 불가.

★ **Strategy C 실현 불가** — algorithmic 제약.

### Strategy C' — Multi-task fused (M expand within seq)

**대안 개념**: 한 seq 의 multiple block 을 stack. M=8 fixed 이지만 **N expand** — N=16 → N=32 (2 block 동시).

**변경**: 한 dpbf16ps 호출에서 2 block 의 K^T (32 token) 와 Q matmul → output [8, 32] FP32 = 1 tile.

**Win 정량**:
- 한 dpbf16ps 가 16 token → 32 token (2 block) 처리. imax/2 호출.
- Setup overhead /2 (tile_loadd / dpbf16 / stored 호출 빈도 ↓)
- per-block average cycle ~650 → ~350 (-46%)

**Effort**: 2-3 일
- K^T pre-pack layout 변경 (2 block 의 16 col × 2 byte = 64 byte = colsb)
- block boundary 처리 (seq_len 끝의 partial block padding)
- accuracy 검증

**Risk**: 중 — partial block padding 처리 정확도

**예상 net throughput win**: **+3-6%** (setup overhead 의 50% 감소)

---

### Strategy D — Tile config persistence (per-thread once)

**개념**: 현재 `ensure_amx_init` thread_local flag 으로 1 회 init. 단 ISPC 와 AMX 가 같은 thread 에서 동시 실행 시 tile state 충돌 가능. ISPC 가 zmm 만 사용 (tile 미사용) 으로 confirm 필요.

**검증**:
- ISPC `avx512spr-x16` target = ZMM register 만 사용, AMX tile 미사용 → tile state 보존 ✓
- `_tile_release()` 호출 시 tile state 해제. dropin 의 `_tile_release()` 없음 → tile config 유지 ✓

**Effort**: 0 (이미 구현됨, 검증만 필요)

**Win**: per-call setup 의 일부 제거. 이미 적용 중. 별도 win 없음.

---

### Strategy E — AMX softmax + av (full AMX path)

**개념**: 현재 AMX 는 qk 만, softmax/av 는 ISPC. softmax 도 AMX 으로 — exp polynomial 의 BF16 micro-gemm 형태 활용 불가능 (transcendental). **av_product 만 AMX 으로**.

**av_product = A[seq_len, NUM_Q_HEADS] × V[seq_len, HEAD_DIM]** matmul → o[NUM_Q_HEADS, HEAD_DIM].

NEO 의 av size:
- M = NUM_Q_HEADS = 8 (작음, qk 와 동일)
- N = HEAD_DIM = 128 (큼 — 4 tile × 32 col 분할 가능)
- K = seq_len (가변, 32-2048 per block dependent)

**Win 정량**: av 는 K=seq_len 가 가변. seq_len 128 → 16 round, seq_len 2048 → 64 round. Per-block 의 dpbf16ps 16-64회. AMX work cycle / K-round = 16 cycle. setup overhead 동일.
- AMX av total ~700-1500 cycle vs ISPC av ~600-1000 → 비슷, win 0~10%

**Effort**: 1-2 일 — qk_amx 와 유사한 구조

**Risk**: 정확도 (BF16) — qk 와 같음

**예상 net throughput win**: **+0-3%** (av 도 AMX 으로 변환해도 작은 size 한계 동일)

---

### Strategy F — Reordered Q × K^T batched across blocks within one seq

**개념**: 한 seq 의 N=imax blocks (각 16 token) 의 K^T 를 **하나의 큰 K** 으로 concatenate. M=8, N=imax×16, K=128. 한 attn_one_seq call 에서 1 회 batched matmul.

**현재**: imax 회 per-block matmul.
**변경**: 1 회 큰 matmul (Q × K^T_concat).

**Win 정량**:
- Setup overhead /imax — imax=8 시 8×↓, imax=128 시 128×↓
- per attn_one_seq cycle: imax × 650 → (imax × 64) + 100 setup = setup 줄임 + tile reuse

**Effort**: 3-5 일
- K^T pre-pack 의 큰 buffer (imax × 16 col × 32 BF16 = imax × 1024 byte). MAX_TOK_NUM=1048576 시 ~1 GB — heap alloc 필요. cache miss 위험.
- block_table indirection 의 sequential 처리 (NEO 의 paged 특성 유지)
- 정확도 (cumulative numerical drift)
- partial block 처리

**Risk**: 큼 — K^T concat buffer 의 memory bandwidth + cache miss

**예상 net throughput win**: **+5-15%** (setup amortize 큼, 그러나 memory pressure ↑)

---

### Strategy G — Persistent K block in tile (cross-block reuse)

**개념**: 인접 block (block_table 의 연속 entry) 의 K 가 같은 layer 의 KV cache 의 연속 위치 — **cache locality 활용**. tile reuse 안 됨 (K block 별 다른 token group) 단 L1/L2 cache 의 prefetch 효과.

**Win**: indirect — software prefetch + cache-friendly access. 정량화 어려움.

**Effort**: 1-2 일 — `__builtin_prefetch` 추가, K^T pre-pack 의 sequential pattern.

**예상 net throughput win**: **+1-3%**

---

## 3. 7 strategy 종합 ranking

| Strategy | 변경 영역 | Effort | Win 추정 | Risk | ROI (win/effort) |
|---|---|---:|---:|---|---:|
| **A** K cache BF16 store | data layout + dtype | 1-2 일 | **+1~5%** | 정확도 (3 bit drop) | **★★★ 최고** |
| **B** Q hoist | amx_kernel + core.h | 0.5 일 | +0.5~3% | 0 | **★★★ 매우 cheap** |
| **C'** Multi-task fused N=32 | K^T pre-pack layout | 2-3 일 | +3-6% | 중 (partial block) | ★★ |
| **F** Q × K^T full batched | K^T concat + memory pressure | 3-5 일 | +5-15% | 큼 (memory) | ★ (high risk) |
| **G** K prefetch | cache hint only | 1-2 일 | +1-3% | 0 | ★★ |
| **E** AMX av_product | qk_amx 와 유사 | 1-2 일 | +0~3% | 중 (정확도) | ★ |
| **C** Multi-seq batched | algorithmic 불가 | — | — | — | (불가) |

## 4. 진정한 win path — Strategy A + B 조합 (권고)

### 권고 sequence

**Step 1: Strategy B (Q hoist)** — 0.5 일, 정확도 risk 0, +0.5-3% win
- 가장 cheap + 정확도 안전
- amx_kernel.cpp 의 qk_amx signature 만 변경, core.h caller 에서 Q 변환 1회

**Step 2: Strategy A (K cache BF16 store)** — 1-2 일, +1-5% win
- B 적용 후 K 변환이 setup overhead 의 dominant
- swap-out path 에서 GPU→host 변환 시 BF16 store
- 정확도 검증 (TST_003 verdict)

**예상 net (A+B 적용 후)**: AMX path 가 baseline 과 **동등 또는 +1-3%** 가능. 단 여전히 작은 matmul size 한계로 큰 win 어려움.

### Step 3 (선택): Strategy C' (Multi-task fused N=32)

만약 Step 1+2 가 baseline 대비 +0% 또는 약간 win 시 — Strategy C' 적용으로 +3-6% 추가.
2-3 일 effort. 단 BF16 store + C' 의 효과 cumulative 가능.

**최대 가능 net win (A + B + C')**: **+5-10%** vs baseline (long workload 기준).

### 비추천 (high risk, low reliable win)

- **F (Q × K^T full batched)**: memory pressure 큼 + cache miss 위험. 단발 시도 가치 의문.
- **E (AMX av)**: av size 도 작아서 win 작음. 정확도 risk 있는데 win 1-3% — 비효율.

---

## 5. 본질적 한계 (Long-term view)

### NEO 의 size mismatch 는 fundamental

- AMX 의 sweet spot = M ≥ 32, N ≥ 32, K large.
- NEO 의 decode attention = M ≤ 8 (NUM_Q_HEADS / TP 8), N = 16 (BLOCK_SIZE), K = 128 (HEAD_DIM).
- Strategy A+B+C' 적용 후에도 M=8 fixed — tile occupancy 50% 이하.

### 진정한 큰 win path

본 매개 한계 안에서 큰 win 어렵.AMX 의 진정한 sweet spot 활용 =:
1. **TP 줄이기** (TP=8 → TP=4) — M = 16, occupancy 100%. 단 model parallelism 줄어들면 GPU 측 throughput 영향.
2. **prefill 만 AMX** — large GEMM (M = num_tokens 1000+) 에서 AMX 진정 활용. 단 vllm 자체 cpu_attn_amx 가 이미 prefill path 처리.
3. **Block size 줄이기** (BLOCK_SIZE 16 → 32) — N=32 tile full. 단 paging granularity 변경, NEO 의 block_table indirection 영향.

이상은 NEO/vllm core design 변경 영역 — 별도 long-haul work.

---

## 6. 결론

**단순 AMX dropin (Phase 3 A) 의 -4.3% 회귀는 fundamental size mismatch + setup overhead-dominant 의 inherent 결과**. 제대로 된 win 위해서는 **data layout 변경 (Strategy A) + setup amortize (Strategy B)** 가 필요.

### 권고 다음 task

1. **Strategy B (Q hoist) 즉시 적용** — 0.5 일, 정확도 risk 0
2. **Strategy A (K cache BF16 store) follow-up** — 1-2 일, 정확도 검증 동반
3. (선택) Strategy C' (Multi-task fused N=32) — 2-3 일

총 effort 3-5 일. 예상 net win +1-5% (단순 dropin -4.3% 의 회복 + 약간 win).

큰 win (>5%) 은 NEO/vllm core design 변경 (TP, BLOCK_SIZE 등) 영역 — 별도 long-haul work.
