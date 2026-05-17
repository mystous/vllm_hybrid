# H — 정적 분석 (SUB_015-Phase 1)

> 2026-05-17 KST. branch `feat/neo-amx-apply` HEAD `c698b971c`. read-only source 분석.
>
> 목적: pacpu kernel 의 FLOPs/bytes/AI 정량, OMP fork-join pattern, 정확도 영향도, 가속 후보 (AMX/AVX-512 / fast_exp / barrier 제거) 의 정량 backing.
>
> 보완 대상: D_bottleneck_table.md / E_amx_avx_applicability.md 의 stale 가정 (libgomp 8.26% / pacpu invisible). 본 분석의 fact 로 정정.

---

## 1. Llama-3.3-70B TP=8 hyperparam (`csrc/cpu/pacpu/dtype.h:33-44`)

| 매크로 | 값 | 비고 |
|---|---|---|
| NUM_LAYERS | 80 | per-call 은 `cur_layer` 1 layer 만 |
| NUM_Q_HEADS | 64/8 = **8** | TP=8 split |
| NUM_KV_HEADS | 8/8 = **1** | ★ TP=8 에서 1 KV head — KV fan-out 매우 작음 |
| QH_PER_KVH | 8/1 = 8 | GQA 8× factor |
| HEAD_DIM | 128 | constant |
| BLOCK_SIZE | 16 | constant |
| BLOCK_NELEM | 1×16×128 = 2,048 elem | K/V cache block size |

### dtype (`dtype.h:5-15`)

| 자료 | 형 | byte | 위치 |
|---|---|---|---|
| `data_t` | float16 (FP16) | 2 | q/k/v, k_cache/v_cache |
| `itmd_t` | float (FP32) | 4 | attn_score (qk 결과, softmax in/out) |
| `otpt_t` | float (FP32) | 4 | output buffer |

★ Input/cache = FP16, intermediate = FP32. AMX micro-GEMM (BF16 input, FP32 accumulator) 와 dtype 호환성:
- FP16→BF16 변환 cost (vCVTPH2PS + vCVTPS2BF16 = 2 round trip 또는 SPR 의 vCVTPH2BF16 부재).
- BF16 mantissa 7 bit vs FP16 mantissa 10 bit — 3 bit precision drop. softmax 정확도 영향 검증 필요.

---

## 2. qk_product FLOPs / bytes / AI (`pacpu.ispc:5-69`)

### Per sequence per layer (seq_len = N tokens, imax = ⌈N/16⌉)

#### 산술량
inner work per K_TILE (BLOCK_SIZE=16, K_TILE_WIDTH=2, NUM_KV_HEADS=1 의 TP=8):
- `foreach (l = 0 ... HEAD_DIM)` × QH_PER_KVH × K_TILE_WIDTH FMA
- = 128 × 8 × 2 = **2,048 FMA** = 4,096 FLOP / K_TILE

K_TILE 수 per block: BLOCK_SIZE / K_TILE_WIDTH = 16 / 2 = 8 K_TILE
→ per block FLOP = 8 × 4,096 = **32,768 FLOP** (NUM_KV_HEADS=1)

총 FLOP per seq per layer = **imax × 32,768**

#### 메모리 traffic
- q read: NUM_Q_HEADS × HEAD_DIM × 2 = 8 × 128 × 2 = 2,048 byte (전체 imax 동안 1 회 만, L1 hit)
- k read: imax × NUM_KV_HEADS × BLOCK_SIZE × HEAD_DIM × 2 = imax × 4,096 byte
- a write: imax × BLOCK_SIZE × NUM_Q_HEADS × 4 = imax × 512 byte

총 traffic = 2,048 + imax × 4,608 byte

#### AI (arithmetic intensity)
AI = (imax × 32,768) / (2,048 + imax × 4,608)
- large imax limit: 32,768 / 4,608 = **7.11 FLOP/byte**
- imax = 128 (seq_len ≈ 2048): 4,194,304 / 591,872 = **7.09 FLOP/byte**

→ AI **7.1 FLOP/byte**. 대부분 K cache stream 의 cost.

### Roofline (SPR per core)
- AVX-512 FP16 peak (16 lane × 2 FMA = 32 FLOP/cycle × 4 GHz) = **128 GFLOP/sec**
- L1 BW: ~200 GB/s/core
- L2 BW: ~50 GB/s/core
- DRAM BW: ~5 GB/s/core (worst-case shared L3 contention)

Roofline 한계:
- L1: AI 7.1 × 200 = 1,420 GFLOP/sec → compute-bound (peak 128 보다 빠름)
- L2: AI 7.1 × 50 = 355 GFLOP/sec → compute-bound
- DRAM: AI 7.1 × 5 = 35.5 GFLOP/sec → **memory-bound** (peak 의 27.7%)

→ K cache 가 L2 안에 들어가는지가 결정적.

### K cache working set
- 1 layer × 1 KV head × seq_len × HEAD_DIM × 2 byte = N × 256 byte/seq.
- seq_len 2,048 → 512 KB/seq. 1 worker thread 가 batch_size=4 seq 처리 → 2 MB working set. **SPR L2 = 2 MB/core 와 boundary**.
- batch_size=30 / ws=14 thread = 약 2.1 seq/thread → 1.1 MB/thread. L2 fit.

→ qk_product 의 실제 bottleneck = **L2 in-bounds 시 compute peak (128 GFLOP/sec) 의 일부**. AMX 적용 시 BF16 peak = 1,024 GFLOP/sec → 8× upside, 단 L2 BW (50 GB/s/core × 7.1 = 355 GFLOP/sec) 가 ceiling.

★ **AMX qk_product 실효 speedup ceiling ≈ 355 / 128 = 2.77×** (L2 BW limit). 32× 이론은 L1 fit 시.

---

## 3. softmax FLOPs / bytes (`pacpu.ispc:109-140`)

### Per seq per layer (seq_len = N, NUM_Q_HEADS = 8)

#### 산술량 (3 pass)
- Pass 1: scale + max — 1 mul + 1 cmp per (head, token) = N × 8 × 2 = **16N FLOP**
- Pass 2: exp + sum — 1 sub + 1 exp + 1 add per (head, token) = N × 8 × (2 + EXP) FLOP
- Pass 3: div + log — 1 div per (head, token) + 1 log per head = N × 8 × 1 + 8 × LOG FLOP

EXP cost (ISPC vectorized polynomial, AVX-512): ~25 cycle/element ≈ 25 FLOP equivalent
LOG cost: ~20 cycle/element ≈ 20 FLOP equivalent

총 ≈ N × 8 × (2 + 25 + 2 + 1) + 8 × 20 ≈ **240N + 160 FLOP** (transcendental dominant)
seq_len 2048 시 ≈ **491,680 FLOP / seq / layer**

#### 메모리 traffic
- a read+write: 3 pass × N × NUM_Q_HEADS × 4 byte = 3 × N × 32 = 96N byte
- asb/amb: NUM_Q_HEADS × 4 = 32 byte (L1 hit)

총 ≈ 96N byte. seq_len 2048 → 196,608 byte.

#### AI
AI = 491,680 / 196,608 = **2.5 FLOP/byte**

★ 매우 낮은 AI — DRAM-bound 처럼 보이지만 실제로는 exp/log latency 가 dominant (vectorized exp 의 throughput 은 SVML/SLEEF polynomial 이 issue).

### Softmax 의 진짜 bottleneck
- exp() / log() = AVX-512 polynomial approximation. 각 ~25 cycle latency × 16 lane SIMD = 25 cycle / 16 element ≈ 1.5 cycle/element compute.
- 그러나 polynomial = multiple FMA chain + table lookup. SVML 의 vCVTPS2PH 후 fast_exp_intrin 의 chained dependency 로 ILP 제한.

★ **fast_exp 재사용** (`csrc/cpu/cpu_arch_macros.h`) — AVX-512 intrinsic 직접 polynomial. ISPC `exp()` 대비 2-4× speedup 가능.

---

## 4. av_product FLOPs / bytes / AI (`pacpu.ispc:71-107`)

qk_product 와 동일 구조 (FMA 패턴). 차이:
- v_cache 가 K_TILE 없이 plain loop — `foreach (l = 0 ... HEAD_DIM)` + `for (h)` only.
- 결과 누적 (memset 후 `+=`).

### 산술량
inner: HEAD_DIM × QH_PER_KVH = 128 × 8 = 1,024 FMA = 2,048 FLOP / (block, head, t)
per block: 16 × 2,048 = **32,768 FLOP** (NUM_KV_HEADS=1)

총 FLOP per seq per layer = imax × 32,768 (qk 와 동일)

### 메모리 traffic
- a read: imax × BLOCK_SIZE × NUM_Q_HEADS × 4 = imax × 512 byte
- v read: imax × NUM_KV_HEADS × BLOCK_SIZE × HEAD_DIM × 2 = imax × 4,096 byte
- o read+write: NUM_Q_HEADS × HEAD_DIM × 4 = 4,096 byte (전체 imax 동안 1 회)

총 = 4,096 + imax × 4,608 byte → **AI ≈ 7.11 FLOP/byte** (qk 와 동일)

→ qk와 av 는 동일한 roofline 특성. AMX 적용 시 두 kernel 모두 동일 ceiling.

---

## 5. OMP fork-join pattern (`core.h:308-363`)

### 구조
```cpp
#pragma omp parallel        // [a] fork
{
  // Step 0: store_kv
  #pragma omp barrier       // [b] barrier #1
  // Step 1: attn_one_seq (qk + softmax + av)
  #pragma omp barrier       // [c] barrier #2
  // Step 2: gather_output
}                            // [d] join (implicit barrier)
```

### Synchronization 빈도

매 layer 의 `paged_attention_cpu` 호출마다:
- fork (a) + barrier (b) + barrier (c) + join (d) = **4 sync point / call**

### Call rate (per worker)
- step rate (실측): throughput 2,157 tps / batch_size ≈ 30 = **72 step/sec**
- layer 80 × 72 step/sec = **5,760 call/sec/worker**
- × 4 sync = **23,040 sync/sec/worker**
- × 8 worker = **184,320 sync/sec total**

### Persistent OMP team (Phase 3.1 적용됨, `core.h:244-249`)
```cpp
static thread_local bool _omp_persistent_init = false;
if (!_omp_persistent_init) {
  omp_set_dynamic(0);
  omp_set_max_active_levels(1);
  _omp_persistent_init = true;
}
```
- 효과: thread count 변동 회피, nested parallel 비활성.
- thread pool 자체는 `KMP_BLOCKTIME` 안에 spinning 유지 (`KMP_BLOCKTIME=200` ms 가 Phase 3.1+KMP=200 best).

### barrier 의 정량적 비용

barrier 의 cost = 가장 늦은 thread 가 도달할 때까지 wait. ws = 14 thread.
- thread 간 load imbalance → barrier wait 시간 비례 증가.
- Step 1 의 task partition (`thrd_rst_blks[]` line 269-272) 은 sequence 의 block 수 기반 — 매우 균등하나 seq_len 분포가 비균등 시 imbalance.
- Step 0 (store_kv) 와 Step 2 (gather_output) 는 `bch_blk_size = (batch_size-1)/ws + 1` 기반 균등 분배.

★ barrier wait 가 libgomp 의 hot path 가 되는 mechanism — Phase 1 동적 분석에서 확인.

---

## 6. Step 1 (attn_one_seq) 의 thread 분배

`tasks[]` vector (line 275):
- 각 task = (batch_id, seq_offs, seg_len, cum_seg_len) tuple.
- 매 sequence 를 block-level 로 분할하여 thread 별 일정한 block 수 배정.

```
batch_size=30, seq_len 분포 [1024, 2048, ...]
total_blocks = Σ ⌈seq_len/16⌉ 
ws=14 → 각 thread 가 ~total_blocks/14 block 처리
```

### 문제점 — sequence 단위 atomicity

`gather_output_one_seq` (Step 2, line 350-361) 는 한 sequence 의 모든 task 결과를 combine. 한 seq 의 multiple thread 결과를 모음.
- ★ Step 1 의 thread 별 task 분배가 seq 단위 atomic 이 아니어서 barrier #2 필요.
- 만약 한 seq = 한 thread 로 묶으면 barrier #2 제거 가능 (단 imbalance ↑).

### 잠재적 변형

**Option 1**: Step 1+2 를 통합 — 한 thread 가 한 seq 의 모든 block 처리 + 즉시 gather. barrier #2 제거.
- 단점: long seq 의 imbalance (max seq_len = 2048, min = 32 시 64× imbalance) → barrier #1 의 wait 가 커짐. Net 효과 미확정.

**Option 2**: `#pragma omp barrier` 제거 + sync 를 atomic memory order 로 대체.
- 위험: race 가능 (Step 2 가 Step 1 의 write 를 read).

**Option 3**: `omp_set_num_threads(N)` 줄이기 — barrier wait 의 worst-case wait time 감소 + cdec executor max_workers cap 안에서 14→8 변경 검토.
- KMP_BLOCKTIME sweep 결과 (Phase 3.2/3.1+KMP) 와 비교 입력.

---

## 7. K_TILE_WIDTH 분석 (`pacpu.ispc:4`)

```c
#define K_TILE_WIDTH 2
```

- AVX-512 zmm register file: 32 × 64 byte = 2,048 byte / core.
- K_TILE_WIDTH × QH_PER_KVH = 2 × 8 = 16 partial sum (per FP32 = 64 byte → 1 zmm) — register fit.
- K_TILE_WIDTH = 4 시 4 × 8 = 32 partial sum (2 zmm) — fit. 단 reduce_add 4 번 호출 (현재 2 번).
- K_TILE_WIDTH = 8 시 8 × 8 = 64 partial sum (4 zmm) — fit. reduce_add 8 번.

### 가능성

K_TILE_WIDTH ↑ → loop trip count ↓ (BLOCK_SIZE 16/W) + reduce_add 호출 빈도 ↑.
- W=2 → tmax/2 ≈ 8 iter × 1 reduce per tile = 8 reduce
- W=4 → tmax/4 ≈ 4 iter × 1 reduce per tile = 4 reduce
- W=8 → tmax/8 ≈ 2 iter × 1 reduce per tile = 2 reduce

**잠재 win**: reduce_add 의 cross-lane shuffle 이 critical path → W ↑ 시 reduce 횟수 ↓ → 약 5-15% qk_product speedup 가능. softmax / av 에는 영향 없음.

→ 단순 변경 / 정확도 영향 0 (수치 결과 동일) / build only 변경.

---

## 8. 기존 vllm AMX/AVX 자산 (재사용 후보)

### `csrc/cpu/cpu_arch_macros.h`
- `fast_exp` — AVX-512 polynomial expansion. softmax 대체 후보.
- `fast_div`, `fast_sqrt` 등 reciprocal 근사.

### `csrc/cpu/micro_gemm/cpu_micro_gemm_amx.hpp`
- AMX tile config (16×16×64 BF16 input → 16×16 FP32 output).
- `_tile_dpbf16ps` 사용 — 1 cycle 에 256 BF16 FMA = 512 FLOP.
- ★ pacpu 의 K_TILE=2 가 16×8 partial 인데 AMX tile=16×16 — head 단위 fan-out 필요.

### `csrc/cpu/cpu_attn_amx.hpp`
- prefill 용 large GEMM-based attention. decode 용 paged 와 별개 contract.
- Mask, scale, head 처리 패턴 다름 — 직접 재사용 불가, **참조 코드 only**.

### `csrc/cpu/quant_q8_0.cpp`
- AVX-512 VNNI 참조 (INT8 dot product). AMX INT8 path 와 유사 패턴.

---

## 9. 정적 분석 핵심 결론

### 9.1 pacpu kernel 비용 추정

per layer per seq (seq_len 2048, imax 128) FLOP 합계:
- qk: 128 × 32,768 = **4.19 MFLOP**
- softmax: ~491K = **0.49 MFLOP**
- av: 128 × 32,768 = **4.19 MFLOP**
- 합계: **8.87 MFLOP / seq / layer**

per worker: batch 30 seq × 80 layer × 8.87 MFLOP = **21.3 GFLOP / step**
worker 가 14 thread, peak 128 GFLOP/sec/core × 14 = 1,792 GFLOP/sec (이론)
→ 이상적 step time = 21.3 / 1,792 = **11.9 ms / step** (compute only, no sync)

실측: 1 / 72 step/sec = 13.9 ms / step → ★ **compute 한계의 85.6%** 도달 중. 추가 win 작음.

### 9.2 가속 가능 영역 (예상 win 정량)

| Lever | 대상 (% of total) | 변경 | 이론 speedup | Amdahl cap (effective gain) |
|---|---|---|---|---|
| **G** libgomp barrier | 43.75% | barrier 제거 / OMP_NUM_THREADS ↓ / sync 패턴 변경 | 2-5× (busy-wait ↓) | **5-15%** (cdec cap 효과) |
| **B** softmax fast_exp | 9.73% | ISPC exp() → AVX-512 polynomial intrinsic | 2-4× | **3-5%** |
| **A** AMX qk+av | 16.65% | BF16 micro-GEMM tile | 8× theoretical, **2.77× L2-bound** | **5-10%** |
| **C** K_TILE_WIDTH ↑ | qk only (8.75%) | constant 변경 | 1.05-1.15× | **1-2%** |

### 9.3 우선순위 (예상 effort/win)

| 순위 | Lever | Effort | Win | 비고 |
|---|---|---|---|---|
| **1** | G barrier wait | 중 (constant + sync rearrange) | 5-15% | 가장 cheap + 큰 영역 |
| **2** | B fast_exp | 중 (intrinsic 작성) | 3-5% | 정확도 검증 필요 |
| **3** | A AMX | 고 (BF16 변환 + 정확도 + build 복잡) | 5-10% | dev 검증 불가 (SPR-only) |
| **4** | C K_TILE_WIDTH | 저 (constant + rebuild) | 1-2% | quick win |

### 9.4 정정 사항

D_bottleneck_table.md 및 E_amx_avx_applicability.md 의 다음 stale fact:

1. "libpacpu symbol 0건 — pacpu 함수 flamegraph 안 보임" → **틀림**. perf 의 dso=libpacpu 에서 softmax 9.73% + qk_product 8.75% + av_product 7.90% **명시 관측**.
2. "libgomp.so.1 의 OMP pool 8.26%" → **틀림**. 실제 43.75%. py-spy flamegraph 가 native frame 정확 캡처 못함 (resolution issue).
3. "libgomp 의 source = ATen index_kernel" → **틀림**. ATen index_kernel 은 libtorch_cpu (4.85% + 2.12%) 에 별개로 잡힘. libgomp 의 hot path 는 pacpu 의 `ispc_attention_tasks` 의 `omp parallel` + barrier × 2 에서 호출.

→ D/E 문서의 lever ranking 재조정 필요. lever G (libgomp) 가 최우선.
