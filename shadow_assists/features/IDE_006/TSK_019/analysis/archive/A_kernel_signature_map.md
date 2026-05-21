# Phase A — pacpu kernel signature map

> 분석 대상: `csrc/cpu/pacpu/pacpu.ispc` (NEO upstream 동일 구조)
> 산출물 유형: kernel 별 input shape / dtype / 메모리 access pattern

---

## 0. 데이터 타입 (dtype.h)

| 별칭 | C++ side | ISPC side | 비고 |
|---|---|---|---|
| `data_t` | `__fp16` (`_Float16` GCC extension) | `float16` | Q/K/V/cache 저장형 |
| `itmd_t` | `float` (FP32) | `float` | attention scores accumulator |
| `otpt_t` | `float` (FP32) | `float` | output accumulator |

→ **storage = FP16 / compute = FP32 accumulator**. dot product 안에서 FP16 × FP16 → FP32 accumulate.

---

## 1. Llama-3.3-70B (prod target, TP=8) 의 macros

| 매크로 | 값 (TP=8) |
|---|---:|
| `NUM_LAYERS` | 80 |
| `NUM_Q_HEADS` | 8 (=64/8) |
| `NUM_KV_HEADS` | 1 (=8/8) |
| `QH_PER_KVH` | 8 |
| `HEAD_DIM` | 128 (전 모델 공통) |
| `BLOCK_SIZE` | 16 (전 모델 공통) |
| `BLOCK_NELEM` | `NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM` = 1 × 16 × 128 = **2048 elem** = **4 KiB** (FP16) |

---

## 2. `qk_product` (Q · K^T) — pacpu.ispc:5-69

### Signature

```c
export void qk_product(
  const uniform int cur_layer,
  const uniform int num_blocks,
  const uniform int seq_len,
  const uniform data_t q[],          // [NUM_Q_HEADS, HEAD_DIM]
  const uniform data_t k_cache[],    // [..., NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
  const uniform int block_table[],   // [seq_len]
  uniform itmd_t a[]                 // [seq_len, NUM_KV_HEADS, HEAD_DIM]  ← 실제로는 [seq_len, NUM_KV_HEADS, QH_PER_KVH] = [seq_len, NUM_Q_HEADS]
);
```

### Input shape (Llama-70B TP=8)

- q: [8, 128] = 1 KiB
- k_cache 한 layer slice: [num_blocks, 1, 16, 128] (FP16) = per block 4 KiB
- block_table: [seq_len/16 ceil] int32
- a (output): [seq_len, 1, 8] = seq_len × 32 bytes

### Compute pattern (block 단위)

```
for each block i:                       # imax = ceil(seq_len/16)
  for each KV head j (1 head only):
    for tile t in [0..BLOCK_SIZE-1] (K_TILE_WIDTH=2 unrolled):
      foreach l in [0..HEAD_DIM-1] (= 128):  ← SIMD 영역
        for each h in [0..QH_PER_KVH-1] (=8):
          sum[h][g] += q[q_off + h*128 + l] * k[k_off + g*128 + l]
      a[t] = reduce_add(sum)
```

### FLOPs / call

- inner: 1 mul + 1 add per (h, g, l) = 128 (HEAD_DIM) × 8 (QH_PER_KVH) × 2 (K_TILE_WIDTH) × 2 (FMA) = **4,096 FLOPs / inner tile**
- per token: 4,096 × ceil(BLOCK_SIZE/K_TILE_WIDTH) = 4,096 × 8 = **32,768 FLOPs / token**
- per seq (seq_len tokens): 32,768 × seq_len FLOPs

### Bytes / call

- q reused (broadcast)
- k_cache read: seq_len × 128 × 2 bytes (FP16) = **256 × seq_len bytes**
- a write: seq_len × 32 bytes
- 거의 K read 가 dominant

### Arithmetic Intensity

- 32,768 FLOPs / (256 + 32) bytes ≈ **114 FLOPs/byte** per token
- 그러나 q broadcast 와 reuse 고려 시 effective AI ≈ ~30-50 FLOPs/byte (memory-bound 영역)

### SIMD 활용

- `foreach (l = 0 ... HEAD_DIM)` — HEAD_DIM=128 / lane=16 → 8 iter, 16-way SIMD 완전 활용
- `reduce_add(sum[h][g])` — 8-element reduction, ISPC 가 horizontal add 적용
- 현재 `avx512spr-x16` target → `vfmadd231ps` + `vextractf32x8` 패턴 추정

---

## 3. `av_product` (attention_weights · V) — pacpu.ispc:71-107

### Signature

```c
export void av_product(
  const uniform int cur_layer, num_blocks, seq_len,
  const uniform itmd_t a[],          // [seq_len, NUM_KV_HEADS, QH_PER_KVH]
  const uniform data_t v_cache[],    // [..., NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
  const uniform int block_table[],   // [seq_len]
  uniform otpt_t o[]                 // [NUM_Q_HEADS, HEAD_DIM]
);
```

### Compute pattern

```
memset(o, 0, NUM_Q_HEADS * HEAD_DIM * 4)   # FP32 output zero
for each block i:
  for each KV head j:
    for t in [0..tmax-1]:
      foreach l in [0..HEAD_DIM-1] (=128):     ← SIMD
        for h in [0..QH_PER_KVH-1] (=8):
          o[h*128 + l] += v[l] * a[h]
```

### FLOPs / call

- per token: 2 (FMA) × 128 × 8 = **2,048 FLOPs / token**
- per seq: 2,048 × seq_len

### Bytes / call

- v_cache read: 같은 V block (seq_len × 256 bytes)
- a read: seq_len × 32 bytes
- o read+write (FP32 accumulator): 4 KiB total (constant per call)

### AI

- 2,048 / (256 + 32) ≈ **7 FLOPs/byte** per token — **lower than qk_product** (V 가 한 번만 사용됨, no reuse)
- memory-bandwidth bound

### SIMD 활용

- `foreach (l = 0 ... HEAD_DIM)` — 동일 16-way 완전 활용
- inner `for h in QH_PER_KVH` 가 scalar — 8 multiplications scalar (per token) 잔존

---

## 4. `softmax` — pacpu.ispc:109-140

### Signature

```c
void softmax(
  const uniform int seq_len,
  const uniform itmd_t softmax_scale,
  uniform itmd_t a[],     // [seq_len, NUM_Q_HEADS] - modified in-place
  uniform itmd_t asb[]    // [NUM_Q_HEADS] - LSE (?)
);
```

### Compute pattern (3-pass)

1. max reduction (per head)
2. exp((x - max) * scale)
3. sum 후 div

### FLOPs / token

- 1 exp + 1 sub + 1 mul + 1 div = ~4 ops/token (exp 가 expensive)
- per call: ~4 × seq_len × NUM_Q_HEADS = 4 × seq_len × 8 = **32 × seq_len ops**

### AI

- 매우 낮음 (read seq_len × NUM_Q_HEADS 4-byte = 32 × seq_len bytes, write 동일)
- AI ≈ 0.5 FLOPs/byte (단, exp 가 multi-cycle op)

---

## 5. `attn_one_seq` — pacpu.ispc:142-160

3 kernel 의 단순 sequential composition:
```
qk_product → softmax → av_product
```

ISPC level 에서 task launch 영역 — host 의 `core.h` 에서 OMP parallel 로 attn_one_seq 를 seq 별로 분산.

---

## 6. 메모리 footprint 종합 (Llama-70B TP=8, seq_len=8192 기준)

| 자료 | 크기 (1 seq) |
|---|---|
| K cache slice (per layer) | 8192/16 × 4 KiB = **2 MiB** |
| V cache slice (per layer) | **2 MiB** |
| a (attention scores) | 8192 × 32 = **256 KiB** |
| o (output) | 4 KiB |
| q | 1 KiB |

→ per-call working set ≈ **4 MiB** (K + V dominant). L2 ~ 2 MiB, L3 ~ 100 MiB per CPU. **L3 fit but not L2** — main memory bandwidth 가 bottleneck.

---

## 7. AMX 적용 가능성 — 1차 hint

| kernel | 주 연산 | tile dim 적합 | dtype 호환 | comment |
|---|---|:-:|:-:|---|
| `qk_product` | Q · K^T (= GEMM with M=NUM_Q_HEADS=8, N=BLOCK_SIZE=16, K=HEAD_DIM=128) | ⚠️ M=8 → AMX tile rows 16 의 절반 | FP16 ↛ AMX FP16 미지원, BF16 변환 필요 | tile fit 일부 / dtype 변환 cost |
| `av_product` | A · V (= GEMM with M=NUM_Q_HEADS=8, N=HEAD_DIM=128, K=seq_len) | ✅ K dim 큼 | FP16 ↛ AMX 동일 cost | tile fit 좋음 |
| `softmax` | exp + reduce | ❌ (GEMM 아님) | N/A | AMX 부적합 |

→ 자세한 적용성 판정은 Phase D-E 에서. 본 phase 는 facts only.

---

## Cross-ref

- 본 문서 산출치 (FLOPs / bytes / AI) 는 Phase D 의 roofline 분석 input
- dtype FP16 ↔ AMX BF16 변환 비용은 Phase C 의 기존 BF16 경로 인용 가능
