# SUB_039 — av_product AMX 확장 plan (av_amx) — refined v2 (turn N+2)

## 0. Refinement (turn N+2)

SUB_042 (prefill/decode 분리 측정) 가 background 로 ~2-3hr 진행 중. 본 turn 에 SUB_039 의 **plan 정제 + 코드 skeleton 작성** + 명시적 risk register 보강. 실제 av_amx 적재 + 측정 + 정확도 verify 는 SUB_042 결과 본 후 별도 turn (사용자 estimate "2일").



> **parent**: TSK_019 / O 분석 §7.2 — step1 (ISPC compute) 70% wall dominant 의 직접 가속
> **출처**: 사용자 명시 (turn 22) — "3개 전부 다 진행"
> **선행**: SUB_036 (Path A 500p baseline) 결과 — 500p 워크로드 에서 lever signal 검출 후 진입.

---

## 1. 배경

### 1.1 현 AMX path 의 한계

| Kernel | 현 구현 | 위치 |
|---|---|---|
| qk_product | **AMX BF16** (`tile_dpbf16ps`) | `amx_kernel.cpp::qk_amx*` |
| softmax | ISPC FP32 (3-pass + online via SUB_033 env-gated) | `pacpu.ispc::softmax` |
| **av_product** | **ISPC FP32** (broadcast-vec product) | `pacpu.ispc::av_product` |

SUB_035 C1a 결과: step1 (ISPC compute) = 70% wall — 이 중 **av_product 가 30-40%** (softmax 가 10-20%, qk 는 AMX 라 빠름).
→ av_product 의 AMX 변환이 step1 의 가장 큰 직접 lever.

### 1.2 av_product 의 구조 (pacpu.ispc:71-110)

```c
// 의사 코드 — 1 task, 1 seq segment
for (i in 0..imax):  // KV blocks
  for (j in 0..NUM_KV_HEADS):  // KV heads (j × QH_PER_KVH = q heads)
    for (t in 0..tmax):  // tokens in block (≤BLOCK_SIZE=16)
      for (l in 0..HEAD_DIM):  // SIMD vectorize via foreach
        for (h in 0..QH_PER_KVH):  // q heads per kv head
          o[j × QH_PER_KVH × HEAD_DIM + h × HEAD_DIM + l]
            += V[i × ... + l] × a[t × NUM_Q_HEADS + j × QH_PER_KVH + h]
```

**핵심 형상**:
- A (attention weight) = `[seq_len_segment, NUM_Q_HEADS=64]` FP32
- V (value cache) = `[seq_len_segment, NUM_KV_HEADS=8, HEAD_DIM=128]` FP16
- O (output) = `[NUM_Q_HEADS=64, HEAD_DIM=128]` FP32 accumulate

### 1.3 AMX matmul 형태로 변환

A^T @ V (per kv_head, transpose 후):
```
A^T[QH_PER_KVH=8, seq_len_segment] @ V[seq_len_segment, HEAD_DIM=128]
  = O[QH_PER_KVH=8, HEAD_DIM=128]
```
이건 GEMM (M=8, K=seq_len, N=128). AMX tile (M=16, K=32 BF16, N=16) 으로 분할:
- M tile (8 rows): 1 tile (사용 8/16, padding)
- N tile (128 cols): 8 tile (16 cols each)
- K tile (seq_len BF16): ceil(seq_len/32) tile

→ **AMX 형태로 변환 가능**. 단 (1) A FP32 → BF16 cast 필요, (2) V FP16 → BF16 cast 필요.

## 2. 변경안

### 2.1 새 함수 av_amx 추가 (`amx_kernel.cpp`)

```cpp
extern "C" void av_amx(
    int cur_layer, int num_blocks, int seq_len,
    const float* a,          // FP32 [seq_len, NUM_Q_HEADS]
    const _Float16* v_cache, // FP16
    const int* block_table,
    float* o                 // FP32 [NUM_Q_HEADS, HEAD_DIM]
) {
    // Step 1: A FP32 → BF16 cast (per kv_head slice)
    // Step 2: V FP16 → BF16 cast (per block)
    // Step 3: AMX matmul A^T @ V → O (FP32 accumulate)
}
```

### 2.2 dispatcher 변경 (`amx_kernel.cpp::attn_one_seq_amx*`)

```cpp
qk_amx(...);
_softmax_dispatch()(...);
// SUB_039 — env-gated av_amx
const char* _av_env = std::getenv("VLLM_NEO_AV_AMX");
if (_av_env && _av_env[0] && _av_env[0] != '0') {
  av_amx(cur_layer, num_blocks, seq_len, a, v_cache, block_table, o);
} else {
  ispc::av_product(cur_layer, num_blocks, seq_len, a, v_cache, block_table, o);
}
```

### 2.3 정확도 검증

- av_product 의 FP32 accumulate vs av_amx 의 BF16 matmul → FP32 → relative error 가 게이트 안인지
- TST_003 의 base prompts 사용

## 3. 적재 step

| Step | 작업 | site | effort |
|---|---|---|:-:|
| 3.1 | av_amx 신규 함수 작성 (M=8, K=seq, N=128 AMX matmul) | `amx_kernel.cpp` | 4-6 hr |
| 3.2 | tile config (TileConfig) + A BF16 cast cache + V BF16 cast cache | `amx_kernel.cpp` | 2-3 hr |
| 3.3 | env-gated dispatcher | `amx_kernel.cpp::attn_one_seq_amx*` | 30 min |
| 3.4 | rebuild | build | 5 min |
| 3.5 | 정확도 verify (BF16 cast accuracy) | eval | 1-2 hr |
| 3.6 | 500p × 8192 측정 (env-OFF / env-ON / env-OFF) 3-way | eval | 1.5 hr |
| 3.7 | winner 3-run avg | eval | 4.5 hr |

**총 effort**: **2 일** (Step 3.1-3.3 = 7-10 hr 코드 + 3.4-3.7 = 측정/verify)

## 4. 위험

| risk | mitigation |
|---|---|
| BF16 cast 비용 > AMX 가속 win | A/V cast 를 thread_local cache 로 1회만 (qk_amx 의 Q BF16 cache 패턴) |
| QH_PER_KVH=8 의 M tile padding 부족 | M=16 tile 의 절반만 사용, padding 0 — AMX 효율 50% |
| 정확도 게이트 깨짐 (BF16 errors) | FP32 accumulator 유지 + per-token rel error 측정 후 threshold |
| seg_len 가짧을 때 (decode) AMX overhead | SUB_037 의 ARI dispatch 와 결합 — seq_len ≥ threshold 시만 av_amx |

## 5. SUB_036 결과 의존성

| SUB_036 결과 | SUB_039 진입 결정 |
|---|---|
| NEO 500p 가 noise 위 (lever signal 가능) | SUB_039 정식 진입 — 2 일 effort 정당화 |
| NEO 500p 가 noise 안 (saturated) | SUB_039 후순위 — 워크로드 자체가 throughput 가속 가치 없음 |
| vanilla 500p >> NEO 500p | NEO 워크로드 재정의 (vanilla OOM 영역) 후 SUB_039 |

## 6. SUB_037 vs SUB_039 sequencing

- SUB_037 (B4 SPARAMX) = 30 min 코드 + 2.5 hr 측정 — **저비용 high-signal**
- SUB_039 (av_amx) = 2 일 — **고비용 high-impact**
- 권장 순서: SUB_037 → SUB_039
- SUB_037 ARI dispatch 가 ≥3% win → SUB_039 도 win 가능성 증가
- SUB_037 win 0% → SUB_039 의 step1 70% 가 의미 — 더 큰 effort 정당화
