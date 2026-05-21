# Phase C — pacpu vs cpu_attn_amx contract diff

> 분석 시각: KST 2026-05-15 ~
> 산출 목적: 두 경로의 입력/출력 contract 차이 → 재사용 시 필요한 adapter 정의

---

## C.10 두 경로의 위치 + 호출 entry

| | NEO pacpu | vllm native CPU attn |
|---|---|---|
| 폴더 | `csrc/cpu/pacpu/` | `csrc/cpu/` |
| 빌드 산출 | `libpacpu-llama3_3_70b-tp8.so` (별도 .so) | `vllm/_C.abi3.so` 안에 합쳐짐 |
| Python entry | `torch.ops.pacpu.paged_attention_cpu(...)` | `torch.ops.vllm.cpu_attention_with_kv_cache(...)` |
| 호출 site | `vllm/model_executor/layers/attention/attention.py:1014` (cdec_future.submit) | (vllm CPU 전용 backend 활성 시) |

→ **서로 완전히 분리**. NEO 의 cdec dispatch 가 vllm CPU attention 영역 거치지 않음.

---

## C.11 입력 contract diff

### NEO pacpu signature (`csrc/cpu/pacpu/pacpu.cpp:78-100`)

```cpp
void paged_attention_cpu(
  int64_t cur_layer,                              // layer index (per-layer call)
  double softmax_scale,                            // 1/sqrt(head_dim)
  const std::vector<int64_t>& seq_ids,             // 어느 req
  const std::vector<int64_t>& seq_lengths,         // 각 req 의 seq_len
  at::Tensor q,                                    // [num_seq, NUM_Q_HEADS, HEAD_DIM]
  at::Tensor k,                                    // [num_seq, NUM_KV_HEADS, HEAD_DIM]  ← new K
  at::Tensor v,                                    // [num_seq, NUM_KV_HEADS, HEAD_DIM]  ← new V
  at::Tensor k_cache,                              // [..., NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
  at::Tensor v_cache,                              // same layout
  at::Tensor block_table,                          // [num_seq, max_blocks]
  at::Tensor o                                     // [num_seq, NUM_Q_HEADS, HEAD_DIM]  ← output
)
```

특징:
- **per-layer 호출** (cur_layer arg)
- block_table = NEO 자체 형식 (BLOCK_SIZE=16 hardcoded)
- dtype = FP16 (k_cache/v_cache `at::Half`)
- output 직접 write 가 아니라 attention 결과를 GPU 가 받아 LSE merge 안 함 (NEO 는 exclusive request → merge 불필요)

### vllm cpu_attention_with_kv_cache signature (`csrc/cpu/torch_bindings.cpp:390`)

```cpp
m.def("cpu_attention_with_kv_cache(Tensor query, Tensor key_cache, Tensor "
      "value_cache, Tensor scheduler_metadata, ...) -> Tensor");
```

특징:
- **모든 layer 한 번에** (?) — query 를 어떻게 처리하는지 확인 필요
- vllm 의 block_table 형식 (different layout)
- dtype = BF16 / FP16 / INT8 (template)
- `scheduler_metadata` 가 별도 — `get_scheduler_metadata()` 가 미리 계산

→ **두 API 가 fundamentally 다른 추상화 단계**. NEO pacpu 는 per-layer + ad-hoc seq lengths. vllm CPU attn 은 scheduler-driven + KV cache layout 의존.

---

## C.12 dtype / layout 차이

| 항목 | NEO pacpu | vllm cpu_attn |
|---|---|---|
| K/V dtype | FP16 (`at::Half`) | template (BF16/FP16/INT8) |
| K/V layout | `[..., L, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]` | vllm 표준 layout (HND or similar) |
| BLOCK_SIZE | 16 (constant) | runtime config |
| HEAD_DIM | 128 (constant) | runtime config |
| Q 입력 | [num_seq, NUM_Q_HEADS, HEAD_DIM] | scheduler_metadata 가 prepare |
| accumulator | FP32 (`itmd_t = float`) | BF16 / FP32 (varies) |
| softmax | inline (3-pass within attn_one_seq) | 별도 op? (확인 필요) |

→ NEO pacpu 가 더 **rigid + closed** (모델별 빌드, BLOCK_SIZE/HEAD_DIM 컴파일 시점 고정). vllm cpu_attn 이 더 **generic** (runtime dispatch + template).

---

## C.13 mask / softmax / scale 처리 위치

| 처리 | NEO pacpu | vllm cpu_attn |
|---|---|---|
| softmax_scale | 사용자 인자 (`double`) | scheduler_metadata 안 |
| mask (causal) | implicit (cdec 는 last token query → 항상 causal) | scheduler_metadata `casual` bool |
| sliding window | 미지원 | 지원 (`window_size` arg) |
| GQA broadcast | inline (loop 안 `QH_PER_KVH`) | template specialization |

→ NEO 는 cdec 의 **incremental decode 특화** (생성 중 1 token query 의 attention). vllm cpu_attn 은 **generic** (prefill 도 가능).

---

## C.14 head 처리 차이

| | NEO pacpu | vllm cpu_attn |
|---|---|---|
| GQA Q heads | NUM_Q_HEADS=8 (Llama-70B TP=8) | runtime `num_heads_q` |
| KV heads | NUM_KV_HEADS=1 (Llama-70B TP=8) | runtime `num_heads_kv` |
| Q broadcast | inline loop (8 Q per KV) | template |
| TP 처리 | 컴파일 시점 TP=8 fixed | runtime |

→ NEO 가 모델/TP 별 **separate .so** 빌드 (`libpacpu-llama3_3_70b-tp8.so`) 인 이유. 각 모델/TP 조합마다 macro 가 다름. vllm cpu_attn 은 단일 .so 다 처리.

---

## C.15 재사용 시 필요한 adapter

NEO pacpu 자리에 vllm cpu_attn 의 AMX kernel 을 그대로 호출하려면:

1. **dtype 변환**: K/V FP16 → BF16 (storage 변환 + AMX 가 BF16 native)
2. **layout 변환**: `[L, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]` → vllm 표준 layout (확인 필요)
3. **scheduler_metadata 구성**: NEO 는 `seq_lengths` 직접, vllm 은 `query_start_loc` 등 별도 계산
4. **per-layer → batched layer 변환**: NEO 는 매 layer 별 call, vllm 은 한 번에 가능?
5. **output write 형식 변환**: NEO 는 `[num_seq, NUM_Q_HEADS, HEAD_DIM]` 출력, vllm 은 (확인 필요)

→ 전체 재포장 cost 큼. **부분 재사용 (예: micro_gemm/cpu_micro_gemm_amx.hpp 의 TileGemm224 만 가져다 NEO pacpu kernel 의 GEMM 부분에 끼워 넣기)** 가 합리적.

---

## C.16 결론 — gap analysis

| 재사용 영역 | gap 크기 | 권장 |
|---|---|---|
| vllm `cpu_attention_with_kv_cache` 통째 | 매우 큼 (3-5 단계 adapter) | ❌ 비추천 |
| **`cpu_attn_amx.hpp` TileGemm224** | 큼 (layout + dtype) | △ 신중 |
| **`micro_gemm/cpu_micro_gemm_amx.hpp` TileGemm224** | 보통 (BF16 변환만) | ◎ **권장** |
| `dnnl_kernels.cpp` onednn_mm | 큼 (layout 통째 변환) | △ overhead 측정 후 |
| `cpu_arch_macros.h` fast_exp | 작음 (just include + call) | ◎ **권장** |

→ NEO pacpu kernel 의 **AMX/AVX 가속 통합 전략**:
1. **fast_exp (cpu_arch_macros.h)** 를 ISPC kernel 의 softmax 영역에 inject (작은 PR)
2. **TileGemm224 (micro_gemm)** 를 ISPC kernel 의 qk_product / av_product GEMM 부분에 inject (중간 PR)
3. **dnnl onednn_mm** 은 큰 변환 cost — 측정 후 결정 (큰 PR)
4. **vllm `cpu_attention_with_kv_cache`** 통째 교체는 NEO pacpu 의 의미 (per-layer cdec dispatch) 상실 — 비추천

다음 Phase D 에서 flamegraph 재분석으로 위 후보 각각의 실효 영향 정량화.
