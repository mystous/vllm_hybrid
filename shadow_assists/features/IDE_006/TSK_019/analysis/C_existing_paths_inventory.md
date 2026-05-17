# Phase C — vllm_hybrid 기존 AMX/AVX 경로 inventory

> 분석 시각: KST 2026-05-15 ~
> 산출 목적: 우리 repo 의 기존 AMX/AVX 경로 nature + 재사용 가능성

---

## C.1 vllm CPU 영역 (`csrc/cpu/`) 전체 file inventory

| file | lines | role | ISA-related |
|---|---:|---|---|
| `cpu_attn.cpp` | 189 | CPU attention entry + ISA dispatch | dispatch hub |
| `cpu_attn_impl.hpp` | 1,970 | impl main + `ISA enum` + switch | dispatch hub |
| **`cpu_attn_amx.hpp`** | **511** | **AMX BF16 attention kernel** | **AMX** |
| `cpu_attn_vec.hpp` | 248 | generic vector kernel | AVX |
| `cpu_attn_vec16.hpp` | 171 | VEC16 path | AVX-512 16-lane |
| `cpu_attn_neon.hpp` | 401 | ARM Neon path | NEON |
| `cpu_attn_neon_bfmmla.hpp` | 682 | ARM Neon BF16 matmul | NEON BF16 |
| `cpu_types_x86.hpp` | 802 | x86 vector type abstraction | AVX/AMX 영역 |
| `cpu_types_arm.hpp` | 925 | ARM type abstraction | NEON |
| `cpu_types_vxe.hpp` | 1,188 | s390x VXE type abstraction | VXE |
| `cpu_arch_macros.h` | 113 | `fast_exp` 등 macro | AVX-512 |
| `cpu_wna16.cpp` | 402 | W4A16 etc weight-only quant | AVX-512 |
| **`cpu_fused_moe.cpp`** | **776** | MoE kernel | AMX/AVX 가능 |
| **`dnnl_kernels.cpp`** | **570** | oneDNN integration | AMX (oneDNN) |
| `dnnl_helper.h` | 220 | oneDNN dispatch helper | AMX (oneDNN) |
| `mla_decode.cpp` | 390 | MLA decode | (vector) |
| `activation_lut_bf16.cpp` | 71 | BF16 LUT activation | AVX-512 |
| `spec_decode_utils.cpp` | 409 | speculative decode | (vector) |
| `micro_gemm/cpu_micro_gemm_amx.hpp` | (별도) | micro GEMM AMX | **AMX** |
| `micro_gemm/cpu_micro_gemm_vec.hpp` | (별도) | micro GEMM vector | AVX |
| `micro_gemm/cpu_micro_gemm_impl.hpp` | (별도) | micro GEMM dispatch | dispatch |
| `pacpu/` (별도 subdir) | 982 (5 files) | **NEO cherry-pick** | ISPC AVX-512 |

→ vllm CPU 영역 **3가지 path 공존**:
1. **vllm native CPU attention** (`cpu_attn_*.hpp`) — AMX/VEC/VEC16/NEON/VXE dispatch + ISA enum
2. **vllm oneDNN integration** (`dnnl_kernels.cpp` + `dnnl_helper.h`) — Intel oneDNN library
3. **NEO pacpu cherry-pick** (`pacpu/`) — ISPC kernel, **vllm 다른 경로와 분리**

---

## C.2 vllm native CPU attention — ISA enum + dispatch

### dispatch entry (`cpu_attn.cpp:1-50`)

```cpp
torch::Tensor get_scheduler_metadata(
    ...
    const std::string& isa_hint,    // ← caller 에서 "amx" / "vec" / "vec16" / "neon" / "vxe" string 전달
    ...
) {
  cpu_attention::ISA isa;
  if (isa_hint == "amx") {
    isa = cpu_attention::ISA::AMX;
  } else if (isa_hint == "vec") {
    isa = cpu_attention::ISA::VEC;
  } else if (isa_hint == "vec16") {
    isa = cpu_attention::ISA::VEC16;
  } ...
}
```

### ISA enum (`cpu_attn_impl.hpp:15`)

```cpp
enum class ISA { AMX, VEC, VEC16, NEON, VXE };
```

### dispatch switch (`cpu_attn_impl.hpp:137-150`)

```cpp
switch (isa) {
  case ISA::AMX:
    // 호출 cpu_attn_amx.hpp 의 kernel
  case ISA::VEC:
    // 호출 cpu_attn_vec.hpp
  case ISA::VEC16:
    // 호출 cpu_attn_vec16.hpp
  case ISA::NEON:
    // 호출 cpu_attn_neon.hpp
  case ISA::VXE:
    // 호출 cpu_attn_vxe.hpp (있다면)
}
```

### torch bindings (`torch_bindings.cpp:99-395`)

```cpp
m.def("cpu_attn_reshape_and_cache(...)", &cpu_attn_reshape_and_cache);
m.def("cpu_attention_with_kv_cache(...)", &cpu_attention_with_kv_cache);
```

→ vllm 에서 정식 등록된 ops 이름: `cpu_attn_reshape_and_cache`, `cpu_attention_with_kv_cache`. **NEO 의 `pacpu.paged_attention_cpu` 와 분리된 namespace**.

---

## C.3 cpu_attn_amx.hpp 의 AMX 구현 detail

### tile config (`cpu_attn_amx.hpp:8-19`)

```cpp
constexpr static int64_t AMX_TILE_ROW_BYTES = 64;
constexpr static int64_t AMX_TILE_ROW_NUM = 16;
constexpr static int64_t AMX_TILE_BYTES = 64 * 16 = 1024;

typedef struct __tile_config {
  uint8_t palette_id = 1;
  uint8_t start_row = 0;
  uint8_t reserved_0[14] = {0};
  uint16_t colsb[16] = {0};    // columns per tile (bytes)
  uint8_t rows[16] = {0};      // rows per tile
} __tilecfg;
```

### tile register 할당 (line 22-24)

```cpp
// TILE 0, 1: load A matrix, row num should be 16, m - 16
// TILE 2, 3: load B matrix, row num should be 16
// TILE 4, 5, 6, 7: store results C matrix, row num should be 16, 16, m - 16, m - 16
```

→ AMX-TMUL 8 tile registers 의 표준 사용 패턴 (2 A + 2 B + 4 C). M-K-N 의 K dimension 을 outer loop 로.

### gemm template (line 26-50)

```cpp
template <typename kv_cache_t>
class TileGemm224 {
 public:
  FORCE_INLINE static void gemm(
    const int32_t m_size,
    void* __restrict__ a_tile,
    void* __restrict__ b_tile,
    float* __restrict__ c_tile,
    const int64_t lda,
    ...
  );
  FORCE_INLINE static void init_tile_config(int32_t m, __tilecfg& config);
};
```

**dtype**: `kv_cache_t` template param — BF16 / FP16 / INT8 셋 다 가능 (Intel AMX-TMUL spec).

---

## C.4 micro_gemm/cpu_micro_gemm_amx.hpp

`cpu_attn_amx.hpp` 와 유사 구조 (`TileGemm224` 동일 이름). 별도 namespace 인데 micro-GEMM 영역에서 재사용 가능 형태. MoE / linear projection 등 다른 영역에서도 활용 가능.

---

## C.5 oneDNN 통합 (`dnnl_kernels.cpp`)

| 함수 | role |
|---|---|
| `onednn_scaled_mm` (line 354) | scaled GEMM (INT8 → BF16/FP32) |
| `onednn_mm` (line 519) | generic matmul (BF16) |
| `static_scaled_int8_quant_impl` (line 36) | int8 quant |
| `dynamic_scaled_int8_quant_impl` (line 92) | int8 quant (dynamic scale) |
| `dynamic_quant_epilogue` (line 195) | quant epilogue |

→ Intel oneDNN library 직접 호출. **AMX 가 oneDNN backend 안에서 자동 활성** (env: `ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX`). 우리 측정 (Tech_done.md) 에서 H100x4 + Xeon 8480+ 환경에서 `brg_matmul:avx10_1_512_amx` dispatch 확정.

oneDNN 은 attention 자체 op 가 아닌 GEMM op. **NEO 의 cdec attention 영역에 직접 적용은 가능하지만 layout 변환 cost 동반**.

---

## C.6 cpu_arch_macros.h — fast_exp helper

```cpp
// AVX-512 F + scalar fallback
inline __m512 _mm512_fast_exp_ps(__m512 x);   // 5-degree polynomial approximation
```

NEO 의 softmax exp 부분 가속 후보. 현재 ISPC 가 자동 lower (`__svml_expf16` 또는 polynomial) 하지만, 명시적 `_mm512_fast_exp_ps` 가 cycle 정확.

---

## C.7 cpu_wna16.cpp — AVX-512 VNNI INT8 dot product 참조

```cpp
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
  _mm512_cvtepi8_epi16        // INT8 → INT16 (sign extension)
  _mm512_madd_epi16           // INT16 × INT16 → INT32 mul-add
  _mm512_dpbusd_epi32         // u8×s8 dot product (VNNI 1 instruction)
  _mm512_reduce_add_epi32     // horizontal reduce
#endif
```

→ NEO 의 FP16 dot product 영역에 직접 적용 불가 (INT 영역). 그러나 **패턴 참조** 로 AVX-512 intrinsic 작성법 reference.

---

## C.8 TSK_003 의 `partial_attention_avx512.cpp` / `partial_attention_amx.cpp`

shadow_assists 의 TSK_003.md 참조:

| 파일 | ISA | dtype | prod 검증 |
|---|---|---|---|
| `partial_attention_avx512.cpp` | AVX-512F | BF16 (with `-mavx512bf16`) | TST_004 152 PASS |
| `partial_attention_amx.cpp` | AMX-TILE + AMX-BF16 | BF16 | TST_004 152 PASS |

`-O3 -mavx512f -mavx512bf16` / `-O3 -mamx-tile -mamx-bf16` 빌드 flag. cpuid `_has_avx512()` / `_has_amx()` gate. JIT load 첫 호출.

**그러나 이 두 파일은 본 repo 의 `csrc/cpu/` 에 직접 보이지 않음** (TSK_003 doc 가 "Phase 1 완료" 라 했으나 실제 source 위치 미상). Phase D 에서 확인 필요.

---

## C.9 결론 — 기존 가속 자산 인벤토리 + 재사용 가능성

| 자산 | 위치 | ISA | dtype | 재사용 가능성 | comment |
|---|---|---|---|:-:|---|
| `cpu_attn_amx.hpp` TileGemm224 | `csrc/cpu/` | AMX | BF16/FP16/INT8 | △ | NEO pacpu 와 다른 op (attention with kv cache). API 재포장 필요 |
| `micro_gemm/cpu_micro_gemm_amx.hpp` TileGemm224 | `csrc/cpu/micro_gemm/` | AMX | template | ○ | micro-GEMM 형태 — pacpu 의 qk_product / av_product 에 직접 적용 가능 |
| `dnnl_kernels.cpp` onednn_mm | `csrc/cpu/` | oneDNN dispatch (AMX 자동) | BF16 | ◎ | qk_product 의 GEMM 부분을 oneDNN 으로 위임. layout cost 필요 |
| `cpu_arch_macros.h` fast_exp | `csrc/cpu/` | AVX-512 | FP32 | ○ | NEO softmax 의 exp 영역 ports |
| `cpu_wna16.cpp` VNNI 패턴 | `csrc/cpu/` | AVX-512 VNNI | INT8 | △ | dtype 다름. 패턴만 참조 |
| TSK_003 `partial_attention_amx.cpp` | (위치 미상) | AMX BF16 | BF16 | ⏳ | prod 검증 152 PASS — 위치 확인 필요 |

→ **재사용 강한 후보**:
- **dnnl onednn_mm** 으로 qk_product GEMM 부분 위임 (cost: FP16 → BF16 변환)
- **micro_gemm/cpu_micro_gemm_amx.hpp** 의 TileGemm224 template 직접 사용
- **cpu_arch_macros.h** fast_exp 로 softmax 가속

**신규 작성 필요 영역**:
- pacpu 의 ISPC kernel 을 AMX intrinsic 으로 rewrite (qk_product / av_product 자체)
- layout 변환 (NEO 의 [BLOCK_SIZE, HEAD_DIM] ↔ AMX tile expected layout)
