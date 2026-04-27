// SPDX-License-Identifier: Apache-2.0
// Cold-KV CPU Partial Attention — AVX-512 kernel (TSK_003 §4.2a).
//
// Algorithmically identical to the portable C++ kernel
// (`partial_attention_portable.cpp`). The hot path — the inner
// head_dim dot product against each cold KV token — is replaced with
// AVX-512 intrinsics. Everything else (the per-sequence loop, the
// online softmax LSE return) stays in scalar C++ so the cross-check
// against the portable kernel (TST_004 §B(ii)) is a 1:1 numerical
// comparison up to BF16 round-off.
//
// For BF16 inputs we exploit AVX512_BF16 (`vdpbf16ps`) when the
// compiler is invoked with ``-mavx512bf16``: 32 BF16 multiplies plus
// 16 fp32 accumulators in a single instruction. Without that flag we
// fall back to upcast-and-fma (load 16 BF16 → zero-extend to int32 →
// shift-by-16 → reinterpret as fp32 → ``_mm512_fmadd_ps``).
//
// FP16 and FP32 use the corresponding AVX-512F path (``_mm512_cvtph
// _ps`` for FP16, native ``_mm512_loadu_ps`` for FP32). The softmax
// max-tracking, exp, and weighted V sum stay in scalar f32 — the
// vectorisable part of those is the same head_dim sweep, but until
// PLN_001 §4.2 sweep tells us the dot product is no longer the
// bottleneck we keep the change surface small.
//
// Dispatch policy and runtime cpuid gate live in the Python wrapper
// (`vllm/v1/attention/ops/cpu_partial_attention.py:_has_avx512_kernel`).
// This file is built only when the build environment is told to
// emit AVX-512 — see the JIT compile flags or the ahead-of-time
// CMake recipe.

#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#define VLLM_CPU_PARTIAL_HAS_AVX512 1
#else
#define VLLM_CPU_PARTIAL_HAS_AVX512 0
#endif

#if defined(__AVX512BF16__)
#define VLLM_CPU_PARTIAL_HAS_AVX512_BF16 1
#else
#define VLLM_CPU_PARTIAL_HAS_AVX512_BF16 0
#endif

namespace cpu_partial_attn_avx512 {

#if VLLM_CPU_PARTIAL_HAS_AVX512

// Horizontal sum of 16 fp32 lanes → scalar. AVX-512F-only path
// (avoids _mm512_extractf32x8_ps which requires AVX-512DQ). We split
// the 512-bit register into four 128-bit lanes via the F-only
// _mm512_extractf32x4_ps and reduce.
static inline float hsum_ps_512(__m512 v) {
  __m128 lane0 = _mm512_castps512_ps128(v);
  __m128 lane1 = _mm512_extractf32x4_ps(v, 1);
  __m128 lane2 = _mm512_extractf32x4_ps(v, 2);
  __m128 lane3 = _mm512_extractf32x4_ps(v, 3);
  __m128 sum01 = _mm_add_ps(lane0, lane1);
  __m128 sum23 = _mm_add_ps(lane2, lane3);
  __m128 q = _mm_add_ps(sum01, sum23);
  q = _mm_add_ps(q, _mm_movehl_ps(q, q));
  q = _mm_add_ss(q, _mm_shuffle_ps(q, q, 0x55));
  return _mm_cvtss_f32(q);
}

// ---- BF16 dot product ----
// at::BFloat16 has a single ``uint16_t x`` field so a tensor of N BF16
// elements is a contiguous ``uint16_t[N]`` we can load with vector
// instructions.
static inline float dot_bf16_avx512(const at::BFloat16* a,
                                    const at::BFloat16* b,
                                    int64_t n) {
  __m512 acc = _mm512_setzero_ps();
  int64_t d = 0;
#if VLLM_CPU_PARTIAL_HAS_AVX512_BF16
  // Native vdpbf16ps when AVX512_BF16 is available. Each iteration
  // consumes 32 BF16 elements (one __m512bh from each side).
  for (; d + 32 <= n; d += 32) {
    __m512bh va = (__m512bh)_mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(a + d));
    __m512bh vb = (__m512bh)_mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(b + d));
    acc = _mm512_dpbf16_ps(acc, va, vb);
  }
#endif
  // Generic AVX-512F path: 16 BF16 elements at a time. Upcast by
  // shifting the 16-bit pattern left by 16 bits to land in the
  // fp32 mantissa/exponent layout.
  for (; d + 16 <= n; d += 16) {
    __m256i ai16 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(a + d));
    __m256i bi16 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(b + d));
    __m512i ai32 = _mm512_cvtepu16_epi32(ai16);
    __m512i bi32 = _mm512_cvtepu16_epi32(bi16);
    __m512 af = _mm512_castsi512_ps(_mm512_slli_epi32(ai32, 16));
    __m512 bf = _mm512_castsi512_ps(_mm512_slli_epi32(bi32, 16));
    acc = _mm512_fmadd_ps(af, bf, acc);
  }
  float result = hsum_ps_512(acc);
  // Scalar tail.
  for (; d < n; ++d) {
    result += static_cast<float>(a[d]) * static_cast<float>(b[d]);
  }
  return result;
}

// ---- FP16 dot product ----
static inline float dot_fp16_avx512(const at::Half* a,
                                    const at::Half* b,
                                    int64_t n) {
  __m512 acc = _mm512_setzero_ps();
  int64_t d = 0;
  // 16 FP16 → fp32 per iter via cvtph_ps.
  for (; d + 16 <= n; d += 16) {
    __m256i ai16 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(a + d));
    __m256i bi16 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(b + d));
    __m512 af = _mm512_cvtph_ps(ai16);
    __m512 bf = _mm512_cvtph_ps(bi16);
    acc = _mm512_fmadd_ps(af, bf, acc);
  }
  float result = hsum_ps_512(acc);
  for (; d < n; ++d) {
    result += static_cast<float>(a[d]) * static_cast<float>(b[d]);
  }
  return result;
}

// ---- FP32 dot product ----
static inline float dot_fp32_avx512(const float* a, const float* b,
                                    int64_t n) {
  __m512 acc = _mm512_setzero_ps();
  int64_t d = 0;
  for (; d + 16 <= n; d += 16) {
    __m512 af = _mm512_loadu_ps(a + d);
    __m512 bf = _mm512_loadu_ps(b + d);
    acc = _mm512_fmadd_ps(af, bf, acc);
  }
  float result = hsum_ps_512(acc);
  for (; d < n; ++d) {
    result += a[d] * b[d];
  }
  return result;
}

#endif  // VLLM_CPU_PARTIAL_HAS_AVX512

// ---- Generic scalar fallback (used only when SIMD is unavailable
//     at compile time — should never hit at runtime when this
//     translation unit is selected by the wrapper) ------------------
template <typename T>
static inline float dot_scalar(const T* a, const T* b, int64_t n) {
  float acc = 0.0f;
  for (int64_t d = 0; d < n; ++d) {
    acc += static_cast<float>(a[d]) * static_cast<float>(b[d]);
  }
  return acc;
}

// Type dispatch for the inner dot product.
template <typename T>
static inline float dot_avx512(const T* a, const T* b, int64_t n);

template <>
inline float dot_avx512<at::BFloat16>(const at::BFloat16* a,
                                      const at::BFloat16* b,
                                      int64_t n) {
#if VLLM_CPU_PARTIAL_HAS_AVX512
  return dot_bf16_avx512(a, b, n);
#else
  return dot_scalar(a, b, n);
#endif
}

template <>
inline float dot_avx512<at::Half>(const at::Half* a, const at::Half* b,
                                  int64_t n) {
#if VLLM_CPU_PARTIAL_HAS_AVX512
  return dot_fp16_avx512(a, b, n);
#else
  return dot_scalar(a, b, n);
#endif
}

template <>
inline float dot_avx512<float>(const float* a, const float* b,
                               int64_t n) {
#if VLLM_CPU_PARTIAL_HAS_AVX512
  return dot_fp32_avx512(a, b, n);
#else
  return dot_scalar(a, b, n);
#endif
}

// ---- V weighted sum SIMD --------------------------------------------
// AVX-512 fmadd of ``out[d] += w * v_ptr[d]`` over head_dim. Same
// dispatch shape as the dot helpers above. Caller's ``out`` is fp32.

#if VLLM_CPU_PARTIAL_HAS_AVX512
template <typename T>
static inline void v_fmadd_avx512(float w, const T* v_ptr, float* out,
                                  int64_t n);

template <>
inline void v_fmadd_avx512<at::BFloat16>(float w, const at::BFloat16* v_ptr,
                                         float* out, int64_t n) {
  __m512 wv = _mm512_set1_ps(w);
  int64_t d = 0;
  for (; d + 16 <= n; d += 16) {
    __m256i bi16 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(v_ptr + d));
    __m512 vf = _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_cvtepu16_epi32(bi16), 16));
    __m512 of = _mm512_loadu_ps(out + d);
    of = _mm512_fmadd_ps(wv, vf, of);
    _mm512_storeu_ps(out + d, of);
  }
  for (; d < n; ++d) {
    out[d] += w * static_cast<float>(v_ptr[d]);
  }
}

template <>
inline void v_fmadd_avx512<at::Half>(float w, const at::Half* v_ptr,
                                     float* out, int64_t n) {
  __m512 wv = _mm512_set1_ps(w);
  int64_t d = 0;
  for (; d + 16 <= n; d += 16) {
    __m256i hi16 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(v_ptr + d));
    __m512 vf = _mm512_cvtph_ps(hi16);
    __m512 of = _mm512_loadu_ps(out + d);
    of = _mm512_fmadd_ps(wv, vf, of);
    _mm512_storeu_ps(out + d, of);
  }
  for (; d < n; ++d) {
    out[d] += w * static_cast<float>(v_ptr[d]);
  }
}

template <>
inline void v_fmadd_avx512<float>(float w, const float* v_ptr, float* out,
                                  int64_t n) {
  __m512 wv = _mm512_set1_ps(w);
  int64_t d = 0;
  for (; d + 16 <= n; d += 16) {
    __m512 vf = _mm512_loadu_ps(v_ptr + d);
    __m512 of = _mm512_loadu_ps(out + d);
    of = _mm512_fmadd_ps(wv, vf, of);
    _mm512_storeu_ps(out + d, of);
  }
  for (; d < n; ++d) {
    out[d] += w * v_ptr[d];
  }
}

#else
template <typename T>
static inline void v_fmadd_avx512(float w, const T* v_ptr, float* out,
                                  int64_t n) {
  for (int64_t d = 0; d < n; ++d)
    out[d] += w * static_cast<float>(v_ptr[d]);
}
#endif

template <typename T>
static std::vector<torch::Tensor> forward_partial_impl(
    torch::Tensor query,
    torch::Tensor cold_kv_cache,
    torch::Tensor cold_kv_cache_v,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_block_bytes,
    torch::Tensor cold_block_ids,
    torch::Tensor cold_block_lens,
    torch::Tensor cu_seqlens_q,
    torch::Tensor query_positions,
    double softmax_scale,
    bool causal) {
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(cold_kv_cache.is_contiguous(), "cold_kv_cache must be contiguous");
  TORCH_CHECK(cold_kv_cache.scalar_type() == torch::kInt8,
              "cold_kv_cache must be int8");
  TORCH_CHECK(query.dim() == 3, "query must be 3D");

  const bool split_kv = cold_kv_cache_v.numel() > 0;
  if (split_kv) {
    TORCH_CHECK(cold_kv_cache_v.is_contiguous(),
                "cold_kv_cache_v must be contiguous");
    TORCH_CHECK(cold_kv_cache_v.scalar_type() == torch::kInt8,
                "cold_kv_cache_v must be int8");
    TORCH_CHECK(cold_kv_cache_v.size(0) == cold_kv_cache.size(0),
                "cold_kv_cache and cold_kv_cache_v must agree on num_blocks");
  }

  const int64_t num_tokens = query.size(0);
  const int64_t num_q_heads = query.size(1);
  TORCH_CHECK(query.size(2) == head_dim, "query head_dim mismatch");
  TORCH_CHECK(num_q_heads % num_kv_heads == 0,
              "num_q_heads must be divisible by num_kv_heads");
  const int64_t q_per_kv = num_q_heads / num_kv_heads;
  const int64_t num_seqs = cu_seqlens_q.size(0) - 1;
  const int64_t page_size_bytes = cold_kv_cache.size(1);

  if (split_kv) {
    TORCH_CHECK(page_size_bytes >= kv_block_bytes,
                "split-K cache page too small for K");
    TORCH_CHECK(cold_kv_cache_v.size(1) >= kv_block_bytes,
                "split-V cache page too small for V");
  } else {
    TORCH_CHECK(page_size_bytes >= 2 * kv_block_bytes,
                "combined page too small for K + V");
  }
  TORCH_CHECK(static_cast<size_t>(kv_block_bytes) % sizeof(T) == 0,
              "kv_block_bytes not aligned to dtype itemsize");
  const int64_t k_block_stride_elems =
      page_size_bytes / static_cast<int64_t>(sizeof(T));
  const int64_t v_block_stride_elems =
      split_kv
          ? (cold_kv_cache_v.size(1) / static_cast<int64_t>(sizeof(T)))
          : k_block_stride_elems;
  const int64_t v_intra_block_offset_elems =
      split_kv ? 0 : (kv_block_bytes / static_cast<int64_t>(sizeof(T)));

  const T* k_data = reinterpret_cast<const T*>(
      cold_kv_cache.data_ptr<int8_t>());
  const T* v_data = split_kv
      ? reinterpret_cast<const T*>(cold_kv_cache_v.data_ptr<int8_t>())
      : k_data;

  auto O = torch::zeros({num_tokens, num_q_heads, head_dim}, query.options());
  auto LSE = torch::full(
      {num_q_heads, num_tokens},
      -std::numeric_limits<float>::infinity(),
      query.options().dtype(torch::kFloat32));

  auto query_a = query.accessor<T, 3>();
  auto O_a = O.accessor<T, 3>();
  auto LSE_a = LSE.accessor<float, 2>();
  auto cold_block_ids_a = cold_block_ids.accessor<int32_t, 2>();
  auto cold_block_lens_a = cold_block_lens.accessor<int32_t, 1>();
  auto cu_seqlens_q_a = cu_seqlens_q.accessor<int32_t, 1>();
  auto query_positions_a = query_positions.accessor<int32_t, 1>();

  const float scale_f = static_cast<float>(softmax_scale);
  const float NEG_INF = -std::numeric_limits<float>::infinity();

  for (int64_t s = 0; s < num_seqs; ++s) {
    const int64_t q_start = cu_seqlens_q_a[s];
    const int64_t q_end = cu_seqlens_q_a[s + 1];
    const int64_t n_cold_blocks = cold_block_lens_a[s];
    if (q_end <= q_start || n_cold_blocks <= 0) continue;
    const int64_t n_cold_kv = n_cold_blocks * block_size;

    #pragma omp parallel default(none) \
        firstprivate(q_start, q_end, n_cold_kv, num_q_heads, q_per_kv, \
                     num_kv_heads, head_dim, block_size, \
                     k_block_stride_elems, v_block_stride_elems, \
                     v_intra_block_offset_elems, scale_f, NEG_INF, \
                     causal, k_data, v_data, s) \
        shared(query_a, O_a, LSE_a, cold_block_ids_a, query_positions_a)
    {
    std::vector<float> scores(static_cast<size_t>(n_cold_kv));

    #pragma omp for collapse(2) schedule(static) nowait
    for (int64_t t = q_start; t < q_end; ++t) {
      for (int64_t h = 0; h < num_q_heads; ++h) {
        const int64_t q_pos = query_positions_a[t];
        const int64_t kv_h = h / q_per_kv;
        // Pointer to query[t, h, :] for the inner dot product.
        const T* q_ptr = &query_a[t][h][0];

        // ---- Pass 1: scores + max ----
        float m_val = NEG_INF;
        for (int64_t k = 0; k < n_cold_kv; ++k) {
          const int64_t block_idx_in_seq = k / block_size;
          const int64_t token_in_block = k % block_size;
          const int64_t real_block_id =
              cold_block_ids_a[s][block_idx_in_seq];

          if (causal && q_pos < k) {
            scores[k] = NEG_INF;
            continue;
          }

          const T* k_ptr = k_data
              + real_block_id * k_block_stride_elems
              + token_in_block * num_kv_heads * head_dim
              + kv_h * head_dim;

          const float dot = dot_avx512<T>(q_ptr, k_ptr, head_dim);
          const float score = dot * scale_f;
          scores[k] = score;
          if (score > m_val) m_val = score;
        }

        if (m_val == NEG_INF) continue;

        // ---- Pass 2: exp(score - m), accumulate sum ----
        float sum_exp = 0.0f;
        for (int64_t k = 0; k < n_cold_kv; ++k) {
          if (scores[k] == NEG_INF) {
            scores[k] = 0.0f;
            continue;
          }
          const float ex = std::exp(scores[k] - m_val);
          scores[k] = ex;
          sum_exp += ex;
        }

        // ---- Pass 3: weighted V sum ----
        float out[1024];
        TORCH_CHECK(head_dim <= 1024, "head_dim too large for stack buffer");
        for (int64_t d = 0; d < head_dim; ++d) out[d] = 0.0f;

        const float inv_sum = 1.0f / sum_exp;
        for (int64_t k = 0; k < n_cold_kv; ++k) {
          const float w = scores[k] * inv_sum;
          if (w == 0.0f) continue;
          const int64_t block_idx_in_seq = k / block_size;
          const int64_t token_in_block = k % block_size;
          const int64_t real_block_id =
              cold_block_ids_a[s][block_idx_in_seq];
          const T* v_ptr = v_data
              + real_block_id * v_block_stride_elems
              + v_intra_block_offset_elems
              + token_in_block * num_kv_heads * head_dim
              + kv_h * head_dim;
          v_fmadd_avx512<T>(w, v_ptr, out, head_dim);
        }

        for (int64_t d = 0; d < head_dim; ++d) {
          O_a[t][h][d] = static_cast<T>(out[d]);
        }
        LSE_a[h][t] = m_val + std::log(sum_exp);
      }    // close ``for h`` (omp for collapse=2)
    }      // close ``for t``
    }      // close ``#pragma omp parallel``
  }        // close ``for s``

  return {O, LSE};
}

std::vector<torch::Tensor> forward_partial_with_lse_avx512(
    torch::Tensor query,
    torch::Tensor cold_kv_cache,
    torch::Tensor cold_kv_cache_v,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_block_bytes,
    torch::Tensor cold_block_ids,
    torch::Tensor cold_block_lens,
    torch::Tensor cu_seqlens_q,
    torch::Tensor query_positions,
    double softmax_scale,
    bool causal) {
  const auto dt = query.scalar_type();
  if (dt == torch::kBFloat16) {
    return forward_partial_impl<at::BFloat16>(
        query, cold_kv_cache, cold_kv_cache_v, block_size, num_kv_heads,
        head_dim, kv_block_bytes, cold_block_ids, cold_block_lens,
        cu_seqlens_q, query_positions, softmax_scale, causal);
  } else if (dt == torch::kHalf) {
    return forward_partial_impl<at::Half>(
        query, cold_kv_cache, cold_kv_cache_v, block_size, num_kv_heads,
        head_dim, kv_block_bytes, cold_block_ids, cold_block_lens,
        cu_seqlens_q, query_positions, softmax_scale, causal);
  } else if (dt == torch::kFloat) {
    return forward_partial_impl<float>(
        query, cold_kv_cache, cold_kv_cache_v, block_size, num_kv_heads,
        head_dim, kv_block_bytes, cold_block_ids, cold_block_lens,
        cu_seqlens_q, query_positions, softmax_scale, causal);
  } else {
    TORCH_CHECK(false,
                "forward_partial_with_lse_avx512: unsupported dtype ",
                dt);
  }
}

}  // namespace cpu_partial_attn_avx512

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_partial_with_lse_avx512",
        &cpu_partial_attn_avx512::forward_partial_with_lse_avx512,
        "Cold-KV CPU Partial Attention (AVX-512)");
}
