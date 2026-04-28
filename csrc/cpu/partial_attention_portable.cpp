// SPDX-License-Identifier: Apache-2.0
// Cold-KV CPU Partial Attention — portable C++ fallback (TSK_001 §4.2c).
//
// Pure C++ scalar/auto-vectorized partial attention with online softmax
// LSE return. Works on any x86 / ARM / etc. machine (no SIMD intrinsics
// — relies on `-O3 -ftree-vectorize` for the compiler to vectorize the
// inner head_dim dot product).
//
// Built via torch.utils.cpp_extension.load (JIT) — see
// vllm/v1/attention/ops/cpu_partial_attention.py.

#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <sched.h>

// vLLM's V1 multiproc executor sets ``OMP_NUM_THREADS=1`` in worker
// subprocesses. Resolve thread count as:
//   baseline = ``CPU_COUNT(sched_getaffinity)`` (TSK_004 NUMA bind),
//   then ``VLLM_PARTIAL_ATTN_THREADS`` env (if set, positive) is
//   clamped to ``min(env, baseline)`` so an oversized operator override
//   cannot resurrect the pthread_create EAGAIN storm. Cached function-
//   static so the cost is paid once per worker process.
static int vllm_partial_attn_thread_count() {
  // Thread-local: omp_get_max_threads() reflects the calling thread's
  // OpenMP nthreads-var ICV (set via omp_set_num_threads). TSK_010
  // sub-batching has the Python wrapper invoke omp_set_num_threads on
  // each sub-batch worker thread to limit the OMP team size and avoid
  // oversubscription. When the calling thread has set a value > 1, use
  // it. Otherwise fall back to the cached baseline (preserves the
  // pre-TSK_010 single-batch behaviour where vLLM's V1 multiproc
  // executor sets OMP_NUM_THREADS=1 and we override via the cached
  // sched_getaffinity / VLLM_PARTIAL_ATTN_THREADS path).
#ifdef _OPENMP
  int omp_max = omp_get_max_threads();
  if (omp_max > 1) {
    return omp_max;
  }
#endif
  static int cached = []() {
    int baseline;
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
      int n = CPU_COUNT(&mask);
      baseline = (n > 0) ? n : 1;
    } else {
#ifdef _OPENMP
      baseline = omp_get_num_procs();
#else
      baseline = 1;
#endif
    }
    if (const char* env = std::getenv("VLLM_PARTIAL_ATTN_THREADS")) {
      int v = std::atoi(env);
      if (v > 0) {
        return (v < baseline) ? v : baseline;
      }
    }
    return baseline;
  }();
  return cached;
}

namespace cpu_partial_attn {

template <typename T>
static std::vector<torch::Tensor> forward_partial_impl(
    torch::Tensor query,           // [num_tokens, num_q_heads, head_dim], T
    torch::Tensor cold_kv_cache,   // [num_blocks, page_size_bytes],     int8.
                                   //   Combined mode: page = K+V back-to-back.
                                   //   Split mode: page = K only.
    torch::Tensor cold_kv_cache_v, // [num_blocks, kv_block_bytes], int8.
                                   //   Empty (numel == 0) -> combined mode.
                                   //   Non-empty -> split mode (V-only buffer
                                   //   paired with K-only cold_kv_cache).
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_block_bytes,        // bytes used by ONE of {K, V} per page
    torch::Tensor cold_block_ids,  // [num_seqs, max_cold_blocks], int32
    torch::Tensor cold_block_lens, // [num_seqs], int32
    torch::Tensor cu_seqlens_q,    // [num_seqs + 1], int32
    torch::Tensor query_positions, // [num_tokens], int32
    double softmax_scale,
    bool causal) {
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(cold_kv_cache.is_contiguous(), "cold_kv_cache must be contiguous");
  TORCH_CHECK(cold_kv_cache.scalar_type() == torch::kInt8,
              "cold_kv_cache must be int8");
  TORCH_CHECK(query.dim() == 3, "query must be 3D");

  // Detect split-K/V layout. FlashAttention's OffloadingConnector mirror
  // surfaces K and V as two distinct (num_blocks, kv_block_bytes) buffers
  // (rather than one combined K+V page) so the C++ kernels need to handle
  // both modes. Empty cold_kv_cache_v (numel == 0) is the legacy combined
  // mode signal.
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
  TORCH_CHECK(query.size(2) == head_dim,
              "query head_dim mismatch");
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
  // Per-block stride within K cache: full row width of cold_kv_cache.
  const int64_t k_block_stride_elems =
      page_size_bytes / static_cast<int64_t>(sizeof(T));
  // Per-block stride within V cache: full row width of cold_kv_cache_v in
  // split mode, equals K stride in combined mode.
  const int64_t v_block_stride_elems =
      split_kv
          ? (cold_kv_cache_v.size(1) / static_cast<int64_t>(sizeof(T)))
          : k_block_stride_elems;
  // Intra-block offset for V data: 0 in split mode (V starts at row 0 of
  // cold_kv_cache_v), kv_block_bytes worth of elements in combined mode
  // (V is back-to-back after K).
  const int64_t v_intra_block_offset_elems =
      split_kv ? 0 : (kv_block_bytes / static_cast<int64_t>(sizeof(T)));
  const int64_t elements_per_kv_block =
      kv_block_bytes / static_cast<int64_t>(sizeof(T));

  // Reinterpret canonical int8 storage as typed T pointer.
  const T* k_data = reinterpret_cast<const T*>(cold_kv_cache.data_ptr<int8_t>());
  const T* v_data = split_kv
      ? reinterpret_cast<const T*>(cold_kv_cache_v.data_ptr<int8_t>())
      : k_data;

  auto O = torch::zeros({num_tokens, num_q_heads, head_dim}, query.options());
  auto LSE = torch::full(
      {num_q_heads, num_tokens},
      -std::numeric_limits<float>::infinity(),
      query.options().dtype(torch::kFloat32));

  // Accessors (CPU only).
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
        num_threads(vllm_partial_attn_thread_count()) \
        firstprivate(q_start, q_end, n_cold_kv, num_q_heads, q_per_kv, \
                     num_kv_heads, head_dim, block_size, \
                     k_block_stride_elems, v_block_stride_elems, \
                     v_intra_block_offset_elems, scale_f, NEG_INF, \
                     causal, k_data, v_data, s) \
        shared(query_a, O_a, LSE_a, cold_block_ids_a, query_positions_a)
    {
    // Per-thread reusable score buffer.
    std::vector<float> scores(static_cast<size_t>(n_cold_kv));

    #pragma omp for collapse(2) schedule(static) nowait
    for (int64_t t = q_start; t < q_end; ++t) {
      for (int64_t h = 0; h < num_q_heads; ++h) {
        const int64_t q_pos = query_positions_a[t];
        const int64_t kv_h = h / q_per_kv;

        // ---- Pass 1: compute scores, track max ----
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

          // Inner head_dim dot product. Compiler should auto-vectorize
          // this loop with `-O3 -ftree-vectorize`.
          float dot = 0.0f;
          for (int64_t d = 0; d < head_dim; ++d) {
            dot += static_cast<float>(query_a[t][h][d])
                 * static_cast<float>(k_ptr[d]);
          }
          const float score = dot * scale_f;
          scores[k] = score;
          if (score > m_val) m_val = score;
        }

        if (m_val == NEG_INF) {
          // All entries were masked; LSE stays -inf and O stays zero.
          continue;
        }

        // ---- Pass 2: exp(score - m), accumulate sum ----
        float sum_exp = 0.0f;
        for (int64_t k = 0; k < n_cold_kv; ++k) {
          if (scores[k] == NEG_INF) {
            scores[k] = 0.0f;  // re-purpose as exp value (0 here)
            continue;
          }
          const float ex = std::exp(scores[k] - m_val);
          scores[k] = ex;
          sum_exp += ex;
        }

        // ---- Pass 3: weighted V sum ----
        // O[t, h, :] = sum_k (scores[k] / sum_exp) * V[real_block, tok, kv_h, :]
        float out[1024];  // head_dim cap (Qwen2.5-7B uses 128). Defensive.
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
          for (int64_t d = 0; d < head_dim; ++d) {
            out[d] += w * static_cast<float>(v_ptr[d]);
          }
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

std::vector<torch::Tensor> forward_partial_with_lse_portable(
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
                "forward_partial_with_lse_portable: unsupported dtype ",
                dt);
  }
}

}  // namespace cpu_partial_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_partial_with_lse_portable",
        &cpu_partial_attn::forward_partial_with_lse_portable,
        "Cold-KV CPU Partial Attention (portable C++)");
}
