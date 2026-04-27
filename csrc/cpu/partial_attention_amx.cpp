// SPDX-License-Identifier: Apache-2.0
// Cold-KV CPU Partial Attention — AMX kernel (TSK_003 §4.2b).
//
// Same outer loop / online softmax / LSE return as the portable and
// AVX-512 kernels (TSK_001 §4.2c, TSK_003 §4.2a) — what changes is
// the inner Q · K^T score computation, which now uses Intel AMX BF16
// tile matmul (`_tile_dpbf16ps`) to evaluate 16 K rows of dot product
// per tile call. AMX is a Sapphire Rapids+ feature; on a host that
// does not expose AMX-BF16 the wrapper's cpuid gate
// (`_has_amx_kernel`) prevents this translation unit from ever
// loading, so neither compile-time `-mamx-bf16` nor the static
// initializer of this .so can SIGILL on the dev box (Alder Lake
// 12900KF, no AMX hardware).
//
// AMX tile layout used here (palette 1, BF16 mode):
//
//   tile 0 (C, fp32 accumulator)
//     M = 1   (one query row at a time)
//     N = 16  (16 K rows in one matmul)
//     => 1 × 16 fp32 = 64 bytes
//
//   tile 1 (A, BF16 query)
//     M = 1
//     K_paired = up to 16 (= 32 BF16 elements per row)
//     => 1 × 64 bytes = 64 bytes
//
//   tile 2 (B, BF16 keys staged in A·B^T-friendly layout)
//     K_paired = up to 16
//     N = 16
//     => 16 pair-rows × 64 bytes = 1024 bytes
//
// For ``head_dim`` larger than 32 BF16 (the per-tile K limit) we
// accumulate across head_dim in chunks of 32. Llama-3.3-70B uses
// head_dim = 128 → 4 accumulating tile calls per (query row, K
// batch).
//
// The K batch is laid out by a thin staging step that copies 16 K
// rows from the canonical cache into a B-tile-friendly buffer
// (transpose from row-major (n_K × head_dim) to
// (head_dim/2 × n_K × bf16-pair) so consecutive head_dim pair-rows
// load with a 32-byte stride). The staging cost is one read of the
// K block per AMX tile call and is a small constant relative to the
// dpbf16ps throughput.
//
// FP16 / FP32 dispatch falls back to the AVX-512 path: AMX BF16 only
// helps the BF16 case, and the prod scope (PLN_001 §3) is BF16/FP16
// — FP16 routes via the AVX-512 kernel (which itself uses
// `_mm512_cvtph_ps`).

#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/syscall.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AMX_TILE__) && defined(__AMX_BF16__) && defined(__AVX512F__)
#include <immintrin.h>
#define VLLM_CPU_PARTIAL_HAS_AMX 1
#else
#define VLLM_CPU_PARTIAL_HAS_AMX 0
#endif

// IDE_006 / TSK_003 §4.2b — diagnostic checkpoint printer. Unbuffered
// stderr so even a SIGILL after a checkpoint leaves a "last seen at"
// breadcrumb in the captured run log. Off by default — enable with
// ``VLLM_AMX_TRACE=1`` in the environment to revive the breadcrumbs
// for re-debugging. The static flag is read once at .so load and
// then the per-call cost collapses to a single predictable branch.
static const bool vllm_amx_trace_enabled = []() {
  const char* env = std::getenv("VLLM_AMX_TRACE");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}();

static inline void vllm_amx_trace(const char* tag) {
  if (!vllm_amx_trace_enabled) return;
  static thread_local int call_count = 0;
  if (call_count >= 64) return;  // hard cap per worker thread
  ++call_count;
  pid_t pid = getpid();
  long tid = static_cast<long>(syscall(SYS_gettid));
  fprintf(stderr,
          "[AMX trace pid=%d tid=%ld] %s\n",
          static_cast<int>(pid), tid, tag);
  fflush(stderr);
}

namespace cpu_partial_attn_amx {

#if VLLM_CPU_PARTIAL_HAS_AMX

// ---- AMX tile config -----------------------------------------------
// One palette-1 record per thread. The palette byte is followed by 7
// reserved bytes, then 8 (rows, colsb) pairs for the 8 tile registers.

struct __attribute__((packed)) TileConfig {
  uint8_t palette;
  uint8_t start_row;
  uint8_t reserved0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
};

// AMX tile register indices. The intrinsics require these as
// literal compile-time integers (the assembler emits ``tmm0``,
// ``tmm1``, etc. directly), so we use preprocessor macros rather
// than ``constexpr int`` — a ``static constexpr int TMM_C = 0;``
// is folded by the optimiser but the inline asm template needs the
// literal token at preprocessing time.
#define VLLM_TMM_C 0
#define VLLM_TMM_A 1
#define VLLM_TMM_B 2

static inline void configure_tiles_for_dot(int head_dim_chunk_bf16) {
  vllm_amx_trace("configure_tiles_for_dot:enter");
  // K_paired = head_dim_chunk_bf16 / 2 (each pair = 2 BF16 = 4 bytes)
  TileConfig cfg{};
  cfg.palette = 1;
  // tile 0 (C): 1 × 16 fp32 = 64 bytes per row, 1 row
  cfg.rows[VLLM_TMM_C] = 1;
  cfg.colsb[VLLM_TMM_C] = 16 * 4;  // 64 bytes
  // tile 1 (A): 1 × head_dim_chunk_bf16 = 1 row × (chunk * 2) bytes
  cfg.rows[VLLM_TMM_A] = 1;
  cfg.colsb[VLLM_TMM_A] = head_dim_chunk_bf16 * 2;  // up to 64 bytes
  // tile 2 (B): K_paired pair-rows × 16 cols × 4 bytes = ... bytes per row
  // (16 cols × 2 BF16 pair = 4 bytes per col, 16 cols = 64 bytes per row)
  cfg.rows[VLLM_TMM_B] = head_dim_chunk_bf16 / 2;  // K_paired
  cfg.colsb[VLLM_TMM_B] = 16 * 4;  // 64 bytes per pair-row
  vllm_amx_trace("configure_tiles_for_dot:about_to_tile_loadconfig");
  _tile_loadconfig(&cfg);
  vllm_amx_trace("configure_tiles_for_dot:tile_loadconfig_returned");
}

// Pack 16 K rows (each head_dim BF16) from canonical layout into the
// AMX B-tile layout: K_paired pair-rows × 16 cols × 2 BF16 each.
//
// canonical[i, d] = K row i element d.
// B[paired_row p, col c, pair s] = canonical[c, 2p + s]   for s ∈ {0,1}
//
// I.e. each pair-row p of B holds the (2p)-th and (2p+1)-th BF16
// element of each of the 16 K rows. Stride within a pair-row is
// 4 bytes (one (BF16,BF16) pair per K row, 16 K rows = 64 bytes).
static inline void pack_K_for_B_tile(
    const at::BFloat16* canonical_K_block_ptr,
    int64_t k_row_stride_bytes,
    int head_dim,
    uint16_t* B_buf  // B_buf[K_paired × 16 × 2] BF16
) {
  const int K_paired = head_dim / 2;
  for (int p = 0; p < K_paired; ++p) {
    for (int c = 0; c < 16; ++c) {
      const at::BFloat16* row = reinterpret_cast<const at::BFloat16*>(
          reinterpret_cast<const uint8_t*>(canonical_K_block_ptr)
          + c * k_row_stride_bytes);
      // Pair row p has elements (2p, 2p+1) of K row c.
      B_buf[p * 16 * 2 + c * 2 + 0] =
          *reinterpret_cast<const uint16_t*>(&row[2 * p]);
      B_buf[p * 16 * 2 + c * 2 + 1] =
          *reinterpret_cast<const uint16_t*>(&row[2 * p + 1]);
    }
  }
}

#endif  // VLLM_CPU_PARTIAL_HAS_AMX

// ---- AVX-512 dot product (BF16 fallback path used by FP16 / FP32 and
// when AMX path completes a 16-batch and we still have a < 16 tail) --

#if defined(__AVX512F__)
#include <immintrin.h>

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

template <typename T>
static inline float dot_avx512_kt(const T* a, const T* b, int64_t n);

template <>
inline float dot_avx512_kt<at::BFloat16>(const at::BFloat16* a,
                                         const at::BFloat16* b,
                                         int64_t n) {
  __m512 acc = _mm512_setzero_ps();
  int64_t d = 0;
  for (; d + 16 <= n; d += 16) {
    __m256i ai16 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(a + d));
    __m256i bi16 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(b + d));
    __m512 af = _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_cvtepu16_epi32(ai16), 16));
    __m512 bf = _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_cvtepu16_epi32(bi16), 16));
    acc = _mm512_fmadd_ps(af, bf, acc);
  }
  float result = hsum_ps_512(acc);
  for (; d < n; ++d) {
    result += static_cast<float>(a[d]) * static_cast<float>(b[d]);
  }
  return result;
}

template <>
inline float dot_avx512_kt<at::Half>(const at::Half* a, const at::Half* b,
                                     int64_t n) {
  __m512 acc = _mm512_setzero_ps();
  int64_t d = 0;
  for (; d + 16 <= n; d += 16) {
    __m256i ai16 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(a + d));
    __m256i bi16 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(b + d));
    acc = _mm512_fmadd_ps(_mm512_cvtph_ps(ai16),
                          _mm512_cvtph_ps(bi16), acc);
  }
  float result = hsum_ps_512(acc);
  for (; d < n; ++d) {
    result += static_cast<float>(a[d]) * static_cast<float>(b[d]);
  }
  return result;
}

template <>
inline float dot_avx512_kt<float>(const float* a, const float* b,
                                  int64_t n) {
  __m512 acc = _mm512_setzero_ps();
  int64_t d = 0;
  for (; d + 16 <= n; d += 16) {
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + d),
                          _mm512_loadu_ps(b + d), acc);
  }
  float result = hsum_ps_512(acc);
  for (; d < n; ++d) {
    result += a[d] * b[d];
  }
  return result;
}

#else
template <typename T>
static inline float dot_avx512_kt(const T* a, const T* b, int64_t n) {
  float acc = 0.0f;
  for (int64_t d = 0; d < n; ++d)
    acc += static_cast<float>(a[d]) * static_cast<float>(b[d]);
  return acc;
}
#endif

// IDE_006 / TSK_003 §4.2b — V weighted sum SIMD. Replaces the scalar
// ``out[d] += w * v_ptr[d]`` inner loop with AVX-512 fmadd against a
// broadcast scale, then BF16 / FP16 / FP32 specialisations match the
// dot product helper above. All three accumulate into an fp32 output
// buffer (``out[head_dim]``) regardless of the source dtype — the
// caller does the final fp32 → T cast once per token.

#if defined(__AVX512F__)
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

// ---- BF16-specialised kernel using AMX for the score batch ---------
//
// For BF16 we batch 16 K rows per AMX matmul. Tail (n_cold_kv % 16)
// uses the AVX-512 scalar dot path.

static std::vector<torch::Tensor> forward_partial_bf16_amx(
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
#if !VLLM_CPU_PARTIAL_HAS_AMX
  TORCH_CHECK(false,
              "AMX kernel built without __AMX_TILE__ / __AMX_BF16__ — "
              "rebuild with the proper compile flags.");
#else
  vllm_amx_trace("forward_partial_bf16_amx:enter");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(cold_kv_cache.is_contiguous(), "cold_kv_cache must be contiguous");
  TORCH_CHECK(cold_kv_cache.scalar_type() == torch::kInt8,
              "cold_kv_cache must be int8");
  TORCH_CHECK(query.dim() == 3, "query must be 3D");
  TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even for BF16 pair packing");

  const bool split_kv = cold_kv_cache_v.numel() > 0;
  if (split_kv) {
    TORCH_CHECK(cold_kv_cache_v.is_contiguous(),
                "cold_kv_cache_v must be contiguous");
    TORCH_CHECK(cold_kv_cache_v.scalar_type() == torch::kInt8,
                "cold_kv_cache_v must be int8");
    TORCH_CHECK(cold_kv_cache_v.size(0) == cold_kv_cache.size(0),
                "cold_kv_cache and cold_kv_cache_v must agree on num_blocks");
  }
  vllm_amx_trace("forward_partial_bf16_amx:torch_checks_passed");

  using T = at::BFloat16;
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

  // head_dim chunks of 32 BF16 elements per AMX tile call.
  const int chunk_bf16 = 32;
  const int n_chunks = static_cast<int>((head_dim + chunk_bf16 - 1) / chunk_bf16);
  TORCH_CHECK(head_dim % chunk_bf16 == 0,
              "head_dim must be a multiple of 32 for AMX BF16 (got ", head_dim, ")");

  // Configure AMX tiles for chunk_bf16 = 32 BF16 K dim per call.
  // Print kernel-level shape so we can correlate with the e2e workload
  // before the first AMX instruction fires.
  {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "forward_partial_bf16_amx:about_to_configure num_seqs=%lld "
             "num_tokens=%lld num_q_heads=%lld num_kv_heads=%lld "
             "head_dim=%lld block_size=%lld",
             static_cast<long long>(num_seqs),
             static_cast<long long>(num_tokens),
             static_cast<long long>(num_q_heads),
             static_cast<long long>(num_kv_heads),
             static_cast<long long>(head_dim),
             static_cast<long long>(block_size));
    vllm_amx_trace(buf);
  }
  // Tile config + scratch buffers (B_buf, C_buf, scores) are PER-THREAD
  // because AMX tile state and ``alignas(64)`` stack scratch are
  // per-thread and the OpenMP region below parallelises across
  // (token, head) pairs. We open the parallel region around the seq
  // loop so each thread does ``configure_tiles_for_dot`` exactly once
  // and reuses the same B_buf / C_buf / scores allocation across all
  // (s, t, h) iterations it executes.

  for (int64_t s = 0; s < num_seqs; ++s) {
    const int64_t q_start = cu_seqlens_q_a[s];
    const int64_t q_end = cu_seqlens_q_a[s + 1];
    const int64_t n_cold_blocks = cold_block_lens_a[s];
    if (q_end <= q_start || n_cold_blocks <= 0) continue;
    const int64_t n_cold_kv = n_cold_blocks * block_size;

    #pragma omp parallel default(none) \
        firstprivate(q_start, q_end, n_cold_kv, n_cold_blocks, num_q_heads, \
                     q_per_kv, num_kv_heads, head_dim, block_size, \
                     k_block_stride_elems, v_block_stride_elems, \
                     v_intra_block_offset_elems, scale_f, NEG_INF, \
                     causal, n_chunks, chunk_bf16, k_data, v_data, s) \
        shared(query_a, O_a, LSE_a, cold_block_ids_a, query_positions_a)
    {
      configure_tiles_for_dot(chunk_bf16);
      alignas(64) uint16_t B_buf[16 * 32];
      alignas(64) float C_buf[16];
      std::vector<float> scores(static_cast<size_t>(n_cold_kv));

      #pragma omp for collapse(2) schedule(static) nowait
      for (int64_t t = q_start; t < q_end; ++t) {
        for (int64_t h = 0; h < num_q_heads; ++h) {
      const int64_t q_pos = query_positions_a[t];

      {
        // Body kept indented inside this brace block to minimise diff
        // surface; the original (s, t, h) loop body follows verbatim.
        (void)0;
        const int64_t kv_h = h / q_per_kv;
        const T* q_ptr = &query_a[t][h][0];

        // ---- Pass 1: scores + max ----
        float m_val = NEG_INF;

        // Process 16 cold-KV rows at a time using AMX.
        int64_t k = 0;
        for (; k + 16 <= n_cold_kv; k += 16) {
          // Check causal mask coverage: if any of the 16 rows is masked
          // out (q_pos < k+i), fall back to scalar for this batch to
          // keep the masking logic identical to portable.
          bool any_masked = false;
          if (causal) {
            for (int i = 0; i < 16; ++i) {
              if (q_pos < k + i) { any_masked = true; break; }
            }
          }
          if (any_masked) {
            for (int i = 0; i < 16; ++i) {
              const int64_t kk = k + i;
              const int64_t bidx = kk / block_size;
              const int64_t tib = kk % block_size;
              const int64_t real_block_id = cold_block_ids_a[s][bidx];
              if (causal && q_pos < kk) {
                scores[kk] = NEG_INF;
                continue;
              }
              const T* k_ptr = k_data
                  + real_block_id * k_block_stride_elems
                  + tib * num_kv_heads * head_dim
                  + kv_h * head_dim;
              const float dot = dot_avx512_kt<T>(q_ptr, k_ptr, head_dim);
              const float score = dot * scale_f;
              scores[kk] = score;
              if (score > m_val) m_val = score;
            }
            continue;
          }

          // Zero accumulator tile. First batch only — log so we know the
          // first AMX op completed without SIGILL.
          vllm_amx_trace("about_to_tile_zero");
          _tile_zero(VLLM_TMM_C);
          vllm_amx_trace("tile_zero_returned");

          // For each chunk of 32 BF16 (= 16 BF16 pairs) in head_dim,
          // load A and B tiles and accumulate.
          for (int chunk = 0; chunk < n_chunks; ++chunk) {
            const int d0 = chunk * chunk_bf16;

            // A tile: 1 × chunk_bf16 BF16 = (chunk_bf16 * 2) bytes.
            // We can load directly from query memory; contiguous already.
            vllm_amx_trace("about_to_tile_loadd_A");
            _tile_loadd(
                VLLM_TMM_A,
                reinterpret_cast<const void*>(q_ptr + d0),
                chunk_bf16 * 2  // stride for a 1-row tile: row size in bytes
            );
            vllm_amx_trace("tile_loadd_A_returned");

            // Stage 16 K rows × chunk_bf16 BF16 into B_buf in pair-row
            // layout. K row stride = num_kv_heads * head_dim * 2 bytes.
            // Compute the K block base for this batch of 16 rows. The
            // 16 rows are k..k+15, each at (block, token_in_block) =
            // ((k+i)/block_size, (k+i)%block_size). They share the same
            // block IFF k % block_size == 0 AND k+15 < (block+1)*block_size,
            // which holds when block_size is a multiple of 16 — true for
            // PLN_001 §3 scope (block_size ∈ {16, 32, 64}).
            // For safety we iterate per-row anyway.
            for (int i = 0; i < 16; ++i) {
              const int64_t kk = k + i;
              const int64_t bidx = kk / block_size;
              const int64_t tib = kk % block_size;
              const int64_t real_block_id = cold_block_ids_a[s][bidx];
              const at::BFloat16* row = k_data
                  + real_block_id * k_block_stride_elems
                  + tib * num_kv_heads * head_dim
                  + kv_h * head_dim
                  + d0;  // chunk offset
              // Pack into B_buf at column i of every pair-row.
              for (int p = 0; p < chunk_bf16 / 2; ++p) {
                B_buf[p * 16 * 2 + i * 2 + 0] =
                    *reinterpret_cast<const uint16_t*>(&row[2 * p]);
                B_buf[p * 16 * 2 + i * 2 + 1] =
                    *reinterpret_cast<const uint16_t*>(&row[2 * p + 1]);
              }
            }

            // Load B tile and dpbf16ps accumulate into C.
            vllm_amx_trace("about_to_tile_loadd_B");
            _tile_loadd(VLLM_TMM_B, B_buf, 16 * 4);  // 64-byte stride per pair-row
            vllm_amx_trace("tile_loadd_B_returned");
            vllm_amx_trace("about_to_tile_dpbf16ps");
            _tile_dpbf16ps(VLLM_TMM_C, VLLM_TMM_A, VLLM_TMM_B);
            vllm_amx_trace("tile_dpbf16ps_returned");
          }

          // Read C tile (1 × 16 fp32) into C_buf.
          vllm_amx_trace("about_to_tile_stored");
          _tile_stored(VLLM_TMM_C, C_buf, 16 * 4);
          vllm_amx_trace("tile_stored_returned");

          // Apply softmax_scale and update scores + max.
          for (int i = 0; i < 16; ++i) {
            const float score = C_buf[i] * scale_f;
            scores[k + i] = score;
            if (score > m_val) m_val = score;
          }
        }

        // Tail (n_cold_kv % 16) — scalar AVX-512 dot.
        for (; k < n_cold_kv; ++k) {
          const int64_t bidx = k / block_size;
          const int64_t tib = k % block_size;
          const int64_t real_block_id = cold_block_ids_a[s][bidx];
          if (causal && q_pos < k) {
            scores[k] = NEG_INF;
            continue;
          }
          const T* k_ptr = k_data
              + real_block_id * k_block_stride_elems
              + tib * num_kv_heads * head_dim
              + kv_h * head_dim;
          const float dot = dot_avx512_kt<T>(q_ptr, k_ptr, head_dim);
          const float score = dot * scale_f;
          scores[k] = score;
          if (score > m_val) m_val = score;
        }

        if (m_val == NEG_INF) continue;

        // ---- Pass 2: exp(score - m) ----
        float sum_exp = 0.0f;
        for (int64_t kk = 0; kk < n_cold_kv; ++kk) {
          if (scores[kk] == NEG_INF) {
            scores[kk] = 0.0f;
            continue;
          }
          const float ex = std::exp(scores[kk] - m_val);
          scores[kk] = ex;
          sum_exp += ex;
        }

        // ---- Pass 3: weighted V sum — AVX-512 fmadd over head_dim,
        // accumulating into the fp32 ``out`` buffer. The scalar fall-
        // through inside ``v_fmadd_avx512`` handles head_dim tails.
        float out[1024];
        TORCH_CHECK(head_dim <= 1024, "head_dim too large for stack buffer");
        for (int64_t d = 0; d < head_dim; ++d) out[d] = 0.0f;

        const float inv_sum = 1.0f / sum_exp;
        for (int64_t kk = 0; kk < n_cold_kv; ++kk) {
          const float w = scores[kk] * inv_sum;
          if (w == 0.0f) continue;
          const int64_t bidx = kk / block_size;
          const int64_t tib = kk % block_size;
          const int64_t real_block_id = cold_block_ids_a[s][bidx];
          const at::BFloat16* v_ptr = v_data
              + real_block_id * v_block_stride_elems
              + v_intra_block_offset_elems
              + tib * num_kv_heads * head_dim
              + kv_h * head_dim;
          v_fmadd_avx512<at::BFloat16>(w, v_ptr, out, head_dim);
        }

        for (int64_t d = 0; d < head_dim; ++d) {
          O_a[t][h][d] = static_cast<T>(out[d]);
        }
        LSE_a[h][t] = m_val + std::log(sum_exp);
      }    // close (void)0 body block
        }  // close inner ``for h`` (omp for collapse=2)
      }    // close ``for t``
      // Each OpenMP thread releases its own AMX tile state before
      // exiting the parallel region.
      vllm_amx_trace("about_to_tile_release");
      _tile_release();
    }      // close ``#pragma omp parallel``
  }        // close ``for s``

  vllm_amx_trace("forward_partial_bf16_amx:exit");

  return {O, LSE};
#endif
}

// FP16 / FP32 / unsupported dtypes route to a scalar AVX-512 path
// (algorithmically identical to the AVX-512 kernel). AMX BF16 doesn't
// help non-BF16 inputs, so this path is the natural fallback.
template <typename T>
static std::vector<torch::Tensor> forward_partial_avx512_fallback(
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
  const int64_t q_per_kv = num_q_heads / num_kv_heads;
  const int64_t num_seqs = cu_seqlens_q.size(0) - 1;
  const int64_t page_size_bytes = cold_kv_cache.size(1);
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
        const T* q_ptr = &query_a[t][h][0];

        float m_val = NEG_INF;
        for (int64_t k = 0; k < n_cold_kv; ++k) {
          const int64_t bidx = k / block_size;
          const int64_t tib = k % block_size;
          const int64_t real_block_id = cold_block_ids_a[s][bidx];
          if (causal && q_pos < k) { scores[k] = NEG_INF; continue; }
          const T* k_ptr = k_data + real_block_id * k_block_stride_elems
              + tib * num_kv_heads * head_dim + kv_h * head_dim;
          const float dot = dot_avx512_kt<T>(q_ptr, k_ptr, head_dim);
          const float score = dot * scale_f;
          scores[k] = score;
          if (score > m_val) m_val = score;
        }
        if (m_val == NEG_INF) continue;
        float sum_exp = 0.0f;
        for (int64_t k = 0; k < n_cold_kv; ++k) {
          if (scores[k] == NEG_INF) { scores[k] = 0.0f; continue; }
          const float ex = std::exp(scores[k] - m_val);
          scores[k] = ex;
          sum_exp += ex;
        }
        float out[1024];
        TORCH_CHECK(head_dim <= 1024, "head_dim too large for stack buffer");
        for (int64_t d = 0; d < head_dim; ++d) out[d] = 0.0f;
        const float inv_sum = 1.0f / sum_exp;
        for (int64_t k = 0; k < n_cold_kv; ++k) {
          const float w = scores[k] * inv_sum;
          if (w == 0.0f) continue;
          const int64_t bidx = k / block_size;
          const int64_t tib = k % block_size;
          const int64_t real_block_id = cold_block_ids_a[s][bidx];
          const T* v_ptr = v_data + real_block_id * v_block_stride_elems
              + v_intra_block_offset_elems + tib * num_kv_heads * head_dim
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

std::vector<torch::Tensor> forward_partial_with_lse_amx(
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
    return forward_partial_bf16_amx(
        query, cold_kv_cache, cold_kv_cache_v, block_size, num_kv_heads,
        head_dim, kv_block_bytes, cold_block_ids, cold_block_lens,
        cu_seqlens_q, query_positions, softmax_scale, causal);
  } else if (dt == torch::kHalf) {
    return forward_partial_avx512_fallback<at::Half>(
        query, cold_kv_cache, cold_kv_cache_v, block_size, num_kv_heads,
        head_dim, kv_block_bytes, cold_block_ids, cold_block_lens,
        cu_seqlens_q, query_positions, softmax_scale, causal);
  } else if (dt == torch::kFloat) {
    return forward_partial_avx512_fallback<float>(
        query, cold_kv_cache, cold_kv_cache_v, block_size, num_kv_heads,
        head_dim, kv_block_bytes, cold_block_ids, cold_block_lens,
        cu_seqlens_q, query_positions, softmax_scale, causal);
  } else {
    TORCH_CHECK(false,
                "forward_partial_with_lse_amx: unsupported dtype ", dt);
  }
}

}  // namespace cpu_partial_attn_amx

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_partial_with_lse_amx",
        &cpu_partial_attn_amx::forward_partial_with_lse_amx,
        "Cold-KV CPU Partial Attention (AMX BF16 + AVX-512 fallback)");
}
