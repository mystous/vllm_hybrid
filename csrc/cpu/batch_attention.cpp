// Batch Attention Optimization for AVX-512
//
// Groups up to 16 sequences per batch for improved cache locality.
// Within each batch, sequences are iterated sequentially; AVX-512's
// 16-wide SIMD lanes are applied to the head-dimension vector
// operations (Q*K dot product, V weighted sum) within each sequence.
//
// Key optimizations:
// - OpenMP parallel across (batch_idx, head_idx) pairs for multi-core
// - AVX-512 FMA for head-dimension dot products (16 floats at a time)
// - KV cache L2 prefetching via _MM_HINT_T1 for next-block lookahead

#include <immintrin.h>
#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef __AVX512F__

namespace {

constexpr int BATCH16 = 16;

// ============================================================================
// BF16 to FP32 conversion (no AVX512BF16 needed)
// ============================================================================
inline __m512 bf16x16_to_fp32_attn(__m256i bf16_vals) {
  __m512i expanded = _mm512_cvtepu16_epi32(bf16_vals);
  __m512i shifted = _mm512_slli_epi32(expanded, 16);
  return _mm512_castsi512_ps(shifted);
}

// FP32 to BF16 with round-to-nearest-even
inline __m256i fp32x16_to_bf16(__m512 fp32_vals) {
  __m512i fi = _mm512_castps_si512(fp32_vals);
  __m512i round_bias = _mm512_set1_epi32(0x00008000);
  fi = _mm512_add_epi32(fi, round_bias);
  __m512i shifted = _mm512_srli_epi32(fi, 16);
  return _mm512_cvtepi32_epi16(shifted);
}

}  // anonymous namespace

// ============================================================================
// Batch-16 Paged Attention V1
// Processes up to 16 sequences simultaneously
//
// When num_seqs >= 16, groups sequences into batches of 16 and processes
// them together. Remaining sequences are processed individually.
// ============================================================================
void batch16_paged_attention_v1(
    torch::Tensor& output,       // [num_seqs, num_heads, head_size]
    const torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    const torch::Tensor& key_cache,    // [num_blocks, num_kv_heads, D/x, B, x]
    const torch::Tensor& value_cache,  // [num_blocks, num_kv_heads, D, B]
    const torch::Tensor& block_tables,    // [num_seqs, max_blocks_per_seq]
    const torch::Tensor& context_lens,    // [num_seqs]
    int num_heads, int head_size, int block_size, int max_blocks_per_seq,
    float scale, int num_kv_heads) {
  TORCH_CHECK(query.dtype() == torch::kBFloat16 ||
                  query.dtype() == torch::kFloat32,
              "batch16_paged_attention: only BF16 and FP32 supported");

  const int num_seqs = query.size(0);
  const int num_queries_per_kv = num_heads / num_kv_heads;

  const int* block_tables_ptr = block_tables.data_ptr<int>();
  const int* context_lens_ptr = context_lens.data_ptr<int>();

  const int kv_block_stride = key_cache.stride(0);
  const int kv_head_stride = key_cache.stride(1);

  // Process sequences in groups of BATCH16
  const int num_full_batches = num_seqs / BATCH16;
  const int remaining_seqs = num_seqs % BATCH16;

  if (query.dtype() == torch::kBFloat16) {
    const c10::BFloat16* q_ptr = query.data_ptr<c10::BFloat16>();
    const c10::BFloat16* k_ptr = key_cache.data_ptr<c10::BFloat16>();
    const c10::BFloat16* v_ptr = value_cache.data_ptr<c10::BFloat16>();
    c10::BFloat16* out_ptr = output.data_ptr<c10::BFloat16>();

    const int q_stride = query.stride(0);

    // Allocate per-thread logits buffer
    const int max_seq_len =
        *std::max_element(context_lens_ptr, context_lens_ptr + num_seqs);
    const int max_seq_len_padded = (max_seq_len + 15) & ~15;

    const int num_threads = omp_get_max_threads();
    const size_t logits_per_thread = max_seq_len_padded;

    // Each thread needs BATCH16 logit buffers (one per sequence in batch)
    float* logits_buf = static_cast<float*>(
        std::aligned_alloc(64, num_threads * BATCH16 * logits_per_thread *
                                   sizeof(float)));

    // Process full batches of 16 sequences
#pragma omp parallel for schedule(dynamic, 1)
    for (int batch_head_idx = 0;
         batch_head_idx < num_full_batches * num_heads; ++batch_head_idx) {
      const int batch_idx = batch_head_idx / num_heads;
      const int head_idx = batch_head_idx % num_heads;
      const int kv_head_idx = head_idx / num_queries_per_kv;
      const int seq_base = batch_idx * BATCH16;
      const int thread_id = omp_get_thread_num();

      // For each of the 16 sequences in this batch, compute attention
      // We process all 16 sequences for this head in parallel

      for (int s = 0; s < BATCH16; ++s) {
        const int seq_idx = seq_base + s;
        const int seq_len = context_lens_ptr[seq_idx];
        const int block_num = (seq_len + block_size - 1) / block_size;
        const int last_block_tokens = seq_len - (block_num - 1) * block_size;

        const c10::BFloat16* q_vec =
            q_ptr + seq_idx * q_stride + head_idx * head_size;
        const int* seq_block_table =
            block_tables_ptr + seq_idx * max_blocks_per_seq;

        float* seq_logits =
            logits_buf +
            (thread_id * BATCH16 + s) * logits_per_thread;

        // Compute QK logits for all blocks
        for (int bi = 0; bi < block_num; ++bi) {
          const int physical_block = seq_block_table[bi];
          const c10::BFloat16* k_block =
              k_ptr + physical_block * kv_block_stride +
              kv_head_idx * kv_head_stride;

          // Prefetch next block's K data to L2
          if (bi + 1 < block_num) {
            const char* next_k = reinterpret_cast<const char*>(
                k_ptr + seq_block_table[bi + 1] * kv_block_stride +
                kv_head_idx * kv_head_stride);
            _mm_prefetch(next_k, _MM_HINT_T1);
            _mm_prefetch(next_k + 64, _MM_HINT_T1);
            _mm_prefetch(next_k + 128, _MM_HINT_T1);
            _mm_prefetch(next_k + 192, _MM_HINT_T1);
          }

          const int tokens_in_block =
              (bi == block_num - 1) ? last_block_tokens : block_size;

          // Compute dot product for each token in this block
          for (int t = 0; t < tokens_in_block; ++t) {
            float dot = 0.0f;
            __m512 acc = _mm512_setzero_ps();

            const int x = 16 / sizeof(c10::BFloat16);  // 8

            // Q * K dot product
            for (int d = 0; d < head_size; d += 16) {
              __m256i vq = _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(q_vec + d));
              __m512 fq = bf16x16_to_fp32_attn(vq);

              // K cache layout: [D/x, block_size, x]
              // For element d, token t:
              //   k_block[(d/x) * block_size * x + t * x + (d % x)]
              // We need to gather K values for consecutive d dimensions
              // This is complex due to the blocked layout

              // Gather K values for 16 consecutive head dimensions
              alignas(64) float k_vals[16] = {0.0f};
              for (int dd = 0; dd < 16 && (d + dd) < head_size; ++dd) {
                const int d_idx = d + dd;
                const int d_outer = d_idx / x;
                const int d_inner = d_idx % x;
                const int k_offset =
                    d_outer * block_size * x + t * x + d_inner;
                k_vals[dd] =
                    static_cast<float>(k_block[k_offset]);
              }
              __m512 fk = _mm512_loadu_ps(k_vals);
              acc = _mm512_fmadd_ps(fq, fk, acc);
            }
            dot = _mm512_reduce_add_ps(acc) * scale;
            seq_logits[bi * block_size + t] = dot;
          }

          // Zero-pad remaining slots in block
          for (int t = tokens_in_block; t < block_size; ++t) {
            seq_logits[bi * block_size + t] = -std::numeric_limits<float>::infinity();
          }
        }

        // Softmax over logits
        const int total_tokens = block_num * block_size;
        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < seq_len; ++i) {
          max_val = std::max(max_val, seq_logits[i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
          seq_logits[i] = std::exp(seq_logits[i] - max_val);
          sum += seq_logits[i];
        }
        if (sum > 0.0f) {
          float inv_sum = 1.0f / sum;
          for (int i = 0; i < seq_len; ++i) {
            seq_logits[i] *= inv_sum;
          }
        }
        for (int i = seq_len; i < total_tokens; ++i) {
          seq_logits[i] = 0.0f;
        }

        // Compute weighted value sum
        c10::BFloat16* out_vec =
            out_ptr + seq_idx * num_heads * head_size + head_idx * head_size;

        for (int d = 0; d < head_size; ++d) {
          float val_acc = 0.0f;
          for (int bi = 0; bi < block_num; ++bi) {
            const int physical_block = seq_block_table[bi];
            const c10::BFloat16* v_block =
                v_ptr + physical_block * kv_block_stride +
                kv_head_idx * kv_head_stride;

            // Prefetch next block's V data to L2
            if (bi + 1 < block_num) {
              const char* next_v = reinterpret_cast<const char*>(
                  v_ptr + seq_block_table[bi + 1] * kv_block_stride +
                  kv_head_idx * kv_head_stride);
              _mm_prefetch(next_v, _MM_HINT_T1);
              _mm_prefetch(next_v + 64, _MM_HINT_T1);
              _mm_prefetch(next_v + 128, _MM_HINT_T1);
              _mm_prefetch(next_v + 192, _MM_HINT_T1);
            }

            const int tokens_in_block =
                (bi == block_num - 1) ? last_block_tokens : block_size;

            // V cache layout: [D, block_size]
            for (int t = 0; t < tokens_in_block; ++t) {
              float prob = seq_logits[bi * block_size + t];
              float v_val =
                  static_cast<float>(v_block[d * block_size + t]);
              val_acc += prob * v_val;
            }
          }
          out_vec[d] = static_cast<c10::BFloat16>(val_acc);
        }
      }
    }

    // Process remaining sequences individually
    for (int s = 0; s < remaining_seqs; ++s) {
      const int seq_idx = num_full_batches * BATCH16 + s;
      const int seq_len = context_lens_ptr[seq_idx];
      const int block_num = (seq_len + block_size - 1) / block_size;
      const int last_block_tokens = seq_len - (block_num - 1) * block_size;
      const int* seq_block_table =
          block_tables_ptr + seq_idx * max_blocks_per_seq;

#pragma omp parallel for schedule(dynamic, 1)
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        const int kv_head_idx = head_idx / num_queries_per_kv;
        const c10::BFloat16* q_vec =
            q_ptr + seq_idx * q_stride + head_idx * head_size;
        const int thread_id = omp_get_thread_num();
        float* seq_logits =
            logits_buf + thread_id * BATCH16 * logits_per_thread;

        const int x = 16 / sizeof(c10::BFloat16);

        // QK
        for (int bi = 0; bi < block_num; ++bi) {
          const int physical_block = seq_block_table[bi];
          const c10::BFloat16* k_block =
              k_ptr + physical_block * kv_block_stride +
              kv_head_idx * kv_head_stride;

          // Prefetch next block's K data to L2
          if (bi + 1 < block_num) {
            const char* next_k = reinterpret_cast<const char*>(
                k_ptr + seq_block_table[bi + 1] * kv_block_stride +
                kv_head_idx * kv_head_stride);
            _mm_prefetch(next_k, _MM_HINT_T1);
            _mm_prefetch(next_k + 64, _MM_HINT_T1);
            _mm_prefetch(next_k + 128, _MM_HINT_T1);
            _mm_prefetch(next_k + 192, _MM_HINT_T1);
          }

          const int tokens_in_block =
              (bi == block_num - 1) ? last_block_tokens : block_size;

          for (int t = 0; t < tokens_in_block; ++t) {
            __m512 acc = _mm512_setzero_ps();
            for (int d = 0; d < head_size; d += 16) {
              __m256i vq = _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(q_vec + d));
              __m512 fq = bf16x16_to_fp32_attn(vq);

              alignas(64) float k_vals[16] = {0.0f};
              for (int dd = 0; dd < 16 && (d + dd) < head_size; ++dd) {
                const int d_idx = d + dd;
                const int d_outer = d_idx / x;
                const int d_inner = d_idx % x;
                k_vals[dd] = static_cast<float>(
                    k_block[d_outer * block_size * x + t * x + d_inner]);
              }
              __m512 fk = _mm512_loadu_ps(k_vals);
              acc = _mm512_fmadd_ps(fq, fk, acc);
            }
            seq_logits[bi * block_size + t] =
                _mm512_reduce_add_ps(acc) * scale;
          }
          for (int t = tokens_in_block; t < block_size; ++t) {
            seq_logits[bi * block_size + t] =
                -std::numeric_limits<float>::infinity();
          }
        }

        // Softmax
        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < seq_len; ++i)
          max_val = std::max(max_val, seq_logits[i]);
        float sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
          seq_logits[i] = std::exp(seq_logits[i] - max_val);
          sum += seq_logits[i];
        }
        if (sum > 0.0f) {
          float inv_sum = 1.0f / sum;
          for (int i = 0; i < seq_len; ++i) seq_logits[i] *= inv_sum;
        }

        // Value
        c10::BFloat16* out_vec =
            out_ptr + seq_idx * num_heads * head_size + head_idx * head_size;
        for (int d = 0; d < head_size; ++d) {
          float val_acc = 0.0f;
          for (int bi = 0; bi < block_num; ++bi) {
            const int physical_block = seq_block_table[bi];
            const c10::BFloat16* v_block =
                v_ptr + physical_block * kv_block_stride +
                kv_head_idx * kv_head_stride;

            // Prefetch next block's V data to L2
            if (bi + 1 < block_num) {
              const char* next_v = reinterpret_cast<const char*>(
                  v_ptr + seq_block_table[bi + 1] * kv_block_stride +
                  kv_head_idx * kv_head_stride);
              _mm_prefetch(next_v, _MM_HINT_T1);
              _mm_prefetch(next_v + 64, _MM_HINT_T1);
              _mm_prefetch(next_v + 128, _MM_HINT_T1);
              _mm_prefetch(next_v + 192, _MM_HINT_T1);
            }

            const int tokens_in_block =
                (bi == block_num - 1) ? last_block_tokens : block_size;
            for (int t = 0; t < tokens_in_block; ++t) {
              val_acc +=
                  seq_logits[bi * block_size + t] *
                  static_cast<float>(v_block[d * block_size + t]);
            }
          }
          out_vec[d] = static_cast<c10::BFloat16>(val_acc);
        }
      }
    }

    std::free(logits_buf);
  } else {
    // FP32 path - similar structure but with float pointers
    const float* q_ptr = query.data_ptr<float>();
    const float* k_ptr = key_cache.data_ptr<float>();
    const float* v_ptr = value_cache.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    const int q_stride = query.stride(0);
    const int max_seq_len =
        *std::max_element(context_lens_ptr, context_lens_ptr + num_seqs);
    const int max_seq_len_padded = (max_seq_len + 15) & ~15;
    const int num_threads = omp_get_max_threads();

    float* logits_buf = static_cast<float*>(
        std::aligned_alloc(64, num_threads * max_seq_len_padded *
                                   sizeof(float)));

#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        const int kv_head_idx = head_idx / num_queries_per_kv;
        const int seq_len = context_lens_ptr[seq_idx];
        const int block_num = (seq_len + block_size - 1) / block_size;
        const int last_block_tokens = seq_len - (block_num - 1) * block_size;
        const int* seq_block_table =
            block_tables_ptr + seq_idx * max_blocks_per_seq;
        const float* q_vec =
            q_ptr + seq_idx * q_stride + head_idx * head_size;
        const int thread_id = omp_get_thread_num();
        float* seq_logits = logits_buf + thread_id * max_seq_len_padded;

        const int x = 16 / sizeof(float);  // 4

        // QK
        for (int bi = 0; bi < block_num; ++bi) {
          const int physical_block = seq_block_table[bi];
          const float* k_block =
              k_ptr + physical_block * kv_block_stride +
              kv_head_idx * kv_head_stride;

          // Prefetch next block's K data to L2
          if (bi + 1 < block_num) {
            const char* next_k = reinterpret_cast<const char*>(
                k_ptr + seq_block_table[bi + 1] * kv_block_stride +
                kv_head_idx * kv_head_stride);
            _mm_prefetch(next_k, _MM_HINT_T1);
            _mm_prefetch(next_k + 64, _MM_HINT_T1);
            _mm_prefetch(next_k + 128, _MM_HINT_T1);
            _mm_prefetch(next_k + 192, _MM_HINT_T1);
          }

          const int tokens = (bi == block_num - 1) ? last_block_tokens : block_size;

          for (int t = 0; t < tokens; ++t) {
            __m512 acc = _mm512_setzero_ps();
            for (int d = 0; d < head_size; d += 16) {
              __m512 vq = _mm512_loadu_ps(q_vec + d);
              alignas(64) float k_vals[16] = {0.0f};
              for (int dd = 0; dd < 16 && (d + dd) < head_size; ++dd) {
                const int d_idx = d + dd;
                const int d_outer = d_idx / x;
                const int d_inner = d_idx % x;
                k_vals[dd] = k_block[d_outer * block_size * x + t * x + d_inner];
              }
              __m512 fk = _mm512_loadu_ps(k_vals);
              acc = _mm512_fmadd_ps(vq, fk, acc);
            }
            seq_logits[bi * block_size + t] =
                _mm512_reduce_add_ps(acc) * scale;
          }
          for (int t = tokens; t < block_size; ++t) {
            seq_logits[bi * block_size + t] =
                -std::numeric_limits<float>::infinity();
          }
        }

        // Softmax
        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < seq_len; ++i)
          max_val = std::max(max_val, seq_logits[i]);
        float sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
          seq_logits[i] = std::exp(seq_logits[i] - max_val);
          sum += seq_logits[i];
        }
        if (sum > 0.0f) {
          float inv_sum = 1.0f / sum;
          for (int i = 0; i < seq_len; ++i) seq_logits[i] *= inv_sum;
        }

        // Value
        float* out_vec =
            out_ptr + seq_idx * num_heads * head_size + head_idx * head_size;
        for (int d = 0; d < head_size; ++d) {
          float val_acc = 0.0f;
          for (int bi = 0; bi < block_num; ++bi) {
            const int physical_block = seq_block_table[bi];
            const float* v_block =
                v_ptr + physical_block * kv_block_stride +
                kv_head_idx * kv_head_stride;

            // Prefetch next block's V data to L2
            if (bi + 1 < block_num) {
              const char* next_v = reinterpret_cast<const char*>(
                  v_ptr + seq_block_table[bi + 1] * kv_block_stride +
                  kv_head_idx * kv_head_stride);
              _mm_prefetch(next_v, _MM_HINT_T1);
              _mm_prefetch(next_v + 64, _MM_HINT_T1);
              _mm_prefetch(next_v + 128, _MM_HINT_T1);
              _mm_prefetch(next_v + 192, _MM_HINT_T1);
            }

            const int tokens = (bi == block_num - 1) ? last_block_tokens : block_size;
            for (int t = 0; t < tokens; ++t) {
              val_acc += seq_logits[bi * block_size + t] *
                         v_block[d * block_size + t];
            }
          }
          out_vec[d] = val_acc;
        }
      }
    }

    std::free(logits_buf);
  }
}

#endif  // __AVX512F__
