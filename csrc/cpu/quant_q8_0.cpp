// Q8_0 Quantization Kernels for AVX-512 VNNI
//
// Q8_0 format (compatible with llama.cpp):
//   Block size = 32 elements
//   Each block: FP16 scale (2 bytes) + int8 quants[32] (32 bytes) = 34 bytes
//
// This implementation provides:
// - Q8_0 GEMV using VNNI instructions (decode path, M=1)
// - Q8_0 Linear for small batch sizes (M<=32)
// - Block-level scale application for accuracy

#include <immintrin.h>
#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__AVX512F__) && defined(__AVX512VNNI__)

namespace {

// Q8_0 block structure (llama.cpp compatible)
constexpr int Q8_0_BLOCK_SIZE = 32;

struct Q8_0Block {
  uint16_t scale_fp16;    // FP16 scale factor
  int8_t quants[Q8_0_BLOCK_SIZE];  // Quantized values
};
static_assert(sizeof(Q8_0Block) == 34, "Q8_0Block must be 34 bytes");

// Convert FP16 (IEEE 754 half) to FP32
inline float fp16_to_fp32(uint16_t h) {
  // Use F16C instruction
  __m128i vh = _mm_set1_epi16(static_cast<short>(h));
  __m128 vf = _mm_cvtph_ps(vh);
  return _mm_cvtss_f32(vf);
}

// Convert FP32 to FP16
inline uint16_t fp32_to_fp16(float f) {
  __m128 vf = _mm_set_ss(f);
  __m128i vh = _mm_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  return static_cast<uint16_t>(_mm_extract_epi16(vh, 0));
}

// ============================================================================
// Q8_0 GEMV: y[n] = sum_k(x_quant[k] * W_q8_0[n][k]) * x_scale * W_scale[n]
// Uses per-block scales for accuracy
// ============================================================================
void q8_0_gemv_vnni_impl(const int8_t* x_quant, float x_scale,
                         const Q8_0Block* weight, float* output,
                         int N, int K) {
  const int n_blocks_per_row = (K + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;

#pragma omp parallel for schedule(static)
  for (int n = 0; n < N; ++n) {
    const Q8_0Block* w_row = weight + n * n_blocks_per_row;

    // Accumulate in FP32 with per-block scale application
    float acc = 0.0f;

    for (int b = 0; b < n_blocks_per_row; ++b) {
      const Q8_0Block& block = w_row[b];
      const float w_scale = fp16_to_fp32(block.scale_fp16);
      const int k_start = b * Q8_0_BLOCK_SIZE;
      const int k_end = std::min(k_start + Q8_0_BLOCK_SIZE, K);
      const int k_len = k_end - k_start;

      // INT32 dot product using VNNI
      int32_t block_acc = 0;

      if (k_len == Q8_0_BLOCK_SIZE) {
        // Full block: use AVX-512 VNNI
        // Load 32 int8 values from x and weight
        // VNNI processes 4 int8 pairs per element, 16 elements per register
        // For 32 elements, we need 2 iterations of 16

        // Method: use _mm512_dpbusd_epi32 with u8*s8
        // We need x as u8 and w as s8 (or vice versa)
        // Since both are s8, use the s8s8 compensation trick:
        // x_u8 = x_s8 + 128, then subtract compensation

        // Pack 32 elements into 256 bits (2 x 128-bit halves)
        // But VNNI works on 512-bit, so pad to 64 elements

        // Simple approach: two 16-element groups
        __m512i vx_lo, vx_hi, vw_lo, vw_hi;
        __m512i acc_lo = _mm512_setzero_si512();

        // Load 32 bytes of x, zero-extend to fill 512 bits
        // We'll process in chunks of 16 int8 elements (64 bytes of packed data)
        // Actually, VNNI dpbusd processes 4 bytes at a time across 16 lanes

        // Simpler approach: scalar + VNNI for 32 elements
        // Load first 16 quads (16 lanes x 4 bytes = 64 bytes)
        // But we only have 32 elements, so use two passes of 16

        // Two passes of 16 elements each, using dpbusd on 4-element groups
        // Pass 1: elements 0-15
        {
          // Prepare: pack 16 x values into positions [0..3] of each lane
          // We have 16 elements, dpbusd expects 4 bytes per lane
          // So we'd need 4 elements per lane = 16/4 = 4 active lanes
          // This doesn't map well. Let's use a simpler approach.

          // Alternative: multiply-add manually using madd
          __m256i vx_256 = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(x_quant + k_start));
          __m256i vw_256 = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(block.quants));

          // Sign-extend to 16-bit and multiply
          __m512i vx_16 = _mm512_cvtepi8_epi16(vx_256);
          __m512i vw_16 = _mm512_cvtepi8_epi16(vw_256);
          __m512i vprod = _mm512_madd_epi16(vx_16, vw_16);

          block_acc = _mm512_reduce_add_epi32(vprod);
        }
      } else {
        // Partial block: scalar fallback
        for (int k = 0; k < k_len; ++k) {
          block_acc += static_cast<int32_t>(x_quant[k_start + k]) *
                       static_cast<int32_t>(block.quants[k]);
        }
      }

      // Apply per-block scale
      acc += static_cast<float>(block_acc) * w_scale * x_scale;
    }

    output[n] = acc;
  }
}

// ============================================================================
// Q8_0 GEMM (batch-aware, small M) — §06-1 Phase 1
//
// Scope: 1 < M < 16.
//
// Why this exists: the baseline `q8_0_linear_impl` calls `q8_0_gemv_vnni_impl`
// once per m row. That reloads every weight block M times from DDR, making
// wall time linear in M. This GEMM kernel flips the loop so that each weight
// block is loaded **once** and then dot-producted against all M activation
// rows while it sits in registers/L1. This amortizes weight DDR BW across
// the batch and restores the expected weight-BW-halved advantage of Q8_0
// for M > 1.
//
// Loop structure:
//   n (output ch)      — parallel
//     b (K block)      — weight block loaded once per (n, b)
//       m              — M activation rows share the same weight block
//
// Layout assumptions (same as q8_0_gemv_vnni_impl):
//   x_quant : [M, K] INT8 row-major, per-row dynamic quant by x_scales
//   weight  : [N, n_blocks_per_row] of Q8_0Block (per-block fp16 scale)
//   output  : [M, N] FP32 row-major
//
// Revisions (§06-1 kernel evolution):
//   v1 — weight reuse across M rows (madd_epi16 + reduce)
//   v2 — VNNI vpdpbusd intrinsic + s8s8 compensation + prefetch  ← CURRENT
//   v3 — (future) two-block packing to recover upper-lane waste
//        and/or row-packed microkernel to amortize horizontal reduce
//
// M >= 16 is intentionally not covered here: that regime enters the
// compute-bound band where the VNNI peak throughput is the new ceiling,
// and extending into AMX-INT8 requires a format redesign (§24 territory).
// ============================================================================
void q8_0_gemm_vnni_impl(const int8_t* x_quant, const float* x_scales,
                         const Q8_0Block* weight, float* output,
                         int M, int N, int K) {
  constexpr int M_MAX = 16;  // Scope guard: caller ensures M < 16
  TORCH_CHECK(M > 0 && M < M_MAX,
              "q8_0_gemm_vnni_impl expects 0 < M < 16, got M=", M);

  const int n_blocks_per_row = (K + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;

#pragma omp parallel for schedule(static)
  for (int n = 0; n < N; ++n) {
    const Q8_0Block* w_row = weight + n * n_blocks_per_row;

    // Per-m accumulator (FP32). Stack-allocated; M < 16.
    float acc[M_MAX] = {0.0f};

    for (int b = 0; b < n_blocks_per_row; ++b) {
      // v4 (prefetch): pull the weight block 2 ahead into L1 while the
      // current block is computed. Masks DDR→L1 latency for the streaming
      // pass over w_row. Q8_0Block = 34 bytes so one prefetch covers it.
      if (b + 2 < n_blocks_per_row) {
        _mm_prefetch(
            reinterpret_cast<const char*>(&w_row[b + 2]), _MM_HINT_T0);
      }

      const Q8_0Block& block = w_row[b];
      const float w_scale = fp16_to_fp32(block.scale_fp16);
      const int k_start = b * Q8_0_BLOCK_SIZE;
      const int k_end = std::min(k_start + Q8_0_BLOCK_SIZE, K);
      const int k_len = k_end - k_start;

      if (k_len == Q8_0_BLOCK_SIZE) {
        // v2 (VNNI dot): replace the 16-bit madd+reduce path with VNNI
        // vpdpbusd (u8 × s8 → s32, 4-byte pairs across 16 lanes). Because
        // VNNI ingests u8 on side A, shift the activation s8→u8 (add 128)
        // and correct with the standard s8s8 compensation:
        //     sum(s8_x * s8_w) = sum(u8_x * s8_w) - 128 * sum(s8_w)
        //                     = dpbusd_dot       - 128 * comp
        //
        // Only lanes 0–7 of the zmm carry real data here (32 bytes fit a
        // half-tile); lanes 8–15 run with zero feed. That halves VNNI's
        // peak throughput at this stage — to be recovered in v3 by
        // packing two consecutive blocks into a full 64-byte zmm.

        // Load the 32-s8 weight block (lower half of zmm; upper half = 0).
        __m256i vw_256 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(block.quants));
        __m512i vw =
            _mm512_inserti32x8(_mm512_setzero_si512(), vw_256, 0);

        // comp = sum over the 32 s8 weight values. Block-level scalar sum
        // is typically auto-vectorized and amortizes across the M rows.
        int32_t comp = 0;
        for (int k = 0; k < Q8_0_BLOCK_SIZE; ++k) {
          comp += block.quants[k];
        }

        const __m256i v_shift128 =
            _mm256_set1_epi8(static_cast<char>(-128));

        for (int m = 0; m < M; ++m) {
          __m256i vx_256 = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(x_quant + m * K + k_start));
          // s8 → u8 shift (XOR with 0x80 via add -128, wraps).
          __m256i vx_u = _mm256_add_epi8(vx_256, v_shift128);
          __m512i vx =
              _mm512_inserti32x8(_mm512_setzero_si512(), vx_u, 0);

          __m512i vacc = _mm512_setzero_si512();
          vacc = _mm512_dpbusd_epi32(vacc, vx, vw);
          int32_t uw_dot = _mm512_reduce_add_epi32(vacc);

          // Compensation undoes the u8 shift.
          int32_t block_dot = uw_dot - 128 * comp;

          acc[m] += static_cast<float>(block_dot) * w_scale * x_scales[m];
        }
      } else {
        // Tail block (K not a multiple of 32): scalar fallback.
        for (int m = 0; m < M; ++m) {
          int32_t block_dot = 0;
          for (int k = 0; k < k_len; ++k) {
            block_dot += static_cast<int32_t>(x_quant[m * K + k_start + k]) *
                         static_cast<int32_t>(block.quants[k]);
          }
          acc[m] += static_cast<float>(block_dot) * w_scale * x_scales[m];
        }
      }
    }

    // Scatter accumulators back to row-major [M, N] output.
    for (int m = 0; m < M; ++m) {
      output[m * N + n] = acc[m];
    }
  }
}

// ============================================================================
// Q8_0 Linear: general M x N with Q8_0 weight
// §06-1 M-aware dispatch:
//   M == 1      → GEMV (memory-bound decode, §06 seqs=1 gain preserved)
//   1 < M < 16  → GEMM with block-wise weight reuse across M (§06-1 Phase 1)
//   M >= 16     → GEMV loop (unchanged, §06-1 scope 밖; AMX path = §24)
// ============================================================================
void q8_0_linear_impl(torch::Tensor& output, const torch::Tensor& input,
                      const torch::Tensor& qweight,
                      const std::optional<torch::Tensor>& bias) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(qweight.is_contiguous(), "qweight must be contiguous");

  const auto input_dtype = input.scalar_type();
  TORCH_CHECK(input_dtype == at::ScalarType::Float ||
                  input_dtype == at::ScalarType::BFloat16,
              "input must be FP32 or BF16");

  const int M = input.size(0);
  const int K = input.size(1);
  const int n_blocks_per_row = (K + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
  const int N = qweight.numel() / (n_blocks_per_row * sizeof(Q8_0Block));

  TORCH_CHECK(output.size(0) == M && output.size(1) == N,
              "output shape mismatch");

  const Q8_0Block* w_ptr =
      reinterpret_cast<const Q8_0Block*>(qweight.data_ptr());

  // Quantize input to INT8 (per-row dynamic quantization)
  auto x_quant = torch::empty({M, K}, torch::kInt8);
  auto x_scales = torch::empty({M}, torch::kFloat32);
  int8_t* xq_ptr = x_quant.data_ptr<int8_t>();
  float* xs_ptr = x_scales.data_ptr<float>();

  // Dynamic quantization of input
#pragma omp parallel for schedule(static)
  for (int m = 0; m < M; ++m) {
    float max_abs = 0.0f;

    if (input_dtype == at::ScalarType::Float) {
      const float* x_row = input.data_ptr<float>() + m * K;
      // Find max absolute value using AVX-512
      __m512 vmax = _mm512_setzero_ps();
      const __m512 vsign_mask =
          _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
      int k = 0;
      for (; k + 16 <= K; k += 16) {
        __m512 vx = _mm512_loadu_ps(x_row + k);
        vx = _mm512_and_ps(vx, vsign_mask);  // abs
        vmax = _mm512_max_ps(vmax, vx);
      }
      max_abs = _mm512_reduce_max_ps(vmax);
      for (; k < K; ++k) {
        max_abs = std::max(max_abs, std::fabs(x_row[k]));
      }

      // Quantize
      float scale = max_abs / 127.0f;
      if (scale == 0.0f) scale = 1.0f;
      float inv_scale = 1.0f / scale;
      xs_ptr[m] = scale;

      __m512 vinv_scale = _mm512_set1_ps(inv_scale);
      k = 0;
      for (; k + 16 <= K; k += 16) {
        __m512 vx = _mm512_loadu_ps(x_row + k);
        __m512 vq = _mm512_mul_ps(vx, vinv_scale);
        // Round and clamp to [-127, 127]
        __m512 vmin = _mm512_set1_ps(-127.0f);
        __m512 vmx = _mm512_set1_ps(127.0f);
        vq = _mm512_max_ps(vq, vmin);
        vq = _mm512_min_ps(vq, vmx);
        __m512i vi32 = _mm512_cvtps_epi32(vq);

        // Pack to int8: 512-bit -> 128-bit
        // First to 16-bit
        __m256i vi16 = _mm512_cvtsepi32_epi16(vi32);
        // Then to 8-bit
        __m128i vi8 = _mm256_cvtsepi16_epi8(vi16);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(xq_ptr + m * K + k), vi8);
      }
      for (; k < K; ++k) {
        float q = x_row[k] * inv_scale;
        q = std::max(-127.0f, std::min(127.0f, q));
        xq_ptr[m * K + k] = static_cast<int8_t>(std::roundf(q));
      }
    } else {
      // BF16 input
      const c10::BFloat16* x_row = input.data_ptr<c10::BFloat16>() + m * K;
      for (int k = 0; k < K; ++k) {
        max_abs = std::max(max_abs, std::fabs(static_cast<float>(x_row[k])));
      }
      float scale = max_abs / 127.0f;
      if (scale == 0.0f) scale = 1.0f;
      float inv_scale = 1.0f / scale;
      xs_ptr[m] = scale;

      for (int k = 0; k < K; ++k) {
        float q = static_cast<float>(x_row[k]) * inv_scale;
        q = std::max(-127.0f, std::min(127.0f, q));
        xq_ptr[m * K + k] = static_cast<int8_t>(std::roundf(q));
      }
    }
  }

  // Compute output — §06-1 M-aware dispatch.
  auto output_f32 = torch::empty({M, N}, torch::kFloat32);
  float* out_f32 = output_f32.data_ptr<float>();

  if (M == 1) {
    // Decode (memory-bound). GEMV already optimal for single-row.
    q8_0_gemv_vnni_impl(xq_ptr, xs_ptr[0], w_ptr, out_f32, N, K);
  } else if (M < 16) {
    // §06-1 Phase 1 target regime: weight reuse across M rows.
    q8_0_gemm_vnni_impl(xq_ptr, xs_ptr, w_ptr, out_f32, M, N, K);
  } else {
    // M >= 16: compute-bound band. Kept on original GEMV loop until
    // AMX-INT8 / format redesign is decided (§24 territory).
    for (int m = 0; m < M; ++m) {
      q8_0_gemv_vnni_impl(xq_ptr + m * K, xs_ptr[m], w_ptr, out_f32 + m * N,
                          N, K);
    }
  }

  // Add bias if present
  if (bias.has_value()) {
    const float* bias_ptr = bias->data_ptr<float>();
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; n += 16) {
        const int n_actual = std::min(16, N - n);
        if (n_actual == 16) {
          __m512 vout = _mm512_loadu_ps(out_f32 + m * N + n);
          __m512 vbias = _mm512_loadu_ps(bias_ptr + n);
          _mm512_storeu_ps(out_f32 + m * N + n, _mm512_add_ps(vout, vbias));
        } else {
          for (int i = 0; i < n_actual; ++i) {
            out_f32[m * N + n + i] += bias_ptr[n + i];
          }
        }
      }
    }
  }

  // Convert to output dtype
  if (output.scalar_type() == at::ScalarType::Float) {
    std::memcpy(output.data_ptr<float>(), out_f32, M * N * sizeof(float));
  } else if (output.scalar_type() == at::ScalarType::BFloat16) {
    c10::BFloat16* out_bf16 = output.data_ptr<c10::BFloat16>();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M * N; ++i) {
      out_bf16[i] = static_cast<c10::BFloat16>(out_f32[i]);
    }
  } else if (output.scalar_type() == at::ScalarType::Half) {
    c10::Half* out_fp16 = output.data_ptr<c10::Half>();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M * N; ++i) {
      out_fp16[i] = static_cast<c10::Half>(out_f32[i]);
    }
  }
}

// ============================================================================
// Quantize FP32/BF16 weight to Q8_0 format
// ============================================================================
void quantize_to_q8_0_impl(const torch::Tensor& weight,
                           torch::Tensor& qweight) {
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D [N, K]");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int N = weight.size(0);
  const int K = weight.size(1);
  const int n_blocks_per_row = (K + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;

  Q8_0Block* out_ptr = reinterpret_cast<Q8_0Block*>(qweight.data_ptr());

#pragma omp parallel for schedule(static)
  for (int n = 0; n < N; ++n) {
    for (int b = 0; b < n_blocks_per_row; ++b) {
      Q8_0Block& block = out_ptr[n * n_blocks_per_row + b];
      const int k_start = b * Q8_0_BLOCK_SIZE;
      const int k_end = std::min(k_start + Q8_0_BLOCK_SIZE, K);

      // Find max absolute value in block
      float max_abs = 0.0f;
      for (int k = k_start; k < k_end; ++k) {
        float val;
        if (weight.scalar_type() == at::ScalarType::Float) {
          val = weight.data_ptr<float>()[n * K + k];
        } else {
          val = static_cast<float>(
              weight.data_ptr<c10::BFloat16>()[n * K + k]);
        }
        max_abs = std::max(max_abs, std::fabs(val));
      }

      // Compute scale
      float scale = max_abs / 127.0f;
      if (scale == 0.0f) scale = 1.0f;  // Avoid division by zero
      float inv_scale = 1.0f / scale;
      block.scale_fp16 = fp32_to_fp16(scale);

      // Quantize block
      for (int k = k_start; k < k_end; ++k) {
        float val;
        if (weight.scalar_type() == at::ScalarType::Float) {
          val = weight.data_ptr<float>()[n * K + k];
        } else {
          val = static_cast<float>(
              weight.data_ptr<c10::BFloat16>()[n * K + k]);
        }
        float q = val * inv_scale;
        q = std::max(-127.0f, std::min(127.0f, q));
        block.quants[k - k_start] = static_cast<int8_t>(std::roundf(q));
      }

      // Zero-pad if partial block
      for (int k = k_end - k_start; k < Q8_0_BLOCK_SIZE; ++k) {
        block.quants[k] = 0;
      }
    }
  }
}

}  // anonymous namespace

// ============================================================================
// Torch entry points
// ============================================================================

void q8_0_linear(torch::Tensor& output, const torch::Tensor& input,
                 const torch::Tensor& qweight,
                 const std::optional<torch::Tensor>& bias) {
  q8_0_linear_impl(output, input, qweight, bias);
}

void q8_0_quantize_weight(torch::Tensor& qweight,
                          const torch::Tensor& weight) {
  quantize_to_q8_0_impl(weight, qweight);
}

#endif  // __AVX512F__ && __AVX512VNNI__
