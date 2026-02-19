// Decode GEMV Optimization for AVX-512
//
// Optimized GEMV (General Matrix-Vector multiply) for the decode phase
// where batch size is typically 1 or very small (<=32).
//
// Key optimizations:
// - Software prefetch for weight rows
// - BF16 -> FP32 conversion using shift (no AVX512BF16 needed)
// - FP32 FMA accumulation with horizontal reduce
// - Batch GEMV for small batches (B<=32)

#include <immintrin.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#ifdef __AVX512F__

namespace {

// ============================================================================
// BF16 to FP32 conversion without AVX512BF16
// ============================================================================
inline __m512 bf16x16_to_fp32(__m256i bf16_vals) {
  // BF16 is upper 16 bits of FP32, so shift left by 16
  __m512i expanded = _mm512_cvtepu16_epi32(bf16_vals);
  __m512i shifted = _mm512_slli_epi32(expanded, 16);
  return _mm512_castsi512_ps(shifted);
}

// ============================================================================
// Single-row BF16 GEMV: y[n] = sum_k(x[k] * W[n, k])
// With software prefetch for next row
// ============================================================================
inline float bf16_dot_product_avx512(const c10::BFloat16* x,
                                     const c10::BFloat16* w_row, int K,
                                     const c10::BFloat16* w_next_row) {
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();

  int k = 0;
  // Main loop: process 32 elements per iteration (2x16 unrolled)
  for (; k + 32 <= K; k += 32) {
    // Load 16 BF16 values from x
    __m256i vx0 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(x + k));
    __m256i vx1 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(x + k + 16));

    // Load 16 BF16 values from weight
    __m256i vw0 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(w_row + k));
    __m256i vw1 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(w_row + k + 16));

    // Convert BF16 to FP32
    __m512 fx0 = bf16x16_to_fp32(vx0);
    __m512 fx1 = bf16x16_to_fp32(vx1);
    __m512 fw0 = bf16x16_to_fp32(vw0);
    __m512 fw1 = bf16x16_to_fp32(vw1);

    // FMA: acc += x * w
    acc0 = _mm512_fmadd_ps(fx0, fw0, acc0);
    acc1 = _mm512_fmadd_ps(fx1, fw1, acc1);

    // Prefetch next row
    if (w_next_row) {
      _mm_prefetch(reinterpret_cast<const char*>(w_next_row + k), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(w_next_row + k + 32),
                   _MM_HINT_T0);
    }
  }

  // Handle remaining elements (16 at a time)
  for (; k + 16 <= K; k += 16) {
    __m256i vx = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(x + k));
    __m256i vw = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(w_row + k));
    __m512 fx = bf16x16_to_fp32(vx);
    __m512 fw = bf16x16_to_fp32(vw);
    acc0 = _mm512_fmadd_ps(fx, fw, acc0);
  }

  // Combine accumulators
  acc0 = _mm512_add_ps(acc0, acc1);
  float result = _mm512_reduce_add_ps(acc0);

  // Handle tail elements
  for (; k < K; ++k) {
    result += static_cast<float>(x[k]) * static_cast<float>(w_row[k]);
  }

  return result;
}

// ============================================================================
// FP32 GEMV: y[n] = sum_k(x[k] * W[n, k])
// ============================================================================
inline float fp32_dot_product_avx512(const float* x, const float* w_row,
                                     int K, const float* w_next_row) {
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();

  int k = 0;
  for (; k + 32 <= K; k += 32) {
    __m512 vx0 = _mm512_loadu_ps(x + k);
    __m512 vx1 = _mm512_loadu_ps(x + k + 16);
    __m512 vw0 = _mm512_loadu_ps(w_row + k);
    __m512 vw1 = _mm512_loadu_ps(w_row + k + 16);

    acc0 = _mm512_fmadd_ps(vx0, vw0, acc0);
    acc1 = _mm512_fmadd_ps(vx1, vw1, acc1);

    if (w_next_row) {
      _mm_prefetch(reinterpret_cast<const char*>(w_next_row + k), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(w_next_row + k + 64),
                   _MM_HINT_T0);
    }
  }

  for (; k + 16 <= K; k += 16) {
    __m512 vx = _mm512_loadu_ps(x + k);
    __m512 vw = _mm512_loadu_ps(w_row + k);
    acc0 = _mm512_fmadd_ps(vx, vw, acc0);
  }

  acc0 = _mm512_add_ps(acc0, acc1);
  float result = _mm512_reduce_add_ps(acc0);

  for (; k < K; ++k) {
    result += x[k] * w_row[k];
  }

  return result;
}

}  // anonymous namespace

// ============================================================================
// BF16 Decode GEMV
// x: [K] BF16
// W: [N, K] BF16 (row-major)
// y: [N] BF16
// ============================================================================
void bf16_decode_gemv(const c10::BFloat16* x, const c10::BFloat16* W,
                      c10::BFloat16* y, int N, int K) {
#pragma omp parallel for schedule(static)
  for (int n = 0; n < N; ++n) {
    const c10::BFloat16* w_row = W + n * K;
    const c10::BFloat16* w_next = (n + 1 < N) ? W + (n + 1) * K : nullptr;
    float result = bf16_dot_product_avx512(x, w_row, K, w_next);
    y[n] = static_cast<c10::BFloat16>(result);
  }
}

// ============================================================================
// BF16 Batch GEMV (B <= 32)
// X: [B, K] BF16
// W: [N, K] BF16
// Y: [B, N] BF16
// ============================================================================
void bf16_batch_gemv(const c10::BFloat16* X, const c10::BFloat16* W,
                     c10::BFloat16* Y, int B, int N, int K) {
  // For each output row n, process all B inputs
#pragma omp parallel for schedule(dynamic)
  for (int n = 0; n < N; ++n) {
    const c10::BFloat16* w_row = W + n * K;
    const c10::BFloat16* w_next = (n + 1 < N) ? W + (n + 1) * K : nullptr;

    for (int b = 0; b < B; ++b) {
      const c10::BFloat16* x_row = X + b * K;
      float result = bf16_dot_product_avx512(x_row, w_row, K, w_next);
      Y[b * N + n] = static_cast<c10::BFloat16>(result);
    }
  }
}

// ============================================================================
// Torch entry point: decode_gemv
// Supports BF16 and FP32 input/weight
// ============================================================================
void decode_gemv(torch::Tensor& output, const torch::Tensor& input,
                 const torch::Tensor& weight,
                 const std::optional<torch::Tensor>& bias) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D [N, K]");

  const int M = input.size(0);
  const int K = input.size(1);
  const int N = weight.size(0);

  TORCH_CHECK(weight.size(1) == K, "weight K dimension mismatch");
  TORCH_CHECK(output.size(0) == M && output.size(1) == N,
              "output shape mismatch");

  const auto dtype = input.scalar_type();
  TORCH_CHECK(dtype == weight.scalar_type(),
              "input and weight must have same dtype");

  if (dtype == at::ScalarType::BFloat16) {
    const c10::BFloat16* x_ptr = input.data_ptr<c10::BFloat16>();
    const c10::BFloat16* w_ptr = weight.data_ptr<c10::BFloat16>();
    c10::BFloat16* y_ptr = output.data_ptr<c10::BFloat16>();

    if (M == 1) {
      bf16_decode_gemv(x_ptr, w_ptr, y_ptr, N, K);
    } else {
      bf16_batch_gemv(x_ptr, w_ptr, y_ptr, M, N, K);
    }

    // Add bias
    if (bias.has_value()) {
      const c10::BFloat16* bias_ptr = bias->data_ptr<c10::BFloat16>();
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; n += 16) {
          const int n_actual = std::min(16, N - n);
          if (n_actual == 16) {
            __m256i vy = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(y_ptr + m * N + n));
            __m256i vb = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(bias_ptr + n));
            __m512 fy = bf16x16_to_fp32(vy);
            __m512 fb = bf16x16_to_fp32(vb);
            __m512 fr = _mm512_add_ps(fy, fb);
            // Convert back to BF16 (truncate upper 16 bits)
            __m512i fi = _mm512_castps_si512(fr);
            __m512i shifted = _mm512_srli_epi32(fi, 16);
            // Round-to-nearest: add rounding bias
            __m512i round_bias = _mm512_set1_epi32(0x00008000);
            fi = _mm512_add_epi32(_mm512_castps_si512(fr), round_bias);
            shifted = _mm512_srli_epi32(fi, 16);
            __m256i result = _mm512_cvtepi32_epi16(shifted);
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(y_ptr + m * N + n), result);
          } else {
            for (int i = 0; i < n_actual; ++i) {
              float val = static_cast<float>(y_ptr[m * N + n + i]) +
                          static_cast<float>(bias_ptr[n + i]);
              y_ptr[m * N + n + i] = static_cast<c10::BFloat16>(val);
            }
          }
        }
      }
    }
  } else if (dtype == at::ScalarType::Float) {
    const float* x_ptr = input.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    float* y_ptr = output.data_ptr<float>();

#pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < N; ++n) {
      const float* w_row = w_ptr + n * K;
      const float* w_next = (n + 1 < N) ? w_ptr + (n + 1) * K : nullptr;
      for (int m = 0; m < M; ++m) {
        y_ptr[m * N + n] =
            fp32_dot_product_avx512(x_ptr + m * K, w_row, K, w_next);
      }
    }

    // Add bias
    if (bias.has_value()) {
      const float* bias_ptr = bias->data_ptr<float>();
#pragma omp parallel for schedule(static)
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; n += 16) {
          const int n_actual = std::min(16, N - n);
          if (n_actual == 16) {
            __m512 vy = _mm512_loadu_ps(y_ptr + m * N + n);
            __m512 vb = _mm512_loadu_ps(bias_ptr + n);
            _mm512_storeu_ps(y_ptr + m * N + n, _mm512_add_ps(vy, vb));
          } else {
            for (int i = 0; i < n_actual; ++i) {
              y_ptr[m * N + n + i] += bias_ptr[n + i];
            }
          }
        }
      }
    }
  } else {
    TORCH_CHECK(false, "decode_gemv: unsupported dtype ", dtype);
  }
}

#endif  // __AVX512F__
