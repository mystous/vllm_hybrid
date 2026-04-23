// AVX-512 VNNI INT8 GEMM Implementation
//
// Provides high-performance INT8 GEMM for AVX512F+VNNI environments
// without requiring AVX512BF16 or AMX.
//
// Key design decisions:
// - 6x16 micro-kernel: 6 ZMM accumulators + 1 broadcast + 1 B load = 8 ZMM
// - u8s8 format with s8s8 compensation (same as SGL kernels)
// - 3-level cache blocking: MC=72, NC=256, KC=256
// - OpenMP parallelization over M tiles

#include "gemm_vnni.hpp"

#if defined(__AVX512F__) && defined(__AVX512VNNI__)

#include <algorithm>
#include <cstring>

namespace vllm {
namespace vnni {

// ============================================================================
// Weight Packing: [N, K] row-major s8 -> VNNI format [N/16, K/4, 16, 4]
// ============================================================================
void pack_weight_vnni(const int8_t* src, int8_t* dst,
                      int32_t* comp, int N, int K) {
  const int N_blocks = (N + NR - 1) / NR;
  const int K4 = (K + 3) / 4;  // Round up K to multiple of 4

  // Zero out compensation
  std::memset(comp, 0, N * sizeof(int32_t));

  for (int nb = 0; nb < N_blocks; ++nb) {
    const int n_start = nb * NR;
    const int n_end = std::min(n_start + NR, N);

    for (int k4 = 0; k4 < K4; ++k4) {
      // Output: dst[nb * K4 * NR * 4 + k4 * NR * 4 + n_local * 4 + k_local]
      // This is [K/4, NR, 4] layout within each N-block
      int8_t* out_ptr = dst + (nb * K4 * NR * 4) + (k4 * NR * 4);

      for (int n_local = 0; n_local < NR; ++n_local) {
        const int n = n_start + n_local;
        for (int k_local = 0; k_local < 4; ++k_local) {
          const int k = k4 * 4 + k_local;
          int8_t val = 0;
          if (n < N && k < K) {
            val = src[n * K + k];
          }
          out_ptr[n_local * 4 + k_local] = val;

          // Accumulate compensation: sum_k(B[n][k]) * 128
          if (n < N && k < K) {
            comp[n] += static_cast<int32_t>(val) * 128;
          }
        }
      }
    }
  }
}

// ============================================================================
// 6x16 Micro-kernel
// C[6, 16] += A_u8[6, K] * B_packed[K/4, 16, 4]
// Then subtract compensation: C[m][n] -= comp[n]
// ============================================================================
void vnni_micro_kernel_6x16(const uint8_t* A, const int8_t* B_packed,
                            int32_t* C, int K, int lda, int ldc,
                            const int32_t* comp) {
  // 6 accumulators (one per row)
  __m512i c0 = _mm512_setzero_si512();
  __m512i c1 = _mm512_setzero_si512();
  __m512i c2 = _mm512_setzero_si512();
  __m512i c3 = _mm512_setzero_si512();
  __m512i c4 = _mm512_setzero_si512();
  __m512i c5 = _mm512_setzero_si512();

  const int K4 = K / 4;
  const int32_t* a0_ptr = reinterpret_cast<const int32_t*>(A + 0 * lda);
  const int32_t* a1_ptr = reinterpret_cast<const int32_t*>(A + 1 * lda);
  const int32_t* a2_ptr = reinterpret_cast<const int32_t*>(A + 2 * lda);
  const int32_t* a3_ptr = reinterpret_cast<const int32_t*>(A + 3 * lda);
  const int32_t* a4_ptr = reinterpret_cast<const int32_t*>(A + 4 * lda);
  const int32_t* a5_ptr = reinterpret_cast<const int32_t*>(A + 5 * lda);

  // B_packed layout: [K/4, 16, 4] -> stride between k4 groups is 16*4 = 64
  // bytes But since we load as __m512i (64 bytes), stride is 16 int32s
  const int32_t* b_ptr = reinterpret_cast<const int32_t*>(B_packed);

  for (int k4 = 0; k4 < K4; ++k4) {
    // Load B: 16 packed int32 values (each containing 4 int8s)
    __m512i vb = _mm512_loadu_si512(b_ptr + k4 * 16);

    // Broadcast A values (4 uint8s packed into one int32)
    __m512i va0 = _mm512_set1_epi32(a0_ptr[k4]);
    c0 = _mm512_dpbusd_epi32(c0, va0, vb);

    __m512i va1 = _mm512_set1_epi32(a1_ptr[k4]);
    c1 = _mm512_dpbusd_epi32(c1, va1, vb);

    __m512i va2 = _mm512_set1_epi32(a2_ptr[k4]);
    c2 = _mm512_dpbusd_epi32(c2, va2, vb);

    __m512i va3 = _mm512_set1_epi32(a3_ptr[k4]);
    c3 = _mm512_dpbusd_epi32(c3, va3, vb);

    __m512i va4 = _mm512_set1_epi32(a4_ptr[k4]);
    c4 = _mm512_dpbusd_epi32(c4, va4, vb);

    __m512i va5 = _mm512_set1_epi32(a5_ptr[k4]);
    c5 = _mm512_dpbusd_epi32(c5, va5, vb);
  }

  // Subtract s8s8 compensation
  __m512i vcomp = _mm512_loadu_si512(comp);
  c0 = _mm512_sub_epi32(c0, vcomp);
  c1 = _mm512_sub_epi32(c1, vcomp);
  c2 = _mm512_sub_epi32(c2, vcomp);
  c3 = _mm512_sub_epi32(c3, vcomp);
  c4 = _mm512_sub_epi32(c4, vcomp);
  c5 = _mm512_sub_epi32(c5, vcomp);

  // Store results
  _mm512_storeu_si512(C + 0 * ldc, c0);
  _mm512_storeu_si512(C + 1 * ldc, c1);
  _mm512_storeu_si512(C + 2 * ldc, c2);
  _mm512_storeu_si512(C + 3 * ldc, c3);
  _mm512_storeu_si512(C + 4 * ldc, c4);
  _mm512_storeu_si512(C + 5 * ldc, c5);
}

// ============================================================================
// Partial micro-kernel for M < 6 or N < 16 edge cases
// ============================================================================
static void vnni_micro_kernel_partial(const uint8_t* A, const int8_t* B_packed,
                                      int32_t* C, int M_actual, int N_actual,
                                      int K, int lda, int ldc,
                                      const int32_t* comp) {
  // Use a temporary buffer for full 6x16 computation, then copy partial
  alignas(64) int32_t tmp[MR * NR];
  const int tmp_ldc = NR;

  // Zero out the temp buffer
  std::memset(tmp, 0, sizeof(tmp));

  // Compute as many full rows as we have
  const int K4 = K / 4;
  __m512i accums[MR];
  for (int m = 0; m < MR; ++m) {
    accums[m] = _mm512_setzero_si512();
  }

  const int32_t* b_ptr = reinterpret_cast<const int32_t*>(B_packed);

  for (int k4 = 0; k4 < K4; ++k4) {
    __m512i vb = _mm512_loadu_si512(b_ptr + k4 * 16);

    for (int m = 0; m < M_actual; ++m) {
      const int32_t* a_ptr = reinterpret_cast<const int32_t*>(A + m * lda);
      __m512i va = _mm512_set1_epi32(a_ptr[k4]);
      accums[m] = _mm512_dpbusd_epi32(accums[m], va, vb);
    }
  }

  // Subtract compensation and store to temp
  __m512i vcomp = _mm512_loadu_si512(comp);
  for (int m = 0; m < M_actual; ++m) {
    accums[m] = _mm512_sub_epi32(accums[m], vcomp);
    _mm512_storeu_si512(tmp + m * tmp_ldc, accums[m]);
  }

  // Copy partial results to output
  for (int m = 0; m < M_actual; ++m) {
    for (int n = 0; n < N_actual; ++n) {
      C[m * ldc + n] = tmp[m * tmp_ldc + n];
    }
  }
}

// ============================================================================
// M=1 GEMV specialization
// ============================================================================
void int8_gemv_vnni(const uint8_t* x, const int8_t* B_packed,
                    const int32_t* comp, int32_t* y, int N, int K) {
  const int K4 = K / 4;
  const int N_blocks = (N + NR - 1) / NR;

#pragma omp parallel for schedule(static)
  for (int nb = 0; nb < N_blocks; ++nb) {
    const int n_start = nb * NR;
    const int n_actual = std::min(NR, N - n_start);

    // Accumulator
    __m512i acc = _mm512_setzero_si512();

    const int32_t* x_ptr = reinterpret_cast<const int32_t*>(x);
    const int32_t* b_ptr =
        reinterpret_cast<const int32_t*>(B_packed + nb * K4 * NR * 4);

    for (int k4 = 0; k4 < K4; ++k4) {
      __m512i vb = _mm512_loadu_si512(b_ptr + k4 * 16);
      __m512i va = _mm512_set1_epi32(x_ptr[k4]);
      acc = _mm512_dpbusd_epi32(acc, va, vb);

      // Software prefetch
      if (k4 + 4 < K4) {
        _mm_prefetch(reinterpret_cast<const char*>(b_ptr + (k4 + 4) * 16),
                     _MM_HINT_T0);
      }
    }

    // Subtract compensation
    __m512i vcomp = _mm512_loadu_si512(comp + n_start);
    acc = _mm512_sub_epi32(acc, vcomp);

    // Store (handle N tail)
    if (n_actual == NR) {
      _mm512_storeu_si512(y + n_start, acc);
    } else {
      alignas(64) int32_t tmp[NR];
      _mm512_storeu_si512(tmp, acc);
      std::memcpy(y + n_start, tmp, n_actual * sizeof(int32_t));
    }
  }
}

// ============================================================================
// INT8 GEMM with 3-level cache blocking
// ============================================================================
void int8_gemm_vnni(const uint8_t* A, const int8_t* B_packed,
                    const int32_t* comp, int32_t* C, int M, int N, int K,
                    int lda, int ldc) {
  // M=1 fast path
  if (M == 1) {
    int8_gemv_vnni(A, B_packed, comp, C, N, K);
    return;
  }

  const int K4 = (K + 3) / 4;

  // Level 1: Tile over N (NC blocks)
  for (int nc = 0; nc < N; nc += NC) {
    const int nc_actual = std::min(NC, N - nc);
    const int nb_start = nc / NR;

    // Level 2: Tile over M (MC blocks) with OpenMP
#pragma omp parallel for schedule(dynamic)
    for (int mc = 0; mc < M; mc += MC) {
      const int mc_actual = std::min(MC, M - mc);

      // Level 3: Tile over K (KC blocks)
      // For K tiling, we accumulate into C
      for (int kc = 0; kc < K; kc += KC) {
        const int kc_actual = std::min(KC, K - kc);
        const int kc4 = kc / 4;
        const int kc4_actual = kc_actual / 4;

        // Process micro-kernels within this tile
        for (int mr = 0; mr < mc_actual; mr += MR) {
          const int mr_actual = std::min(MR, mc_actual - mr);
          const int m = mc + mr;

          for (int nr = 0; nr < nc_actual; nr += NR) {
            const int nr_actual = std::min(NR, nc_actual - nr);
            const int n = nc + nr;
            const int nb = n / NR;

            // Pointer to A tile: A[m, kc]
            const uint8_t* A_tile = A + m * lda + kc;

            // Pointer to B packed tile: B[nb, kc/4, ...]
            // B_packed layout: [N_blocks, K4, NR, 4]
            const int8_t* B_tile =
                B_packed + nb * K4 * NR * 4 + kc4 * NR * 4;

            // For K tiling: first kc block initializes, subsequent adds
            if (mr_actual == MR && nr_actual == NR) {
              if (kc == 0) {
                // First K tile: write to C directly
                vnni_micro_kernel_6x16(A_tile, B_tile, C + m * ldc + n,
                                       kc_actual, lda, ldc, comp + n);
              } else {
                // Subsequent K tiles: accumulate into temp, then add
                alignas(64) int32_t tmp[MR * NR];
                vnni_micro_kernel_6x16(A_tile, B_tile, tmp, kc_actual, lda, NR,
                                       comp + n);
                // Add to existing C (compensation was already subtracted in
                // first tile, so add it back here to avoid double subtraction)
                __m512i vcomp = _mm512_loadu_si512(comp + n);
                for (int i = 0; i < MR; ++i) {
                  __m512i existing = _mm512_loadu_si512(C + (m + i) * ldc + n);
                  __m512i partial = _mm512_loadu_si512(tmp + i * NR);
                  // tmp already has comp subtracted; add comp back since
                  // it should only be subtracted once total
                  partial = _mm512_add_epi32(partial, vcomp);
                  existing = _mm512_add_epi32(existing, partial);
                  _mm512_storeu_si512(C + (m + i) * ldc + n, existing);
                }
              }
            } else {
              // Edge case: partial micro-kernel
              if (kc == 0) {
                vnni_micro_kernel_partial(A_tile, B_tile, C + m * ldc + n,
                                          mr_actual, nr_actual, kc_actual, lda,
                                          ldc, comp + n);
              } else {
                alignas(64) int32_t tmp[MR * NR];
                vnni_micro_kernel_partial(A_tile, B_tile, tmp, mr_actual,
                                          nr_actual, kc_actual, lda, NR,
                                          comp + n);
                for (int i = 0; i < mr_actual; ++i) {
                  for (int j = 0; j < nr_actual; ++j) {
                    // Add comp back to avoid double subtraction
                    C[(m + i) * ldc + n + j] += tmp[i * NR + j] + comp[n + j];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// ============================================================================
// Dequantize INT32 -> BF16/FP32
// output[m][n] = (int32[m][n]) * a_scale[m] * b_scale[n] + bias[n]
// ============================================================================
void dequant_int32_to_output(const int32_t* input, void* output,
                             at::ScalarType dtype, const float* a_scales,
                             const float* b_scales, const float* bias, int M,
                             int N, int ldc, bool per_token_a,
                             bool per_channel_b) {
#pragma omp parallel for schedule(static)
  for (int m = 0; m < M; ++m) {
    const float a_s = per_token_a ? a_scales[m] : a_scales[0];
    __m512 va_scale = _mm512_set1_ps(a_s);

    for (int n = 0; n < N; n += 16) {
      const int n_actual = std::min(16, N - n);

      // Load int32 accumulator
      __m512i vi32 = _mm512_loadu_si512(input + m * ldc + n);

      // Convert to FP32
      __m512 vf32 = _mm512_cvtepi32_ps(vi32);

      // Scale: result = int32 * a_scale * b_scale
      __m512 vb_scale;
      if (per_channel_b) {
        vb_scale = _mm512_loadu_ps(b_scales + n);
      } else {
        vb_scale = _mm512_set1_ps(b_scales[0]);
      }

      vf32 = _mm512_mul_ps(vf32, va_scale);
      vf32 = _mm512_mul_ps(vf32, vb_scale);

      // Add bias if present
      if (bias) {
        __m512 vbias = _mm512_loadu_ps(bias + n);
        vf32 = _mm512_add_ps(vf32, vbias);
      }

      // Store based on dtype
      if (dtype == at::ScalarType::Float) {
        float* out_f32 = static_cast<float*>(output);
        if (n_actual == 16) {
          _mm512_storeu_ps(out_f32 + m * N + n, vf32);
        } else {
          __mmask16 mask =
              _cvtu32_mask16((1u << n_actual) - 1);
          _mm512_mask_storeu_ps(out_f32 + m * N + n, mask, vf32);
        }
      } else if (dtype == at::ScalarType::BFloat16) {
        // BF16 conversion without AVX512BF16:
        // Shift FP32 right by 16 bits to get BF16
        c10::BFloat16* out_bf16 = static_cast<c10::BFloat16*>(output);
        alignas(64) float tmp_f32[16];
        if (n_actual == 16) {
          _mm512_storeu_ps(tmp_f32, vf32);
        } else {
          __mmask16 mask =
              _cvtu32_mask16((1u << n_actual) - 1);
          _mm512_mask_storeu_ps(tmp_f32, mask, vf32);
        }
        for (int i = 0; i < n_actual; ++i) {
          // Round-to-nearest-even BF16 conversion
          uint32_t bits;
          std::memcpy(&bits, &tmp_f32[i], sizeof(bits));
          uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
          bits += rounding_bias;
          uint16_t bf16_bits = static_cast<uint16_t>(bits >> 16);
          std::memcpy(&out_bf16[m * N + n + i], &bf16_bits, sizeof(bf16_bits));
        }
      } else if (dtype == at::ScalarType::Half) {
        // FP16 conversion
        c10::Half* out_fp16 = static_cast<c10::Half*>(output);
        __m256i vf16 = _mm512_cvtps_ph(
            vf32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        if (n_actual == 16) {
          _mm256_storeu_si256(
              reinterpret_cast<__m256i*>(out_fp16 + m * N + n), vf16);
        } else {
          alignas(32) uint16_t tmp_f16[16];
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_f16), vf16);
          std::memcpy(out_fp16 + m * N + n, tmp_f16,
                      n_actual * sizeof(uint16_t));
        }
      }
    }
  }
}

// ============================================================================
// Torch entry point: VNNI INT8 GEMM
// ============================================================================
void vnni_int8_gemm(torch::Tensor& out, const torch::Tensor& a,
                    const torch::Tensor& b_packed,
                    const torch::Tensor& b_comp,
                    const torch::Tensor& a_scales,
                    const torch::Tensor& b_scales,
                    const std::optional<torch::Tensor>& bias) {
  TORCH_CHECK(a.dtype() == torch::kInt8, "a must be INT8");
  TORCH_CHECK(b_packed.dtype() == torch::kInt8, "b_packed must be INT8");
  TORCH_CHECK(b_comp.dtype() == torch::kInt32, "b_comp must be INT32");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b_packed.is_contiguous(), "b_packed must be contiguous");

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b_comp.size(0);
  const int lda = K;
  const int ldc = N;

  TORCH_CHECK(out.size(0) == M && out.size(1) == N,
              "out must be [M, N], got [", out.size(0), ", ", out.size(1), "]");

  // Shift A from s8 to u8: u8 = s8 + 128
  auto a_u8 = torch::empty_like(a, torch::kUInt8);
  const int8_t* a_s8 = a.data_ptr<int8_t>();
  uint8_t* a_u8_ptr = a_u8.data_ptr<uint8_t>();

  const int a_numel = M * K;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < a_numel; i += 64) {
    const int count = std::min(64, a_numel - i);
    if (count == 64) {
      __m512i vs8 = _mm512_loadu_si512(a_s8 + i);
      __m512i offset = _mm512_set1_epi8(static_cast<char>(-128));
      __m512i vu8 = _mm512_sub_epi8(vs8, offset);
      _mm512_storeu_si512(a_u8_ptr + i, vu8);
    } else {
      for (int j = 0; j < count; ++j) {
        a_u8_ptr[i + j] = static_cast<uint8_t>(static_cast<int16_t>(a_s8[i + j]) + 128);
      }
    }
  }

  // Allocate int32 accumulator
  auto c_i32 = torch::zeros({M, N}, torch::kInt32);
  int32_t* c_ptr = c_i32.data_ptr<int32_t>();

  // Run GEMM
  int8_gemm_vnni(a_u8_ptr, b_packed.data_ptr<int8_t>(),
                 b_comp.data_ptr<int32_t>(), c_ptr, M, N, K, lda, ldc);

  // Dequantize to output
  const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  const bool per_token_a = (a_scales.numel() > 1);
  const bool per_channel_b = (b_scales.numel() > 1);

  dequant_int32_to_output(c_ptr, out.data_ptr(), out.scalar_type(),
                          a_scales.data_ptr<float>(),
                          b_scales.data_ptr<float>(), bias_ptr, M, N, ldc,
                          per_token_a, per_channel_b);
}

// ============================================================================
// Torch entry point: Weight packing
// ============================================================================
void vnni_pack_weight(torch::Tensor& packed, torch::Tensor& comp,
                      const torch::Tensor& weight) {
  TORCH_CHECK(weight.dtype() == torch::kInt8, "weight must be INT8");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D [N, K]");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int N = weight.size(0);
  const int K = weight.size(1);

  TORCH_CHECK(comp.size(0) == N, "comp must be [N]");
  TORCH_CHECK(comp.dtype() == torch::kInt32, "comp must be INT32");

  pack_weight_vnni(weight.data_ptr<int8_t>(), packed.data_ptr<int8_t>(),
                   comp.data_ptr<int32_t>(), N, K);
}

}  // namespace vnni
}  // namespace vllm

#endif  // __AVX512F__ && __AVX512VNNI__
