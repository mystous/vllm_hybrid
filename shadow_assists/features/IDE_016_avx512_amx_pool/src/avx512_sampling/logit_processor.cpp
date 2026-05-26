// IDE_016 / TSK_025 Step 2 — AVX-512 logit processor chain
//
// SUB_161: logits_processor.py = 27% / penalty = 23% of TP0 sampler CPU 시간.
// 본 TU 는 temperature / logit-bias / softmax 의 AVX-512 vectorize.
//
// build: g++ -O3 -mavx512f -mavx512vl -mavx512bf16 -march=sapphirerapids -fPIC -c

#include "sampling_kernels.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace vllm_hybrid_avx512 {

static constexpr int CHUNK_FP32 = 16;
static constexpr int CHUNK_BF16 = 32;

// BF16 ↔ FP32 helpers (AVX-512F only; AVX-512_BF16 의 cvt 대신 manual shift)
static inline __m512 bf16_to_fp32_16(const uint16_t* p) {
    __m256i bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    __m512i u32 = _mm512_cvtepu16_epi32(bf16);
    u32 = _mm512_slli_epi32(u32, 16);
    return _mm512_castsi512_ps(u32);
}
static inline void fp32_to_bf16_16(uint16_t* dst, __m512 v) {
    // round-to-nearest-even via add 0x8000 + arith shift
    __m512i u32 = _mm512_castps_si512(v);
    __m512i bias = _mm512_set1_epi32(0x00008000);
    // round to nearest even: detect lsb of upper 16 bits, conditional add
    __m512i upper_lsb = _mm512_and_si512(_mm512_srli_epi32(u32, 16),
                                         _mm512_set1_epi32(1));
    __m512i rounded = _mm512_add_epi32(_mm512_add_epi32(u32, bias), upper_lsb);
    __m512i bf = _mm512_srli_epi32(rounded, 16);
    __m256i bf16 = _mm512_cvtepi32_epi16(bf);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), bf16);
}

// ──────────────────────────────────────────────────────────────────────
// Temperature
// ──────────────────────────────────────────────────────────────────────

void apply_temperature_avx512_fp32(float* logits, int N, float temperature) {
    if (temperature <= 0.0f || temperature == 1.0f) return;
    float inv_T = 1.0f / temperature;
    __m512 v_invT = _mm512_set1_ps(inv_T);
    int i = 0;
    for (; i + CHUNK_FP32 <= N; i += CHUNK_FP32) {
        __m512 x = _mm512_loadu_ps(logits + i);
        x = _mm512_mul_ps(x, v_invT);
        _mm512_storeu_ps(logits + i, x);
    }
    for (; i < N; ++i) logits[i] *= inv_T;
}

void apply_temperature_avx512_bf16(uint16_t* logits, int N, float temperature) {
    if (temperature <= 0.0f || temperature == 1.0f) return;
    float inv_T = 1.0f / temperature;
    __m512 v_invT = _mm512_set1_ps(inv_T);
    int i = 0;
    for (; i + 16 <= N; i += 16) {
        __m512 x = bf16_to_fp32_16(logits + i);
        x = _mm512_mul_ps(x, v_invT);
        fp32_to_bf16_16(logits + i, x);
    }
    for (; i < N; ++i) {
        uint32_t bits = static_cast<uint32_t>(logits[i]) << 16;
        float fv;
        std::memcpy(&fv, &bits, sizeof(float));
        fv *= inv_T;
        uint32_t obits;
        std::memcpy(&obits, &fv, sizeof(float));
        // round-to-nearest-even
        uint32_t lsb = (obits >> 16) & 1;
        obits = obits + 0x8000 + lsb;
        logits[i] = static_cast<uint16_t>(obits >> 16);
    }
}

// ──────────────────────────────────────────────────────────────────────
// Logit bias (dense)
// ──────────────────────────────────────────────────────────────────────

void apply_logit_bias_avx512_fp32(float* logits, const float* bias,
                                 int B, int V) {
    if (!bias) return;
    for (int b = 0; b < B; ++b) {
        float* row = logits + static_cast<size_t>(b) * V;
        // bias [B, V] or [V] broadcast — caller guarantees layout.
        const float* brow = bias + static_cast<size_t>(b) * V;
        int v = 0;
        for (; v + CHUNK_FP32 <= V; v += CHUNK_FP32) {
            __m512 x = _mm512_loadu_ps(row + v);
            __m512 y = _mm512_loadu_ps(brow + v);
            _mm512_storeu_ps(row + v, _mm512_add_ps(x, y));
        }
        for (; v < V; ++v) row[v] += brow[v];
    }
}

// ──────────────────────────────────────────────────────────────────────
// Logit bias (sparse) — vLLM 의 logit_bias dict 패턴
// ──────────────────────────────────────────────────────────────────────

void apply_logit_bias_sparse_avx512(float* logits, int V,
                                   const int32_t* idx,
                                   const float* bias,
                                   int batch_row, int Nb) {
    (void)V;
    float* row = logits;
    if (batch_row > 0) {
        // caller passed batch-prefix pointer already adjusted, but allow
        // safety pattern: logits points to row 0 of B-V matrix
        row = logits + static_cast<size_t>(batch_row) * V;
    }
    // sparse: tiny — scalar gather
    for (int n = 0; n < Nb; ++n) {
        row[idx[n]] += bias[n];
    }
}

// ──────────────────────────────────────────────────────────────────────
// Softmax (FP32, stable)
// ──────────────────────────────────────────────────────────────────────
//
// row-wise: 1) max reduce 2) exp(x - max) accumulate 3) divide by sum
// AVX-512 reduce_max_ps / reduce_add_ps available natively.

void softmax_avx512_fp32(const float* logits, float* probs, int B, int V) {
    for (int b = 0; b < B; ++b) {
        const float* row = logits + static_cast<size_t>(b) * V;
        float* out = probs + static_cast<size_t>(b) * V;

        // max
        __m512 vmax = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        int v = 0;
        for (; v + CHUNK_FP32 <= V; v += CHUNK_FP32) {
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(row + v));
        }
        float rowmax = _mm512_reduce_max_ps(vmax);
        for (; v < V; ++v) {
            if (row[v] > rowmax) rowmax = row[v];
        }

        // exp(x - max) + sum
        __m512 vmaxb = _mm512_set1_ps(rowmax);
        __m512 vsum = _mm512_setzero_ps();
        v = 0;
        for (; v + CHUNK_FP32 <= V; v += CHUNK_FP32) {
            __m512 x = _mm512_sub_ps(_mm512_loadu_ps(row + v), vmaxb);
            alignas(64) float tmp[CHUNK_FP32];
            _mm512_storeu_ps(tmp, x);
            for (int j = 0; j < CHUNK_FP32; ++j) tmp[j] = std::exp(tmp[j]);
            __m512 ex = _mm512_loadu_ps(tmp);
            _mm512_storeu_ps(out + v, ex);
            vsum = _mm512_add_ps(vsum, ex);
        }
        float total = _mm512_reduce_add_ps(vsum);
        for (; v < V; ++v) {
            float e = std::exp(row[v] - rowmax);
            out[v] = e;
            total += e;
        }
        if (total <= 0.0f) total = 1.0f;
        float inv = 1.0f / total;
        __m512 vinv = _mm512_set1_ps(inv);
        v = 0;
        for (; v + CHUNK_FP32 <= V; v += CHUNK_FP32) {
            __m512 e = _mm512_loadu_ps(out + v);
            _mm512_storeu_ps(out + v, _mm512_mul_ps(e, vinv));
        }
        for (; v < V; ++v) out[v] *= inv;
    }
}

}  // namespace vllm_hybrid_avx512
