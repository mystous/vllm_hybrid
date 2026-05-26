// IDE_016 / TSK_025 Step 2 — Penalty operations (repetition / frequency / presence)
//
// SUB_161: penalties = 23% of trident TP0 sampler CPU 시간.
// 본 TU 는 HuggingFace-식 repetition penalty 와 OpenAI-식 frequency / presence
// penalty 의 AVX-512 vectorize.
//
// build: g++ -O3 -mavx512f -mavx512dq -mavx512vl -march=sapphirerapids -fPIC -c

#include "sampling_kernels.h"

#include <immintrin.h>
#include <cstdint>

namespace vllm_hybrid_avx512 {

static constexpr int CHUNK_FP32 = 16;

// ──────────────────────────────────────────────────────────────────────
// Repetition penalty (HuggingFace)
// ──────────────────────────────────────────────────────────────────────
//
// 각 token_ids[b, :lengths[b]] 위치에서 logits 를 penalty 로 곱/나눔.
// dense path 는 sparse (≤ context len) 이므로 scalar gather 가 효율적.
// vectorize 가능한 영역 = penalty 적용 분기.

void apply_repetition_penalty_avx512(float* logits, int B, int V,
                                    const int32_t* token_ids,
                                    const int32_t* lengths,
                                    int max_seen,
                                    float penalty) {
    if (penalty == 1.0f) return;
    const float inv_pen = 1.0f / penalty;
    for (int b = 0; b < B; ++b) {
        float* row = logits + static_cast<size_t>(b) * V;
        const int32_t* ids = token_ids + static_cast<size_t>(b) * max_seen;
        int L = lengths[b];
        // dedup overhead is caller's responsibility; we just apply per index
        for (int n = 0; n < L; ++n) {
            int32_t t = ids[n];
            if (t < 0 || t >= V) continue;
            float v = row[t];
            row[t] = (v > 0.0f) ? (v * inv_pen) : (v * penalty);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Frequency penalty: logits -= freq[v] * alpha
// ──────────────────────────────────────────────────────────────────────

void apply_frequency_penalty_avx512(float* logits, int B, int V,
                                   const int32_t* freq, float alpha) {
    if (alpha == 0.0f) return;
    __m512 valpha = _mm512_set1_ps(alpha);
    for (int b = 0; b < B; ++b) {
        float* row = logits + static_cast<size_t>(b) * V;
        const int32_t* fr = freq + static_cast<size_t>(b) * V;
        int v = 0;
        for (; v + CHUNK_FP32 <= V; v += CHUNK_FP32) {
            __m512i f = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(fr + v));
            __m512 ff = _mm512_cvtepi32_ps(f);
            __m512 x = _mm512_loadu_ps(row + v);
            x = _mm512_sub_ps(x, _mm512_mul_ps(ff, valpha));
            _mm512_storeu_ps(row + v, x);
        }
        for (; v < V; ++v) {
            row[v] -= static_cast<float>(fr[v]) * alpha;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Presence penalty: logits -= alpha if freq[v] > 0 else 0
// ──────────────────────────────────────────────────────────────────────

void apply_presence_penalty_avx512(float* logits, int B, int V,
                                  const int32_t* freq, float alpha) {
    if (alpha == 0.0f) return;
    __m512 valpha = _mm512_set1_ps(alpha);
    __m512i vzero = _mm512_setzero_si512();
    for (int b = 0; b < B; ++b) {
        float* row = logits + static_cast<size_t>(b) * V;
        const int32_t* fr = freq + static_cast<size_t>(b) * V;
        int v = 0;
        for (; v + CHUNK_FP32 <= V; v += CHUNK_FP32) {
            __m512i f = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(fr + v));
            __mmask16 m = _mm512_cmpgt_epi32_mask(f, vzero);
            __m512 x = _mm512_loadu_ps(row + v);
            // conditional subtract alpha where mask set
            x = _mm512_mask_sub_ps(x, m, x, valpha);
            _mm512_storeu_ps(row + v, x);
        }
        for (; v < V; ++v) {
            if (fr[v] > 0) row[v] -= alpha;
        }
    }
}

}  // namespace vllm_hybrid_avx512
