// SUB_178 — Cold-KV CPU AVX-512 dequant kernels
//
// INT8 → BF16  and  INT4 → BF16  dequant for vLLM KV blocks held on CPU.
//
// Layout assumption (matches Qwen 32B / TP=4 KV layout shape):
//   - one KV "chunk" = block_size (16) × num_heads_kv_per_rank × head_dim
//     e.g. Qwen32B head_dim=128, num_heads_kv=8, TP=4 → 2 heads/rank →
//          16 × 2 × 128 = 4096 elems / 8 KB BF16 / 4 KB INT8 / 2 KB INT4
//   - scale stored per (block, head) → length = block_size * num_heads * 2 bytes (BF16)
//
// Public C-ABI:
//   cold_kv_int8_to_bf16(const int8_t* q, const uint16_t* scale_bf16,
//                        uint16_t* out_bf16, int n_elems, int scale_group_size);
//   cold_kv_int4_to_bf16(const uint8_t* q_packed, const uint16_t* scale_bf16,
//                        uint16_t* out_bf16, int n_elems, int scale_group_size);
//
// Build:
//   g++ -O3 -mavx512f -mavx512bw -mavx512vl -mavx512bf16 -fPIC -shared \
//       cold_kv_decompress.cpp -o libcold_kv.so

#include <cstdint>
#include <cstring>
#include <immintrin.h>

extern "C" {

// ----------------------------------------------------------------------------
// Helper: BF16 → FP32 (single elem). BF16 is stored as uint16_t holding the
// high 16 bits of an FP32. So we just shift left.
// ----------------------------------------------------------------------------
static inline float bf16_to_fp32_scalar(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ----------------------------------------------------------------------------
// INT8 → BF16, AVX-512 path
//
// Process 32 elements per loop iteration:
//   - load 32 int8 → vpmovsxbw → 32×int16
//   - split low/high 16×int16 → cvtepi16_epi32 → 16×int32 each
//   - cvtepi32_ps → 16×fp32 each
//   - multiply by scale (broadcast per group)
//   - pack two 16×fp32 → 32×bf16 via _mm512_cvtne2ps_pbh (AVX512_BF16) when
//     available, else manual high-half extraction (fallback).
// ----------------------------------------------------------------------------
void cold_kv_int8_to_bf16(const int8_t* __restrict__ q,
                          const uint16_t* __restrict__ scale_bf16,
                          uint16_t* __restrict__ out_bf16,
                          int n_elems,
                          int scale_group_size) {
    if (scale_group_size <= 0) scale_group_size = n_elems;
    int g = 0;
    int i = 0;
    // 32-elem chunks
    while (i + 32 <= n_elems) {
        // Find current scale (each group is scale_group_size contiguous elems)
        g = i / scale_group_size;
        float s = bf16_to_fp32_scalar(scale_bf16[g]);
        __m512 vs = _mm512_set1_ps(s);

        // load 32 int8 -> 32x int16 (sign-extend)
        __m256i q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + i));
        __m512i q16 = _mm512_cvtepi8_epi16(q8);

        // split low/high 16x int16
        __m256i lo16 = _mm512_extracti64x4_epi64(q16, 0);
        __m256i hi16 = _mm512_extracti64x4_epi64(q16, 1);

        __m512i lo32 = _mm512_cvtepi16_epi32(lo16);
        __m512i hi32 = _mm512_cvtepi16_epi32(hi16);

        __m512 lof = _mm512_cvtepi32_ps(lo32);
        __m512 hif = _mm512_cvtepi32_ps(hi32);

        lof = _mm512_mul_ps(lof, vs);
        hif = _mm512_mul_ps(hif, vs);

#ifdef __AVX512BF16__
        // pack 32x fp32 → 32x bf16 with stochastic-free RNE conversion
        __m512bh packed = _mm512_cvtne2ps_pbh(hif, lof);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(out_bf16 + i),
                            reinterpret_cast<__m512i&>(packed));
#else
        // Fallback: extract high 16 bits manually (truncation toward zero)
        __m512i lo_bits = _mm512_castps_si512(lof);
        __m512i hi_bits = _mm512_castps_si512(hif);
        __m256i lo_bf = _mm512_cvtepi32_epi16(_mm512_srli_epi32(lo_bits, 16));
        __m256i hi_bf = _mm512_cvtepi32_epi16(_mm512_srli_epi32(hi_bits, 16));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_bf16 + i), lo_bf);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_bf16 + i + 16), hi_bf);
#endif
        i += 32;
    }
    // Tail (rare for KV chunks aligned to 32)
    for (; i < n_elems; ++i) {
        g = i / scale_group_size;
        float s = bf16_to_fp32_scalar(scale_bf16[g]);
        float v = (float)q[i] * s;
        uint32_t u;
        std::memcpy(&u, &v, sizeof(u));
        out_bf16[i] = (uint16_t)(u >> 16);
    }
}

// ----------------------------------------------------------------------------
// INT4 → BF16, AVX-512 path
//
// Input packed: each byte = 2 INT4 nibbles (signed [-8, 7]).
//   - low nibble in bits [3:0], high nibble in bits [7:4]
// Process 64 elements (32 bytes) per iteration. We unpack to 64 int8 then
// reuse the int8 path internally (manual inline for speed).
// ----------------------------------------------------------------------------
void cold_kv_int4_to_bf16(const uint8_t* __restrict__ q_packed,
                          const uint16_t* __restrict__ scale_bf16,
                          uint16_t* __restrict__ out_bf16,
                          int n_elems,
                          int scale_group_size) {
    if (scale_group_size <= 0) scale_group_size = n_elems;
    int i = 0;
    int g = 0;

    // Mask to extract low nibble
    const __m512i lo_mask = _mm512_set1_epi8(0x0F);
    // Add 8 then subtract 8 trick for sign-extension of 4-bit
    const __m512i sign_bias = _mm512_set1_epi8(0x08);

    while (i + 64 <= n_elems) {
        g = i / scale_group_size;
        float s = bf16_to_fp32_scalar(scale_bf16[g]);
        __m512 vs = _mm512_set1_ps(s);

        // Load 32 bytes = 64 nibbles
        __m256i packed = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(q_packed + i / 2));

        // Promote to 512-bit: even indices = low nibbles, odd = high
        __m512i p512 = _mm512_cvtepu8_epi16(packed);  // 32x uint16 (each holds 1 byte)
        // low nibbles = p512 & 0x0F
        __m512i lo_nib = _mm512_and_si512(p512, _mm512_set1_epi16(0x000F));
        // high nibbles = (p512 >> 4) & 0x0F
        __m512i hi_nib = _mm512_and_si512(_mm512_srli_epi16(p512, 4),
                                          _mm512_set1_epi16(0x000F));
        // sign-extend nibble: value = (n ^ 8) - 8
        __m512i lo_se = _mm512_sub_epi16(
            _mm512_xor_si512(lo_nib, _mm512_set1_epi16(0x0008)),
            _mm512_set1_epi16(0x0008));
        __m512i hi_se = _mm512_sub_epi16(
            _mm512_xor_si512(hi_nib, _mm512_set1_epi16(0x0008)),
            _mm512_set1_epi16(0x0008));

        // interleave: out[k*2]=lo[k], out[k*2+1]=hi[k] for k=0..31
        // _mm512_unpacklo_epi16 / _mm512_unpackhi_epi16 work per 128-bit lane,
        // so a direct unpack yields lane-shuffled output. We use permutex2var
        // with explicit indices to produce the true interleave.
        //
        // Desired layout in two 512-bit halves:
        //   lo_int = [lo0 hi0 lo1 hi1 ... lo15 hi15]   (nibbles 0..31)
        //   hi_int = [lo16 hi16 ... lo31 hi31]         (nibbles 32..63)
        //
        // idx_low_pat selects from lo_se (low halves) at even positions and
        // from hi_se (high halves) at odd positions.
        // _mm512_set_epi16 is reverse order; we want logical index k → element k.
        // Use reverse listing for the desired interleave.
        static const __m512i idx_low_pat = _mm512_set_epi16(
            47, 15, 46, 14, 45, 13, 44, 12,
            43, 11, 42, 10, 41,  9, 40,  8,
            39,  7, 38,  6, 37,  5, 36,  4,
            35,  3, 34,  2, 33,  1, 32,  0);
        static const __m512i idx_high_pat = _mm512_set_epi16(
            63, 31, 62, 30, 61, 29, 60, 28,
            59, 27, 58, 26, 57, 25, 56, 24,
            55, 23, 54, 22, 53, 21, 52, 20,
            51, 19, 50, 18, 49, 17, 48, 16);
        __m512i lo_int = _mm512_permutex2var_epi16(lo_se, idx_low_pat,  hi_se);
        __m512i hi_int = _mm512_permutex2var_epi16(lo_se, idx_high_pat, hi_se);

        // Now each int16 holds a small signed value [-8, 7].
        // Convert lo_int 32x int16 → 16x int32 (low half) + 16x int32 (high half)
        // Then to fp32, mul by scale, pack bf16.
        auto convert_and_store = [&](__m512i v_i16, int offset) {
            __m256i v0 = _mm512_extracti64x4_epi64(v_i16, 0);
            __m256i v1 = _mm512_extracti64x4_epi64(v_i16, 1);
            __m512i i32a = _mm512_cvtepi16_epi32(v0);
            __m512i i32b = _mm512_cvtepi16_epi32(v1);
            __m512 fa = _mm512_mul_ps(_mm512_cvtepi32_ps(i32a), vs);
            __m512 fb = _mm512_mul_ps(_mm512_cvtepi32_ps(i32b), vs);
#ifdef __AVX512BF16__
            __m512bh p = _mm512_cvtne2ps_pbh(fb, fa);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(out_bf16 + offset),
                                reinterpret_cast<__m512i&>(p));
#else
            __m256i ba = _mm512_cvtepi32_epi16(
                _mm512_srli_epi32(_mm512_castps_si512(fa), 16));
            __m256i bb = _mm512_cvtepi32_epi16(
                _mm512_srli_epi32(_mm512_castps_si512(fb), 16));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_bf16 + offset), ba);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_bf16 + offset + 16), bb);
#endif
        };
        convert_and_store(lo_int, i);
        convert_and_store(hi_int, i + 32);
        i += 64;
    }

    // Tail
    for (; i < n_elems; ++i) {
        g = i / scale_group_size;
        float s = bf16_to_fp32_scalar(scale_bf16[g]);
        uint8_t byte = q_packed[i / 2];
        int8_t nib = (i & 1) ? (int8_t)(byte >> 4) : (int8_t)(byte & 0x0F);
        // sign-extend 4-bit
        if (nib & 0x08) nib |= 0xF0;
        float v = (float)nib * s;
        uint32_t u;
        std::memcpy(&u, &v, sizeof(u));
        out_bf16[i] = (uint16_t)(u >> 16);
    }
    (void)lo_mask;
    (void)sign_bias;
}

// ----------------------------------------------------------------------------
// Reference scalar paths for accuracy verification.
// ----------------------------------------------------------------------------
void cold_kv_int8_to_bf16_ref(const int8_t* q, const uint16_t* scale_bf16,
                              uint16_t* out_bf16, int n_elems,
                              int scale_group_size) {
    if (scale_group_size <= 0) scale_group_size = n_elems;
    for (int i = 0; i < n_elems; ++i) {
        int g = i / scale_group_size;
        float s = bf16_to_fp32_scalar(scale_bf16[g]);
        float v = (float)q[i] * s;
        uint32_t u;
        std::memcpy(&u, &v, sizeof(u));
        out_bf16[i] = (uint16_t)(u >> 16);
    }
}

void cold_kv_int4_to_bf16_ref(const uint8_t* q_packed, const uint16_t* scale_bf16,
                              uint16_t* out_bf16, int n_elems,
                              int scale_group_size) {
    if (scale_group_size <= 0) scale_group_size = n_elems;
    for (int i = 0; i < n_elems; ++i) {
        int g = i / scale_group_size;
        float s = bf16_to_fp32_scalar(scale_bf16[g]);
        uint8_t byte = q_packed[i / 2];
        int8_t nib = (i & 1) ? (int8_t)(byte >> 4) : (int8_t)(byte & 0x0F);
        if (nib & 0x08) nib |= 0xF0;
        float v = (float)nib * s;
        uint32_t u;
        std::memcpy(&u, &v, sizeof(u));
        out_bf16[i] = (uint16_t)(u >> 16);
    }
}

}  // extern "C"
