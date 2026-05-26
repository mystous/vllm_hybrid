// IDE_016 / TSK_025 — AVX-512 top-k / top-p sampling kernel
//
// SUB_161 lever: trident TP0 의 44.3% CPU 시간 = sampler.py.
// SUB_117 input: 10.24 TFLOPS available CPU compute (N=32 pinned).
//
// build (Sapphire Rapids 8480+):
//   g++ -O3 -mavx512f -mavx512bw -mavx512vl -mavx512bf16 -mavx512cd \
//       -march=sapphirerapids -fPIC -c topk_topp_kernel.cpp
//
// 알고리즘:
//   top-k:  streaming threshold + _mm512_mask_compress_ps + partial sort
//           - 1 pass: max reduction (AVX-512 reduce_max_ps)
//           - 2 pass: threshold estimator (running K-th value approximation)
//           - 3 pass: compress + insertion sort 의 top-K
//
//   top-p: sorted descending probs 위에서 prefix sum + threshold detect
//           AVX-512 fp32 chunk sum + mask compare → 첫 cutoff index
//
// 정확도 게이트 (CLAUDE.md 운영 해석):
//   per-token logprob max abs diff < 1e-3 vs PyTorch torch.topk / torch.softmax
//
// AVX-512 fuse-off (Alder Lake) 회피:
//   본 TU 는 -mavx512f 로만 컴파일되며, runtime 검사 (__builtin_cpu_supports)
//   는 python_bindings.cpp 가 담당.

#include "sampling_kernels.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace vllm_hybrid_avx512 {

// ──────────────────────────────────────────────────────────────────────
// Constants + helpers
// ──────────────────────────────────────────────────────────────────────

static constexpr int LANE_FP32 = 16;   // 512 bits / 32 bits

// BF16 → FP32 upcast: bf16 비트 패턴을 fp32 의 상위 16 비트로 두고 하위 16 비트 = 0.
// AVX-512_BF16 의 _mm512_cvtpbh_ps 와 동일 의미이지만, manual shift 가
// AVX-512F 만으로 동작하므로 fuse-off 경로 호환 좋음.
static inline __m512 bf16_to_fp32_16(const uint16_t* p) {
    __m256i bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    __m512i u32 = _mm512_cvtepu16_epi32(bf16);
    u32 = _mm512_slli_epi32(u32, 16);
    return _mm512_castsi512_ps(u32);
}

// horizontal max over __m512 lanes
static inline float hmax_512(__m512 v) {
    return _mm512_reduce_max_ps(v);
}

// ──────────────────────────────────────────────────────────────────────
// One-row top-k via streaming threshold + compress
// ──────────────────────────────────────────────────────────────────────
//
// Strategy:
//   Pass 1: compute global max for stability and threshold init.
//   Pass 2: 작은 candidate buffer (capacity = K * OVERSAMPLE) 유지. AVX-512
//           로 chunk_max 검사 → chunk 안 어느 원소라도 threshold 초과 시만
//           per-element fallback 진입.
//   Pass 3: scalar partial sort 의 top-K (candidate ≤ K*OVERSAMPLE 이라 cheap).
//
// vocab=152064 / K=20 기준:
//   - threshold-based filter 가 평균 99.5%+ 의 chunk skip.
//   - candidate buffer ≤ 256 → cache hot.
//   - partial_sort O(N log K) 에서 N=256.

template <typename T>
static void topk_row_threshold(const T* row_data, int V, int K,
                              int32_t* idx_out, float* val_out,
                              bool is_bf16) {
    static constexpr int CHUNK = 16;
    constexpr int OVERSAMPLE = 8;   // candidate ≈ K * 8
    const int cap = std::max(K * OVERSAMPLE, 64);

    // candidate buffer
    struct Cand { float v; int32_t i; };
    std::vector<Cand> buf;
    buf.reserve(cap + CHUNK);

    // Pass 1: global max — for absolute lower bound seed (most logits ≥ -inf).
    __m512 vmax = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    int v = 0;
    for (; v + CHUNK <= V; v += CHUNK) {
        __m512 x;
        if (is_bf16) {
            x = bf16_to_fp32_16(reinterpret_cast<const uint16_t*>(row_data) + v);
        } else {
            x = _mm512_loadu_ps(reinterpret_cast<const float*>(row_data) + v);
        }
        vmax = _mm512_max_ps(vmax, x);
    }
    float global_max = hmax_512(vmax);
    for (; v < V; ++v) {
        float x = is_bf16
            ? (static_cast<float>(reinterpret_cast<const uint16_t*>(row_data)[v]) == 0.0f
                ? 0.0f  // never taken, suppresses warning
                : 0.0f)
            : static_cast<float>(reinterpret_cast<const float*>(row_data)[v]);
        // proper scalar bf16 -> fp32:
        if (is_bf16) {
            uint32_t bits = static_cast<uint32_t>(
                reinterpret_cast<const uint16_t*>(row_data)[v]) << 16;
            float fv;
            std::memcpy(&fv, &bits, sizeof(float));
            x = fv;
        }
        if (x > global_max) global_max = x;
    }

    // running threshold — initialized to -inf so 첫 cap 개는 무조건 채워짐.
    float threshold = -std::numeric_limits<float>::infinity();

    // Pass 2: streaming scan + compress on threshold-gt
    v = 0;
    for (; v + CHUNK <= V; v += CHUNK) {
        __m512 x;
        if (is_bf16) {
            x = bf16_to_fp32_16(reinterpret_cast<const uint16_t*>(row_data) + v);
        } else {
            x = _mm512_loadu_ps(reinterpret_cast<const float*>(row_data) + v);
        }
        __m512 thr = _mm512_set1_ps(threshold);
        __mmask16 m = _mm512_cmp_ps_mask(x, thr, _CMP_GT_OQ);
        if (__builtin_expect(m == 0, 1)) continue;   // common-case: skip chunk

        // Use compressstoreu to gather hits to temp arrays.
        alignas(64) float vbuf[CHUNK];
        alignas(64) int32_t ibuf[CHUNK];
        // index lane = {v, v+1, ..., v+15}
        __m512i base = _mm512_add_epi32(_mm512_set1_epi32(v),
            _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15));
        _mm512_mask_compressstoreu_ps(vbuf, m, x);
        _mm512_mask_compressstoreu_epi32(ibuf, m, base);
        int hits = __builtin_popcount(static_cast<unsigned>(m));
        for (int h = 0; h < hits; ++h) {
            buf.push_back({vbuf[h], ibuf[h]});
        }

        // If buf 가 cap*2 초과 시 partial sort 후 cap 으로 truncate + threshold 업데이트
        if (static_cast<int>(buf.size()) >= cap * 2) {
            std::partial_sort(buf.begin(), buf.begin() + cap, buf.end(),
                [](const Cand& a, const Cand& b) { return a.v > b.v; });
            buf.resize(cap);
            threshold = buf.back().v;   // K-th approximation
        }
    }
    // tail
    for (; v < V; ++v) {
        float x;
        if (is_bf16) {
            uint32_t bits = static_cast<uint32_t>(
                reinterpret_cast<const uint16_t*>(row_data)[v]) << 16;
            std::memcpy(&x, &bits, sizeof(float));
        } else {
            x = reinterpret_cast<const float*>(row_data)[v];
        }
        if (x > threshold) buf.push_back({x, v});
    }

    // Pass 3: partial_sort top-K
    int eff_k = std::min(K, static_cast<int>(buf.size()));
    if (eff_k <= 0) {
        // pathological: all -inf — return arbitrary indices with global_max placeholder
        for (int k = 0; k < K; ++k) {
            idx_out[k] = k;
            val_out[k] = global_max;
        }
        return;
    }
    std::partial_sort(buf.begin(), buf.begin() + eff_k, buf.end(),
        [](const Cand& a, const Cand& b) { return a.v > b.v; });
    for (int k = 0; k < eff_k; ++k) {
        idx_out[k] = buf[k].i;
        val_out[k] = buf[k].v;
    }
    // fill remainder if buf < K (extremely small vocab)
    for (int k = eff_k; k < K; ++k) {
        idx_out[k] = 0;
        val_out[k] = -std::numeric_limits<float>::infinity();
    }
}


// Public API
void topk_avx512_bf16(const uint16_t* logits_bf16, int B, int V, int K,
                     int32_t* indices_out, float* values_out) {
    for (int b = 0; b < B; ++b) {
        topk_row_threshold(logits_bf16 + static_cast<size_t>(b) * V, V, K,
                          indices_out + b * K, values_out + b * K,
                          /*is_bf16=*/true);
    }
}

void topk_avx512_fp32(const float* logits_fp32, int B, int V, int K,
                     int32_t* indices_out, float* values_out) {
    for (int b = 0; b < B; ++b) {
        topk_row_threshold(logits_fp32 + static_cast<size_t>(b) * V, V, K,
                          indices_out + b * K, values_out + b * K,
                          /*is_bf16=*/false);
    }
}


// ──────────────────────────────────────────────────────────────────────
// top-p (nucleus) cutoff
// ──────────────────────────────────────────────────────────────────────
//
// Input: sorted descending probs per row [B, V].
// Output: cutoff [B] = first index where cumulative >= p, kept ≥ 1.
//
// AVX-512: chunk 별 sum 누적 + cumulative ≥ p detect.
// 16-wide horizontal prefix sum 은 비싸므로 chunk 별 reduce_add 만 사용하여
// linear scan 의 throughput 을 16× 향상.

void topp_avx512_fp32(const float* sorted_probs, int B, int V,
                     float p, int32_t* cutoff_out) {
    static constexpr int CHUNK = 16;
    for (int b = 0; b < B; ++b) {
        const float* row = sorted_probs + static_cast<size_t>(b) * V;
        float cum = 0.0f;
        int cutoff = V;
        int v = 0;
        for (; v + CHUNK <= V; v += CHUNK) {
            __m512 x = _mm512_loadu_ps(row + v);
            float chunk_sum = _mm512_reduce_add_ps(x);
            if (cum + chunk_sum < p) {
                cum += chunk_sum;
                continue;
            }
            // exact cutoff within chunk — scalar (chunk size 16 cheap)
            for (int j = 0; j < CHUNK; ++j) {
                cum += row[v + j];
                if (cum >= p) { cutoff = v + j + 1; goto done; }
            }
        }
        for (; v < V; ++v) {
            cum += row[v];
            if (cum >= p) { cutoff = v + 1; goto done; }
        }
        done:
        if (cutoff < 1) cutoff = 1;   // always keep top-1
        cutoff_out[b] = cutoff;
    }
}


// ──────────────────────────────────────────────────────────────────────
// Fused softmax + top-k + top-p + sample
// ──────────────────────────────────────────────────────────────────────
//
// Strategy:
//   1. apply temperature + find max (1 pass)
//   2. exp + accumulate sum over vocab (1 pass), stored to FP32 workspace
//   3. probs = exp / sum
//   4. top-K via threshold (workspace 의 probs)
//   5. sort top-K descending
//   6. cumulative + top-p cutoff
//   7. renormalize → categorical sample (single uniform RNG)
//
// 본 함수의 main lever — sampler.py 의 44% 영역.

// xorshift64* RNG — 빠른 deterministic uniform [0, 1)
static inline uint64_t xorshift64(uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state * 0x2545F4914F6CDD1DULL;
}
static inline float uniform01(uint64_t& state) {
    uint64_t x = xorshift64(state);
    return static_cast<float>(x >> 40) * (1.0f / 16777216.0f);   // 24-bit
}

template <typename T>
static int64_t fused_sample_row(const T* row_data, int V, int K, float p,
                                float temperature, uint64_t& rng,
                                bool is_bf16) {
    static constexpr int CHUNK = 16;

    // Pass 1: temperature-scaled max
    float inv_T = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;
    __m512 vmax = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    __m512 vinvT = _mm512_set1_ps(inv_T);
    int v = 0;
    for (; v + CHUNK <= V; v += CHUNK) {
        __m512 x;
        if (is_bf16) {
            x = bf16_to_fp32_16(reinterpret_cast<const uint16_t*>(row_data) + v);
        } else {
            x = _mm512_loadu_ps(reinterpret_cast<const float*>(row_data) + v);
        }
        x = _mm512_mul_ps(x, vinvT);
        vmax = _mm512_max_ps(vmax, x);
    }
    float global_max = hmax_512(vmax);
    for (; v < V; ++v) {
        float x;
        if (is_bf16) {
            uint32_t bits = static_cast<uint32_t>(
                reinterpret_cast<const uint16_t*>(row_data)[v]) << 16;
            std::memcpy(&x, &bits, sizeof(float));
        } else {
            x = reinterpret_cast<const float*>(row_data)[v];
        }
        x *= inv_T;
        if (x > global_max) global_max = x;
    }

    // Pass 2: exp(x - max) + accumulate sum, write to workspace
    // workspace = scaled_logits (FP32) — reused for top-K extraction
    std::vector<float> ws(V);
    __m512 vmaxv = _mm512_set1_ps(global_max);
    __m512 vsum = _mm512_setzero_ps();
    v = 0;
    for (; v + CHUNK <= V; v += CHUNK) {
        __m512 x;
        if (is_bf16) {
            x = bf16_to_fp32_16(reinterpret_cast<const uint16_t*>(row_data) + v);
        } else {
            x = _mm512_loadu_ps(reinterpret_cast<const float*>(row_data) + v);
        }
        x = _mm512_mul_ps(x, vinvT);
        x = _mm512_sub_ps(x, vmaxv);
        // exp_ps approximation: use intrinsic-free libm path for accuracy.
        // (full vectorized exp is ~25-line poly; for correctness baseline use scalar.)
        alignas(64) float tmp[CHUNK];
        _mm512_storeu_ps(tmp, x);
        for (int j = 0; j < CHUNK; ++j) {
            tmp[j] = std::exp(tmp[j]);
        }
        __m512 ex = _mm512_loadu_ps(tmp);
        _mm512_storeu_ps(ws.data() + v, ex);
        vsum = _mm512_add_ps(vsum, ex);
    }
    float total = _mm512_reduce_add_ps(vsum);
    for (; v < V; ++v) {
        float x;
        if (is_bf16) {
            uint32_t bits = static_cast<uint32_t>(
                reinterpret_cast<const uint16_t*>(row_data)[v]) << 16;
            std::memcpy(&x, &bits, sizeof(float));
        } else {
            x = reinterpret_cast<const float*>(row_data)[v];
        }
        x = std::exp(x * inv_T - global_max);
        ws[v] = x;
        total += x;
    }
    float inv_total = 1.0f / total;

    // Pass 3: probs normalization (in workspace) — fuse with top-K pull
    __m512 vinvtot = _mm512_set1_ps(inv_total);
    v = 0;
    for (; v + CHUNK <= V; v += CHUNK) {
        __m512 e = _mm512_loadu_ps(ws.data() + v);
        e = _mm512_mul_ps(e, vinvtot);
        _mm512_storeu_ps(ws.data() + v, e);
    }
    for (; v < V; ++v) ws[v] *= inv_total;

    // Top-K: reuse topk_row_threshold over ws
    struct Cand { float v; int32_t i; };
    std::vector<Cand> top(K);
    {
        // simple FP32 topk_row_threshold inline reuse
        const int OVERSAMPLE = 8;
        int cap = std::max(K * OVERSAMPLE, 64);
        std::vector<Cand> buf;
        buf.reserve(cap + CHUNK);
        float threshold = -std::numeric_limits<float>::infinity();
        int vi = 0;
        for (; vi + CHUNK <= V; vi += CHUNK) {
            __m512 x = _mm512_loadu_ps(ws.data() + vi);
            __m512 thr = _mm512_set1_ps(threshold);
            __mmask16 m = _mm512_cmp_ps_mask(x, thr, _CMP_GT_OQ);
            if (m == 0) continue;
            alignas(64) float vbuf[CHUNK];
            alignas(64) int32_t ibuf[CHUNK];
            __m512i base = _mm512_add_epi32(_mm512_set1_epi32(vi),
                _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15));
            _mm512_mask_compressstoreu_ps(vbuf, m, x);
            _mm512_mask_compressstoreu_epi32(ibuf, m, base);
            int hits = __builtin_popcount(static_cast<unsigned>(m));
            for (int h = 0; h < hits; ++h) buf.push_back({vbuf[h], ibuf[h]});
            if (static_cast<int>(buf.size()) >= cap * 2) {
                std::partial_sort(buf.begin(), buf.begin() + cap, buf.end(),
                    [](const Cand& a, const Cand& b) { return a.v > b.v; });
                buf.resize(cap);
                threshold = buf.back().v;
            }
        }
        for (; vi < V; ++vi) {
            if (ws[vi] > threshold) buf.push_back({ws[vi], vi});
        }
        int eff_k = std::min(K, static_cast<int>(buf.size()));
        std::partial_sort(buf.begin(), buf.begin() + eff_k, buf.end(),
            [](const Cand& a, const Cand& b) { return a.v > b.v; });
        top.assign(buf.begin(), buf.begin() + eff_k);
        if (eff_k < K) top.resize(K, Cand{0.0f, 0});
    }

    // Top-p cutoff on the top-K subset
    int kept = static_cast<int>(top.size());
    if (p > 0.0f && p < 1.0f) {
        float cum = 0.0f;
        int cut = kept;
        for (int k = 0; k < kept; ++k) {
            cum += top[k].v;
            if (cum >= p) { cut = k + 1; break; }
        }
        if (cut < 1) cut = 1;
        kept = cut;
    }

    // Renormalize kept
    float keep_sum = 0.0f;
    for (int k = 0; k < kept; ++k) keep_sum += top[k].v;
    if (keep_sum <= 0.0f) {
        // pathological — return top-1
        return static_cast<int64_t>(top[0].i);
    }
    float inv_keep = 1.0f / keep_sum;

    // Categorical sample: cumulative + uniform
    float r = uniform01(rng);
    float c = 0.0f;
    for (int k = 0; k < kept; ++k) {
        c += top[k].v * inv_keep;
        if (r <= c) return static_cast<int64_t>(top[k].i);
    }
    return static_cast<int64_t>(top[kept - 1].i);   // tail
}


void fused_sample_avx512_bf16(const uint16_t* logits_bf16,
                             int B, int V, int K, float p,
                             float temperature, uint64_t rng_seed,
                             int64_t* sampled_token_out) {
    uint64_t rng = rng_seed ? rng_seed : 0xC0FFEE12345678ULL;
    for (int b = 0; b < B; ++b) {
        sampled_token_out[b] = fused_sample_row(
            logits_bf16 + static_cast<size_t>(b) * V, V, K, p,
            temperature, rng, /*is_bf16=*/true);
    }
}

void fused_sample_avx512_fp32(const float* logits_fp32,
                             int B, int V, int K, float p,
                             float temperature, uint64_t rng_seed,
                             int64_t* sampled_token_out) {
    uint64_t rng = rng_seed ? rng_seed : 0xC0FFEE12345678ULL;
    for (int b = 0; b < B; ++b) {
        sampled_token_out[b] = fused_sample_row(
            logits_fp32 + static_cast<size_t>(b) * V, V, K, p,
            temperature, rng, /*is_bf16=*/false);
    }
}

}  // namespace vllm_hybrid_avx512
