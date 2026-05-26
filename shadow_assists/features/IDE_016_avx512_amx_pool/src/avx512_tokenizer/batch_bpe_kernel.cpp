// IDE_016 / TSK_024 / SUB_171 — AVX-512 batch tokenizer kernel impl
//
// 본 파일은 detokenize hot path (token_id → bytes lookup + concat) 를 AVX-512
// 로 vectorize 한다. Qwen 2.5 의 vocab 152,064 / piece 평균 4-5 bytes 환경에서
// pure Python loop 의 byte concat 대비 16× lane parallelism + 64-byte stream
// memcpy 로 latency 감소 목표.

#include "tokenizer_kernels.h"

#include <immintrin.h>
#include <cstring>
#include <algorithm>
#include <cstdio>

namespace vllm_hybrid_tok {

// ──────────────────────────────────────────────────────────────────────
// Scalar fallback (correctness ground-truth + non-AVX-512 path)
// ──────────────────────────────────────────────────────────────────────

void batch_detokenize_bytes_scalar(
    const VocabTable& table,
    const int32_t*    token_ids,
    const int32_t*    seq_offsets,
    int               B,
    uint8_t*          out_bytes,
    int32_t*          out_byte_offsets,
    int32_t*          out_byte_lengths) {

    out_byte_offsets[0] = 0;
    int32_t write_cursor = 0;

    for (int b = 0; b < B; ++b) {
        int32_t tok_lo = seq_offsets[b];
        int32_t tok_hi = seq_offsets[b + 1];
        int32_t seq_start = write_cursor;

        for (int32_t t = tok_lo; t < tok_hi; ++t) {
            int32_t tid = token_ids[t];
            if (tid < 0 || tid >= table.V) {
                // unknown token — emit nothing (vLLM contract: skip)
                continue;
            }
            int32_t off = table.offsets[tid];
            int32_t sz  = table.sizes[tid];
            if (sz > 0) {
                std::memcpy(out_bytes + write_cursor,
                            table.pieces + off,
                            static_cast<size_t>(sz));
                write_cursor += sz;
            }
        }

        out_byte_lengths[b] = write_cursor - seq_start;
        out_byte_offsets[b + 1] = write_cursor;
    }
}


// ──────────────────────────────────────────────────────────────────────
// AVX-512 path: per-sequence 16-wide size gather + prefix sum + memcpy
// ──────────────────────────────────────────────────────────────────────
//
// 동작:
//   sequence 단위로 token chunks 16개씩 처리.
//   1. _mm512_i32gather_epi32 로 sizes[tid_0..tid_15] 를 한 번에 fetch.
//   2. _mm512_i32gather_epi32 로 offsets[tid_0..tid_15] 도 fetch (next-step
//      memcpy source pointer 계산용).
//   3. AVX-512 prefix sum (Kogge-Stone 4 stage shift-add) 로 local
//      write_cursor 16개 결정 → SoA store.
//   4. 각 lane 의 piece 를 memcpy. SIMD copy 는 piece 길이가 짧으면
//      (Qwen 평균 4-5 bytes) overhead 가 더 커서 scalar memcpy 가 더 빠름.
//      하지만 size>=64 path 는 vmovdqu64 stream 사용.

#if defined(__AVX512F__) && defined(__AVX512BW__)

static inline __m512i prefix_sum_epi32(__m512i v) {
    // Hillis-Steele inclusive prefix sum, 4 stages for 16 lanes.
    __m512i s;
    s = _mm512_alignr_epi32(v, _mm512_setzero_si512(), 15);  // shift right 1
    v = _mm512_add_epi32(v, s);
    s = _mm512_alignr_epi32(v, _mm512_setzero_si512(), 14);  // shift right 2
    v = _mm512_add_epi32(v, s);
    s = _mm512_alignr_epi32(v, _mm512_setzero_si512(), 12);  // shift right 4
    v = _mm512_add_epi32(v, s);
    s = _mm512_alignr_epi32(v, _mm512_setzero_si512(), 8);   // shift right 8
    v = _mm512_add_epi32(v, s);
    return v;
}

static inline void copy_piece_simd(uint8_t*       dst,
                                   const uint8_t* src,
                                   int32_t        n) {
    // 64-byte block stream copy until tail, then masked store.
    int32_t i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i), v);
    }
    if (i < n) {
        int32_t rem = n - i;
        __mmask64 m = (rem >= 64) ? ~0ULL : ((1ULL << rem) - 1);
        __m512i v = _mm512_maskz_loadu_epi8(m, src + i);
        _mm512_mask_storeu_epi8(dst + i, m, v);
    }
}

void batch_detokenize_bytes_avx512(
    const VocabTable& table,
    const int32_t*    token_ids,
    const int32_t*    seq_offsets,
    int               B,
    uint8_t*          out_bytes,
    int32_t*          out_byte_offsets,
    int32_t*          out_byte_lengths) {

    out_byte_offsets[0] = 0;
    int32_t write_cursor = 0;
    const int32_t V = table.V;

    alignas(64) int32_t sizes_buf[16];
    alignas(64) int32_t offsets_buf[16];
    alignas(64) int32_t prefix_buf[16];

    for (int b = 0; b < B; ++b) {
        int32_t tok_lo   = seq_offsets[b];
        int32_t tok_hi   = seq_offsets[b + 1];
        int32_t seq_start = write_cursor;
        int32_t t = tok_lo;

        // Vectorized chunks of 16 token_ids
        for (; t + 16 <= tok_hi; t += 16) {
            __m512i tids = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(token_ids + t));

            // mask out OOB token ids
            __mmask16 valid = _mm512_cmplt_epi32_mask(
                tids, _mm512_set1_epi32(V));
            valid &= _mm512_cmpge_epi32_mask(
                tids, _mm512_setzero_si512());

            // Gather sizes[tids] and offsets[tids] (lanes invalid → 0)
            __m512i sizes_v   = _mm512_mask_i32gather_epi32(
                _mm512_setzero_si512(), valid, tids, table.sizes,   4);
            __m512i offsets_v = _mm512_mask_i32gather_epi32(
                _mm512_setzero_si512(), valid, tids, table.offsets, 4);

            // store gathered offsets for memcpy step
            _mm512_store_si512(reinterpret_cast<__m512i*>(sizes_buf),   sizes_v);
            _mm512_store_si512(reinterpret_cast<__m512i*>(offsets_buf), offsets_v);

            // local prefix sum (exclusive) → write positions
            __m512i pref_inc = prefix_sum_epi32(sizes_v);
            // shift to exclusive: subtract sizes_v
            __m512i pref_exc = _mm512_sub_epi32(pref_inc, sizes_v);
            _mm512_store_si512(reinterpret_cast<__m512i*>(prefix_buf), pref_exc);

            // total sum (last inclusive lane)
            int32_t chunk_total =
                _mm_extract_epi32(_mm512_extracti32x4_epi32(pref_inc, 3), 3);

            // Per-lane memcpy
            uint8_t* dst_chunk = out_bytes + write_cursor;
            for (int lane = 0; lane < 16; ++lane) {
                int32_t sz = sizes_buf[lane];
                if (sz == 0) continue;
                copy_piece_simd(dst_chunk + prefix_buf[lane],
                                table.pieces + offsets_buf[lane],
                                sz);
            }
            write_cursor += chunk_total;
        }

        // Scalar tail
        for (; t < tok_hi; ++t) {
            int32_t tid = token_ids[t];
            if (tid < 0 || tid >= V) continue;
            int32_t sz  = table.sizes[tid];
            int32_t off = table.offsets[tid];
            if (sz > 0) {
                std::memcpy(out_bytes + write_cursor,
                            table.pieces + off,
                            static_cast<size_t>(sz));
                write_cursor += sz;
            }
        }

        out_byte_lengths[b] = write_cursor - seq_start;
        out_byte_offsets[b + 1] = write_cursor;
    }
}

int64_t batch_detokenize_byte_total(const VocabTable& table,
                                    const int32_t*    token_ids,
                                    int               total_tokens) {
    const int32_t V = table.V;
    __m512i acc = _mm512_setzero_si512();
    int t = 0;
    for (; t + 16 <= total_tokens; t += 16) {
        __m512i tids = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(token_ids + t));
        __mmask16 valid = _mm512_cmplt_epi32_mask(tids, _mm512_set1_epi32(V));
        valid &= _mm512_cmpge_epi32_mask(tids, _mm512_setzero_si512());
        __m512i sizes_v = _mm512_mask_i32gather_epi32(
            _mm512_setzero_si512(), valid, tids, table.sizes, 4);
        acc = _mm512_add_epi32(acc, sizes_v);
    }
    int64_t total = _mm512_reduce_add_epi32(acc);
    for (; t < total_tokens; ++t) {
        int32_t tid = token_ids[t];
        if (tid >= 0 && tid < V) total += table.sizes[tid];
    }
    return total;
}

#else

// AVX-512 absent — alias to scalar (e.g. dev machine fuse-off path).
void batch_detokenize_bytes_avx512(
    const VocabTable& table,
    const int32_t*    token_ids,
    const int32_t*    seq_offsets,
    int               B,
    uint8_t*          out_bytes,
    int32_t*          out_byte_offsets,
    int32_t*          out_byte_lengths) {
    batch_detokenize_bytes_scalar(table, token_ids, seq_offsets, B,
                                  out_bytes, out_byte_offsets, out_byte_lengths);
}

int64_t batch_detokenize_byte_total(const VocabTable& table,
                                    const int32_t*    token_ids,
                                    int               total_tokens) {
    const int32_t V = table.V;
    int64_t total = 0;
    for (int t = 0; t < total_tokens; ++t) {
        int32_t tid = token_ids[t];
        if (tid >= 0 && tid < V) total += table.sizes[tid];
    }
    return total;
}

#endif  // __AVX512F__


// ──────────────────────────────────────────────────────────────────────
// BPE pair min-rank scan — skeleton only (SUB_172 canonical 측정 시 활성)
// ──────────────────────────────────────────────────────────────────────

void batch_bpe_min_rank_avx512(
    const int32_t* rank_table_flat,
    int32_t        rank_table_dim,
    const int32_t* pair_ids,
    int            B,
    int            num_pairs,
    int32_t*       best_idx) {

#if defined(__AVX512F__)
    const __m512i INFINITY_V = _mm512_set1_epi32(INT32_MAX);
    const int32_t dim_sq = rank_table_dim * rank_table_dim;

    for (int b = 0; b < B; ++b) {
        const int32_t* row = pair_ids + b * num_pairs;
        __m512i best_rank = INFINITY_V;
        __m512i best_lane = _mm512_set1_epi32(-1);
        __m512i lane_id   = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
        __m512i lane_step = _mm512_set1_epi32(16);

        int p = 0;
        for (; p + 16 <= num_pairs; p += 16) {
            __m512i pid = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(row + p));
            __mmask16 valid = _mm512_cmplt_epi32_mask(pid,
                              _mm512_set1_epi32(dim_sq));
            valid &= _mm512_cmpge_epi32_mask(pid, _mm512_setzero_si512());
            __m512i rank = _mm512_mask_i32gather_epi32(
                INFINITY_V, valid, pid, rank_table_flat, 4);

            __mmask16 better = _mm512_cmplt_epi32_mask(rank, best_rank);
            best_rank = _mm512_mask_mov_epi32(best_rank, better, rank);
            best_lane = _mm512_mask_mov_epi32(best_lane, better, lane_id);
            lane_id   = _mm512_add_epi32(lane_id, lane_step);
        }

        // reduce 16-lane min
        alignas(64) int32_t rb[16], lb[16];
        _mm512_store_si512(reinterpret_cast<__m512i*>(rb), best_rank);
        _mm512_store_si512(reinterpret_cast<__m512i*>(lb), best_lane);
        int32_t bestr = INT32_MAX;
        int32_t bestl = -1;
        for (int k = 0; k < 16; ++k) {
            if (rb[k] < bestr) { bestr = rb[k]; bestl = lb[k]; }
        }
        for (; p < num_pairs; ++p) {
            int32_t pid = row[p];
            if (pid < 0 || pid >= dim_sq) continue;
            int32_t r = rank_table_flat[pid];
            if (r < bestr) { bestr = r; bestl = p; }
        }
        best_idx[b] = (bestr == INT32_MAX) ? -1 : bestl;
    }
#else
    for (int b = 0; b < B; ++b) {
        const int32_t* row = pair_ids + b * num_pairs;
        int32_t bestr = INT32_MAX, bestl = -1;
        const int32_t dim_sq = rank_table_dim * rank_table_dim;
        for (int p = 0; p < num_pairs; ++p) {
            int32_t pid = row[p];
            if (pid < 0 || pid >= dim_sq) continue;
            int32_t r = rank_table_flat[pid];
            if (r < bestr) { bestr = r; bestl = p; }
        }
        best_idx[b] = (bestr == INT32_MAX) ? -1 : bestl;
    }
#endif
}

}  // namespace vllm_hybrid_tok
