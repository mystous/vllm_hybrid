// IDE_016 / TSK_024 / SUB_171 — AVX-512 batch tokenizer kernel public API
//
// Scope (본 SUB):
//   batch detokenize 의 hot path 를 vectorize 한다. vLLM 의 detokenize 경로는
//   `vllm/v1/engine/detokenizer.py` 의 `FastIncrementalDetokenizer` 가 사용하는
//   tokenizers (Rust) DecodeStream 가 majority 이고, 그 안에서 일어나는
//   주요 hot path 가
//     (1) token_id → vocab piece bytes lookup,
//     (2) byte concat + length sum,
//     (3) BPE pair merge rank scan (encode 시).
//   본 kernel 은 (1)+(2) 를 batch 32 sequence × variable length 로 처리하는
//   AVX-512 SoA path 를 제공한다 (가장 hot 한 detokenize 경로).
//
// dtype 약속:
//   token_ids   : int32 (vocab ≤ 2³¹).  Qwen 2.5 = 152,064 vocab.
//   piece bytes : uint8 (vocab table flatten — `pieces` + `offsets`).
//   length sums : int32.
//
// 별도 batch_bpe_search_avx512 는 BPE merge rank lookup 의 16-way SIMD
// gather 경로 (encode 보조). 현재는 detokenize 가 우선이므로 stub 만 둔다.

#pragma once

#include <cstdint>
#include <cstddef>

namespace vllm_hybrid_tok {

// ──────────────────────────────────────────────────────────────────────
// Vocab table layout
// ──────────────────────────────────────────────────────────────────────
//
// 호출자가 build_vocab_table 로 한 번 build 한 뒤 batch_detokenize_bytes 가
// 그 table 을 재사용한다. table 은 read-only 라 thread-safe.
//
//   pieces  : flat byte stream of all piece UTF-8 bytes (length = total_bytes)
//   offsets : [V+1] int32 ; piece i 는 pieces[offsets[i] .. offsets[i+1]).
//   sizes   : [V]   int32 ; sizes[i] = offsets[i+1] - offsets[i].  (성능 cache)

struct VocabTable {
    const uint8_t* pieces;     // length = total_bytes
    const int32_t* offsets;    // length = V + 1
    const int32_t* sizes;      // length = V
    int32_t V;
    int32_t total_bytes;
};


// ──────────────────────────────────────────────────────────────────────
// Batch detokenize — primary entry point
// ──────────────────────────────────────────────────────────────────────

/// AVX-512 batch detokenize.
/// Input:
///   table     : pre-built VocabTable (vocab_size V).
///   token_ids : flat int32 [total_tokens], sequence i 의 token_ids 는
///               token_ids[seq_offsets[i] .. seq_offsets[i+1]).
///   seq_offsets : [B+1] int32 ; per-sequence prefix offset.
///   B         : batch size.
/// Output:
///   out_bytes : pre-allocated [B][max_seq_bytes] flat write buffer
///               (호출자가 conservative size 로 alloc, e.g. sum sizes).
///   out_byte_offsets : [B+1] int32 ; output prefix per-sequence.
///   out_byte_lengths : [B]   int32 ; final per-sequence byte length.
///
/// 알고리즘:
///   1. per-sequence pass (parallelize across B with OpenMP if available).
///   2. AVX-512 prefix-sum (16-wide vpsadbw / vpermilps) on token sizes.
///   3. memcpy per-piece (cache-friendly stream) → write into out_bytes.
///
/// SIMD lanes:
///   - sizes gather: VPSCATTERDD-paired VPGATHERDD on offsets.
///   - prefix sum: hillis–steele 16-wide on int32.
///   - byte memcpy: 64-byte vmovdqu64 stream when piece >= 64, else scalar tail.
///
/// 정확도: token-level exact match vs python tokenizer.decode (BPE/SP 는
/// deterministic; vocab bytes 가 같으면 결과 동일).
void batch_detokenize_bytes_avx512(
    const VocabTable& table,
    const int32_t*    token_ids,
    const int32_t*    seq_offsets,
    int               B,
    uint8_t*          out_bytes,
    int32_t*          out_byte_offsets,
    int32_t*          out_byte_lengths);


/// Scalar fallback (correctness reference, also used when CPUID has no avx512).
void batch_detokenize_bytes_scalar(
    const VocabTable& table,
    const int32_t*    token_ids,
    const int32_t*    seq_offsets,
    int               B,
    uint8_t*          out_bytes,
    int32_t*          out_byte_offsets,
    int32_t*          out_byte_lengths);


// ──────────────────────────────────────────────────────────────────────
// AVX-512 batch BPE merge-rank search (encode 보조 — stub for SUB_172)
// ──────────────────────────────────────────────────────────────────────

/// BPE pair merge rank lookup. Given B sequence of pair-ids, returns best
/// (min-rank) pair index per sequence. 본 SUB 는 skeleton 만 제공 — 실제
/// encode hook 은 SUB_172 의 canonical integration 에서 수행.
///
/// rank_table : [V*V] int32 — packed (left << 17) | right keyed rank lookup
///              (open-addressing hash bucket; 본 SUB 는 dense placeholder).
/// pair_ids   : [B][num_pairs] int32
/// best_idx   : [B] int32 — argmin pair index ; -1 if none below INT32_MAX.
void batch_bpe_min_rank_avx512(
    const int32_t* rank_table_flat,
    int32_t        rank_table_dim,   // square dim
    const int32_t* pair_ids,
    int            B,
    int            num_pairs,
    int32_t*       best_idx);


// ──────────────────────────────────────────────────────────────────────
// Utility
// ──────────────────────────────────────────────────────────────────────

/// Total output byte length for batch (used to size out_bytes alloc).
/// Cheap O(total_tokens) AVX-512 sum.
int64_t batch_detokenize_byte_total(const VocabTable& table,
                                    const int32_t*    token_ids,
                                    int               total_tokens);

}  // namespace vllm_hybrid_tok
