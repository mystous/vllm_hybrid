#include <torch/library.h>
#include <ATen/Tensor.h>
#include <cstring>   // SUB_021 — std::memcpy

#include "dtype.h"
#include "core.h"

typedef at::Half at_data_t;

// #define USE_ATEN_OPER
#ifdef USE_ATEN_OPER
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>

void paged_attention_cpu_torch(
  int64_t cur_layer,
  double softmax_scale,
	const std::vector<int64_t> &decoding_seq_ids,
	const std::vector<int64_t> &decoding_seq_lengths,

  at::Tensor q,
  at::Tensor k,
  at::Tensor v,
	at::Tensor k_cache,
	at::Tensor v_cache,
  at::Tensor block_table,
  at::Tensor o
) {
  for (auto i = 0; i < decoding_seq_ids.size(); i++) {
    auto seq_id = decoding_seq_ids[i];
    auto seq_len = decoding_seq_lengths[i];
    at::Tensor qi = q.index({i});
    at::Tensor ki = k.index({i});
    at::Tensor vi = v.index({i});
    auto blkid = (seq_len - 1) / BLOCK_SIZE;
    auto blkoff = (seq_len - 1) % BLOCK_SIZE;
    std::cout << k_cache.sizes() << std::endl;
    k_cache.index_put_({cur_layer, blkid, at::indexing::Slice(), blkoff, at::indexing::Slice()}, ki);
    v_cache.index_put_({cur_layer, blkid, at::indexing::Slice(), blkoff, at::indexing::Slice()}, vi);
    at::Tensor nk = k_cache.index({cur_layer, block_ids}).permute({1, 2, 0}).to(at::kFloat);
    at::Tensor nv = v_cache.index({cur_layer, block_ids}).permute({1, 0, 2}).to(at::kFloat);
    at::Tensor attn_score = (at::bmm(qi, nk) * softmax_scale).softmax(-1);
    o.index_put_({i}, at::bmm(attn_score, nv).view(-1));
  }
}
#endif

void assert_hyper_params_expected(int num_q_heads, int num_kv_heads, int num_layers, int head_dim, int block_size) {
  if (num_q_heads != NUM_Q_HEADS) {
    throw std::invalid_argument("expected num_q_heads to be " + std::to_string(NUM_Q_HEADS) + ", but got " + std::to_string(num_q_heads));
  }
  if (num_kv_heads != NUM_KV_HEADS) {
    throw std::invalid_argument("expected num_kv_heads to be " + std::to_string(NUM_KV_HEADS) + ", but got " + std::to_string(num_kv_heads));
  }
  // vLLM 은 per-layer 별로 KV cache 를 분리 보관 — 한 호출에서는 한 layer
  // 만 보임 (caller 가 cur_layer=0 + num_layers=1 로 전달). NEO 원본은
  // multi-layer KV 를 한 buffer 에 두므로 num_layers == NUM_LAYERS 강제
  // 였지만, kernel 의 실제 offset 계산은 ``cur_layer * num_blocks`` 라
  // num_layers 자체가 필수 NUM_LAYERS 일 필요는 없음. 완화: cur_layer <
  // num_layers 만 보장.
  if (num_layers <= 0 || num_layers > NUM_LAYERS) {
    throw std::invalid_argument("expected num_layers to be in (0, " + std::to_string(NUM_LAYERS) + "], but got " + std::to_string(num_layers));
  }
  if (head_dim != HEAD_DIM) {
    throw std::invalid_argument("expected head_dim to be " + std::to_string(HEAD_DIM) + ", but got " + std::to_string(head_dim));
  }
  if (block_size != BLOCK_SIZE) {
    throw std::invalid_argument("expected block_size to be " + std::to_string(BLOCK_SIZE) + ", but got " + std::to_string(block_size));
  }
}

/*
 * Paged attention, contains 3 implementations:
 */

#define USE_ISPC_TASKS_OPER
// #define USE_ISPC_OPER

void paged_attention_cpu(
  int64_t cur_layer,
  double softmax_scale,
	const std::vector<int64_t> &seq_ids,
	const std::vector<int64_t> &seq_lengths,

  at::Tensor q, // [batch_size, num_q_heads, head_dim]
  at::Tensor k, // [batch_size, num_kv_heads, head_dim]
  at::Tensor v, // [batch_size, num_kv_heads, head_dim]
	at::Tensor k_cache, // [..., num_layers, num_kv_heads, block_size, head_dim]
	at::Tensor v_cache, // [..., num_layers, num_kv_heads, block_size, head_dim]
  at::Tensor block_table, // [..., max_seq_len]
  at::Tensor o // [batch_size, num_kv_heads * qh_per_kvh * head_dim]
) {
  int batch_size = q.size(0);
  int num_q_heads = q.size(1);
  int num_layers = k_cache.size(0);
  int num_blocks = k_cache.size(1);
  int num_kv_heads = k_cache.size(2);
  int block_size = k_cache.size(3);
  int head_dim = k_cache.size(4);
  int block_table_width = block_table.size(1);

  assert_hyper_params_expected(num_q_heads, num_kv_heads, num_layers, head_dim, block_size);

  auto qbatch_p = (data_t*) q.data_ptr<at_data_t>();
  auto kbatch_p = (data_t*) k.data_ptr<at_data_t>();
  auto vbatch_p = (data_t*) v.data_ptr<at_data_t>();
  auto obatch_p = o.data_ptr<otpt_t>();
  auto kcache_p = (data_t*) k_cache.data_ptr<at_data_t>();
  auto vcache_p = (data_t*) v_cache.data_ptr<at_data_t>();
  auto block_table_p = block_table.data_ptr<int32_t>(); // [batch_size, max_seq_len]

  #ifdef USE_BRUTE_OPER
    brute_attention(
      cur_layer, num_blocks, batch_size, block_table_width, softmax_scale,
      seq_ids, seq_lengths,
      qbatch_p, kbatch_p, vbatch_p, obatch_p, kcache_p, vcache_p, block_table_p
    );
  #elifdef USE_ISPC_OPER
    ispc_attention(
      cur_layer, num_blocks, batch_size, block_table_width, softmax_scale,
      seq_ids, seq_lengths,
      qbatch_p, kbatch_p, vbatch_p, obatch_p, kcache_p, vcache_p, block_table_p
    );
  #elifdef USE_ISPC_TASKS_OPER
    ispc_attention_tasks(
      cur_layer, num_blocks, batch_size, block_table_width, softmax_scale,
      seq_ids, seq_lengths,
      qbatch_p, kbatch_p, vbatch_p, obatch_p, kcache_p, vcache_p, block_table_p
    );
  #endif
}

// =====================================================================
// SUB_021 — Layer (iii) #2: C extension batched copy.
//
// Replaces SUB_028's Python-side CPU scatter (``copy_all_layers_in_from_staged``)
// with a C++ + OpenMP parallelized version. The H2D itself remains in
// PyTorch (cudaMemcpyAsync via ``.copy_()``), but the *CPU scatter*
// from pinned staging tensor → NEO buffer (per-layer index_put_) is
// where Python loop + ATen scatter has per-layer overhead. With 80
// layers × N blocks per req, parallel scatter across layers (OMP) can
// saturate DDR4 bandwidth (~50 GB/s) where the serial loop typically
// hits ~10 GB/s due to per-iteration ATen dispatch cost.
//
// NEO reference: ``swiftllm/worker/block_swapper.py:42-54`` —
// pinned dst + non_blocking H2D batched. NEO 의 C ext 는 H2D 까지
// 직접 호출 (cudaMemcpyAsync). PyTorch 기반 vllm_hybrid 는 H2D 가 이미
// ATen 의 fast path 라 추가 이득 없음 → CPU scatter 만 C++ 화.
// =====================================================================

void scatter_layers_into_buf(
    at::Tensor k_buf_full,    // (num_layers, total_blocks, num_kv_heads, block_size, head_dim) — pinned CPU
    at::Tensor v_buf_full,
    at::Tensor k_staged,       // (num_layers, max_blocks_per_req, ...) — pinned CPU, H2D 완료된 source
    at::Tensor v_staged,
    const std::vector<int64_t> &block_ids,  // NEO buffer 내 destination block ids
    int64_t n_blocks            // 유효한 staged blocks (≤ k_staged.size(1))
) {
  TORCH_CHECK(k_buf_full.dim() == 5 && v_buf_full.dim() == 5,
              "k_buf_full / v_buf_full must be 5D (num_layers, total_blocks, num_kv_heads, block_size, head_dim)");
  TORCH_CHECK(k_staged.dim() == 5 && v_staged.dim() == 5,
              "k_staged / v_staged must be 5D");
  TORCH_CHECK(k_buf_full.device().is_cpu() && v_buf_full.device().is_cpu()
              && k_staged.device().is_cpu() && v_staged.device().is_cpu(),
              "all tensors must be on CPU (pinned)");
  TORCH_CHECK(k_buf_full.scalar_type() == k_staged.scalar_type(),
              "k_buf_full and k_staged dtype must match");
  TORCH_CHECK(v_buf_full.scalar_type() == v_staged.scalar_type(),
              "v_buf_full and v_staged dtype must match");
  TORCH_CHECK(static_cast<int64_t>(block_ids.size()) >= n_blocks,
              "block_ids size must be >= n_blocks");

  int64_t num_layers = k_buf_full.size(0);
  int64_t per_block_elems = k_buf_full.size(2) * k_buf_full.size(3) * k_buf_full.size(4);
  int64_t total_blocks = k_buf_full.size(1);
  int64_t elem_size = k_buf_full.element_size();
  int64_t per_block_bytes = per_block_elems * elem_size;

  // staged stride along layer axis (in elements then bytes).
  int64_t staged_layer_stride_elems = k_staged.size(1) * per_block_elems;
  int64_t staged_layer_stride_bytes = staged_layer_stride_elems * elem_size;
  // staged stride along block axis (within one layer).
  int64_t staged_block_stride_bytes = per_block_elems * elem_size;

  // dst (k_buf_full) stride along layer axis.
  int64_t dst_layer_stride_bytes = total_blocks * per_block_elems * elem_size;
  int64_t dst_block_stride_bytes = per_block_elems * elem_size;

  uint8_t *k_dst_base = reinterpret_cast<uint8_t*>(k_buf_full.data_ptr());
  uint8_t *v_dst_base = reinterpret_cast<uint8_t*>(v_buf_full.data_ptr());
  const uint8_t *k_src_base = reinterpret_cast<const uint8_t*>(k_staged.data_ptr());
  const uint8_t *v_src_base = reinterpret_cast<const uint8_t*>(v_staged.data_ptr());

  // OMP parallel across (layer, block_id) pairs. Memory bandwidth bound,
  // 80 × n_blocks tasks. Each task copies per_block_bytes (typically 4-8 KB)
  // from contiguous staged region into scattered NEO buffer rows.
  // Threads are limited by OMP_NUM_THREADS (default 10 in NEO production).
  #pragma omp parallel for collapse(2) schedule(static)
  for (int64_t layer = 0; layer < num_layers; layer++) {
    for (int64_t b = 0; b < n_blocks; b++) {
      int64_t dst_blk = block_ids[b];
      // K
      const uint8_t *k_src = k_src_base + layer * staged_layer_stride_bytes
                                        + b * staged_block_stride_bytes;
      uint8_t *k_dst = k_dst_base + layer * dst_layer_stride_bytes
                                  + dst_blk * dst_block_stride_bytes;
      std::memcpy(k_dst, k_src, per_block_bytes);
      // V
      const uint8_t *v_src = v_src_base + layer * staged_layer_stride_bytes
                                        + b * staged_block_stride_bytes;
      uint8_t *v_dst = v_dst_base + layer * dst_layer_stride_bytes
                                  + dst_blk * dst_block_stride_bytes;
      std::memcpy(v_dst, v_src, per_block_bytes);
    }
  }
}

TORCH_LIBRARY(pacpu, m) {
#ifdef USE_ATEN_OPER
  m.def("paged_attention_cpu_torch", &paged_attention_cpu_torch);
#endif
  m.def("paged_attention_cpu", &paged_attention_cpu);
  m.def("scatter_layers_into_buf", &scatter_layers_into_buf);  // SUB_021
}