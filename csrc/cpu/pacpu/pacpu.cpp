#include <torch/library.h>
#include <ATen/Tensor.h>

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

  // P3 (F3) — K dtype 분기. BF16 시 host store 그대로 (bit-pattern preserve,
  //   sizeof 동일). Q, V 는 FP16 hard-coded (Q AMX 안 BF16 conv, V ISPC).
  bool k_is_bf16 = (k.scalar_type() == at::kBFloat16);
  if (k_is_bf16) {
    TORCH_CHECK(k_cache.scalar_type() == at::kBFloat16,
                "P3: k staging BF16 then k_cache must also be BF16");
    TORCH_CHECK(q.scalar_type() == at::kHalf,
                "P3: q must be FP16 (AMX path converts internally)");
    TORCH_CHECK(v.scalar_type() == at::kHalf,
                "P3: v must be FP16 (ISPC av_product hard-coded)");
    TORCH_CHECK(v_cache.scalar_type() == at::kHalf,
                "P3: v_cache must be FP16 (av_product reads FP16)");
  }

  auto qbatch_p = (data_t*) q.data_ptr<at_data_t>();
  auto vbatch_p = (data_t*) v.data_ptr<at_data_t>();
  auto obatch_p = o.data_ptr<otpt_t>();
  auto vcache_p = (data_t*) v_cache.data_ptr<at_data_t>();
  auto block_table_p = block_table.data_ptr<int32_t>(); // [batch_size, max_seq_len]
  // K pointer: BF16 시 raw bit pointer (사용은 ispc_attention_tasks 안의
  //   store_kv memcpy + AMX BF16 path 만, ISPC qk path 진입 X).
  data_t* kbatch_p;
  data_t* kcache_p;
  if (k_is_bf16) {
    kbatch_p = reinterpret_cast<data_t*>(k.data_ptr<at::BFloat16>());
    kcache_p = reinterpret_cast<data_t*>(k_cache.data_ptr<at::BFloat16>());
  } else {
    kbatch_p = (data_t*) k.data_ptr<at_data_t>();
    kcache_p = (data_t*) k_cache.data_ptr<at_data_t>();
  }

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
      qbatch_p, kbatch_p, vbatch_p, obatch_p, kcache_p, vcache_p, block_table_p,
      k_is_bf16   // P3 flag
    );
  #endif
}

// SUB_021 (scatter_layers_into_buf) 제거됨 (2026-05-14):
// 500p 측정 시 -4% 회귀 (OMP 경합 with cdec). ATen 의 vectorized index_put_
// 이 naive memcpy OMP 보다 빠른 것 입증.

TORCH_LIBRARY(pacpu, m) {
#ifdef USE_ATEN_OPER
  m.def("paged_attention_cpu_torch", &paged_attention_cpu_torch);
#endif
  m.def("paged_attention_cpu", &paged_attention_cpu);
}