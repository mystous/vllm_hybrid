#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include "dtype.h"
#include "pacpu_ispc.h"

// SUB_015-Phase 3 A: AMX kernel forward decl (defined in amx_kernel.cpp).
namespace amx_kernel {
extern "C" void attn_one_seq_amx(
    int cur_layer, int num_blocks, int seq_len, float softmax_scale,
    const data_t* q, const data_t* k_cache, const data_t* v_cache,
    const int* block_table, itmd_t* a, otpt_t* o, itmd_t* asb);
// P3 (F3) — BF16-K variant. K BF16 raw bit pointer, V FP16 그대로.
extern "C" void attn_one_seq_amx_bf16(
    int cur_layer, int num_blocks, int seq_len, float softmax_scale,
    const data_t* q, const uint16_t* k_cache_bf16, const data_t* v_cache,
    const int* block_table, itmd_t* a, otpt_t* o, itmd_t* asb);
bool ensure_amx_init();
}

namespace brute {

void store_kv(
  int cur_layer,
  int num_blocks,
  int seq_len,
  data_t k[],
  data_t v[],
  data_t k_cache[],
  data_t v_cache[],
  int block_table[]
) {
  int block_pos = (seq_len - 1) / BLOCK_SIZE;
  int block_id = block_table[block_pos];
  int block_off = (seq_len - 1) % BLOCK_SIZE;
  int64_t cache_off = (1ll * cur_layer * num_blocks + block_id) * BLOCK_NELEM + block_off * HEAD_DIM;
  data_t* kp = k_cache + cache_off;
  data_t* vp = v_cache + cache_off;
  for (int i = 0; i < NUM_KV_HEADS; i++) {
    memcpy(kp + i * BLOCK_SIZE * HEAD_DIM, k + i * HEAD_DIM, HEAD_DIM * sizeof(data_t));
    memcpy(vp + i * BLOCK_SIZE * HEAD_DIM, v + i * HEAD_DIM, HEAD_DIM * sizeof(data_t));
  }
}

void qk_product(
  int cur_layer,
  int num_blocks,
  int seq_len,

  data_t q[],
  data_t k_cache[],
  int block_table[],

  itmd_t a[]
) {
  for (auto j = 0; j < seq_len; j += BLOCK_SIZE) {
    auto kp = k_cache + (1ll * cur_layer * num_blocks + block_table[j / BLOCK_SIZE]) * BLOCK_NELEM;
    auto tlim = std::min(BLOCK_SIZE, seq_len - j);
    for (auto h = 0; h < NUM_KV_HEADS; h++) {
      for (auto t = 0; t < tlim; t++) {
        for (auto d = 0; d < HEAD_DIM; d++) {
          for (auto l = 0; l < QH_PER_KVH; l++) {
            a[(j + t) * NUM_Q_HEADS + h * QH_PER_KVH + l] += 
              q[(h * QH_PER_KVH + l) * HEAD_DIM + d] * kp[(h * BLOCK_SIZE + t) * HEAD_DIM + d];
          }
        }
      }
    }
  }
}

void av_product(
  int cur_layer,
  int num_blocks,
  int seq_len,

  itmd_t a[],
  data_t v_cache[],
  int block_table[],

  otpt_t o[]
) {
  memset(o, 0, NUM_Q_HEADS * HEAD_DIM * sizeof(otpt_t));
  for (auto j = 0; j < seq_len; j += BLOCK_SIZE) {
    auto vjp = v_cache + (1ll * cur_layer * num_blocks + block_table[j / BLOCK_SIZE]) * BLOCK_NELEM;
    auto vp = vjp;
    auto tlim = std::min(BLOCK_SIZE, seq_len - j);
    for (auto h = 0; h < NUM_KV_HEADS; h++) {
      for (auto t = 0; t < tlim; t++) {
        for (auto d = 0; d < HEAD_DIM; d++) {
          for (auto l = 0; l < QH_PER_KVH; l++) {
            o[(h * QH_PER_KVH + l) * HEAD_DIM + d] += 
              a[(j + t) * NUM_Q_HEADS + h * QH_PER_KVH + l] * vp[(h * BLOCK_SIZE + t) * HEAD_DIM + d];
          }
        }
      }
    }
  }
}

void softmax(
  int seq_len,
  itmd_t softmax_scale,
  itmd_t a[],
  itmd_t s[],
  itmd_t m[]
) {
  for (auto h = 0; h < NUM_Q_HEADS; h++) {
    s[h] = 0;
    m[h] = -1e20;
  }
  
  auto ap = a;
  for (auto j = 0; j < seq_len; j++) {
    for (auto h = 0; h < NUM_Q_HEADS; h++) {
      ap[h] *= softmax_scale;
      m[h] = std::max(m[h], ap[h]);
    }
    ap += NUM_Q_HEADS;
  }

  ap = a;
  for (auto j = 0; j < seq_len; j++) {
    for (auto h = 0; h < NUM_Q_HEADS; h++) {
      ap[h] = std::exp(ap[h] - m[h]);
      s[h] += ap[h];
    }
    ap += NUM_Q_HEADS;
  }

  ap = a;
  for (auto j = 0; j < seq_len; j++) {
    for (auto h = 0; h < NUM_Q_HEADS; h++) {
      ap[h] /= s[h];
    }
    ap += NUM_Q_HEADS;
  }
}

}

void brute_attention(
  int cur_layer,
  int num_blocks,
  int batch_size,
  int block_table_width,
  double softmax_scale,
  const std::vector<int64_t> &seq_ids,
	const std::vector<int64_t> &seq_lengths,
  
  data_t qbatch_p[],
  data_t kbatch_p[],
  data_t vbatch_p[],
  otpt_t obatch_p[],
  data_t kcache_p[],
  data_t vcache_p[],
  int block_table_p[]
){
  auto max_seq_len = *std::max_element(seq_lengths.begin(), seq_lengths.end());
  itmd_t* attn_score_buf = new itmd_t[max_seq_len * NUM_Q_HEADS];
  itmd_t* attn_sum_buf = new itmd_t[NUM_Q_HEADS];
  itmd_t* attn_max_buf = new itmd_t[NUM_Q_HEADS];

  for (auto i = 0; i < batch_size; i++) {
    int seq_id = seq_ids[i];
    int seq_len = seq_lengths[i];
    auto qip = qbatch_p + i * NUM_Q_HEADS * HEAD_DIM;
    auto kip = kbatch_p + i * NUM_KV_HEADS * HEAD_DIM;
    auto vip = vbatch_p + i * NUM_KV_HEADS * HEAD_DIM;
    auto oip = obatch_p + i * NUM_Q_HEADS * HEAD_DIM;
    auto btp = block_table_p + seq_id * block_table_width;
    memset(attn_score_buf, 0, seq_len * NUM_Q_HEADS * sizeof(itmd_t));

    brute::store_kv(cur_layer, num_blocks, seq_len, kip, vip, kcache_p, vcache_p, btp);
    brute::qk_product(cur_layer, num_blocks, seq_len, qip, kcache_p, btp, attn_score_buf);
    brute::softmax(seq_len, softmax_scale, attn_score_buf, attn_sum_buf, attn_max_buf);
    brute::av_product(cur_layer, num_blocks, seq_len, attn_score_buf, vcache_p, btp, oip);
  }
  delete [] attn_score_buf;
  delete [] attn_sum_buf;
  delete [] attn_max_buf;
}

void ispc_attention(
  int cur_layer,
  int num_blocks,
  int batch_size,
  int block_table_width,
  double softmax_scale,
  const std::vector<int64_t> &seq_ids,
  const std::vector<int64_t> &seq_lengths,
  
  data_t qbatch_p[],
  data_t kbatch_p[],
  data_t vbatch_p[],
  otpt_t obatch_p[],
  data_t kcache_p[],
  data_t vcache_p[],
  int block_table_p[]
) {
  int max_seq_len = *std::max_element(seq_lengths.begin(), seq_lengths.end());
  itmd_t* attn_score_buf = new itmd_t[max_seq_len * NUM_Q_HEADS];
  itmd_t attn_sum_buf[NUM_Q_HEADS];

  for (auto i = 0; i < batch_size; i++) {
    int seq_id = seq_ids[i];
    int seq_len = seq_lengths[i];
    auto qip = qbatch_p + i * NUM_Q_HEADS * HEAD_DIM;
    auto kip = kbatch_p + i * NUM_KV_HEADS * HEAD_DIM;
    auto vip = vbatch_p + i * NUM_KV_HEADS * HEAD_DIM;
    auto oip = obatch_p + i * NUM_Q_HEADS * HEAD_DIM;
    auto btp = block_table_p + seq_id * block_table_width;

    brute::store_kv(
      cur_layer, num_blocks, seq_len, 
      kip, vip, kcache_p, vcache_p, btp
    );

    ispc::attn_one_seq(
      cur_layer, num_blocks, seq_len, softmax_scale,
      qip, kcache_p, vcache_p, btp, 
      attn_score_buf, oip, attn_sum_buf
    );
  }

  delete [] attn_score_buf;
}

// TSK_019 plan F12 진짜 fix — global buffer 폐기. multi-dispatch
// (max_workers > 1 cdec executor) 시 동일 global buffer 동시 쓰기로
// race + segfault. thread_local std::vector 사용 — 각 caller thread
// 별 별 buffer. 함수 내부 정의로 header multiple-definition 회피.
void ispc_attention_tasks(
  int cur_layer,
  int num_blocks,
  int batch_size,
  int block_table_width,
  double softmax_scale,
  const std::vector<int64_t> &seq_ids,
  const std::vector<int64_t> &seq_lengths,

  data_t qbatch_p[],
  data_t kbatch_p[],
  data_t vbatch_p[],
  otpt_t obatch_p[],
  data_t kcache_p[],
  data_t vcache_p[],
  int block_table_p[],
  bool k_is_bf16 = false   // P3 (F3) — host K BF16 store flag.
                            // K input + K cache 모두 BF16 bit pattern.
                            // store_kv 는 pure memcpy 라 dtype-agnostic.
                            // AMX path 의 attn_one_seq_amx_bf16 호출 분기.
                            // ISPC path 는 BF16 지원 X → AMX 강제 시 진입.
) {
  // Phase 3.1 — Persistent OMP team settings.
  // omp_set_dynamic(0): runtime 의 dynamic thread 수 조정 비활성화 (매 call thread
  //   count 변동 회피 → thread pool 안정성 + launch overhead 절감).
  // omp_set_max_active_levels(1): nested parallel 비활성화 (본 path single-level).
  // 매 call 호출되나 thread_local flag 로 1회만 실제 setter 호출 (cheap).
  static thread_local bool _omp_persistent_init = false;
  if (!_omp_persistent_init) {
    omp_set_dynamic(0);
    omp_set_max_active_levels(1);
    _omp_persistent_init = true;
  }

  // F12 fix — thread_local heap-alloc. 각 caller thread (cdec executor
  // worker) 별 별 buffer 1 회 init 후 재사용. race 0.
  static thread_local std::vector<itmd_t> _tl_attn_score_buf_vec(
    MAX_TOK_NUM * NUM_Q_HEADS);
  static thread_local std::vector<otpt_t> _tl_o_buf_vec(
    MAX_TASK_NUM * NUM_Q_HEADS * HEAD_DIM);
  static thread_local std::vector<itmd_t> _tl_attn_sum_buf_vec(
    MAX_TASK_NUM * NUM_Q_HEADS);
  itmd_t* attn_score_buf = _tl_attn_score_buf_vec.data();
  otpt_t* o_buf = _tl_o_buf_vec.data();
  itmd_t* attn_sum_buf = _tl_attn_sum_buf_vec.data();
  int ws = omp_get_max_threads();
  int bch_blk_size = (batch_size - 1) / ws + 1;
  int tot_blks = 0;
  for (auto i = 0; i < batch_size; i++) {
    tot_blks += (seq_lengths[i] - 1) / BLOCK_SIZE + 1;
  }

  int thrd_rst_blks[MAX_WS];
  for (auto i = 0; i < ws; i++) {
    thrd_rst_blks[i] = tot_blks / ws + (i < tot_blks % ws);
  }

  // Distribute tasks to threads, each thread processes no more than thrd_max_blks blocks
  std::vector<std::tuple<int, int, int, int> > tasks; // specs of each task (batch_id, seq_offs, seg_len, cum_seg_len)
  int* thrd_start_task = new int[ws + 1];             // starting task id for each thread
  int* seq_start_task = new int[batch_size + 1];      // starting task id for each sequence
  int cur_thrd = 0;
  int cum_seg_len = 0;
  thrd_start_task[0] = 0;
  for (int i = 0; i < batch_size; i++) {
    int seq_offs = 0;
    int seq_len = seq_lengths[i];
    seq_start_task[i] = tasks.size();
    while(seq_offs < seq_len) {
      if (thrd_rst_blks[cur_thrd] == 0) {
        thrd_start_task[++cur_thrd] = tasks.size();
      }
      int seg_len = std::min(seq_len - seq_offs, thrd_rst_blks[cur_thrd] * BLOCK_SIZE);
      tasks.emplace_back(i, seq_offs, seg_len, cum_seg_len);
      seq_offs += seg_len;
      cum_seg_len += seg_len;
      thrd_rst_blks[cur_thrd] -= (seg_len - 1) / BLOCK_SIZE + 1;
    }
  }
  seq_start_task[batch_size] = tasks.size();
  for (;cur_thrd < ws; cur_thrd++) {
    thrd_start_task[cur_thrd + 1] = tasks.size();
  }

  // for (int i = 0; i <= batch_size; i++) {
  //   printf("seq_start_task[%d] = %d\n", i, seq_start_task[i]);
  // }
  // for (int i = 0; i <= ws; i++) {
  //   printf("thrd_start_task[%d] = %d\n", i, thrd_start_task[i]);
  // }

  // [PROFILE OMP] turn 7 → 8 instrumentation —
  // OMP barrier #1/#2 의 thread-level wait time 측정 (py-spy 의 한계 해소).
  // env VLLM_NEO_OMP_PROFILE=1 활성 시만 enable. 비활성 시 omp_get_wtime()
  // 호출 자체 skip (overhead 0).
  static int _omp_prof_decided = 0;  // 0=undecided, 1=on, -1=off
  if (_omp_prof_decided == 0) {
    const char* _omp_pe = std::getenv("VLLM_NEO_OMP_PROFILE");
    _omp_prof_decided = (_omp_pe && _omp_pe[0] && _omp_pe[0] != '0') ? 1 : -1;
  }
  const bool _omp_prof_on = (_omp_prof_decided == 1);
  // per-tid barrier wait time 누적 영역 (call 별 reset)
  static thread_local double _omp_b1_wait_sum_ms = 0.0;
  static thread_local double _omp_b2_wait_sum_ms = 0.0;
  static thread_local double _omp_step0_sum_ms = 0.0;
  static thread_local double _omp_step1_sum_ms = 0.0;
  static thread_local double _omp_step2_sum_ms = 0.0;
  static thread_local uint64_t _omp_call_count = 0;
  static thread_local double _omp_b1_max_ms = 0.0;
  static thread_local double _omp_b2_max_ms = 0.0;

  // [A3] KV prefetch L2 — turn 13 (Stage 1 A3 영역).
  // env VLLM_NEO_KV_PREFETCH=1 활성 시 store_kv 직전 + attn 영역 진입 시
  // 다음 block 의 K cache 영역 영역 L2 prefetch (_mm_prefetch).
  // 비활성 시 prefetch hint 영역 호출 영역 skip (overhead 0).
  static int _kv_prefetch_decided = 0;  // 0=undecided, 1=on, -1=off
  if (_kv_prefetch_decided == 0) {
    const char* _kv_pe = std::getenv("VLLM_NEO_KV_PREFETCH");
    _kv_prefetch_decided = (_kv_pe && _kv_pe[0] && _kv_pe[0] != '0') ? 1 : -1;
  }
  const bool _kv_prefetch_on = (_kv_prefetch_decided == 1);

  // [C1a / SUB_035] OMP team launch overhead 측정 — env VLLM_NEO_OMP_LAUNCH_PROFILE=1
  //   parallel region 진입 직전 / 종료 직후 wall 차이 = team launch + join cost.
  //   기존 step0+b1+step1+b2+step2 합과 비교해서 진정한 launch overhead 분리.
  static int _omp_launch_prof_decided = 0;
  if (_omp_launch_prof_decided == 0) {
    const char* _olp_pe = std::getenv("VLLM_NEO_OMP_LAUNCH_PROFILE");
    _omp_launch_prof_decided = (_olp_pe && _olp_pe[0] && _olp_pe[0] != '0') ? 1 : -1;
  }
  const bool _omp_launch_prof_on = (_omp_launch_prof_decided == 1);
  static thread_local double _omp_launch_total_ms = 0.0;
  static thread_local double _omp_launch_max_ms = 0.0;
  static thread_local uint64_t _omp_launch_count = 0;
  double _olp_t0 = _omp_launch_prof_on ? omp_get_wtime() : 0.0;

  # pragma omp parallel
  {
    // Step 0:
    //   store the kv_cache
    int tid = omp_get_thread_num();
    int l = tid * bch_blk_size, r = std::min((tid + 1) * bch_blk_size, batch_size);
    // NOTE: l >= r when batch_size < omp_get_max_threads()
    double _t_start = _omp_prof_on ? omp_get_wtime() : 0.0;
    for (auto i = l; i < r; i++) {
      int seq_id = seq_ids[i];
      int seq_len = seq_lengths[i];
      auto kip = kbatch_p + i * NUM_KV_HEADS * HEAD_DIM;
      auto vip = vbatch_p + i * NUM_KV_HEADS * HEAD_DIM;
      auto btp = block_table_p + seq_id * block_table_width;

      // [A3] L2 prefetch — next sequence's K head + block_table 영역 미리 영역
      // L2 영역 hint. (i+1) 영역 hint, 마지막 seq 영역 skip.
      if (_kv_prefetch_on && i + 1 < r) {
        __builtin_prefetch(
          kbatch_p + (i + 1) * NUM_KV_HEADS * HEAD_DIM, 0, 2);  // K head, L2 hint
        __builtin_prefetch(
          vbatch_p + (i + 1) * NUM_KV_HEADS * HEAD_DIM, 0, 2);  // V head, L2 hint
        __builtin_prefetch(
          block_table_p + seq_ids[i + 1] * block_table_width, 0, 2);
      }

      brute::store_kv(
        cur_layer, num_blocks, seq_len, kip, vip, kcache_p, vcache_p, btp
      );
    }
    double _t_after_step0 = _omp_prof_on ? omp_get_wtime() : 0.0;
    # pragma omp barrier
    double _t_after_b1 = _omp_prof_on ? omp_get_wtime() : 0.0;
    if (_omp_prof_on) {
      double _step0_ms = (_t_after_step0 - _t_start) * 1000.0;
      double _b1_wait_ms = (_t_after_b1 - _t_after_step0) * 1000.0;
      _omp_step0_sum_ms += _step0_ms;
      _omp_b1_wait_sum_ms += _b1_wait_ms;
      if (_b1_wait_ms > _omp_b1_max_ms) _omp_b1_max_ms = _b1_wait_ms;
    }

    // Step 1:
    //   compute intermediate output for each sequence block
    //   output is stored in o_buf and attn_sum_buf
    // SUB_015-Phase 3 A: env-toggle AMX path (qk = AMX BF16, softmax/av = ISPC).
    //   _amx_init flag thread_local — AMX permission + tile config 1 회.
    static thread_local int _amx_decided = 0;  // 0=undecided, 1=use_amx, -1=use_ispc
    if (_amx_decided == 0) {
      const char* env = std::getenv("VLLM_NEO_USE_AMX");
      bool want_amx = (env && env[0] && env[0] != '0');
      if (want_amx && amx_kernel::ensure_amx_init()) {
        _amx_decided = 1;
      } else {
        _amx_decided = -1;
      }
    }
    // P2 F6 (dynamic schedule) 단독 측정 = -1.4% 회귀 (564.9 vs 573.0 baseline).
    //   atomic counter overhead 가 imbalance ↓ win 보다 큼. static partition 복원.
    for (auto t = thrd_start_task[tid]; t < thrd_start_task[tid + 1]; t++) {
      int i, seq_offs, seg_len, cum_seg_len;
      std::tie(i, seq_offs, seg_len, cum_seg_len) = tasks[t];
      auto qip = qbatch_p + i * NUM_Q_HEADS * HEAD_DIM;
      auto oip = obatch_p + i * NUM_Q_HEADS * HEAD_DIM;
      auto btp = block_table_p + seq_ids[i] * block_table_width;
      itmd_t* a_ptr = attn_score_buf + cum_seg_len * NUM_Q_HEADS;
      otpt_t* o_ptr = (seg_len == seq_lengths[i] ? oip : o_buf + t * NUM_Q_HEADS * HEAD_DIM);
      itmd_t* asb_ptr = attn_sum_buf + t * NUM_Q_HEADS;
      if (_amx_decided == 1) {
        // P3 (F3) — K BF16 시 BF16-native variant 호출 (Step 1 vec K conv
        //   skip). Q FP16 그대로, V FP16 그대로 (ISPC av_product).
        if (k_is_bf16) {
          amx_kernel::attn_one_seq_amx_bf16(
            cur_layer, num_blocks, seg_len, (float)softmax_scale,
            qip, reinterpret_cast<const uint16_t*>(kcache_p),
            vcache_p, btp + seq_offs / BLOCK_SIZE,
            a_ptr, o_ptr, asb_ptr);
        } else {
          amx_kernel::attn_one_seq_amx(
            cur_layer, num_blocks, seg_len, (float)softmax_scale,
            qip, kcache_p, vcache_p, btp + seq_offs / BLOCK_SIZE,
            a_ptr, o_ptr, asb_ptr);
        }
      } else {
        // ISPC path = K FP16 hard-coded. K BF16 시 진입 안 함 (caller 가
        //   USE_AMX=1 + HOST_K_BF16=1 동시 활성 만 허용). 만약 dispatch
        //   fail (AMX init fail) + K BF16 시 결과 garbage 위험 — fallback
        //   throw 또는 forward 진입 막기 위한 caller-side check 필요.
        ispc::attn_one_seq(
          cur_layer, num_blocks, seg_len, softmax_scale,
          qip, kcache_p, vcache_p, btp + seq_offs / BLOCK_SIZE,
          a_ptr, o_ptr, asb_ptr);
      }
    }
    double _t_after_step1 = _omp_prof_on ? omp_get_wtime() : 0.0;
    # pragma omp barrier
    double _t_after_b2 = _omp_prof_on ? omp_get_wtime() : 0.0;
    if (_omp_prof_on) {
      double _step1_ms = (_t_after_step1 - _t_after_b1) * 1000.0;
      double _b2_wait_ms = (_t_after_b2 - _t_after_step1) * 1000.0;
      _omp_step1_sum_ms += _step1_ms;
      _omp_b2_wait_sum_ms += _b2_wait_ms;
      if (_b2_wait_ms > _omp_b2_max_ms) _omp_b2_max_ms = _b2_wait_ms;
    }

    // Step 2:
    // Gather intermediate output to final output
    for (auto i = l; i < r; i++) {
      int num_tasks = seq_start_task[i + 1] - seq_start_task[i];
      if (num_tasks > 1) {
        int o_off = seq_start_task[i] * NUM_Q_HEADS * HEAD_DIM;
        int as_off = seq_start_task[i] * NUM_Q_HEADS;
        auto oip = obatch_p + i * NUM_Q_HEADS * HEAD_DIM;
        ispc::gather_output_one_seq(
          num_tasks,
          o_buf + o_off,
          attn_sum_buf + as_off,
          oip
        );
      }
    }
    double _t_after_step2 = _omp_prof_on ? omp_get_wtime() : 0.0;
    if (_omp_prof_on) {
      double _step2_ms = (_t_after_step2 - _t_after_b2) * 1000.0;
      _omp_step2_sum_ms += _step2_ms;
    }
  }
  // [PROFILE OMP] tid=0 영역 (master) 만 summary emit (log_freq=500 calls)
  if (_omp_prof_on) {
    _omp_call_count++;
    if (_omp_call_count % 500 == 0) {
      fprintf(stderr,
        "[PROFILE OMP tid=0] calls=%lu step0_avg=%.3fms b1_wait_avg=%.3fms "
        "b1_wait_max=%.3fms step1_avg=%.3fms b2_wait_avg=%.3fms "
        "b2_wait_max=%.3fms step2_avg=%.3fms\n",
        (unsigned long)_omp_call_count,
        _omp_step0_sum_ms / _omp_call_count,
        _omp_b1_wait_sum_ms / _omp_call_count,
        _omp_b1_max_ms,
        _omp_step1_sum_ms / _omp_call_count,
        _omp_b2_wait_sum_ms / _omp_call_count,
        _omp_b2_max_ms,
        _omp_step2_sum_ms / _omp_call_count);
      fflush(stderr);
    }
  }

  // [C1a / SUB_035] OMP team launch overhead summary emit (log_freq=500 calls)
  if (_omp_launch_prof_on) {
    double _olp_ms = (omp_get_wtime() - _olp_t0) * 1000.0;
    _omp_launch_total_ms += _olp_ms;
    if (_olp_ms > _omp_launch_max_ms) _omp_launch_max_ms = _olp_ms;
    _omp_launch_count++;
    if (_omp_launch_count % 500 == 0) {
      double _step_sum = _omp_step0_sum_ms + _omp_b1_wait_sum_ms
                       + _omp_step1_sum_ms + _omp_b2_wait_sum_ms
                       + _omp_step2_sum_ms;
      double _overhead = _omp_launch_total_ms - _step_sum;
      double _overhead_pct = (_omp_launch_total_ms > 0)
                           ? (_overhead / _omp_launch_total_ms) * 100.0 : 0.0;
      fprintf(stderr,
        "[PROFILE OMP LAUNCH] calls=%lu parallel_avg_ms=%.3f parallel_max_ms=%.3f "
        "step_sum_ms=%.3f launch_overhead_ms=%.3f overhead_pct=%.2f%%\n",
        (unsigned long)_omp_launch_count,
        _omp_launch_total_ms / _omp_launch_count,
        _omp_launch_max_ms,
        _step_sum / _omp_launch_count,
        _overhead / _omp_launch_count,
        _overhead_pct);
      fflush(stderr);
    }
  }
}
