// IDE_018 / TSK_032 + TSK_033 — Task pool handles (shared ABI)

#pragma once

#include "scheduler.h"

#include <cstddef>
#include <cstdint>

namespace vllm_hybrid_phase {

// ── Attention-phase handles ────────────────────────────────────────

struct SchedulerHandle {
    void (*prepare_next_batch)(uint64_t step_id);
};

struct DetokenizerHandle {
    void (*detokenize_batch)(const int32_t* token_ids, size_t n_tokens, uint64_t step_id);
};

struct GrammarHandle {
    void (*advance_and_mask)(int32_t prev_token, uint64_t step_id);
};

struct ClassifierHandle {
    void (*classify)(const char* prompt_utf8, size_t prompt_len, int request_id);
};

struct AttentionPoolHandles {
    SchedulerHandle*  scheduler;
    DetokenizerHandle* detok;
    GrammarHandle*    grammar;
    ClassifierHandle* classifier;
};

struct PromptRef {
    const char* utf8;
    size_t       len;
    int          id;
};

struct AttentionStepInput {
    const int32_t* prev_tokens = nullptr;       // task B input
    size_t          prev_token_count = 0;
    int32_t         last_emitted_token = -1;    // task C input
    bool            constrained_active = false; // task C gate
    const PromptRef* new_request_prompts = nullptr;  // task D input
    int             new_request_count = 0;
};

namespace attention_pool {

Task produce_task_A_schedule(uint64_t step_id, SchedulerHandle* sched_handle);
Task produce_task_B_detokenize(uint64_t step_id, DetokenizerHandle* detok,
                               const int32_t* token_ids, size_t n_tokens);
Task produce_task_C_grammar(uint64_t step_id, GrammarHandle* grammar, int32_t prev_token);
Task produce_task_D_classify(uint64_t step_id, ClassifierHandle* clf,
                             const char* prompt_utf8, size_t prompt_len, int request_id);

size_t enqueue_attention_phase_tasks(PhaseBurstScheduler& sched,
                                     const AttentionPoolHandles& h,
                                     uint64_t step_id,
                                     const AttentionStepInput& in);

AttentionPoolHandles make_stub_handles();
uint64_t stub_invocation_count();

}  // namespace attention_pool

// ── Linear-phase handles ───────────────────────────────────────────

struct KVPrefetchHandle {
    // GPU → CPU pinned 으로 cold-KV chunk pull (TSK_028 pinned pool).
    void (*prefetch_chunk)(int layer, int chunk_id, uint64_t step_id);
};

struct DraftHeadHandle {
    // AMX-based draft head; output: draft token ids appended to a draft buffer.
    void (*draft_step)(uint64_t step_id, int batch_size);
};

struct ColdKVHandle {
    // TSK_030 cold-KV decompress (per-chunk 5-20 ms).
    void (*decompress_chunk)(int layer, int chunk_id, uint64_t step_id);
};

struct LinearPoolHandles {
    KVPrefetchHandle* kv_prefetch;
    DraftHeadHandle*  draft;
    ColdKVHandle*     coldkv;
};

struct LinearStepInput {
    // KV prefetch — list of (layer, chunk_id) about to be needed next attention.
    const int* prefetch_layers = nullptr;
    const int* prefetch_chunks = nullptr;
    int         prefetch_count = 0;

    // Draft — fire once per linear phase, decoded from current GPU state.
    bool        draft_enabled = false;
    int         draft_batch_size = 0;

    // Cold-KV — list of (layer, chunk_id) needing decompress.
    const int* coldkv_layers = nullptr;
    const int* coldkv_chunks = nullptr;
    int         coldkv_count = 0;
};

namespace linear_pool {

Task produce_task_E_kv_prefetch(uint64_t step_id, KVPrefetchHandle* h,
                                int layer, int chunk_id);
Task produce_task_F_draft(uint64_t step_id, DraftHeadHandle* h, int batch_size);
Task produce_task_G_coldkv(uint64_t step_id, ColdKVHandle* h,
                           int layer, int chunk_id);

size_t enqueue_linear_phase_tasks(PhaseBurstScheduler& sched,
                                  const LinearPoolHandles& h,
                                  uint64_t step_id,
                                  const LinearStepInput& in);

LinearPoolHandles make_stub_handles();
uint64_t stub_invocation_count();

}  // namespace linear_pool

// ── SUB_184 dummy-fill ─────────────────────────────────────────────
namespace dummy_fill {

size_t enqueue_dummy_attention_burst(PhaseBurstScheduler& sched,
                                     uint64_t step_id, int count, int iters);
size_t enqueue_dummy_linear_burst(PhaseBurstScheduler& sched,
                                  uint64_t step_id, int count, int iters);
uint64_t invocation_count();
uint64_t total_iters();

}  // namespace dummy_fill

}  // namespace vllm_hybrid_phase
