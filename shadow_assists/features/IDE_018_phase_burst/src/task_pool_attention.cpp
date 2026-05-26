// IDE_018 / TSK_032 — Attention-phase CPU task pool
//
// Tasks fired during GPU's memory-bound attention phase (CPU SM idle window).
//
// paper Table 1b (SUB_168) attention column:
//   Task A schedule next batch (1-3 ms)
//   Task B detokenize previous step (1-5 ms)
//   Task C grammar / constraint check (2-10 ms / token)
//   Task D request classifier (1 ms / request)
//
// Each "produce_task_X" returns a Task ready for PhaseBurstScheduler::enqueue.
// The compute kernels themselves live in:
//   - TSK_024 AVX-512 scheduler vectorize (task A)
//   - TSK_024 AVX-512 tokenizer (task B)
//   - existing XGrammar (task C, via FFI)
//   - IDE_012 / SUB_076 classifier (task D)
// 본 file 은 dispatch glue 와 stub fn (실측 wiring 은 IDE_016 / IDE_012 turn).

#include "phase_detector.h"
#include "scheduler.h"
#include "task_pool.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

namespace vllm_hybrid_phase {
namespace attention_pool {

// ──────────────────────────────────────────────────────────────────────
// Task A — Schedule next batch (AVX-512 scheduler)
// ──────────────────────────────────────────────────────────────────────
//
// vLLM Scheduler.schedule_next_batch() 의 CPU work 를 attention window 내
// 미리 수행. ENV/extension 의 prebatch_handle 을 통해 main loop 와 동기화.
//
// target: per-step 1-3 ms (depends on batch size).

Task produce_task_A_schedule(uint64_t step_id,
                             SchedulerHandle* sched_handle) {
    return Task{
        TASK_A_SCHEDULE,
        step_id,
        /*applicable_phases=*/ MASK_ATTN | MASK_LINEAR | MASK_TP_AR,
        /*fn=*/ [sched_handle, step_id]() {
            if (!sched_handle || !sched_handle->prepare_next_batch) return;
            sched_handle->prepare_next_batch(step_id);
        },
        /*enqueued_ns=*/ 0,
    };
}

// ──────────────────────────────────────────────────────────────────────
// Task B — Detokenize previous step output (AVX-512 BPE)
// ──────────────────────────────────────────────────────────────────────
//
// previous-step output token ids 가 ready 되는 즉시 detokenize.
// vLLM 의 default path 는 sample 종료 후 sync detok — 본 task 는 sample
// completion 과 동시에 attention window 에서 CPU 측 detokenize 를 시작.
//
// SUB_161 measurement: sampler.py 44%, logits 27% — sampler 종료 후 detok
// 즉시 시작 시 1-step 의 critical path 에서 1-5 ms 절약.

Task produce_task_B_detokenize(uint64_t step_id,
                               DetokenizerHandle* detok,
                               const int32_t* token_ids,
                               size_t n_tokens) {
    // capture small by-value — token_ids 는 caller 가 별도 lifetime 관리.
    return Task{
        TASK_B_DETOKENIZE,
        step_id,
        /*applicable_phases=*/ MASK_ATTN | MASK_LINEAR | MASK_IDLE | MASK_POST,
        /*fn=*/ [detok, token_ids, n_tokens, step_id]() {
            if (!detok || !detok->detokenize_batch) return;
            detok->detokenize_batch(token_ids, n_tokens, step_id);
        },
        /*enqueued_ns=*/ 0,
    };
}

// ──────────────────────────────────────────────────────────────────────
// Task C — Grammar / constraint check (XGrammar offload)
// ──────────────────────────────────────────────────────────────────────
//
// XGrammar 의 next-token mask 계산 — 다음 sample phase 의 logits mask 로
// 미리 준비. constrained workload (function calling, JSON mode) 에서 효과
// 큼. 일반 chat workload 에서는 skip (caller 가 enqueue 안 함).
//
// target: per-token 2-10 ms.

Task produce_task_C_grammar(uint64_t step_id,
                            GrammarHandle* grammar,
                            int32_t prev_token) {
    return Task{
        TASK_C_GRAMMAR,
        step_id,
        /*applicable_phases=*/ MASK_ATTN | MASK_LINEAR,
        /*fn=*/ [grammar, prev_token, step_id]() {
            if (!grammar || !grammar->advance_and_mask) return;
            grammar->advance_and_mask(prev_token, step_id);
        },
        /*enqueued_ns=*/ 0,
    };
}

// ──────────────────────────────────────────────────────────────────────
// Task D — Request classifier (IDE_012 / SUB_076)
// ──────────────────────────────────────────────────────────────────────
//
// 신규 request 의 workload class 를 inline 으로 예측 → routing 에 활용.
// SUB_076 PoC: macro accuracy 1.000 on local 3 workload × 500 prompt.
// CPU only, lightweight (1 ms / request).

Task produce_task_D_classify(uint64_t step_id,
                             ClassifierHandle* clf,
                             const char* prompt_utf8,
                             size_t prompt_len,
                             int request_id) {
    return Task{
        TASK_D_CLASSIFY,
        step_id,
        /*applicable_phases=*/ MASK_ATTN | MASK_LINEAR | MASK_IDLE,
        /*fn=*/ [clf, prompt_utf8, prompt_len, request_id]() {
            if (!clf || !clf->classify) return;
            clf->classify(prompt_utf8, prompt_len, request_id);
        },
        /*enqueued_ns=*/ 0,
    };
}

// ──────────────────────────────────────────────────────────────────────
// Bulk producer — vLLM main loop 가 1 step 마다 호출
// ──────────────────────────────────────────────────────────────────────

size_t enqueue_attention_phase_tasks(PhaseBurstScheduler& sched,
                                     const AttentionPoolHandles& h,
                                     uint64_t step_id,
                                     const AttentionStepInput& in) {
    size_t enqueued = 0;
    if (h.scheduler) {
        sched.enqueue(produce_task_A_schedule(step_id, h.scheduler));
        enqueued++;
    }
    if (h.detok && in.prev_tokens && in.prev_token_count > 0) {
        sched.enqueue(produce_task_B_detokenize(step_id, h.detok,
                                                in.prev_tokens, in.prev_token_count));
        enqueued++;
    }
    if (h.grammar && in.constrained_active && in.last_emitted_token >= 0) {
        sched.enqueue(produce_task_C_grammar(step_id, h.grammar,
                                             in.last_emitted_token));
        enqueued++;
    }
    if (h.classifier && in.new_request_count > 0 && in.new_request_prompts) {
        for (int i = 0; i < in.new_request_count; ++i) {
            sched.enqueue(produce_task_D_classify(step_id, h.classifier,
                in.new_request_prompts[i].utf8,
                in.new_request_prompts[i].len,
                in.new_request_prompts[i].id));
            enqueued++;
        }
    }
    return enqueued;
}

// ──────────────────────────────────────────────────────────────────────
// Stub fns — placeholder until IDE_016 / IDE_012 wiring lands
// ──────────────────────────────────────────────────────────────────────
//
// 실제 production 에서는 본 stub 들이 AVX-512 kernel / XGrammar / classifier
// 의 entry point 로 교체된다. 본 stub 들은 microbench / scheduler test 의
// "task가 실행되었나" 신호 (atomic counter increment) 만 제공.

std::atomic<uint64_t> g_stub_invocation_count{0};

void stub_prepare_next_batch(uint64_t /*step*/) {
    // simulate 1.5 ms work
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::microseconds(1500)) {
        asm volatile("pause" ::: "memory");
    }
    g_stub_invocation_count.fetch_add(1, std::memory_order_relaxed);
}

void stub_detokenize_batch(const int32_t* /*ids*/, size_t /*n*/, uint64_t /*step*/) {
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::microseconds(2500)) {
        asm volatile("pause" ::: "memory");
    }
    g_stub_invocation_count.fetch_add(1, std::memory_order_relaxed);
}

void stub_grammar_advance(int32_t /*tok*/, uint64_t /*step*/) {
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::microseconds(5000)) {
        asm volatile("pause" ::: "memory");
    }
    g_stub_invocation_count.fetch_add(1, std::memory_order_relaxed);
}

void stub_classify(const char* /*p*/, size_t /*l*/, int /*rid*/) {
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::microseconds(1000)) {
        asm volatile("pause" ::: "memory");
    }
    g_stub_invocation_count.fetch_add(1, std::memory_order_relaxed);
}

AttentionPoolHandles make_stub_handles() {
    static SchedulerHandle  sh{ &stub_prepare_next_batch };
    static DetokenizerHandle dh{ &stub_detokenize_batch };
    static GrammarHandle    gh{ &stub_grammar_advance };
    static ClassifierHandle ch{ &stub_classify };
    return AttentionPoolHandles{ &sh, &dh, &gh, &ch };
}

uint64_t stub_invocation_count() {
    return g_stub_invocation_count.load(std::memory_order_relaxed);
}

}  // namespace attention_pool
}  // namespace vllm_hybrid_phase
