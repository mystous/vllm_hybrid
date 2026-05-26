// IDE_018 / TSK_033 — Linear-phase CPU task pool
//
// Tasks fired during GPU's compute-bound linear (matmul) phase
// (CPU SM-equivalent idle, but GPU memory bus available).
//
// paper Table 1b (SUB_168) linear column:
//   Task E KV cache DMA prefetch (per-chunk 60 μs — SUB_166 1 MB crossover)
//   Task F AMX speculative draft head (≤ 5 ms / batch)
//   Task G Cold-KV decompress (5-20 ms / chunk)
//
// Compute kernel ownership:
//   - TSK_028 pinned host buffer pool (task E data plane)
//   - TSK_026 AMX draft head (task F compute) — IDE_016
//   - TSK_030 cold-KV decompress (task G) — IDE_017

#include "phase_detector.h"
#include "scheduler.h"
#include "task_pool.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace vllm_hybrid_phase {
namespace linear_pool {

// ──────────────────────────────────────────────────────────────────────
// Task E — KV cache DMA prefetch
// ──────────────────────────────────────────────────────────────────────
//
// linear phase 중 다음 attention 에 필요한 KV chunk 를 미리 device 또는
// pinned host 로 prefetch. SUB_166 의 1 MB crossover / 54 GB/s 로 한 chunk
// ≈ 60 μs (1 MB).

Task produce_task_E_kv_prefetch(uint64_t step_id,
                                KVPrefetchHandle* h,
                                int layer, int chunk_id) {
    return Task{
        TASK_E_KV_PREFETCH,
        step_id,
        /*applicable_phases=*/ MASK_LINEAR | MASK_TP_AR | MASK_IDLE,
        /*fn=*/ [h, layer, chunk_id, step_id]() {
            if (!h || !h->prefetch_chunk) return;
            h->prefetch_chunk(layer, chunk_id, step_id);
        },
        /*enqueued_ns=*/ 0,
    };
}

// ──────────────────────────────────────────────────────────────────────
// Task F — AMX speculative draft head
// ──────────────────────────────────────────────────────────────────────
//
// linear phase 의 GPU matmul 과 parallel 로 AMX-based draft head 가
// k tokens 의 spec candidate 생성. paper IDE_019 (multi-source drafter) 의
// AMX source.
//
// target: per-batch ≤ 5 ms; gate: AMX 가용 (prod Xeon SPR+).

Task produce_task_F_draft(uint64_t step_id,
                          DraftHeadHandle* h,
                          int batch_size) {
    return Task{
        TASK_F_DRAFT,
        step_id,
        /*applicable_phases=*/ MASK_LINEAR | MASK_TP_AR,
        /*fn=*/ [h, batch_size, step_id]() {
            if (!h || !h->draft_step) return;
            h->draft_step(step_id, batch_size);
        },
        /*enqueued_ns=*/ 0,
    };
}

// ──────────────────────────────────────────────────────────────────────
// Task G — Cold-KV decompress
// ──────────────────────────────────────────────────────────────────────
//
// IDE_017 / TSK_030 의 cold-KV decompress (5-20 ms / chunk). attention 이
// 아닌 phase 에서만 안전 (KV mutate 충돌 회피).

Task produce_task_G_coldkv(uint64_t step_id,
                           ColdKVHandle* h,
                           int layer, int chunk_id) {
    return Task{
        TASK_G_COLDKV,
        step_id,
        /*applicable_phases=*/ MASK_LINEAR | MASK_TP_AR | MASK_IDLE,
        /*fn=*/ [h, layer, chunk_id, step_id]() {
            if (!h || !h->decompress_chunk) return;
            h->decompress_chunk(layer, chunk_id, step_id);
        },
        /*enqueued_ns=*/ 0,
    };
}

// ──────────────────────────────────────────────────────────────────────
// Bulk producer
// ──────────────────────────────────────────────────────────────────────

size_t enqueue_linear_phase_tasks(PhaseBurstScheduler& sched,
                                  const LinearPoolHandles& h,
                                  uint64_t step_id,
                                  const LinearStepInput& in) {
    size_t enqueued = 0;

    if (h.kv_prefetch && in.prefetch_count > 0 && in.prefetch_layers && in.prefetch_chunks) {
        for (int i = 0; i < in.prefetch_count; ++i) {
            sched.enqueue(produce_task_E_kv_prefetch(step_id, h.kv_prefetch,
                                                     in.prefetch_layers[i],
                                                     in.prefetch_chunks[i]));
            enqueued++;
        }
    }

    if (h.draft && in.draft_enabled && in.draft_batch_size > 0) {
        sched.enqueue(produce_task_F_draft(step_id, h.draft, in.draft_batch_size));
        enqueued++;
    }

    if (h.coldkv && in.coldkv_count > 0 && in.coldkv_layers && in.coldkv_chunks) {
        for (int i = 0; i < in.coldkv_count; ++i) {
            sched.enqueue(produce_task_G_coldkv(step_id, h.coldkv,
                                                in.coldkv_layers[i],
                                                in.coldkv_chunks[i]));
            enqueued++;
        }
    }

    return enqueued;
}

// ──────────────────────────────────────────────────────────────────────
// Stub fns — placeholder until IDE_016 / IDE_017 / IDE_019 wiring lands
// ──────────────────────────────────────────────────────────────────────

std::atomic<uint64_t> g_stub_invocation_count{0};

void stub_kv_prefetch(int /*layer*/, int /*chunk*/, uint64_t /*step*/) {
    // SUB_166 60 μs / chunk model
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::microseconds(60)) {
        asm volatile("pause" ::: "memory");
    }
    g_stub_invocation_count.fetch_add(1, std::memory_order_relaxed);
}

void stub_draft_step(uint64_t /*step*/, int /*bs*/) {
    // ~3 ms work
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::microseconds(3000)) {
        asm volatile("pause" ::: "memory");
    }
    g_stub_invocation_count.fetch_add(1, std::memory_order_relaxed);
}

void stub_coldkv_decompress(int /*layer*/, int /*chunk*/, uint64_t /*step*/) {
    // ~10 ms work
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::microseconds(10000)) {
        asm volatile("pause" ::: "memory");
    }
    g_stub_invocation_count.fetch_add(1, std::memory_order_relaxed);
}

LinearPoolHandles make_stub_handles() {
    static KVPrefetchHandle kh{ &stub_kv_prefetch };
    static DraftHeadHandle  dh{ &stub_draft_step };
    static ColdKVHandle     ch{ &stub_coldkv_decompress };
    return LinearPoolHandles{ &kh, &dh, &ch };
}

uint64_t stub_invocation_count() {
    return g_stub_invocation_count.load(std::memory_order_relaxed);
}

}  // namespace linear_pool
}  // namespace vllm_hybrid_phase
