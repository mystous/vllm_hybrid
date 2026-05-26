// IDE_018 / TSK_031 — Phase detection mechanism (full impl)
//
// CUDA event hook + IPC primitive for phase-burst scheduler.
// paper main contribution 의 phase signal source. vLLM forward path 에 patch 로
// 삽입되어 attention/linear/sample/tp_allreduce/post_step 의 phase boundary
// 마다 atomic counter 갱신 + eventfd notify.
//
// Phase A input:
//   - SUB_148: VLLM thread default full-mask — OS-coordination 가능
//   - SUB_162: VLLM Python threads 96-100% S — CPU 측 dispatch 안전
//   - SUB_166: DMA 35 μs overhead — phase signal latency 목표 < 50 μs 와 정합
//
// IPC 설계 선택:
//   1) shared atomic counter (lockless, single-cacheline) — primary signal
//   2) eventfd notify — blocking wait fast path (read/write 1 byte → 2-5 μs)
//   3) cudaStreamAddCallback — driver-managed thread (overhead 25-40 μs)
//   본 file 은 (1)+(2) 를 main path, (3) 은 fallback / 비교 reference 로 둠.
//
// Build: -fPIC -O3 -std=c++17. cuda_runtime 의 weak link (CUDA 없는 환경에서도
// scheduler unit test 동작하도록 ENABLE_CUDA macro guard).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

// CPU relax — x86_64 PAUSE, fallback to nop on other archs.
#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define CPU_RELAX() _mm_pause()
#else
#  define CPU_RELAX() do { asm volatile("" ::: "memory"); } while (0)
#endif

#ifdef ENABLE_CUDA
#  include <cuda_runtime.h>
#endif

#include "phase_detector.h"

namespace vllm_hybrid_phase {

// ──────────────────────────────────────────────────────────────────────
// Time helper
// ──────────────────────────────────────────────────────────────────────

uint64_t PhaseSignal::current_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1'000'000'000ULL + uint64_t(ts.tv_nsec);
}

// ──────────────────────────────────────────────────────────────────────
// PhaseSignal — shared atomic counter + eventfd
// ──────────────────────────────────────────────────────────────────────

PhaseSignal* PhaseSignal::create_anonymous() {
    // anonymous mmap — for single-process (vLLM driver + CPU pool 같은 process)
    void* p = ::mmap(nullptr, sizeof(PhaseSignal),
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        throw std::runtime_error("PhaseSignal mmap failed");
    }
    auto* sig = new (p) PhaseSignal();
    sig->step_id.store(0);
    sig->phase.store(PHASE_IDLE);
    sig->phase_start_ns.store(current_time_ns());
    sig->total_updates.store(0);
    sig->signal_drops.store(0);
    sig->eventfd_ = eventfd(0, EFD_NONBLOCK | EFD_SEMAPHORE);
    if (sig->eventfd_ < 0) {
        throw std::runtime_error("eventfd creation failed");
    }
    return sig;
}

void PhaseSignal::destroy(PhaseSignal* sig) {
    if (!sig) return;
    if (sig->eventfd_ >= 0) ::close(sig->eventfd_);
    sig->~PhaseSignal();
    ::munmap(sig, sizeof(PhaseSignal));
}

void PhaseSignal::update(uint8_t new_phase, uint64_t new_step_id) {
    uint64_t now = current_time_ns();
    phase_start_ns.store(now, std::memory_order_relaxed);
    if (new_step_id != 0) {
        step_id.store(new_step_id, std::memory_order_relaxed);
    }
    // release: ensures downstream observer sees consistent timestamp
    phase.store(new_phase, std::memory_order_release);
    total_updates.fetch_add(1, std::memory_order_relaxed);

    // eventfd notify — non-blocking; if counter saturated drop signal
    // (consumer was slow but atomic counter is the truth, eventfd is wake-up only)
    uint64_t one = 1;
    ssize_t n = ::write(eventfd_, &one, sizeof(one));
    if (n != sizeof(one)) {
        signal_drops.fetch_add(1, std::memory_order_relaxed);
    }
}

uint8_t PhaseSignal::wait_next(int timeout_us) {
    // Hybrid wait: short spin then eventfd block.
    // spin minimizes latency for back-to-back phase signals (linear → sample
    // 은 GPU 측에서 μs 단위로 빠르게 옴, 50 μs 안쪽).
    uint64_t spin_start = current_time_ns();
    const uint64_t spin_budget_ns = 20'000ULL;   // 20 μs spin
    uint64_t val;
    while (current_time_ns() - spin_start < spin_budget_ns) {
        ssize_t n = ::read(eventfd_, &val, sizeof(val));
        if (n == sizeof(val)) {
            return phase.load(std::memory_order_acquire);
        }
        CPU_RELAX();
    }
    // Spin exhausted — fall back to poll() with timeout.
    struct pollfd pfd;
    pfd.fd = eventfd_;
    pfd.events = POLLIN;
    int ms;
    if (timeout_us < 0) {
        ms = -1;                            // block forever
    } else if (timeout_us == 0) {
        ms = 0;                             // non-blocking
    } else {
        ms = (timeout_us + 999) / 1000;     // round-up to ms
    }
    int rc = ::poll(&pfd, 1, ms);
    if (rc > 0 && (pfd.revents & POLLIN)) {
        ::read(eventfd_, &val, sizeof(val));   // drain
    }
    return phase.load(std::memory_order_acquire);
}

uint8_t PhaseSignal::current() const {
    return phase.load(std::memory_order_acquire);
}

uint64_t PhaseSignal::current_step() const {
    return step_id.load(std::memory_order_acquire);
}

uint64_t PhaseSignal::ns_in_phase() const {
    return current_time_ns() - phase_start_ns.load(std::memory_order_acquire);
}

// ──────────────────────────────────────────────────────────────────────
// CUDA event callback path (fallback / reference)
// ──────────────────────────────────────────────────────────────────────

#ifdef ENABLE_CUDA

namespace {
// Arena-allocated callback contexts (avoid malloc per signal).
// Driver-managed thread frees the ctx by index swap.
constexpr int kCtxArenaSize = 4096;
struct CtxArena {
    PhaseHookCtx slots[kCtxArenaSize];
    std::atomic<uint32_t> head{0};
    PhaseHookCtx* acquire() {
        uint32_t i = head.fetch_add(1, std::memory_order_relaxed) % kCtxArenaSize;
        return &slots[i];
    }
};
static CtxArena g_arena;
}  // anonymous

void CUDART_CB phase_signal_callback(cudaStream_t /*stream*/,
                                     cudaError_t /*status*/,
                                     void* userData) {
    auto* ctx = static_cast<PhaseHookCtx*>(userData);
    // driver thread — must be fast, no cuda calls, no malloc.
    ctx->signal->update(ctx->phase, ctx->step_id);
}

void insert_phase_signal(cudaStream_t stream,
                         PhaseSignal* signal,
                         uint8_t phase,
                         uint64_t step_id) {
    PhaseHookCtx* ctx = g_arena.acquire();
    ctx->signal = signal;
    ctx->phase = phase;
    ctx->step_id = step_id;
    cudaStreamAddCallback(stream, phase_signal_callback, ctx, 0);
}

// ──────────────────────────────────────────────────────────────────────
// Event-pair poll path (lower overhead than callback)
// ──────────────────────────────────────────────────────────────────────

PhaseEventPair* PhaseEventPair::create() {
    auto* p = new PhaseEventPair();
    cudaEventCreateWithFlags(&p->start, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&p->stop, cudaEventBlockingSync);
    p->phase = PHASE_IDLE;
    return p;
}

void PhaseEventPair::destroy(PhaseEventPair* p) {
    if (!p) return;
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->stop);
    delete p;
}

void record_phase_start(cudaStream_t stream, PhaseEventPair* p, uint8_t phase) {
    p->phase = phase;
    cudaEventRecord(p->start, stream);
}

void record_phase_stop(cudaStream_t stream, PhaseEventPair* p) {
    cudaEventRecord(p->stop, stream);
}

bool poll_phase_done(PhaseEventPair* p) {
    return cudaEventQuery(p->stop) == cudaSuccess;
}

#endif  // ENABLE_CUDA

// ──────────────────────────────────────────────────────────────────────
// PhaseSignal singleton for Python binding access
// ──────────────────────────────────────────────────────────────────────

static PhaseSignal* g_singleton = nullptr;

PhaseSignal* get_or_create_global_signal() {
    if (g_singleton == nullptr) {
        g_singleton = PhaseSignal::create_anonymous();
    }
    return g_singleton;
}

void release_global_signal() {
    if (g_singleton) {
        PhaseSignal::destroy(g_singleton);
        g_singleton = nullptr;
    }
}

}  // namespace vllm_hybrid_phase
