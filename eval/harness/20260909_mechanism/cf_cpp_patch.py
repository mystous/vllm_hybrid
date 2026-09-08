#!/usr/bin/env python3
"""IDE_033 Callback-free CPU<->GPU handoff — C++ patch for kt-kernel cpuinfer.h + ext_bindings.cpp.

Mechanism:
  - arm_packet(imm_task, def_task_or_none) -> slot : registers task fn/args (persist for graph replay) in packet table.
  - go_on_stream(stream, slot): stream memop writes go_flag[slot]=1 (replaces submit host nodes).
  - poller thread: spins over armed slots; on go_flag==1 -> reset, enqueue [imm, done-signal, def] into task queue.
  - wait: existing wait_signal_on_stream(stream, done_slot) (memop wait>=1, write 0) with done_slot = slot (separate array).
No cudaLaunchHostFunc anywhere on the path.
"""
import re
P = "/sgl-workspace/ktransformers/kt-kernel/cpu_backend/cpuinfer.h"
s = open(P).read()

anchor = "  // ---- Non-blocking completion signaling (interleaved dual-graph support) ----"
assert s.count(anchor) == 1
cf = r'''
  // ---- IDE_033: Callback-free handoff (GPU->CPU via mapped go-flags + poller; CPU->GPU via done-flags memop) ----
#ifndef KTRANSFORMERS_CPU_ONLY
  static constexpr int kMaxPackets = 16384;
  struct Packet {
    void (*imm_fn)(void*) = nullptr; void* imm_args = nullptr;
    void (*def_fn)(void*) = nullptr; void* def_args = nullptr;
    bool armed = false;
  };
  Packet packets_[kMaxPackets];
  std::atomic<int> n_packets_{0};
  volatile unsigned int* go_host_ = nullptr;   unsigned int* go_dev_ = nullptr;
  volatile unsigned int* done_host_ = nullptr; unsigned int* done_dev_ = nullptr;
  std::thread poller_; std::atomic<bool> poller_run_{false};

  void ensure_cf_() {
    if (go_host_ != nullptr) return;
    cudaError_t e;
    e = cudaHostAlloc((void**)&go_host_, kMaxPackets * sizeof(unsigned int), cudaHostAllocMapped | cudaHostAllocPortable);
    if (e != cudaSuccess) { fprintf(stderr, "[kt-cf] cudaHostAlloc go FAILED %d\n", (int)e); fflush(stderr); abort(); }
    e = cudaHostAlloc((void**)&done_host_, kMaxPackets * sizeof(unsigned int), cudaHostAllocMapped | cudaHostAllocPortable);
    if (e != cudaSuccess) { fprintf(stderr, "[kt-cf] cudaHostAlloc done FAILED %d\n", (int)e); fflush(stderr); abort(); }
    for (int i = 0; i < kMaxPackets; ++i) { go_host_[i] = 0; done_host_[i] = 0; }
    cudaHostGetDevicePointer((void**)&go_dev_, (void*)go_host_, 0);
    cudaHostGetDevicePointer((void**)&done_dev_, (void*)done_host_, 0);
    ensure_driver_fns_();
    poller_run_ = true;
    poller_ = std::thread([this]() { this->poll_loop_(); });
    fprintf(stderr, "[kt-cf] callback-free handoff ready (ci=%p)\n", (void*)this); fflush(stderr);
  }

  // Register a packet (tasks persist: graph replays re-trigger the same slot). Returns slot id.
  int arm_packet(std::pair<intptr_t, intptr_t> imm, std::pair<intptr_t, intptr_t> def, bool has_def) {
    ensure_cf_();
    int slot = n_packets_.fetch_add(1);
    if (slot >= kMaxPackets) { fprintf(stderr, "[kt-cf] packet table full\n"); fflush(stderr); abort(); }
    Packet& p = packets_[slot];
    p.imm_fn = (void (*)(void*))imm.first; p.imm_args = (void*)imm.second;
    if (p.imm_args) *((CPUInfer**)p.imm_args) = this;
    if (has_def) { p.def_fn = (void (*)(void*))def.first; p.def_args = (void*)def.second; if (p.def_args) *((CPUInfer**)p.def_args) = this; }
    else { p.def_fn = nullptr; p.def_args = nullptr; }
    __atomic_store_n(&p.armed, true, __ATOMIC_RELEASE);
    return slot;
  }
  // Re-arm an existing slot with (possibly) new task args (eager path reuse).
  void rearm_packet(int slot, std::pair<intptr_t, intptr_t> imm, std::pair<intptr_t, intptr_t> def, bool has_def) {
    Packet& p = packets_[slot];
    p.imm_fn = (void (*)(void*))imm.first; p.imm_args = (void*)imm.second; if (p.imm_args) *((CPUInfer**)p.imm_args) = this;
    if (has_def) { p.def_fn = (void (*)(void*))def.first; p.def_args = (void*)def.second; if (p.def_args) *((CPUInfer**)p.def_args) = this; }
    else { p.def_fn = nullptr; p.def_args = nullptr; }
    __atomic_store_n(&p.armed, true, __ATOMIC_RELEASE);
  }
  // GPU-side trigger: stream memop writes go_flag[slot]=1 (capturable, replay-safe: poller resets to 0).
  void go_on_stream(intptr_t user_cuda_stream, int slot) {
    ensure_cf_();
    int r = cu_write_val32_((void*)user_cuda_stream, (unsigned long long)(uintptr_t)(go_dev_ + slot), 1u, 0u);
    if (r != 0) { fprintf(stderr, "[kt-cf] go memop FAILED slot=%d r=%d\n", slot, r); fflush(stderr); abort(); }
  }
  // GPU waits until done_flag[slot]>=1 then resets to 0.
  void wait_done_on_stream(intptr_t user_cuda_stream, int slot) {
    ensure_cf_();
    int r1 = cu_wait_val32_((void*)user_cuda_stream, (unsigned long long)(uintptr_t)(done_dev_ + slot), 1u, 1u);
    int r2 = cu_write_val32_((void*)user_cuda_stream, (unsigned long long)(uintptr_t)(done_dev_ + slot), 0u, 0u);
    if (r1 != 0 || r2 != 0) { fprintf(stderr, "[kt-cf] done memop FAILED slot=%d %d %d\n", slot, r1, r2); fflush(stderr); abort(); }
  }
  void poll_loop_() {
    while (poller_run_.load(std::memory_order_acquire)) {
      int n = n_packets_.load(std::memory_order_acquire);
      bool any = false;
      for (int slot = 0; slot < n; ++slot) {
        if (__atomic_load_n(&go_host_[slot], __ATOMIC_ACQUIRE) != 1u) continue;
        __atomic_store_n(&go_host_[slot], 0u, __ATOMIC_RELEASE);
        Packet& p = packets_[slot]; any = true;
        // immediate task -> done signal -> deferred task  (== KT sync(allow_pending=1) semantics)
        if (p.imm_fn) p.imm_fn(p.imm_args);   // enqueues into task_queue_
        volatile unsigned int* d = done_host_ + slot;
        task_queue_->enqueue([d]() { __atomic_store_n(d, 1u, __ATOMIC_RELEASE); });
        if (p.def_fn) p.def_fn(p.def_args);
      }
      if (!any) { for (int k = 0; k < 32; ++k) __builtin_ia32_pause(); }
    }
  }
#endif
'''
s = s.replace(anchor, cf + "\n" + anchor)
# includes
if "#include <thread>" not in s:
    s = s.replace("#include <mutex>", "#include <mutex>\n#include <thread>\n#include <atomic>", 1) if "#include <mutex>" in s else "#include <thread>\n#include <atomic>\n" + s
open(P, "w").write(s)

B = "/sgl-workspace/ktransformers/kt-kernel/ext_bindings.cpp"
b = open(B).read()
anchor_b = '''      .def("write_flag_on_stream", &CPUInfer::write_flag_on_stream,
           py::arg("user_cuda_stream"), py::arg("slot"), py::arg("val"))'''
assert b.count(anchor_b) == 1
b = b.replace(anchor_b, anchor_b + '''
      .def("arm_packet", &CPUInfer::arm_packet, py::arg("imm"), py::arg("def"), py::arg("has_def"))
      .def("rearm_packet", &CPUInfer::rearm_packet, py::arg("slot"), py::arg("imm"), py::arg("def"), py::arg("has_def"))
      .def("go_on_stream", &CPUInfer::go_on_stream, py::arg("user_cuda_stream"), py::arg("slot"))
      .def("wait_done_on_stream", &CPUInfer::wait_done_on_stream, py::arg("user_cuda_stream"), py::arg("slot"))''')
open(B, "w").write(b)
print("IDE_033 C++ patch applied")
