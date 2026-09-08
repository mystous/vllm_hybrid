#!/usr/bin/env python3
"""IDE_036: 폴러 수준 빈 deferred 생략. arm/rearm 에 (mask_ptr, n_experts, out_row_bytes) 추가. KT_CF_SKIP_EMPTY_DEF=1 이면
폴러가 def_args 의 expert_ids 를 훑어 cold(마스크 0) 이고 ≥0 인 id 가 하나도 없으면 def_fn 을 건너뛰고 출력 슬롯(qlen×row_bytes)을 0 으로 채움."""
P="/sgl-workspace/ktransformers/kt-kernel/cpu_backend/cpuinfer.h"; s=open(P).read()
if "KT_CF_SKIP_EMPTY_DEF" in s: print("already"); raise SystemExit
s=s.replace("""    void (*def_fn)(void*) = nullptr; void* def_args = nullptr;
    bool armed = false;
  };""","""    void (*def_fn)(void*) = nullptr; void* def_args = nullptr;
    const uint8_t* gpu_mask = nullptr; int n_experts = 0; size_t out_row_bytes = 0;
    bool armed = false;
  };
  struct FwdArgsView { void* ci; void* moe; intptr_t qlen_ptr; int k; intptr_t expert_ids; intptr_t weights; intptr_t input; intptr_t output; bool incremental; };
  static bool skip_empty_def_on_() { static bool v = std::getenv("KT_CF_SKIP_EMPTY_DEF") != nullptr; return v; }
  std::atomic<long> n_def_skipped_{0}, n_def_run_{0};
  // deferred 작업에 cold 작업이 하나도 없으면 true (출력 슬롯 0 채움 포함)
  bool def_is_empty_(Packet& p) {
    if (!p.gpu_mask || !p.def_args) return false;
    FwdArgsView* a = (FwdArgsView*)p.def_args; int qlen = *(int*)a->qlen_ptr; const int64_t* ids = (const int64_t*)a->expert_ids;
    for (int i = 0; i < qlen * a->k; ++i) { int64_t e = ids[i]; if (e >= 0 && e < p.n_experts && !p.gpu_mask[e]) return false; }
    memset((void*)a->output, 0, (size_t)qlen * p.out_row_bytes);
    return true;
  }""")
s=s.replace("""  int arm_packet(std::pair<intptr_t, intptr_t> imm, std::pair<intptr_t, intptr_t> def, bool has_def) {
    ensure_cf_();
    int slot = n_packets_.fetch_add(1);
    if (slot >= kMaxPackets) { fprintf(stderr, "[kt-cf] packet table full\\n"); fflush(stderr); abort(); }
    Packet& p = packets_[slot];""","""  int arm_packet(std::pair<intptr_t, intptr_t> imm, std::pair<intptr_t, intptr_t> def, bool has_def, intptr_t mask_ptr = 0, int n_experts = 0, size_t out_row_bytes = 0) {
    ensure_cf_();
    int slot = n_packets_.fetch_add(1);
    if (slot >= kMaxPackets) { fprintf(stderr, "[kt-cf] packet table full\\n"); fflush(stderr); abort(); }
    Packet& p = packets_[slot]; p.gpu_mask = (const uint8_t*)mask_ptr; p.n_experts = n_experts; p.out_row_bytes = out_row_bytes;""")
s=s.replace("""  void rearm_packet(int slot, std::pair<intptr_t, intptr_t> imm, std::pair<intptr_t, intptr_t> def, bool has_def) {
    Packet& p = packets_[slot];""","""  void rearm_packet(int slot, std::pair<intptr_t, intptr_t> imm, std::pair<intptr_t, intptr_t> def, bool has_def, intptr_t mask_ptr = 0, int n_experts = 0, size_t out_row_bytes = 0) {
    Packet& p = packets_[slot]; p.gpu_mask = (const uint8_t*)mask_ptr; p.n_experts = n_experts; p.out_row_bytes = out_row_bytes;""")
old="""        if (p.imm_fn) p.imm_fn(p.imm_args);   // enqueues into task_queue_
        volatile unsigned int* d = done_host_ + slot;
        task_queue_->enqueue([d]() { __atomic_store_n(d, 1u, __ATOMIC_RELEASE); });
        if (p.def_fn) p.def_fn(p.def_args);"""
new="""        if (p.imm_fn) p.imm_fn(p.imm_args);   // enqueues into task_queue_
        volatile unsigned int* d = done_host_ + slot;
        bool run_def = (p.def_fn != nullptr);
        if (run_def && skip_empty_def_on_() && def_is_empty_(p)) { run_def = false; n_def_skipped_.fetch_add(1, std::memory_order_relaxed); }
        task_queue_->enqueue([d]() { __atomic_store_n(d, 1u, __ATOMIC_RELEASE); });
        if (run_def) { p.def_fn(p.def_args); n_def_run_.fetch_add(1, std::memory_order_relaxed); }"""
assert s.count(old)==1; s=s.replace(old,new)
if "#include <cstring>" not in s: s=s.replace("#include <thread>","#include <thread>\n#include <cstring>",1)
open(P,"w").write(s)
B="/sgl-workspace/ktransformers/kt-kernel/ext_bindings.cpp"; b=open(B).read()
b=b.replace('''      .def("arm_packet", &CPUInfer::arm_packet, py::arg("imm"), py::arg("def"), py::arg("has_def"))
      .def("rearm_packet", &CPUInfer::rearm_packet, py::arg("slot"), py::arg("imm"), py::arg("def"), py::arg("has_def"))''',
'''      .def("arm_packet", &CPUInfer::arm_packet, py::arg("imm"), py::arg("def"), py::arg("has_def"), py::arg("mask_ptr") = 0, py::arg("n_experts") = 0, py::arg("out_row_bytes") = 0)
      .def("rearm_packet", &CPUInfer::rearm_packet, py::arg("slot"), py::arg("imm"), py::arg("def"), py::arg("has_def"), py::arg("mask_ptr") = 0, py::arg("n_experts") = 0, py::arg("out_row_bytes") = 0)
      .def("cf_def_stats", [](CPUInfer& ci) { return std::make_pair(ci.n_def_skipped_.load(), ci.n_def_run_.load()); })''')
assert "cf_def_stats" in b; open(B,"w").write(b); print("IDE_036 C++ patch applied")
