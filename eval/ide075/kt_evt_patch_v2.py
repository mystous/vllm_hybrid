#!/usr/bin/env python3
"""IDE_075 A03 — kt-kernel 저간섭 기록기 v2 (컨테이너 sgl-kt 안에서 실행). v1(IDE_074) 의 stamp 태스크 삽입을 제거한다.
원칙(지시서 §6.2): 모델 FIFO 에 기록 전용 task 추가 금지 / 사전 할당 bounded buffer / hot path 에 파일 I/O·할당 없음 / 전역 mutex 없음 / 레코드는 immutable 스냅샷 / 누락 계수 / flush 는 별도 스레드.
기록 지점:
  (1) TaskQueue (cpu_backend/task_queue.{h,cpp}): task_seq, kind(호출 스레드가 지정), pending_at_enqueue, running_seq_at_enqueue, enqueue bracket(before/after 가시화), dequeue, exec start/end, worker tid → KtTaskRec 링(seq%cap)
  (2) poller (cpuinfer.h::poll_loop_): go 관측, imm/done/def enqueue bracket + 각 task_seq, qlen·n_cold_ids 스냅샷; done-setter(기존 업무 task) 안에서 store 전후 bracket
  (3) deferred/imm 작업 본체 (moe-tp.hpp::TP_MOE_Common::forward): 진입/반환, NUMA 서브잡 시작/종료, merge 종료 — poller→worker SPSC 로 전달된 (rec, task_seq) 를 현재 task_seq 와 대조해서만 연결
  (4) 단계 타이머 (moe_base.hpp forward_prefill/forward_decode 의 기존 FORWARD_TIME_PROFILE 값) 와 activated_expert/max rows/sum rows 를 rec 에 기록 (printf 대신; 계산은 기존과 동일)
  (5) 실제 GEMM 분기 (moe.hpp do_gate_up_gemm/do_down_gemm, ith==0 만): AMX/AVX 호출 수
활성: env KT_EVT=<csv>. 미설정 시 rec/ring 이 nullptr 이라 모든 분기가 비활성 (task_seq 카운터·Node.seq 필드는 상시).
사용: kt_evt_patch_v2.py apply | revert | status   (apply 는 먼저 v1 을 .orig 로 되돌린 뒤 적용)"""
import os, sys, shutil, hashlib, re
KT = "/sgl-workspace/ktransformers/kt-kernel"; PY = "/usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py"; BK = "/sgl-workspace/ide074_backup"
F_CI = f"{KT}/cpu_backend/cpuinfer.h"; F_TQH = f"{KT}/cpu_backend/task_queue.h"; F_TQC = f"{KT}/cpu_backend/task_queue.cpp"; F_TP = f"{KT}/operators/moe-tp.hpp"; F_MB = f"{KT}/operators/amx/moe_base.hpp"; F_MOE = f"{KT}/operators/amx/moe.hpp"
FILES = [F_CI, F_TQH, F_TQC, F_TP, F_MB, F_MOE, PY]; HDR = f"{KT}/cpu_backend/kt_evt.h"; MARK = "IDE_075"

KT_EVT_H = r'''// IDE_075: 저간섭 CPU-side 기록기 (CLOCK_REALTIME ns). 기록 전용 task 없음. env KT_EVT 미설정 시 모든 포인터 nullptr.
#pragma once
#include <time.h>
#include <atomic>
#include <cstdint>
struct KtEvtRec {   // 패킷 go 1회당 1 레코드 (poller 가 생성, 이후 필드는 각 스레드가 자기 몫만 기록)
  int slot; unsigned epoch; int qlen; int n_cold_ids; unsigned char imm_present, def_present, def_skipped, pad_;
  long long t_go, t_imm_enq_b, t_imm_enq_a, t_done_enq_b, t_done_enq_a, t_def_enq_b, t_def_enq_a;
  unsigned long long imm_task_seq, done_task_seq, def_task_seq;
  long long t_done_store_b, t_done_store_a;                       // done-setter task (worker) 안의 store 전후
  long long t_fwd_entry, t_fwd_exit, t_imm_fwd_entry, t_imm_fwd_exit, t_numa_start[2], t_numa_end[2], t_merge_end;
  int activated_expert[2], max_local_num[2], sum_rows[2], n_amx[2], n_avx[2], qlen_seen[2]; long stage_us[2][9];
};
struct KtTaskRec { unsigned long long seq, running_seq_at_enq; long long t_enq_b, t_enq_a, t_dequeue, t_exec_start, t_exec_end; unsigned pending_at_enq, tid; unsigned char kind; };
struct KtEvtPend { KtEvtRec* rec; unsigned long long seq; int which; };   // poller → task worker (enqueue 순)
inline long long kt_evt_now() { timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec; }
inline KtTaskRec* kt_task_ring = nullptr; inline unsigned kt_task_cap = 0;
inline thread_local unsigned char kt_evt_next_kind = 0;       // enqueue 호출 스레드가 지정: 1 imm, 2 done_setter, 3 deferred, 0 기타
inline thread_local unsigned long long kt_evt_last_seq = 0;   // 이 스레드의 마지막 enqueue seq
inline thread_local unsigned long long kt_cur_task_seq = 0;   // task worker 스레드: 실행 중 task seq
inline std::atomic<unsigned long long> kt_running_seq{0};
constexpr unsigned KT_EVT_SPSC_CAP = 16384;
inline KtEvtPend kt_evt_spsc[KT_EVT_SPSC_CAP]; inline std::atomic<unsigned> kt_evt_spsc_head{0}, kt_evt_spsc_tail{0}; inline std::atomic<unsigned> kt_evt_spsc_full{0}, kt_evt_spsc_stale{0};
inline void kt_evt_push(KtEvtRec* r, unsigned long long seq, int which) {
  unsigned t = kt_evt_spsc_tail.load(std::memory_order_relaxed), n = (t + 1) % KT_EVT_SPSC_CAP;
  if (n == kt_evt_spsc_head.load(std::memory_order_acquire)) { kt_evt_spsc_full.fetch_add(1, std::memory_order_relaxed); return; }
  kt_evt_spsc[t] = KtEvtPend{r, seq, which}; kt_evt_spsc_tail.store(n, std::memory_order_release);
}
inline KtEvtPend kt_evt_pop_for(unsigned long long seq) {   // 현재 task seq 에 해당하는 pend 만 반환; 더 오래된 pend 는 stale 로 버림
  for (;;) {
    unsigned h = kt_evt_spsc_head.load(std::memory_order_relaxed);
    if (h == kt_evt_spsc_tail.load(std::memory_order_acquire)) return KtEvtPend{nullptr, 0, -1};
    KtEvtPend p = kt_evt_spsc[h];
    if (p.seq == seq) { kt_evt_spsc_head.store((h + 1) % KT_EVT_SPSC_CAP, std::memory_order_release); return p; }
    if (p.seq < seq) { kt_evt_spsc_head.store((h + 1) % KT_EVT_SPSC_CAP, std::memory_order_release); kt_evt_spsc_stale.fetch_add(1, std::memory_order_relaxed); continue; }
    return KtEvtPend{nullptr, 0, -1};
  }
}
'''


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def rep(s, old, new, n=1):
    assert s.count(old) == n, (old[:80], s.count(old)); return s.replace(old, new)


def patch_tqh(s):
    s = rep(s, '#include <vector>\n', '#include <vector>\n#include "kt_evt.h"   // IDE_075\n')
    s = rep(s, "    std::function<void()> task;\n    std::atomic<Node*> next;\n", "    std::function<void()> task;\n    std::atomic<Node*> next;\n    unsigned long long seq = 0;   // IDE_075\n")
    s = rep(s, "  std::atomic<size_t> pending;\n", "  std::atomic<size_t> pending;\n  std::atomic<unsigned long long> seq_ctr_{0};   // IDE_075\n")
    return s


def patch_tqc(s):
    old = '''void TaskQueue::enqueue(std::function<void()> task) {
  pending.fetch_add(1, std::memory_order_acq_rel);
  Node* node = new Node(task);
  Node* prev = tail.exchange(node, std::memory_order_acq_rel);
  prev->next.store(node, std::memory_order_release);
'''
    new = '''void TaskQueue::enqueue(std::function<void()> task) {
  // IDE_075: task_seq + enqueue bracket (기록 전용 task 없음; ring 미할당(OFF) 시 r==nullptr)
  unsigned long long seq = seq_ctr_.fetch_add(1, std::memory_order_acq_rel) + 1; KtTaskRec* r = nullptr;
  if (kt_task_ring) { r = &kt_task_ring[seq % kt_task_cap]; r->seq = seq; r->kind = kt_evt_next_kind; r->pending_at_enq = (unsigned)pending.load(std::memory_order_acquire); r->running_seq_at_enq = kt_running_seq.load(std::memory_order_relaxed); r->t_dequeue = r->t_exec_start = r->t_exec_end = 0; r->tid = 0; r->t_enq_b = kt_evt_now(); }
  pending.fetch_add(1, std::memory_order_acq_rel);
  Node* node = new Node(task); node->seq = seq;
  Node* prev = tail.exchange(node, std::memory_order_acq_rel);
  prev->next.store(node, std::memory_order_release);
  if (r) r->t_enq_a = kt_evt_now();
  kt_evt_last_seq = seq;
'''
    s = rep(s, old, new)
    old = '''      if (next->task) {
        try {
          if (tq_on) tq_before_();
          next->task();
          if (tq_on) tq_after_();
        } catch (...) {'''
    new = '''      KtTaskRec* r = kt_task_ring ? &kt_task_ring[next->seq % kt_task_cap] : nullptr;   // IDE_075
      if (r) r->t_dequeue = kt_evt_now();
      kt_cur_task_seq = next->seq; kt_running_seq.store(next->seq, std::memory_order_relaxed);
      if (next->task) {
        try {
          if (tq_on) tq_before_();
          if (r) r->t_exec_start = kt_evt_now();
          next->task();
          if (r) { r->t_exec_end = kt_evt_now(); static thread_local unsigned _tid = (unsigned)syscall(SYS_gettid); r->tid = _tid; }
          if (tq_on) tq_after_();
        } catch (...) {'''
    s = rep(s, old, new)
    s = rep(s, "#include <chrono>\n", "#include <chrono>\n#include <unistd.h>\n#include <sys/syscall.h>\n", 1)
    return s


def patch_cpuinfer(s):
    s = rep(s, '#include "task_queue.h"\n', '#include "task_queue.h"\n#include "kt_evt.h"   // IDE_075\n')
    old = "  Packet packets_[kMaxPackets];\n  std::atomic<int> n_packets_{0};\n"
    new = old + r'''  // ---- IDE_075: 저간섭 기록기. env KT_EVT=<csv> 로만 활성 ----
  static constexpr int kEvtCap = 1 << 20; static constexpr unsigned kTaskCap = 1u << 20;
  KtEvtRec* evt_ = nullptr; unsigned evt_n_ = 0; std::atomic<unsigned> evt_dropped_{0}; unsigned evt_written_ = 0; unsigned long long task_written_ = 0;
  unsigned slot_epoch_[kMaxPackets] = {0};
  std::thread evt_flusher_; std::atomic<bool> evt_run_{false}; FILE* evt_fp_ = nullptr; FILE* task_fp_ = nullptr;
  static const char* evt_path_() { static const char* v = std::getenv("KT_EVT"); return v; }
  static bool evt_complete_(const KtEvtRec& r) { return (r.def_present && !r.def_skipped) ? (r.t_fwd_exit != 0) : (r.t_done_store_a != 0); }
  void evt_write_(const KtEvtRec& r) {
    fprintf(evt_fp_, "%d,%u,%d,%d,%u,%u,%u,%lld,%lld,%lld,%llu,%lld,%lld,%llu,%lld,%lld,%llu,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
            r.slot, r.epoch, r.qlen, r.n_cold_ids, r.imm_present, r.def_present, r.def_skipped, r.t_go, r.t_imm_enq_b, r.t_imm_enq_a, r.imm_task_seq, r.t_done_enq_b, r.t_done_enq_a, r.done_task_seq, r.t_def_enq_b, r.t_def_enq_a, r.def_task_seq,
            r.t_done_store_b, r.t_done_store_a, r.t_fwd_entry, r.t_fwd_exit, r.t_imm_fwd_entry, r.t_imm_fwd_exit, r.t_numa_start[0], r.t_numa_end[0], r.t_numa_start[1], r.t_numa_end[1], r.t_merge_end,
            r.activated_expert[0], r.activated_expert[1], r.max_local_num[0], r.max_local_num[1], r.sum_rows[0], r.sum_rows[1], r.n_amx[0], r.n_amx[1], r.n_avx[0], r.n_avx[1], r.qlen_seen[0], r.qlen_seen[1]);
    for (int s = 0; s < 2; ++s) for (int k = 0; k < 9; ++k) fprintf(evt_fp_, ",%ld", r.stage_us[s][k]);
    fputc('\n', evt_fp_);
  }
  void evt_flush_(bool final_) {
    long long now = kt_evt_now();
    unsigned n = __atomic_load_n(&evt_n_, __ATOMIC_ACQUIRE); if (n > (unsigned)kEvtCap) n = kEvtCap;
    while (evt_written_ < n) { KtEvtRec& r = evt_[evt_written_]; if (!final_ && !evt_complete_(r) && now - r.t_go < 5000000000LL) break; evt_write_(r); ++evt_written_; }
    fflush(evt_fp_);
    if (kt_task_ring && task_fp_) {
      unsigned long long last = kt_running_seq.load(std::memory_order_relaxed);
      while (task_written_ < last) { KtTaskRec& t = kt_task_ring[(task_written_ + 1) % kt_task_cap]; if (t.seq != task_written_ + 1) { fprintf(task_fp_, "%llu,,,,,,,,,OVERWRITTEN_OR_MISSING\n", task_written_ + 1); ++task_written_; continue; }
        if (!final_ && t.t_exec_end == 0 && now - t.t_enq_b < 5000000000LL) break;
        fprintf(task_fp_, "%llu,%u,%u,%llu,%lld,%lld,%lld,%lld,%lld,%u\n", t.seq, t.kind, t.pending_at_enq, t.running_seq_at_enq, t.t_enq_b, t.t_enq_a, t.t_dequeue, t.t_exec_start, t.t_exec_end, t.tid); ++task_written_; }
      fflush(task_fp_);
    }
    if (final_) { fprintf(evt_fp_, "#dropped=%u,written=%u,spsc_full=%u,spsc_stale=%u\n", evt_dropped_.load(), evt_written_, kt_evt_spsc_full.load(), kt_evt_spsc_stale.load()); fflush(evt_fp_); }
  }
  void evt_ensure_() {
    if (evt_ != nullptr || evt_path_() == nullptr) return;
    evt_ = (KtEvtRec*)calloc((size_t)kEvtCap, sizeof(KtEvtRec));
    kt_task_ring = (KtTaskRec*)calloc((size_t)kTaskCap, sizeof(KtTaskRec)); kt_task_cap = kTaskCap;
    evt_fp_ = fopen(evt_path_(), "a"); std::string tp = std::string(evt_path_()) + ".tasks"; task_fp_ = fopen(tp.c_str(), "a");
    if (!evt_fp_ || !task_fp_) { fprintf(stderr, "[kt-evt] fopen failed\n"); fflush(stderr); free(evt_); evt_ = nullptr; free(kt_task_ring); kt_task_ring = nullptr; kt_task_cap = 0; return; }
    fprintf(evt_fp_, "slot,epoch,qlen,n_cold_ids,imm_present,def_present,def_skipped,t_go,t_imm_enq_b,t_imm_enq_a,imm_task_seq,t_done_enq_b,t_done_enq_a,done_task_seq,t_def_enq_b,t_def_enq_a,def_task_seq,t_done_store_b,t_done_store_a,t_fwd_entry,t_fwd_exit,t_imm_fwd_entry,t_imm_fwd_exit,t_numa0_start,t_numa0_end,t_numa1_start,t_numa1_end,t_merge_end,act0,act1,maxrows0,maxrows1,sumrows0,sumrows1,namx0,namx1,navx0,navx1,qlen0,qlen1,s0_prepare,s0_cpy_input,s0_q_input,s0_up_gate,s0_act,s0_q_down,s0_down,s0_weight,s0_total,s1_prepare,s1_cpy_input,s1_q_input,s1_up_gate,s1_act,s1_q_down,s1_down,s1_weight,s1_total\n");
    fprintf(task_fp_, "seq,kind,pending_at_enq,running_seq_at_enq,t_enq_b,t_enq_a,t_dequeue,t_exec_start,t_exec_end,tid\n"); fflush(evt_fp_); fflush(task_fp_);
    evt_run_ = true;
    evt_flusher_ = std::thread([this]() { while (evt_run_.load(std::memory_order_acquire)) { evt_flush_(false); std::this_thread::sleep_for(std::chrono::milliseconds(200)); } evt_flush_(true); });
    fprintf(stderr, "[kt-evt] v2 event log ready: %s (+.tasks) cap %d/%u\n", evt_path_(), kEvtCap, kTaskCap); fflush(stderr);
  }
'''
    s = rep(s, old, new)
    s = rep(s, "    poller_run_ = true;\n    poller_ = std::thread([this]() { this->poll_loop_(); });\n", "    evt_ensure_();   // IDE_075\n    poller_run_ = true;\n    poller_ = std::thread([this]() { this->poll_loop_(); });\n")
    old = '''        Packet& p = packets_[slot]; any = true;
        // immediate task -> done signal -> deferred task  (== KT sync(allow_pending=1) semantics)
        if (p.imm_fn) p.imm_fn(p.imm_args);   // enqueues into task_queue_
        volatile unsigned int* d = done_host_ + slot;
        bool run_def = (p.def_fn != nullptr);
        if (run_def && skip_empty_def_on_() && def_is_empty_(p)) { run_def = false; n_def_skipped_.fetch_add(1, std::memory_order_relaxed); }
        task_queue_->enqueue([d]() { __atomic_store_n(d, 1u, __ATOMIC_RELEASE); });
        if (run_def) { p.def_fn(p.def_args); n_def_run_.fetch_add(1, std::memory_order_relaxed); }
'''
    new = '''        Packet& p = packets_[slot]; any = true;
        KtEvtRec* rec = nullptr;   // IDE_075 (KT_EVT 미설정 → nullptr)
        if (evt_) {
          unsigned i = evt_n_; __atomic_store_n(&evt_n_, i + 1, __ATOMIC_RELEASE);
          if (i < (unsigned)kEvtCap) {
            rec = &evt_[i]; rec->slot = slot; rec->epoch = ++slot_epoch_[slot]; rec->t_go = kt_evt_now();
            rec->imm_present = p.imm_fn != nullptr; rec->def_present = p.def_fn != nullptr;
            if (p.def_args) { FwdArgsView* a = (FwdArgsView*)p.def_args; rec->qlen = *(int*)a->qlen_ptr; int nc = 0; const int64_t* ids = (const int64_t*)a->expert_ids;
              for (int t = 0; t < rec->qlen * a->k; ++t) { int64_t e = ids[t]; if (e >= 0 && e < p.n_experts && (!p.gpu_mask || !p.gpu_mask[e])) ++nc; } rec->n_cold_ids = nc; }
          } else evt_dropped_.fetch_add(1, std::memory_order_relaxed);
        }
        // immediate task -> done signal -> deferred task  (== KT sync(allow_pending=1) semantics)  [task 구성은 OFF 와 동일]
        if (p.imm_fn) { if (rec) { kt_evt_next_kind = 1; rec->t_imm_enq_b = kt_evt_now(); } p.imm_fn(p.imm_args); if (rec) { rec->t_imm_enq_a = kt_evt_now(); rec->imm_task_seq = kt_evt_last_seq; kt_evt_push(rec, kt_evt_last_seq, 0); } }
        volatile unsigned int* d = done_host_ + slot;
        bool run_def = (p.def_fn != nullptr);
        if (run_def && skip_empty_def_on_() && def_is_empty_(p)) { run_def = false; n_def_skipped_.fetch_add(1, std::memory_order_relaxed); if (rec) rec->def_skipped = 1; }
        if (rec) { kt_evt_next_kind = 2; rec->t_done_enq_b = kt_evt_now(); }
        task_queue_->enqueue([d, rec]() { if (rec) rec->t_done_store_b = kt_evt_now(); __atomic_store_n(d, 1u, __ATOMIC_RELEASE); if (rec) rec->t_done_store_a = kt_evt_now(); });
        if (rec) { rec->t_done_enq_a = kt_evt_now(); rec->done_task_seq = kt_evt_last_seq; }
        if (run_def) { if (rec) { kt_evt_next_kind = 3; rec->t_def_enq_b = kt_evt_now(); } p.def_fn(p.def_args); n_def_run_.fetch_add(1, std::memory_order_relaxed); if (rec) { rec->t_def_enq_a = kt_evt_now(); rec->def_task_seq = kt_evt_last_seq; kt_evt_push(rec, kt_evt_last_seq, 1); } }
        kt_evt_next_kind = 0;
'''
    s = rep(s, old, new)
    s = rep(s, "  ~CPUInfer() {\n    printf(\"CPUInfer[0x%lx]: Goodbye\\n\", (intptr_t)this);\n", "  ~CPUInfer() {\n    printf(\"CPUInfer[0x%lx]: Goodbye\\n\", (intptr_t)this);\n#ifndef KTRANSFORMERS_CPU_ONLY\n    if (evt_run_.load()) { evt_run_ = false; if (evt_flusher_.joinable()) evt_flusher_.join(); if (evt_fp_) fclose(evt_fp_); if (task_fp_) fclose(task_fp_); }   // IDE_075\n#endif\n")
    if "#include <string>" not in s: s = rep(s, "#include <vector>\n", "#include <vector>\n#include <string>\n", 1)
    return s


def patch_moetp(s):
    s = rep(s, '#include "../cpu_backend/shared_mem_buffer.h"\n', '#include "../cpu_backend/shared_mem_buffer.h"\n#include "../cpu_backend/kt_evt.h"   // IDE_075\n')
    old = '''    pool->dispense_backend()->do_numa_job([this, pool, qlen, k, expert_ids, input, weights](int numa_id) {
      tps[numa_id]->forward(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id]);
    });
    auto _wp_t1 = std::chrono::steady_clock::now();
    merge_results(qlen, output, incremental);'''
    new = '''    KtEvtPend _pd = kt_evt_pop_for(kt_cur_task_seq);   // IDE_075: 이 task 에 대응하는 레코드만 (OFF: rec==nullptr)
    KtEvtRec* _r = _pd.rec; if (_r && _pd.which == 0) { _r->t_imm_fwd_entry = kt_evt_now(); }
    KtEvtRec* _rd = (_r && _pd.which == 1) ? _r : nullptr; if (_rd) _rd->t_fwd_entry = kt_evt_now();
    pool->dispense_backend()->do_numa_job([this, pool, qlen, k, expert_ids, input, weights, _rd](int numa_id) {
      if constexpr (requires (T& x) { x.kt_rec_; }) tps[numa_id]->kt_rec_ = _rd;
      if (_rd && numa_id >= 0 && numa_id < 2) _rd->t_numa_start[numa_id] = kt_evt_now();
      tps[numa_id]->forward(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id]);
      if (_rd && numa_id >= 0 && numa_id < 2) _rd->t_numa_end[numa_id] = kt_evt_now();
      if constexpr (requires (T& x) { x.kt_rec_; }) tps[numa_id]->kt_rec_ = nullptr;
    });
    auto _wp_t1 = std::chrono::steady_clock::now();
    merge_results(qlen, output, incremental);
    if (_rd) { _rd->t_merge_end = kt_evt_now(); _rd->t_fwd_exit = _rd->t_merge_end; }
    if (_r && _pd.which == 0) _r->t_imm_fwd_exit = kt_evt_now();'''
    return rep(s, old, new)


def patch_moebase(s):
    s = rep(s, "#include <fstream>\n", "#include <fstream>\n#include \"../../cpu_backend/kt_evt.h\"   // IDE_075\n", 1)
    s = rep(s, "  int tp_part_idx = 0;\n", "  int tp_part_idx = 0;\n  KtEvtRec* kt_rec_ = nullptr;   // IDE_075: 현재 작업 레코드 (TP_MOE_Common::forward 가 설정)\n", 1)
    # prefill: printf 지점 앞에 rec 기록 (printf 는 KT_PHASE_PROF 게이트 그대로)
    old = '''    static bool _pp_on = std::getenv("KT_PHASE_PROF") != nullptr; static thread_local unsigned _pp_cnt = 0;'''
    new = '''    if (kt_rec_ && tp_part_idx >= 0 && tp_part_idx < 2) {   // IDE_075: 단계 µs·규모를 레코드에 기록 (계산은 기존 타이머 그대로)
      int _s = tp_part_idx; long* _st = kt_rec_->stage_us[_s]; _st[0] = prepare_time; _st[1] = cpy_input_time; _st[2] = q_input_time; _st[3] = up_gate_time; _st[4] = act_time; _st[5] = q_down_time; _st[6] = down_time; _st[7] = weight_time; _st[8] = forward_total_time;
      kt_rec_->activated_expert[_s] = activated_expert; kt_rec_->max_local_num[_s] = max_local_num; kt_rec_->qlen_seen[_s] = qlen;
      int _sum = 0; for (int _e = 0; _e < config_.expert_num; ++_e) _sum += m_local_num_[_e]; kt_rec_->sum_rows[_s] = _sum;
    }
    static bool _pp_on = std::getenv("KT_PHASE_PROF") != nullptr; static thread_local unsigned _pp_cnt = 0;'''
    s = rep(s, old, new)
    # decode: v1 게이트 유지 + rec 기록
    old = '''    static bool _pd_on = std::getenv("KT_PHASE_PROF") != nullptr; static thread_local unsigned _pd_cnt = 0;   // IDE_074: decode printf 게이트 (기존: 매 호출 무조건 출력)'''
    new = '''    if (kt_rec_ && tp_part_idx >= 0 && tp_part_idx < 2) {   // IDE_075
      int _s = tp_part_idx; long* _st = kt_rec_->stage_us[_s]; _st[0] = 0; _st[1] = 0; _st[2] = q_input_time; _st[3] = up_gate_time; _st[4] = act_time; _st[5] = q_down_time; _st[6] = down_time; _st[7] = weight_time; _st[8] = forward_total_time;
      kt_rec_->activated_expert[_s] = activated_expert; kt_rec_->qlen_seen[_s] = 1; kt_rec_->sum_rows[_s] = activated_expert;
    }
    static bool _pd_on = std::getenv("KT_PHASE_PROF") != nullptr; static thread_local unsigned _pd_cnt = 0;   // IDE_074: decode printf 게이트 (기존: 매 호출 무조건 출력)'''
    if old in s: s = rep(s, old, new)
    else:
        # v1 미적용 원본이면 v1 과 같은 게이트를 함께 적용
        old0 = '''    auto end_time = std::chrono::high_resolution_clock::now();
    auto forward_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    printf(
        "Profiling Results (numa[%d]): activated_expert: %d, q_input: %ld us, "'''
        new0 = '''    auto end_time = std::chrono::high_resolution_clock::now();
    auto forward_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
''' + new.replace("    static bool _pd_on", "    static bool _pd_on") + '''
    if (_pd_on && ((++_pd_cnt) & 63) == 0) printf(
        "Profiling Results (numa[%d]): activated_expert: %d, q_input: %ld us, "'''
        s = rep(s, old0, new0)
    return s


def patch_moe(s):
    old = '''    if (qlen > amx_min_qlen_() || m >= amx_min_rows_()) {
      amx::mat_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    }'''
    new = '''    if (qlen > amx_min_qlen_() || m >= amx_min_rows_()) {
      if (this->kt_rec_ && ith == 0 && tp_part_idx < 2) __atomic_fetch_add(&this->kt_rec_->n_amx[tp_part_idx], 1, __ATOMIC_RELAXED);   // IDE_075 실제 분기 계수 (expert·gemm 당 1)
      amx::mat_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    } else {
      if (this->kt_rec_ && ith == 0 && tp_part_idx < 2) __atomic_fetch_add(&this->kt_rec_->n_avx[tp_part_idx], 1, __ATOMIC_RELAXED);
      amx::vec_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    }'''
    s = rep(s, old, new)
    old = '''    if (qlen > amx_min_qlen_() || m >= amx_min_rows_()) {
      amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
    }'''
    new = '''    if (qlen > amx_min_qlen_() || m >= amx_min_rows_()) {
      if (this->kt_rec_ && ith == 0 && tp_part_idx < 2) __atomic_fetch_add(&this->kt_rec_->n_amx[tp_part_idx], 1, __ATOMIC_RELAXED);
      amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
    } else {
      if (this->kt_rec_ && ith == 0 && tp_part_idx < 2) __atomic_fetch_add(&this->kt_rec_->n_avx[tp_part_idx], 1, __ATOMIC_RELAXED);
      amx::vec_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
    }'''
    return rep(s, old, new)


def patch_py(s):
    old = '''            _ci.go_on_stream(self._cf_pending_stream, _slot)
            self._cf_last_slot = _slot
'''
    new = '''            _evt_path = _os_cf.environ.get("KT_EVT")   # IDE_075: slot 맵 (effective_from = 기록 ns; go 는 이 뒤에 GPU 에서 발생)
            if _evt_path:
                import time as _t_ev
                with open(_evt_path + ".map", "a") as _f_ev: _f_ev.write(f"{_slot},{self.layer_idx},{int(output_cpu[next_slot].shape[0])},{int(torch.cuda.is_current_stream_capturing())},{_t_ev.time_ns()}\\n")
            _ci.go_on_stream(self._cf_pending_stream, _slot)
            self._cf_last_slot = _slot
'''
    return rep(s, old, new)


def restore_orig():
    for f in FILES:
        b = f"{BK}/{os.path.basename(f)}.orig"
        if os.path.exists(b): shutil.copy(b, f)
        else: shutil.copy(f, b)   # v1 이 건드리지 않은 파일(task_queue.*, moe.hpp)은 현재 상태를 .orig 로 보존
    if os.path.exists(HDR): os.remove(HDR)


def apply():
    os.makedirs(BK, exist_ok=True); restore_orig()
    for f in FILES: assert "IDE_074" not in open(f).read() and "IDE_075" not in open(f).read(), f
    open(HDR, "w").write(KT_EVT_H)
    for f, fn in ((F_TQH, patch_tqh), (F_TQC, patch_tqc), (F_CI, patch_cpuinfer), (F_TP, patch_moetp), (F_MB, patch_moebase), (F_MOE, patch_moe), (PY, patch_py)):
        s = fn(open(f).read()); open(f, "w").write(s); print("patched", os.path.basename(f), sha(f)[:12])


def revert(): restore_orig(); print("reverted to .orig")
def status():
    for f in FILES: t = open(f).read(); print(("v2 " if MARK in t else ("v1 " if "IDE_074" in t else "orig")), sha(f)[:12], f)
    print("hdr", os.path.exists(HDR))


if __name__ == "__main__": {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()
