#!/usr/bin/env python3
"""IDE_074 M3/M4 — kt-kernel CPU-side 이벤트 타임스탬프 프로브 (컨테이너 sgl-kt 안에서 실행).
변경 (최소 diff, env KT_EVT=<파일경로> 미설정 시 기존 경로와 동일):
  1) cpu_backend/kt_evt.h (신규): KtEvtRec, kt_evt_now() (CLOCK_REALTIME ns), thread_local kt_evt_cur
  2) cpu_backend/cpuinfer.h: poll_loop_ 에서 go 관측·imm/def enqueue·done-setter 실행·deferred 작업 시작/종료 시각 기록,
     slot 별 epoch 카운터, 레코드 링(1<<20)·플러셔 스레드(200 ms 주기 CSV append)·누락 카운터
  3) operators/moe-tp.hpp: NUMA 서브풀 시작/종료 시각 (task worker 스레드가 kt_evt_cur 를 읽어 람다에 캡처)
  4) operators/amx/moe_base.hpp: forward_decode 의 게이트 없는 printf 를 KT_PHASE_PROF + 64회 1회 로 게이트 (CONFIG_DIFF 기록)
  5) site-packages kt_kernel/experts_base.py: arm/rearm 시 slot→(layer_idx, rows, capturing, ns) 를 KT_EVT.map 에 append
원본은 /sgl-workspace/ide074_backup/ 에 보존. 사용: kt_evt_patch.py apply | revert | status"""
import os, sys, shutil, hashlib, re
KT = "/sgl-workspace/ktransformers/kt-kernel"; PY = "/usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py"; BK = "/sgl-workspace/ide074_backup"
FILES = [f"{KT}/cpu_backend/cpuinfer.h", f"{KT}/operators/moe-tp.hpp", f"{KT}/operators/amx/moe_base.hpp", PY]
MARK = "IDE_074"
HDR = f"{KT}/cpu_backend/kt_evt.h"
KT_EVT_H = r'''// IDE_074: CPU-side per-packet event record (CLOCK_REALTIME ns; kineto GPU 트레이스와 같은 host 시계 도메인)
#pragma once
#include <time.h>
struct KtEvtRec {
  int slot; unsigned epoch; int qlen; int n_cold_ids;
  long long t_go, t_imm_enq, t_done_enq, t_def_enq, t_def_start, t_numa_start[2], t_numa_end[2], t_def_end, t_done_set;
  unsigned char imm_present, def_present, def_skipped, written;
};
inline long long kt_evt_now() { timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec; }
inline thread_local KtEvtRec* kt_evt_cur = nullptr;   // task worker 스레드: 현재 실행 중인 deferred 작업의 레코드
'''


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def patch_cpuinfer(s):
    assert MARK not in s
    s = s.replace('#include "task_queue.h"\n', '#include "task_queue.h"\n#include "kt_evt.h"   // IDE_074\n', 1)
    # 멤버·플러셔
    old = "  Packet packets_[kMaxPackets];\n  std::atomic<int> n_packets_{0};\n"
    new = old + r'''  // ---- IDE_074: per-packet CPU-side event timestamps. env KT_EVT=<csv path> 로만 활성 (미설정 시 rec==nullptr, 기존 경로 동일) ----
  static constexpr int kEvtCap = 1 << 20;
  KtEvtRec* evt_ = nullptr; unsigned evt_n_ = 0; std::atomic<unsigned> evt_dropped_{0}; unsigned evt_written_ = 0;
  unsigned slot_epoch_[kMaxPackets] = {0};
  std::thread evt_flusher_; std::atomic<bool> evt_run_{false}; FILE* evt_fp_ = nullptr;
  static const char* evt_path_() { static const char* v = std::getenv("KT_EVT"); return v; }
  static bool evt_complete_(const KtEvtRec& r) { return r.def_present ? (r.t_def_end != 0) : (r.t_done_set != 0); }
  void evt_write_(const KtEvtRec& r) {
    fprintf(evt_fp_, "%d,%u,%d,%d,%u,%u,%u,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld\n", r.slot, r.epoch, r.qlen, r.n_cold_ids, r.imm_present, r.def_present, r.def_skipped,
            r.t_go, r.t_imm_enq, r.t_done_enq, r.t_def_enq, r.t_def_start, r.t_numa_start[0], r.t_numa_end[0], r.t_numa_start[1], r.t_numa_end[1], r.t_def_end, r.t_done_set);
  }
  void evt_flush_(bool final_) {
    unsigned n = __atomic_load_n(&evt_n_, __ATOMIC_ACQUIRE); if (n > (unsigned)kEvtCap) n = kEvtCap;
    while (evt_written_ < n) { KtEvtRec& r = evt_[evt_written_]; if (!final_ && !evt_complete_(r)) break; evt_write_(r); ++evt_written_; }
    fflush(evt_fp_);
    if (final_) { fprintf(evt_fp_, "#dropped=%u,written=%u\n", evt_dropped_.load(), evt_written_); fflush(evt_fp_); }
  }
  void evt_ensure_() {
    if (evt_ != nullptr || evt_path_() == nullptr) return;
    evt_ = (KtEvtRec*)calloc((size_t)kEvtCap, sizeof(KtEvtRec));
    evt_fp_ = fopen(evt_path_(), "a");
    if (!evt_fp_) { fprintf(stderr, "[kt-evt] fopen %s failed\n", evt_path_()); fflush(stderr); free(evt_); evt_ = nullptr; return; }
    fprintf(evt_fp_, "slot,epoch,qlen,n_cold_ids,imm_present,def_present,def_skipped,t_go,t_imm_enq,t_done_enq,t_def_enq,t_def_start,t_numa0_start,t_numa0_end,t_numa1_start,t_numa1_end,t_def_end,t_done_set\n"); fflush(evt_fp_);
    evt_run_ = true;
    evt_flusher_ = std::thread([this]() { while (evt_run_.load(std::memory_order_acquire)) { evt_flush_(false); std::this_thread::sleep_for(std::chrono::milliseconds(200)); } evt_flush_(true); });
    fprintf(stderr, "[kt-evt] event log ready: %s (cap %d)\n", evt_path_(), kEvtCap); fflush(stderr);
  }
'''
    assert old in s; s = s.replace(old, new, 1)
    # ensure_cf_ 에서 evt_ensure_
    old = "    poller_run_ = true;\n    poller_ = std::thread([this]() { this->poll_loop_(); });\n"
    assert old in s; s = s.replace(old, "    evt_ensure_();   // IDE_074\n" + old, 1)
    # poll_loop_ 본문
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
        // IDE_074: 레코드 확보 (KT_EVT 미설정이면 evt_==nullptr → rec==nullptr, 아래 분기 전부 비활성)
        KtEvtRec* rec = nullptr;
        if (evt_) {
          unsigned i = evt_n_; __atomic_store_n(&evt_n_, i + 1, __ATOMIC_RELEASE);
          if (i < (unsigned)kEvtCap) {
            rec = &evt_[i]; rec->slot = slot; rec->epoch = ++slot_epoch_[slot]; rec->t_go = kt_evt_now();
            rec->imm_present = p.imm_fn != nullptr; rec->def_present = p.def_fn != nullptr;
            if (p.def_args) { FwdArgsView* a = (FwdArgsView*)p.def_args; rec->qlen = *(int*)a->qlen_ptr; int nc = 0; const int64_t* ids = (const int64_t*)a->expert_ids;
              for (int t = 0; t < rec->qlen * a->k; ++t) { int64_t e = ids[t]; if (e >= 0 && e < p.n_experts && (!p.gpu_mask || !p.gpu_mask[e])) ++nc; } rec->n_cold_ids = nc; }
          } else evt_dropped_.fetch_add(1, std::memory_order_relaxed);
        }
        // immediate task -> done signal -> deferred task  (== KT sync(allow_pending=1) semantics)
        if (p.imm_fn) p.imm_fn(p.imm_args);   // enqueues into task_queue_
        if (rec) rec->t_imm_enq = kt_evt_now();
        volatile unsigned int* d = done_host_ + slot;
        bool run_def = (p.def_fn != nullptr);
        if (run_def && skip_empty_def_on_() && def_is_empty_(p)) { run_def = false; n_def_skipped_.fetch_add(1, std::memory_order_relaxed); if (rec) rec->def_skipped = 1; }
        task_queue_->enqueue([d, rec]() { __atomic_store_n(d, 1u, __ATOMIC_RELEASE); if (rec) rec->t_done_set = kt_evt_now(); });
        if (rec) rec->t_done_enq = kt_evt_now();
        if (run_def) {
          if (rec) task_queue_->enqueue([rec]() { rec->t_def_start = kt_evt_now(); kt_evt_cur = rec; });
          p.def_fn(p.def_args); n_def_run_.fetch_add(1, std::memory_order_relaxed);
          if (rec) { task_queue_->enqueue([rec]() { rec->t_def_end = kt_evt_now(); kt_evt_cur = nullptr; }); rec->t_def_enq = kt_evt_now(); }
        } else if (rec) { rec->t_def_end = 0; rec->def_present = 0; }   // deferred 없음/skip → done_set 으로 완료 판정
'''
    assert old in s; s = s.replace(old, new, 1)
    # 소멸자: 플러셔 종료
    old = "  ~CPUInfer() {\n    printf(\"CPUInfer[0x%lx]: Goodbye\\n\", (intptr_t)this);\n"
    assert old in s; s = s.replace(old, old + "#ifndef KTRANSFORMERS_CPU_ONLY\n    if (evt_run_.load()) { evt_run_ = false; if (evt_flusher_.joinable()) evt_flusher_.join(); if (evt_fp_) fclose(evt_fp_); }   // IDE_074\n#endif\n", 1)
    return s


def patch_moetp(s):
    assert MARK not in s
    s = s.replace('#include "../cpu_backend/shared_mem_buffer.h"\n', '#include "../cpu_backend/shared_mem_buffer.h"\n#include "../cpu_backend/kt_evt.h"   // IDE_074\n', 1)
    old = '''    pool->dispense_backend()->do_numa_job([this, pool, qlen, k, expert_ids, input, weights](int numa_id) {
      tps[numa_id]->forward(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id]);
    });'''
    new = '''    KtEvtRec* _evt_r = kt_evt_cur;   // IDE_074: task worker 스레드의 현재 레코드 (KT_EVT 미설정이면 nullptr)
    pool->dispense_backend()->do_numa_job([this, pool, qlen, k, expert_ids, input, weights, _evt_r](int numa_id) {
      if (_evt_r && numa_id >= 0 && numa_id < 2) _evt_r->t_numa_start[numa_id] = kt_evt_now();
      tps[numa_id]->forward(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id]);
      if (_evt_r && numa_id >= 0 && numa_id < 2) _evt_r->t_numa_end[numa_id] = kt_evt_now();
    });'''
    assert old in s; s = s.replace(old, new, 1)
    return s


def patch_moebase(s):
    assert MARK not in s
    old = '''    auto end_time = std::chrono::high_resolution_clock::now();
    auto forward_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    printf(
        "Profiling Results (numa[%d]): activated_expert: %d, q_input: %ld us, "'''
    new = '''    auto end_time = std::chrono::high_resolution_clock::now();
    auto forward_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    static bool _pd_on = std::getenv("KT_PHASE_PROF") != nullptr; static thread_local unsigned _pd_cnt = 0;   // IDE_074: decode printf 게이트 (기존: 매 호출 무조건 출력)
    if (_pd_on && ((++_pd_cnt) & 63) == 0) printf(
        "Profiling Results (numa[%d]): activated_expert: %d, q_input: %ld us, "'''
    assert s.count(old) == 1, s.count(old); s = s.replace(old, new, 1)
    return s


def patch_py(s):
    assert MARK not in s
    old = '''            _ci.go_on_stream(self._cf_pending_stream, _slot)
            self._cf_last_slot = _slot
'''
    new = '''            _evt_path = _os_cf.environ.get("KT_EVT")   # IDE_074: slot → (layer, rows, capturing) 맵
            if _evt_path:
                import time as _t_ev
                with open(_evt_path + ".map", "a") as _f_ev: _f_ev.write(f"{_slot},{self.layer_idx},{int(output_cpu[next_slot].shape[0])},{int(torch.cuda.is_current_stream_capturing())},{_t_ev.time_ns()}\\n")
            _ci.go_on_stream(self._cf_pending_stream, _slot)
            self._cf_last_slot = _slot
'''
    assert s.count(old) == 1; s = s.replace(old, new, 1)
    return s


def apply():
    os.makedirs(BK, exist_ok=True)
    for f in FILES:
        b = f"{BK}/{os.path.basename(f)}.orig"
        if not os.path.exists(b): shutil.copy(f, b)
    open(HDR, "w").write(KT_EVT_H)
    for f, fn in ((FILES[0], patch_cpuinfer), (FILES[1], patch_moetp), (FILES[2], patch_moebase), (FILES[3], patch_py)):
        s = open(f).read(); open(f, "w").write(fn(s)); print("patched", f, sha(f)[:12])


def revert():
    for f in FILES:
        b = f"{BK}/{os.path.basename(f)}.orig"
        if os.path.exists(b): shutil.copy(b, f); print("reverted", f)
    if os.path.exists(HDR): os.remove(HDR)


def status():
    for f in FILES: print(("PATCHED " if MARK in open(f).read() else "orig    "), sha(f)[:12], f)
    print("hdr", os.path.exists(HDR))


if __name__ == "__main__": {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()
