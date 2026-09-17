#!/usr/bin/env python3
"""IDE_075 확장 — 기록기 v3 = v2 + (1) SPSC 대신 Node 에 rec 포인터 전달(경쟁 조건 제거: v2 에서 큐가 비어 있을 때 worker 가 push 전에 forward 에 진입해 54건 미기록),
(2) 스레드 이름(kt-cf-poll / kt-evt-flush / kt-task-worker) → TID 역할 식별, (3) expert 단위 rows 표본 링 (prefill 경로, (slot+epoch)%16==0 층화, .experts CSV).
사용: kt_evt_patch_v3.py apply | revert | status  (apply 는 .orig 로 되돌린 뒤 v2 패처를 재사용해 적용하고 v3 치환을 덧입힘)"""
import os, sys, importlib.util
spec = importlib.util.spec_from_file_location("v2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kt_evt_patch_v2.py")); v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
MARK3 = "IDE_075v3"


def rep(s, old, new, n=1):
    assert s.count(old) == n, (old[:70], s.count(old)); return s.replace(old, new)


HDR_V3 = v2.KT_EVT_H.replace("// IDE_075: 저간섭", "// IDE_075v3: 저간섭") + r'''
// ---- v3: Node 경유 rec 전달 (SPSC 대체) + expert 표본 링 + 스레드 이름 ----
inline thread_local KtEvtRec* kt_evt_next_rec = nullptr; inline thread_local int kt_evt_next_which = -1;   // enqueue 호출 스레드가 지정 (0 imm, 1 def)
inline thread_local KtEvtRec* kt_cur_rec = nullptr; inline thread_local int kt_cur_which = -1;              // task worker: 실행 중 task 의 rec
struct KtExpertRec { int slot; unsigned epoch; int numa; int n; int qlen; unsigned short expert[64]; unsigned short rows[64]; std::atomic<int> done; };
inline KtExpertRec* kt_expert_ring = nullptr; inline unsigned kt_expert_cap = 0; inline std::atomic<unsigned> kt_expert_n{0}; inline std::atomic<unsigned> kt_expert_dropped{0};
'''


def apply():
    v2.restore_orig()
    for f in v2.FILES: assert "IDE_074" not in open(f).read() and "IDE_075" not in open(f).read(), f
    open(v2.HDR, "w").write(HDR_V3)
    # --- task_queue.h: Node 에 rec/which
    s = v2.patch_tqh(open(v2.F_TQH).read())
    s = rep(s, "    unsigned long long seq = 0;   // IDE_075\n", "    unsigned long long seq = 0;   // IDE_075\n    KtEvtRec* rec = nullptr; int which = -1;   // IDE_075v3\n")
    open(v2.F_TQH, "w").write(s)
    # --- task_queue.cpp: enqueue 가 rec 저장, worker 가 kt_cur_rec 설정 + 스레드 이름
    s = v2.patch_tqc(open(v2.F_TQC).read())
    s = rep(s, "  Node* node = new Node(task); node->seq = seq;\n", "  Node* node = new Node(task); node->seq = seq; node->rec = kt_evt_next_rec; node->which = kt_evt_next_which;   // IDE_075v3\n")
    s = rep(s, "      kt_cur_task_seq = next->seq; kt_running_seq.store(next->seq, std::memory_order_relaxed);\n", "      kt_cur_task_seq = next->seq; kt_running_seq.store(next->seq, std::memory_order_relaxed); kt_cur_rec = next->rec; kt_cur_which = next->which;   // IDE_075v3\n")
    s = rep(s, "void TaskQueue::worker() {\n  Node* curr = head.load(std::memory_order_relaxed);\n", "void TaskQueue::worker() {\n  pthread_setname_np(pthread_self(), \"kt-task-worker\");   // IDE_075v3\n  Node* curr = head.load(std::memory_order_relaxed);\n")
    s = rep(s, "#include <unistd.h>\n#include <sys/syscall.h>\n", "#include <unistd.h>\n#include <sys/syscall.h>\n#include <pthread.h>\n")
    # task 종료 후 rec 해제
    s = rep(s, "          if (r) { r->t_exec_end = kt_evt_now(); static thread_local unsigned _tid = (unsigned)syscall(SYS_gettid); r->tid = _tid; }\n", "          if (r) { r->t_exec_end = kt_evt_now(); static thread_local unsigned _tid = (unsigned)syscall(SYS_gettid); r->tid = _tid; }\n          kt_cur_rec = nullptr; kt_cur_which = -1;   // IDE_075v3\n")
    open(v2.F_TQC, "w").write(s)
    # --- cpuinfer.h: poller 가 next_rec 지정 (push 제거), 스레드 이름, expert 링 할당·flush
    s = v2.patch_cpuinfer(open(v2.F_CI).read())
    s = rep(s, "if (p.imm_fn) { if (rec) { kt_evt_next_kind = 1; rec->t_imm_enq_b = kt_evt_now(); } p.imm_fn(p.imm_args); if (rec) { rec->t_imm_enq_a = kt_evt_now(); rec->imm_task_seq = kt_evt_last_seq; kt_evt_push(rec, kt_evt_last_seq, 0); } }",
            "if (p.imm_fn) { if (rec) { kt_evt_next_kind = 1; kt_evt_next_rec = rec; kt_evt_next_which = 0; rec->t_imm_enq_b = kt_evt_now(); } p.imm_fn(p.imm_args); if (rec) { rec->t_imm_enq_a = kt_evt_now(); rec->imm_task_seq = kt_evt_last_seq; } kt_evt_next_rec = nullptr; kt_evt_next_which = -1; }")
    s = rep(s, "if (run_def) { if (rec) { kt_evt_next_kind = 3; rec->t_def_enq_b = kt_evt_now(); } p.def_fn(p.def_args); n_def_run_.fetch_add(1, std::memory_order_relaxed); if (rec) { rec->t_def_enq_a = kt_evt_now(); rec->def_task_seq = kt_evt_last_seq; kt_evt_push(rec, kt_evt_last_seq, 1); } }",
            "if (run_def) { if (rec) { kt_evt_next_kind = 3; kt_evt_next_rec = rec; kt_evt_next_which = 1; rec->t_def_enq_b = kt_evt_now(); } p.def_fn(p.def_args); n_def_run_.fetch_add(1, std::memory_order_relaxed); if (rec) { rec->t_def_enq_a = kt_evt_now(); rec->def_task_seq = kt_evt_last_seq; } kt_evt_next_rec = nullptr; kt_evt_next_which = -1; }")
    s = rep(s, "  void poll_loop_() {\n    while (poller_run_.load(std::memory_order_acquire)) {\n", "  void poll_loop_() {\n    pthread_setname_np(pthread_self(), \"kt-cf-poll\");   // IDE_075v3\n    while (poller_run_.load(std::memory_order_acquire)) {\n")
    s = rep(s, "    evt_flusher_ = std::thread([this]() { while (evt_run_.load(std::memory_order_acquire)) { evt_flush_(false); std::this_thread::sleep_for(std::chrono::milliseconds(200)); } evt_flush_(true); });",
            "    kt_expert_ring = (KtExpertRec*)calloc((size_t)(1u << 16), sizeof(KtExpertRec)); kt_expert_cap = 1u << 16; std::string xp = std::string(evt_path_()) + \".experts\"; expert_fp_ = fopen(xp.c_str(), \"a\"); if (expert_fp_) { fprintf(expert_fp_, \"slot,epoch,numa,qlen,n,pairs(expert:rows;...)\\n\"); fflush(expert_fp_); }\n    evt_flusher_ = std::thread([this]() { pthread_setname_np(pthread_self(), \"kt-evt-flush\"); while (evt_run_.load(std::memory_order_acquire)) { evt_flush_(false); std::this_thread::sleep_for(std::chrono::milliseconds(200)); } evt_flush_(true); });")
    s = rep(s, "  std::thread evt_flusher_; std::atomic<bool> evt_run_{false}; FILE* evt_fp_ = nullptr; FILE* task_fp_ = nullptr;\n", "  std::thread evt_flusher_; std::atomic<bool> evt_run_{false}; FILE* evt_fp_ = nullptr; FILE* task_fp_ = nullptr; FILE* expert_fp_ = nullptr; unsigned expert_written_ = 0;   // v3\n")
    s = rep(s, "    if (final_) { fprintf(evt_fp_, \"#dropped=%u,written=%u,spsc_full=%u,spsc_stale=%u\\n\", evt_dropped_.load(), evt_written_, kt_evt_spsc_full.load(), kt_evt_spsc_stale.load()); fflush(evt_fp_); }",
            "    if (kt_expert_ring && expert_fp_) {\n      unsigned en = kt_expert_n.load(std::memory_order_acquire); if (en > kt_expert_cap) en = kt_expert_cap;\n      while (expert_written_ < en) { KtExpertRec& x = kt_expert_ring[expert_written_]; if (!x.done.load(std::memory_order_acquire)) { if (!final_) break; }\n        fprintf(expert_fp_, \"%d,%u,%d,%d,%d,\", x.slot, x.epoch, x.numa, x.qlen, x.n); for (int i = 0; i < x.n && i < 64; ++i) fprintf(expert_fp_, \"%u:%u;\", x.expert[i], x.rows[i]); fputc('\\n', expert_fp_); ++expert_written_; }\n      fflush(expert_fp_);\n    }\n    if (final_) { fprintf(evt_fp_, \"#dropped=%u,written=%u,expert_samples=%u,expert_dropped=%u\\n\", evt_dropped_.load(), evt_written_, expert_written_, kt_expert_dropped.load()); fflush(evt_fp_); }")
    s = rep(s, "if (evt_fp_) fclose(evt_fp_); if (task_fp_) fclose(task_fp_); }   // IDE_075\n", "if (evt_fp_) fclose(evt_fp_); if (task_fp_) fclose(task_fp_); if (expert_fp_) fclose(expert_fp_); }   // IDE_075\n")
    s = rep(s, "#include <vector>\n#include <string>\n", "#include <vector>\n#include <string>\n#include <pthread.h>\n") if "#include <vector>\n#include <string>\n" in s else rep(s, "#include <vector>\n", "#include <vector>\n#include <string>\n#include <pthread.h>\n", 1)
    open(v2.F_CI, "w").write(s)
    # --- moe-tp.hpp: SPSC pop 대신 kt_cur_rec
    s = v2.patch_moetp(open(v2.F_TP).read())
    s = rep(s, "    KtEvtPend _pd = kt_evt_pop_for(kt_cur_task_seq);   // IDE_075: 이 task 에 대응하는 레코드만 (OFF: rec==nullptr)\n    KtEvtRec* _r = _pd.rec; if (_r && _pd.which == 0) { _r->t_imm_fwd_entry = kt_evt_now(); }\n    KtEvtRec* _rd = (_r && _pd.which == 1) ? _r : nullptr; if (_rd) _rd->t_fwd_entry = kt_evt_now();",
            "    KtEvtPend _pd{kt_cur_rec, kt_cur_task_seq, kt_cur_which};   // IDE_075v3: Node 경유 (경쟁 없음; OFF: rec==nullptr)\n    KtEvtRec* _r = _pd.rec; if (_r && _pd.which == 0) { _r->t_imm_fwd_entry = kt_evt_now(); }\n    KtEvtRec* _rd = (_r && _pd.which == 1) ? _r : nullptr; if (_rd) _rd->t_fwd_entry = kt_evt_now();")
    open(v2.F_TP, "w").write(s)
    # --- moe_base.hpp: v2 + expert 표본 (prefill 경로, 히스토그램 뒤)
    s = v2.patch_moebase(open(v2.F_MB).read())
    old = "    if (kt_rec_ && tp_part_idx >= 0 && tp_part_idx < 2) {   // IDE_075: 단계 µs·규모를 레코드에 기록 (계산은 기존 타이머 그대로)"
    new = '''    if (kt_rec_ && tp_part_idx >= 0 && tp_part_idx < 2 && kt_expert_ring && ((kt_rec_->slot + (int)kt_rec_->epoch) % 16 == 0)) {   // IDE_075v3: expert 단위 rows 표본 (slot 별 1/16 층화)
      unsigned xi = kt_expert_n.fetch_add(1, std::memory_order_acq_rel);
      if (xi < kt_expert_cap) { KtExpertRec& x = kt_expert_ring[xi]; x.slot = kt_rec_->slot; x.epoch = kt_rec_->epoch; x.numa = tp_part_idx; x.qlen = qlen; int c = 0;
        for (int e = 0; e < config_.expert_num && c < 64; ++e) if (m_local_num_[e] > 0) { x.expert[c] = (unsigned short)e; x.rows[c] = (unsigned short)m_local_num_[e]; ++c; }
        x.n = c; x.done.store(1, std::memory_order_release); } else kt_expert_dropped.fetch_add(1, std::memory_order_relaxed);
    }
''' + old
    s = rep(s, old, new); open(v2.F_MB, "w").write(s)
    s = v2.patch_moe(open(v2.F_MOE).read()); open(v2.F_MOE, "w").write(s)
    s = v2.patch_py(open(v2.PY).read()); open(v2.PY, "w").write(s)
    for f in v2.FILES: print("patched", os.path.basename(f), v2.sha(f)[:12])


def status():
    for f in v2.FILES: t = open(f).read(); print(("v3 " if (MARK3 in t or ("IDE_075" in t and "kt_evt_next_rec" in open(v2.HDR).read())) else ("v2 " if "IDE_075" in t else "orig")), v2.sha(f)[:12], f)


if __name__ == "__main__": {"apply": apply, "revert": v2.revert, "status": status}[sys.argv[1]]()
