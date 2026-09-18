#!/usr/bin/env python3
"""IDE_076 C09 — A2: 같은 Cold job 안의 S1(gate/up A 양자화)~S5(down GEMM) 를 expert 단위 readiness 로 한 번의 do_work_stealing_job 에 공급.
컨테이너 sgl-kt 에서 apply | revert | status. 전제: kt_opt.h(cnt2) 존재.
계약(PLAN.md §7.2~7.4): 상위 FIFO·job 순서·done 게시·H2D·결합·expert/rows·NUMA-local shard·기존 pool 불변. 각 stage 의 계산 함수(from_mat, do_gate_up_gemm+to_mat, act, from_mat(down), do_down_gemm+to_mat) 호출 인자 동일 → 출력 bitwise 동일.
descriptor 순서(위상 정렬): QA(e…) → GU(e,ith,gate/up…) → ACT(e,ith…) → QD(e…) → DN(e,ith…). 의존 카운터: QA(e)=1 → GU(e,*) ; GU(e,ith)=2 → ACT(e,ith) ; ACT(e,*)=nth_gu → QD(e) ; QD(e)=1 → DN(e,*).
준비 안 된 descriptor 를 뽑은 worker 는 낮은 id 의 task(실행 중인 다른 worker) 를 spin 대기 — 의존이 항상 낮은 id 이므로 교착 없음. 새 스레드·NUMA 간 이동·다음 step 대기 없음.
KT_OPT_A2_ENABLE=1 일 때만; OFF 는 기존 코드 그대로."""
import sys, os, shutil, hashlib
ROOT = "/sgl-workspace/ktransformers/kt-kernel"; BK = "/sgl-workspace/ide076_backup"; MB = f"{ROOT}/operators/amx/moe_base.hpp"

INC_OLD = '#include "../../cpu_backend/kt_evt.h"   // IDE_075'
INC_NEW = '#include "../../cpu_backend/kt_evt.h"   // IDE_075\n#include "../../cpu_backend/kt_opt.h"   // IDE_076\n#include <atomic>\n#include <memory>'

FN_ANCHOR = "  void forward_prefill(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,"
FN_NEW = r'''  // ---- IDE_076 A2: S1~S5 expert-pipelined single dispatch (PLAN.md §7) ----
  std::unique_ptr<std::atomic<int>[]> a2_qa_, a2_gu_, a2_act_, a2_qd_; size_t a2_cap_gu_ = 0;
  void forward_prefill_a2_(int activated_expert, int qlen, bool a_quantized) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const int nth_gu = T::recommended_nth(config_.intermediate_size), nth_dn = T::recommended_nth(config_.hidden_size), E = config_.expert_num;
    if (a2_cap_gu_ < (size_t)E * nth_gu) { a2_qa_.reset(new std::atomic<int>[E]); a2_gu_.reset(new std::atomic<int>[(size_t)E * nth_gu]); a2_act_.reset(new std::atomic<int>[E]); a2_qd_.reset(new std::atomic<int>[E]); a2_cap_gu_ = (size_t)E * nth_gu; }
    for (int i = 0; i < activated_expert; i++) { int e = m_expert_id_map_[i]; a2_qa_[e].store(a_quantized ? 1 : 0, std::memory_order_relaxed); a2_act_[e].store(0, std::memory_order_relaxed); a2_qd_[e].store(0, std::memory_order_relaxed); for (int j = 0; j < nth_gu; j++) a2_gu_[(size_t)e * nth_gu + j].store(0, std::memory_order_relaxed); }
    const int nQA = a_quantized ? 0 : activated_expert, nGU = activated_expert * nth_gu * 2, nACT = activated_expert * nth_gu, nQD = activated_expert, nDN = activated_expert * nth_dn;
    const int total = nQA + nGU + nACT + nQD + nDN;
    if (total == 0) return;
    std::atomic_thread_fence(std::memory_order_release);
    pool->do_work_stealing_job(total, [](int _) { T::config(); }, [this, nQA, nGU, nACT, nQD, nth_gu, nth_dn, qlen, a_quantized](int t) {
      auto spin = [](std::atomic<int>& v, int target) { while (v.load(std::memory_order_acquire) < target) _mm_pause(); };
      if (t < nQA) { int e = m_expert_id_map_[t]; gate_up_ba_[e]->from_mat(m_local_num_[e], m_local_input_ptr_[e], 0, 1); a2_qa_[e].store(1, std::memory_order_release); return; }
      t -= nQA;
      if (t < nGU) {
        int task_id = t / 2; bool do_up = t % 2; int e = m_expert_id_map_[task_id / nth_gu]; int ith = task_id % nth_gu;
        if (!a_quantized) spin(a2_qa_[e], 1);
        derived()->do_gate_up_gemm(do_up, e, ith, nth_gu, qlen);
        if (do_up) up_bc_[e]->to_mat(m_local_num_[e], m_local_up_output_ptr_[e], ith, nth_gu); else gate_bc_[e]->to_mat(m_local_num_[e], m_local_gate_output_ptr_[e], ith, nth_gu);
        a2_gu_[(size_t)e * nth_gu + ith].fetch_add(1, std::memory_order_release); return;
      }
      t -= nGU;
      if (t < nACT) {
        int e = m_expert_id_map_[t / nth_gu]; int ith = t % nth_gu; spin(a2_gu_[(size_t)e * nth_gu + ith], 2);
        auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth_gu);
        for (int i = 0; i < m_local_num_[e]; i++) {
          ggml_bf16_t* gate_output_ptr = &m_local_gate_output_ptr_[e][i * config_.intermediate_size];
          ggml_bf16_t* up_output_ptr = &m_local_up_output_ptr_[e][i * config_.intermediate_size];
          ggml_bf16_t* destination_ptr = gate_output_ptr;
          for (int j = n_start; j < n_end; j += 32) {
            __m512 gate_val0, gate_val1, up_val0, up_val1;
            avx512_32xbf16_to_32xfp32((__m512i*)(gate_output_ptr + j), &gate_val0, &gate_val1);
            avx512_32xbf16_to_32xfp32((__m512i*)(up_output_ptr + j), &up_val0, &up_val1);
            __m512 result0 = amx::act_fn(gate_val0, up_val0, config_.swiglu_limit, config_.swiglu_alpha);
            __m512 result1 = amx::act_fn(gate_val1, up_val1, config_.swiglu_limit, config_.swiglu_alpha);
            avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i*)(destination_ptr + j));
          }
        }
        a2_act_[e].fetch_add(1, std::memory_order_release); return;
      }
      t -= nACT;
      if (t < nQD) { int e = m_expert_id_map_[t]; spin(a2_act_[e], nth_gu); down_ba_[e]->from_mat(m_local_num_[e], m_local_gate_output_ptr_[e], 0, 1); a2_qd_[e].store(1, std::memory_order_release); return; }
      t -= nQD;
      { int e = m_expert_id_map_[t / nth_dn]; int ith = t % nth_dn; spin(a2_qd_[e], 1); derived()->do_down_gemm(e, ith, nth_dn, qlen); down_bc_[e]->to_mat(m_local_num_[e], m_local_down_output_ptr_[e], ith, nth_dn); }
    }, nullptr);
  }

'''
S1_ANCHOR = "    bool a_quantized_ = gathered_fused_;   // IDE_051: fused gather 가 실제 수행된 경우에만 양자화 생략"
S1_NEW = "    if (kt_opt::a2_enabled()) {   // IDE_076 A2: S1~S5 fused dispatch (proof: bundled job 계수)\n      if (kt_opt::proof_on()) { auto& c = kt_opt::counters(); c.a2_eligible_jobs++; c.a2_bundled_jobs++; }\n      forward_prefill_a2_(activated_expert, qlen, gathered_fused_);\n    } else {\n" + S1_ANCHOR
S5_END = """          derived()->do_down_gemm(expert_idx, ith, nth, qlen);
          down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);
"""
S5_END_NEW = S5_END + "    }   // IDE_076 A2 else-end\n"


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def apply():
    b = f"{BK}/moe_base.hpp.pre_a2"
    if not os.path.exists(b): shutil.copy(MB, b)
    s = open(MB).read()
    if "forward_prefill_a2_" in s: print("already"); return
    assert s.count(INC_OLD) == 1 and s.count(FN_ANCHOR) == 1 and s.count(S1_ANCHOR) == 1 and s.count(S5_END) == 1, (s.count(INC_OLD), s.count(FN_ANCHOR), s.count(S1_ANCHOR), s.count(S5_END))
    s = s.replace(INC_OLD, INC_NEW).replace(FN_ANCHOR, FN_NEW + FN_ANCHOR).replace(S1_ANCHOR, S1_NEW)
    i = s.index(S5_END); s = s[:i] + S5_END_NEW + s[i + len(S5_END):]   # 첫 번째(prefill) 만
    open(MB, "w").write(s); print("applied", sha(MB))


def revert():
    b = f"{BK}/moe_base.hpp.pre_a2"
    if os.path.exists(b): shutil.copy(b, MB); print("reverted", sha(MB))


def status(): print("moe_base.hpp", sha(MB), "a2" if "forward_prefill_a2_" in open(MB).read() else "no-a2")


if __name__ == "__main__": {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()
