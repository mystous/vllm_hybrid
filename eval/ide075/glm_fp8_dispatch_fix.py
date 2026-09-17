#!/usr/bin/env python3
"""IDE_075 G00 — kt-kernel FP8/BF16 CPU 경로 0 출력 근본 원인 수정 (컨테이너 sgl-kt 안에서 실행: apply | revert | status).

원인 (bisect3 C2 + 코드 검토): operators/amx/moe_base.hpp 의 IDE_046-b/IDE_051 hunk 3곳이
    if constexpr (requires(... x_.from_mat_row/from_mat_block ...)) if (env_flag) { NEW } else ORIGINAL;
형태. C++ 문법상 `else` 는 안쪽 `if (env_flag)` 에 붙으므로 ORIGINAL 도 `if constexpr` 본문에 속하고,
BufferA 에 from_mat_row/from_mat_block 이 없는 커널(FP8·FP8PerChannel·BF16 = BufferABF16Impl) 에서는
조건이 거짓 → NEW 와 ORIGINAL 이 모두 컴파일에서 제거됨 → 입력 gather·A 양자화·down A 양자화가 전부 생략 → 출력 0.
INT4 계열(BufferAImpl 등) 은 조건이 참이라 정상 동작했으므로 Qwen 은 영향 없음.

수정: 3곳을 플래그 + 중괄호로 재구성 (env 미설정 기본 경로는 INT4 에서 바이트 동일 동작).
"""
import sys, hashlib, shutil, os
P = "/sgl-workspace/ktransformers/kt-kernel/operators/amx/moe_base.hpp"; BK = "/sgl-workspace/ide074_backup/moe_base.hpp.pre_fp8fix"

S1_OLD = """    const bool fuse_qin_ = kt_fuse_qin_() && qlen >= 10;
    if constexpr (requires(typename T::BufferA& x_, const ggml_bf16_t* p_) { x_.from_mat_row(1, p_, 0); }) if (fuse_qin_) {
      pool->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int j = 0; j < k; j++) {
          int e = expert_ids[i * k + j];
          if (config_.should_skip_expert(e)) continue;
          gate_up_ba_[e]->from_mat_row(m_local_num_[e], (const ggml_bf16_t*)input + i * config_.hidden_size, m_local_pos_[i][j]);
        }
      }, nullptr);
    } else
    direct_or_pool(qlen, [&](int i) {"""
S1_NEW = """    const bool fuse_qin_ = kt_fuse_qin_() && qlen >= 10;
    bool gathered_fused_ = false;   // IDE_075 G00 fix: if-constexpr 안의 dangling else 제거 (FP8/BF16 BufferA 에서 원본 경로가 사라지던 문제)
    if constexpr (requires(typename T::BufferA& x_, const ggml_bf16_t* p_) { x_.from_mat_row(1, p_, 0); }) { if (fuse_qin_) {
      pool->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int j = 0; j < k; j++) {
          int e = expert_ids[i * k + j];
          if (config_.should_skip_expert(e)) continue;
          gate_up_ba_[e]->from_mat_row(m_local_num_[e], (const ggml_bf16_t*)input + i * config_.hidden_size, m_local_pos_[i][j]);
        }
      }, nullptr);
      gathered_fused_ = true;
    } }
    if (!gathered_fused_)
    direct_or_pool(qlen, [&](int i) {"""
S2_OLD = """    if (fuse_qin_) {
      // IDE_051: gather 단계에서 이미 양자화됨
    } else
    if constexpr (requires(typename T::BufferA& x_, ggml_bf16_t* p_) { x_.from_mat_block(1, p_, 0); }) if (kt_qa_par_() && qlen >= 10) {
      const int mb = (max_local_num + T::M_STEP - 1) / T::M_STEP;
      pool->do_work_stealing_job(activated_expert * mb, nullptr, [this, mb](int t) {
        int expert_idx = m_expert_id_map_[t / mb]; int m_begin = (t % mb) * T::M_STEP;
        if (m_begin < m_local_num_[expert_idx])
          gate_up_ba_[expert_idx]->from_mat_block(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], m_begin);
      }, nullptr);
    } else
    direct_or_pool(activated_expert, [this](int task_id) {"""
S2_NEW = """    bool a_quantized_ = gathered_fused_;   // IDE_051: fused gather 가 실제 수행된 경우에만 양자화 생략
    if constexpr (requires(typename T::BufferA& x_, ggml_bf16_t* p_) { x_.from_mat_block(1, p_, 0); }) { if (!a_quantized_ && kt_qa_par_() && qlen >= 10) {
      const int mb = (max_local_num + T::M_STEP - 1) / T::M_STEP;
      pool->do_work_stealing_job(activated_expert * mb, nullptr, [this, mb](int t) {
        int expert_idx = m_expert_id_map_[t / mb]; int m_begin = (t % mb) * T::M_STEP;
        if (m_begin < m_local_num_[expert_idx])
          gate_up_ba_[expert_idx]->from_mat_block(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], m_begin);
      }, nullptr);
      a_quantized_ = true;
    } }
    if (!a_quantized_)
    direct_or_pool(activated_expert, [this](int task_id) {"""
S3_OLD = """    if constexpr (requires(typename T::BufferA& x_, ggml_bf16_t* p_) { x_.from_mat_block(1, p_, 0); }) if (kt_qa_par_() && qlen >= 10) {
      const int mb = (max_local_num + T::M_STEP - 1) / T::M_STEP;
      pool->do_work_stealing_job(activated_expert * mb, nullptr, [this, mb](int t) {
        int expert_idx = m_expert_id_map_[t / mb]; int m_begin = (t % mb) * T::M_STEP;
        if (m_begin < m_local_num_[expert_idx])
          down_ba_[expert_idx]->from_mat_block(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], m_begin);
      }, nullptr);
    } else
    pool->do_work_stealing_job("""
S3_NEW = """    bool down_quantized_ = false;   // IDE_075 G00 fix
    if constexpr (requires(typename T::BufferA& x_, ggml_bf16_t* p_) { x_.from_mat_block(1, p_, 0); }) { if (kt_qa_par_() && qlen >= 10) {
      const int mb = (max_local_num + T::M_STEP - 1) / T::M_STEP;
      pool->do_work_stealing_job(activated_expert * mb, nullptr, [this, mb](int t) {
        int expert_idx = m_expert_id_map_[t / mb]; int m_begin = (t % mb) * T::M_STEP;
        if (m_begin < m_local_num_[expert_idx])
          down_ba_[expert_idx]->from_mat_block(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], m_begin);
      }, nullptr);
      down_quantized_ = true;
    } }
    if (!down_quantized_)
    pool->do_work_stealing_job("""
PAIRS = [("S1", S1_OLD, S1_NEW), ("S2", S2_OLD, S2_NEW), ("S3", S3_OLD, S3_NEW)]


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def apply():
    s = open(P).read()
    if "IDE_075 G00 fix" in s: print("already applied", sha(P)); return
    if not os.path.exists(BK): shutil.copy(P, BK)
    for name, old, new in PAIRS:
        assert s.count(old) == 1, (name, s.count(old)); s = s.replace(old, new)
    open(P, "w").write(s); print("applied", sha(P), "backup", BK)


def revert():
    if os.path.exists(BK): shutil.copy(BK, P); print("reverted", sha(P))
    else: print("no backup")


def status():
    s = open(P).read(); print("moe_base.hpp", sha(P), "fix" if "IDE_075 G00 fix" in s else "unfixed", "dangling_sites", sum(1 for _, old, _ in PAIRS if old in s))


if __name__ == "__main__": {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()
