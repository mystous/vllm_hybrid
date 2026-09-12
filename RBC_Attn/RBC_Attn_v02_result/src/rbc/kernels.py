"""Triton GPU 실행기 — multi-block attention + compact partial merge.

v0.1 대비 변경 (v0.2 지시서):
  P2  단일 소유(degree==1) 행은 main 이 정규화한 최종 출력·LSE 를 직접 기록한다.
      partial workspace 쓰기·읽기와 merge 실행을 그만큼 없앤다. Q-outer 에도 동등 적용.
  P3  (a) 입력 mask block 크기 BK=128 은 고정하고 GPU 내부 처리 타일 BN 만 {32,64,128}
          에서 고른다. mask 를 잘게 바꾸거나 선택 KV 를 줄이는 것이 아니다.
      (b) V 적재를 PV 직전으로 늦추는 variant (`late_v`).
      (c) task 의 모든 KV block 이 모든 실제 query 에 연결되면 FULL 경로로 membership
          비트 검사만 생략한다. padding·NK 경계·causal 은 계속 적용한다.

FP32 online softmax 상태를 CTA 안에 유지하고 작업 끝에만 기록하는 구조는 그대로다.
구현은 `tl.load`/`tl.dot` 기반이며 수동 TMA 나 warp specialization 은 쓰지 않는다.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

NEG_INF = tl.constexpr(float("-inf"))   # Triton 은 전역을 constexpr 로만 허용한다


@triton.jit
def _rbc_main_kernel(
    Q, K, V, OUT, LSE, OUT_U, OUT_M, OUT_L,
    task_qptr, task_q, task_kptr, task_k, task_kmask, slot_mode,
    q_positions, scale,
    stride_qn, stride_qg, stride_kn, stride_vn,
    G: tl.constexpr, D: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
    NK: tl.constexpr, CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr, FULL: tl.constexpr, LATE_V: tl.constexpr,
):
    tid = tl.program_id(0)
    qtile = tl.program_id(1)

    q_lo = tl.load(task_qptr + tid)
    q_hi = tl.load(task_qptr + tid + 1)
    k_lo = tl.load(task_kptr + tid)
    k_hi = tl.load(task_kptr + tid + 1)
    n_q = q_hi - q_lo
    n_k = k_hi - k_lo

    slot0 = qtile * BLOCK_M
    if slot0 >= n_q:
        return

    offs_m = slot0 + tl.arange(0, BLOCK_M)
    valid_m = offs_m < n_q
    q_idx = tl.load(task_q + q_lo + offs_m, mask=valid_m, other=0)

    offs_d = tl.arange(0, BLOCK_D)
    valid_d = offs_d < D
    offs_g = tl.arange(0, G)

    q = tl.load(Q + q_idx[:, None, None] * stride_qn
                + offs_g[None, :, None] * stride_qg + offs_d[None, None, :],
                mask=valid_m[:, None, None] & valid_d[None, None, :], other=0.0)
    q = tl.reshape(q, (BLOCK_M * G, BLOCK_D)).to(tl.float32)

    qpos = tl.load(q_positions + q_idx, mask=valid_m, other=0)
    qpos_f = tl.reshape(tl.broadcast_to(qpos[:, None], (BLOCK_M, G)), (BLOCK_M * G,))
    rowv = tl.reshape(tl.broadcast_to(valid_m[:, None], (BLOCK_M, G)), (BLOCK_M * G,))

    m_i = tl.full((BLOCK_M * G,), NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M * G,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M * G, BLOCK_D), dtype=tl.float32)

    offs_n = tl.arange(0, BN)
    N_SUB: tl.constexpr = BK // BN

    for ki in range(0, n_k):
        b = tl.load(task_k + k_lo + ki)
        if FULL:
            edge_f = rowv                       # P3-c: membership 검사만 생략
        else:
            kmask = tl.load(task_kmask + k_lo + ki)
            has_edge = ((kmask >> offs_m.to(tl.int64)) & 1) == 1
            edge_f = tl.reshape(
                tl.broadcast_to(has_edge[:, None], (BLOCK_M, G)), (BLOCK_M * G,)) & rowv

        # P3-a: 한 mask block(BK) 을 GPU 내부 타일 BN 으로 나눠 처리한다
        for sub in range(0, N_SUB):
            kv_pos = b * BK + sub * BN + offs_n
            kv_valid = kv_pos < NK

            kblk = tl.load(K + kv_pos[:, None] * stride_kn + offs_d[None, :],
                           mask=kv_valid[:, None] & valid_d[None, :], other=0.0)
            vblk = tl.zeros((BN, BLOCK_D), dtype=kblk.dtype)
            if not LATE_V:
                vblk = tl.load(V + kv_pos[:, None] * stride_vn + offs_d[None, :],
                               mask=kv_valid[:, None] & valid_d[None, :], other=0.0)

            s = tl.dot(q, tl.trans(kblk.to(tl.float32))) * scale

            keep = kv_valid[None, :] & edge_f[:, None]
            if CAUSAL:
                keep = keep & (kv_pos[None, :] <= qpos_f[:, None])
            s = tl.where(keep, s, NEG_INF)

            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == NEG_INF, 0.0, m_new)
            p = tl.where(keep, tl.exp(s - m_safe[:, None]), 0.0)
            alpha = tl.where(m_i == NEG_INF, 0.0, tl.exp(m_i - m_safe))
            l_i = l_i * alpha + tl.sum(p, 1)

            if LATE_V:                          # P3-b: PV 직전에 V 적재
                vblk = tl.load(V + kv_pos[:, None] * stride_vn + offs_d[None, :],
                               mask=kv_valid[:, None] & valid_d[None, :], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(vblk.dtype), vblk).to(tl.float32)
            m_i = m_new

    # --- P2: 소유권에 따라 최종 출력 또는 partial 로 기록 ---
    slot = q_lo + offs_m
    mode = tl.load(slot_mode + slot, mask=valid_m, other=0)
    dm = tl.broadcast_to((mode == 1)[:, None], (BLOCK_M, G))

    pos = l_i > 0
    inv = tl.where(pos, 1.0 / tl.where(pos, l_i, 1.0), 0.0)
    final = acc * inv[:, None]
    m_safe_o = tl.where(m_i == NEG_INF, 0.0, m_i)
    lse_v = tl.where(pos, m_safe_o + tl.log(tl.where(pos, l_i, 1.0)), NEG_INF)

    o_ptr = (OUT + (q_idx[:, None, None] * G + offs_g[None, :, None]) * D
             + offs_d[None, None, :])
    tl.store(o_ptr, tl.reshape(final, (BLOCK_M, G, BLOCK_D)),
             mask=valid_m[:, None, None] & dm[:, :, None] & valid_d[None, None, :])
    tl.store(LSE + q_idx[:, None] * G + offs_g[None, :],
             tl.reshape(lse_v, (BLOCK_M, G)), mask=valid_m[:, None] & dm)

    u_ptr = (OUT_U + (slot[:, None, None] * G + offs_g[None, :, None]) * D
             + offs_d[None, None, :])
    pm = valid_m[:, None] & (dm == 0)
    tl.store(u_ptr, tl.reshape(acc, (BLOCK_M, G, BLOCK_D)),
             mask=pm[:, :, None] & valid_d[None, None, :])
    tl.store(OUT_M + slot[:, None] * G + offs_g[None, :],
             tl.reshape(m_i, (BLOCK_M, G)), mask=pm)
    tl.store(OUT_L + slot[:, None] * G + offs_g[None, :],
             tl.reshape(l_i, (BLOCK_M, G)), mask=pm)


@triton.jit
def _rbc_merge_kernel(
    OUT, LSE, U, M, L, red_ptr, red_slot, multi_rows,
    G: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """degree>=2 인 행만 결합한다 (P2: compact merge)."""
    r = tl.program_id(0)
    g = tl.program_id(1)
    qi = tl.load(multi_rows + r)
    r0 = tl.load(red_ptr + r)
    r1 = tl.load(red_ptr + r + 1)
    offs_d = tl.arange(0, BLOCK_D)
    valid_d = offs_d < D

    m_acc = NEG_INF.value
    for t in range(r0, r1):
        slot = tl.load(red_slot + t)
        m_acc = tl.maximum(m_acc, tl.load(M + slot * G + g))
    m_safe = tl.where(m_acc == NEG_INF, 0.0, m_acc)

    l_acc = 0.0
    u_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for t in range(r0, r1):
        slot = tl.load(red_slot + t)
        m_r = tl.load(M + slot * G + g)
        w = tl.where(m_r == NEG_INF, 0.0, tl.exp(m_r - m_safe))
        l_acc += w * tl.load(L + slot * G + g)
        u_acc += w * tl.load(U + (slot * G + g) * D + offs_d, mask=valid_d, other=0.0)

    pos = l_acc > 0
    tl.store(OUT + (qi * G + g) * D + offs_d,
             tl.where(pos, u_acc / tl.where(pos, l_acc, 1.0), 0.0), mask=valid_d)
    tl.store(LSE + qi * G + g,
             tl.where(pos, m_safe + tl.log(tl.where(pos, l_acc, 1.0)), NEG_INF))


@triton.jit
def _rbc_empty_kernel(OUT, LSE, empty_rows, G: tl.constexpr, D: tl.constexpr,
                      BLOCK_D: tl.constexpr):
    """빈 행을 out=0, LSE=-inf 로 채운다 (버퍼 재사용 시 잔값 방지)."""
    r = tl.program_id(0)
    g = tl.program_id(1)
    qi = tl.load(empty_rows + r)
    offs_d = tl.arange(0, BLOCK_D)
    tl.store(OUT + (qi * G + g) * D + offs_d, tl.zeros((BLOCK_D,), dtype=tl.float32),
             mask=offs_d < D)
    tl.store(LSE + qi * G + g, NEG_INF.value)


def run_plan(dplan, q, k, v, q_positions, *, bk, causal, scale,
             block_m=None, bn=None, num_warps=4, num_stages=2, workspace=None,
             late_v=False, allow_full=True, return_counts=False,
             stages=("main", "merge", "empty")):
    """계획 하나를 실행해 (out, lse) 를 돌려준다.

    `stages` 는 E1 표의 main_us / merge_us 분리 측정을 위한 것이다. 기본값은 전 단계
    실행이고, 부분 실행 결과는 출력이 불완전하므로 시간 측정에만 쓴다.
    """
    nq, G, D = q.shape
    NK = k.shape[0]
    dev = q.device
    n_slots = max(dplan.n_slots, 1)

    block_m = dplan.block_m if block_m is None else int(block_m)
    bn = bk if bn is None else int(bn)
    if bk % bn != 0:
        raise ValueError(f"BK({bk}) 는 BN({bn}) 의 배수여야 한다")
    block_d = triton.next_power_of_2(D)
    full = bool(allow_full and dplan.full_membership)

    if workspace is None:
        U = torch.empty((n_slots, G, D), device=dev, dtype=torch.float32)
        Mst = torch.empty((n_slots, G), device=dev, dtype=torch.float32)
        Lst = torch.empty((n_slots, G), device=dev, dtype=torch.float32)
        out = torch.empty((nq, G, D), device=dev, dtype=torch.float32)
        lse = torch.empty((nq, G), device=dev, dtype=torch.float32)
    else:
        U, Mst, Lst, out, lse = workspace.get(n_slots, nq, G, D)

    counts = dict(main_launches=0, merge_launches=0, empty_launches=0,
                  partial_slots=dplan.n_partial_slots, full_membership=full, bn=bn)

    if dplan.n_tasks > 0 and "main" in stages:
        _rbc_main_kernel[(dplan.n_tasks, dplan.max_qtiles)](
            q, k, v, out, lse, U, Mst, Lst,
            dplan.task_qptr, dplan.task_q, dplan.task_kptr, dplan.task_k,
            dplan.task_kmask, dplan.slot_mode, q_positions, scale,
            q.stride(0), q.stride(1), k.stride(0), v.stride(0),
            G=G, D=D, BK=bk, BN=bn, NK=NK, CAUSAL=causal,
            BLOCK_M=block_m, BLOCK_D=block_d, FULL=full, LATE_V=bool(late_v),
            num_warps=num_warps, num_stages=num_stages)
        counts["main_launches"] = 1

    if dplan.n_multi_rows > 0 and "merge" in stages:
        _rbc_merge_kernel[(dplan.n_multi_rows, G)](
            out, lse, U, Mst, Lst, dplan.red_ptr, dplan.red_slot, dplan.multi_rows,
            G=G, D=D, BLOCK_D=block_d, num_warps=2)
        counts["merge_launches"] = 1

    if dplan.n_empty_rows > 0 and "empty" in stages:
        _rbc_empty_kernel[(dplan.n_empty_rows, G)](
            out, lse, dplan.empty_row_ids, G=G, D=D, BLOCK_D=block_d, num_warps=1)
        counts["empty_launches"] = 1

    if return_counts:
        return out, lse, counts
    return out, lse
