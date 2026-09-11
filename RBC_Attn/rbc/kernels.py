"""Triton GPU 실행기 — multi-block attention + partial merge.

설계 §2, §4:
  - 한 CTA 가 하나의 task(= query 묶음 × KV block 묶음)를 맡는다.
  - CTA 안에서 여러 KV block 을 순회하며 FP32 online softmax 상태 (u, m, l) 를 유지한다.
  - **작업이 끝날 때만** partial 을 global memory 에 쓴다. (KV-outer 는 block 마다 쓴다)
  - 실제 sparse edge 가 아닌 (query, block) 짝은 플래너가 준 비트마스크로 제외한다.
  - 절대 위치 causal mask 를 원소 단위로 적용한다.

구현은 `tl.load` / `tl.dot` 기반이다. 수동 TMA 나 warp specialization 은 쓰지 않았다.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

NEG_INF = tl.constexpr(float("-inf"))   # Triton 은 전역을 constexpr 로만 허용한다


@triton.jit
def _rbc_main_kernel(
    Q, K, V, OUT_U, OUT_M, OUT_L,
    task_qptr, task_q, task_kptr, task_k, task_kmask,
    q_positions, scale,
    stride_qn, stride_qg,
    stride_kn, stride_vn,
    G: tl.constexpr, D: tl.constexpr, BK: tl.constexpr, NK: tl.constexpr,
    CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """task 하나(의 query 타일 하나)를 CTA 하나가 처리한다."""
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

    offs_m = slot0 + tl.arange(0, BLOCK_M)            # task 내 query 슬롯 번호
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

    offs_bk = tl.arange(0, BK)

    # --- 이 CTA 의 KV block 들을 순회하며 상태를 누적 (알고리즘의 핵심) ---
    for ki in range(0, n_k):
        b = tl.load(task_k + k_lo + ki)
        kmask = tl.load(task_kmask + k_lo + ki)        # int64 비트마스크
        kv_pos = b * BK + offs_bk
        kv_valid = kv_pos < NK

        kblk = tl.load(K + kv_pos[:, None] * stride_kn + offs_d[None, :],
                       mask=kv_valid[:, None] & valid_d[None, :], other=0.0)
        vblk = tl.load(V + kv_pos[:, None] * stride_vn + offs_d[None, :],
                       mask=kv_valid[:, None] & valid_d[None, :], other=0.0)

        s = tl.dot(q, tl.trans(kblk.to(tl.float32))) * scale     # [BLOCK_M*G, BK]

        # 이 block 이 실제로 연결된 query 슬롯만 남긴다 (병합 직사각형의 빈 칸 제외)
        has_edge = ((kmask >> offs_m.to(tl.int64)) & 1) == 1
        edge_f = tl.reshape(tl.broadcast_to(has_edge[:, None], (BLOCK_M, G)), (BLOCK_M * G,))

        keep = kv_valid[None, :] & edge_f[:, None] & rowv[:, None]
        if CAUSAL:
            keep = keep & (kv_pos[None, :] <= qpos_f[:, None])
        s = tl.where(keep, s, NEG_INF)

        # --- online softmax 갱신 (FP32 상태를 레지스터에 유지) ---
        blk_max = tl.max(s, 1)
        m_new = tl.maximum(m_i, blk_max)
        m_safe = tl.where(m_new == NEG_INF, 0.0, m_new)
        p = tl.where(keep, tl.exp(s - m_safe[:, None]), 0.0)
        alpha = tl.where(m_i == NEG_INF, 0.0, tl.exp(m_i - m_safe))
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vblk.dtype), vblk).to(tl.float32)
        m_i = m_new

    # --- 작업이 끝날 때만 partial 기록 ---
    slot = q_lo + offs_m
    tl.store(OUT_U + (slot[:, None, None] * G + offs_g[None, :, None]) * D + offs_d[None, None, :],
             tl.reshape(acc, (BLOCK_M, G, BLOCK_D)),
             mask=valid_m[:, None, None] & valid_d[None, None, :])
    tl.store(OUT_M + slot[:, None] * G + offs_g[None, :],
             tl.reshape(m_i, (BLOCK_M, G)), mask=valid_m[:, None])
    tl.store(OUT_L + slot[:, None] * G + offs_g[None, :],
             tl.reshape(l_i, (BLOCK_M, G)), mask=valid_m[:, None])


@triton.jit
def _rbc_merge_kernel(
    OUT, LSE, U, M, L, red_ptr, red_slot,
    G: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """query 하나의 partial 상태들을 결합한다 (설계 §2.2 수식).

    m = max m_r,  l = sum e^{m_r-m} l_r,  u = sum e^{m_r-m} u_r,  O = u/l
    """
    qi = tl.program_id(0)
    g = tl.program_id(1)
    r0 = tl.load(red_ptr + qi)
    r1 = tl.load(red_ptr + qi + 1)
    offs_d = tl.arange(0, BLOCK_D)
    valid_d = offs_d < D

    # 스칼라 상태로 유지한다 (블록 모양이면 스칼라 포인터에 store 할 수 없다)
    m_acc = NEG_INF.value
    for r in range(r0, r1):
        slot = tl.load(red_slot + r)
        m_acc = tl.maximum(m_acc, tl.load(M + slot * G + g))

    m_safe = tl.where(m_acc == NEG_INF, 0.0, m_acc)
    l_acc = 0.0
    u_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for r in range(r0, r1):
        slot = tl.load(red_slot + r)
        m_r = tl.load(M + slot * G + g)
        l_r = tl.load(L + slot * G + g)
        w = tl.where(m_r == NEG_INF, 0.0, tl.exp(m_r - m_safe))
        l_acc += w * l_r
        u_acc += w * tl.load(U + (slot * G + g) * D + offs_d, mask=valid_d, other=0.0)

    pos = l_acc > 0
    out = tl.where(pos, u_acc / tl.where(pos, l_acc, 1.0), 0.0)
    tl.store(OUT + (qi * G + g) * D + offs_d, out, mask=valid_d)
    tl.store(LSE + qi * G + g,
             tl.where(pos, m_safe + tl.log(tl.where(pos, l_acc, 1.0)), NEG_INF))


def run_plan(dplan, q, k, v, q_positions, *, bk, causal, scale,
             block_m=None, num_warps=4, num_stages=2, workspace=None):
    """계획 하나를 실행해 (out, lse) 를 돌려준다.

    dplan: `runtime.DevicePlan` (descriptor 가 이미 GPU 에 올라간 상태)
    q: [Nq, G, D] (단일 KV head), k/v: [Nk, D]. 전부 CUDA tensor.
    """
    nq, G, D = q.shape
    NK = k.shape[0]
    dev = q.device
    n_slots = dplan.n_slots

    if block_m is None:
        block_m = dplan.block_m
    block_d = triton.next_power_of_2(D)

    if workspace is None:
        U = torch.empty((n_slots, G, D), device=dev, dtype=torch.float32)
        Mst = torch.empty((n_slots, G), device=dev, dtype=torch.float32)
        Lst = torch.empty((n_slots, G), device=dev, dtype=torch.float32)
        out = torch.empty((nq, G, D), device=dev, dtype=torch.float32)
        lse = torch.empty((nq, G), device=dev, dtype=torch.float32)
    else:
        U, Mst, Lst, out, lse = workspace.get(n_slots, nq, G, D)

    grid = (dplan.n_tasks, dplan.max_qtiles)
    _rbc_main_kernel[grid](
        q, k, v, U, Mst, Lst,
        dplan.task_qptr, dplan.task_q, dplan.task_kptr, dplan.task_k, dplan.task_kmask,
        q_positions, scale,
        q.stride(0), q.stride(1), k.stride(0), v.stride(0),
        G=G, D=D, BK=bk, NK=NK, CAUSAL=causal,
        BLOCK_M=block_m, BLOCK_D=block_d,
        num_warps=num_warps, num_stages=num_stages,
    )
    _rbc_merge_kernel[(nq, G)](
        out, lse, U, Mst, Lst, dplan.red_ptr, dplan.red_slot,
        G=G, D=D, BLOCK_D=block_d, num_warps=2,
    )
    return out, lse
