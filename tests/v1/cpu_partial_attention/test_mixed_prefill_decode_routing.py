# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TSK_002 §4.5c reload-fallback (X4/W1) — *진짜 동작* 통합 회귀.

`test_prefill_cold_mask.py` 가 helper 의 입출력 패턴만 검증한다면, 본
파일은 helper 의 결과가 `hot_cold_attention` 에 *실제로 전달*되었을 때
다음을 검증한다:

1. **mask 후 §4.5b decode-only gate 가 발동 안 함** — prefill+cold seq 의
   num_cold_blocks 가 0 으로 mask 되었으므로 RuntimeError 가 raise 안 됨.
2. **mixed batch (prefill+decode) 의 출력이 ground truth (full attention)
   와 일치** — prefill seq 는 hot only path, decode seq 는 IDE_006 cold
   path 로 routing 되어 LSE merge 결과가 분포 동등.
3. **decode seq 의 cold path 가 그대로 firing** — README §5.1 의 GPU/CPU
   overlap throughput 가치 보존 (즉 mask 가 *일부만* 적용되어 decode seq
   의 IDE_006 path 가 살아 있음).

Setup: phase3b 의 mixed batch 패턴을 가져오되, 일부 seq 의 q_len > 1
(prefill chunk) 으로 만들어 §4.5b gate 의 raise 시나리오를 재현. 그 다음
helper 로 mask 를 적용하고 hot_cold_attention 에 넘김.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="hot_cold_attention 의 hot path 는 CUDA + flash-attn 필요",
)


@cuda_required
def test_mixed_batch_mask_then_routing_matches_full_attention():
    """End-to-end: mixed batch (prefill+decode 모두 cold-bearing) → mask
    helper 적용 → hot_cold_attention 호출 → ground truth 와 일치.

    이 테스트가 통과하면 §4.5c 의 *알고리즘 흐름 전체* 가 동작:
      (a) helper 가 prefill seq 의 cold count 를 0 mask
      (b) §4.5b gate 가 mask 결과를 보고 raise 하지 않음
      (c) hot_cold_attention 이 prefill seq 는 hot only, decode seq 는
          cold path 로 routing
      (d) merge_attn_states 가 두 path 를 LSE 로 합산하여 ground truth
          와 분포 동등 결과 산출
    """
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )
    from vllm.v1.attention.backends.flash_attn import hot_cold_attention
    from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter
    from vllm.v1.worker.gpu_model_runner import _mask_prefill_cold_blocks

    device = torch.device("cuda")
    torch.manual_seed(31)

    # ---------- mixed batch 설정 ----------
    # seq 0: prefill chunk (q_len=4, cold=2)  ← §4.5b 가 raise 했던 패턴
    # seq 1: decode      (q_len=1, cold=3)  ← IDE_006 cold path 그대로 firing
    # seq 2: prefill chunk (q_len=2, cold=1)  ← raise 패턴
    # seq 3: decode      (q_len=1, cold=0)  ← cold 없음, hot only
    seq_lens = [128, 192, 96, 64]
    cold_per_seq_raw = [2, 3, 1, 0]
    query_lens = [4, 1, 2, 1]
    block_size = 16
    num_kv_heads = 4
    num_q_heads = 8
    head_dim = 128
    kv_dtype = torch.bfloat16
    atol = 5e-3

    batch_size = len(seq_lens)
    num_tokens = sum(query_lens)
    cu_query_lens = torch.tensor(
        [0] + list(np.cumsum(query_lens)),
        dtype=torch.int32, device=device,
    )
    seqused_k_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    max_seqlen_k = max(seq_lens)
    max_query_len = max(query_lens)

    # query_positions: token 별 absolute position
    query_positions = torch.empty(num_tokens, dtype=torch.int32, device="cpu")
    pos = 0
    for s, q_len in enumerate(query_lens):
        base = seq_lens[s] - q_len
        for j in range(q_len):
            query_positions[pos] = base + j
            pos += 1

    query = torch.randn(
        num_tokens, num_q_heads, head_dim, dtype=kv_dtype, device=device,
    )

    # ---------- KV cache (cold + hot 모두 GPU; vLLM reload 후 상태 흉내) ----------
    nblocks_per_seq = [(sl + block_size - 1) // block_size for sl in seq_lens]
    n_cold_total = sum(cold_per_seq_raw)
    n_hot_total = sum(nblocks_per_seq) - n_cold_total
    max_blocks = max(nblocks_per_seq)
    max_cold = max(max(cold_per_seq_raw), 1)

    total_blocks_gpu = n_cold_total + n_hot_total + 4
    key_cache_gpu = torch.randn(
        total_blocks_gpu, block_size, num_kv_heads, head_dim,
        dtype=kv_dtype, device=device,
    )
    value_cache_gpu = torch.randn_like(key_cache_gpu)

    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=block_size, dtype=kv_dtype,
    )
    cpu_canonical = torch.zeros(
        max(n_cold_total, 1), layout.page_size_bytes,
        dtype=torch.int8, device="cpu",
    )
    if n_cold_total > 0:
        adapter = KVViewAdapter(cpu_canonical, layout)
        adapter.k_view().copy_(key_cache_gpu[:n_cold_total].cpu())
        adapter.v_view().copy_(value_cache_gpu[:n_cold_total].cpu())

    block_table = torch.zeros(
        (batch_size, max_blocks), dtype=torch.int32, device=device,
    )
    cold_idx, hot_idx = 0, n_cold_total
    for s in range(batch_size):
        for j in range(cold_per_seq_raw[s]):
            block_table[s, j] = cold_idx
            cold_idx += 1
        n_hot_s = nblocks_per_seq[s] - cold_per_seq_raw[s]
        for j in range(n_hot_s):
            block_table[s, cold_per_seq_raw[s] + j] = hot_idx
            hot_idx += 1

    cold_block_ids_cpu = torch.zeros(
        (batch_size, max_cold), dtype=torch.int32, device="cpu",
    )
    cold_idx = 0
    for s in range(batch_size):
        for j in range(cold_per_seq_raw[s]):
            cold_block_ids_cpu[s, j] = cold_idx
            cold_idx += 1

    softmax_scale = 1.0 / (head_dim ** 0.5)
    fa_version = get_flash_attn_version()

    # ---------- Ground truth: 전체 KV 가 GPU 에 있을 때 full attention ----------
    ref_out, _ = flash_attn_varlen_func(
        q=query,
        k=key_cache_gpu, v=value_cache_gpu,
        cu_seqlens_q=cu_query_lens,
        seqused_k=seqused_k_t,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=True, window_size=[-1, -1],
        block_table=block_table,
        softcap=0.0,
        return_softmax_lse=True,
        fa_version=fa_version,
    )

    # ---------- §4.5c: helper 로 mask 적용 ----------
    num_cold_blocks_np = np.array(cold_per_seq_raw, dtype=np.int32)
    cu_q_cpu = np.array(
        [0] + list(np.cumsum(query_lens)), dtype=np.int32,
    )
    _mask_prefill_cold_blocks(num_cold_blocks_np, cu_q_cpu, batch_size)

    # mask 후: seq 0 (prefill, cold=2 → 0), seq 1 (decode, cold=3 → 3),
    #         seq 2 (prefill, cold=1 → 0), seq 3 (decode, cold=0 → 0)
    np.testing.assert_array_equal(
        num_cold_blocks_np, np.array([0, 3, 0, 0], dtype=np.int32),
        err_msg="mask 후 prefill seq 만 0, decode seq 는 보존되어야",
    )
    max_cold_after_mask = int(num_cold_blocks_np.max(initial=0))
    assert max_cold_after_mask == 3  # seq 1 의 decode cold count 보존

    num_cold_blocks_t = torch.from_numpy(num_cold_blocks_np).to(device=device)

    # ---------- mask 된 input 으로 hot_cold_attention 호출 ----------
    # §4.5b gate 가 mask 결과를 보고 RuntimeError 발동 안 함이 핵심.
    # 호출 자체가 raise 없이 끝나는 것이 (a)+(b) contract 검증.
    output = torch.empty_like(ref_out)
    hot_cold_attention(
        output=output,
        query=query,
        key_cache=key_cache_gpu, value_cache=value_cache_gpu,
        cu_query_lens=cu_query_lens, max_query_len=max_query_len,
        seqused_k=seqused_k_t, max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        sliding_window=(-1, -1), logits_soft_cap=0.0,
        block_table=block_table, block_size=block_size,
        num_cold_blocks=num_cold_blocks_t,
        max_num_cold_blocks=max_cold_after_mask,
        fa_version=fa_version, causal=True,
        cpu_kv_cache=[cpu_canonical], cold_kv_layout=layout,
        cold_block_ids=cold_block_ids_cpu,
        query_positions=query_positions,
    )
    torch.cuda.synchronize()

    # ---------- (c)+(d) ground truth 와 분포 동등 ----------
    torch.testing.assert_close(output, ref_out, atol=atol, rtol=atol)


@cuda_required
def test_all_prefill_batch_mask_falls_through_to_fast_path():
    """모든 seq 가 prefill+cold 인 batch — mask 후 max_cold == 0 → fast path.

    §4.5c 가 prefill-only batch 에서 IDE_006 path 를 *완전히 우회* 하여
    standard hot FA 로만 처리하는 것을 검증. cold path 안에서 어떤 작업도
    일어나지 않으므로 hang/raise 없이 결과가 바로 나옴.
    """
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )
    from vllm.v1.attention.backends.flash_attn import hot_cold_attention
    from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter
    from vllm.v1.worker.gpu_model_runner import _mask_prefill_cold_blocks

    device = torch.device("cuda")
    torch.manual_seed(41)

    # 모든 prefill, 모든 cold-bearing
    seq_lens = [256, 192]
    cold_per_seq_raw = [4, 3]
    query_lens = [8, 6]  # 모두 q_len > 1
    block_size = 16
    num_kv_heads = 4
    num_q_heads = 8
    head_dim = 128
    kv_dtype = torch.bfloat16
    atol = 5e-3

    batch_size = len(seq_lens)
    num_tokens = sum(query_lens)
    cu_query_lens = torch.tensor(
        [0] + list(np.cumsum(query_lens)),
        dtype=torch.int32, device=device,
    )
    seqused_k_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    max_seqlen_k = max(seq_lens)
    max_query_len = max(query_lens)

    query_positions = torch.empty(num_tokens, dtype=torch.int32, device="cpu")
    pos = 0
    for s, q_len in enumerate(query_lens):
        base = seq_lens[s] - q_len
        for j in range(q_len):
            query_positions[pos] = base + j
            pos += 1

    query = torch.randn(
        num_tokens, num_q_heads, head_dim, dtype=kv_dtype, device=device,
    )

    nblocks_per_seq = [(sl + block_size - 1) // block_size for sl in seq_lens]
    n_cold_total = sum(cold_per_seq_raw)
    n_hot_total = sum(nblocks_per_seq) - n_cold_total
    max_blocks = max(nblocks_per_seq)
    max_cold = max(cold_per_seq_raw)

    total_blocks_gpu = n_cold_total + n_hot_total + 4
    key_cache_gpu = torch.randn(
        total_blocks_gpu, block_size, num_kv_heads, head_dim,
        dtype=kv_dtype, device=device,
    )
    value_cache_gpu = torch.randn_like(key_cache_gpu)

    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=block_size, dtype=kv_dtype,
    )
    cpu_canonical = torch.zeros(
        n_cold_total, layout.page_size_bytes,
        dtype=torch.int8, device="cpu",
    )
    adapter = KVViewAdapter(cpu_canonical, layout)
    adapter.k_view().copy_(key_cache_gpu[:n_cold_total].cpu())
    adapter.v_view().copy_(value_cache_gpu[:n_cold_total].cpu())

    block_table = torch.zeros(
        (batch_size, max_blocks), dtype=torch.int32, device=device,
    )
    cold_idx, hot_idx = 0, n_cold_total
    for s in range(batch_size):
        for j in range(cold_per_seq_raw[s]):
            block_table[s, j] = cold_idx
            cold_idx += 1
        n_hot_s = nblocks_per_seq[s] - cold_per_seq_raw[s]
        for j in range(n_hot_s):
            block_table[s, cold_per_seq_raw[s] + j] = hot_idx
            hot_idx += 1

    cold_block_ids_cpu = torch.zeros(
        (batch_size, max_cold), dtype=torch.int32, device="cpu",
    )
    cold_idx = 0
    for s in range(batch_size):
        for j in range(cold_per_seq_raw[s]):
            cold_block_ids_cpu[s, j] = cold_idx
            cold_idx += 1

    softmax_scale = 1.0 / (head_dim ** 0.5)
    fa_version = get_flash_attn_version()

    # Ground truth
    ref_out, _ = flash_attn_varlen_func(
        q=query,
        k=key_cache_gpu, v=value_cache_gpu,
        cu_seqlens_q=cu_query_lens,
        seqused_k=seqused_k_t,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=True, window_size=[-1, -1],
        block_table=block_table,
        softcap=0.0,
        return_softmax_lse=True,
        fa_version=fa_version,
    )

    # mask 적용 — 모든 seq prefill 이라 모두 0 으로 force
    num_cold_blocks_np = np.array(cold_per_seq_raw, dtype=np.int32)
    cu_q_cpu = np.array(
        [0] + list(np.cumsum(query_lens)), dtype=np.int32,
    )
    _mask_prefill_cold_blocks(num_cold_blocks_np, cu_q_cpu, batch_size)
    np.testing.assert_array_equal(
        num_cold_blocks_np, np.zeros(batch_size, dtype=np.int32),
        err_msg="all-prefill batch — 모두 0 으로 mask",
    )
    max_cold_after_mask = int(num_cold_blocks_np.max(initial=0))
    assert max_cold_after_mask == 0

    num_cold_blocks_t = torch.from_numpy(num_cold_blocks_np).to(device=device)

    # max_num_cold_blocks==0 fast path 진입 — cold path 자체가 bypass.
    output = torch.empty_like(ref_out)
    hot_cold_attention(
        output=output,
        query=query,
        key_cache=key_cache_gpu, value_cache=value_cache_gpu,
        cu_query_lens=cu_query_lens, max_query_len=max_query_len,
        seqused_k=seqused_k_t, max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        sliding_window=(-1, -1), logits_soft_cap=0.0,
        block_table=block_table, block_size=block_size,
        num_cold_blocks=num_cold_blocks_t,
        max_num_cold_blocks=max_cold_after_mask,
        fa_version=fa_version, causal=True,
        cpu_kv_cache=[cpu_canonical], cold_kv_layout=layout,
        cold_block_ids=cold_block_ids_cpu,
        query_positions=query_positions,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_out, atol=atol, rtol=atol)


@cuda_required
def test_without_mask_prefill_with_cold_still_raises():
    """§4.5b defensive guard 검증 — mask 가 *적용 안 된 경우* (즉 §4.5c
    routing 이 깨진 bug 시나리오) hot_cold_attention 진입 시 §4.5b gate 가
    여전히 RuntimeError 로 막아준다.

    이게 §4.5b 와 §4.5c 의 layering 핵심 — §4.5c 가 정상 흐름에서 mask 로
    routing 하고, §4.5b 가 *defensive guard* 로 mask 누락 / 새 routing path
    추가 등 *코드 bug* 를 잡는다.
    """
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    from vllm.v1.attention.backends.flash_attn import hot_cold_attention
    from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter

    device = torch.device("cuda")
    torch.manual_seed(53)

    # mask 를 *일부러 적용 안 함* — prefill+cold 가 그대로 들어옴
    block_size = 16
    num_kv_heads = 4
    num_q_heads = 8
    head_dim = 128
    kv_dtype = torch.bfloat16
    seq_len = 192
    cold_blocks = 2
    q_len = 4  # prefill chunk

    cu_query_lens = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    seqused_k_t = torch.tensor([seq_len], dtype=torch.int32, device=device)
    query_positions = torch.tensor(
        list(range(seq_len - q_len, seq_len)),
        dtype=torch.int32, device="cpu",
    )
    query = torch.randn(
        q_len, num_q_heads, head_dim, dtype=kv_dtype, device=device,
    )
    nblocks = (seq_len + block_size - 1) // block_size
    total_blocks_gpu = nblocks + 4
    key_cache = torch.randn(
        total_blocks_gpu, block_size, num_kv_heads, head_dim,
        dtype=kv_dtype, device=device,
    )
    value_cache = torch.randn_like(key_cache)
    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=block_size, dtype=kv_dtype,
    )
    cpu_canonical = torch.zeros(
        cold_blocks, layout.page_size_bytes, dtype=torch.int8, device="cpu",
    )
    adapter = KVViewAdapter(cpu_canonical, layout)
    adapter.k_view().copy_(key_cache[:cold_blocks].cpu())
    adapter.v_view().copy_(value_cache[:cold_blocks].cpu())
    block_table = torch.zeros((1, nblocks), dtype=torch.int32, device=device)
    for j in range(nblocks):
        block_table[0, j] = j
    cold_block_ids = torch.zeros((1, cold_blocks), dtype=torch.int32, device="cpu")
    for j in range(cold_blocks):
        cold_block_ids[0, j] = j

    out = torch.empty(
        (q_len, num_q_heads, head_dim), dtype=kv_dtype, device=device,
    )
    # mask 안 함 — num_cold_blocks 그대로
    with pytest.raises(RuntimeError, match=r"prefill chunk with cold blocks"):
        hot_cold_attention(
            output=out, query=query,
            key_cache=key_cache, value_cache=value_cache,
            cu_query_lens=cu_query_lens, max_query_len=q_len,
            seqused_k=seqused_k_t, max_seqlen_k=seq_len,
            softmax_scale=1.0 / (head_dim ** 0.5),
            sliding_window=(-1, -1), logits_soft_cap=0.0,
            block_table=block_table, block_size=block_size,
            num_cold_blocks=torch.tensor(
                [cold_blocks], dtype=torch.int32, device=device,
            ),
            max_num_cold_blocks=cold_blocks,
            fa_version=get_flash_attn_version(), causal=True,
            cpu_kv_cache=[cpu_canonical], cold_kv_layout=layout,
            cold_block_ids=cold_block_ids, query_positions=query_positions,
        )
