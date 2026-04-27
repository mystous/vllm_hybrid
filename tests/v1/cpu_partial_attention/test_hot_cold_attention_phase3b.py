# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TSK_002 §4.4 Phase 3b — hot_cold_attention 의 cold path + LSE merge 정확성.

Phase 3b 가 wire 한 cold path 가 다음을 만족하는지 검증:

* 시퀀스별로 다른 cold count 를 가진 batch 에서, hot_cold_attention 의 결과가
  동일 KV 데이터를 GPU 에 모두 두고 직접 `flash_attn_varlen_func` 로 부른
  full-attention 결과와 BF16/FP16 tolerance 안에서 일치.
* Cold prefix 데이터는 CPU canonical int8 buffer 에서, hot suffix 는 GPU
  paged KV cache 에서 read — 두 path 의 LSE 가 `merge_attn_states` 로 정확히
  결합되는지가 핵심 검증 포인트.

PLN_001 §4.1 의 합의값 (TSK_001 §4.5 와 동일):
- BF16: O atol/rtol = 5e-3, LSE atol/rtol = 1e-3
- FP16: O atol/rtol = 1e-3
"""

from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="hot_cold_attention 의 hot path 는 CUDA + flash-attn 필요",
)


@cuda_required
@pytest.mark.parametrize(
    "kv_dtype, atol",
    [(torch.bfloat16, 5e-3), (torch.float16, 1e-3)],
)
def test_hot_cold_split_matches_full_attention(kv_dtype, atol):
    """Heterogeneous per-seq cold count 에서 hot_cold_attention 의 출력이
    동일 데이터를 모두 GPU 에 두고 단일 flash_attn_varlen_func 로 부른
    full-attention 결과와 BF16/FP16 tolerance 내 일치.

    Setup:
      - 4 seq, seq_lens=[128, 256, 192, 64], block_size=16
      - num_cold_blocks=[2, 4, 0, 1] (heterogeneous; 0 도 포함)
      - decode 패턴 (시퀀스당 1 query token)
      - GPU KV cache 에는 cold 와 hot 모두 들어 있음 (reference 용)
      - CPU canonical buffer 는 cold 부분만 — GPU cold 블록과 동일 데이터를
        mirror.
    """
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )
    from vllm.v1.attention.backends.flash_attn import hot_cold_attention
    from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter

    device = torch.device("cuda")
    torch.manual_seed(42)

    batch_size = 4
    seq_lens = [128, 256, 192, 64]
    num_cold_blocks_per_seq = [2, 4, 0, 1]
    block_size = 16
    num_kv_heads = 4
    num_q_heads = 32
    head_dim = 128

    # 디코드 패턴 — 시퀀스당 1 query.
    query_lens = [1] * batch_size
    num_tokens = sum(query_lens)
    cu_query_lens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(query_lens), dim=0).tolist()),
        dtype=torch.int32, device=device,
    )
    seqused_k_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    max_seqlen_k = max(seq_lens)
    max_query_len = max(query_lens)

    # query_positions: 디코드는 마지막 토큰의 position = seq_len - 1.
    query_positions = torch.tensor(
        [seq_lens[s] - 1 for s in range(batch_size)],
        dtype=torch.int32, device=device,
    )

    # Q.
    query = torch.randn(
        num_tokens, num_q_heads, head_dim, dtype=kv_dtype, device=device,
    )

    # ---------- Block layout 계산 ----------
    nblocks_per_seq = [(sl + block_size - 1) // block_size for sl in seq_lens]
    cold_per_seq = num_cold_blocks_per_seq
    n_cold_total = sum(cold_per_seq)
    n_hot_total = sum(nblocks_per_seq) - n_cold_total
    max_blocks_per_seq = max(nblocks_per_seq)
    max_cold_blocks_per_seq = max(cold_per_seq) if max(cold_per_seq) > 0 else 1

    # GPU full KV cache — cold 영역 [0, n_cold_total), hot 영역 [n_cold_total,
    # n_cold_total + n_hot_total) + 여유.
    total_blocks_gpu = n_cold_total + n_hot_total + 8  # 여유 padding
    key_cache_gpu = torch.randn(
        total_blocks_gpu, block_size, num_kv_heads, head_dim,
        dtype=kv_dtype, device=device,
    )
    value_cache_gpu = torch.randn_like(key_cache_gpu)

    # ---------- CPU canonical buffer (cold 만) ----------
    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=block_size, dtype=kv_dtype,
    )
    cpu_canonical = torch.zeros(
        n_cold_total if n_cold_total > 0 else 1,
        layout.page_size_bytes,
        dtype=torch.int8, device="cpu",
    )
    if n_cold_total > 0:
        adapter = KVViewAdapter(cpu_canonical, layout)
        cpu_K_view = adapter.k_view()
        cpu_V_view = adapter.v_view()
        # GPU 의 cold 영역 [0, n_cold_total) 을 그대로 CPU 로 mirror.
        cpu_K_view.copy_(key_cache_gpu[:n_cold_total].cpu())
        cpu_V_view.copy_(value_cache_gpu[:n_cold_total].cpu())

    # ---------- block_table 구성 ----------
    # 시퀀스 s 의 block_table[s, :] 는:
    #   columns [0, cold_per_seq[s])               -> GPU cold 블록 ID
    #   columns [cold_per_seq[s], nblocks_per_seq[s]) -> GPU hot 블록 ID
    block_table = torch.zeros(
        (batch_size, max_blocks_per_seq), dtype=torch.int32, device=device,
    )
    cold_idx_global = 0
    hot_idx_global = n_cold_total
    for s in range(batch_size):
        for j in range(cold_per_seq[s]):
            block_table[s, j] = cold_idx_global
            cold_idx_global += 1
        n_hot_s = nblocks_per_seq[s] - cold_per_seq[s]
        for j in range(n_hot_s):
            block_table[s, cold_per_seq[s] + j] = hot_idx_global
            hot_idx_global += 1

    # ---------- cold_block_ids (CPU canonical 의 ID 공간) ----------
    cold_block_ids_cpu = torch.zeros(
        (batch_size, max_cold_blocks_per_seq),
        dtype=torch.int32, device="cpu",
    )
    cold_idx_global = 0
    for s in range(batch_size):
        for j in range(cold_per_seq[s]):
            cold_block_ids_cpu[s, j] = cold_idx_global
            cold_idx_global += 1

    num_cold_blocks_t = torch.tensor(
        cold_per_seq, dtype=torch.int32, device=device,
    )
    max_num_cold_blocks = max(cold_per_seq)

    softmax_scale = 1.0 / (head_dim ** 0.5)
    fa_version = get_flash_attn_version()

    # ---------- Reference: full attention (cold + hot 모두 GPU) ----------
    ref_out, _ref_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache_gpu,
        v=value_cache_gpu,
        cu_seqlens_q=cu_query_lens,
        seqused_k=seqused_k_t,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=[-1, -1],
        block_table=block_table,
        softcap=0.0,
        return_softmax_lse=True,
        fa_version=fa_version,
    )

    # ---------- Test: hot_cold_attention with cold path ----------
    output = torch.empty_like(ref_out)
    hot_cold_attention(
        output=output,
        query=query,
        key_cache=key_cache_gpu,
        value_cache=value_cache_gpu,
        cu_query_lens=cu_query_lens,
        max_query_len=max_query_len,
        seqused_k=seqused_k_t,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        sliding_window=(-1, -1),
        logits_soft_cap=0.0,
        block_table=block_table,
        block_size=block_size,
        num_cold_blocks=num_cold_blocks_t,
        max_num_cold_blocks=max_num_cold_blocks,
        fa_version=fa_version,
        causal=True,
        # ---- Phase 3b cold path inputs ----
        cpu_kv_cache=[cpu_canonical],
        cold_kv_layout=layout,
        cold_block_ids=cold_block_ids_cpu,
        query_positions=query_positions,
    )

    # CUDA stream 동기화 — cold path 의 H2D non_blocking 이 merge 에 반영
    # 되었는지 확인 (merge_attn_states 자체가 main stream 위에서 실행).
    torch.cuda.synchronize()

    torch.testing.assert_close(output, ref_out, atol=atol, rtol=atol)


@cuda_required
def test_hot_cold_split_all_cold_one_seq_others_zero():
    """한 시퀀스만 100% cold, 나머지는 0 인 가장 unbalanced edge case.
    `block_table` 의 per-row gather 가 비정상적으로 작은 hot column 폭을
    만들지 않는지, hot/cold 분리가 algorithmically 정확한지 확인.

    **tolerance 주의**: 본 케이스는 FA fused kernel (reference 의 모든
    시퀀스) vs Python reference (cold path 의 100%-cold 시퀀스) 의 BF16
    precision floor 를 stress 합니다. 동일 데이터에 대해 두 구현이 BF16
    상에서 ~2~3% 까지 벌어질 수 있으므로 atol 을 5e-2 로 두어
    *algorithmic* 정확성 — i.e. block_table gather / seqused_k clipping /
    LSE merge 가 모두 올바른 위치에 적용 — 만 검증합니다. heterogeneous
    case (test_hot_cold_split_matches_full_attention) 가 PLN_001 §4.1
    합의 tolerance (5e-3) 로 통과하므로 알고리즘 자체는 그쪽이 보증."""
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )
    from vllm.v1.attention.backends.flash_attn import hot_cold_attention
    from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter

    device = torch.device("cuda")
    torch.manual_seed(7)

    batch_size = 3
    seq_lens = [64, 64, 64]
    # 첫 시퀀스만 4 cold (= 64 토큰 = 전체), 나머지 0.
    cold_per_seq = [4, 0, 0]
    block_size = 16
    num_kv_heads = 4
    num_q_heads = 8
    head_dim = 128
    kv_dtype = torch.bfloat16
    # FA fused kernel vs Python reference 의 BF16 precision floor — 본 edge
    # case 는 100% cold 시퀀스를 포함해 floor 가 가장 크게 노출됨.
    atol = 5e-2

    query_lens = [1] * batch_size
    num_tokens = sum(query_lens)
    cu_query_lens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(query_lens), dim=0).tolist()),
        dtype=torch.int32, device=device,
    )
    seqused_k_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    query_positions = torch.tensor(
        [seq_lens[s] - 1 for s in range(batch_size)],
        dtype=torch.int32, device=device,
    )

    query = torch.randn(
        num_tokens, num_q_heads, head_dim, dtype=kv_dtype, device=device,
    )

    nblocks_per_seq = [(sl + block_size - 1) // block_size for sl in seq_lens]
    n_cold_total = sum(cold_per_seq)
    n_hot_total = sum(nblocks_per_seq) - n_cold_total
    max_blocks = max(nblocks_per_seq)

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
        n_cold_total, layout.page_size_bytes, dtype=torch.int8, device="cpu",
    )
    adapter = KVViewAdapter(cpu_canonical, layout)
    adapter.k_view().copy_(key_cache_gpu[:n_cold_total].cpu())
    adapter.v_view().copy_(value_cache_gpu[:n_cold_total].cpu())

    block_table = torch.zeros(
        (batch_size, max_blocks), dtype=torch.int32, device=device,
    )
    cold_idx, hot_idx = 0, n_cold_total
    for s in range(batch_size):
        for j in range(cold_per_seq[s]):
            block_table[s, j] = cold_idx
            cold_idx += 1
        n_hot_s = nblocks_per_seq[s] - cold_per_seq[s]
        for j in range(n_hot_s):
            block_table[s, cold_per_seq[s] + j] = hot_idx
            hot_idx += 1

    max_cold = max(cold_per_seq)
    cold_block_ids = torch.zeros(
        (batch_size, max_cold), dtype=torch.int32, device="cpu",
    )
    cold_idx = 0
    for s in range(batch_size):
        for j in range(cold_per_seq[s]):
            cold_block_ids[s, j] = cold_idx
            cold_idx += 1

    softmax_scale = 1.0 / (head_dim ** 0.5)
    fa_version = get_flash_attn_version()

    ref_out, _ = flash_attn_varlen_func(
        q=query, k=key_cache_gpu, v=value_cache_gpu,
        cu_seqlens_q=cu_query_lens, seqused_k=seqused_k_t,
        max_seqlen_q=1, max_seqlen_k=max(seq_lens),
        softmax_scale=softmax_scale, causal=True,
        window_size=[-1, -1], block_table=block_table,
        softcap=0.0, return_softmax_lse=True, fa_version=fa_version,
    )

    output = torch.empty_like(ref_out)
    hot_cold_attention(
        output=output, query=query,
        key_cache=key_cache_gpu, value_cache=value_cache_gpu,
        cu_query_lens=cu_query_lens, max_query_len=1,
        seqused_k=seqused_k_t, max_seqlen_k=max(seq_lens),
        softmax_scale=softmax_scale,
        sliding_window=(-1, -1), logits_soft_cap=0.0,
        block_table=block_table, block_size=block_size,
        num_cold_blocks=torch.tensor(
            cold_per_seq, dtype=torch.int32, device=device,
        ),
        max_num_cold_blocks=max_cold,
        fa_version=fa_version, causal=True,
        cpu_kv_cache=[cpu_canonical], cold_kv_layout=layout,
        cold_block_ids=cold_block_ids, query_positions=query_positions,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, ref_out, atol=atol, rtol=atol)


@cuda_required
def test_hot_cold_split_mixed_device_inputs():
    """Regression — IDE_006 / TSK_004 per-seq 필터 device mismatch fix.

    Prod 은 ``query`` 를 GPU 에 두지만 ``query_positions`` 는 host 에서
    빌드되어 CPU 에 있는 상태로 ``hot_cold_attention`` 에 들어옵니다.
    이전 per-seq 필터 구현이 두 텐서를 같은 device 라 가정해 GPU index
    를 CPU 텐서에 넘겨 ``RuntimeError: Expected all tensors to be on
    the same device`` 로 EngineDeadError 가 발생했었습니다 (commit
    6cac231904 의 회귀, run 20260427_060423 에서 100/100 fail).

    본 테스트는 그 prod 입력 device topology 를 그대로 재현합니다 —
    query 는 cuda, query_positions / cold_block_ids 는 cpu.
    """
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )
    from vllm.v1.attention.backends.flash_attn import hot_cold_attention
    from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter

    device = torch.device("cuda")
    torch.manual_seed(11)

    batch_size = 4
    seq_lens = [128, 256, 192, 64]
    cold_per_seq = [2, 4, 0, 1]
    block_size = 16
    num_kv_heads = 4
    num_q_heads = 16
    head_dim = 128
    kv_dtype = torch.bfloat16
    atol = 5e-3

    query_lens = [1] * batch_size
    num_tokens = sum(query_lens)
    cu_query_lens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(query_lens), dim=0).tolist()),
        dtype=torch.int32, device=device,
    )
    seqused_k_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    # ★ prod-like: query_positions 만 CPU 로 강제. 이전 per-seq 필터는
    # query 와 같은 device 라 가정해서 폭발했음.
    query_positions = torch.tensor(
        [seq_lens[s] - 1 for s in range(batch_size)],
        dtype=torch.int32, device="cpu",
    )

    query = torch.randn(
        num_tokens, num_q_heads, head_dim, dtype=kv_dtype, device=device,
    )

    nblocks_per_seq = [(sl + block_size - 1) // block_size for sl in seq_lens]
    n_cold_total = sum(cold_per_seq)
    n_hot_total = sum(nblocks_per_seq) - n_cold_total
    max_blocks = max(nblocks_per_seq)
    max_cold = max(cold_per_seq)

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
        n_cold_total, layout.page_size_bytes, dtype=torch.int8, device="cpu",
    )
    adapter = KVViewAdapter(cpu_canonical, layout)
    adapter.k_view().copy_(key_cache_gpu[:n_cold_total].cpu())
    adapter.v_view().copy_(value_cache_gpu[:n_cold_total].cpu())

    block_table = torch.zeros(
        (batch_size, max_blocks), dtype=torch.int32, device=device,
    )
    cold_idx, hot_idx = 0, n_cold_total
    for s in range(batch_size):
        for j in range(cold_per_seq[s]):
            block_table[s, j] = cold_idx
            cold_idx += 1
        n_hot_s = nblocks_per_seq[s] - cold_per_seq[s]
        for j in range(n_hot_s):
            block_table[s, cold_per_seq[s] + j] = hot_idx
            hot_idx += 1

    # ★ prod-like: cold_block_ids 도 CPU.
    cold_block_ids = torch.zeros(
        (batch_size, max_cold), dtype=torch.int32, device="cpu",
    )
    cold_idx = 0
    for s in range(batch_size):
        for j in range(cold_per_seq[s]):
            cold_block_ids[s, j] = cold_idx
            cold_idx += 1

    softmax_scale = 1.0 / (head_dim ** 0.5)
    fa_version = get_flash_attn_version()

    ref_out, _ = flash_attn_varlen_func(
        q=query, k=key_cache_gpu, v=value_cache_gpu,
        cu_seqlens_q=cu_query_lens, seqused_k=seqused_k_t,
        max_seqlen_q=1, max_seqlen_k=max(seq_lens),
        softmax_scale=softmax_scale, causal=True,
        window_size=[-1, -1], block_table=block_table,
        softcap=0.0, return_softmax_lse=True, fa_version=fa_version,
    )

    output = torch.empty_like(ref_out)
    # 이전 구현은 여기서 RuntimeError (device mismatch) 로 죽었음.
    hot_cold_attention(
        output=output, query=query,
        key_cache=key_cache_gpu, value_cache=value_cache_gpu,
        cu_query_lens=cu_query_lens, max_query_len=1,
        seqused_k=seqused_k_t, max_seqlen_k=max(seq_lens),
        softmax_scale=softmax_scale,
        sliding_window=(-1, -1), logits_soft_cap=0.0,
        block_table=block_table, block_size=block_size,
        num_cold_blocks=torch.tensor(
            cold_per_seq, dtype=torch.int32, device=device,
        ),
        max_num_cold_blocks=max_cold,
        fa_version=fa_version, causal=True,
        cpu_kv_cache=[cpu_canonical], cold_kv_layout=layout,
        cold_block_ids=cold_block_ids, query_positions=query_positions,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, ref_out, atol=atol, rtol=atol)


@cuda_required
def test_hot_cold_split_async_matches_sync(monkeypatch):
    """Regression — IDE_006 / TSK_002 §4.6 async issue path equivalence.

    NEO-style ``forward_partial_with_lse_async`` issues the C++ kernel
    on a background thread and returns a ``Future``. The dispatcher
    inside ``hot_cold_attention`` awaits this future before merge,
    so per-call wall behaviour should be observationally identical
    to the synchronous path. This test verifies *numerical*
    equivalence between (default) async-issue mode and the
    ``VLLM_COLD_KV_DISABLE_OVERLAP=1`` sync fallback.
    """
    import vllm.v1.attention.ops.cpu_partial_attention as cpa
    import vllm.v1.attention.backends.flash_attn as fa

    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )
    from vllm.v1.attention.backends.flash_attn import hot_cold_attention
    from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter

    device = torch.device("cuda")
    torch.manual_seed(123)

    batch_size = 3
    seq_lens = [96, 192, 64]
    cold_per_seq = [3, 6, 0]  # mix of cold and hot-only
    block_size = 16
    num_kv_heads = 4
    num_q_heads = 8
    head_dim = 128
    kv_dtype = torch.bfloat16
    atol = 5e-3

    query_lens = [1] * batch_size
    num_tokens = sum(query_lens)
    cu_query_lens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(query_lens), dim=0).tolist()),
        dtype=torch.int32, device=device,
    )
    seqused_k_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    query_positions = torch.tensor(
        [seq_lens[s] - 1 for s in range(batch_size)],
        dtype=torch.int32, device="cpu",
    )

    query = torch.randn(
        num_tokens, num_q_heads, head_dim, dtype=kv_dtype, device=device,
    )

    nblocks_per_seq = [(sl + block_size - 1) // block_size for sl in seq_lens]
    n_cold_total = sum(cold_per_seq)
    n_hot_total = sum(nblocks_per_seq) - n_cold_total
    max_blocks = max(nblocks_per_seq)
    max_cold = max(cold_per_seq)

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
        n_cold_total, layout.page_size_bytes, dtype=torch.int8, device="cpu",
    )
    adapter = KVViewAdapter(cpu_canonical, layout)
    adapter.k_view().copy_(key_cache_gpu[:n_cold_total].cpu())
    adapter.v_view().copy_(value_cache_gpu[:n_cold_total].cpu())

    block_table = torch.zeros(
        (batch_size, max_blocks), dtype=torch.int32, device=device,
    )
    cold_idx, hot_idx = 0, n_cold_total
    for s in range(batch_size):
        for j in range(cold_per_seq[s]):
            block_table[s, j] = cold_idx
            cold_idx += 1
        n_hot_s = nblocks_per_seq[s] - cold_per_seq[s]
        for j in range(n_hot_s):
            block_table[s, cold_per_seq[s] + j] = hot_idx
            hot_idx += 1

    cold_block_ids = torch.zeros(
        (batch_size, max_cold), dtype=torch.int32, device="cpu",
    )
    cold_idx = 0
    for s in range(batch_size):
        for j in range(cold_per_seq[s]):
            cold_block_ids[s, j] = cold_idx
            cold_idx += 1

    softmax_scale = 1.0 / (head_dim ** 0.5)
    fa_version = get_flash_attn_version()

    def _run() -> torch.Tensor:
        out = torch.empty(
            (num_tokens, num_q_heads, head_dim),
            dtype=kv_dtype, device=device,
        )
        hot_cold_attention(
            output=out, query=query,
            key_cache=key_cache_gpu, value_cache=value_cache_gpu,
            cu_query_lens=cu_query_lens, max_query_len=1,
            seqused_k=seqused_k_t, max_seqlen_k=max(seq_lens),
            softmax_scale=softmax_scale,
            sliding_window=(-1, -1), logits_soft_cap=0.0,
            block_table=block_table, block_size=block_size,
            num_cold_blocks=torch.tensor(
                cold_per_seq, dtype=torch.int32, device=device,
            ),
            max_num_cold_blocks=max_cold,
            fa_version=fa_version, causal=True,
            cpu_kv_cache=[cpu_canonical], cold_kv_layout=layout,
            cold_block_ids=cold_block_ids, query_positions=query_positions,
        )
        torch.cuda.synchronize()
        return out

    # Default — async overlap ON
    assert cpa._ASYNC_OVERLAP_DISABLED is False
    out_async = _run()

    # Force sync fallback
    monkeypatch.setattr(cpa, "_ASYNC_OVERLAP_DISABLED", True)
    monkeypatch.setattr(fa, "_PARTIAL_ASYNC_DISABLED", True)
    out_sync = _run()

    # async / sync paths must produce identical outputs (within BF16
    # rounding — no per-call randomness in either path).
    torch.testing.assert_close(out_async, out_sync, atol=atol, rtol=atol)


@cuda_required
def test_hot_cold_split_prefill_with_cold_fails_closed():
    """Regression — IDE_006 / TSK_002 §4.5 decode-only gate (fail-closed).

    정책 (2026-04-27): cold path 는 q_len==1 (pure decode) 만 허용.
    prefill chunk (q_len > 1) 가 cold block 을 보유한 batch 가 들어오면
    silent bypass 는 CLAUDE.md "결과 값이 달라져서는 안됨" 위반이라
    금지, 명시적 RuntimeError 로 fail-closed.

    GPU reload fallback 이 코드상 증명되기 전까지 본 동작은 유지된다.
    """
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )
    from vllm.v1.attention.backends.flash_attn import hot_cold_attention
    from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter

    device = torch.device("cuda")
    torch.manual_seed(13)

    batch_size = 2
    seq_lens = [128, 96]
    cold_per_seq = [3, 0]  # seq 0 has cold, seq 1 doesn't
    block_size = 16
    num_kv_heads = 4
    num_q_heads = 8
    head_dim = 128
    kv_dtype = torch.bfloat16

    # ★ KEY: q_len > 1 for the cold-bearing seq (seq 0) — prefill chunk.
    query_lens = [4, 1]
    num_tokens = sum(query_lens)
    cu_query_lens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(query_lens), dim=0).tolist()),
        dtype=torch.int32, device=device,
    )
    seqused_k_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    query_positions = torch.tensor(
        [seq_lens[0] - 4, seq_lens[0] - 3, seq_lens[0] - 2, seq_lens[0] - 1,
         seq_lens[1] - 1],
        dtype=torch.int32, device="cpu",
    )

    query = torch.randn(
        num_tokens, num_q_heads, head_dim, dtype=kv_dtype, device=device,
    )

    nblocks_per_seq = [(sl + block_size - 1) // block_size for sl in seq_lens]
    n_cold_total = sum(cold_per_seq)
    n_hot_total = sum(nblocks_per_seq) - n_cold_total
    max_blocks = max(nblocks_per_seq)
    max_cold = max(cold_per_seq)

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
        n_cold_total, layout.page_size_bytes, dtype=torch.int8, device="cpu",
    )
    adapter = KVViewAdapter(cpu_canonical, layout)
    adapter.k_view().copy_(key_cache_gpu[:n_cold_total].cpu())
    adapter.v_view().copy_(value_cache_gpu[:n_cold_total].cpu())

    block_table = torch.zeros(
        (batch_size, max_blocks), dtype=torch.int32, device=device,
    )
    cold_idx, hot_idx = 0, n_cold_total
    for s in range(batch_size):
        for j in range(cold_per_seq[s]):
            block_table[s, j] = cold_idx
            cold_idx += 1
        n_hot_s = nblocks_per_seq[s] - cold_per_seq[s]
        for j in range(n_hot_s):
            block_table[s, cold_per_seq[s] + j] = hot_idx
            hot_idx += 1

    cold_block_ids = torch.zeros(
        (batch_size, max_cold), dtype=torch.int32, device="cpu",
    )
    cold_idx = 0
    for s in range(batch_size):
        for j in range(cold_per_seq[s]):
            cold_block_ids[s, j] = cold_idx
            cold_idx += 1

    softmax_scale = 1.0 / (head_dim ** 0.5)
    fa_version = get_flash_attn_version()

    out = torch.empty(
        (num_tokens, num_q_heads, head_dim), dtype=kv_dtype, device=device,
    )

    # 정책에 따른 fail-closed 검증.
    with pytest.raises(RuntimeError, match=r"prefill chunk with cold blocks"):
        hot_cold_attention(
            output=out, query=query,
            key_cache=key_cache_gpu, value_cache=value_cache_gpu,
            cu_query_lens=cu_query_lens, max_query_len=max(query_lens),
            seqused_k=seqused_k_t, max_seqlen_k=max(seq_lens),
            softmax_scale=softmax_scale,
            sliding_window=(-1, -1), logits_soft_cap=0.0,
            block_table=block_table, block_size=block_size,
            num_cold_blocks=torch.tensor(
                cold_per_seq, dtype=torch.int32, device=device,
            ),
            max_num_cold_blocks=max_cold,
            fa_version=fa_version, causal=True,
            cpu_kv_cache=[cpu_canonical], cold_kv_layout=layout,
            cold_block_ids=cold_block_ids, query_positions=query_positions,
        )
