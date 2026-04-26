# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TSK_002 §4.4 Phase 3a — hot_cold_attention 의 GPU hot path 정확성.

Phase 3a 는 cold path 를 미구현 (NotImplementedError). 따라서 본 unit test 는
다음 두 가지만 검증:

1. `max_num_cold_blocks == 0` (배치 전체에 cold 가 없는 degenerate 케이스):
   `hot_cold_attention` 의 출력이 동일 입력에 대한 직접 `flash_attn_varlen_func`
   호출 결과와 bit-identical 일치 (atol=0, rtol=0). 이게 깨지면 hot path 의
   block_table slicing 또는 seqused_k clipping 이 잘못된 것.

2. `max_num_cold_blocks > 0` (cold 가 있는 케이스): NotImplementedError 가
   Phase 3b 안내 메시지와 함께 raise.
"""

from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="hot_cold_attention의 hot path 는 CUDA + flash-attn 필요",
)


def _build_synthetic_batch(
    batch_size: int,
    seq_lens: list[int],
    block_size: int,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    kv_dtype: torch.dtype,
    device: torch.device,
):
    assert len(seq_lens) == batch_size
    max_seqlen_k = max(seq_lens)
    max_blocks_per_seq = (max_seqlen_k + block_size - 1) // block_size

    # 디코드 패턴: 시퀀스당 1 query token.
    query_lens = [1] * batch_size
    num_tokens = sum(query_lens)
    cu_query_lens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(query_lens), dim=0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    seqused_k_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    query = torch.randn(
        num_tokens, num_q_heads, head_dim, dtype=kv_dtype, device=device
    )

    # 여유 블록을 두어 block_table 안에 잘못된 인덱스가 들어가지 않게.
    total_blocks = batch_size * max_blocks_per_seq + 8
    key_cache = torch.randn(
        total_blocks, block_size, num_kv_heads, head_dim,
        dtype=kv_dtype, device=device,
    )
    value_cache = torch.randn_like(key_cache)

    block_table = torch.zeros(
        (batch_size, max_blocks_per_seq), dtype=torch.int32, device=device
    )
    block_id = 0
    for i, sl in enumerate(seq_lens):
        nblocks = (sl + block_size - 1) // block_size
        for j in range(nblocks):
            block_table[i, j] = block_id
            block_id += 1

    return dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        cu_query_lens=cu_query_lens,
        seqused_k=seqused_k_t,
        max_query_len=max(query_lens),
        max_seqlen_k=max_seqlen_k,
        block_table=block_table,
        block_size=block_size,
    )


@cuda_required
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float16])
def test_hot_cold_zero_matches_flash_attn_direct_call(kv_dtype):
    """`num_cold_blocks=zeros` + `max_num_cold_blocks=0` 일 때 hot_cold_attention
    의 출력이 동일 입력의 직접 `flash_attn_varlen_func` 호출과 atol=0, rtol=0
    으로 일치해야 합니다 — hot path 의 slicing / clipping 정확성 검증."""

    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )
    from vllm.v1.attention.backends.flash_attn import hot_cold_attention

    device = torch.device("cuda")
    torch.manual_seed(0)
    inputs = _build_synthetic_batch(
        batch_size=4,
        seq_lens=[128, 256, 192, 64],
        block_size=16,
        num_kv_heads=4,
        num_q_heads=32,
        head_dim=128,
        kv_dtype=kv_dtype,
        device=device,
    )
    softmax_scale = 1.0 / (inputs["query"].shape[-1] ** 0.5)
    fa_version = get_flash_attn_version()

    # Reference: 직접 flash_attn_varlen_func.
    ref_out, _ref_lse = flash_attn_varlen_func(
        q=inputs["query"],
        k=inputs["key_cache"],
        v=inputs["value_cache"],
        cu_seqlens_q=inputs["cu_query_lens"],
        seqused_k=inputs["seqused_k"],
        max_seqlen_q=inputs["max_query_len"],
        max_seqlen_k=inputs["max_seqlen_k"],
        softmax_scale=softmax_scale,
        causal=True,
        window_size=[-1, -1],
        block_table=inputs["block_table"],
        softcap=0.0,
        return_softmax_lse=True,
        fa_version=fa_version,
    )

    # 테스트: num_cold_blocks 전부 0.
    output = torch.empty_like(ref_out)
    num_cold_blocks = torch.zeros(4, dtype=torch.int32, device=device)
    hot_cold_attention(
        output=output,
        query=inputs["query"],
        key_cache=inputs["key_cache"],
        value_cache=inputs["value_cache"],
        cu_query_lens=inputs["cu_query_lens"],
        max_query_len=inputs["max_query_len"],
        seqused_k=inputs["seqused_k"],
        max_seqlen_k=inputs["max_seqlen_k"],
        softmax_scale=softmax_scale,
        sliding_window=(-1, -1),
        logits_soft_cap=0.0,
        block_table=inputs["block_table"],
        block_size=inputs["block_size"],
        num_cold_blocks=num_cold_blocks,
        max_num_cold_blocks=0,
        fa_version=fa_version,
        causal=True,
    )
    torch.testing.assert_close(output, ref_out, atol=0, rtol=0)


@cuda_required
def test_hot_cold_nonzero_raises_phase3b_notimplemented():
    """`max_num_cold_blocks > 0` 케이스는 Phase 3b 가 처리할 cold path 가 없으므로
    NotImplementedError 로 명시적 실패 — 안내 메시지에 'Phase 3b' 포함."""

    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    from vllm.v1.attention.backends.flash_attn import hot_cold_attention

    device = torch.device("cuda")
    torch.manual_seed(0)
    inputs = _build_synthetic_batch(
        batch_size=4,
        seq_lens=[128, 256, 192, 64],
        block_size=16,
        num_kv_heads=4,
        num_q_heads=32,
        head_dim=128,
        kv_dtype=torch.bfloat16,
        device=device,
    )
    softmax_scale = 1.0 / (inputs["query"].shape[-1] ** 0.5)

    # 첫 시퀀스만 cold 2 블록 (32 토큰), 나머지는 0.
    num_cold_blocks = torch.tensor([2, 0, 0, 0], dtype=torch.int32, device=device)
    output = torch.empty(
        inputs["query"].shape, dtype=torch.bfloat16, device=device
    )
    with pytest.raises(NotImplementedError, match="Phase 3b"):
        hot_cold_attention(
            output=output,
            query=inputs["query"],
            key_cache=inputs["key_cache"],
            value_cache=inputs["value_cache"],
            cu_query_lens=inputs["cu_query_lens"],
            max_query_len=inputs["max_query_len"],
            seqused_k=inputs["seqused_k"],
            max_seqlen_k=inputs["max_seqlen_k"],
            softmax_scale=softmax_scale,
            sliding_window=(-1, -1),
            logits_soft_cap=0.0,
            block_table=inputs["block_table"],
            block_size=inputs["block_size"],
            num_cold_blocks=num_cold_blocks,
            max_num_cold_blocks=2,
            fa_version=get_flash_attn_version(),
            causal=True,
        )
