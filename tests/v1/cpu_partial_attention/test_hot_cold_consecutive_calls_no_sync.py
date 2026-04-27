# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TSK_002 §4.6 — _COLD_SCATTER_BUFS 의 cross-call CUDA stream race 회귀.

문제 모양:

    Layer N (Call A):
      cold_stream:   index_fill_(stale -inf) → index_copy_(A's data)
      default_str:                              wait_stream(cold) → merge_attn_states(buf)
                                                                          ↑ buf 에서 read 중
      function returns
    Layer N+1 (Call B):
      cold_stream:   index_fill_(A_dirty -inf) → index_copy_(B's data)
                          ↑ Call A 의 merge 가 default 에서 아직 실행 중일 수 있음 — race

기존 phase3b 회귀는 매 호출 끝에 ``torch.cuda.synchronize()`` 를 해서 이
race 를 가립니다. 본 파일은 *연속 두 호출 사이에 sync 없이* 두 결과가
모두 ground truth 와 일치해야 한다는 contract 를 검증합니다.

**중요한 환경 의존성 — dev 에서 race 가 reliably evoke 안 됨**:
이 머신의 PyTorch default stream 은 *CUDA legacy default stream*
(ID 0x0) 이라 다른 stream 들과 *implicit sync* 되고, RTX 3090 의
hot_cold_attention timing 도 race window 를 좁힙니다. 본 테스트는
main 작업을 non-default stream 으로 옮겨 implicit sync 를 우회하지만,
그래도 dev 에서는 race 가 deterministic 하게 evoke 가 되지 않을 수
있습니다. 그럼에도 다음 두 역할을 합니다:

1. *대표 시나리오 회귀 가드* — fix 가 이상한 형태로 깨지면 (예:
   wait_event 인자가 잘못 — 다른 stream 의 event 등) deterministic 하게
   fail.
2. PyTorch per-thread default stream 모드 / prod 의 다른 stream policy
   환경에서는 *현재 fix 가 없으면* 즉시 race 노출. 즉 fix 의 환경 횡단
   견고성을 prod 에서 자연스럽게 검증.

Race fix 의 메커니즘 자체는 ``test_after_first_call_records_merge_event``
가 white-box 로 검증 — fix 적용 후 cache entry 의 last_merge_event
슬롯에 실제 ``torch.cuda.Event`` 가 record 되었는지.

Race 를 evoke 하기 위한 setup:

* Main 작업을 non-default ``test_stream`` 에서 실행 (legacy implicit
  sync 우회).
* ``merge_attn_states`` 를 ``slow_merge`` 로 monkeypatch — 호출 전에
  ``torch.cuda._sleep(50M cycles ≈ 30 ms)`` 를 default stream 에
  enqueue 해 merge 가 늦게 끝나도록 함 (race window 확장).
* Call A 와 Call B 의 cache key 가 정확히 동일 — 같은
  ``_COLD_SCATTER_BUFS`` entry 를 hit.
* Call A 와 Call B 의 cold-bearing row 가 disjoint — Call B 의
  ``index_fill_(1, A_dirty, -inf)`` 가 Call A 의 cold_lse 를 덮으면
  Call A 의 merge 가 cold side 를 drop → mismatch.
* N=30 trial 반복.

Race fix 후 (cache entry 에 last_merge_event 저장 + 다음 cold-side write
직전 ``cold_stream.wait_event(prev_event)``) 에는 deterministic 하게
통과해야 합니다.
"""

from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="hot_cold_attention 의 hot path 는 CUDA + flash-attn 필요",
)


@pytest.fixture(autouse=True)
def _clear_scatter_cache():
    """각 테스트가 깨끗한 cache 상태에서 시작."""
    from vllm.v1.attention.backends.flash_attn import _COLD_SCATTER_BUFS

    saved = dict(_COLD_SCATTER_BUFS)
    _COLD_SCATTER_BUFS.clear()
    try:
        yield
    finally:
        _COLD_SCATTER_BUFS.clear()
        _COLD_SCATTER_BUFS.update(saved)


def _build_batch(*, cold_per_seq: list[int], device: torch.device, kv_dtype):
    """4-seq decode batch — cold_per_seq 만 다르고 나머지 dim 은 고정."""
    from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter

    seq_lens = [128, 256, 192, 64]
    batch_size = 4
    block_size = 16
    num_kv_heads = 4
    num_q_heads = 32
    head_dim = 128
    query_lens = [1] * batch_size
    num_tokens = sum(query_lens)

    cu_query_lens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(query_lens), dim=0).tolist()),
        dtype=torch.int32, device=device,
    )
    seqused_k_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    max_seqlen_k = max(seq_lens)
    max_query_len = max(query_lens)
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
    max_blocks_per_seq = max(nblocks_per_seq)
    max_cold_blocks_per_seq = max(max(cold_per_seq), 1)

    total_blocks_gpu = n_cold_total + n_hot_total + 8
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
        n_cold_total if n_cold_total > 0 else 1,
        layout.page_size_bytes,
        dtype=torch.int8, device="cpu",
    )
    if n_cold_total > 0:
        adapter = KVViewAdapter(cpu_canonical, layout)
        adapter.k_view().copy_(key_cache[:n_cold_total].cpu())
        adapter.v_view().copy_(value_cache[:n_cold_total].cpu())

    block_table = torch.zeros(
        (batch_size, max_blocks_per_seq), dtype=torch.int32, device=device,
    )
    cold_idx_g = 0
    hot_idx_g = n_cold_total
    for s in range(batch_size):
        for j in range(cold_per_seq[s]):
            block_table[s, j] = cold_idx_g
            cold_idx_g += 1
        n_hot_s = nblocks_per_seq[s] - cold_per_seq[s]
        for j in range(n_hot_s):
            block_table[s, cold_per_seq[s] + j] = hot_idx_g
            hot_idx_g += 1

    cold_block_ids = torch.zeros(
        (batch_size, max_cold_blocks_per_seq), dtype=torch.int32, device="cpu",
    )
    cold_idx_g = 0
    for s in range(batch_size):
        for j in range(cold_per_seq[s]):
            cold_block_ids[s, j] = cold_idx_g
            cold_idx_g += 1

    num_cold_blocks_t = torch.tensor(
        cold_per_seq, dtype=torch.int32, device=device,
    )

    return dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        cu_query_lens=cu_query_lens,
        max_query_len=max_query_len,
        seqused_k=seqused_k_t,
        max_seqlen_k=max_seqlen_k,
        block_table=block_table,
        block_size=block_size,
        num_cold_blocks=num_cold_blocks_t,
        max_num_cold_blocks=max(max(cold_per_seq), 1),
        cpu_kv_cache=[cpu_canonical],
        cold_kv_layout=layout,
        cold_block_ids=cold_block_ids,
        query_positions=query_positions,
        # for ground truth
        _ref_args=(query, key_cache, value_cache, cu_query_lens,
                   seqused_k_t, max_query_len, max_seqlen_k, block_table),
    )


@cuda_required
def test_two_consecutive_hot_cold_calls_no_sync_in_between(monkeypatch):
    """Call A (cold rows 0~1) 직후 Call B (cold rows 2~3) — 사이에 sync 없음.

    두 호출의 cache key 가 동일 (4 tokens × 32 heads × 128 dim, BF16) 하므로
    같은 ``_COLD_SCATTER_BUFS`` entry 를 hit. dirty_idx 가 서로 다르므로
    Call B 의 ``index_fill_(1, A_dirty, -inf)`` 가 Call A 의 default-stream
    merge 가 buf 를 다 읽기 전에 cold-stream 에서 실행되면 race.

    fix 전: 비결정적 mismatch (race 가 trial 마다 다른 timing 에 발생).
    fix 후: deterministic match — last_merge_event 가 Call B 의 cold write
            를 차단.
    """
    from vllm.v1.attention.backends import flash_attn as fa_mod
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )
    from vllm.v1.attention.backends.flash_attn import hot_cold_attention

    device = torch.device("cuda")
    torch.manual_seed(0)
    kv_dtype = torch.bfloat16

    # Race window 를 deterministic 하게 벌리기 위해 default-stream merge 를
    # monkeypatch — merge 직전에 ``torch.cuda._sleep`` 으로 default stream 을
    # 수십 ms 동안 stall 시킨다. 이러면 Call A 의 merge 가 buf 를 아직 안 읽었는데
    # Call B 의 cold-stream index_fill_ 이 같은 buf 를 -inf 로 덮어 쓴다.
    real_merge = fa_mod.merge_attn_states
    merge_call_count = {"n": 0}

    def slow_merge(*args, **kwargs):
        # default stream 에 50M cycles ≈ 30 ms (3090) sleep 을 enqueue
        merge_call_count["n"] += 1
        torch.cuda._sleep(50_000_000)
        return real_merge(*args, **kwargs)

    monkeypatch.setattr(fa_mod, "merge_attn_states", slow_merge)

    # ---- 두 배치 — cache key 동일, dirty_idx 만 다름 ----
    batch_a = _build_batch(
        cold_per_seq=[2, 4, 0, 0], device=device, kv_dtype=kv_dtype,
    )
    batch_b = _build_batch(
        cold_per_seq=[0, 0, 3, 1], device=device, kv_dtype=kv_dtype,
    )

    fa_version = get_flash_attn_version()
    softmax_scale = 1.0 / (128 ** 0.5)

    def _ground_truth(b):
        q, k, v, cu_q, sk, mq, mk, bt = b["_ref_args"]
        out, _ = flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_q,
            seqused_k=sk,
            max_seqlen_q=mq,
            max_seqlen_k=mk,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=[-1, -1],
            block_table=bt,
            softcap=0.0,
            return_softmax_lse=True,
            fa_version=fa_version,
        )
        return out

    truth_a = _ground_truth(batch_a)
    truth_b = _ground_truth(batch_b)
    torch.cuda.synchronize()

    # Non-default stream 위에서 main 작업 실행 — legacy default 의 implicit
    # sync 를 우회해야 race window 가 실제로 열린다. ``slow_merge`` 의 sleep
    # 이 default stream 에서 진행되는 동안 cold_stream 의 다음 호출
    # ``index_fill_`` 가 같은 buffer 를 -inf 로 덮어쓰는 race.
    test_stream = torch.cuda.Stream(device=device)

    n_trials = 30
    for trial in range(n_trials):
        out_a = torch.empty_like(truth_a)
        out_b = torch.empty_like(truth_b)

        with torch.cuda.stream(test_stream):
            # ---- Call A ----
            kw_a = {k: v for k, v in batch_a.items() if not k.startswith("_")}
            hot_cold_attention(
                output=out_a,
                softmax_scale=softmax_scale,
                sliding_window=(-1, -1),
                logits_soft_cap=0.0,
                fa_version=fa_version,
                causal=True,
                **kw_a,
            )

            # ---- Call B (no sync between A and B) ----
            kw_b = {k: v for k, v in batch_b.items() if not k.startswith("_")}
            hot_cold_attention(
                output=out_b,
                softmax_scale=softmax_scale,
                sliding_window=(-1, -1),
                logits_soft_cap=0.0,
                fa_version=fa_version,
                causal=True,
                **kw_b,
            )

        torch.cuda.synchronize()

        # 두 결과 모두 deterministic 하게 ground truth 와 일치해야.
        torch.testing.assert_close(
            out_a, truth_a, atol=5e-3, rtol=5e-3,
            msg=f"Call A mismatch on trial {trial} — likely cross-call "
                f"buffer race (Call B 의 cold-stream write 가 Call A 의 "
                f"default-stream merge 보다 먼저 실행됨)",
        )
        torch.testing.assert_close(
            out_b, truth_b, atol=5e-3, rtol=5e-3,
            msg=f"Call B mismatch on trial {trial}",
        )

    # sanity — slow_merge 가 진짜 hooking 되었는가? 매 trial 당 2회 (Call A + B)
    expected = n_trials * 2
    assert merge_call_count["n"] == expected, (
        f"slow_merge hooked {merge_call_count['n']}회 (expected {expected}) — "
        "monkeypatch 가 실제 호출 경로에 닿지 않으면 race window 가 evoke 되지 않음"
    )


@cuda_required
def test_after_first_call_records_merge_event():
    """fix 의 white-box contract — 첫 hot_cold_attention 호출이 끝나면
    ``_COLD_SCATTER_BUFS`` 의 cache entry 4번째 슬롯에 실제
    ``torch.cuda.Event`` 가 record 되어 있어야 한다. 다음 호출의
    cold-stream write 직전에 이 event 를 ``wait_event`` 함으로써
    cross-call buffer race 가 차단된다.

    dev 에서 race 자체를 deterministic 하게 evoke 가 어려우므로,
    fix 의 *메커니즘이 살아 있다* 는 사실을 직접 검증한다."""
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    from vllm.v1.attention.backends.flash_attn import (
        _COLD_SCATTER_BUFS, hot_cold_attention,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    kv_dtype = torch.bfloat16

    batch = _build_batch(
        cold_per_seq=[2, 4, 0, 1], device=device, kv_dtype=kv_dtype,
    )
    fa_version = get_flash_attn_version()

    out = torch.empty(
        batch["query"].shape, dtype=kv_dtype, device=device,
    )
    kw = {k: v for k, v in batch.items() if not k.startswith("_")}
    hot_cold_attention(
        output=out,
        softmax_scale=1.0 / (128 ** 0.5),
        sliding_window=(-1, -1),
        logits_soft_cap=0.0,
        fa_version=fa_version,
        causal=True,
        **kw,
    )
    torch.cuda.synchronize()

    # cold path 가 firing 됐고, cache entry 가 만들어졌어야.
    assert len(_COLD_SCATTER_BUFS) >= 1, (
        "hot_cold_attention 이 cold path 를 firing 하지 않으면 본 테스트 의미 없음"
    )
    *_, last_event = next(iter(_COLD_SCATTER_BUFS.values()))
    assert last_event is not None, (
        "fix 가 동작하면 첫 호출 후 cache entry 의 last_merge_event 슬롯에 "
        "실제 torch.cuda.Event 가 들어가 있어야"
    )
    assert isinstance(last_event, torch.cuda.Event), (
        f"last_merge_event 가 torch.cuda.Event 여야 하는데 "
        f"{type(last_event).__name__} 가 들어 있음"
    )
    # event 가 진짜 record 됐는지 (즉 query 가능한 상태)
    # query() 는 None/already-completed 상태든 record 안 된 raw event 는
    # CUDA error 를 낼 수 있음. record 된 event 는 항상 query 가능.
    _ = last_event.query()  # raise 없이 통과해야
