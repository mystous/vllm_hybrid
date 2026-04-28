"""TSK_007 GQA broadcast 옵션 A vs B microbench.

옵션 A (current): K/V 를 KV-head 단위 (compact) 로 저장. kernel 내부에서
                  Q-head 인덱스 h 가 KV-head 인덱스 ``h // q_per_kv`` 로
                  매핑 (broadcast). 메모리 절약.
옵션 B (pre-expand): K/V 를 Q-head 단위로 미리 expand. kernel 의
                     num_kv_heads = num_q_heads (q_per_kv = 1). 메모리 *q_per_kv*
                     배 + cache locality 단순.

Llama-3.3-70B GQA = Q heads 64 / KV heads 8 / q_per_kv 8. TP=8 worker per:
  num_q_heads=8, num_kv_heads=1 (option A) / num_kv_heads=8 (option B).
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import torch

# Ensure prod libcuda is preloaded for transitive imports.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm.v1.attention.ops.cpu_partial_attention import (  # noqa: E402
    forward_partial_with_lse,
)
from vllm.v1.attention.ops.kv_view_adapter import (  # noqa: E402
    KVPageLayout,
    KVViewAdapter,
)


def build_inputs(
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    block_size: int,
    n_cold_blocks: int,
    num_seqs: int,
    dtype: torch.dtype,
):
    n_cold_kv = n_cold_blocks * block_size

    layout = KVPageLayout(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        dtype=dtype,
    )

    total_blocks = n_cold_blocks * num_seqs + 4
    cpu_buf = torch.empty(total_blocks, layout.page_size_bytes, dtype=torch.int8)
    adapter = KVViewAdapter(cpu_buf, layout)

    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        adapter.k_view().copy_(
            torch.randn(
                total_blocks, block_size, num_kv_heads, head_dim,
                generator=g, dtype=dtype,
            )
        )
        g.manual_seed(2)
        adapter.v_view().copy_(
            torch.randn(
                total_blocks, block_size, num_kv_heads, head_dim,
                generator=g, dtype=dtype,
            )
        )

    num_tokens = num_seqs  # decode-only
    query = torch.randn(num_tokens, num_q_heads, head_dim, dtype=dtype)
    cu_seqlens_q = torch.arange(0, num_tokens + 1, dtype=torch.int32)
    seq_lens_total = torch.tensor([n_cold_kv] * num_seqs, dtype=torch.int32)
    query_positions = torch.tensor([n_cold_kv - 1] * num_seqs, dtype=torch.int32)
    cold_block_lens = torch.tensor([n_cold_blocks] * num_seqs, dtype=torch.int32)
    cold_block_ids = torch.zeros(num_seqs, n_cold_blocks, dtype=torch.int32)
    for s in range(num_seqs):
        for b in range(n_cold_blocks):
            cold_block_ids[s, b] = s * n_cold_blocks + b

    softmax_scale = 1.0 / (head_dim ** 0.5)

    return dict(
        query=query,
        cold_kv_cache=cpu_buf,
        cold_kv_layout=layout,
        cold_block_ids=cold_block_ids,
        cold_block_lens=cold_block_lens,
        cu_seqlens_q=cu_seqlens_q,
        seq_lens_total=seq_lens_total,
        query_positions=query_positions,
        softmax_scale=softmax_scale,
        causal=True,
    )


def benchmark_call(inputs: dict, n_warmup: int = 5, n_iter: int = 30) -> list[float]:
    for _ in range(n_warmup):
        forward_partial_with_lse(**inputs)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        forward_partial_with_lse(**inputs)
        times.append(time.perf_counter() - t0)
    return times


def fmt(times: list[float]) -> str:
    median = statistics.median(times) * 1000
    mean = statistics.mean(times) * 1000
    p90 = sorted(times)[int(0.9 * len(times))] * 1000
    p99 = sorted(times)[int(0.99 * len(times))] * 1000 if len(times) >= 100 else p90
    return f"median={median:6.2f} mean={mean:6.2f} p90={p90:6.2f} p99={p99:6.2f} ms"


def main():
    # Llama-3.3-70B + TP=8 worker per:
    #   num_q_heads = 64 / 8 = 8
    #   num_kv_heads = 8 / 8 = 1   (option A)
    # q_per_kv = num_q_heads / num_kv_heads_A = 8

    NUM_Q_HEADS = 8
    NUM_KV_HEADS_A = 1  # compact
    NUM_KV_HEADS_B = NUM_Q_HEADS  # pre-expanded
    HEAD_DIM = 128
    BLOCK_SIZE = 64
    DTYPE = torch.bfloat16

    print(
        f"GQA layout: q_heads={NUM_Q_HEADS} kv_heads_A={NUM_KV_HEADS_A} "
        f"kv_heads_B={NUM_KV_HEADS_B} q_per_kv={NUM_Q_HEADS // NUM_KV_HEADS_A}"
    )
    print(f"head_dim={HEAD_DIM} block_size={BLOCK_SIZE} dtype={DTYPE}")
    print()

    print(f"{'n_cold_blocks':>14} {'num_seqs':>9} | {'A (compact)':<48} | {'B (expand)':<48} | {'B/A':>6}")
    print("-" * 134)

    for n_cold_blocks in [50, 100, 200]:
        for num_seqs in [1, 4, 16]:
            inputs_a = build_inputs(
                num_kv_heads=NUM_KV_HEADS_A,
                num_q_heads=NUM_Q_HEADS,
                head_dim=HEAD_DIM,
                block_size=BLOCK_SIZE,
                n_cold_blocks=n_cold_blocks,
                num_seqs=num_seqs,
                dtype=DTYPE,
            )
            inputs_b = build_inputs(
                num_kv_heads=NUM_KV_HEADS_B,
                num_q_heads=NUM_Q_HEADS,
                head_dim=HEAD_DIM,
                block_size=BLOCK_SIZE,
                n_cold_blocks=n_cold_blocks,
                num_seqs=num_seqs,
                dtype=DTYPE,
            )

            times_a = benchmark_call(inputs_a, n_warmup=5, n_iter=30)
            times_b = benchmark_call(inputs_b, n_warmup=5, n_iter=30)

            ma = statistics.median(times_a) * 1000
            mb = statistics.median(times_b) * 1000
            ratio = mb / ma if ma > 0 else float("nan")

            print(
                f"{n_cold_blocks:>14} {num_seqs:>9} | "
                f"{fmt(times_a):<48} | {fmt(times_b):<48} | {ratio:>5.2f}×"
            )

    print()
    print("결과 해석")
    print("- B/A < 1.0: 옵션 B (pre-expand) 가 빠름 — kernel 의 cache pattern 이 expanded layout 에 유리")
    print("- B/A > 1.0: 옵션 A (compact + broadcast) 가 빠름 — broadcast 비용 < cache miss 비용")


if __name__ == "__main__":
    main()
