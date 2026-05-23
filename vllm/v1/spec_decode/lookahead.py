# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SUB_052 — Lookahead Decoding (CPU Jacobi) Proposer skeleton.

reference:
    arXiv 2402.02057 — "Break the Sequential Dependency of LLM Inference
    Using Lookahead Decoding"
    GitHub hao-ai-lab/LookaheadDecoding

본 file: Proposer interface skeleton 영역 — Jacobi kernel 영역 영역 후 implement.
status: 진입 (skeleton 영역, 영역 algorithm 영역 영역).

Plan:
    shadow_assists/features/IDE_006/TSK_020/planning/SUB_052_lookahead_decoding.md
"""
import os

import numpy as np
import torch

from vllm.config import VllmConfig


class LookaheadProposer:
    """SUB_052 Lookahead Decoding (CPU Jacobi) Proposer — skeleton.

    Mechanism:
        Jacobi iteration generates n-gram pool fully parallel on CPU.
        - window W (lookahead steps)
        - ngram_size N (matched chain length)
        - n_gram pool W × N matrix
        Each step finds best matching chain → GPU verify.

    Current status: skeleton — propose() returns empty list, no spec gain.
    Activate via env:
        VLLM_LOOKAHEAD_ENABLE=1  (default 0 = inactive)
        VLLM_LOOKAHEAD_WINDOW=15  (default 15)
        VLLM_LOOKAHEAD_NGRAM_SIZE=5  (default 5)

    Integration TODO (follow-up work, 2-3 days):
        1. Jacobi iteration numba kernel — generate n-gram pool W × N
        2. n-gram cache lookup (per-prompt local cache, radix or hash)
        3. Best chain selection → return K spec tokens
        4. register in gpu_model_runner.py `method == "lookahead"` branch
        5. SpeculativeConfig.method = "lookahead" 영역 support 추가
    """

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None

        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

        # SUB_052 env config
        self.enable = bool(int(os.environ.get("VLLM_LOOKAHEAD_ENABLE", "0")))
        self.window = int(os.environ.get("VLLM_LOOKAHEAD_WINDOW", "15"))
        self.ngram_size = int(os.environ.get("VLLM_LOOKAHEAD_NGRAM_SIZE", "5"))

        if self.enable:
            import warnings
            warnings.warn(
                "VLLM_LOOKAHEAD_ENABLE=1 set but LookaheadProposer is a skeleton — "
                "Jacobi kernel not yet implemented. Returns empty drafts. "
                "See shadow_assists/features/IDE_006/TSK_020/planning/"
                "SUB_052_lookahead_decoding.md",
                stacklevel=2,
            )

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> list[list[int]]:
        """SUB_052 propose — currently returns empty (skeleton).

        Full implementation (TODO):
            1. Maintain per-request n-gram cache (prefix_window → next-token list)
            2. For each request: run lookahead window with main model on GPU
               (requires gpu_model_runner.py integration)
            3. n-gram pool 영역 update from window output
            4. Match best chain from cache against current suffix
            5. Return chain as draft tokens
        """
        # Placeholder — returns empty drafts (no spec gain).
        # Real integration requires gpu_model_runner.py changes for Jacobi window.
        return [[] for _ in sampled_token_ids]

    def load_model(self, *args, **kwargs):
        """No model to load (n-gram based, prompt-local cache)."""
        pass


# === SUB_052 — n-gram pool match kernel (CPU side, partial impl) ===
# Reference: arXiv 2402.02057 §3 "Lookahead Decoding"
#
# 본 kernel: lookahead_pool 영역 lookup-table 영역 영역 영역 suffix 영역 best match 영역 찾음.
# real Jacobi (W positions parallel model forward) 영역 GPU side — gpu_model_runner.py
# 영역 deep change 필요.
import numpy as np
from numba import njit


@njit
def lookahead_match_kernel(
    prompt_tokens: np.ndarray,  # shape (T,)
    pool: np.ndarray,           # shape (P, N) — N-gram pool (P chains × N tokens)
    pool_count: int,            # actual P (chains in pool)
    suffix_len: int,            # match window 영역 suffix 영역 length
    k: int,                     # draft tokens count
) -> np.ndarray:
    """SUB_052 — find best matching chain in lookahead pool.

    Match prompt's suffix (last `suffix_len` tokens) against the first
    `suffix_len` tokens of each chain in pool. If found, return chain's
    next K tokens.

    Returns shape (k,) chain or empty.
    """
    T = prompt_tokens.shape[0]
    if T < suffix_len or pool_count == 0 or suffix_len >= pool.shape[1]:
        return np.empty((0,), dtype=prompt_tokens.dtype)

    suffix = prompt_tokens[T - suffix_len:]
    best_p = -1
    for p in range(pool_count):
        match = True
        for i in range(suffix_len):
            if pool[p, i] != suffix[i]:
                match = False
                break
        if match:
            best_p = p
            break

    if best_p < 0:
        return np.empty((0,), dtype=prompt_tokens.dtype)

    avail = pool.shape[1] - suffix_len
    n_out = min(k, avail)
    out = np.zeros(n_out, dtype=prompt_tokens.dtype)
    for j in range(n_out):
        out[j] = pool[best_p, suffix_len + j]
    return out


# TODO (gpu_model_runner.py integration):
#   - run main model forward on lookahead window (W positions) parallel
#   - extract argmax tokens for each position → update pool
#   - call lookahead_match_kernel for each request's spec proposal
