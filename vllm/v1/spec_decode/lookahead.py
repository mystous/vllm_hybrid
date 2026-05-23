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
        """SUB_052 propose — currently returns empty (skeleton)."""
        # TODO: implement Jacobi iteration n-gram pool generation here.
        # For now, return empty drafts for each request (no spec gain).
        return [[] for _ in sampled_token_ids]

    def load_model(self, *args, **kwargs):
        """No model to load (n-gram based, prompt-local cache)."""
        pass


# === Jacobi iteration kernel (TODO) ===
# Reference impl:
#   GitHub hao-ai-lab/LookaheadDecoding 의 lookahead/decoding.py
#   핵심: jacobi_window 함수가 W positions 의 token 추정을 parallel update.
#
# Numba kernel sketch:
#   @njit(parallel=True)
#   def jacobi_window_kernel(
#       prompt_tokens: np.ndarray,  # shape (T,)
#       window: int,                # W
#       ngram_size: int,            # N
#       max_iter: int,              # convergence iter cap
#   ) -> np.ndarray:                # shape (W, N) n-gram pool
#       ...
#
# n-gram cache lookup (per-prompt, simple hash table):
#   key = tuple of N tokens (prefix of length N-1)
#   value = next token candidates
#
# best chain selection:
#   for each candidate chain in pool, score by: (length × confidence).
#   pick top-1 (또는 top-M with SUB_057 integration).
