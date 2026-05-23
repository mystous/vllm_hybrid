# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import numpy as np
import torch
from numba import get_num_threads, jit, njit, prange, set_num_threads

from vllm.config import VllmConfig


class NgramProposer:
    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        # Minimum length of the n-gram to match.
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        # Maximum length of the n-gram to match.
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # Number of tokens follow the match. If there are less than k
        # tokens follow the match, we will return the maximum amount of
        # tokens until the end.
        self.k = vllm_config.speculative_config.num_speculative_tokens
        # Maximum length of the model.
        self.max_model_len = vllm_config.model_config.max_model_len

        # Pre-allocate buffers for numba batch propose.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros((max_num_seqs), dtype=np.int32)

        # Threshold of total number of tokens in the batch to enable
        # multi-threading in numba batch propose.
        self.num_tokens_threshold = 8192
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        cpu_count = os.cpu_count()
        # SUB_047 (Tier 1 B) — env-tunable ngram numba thread cap.
        #   Original vLLM cap = min(1, cpu_count//2) → effectively 1 thread!
        #   TODO(ekagra-ranjan): bump up the cap from 1 to 8 when TP parallelization
        #   for ngram is implemented.
        #   본 lever: VLLM_NGRAM_NUM_THREADS_CAP (default 1, recommended 8 or higher)
        #             VLLM_NGRAM_DIVIDE_BY_TP (default 1 = divide, 0 = skip — rank 영역 thread 영역 ↑)
        cap = int(os.environ.get("VLLM_NGRAM_NUM_THREADS_CAP", "1"))
        divide_by_tp = int(os.environ.get("VLLM_NGRAM_DIVIDE_BY_TP", "1"))
        # SUB_057 (Tier C) — ngram tree expansion (top-M multi-chain candidates).
        #   default 1 = current SUB_047 behavior (single longest chain).
        #   > 1 = generate top-M chains via _find_top_m_matched_ngrams_and_propose_tokens,
        #         currently 1st chain (longest) 영역 영역 사용 — rejection_sampler tree verify
        #         적재 영역 영역 영역 same behavior 영역 top-1 (numba kernel time 영역 약간 ↑).
        #   plan: shadow_assists/features/IDE_006/TSK_020/planning/SUB_057_ngram_tree_expansion.md
        #   surface: (1) batch_propose numba kernel — top-M heap [✓ wired], (2) rejection_sampler — tree path [✗ TODO]
        self.ngram_top_m = max(1, int(os.environ.get("VLLM_NGRAM_TOP_M", "1")))
        if self.ngram_top_m > 1:
            # Pre-allocate top-M buffer (3D: max_num_seqs × top_m × k)
            self.valid_ngram_draft_topm = np.zeros(
                (max_num_seqs, self.ngram_top_m, self.k), dtype=np.int32
            )
            self.valid_ngram_topm_count = np.zeros((max_num_seqs), dtype=np.int32)
            # log activation
            import warnings
            warnings.warn(
                f"SUB_057: VLLM_NGRAM_TOP_M={self.ngram_top_m} wired — top-M numba "
                f"kernel active, but only chain 0 (longest) used until "
                f"rejection_sampler tree verify is implemented.",
                stacklevel=2,
            )
        # Max number of threads for numba parallel processing.
        if cpu_count:
            # Divide by 2 to use physical cores
            # and not logical cores (hyper-threading).
            self.num_numba_thread_available = max(1, min(cap, (cpu_count // 2)))
            if divide_by_tp:
                # Divide by tp_size to ensure each tensor parallel rank
                # has some threads since all ranks will run this.
                self.num_numba_thread_available //= tp_size
            self.num_numba_thread_available = max(1, self.num_numba_thread_available)
        else:
            self.num_numba_thread_available = 1

        # Trigger Numba JIT compilation for N-gram proposer.
        # This usually takes less than 1 second.
        self.propose(
            [[]] * 1024,
            np.zeros(1024, dtype=np.int32),
            np.zeros((1024, self.max_model_len), dtype=np.int32),
        )

    def batch_propose(
        self,
        num_requests: int,
        valid_ngram_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        """Batch version of ngram proposer using numba for acceleration.

        Args:
            valid_ngram_requests:
                Set of indices of requests that need ngram proposals.
            num_tokens_no_spec:
                Numpy array of shape (batch_size,) representing the number
                of tokens without speculative tokens for each request.
            token_ids_cpu:
                Numpy array of shape (batch_size, max_model_len)
                representing the token IDs for each request.

        Returns:
            list[list[int]]:
                A list where each element is a list of proposed
                token IDs for the corresponding request.
        """
        draft_token_ids: list[list[int]] = []

        # Only run batch propose if there are requests needing ngram proposals.
        # avoid calling numba function with empty list which causes error
        # ValueError: cannot compute fingerprint of empty list
        if num_ngram_requests := len(valid_ngram_requests):
            original_num_numba_threads = get_num_threads()
            # Ensure we use at least one thread.
            # If total tokens is small, using multiple threads
            # may slow down due to overhead.
            total_tokens = np.sum(num_tokens_no_spec)
            if total_tokens >= self.num_tokens_threshold:
                final_num_threads = max(
                    1, min(self.num_numba_thread_available, num_ngram_requests)
                )
                set_num_threads(final_num_threads)
            else:
                set_num_threads(1)

            if self.ngram_top_m > 1:
                # SUB_057 top-M path — generate top-M chains, use chain 0 only
                # (rejection_sampler tree verify TODO).
                batch_propose_numba_topm(
                    valid_ngram_requests,
                    num_tokens_no_spec,
                    token_ids_cpu,
                    self.min_n,
                    self.max_n,
                    self.max_model_len,
                    self.k,
                    self.ngram_top_m,
                    self.valid_ngram_draft_topm,
                    self.valid_ngram_topm_count,
                )
            else:
                batch_propose_numba(
                    valid_ngram_requests,
                    num_tokens_no_spec,
                    token_ids_cpu,
                    self.min_n,
                    self.max_n,
                    self.max_model_len,
                    self.k,
                    self.valid_ngram_draft,
                    self.valid_ngram_num_drafts,
                )

            # Restore original number of threads.
            set_num_threads(original_num_numba_threads)

        if self.ngram_top_m > 1:
            # Use chain 0 (longest) — same length as top-1 best (output 영역 동일).
            for i in range(num_requests):
                if i in valid_ngram_requests and self.valid_ngram_topm_count[i] > 0:
                    # chain 0 from top-M output (longest)
                    draft_token_ids.append(
                        self.valid_ngram_draft_topm[i, 0, : self.k].tolist()
                    )
                else:
                    draft_token_ids.append([])
        else:
            for i in range(num_requests):
                if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:
                    draft_token_ids.append(
                        self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()
                    )
                else:
                    draft_token_ids.append([])

        return draft_token_ids

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        # find which requests need ngram proposals
        valid_ngram_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                continue

            num_tokens = num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                continue

            valid_ngram_requests.append(i)

        draft_token_ids = self.batch_propose(
            len(sampled_token_ids),
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
        )

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass


@njit(parallel=True)
def batch_propose_numba(
    valid_ngram_requests: list,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
):
    for i in prange(len(valid_ngram_requests)):
        idx = valid_ngram_requests[i]
        num_tokens = num_tokens_no_spec[idx]
        context_token_ids = token_ids_cpu[idx, :num_tokens]
        drafter_output = _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=context_token_ids,
            min_ngram=min_n,
            max_ngram=max_n,
            max_model_len=max_model_len,
            k=k,
        )

        valid_ngram_num_drafts[idx] = drafter_output.shape[0]
        if len(drafter_output):
            valid_ngram_draft[idx, : drafter_output.shape[0]] = drafter_output


# SUB_057 — batch_propose top-M variant. Generates top-M chains per request,
# stores in valid_ngram_draft_topm[idx, m, :k]. Caller uses chain 0 only
# until rejection_sampler tree verify is wired (TODO).
@njit(parallel=True)
def batch_propose_numba_topm(
    valid_ngram_requests: list,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
    top_m: int,
    valid_ngram_draft_topm: np.ndarray,  # shape (max_num_seqs, top_m, k)
    valid_ngram_topm_count: np.ndarray,  # shape (max_num_seqs,)
):
    for i in prange(len(valid_ngram_requests)):
        idx = valid_ngram_requests[i]
        num_tokens = num_tokens_no_spec[idx]
        context_token_ids = token_ids_cpu[idx, :num_tokens]
        # output shape: (M_actual, k)
        topm_output = _find_top_m_matched_ngrams_and_propose_tokens(
            origin_tokens=context_token_ids,
            min_ngram=min_n,
            max_ngram=max_n,
            max_model_len=max_model_len,
            k=k,
            top_m=top_m,
        )
        m_actual = topm_output.shape[0]
        valid_ngram_topm_count[idx] = m_actual
        if m_actual > 0:
            valid_ngram_draft_topm[idx, :m_actual, :] = topm_output


@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """
    Find the longest n-gram which matches the suffix of the given tokens
    whose length is within [min_ngram, max_ngram] (inclusive).

    If found, we will extract k right after the matched ngram.
    """
    # Do not generate draft tokens is context is shorter than minimum n-gram
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Do not generate draft tokens beyond the max model length.
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip tokens, and the goal become to find longest ngram
    # on the rightmost position which matches the prefix with
    # length [min_n, max_n] (inclusive).
    tokens = origin_tokens[::-1]

    # Longest prefix (not including itself) which is a suffix of
    # the current position.
    #   lps[i] = max{v, where tokens[0:v] == tokens[i+1-v:i+1]}
    #
    # As ngram is capped by max_ngram to save memory, we only need to
    # store lps for the first max_ngram prefix.
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    # lps[0] always equal to 0, we start with index 1
    prev_lps = 0
    i = 1
    while i < total_token:
        # tokens[:prev_lps] is the longest prefix as a suffix of tokens[:i]
        if tokens[prev_lps] == tokens[i]:
            # Token match: tokens[:prev_lps+1] is the longest prefix as
            # a suffix of tokens[:i+1]
            prev_lps += 1
            # Check if we found a longer valid ngram.
            #
            # Update position when longest_ngram matched prev_lps,
            # as we want to get the target n-gram of the earliest position
            # in the original tokens (i.e.
            # latest position in the reversed tokens)
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                # Store LPS for the first max_ngram prefix
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                # When prev_lps reached max_ngram, update prev_lps
                # to lps[max_ngram-1] to avoid matching ngram
                # longer than max_ngram
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            # Token mismatch: try the second-longest prefix
            # among all suffix of tokens[:i],
            # which is the longest prefix of tokens[:prev_lps]
            prev_lps = lps[prev_lps - 1]
        else:
            # Token mismatch, and no more prefix (except empty string)
            # as a suffix of tokens[:i]
            i += 1

    if longest_ngram < min_ngram:
        # No valid ngram is found
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip the position back, so in origin_tokens,
    # origin_tokens[total_token-1-position:total_token-1-position+longest_ngram]
    # is the matched ngram, so we should start drafting tokens from
    # total_token-1-position+longest_ngram
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position : start_position + k]


# SUB_057 (Tier C) — top-M chain ngram extraction.
#   기존 _find_longest_matched_ngram_and_propose_tokens 영역 longest 1 chain 만 반환.
#   본 영역 longest 영역 영역 top_m 영역 후보 chain 영역 2D 영역 (M, K) 반환.
#   현 상태: 영역 numba kernel 영역 작동 — chain 0 (longest) 영역 영역 기존 behavior 영역 영역 영역,
#   chain 1~M-1 영역 영역 영역 alternative.
#   사용: 영역 rejection_sampler 영역 tree path 영역 지원 영역 영역 영역 영역 영역 (현재 미지원).
@jit(nopython=True)
def _find_top_m_matched_ngrams_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
    top_m: int,
) -> np.ndarray:
    """
    SUB_057: Find top-M longest n-grams matching the suffix of `origin_tokens`,
    return up to M chains of K candidate tokens each as a 2D array (M_actual, K).

    M_actual ≤ top_m (영역 영역 less 영역 영역 distinct candidate 영역 영역 영역 영역).
    Returns shape (0, K) if no valid ngram.
    """
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram or top_m < 1:
        return np.empty((0, k), dtype=origin_tokens.dtype)

    k_actual = min(k, max_model_len - total_token)
    if k_actual <= 0:
        return np.empty((0, k), dtype=origin_tokens.dtype)

    # Reverse for KMP-style scan
    tokens = origin_tokens[::-1]
    lps = np.zeros(max_ngram, dtype=np.int32)

    # Track top-M (ngram_len, position) pairs — sorted ascending by length.
    # 영역 영역 영역 영역 영역 새 후보 영역 영역 (ngram_len 더 큼 또는 영역 length 영역 다른 position).
    cand_ngrams = np.zeros(top_m, dtype=np.int32)  # length
    cand_positions = np.zeros(top_m, dtype=np.int32)  # rightmost matching position in reversed
    n_cand = 0

    prev_lps = 0
    i = 1
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            if prev_lps >= min_ngram:
                # Insert candidate sorted by length (descending).
                # We keep candidates with distinct end positions (i 영역 unique).
                # Simple O(M) insertion sort 영역 (top_m 영역 small).
                inserted = False
                # Check if already have this position (deduplicate by position too)
                dup = False
                for j in range(n_cand):
                    if cand_positions[j] == i:
                        dup = True
                        break
                if not dup:
                    if n_cand < top_m:
                        # Append + bubble-sort by length descending
                        cand_ngrams[n_cand] = prev_lps
                        cand_positions[n_cand] = i
                        n_cand += 1
                        # bubble-sort (descending by length)
                        for jj in range(n_cand - 1, 0, -1):
                            if cand_ngrams[jj] > cand_ngrams[jj - 1]:
                                # swap
                                tmp_l = cand_ngrams[jj]
                                tmp_p = cand_positions[jj]
                                cand_ngrams[jj] = cand_ngrams[jj - 1]
                                cand_positions[jj] = cand_positions[jj - 1]
                                cand_ngrams[jj - 1] = tmp_l
                                cand_positions[jj - 1] = tmp_p
                            else:
                                break
                    elif prev_lps > cand_ngrams[top_m - 1]:
                        # Replace smallest + re-sort
                        cand_ngrams[top_m - 1] = prev_lps
                        cand_positions[top_m - 1] = i
                        for jj in range(top_m - 1, 0, -1):
                            if cand_ngrams[jj] > cand_ngrams[jj - 1]:
                                tmp_l = cand_ngrams[jj]
                                tmp_p = cand_positions[jj]
                                cand_ngrams[jj] = cand_ngrams[jj - 1]
                                cand_positions[jj] = cand_positions[jj - 1]
                                cand_ngrams[jj - 1] = tmp_l
                                cand_positions[jj - 1] = tmp_p
                            else:
                                break

            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if n_cand == 0:
        return np.empty((0, k), dtype=origin_tokens.dtype)

    # Build output 2D array — M_actual chains × k tokens
    out = np.zeros((n_cand, k), dtype=origin_tokens.dtype)
    for m in range(n_cand):
        ngram_len = cand_ngrams[m]
        position = cand_positions[m]
        start_position = total_token - 1 - position + ngram_len
        k_avail = min(k, total_token - start_position)
        if k_avail > 0:
            out[m, :k_avail] = origin_tokens[start_position : start_position + k_avail]

    return out
