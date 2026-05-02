# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Performance predictor for NEO-style asymmetric GPU/CPU pipelining.

The algorithms in this module are adapted from the NEO project
(https://github.com/NEO-MLSys25/NEO, MLSys 2025, Apache 2.0).
Only algorithms are reused; no code is copied verbatim.

Used by ``vllm/v1/core/sched/scheduler.py`` (NEO-style mode selection)
to decide whether to run a sequential single-batch or two pipelined
sub-batches in the next iteration.

See ``shadow_assists/features/IDE_006/NEO_code_deepdive.md`` §6 for
the algorithm reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class PerfPredictor:
    """Abstract performance predictor.

    Predicts the wall time of the four primary sub-stages of a
    NEO-style sub-batch:

    - ``get_linr_T(S)`` — linear / FFN / projection time for iter width S
    - ``get_pref_T(S)`` — GPU prefilling time for total prefill tokens S
    - ``get_gdec_T(N)`` — GPU paged-attention decoding time for total
      KV tokens N
    - ``get_cdec_T(S, N)`` — CPU paged-attention decoding time for
      batch S and total KV tokens N (bilinear)
    - ``get_lnch_T()`` — constant launch overhead (kernel dispatch)
    """

    def get_linr_T(self, S: int) -> float:
        raise NotImplementedError

    def get_pref_T(self, S: int) -> float:
        raise NotImplementedError

    def get_gdec_T(self, N: int) -> float:
        raise NotImplementedError

    def get_cdec_T(self, S: int, N: int) -> float:
        raise NotImplementedError

    def get_lnch_T(self) -> float:
        raise NotImplementedError


class ZeroPerfPredictor(PerfPredictor):
    """Always returns zero. Used in unit tests and as the default
    when profiling has not yet completed."""

    def get_linr_T(self, S: int) -> float:
        return 0.0

    def get_pref_T(self, S: int) -> float:
        return 0.0

    def get_gdec_T(self, N: int) -> float:
        return 0.0

    def get_cdec_T(self, S: int, N: int) -> float:
        return 0.0

    def get_lnch_T(self) -> float:
        return 0.0


class TablePerfPredictor(PerfPredictor):
    """Table-based predictor with linear interpolation.

    The four ``*_S_list`` / ``*_N_list`` index arrays are seeded with
    powers of two plus a few interleaving points. ``ModelProfiler``
    fills the matching ``*_T_list`` arrays at startup. After that,
    ``get_*_T`` returns ``ys[idx]`` for an exact match or a linear
    interpolation between adjacent points.

    The 2D ``get_cdec_T(S, N)`` performs bilinear interpolation:
    interpolate along ``N`` for each adjacent ``S`` row, then
    interpolate the two row values along ``S``.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        sched = vllm_config.scheduler_config
        cache = vllm_config.cache_config
        model = vllm_config.model_config

        max_tokens_in_batch = sched.max_num_batched_tokens
        max_batch_size = sched.max_num_seqs
        block_size = cache.block_size
        # Conservative GPU/CPU token caps if the offload connector
        # has not been initialized yet. Profiler can override them.
        max_gpu_tokens = max(max_tokens_in_batch, 1)
        max_cpu_tokens = max(max_tokens_in_batch * 4, 1)
        max_seq_len = model.max_model_len

        # ── linr_S_list — fine-grained for the first 512 tokens then
        # power-of-two until max_tokens_in_batch
        self.linr_S_list = list(range(1, min(512, max_tokens_in_batch))) + [
            2**i for i in range(9, max(9, (max_tokens_in_batch - 1).bit_length()))
        ] + [max_tokens_in_batch]
        self.linr_T_list: list[float] | None = None
        self.linr_S_lb_idx = self._get_lb_idx_list(self.linr_S_list)
        # NEO heuristic: above this iter width we generally would not
        # accept further CPU prefills.
        self.linr_S_threshold = 128

        # ── pref_S_list — power-of-two with 3 * 2^(i-2) interleave
        block_log = max(1, (block_size - 1).bit_length())
        max_log = max(block_log + 1, (max_tokens_in_batch - 1).bit_length())
        self.pref_S_list = sum(
            [[2 ** (i - 2) * 3, 2**i] for i in range(block_log, max_log)],
            [],
        ) + [max_tokens_in_batch]
        self.pref_T_list: list[float] | None = None
        self.pref_S_lb_idx = self._get_lb_idx_list(self.pref_S_list)

        # ── gdec_N_list — same shape as pref but bounded by max_gpu_tokens
        gpu_log = max(block_log + 1, (max_gpu_tokens - 1).bit_length())
        self.gdec_N_list = sum(
            [[2 ** (i - 2) * 3, 2**i] for i in range(block_log, gpu_log)],
            [],
        ) + [max_gpu_tokens]
        self.gdec_T_list: list[float] | None = None
        self.gdec_N_lb_idx = self._get_lb_idx_list(self.gdec_N_list)

        # ── cdec — 2D table indexed by (batch_S, total_N)
        bs_log = max(1, (max_batch_size - 1).bit_length())
        self.cdec_S_list = [2**i for i in range(0, bs_log)] + [max_batch_size]
        self.cdec_N_lists: list[list[int]] = []
        for S in self.cdec_S_list:
            min_N = S * block_size
            max_N = min(S * max_seq_len, max_cpu_tokens)
            top = max(min_N + 1, max_N)
            entries = [min_N]
            for i in range(min_N.bit_length(), max(min_N.bit_length(), (top - 1).bit_length())):
                v = 2**i
                if v > min_N and v < top:
                    entries.append(v)
            entries.append(top)
            self.cdec_N_lists.append(entries)
        self.cdec_N_list_agg = sorted({n for row in self.cdec_N_lists for n in row})
        self.cdec_T_lists: list[list[float] | None] = [
            None for _ in self.cdec_S_list
        ]
        self.cdec_S_lb_idx = self._get_lb_idx_list(self.cdec_S_list)
        self.cdec_N_lb_idx = self._get_lb_idx_list(self.cdec_N_list_agg)

        # ── launch overhead constant (ms). Overridden by ModelProfiler
        # if measured. NEO uses 0.8 ms.
        self.lnch_T = 0.8

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_lb_idx_list(input_list: list[int]) -> list[int]:
        """For each integer x in ``[0, input_list[-1]]`` returns the
        smallest j such that ``input_list[j] >= x``. Lookup is O(1)
        after construction."""
        if not input_list:
            return []
        result: list[int] = [0] * (input_list[0] + 1)
        for i in range(len(input_list) - 1):
            gap = input_list[i + 1] - input_list[i]
            if gap > 0:
                result.extend([i + 1] * gap)
        return result

    @staticmethod
    def _interp(x: int, x0: int, x1: int, y0: float, y1: float) -> float:
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def _interp_1d(
        self,
        x: int,
        xs: list[int],
        ys: list[float] | None,
        x_lb_idx: list[int],
    ) -> float:
        if ys is None:
            return 0.0
        if x <= 0:
            return 0.0
        if x > xs[-1]:
            x = xs[-1]
        idx = x_lb_idx[x] if x < len(x_lb_idx) else len(xs) - 1
        if idx >= len(xs):
            idx = len(xs) - 1
        if idx == 0 or x == xs[idx]:
            return ys[idx]
        return self._interp(x, xs[idx - 1], xs[idx], ys[idx - 1], ys[idx])

    # ------------------------------------------------------------------
    # PerfPredictor interface
    # ------------------------------------------------------------------
    def get_linr_T(self, S: int) -> float:
        return self._interp_1d(S, self.linr_S_list, self.linr_T_list,
                               self.linr_S_lb_idx)

    def get_pref_T(self, S: int) -> float:
        return self._interp_1d(S, self.pref_S_list, self.pref_T_list,
                               self.pref_S_lb_idx)

    def get_gdec_T(self, N: int) -> float:
        return self._interp_1d(N, self.gdec_N_list, self.gdec_T_list,
                               self.gdec_N_lb_idx)

    def get_cdec_T(self, S: int, N: int) -> float:
        if S <= 0 or N <= 0:
            return 0.0
        if self.cdec_S_list[-1] < S:
            S = self.cdec_S_list[-1]
        s_idx = self.cdec_S_lb_idx[S] if len(self.cdec_S_lb_idx) > S else \
            len(self.cdec_S_list) - 1
        if s_idx >= len(self.cdec_S_list):
            s_idx = len(self.cdec_S_list) - 1

        ts1_row = self.cdec_T_lists[s_idx]
        if s_idx == 0 or self.cdec_S_list[s_idx] == S:
            return self._interp_1d(N, self.cdec_N_list_agg, ts1_row,
                                   self.cdec_N_lb_idx)
        ts0_row = self.cdec_T_lists[s_idx - 1]
        ts0 = self._interp_1d(N, self.cdec_N_list_agg, ts0_row,
                              self.cdec_N_lb_idx)
        ts1 = self._interp_1d(N, self.cdec_N_list_agg, ts1_row,
                              self.cdec_N_lb_idx)
        return self._interp(S, self.cdec_S_list[s_idx - 1],
                            self.cdec_S_list[s_idx], ts0, ts1)

    def get_lnch_T(self) -> float:
        return self.lnch_T
