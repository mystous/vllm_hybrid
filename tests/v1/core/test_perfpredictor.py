# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm/v1/core/sched/perfpredictor.py``.

Tests do not exercise any actual GPU or CPU forward; they verify the
shape of the index arrays, lower-bound lookup, 1D and bilinear
interpolation, and the ZeroPerfPredictor stub.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm.v1.core.sched.perfpredictor import (
    PerfPredictor,
    TablePerfPredictor,
    ZeroPerfPredictor,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
@dataclass
class _SchedCfg:
    max_num_batched_tokens: int = 1024
    max_num_seqs: int = 32


@dataclass
class _CacheCfg:
    block_size: int = 16


@dataclass
class _ModelCfg:
    max_model_len: int = 4096


@dataclass
class _VllmCfg:
    scheduler_config: _SchedCfg
    cache_config: _CacheCfg
    model_config: _ModelCfg


def _make_predictor(**overrides) -> TablePerfPredictor:
    cfg = _VllmCfg(_SchedCfg(), _CacheCfg(), _ModelCfg())
    if "max_num_batched_tokens" in overrides:
        cfg.scheduler_config.max_num_batched_tokens = \
            overrides["max_num_batched_tokens"]
    if "max_num_seqs" in overrides:
        cfg.scheduler_config.max_num_seqs = overrides["max_num_seqs"]
    if "block_size" in overrides:
        cfg.cache_config.block_size = overrides["block_size"]
    if "max_model_len" in overrides:
        cfg.model_config.max_model_len = overrides["max_model_len"]
    return TablePerfPredictor(cfg)


# ----------------------------------------------------------------------
# Abstract & Zero
# ----------------------------------------------------------------------
def test_abstract_perfpredictor_raises():
    p = PerfPredictor()
    for fn in (lambda: p.get_linr_T(1),
               lambda: p.get_pref_T(1),
               lambda: p.get_gdec_T(1),
               lambda: p.get_cdec_T(1, 1),
               lambda: p.get_lnch_T()):
        with pytest.raises(NotImplementedError):
            fn()


def test_zero_perfpredictor_returns_zero():
    p = ZeroPerfPredictor()
    assert p.get_linr_T(100) == 0.0
    assert p.get_pref_T(100) == 0.0
    assert p.get_gdec_T(100) == 0.0
    assert p.get_cdec_T(8, 100) == 0.0
    assert p.get_lnch_T() == 0.0


# ----------------------------------------------------------------------
# Index arrays
# ----------------------------------------------------------------------
def test_index_arrays_are_sorted_unique_and_bounded():
    p = _make_predictor(max_num_batched_tokens=2048, max_num_seqs=64)
    for name, arr in (("linr_S", p.linr_S_list),
                      ("pref_S", p.pref_S_list),
                      ("gdec_N", p.gdec_N_list),
                      ("cdec_S", p.cdec_S_list)):
        assert len(arr) > 0, name
        assert arr == sorted(arr), name
        assert all(arr[i] != arr[i + 1] for i in range(len(arr) - 1)) or \
               len(set(arr)) == len(arr), name
        assert arr[-1] >= arr[0], name


def test_lb_idx_lookup_correctness():
    """For every integer x, _get_lb_idx_list returns the smallest j
    such that input_list[j] >= x."""
    arr = [1, 4, 8, 16]
    lb = TablePerfPredictor._get_lb_idx_list(arr)
    # lb[0] points to first valid index — by construction j=0 entries
    # are 0 (input_list[0] + 1 zeros prepended to handle x in [0, arr[0]])
    assert lb[0] == 0
    assert lb[1] == 0  # arr[0] == 1 >= 1
    assert lb[2] == 1  # arr[1] == 4 >= 2
    assert lb[4] == 1  # arr[1] == 4 >= 4
    assert lb[5] == 2  # arr[2] == 8 >= 5
    assert lb[16] == 3 # arr[3] == 16 >= 16


# ----------------------------------------------------------------------
# 1D interpolation
# ----------------------------------------------------------------------
def test_interp_two_points():
    assert TablePerfPredictor._interp(5, 0, 10, 0.0, 1.0) == pytest.approx(0.5)
    assert TablePerfPredictor._interp(0, 0, 10, 0.0, 1.0) == 0.0
    assert TablePerfPredictor._interp(10, 0, 10, 0.0, 1.0) == 1.0
    # Degenerate (x0 == x1): return y0 to avoid div-by-zero
    assert TablePerfPredictor._interp(5, 5, 5, 1.5, 99.0) == 1.5


def test_get_linr_T_interpolation():
    p = _make_predictor()
    p.linr_T_list = [float(s) for s in p.linr_S_list]   # T = S identity
    # exact-match points return the table value
    for s in p.linr_S_list[:5]:
        assert p.get_linr_T(s) == float(s)
    # zero returns zero
    assert p.get_linr_T(0) == 0.0
    # last bucket
    assert p.get_linr_T(p.linr_S_list[-1]) == float(p.linr_S_list[-1])


def test_get_linr_T_zero_when_not_profiled():
    p = _make_predictor()
    # linr_T_list is None until ModelProfiler.run() populates it
    assert p.linr_T_list is None
    assert p.get_linr_T(64) == 0.0


def test_get_gdec_T_interpolation_between_points():
    p = _make_predictor()
    p.gdec_T_list = [float(n) * 2.0 for n in p.gdec_N_list]   # T = 2N
    n0, n1 = p.gdec_N_list[1], p.gdec_N_list[2]
    midpoint = (n0 + n1) // 2
    expected = 2.0 * midpoint
    actual = p.get_gdec_T(midpoint)
    # Allow up to 1 unit of difference because of integer midpoint rounding
    assert abs(actual - expected) < 2.0


# ----------------------------------------------------------------------
# 2D bilinear (cdec)
# ----------------------------------------------------------------------
def test_get_cdec_T_zero_input():
    p = _make_predictor()
    # cdec_T_lists is initialised to [None, None, ...]
    assert p.get_cdec_T(0, 100) == 0.0
    assert p.get_cdec_T(4, 0) == 0.0


def test_get_cdec_T_with_table_populated():
    p = _make_predictor(max_num_seqs=16, max_model_len=512)
    # T(S, N) = S * N (linear in both dimensions for predictability)
    p.cdec_T_lists = [
        [float(S * N) for N in p.cdec_N_list_agg]
        for S in p.cdec_S_list
    ]
    # Exact match points
    S0 = p.cdec_S_list[0]
    N0 = p.cdec_N_list_agg[0]
    assert p.get_cdec_T(S0, N0) == pytest.approx(float(S0 * N0))
    # Bilinear is not exact for the product T = S*N — verify
    # the result lies between the four surrounding points
    S = p.cdec_S_list[1]
    N = p.cdec_N_list_agg[1]
    val = p.get_cdec_T(S, N)
    assert val == pytest.approx(float(S * N))


# ----------------------------------------------------------------------
# Launch overhead
# ----------------------------------------------------------------------
def test_lnch_T_default_is_neo_constant():
    p = _make_predictor()
    assert p.get_lnch_T() == pytest.approx(0.8)
    p.lnch_T = 1.5
    assert p.get_lnch_T() == 1.5
