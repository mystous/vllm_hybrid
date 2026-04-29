# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm/v1/metrics/profiler.py`` (NEO ModelProfiler)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm.v1.core.sched.perfpredictor import TablePerfPredictor
from vllm.v1.metrics.profiler import ModelProfiler


@dataclass
class _SchedCfg:
    max_num_batched_tokens: int = 256
    max_num_seqs: int = 8


@dataclass
class _CacheCfg:
    block_size: int = 16


@dataclass
class _ModelCfg:
    max_model_len: int = 512


@dataclass
class _VllmCfg:
    scheduler_config: _SchedCfg
    cache_config: _CacheCfg
    model_config: _ModelCfg


def _stub_measure_constant(value: float):
    """Return a measure_fn that always returns ``value``."""
    calls = []

    def fn(kind: str, p1: int, p2: int = 0) -> float:
        calls.append((kind, p1, p2))
        return value

    fn.calls = calls   # type: ignore[attr-defined]
    return fn


def test_profiler_populates_all_tables_with_constant_measure():
    pred = TablePerfPredictor(_VllmCfg(_SchedCfg(), _CacheCfg(), _ModelCfg()))
    measure = _stub_measure_constant(1.5)
    profiler = ModelProfiler(pred, measure, nwarmup=1, nrepeat=2)

    profiler.run()

    assert pred.linr_T_list is not None
    assert pred.pref_T_list is not None
    assert pred.gdec_T_list is not None
    assert pred.cdec_T_lists is not None
    assert all(row is not None for row in pred.cdec_T_lists)

    # All measurements equal the constant
    assert all(t == pytest.approx(1.5) for t in pred.linr_T_list)
    assert all(t == pytest.approx(1.5) for t in pred.pref_T_list)
    assert all(t == pytest.approx(1.5) for t in pred.gdec_T_list)
    for row in pred.cdec_T_lists:
        assert all(t == pytest.approx(1.5) for t in row)


def test_profiler_warmup_and_repeat_call_counts():
    pred = TablePerfPredictor(_VllmCfg(_SchedCfg(), _CacheCfg(), _ModelCfg()))
    measure = _stub_measure_constant(1.0)
    profiler = ModelProfiler(pred, measure, nwarmup=2, nrepeat=3)

    profiler.run()

    n_linr = len(pred.linr_S_list)
    n_pref = len(pred.pref_S_list)
    n_gdec = len(pred.gdec_N_list)
    n_cdec = len(pred.cdec_S_list) * len(pred.cdec_N_list_agg)
    n_lnch = 1
    expected = (n_linr + n_pref + n_gdec + n_cdec + n_lnch) * (2 + 3)
    assert len(measure.calls) == expected


def test_profiler_lnch_override_default():
    pred = TablePerfPredictor(_VllmCfg(_SchedCfg(), _CacheCfg(), _ModelCfg()))
    initial_lnch = pred.get_lnch_T()
    measure = _stub_measure_constant(2.0)
    profiler = ModelProfiler(pred, measure, nwarmup=1, nrepeat=1)

    profiler.run()

    assert pred.get_lnch_T() == pytest.approx(2.0)
    assert pred.get_lnch_T() != initial_lnch


def test_profiler_get_predictions_after_run():
    """End-to-end check: profile, then ask predictor for points, both
    exact-match and interpolated."""
    pred = TablePerfPredictor(_VllmCfg(_SchedCfg(), _CacheCfg(), _ModelCfg()))
    measure = _stub_measure_constant(0.42)
    profiler = ModelProfiler(pred, measure, nwarmup=0, nrepeat=1)

    profiler.run()

    # Exact-match index — returns 0.42
    assert pred.get_linr_T(pred.linr_S_list[0]) == pytest.approx(0.42)
    # Interpolated point — still 0.42 because table is constant
    assert pred.get_linr_T(50) == pytest.approx(0.42)
    # cdec bilinear over a constant table is constant
    assert pred.get_cdec_T(2, 64) == pytest.approx(0.42)
