# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TSK_017 — NEO PerfPredictor + disk cache (Step 1.8).

Coverage
--------
* ``ZeroPerfPredictor`` — always returns 0.0 (fallback path).
* ``TablePerfPredictor`` — 1D linear interpolation accuracy across
  exact-match / interior / clamp boundaries.
* ``TablePerfPredictor`` — 2D bilinear interpolation accuracy
  for ``get_cdec_T(S, N)``.
* ``neo_perfpredictor_cache`` — save / load roundtrip; key stability
  under config equivalence; corrupt entry handling; schema version.

See ``shadow_assists/features/IDE_006/TSK_017.md`` Step 1.7 / 1.8.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from vllm.v1.core.sched.perfpredictor import (
    TablePerfPredictor,
    ZeroPerfPredictor,
)


# ----------------------------------------------------------------------
# Shared minimal VllmConfig stub — lighter than constructing the real
# one (which pulls model/cache/parallel/scheduler dataclasses).
# ----------------------------------------------------------------------
@dataclass
class _StubModel:
    model: str = "test/dummy-1b"
    dtype: str = "bfloat16"
    max_model_len: int = 256


@dataclass
class _StubCache:
    block_size: int = 16


@dataclass
class _StubParallel:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1


@dataclass
class _StubSched:
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 256


@dataclass
class _StubVllmConfig:
    model_config: _StubModel
    cache_config: _StubCache
    parallel_config: _StubParallel
    scheduler_config: _StubSched


def _mini_vllm_config(**overrides) -> _StubVllmConfig:
    cfg = _StubVllmConfig(
        model_config=_StubModel(),
        cache_config=_StubCache(),
        parallel_config=_StubParallel(),
        scheduler_config=_StubSched(),
    )
    for k, v in overrides.items():
        section, _, attr = k.partition(".")
        setattr(getattr(cfg, section), attr, v)
    return cfg


# ----------------------------------------------------------------------
# ZeroPerfPredictor — fallback returns 0
# ----------------------------------------------------------------------
def test_zero_predictor_returns_zero_for_all_apis():
    p = ZeroPerfPredictor()
    assert p.get_linr_T(128) == 0.0
    assert p.get_pref_T(2048) == 0.0
    assert p.get_gdec_T(4096) == 0.0
    assert p.get_cdec_T(8, 4096) == 0.0
    assert p.get_lnch_T() == 0.0


# ----------------------------------------------------------------------
# TablePerfPredictor — 1D linear interpolation
# ----------------------------------------------------------------------
def _make_seeded_predictor() -> TablePerfPredictor:
    """Construct a predictor + populate the four 1D tables manually
    with a synthetic linear function ``y = 2*x + 1`` so interpolation
    is checkable against the closed-form."""
    cfg = _mini_vllm_config()
    p = TablePerfPredictor(cfg)
    # Replace the fixed grids with a tiny known set so the interp
    # math is trivially verifiable.
    p.linr_S_list = [1, 4, 16, 64]
    p.linr_T_list = [2 * s + 1 for s in p.linr_S_list]
    p.linr_S_lb_idx = TablePerfPredictor._get_lb_idx_list(p.linr_S_list)

    p.pref_S_list = [4, 16, 64]
    p.pref_T_list = [2 * s + 1 for s in p.pref_S_list]
    p.pref_S_lb_idx = TablePerfPredictor._get_lb_idx_list(p.pref_S_list)

    p.gdec_N_list = [16, 256, 4096]
    p.gdec_T_list = [s * 0.001 for s in p.gdec_N_list]
    p.gdec_N_lb_idx = TablePerfPredictor._get_lb_idx_list(p.gdec_N_list)

    p.lnch_T = 0.42
    return p


def test_table_predictor_exact_match_returns_table_value():
    p = _make_seeded_predictor()
    # exact match at every grid point
    for s, expected in zip(p.linr_S_list, p.linr_T_list):
        assert p.get_linr_T(s) == pytest.approx(expected)


def test_table_predictor_linear_interp_midpoint():
    p = _make_seeded_predictor()
    # x=2 sits between (1, 3) and (4, 9). Linear interp:
    # 3 + (9-3) * (2-1)/(4-1) = 3 + 2 = 5
    assert p.get_linr_T(2) == pytest.approx(5.0)
    # x=8 between (4, 9) and (16, 33). Interp:
    # 9 + (33-9) * (8-4)/(16-4) = 9 + 24*4/12 = 9 + 8 = 17
    assert p.get_linr_T(8) == pytest.approx(17.0)


def test_table_predictor_clamps_above_max_to_table_max():
    p = _make_seeded_predictor()
    # x>max → clamped to xs[-1] = 64 → ys[-1] = 129
    assert p.get_linr_T(1000) == pytest.approx(129.0)


def test_table_predictor_zero_or_negative_input_returns_zero():
    p = _make_seeded_predictor()
    assert p.get_linr_T(0) == 0.0
    assert p.get_linr_T(-5) == 0.0


def test_table_predictor_unpopulated_table_returns_zero():
    cfg = _mini_vllm_config()
    p = TablePerfPredictor(cfg)
    # T_list is None until ModelProfiler.run() — interp must short-circuit.
    assert p.linr_T_list is None
    assert p.get_linr_T(128) == 0.0


def test_table_predictor_lnch_constant():
    p = _make_seeded_predictor()
    assert p.get_lnch_T() == 0.42


# ----------------------------------------------------------------------
# TablePerfPredictor — 2D bilinear cdec_T
# ----------------------------------------------------------------------
def test_table_predictor_cdec_2d_interpolation():
    """Seed the cdec table with a known bilinear surface
    ``f(S, N) = S + 0.001 * N`` and check interior + edge cases."""
    cfg = _mini_vllm_config()
    p = TablePerfPredictor(cfg)
    p.cdec_S_list = [1, 8]
    p.cdec_N_list_agg = [16, 256]
    p.cdec_T_lists = [
        [1 + 0.001 * 16,   1 + 0.001 * 256],          # S=1
        [8 + 0.001 * 16,   8 + 0.001 * 256],          # S=8
    ]
    p.cdec_S_lb_idx = TablePerfPredictor._get_lb_idx_list(p.cdec_S_list)
    p.cdec_N_lb_idx = TablePerfPredictor._get_lb_idx_list(p.cdec_N_list_agg)

    # exact corner
    assert p.get_cdec_T(1, 16) == pytest.approx(1 + 0.001 * 16)
    assert p.get_cdec_T(8, 256) == pytest.approx(8 + 0.001 * 256)
    # interior — bilinear at (S=4, N=128)
    # interp along N first per S row, then along S between the two rows
    # row S=1 at N=128: 1 + 0.001*16 + (1+0.001*256 - 1-0.001*16) * (128-16)/(256-16)
    #                 = 1.016 + 0.24 * 112/240 = 1.016 + 0.112 = 1.128
    # row S=8 at N=128: 8.016 + 0.24 * 112/240 = 8.128
    # then S interp: 1.128 + (8.128 - 1.128) * (4-1)/(8-1) = 1.128 + 3.0 = 4.128
    assert p.get_cdec_T(4, 128) == pytest.approx(4.128)


def test_table_predictor_cdec_zero_inputs_short_circuit():
    cfg = _mini_vllm_config()
    p = TablePerfPredictor(cfg)
    assert p.get_cdec_T(0, 100) == 0.0
    assert p.get_cdec_T(5, 0) == 0.0
    assert p.get_cdec_T(-1, 100) == 0.0


# ----------------------------------------------------------------------
# Disk cache — save / load roundtrip
# ----------------------------------------------------------------------
def _sample_profile_data() -> dict:
    return {
        "linr_T_pairs": [(1, 0.42), (16, 1.5), (256, 8.0)],
        "pref_T_pairs": [(16, 0.5), (256, 4.0), (1024, 16.0)],
        "gdec_T_pairs": [(16, 0.05), (4096, 1.0)],
        "lnch_T": 0.8,
    }


def test_cache_key_deterministic_for_same_config(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_NEO_PREDICTOR_CACHE_DIR", str(tmp_path))
    from vllm.v1.core.sched import neo_perfpredictor_cache as ppc
    cfg_a = _mini_vllm_config()
    cfg_b = _mini_vllm_config()
    assert ppc.compute_cache_key(cfg_a) == ppc.compute_cache_key(cfg_b)


def test_cache_key_changes_when_binding_field_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_NEO_PREDICTOR_CACHE_DIR", str(tmp_path))
    from vllm.v1.core.sched import neo_perfpredictor_cache as ppc
    base = _mini_vllm_config()
    base_key = ppc.compute_cache_key(base)
    # different model
    other = _mini_vllm_config(**{"model_config.model": "test/other-7b"})
    assert ppc.compute_cache_key(other) != base_key
    # different TP
    tp4 = _mini_vllm_config(**{"parallel_config.tensor_parallel_size": 4})
    assert ppc.compute_cache_key(tp4) != base_key
    # different block_size
    bs8 = _mini_vllm_config(**{"cache_config.block_size": 8})
    assert ppc.compute_cache_key(bs8) != base_key


def test_cache_load_miss_when_file_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_NEO_PREDICTOR_CACHE_DIR", str(tmp_path))
    from vllm.v1.core.sched import neo_perfpredictor_cache as ppc
    cfg = _mini_vllm_config()
    assert ppc.load(cfg) is None


def test_cache_save_then_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_NEO_PREDICTOR_CACHE_DIR", str(tmp_path))
    from vllm.v1.core.sched import neo_perfpredictor_cache as ppc
    cfg = _mini_vllm_config()
    profile = _sample_profile_data()
    saved = ppc.save(cfg, profile)
    assert saved is not None and saved.is_file()

    loaded = ppc.load(cfg)
    assert loaded is not None
    # JSON roundtrip preserves values; pairs come back as tuples (cache
    # internally normalizes ``[S, ms]`` → ``(S, ms)``).
    assert loaded["linr_T_pairs"] == profile["linr_T_pairs"]
    assert loaded["pref_T_pairs"] == profile["pref_T_pairs"]
    assert loaded["gdec_T_pairs"] == profile["gdec_T_pairs"]
    assert loaded["lnch_T"] == profile["lnch_T"]


def test_cache_save_invalid_profile_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_NEO_PREDICTOR_CACHE_DIR", str(tmp_path))
    from vllm.v1.core.sched import neo_perfpredictor_cache as ppc
    cfg = _mini_vllm_config()
    # missing required keys → silent skip
    assert ppc.save(cfg, {"linr_T_pairs": []}) is None


def test_cache_load_corrupt_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_NEO_PREDICTOR_CACHE_DIR", str(tmp_path))
    from vllm.v1.core.sched import neo_perfpredictor_cache as ppc
    cfg = _mini_vllm_config()
    path = tmp_path / f"{ppc.compute_cache_key(cfg)}.json"
    path.write_text("{not-valid-json")
    assert ppc.load(cfg) is None


def test_cache_load_schema_mismatch_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_NEO_PREDICTOR_CACHE_DIR", str(tmp_path))
    from vllm.v1.core.sched import neo_perfpredictor_cache as ppc
    cfg = _mini_vllm_config()
    path = tmp_path / f"{ppc.compute_cache_key(cfg)}.json"
    path.write_text(json.dumps({
        "version": 99999,
        "profile_data": _sample_profile_data(),
    }))
    assert ppc.load(cfg) is None


def test_cache_isolation_per_key(tmp_path, monkeypatch):
    """Saving against config A must not affect lookup for config B."""
    monkeypatch.setenv("VLLM_NEO_PREDICTOR_CACHE_DIR", str(tmp_path))
    from vllm.v1.core.sched import neo_perfpredictor_cache as ppc
    cfg_a = _mini_vllm_config()
    cfg_b = _mini_vllm_config(**{"model_config.model": "test/other-7b"})
    ppc.save(cfg_a, _sample_profile_data())
    assert ppc.load(cfg_b) is None
    assert ppc.load(cfg_a) is not None


def test_cache_atomic_write_no_partial_file(tmp_path, monkeypatch):
    """After a successful save, only the final entry exists — no
    leftover .tmp files (atomic via os.replace)."""
    monkeypatch.setenv("VLLM_NEO_PREDICTOR_CACHE_DIR", str(tmp_path))
    from vllm.v1.core.sched import neo_perfpredictor_cache as ppc
    cfg = _mini_vllm_config()
    ppc.save(cfg, _sample_profile_data())
    leftover = list(Path(tmp_path).glob("*.tmp"))
    assert leftover == []
