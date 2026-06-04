# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SUB_201 A2 — PinnedPool loader (engine-side).

The pinned-host ctypes wrapper currently lives under
``shadow_assists/features/IDE_017_dma_zero_copy/SUB_201_A2_kvtier_poc/``.
This thin loader resolves the path lazily so the engine
(``vllm/v1/core/kv_dram_tiering.py``) does not hardcode a relative
shadow_assists path. Order of resolution:

  1. ``VLLM_KV_TIERING_POOL_WRAPPER`` env var (file path to
     ``pinned_pool_wrapper.py``) — for non-standard installs.
  2. The repo-anchored shadow_assists location (PoC default).
  3. Raise ``ImportError`` if neither resolves.

Library file (.so) resolution mirrors the same order via
``VLLM_KV_TIERING_POOL_LIB`` or the shadow_assists ``build/`` dir.

This module is **only** imported when
``VLLM_KV_TIERING_DRAM`` is enabled, so unaffected installs pay zero
cost (file isn't touched at boot).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# Path within the repo to the PoC wrapper. Engine-side anchor: this
# file is at ``vllm/v1/core/_kv_tier_pool_loader.py``. The repo root is
# 4 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_WRAPPER = (
    _REPO_ROOT
    / "shadow_assists"
    / "features"
    / "IDE_017_dma_zero_copy"
    / "SUB_201_A2_kvtier_poc"
    / "pinned_pool_wrapper.py"
)
_DEFAULT_LIB = (
    _REPO_ROOT
    / "shadow_assists"
    / "features"
    / "IDE_017_dma_zero_copy"
    / "build"
    / "libpinned_pool.so"
)


def _resolve_wrapper_path() -> Path:
    env = os.environ.get("VLLM_KV_TIERING_POOL_WRAPPER")
    if env:
        p = Path(env)
        if not p.exists():
            raise ImportError(
                f"VLLM_KV_TIERING_POOL_WRAPPER points to non-existent path: {p}"
            )
        return p
    if not _DEFAULT_WRAPPER.exists():
        raise ImportError(
            f"PinnedPool wrapper not found at default {_DEFAULT_WRAPPER}; "
            "set VLLM_KV_TIERING_POOL_WRAPPER to override"
        )
    return _DEFAULT_WRAPPER


def _resolve_lib_path() -> str | None:
    env = os.environ.get("VLLM_KV_TIERING_POOL_LIB")
    if env:
        return env
    if _DEFAULT_LIB.exists():
        return str(_DEFAULT_LIB)
    return None  # PinnedPool wrapper has its own default fallback


def _load_wrapper_module():
    """Import the pinned_pool_wrapper module from its file path under
    a stable name so re-imports return the same module."""
    mod_name = "vllm._kv_tier_pinned_pool_wrapper"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    wrapper_path = _resolve_wrapper_path()
    spec = importlib.util.spec_from_file_location(mod_name, wrapper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build import spec for {wrapper_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_pinned_pool(
    total_limit_bytes: int,
    numa_node: int = -1,
    lib_path: str | None = None,
):
    """Build and return a PinnedPool instance.

    Raises on any failure (caller — ``try_build_tier`` — turns this
    into a logged warning and the no-op tier path).
    """
    mod = _load_wrapper_module()
    lib_path = lib_path or _resolve_lib_path()
    return mod.PinnedPool(
        total_limit_bytes=total_limit_bytes,
        numa_node=numa_node,
        lib_path=lib_path,
    )
