# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk cache for NEO ``profile_neo_predictor`` results — TSK_017 Step 1.7.

The profiler RPC (``GPUModelRunner.profile_neo_predictor``) measures
~60 forward calls and runs to ~21s on Llama-3.3-70B + TP=8 / ~4s on
Qwen2.5-1.5B + TP=1 / RTX 3090. Each engine startup currently pays
this cost. Since the measurement is deterministic for a fixed
``(model, TP, dtype, max_num_seqs, max_num_batched_tokens, block_size)``
tuple, persisting the resulting dict to disk lets subsequent startups
skip the RPC entirely.

Public API
----------
* ``compute_cache_key(vllm_config) -> str`` — hex digest derived from
  the binding config knobs that affect the measurement.
* ``load(vllm_config) -> dict | None`` — return the cached
  ``profile_data`` dict on hit, ``None`` on miss / corrupt entry.
* ``save(vllm_config, profile_data) -> Path | None`` — write the
  dict atomically. Returns the file path on success, ``None`` on
  silent failure (cache is best-effort — never fails startup).

The cache lives at ``~/.cache/vllm/neo_predictor/<key>.json`` by
default. Override with the ``VLLM_NEO_PREDICTOR_CACHE_DIR`` env var.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

# Bumped when the on-disk schema changes. Old entries are silently
# ignored (treated as miss) so a downgrade or upgrade never hangs
# startup on a corrupt cache.
# v2 (TSK_019 SUB_016) — cdec_T_pairs (2D grid) added.
_SCHEMA_VERSION = 2

# Required keys in the profile_data dict. A cache entry missing any
# of them is treated as miss (forces a fresh measurement).
_REQUIRED_PROFILE_KEYS = ("linr_T_pairs", "pref_T_pairs",
                          "gdec_T_pairs", "cdec_T_pairs", "lnch_T")


def _cache_dir() -> Path:
    override = os.environ.get("VLLM_NEO_PREDICTOR_CACHE_DIR")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "vllm" / "neo_predictor"


def _key_meta(vllm_config: VllmConfig) -> dict[str, Any]:
    """Extract the binding config knobs that determine the measurement.

    Anything that changes ``ModelProfiler`` 's S/N grids or the dummy
    forward shape must be captured here. Keep the field names stable
    across versions — schema bumps are explicit via ``_SCHEMA_VERSION``.
    """
    sched = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    return {
        "model": model.model,
        "dtype": str(model.dtype),
        "max_model_len": model.max_model_len,
        "tp": parallel.tensor_parallel_size,
        "pp": parallel.pipeline_parallel_size,
        "max_num_seqs": sched.max_num_seqs,
        "max_num_batched_tokens": sched.max_num_batched_tokens,
        "block_size": cache.block_size,
    }


def compute_cache_key(vllm_config: VllmConfig) -> str:
    meta = _key_meta(vllm_config)
    payload = json.dumps(meta, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _entry_path(vllm_config: VllmConfig) -> Path:
    return _cache_dir() / f"{compute_cache_key(vllm_config)}.json"


def _is_valid_profile_data(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    return all(k in data for k in _REQUIRED_PROFILE_KEYS)


def load(vllm_config: VllmConfig) -> dict | None:
    """Read the cached profile dict if present + valid + schema-matched."""
    path = _entry_path(vllm_config)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("NEO predictor cache: failed to read %s (%s); "
                       "treating as miss", path, e)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != _SCHEMA_VERSION:
        logger.info("NEO predictor cache: schema mismatch at %s "
                    "(have=%r, want=%r); miss", path,
                    payload.get("version"), _SCHEMA_VERSION)
        return None
    profile = payload.get("profile_data")
    if not _is_valid_profile_data(profile):
        logger.warning("NEO predictor cache: malformed entry %s; miss", path)
        return None
    # JSON encodes (S, ms) tuples as [S, ms] lists. Convert back so
    # the adapter's downstream consumers see the original shape.
    # cdec is 3-tuple (S, N, ms).
    for k in ("linr_T_pairs", "pref_T_pairs", "gdec_T_pairs",
              "cdec_T_pairs"):
        if k in profile:
            profile[k] = [tuple(p) for p in profile[k]]
    logger.info("NEO predictor cache: HIT %s", path)
    return profile


def save(vllm_config: VllmConfig, profile_data: dict) -> Path | None:
    """Persist ``profile_data`` atomically. Returns the path or ``None``
    on silent failure (the cache is best-effort and must never break
    engine startup)."""
    if not _is_valid_profile_data(profile_data):
        logger.debug("NEO predictor cache: skip save (invalid profile_data)")
        return None
    path = _entry_path(vllm_config)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _SCHEMA_VERSION,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "key_meta": _key_meta(vllm_config),
            "profile_data": profile_data,
        }
        # Atomic write — open NamedTemporaryFile in the same dir then
        # os.replace, so a crash mid-write never leaves a partial file.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(path.parent),
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as tmp:
            json.dump(payload, tmp, separators=(",", ":"))
            tmp_name = tmp.name
        os.replace(tmp_name, path)
    except OSError as e:
        logger.warning("NEO predictor cache: failed to save %s (%s)", path, e)
        return None
    logger.info("NEO predictor cache: SAVED %s", path)
    return path
