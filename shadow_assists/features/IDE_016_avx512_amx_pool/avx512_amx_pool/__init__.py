"""IDE_016 — AVX-512 + AMX CPU SIMD Acceleration Pool.

Python package that re-exports the C++ pybind11 module (avx512_amx_pool._core)
together with the high-level Python wrapper at src/_python/avx512_sampling.py.

Usage::

    from avx512_amx_pool import sampling
    token_ids = sampling.fused_sample(logits, k=20, p=0.95, temperature=0.7)

    from avx512_amx_pool import matmul
    C = matmul.amx_matmul(A, matmul.amx_repack_b(B))
"""
from __future__ import annotations

import importlib.util
import os
import sys

# Resolve src/_python/avx512_sampling.py without polluting Python path.
_here = os.path.dirname(__file__)
_wrapper_path = os.path.abspath(
    os.path.join(_here, "..", "src", "_python", "avx512_sampling.py"))

_spec = importlib.util.spec_from_file_location(
    "avx512_amx_pool._wrapper", _wrapper_path)
_wrapper = importlib.util.module_from_spec(_spec)
# Make the wrapper able to import avx512_amx_pool._core
sys.modules["avx512_amx_pool._wrapper"] = _wrapper
_spec.loader.exec_module(_wrapper)   # type: ignore[union-attr]


# Re-export under module namespaces for clarity.
class _SamplingFacade:
    is_available = staticmethod(_wrapper.is_available)
    cpu_has_avx512 = staticmethod(_wrapper.cpu_has_avx512)
    fused_sample = staticmethod(_wrapper.fused_sample)
    topk_topp = staticmethod(_wrapper.topk_topp)
    topk = staticmethod(_wrapper.topk)
    topk_only = staticmethod(_wrapper.topk_only)
    topp_cutoff = staticmethod(_wrapper.topp_cutoff)
    apply_temperature = staticmethod(_wrapper.apply_temperature)
    apply_logit_bias = staticmethod(_wrapper.apply_logit_bias)
    softmax = staticmethod(_wrapper.softmax)
    apply_repetition_penalty = staticmethod(_wrapper.apply_repetition_penalty)
    apply_frequency_penalty = staticmethod(_wrapper.apply_frequency_penalty)
    apply_presence_penalty = staticmethod(_wrapper.apply_presence_penalty)
    _torch_fallback_fused_sample = staticmethod(_wrapper._torch_fallback_fused_sample)
    _torch_fallback_topk_topp = staticmethod(_wrapper._torch_fallback_topk_topp)


class _MatmulFacade:
    amx_is_available = staticmethod(_wrapper.amx_is_available)
    request_amx_permission = staticmethod(_wrapper.request_amx_permission)
    amx_matmul = staticmethod(_wrapper.amx_matmul)
    amx_repack_b = staticmethod(_wrapper.amx_repack_b)


sampling = _SamplingFacade()
matmul = _MatmulFacade()


# ── Tokenizer (SUB_171) ───────────────────────────────────────────
_tok_wrapper_path = os.path.abspath(
    os.path.join(_here, "..", "src", "_python", "tokenizer.py"))
_tok_spec = importlib.util.spec_from_file_location(
    "avx512_amx_pool._tokenizer", _tok_wrapper_path)
_tok = importlib.util.module_from_spec(_tok_spec)
sys.modules["avx512_amx_pool._tokenizer"] = _tok
_tok_spec.loader.exec_module(_tok)   # type: ignore[union-attr]

BatchDetokenizer = _tok.BatchDetokenizer
tokenizer = _tok


# convenience top-level
is_available = _wrapper.is_available
amx_is_available = _wrapper.amx_is_available

__all__ = [
    "sampling", "matmul", "tokenizer", "BatchDetokenizer",
    "is_available", "amx_is_available",
]
