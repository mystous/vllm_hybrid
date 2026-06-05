# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SUB_201 / B2(jump_forward) — unit tests for the xgrammar jump-forward
decoding integration added in vllm.v1.structured_output.backend_xgrammar.

These tests do **not** spin up an engine; they exercise the wrapper
``XgrammarGrammar.try_jump_forward()`` directly against a real
``xgrammar.GrammarCompiler`` + a real HF tokenizer. The integration with
``EngineCore.step`` is exercised in the e2e measurement phase.
"""
from __future__ import annotations

import json

import pytest

xgr = pytest.importorskip("xgrammar")
transformers = pytest.importorskip("transformers")
from transformers import AutoTokenizer  # noqa: E402  (after importorskip)

from vllm.v1.structured_output.backend_xgrammar import (  # noqa: E402
    _JF_TOKENIZER_BY_COMPILER,
    XgrammarGrammar,
)


JSON_SCHEMA_5KEY = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
        "city": {"type": "string"},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "email", "city", "active"],
}


@pytest.fixture(scope="module")
def tokenizer():
    # gpt2 is small and ubiquitous; cached on B200 dev env.
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def compiler(tokenizer):
    tinfo = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=50257)
    return xgr.GrammarCompiler(tinfo)


def _grammar(compiler, tokenizer, schema_dict):
    ctx = compiler.compile_json_schema(
        json.dumps(schema_dict), any_whitespace=False
    )
    matcher = xgr.GrammarMatcher(ctx)
    # Register tokenizer for this compiler id (same path XgrammarBackend uses).
    _JF_TOKENIZER_BY_COMPILER[id(compiler)] = tokenizer
    return XgrammarGrammar(
        vocab_size=50257,
        matcher=matcher,
        ctx=ctx,
        _compiler_id=id(compiler),
    )


def test_initial_jfs_returns_object_open_brace(compiler, tokenizer):
    """At grammar start the schema forces `{"name": "` deterministically."""
    g = _grammar(compiler, tokenizer, JSON_SCHEMA_5KEY)
    ids = g.try_jump_forward()
    assert ids, "expected non-empty JFS at schema start"
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    assert decoded == '{"name": "', (
        f"JFS roundtrip mismatch: got {decoded!r}"
    )


def test_try_jump_forward_advances_matcher(compiler, tokenizer):
    """Once the JFS is consumed, the next call must return empty until
    free-form value characters are accepted."""
    g = _grammar(compiler, tokenizer, JSON_SCHEMA_5KEY)
    first = g.try_jump_forward()
    assert first
    # Without consuming a value, no further deterministic JFS is available.
    again = g.try_jump_forward()
    assert again == [], f"second call should be empty, got {again}"


def test_no_jump_forward_when_terminated(compiler, tokenizer):
    """A trivial single-int schema; after we consume `1` the matcher cannot
    be terminated without the stop token, but try_jump_forward must not
    raise and must be safe to call on already-terminated state."""
    g = _grammar(compiler, tokenizer, {"type": "integer"})
    # Pretend the matcher is terminated.
    g._is_terminated = True
    assert g.try_jump_forward() == []


def test_byte_equivalence_round_trip(compiler, tokenizer):
    """The JFS bytes returned by xgrammar must match tokenizer.decode of
    the token-id list we emit — otherwise we would silently corrupt the
    request output. The wrapper validates this and returns [] on mismatch."""
    g = _grammar(compiler, tokenizer, JSON_SCHEMA_5KEY)
    ids = g.try_jump_forward()
    assert ids
    decoded_bytes = tokenizer.decode(ids, skip_special_tokens=False).encode(
        "utf-8"
    )
    assert decoded_bytes == b'{"name": "'


def test_jump_forward_does_not_overshoot_matcher(compiler, tokenizer):
    """Advancing the matcher by the JFS must keep it in a state where the
    *next* logically forced character (a free-form value byte) is still
    accepted — i.e. we have not over-consumed."""
    g = _grammar(compiler, tokenizer, JSON_SCHEMA_5KEY)
    g.try_jump_forward()
    # After `{"name": "` the next byte should be a string-body byte, e.g. 'A'.
    assert g.matcher.accept_string("Alice"), (
        "matcher refused 'Alice' after consuming JFS — jump-forward overshot"
    )


def test_module_cache_populated(compiler, tokenizer):
    """The wrapper relies on a module-level dict that maps compiler id ->
    tokenizer. The fixture registers it explicitly; the production path does
    so in XgrammarBackend.__post_init__."""
    _grammar(compiler, tokenizer, JSON_SCHEMA_5KEY)
    assert id(compiler) in _JF_TOKENIZER_BY_COMPILER
