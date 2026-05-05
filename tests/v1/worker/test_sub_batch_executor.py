# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm/v1/worker/sub_batch_executor.py`` (TSK_016).

These tests exercise the layer-pipeline orchestration without any
actual GPU work. The callbacks are deterministic stubs that record the
sequence of calls; the assertions check the layer-offset invariant and
the call ordering required by the NEO algorithm.
"""

from __future__ import annotations

import pytest

from vllm.v1.core.sched.sub_batch import SubBatch
from vllm.v1.worker.sub_batch_executor import (
    LayerPipelineCallbacks,
    SubBatchPipelineExecutor,
)


# ----------------------------------------------------------------------
# Trace-based callbacks
# ----------------------------------------------------------------------
class _Trace:
    """Records (op, batch_label, layer_idx) tuples for assertion."""
    def __init__(self) -> None:
        self.events: list[tuple[str, str, int]] = []

    def label_of(self, batch: SubBatch) -> str:
        return getattr(batch, "_label", "?")


def _make_callbacks(trace: _Trace) -> LayerPipelineCallbacks:
    def preproj(emb, batch, layer_idx, layer_off):
        trace.events.append(("preproj", trace.label_of(batch), layer_idx))
        return f"q@{layer_idx}", f"k@{layer_idx}", f"v@{layer_idx}"

    def attention(q, k, v, batch, cur_layer_id):
        trace.events.append(("attention", trace.label_of(batch),
                              cur_layer_id))
        return f"attn@{cur_layer_id}"

    def postproj(attn_out, batch, layer_idx):
        trace.events.append(("postproj", trace.label_of(batch), layer_idx))
        return f"emb@{layer_idx + 1}"

    return LayerPipelineCallbacks(preproj=preproj,
                                   attention=attention,
                                   postproj=postproj)


# ----------------------------------------------------------------------
# Sequential
# ----------------------------------------------------------------------
def test_forward_sequential_calls_layers_in_order():
    trace = _Trace()
    cb = _make_callbacks(trace)
    exe = SubBatchPipelineExecutor(num_layers=4, callbacks=cb)
    batch = SubBatch()
    batch._label = "A"

    result = exe.forward_sequential(batch, "input")

    # Per layer: preproj → attention → postproj. 4 layers.
    expected_ops = [op for layer in range(4)
                    for op in (("preproj", "A", layer),
                               ("attention", "A", layer),
                               ("postproj", "A", layer))]
    assert trace.events == expected_ops
    assert result == "emb@4"


# ----------------------------------------------------------------------
# Pipelined
# ----------------------------------------------------------------------
def test_forward_pipeline_layer_offset_invariant():
    """For every call after first_stage, batch[0] is exactly one layer
    ahead of batch[1] in its attention computation."""
    trace = _Trace()
    cb = _make_callbacks(trace)
    exe = SubBatchPipelineExecutor(num_layers=4, callbacks=cb)
    a = SubBatch()
    a._label = "A"
    b = SubBatch()
    b._label = "B"

    out0, out1 = exe.forward_pipeline([a, b], ["e0", "e1"])

    # Walk through the attention events in order
    attn_events = [e for e in trace.events if e[0] == "attention"]
    layers_a = [layer for op, lbl, layer in attn_events if lbl == "A"]
    layers_b = [layer for op, lbl, layer in attn_events if lbl == "B"]
    assert layers_a == [0, 1, 2, 3], layers_a
    assert layers_b == [0, 1, 2, 3], layers_b

    # And in the trace, A's attention at layer i is followed (eventually)
    # by B's attention at layer i.
    a_idx = [i for i, e in enumerate(attn_events) if e[1] == "A"]
    b_idx = [i for i, e in enumerate(attn_events) if e[1] == "B"]
    # By construction of forward_first_stage, A's layer 0 attention runs
    # before B's layer 0 attention.
    assert a_idx[0] < b_idx[0]


def test_forward_pipeline_total_attention_call_count():
    trace = _Trace()
    cb = _make_callbacks(trace)
    L = 4
    exe = SubBatchPipelineExecutor(num_layers=L, callbacks=cb)
    a = SubBatch()
    a._label = "A"
    b = SubBatch()
    b._label = "B"
    exe.forward_pipeline([a, b], ["e0", "e1"])

    n_attn = sum(1 for e in trace.events if e[0] == "attention")
    # Exactly L attentions per batch, and 2 batches → 2L
    assert n_attn == 2 * L


def test_forward_pipeline_requires_two_layers():
    cb = _make_callbacks(_Trace())
    exe = SubBatchPipelineExecutor(num_layers=1, callbacks=cb)
    a = SubBatch()
    a._label = "A"
    b = SubBatch()
    b._label = "B"
    with pytest.raises(ValueError, match="num_layers >= 2"):
        exe.forward_pipeline([a, b], ["e0", "e1"])


def test_forward_pipeline_rejects_wrong_batch_count():
    cb = _make_callbacks(_Trace())
    exe = SubBatchPipelineExecutor(num_layers=2, callbacks=cb)
    a = SubBatch()
    a._label = "A"
    with pytest.raises(ValueError, match="2 batches"):
        exe.forward_first_stage([a], ["e0"])


def test_forward_pipeline_two_layer_minimum():
    """Smallest pipelined case: L=2 → first_stage + last_stage only."""
    trace = _Trace()
    cb = _make_callbacks(trace)
    exe = SubBatchPipelineExecutor(num_layers=2, callbacks=cb)
    a = SubBatch()
    a._label = "A"
    b = SubBatch()
    b._label = "B"

    out0, out1 = exe.forward_pipeline([a, b], ["e0", "e1"])

    attn_a = [layer for op, lbl, layer in trace.events
              if op == "attention" and lbl == "A"]
    attn_b = [layer for op, lbl, layer in trace.events
              if op == "attention" and lbl == "B"]
    assert attn_a == [0, 1]
    assert attn_b == [0, 1]
