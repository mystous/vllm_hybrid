# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NEO-style asymmetric layer pipeline: layer-offset ping-pong.

Adapted from NEO ``swiftllm/worker/model.py:_forward_pipeline`` and
``swiftllm/worker/layers/transformer_layer.py`` (MLSys 2025,
Apache 2.0). Only the algorithm is reused.

The fundamental observation (NEO §4.4): a single ``forward_double``
processes two sub-batches at *different* layer indices —
``batch[1]`` runs layer i and ``batch[0]`` runs layer i+1. The two
sub-batches stay one layer apart for the entire forward pass, which
breaks the *intra-layer* attention dependency that would otherwise
serialise the GPU and CPU paths.

This module is an abstract *hook layer*: it does not call any vLLM
attention kernels itself. The engine wires concrete callables for

* ``preproj`` — RMSNorm + QKV projection + RoPE
* ``attention`` — actual attention dispatch (3-way: prefill /
  GPU decode / CPU decode)
* ``postproj`` — output projection + residual + FFN

into ``LayerPipelineCallbacks``. Tests can substitute deterministic
stubs to verify the layer-offset and ordering invariants without GPU
or CUDA streams.

Two modes:

* ``forward_sequential(batch)`` — single-batch path (vanilla)
* ``forward_pipeline(batches[0..1])`` — two-batch ping-pong with
  layer offset

See ``shadow_assists/features/IDE_006/NEO_code_deepdive.md`` §4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from vllm.v1.core.sched.sub_batch import SubBatch


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LayerPipelineCallbacks:
    """Function hooks that the engine plugs into the pipeline.

    All callbacks are responsible for any GPU stream coordination they
    need (e.g. recording / waiting on QKV-transfer events). The pipeline
    runner only sequences the calls; it never inspects tensors.

    Signatures use ``Any`` because real call sites pass tensors / model
    weights / sub-batch metadata, while unit tests pass plain values.
    """

    preproj: Callable[[Any, SubBatch, int, int], Any]
    """``preproj(embeddings, batch, layer_idx, layer_off) -> (q, k, v)``.

    ``layer_off`` is 1 when the call site is preparing the *next*
    layer's input as part of the ping-pong (NEO behaviour); 0 for the
    sequential path."""

    attention: Callable[[Any, Any, Any, SubBatch, int], Any]
    """``attention(q, k, v, batch, cur_layer_id) -> output``.

    The 3-way dispatch (prefill / GPU decode / CPU decode) lives inside
    this callback."""

    postproj: Callable[[Any, SubBatch, int], Any]
    """``postproj(attention_output, batch, layer_idx) -> next_embeddings``."""


# ----------------------------------------------------------------------
# Pipeline runner
# ----------------------------------------------------------------------
class SubBatchPipelineExecutor:
    """Drive the NEO layer pipeline for one or two sub-batches.

    Parameters
    ----------
    num_layers
        Number of transformer layers in the model (``L``).
    callbacks
        Engine-supplied hooks (see ``LayerPipelineCallbacks``).
    """

    def __init__(
        self,
        *,
        num_layers: int,
        callbacks: LayerPipelineCallbacks,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_layers = num_layers
        self.cb = callbacks

    # ------------------------------------------------------------------
    # Sequential path — single sub-batch
    # ------------------------------------------------------------------
    def forward_sequential(self, batch: SubBatch, embeddings: Any) -> Any:
        """Run all ``num_layers`` layers for a single sub-batch."""
        x = embeddings
        for layer_idx in range(self.num_layers):
            q, k, v = self.cb.preproj(x, batch, layer_idx, 0)
            attn_out = self.cb.attention(q, k, v, batch, layer_idx)
            x = self.cb.postproj(attn_out, batch, layer_idx)
        return x

    # ------------------------------------------------------------------
    # Pipelined path — two sub-batches with layer offset
    # ------------------------------------------------------------------
    def forward_first_stage(
        self,
        batches: Sequence[SubBatch],
        embeddings: Sequence[Any],
    ) -> tuple[Any, Any, Any]:
        """Prepare the pipeline:

        1. Run batch[0]'s pre-projection + attention for layer 0
           (and post-projection-to-next-layer-input).
        2. Compute batch[1]'s layer-0 (q, k, v) so that the main loop
           can begin its first ``forward_double`` with batch[1] already
           one stage ahead.
        """
        if len(batches) != 2:
            raise ValueError("forward_first_stage requires exactly 2 batches")
        if len(embeddings) != 2:
            raise ValueError("forward_first_stage requires 2 embeddings")

        # batch[0] runs layer 0 immediately.
        q0, k0, v0 = self.cb.preproj(embeddings[0], batches[0], 0, 0)
        attn0 = self.cb.attention(q0, k0, v0, batches[0], 0)
        next_emb0 = self.cb.postproj(attn0, batches[0], 0)
        # We will hand next_emb0 into batch[0]'s layer-1 preproj inside
        # the main loop (cur_stage=1), so we keep it on a side channel.
        # We expose it via the second slot of the tuple below; the
        # forward_pipeline orchestration code threads it through.
        # batch[1] only computes layer-0 q/k/v at this stage.
        q1, k1, v1 = self.cb.preproj(embeddings[1], batches[1], 0, 0)
        return (q1, k1, v1), next_emb0

    def forward_double(
        self,
        batches: Sequence[SubBatch],
        layer_idx: int,
        q1: Any,
        k1: Any,
        v1: Any,
        next_emb0: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """One iteration of the layer ping-pong (NEO §4.4).

        Stage 0:
            * batch[1] runs ``attention`` at layer ``layer_idx``
              (consuming q1/k1/v1).
            * batch[0] runs ``preproj`` for layer ``layer_idx + 1``
              (consuming ``next_emb0``).
        Stage 1:
            * batch[0] runs ``attention`` at layer ``layer_idx + 1``
              (consuming the q/k/v just produced).
            * batch[1] runs ``postproj`` for layer ``layer_idx`` and
              ``preproj`` for layer ``layer_idx + 1``.

        Returns ``(q1', k1', v1', next_emb0')`` for the next call.
        """
        # Stage 0
        attn1 = self.cb.attention(q1, k1, v1, batches[1], layer_idx)
        # Use layer_off=0 because next_emb0 represents the *output* of
        # batch[0]'s previous layer; the upcoming preproj is for that
        # layer's successor (which is layer_idx + 1).
        q0_next, k0_next, v0_next = self.cb.preproj(
            next_emb0, batches[0], layer_idx + 1, 0
        )

        # Stage 1
        attn0_next = self.cb.attention(q0_next, k0_next, v0_next,
                                       batches[0], layer_idx + 1)
        next_emb0_new = self.cb.postproj(attn0_next, batches[0],
                                         layer_idx + 1)
        emb1 = self.cb.postproj(attn1, batches[1], layer_idx)
        q1_new, k1_new, v1_new = self.cb.preproj(emb1, batches[1],
                                                 layer_idx + 1, 0)
        return q1_new, k1_new, v1_new, next_emb0_new

    def forward_last_stage(
        self,
        batches: Sequence[SubBatch],
        q1: Any,
        k1: Any,
        v1: Any,
        next_emb0: Any,
    ) -> Any:
        """Drain the pipeline.

        At the start of this call:
            * batch[0]'s layer ``L-1`` post-projection has just produced
              ``next_emb0`` — which is the model's final hidden state
              for batch[0].
            * batch[1] has q/k/v for layer ``L-1`` waiting to be run.
        """
        last = self.num_layers - 1
        attn1 = self.cb.attention(q1, k1, v1, batches[1], last)
        emb1 = self.cb.postproj(attn1, batches[1], last)
        return next_emb0, emb1

    # ------------------------------------------------------------------
    # High-level entry
    # ------------------------------------------------------------------
    def forward_pipeline(
        self,
        batches: Sequence[SubBatch],
        embeddings: Sequence[Any],
    ) -> tuple[Any, Any]:
        """Run the full asymmetric pipeline for two sub-batches."""
        if self.num_layers < 2:
            raise ValueError("forward_pipeline requires num_layers >= 2")

        (q1, k1, v1), next_emb0 = self.forward_first_stage(batches, embeddings)
        # forward_double iterates from layer 0 onwards; each call advances
        # batch[1] by one layer (its current layer becomes the "i" in the
        # docstring) and batch[0] by one layer (becomes "i+1").
        # We need num_layers - 1 calls total: after the first_stage,
        # batch[0] is at layer 1 ready and batch[1] is at layer 0 ready.
        # Each forward_double brings batch[1] up by one layer (so it
        # finishes at layer L-2) and batch[0] up by one (so it finishes
        # at layer L-1).
        # Each forward_double advances batch[1] by one layer (running its
        # ``layer_idx`` attention) and batch[0] by one layer (running its
        # ``layer_idx + 1`` attention). After ``L - 1`` calls,
        # batch[1] has attended layers 0..L-2 and batch[0] has attended
        # layers 0..L-1. ``forward_last_stage`` then runs batch[1]'s
        # final layer (L-1).
        for layer_idx in range(0, self.num_layers - 1):
            q1, k1, v1, next_emb0 = self.forward_double(
                batches, layer_idx, q1, k1, v1, next_emb0
            )
        # After the loop:
        #   - batch[1] q1/k1/v1 are for layer L-2 attention
        #   - next_emb0 is batch[0]'s post layer L-1 output (final)
        # Run the last stage to compute batch[1] layer L-1 attention.
        return self.forward_last_stage(batches, q1, k1, v1, next_emb0)
