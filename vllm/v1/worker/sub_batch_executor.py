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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.v1.core.sched.sub_batch import SubBatch


def _noop_transfer(q: Any, k: Any, v: Any, batch: SubBatch,
                   layer_idx: int) -> None:
    """TSK_019 plan B1 — default no-op transfer callback. cdec dispatch
    의 H2D transfer 가 attention callback 안 implicit 으로 처리되는
    현재 구현에서는 별도 동작 X. NEO 원본 `_transfer_qkv` 동등 callback
    contract 만 적재."""
    return None


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

    transfer: Callable[[Any, Any, Any, SubBatch, int], Any] = field(
        default=_noop_transfer
    )
    """TSK_019 plan B1 — ``transfer(q, k, v, batch, layer_idx) -> None``.

    NEO `swiftllm/worker/layers/transformer_layer.py:158-178`
    `_transfer_qkv` 동등. preproj 직후 / attention 직전 별 stage 로
    cdec H2D 시작 — attention callback 이 *별도 host fence* 없이 GPU
    forward launch 가능. 본 callback 의 default 는 no-op (현재
    attention.py 의 cdec dispatch 가 implicit transfer 처리). model 별
    wiring 시점에 explicit transfer 할당."""


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
        # S7 (NEO 원본 rewrite plan G) — 별 stream 영역 (s0/s1) 제거.
        # NEO 원본 forward_pipeline 은 default stream + cpu_communication_stream
        # (1 개 추가) 만 사용. s0/s1 priority 효과 (Phase 3.3) 영역 제거.
        # `_get_batch_streams` dead code 정리.

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
        # IDE_006 async cdec — drain attn0's pending cdec before postproj.
        from vllm.model_executor.layers.attention.attention import (
            _neo_drain_pending_cdec as _drain_pending,
        )
        _drain_pending()
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
        # S8 (NEO 원본 rewrite plan G) — _forward_pipeline_stage(cur_stage) 패턴 정합.
        # NEO 원본 `swiftllm/worker/layers/transformer_layer.py:_forward_pipeline_stage`
        # ordering: batches[cur_stage] 의 postproj+preproj 가 batches[other] 의
        # attention 보다 *먼저* 실행 → batches[other] attention 시점에 batches[cur_stage]
        # 의 모든 work 이미 끝남 → batch interleave 의 wall hide 정합.

        # ── Stage 0 (cur_stage=0, other=1) ──
        #   transfer(b1) → preproj(b0, next layer) → attention(b1, this layer)
        self.cb.transfer(q1, k1, v1, batches[1], layer_idx)
        q0_next, k0_next, v0_next = self.cb.preproj(
            next_emb0, batches[0], layer_idx + 1, 0
        )
        attn1 = self.cb.attention(q1, k1, v1, batches[1], layer_idx)

        # ── Stage 1 (cur_stage=1, other=0) ──
        #   transfer(b0) → postproj(b1, this) + preproj(b1, next) → attention(b0, next)
        self.cb.transfer(q0_next, k0_next, v0_next, batches[0], layer_idx + 1)
        emb1 = self.cb.postproj(attn1, batches[1], layer_idx)
        q1_new, k1_new, v1_new = self.cb.preproj(
            emb1, batches[1], layer_idx + 1, 0
        )
        attn0_next = self.cb.attention(
            q0_next, k0_next, v0_next, batches[0], layer_idx + 1
        )
        next_emb0_new = self.cb.postproj(attn0_next, batches[0], layer_idx + 1)
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
        # IDE_006 async cdec — drain before postproj reads attn1.
        from vllm.model_executor.layers.attention.attention import (
            _neo_drain_pending_cdec as _drain_pending,
        )
        _drain_pending()
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

        # IDE_006 — async cdec is the algorithm-correct NEO §4.4 path
        # (defers cdec wait so batches[1] CPU attention overlaps with
        # batches[0] preproj on s0). Empirically on the current hardware
        # the 2-concurrent-cdec pattern saturates the OMP pool and
        # regresses throughput; gate it behind VLLM_NEO_ASYNC_CDEC so
        # the implementation stays available without being on by default.
        import os as _os_async
        if _os_async.environ.get("VLLM_NEO_ASYNC_CDEC", "0") == "1":
            from vllm.model_executor.layers.attention.attention import (
                neo_async_cdec_scope as _neo_async_cdec_scope,
            )
            with _neo_async_cdec_scope():
                return self._forward_pipeline_inner(batches, embeddings)
        return self._forward_pipeline_inner(batches, embeddings)

    def _forward_pipeline_inner(
        self,
        batches: Sequence[SubBatch],
        embeddings: Sequence[Any],
    ) -> tuple[Any, Any]:
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
