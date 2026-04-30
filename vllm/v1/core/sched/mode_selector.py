# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NEO-style mode selector: pick pipelined vs sequential per iteration.

Adapted from NEO ``swiftllm/server/scheduler.py:_decide_mode_and_gen_batch``
(MLSys 2025, Apache 2.0). Algorithms only — no code copied.

Given the candidate prefill / decode requests for the next iteration,
``decide_mode`` returns one of:

* ``[gpu_only_batch]`` — single sub-batch (sequential mode, vanilla)
* ``[batch0, batch1]`` — two sub-batches (pipelined mode, asymmetric)

The decision criterion is the per-time request rate: pick the
configuration whose ``rate = num_reqs / wall_time`` is larger.

See ``shadow_assists/features/IDE_006/NEO_code_deepdive.md`` §3.1.
"""

from __future__ import annotations

import math
from typing import Iterable, Protocol

from vllm.v1.core.sched.perfpredictor import PerfPredictor
from vllm.v1.core.sched.sub_batch import SubBatch


class _ReqLike(Protocol):
    request_id: int
    @property
    def prompt_len(self) -> int: ...
    @property
    def num_tokens(self) -> int: ...


class ScheduleBudget:
    """Tracks the remaining batch-size and token budget while the
    scheduler is composing the next iteration."""

    __slots__ = ("remaining_batch_size", "remaining_tokens_in_batch",
                 "max_batch_size", "max_tokens_in_batch")

    def __init__(self, max_batch_size: int, max_tokens_in_batch: int) -> None:
        self.max_batch_size = max_batch_size
        self.max_tokens_in_batch = max_tokens_in_batch
        self.remaining_batch_size = max_batch_size
        self.remaining_tokens_in_batch = max_tokens_in_batch

    @property
    def overspent(self) -> bool:
        return (self.remaining_batch_size < 0
                or self.remaining_tokens_in_batch < 0)

    def check_and_substract(self, num_tokens: int) -> bool:
        if (self.remaining_batch_size < 1
                or self.remaining_tokens_in_batch < num_tokens):
            return False
        self.remaining_batch_size -= 1
        self.remaining_tokens_in_batch -= num_tokens
        return True

    def add(self, num_tokens: int) -> None:
        self.remaining_batch_size += 1
        self.remaining_tokens_in_batch += num_tokens


def _get_remains(batches: list[SubBatch]) -> list[float]:
    """For each batch j, how much *headroom* does the CPU have before
    becoming the critical path?

        remains[j] = batches[j^1].linr_T
                   + batches[j].pref_T
                   + batches[j].gdec_T
                   - batches[j].cpu_time

    Positive ⇒ GPU is busy longer; CPU has slack and can absorb more
    cdec requests. Negative ⇒ CPU is on the critical path.
    """
    assert len(batches) == 2
    out: list[float] = []
    for j in range(2):
        other = j ^ 1
        out.append(
            batches[other].perfdata.linr_T
            + batches[j].perfdata.pref_T
            + batches[j].perfdata.gdec_T
            - batches[j].perfdata.cpu_time
        )
    return out


def decide_mode(
    *,
    gpu_prefill_reqs: list[_ReqLike],
    cpu_prefill_reqs: list[_ReqLike],
    gpu_decoding_q: list[_ReqLike],
    cpu_decoding_q: Iterable[_ReqLike],
    budget: ScheduleBudget,
    predictor: PerfPredictor,
    num_layers: int,
    num_gpu_blocks: int,
    linr_S_threshold: int = 128,
) -> list[SubBatch]:
    """Run NEO's 5-step batch composition. Returns either one or two
    SubBatches depending on whether pipelined mode wins.
    """
    batches = [SubBatch(predictor) for _ in range(2)]
    gpu_only_batch = SubBatch(predictor)

    # Step 1 — seed batch[0] and gpu_only with all prefill + GPU decode
    for req in gpu_prefill_reqs:
        batches[0].add_pref(req, is_gpu=True)
        gpu_only_batch.add_pref(req, is_gpu=True)
    for req in cpu_prefill_reqs:
        batches[0].add_pref(req, is_gpu=False)
        gpu_only_batch.add_pref(req, is_gpu=False)
    for req in gpu_decoding_q:
        batches[0].add_gdec(req)
        gpu_only_batch.add_gdec(req)

    if not batches[0] and num_gpu_blocks > 0:
        return []

    # Step 2 — trim CPU prefill in gpu_only when iter width is large
    while gpu_only_batch.get_num_prefs():
        req, is_gpu = gpu_only_batch.pop_pref()
        if is_gpu or gpu_only_batch.perfdata.s < linr_S_threshold:
            gpu_only_batch.add_pref(req, is_gpu=is_gpu)
            break

    # Step 3 — distribute CPU-decode requests between batches[0]/[1]
    min_out_cpu_len = math.inf
    next_batch_idx = 1
    for req in cpu_decoding_q:
        if not budget.check_and_substract(1):
            break
        if req.num_tokens >= min_out_cpu_len:
            budget.add(1)
            continue
        batches[next_batch_idx].add_cdec(req)
        remains = _get_remains(batches)
        if min(remains) < 0 and num_gpu_blocks > 0:
            min_out_cpu_len = req.num_tokens
            budget.add(1)
            batches[next_batch_idx].pop_cdec()
            continue
        next_batch_idx = int(remains[1] > remains[0])

    # Edge case: no cdec was packed → fall back to GPU-only.
    # When ``VLLM_NEO_FORCE_PIPELINED=1`` we instead split batches[0]
    # before falling back, so the forced two-sub-batch shape lands
    # downstream and the dual-forward path actually fires.
    if not batches[1] and num_gpu_blocks > 0:
        import os as _os
        if _os.environ.get("VLLM_NEO_FORCE_PIPELINED") == "1":
            half_gprf = len(batches[0].gprf_reqs) // 2
            for _ in range(half_gprf):
                req = batches[0].gprf_reqs.pop()
                batches[0].perfdata.pop_pref(req.prompt_len)
                batches[1].add_pref(req, is_gpu=True)
            half_gdec = batches[0].num_gdecs // 2
            for _ in range(half_gdec):
                req = batches[0].pop_gdec()
                batches[1].add_gdec(req)
        if not batches[1]:
            return [gpu_only_batch]

    # When num_gpu_blocks == 0 we're in CPU-only mode; return the
    # composed batches directly.
    if num_gpu_blocks == 0:
        return [b for b in batches if len(b) > 0]

    # Step 4 — trim batches[0] prefill if CPU is idling
    while batches[0].get_num_prefs():
        req, is_gpu = batches[0].pop_pref()
        too_small = batches[0].perfdata.s < linr_S_threshold
        cpu_late = min(_get_remains(batches)) < 0
        if is_gpu or too_small or cpu_late:
            batches[0].add_pref(req, is_gpu=is_gpu)
            break

    # Step 5 — pipelined vs sequential by request-rate comparison
    sequential_time = gpu_only_batch.gpu_time * num_layers
    pipelined_time = (batches[0].gpu_time + batches[1].gpu_time) * num_layers

    # Experimental override (env: ``VLLM_NEO_FORCE_PIPELINED=1``).
    # When active, force the pipelined two-sub-batch return so the
    # downstream ``forward_neo_pipelined`` path actually fires —
    # used during NEO data-path bring-up (Step 5.4) to surface the
    # first runtime mismatch with vLLM's single-batch forward
    # context. Has no effect once a real predictor is wired.
    import os as _os_force
    if _os_force.environ.get("VLLM_NEO_FORCE_PIPELINED") == "1":
        import logging as _lg
        _lg.getLogger(__name__).info(
            "[NEO-DEBUG] decide_mode FORCE entry: "
            "batches[0] gprf=%d cprf=%d gdec=%d cdec=%d / batches[1] gprf=%d gdec=%d cdec=%d",
            batches[0].num_gprfs, batches[0].num_cprfs,
            batches[0].num_gdecs, batches[0].num_cdecs,
            batches[1].num_gprfs, batches[1].num_gdecs, batches[1].num_cdecs,
        )
        if not len(batches[1]):
            # GPU prefill 분할
            half_gprf = len(batches[0].gprf_reqs) // 2
            for _ in range(half_gprf):
                req = batches[0].gprf_reqs.pop()
                batches[0].perfdata.pop_pref(req.prompt_len)
                batches[1].add_pref(req, is_gpu=True)
            # GPU decode 분할
            half_gdec = batches[0].num_gdecs // 2
            for _ in range(half_gdec):
                req = batches[0].pop_gdec()
                batches[1].add_gdec(req)
        if len(batches[1]):
            # Force pipelined return, bypassing Step 5's rate compare
            # which is meaningless under ZeroPerfPredictor.
            return batches

    if sequential_time <= 0:
        return batches if pipelined_time > 0 else [gpu_only_batch]
    if pipelined_time <= 0:
        return [gpu_only_batch]
    sequential_rate = len(gpu_only_batch) / sequential_time
    pipelined_rate = (len(batches[0]) + len(batches[1])) / pipelined_time
    return batches if pipelined_rate > sequential_rate else [gpu_only_batch]
