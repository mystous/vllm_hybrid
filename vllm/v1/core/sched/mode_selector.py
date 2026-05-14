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

import itertools
import logging
import math
import os
from collections.abc import Iterable
from typing import Protocol

from vllm.v1.core.sched.perfpredictor import PerfPredictor
from vllm.v1.core.sched.sub_batch import SubBatch

# os.environ lookup is *not* free — Python eagerly snapshots all env on
# first import, but ``os.environ.get`` still does a dict lookup each
# call. ``decide_mode`` runs every schedule, so cache the flag once at
# module import. Setting the env var after Python startup will not take
# effect — that is intentional (used only for dev-time bring-up).
_FORCE_PIPELINED: bool = os.environ.get("VLLM_NEO_FORCE_PIPELINED") == "1"
# [TSK_019 v3 / Phase A-0] Kill switch — set to "1" 면 force-pipelined
# 효과 전체 무시 (env + scheduler_config.enable_neo_force_pipelined 모두).
# 시스템 hang 시 비상 회피용.
_DISABLE_FORCE_PIPELINED: bool = (
    os.environ.get("VLLM_NEO_DISABLE_FORCE_PIPELINED") == "1"
)
# [Phase A-0] forward_double 진입 전 보장할 GPU free memory 마진 (bytes).
# default 1 GiB / shard — 부족 시 sequential 자동 폴백.
_NEO_GPU_FREE_MARGIN_BYTES: int = int(
    os.environ.get("VLLM_NEO_GPU_FREE_MARGIN_BYTES", str(1024 ** 3))
)

logger = logging.getLogger(__name__)


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
    force_pipelined: bool = False,
) -> list[SubBatch]:
    """Run NEO's 5-step batch composition. Returns either one or two
    SubBatches depending on whether pipelined mode wins.

    Args:
        force_pipelined: [TSK_019 v3 / Phase A-0] caller (NeoScheduler /
            adapter) 가 ``scheduler_config.enable_neo_force_pipelined``
            기반으로 set. True 면 cdec_q empty 시에도 pipelined 시도.
            ``VLLM_NEO_DISABLE_FORCE_PIPELINED=1`` env 시 무시.
    """
    # [Phase A-0] Kill switch — 본 env 가 set 되면 모든 force-pipelined
    # 효과 (env + caller-passed flag) 를 비활성. 시스템 hang 시 비상
    # 회피.
    effective_force_pipelined = (
        (_FORCE_PIPELINED or force_pipelined) and not _DISABLE_FORCE_PIPELINED
    )

    # [Phase A-0] GPU memory guard — pipelined 진입 전 forward_double 의
    # 추가 alloc (preproj/postproj workspace) 위한 free margin 확인.
    # margin 부족 시 effective_force_pipelined 무력화 → sequential 폴백.
    if effective_force_pipelined:
        try:
            import torch as _torch_guard
            if _torch_guard.cuda.is_available():
                free_bytes, _total = _torch_guard.cuda.mem_get_info()
                if free_bytes < _NEO_GPU_FREE_MARGIN_BYTES:
                    logger.warning(
                        "decide_mode: GPU free %.0f MiB < margin %.0f MiB "
                        "— forced pipelined disabled, fall back to sequential.",
                        free_bytes / (1024 ** 2),
                        _NEO_GPU_FREE_MARGIN_BYTES / (1024 ** 2),
                    )
                    effective_force_pipelined = False
        except (ImportError, RuntimeError):
            # cuda 미초기화 또는 torch 미설치 — guard skip
            pass

    # Fast-path: 의미 있게 pipelined 가 발화하려면 (a) cpu_decoding_q 가
    # non-empty (Step 3 가 cdec 배포 가능) 또는 (b) FORCE_PIPELINED 활성.
    # 둘 다 아니면 batches[1] 가 끝까지 비어있어 line 177 의
    # ``return [gpu_only_batch]`` 로 떨어진다 — 이때 batches[0] 와
    # gpu_only_batch 양쪽에 add_gdec/add_pref 를 *모두* 한 작업이 통째로
    # 폐기. 이 fast-path 로 batches[0]/[1] 의 add 자체를 skip 해서 134+
    # add_gdec 호출 비용 제거.
    cdec_iter = iter(cpu_decoding_q)
    has_cdec = False
    try:
        first_cdec = next(cdec_iter)
        has_cdec = True
    except StopIteration:
        first_cdec = None

    if (not has_cdec
            and not effective_force_pipelined
            and num_gpu_blocks > 0):
        gpu_only_batch = SubBatch(predictor)
        for req in gpu_prefill_reqs:
            gpu_only_batch.add_pref(req, is_gpu=True)
        for req in cpu_prefill_reqs:
            gpu_only_batch.add_pref(req, is_gpu=False)
        for req in gpu_decoding_q:
            gpu_only_batch.add_gdec(req)

        if len(gpu_only_batch) == 0:
            return []

        # Step 2 — trim CPU prefill in gpu_only when iter width is large
        while gpu_only_batch.get_num_prefs():
            req, is_gpu = gpu_only_batch.pop_pref()
            if is_gpu or gpu_only_batch.perfdata.s < linr_S_threshold:
                gpu_only_batch.add_pref(req, is_gpu=is_gpu)
                break

        return [gpu_only_batch]

    # Slow path — cpu_decoding_q 가 있거나 FORCE 활성. cdec_iter 를 다시
    # full iterator 로 만들기 위해 첫 element 와 chain.
    if has_cdec:
        cpu_decoding_q = itertools.chain([first_cdec], cdec_iter)

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

    # Step 3 — distribute CPU-decode requests between batches[0]/[1].
    # IDE_006 G3 — NEO 정통 (scheduler.py:184-203) 영역 복원. balance-driven
    # next_batch_idx = remains[1] > remains[0] + overflow guard
    # (min(remains) < 0 시 pop + min_out_cpu_len filter). HeuristicPerf
    # Predictor 가 perfdata 영역 채우므로 _get_remains 영역 비제로 값 반환
    # → §4.4 의 진짜 load-aware 영역 활성.
    # env override: VLLM_NEO_DECIDE_MODE_BALANCE=0 시 strict alternation
    # (이전 fallback 영역, 회귀 시 비상 회피용).
    _balance_mode = os.environ.get("VLLM_NEO_DECIDE_MODE_BALANCE", "1") == "1"
    min_out_cpu_len = math.inf
    next_batch_idx = 1
    for req in cpu_decoding_q:
        if not budget.check_and_substract(1):
            break
        if _balance_mode and req.num_tokens >= min_out_cpu_len:
            # NEO Step 3 FCFS — 이전 req 가 CPU overflow 발화한 seq_len
            # 이상은 skip (큰 seq_len 일수록 cdec 비용 크므로).
            budget.add(1)
            continue
        batches[next_batch_idx].add_cdec(req)
        if _balance_mode:
            remains = _get_remains(batches)
            if min(remains) < 0 and num_gpu_blocks > 0:
                # CPU overflow — pop, 같은 seq_len 이상 skip.
                min_out_cpu_len = req.num_tokens
                budget.add(1)
                batches[next_batch_idx].pop_cdec()
                continue
            # balance-driven: 더 slack 큰 batch 에 next 추가.
            next_batch_idx = 1 if remains[1] > remains[0] else 0
        else:
            # legacy strict alternation fallback.
            next_batch_idx = 1 - next_batch_idx

    # Edge case: no cdec was packed → fall back to GPU-only.
    # When ``VLLM_NEO_FORCE_PIPELINED=1`` *or* has_cdec is True we instead
    # split batches[0] before falling back, so the forced two-sub-batch
    # shape lands downstream and the dual-forward path actually fires.
    if not batches[1] and num_gpu_blocks > 0:
        # IDE_006 — has_cdec 시도 split (predictor 영역 무시).
        # [Phase A-0] caller-passed force_pipelined 도 동등 효과.
        if effective_force_pipelined or has_cdec:
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

    # Experimental override — module-level cached at import (zero hot-path
    # cost when off). Used during NEO bring-up to force pipelined return
    # so ``forward_neo_pipelined`` exercises real workloads even with
    # ZeroPerfPredictor (where Step 5's rate compare is meaningless).
    if effective_force_pipelined:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[NEO] decide_mode FORCE entry: "
                "batches[0] gprf=%d cprf=%d gdec=%d cdec=%d / "
                "batches[1] gprf=%d gdec=%d cdec=%d",
                batches[0].num_gprfs, batches[0].num_cprfs,
                batches[0].num_gdecs, batches[0].num_cdecs,
                batches[1].num_gprfs, batches[1].num_gdecs,
                batches[1].num_cdecs,
            )
        if not len(batches[1]):
            half_gprf = len(batches[0].gprf_reqs) // 2
            for _ in range(half_gprf):
                req = batches[0].gprf_reqs.pop()
                batches[0].perfdata.pop_pref(req.prompt_len)
                batches[1].add_pref(req, is_gpu=True)
            half_gdec = batches[0].num_gdecs // 2
            for _ in range(half_gdec):
                req = batches[0].pop_gdec()
                batches[1].add_gdec(req)
        if len(batches[1]):
            return batches

    if sequential_time <= 0:
        return batches if pipelined_time > 0 else [gpu_only_batch]
    if pipelined_time <= 0:
        return [gpu_only_batch]
    sequential_rate = len(gpu_only_batch) / sequential_time
    pipelined_rate = (len(batches[0]) + len(batches[1])) / pipelined_time
    return batches if pipelined_rate > sequential_rate else [gpu_only_batch]
