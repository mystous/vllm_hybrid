# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, NamedTuple

from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    ReqId,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.abstract import (
    OffloadingManager,
    OffloadKey,
    ReqContext,
    get_offload_block_hash,
    make_offload_key,
)
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import TransferSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    gpu_block_size: int
    offloaded_block_size: int
    hash_block_size_factor: int


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    block_size_factor: int

    @classmethod
    def from_spec(cls, spec: OffloadingSpec) -> "SchedulerOffloadConfig":
        return cls(
            kv_group_configs=tuple(
                GroupOffloadConfig(
                    group_idx=idx,
                    gpu_block_size=gpu_block_size,
                    offloaded_block_size=gpu_block_size * spec.block_size_factor,
                    hash_block_size_factor=(
                        (gpu_block_size * spec.block_size_factor)
                        // spec.hash_block_size
                    ),
                )
                for idx, gpu_block_size in enumerate(spec.gpu_block_size)
            ),
            block_size_factor=spec.block_size_factor,
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # index of next block (of size offloaded_block_size) to offload
    next_stored_block_idx: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    req_context: ReqContext = field(init=False)
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
        self.req_context = ReqContext(kv_transfer_params=self.req.kv_transfer_params)

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hash_block_size_factor * len(group_state.offload_keys)
                + group_config.hash_block_size_factor
                - 1,
                None,
                group_config.hash_block_size_factor,
            ):
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.config = SchedulerOffloadConfig.from_spec(spec)
        self.manager: OffloadingManager = spec.get_manager()

        # Cold-KV CPU partial attention (IDE_006 / TSK_002 Phase 4c) is
        # currently scope-locked to block_size_factor == 1 (one offloaded
        # block == one GPU block). When factor > 1 the partial-attention
        # kernel — which works at GPU-block granularity — has no way to
        # address sub-block slices of a CPU canonical buffer. Refuse the
        # combination at config time so the user gets a clear message
        # instead of an opaque failure deep inside model_runner.
        kv_cfg = spec.vllm_config.kv_transfer_config
        if (
            kv_cfg is not None
            and kv_cfg.enable_cpu_partial_attention
            and self.config.block_size_factor != 1
        ):
            raise ValueError(
                "enable_cpu_partial_attention=True requires "
                "block_size_factor == 1 (offloaded_block_size must equal "
                "gpu_block_size). Got block_size_factor="
                f"{self.config.block_size_factor}. Either drop "
                "kv_connector_extra_config['block_size'] (the default "
                "matches gpu_block_size) or set it to the GPU block size "
                "of the current model."
            )

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        # requests to load for the current scheduler step
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}
        # if GPU prefix caching is enabled,
        # track loaded blocks to avoid redundant loads
        self._blocks_being_loaded: set[OffloadKey] | None = (
            set() if spec.vllm_config.cache_config.enable_prefix_caching else None
        )

        # request ID -> set(offload keys being stored/loaded)
        self._reqs_being_stored = defaultdict[ReqId, set[OffloadKey]](set)
        self._reqs_being_loaded = defaultdict[ReqId, set[OffloadKey]](set)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        if req_status := self._req_status.get(request.request_id):
            # make sure block IDs are cleared
            for group_state in req_status.group_states:
                group_state.block_ids.clear()
        else:
            req_status = RequestOffloadState(config=self.config, req=request)
            req_status.update_offload_keys()
            self._req_status[request.request_id] = req_status

        req_status.num_locally_computed_tokens = num_computed_tokens

        # Below assertions will be removed once this function supports HMA
        assert len(self.config.kv_group_configs) == 1
        assert len(req_status.group_states) == 1
        group_config = self.config.kv_group_configs[0]
        group_state = req_status.group_states[0]

        num_blocks = request.num_tokens // group_config.offloaded_block_size

        assert len(request.block_hashes) // self.config.block_size_factor == num_blocks
        offload_keys = group_state.offload_keys

        self.manager.touch(offload_keys)

        full_block_tokens = group_config.offloaded_block_size * num_blocks
        if full_block_tokens - num_computed_tokens < group_config.offloaded_block_size:
            # we can load less than a block, skip
            return 0, False

        start_block_idx = num_computed_tokens // group_config.offloaded_block_size
        hits = self.manager.lookup(
            offload_keys[start_block_idx:],
            req_status.req_context,
        )
        if hits is None:
            # indicates a lookup that should be tried later
            return None, False
        if hits == 0:
            return 0, False

        num_hit_tokens = (
            group_config.offloaded_block_size * (start_block_idx + hits)
            - num_computed_tokens
        )
        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            request.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )
        if num_hit_tokens < group_config.offloaded_block_size:
            return 0, False

        if self._blocks_being_loaded and any(
            key in self._blocks_being_loaded
            for key in offload_keys[start_block_idx : start_block_idx + hits]
        ):
            # hit blocks are being loaded, delay request
            logger.debug(
                "Delaying request %s since some of its blocks are already being loaded",
                request.request_id,
            )
            return None, False

        return num_hit_tokens, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]
        block_groups = blocks.get_block_ids()

        # Below assertions will be removed once this function supports HMA
        assert len(self.config.kv_group_configs) == 1
        assert len(req_status.group_states) == 1
        assert len(block_groups) == 1
        block_ids = block_groups[0]
        group_config = self.config.kv_group_configs[0]
        group_state = req_status.group_states[0]

        num_computed_gpu_blocks = sum(
            block.block_hash is not None for block in blocks.blocks[0]
        )
        num_computed_tokens = num_computed_gpu_blocks * group_config.gpu_block_size
        full_block_tokens = num_computed_tokens + num_external_tokens
        assert full_block_tokens % group_config.offloaded_block_size == 0

        num_pending_gpu_blocks = len(block_ids) - num_computed_gpu_blocks
        assert (
            num_external_tokens == num_pending_gpu_blocks * group_config.gpu_block_size
        )

        start_block_idx = num_computed_tokens // group_config.offloaded_block_size
        num_blocks = full_block_tokens // group_config.offloaded_block_size

        assert len(request.block_hashes) // self.config.block_size_factor >= num_blocks
        offload_keys = group_state.offload_keys[start_block_idx:num_blocks]

        src_spec = self.manager.prepare_load(offload_keys, req_status.req_context)
        dst_spec = GPULoadStoreSpec(
            block_ids[num_computed_gpu_blocks:],
            group_sizes=(num_pending_gpu_blocks,),
            block_indices=(num_computed_gpu_blocks,),
        )

        self._reqs_to_load[request.request_id] = (src_spec, dst_spec)
        req_blocks_being_loaded = self._reqs_being_loaded[request.request_id]
        req_blocks_being_loaded.update(offload_keys)
        group_state.next_stored_block_idx = num_blocks

        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(req_blocks_being_loaded)

    def _get_reqs_to_store(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[dict[ReqId, TransferSpec], dict[ReqId, list[OffloadKey]]]:
        # Below assertion will be removed once this function supports HMA
        assert len(self.config.kv_group_configs) == 1
        group_config = self.config.kv_group_configs[0]

        reqs_to_store: dict[ReqId, TransferSpec] = {}
        # Parallel structure surfaced to the worker via metadata — per-req
        # OffloadKey list dispatched THIS round so the worker can report
        # completions back via OffloadingConnectorWorkerMetadata.
        reqs_to_store_keys: dict[ReqId, list[OffloadKey]] = {}
        # iterate over both new and cached requests
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                req_status.update_block_id_groups(new_block_id_groups)

            # Below assertion will be removed once this function supports HMA
            assert len(req_status.group_states) == 1
            group_state = req_status.group_states[0]

            block_ids = group_state.block_ids

            req = req_status.req
            new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            expected_tokens = req.num_computed_tokens + new_tokens
            # with async scheduling, some tokens may be missing
            total_tokens = min(expected_tokens, req.num_tokens)
            num_blocks = total_tokens // group_config.offloaded_block_size
            start_block_idx = group_state.next_stored_block_idx
            num_new_blocks = num_blocks - start_block_idx

            if num_new_blocks <= 0:
                continue

            num_gpu_blocks = num_blocks * self.config.block_size_factor
            assert len(req.block_hashes) >= num_gpu_blocks

            new_offload_keys = group_state.offload_keys[start_block_idx:num_blocks]
            store_output = self.manager.prepare_store(
                new_offload_keys, req_status.req_context
            )
            if store_output is None:
                logger.warning(
                    "Request %s: cannot store %s blocks", req_id, num_new_blocks
                )
                continue

            group_state.next_stored_block_idx = num_blocks

            if not store_output.keys_to_store:
                continue
            keys_to_store = set(store_output.keys_to_store)

            self.manager.touch(group_state.offload_keys[:num_blocks])

            dst_spec = store_output.store_spec
            src_block_ids: list[int] = []
            for idx, key in enumerate(new_offload_keys):
                if key not in keys_to_store:
                    continue
                offloaded_block_idx = start_block_idx + idx
                gpu_block_idx = offloaded_block_idx * self.config.block_size_factor
                for i in range(self.config.block_size_factor):
                    src_block_ids.append(block_ids[gpu_block_idx + i])
            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=(len(src_block_ids),)
            )

            reqs_to_store[req_id] = (src_spec, dst_spec)
            self._reqs_being_stored[req_id] |= keys_to_store
            # The keys list passed to the worker must be in the same order
            # as the GPU block IDs in src_block_ids so per-job completion
            # mapping is unambiguous. We build it from new_offload_keys
            # (the original ordered list) filtered through keys_to_store.
            reqs_to_store_keys[req_id] = [
                key for key in new_offload_keys if key in keys_to_store
            ]

            logger.debug(
                "Request %s offloading %s blocks starting from block #%d",
                req_id,
                len(keys_to_store),
                start_block_idx,
            )

        return reqs_to_store, reqs_to_store_keys

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        # _get_reqs_to_store updates next_stored_block_idx in-place; capture
        # the per-request cold-block snapshot AFTER that update so the worker
        # sees the same value the connector just committed.
        reqs_to_store, reqs_to_store_keys = self._get_reqs_to_store(
            scheduler_output
        )

        # Cold-KV CPU partial attention (IDE_006 / TSK_002) input. The
        # offloaded-block-unit count `next_stored_block_idx` is converted
        # to GPU-block units (matching the indexing of `block_table` used
        # by attention kernels) by multiplying by `block_size_factor`.
        # Single-KV-group is enforced elsewhere (line 169-170, 233-234,
        # 275-276); we read group_states[0] accordingly.
        #
        # Phase 4a additionally surfaces the CPU canonical-buffer block
        # IDs for each request's cold prefix via
        # `manager.peek_block_ids(...)` — a no-side-effect lookup that
        # returns the cache-side block ID per offload key. The kernel
        # (TSK_001's forward_partial_with_lse) reads cold KV in place
        # using these IDs, no transfer involved.
        #
        # block_size_factor > 1 is currently not supported on the cold
        # path (one offloaded block would map to multiple GPU blocks but
        # the partial-attention kernel works at GPU-block granularity);
        # we populate cold_cpu_block_ids only when factor == 1, otherwise
        # leave it empty so the worker side falls back to the standard
        # non-cold-split path.
        num_cold_gpu_blocks_per_req: dict[ReqId, int] = {}
        cold_cpu_block_ids: dict[ReqId, list[int]] = {}
        cold_factor_one = self.config.block_size_factor == 1

        # IDE_006 / TSK_002 Phase 4c diagnostic — DEBUG level so the
        # detail only shows under VLLM_LOGGING_LEVEL=DEBUG. The runtime
        # guard against silent dispatcher bypass lives in
        # eval/run_e2e_accuracy.py:_compare_outputs as the
        # ``suspicious_no_cold_path`` detector. Useful for dev-side
        # debugging when (A) connector never asks to store anything
        # (next_stored_block_idx stays 0) needs to be distinguished
        # from (B) transfers queued but never marked ready
        # (peek_block_ids returns None).
        if logger.isEnabledFor(logging.DEBUG):
            if not hasattr(self, "_cold_diag_counter"):
                self._cold_diag_counter = 0
            self._cold_diag_counter += 1
            if self._cold_diag_counter <= 5 or self._cold_diag_counter % 50 == 0:
                _stats: list[str] = []
                for _rid, _rstatus in list(self._req_status.items())[:5]:
                    _ni = _rstatus.group_states[0].next_stored_block_idx
                    _peeked: list[Any] = []
                    if _ni > 0 and cold_factor_one:
                        _ok = _rstatus.group_states[0].offload_keys[:_ni]
                        try:
                            _peeked = list(
                                self.manager.peek_block_ids(
                                    _ok, _rstatus.req_context
                                )
                            )
                        except Exception as _e:
                            _stats.append(f"{_rid[:8]}:next={_ni},peek_err={_e!r}")
                            continue
                    _ready = sum(1 for _b in _peeked if _b is not None)
                    _stats.append(
                        f"{_rid[:8]}:next={_ni},ready={_ready}/{len(_peeked)}"
                    )
                logger.debug(
                    "[IDE_006 diag scheduler call=%d] num_reqs_total=%d "
                    "factor=%d sampled=%s",
                    self._cold_diag_counter,
                    len(self._req_status),
                    self.config.block_size_factor,
                    ", ".join(_stats) or "<none>",
                )

        for req_id, req_status in self._req_status.items():
            next_idx = req_status.group_states[0].next_stored_block_idx
            if next_idx == 0:
                continue
            if cold_factor_one:
                # Reconcile num count and CPU IDs from a single source —
                # the longest contiguous prefix of ready (data has been
                # transferred to CPU) blocks. `next_stored_block_idx` is
                # an *optimistic* counter that increments at prepare_store
                # time (block allocated, not yet ready). The Cold-KV CPU
                # partial attention path can only safely read READY
                # blocks, so we trim back to that prefix here. Otherwise
                # num_cold_blocks and cold_cpu_block_ids would race and
                # the dispatcher would invoke the kernel with stale CPU
                # block IDs (or none at all) → garbage output / CUDA
                # illegal memory access.
                cold_offload_keys = (
                    req_status.group_states[0].offload_keys[:next_idx]
                )
                peeked = self.manager.peek_block_ids(
                    cold_offload_keys, req_status.req_context
                )
                ready_prefix = 0
                for b in peeked:
                    if b is None:
                        break
                    ready_prefix += 1
                if ready_prefix > 0:
                    num_cold_gpu_blocks_per_req[req_id] = ready_prefix
                    cold_cpu_block_ids[req_id] = [
                        int(b) for b in peeked[:ready_prefix]
                    ]
            else:
                # factor != 1 → cold attention path is not supported (we
                # validate and reject this combination at __init__ when
                # enable_cpu_partial_attention is on). Still publish the
                # GPU-block count for any other consumer.
                num_cold_gpu_blocks_per_req[req_id] = (
                    next_idx * self.config.block_size_factor
                )

        meta = OffloadingConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_store=reqs_to_store,
            reqs_to_flush=scheduler_output.preempted_req_ids,
            num_cold_gpu_blocks_per_req=num_cold_gpu_blocks_per_req,
            cold_cpu_block_ids=cold_cpu_block_ids,
            reqs_to_store_keys=reqs_to_store_keys,
        )
        self._reqs_to_load = {}

        # NOTE (orozery): we should move this logic to update_connector_output
        # once KVConnectorOutput allows us to report completed transfers
        for req_id in scheduler_output.preempted_req_ids or ():
            keys = self._reqs_being_stored.get(req_id)
            if keys:
                self.manager.complete_store(keys)
                keys.clear()

        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        # IDE_006 / TSK_002 Phase 4c: surface in-flight store-completion
        # events from the worker BEFORE the request-finished path so that
        # peek_block_ids returns ready cache-side block IDs while the
        # request is still alive — the prerequisite for the Cold-KV CPU
        # partial attention dispatcher (`max_num_cold_blocks_host > 0`).
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
            OffloadingConnectorWorkerMetadata,
        )

        worker_meta = connector_output.kv_connector_worker_meta
        if isinstance(worker_meta, OffloadingConnectorWorkerMetadata):
            if worker_meta.finished_store_keys:
                self.manager.complete_store(worker_meta.finished_store_keys)

        for req_id in connector_output.finished_sending or []:
            keys = self._reqs_being_stored.pop(req_id, None)
            if keys:
                self.manager.complete_store(keys)

        for req_id in connector_output.finished_recving or []:
            keys = self._reqs_being_loaded.pop(req_id, None)
            if keys:
                if self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(keys)
                self.manager.complete_load(keys)

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_id = request.request_id

        # TODO(orozery): possibly kickoff offload for last block
        # which may have been deferred due to async scheduling
        self._req_status.pop(req_id, None)

        request_being_stored = req_id in self._reqs_being_stored
        return request_being_stored, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Take the KV cache events from the connector.

        Returns:
            A list of KV cache events.
        """
        for event in self.manager.take_events():
            block_hashes = [get_offload_block_hash(key) for key in event.keys]
            if event.removed:
                yield BlockRemoved(block_hashes=block_hashes, medium=event.medium)
            else:
                yield BlockStored(
                    block_hashes=block_hashes,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=0,
                    medium=event.medium,
                    lora_name=None,
                )

    def shutdown(self) -> None:
        self.manager.shutdown()
