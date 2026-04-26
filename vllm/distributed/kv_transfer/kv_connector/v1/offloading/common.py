# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.kv_offload.abstract import OffloadKey
from vllm.v1.kv_offload.worker.worker import TransferSpec

ReqId = str


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]
    reqs_to_flush: set[str] | None = None
    # Per-request count of GPU block-table entries whose KV is currently
    # offloaded (i.e. resident on CPU). Populated by
    # OffloadingConnectorScheduler.build_connector_meta. Consumed by the
    # Cold-KV CPU partial attention path (IDE_006 / TSK_002) to slice each
    # request's block_table into a cold prefix and a hot suffix. The
    # default empty dict keeps the pre-IDE_006 behaviour for any consumer
    # that does not opt in.
    num_cold_gpu_blocks_per_req: dict[ReqId, int] = field(default_factory=dict)
    # Per-request CPU canonical-buffer block IDs for the cold prefix, in
    # the same order as the GPU block_table prefix they correspond to.
    # `len(cold_cpu_block_ids[req_id]) == num_cold_gpu_blocks_per_req[req_id]`
    # under the scope-locked single KV group + block_size_factor == 1
    # configuration (PLN_001 §3 + IDE_006 / TSK_002 Phase 4a). The CPU
    # partial-attention kernel (TSK_001's forward_partial_with_lse) uses
    # these IDs together with the per-layer canonical int8 buffer surfaced
    # by Phase 4b to read cold KV in place — no transfer, no GPU side-
    # effect. Default empty dict keeps every existing consumer unaffected.
    cold_cpu_block_ids: dict[ReqId, list[int]] = field(default_factory=dict)
    # Per-request OffloadKey list parallel to ``reqs_to_store`` — the keys
    # whose GPU→CPU transfer is being dispatched THIS round. The worker
    # tracks them per job so that, on transfer completion, it can report
    # the exact keys back to the scheduler-side manager via
    # OffloadingConnectorWorkerMetadata. Without this surface the
    # scheduler's manager has no way to mark in-flight stored blocks as
    # "ready", which silently disables the Cold-KV CPU partial attention
    # path (IDE_006 / TSK_002 Phase 4c) — see scheduler.py:406+ where
    # peek_block_ids is used.
    reqs_to_store_keys: dict[ReqId, list[OffloadKey]] = field(default_factory=dict)


@dataclass
class OffloadingConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker→scheduler surface for in-flight transfer completion (IDE_006
    / TSK_002 Phase 4c).

    `finished_store_keys` accumulates the OffloadKeys whose GPU→CPU async
    transfers have completed since the previous engine step. The scheduler
    side calls ``manager.complete_store(finished_store_keys)`` so that
    subsequent ``manager.peek_block_ids`` calls can return ready cache-side
    block IDs — the prerequisite for the Cold-KV CPU partial attention
    dispatcher to fire (`max_num_cold_blocks_host > 0`).

    With TP > 1, each worker process emits its own metadata; the engine
    aggregates them via :py:meth:`aggregate` before handing them to the
    single scheduler-side connector. Concatenating the two key lists is
    safe because ``manager.complete_store`` deduplicates internally and
    each TP rank stores the same set of keys (just per-rank slices of
    each block).
    """

    finished_store_keys: list[OffloadKey] = field(default_factory=list)

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        if not isinstance(other, OffloadingConnectorWorkerMetadata):
            return self
        return OffloadingConnectorWorkerMetadata(
            finished_store_keys=self.finished_store_keys
            + other.finished_store_keys,
        )
