# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
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
