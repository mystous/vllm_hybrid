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
