# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Disaggregated Serving for CPU-GPU Hybrid Inference.

This module implements Prefill/Decode disaggregation, where:
- Prefill nodes handle compute-intensive prompt processing
- Decode nodes handle memory-bound token generation
- KV Cache is transferred between nodes

Components:
- KVTransfer: KV Cache transfer utilities
- PrefillNode: Prefill-specialized node
- DecodeNode: Decode-specialized node with MoE offload and N-gram support
- DisaggregatedCoordinator: Request routing and orchestration
"""

from vllm.engine.disaggregated.kv_transfer import (
    KVCacheReceiver,
    KVCacheSender,
    KVTransferConfig,
)
from vllm.engine.disaggregated.coordinator import DisaggregatedCoordinator

__all__ = [
    "KVTransferConfig",
    "KVCacheSender",
    "KVCacheReceiver",
    "DisaggregatedCoordinator",
]
