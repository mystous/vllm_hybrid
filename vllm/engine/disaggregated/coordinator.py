# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Disaggregated Serving Coordinator.

This module coordinates Prefill and Decode nodes for disaggregated LLM serving.
It handles:
- Request routing between Prefill and Decode pools
- Load balancing across nodes
- KV cache transfer orchestration
- Integration with MoE offload and N-gram speculative decoding

Architecture:
    Request → Coordinator → Prefill Pool → KV Transfer → Decode Pool → Response
                  ↓
             Load Balancer
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid

import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.disaggregated.kv_transfer import (
    KVCacheReceiver,
    KVCacheSender,
    KVTransferConfig,
    TransferMethod,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.expert_offload import (
    ExpertOffloadConfig,
    ExpertOffloadManager,
)
from vllm.v1.spec_decode.ngram_proposer_dynamic import (
    DynamicNgramConfig,
    HybridNgramWorker,
)

logger = init_logger(__name__)


class NodeType(str, Enum):
    """Type of disaggregated node."""
    PREFILL = "prefill"
    DECODE = "decode"


class LoadBalanceStrategy(str, Enum):
    """Load balancing strategy."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"


@dataclass
class DisaggregatedConfig:
    """Configuration for disaggregated serving."""

    enabled: bool = True
    """Enable disaggregated serving."""

    num_prefill_nodes: int = 1
    """Number of prefill nodes."""

    num_decode_nodes: int = 1
    """Number of decode nodes."""

    prefill_device: str = "cuda"
    """Device for prefill: cuda or cpu."""

    decode_device: str = "cuda"
    """Device for decode: cuda (with MoE offload to CPU)."""

    kv_transfer: KVTransferConfig = field(default_factory=KVTransferConfig)
    """KV cache transfer configuration."""

    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    """Load balancing strategy for request routing."""

    moe_offload: Optional[ExpertOffloadConfig] = None
    """MoE expert offload configuration for decode nodes."""

    ngram_spec: Optional[DynamicNgramConfig] = None
    """N-gram speculative decoding configuration."""


@dataclass
class InferenceRequest:
    """Inference request container."""

    id: str
    """Unique request ID."""

    input_ids: List[int]
    """Input token IDs."""

    max_tokens: int
    """Maximum tokens to generate."""

    sampling_params: Dict[str, Any] = field(default_factory=dict)
    """Sampling parameters."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Request metadata."""


@dataclass
class InferenceResponse:
    """Inference response container."""

    request_id: str
    """Request ID."""

    output_tokens: List[int]
    """Generated token IDs."""

    finish_reason: str = "stop"
    """Finish reason: stop, length, or error."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Response metadata."""


@dataclass
class PrefillResult:
    """Result from prefill node."""

    request_id: str
    """Request ID."""

    num_prompt_tokens: int
    """Number of prompt tokens processed."""

    kv_transfer_handle: str
    """Handle for KV cache transfer."""


@dataclass
class DecodeResult:
    """Result from decode node."""

    request_id: str
    """Request ID."""

    output_tokens: List[int]
    """Generated tokens."""

    finish_reason: str
    """Finish reason."""


class LoadBalancer:
    """Simple load balancer for node selection."""

    def __init__(
        self,
        nodes: List[Any],
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
    ):
        self.nodes = nodes
        self.strategy = strategy
        self._rr_index = 0
        self._node_loads: Dict[int, int] = {i: 0 for i in range(len(nodes))}

    def select(self) -> Tuple[int, Any]:
        """Select a node based on strategy."""
        if not self.nodes:
            raise RuntimeError("No nodes available")

        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            idx = self._rr_index
            self._rr_index = (self._rr_index + 1) % len(self.nodes)

        elif self.strategy == LoadBalanceStrategy.LEAST_LOADED:
            idx = min(self._node_loads.keys(), key=lambda k: self._node_loads[k])

        elif self.strategy == LoadBalanceStrategy.RANDOM:
            import random
            idx = random.randint(0, len(self.nodes) - 1)

        else:
            idx = 0

        return idx, self.nodes[idx]

    def increment_load(self, idx: int):
        """Increment load count for a node."""
        self._node_loads[idx] = self._node_loads.get(idx, 0) + 1

    def decrement_load(self, idx: int):
        """Decrement load count for a node."""
        self._node_loads[idx] = max(0, self._node_loads.get(idx, 0) - 1)


class PrefillNode:
    """
    Prefill-specialized node.

    Handles compute-intensive prompt processing optimized for throughput.
    """

    def __init__(
        self,
        node_id: int,
        model_config: ModelConfig,
        device: str,
        kv_sender: KVCacheSender,
    ):
        self.node_id = node_id
        self.model_config = model_config
        self.device = device
        self.kv_sender = kv_sender

        # Model will be loaded lazily
        self._model = None
        self._tokenizer = None

        logger.info(f"PrefillNode {node_id} initialized on {device}")

    async def run_prefill(
        self,
        request: InferenceRequest,
        target_decode_node: int,
    ) -> PrefillResult:
        """
        Run prefill and send KV cache to decode node.

        Args:
            request: Inference request.
            target_decode_node: Target decode node ID.

        Returns:
            PrefillResult with transfer handle.
        """
        logger.debug(f"PrefillNode {self.node_id}: Processing request {request.id}")

        # Convert to tensor
        input_ids = torch.tensor([request.input_ids], device=self.device)
        num_tokens = len(request.input_ids)

        # Run prefill (placeholder - would use actual model)
        # In production, this would run the model's forward pass
        # and extract the KV cache

        # Simulate KV cache generation
        num_layers = 32  # Placeholder
        hidden_size = 4096  # Placeholder
        num_heads = 32  # Placeholder
        head_dim = hidden_size // num_heads

        # Create dummy KV cache for demonstration
        key_cache = [
            torch.randn(1, num_heads, num_tokens, head_dim, device=self.device)
            for _ in range(num_layers)
        ]
        value_cache = [
            torch.randn(1, num_heads, num_tokens, head_dim, device=self.device)
            for _ in range(num_layers)
        ]

        # Send KV cache to decode node
        transfer_handle = await self.kv_sender.send_async(
            kv_cache=(key_cache, value_cache),
            request_id=request.id,
            num_tokens=num_tokens,
            metadata={
                "target_node": target_decode_node,
                "input_ids": request.input_ids,
            },
        )

        return PrefillResult(
            request_id=request.id,
            num_prompt_tokens=num_tokens,
            kv_transfer_handle=transfer_handle,
        )


class DecodeNode:
    """
    Decode-specialized node with MoE offload and N-gram support.

    Handles memory-bound token generation with CPU-GPU hybrid execution.
    """

    def __init__(
        self,
        node_id: int,
        model_config: ModelConfig,
        device: str,
        kv_receiver: KVCacheReceiver,
        moe_offload_config: Optional[ExpertOffloadConfig] = None,
        ngram_config: Optional[DynamicNgramConfig] = None,
        vllm_config: Optional[VllmConfig] = None,
    ):
        self.node_id = node_id
        self.model_config = model_config
        self.device = device
        self.kv_receiver = kv_receiver

        # Model will be loaded lazily
        self._model = None
        self._tokenizer = None

        # MoE offload manager
        self.moe_offload: Optional[ExpertOffloadManager] = None
        if moe_offload_config and moe_offload_config.enabled:
            # Will be initialized when model is loaded
            self._moe_config = moe_offload_config

        # N-gram speculative decoding worker
        self.ngram_worker: Optional[HybridNgramWorker] = None
        if ngram_config:
            self.ngram_worker = HybridNgramWorker(ngram_config, vllm_config)

        logger.info(
            f"DecodeNode {node_id} initialized on {device}, "
            f"MoE offload: {moe_offload_config is not None}, "
            f"N-gram spec: {ngram_config is not None}"
        )

    async def run_decode(
        self,
        request: InferenceRequest,
    ) -> DecodeResult:
        """
        Run decode with KV cache from prefill node.

        Args:
            request: Inference request.

        Returns:
            DecodeResult with generated tokens.
        """
        logger.debug(f"DecodeNode {self.node_id}: Processing request {request.id}")

        # Wait for KV cache from prefill node
        kv_result = await self.kv_receiver.receive_async(request.id)

        if kv_result is None:
            logger.error(f"KV cache transfer timeout for request {request.id}")
            return DecodeResult(
                request_id=request.id,
                output_tokens=[],
                finish_reason="error",
            )

        key_cache, value_cache, num_prompt_tokens = kv_result

        # Move KV cache to device
        key_cache = [k.to(self.device) for k in key_cache]
        value_cache = [v.to(self.device) for v in value_cache]

        # Decode loop
        output_tokens = []
        context = list(request.input_ids)
        finish_reason = "stop"

        import numpy as np

        for step in range(request.max_tokens):
            # N-gram speculative proposals (CPU)
            spec_tokens = []
            if self.ngram_worker:
                proposals = self.ngram_worker.get_proposals(
                    np.array(context, dtype=np.int32)
                )
                if proposals is not None:
                    spec_tokens = proposals.tolist()

            # Forward pass (with or without speculation)
            if spec_tokens:
                # Speculative forward - verify multiple tokens
                next_tokens, accepted_mask = self._speculative_forward(
                    context, spec_tokens, key_cache, value_cache
                )

                # Record acceptance stats
                if self.ngram_worker:
                    self.ngram_worker.record_verification_result(
                        np.array(spec_tokens),
                        np.array(accepted_mask),
                    )

                output_tokens.extend(next_tokens)
                context.extend(next_tokens)

            else:
                # Standard forward - one token at a time
                next_token = self._forward_one_token(
                    context, key_cache, value_cache
                )
                output_tokens.append(next_token)
                context.append(next_token)

            # Check for EOS
            # TODO: Use actual EOS token from tokenizer
            if len(output_tokens) >= request.max_tokens:
                finish_reason = "length"
                break

        # Update N-gram table with generated tokens
        if self.ngram_worker:
            self.ngram_worker.submit_output(output_tokens)

        return DecodeResult(
            request_id=request.id,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )

    def _speculative_forward(
        self,
        context: List[int],
        spec_tokens: List[int],
        key_cache: List[torch.Tensor],
        value_cache: List[torch.Tensor],
    ) -> Tuple[List[int], List[bool]]:
        """
        Run speculative forward pass with verification.

        Returns:
            Tuple of (accepted_tokens, acceptance_mask).
        """
        # Placeholder: In production, this would run the model
        # and verify speculative tokens

        # Simulate acceptance (50% acceptance rate for demo)
        import random
        accepted = []
        mask = []

        for token in spec_tokens:
            if random.random() < 0.5:
                accepted.append(token)
                mask.append(True)
            else:
                # Generate correct token (placeholder)
                accepted.append(random.randint(0, 32000))
                mask.append(False)
                break

        # If no speculative tokens, generate one
        if not accepted:
            accepted.append(random.randint(0, 32000))
            mask.append(True)

        return accepted, mask

    def _forward_one_token(
        self,
        context: List[int],
        key_cache: List[torch.Tensor],
        value_cache: List[torch.Tensor],
    ) -> int:
        """
        Run single-token forward pass.

        Returns:
            Generated token ID.
        """
        # Placeholder: In production, this would run the model
        import random
        return random.randint(0, 32000)


class DisaggregatedCoordinator:
    """
    Coordinator for disaggregated Prefill/Decode serving.

    Manages request routing between Prefill and Decode node pools
    with load balancing and KV cache transfer orchestration.
    """

    def __init__(
        self,
        config: DisaggregatedConfig,
        model_config: ModelConfig,
        vllm_config: Optional[VllmConfig] = None,
    ):
        """
        Initialize the coordinator.

        Args:
            config: Disaggregated serving configuration.
            model_config: Model configuration.
            vllm_config: Full vLLM configuration.
        """
        self.config = config
        self.model_config = model_config

        # KV transfer components
        self._kv_sender = KVCacheSender(config.kv_transfer)
        self._kv_receiver = KVCacheReceiver(config.kv_transfer)

        # Initialize node pools
        self.prefill_nodes: List[PrefillNode] = []
        self.decode_nodes: List[DecodeNode] = []

        for i in range(config.num_prefill_nodes):
            node = PrefillNode(
                node_id=i,
                model_config=model_config,
                device=config.prefill_device,
                kv_sender=self._kv_sender,
            )
            self.prefill_nodes.append(node)

        for i in range(config.num_decode_nodes):
            node = DecodeNode(
                node_id=i,
                model_config=model_config,
                device=config.decode_device,
                kv_receiver=self._kv_receiver,
                moe_offload_config=config.moe_offload,
                ngram_config=config.ngram_spec,
                vllm_config=vllm_config,
            )
            self.decode_nodes.append(node)

        # Load balancers
        self.prefill_lb = LoadBalancer(
            self.prefill_nodes,
            config.load_balance_strategy,
        )
        self.decode_lb = LoadBalancer(
            self.decode_nodes,
            config.load_balance_strategy,
        )

        logger.info(
            f"DisaggregatedCoordinator initialized: "
            f"{config.num_prefill_nodes} prefill nodes, "
            f"{config.num_decode_nodes} decode nodes"
        )

    async def start(self):
        """Start the coordinator and receivers."""
        await self._kv_receiver.start()
        logger.info("Coordinator started")

    async def process_request(
        self,
        input_ids: List[int],
        max_tokens: int,
        sampling_params: Optional[Dict] = None,
    ) -> InferenceResponse:
        """
        Process an inference request through the disaggregated pipeline.

        Args:
            input_ids: Input token IDs.
            max_tokens: Maximum tokens to generate.
            sampling_params: Optional sampling parameters.

        Returns:
            InferenceResponse with generated tokens.
        """
        request_id = str(uuid.uuid4())

        request = InferenceRequest(
            id=request_id,
            input_ids=input_ids,
            max_tokens=max_tokens,
            sampling_params=sampling_params or {},
        )

        try:
            # Select nodes
            prefill_idx, prefill_node = self.prefill_lb.select()
            decode_idx, decode_node = self.decode_lb.select()

            # Track load
            self.prefill_lb.increment_load(prefill_idx)
            self.decode_lb.increment_load(decode_idx)

            try:
                # Run prefill
                prefill_result = await prefill_node.run_prefill(
                    request, decode_idx
                )

                # Run decode
                decode_result = await decode_node.run_decode(request)

                return InferenceResponse(
                    request_id=request_id,
                    output_tokens=decode_result.output_tokens,
                    finish_reason=decode_result.finish_reason,
                    metadata={
                        "prefill_node": prefill_idx,
                        "decode_node": decode_idx,
                        "num_prompt_tokens": prefill_result.num_prompt_tokens,
                    },
                )

            finally:
                # Release load
                self.prefill_lb.decrement_load(prefill_idx)
                self.decode_lb.decrement_load(decode_idx)

        except Exception as e:
            logger.error(f"Request processing error: {e}")
            return InferenceResponse(
                request_id=request_id,
                output_tokens=[],
                finish_reason="error",
                metadata={"error": str(e)},
            )

    def get_stats(self) -> Dict:
        """Get coordinator statistics."""
        stats = {
            "num_prefill_nodes": len(self.prefill_nodes),
            "num_decode_nodes": len(self.decode_nodes),
            "prefill_loads": dict(self.prefill_lb._node_loads),
            "decode_loads": dict(self.decode_lb._node_loads),
        }

        # Add N-gram stats from decode nodes
        for i, node in enumerate(self.decode_nodes):
            if node.ngram_worker:
                stats[f"decode_node_{i}_ngram"] = node.ngram_worker.get_stats()

        return stats

    def shutdown(self):
        """Shutdown the coordinator."""
        self._kv_sender.close()
        self._kv_receiver.close()

        for node in self.decode_nodes:
            if node.ngram_worker:
                node.ngram_worker.shutdown()

        logger.info("Coordinator shutdown complete")
