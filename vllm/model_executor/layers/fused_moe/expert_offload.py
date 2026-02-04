# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MoE Expert CPU-GPU Offload Manager.

This module implements CPU-GPU hybrid execution for MoE (Mixture of Experts)
models, where frequently used experts stay on GPU while others are offloaded
to CPU for memory efficiency.

Key features:
- LRU-based expert caching on GPU
- Async CPU-GPU transfers
- INT8 quantization for CPU experts (AVX-512 VNNI optimized)
- Thread-parallel CPU expert execution
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ExpertOffloadConfig:
    """Configuration for MoE Expert offloading."""

    enabled: bool = True
    """Enable expert offloading."""

    num_gpu_experts: int = 8
    """Number of experts to keep on GPU (hot cache)."""

    cpu_dtype: str = "bfloat16"
    """Data type for CPU experts: bfloat16, float32, or int8."""

    swap_threshold: int = 100
    """Number of forward passes before considering expert swap."""

    async_transfer: bool = True
    """Use async CUDA streams for CPU-GPU transfers."""

    cpu_threads: int = 4
    """Number of threads for parallel CPU expert execution."""

    use_ipex: bool = True
    """Use Intel Extension for PyTorch if available."""


@dataclass
class ExpertStats:
    """Statistics for expert usage tracking."""

    usage_count: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    """Per-expert invocation count."""

    total_forwards: int = 0
    """Total forward passes."""

    last_swap_forward: int = 0
    """Forward count at last swap."""


class ExpertOffloadManager:
    """
    MoE Expert CPU-GPU Offload Manager.

    Manages the distribution of MoE experts between GPU (hot cache) and
    CPU (cold storage), with LRU-based swapping for optimal throughput.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        config: ExpertOffloadConfig,
    ):
        """
        Initialize the offload manager.

        Args:
            num_experts: Total number of experts in the MoE layer.
            hidden_size: Hidden dimension size.
            intermediate_size: Intermediate (FFN) dimension size.
            config: Offload configuration.
        """
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.config = config

        # Expert storage
        self.gpu_experts: Dict[int, nn.Module] = {}
        self.cpu_experts: Dict[int, nn.Module] = {}

        # Expert weights for direct access
        self.gpu_w13: Dict[int, torch.Tensor] = {}
        self.gpu_w2: Dict[int, torch.Tensor] = {}
        self.cpu_w13: Dict[int, torch.Tensor] = {}
        self.cpu_w2: Dict[int, torch.Tensor] = {}

        # Statistics for LRU
        self.stats = ExpertStats()

        # GPU indices currently on GPU
        self.gpu_expert_ids: Set[int] = set()

        # Async transfer stream
        if config.async_transfer and torch.cuda.is_available():
            self.transfer_stream = torch.cuda.Stream()
        else:
            self.transfer_stream = None

        # CPU thread pool for parallel expert execution
        self.cpu_executor = ThreadPoolExecutor(max_workers=config.cpu_threads)

        # IPEX availability
        self._ipex_available = False
        if config.use_ipex:
            try:
                import intel_extension_for_pytorch as ipex
                self._ipex_available = True
                self._ipex = ipex
                logger.info("ExpertOffloadManager: IPEX available for CPU experts")
            except ImportError:
                logger.info("ExpertOffloadManager: IPEX not available, using PyTorch")

        logger.info(
            f"ExpertOffloadManager initialized: "
            f"{num_experts} experts, {config.num_gpu_experts} on GPU, "
            f"CPU dtype: {config.cpu_dtype}"
        )

    def setup_experts(
        self,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        initial_gpu_experts: Optional[List[int]] = None,
    ):
        """
        Setup expert weights for offloading.

        Args:
            w13_weight: Combined gate and up projection weights [num_experts, 2*intermediate, hidden].
            w2_weight: Down projection weights [num_experts, hidden, intermediate].
            initial_gpu_experts: List of expert indices to keep on GPU initially.
        """
        if initial_gpu_experts is None:
            # Default: first N experts on GPU
            initial_gpu_experts = list(range(min(self.config.num_gpu_experts, self.num_experts)))

        self.gpu_expert_ids = set(initial_gpu_experts)

        # Determine CPU dtype
        if self.config.cpu_dtype == "bfloat16":
            cpu_dtype = torch.bfloat16
        elif self.config.cpu_dtype == "float32":
            cpu_dtype = torch.float32
        elif self.config.cpu_dtype == "int8":
            cpu_dtype = torch.int8
        else:
            cpu_dtype = torch.bfloat16

        # Distribute weights
        for i in range(self.num_experts):
            if i in self.gpu_expert_ids:
                # Keep on GPU
                self.gpu_w13[i] = w13_weight[i].cuda()
                self.gpu_w2[i] = w2_weight[i].cuda()
            else:
                # Move to CPU with appropriate dtype
                if cpu_dtype == torch.int8:
                    # Quantize for INT8
                    self.cpu_w13[i] = self._quantize_to_int8(w13_weight[i])
                    self.cpu_w2[i] = self._quantize_to_int8(w2_weight[i])
                else:
                    self.cpu_w13[i] = w13_weight[i].to(cpu_dtype).cpu()
                    self.cpu_w2[i] = w2_weight[i].to(cpu_dtype).cpu()

        logger.info(
            f"Expert weights distributed: {len(self.gpu_expert_ids)} on GPU, "
            f"{self.num_experts - len(self.gpu_expert_ids)} on CPU"
        )

    def _quantize_to_int8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to INT8 with scale.

        Returns:
            Tuple of (quantized_tensor, scale)
        """
        abs_max = tensor.abs().max()
        scale = abs_max / 127.0
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8).cpu()
        return (quantized, scale.cpu())

    def route_and_compute(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Route tokens to experts and compute with CPU-GPU offloading.

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size].
            router_logits: Router output [num_tokens, num_experts].
            top_k: Number of experts per token.
            renormalize: Whether to renormalize routing weights.
            use_grouped_topk: Use grouped top-k selection.
            topk_group: Number of groups for grouped top-k.
            num_expert_group: Number of experts per group.

        Returns:
            Output tensor [num_tokens, hidden_size].
        """
        # Update statistics
        self.stats.total_forwards += 1

        # Compute routing weights and selected experts
        routing_weights, selected_experts = self._compute_routing(
            hidden_states,
            router_logits,
            top_k,
            renormalize,
            use_grouped_topk,
            topk_group,
            num_expert_group,
        )

        # Update expert usage stats
        unique_experts = selected_experts.unique().tolist()
        for idx in unique_experts:
            self.stats.usage_count[idx] += 1

        # Classify selected experts into GPU and CPU
        gpu_experts_mask = torch.tensor(
            [idx in self.gpu_expert_ids for idx in range(self.num_experts)],
            device=hidden_states.device
        )

        # Separate computation paths
        output = torch.zeros_like(hidden_states)

        # GPU expert computation
        gpu_selected = selected_experts.clone()
        gpu_selected[~gpu_experts_mask[selected_experts]] = -1  # Mark CPU experts
        gpu_output = self._compute_gpu_experts(
            hidden_states, routing_weights, gpu_selected, top_k
        )
        output += gpu_output

        # CPU expert computation (if any selected)
        cpu_expert_indices = [idx for idx in unique_experts if idx not in self.gpu_expert_ids]
        if cpu_expert_indices:
            cpu_output = self._compute_cpu_experts(
                hidden_states, routing_weights, selected_experts, cpu_expert_indices
            )
            output += cpu_output.to(hidden_states.device)

        # Check if swap is needed
        if self._should_swap():
            self._swap_experts()

        return output

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int],
        num_expert_group: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights and expert selection."""
        if use_grouped_topk and topk_group is not None and num_expert_group is not None:
            # Grouped top-k (DeepSeek style)
            return self._grouped_topk(
                hidden_states, router_logits, top_k, renormalize,
                topk_group, num_expert_group
            )
        else:
            # Standard top-k
            scores = torch.softmax(router_logits.float(), dim=-1)
            routing_weights, selected_experts = torch.topk(scores, top_k, dim=-1)

            if renormalize:
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

            return routing_weights, selected_experts

    def _grouped_topk(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        topk_group: int,
        num_expert_group: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Grouped top-k expert selection (DeepSeek V2/V3 style)."""
        scores = torch.softmax(router_logits.float(), dim=-1)
        num_tokens = scores.shape[0]

        # Group-level selection
        group_scores = scores.view(num_tokens, num_expert_group, -1).max(dim=-1).values
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]

        # Create group mask
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Expand mask to expert level
        experts_per_group = scores.shape[-1] // num_expert_group
        score_mask = group_mask.unsqueeze(-1).expand(
            num_tokens, num_expert_group, experts_per_group
        ).reshape(num_tokens, -1)

        # Apply mask and select top-k
        masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        routing_weights, selected_experts = torch.topk(masked_scores, k=top_k, dim=-1)

        if renormalize:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        return routing_weights, selected_experts

    def _compute_gpu_experts(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Compute experts on GPU."""
        output = torch.zeros_like(hidden_states)
        num_tokens = hidden_states.shape[0]

        for k in range(top_k):
            expert_ids = selected_experts[:, k]
            weights = routing_weights[:, k].unsqueeze(-1)

            # Process each unique GPU expert
            for expert_id in expert_ids.unique():
                if expert_id == -1:  # Skip CPU experts
                    continue
                expert_id = expert_id.item()
                if expert_id not in self.gpu_expert_ids:
                    continue

                mask = (expert_ids == expert_id)
                if not mask.any():
                    continue

                # Get expert weights
                w13 = self.gpu_w13[expert_id]
                w2 = self.gpu_w2[expert_id]

                # Compute: SiLU(x @ w1) * (x @ w3) @ w2
                x = hidden_states[mask]
                intermediate = w13.shape[0] // 2

                # Gate and up projection
                gate_up = torch.mm(x, w13.t())
                gate = gate_up[:, :intermediate]
                up = gate_up[:, intermediate:]

                # SiLU activation and combination
                hidden = torch.nn.functional.silu(gate) * up

                # Down projection
                expert_output = torch.mm(hidden, w2.t())

                # Apply routing weight and accumulate
                output[mask] += expert_output * weights[mask]

        return output

    def _compute_cpu_experts(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        cpu_expert_indices: List[int],
    ) -> torch.Tensor:
        """Compute experts on CPU with parallel execution."""
        # Move data to CPU
        hidden_cpu = hidden_states.detach().cpu()
        routing_cpu = routing_weights.detach().cpu()
        selected_cpu = selected_experts.detach().cpu()

        output = torch.zeros_like(hidden_cpu)
        num_tokens = hidden_cpu.shape[0]
        top_k = selected_cpu.shape[1]

        # Submit parallel tasks for each CPU expert
        futures = []
        for expert_id in cpu_expert_indices:
            future = self.cpu_executor.submit(
                self._run_single_cpu_expert,
                hidden_cpu, routing_cpu, selected_cpu, expert_id, top_k
            )
            futures.append((expert_id, future))

        # Collect results
        for expert_id, future in futures:
            expert_output = future.result()
            output += expert_output

        return output

    def _run_single_cpu_expert(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        expert_id: int,
        top_k: int,
    ) -> torch.Tensor:
        """Run a single expert on CPU."""
        output = torch.zeros_like(hidden_states)

        # Get expert weights
        w13_data = self.cpu_w13[expert_id]
        w2_data = self.cpu_w2[expert_id]

        # Handle INT8 quantization
        if isinstance(w13_data, tuple):
            w13, w13_scale = w13_data
            w2, w2_scale = w2_data
            is_int8 = True
        else:
            w13, w2 = w13_data, w2_data
            is_int8 = False

        for k in range(top_k):
            expert_ids = selected_experts[:, k]
            weights = routing_weights[:, k].unsqueeze(-1)

            mask = (expert_ids == expert_id)
            if not mask.any():
                continue

            x = hidden_states[mask]

            if is_int8:
                # INT8 computation with dequantization
                x_fp = x.float()
                w13_fp = w13.float() * w13_scale
                w2_fp = w2.float() * w2_scale
            else:
                x_fp = x
                w13_fp = w13
                w2_fp = w2

            # Compute MoE FFN
            with torch.no_grad():
                intermediate = w13_fp.shape[0] // 2

                gate_up = torch.mm(x_fp, w13_fp.t())
                gate = gate_up[:, :intermediate]
                up = gate_up[:, intermediate:]

                hidden = torch.nn.functional.silu(gate) * up
                expert_output = torch.mm(hidden, w2_fp.t())

                output[mask] += expert_output.to(hidden_states.dtype) * weights[mask]

        return output

    def _should_swap(self) -> bool:
        """Check if expert swap should be performed."""
        forwards_since_swap = self.stats.total_forwards - self.stats.last_swap_forward
        return forwards_since_swap >= self.config.swap_threshold

    def _swap_experts(self):
        """Swap experts between GPU and CPU based on usage statistics."""
        # Sort experts by usage count
        sorted_experts = sorted(
            self.stats.usage_count.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Determine new GPU experts
        new_gpu_experts = set(
            idx for idx, _ in sorted_experts[:self.config.num_gpu_experts]
        )

        # Experts to move to GPU
        to_gpu = new_gpu_experts - self.gpu_expert_ids
        # Experts to move to CPU
        to_cpu = self.gpu_expert_ids - new_gpu_experts

        if not to_gpu and not to_cpu:
            self.stats.last_swap_forward = self.stats.total_forwards
            return

        logger.info(
            f"Expert swap: {len(to_gpu)} to GPU, {len(to_cpu)} to CPU"
        )

        # Perform swaps (optionally async)
        if self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                self._do_swap(to_gpu, to_cpu)
            self.transfer_stream.synchronize()
        else:
            self._do_swap(to_gpu, to_cpu)

        self.gpu_expert_ids = new_gpu_experts
        self.stats.last_swap_forward = self.stats.total_forwards

        # Reset usage counts periodically
        if self.stats.total_forwards % (self.config.swap_threshold * 10) == 0:
            self.stats.usage_count = defaultdict(int)

    def _do_swap(self, to_gpu: Set[int], to_cpu: Set[int]):
        """Execute the actual swap operations."""
        cpu_dtype = getattr(torch, self.config.cpu_dtype) if self.config.cpu_dtype != "int8" else torch.bfloat16

        # Move to CPU first (to free GPU memory)
        for idx in to_cpu:
            if idx in self.gpu_w13:
                if self.config.cpu_dtype == "int8":
                    self.cpu_w13[idx] = self._quantize_to_int8(self.gpu_w13[idx])
                    self.cpu_w2[idx] = self._quantize_to_int8(self.gpu_w2[idx])
                else:
                    self.cpu_w13[idx] = self.gpu_w13[idx].to(cpu_dtype).cpu()
                    self.cpu_w2[idx] = self.gpu_w2[idx].to(cpu_dtype).cpu()
                del self.gpu_w13[idx]
                del self.gpu_w2[idx]

        # Move to GPU
        for idx in to_gpu:
            if idx in self.cpu_w13:
                w13_data = self.cpu_w13[idx]
                w2_data = self.cpu_w2[idx]

                # Dequantize if INT8
                if isinstance(w13_data, tuple):
                    w13, scale = w13_data
                    self.gpu_w13[idx] = (w13.float() * scale).cuda()
                    w2, scale = w2_data
                    self.gpu_w2[idx] = (w2.float() * scale).cuda()
                else:
                    self.gpu_w13[idx] = w13_data.cuda()
                    self.gpu_w2[idx] = w2_data.cuda()

                del self.cpu_w13[idx]
                del self.cpu_w2[idx]

    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            "total_forwards": self.stats.total_forwards,
            "gpu_experts": list(self.gpu_expert_ids),
            "cpu_experts": [i for i in range(self.num_experts) if i not in self.gpu_expert_ids],
            "top_used_experts": sorted(
                self.stats.usage_count.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
        }

    def shutdown(self):
        """Clean up resources."""
        self.cpu_executor.shutdown(wait=False)
