# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPU Model Runner for vLLM V1

Optimized for Intel Xeon processors with:
- NUMA-aware KVCache allocation
- AVX-512 optimized operations
- IPEX integration for accelerated inference
"""
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.cpu_attn import TorchSDPAMetadataBuilderV1
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.platforms.cpu import CpuPlatform
import vllm.attention.selector

# Intel CPU optimization utilities (optional, graceful fallback)
try:
    from vllm.platforms.intel_cpu_utils import (
        NUMAAllocator,
        create_numa_aware_tensor,
        is_ipex_available,
        optimize_model_with_ipex,
    )
    _INTEL_UTILS_AVAILABLE = True
except ImportError:
    _INTEL_UTILS_AVAILABLE = False
    NUMAAllocator = None  # type: ignore
    is_ipex_available = lambda: False  # type: ignore
    optimize_model_with_ipex = lambda m, **kw: m  # type: ignore


if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    """
    CPU Model Runner optimized for Intel Xeon processors.

    Key optimizations:
    - NUMA-aware KVCache allocation for optimal memory bandwidth
    - IPEX integration for accelerated inference
    - AVX-512 optimized tensor operations
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 numa_node: int = -1):
        # Store NUMA node before parent __init__ (which may allocate memory)
        self._numa_node = numa_node
        self._numa_allocator = None

        # Try to set up NUMA binding if available
        if _INTEL_UTILS_AVAILABLE and NUMAAllocator is not None and numa_node >= 0:
            try:
                self._numa_allocator = NUMAAllocator()
                if self._numa_allocator.is_available:
                    self._numa_allocator.bind_to_node(numa_node)
                    logger.info(f"CPUModelRunner bound to NUMA node {numa_node}")
                else:
                    self._numa_allocator = None
                    logger.debug("NUMA not available, using standard allocation")
            except Exception as e:
                logger.debug(f"NUMA setup skipped: {e}")
                self._numa_allocator = None

        super().__init__(vllm_config, device)

        assert device == torch.device("cpu")
        assert self.speculative_config is None, "spec decode is not supported."

        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        self._postprocess_tensors()

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """


        # Attention free models have zero kv_cache_goups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if len(self.kv_cache_config.kv_cache_groups) > 1:
            raise ValueError("Multiple KVCacheGroups is not"
                             "currently supported with CPU model runner.")

        assert isinstance(self.attn_groups[0]
                    [0].metadata_builder, TorchSDPAMetadataBuilderV1)

        self.attn_groups[0][0].metadata_builder.reorder_batch(
            self.input_batch, scheduler_output)

    def _postprocess_tensors(self) -> None:
        # Note: replace device tensors with cpu tensors
        def replace_tensor(obj: Any, cpu_attr_name: str,
                           device_attr_name) -> None:
            cpu_tensor = getattr(obj, cpu_attr_name, None)
            device_tensor = getattr(obj, device_attr_name, None)
            if cpu_tensor is not None and device_tensor is not None:
                assert isinstance(cpu_tensor, torch.Tensor)
                assert isinstance(device_tensor, torch.Tensor)
                setattr(obj, device_attr_name, cpu_tensor)

        for k, v in vars(self).items():
            if k.endswith("_cpu") and isinstance(v, torch.Tensor):
                replace_tensor(self, k, k[:-4])

        for k, v in vars(self.input_batch).items():
            if k.endswith("_cpu_tensor") and isinstance(v, torch.Tensor):
                replace_tensor(self.input_batch, k, k[:-11])

        for block_table in self.input_batch.block_table.block_tables:
            for k, v in vars(block_table).items():
                if k.endswith("_cpu") and isinstance(v, torch.Tensor):
                    replace_tensor(block_table, k, k[:-4])

    def _allocate_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        NUMA-aware KVCache allocation for CPU.

        Allocates KVCache tensors on the local NUMA node for optimal
        memory bandwidth on multi-socket Intel Xeon systems.
        """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}

        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            size = kv_cache_tensor.size

            # Use NUMA-aware allocation if available
            if (_INTEL_UTILS_AVAILABLE and
                self._numa_allocator is not None and
                self._numa_allocator.is_available and
                self._numa_node >= 0):

                # Ensure allocation happens on the correct NUMA node
                self._numa_allocator.bind_to_node(self._numa_node)
                tensor = torch.zeros(size, dtype=torch.int8, device='cpu')
                logger.info(f"Allocated KVCache ({size / (1024**3):.2f} GiB) "
                           f"on NUMA node {self._numa_node}")
            else:
                # Standard allocation (fallback)
                tensor = torch.zeros(size, dtype=torch.int8, device='cpu')
                logger.info(f"Allocated KVCache ({size / (1024**3):.2f} GiB) "
                           f"without NUMA binding")

            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            layer_names.update(group.layer_names)
        assert layer_names == set(kv_cache_raw_tensors.keys()), \
            "Some layers are not correctly initialized"

        return kv_cache_raw_tensors

    def load_model(self, **kwargs) -> None:
        # Override device in config to ensure model loads on CPU
        original_device = self.vllm_config.device_config.device
        self.vllm_config.device_config.device = self.device

        # Patch current_platform in selector to force correct backend
        original_platform = vllm.attention.selector.current_platform
        vllm.attention.selector.current_platform = CpuPlatform
        vllm.attention.selector._cached_get_attn_backend.cache_clear()

        try:
            self.model = get_model(vllm_config=self.vllm_config)
        finally:
            self.vllm_config.device_config.device = original_device
            # Restore original platform and clear cache again
            vllm.attention.selector.current_platform = original_platform
            vllm.attention.selector._cached_get_attn_backend.cache_clear()

        self.model.to(self.device)

        # Apply IPEX optimization if available
        if _INTEL_UTILS_AVAILABLE and is_ipex_available():
            try:
                # Use bfloat16 for Sapphire Rapids with AMX
                dtype = self.model_config.dtype
                if dtype == torch.float32:
                    # bfloat16 is more efficient on Sapphire Rapids
                    dtype = torch.bfloat16
                self.model = optimize_model_with_ipex(self.model, dtype=dtype)
                logger.info(f"Model optimized with IPEX (dtype={dtype})")
            except Exception as e:
                logger.warning(f"IPEX optimization failed: {e}")

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.model_config,
                                              self.scheduler_config,
                                              self.lora_config, self.device)

    def get_model(self) -> nn.Module:
        return self.model

    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        with _set_global_compilation_settings(self.vllm_config):
            self._dummy_run(max(16, self.max_num_reqs))
        logger.info("Warming up done.")

    def _init_device_properties(self) -> None:
        pass

    def _sync_device(self) -> None:
        pass

    def profile_run(self) -> None:
        """
        CPU-specific profile run.
        Simplified version without CUDA graph capture.
        """
        import gc
        from vllm.distributed.parallel_state import get_pp_group

        logger.info(f"CPUModelRunner profile_run start. device={self.device}")

        # Run a dummy forward pass to initialize any lazy modules
        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True)

        logger.info(f"CPUModelRunner profile_run dummy_run finished.")

        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                output = self._dummy_pooler_run(hidden_states)
            else:
                output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None

        del hidden_states, output
        gc.collect()

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        capture_attn_cudagraph: bool = False,
        skip_eplb: bool = False,
        is_profile: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CPU-specific dummy run for profiling.
        Simplified version without CUDA-specific features.
        """
        import numpy as np
        from vllm.distributed.parallel_state import get_pp_group
        from vllm.forward_context import set_forward_context
        from vllm.multimodal.inputs import MultiModalKwargs

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
        num_tokens += num_pad

        logger.info(f"CPUModelRunner _dummy_run start. device={self.device} num_tokens={num_tokens}")

        # Set num_scheduled_tokens
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)

        # No CUDA graph capture on CPU
        attn_metadata = None

        with self.maybe_dummy_run_with_lora(self.lora_config, num_scheduled_tokens):
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
                model_mm_kwargs = self._dummy_mm_kwargs(num_reqs)
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None
                model_mm_kwargs = {}

            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device))
                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens, None, False)

            with self.maybe_randomize_inputs(input_ids), set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **MultiModalKwargs.as_kwargs(
                        model_mm_kwargs,
                        device=self.device,
                    ),
                )
            logger.info(f"CPUModelRunner _dummy_run model execution finished.")

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

        # Skip EPLB for CPU (not applicable)
        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return hidden_states, hidden_states[logit_indices]


@contextmanager
def _set_global_compilation_settings(config: VllmConfig):
    import torch._inductor.config

    inductor_config = config.compilation_config.inductor_compile_config
    try:
        # Note: The MKLDNN and CPPGEMM backend requires freezing parameters.
        freezing_value = torch._inductor.config.freezing
        if inductor_config.get("max_autotune", False):
            torch._inductor.config.freezing = True
        yield
    finally:
        torch._inductor.config.freezing = freezing_value
