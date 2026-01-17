# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from vllm.logger import init_logger
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.interface import (DeviceCapability, Platform, PlatformEnum,
                                      _Backend)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig, ParallelConfig

logger = init_logger(__name__)


class HeterogeneousPlatform(Platform):
    _enum = PlatformEnum.CUDA  # Default to CUDA to pass initial checks, but dynamic dispatch is key
    device_name: str = "heterogeneous"
    device_type: str = "heterogeneous"
    device_type: str = "heterogeneous"
    dispatch_key: str = "CUDA" # Default to CUDA dispatch key for now
    ray_device_key: str = "GPU"
    dist_backend: str = "gloo" # Force Gloo for heterogeneous (CPU/GPU) coordination
    additional_env_vars: List[str] = ["VLLM_HETEROGENEOUS_PLATFORM"] # Ensure platform detection propagates



    # We need to maintain instances of sub-platforms
    _cuda_platform = CudaPlatform()
    _cpu_platform = CpuPlatform()
    
    # Context to know which platform is currently active for the calling thread/worker
    # This is tricky because Platform methods are classmethods. 
    # We might need to rely on the device_type passed in or global worker state.
    # For now, we will default to CUDA for global queries and try to be smart about specific ones.

    @classmethod
    def _get_platform(cls, device: Optional[Union[str, torch.device]] = None):
        # Semi-hack: Use device string to decide
        if device is not None:
            d = torch.device(device)
            if d.type == "cpu":
                return cls._cpu_platform
            if d.type == "cuda":
                return cls._cuda_platform
        return cls._cuda_platform # Default

    @property
    def supported_dtypes(self) -> List[torch.dtype]:
        return self._cuda_platform.supported_dtypes

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        # This is ambiguous in heterogeneous. Assuming GPU 0 for now if asked globally.
        return cls._cuda_platform.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return cls._cuda_platform.get_device_total_memory(device_id)

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        # If any part of the heterogeneous setup is CPU, we might need to be careful.
        # But usually async output is per-worker.
        return cls._cuda_platform.is_async_output_supported(enforce_eager)

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        # We run checks for both? Or just CUDA for now since it's stricter?
        cls._cuda_platform.check_and_update_config(vllm_config)
        # cls._cpu_platform.check_and_update_config(vllm_config) # CPU config might conflict

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        cls._cuda_platform.verify_model_arch(model_arch)

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        cls._cuda_platform.verify_quantization(quant)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.platforms.heterogeneous.HeterogeneousDeviceCommunicator"

    # Proxy other methods dynamically if possible, but they are class methods.
    # We have to implement them explicitly.

    @classmethod
    def is_cuda(cls) -> bool:
        return True # Pretend to be CUDA for compatibility

    @classmethod
    def is_cpu(cls) -> bool:
        return False 
        
    @classmethod
    def has_device_capability(cls, capability: Union[tuple[int, int], int], device_id: int = 0) -> bool:
        return cls._cuda_platform.has_device_capability(capability, device_id)

    @classmethod
    def device_count(cls) -> int:
        # Return CUDA count + 2 to allow for two CPU workers (one per NUMA node)
        return torch.cuda.device_count() + 2

    @classmethod
    def get_ray_placement_group_bundles(
            cls, parallel_config: "ParallelConfig") -> List[dict]:
        num_gpus = torch.cuda.device_count()
        bundles = []
        for i in range(parallel_config.world_size):
            if i < num_gpus:
                bundles.append({"GPU": 1.0})
            else:
                bundles.append({"CPU": 1.0})
        return bundles

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        if device.type == "cuda":
            cls._cuda_platform.set_device(device)
        elif device.type == "cpu":
            cls._cpu_platform.set_device(device)
        else:
            # Fallback or error
            logger.warning(f"Unknown device type for set_device: {device.type}. Defaulting to CUDA platform.")
            cls._cuda_platform.set_device(device)

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla):
        # Default to CUDA platform backend as Heterogeneous is primarily CUDA + CPU offload
        # BUT check if we are in a CPU worker process context
        device_env = os.environ.get("VLLM_HETEROGENEOUS_DEVICE")
        logger.info(f"HeterogeneousPlatform.get_attn_backend_cls called. VLLM_HETEROGENEOUS_DEVICE={device_env}")
        
        if device_env == "cpu":
             logger.info("Delegating to CPU Platform for Attention Backend.")
             return cls._cpu_platform.get_attn_backend_cls(
                selected_backend, head_size, dtype, kv_cache_dtype, block_size,
                use_v1, use_mla)
        
        logger.info("Delegating to CUDA Platform for Attention Backend.")
        return cls._cuda_platform.get_attn_backend_cls(
            selected_backend, head_size, dtype, kv_cache_dtype, block_size,
            use_v1, use_mla)


from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
from vllm.distributed.device_communicators.cpu_communicator import _CPUSHMDistributed
from vllm.logger import init_logger
import os

logger = init_logger(__name__)

class HeterogeneousDeviceCommunicator(DeviceCommunicatorBase):
    def __init__(self, cpu_group, device=None, device_group=None, unique_name=""):
        # We force the use of the base class which relies on standard torch.distributed
        # commands (falling back to Gloo).
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        super().__init__(cpu_group, device, device_group, unique_name)
        
        # Try to initialize Shared Memory distributed backend for performance
        # This is much faster than Gloo for intra-node communication
        self.shm_dist = None
        if os.environ.get("VLLM_DIST_IDENT"):
            try:
                # _CPUSHMDistributed is compatible with our interface
                self.shm_dist = _CPUSHMDistributed(self)
                logger.info("Initialized Shared Memory communication for Heterogeneous Platform.")
            except Exception as e:
                logger.warning(f"Failed to initialize Shared Memory backend: {e}. Falling back to standard Gloo.")

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.shm_dist:
            # Use SHM
            if input_.is_cuda:
                # Copy to CPU, op, Copy back
                # Optimizable with pinned memory ideally, but .cpu() is safe baseline
                input_cpu = input_.cpu()
                self.shm_dist.all_reduce(input_cpu)
                return input_cpu.to(input_.device)
            else:
                self.shm_dist.all_reduce(input_)
                return input_
        else:
            # Fallback to Gloo (CPU Offload)
            orig_device = input_.device
            input_cpu = input_.cpu()
            # Use cpu_group for global communication
            torch.distributed.all_reduce(input_cpu, group=self.cpu_group)
            return input_cpu.to(orig_device)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
         # SHM implementation for all_gather (via all_gather_into_tensor logic mostly)
         # _CPUSHMDistributed has all_gather_into_tensor.
         # DeviceCommunicatorBase.all_gather uses all_gather_into_tensor.
         
         if self.shm_dist:
            orig_device = input_.device
            input_cpu = input_.cpu()
            
            input_size = input_cpu.size()
            output_size = (input_size[0] * self.world_size, ) + input_size[1:]
            output_tensor = torch.empty(output_size, dtype=input_cpu.dtype, device="cpu")
            
            self.shm_dist.all_gather_into_tensor(output_tensor, input_cpu)
            
            # Reshape logic matches base class
            output_tensor = output_tensor.reshape((self.world_size, ) + input_size)
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(input_size[:dim] +
                                                  (self.world_size *
                                                   input_size[dim], ) +
                                                  input_size[dim + 1:])
            
            return output_tensor.to(orig_device)
         else:
            orig_device = input_.device
            input_cpu = input_.cpu()
            # Use cpu_group for global communication
            # We can't use super().all_gather because it uses device_group
            # Implement simple all_gather using torch.distributed.all_gather
            
            world_size = torch.distributed.get_world_size(group=self.cpu_group)
            output_tensors = [torch.empty_like(input_cpu) for _ in range(world_size)]
            torch.distributed.all_gather(output_tensors, input_cpu, group=self.cpu_group)
            
            output_tensor = torch.cat(output_tensors, dim=dim)
            return output_tensor.to(orig_device)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # _CPUSHMDistributed does NOT implement reduce_scatter!
        # cpu_communicator.py only shows all_reduce, gather, all_gather_into_tensor, send, recv.
        # No reduce_scatter in the snippet I viewed (lines 113-202).
        # So we MUST fallback to standard Gloo for this, or emulate it via all_reduce (slow) or gather/reduce.
        # DeviceCommunicatorBase implements reduce_scatter using torch.distributed.reduce_scatter_tensor.
        # So we fallback to super().reduce_scatter(cpu) which uses Gloo.
        
        orig_device = input_.device
        input_cpu = input_.cpu()
        
        # Check if reduce_scatter_tensor is available/supported by Gloo
        # It usually is.
        # But DeviceCommunicatorBase implementation is complex.
        # Let's try to use torch.distributed.reduce_scatter_tensor directly on cpu_group
        
        output_size = list(input_cpu.size())
        output_size[dim] //= self.world_size
        output_tensor = torch.empty(output_size, dtype=input_cpu.dtype, device="cpu")
        
        try:
            torch.distributed.reduce_scatter_tensor(output_tensor, input_cpu, group=self.cpu_group)
        except (AttributeError, RuntimeError):
             # Fallback if reduce_scatter_tensor not supported (e.g. older Gloo?)
             # Naive implementation: AllReduce -> Slice
             torch.distributed.all_reduce(input_cpu, group=self.cpu_group)
             # Slice the part for this rank
             rank = torch.distributed.get_rank(group=self.cpu_group)
             slices = torch.chunk(input_cpu, self.world_size, dim=dim)
             output_tensor.copy_(slices[rank])

        return output_tensor.to(orig_device)

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor:
        # _CPUSHMDistributed doesn't explicitly implement broadcast?
        # Check snippet: it has send, recv, gather, all_gather. No broadcast.
        # So fallback to Gloo.
        
        orig_device = input_.device
        input_cpu = input_.cpu()
        torch.distributed.broadcast(input_cpu, src=src, group=self.cpu_group)
        return input_cpu.to(orig_device)
        
    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        if self.shm_dist:
             # SHM send optimization if available, else Gloo
             pass
        
        tensor_cpu = tensor.cpu()
        torch.distributed.send(tensor_cpu, dst, group=self.cpu_group)

    def recv(self, size: torch.Size, dtype: torch.dtype, src: Optional[int] = None) -> torch.Tensor:
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        tensor = torch.empty(size, dtype=dtype, device="cpu")
        # Use cpu_group
        torch.distributed.recv(tensor, self.ranks[src], self.cpu_group)
        return tensor.to(self.device)





