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
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)


class HeterogeneousPlatform(Platform):
    _enum = PlatformEnum.CUDA  # Default to CUDA to pass initial checks, but dynamic dispatch is key
    device_name: str = "heterogeneous"
    device_type: str = "heterogeneous"
    dispatch_key: str = "CUDA" # Default to CUDA dispatch key for now

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
        # Return generic base, or CUDA one? 
        return cls._cuda_platform.get_device_communicator_cls()

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
