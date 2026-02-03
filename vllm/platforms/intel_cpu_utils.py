# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Intel CPU Optimization Utilities for vLLM

Provides NUMA-aware memory allocation, AVX-512 optimization,
and Intel-specific performance tuning for Xeon processors.

Optimized for Intel Xeon Platinum 8480+ (Sapphire Rapids) and similar.

NOTE: This module is designed to gracefully degrade on systems without:
- AVX-512 support (falls back to AVX2 or standard ops)
- libnuma library (falls back to standard memory allocation)
- IPEX (falls back to standard PyTorch)
"""

import ctypes
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging

import torch

logger = logging.getLogger(__name__)

# ============================================================================
# Feature Detection Flags (set during module import)
# ============================================================================
_LIBNUMA_AVAILABLE = False
_AVX512_AVAILABLE = False
_AMX_AVAILABLE = False

# ============================================================================
# NUMA Utilities
# ============================================================================

@dataclass
class NUMANodeInfo:
    """Information about a NUMA node."""
    node_id: int
    cpu_ids: List[int]
    total_memory_bytes: int
    free_memory_bytes: int


class NUMAAllocator:
    """
    NUMA-aware memory allocator for CPU tensors.

    Uses libnuma for optimal memory placement on multi-socket systems
    like Intel Xeon Platinum 8480+.

    Gracefully falls back to standard allocation if libnuma is not available.
    """

    _instance: Optional['NUMAAllocator'] = None
    _libnuma: Optional[ctypes.CDLL] = None
    _numa_available: bool = False
    _num_nodes: int = 1

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize NUMA support with graceful fallback."""
        global _LIBNUMA_AVAILABLE

        self._numa_available = False
        self._num_nodes = 1
        self._libnuma = None

        if platform.system() != "Linux":
            logger.debug("NUMA support is only available on Linux")
            return

        try:
            self._libnuma = ctypes.CDLL("libnuma.so.1", mode=ctypes.RTLD_GLOBAL)

            # Check if NUMA is available
            numa_available = self._libnuma.numa_available()
            if numa_available == -1:
                logger.debug("NUMA not available on this system (single node or disabled)")
                self._libnuma = None
                return

            self._numa_available = True
            _LIBNUMA_AVAILABLE = True
            self._num_nodes = self._libnuma.numa_num_configured_nodes()

            # Set up function signatures
            self._libnuma.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
            self._libnuma.numa_alloc_onnode.restype = ctypes.c_void_p

            self._libnuma.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            self._libnuma.numa_free.restype = None

            self._libnuma.numa_node_size64.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
            self._libnuma.numa_node_size64.restype = ctypes.c_longlong

            self._libnuma.numa_set_preferred.argtypes = [ctypes.c_int]
            self._libnuma.numa_set_preferred.restype = None

            logger.info(f"NUMA initialized: {self._num_nodes} nodes available")

        except OSError as e:
            logger.debug(f"libnuma not available: {e}. Using standard memory allocation.")
            self._libnuma = None
            self._numa_available = False

    @property
    def is_available(self) -> bool:
        return self._numa_available

    @property
    def num_nodes(self) -> int:
        return self._num_nodes if self._numa_available else 1

    def get_node_info(self, node_id: int) -> Optional[NUMANodeInfo]:
        """Get information about a specific NUMA node."""
        if not self._numa_available:
            return None

        if node_id < 0 or node_id >= self._num_nodes:
            return None

        free_mem = ctypes.c_longlong()
        total_mem = self._libnuma.numa_node_size64(node_id, ctypes.byref(free_mem))

        # Get CPU IDs for this node
        cpu_ids = self._get_cpus_for_node(node_id)

        return NUMANodeInfo(
            node_id=node_id,
            cpu_ids=cpu_ids,
            total_memory_bytes=total_mem,
            free_memory_bytes=free_mem.value
        )

    def _get_cpus_for_node(self, node_id: int) -> List[int]:
        """Get list of CPU IDs belonging to a NUMA node."""
        try:
            result = subprocess.run(
                ["numactl", "--hardware"],
                capture_output=True,
                text=True,
                timeout=5
            )

            for line in result.stdout.split('\n'):
                if line.startswith(f"node {node_id} cpus:"):
                    cpu_str = line.split(':')[1].strip()
                    if cpu_str:
                        return [int(x) for x in cpu_str.split()]
            return []
        except Exception:
            return []

    def allocate_on_node(self, size_bytes: int, node_id: int) -> Optional[int]:
        """
        Allocate memory on a specific NUMA node.

        Returns the memory address or None on failure.
        """
        if not self._numa_available:
            return None

        if node_id < 0 or node_id >= self._num_nodes:
            logger.warning(f"Invalid NUMA node {node_id}, using node 0")
            node_id = 0

        ptr = self._libnuma.numa_alloc_onnode(size_bytes, node_id)
        if ptr == 0 or ptr is None:
            logger.error(f"Failed to allocate {size_bytes} bytes on NUMA node {node_id}")
            return None

        return ptr

    def free(self, ptr: int, size_bytes: int):
        """Free NUMA-allocated memory."""
        if self._numa_available and ptr:
            self._libnuma.numa_free(ptr, size_bytes)

    def get_preferred_node_for_rank(self, rank: int, world_size: int) -> int:
        """
        Get the preferred NUMA node for a given rank.

        Distributes ranks across NUMA nodes for optimal memory bandwidth.
        """
        if not self._numa_available:
            return 0
        return rank % self._num_nodes

    def bind_to_node(self, node_id: int) -> bool:
        """
        Bind current thread's memory allocations to a specific NUMA node.

        Returns True on success.
        """
        if not self._numa_available:
            return False

        try:
            self._libnuma.numa_set_preferred(node_id)
            return True
        except Exception as e:
            logger.warning(f"Failed to bind to NUMA node {node_id}: {e}")
            return False


def create_numa_aware_tensor(
    size: int,
    dtype: torch.dtype,
    numa_node: int,
    pin_memory: bool = False
) -> torch.Tensor:
    """
    Create a tensor with memory allocated on a specific NUMA node.

    Args:
        size: Size in bytes
        dtype: Tensor dtype
        numa_node: Target NUMA node ID
        pin_memory: Whether to pin memory (for faster GPU transfers)

    Returns:
        torch.Tensor allocated on the specified NUMA node
    """
    allocator = NUMAAllocator()

    if not allocator.is_available:
        # Fallback to standard allocation
        return torch.zeros(size, dtype=torch.int8, device='cpu')

    # Bind memory allocation to the target NUMA node
    allocator.bind_to_node(numa_node)

    # Allocate tensor (will use the bound NUMA node)
    tensor = torch.zeros(size, dtype=torch.int8, device='cpu')

    logger.debug(f"Allocated {size} bytes tensor on NUMA node {numa_node}")
    return tensor


# ============================================================================
# AVX-512 and Intel Optimization Utilities
# ============================================================================

@dataclass
class IntelCPUFeatures:
    """Detected Intel CPU features."""
    # AVX-2 (baseline for modern optimization)
    avx2: bool = False
    avx_vnni: bool = False  # AVX-VNNI (Alder Lake+, without AVX-512)

    # AVX-512 (Xeon, some consumer CPUs)
    avx512f: bool = False
    avx512_vnni: bool = False
    avx512_bf16: bool = False

    # AMX (Sapphire Rapids+)
    amx_bf16: bool = False
    amx_int8: bool = False

    # CPU topology
    model_name: str = ""
    num_sockets: int = 1
    cores_per_socket: int = 1
    threads_per_core: int = 1
    l3_cache_mb: float = 0.0


def detect_intel_cpu_features() -> IntelCPUFeatures:
    """
    Detect Intel CPU features for optimization.

    Detects various CPU features including:
    - AVX2 (baseline for modern CPUs)
    - AVX-512 with VNNI (Vector Neural Network Instructions)
    - AVX-512 BF16 (Brain Float 16)
    - AMX (Advanced Matrix Extensions) - Sapphire Rapids+

    Works on any Linux system, gracefully handles missing features.
    """
    global _AVX512_AVAILABLE, _AMX_AVAILABLE

    features = IntelCPUFeatures()

    if platform.system() != "Linux":
        logger.debug("CPU feature detection only supported on Linux")
        return features

    try:
        # Read CPU info
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()

        # Check for AVX-512 features
        if "avx512f" in cpuinfo:
            features.avx512f = True
            _AVX512_AVAILABLE = True
        if "avx512_vnni" in cpuinfo:
            features.avx512_vnni = True
        if "avx512_bf16" in cpuinfo:
            features.avx512_bf16 = True
        if "amx_bf16" in cpuinfo:
            features.amx_bf16 = True
            _AMX_AVAILABLE = True
        if "amx_int8" in cpuinfo:
            features.amx_int8 = True
            _AMX_AVAILABLE = True

        # Check for AVX2 (fallback for non-AVX512 systems)
        if "avx2" in cpuinfo:
            features.avx2 = True

        # Check for AVX-VNNI (available on Alder Lake without AVX-512)
        if "avx_vnni" in cpuinfo:
            features.avx_vnni = True

        # Get model name
        for line in cpuinfo.split('\n'):
            if "model name" in line:
                features.model_name = line.split(':')[1].strip()
                break

        # Get topology info using lscpu
        try:
            result = subprocess.run(
                ["lscpu"],
                capture_output=True,
                text=True,
                timeout=5
            )

            for line in result.stdout.split('\n'):
                if "Socket(s):" in line:
                    features.num_sockets = int(line.split(':')[1].strip())
                elif "Core(s) per socket:" in line:
                    features.cores_per_socket = int(line.split(':')[1].strip())
                elif "Thread(s) per core:" in line:
                    features.threads_per_core = int(line.split(':')[1].strip())
                elif "L3 cache:" in line:
                    cache_str = line.split(':')[1].strip()
                    if 'MiB' in cache_str:
                        features.l3_cache_mb = float(cache_str.replace('MiB', '').strip())
                    elif 'KiB' in cache_str:
                        features.l3_cache_mb = float(cache_str.replace('KiB', '').strip()) / 1024
        except Exception as e:
            logger.debug(f"lscpu not available: {e}")

    except Exception as e:
        logger.debug(f"Failed to detect CPU features: {e}")

    # Log detected features
    logger.info(f"CPU detected: {features.model_name}")
    if features.avx512f:
        logger.info("  AVX-512: Yes")
    elif features.avx2:
        logger.info("  AVX-512: No (AVX2 available)")
    if features.amx_bf16:
        logger.info("  AMX-BF16: Yes (Sapphire Rapids+)")

    return features


def configure_intel_optimizations(features: Optional[IntelCPUFeatures] = None) -> dict:
    """
    Configure environment variables for Intel CPU optimization.

    Works on any Intel/AMD CPU, with enhanced optimization for:
    - Intel Xeon with AVX-512
    - Intel Sapphire Rapids with AMX
    - Consumer CPUs with AVX2

    Returns dict of configured settings.
    """
    if features is None:
        features = detect_intel_cpu_features()

    settings = {}

    # =========================================
    # OpenMP Settings (works on all CPUs)
    # =========================================

    # Use Intel's OpenMP runtime if available
    os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
    settings["KMP_AFFINITY"] = os.environ["KMP_AFFINITY"]

    # Prevent threads from sleeping (better latency)
    os.environ.setdefault("KMP_BLOCKTIME", "1")
    settings["KMP_BLOCKTIME"] = os.environ["KMP_BLOCKTIME"]

    # Use performance barrier patterns
    os.environ.setdefault("KMP_FORKJOIN_BARRIER_PATTERN", "dist,dist")
    os.environ.setdefault("KMP_PLAIN_BARRIER_PATTERN", "dist,dist")
    os.environ.setdefault("KMP_REDUCTION_BARRIER_PATTERN", "dist,dist")

    # Disable CPU frequency throttling hint
    os.environ.setdefault("KMP_TPAUSE", "0")
    settings["KMP_TPAUSE"] = os.environ["KMP_TPAUSE"]

    # =========================================
    # MKL Settings (adaptive to CPU features)
    # =========================================

    # Enable MKL verbose for debugging (disabled by default)
    os.environ.setdefault("MKL_VERBOSE", "0")

    # Set MKL instruction set based on available features
    if features.avx512f:
        os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX512")
        settings["MKL_ENABLE_INSTRUCTIONS"] = "AVX512"
    elif features.avx2:
        os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX2")
        settings["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"

    # =========================================
    # Thread Settings
    # =========================================

    # Set optimal thread count based on physical cores
    total_physical_cores = features.num_sockets * features.cores_per_socket
    if total_physical_cores > 0:
        # Use all physical cores for computation
        os.environ.setdefault("OMP_NUM_THREADS", str(total_physical_cores))
        settings["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

    # =========================================
    # PyTorch Inductor Settings
    # =========================================

    # Enable max autotune for best kernel selection
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "1")
    settings["TORCHINDUCTOR_MAX_AUTOTUNE"] = os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"]

    # Use MKLDNN backend when possible
    os.environ.setdefault("TORCHINDUCTOR_CPP_BACKEND", "1")
    settings["TORCHINDUCTOR_CPP_BACKEND"] = os.environ["TORCHINDUCTOR_CPP_BACKEND"]

    # Log configuration
    logger.info(f"CPU optimization configured: {features.model_name}")
    if features.avx512f:
        logger.info(f"  Features: AVX-512, VNNI={features.avx512_vnni}, "
                    f"BF16={features.avx512_bf16}, AMX={features.amx_bf16}")
    else:
        logger.info(f"  Features: AVX2={features.avx2}, AVX-VNNI={features.avx_vnni}")
    logger.info(f"  Topology: {features.num_sockets} socket(s) x "
                f"{features.cores_per_socket} cores x "
                f"{features.threads_per_core} threads")

    return settings


def configure_pytorch_for_intel(features: Optional[IntelCPUFeatures] = None):
    """
    Configure PyTorch settings optimized for Intel CPUs.

    Works on any CPU, with enhanced settings for AVX-512 systems.

    This includes:
    - Thread settings
    - Memory format preferences
    - Inductor backend settings (adaptive to CPU features)
    """
    if features is None:
        features = detect_intel_cpu_features()

    # Set thread count to physical cores
    total_physical_cores = features.num_sockets * features.cores_per_socket
    if total_physical_cores > 0:
        torch.set_num_threads(total_physical_cores)
        logger.info(f"PyTorch threads set to {total_physical_cores} (physical cores)")

    # Set interop threads for parallel regions
    interop_threads = max(1, features.num_sockets)
    torch.set_num_interop_threads(interop_threads)
    logger.debug(f"PyTorch interop threads set to {interop_threads}")

    # Enable high precision matmul
    torch.set_float32_matmul_precision('high')
    logger.debug("Float32 matmul precision set to 'high'")

    # Configure Inductor for Intel CPUs
    try:
        import torch._inductor.config as inductor_config

        # Enable freezing for better optimization
        inductor_config.freezing = True

        # Enable epilogue fusion
        inductor_config.epilogue_fusion = True

        # Enable pattern matching optimizations
        inductor_config.pattern_matcher = True

        # Use MKLDNN/oneDNN for CPU ops
        inductor_config.cpp_wrapper = True

        # Set SIMD length based on available instruction sets
        if features.avx512f:
            # AVX-512: 512-bit / 32-bit = 16 floats
            inductor_config.cpp.simdlen = 16
            logger.info("Inductor configured for AVX-512 (simdlen=16)")
        elif features.avx2:
            # AVX2: 256-bit / 32-bit = 8 floats
            inductor_config.cpp.simdlen = 8
            logger.info("Inductor configured for AVX2 (simdlen=8)")

    except ImportError:
        logger.debug("torch._inductor.config not available")
    except AttributeError as e:
        # Some inductor config options may not exist in older PyTorch versions
        logger.debug(f"Some Inductor options not available: {e}")
    except Exception as e:
        logger.debug(f"Failed to configure Inductor: {e}")


# ============================================================================
# IPEX Integration
# ============================================================================

_ipex_available: Optional[bool] = None
_ipex_module = None


def is_ipex_available() -> bool:
    """Check if Intel Extension for PyTorch is available."""
    global _ipex_available, _ipex_module

    if _ipex_available is not None:
        return _ipex_available

    try:
        import intel_extension_for_pytorch as ipex
        _ipex_module = ipex
        _ipex_available = True
        logger.info(f"IPEX available: version {ipex.__version__}")
    except (ImportError, AttributeError, Exception) as e:
        _ipex_available = False
        logger.info(f"IPEX not available: {e}. Using standard PyTorch")

    return _ipex_available


def get_ipex_module():
    """Get the IPEX module if available."""
    if is_ipex_available():
        return _ipex_module
    return None


def optimize_model_with_ipex(model: torch.nn.Module, dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    """
    Optimize a PyTorch model using IPEX.

    Args:
        model: The model to optimize
        dtype: Target dtype (bf16 recommended for Sapphire Rapids)

    Returns:
        Optimized model
    """
    if not is_ipex_available():
        logger.warning("IPEX not available, returning original model")
        return model

    ipex = get_ipex_module()

    try:
        # Optimize model with IPEX
        model = ipex.optimize(
            model,
            dtype=dtype,
            inplace=True,
            auto_kernel_selection=True
        )
        logger.info(f"Model optimized with IPEX (dtype={dtype})")
    except Exception as e:
        logger.warning(f"IPEX optimization failed: {e}")

    return model


# ============================================================================
# Combined Setup Function
# ============================================================================

def setup_intel_cpu_environment(
    rank: int = 0,
    world_size: int = 1,
    enable_numa: bool = True,
    enable_avx_optimization: bool = True,
    enable_ipex: bool = True
) -> dict:
    """
    Complete setup for Intel/AMD CPU optimization.

    Call this early in CPU worker initialization.
    Gracefully handles missing features (no AVX-512, no NUMA, no IPEX).

    Args:
        rank: Worker rank for NUMA node assignment
        world_size: Total number of workers
        enable_numa: Enable NUMA-aware allocation (if available)
        enable_avx_optimization: Enable AVX optimizations (AVX2 or AVX-512)
        enable_ipex: Enable IPEX if available

    Returns:
        Dict with configuration info
    """
    config = {
        "numa_enabled": False,
        "numa_node": -1,
        "avx512_enabled": False,
        "avx2_enabled": False,
        "ipex_enabled": False,
        "features": None
    }

    # Detect CPU features
    try:
        features = detect_intel_cpu_features()
        config["features"] = features
        config["avx512_enabled"] = features.avx512f
        config["avx2_enabled"] = features.avx2
    except Exception as e:
        logger.warning(f"Failed to detect CPU features: {e}")
        features = IntelCPUFeatures()
        config["features"] = features

    # Configure Intel optimizations (environment variables)
    if enable_avx_optimization:
        try:
            configure_intel_optimizations(features)
        except Exception as e:
            logger.warning(f"Failed to configure Intel optimizations: {e}")

    # Configure PyTorch settings
    try:
        configure_pytorch_for_intel(features)
    except Exception as e:
        logger.warning(f"Failed to configure PyTorch: {e}")

    # Setup NUMA binding (graceful fallback if not available)
    if enable_numa:
        try:
            allocator = NUMAAllocator()
            if allocator.is_available and allocator.num_nodes > 1:
                numa_node = allocator.get_preferred_node_for_rank(rank, world_size)
                allocator.bind_to_node(numa_node)
                config["numa_enabled"] = True
                config["numa_node"] = numa_node
                logger.info(f"Rank {rank} bound to NUMA node {numa_node}")
            elif allocator.is_available:
                # Single NUMA node system - still set node 0
                config["numa_node"] = 0
                logger.debug("Single NUMA node system, no binding needed")
            else:
                logger.debug("NUMA not available, using standard memory allocation")
        except Exception as e:
            logger.debug(f"NUMA setup skipped: {e}")

    # Check IPEX
    if enable_ipex:
        try:
            config["ipex_enabled"] = is_ipex_available()
        except Exception as e:
            logger.debug(f"IPEX check failed: {e}")
            config["ipex_enabled"] = False

    return config
