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

# torch is imported lazily to allow OMP environment variables
# to be set before OpenMP runtime initialization.
# Use _get_torch() instead of direct torch access at module level.
_torch = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch

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
    dtype,  # torch.dtype — lazy import
    numa_node: int,
    pin_memory: bool = False
):
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
        return _get_torch().zeros(size, dtype=_get_torch().int8, device='cpu')

    # Bind memory allocation to the target NUMA node
    allocator.bind_to_node(numa_node)

    # Allocate tensor (will use the bound NUMA node)
    torch = _get_torch()
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


def _check_kernel_amx_support() -> tuple[bool, str]:
    """
    Check if the Linux kernel supports AMX.

    AMX requires Linux kernel 5.16 or later.
    Returns (supported, reason).
    """
    try:
        result = subprocess.run(
            ["uname", "-r"],
            capture_output=True,
            text=True,
            timeout=5
        )
        kernel_version = result.stdout.strip()

        # Parse version (e.g., "5.19.0-42-generic" -> (5, 19))
        parts = kernel_version.split('.')
        if len(parts) >= 2:
            major = int(parts[0])
            minor = int(parts[1].split('-')[0])

            if major < 5 or (major == 5 and minor < 16):
                return False, f"Kernel {kernel_version} is too old (need 5.16+)"
            return True, f"Kernel {kernel_version} supports AMX"
    except Exception as e:
        return False, f"Could not check kernel version: {e}"

    return True, "Kernel version check passed"


def _detect_amx_from_lscpu() -> tuple[bool, bool, bool]:
    """
    Alternative AMX detection using lscpu.

    Some systems show AMX in lscpu even when cpuinfo doesn't list it.
    Returns (amx_tile, amx_bf16, amx_int8).
    """
    amx_tile = False
    amx_bf16 = False
    amx_int8 = False

    try:
        result = subprocess.run(
            ["lscpu"],
            capture_output=True,
            text=True,
            timeout=5
        )

        output = result.stdout.lower()

        # lscpu shows flags in "Flags:" line
        for line in result.stdout.split('\n'):
            if line.strip().lower().startswith('flags:'):
                flags_str = line.lower()
                if 'amx_tile' in flags_str or 'amx-tile' in flags_str:
                    amx_tile = True
                if 'amx_bf16' in flags_str or 'amx-bf16' in flags_str:
                    amx_bf16 = True
                if 'amx_int8' in flags_str or 'amx-int8' in flags_str:
                    amx_int8 = True
                break

    except Exception as e:
        logger.debug(f"lscpu AMX detection failed: {e}")

    return amx_tile, amx_bf16, amx_int8


def detect_intel_cpu_features() -> IntelCPUFeatures:
    """
    Detect Intel CPU features for optimization.

    Detects various CPU features including:
    - AVX2 (baseline for modern CPUs)
    - AVX-512 with VNNI (Vector Neural Network Instructions)
    - AVX-512 BF16 (Brain Float 16)
    - AMX (Advanced Matrix Extensions) - Sapphire Rapids+

    Works on any Linux system, gracefully handles missing features.

    AMX Detection Notes:
    - Requires Linux kernel 5.16 or later
    - Checks both /proc/cpuinfo and lscpu for AMX flags
    - Supports both underscore (amx_bf16) and dash (amx-bf16) formats
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

        # Normalize: also check for dash-separated versions
        cpuinfo_lower = cpuinfo.lower()

        # Check for AVX-512 features
        if "avx512f" in cpuinfo_lower:
            features.avx512f = True
            _AVX512_AVAILABLE = True
        if "avx512_vnni" in cpuinfo_lower or "avx512-vnni" in cpuinfo_lower:
            features.avx512_vnni = True
        if "avx512_bf16" in cpuinfo_lower or "avx512-bf16" in cpuinfo_lower:
            features.avx512_bf16 = True

        # AMX detection - check multiple formats
        # Also check for amx_tile which is the base AMX feature
        amx_tile_found = "amx_tile" in cpuinfo_lower or "amx-tile" in cpuinfo_lower
        amx_bf16_found = "amx_bf16" in cpuinfo_lower or "amx-bf16" in cpuinfo_lower
        amx_int8_found = "amx_int8" in cpuinfo_lower or "amx-int8" in cpuinfo_lower

        if amx_bf16_found:
            features.amx_bf16 = True
            _AMX_AVAILABLE = True
        if amx_int8_found:
            features.amx_int8 = True
            _AMX_AVAILABLE = True

        # Check for AVX2 (fallback for non-AVX512 systems)
        if "avx2" in cpuinfo_lower:
            features.avx2 = True

        # Check for AVX-VNNI (available on Alder Lake without AVX-512)
        if "avx_vnni" in cpuinfo_lower or "avx-vnni" in cpuinfo_lower:
            features.avx_vnni = True

        # Get model name
        for line in cpuinfo.split('\n'):
            if "model name" in line.lower():
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

        # If AMX not detected from cpuinfo, try lscpu as fallback
        if not features.amx_bf16 and not features.amx_int8:
            lscpu_tile, lscpu_bf16, lscpu_int8 = _detect_amx_from_lscpu()
            if lscpu_bf16:
                features.amx_bf16 = True
                _AMX_AVAILABLE = True
                logger.debug("AMX-BF16 detected via lscpu (not in cpuinfo)")
            if lscpu_int8:
                features.amx_int8 = True
                _AMX_AVAILABLE = True
                logger.debug("AMX-INT8 detected via lscpu (not in cpuinfo)")

        # Check kernel AMX support if CPU has AMX but we haven't detected it
        if not _AMX_AVAILABLE and ("8480" in features.model_name or
                                    "sapphire" in features.model_name.lower() or
                                    "emerald" in features.model_name.lower()):
            kernel_ok, kernel_msg = _check_kernel_amx_support()
            if not kernel_ok:
                logger.warning(f"CPU likely supports AMX but: {kernel_msg}")
            else:
                logger.warning(
                    f"CPU ({features.model_name}) should support AMX but flags not found in cpuinfo. "
                    f"This may be due to kernel configuration or XSTATE permissions. "
                    f"Try: 'grep amx /proc/cpuinfo' and 'uname -r' to diagnose."
                )

    except Exception as e:
        logger.debug(f"Failed to detect CPU features: {e}")

    # Log detected features
    logger.info(f"CPU detected: {features.model_name}")

    # AVX status
    if features.avx512f:
        logger.info("  AVX-512: Yes")
        if features.avx512_vnni:
            logger.info("  AVX-512 VNNI: Yes (INT8 acceleration)")
        if features.avx512_bf16:
            logger.info("  AVX-512 BF16: Yes")
    elif features.avx2:
        logger.info("  AVX-512: No (AVX2 available)")

    # AMX status (Sapphire Rapids+)
    if features.amx_bf16 or features.amx_int8:
        if features.amx_bf16:
            logger.info("  AMX-BF16: Yes (Sapphire Rapids+)")
        if features.amx_int8:
            logger.info("  AMX-INT8: Yes (Sapphire Rapids+)")
    else:
        logger.info("  AMX: No (requires Sapphire Rapids or newer)")

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
    # AMX Settings (Sapphire Rapids+)
    # =========================================

    if features.amx_bf16 or features.amx_int8:
        # Enable AMX in oneDNN/DNNL
        os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX512_CORE_AMX")
        os.environ.setdefault("DNNL_MAX_CPU_ISA", "AVX512_CORE_AMX")
        settings["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_AMX"
        settings["AMX_ENABLED"] = True

        # Enable AMX tiles (Linux kernel 5.16+)
        # AMX requires explicit permission via ARCH_REQ_XCOMP_PERM
        try:
            _enable_amx_tiles()
        except Exception as e:
            logger.debug(f"AMX tile permission setup skipped: {e}")

        logger.info(f"  AMX enabled: BF16={features.amx_bf16}, INT8={features.amx_int8}")
    elif features.avx512f:
        # AVX-512 without AMX
        os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX512_CORE_VNNI")
        os.environ.setdefault("DNNL_MAX_CPU_ISA", "AVX512_CORE_VNNI")
        settings["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_VNNI"
        settings["AMX_ENABLED"] = False

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
    if features.amx_bf16 or features.amx_int8:
        logger.info(f"  Features: AMX-BF16={features.amx_bf16}, AMX-INT8={features.amx_int8}, "
                    f"AVX-512={features.avx512f}")
    elif features.avx512f:
        logger.info(f"  Features: AVX-512, VNNI={features.avx512_vnni}, "
                    f"BF16={features.avx512_bf16}, AMX=No")
    else:
        logger.info(f"  Features: AVX2={features.avx2}, AVX-VNNI={features.avx_vnni}")
    logger.info(f"  Topology: {features.num_sockets} socket(s) x "
                f"{features.cores_per_socket} cores x "
                f"{features.threads_per_core} threads")

    return settings


def _enable_amx_tiles():
    """
    Enable AMX tiles via ARCH_REQ_XCOMP_PERM syscall.

    AMX requires explicit kernel permission on Linux 5.16+.
    This is typically done automatically by IPEX/oneDNN, but we ensure it here.
    """
    import platform
    if platform.system() != "Linux":
        return

    try:
        import ctypes

        # ARCH_REQ_XCOMP_PERM = 0x1023
        # XFEATURE_XTILEDATA = 18
        ARCH_REQ_XCOMP_PERM = 0x1023
        XFEATURE_XTILEDATA = 18

        libc = ctypes.CDLL("libc.so.6", use_errno=True)

        # syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)
        SYS_arch_prctl = 158  # x86_64
        result = libc.syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)

        if result == 0:
            logger.debug("AMX tiles enabled via ARCH_REQ_XCOMP_PERM")
        else:
            errno = ctypes.get_errno()
            logger.debug(f"AMX tile permission request returned {result}, errno={errno}")

    except Exception as e:
        logger.debug(f"AMX tile setup failed: {e}")


def configure_pytorch_for_intel(features: Optional[IntelCPUFeatures] = None):
    """
    Configure PyTorch settings optimized for Intel CPUs.

    Works on any CPU, with enhanced settings for AVX-512 and AMX systems.

    NOTE: OMP env vars must be set before this call (torch import initializes OpenMP).

    This includes:
    - Thread settings
    - Memory format preferences
    - Inductor backend settings (adaptive to CPU features)
    - AMX acceleration (Sapphire Rapids+)
    """
    # torch import here is intentional — OMP env vars must already be set
    torch = _get_torch()

    if features is None:
        features = detect_intel_cpu_features()

    # Set thread count to physical cores, but respect OMP_NUM_THREADS
    # if already set (e.g., by hybrid mode's _setup_cpu_process_env())
    total_physical_cores = features.num_sockets * features.cores_per_socket
    if total_physical_cores > 0:
        omp_env = os.environ.get("OMP_NUM_THREADS")
        if omp_env:
            try:
                target_threads = int(omp_env)
                torch.set_num_threads(target_threads)
                logger.info(f"PyTorch threads set to {target_threads} "
                           f"(from OMP_NUM_THREADS, physical={total_physical_cores})")
            except ValueError:
                torch.set_num_threads(total_physical_cores)
                logger.info(f"PyTorch threads set to {total_physical_cores} (physical cores)")
        else:
            torch.set_num_threads(total_physical_cores)
            logger.info(f"PyTorch threads set to {total_physical_cores} (physical cores)")

    # Set interop threads for parallel regions
    interop_threads = max(1, features.num_sockets)
    torch.set_num_interop_threads(interop_threads)
    logger.debug(f"PyTorch interop threads set to {interop_threads}")

    # Enable high precision matmul
    torch.set_float32_matmul_precision('high')
    logger.debug("Float32 matmul precision set to 'high'")

    # =========================================
    # AMX Configuration (Sapphire Rapids+)
    # =========================================
    if features.amx_bf16 or features.amx_int8:
        _configure_amx_for_pytorch(features)

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
        if features.amx_bf16:
            # AMX uses tiles, but SIMD still uses AVX-512
            inductor_config.cpp.simdlen = 16
            logger.info("Inductor configured for AMX + AVX-512 (simdlen=16)")
        elif features.avx512f:
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


def _configure_amx_for_pytorch(features: IntelCPUFeatures):
    """
    Configure PyTorch for AMX acceleration.

    AMX (Advanced Matrix Extensions) provides significant speedup for
    matrix operations on Sapphire Rapids and newer processors.
    """
    logger.info("Configuring PyTorch for AMX acceleration...")

    # 1. Enable oneDNN with AMX support
    try:
        # PyTorch 2.0+ uses oneDNN by default for CPU
        # Ensure it's enabled and configured for AMX
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
            logger.debug("  oneDNN/MKLDNN enabled")
    except Exception as e:
        logger.debug(f"  MKLDNN setup skipped: {e}")

    # 2. Try to enable BF16 fast math (uses AMX-BF16)
    try:
        # PyTorch 2.1+ has allow_bf16_reduced_precision_reduction
        if hasattr(torch.backends, 'cpu') and hasattr(torch.backends.cpu, 'allow_bf16_reduced_precision_reduction'):
            torch.backends.cpu.allow_bf16_reduced_precision_reduction = True
            logger.debug("  BF16 reduced precision enabled")
    except Exception as e:
        logger.debug(f"  BF16 precision setup skipped: {e}")

    # 3. Configure IPEX for AMX if available
    try:
        import intel_extension_for_pytorch as ipex

        # Set FP32 math mode to BF16 (AMX-BF16 will be used)
        if hasattr(ipex, 'set_fp32_math_mode') and hasattr(ipex, 'FP32MathMode'):
            ipex.set_fp32_math_mode(ipex.FP32MathMode.BF16)
            logger.info("  IPEX FP32 math mode set to BF16 (AMX accelerated)")

        # Enable oneDNN graph fusion for better AMX utilization
        if hasattr(ipex, 'enable_onednn_fusion'):
            ipex.enable_onednn_fusion(True)
            logger.debug("  IPEX oneDNN fusion enabled")

    except ImportError:
        logger.debug("  IPEX not available for AMX configuration")
    except Exception as e:
        logger.debug(f"  IPEX AMX setup skipped: {e}")

    logger.info(f"  AMX configuration complete: BF16={features.amx_bf16}, INT8={features.amx_int8}")


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

        # Check if IPEX can use AMX
        try:
            if hasattr(ipex, 'get_fp32_math_mode'):
                # IPEX 2.0+ has AMX support detection
                logger.info(f"  IPEX AMX optimization: Available")
        except Exception:
            pass
    except ImportError:
        _ipex_available = False
        logger.debug("IPEX not installed. Using standard PyTorch")
    except AttributeError as e:
        # Known IPEX bug: "module 'os' has no attribute 'exit'"
        # This happens due to IPEX internal issues with certain PyTorch versions
        _ipex_available = False
        if "os" in str(e) and "exit" in str(e):
            logger.debug("IPEX has compatibility issue with current PyTorch. Using standard PyTorch")
        else:
            logger.debug(f"IPEX not available (AttributeError): {e}")
    except Exception as e:
        _ipex_available = False
        error_msg = str(e)
        # Simplify common error messages
        if "PyTorch" in error_msg and "version" in error_msg.lower():
            logger.debug(f"IPEX/PyTorch version mismatch. Using standard PyTorch")
        else:
            logger.debug(f"IPEX not available: {e}")

    return _ipex_available


def get_ipex_module():
    """Get the IPEX module if available."""
    if is_ipex_available():
        return _ipex_module
    return None


def _ensure_transformers_beam_search_shim():
    """IPEX 2.8 이 기대하는 transformers 구 API 를 transformers 5.x 에 shim 주입.

    NOTE (§04 기각, 2026-04-19): 본 shim 은 ``ipex.llm.optimize`` import
    단계를 통과시키기 위한 것이었으나, 이어지는 module walking 단계에서
    vLLM 모델 구조 (`num_heads` / `num_kv_heads`, QKVParallelLinear) 가 IPEX
    가 기대하는 HF transformers module 구조와 다르다는 **구조적 비호환**이
    확인되어 §04 자체가 기각되었다. 상세: NinjaGap_Todo/04_ipex_woq_int8.md.
    코드는 히스토리/후속 참고 목적으로 보존하며, ``HYBRID_WOQ_INT8=1`` 은
    dormant flag 로 남아 있다. 재활성화는 §06 VNNI hot path wiring 시 별도
    설계.

    Shim 대상:
    1) `transformers.generation.beam_search.BeamScorer` — 모듈 자체가 사라짐 → 신규 모듈 생성
    2) `transformers.generation.utils` 내 구이름 Output 클래스 → 신이름으로 alias
       - BeamSearch* → GenerateBeam*
       - GreedySearch* / Sample* → Generate{Encoder,Decoder}Output

    vLLM 은 IPEX generation path 를 쓰지 않으므로 실제 호출은 없다.
    import 만 성공하면 `ipex.llm.optimize` 가 WoQ qconfig 로 가중치를 변환한다.
    """
    import sys

    # (1) beam_search 모듈 자체 shim
    key = 'transformers.generation.beam_search'
    if key not in sys.modules:
        try:
            import types
            mod = types.ModuleType(key)

            class BeamScorer:  # minimal abstract stub
                def __init__(self, *a, **kw):
                    pass

                def process(self, *a, **kw):
                    raise RuntimeError(
                        "BeamScorer shim: beam search path not supported")

                def finalize(self, *a, **kw):
                    raise RuntimeError(
                        "BeamScorer shim: beam search path not supported")

            mod.BeamScorer = BeamScorer
            sys.modules[key] = mod
            logger.info(
                "[HYBRID-WOQ] shim injected: %s.BeamScorer", key)
        except Exception as e:
            logger.warning("[HYBRID-WOQ] shim module injection failed: %s", e)

    # (2) generation.utils alias — 구이름 → 신이름
    try:
        from transformers.generation import utils as _u
        alias_map = {
            # Beam (이름 변경)
            'BeamSearchEncoderDecoderOutput': 'GenerateBeamEncoderDecoderOutput',
            'BeamSearchDecoderOnlyOutput': 'GenerateBeamDecoderOnlyOutput',
            # Greedy / Sample — transformers 5.x 에서 Generate{E,D}Output 으로 통합
            'GreedySearchEncoderDecoderOutput': 'GenerateEncoderDecoderOutput',
            'GreedySearchDecoderOnlyOutput': 'GenerateDecoderOnlyOutput',
            'SampleEncoderDecoderOutput': 'GenerateEncoderDecoderOutput',
            'SampleDecoderOnlyOutput': 'GenerateDecoderOnlyOutput',
        }
        injected = []
        for old, new in alias_map.items():
            if hasattr(_u, old):
                continue
            target = getattr(_u, new, None)
            if target is None:
                # 최후 수단: dummy type
                target = type(old, (), {})
            setattr(_u, old, target)
            injected.append(old)
        if injected:
            logger.info(
                "[HYBRID-WOQ] alias injected in generation.utils: %s",
                injected)
    except Exception as e:
        logger.warning("[HYBRID-WOQ] alias injection failed: %s", e)

    # (3) modeling_mllama — IPEX 2.8 이 SDPA variant 별도 class 기대하나
    # transformers 5.x 에서 base Attention 클래스로 통합됨.
    # vLLM 에서 Llama 계열 모델은 mllama 가 아니므로 alias 만 걸어 import 성공시킨다.
    try:
        from transformers.models.mllama import modeling_mllama as _mm
        mllama_alias = {
            'MllamaTextCrossSdpaAttention': 'MllamaTextCrossAttention',
        }
        mm_injected = []
        for old, new in mllama_alias.items():
            if hasattr(_mm, old):
                continue
            target = getattr(_mm, new, None) or type(old, (), {})
            setattr(_mm, old, target)
            mm_injected.append(old)
        if mm_injected:
            logger.info(
                "[HYBRID-WOQ] alias injected in modeling_mllama: %s",
                mm_injected)
    except Exception as e:
        logger.warning(
            "[HYBRID-WOQ] mllama alias injection failed: %s", e)


def _build_woq_qconfig(ipex, weight_dtype: str = "int8"):
    """NinjaGap §04: Weight-Only Quantization qconfig (INT8/INT4).

    HYBRID_WOQ_INT8=1 시 호출. 실패 시 None 반환 → WoQ skip, 기본 BF16 optimize.
    """
    try:
        from intel_extension_for_pytorch.quantization import (  # type: ignore
            WoqLowpMode, WoqWeightDtype)
        wdtype = (WoqWeightDtype.INT8 if weight_dtype == "int8"
                  else WoqWeightDtype.INT4)
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=wdtype,
            lowp_mode=WoqLowpMode.BF16,
            act_quant_mode=None,
            group_size=-1,
        )
        logger.info(
            "[HYBRID-WOQ] qconfig built: weight=%s lowp=BF16 group=-1",
            weight_dtype.upper())
        return qconfig
    except Exception as e:
        logger.warning("[HYBRID-WOQ] qconfig build failed: %s", e)
        return None


def optimize_model_with_ipex(model, dtype=None):
    """
    Optimize a PyTorch model using IPEX.

    Args:
        model: The model to optimize
        dtype: Target dtype (bf16 recommended for Sapphire Rapids)

    Returns:
        Optimized model

    Env vars:
        HYBRID_WOQ_INT8=1  — NinjaGap §04: WoQ INT8 적용
                             (ipex.llm.optimize path 사용, qconfig INT8)
        HYBRID_WOQ_DTYPE   — 'int8' (기본) 또는 'int4' (실험)
    """
    torch = _get_torch()
    if dtype is None:
        dtype = torch.bfloat16
    if not is_ipex_available():
        logger.warning("IPEX not available, returning original model")
        return model

    ipex = get_ipex_module()

    # ── NinjaGap §04: WoQ path ─────────────────────────────────────────────
    import os as _os
    woq_enabled = _os.environ.get("HYBRID_WOQ_INT8", "0") == "1"
    if woq_enabled and hasattr(ipex, "llm") and hasattr(ipex.llm, "optimize"):
        # transformers 5.x 와 IPEX 2.8 호환성 shim
        _ensure_transformers_beam_search_shim()
        weight_dtype = _os.environ.get("HYBRID_WOQ_DTYPE", "int8").lower()
        qconfig = _build_woq_qconfig(ipex, weight_dtype=weight_dtype)
        if qconfig is not None:
            try:
                model = ipex.llm.optimize(
                    model, dtype=dtype, inplace=True,
                    quantization_config=qconfig,
                )
                logger.info(
                    "[HYBRID-WOQ] ipex.llm.optimize applied "
                    "(dtype=%s, weight=%s)", dtype, weight_dtype.upper())
                return model
            except Exception as e:
                logger.warning(
                    "[HYBRID-WOQ] ipex.llm.optimize with WoQ failed: %s. "
                    "Falling back to plain BF16 optimize.", e)
        else:
            logger.warning(
                "[HYBRID-WOQ] qconfig unavailable, falling back to BF16")

    # ── 기본: BF16 optimize (기존 동작) ───────────────────────────────────
    try:
        model = ipex.optimize(
            model,
            dtype=dtype,
            inplace=True,
            auto_kernel_selection=True
        )
        logger.info(f"Model optimized with IPEX (dtype={dtype})")
    except Exception as e:
        logger.warning(f"IPEX optimization failed: {e}, "
                       f"retrying with weights_prepack=False")
        try:
            model = ipex.optimize(
                model,
                dtype=dtype,
                inplace=True,
                weights_prepack=False,
            )
            logger.info(f"Model optimized with IPEX (dtype={dtype}, "
                        f"weights_prepack=False)")
        except Exception as e2:
            logger.warning(f"IPEX optimization fallback also failed: {e2}")

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
        "amx_enabled": False,
        "ipex_enabled": False,
        "features": None
    }

    # Detect CPU features
    try:
        features = detect_intel_cpu_features()
        config["features"] = features
        config["avx512_enabled"] = features.avx512f
        config["avx2_enabled"] = features.avx2
        config["amx_enabled"] = features.amx_bf16 or features.amx_int8
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
