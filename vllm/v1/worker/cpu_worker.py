# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPU Worker for vLLM V1

Optimized for Intel Xeon processors with:
- NUMA-aware memory allocation
- AVX-512 / AMX acceleration
- Intel Extension for PyTorch (IPEX) integration
"""
import os
import platform
from typing import Callable, Optional

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.utils import set_random_seed
from vllm.platforms import CpuArchEnum, current_platform
from vllm.platforms.cpu import CpuPlatform, LogicalCPUInfo
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_worker import (Worker,
                                       init_worker_distributed_environment)

# Intel CPU optimization utilities (optional, graceful fallback if not available)
try:
    from vllm.platforms.intel_cpu_utils import (
        setup_intel_cpu_environment,
        detect_intel_cpu_features,
        NUMAAllocator,
        is_ipex_available,
        IntelCPUFeatures,
    )
    _INTEL_UTILS_AVAILABLE = True
except ImportError as e:
    _INTEL_UTILS_AVAILABLE = False
    IntelCPUFeatures = None  # type: ignore
    NUMAAllocator = None  # type: ignore

logger = init_logger(__name__)


class CPUWorker(Worker):
    """
    CPU Worker optimized for Intel Xeon processors.

    Features:
    - NUMA-aware memory allocation for KVCache
    - AVX-512 / AMX acceleration via PyTorch Inductor
    - Intel Extension for PyTorch (IPEX) integration
    - Optimized thread affinity for multi-socket systems
    """

    def __init__(self,
                 vllm_config: VllmConfig,
                 local_rank: int,
                 rank: int,
                 distributed_init_method: str,
                 is_driver_worker: bool = False):

        # =====================================================
        # Intel CPU Optimization Setup (NUMA, AVX-512, IPEX)
        # =====================================================
        self._intel_config: dict = {}
        self._numa_node: int = -1
        self._cpu_features: Optional[IntelCPUFeatures] = None

        # Detect hybrid mode: _setup_cpu_process_env() already configured
        # Intel optimizations, NUMA binding, IPEX, AMX, etc.
        # CUDA_VISIBLE_DEVICES="" is set by run_cpu_engine_core()
        self._is_hybrid_cpu_process = (
            os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and os.environ.get("VLLM_CPU_KVCACHE_SPACE") is not None
        )

        if _INTEL_UTILS_AVAILABLE and not self._is_hybrid_cpu_process:
            # Non-hybrid (CPU-only) mode: full Intel environment setup
            try:
                self._intel_config = setup_intel_cpu_environment(
                    rank=local_rank,
                    world_size=vllm_config.parallel_config.world_size,
                    enable_numa=True,
                    enable_avx_optimization=True,
                    enable_ipex=True
                )
                self._numa_node = self._intel_config.get("numa_node", -1)
                self._cpu_features = self._intel_config.get("features")

                logger.info(f"CPUWorker[{rank}] Intel optimization initialized:")
                logger.info(f"  NUMA node: {self._numa_node}")
                logger.info(f"  AVX-512: {self._intel_config.get('avx512_enabled', False)}")
                logger.info(f"  IPEX: {self._intel_config.get('ipex_enabled', False)}")

                if self._cpu_features:
                    logger.info(f"  CPU: {self._cpu_features.model_name}")
                    logger.info(f"  Topology: {self._cpu_features.num_sockets} sockets x "
                               f"{self._cpu_features.cores_per_socket} cores")
                    if self._cpu_features.amx_bf16:
                        logger.info("  AMX-BF16: Enabled (Sapphire Rapids)")
            except Exception as e:
                logger.warning(f"Intel CPU optimization setup failed: {e}")
        elif _INTEL_UTILS_AVAILABLE and self._is_hybrid_cpu_process:
            # Hybrid mode: env already configured by _setup_cpu_process_env()
            # Just detect features for logging/downstream use
            try:
                self._cpu_features = detect_intel_cpu_features()
                logger.info(
                    "CPUWorker[%d] running in hybrid mode "
                    "(env pre-configured by _setup_cpu_process_env). "
                    "OMP_THREADS=%s, KVCACHE=%sGB",
                    rank,
                    os.environ.get("OMP_NUM_THREADS", "?"),
                    os.environ.get("VLLM_CPU_KVCACHE_SPACE", "?"),
                )
                if self._cpu_features:
                    logger.info(
                        "  CPU: %s, %d sockets x %d cores, "
                        "AVX512=%s, AMX=%s, VNNI=%s",
                        self._cpu_features.model_name,
                        self._cpu_features.num_sockets,
                        self._cpu_features.cores_per_socket,
                        self._cpu_features.avx512f,
                        self._cpu_features.amx_bf16,
                        self._cpu_features.avx512_vnni,
                    )
            except Exception as e:
                logger.warning(f"CPU feature detection failed: {e}")

            # IPEX availability warning
            try:
                if not is_ipex_available():
                    logger.warning(
                        "IPEX not available! CPU decode performance will be "
                        "significantly degraded (Python loop fallback for "
                        "PagedAttention). Install intel-extension-for-pytorch "
                        "for optimal hybrid inference performance."
                    )
            except Exception:
                pass

        # =====================================================
        # Force CPU Platform
        # =====================================================
        try:
            from vllm.platforms.cpu import CpuPlatform
            import vllm.platforms
            vllm.platforms._current_platform = CpuPlatform()
            from vllm.attention.selector import _cached_get_attn_backend
            _cached_get_attn_backend.cache_clear()
            logger.info("Forced vllm.platforms._current_platform to CpuPlatform for CPUWorker.")
        except Exception as e:
            logger.error(f"Failed to force CpuPlatform: {e}")

        # =====================================================
        # Update config for CPU platform
        # =====================================================
        # WorkerBase.__init__ may have set device_type="heterogeneous"
        # because CUDA_VISIBLE_DEVICES is empty in the CPU EngineCore
        # process (so num_gpus=0, world_size>num_gpus heuristic triggers).
        # CpuPlatform.check_and_update_config asserts device_type=="cpu",
        # so we restore it before the call.  This is safe inside the CPU
        # engine process — it really IS a pure-CPU worker, the
        # heterogeneous flag was a side effect of process isolation.
        try:
            prev_dev_type = vllm_config.device_config.device_type
            if prev_dev_type != "cpu":
                logger.info(
                    "[HYBRID-CPU-WORKER] Forcing device_config.device_type "
                    "%r → 'cpu' (heterogeneous flag was set by WorkerBase "
                    "heuristic because CUDA is hidden in this process).",
                    prev_dev_type)
                vllm_config.device_config.device_type = "cpu"
        except Exception as _e:
            logger.warning(
                "[HYBRID-CPU-WORKER] could not coerce device_type: %s", _e)
        try:
            from vllm.platforms.cpu import CpuPlatform
            CpuPlatform.check_and_update_config(vllm_config)
            logger.info(
                "Updated vllm_config for CPU platform via "
                "check_and_update_config (device_type=%r).",
                vllm_config.device_config.device_type)
        except Exception as e:
            # logger.exception so traceback + exception type are visible
            # — previously str(e) was empty for assertion errors, hiding
            # the root cause.
            logger.exception(
                "Failed to update config for CPU "
                "(device_type=%r, type=%s, msg=%r)",
                vllm_config.device_config.device_type,
                type(e).__name__, str(e))

        # =====================================================
        # PyTorch Compilation Configuration for CPU
        # =====================================================
        from vllm.config import CompilationLevel
        # torch 2.9+: CPU Dynamo/Inductor has compatibility issues.
        # Use eager mode for stability; performance impact is minimal
        # for CPU-bound inference.
        vllm_config.model_config.enforce_eager = True
        vllm_config.compilation_config.level = CompilationLevel.NO_COMPILATION
        logger.info("CPU Worker: Using eager mode (torch.compile disabled)")

        # Reduce logging noise from PyTorch Dynamo/Inductor
        import logging
        logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
        logging.getLogger("torch._inductor").setLevel(logging.ERROR)

        super().__init__(vllm_config,
                         local_rank,
                         rank,
                         distributed_init_method,
                         is_driver_worker=is_driver_worker)

        self.parallel_config.disable_custom_all_reduce = True

        # =====================================================
        # Thread Configuration (NUMA-aware)
        # =====================================================
        self._configure_threads_for_numa()

    def _configure_inductor_for_intel(self, vllm_config: VllmConfig):
        """Configure PyTorch Inductor settings for Intel CPUs."""
        try:
            # Update inductor compile config for Intel optimization
            inductor_config = vllm_config.compilation_config.inductor_compile_config

            # Enable optimizations beneficial for Intel CPUs
            inductor_config.update({
                "dce": True,  # Dead code elimination
                "size_asserts": False,
                "nan_asserts": False,
                "epilogue_fusion": True,  # Fuse epilogue operations
                "max_autotune": True,  # Find best kernel configurations
                "freezing": True,  # Enable parameter freezing for better optimization
            })

            # Set high precision for matmul
            torch.set_float32_matmul_precision('high')

            logger.info("Inductor configured for Intel CPU optimization")

        except Exception as e:
            logger.warning(f"Failed to configure Inductor for Intel: {e}")

    def _configure_threads_for_numa(self):
        """Configure thread count based on NUMA topology or CPU count.

        In hybrid mode, respects the OMP_NUM_THREADS set by
        _setup_cpu_process_env() to avoid overriding the optimized value.

        In CPU-only mode, uses all physical cores on the bound NUMA node
        (or all cores if no NUMA binding).
        """
        import multiprocessing

        # In hybrid mode, respect the pre-configured OMP_NUM_THREADS
        # from _setup_cpu_process_env() — don't override it
        if self._is_hybrid_cpu_process:
            omp_threads_str = os.environ.get("OMP_NUM_THREADS")
            if omp_threads_str:
                try:
                    target = int(omp_threads_str)
                    torch.set_num_threads(target)
                    # interop을 약간 늘려서 op 간 직렬화 방지
                    try:
                        torch.set_num_interop_threads(max(2, target // 8))
                    except RuntimeError:
                        pass  # 이미 사용된 후엔 변경 불가
                    logger.info(
                        "[HYBRID-CPU-WORKER] thread config: "
                        "torch_threads=%d torch_interop=%d "
                        "(from OMP_NUM_THREADS=%s)",
                        torch.get_num_threads(),
                        torch.get_num_interop_threads(),
                        omp_threads_str)
                    return
                except ValueError:
                    pass

        current_threads = torch.get_num_threads()

        # Try to use NUMA-aware thread count if features are detected
        if self._cpu_features and self._cpu_features.cores_per_socket > 0:
            # Calculate cores per NUMA node
            num_numa_nodes = 1
            if _INTEL_UTILS_AVAILABLE and NUMAAllocator is not None:
                try:
                    allocator = NUMAAllocator()
                    if allocator.is_available:
                        num_numa_nodes = max(1, allocator.num_nodes)
                except Exception:
                    pass

            total_cores = (self._cpu_features.cores_per_socket
                           * self._cpu_features.num_sockets)

            # If bound to a NUMA node, use ALL cores on that node
            # If not bound, use ALL physical cores
            if self._numa_node >= 0 and num_numa_nodes > 1:
                # Bound to specific NUMA node — use all its cores
                target_threads = max(4, total_cores // num_numa_nodes)
            else:
                # Not bound — use all physical cores
                target_threads = max(4, total_cores)

            torch.set_num_threads(target_threads)
            if self._numa_node >= 0:
                logger.info(
                    "CPUWorker: Thread count set to %d "
                    "(NUMA node %d, %d total cores, %d nodes)",
                    target_threads, self._numa_node,
                    total_cores, num_numa_nodes)
            else:
                logger.info(
                    "CPUWorker: Thread count set to %d "
                    "(%d total cores)", target_threads, total_cores)
        elif current_threads < 4:
            # Fallback: use all CPU cores
            try:
                target_threads = max(8, multiprocessing.cpu_count())
            except Exception:
                target_threads = 8

            torch.set_num_threads(target_threads)
            logger.info(
                "CPUWorker: Overrode low thread count (%d) "
                "to %d for performance.", current_threads, target_threads)

    def _python_init_cpu_threads_env(self, cpu_ids_str: str) -> None:
        """Python fallback for torch.ops._C_utils.init_cpu_threads_env.

        Sets the process CPU affinity (so all OMP/BLAS threads are
        constrained to these cores) and re-asserts torch thread count.

        Unlike the C++ version, this does NOT pin individual OMP threads
        1-to-1 to single cores via sched_setaffinity inside the OMP
        parallel region. Instead it relies on the Linux scheduler to
        distribute the OMP_NUM_THREADS workers across the affinity mask,
        which is sufficient for typical hybrid deployments without
        AVX-512/AMX builds. The C++ path remains preferred when
        available (built into _C extension on CPU-only builds).

        cpu_ids_str: comma-separated CPU id list, e.g. "0,1,2,3" or
            "0-3,8-11" (numa-style range syntax).
        """
        cpu_ids: list[int] = []
        for part in cpu_ids_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                cpu_ids.extend(range(int(a), int(b) + 1))
            else:
                cpu_ids.append(int(part))
        if not cpu_ids:
            raise ValueError(f"empty cpu_ids parsed from {cpu_ids_str!r}")

        # 1. Process-level affinity (inherited by every thread spawned
        #    by this process from now on, including OMP/BLAS workers).
        os.sched_setaffinity(0, set(cpu_ids))

        # 2. Re-assert torch and OMP thread counts to match #cores.
        nthreads = len(cpu_ids)
        os.environ["OMP_NUM_THREADS"] = str(nthreads)
        os.environ["MKL_NUM_THREADS"] = str(nthreads)
        torch.set_num_threads(nthreads)
        try:
            torch.set_num_interop_threads(max(2, nthreads // 8))
        except RuntimeError:
            # set_num_interop_threads can only be called before any
            # parallel work has started; ignore if too late.
            pass

        # 3. Memory-bind to the NUMA node containing the first core
        #    (matches what the C++ init_cpu_threads_env does).
        try:
            from vllm.platforms.intel_cpu_utils import NUMAAllocator
            alloc = NUMAAllocator()
            if alloc.is_available:
                # Find which NUMA node hosts cpu_ids[0]
                for nid in range(alloc.num_nodes):
                    info = alloc.get_node_info(nid)
                    if info and cpu_ids[0] in info.cpu_ids:
                        alloc.bind_to_node(nid)
                        logger.info(
                            "[HYBRID-CPU-WORKER] Python fallback "
                            "memory-bound to NUMA node %d", nid)
                        break
        except Exception as e:
            logger.debug("NUMA membind in fallback skipped: %s", e)

        logger.info(
            "[HYBRID-CPU-WORKER] Python fallback bound %d cores: %s "
            "(OMP_NUM_THREADS=%d, torch.get_num_threads()=%d)",
            nthreads,
            cpu_ids if nthreads <= 16 else f"{cpu_ids[:8]}...{cpu_ids[-8:]}",
            nthreads, torch.get_num_threads(),
        )

    @property
    def numa_node(self) -> int:
        """Get the NUMA node this worker is bound to."""
        return self._numa_node

    @property
    def intel_config(self) -> dict:
        """Get Intel CPU optimization configuration."""
        return self._intel_config


    def init_device(self):
        # Setup OpenMP threads affinity.
        omp_cpuids = envs.VLLM_CPU_OMP_THREADS_BIND
        if omp_cpuids == "auto" and platform.system() == "Linux":
            if current_platform.get_cpu_architecture() == CpuArchEnum.POWERPC:
                # For POWERPC SMT-8/4/2
                self.local_omp_cpuid = self._get_autobind_cpu_ids(
                    lambda cpus: [cpu for cpu in cpus if cpu.id % 8 < 4])
            elif current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                # For x86 SMT-2, use 1 CPU per core
                self.local_omp_cpuid = self._get_autobind_cpu_ids(
                    lambda cpus: cpus[-1:])
            else:
                self.local_omp_cpuid = "all"
        else:
            self.local_omp_cpuid = omp_cpuids.split("|")[self.rank]

        logger.info(
            "[HYBRID-CPU-WORKER] init_device: VLLM_CPU_OMP_THREADS_BIND=%r "
            "→ local_omp_cpuid=%r (rank=%d, local_rank=%d)",
            omp_cpuids, self.local_omp_cpuid, self.rank, self.local_rank,
        )
        if self.local_omp_cpuid != "all":
            bound_via = None
            try:
                ret = torch.ops._C_utils.init_cpu_threads_env(self.local_omp_cpuid)
                bound_via = "C++ (init_cpu_threads_env)"
                # init_cpu_threads_env가 코어 매핑을 출력 → 항상 INFO로 기록
                logger.info(
                    "[HYBRID-CPU-WORKER] init_cpu_threads_env (C++) returned:\n%s",
                    ret if ret else "(empty)")
            except AttributeError:
                # CUDA 빌드에서는 _C_utils.init_cpu_threads_env가 등록되지
                # 않는다 (cmake/cpu_extension.cmake가 include 안 됨).
                # Python fallback으로 process-level affinity를 직접 설정.
                logger.warning(
                    "[HYBRID-CPU-WORKER] torch.ops._C_utils."
                    "init_cpu_threads_env not registered (CUDA build). "
                    "Falling back to Python sched_setaffinity.")
                try:
                    self._python_init_cpu_threads_env(self.local_omp_cpuid)
                    bound_via = "Python (sched_setaffinity)"
                except Exception as fb_e:
                    logger.error(
                        "[HYBRID-CPU-WORKER] Python affinity fallback "
                        "FAILED: %s. Thread affinity NOT set — expect "
                        "poor CPU utilization.", fb_e)
            except RuntimeError as e:
                # VLLM_NUMA_DISABLED 빌드는 warning string만 반환할 수 있다.
                logger.warning(
                    "[HYBRID-CPU-WORKER] C++ init_cpu_threads_env failed: "
                    "%s. Falling back to Python sched_setaffinity.", e)
                try:
                    self._python_init_cpu_threads_env(self.local_omp_cpuid)
                    bound_via = "Python (sched_setaffinity, fallback)"
                except Exception as fb_e:
                    logger.error(
                        "[HYBRID-CPU-WORKER] Python affinity fallback "
                        "FAILED: %s.", fb_e)
            if bound_via:
                logger.info(
                    "[HYBRID-CPU-WORKER] thread binding established via: %s",
                    bound_via)
        else:
            logger.warning(
                "[HYBRID-CPU-WORKER] local_omp_cpuid='all' → no explicit "
                "thread binding. OMP runtime will choose; check "
                "VLLM_CPU_OMP_THREADS_BIND.")

        # OS 레벨 thread/affinity 진단 (init_cpu_threads_env 직후)
        try:
            import psutil as _ps
            _proc = _ps.Process()
            _aff = sorted(_proc.cpu_affinity())
            _nthr = _proc.num_threads()
            logger.info(
                "[HYBRID-CPU-WORKER] post-init: torch_threads=%d "
                "process_threads=%d cpu_affinity=%d cores %s",
                torch.get_num_threads(), _nthr, len(_aff),
                _aff if len(_aff) <= 32 else f"{_aff[:16]}...{_aff[-16:]}",
            )
        except Exception as _e:
            logger.warning(
                "[HYBRID-CPU-WORKER] failed to read process affinity: %s",
                _e)


        # Note: unique identifier for creating allreduce shared memory
        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(
            ":")[-1]
        # Initialize the distributed environment.
        # Force Gloo backend for CPU workers
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank,
                                            backend="gloo")
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner with NUMA awareness
        self.model_runner: CPUModelRunner = CPUModelRunner(
            self.vllm_config, torch.device("cpu"),
            numa_node=self._numa_node)

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def determine_available_memory(self) -> int:
        # Sync with GPU worker's profile_run to avoid deadlock
        logger.info("DEBUG_AG: CPUWorker determine_available_memory start. Syncing profile_run.")
        self.model_runner.profile_run()
        logger.info("DEBUG_AG: CPUWorker determine_available_memory profile_run done.")

        return self.cache_config.cpu_kvcache_space_bytes  # type: ignore

    def determine_num_available_blocks(self) -> tuple[int, int]:
        logger.info("DEBUG_AG: CPUWorker determine_num_available_blocks start")
        
        # For CPU worker, we don't calculate blocks dynamically based on GPU mem.
        # We rely on the config or default 0.
        # Executor aggregates results.
        # We return (0, 0) or whatever is appropriate. 
        # Actually V1 Logic:
        # GPUWorker returns (num_gpu, num_cpu).
        # We should return (0, available_cpu_blocks).
        
        # Calculate available CPU blocks
        num_cpu_blocks = self.vllm_config.cache_config.num_cpu_blocks
        if num_cpu_blocks is None:
            # If not set, use default or calculate from swap space
            # For now return 0 to avoid breaking logic if unmitigated
            num_cpu_blocks = 0

        logger.info(f"DEBUG_AG: CPUWorker determine_num_available_blocks done. returning 0, {num_cpu_blocks}")
        return 0, num_cpu_blocks

    def compile_or_warm_up_model(self) -> None:
        logger.info("CPUWorker: Entering compile_or_warm_up_model")
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        try:
            set_random_seed(self.model_config.seed)
            logger.info("CPUWorker: set_random_seed completed")
        except Exception as e:
            logger.error(f"CPUWorker: set_random_seed failed: {e}")
        
        # 1. Standard Warmup (Mirror GPUWorker)
        warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x for x in warmup_sizes if x not in
                self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("CPUWorker: Shadowing warmup for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True)

        # 2. Shadow CUDAGraph Capture (Mirror GPUWorker)
        if not self.model_config.enforce_eager:
            # GPUWorker iterates over compilation_cases which is reversed(reversed(config)) = config order effectively
            # GPUWorker: compilation_cases = reversed(self.cudagraph_batch_sizes)
            # where self.cudagraph_batch_sizes = reversed(config.cudagraph_capture_sizes)
            # So compilation_cases = config.cudagraph_capture_sizes
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            num_warmups = self.vllm_config.compilation_config.cudagraph_num_of_warmups
            
            for size in capture_sizes:
                # Shadow warmups
                for i in range(num_warmups):
                     logger.info("CPUWorker: Shadowing capture warmup %d for size %d", i, size)
                     self.model_runner._dummy_run(size, skip_eplb=True)
                # Shadow capture
                logger.info("CPUWorker: Shadowing capture run for size %d", size)
                self.model_runner._dummy_run(size, skip_eplb=True)

        # 3. Final Sampler Warmup (Mirror GPUWorker)
        if get_pp_group().is_last_rank:
            max_num_reqs = min(self.scheduler_config.max_num_seqs,
                               self.scheduler_config.max_num_batched_tokens)
            # CPUModelRunner inherits from GPUModelRunner, so it has _dummy_run
            # We assume it returns the same tuple structure
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                    num_tokens=max_num_reqs, skip_eplb=True)
            
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        set_random_seed(self.model_config.seed)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        # Periodic execute_model trace.
        # Default is OFF (_every=0) to avoid stdout I/O serializing the
        # per-step hot loop under production load. Re-enable with
        #   VLLM_HYBRID_TRACE=1              (every step — smoke only)
        #   VLLM_HYBRID_TRACE_EVERY=N (N>0)  (every N steps)
        import time as _time
        _trace = os.environ.get("VLLM_HYBRID_TRACE", "0") == "1"
        _step = getattr(self, "_hybrid_exec_step", 0) + 1
        self._hybrid_exec_step = _step
        _every = int(os.environ.get("VLLM_HYBRID_TRACE_EVERY", "0"))

        if _trace or (_every > 0 and _step % _every == 0):
            num_scheduled = (
                getattr(scheduler_output, "total_num_scheduled_tokens", None)
                or getattr(scheduler_output, "num_scheduled_tokens", None)
                or 0)
            try:
                if hasattr(num_scheduled, "values"):
                    num_scheduled = sum(num_scheduled.values())
            except Exception:
                pass
            num_reqs = len(getattr(scheduler_output,
                                   "scheduled_new_reqs", []) or [])
            try:
                num_reqs += len(getattr(scheduler_output,
                                        "scheduled_cached_reqs", []) or [])
            except Exception:
                pass
            _t0 = _time.perf_counter()
        else:
            _t0 = None

        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors)

        if _t0 is not None:
            elapsed_ms = (_time.perf_counter() - _t0) * 1000
            logger.info(
                "[HYBRID-CPU-EXEC] step=%d reqs=%s tokens=%s "
                "torch_threads=%d elapsed=%.1fms",
                _step, num_reqs, num_scheduled,
                torch.get_num_threads(), elapsed_ms,
            )

        if not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return None

        assert isinstance(output, ModelRunnerOutput)
        return output if self.is_driver_worker else None

    def _get_autobind_cpu_ids(
        self, cpu_selector: Callable[[list[LogicalCPUInfo]],
                                     list[LogicalCPUInfo]]
    ) -> str:
        """
        Return CPU ids to bind based on NUMA nodes. 
        Currently for rank N, only CPU ids on the N-th node in available NUMA 
        node list will be selected.
        Args:
            cpu_selector: a callable object to select CPUs from a CPU list 
            of a physical core. The input is a LogicalCPUInfo list, sorted by
            the LogicalCPUInfo.id. A selected LogicalCPUInfo list should be 
            returned.
        """

        allowed_numa_nodes, logical_cpu_list = \
            CpuPlatform.get_allowed_cpu_memory_node_list()

        # In hybrid mode with num_cpu_engines>1, every CPU engine is a
        # separate process whose local_rank is 0. The NUMA assignment must
        # come from hybrid_config.numa_bind_node (set by
        # run_cpu_engine_core via numa_node kwarg in launch_hybrid_engines).
        # Without this override, all CPU engines would pin themselves to
        # the same NUMA node and contend for the same cores — fatal for
        # multi-NUMA hosts like H100x8 + Sapphire Rapids 2-socket.
        selected_numa_node = None
        try:
            hc = getattr(self.vllm_config, 'hybrid_config', None)
            if hc is not None:
                bind_node = getattr(hc, 'numa_bind_node', None)
                if bind_node is not None and bind_node in allowed_numa_nodes:
                    selected_numa_node = bind_node
                    logger.info(
                        "[HYBRID-CPU-WORKER] _get_autobind_cpu_ids: "
                        "using hybrid_config.numa_bind_node=%d "
                        "(allowed_nodes=%s)",
                        bind_node, allowed_numa_nodes)
        except Exception as _e:
            logger.debug("numa_bind_node lookup failed: %s", _e)

        if selected_numa_node is None:
            # Fallback: historical local_rank-based selection (single CPU
            # engine, or non-hybrid CPU-only build).
            if len(allowed_numa_nodes) < self.parallel_config.world_size:
                logger.warning(
                    f"Not enough NUMA nodes ({len(allowed_numa_nodes)}) "
                    f"for {self.parallel_config.world_size} workers. "
                    f"Workers will share NUMA nodes.")
            node_idx = self.local_rank % len(allowed_numa_nodes)
            selected_numa_node = allowed_numa_nodes[node_idx]  # type: ignore
            logger.info(
                "[HYBRID-CPU-WORKER] _get_autobind_cpu_ids: "
                "local_rank=%d → node_idx=%d → NUMA node %d",
                self.local_rank, node_idx, selected_numa_node)

        logical_cpu_list = [
            x for x in logical_cpu_list if x.numa_node == selected_numa_node
        ]

        # Select CPUs from each physical core via cpu_selector
        core_to_cpus: dict[int, list[LogicalCPUInfo]] = {}
        for cpu_info in logical_cpu_list:
            if cpu_info.physical_core not in core_to_cpus:
                core_to_cpus[cpu_info.physical_core] = []
            core_to_cpus[cpu_info.physical_core].append(cpu_info)
        logical_cpu_list = []
        for cpu_list in core_to_cpus.values():
            cpu_list = sorted(cpu_list, key=lambda x: x.id)
            logical_cpu_list.extend(cpu_selector(cpu_list))
        logical_cpu_list = sorted(logical_cpu_list, key=lambda x: x.id)

        # Reserve CPUs for other processes
        reserve_cpu_num = envs.VLLM_CPU_NUM_OF_RESERVED_CPU
        if reserve_cpu_num is None:
            reserve_cpu_num = 1 if self.parallel_config.world_size > 1 else 0
        assert len(logical_cpu_list) > reserve_cpu_num, (
            f"VLLM_CPU_NUM_OF_RESERVED_CPU ({reserve_cpu_num}) "
            f"should less than {len(logical_cpu_list)}.")
        if reserve_cpu_num != 0:
            logical_cpu_list = logical_cpu_list[:-reserve_cpu_num]

        # Apply hybrid_config.cpu_core_ratio (0<r<=1): clip the front part
        # of the physical-core list to use only ratio × cores. The C++
        # init_cpu_threads_env pins one OMP thread to each of the returned
        # cores, so shortening the list directly reduces the number of
        # cores the CPU engine actually uses. Leaves the remaining NUMA
        # cores idle for main thread / other processes.
        try:
            hc = getattr(self.vllm_config, 'hybrid_config', None)
            ratio = float(getattr(hc, 'cpu_core_ratio', 1.0) or 1.0)
        except Exception:
            ratio = 1.0
        if 0.0 < ratio < 1.0 and len(logical_cpu_list) > 1:
            keep = max(1, int(len(logical_cpu_list) * ratio))
            dropped = len(logical_cpu_list) - keep
            logical_cpu_list = logical_cpu_list[:keep]
            logger.info(
                "[HYBRID-CPU-WORKER] cpu_core_ratio=%.2f → keep %d / "
                "drop %d physical cores (NUMA %d)",
                ratio, keep, dropped, selected_numa_node)

        logger.info("auto thread-binding list (id, physical core): %s",
                    [(x.id, x.physical_core) for x in logical_cpu_list])
        return ",".join([str(x.id) for x in logical_cpu_list])
