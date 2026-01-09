# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

logger = init_logger(__name__)


class CPUWorker(Worker):

    def __init__(self,
                 vllm_config: VllmConfig,
                 local_rank: int,
                 rank: int,
                 distributed_init_method: str,
                 is_driver_worker: bool = False):
        # Force CPU Platform for this worker to ensure correct backend selection (e.g. Attention)
        try:
            from vllm.platforms.cpu import CpuPlatform
            import vllm.platforms
            vllm.platforms._current_platform = CpuPlatform()
            # Also clear the cached backend function to ensure re-resolution
            from vllm.attention.selector import _cached_get_attn_backend
            _cached_get_attn_backend.cache_clear()
            logger.info("Forced vllm.platforms._current_platform to CpuPlatform for CPUWorker.")
        except Exception as e:
            logger.error(f"Failed to force CpuPlatform: {e}")

        # Ensure the config is compatible with CPU (e.g. downgrades compilation level)
        try:
            from vllm.platforms.cpu import CpuPlatform
            CpuPlatform.check_and_update_config(vllm_config)
            logger.info("Updated vllm_config for CPU platform via check_and_update_config.")
        except Exception as e:
             logger.error(f"Failed to update config for CPU: {str(e)}")
        
        # Force eager execution to bypass compilation on CPU (Dynamo/Inductor on CPU is unstable in this context)
        # Force eager execution to bypass compilation on CPU (Dynamo/Inductor on CPU is unstable in this context)
        vllm_config.model_config.enforce_eager = True
        
        # Explicitly disable compilation in compilation_config as well, since decorators check this directly
        from vllm.config import CompilationLevel
        vllm_config.compilation_config.level = CompilationLevel.NO_COMPILATION


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

        if self.local_omp_cpuid != "all":
            try:
                ret = torch.ops._C_utils.init_cpu_threads_env(self.local_omp_cpuid)
                if ret:
                    logger.info(ret)
            except AttributeError:
                 logger.warning("torch.ops._C_utils.init_cpu_threads_env not found. Skipping thread binding.")


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

        # Construct the model runner
        self.model_runner: CPUModelRunner = CPUModelRunner(
            self.vllm_config, torch.device("cpu"))

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
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors)

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
        
        # In heterogeneous mode (or if we just don't have enough NUMA nodes), we shouldn't crash.
        # Just warn and reuse nodes.
        if len(allowed_numa_nodes) < self.parallel_config.world_size:
             logger.warning(
                 f"Not enough NUMA nodes ({len(allowed_numa_nodes)}) for {self.parallel_config.world_size} workers. "
                 f"Workers will share NUMA nodes."
             )


        # Get CPUs on NUMA node `allowed_numa_nodes[local_rank]``
        # Use modulo to cycle through available nodes if local_rank exceeds available nodes
        node_idx = self.local_rank % len(allowed_numa_nodes)
        selected_numa_node = allowed_numa_nodes[node_idx]  # type: ignore
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

        logger.info("auto thread-binding list (id, physical core): %s",
                    [(x.id, x.physical_core) for x in logical_cpu_list])
        return ",".join([str(x.id) for x in logical_cpu_list])
