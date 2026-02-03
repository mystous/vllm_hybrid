# SPDX-License-Identifier: Apache-2.0
"""
Parallel Batch Executor (APEX 방식)

CPU와 GPU가 독립적으로 서로 다른 배치를 동시에 처리하여
전체 처리량을 향상시킵니다.

사용법:
    vllm serve model --hybrid-mode parallel-batch

IPEX (Intel Extension for PyTorch)를 활용하여 AVX-512 최적화를 적용합니다.
NUMA-aware 메모리 할당 및 스레드 바인딩으로 멀티소켓 시스템 최적화.
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from vllm.config import HybridConfig, VllmConfig
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest

# Intel CPU 최적화 유틸리티
from vllm.platforms.intel_cpu_utils import (
    NUMAAllocator,
    detect_intel_cpu_features,
    setup_intel_cpu_environment,
    configure_intel_optimizations,
    configure_pytorch_for_intel,
    is_ipex_available,
    optimize_model_with_ipex,
    IntelCPUFeatures,
)

logger = init_logger(__name__)

# IPEX 가용성 확인 (intel_cpu_utils 사용)
_IPEX_AVAILABLE = is_ipex_available()
if _IPEX_AVAILABLE:
    import intel_extension_for_pytorch as ipex
    logger.info(f"IPEX available: version {ipex.__version__}")
else:
    logger.warning("IPEX not available. CPU performance will be limited.")


@dataclass
class ProfileResult:
    """프로파일링 결과."""
    gpu_throughput: float  # tok/s
    cpu_throughput: float  # tok/s
    optimal_gpu_ratio: float
    optimal_cpu_ratio: float


class ParallelBatchProfiler:
    """CPU/GPU 처리량 프로파일러."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        gpu_worker: Any,
        cpu_worker: "CPUWorkerWrapper",
    ):
        self.vllm_config = vllm_config
        self.gpu_worker = gpu_worker
        self.cpu_worker = cpu_worker
        self.profile_result: Optional[ProfileResult] = None

        # 프로파일링용 설정
        self.profile_seq_len = 128
        self.profile_batch_size = 1

    def profile(self, num_batches: int = 5) -> ProfileResult:
        """CPU/GPU 처리량 측정."""
        logger.info("Starting parallel-batch profiling...")

        # CPU 워커 초기화 (아직 안됐으면)
        if not self.cpu_worker.initialized:
            try:
                self.cpu_worker.initialize()
            except Exception as e:
                logger.warning(f"CPU worker initialization failed: {e}")

        # GPU 프로파일링 (기존 executor 사용)
        gpu_throughput = self._measure_gpu_throughput(num_batches)

        # CPU 프로파일링 (IPEX 최적화된 모델)
        cpu_throughput = self._measure_cpu_throughput(num_batches)

        # 최적 비율 계산
        total = gpu_throughput + cpu_throughput
        if total > 0:
            gpu_ratio = gpu_throughput / total
            cpu_ratio = cpu_throughput / total
        else:
            # CPU 실패 시 GPU만 사용
            gpu_ratio = 1.0
            cpu_ratio = 0.0

        self.profile_result = ProfileResult(
            gpu_throughput=gpu_throughput,
            cpu_throughput=cpu_throughput,
            optimal_gpu_ratio=gpu_ratio,
            optimal_cpu_ratio=cpu_ratio,
        )

        logger.info(
            f"Profiling complete: GPU={gpu_throughput:.2f} tok/s, "
            f"CPU={cpu_throughput:.2f} tok/s, "
            f"Optimal ratio: GPU={gpu_ratio:.1%}, CPU={cpu_ratio:.1%}"
        )

        return self.profile_result

    def _measure_gpu_throughput(self, num_batches: int) -> float:
        """GPU 처리량 측정."""
        # GPU executor가 없으면 0 반환
        if self.gpu_worker is None:
            return 0.0

        # 실제 측정은 GPU executor에 위임
        # 여기서는 추정값 사용 (실제 구현에서는 측정)
        # H100 기준 대략적인 추정
        estimated_gpu_throughput = 100.0  # tok/s per sequence

        logger.info(f"GPU throughput (estimated): {estimated_gpu_throughput:.2f} tok/s")
        return estimated_gpu_throughput

    def _measure_cpu_throughput(self, num_batches: int) -> float:
        """CPU 처리량 측정 (실제 측정)."""
        if self.cpu_worker is None or self.cpu_worker.model is None:
            logger.warning("CPU model not available for profiling")
            return 0.0

        try:
            # 더미 입력 생성
            dummy_input = torch.randint(
                0, 32000,
                (self.profile_batch_size, self.profile_seq_len),
                device='cpu'
            )

            # 워밍업
            logger.info("CPU profiling warmup...")
            for _ in range(2):
                with torch.no_grad():
                    _ = self.cpu_worker.model(dummy_input)

            # 측정
            logger.info(f"CPU profiling ({num_batches} batches)...")
            start = time.perf_counter()
            total_tokens = 0

            for _ in range(num_batches):
                with torch.no_grad(), torch.cpu.amp.autocast(
                    enabled=(self.cpu_worker.hybrid_config.cpu_dtype == "bfloat16")
                ):
                    _ = self.cpu_worker.model(dummy_input)
                total_tokens += self.profile_seq_len * self.profile_batch_size

            elapsed = time.perf_counter() - start
            throughput = total_tokens / elapsed if elapsed > 0 else 0.0

            logger.info(f"CPU throughput (measured): {throughput:.2f} tok/s")
            return throughput

        except Exception as e:
            logger.warning(f"CPU profiling failed: {e}")
            return 0.0


class ParallelBatchScheduler:
    """CPU/GPU 배치 분할 스케줄러."""

    def __init__(
        self,
        gpu_ratio: float,
        cpu_ratio: float,
    ):
        self.gpu_ratio = gpu_ratio
        self.cpu_ratio = cpu_ratio

        logger.info(
            f"ParallelBatchScheduler initialized: "
            f"GPU={gpu_ratio:.1%}, CPU={cpu_ratio:.1%}"
        )

    def partition_requests(
        self,
        requests: List[Any],
    ) -> Tuple[List[Any], List[Any]]:
        """요청을 GPU/CPU로 분할."""
        n = len(requests)
        if n == 0:
            return [], []

        gpu_count = max(1, int(n * self.gpu_ratio))

        # 긴 요청은 GPU로 (compute bound에서 효율적)
        # 짧은 요청은 CPU로
        sorted_requests = sorted(
            requests,
            key=lambda r: getattr(r, 'num_tokens', 0),
            reverse=True
        )

        gpu_batch = sorted_requests[:gpu_count]
        cpu_batch = sorted_requests[gpu_count:]

        return gpu_batch, cpu_batch


class CPUWorkerWrapper:
    """CPU 워커 래퍼 (IPEX + AVX-512 + NUMA 최적화)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        hybrid_config: HybridConfig,
        worker_rank: int = 0,
        num_cpu_workers: int = 1,
    ):
        self.vllm_config = vllm_config
        self.hybrid_config = hybrid_config
        self.worker_rank = worker_rank
        self.num_cpu_workers = num_cpu_workers
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self.device = torch.device('cpu')

        # NUMA 관련
        self.numa_allocator: Optional[NUMAAllocator] = None
        self.numa_node: int = -1
        self.cpu_features: Optional[IntelCPUFeatures] = None
        self.env_config: dict = {}

        # 스레드 설정은 NUMA 설정 후에 적용
        self._initial_thread_count = hybrid_config.cpu_num_threads

    def initialize(self):
        """CPU 모델 초기화 (IPEX + NUMA 최적화 적용)."""
        if self.initialized:
            return

        logger.info(
            f"Initializing CPU worker (rank={self.worker_rank}) "
            f"with initial threads={self._initial_thread_count}"
        )

        # CPU 환경 설정 (NUMA 포함)
        self._setup_cpu_environment()

        # 모델 로드
        self._load_model()

        self.initialized = True
        features_str = ""
        if self.cpu_features:
            features_str = f", AVX512={self.cpu_features.avx512f}"
            if self.cpu_features.avx512_vnni:
                features_str += ", VNNI"
        logger.info(
            f"CPU worker initialized: NUMA node={self.numa_node}, "
            f"threads={torch.get_num_threads()}{features_str}"
        )

    def _setup_cpu_environment(self):
        """CPU 최적화 환경 설정 (NUMA + AVX-512 + IPEX)."""
        # NUMA 활성화 여부 확인
        enable_numa = self.hybrid_config.numa_aware

        # 1. Intel CPU 환경 전체 설정 (NUMA, AVX-512, PyTorch 설정 포함)
        self.env_config = setup_intel_cpu_environment(
            rank=self.worker_rank,
            world_size=self.num_cpu_workers,
            enable_numa=enable_numa,
            enable_avx_optimization=True,
            enable_ipex=True,
        )

        # 명시적 NUMA 노드 바인딩이 지정된 경우
        if self.hybrid_config.numa_bind_node is not None:
            self.env_config["numa_node"] = self.hybrid_config.numa_bind_node

        # 2. NUMA 정보 저장
        self.numa_node = self.env_config.get("numa_node", -1)
        self.cpu_features = self.env_config.get("features")

        # 3. NUMA Allocator 초기화
        self.numa_allocator = NUMAAllocator()

        # 4. NUMA 기반 스레드 수 최적화
        self._configure_numa_threads()

        # 5. 추가 환경변수 설정
        os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX512_CORE_VNNI")
        os.environ.setdefault("TORCHINDUCTOR_CPP_BACKEND", "1")

        # 로그
        logger.info(
            f"CPU environment configured: "
            f"NUMA node={self.numa_node}, "
            f"NUMA available={self.numa_allocator.is_available if self.numa_allocator else False}, "
            f"num_nodes={self.numa_allocator.num_nodes if self.numa_allocator else 1}"
        )

    def _configure_numa_threads(self):
        """NUMA 토폴로지 기반 스레드 설정."""
        # Fallback 스레드 수 (최소 1)
        fallback_threads = max(1, self._initial_thread_count)

        if not self.numa_allocator or not self.numa_allocator.is_available:
            # NUMA 미지원 시 사용자 설정 사용
            torch.set_num_threads(fallback_threads)
            logger.info(f"NUMA not available, using {fallback_threads} threads")
            return

        # NUMA 노드가 -1이면 노드 0 사용
        target_numa_node = self.numa_node if self.numa_node >= 0 else 0

        # NUMA 노드 정보 가져오기
        node_info = self.numa_allocator.get_node_info(target_numa_node)
        if node_info is None or not node_info.cpu_ids:
            torch.set_num_threads(fallback_threads)
            logger.info(f"NUMA node {target_numa_node} info unavailable, using {fallback_threads} threads")
            return

        num_cpus_in_node = len(node_info.cpu_ids)
        num_numa_nodes = self.numa_allocator.num_nodes

        # 스레드 수 결정 로직:
        # - 단일 워커: 로컬 NUMA 노드의 모든 코어 사용
        # - 멀티 워커: 워커당 NUMA 노드 할당
        if self.num_cpu_workers == 1:
            # 단일 워커: NUMA 인터리빙 없이 로컬 노드만 사용
            # 전체 코어를 사용하되, HT를 고려하여 물리 코어 수 선호
            if self.cpu_features:
                threads_per_core = self.cpu_features.threads_per_core or 1
                physical_cores = num_cpus_in_node // max(1, threads_per_core)
                optimal_threads = max(physical_cores, num_cpus_in_node // 2)
            else:
                optimal_threads = num_cpus_in_node

            # 사용자 설정과 비교
            if self._initial_thread_count > 0 and self._initial_thread_count != 48:
                # 명시적으로 지정한 경우 해당 값 사용
                optimal_threads = min(self._initial_thread_count, num_cpus_in_node)

        else:
            # 멀티 워커: 워커별로 NUMA 노드 분배
            optimal_threads = num_cpus_in_node

        # 최소 1 스레드 보장
        optimal_threads = max(1, optimal_threads)

        torch.set_num_threads(optimal_threads)
        os.environ["OMP_NUM_THREADS"] = str(optimal_threads)

        # CPU 어피니티 설정 (NUMA 노드 내 코어에 바인딩)
        if node_info.cpu_ids:
            cpu_list = ",".join(str(c) for c in node_info.cpu_ids[:optimal_threads])
            os.environ["GOMP_CPU_AFFINITY"] = cpu_list
            os.environ["KMP_AFFINITY"] = f"explicit,proclist=[{cpu_list}],granularity=fine"

        logger.info(
            f"NUMA thread config: node={self.numa_node}, "
            f"cpus_in_node={num_cpus_in_node}, "
            f"threads={optimal_threads}, "
            f"total_numa_nodes={num_numa_nodes}"
        )

        # 메모리 대역폭 정보 로깅
        if node_info.total_memory_bytes > 0:
            total_gb = node_info.total_memory_bytes / (1024**3)
            free_gb = node_info.free_memory_bytes / (1024**3)
            logger.info(
                f"NUMA node {self.numa_node} memory: "
                f"total={total_gb:.1f}GB, free={free_gb:.1f}GB"
            )

    def _load_model(self):
        """모델 로드 및 IPEX + NUMA 최적화 적용."""
        model_config = self.vllm_config.model_config

        logger.info(
            f"Loading model {model_config.model} for CPU inference "
            f"(NUMA node={self.numa_node})..."
        )

        try:
            # NUMA 노드에 메모리 할당 바인딩
            if self.numa_allocator and self.numa_allocator.is_available:
                self.numa_allocator.bind_to_node(self.numa_node)
                logger.info(f"Memory allocation bound to NUMA node {self.numa_node}")

            # Transformers에서 모델 로드
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # dtype 결정
            dtype_str = self.hybrid_config.cpu_dtype
            if dtype_str == "bfloat16":
                dtype = torch.bfloat16
            elif dtype_str == "float16":
                dtype = torch.float16
            elif dtype_str == "int8":
                dtype = torch.float32  # INT8은 나중에 양자화
            else:
                dtype = torch.float32

            # 모델 로드 (NUMA-aware - 현재 바인딩된 노드에 할당됨)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config.model,
                torch_dtype=dtype,
                trust_remote_code=model_config.trust_remote_code,
                low_cpu_mem_usage=True,
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_config.tokenizer or model_config.model,
                trust_remote_code=model_config.trust_remote_code,
            )

            # IPEX 최적화 적용 (intel_cpu_utils 활용)
            if _IPEX_AVAILABLE:
                self._apply_ipex_optimization(dtype_str)
            else:
                logger.warning("IPEX not available, using standard PyTorch")
                # PyTorch 기본 최적화라도 적용
                if self.cpu_features:
                    configure_pytorch_for_intel(self.cpu_features)

            # 모델 메모리 사용량 로깅
            self._log_model_memory()

            logger.info(f"Model loaded: {model_config.model}, dtype={dtype}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _apply_ipex_optimization(self, dtype_str: str):
        """IPEX 최적화 적용 (AMX > AVX-512 VNNI > AVX2 순서로 활용)."""
        try:
            import intel_extension_for_pytorch as ipex

            # AMX 사용 가능 여부 확인
            amx_available = self.cpu_features and (
                self.cpu_features.amx_bf16 or self.cpu_features.amx_int8
            )

            # IPEX에서 AMX 활성화 (가능한 경우)
            if amx_available:
                self._enable_ipex_amx()

            if dtype_str == "int8":
                # INT8 양자화 - AMX-INT8 또는 AVX-512 VNNI로 가속
                accel_type = "AMX-INT8" if (self.cpu_features and self.cpu_features.amx_int8) else "AVX-512 VNNI"
                logger.info(f"Applying IPEX INT8 quantization ({accel_type} accelerated)...")
                try:
                    from intel_extension_for_pytorch.quantization import prepare, convert

                    # 양자화 설정
                    qconfig = ipex.quantization.default_dynamic_qconfig
                    self.model = prepare(self.model, qconfig, inplace=True)
                    self.model = convert(self.model, inplace=True)
                    logger.info(f"IPEX INT8 quantization applied ({accel_type})")
                except ImportError:
                    # IPEX 버전에 따라 API가 다를 수 있음
                    logger.warning("IPEX quantization API not available, using BF16 fallback")
                    self.model = ipex.optimize(
                        self.model,
                        dtype=torch.bfloat16,
                        auto_kernel_selection=True,
                    )

            elif dtype_str == "bfloat16":
                # BF16 최적화 - AMX-BF16 > AVX-512 BF16 순서로 활용
                if self.cpu_features and self.cpu_features.amx_bf16:
                    accel_type = "AMX-BF16"
                elif self.cpu_features and self.cpu_features.avx512_bf16:
                    accel_type = "AVX-512 BF16"
                else:
                    accel_type = "AVX-512"

                logger.info(f"Applying IPEX BF16 optimization ({accel_type})...")
                self.model = ipex.optimize(
                    self.model,
                    dtype=torch.bfloat16,
                    auto_kernel_selection=True,
                    weights_prepack=True,  # 가중치 프리팩 (GEMM 최적화)
                )
                logger.info(f"IPEX BF16 optimization applied ({accel_type})")

            else:
                # FP32 최적화 (AMX 사용 시 BF16으로 내부 연산 가능)
                logger.info("Applying IPEX FP32 optimization...")

                # AMX가 있으면 FP32 입력을 BF16으로 처리하는 옵션 활성화
                if amx_available:
                    try:
                        # IPEX 2.0+: FP32 연산을 BF16으로 수행 (AMX 활용)
                        ipex.set_fp32_math_mode(ipex.FP32MathMode.BF16)
                        logger.info("  FP32 math mode set to BF16 (AMX accelerated)")
                    except AttributeError:
                        pass

                self.model = ipex.optimize(
                    self.model,
                    weights_prepack=True,
                )
                logger.info("IPEX FP32 optimization applied")

        except Exception as e:
            logger.warning(f"IPEX optimization failed: {e}, using standard PyTorch")

    def _enable_ipex_amx(self):
        """IPEX에서 AMX 활성화."""
        try:
            import intel_extension_for_pytorch as ipex

            # IPEX 2.0+에서 AMX 관련 설정
            # 1. FP32 연산을 BF16으로 수행 (AMX-BF16 활용)
            if hasattr(ipex, 'set_fp32_math_mode'):
                # FP32MathMode.BF16: FP32 입력을 BF16으로 변환하여 AMX로 처리
                ipex.set_fp32_math_mode(ipex.FP32MathMode.BF16)
                logger.info("IPEX AMX enabled: FP32 math mode set to BF16")

            # 2. oneDNN graph fusion 활성화 (AMX 커널 사용)
            if hasattr(ipex, 'enable_onednn_fusion'):
                ipex.enable_onednn_fusion(True)
                logger.debug("IPEX oneDNN fusion enabled")

            logger.info(
                f"IPEX AMX configuration complete: "
                f"AMX-BF16={self.cpu_features.amx_bf16}, "
                f"AMX-INT8={self.cpu_features.amx_int8}"
            )

        except Exception as e:
            logger.debug(f"IPEX AMX setup skipped: {e}")

    def _log_model_memory(self):
        """모델 메모리 사용량 로깅."""
        try:
            # 모델 파라미터 메모리 추정
            total_params = sum(p.numel() for p in self.model.parameters())
            param_memory_gb = total_params * 2 / (1024**3)  # BF16 기준

            # NUMA 노드별 메모리 상황
            if self.numa_allocator and self.numa_allocator.is_available:
                node_info = self.numa_allocator.get_node_info(self.numa_node)
                if node_info:
                    free_gb = node_info.free_memory_bytes / (1024**3)
                    logger.info(
                        f"Model memory: ~{param_memory_gb:.2f}GB, "
                        f"NUMA node {self.numa_node} free: {free_gb:.1f}GB"
                    )
                    return

            logger.info(f"Model memory: ~{param_memory_gb:.2f}GB")
        except Exception:
            pass

    def execute_model(self, request: ExecuteModelRequest) -> Any:
        """CPU에서 모델 실행."""
        if not self.initialized:
            self.initialize()

        if self.model is None:
            logger.warning("CPU model not loaded")
            return None

        try:
            # 요청에서 입력 추출
            outputs = []
            for seq_group in request.seq_group_metadata_list:
                seq_data = seq_group.seq_data
                input_ids = list(seq_data.values())[0].get_token_ids()

                # 텐서 변환
                input_tensor = torch.tensor([input_ids], device=self.device)

                # 추론
                with torch.no_grad(), torch.cpu.amp.autocast(
                    enabled=(self.hybrid_config.cpu_dtype == "bfloat16")
                ):
                    output = self.model(input_tensor)
                    logits = output.logits[:, -1, :]

                outputs.append(logits)

            return outputs

        except Exception as e:
            logger.error(f"CPU execution failed: {e}")
            return None

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """토큰 생성 (간단한 greedy decoding)."""
        if not self.initialized:
            self.initialize()

        if self.model is None:
            return None

        input_ids = input_ids.to(self.device)

        with torch.no_grad(), torch.cpu.amp.autocast(
            enabled=(self.hybrid_config.cpu_dtype == "bfloat16")
        ):
            for _ in range(max_new_tokens):
                output = self.model(input_ids)
                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids


class ParallelBatchExecutor:
    """
    Parallel Batch Executor (APEX 방식).

    CPU와 GPU가 독립적으로 서로 다른 배치를 동시에 처리합니다.
    IPEX를 활용하여 CPU에서 AVX-512 최적화를 적용합니다.
    NUMA-aware 멀티 워커로 멀티소켓 시스템 최적화.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        gpu_executor: Any = None,  # 기존 GPU executor
    ):
        self.vllm_config = vllm_config
        self.hybrid_config = vllm_config.hybrid_config
        self.gpu_executor = gpu_executor

        # NUMA 토폴로지 확인
        self.numa_allocator = NUMAAllocator()
        self.num_numa_nodes = self.numa_allocator.num_nodes if self.numa_allocator.is_available else 1

        # CPU 워커 수 결정 (NUMA 노드당 1개 또는 단일)
        self.num_cpu_workers = self._determine_num_cpu_workers()

        # CPU 워커들 (NUMA 노드별)
        self.cpu_workers: List[CPUWorkerWrapper] = []
        for rank in range(self.num_cpu_workers):
            worker = CPUWorkerWrapper(
                vllm_config,
                self.hybrid_config,
                worker_rank=rank,
                num_cpu_workers=self.num_cpu_workers,
            )
            self.cpu_workers.append(worker)

        # 기본 워커 (단일 워커 호환성용)
        self.cpu_worker = self.cpu_workers[0] if self.cpu_workers else None

        # 스케줄러 (프로파일링 후 설정)
        self.scheduler: Optional[ParallelBatchScheduler] = None

        # 비동기 실행용 스레드풀 (NUMA 워커당 1개 스레드)
        self.cpu_thread_pool = ThreadPoolExecutor(
            max_workers=max(1, self.num_cpu_workers),
            thread_name_prefix="cpu_worker"
        )

        # 통계
        self.stats = {
            "gpu_requests": 0,
            "cpu_requests": 0,
            "gpu_tokens": 0,
            "cpu_tokens": 0,
        }

        # NUMA 토폴로지 로깅
        self._log_numa_topology()

        logger.info(
            f"ParallelBatchExecutor created: "
            f"cpu_workers={self.num_cpu_workers}, "
            f"numa_nodes={self.num_numa_nodes}, "
            f"cpu_dtype={self.hybrid_config.cpu_dtype}, "
            f"ipex_available={_IPEX_AVAILABLE}"
        )

    def _determine_num_cpu_workers(self) -> int:
        """NUMA 토폴로지 기반 CPU 워커 수 결정."""
        # NUMA 미지원 또는 단일 노드: 워커 1개
        if not self.numa_allocator.is_available or self.num_numa_nodes <= 1:
            return 1

        # 멀티 NUMA 노드: 노드당 1개 워커 (현재는 단순화)
        # 향후 확장: 각 NUMA 노드에서 별도 모델 인스턴스 실행
        # 현재는 메인 워커 1개만 사용 (메모리 제약)
        # TODO: 멀티 모델 인스턴스 지원 시 num_numa_nodes 반환
        return 1

    def _log_numa_topology(self):
        """NUMA 토폴로지 정보 로깅."""
        if not self.numa_allocator.is_available:
            logger.info("NUMA: Not available (single memory domain)")
            return

        logger.info(f"NUMA topology: {self.num_numa_nodes} nodes detected")

        total_memory_gb = 0
        total_cpus = 0

        for node_id in range(self.num_numa_nodes):
            node_info = self.numa_allocator.get_node_info(node_id)
            if node_info:
                mem_gb = node_info.total_memory_bytes / (1024**3)
                num_cpus = len(node_info.cpu_ids)
                total_memory_gb += mem_gb
                total_cpus += num_cpus
                logger.info(
                    f"  Node {node_id}: {num_cpus} CPUs, "
                    f"{mem_gb:.1f}GB memory, "
                    f"CPUs: {node_info.cpu_ids[:8]}..." if num_cpus > 8 else f"CPUs: {node_info.cpu_ids}"
                )

        logger.info(
            f"  Total: {total_cpus} CPUs, {total_memory_gb:.1f}GB memory"
        )

    def initialize(self):
        """초기화 (프로파일링 포함)."""
        logger.info("Initializing ParallelBatchExecutor...")

        # 모든 CPU 워커 초기화
        initialized_workers = 0
        for i, worker in enumerate(self.cpu_workers):
            try:
                worker.initialize()
                initialized_workers += 1
            except Exception as e:
                logger.error(f"CPU worker {i} initialization failed: {e}")

        if initialized_workers == 0:
            logger.error("All CPU workers failed to initialize")
            # CPU 실패 시 GPU만 사용
            self.scheduler = ParallelBatchScheduler(1.0, 0.0)
            return

        logger.info(f"Initialized {initialized_workers}/{len(self.cpu_workers)} CPU workers")

        # 프로파일링 또는 수동 설정
        if self.hybrid_config.auto_profile:
            self._run_profiling()
        else:
            gpu_ratio = 1.0 - (self.hybrid_config.cpu_ratio or 0.0)
            cpu_ratio = self.hybrid_config.cpu_ratio or 0.0
            self.scheduler = ParallelBatchScheduler(gpu_ratio, cpu_ratio)

        logger.info("ParallelBatchExecutor initialized")

    def _run_profiling(self):
        """프로파일링 실행."""
        logger.info("Running profiling to determine optimal CPU/GPU ratio...")

        profiler = ParallelBatchProfiler(
            self.vllm_config,
            self.gpu_executor,
            self.cpu_worker,
        )

        try:
            result = profiler.profile(
                num_batches=self.hybrid_config.profile_num_batches
            )
            self.scheduler = ParallelBatchScheduler(
                result.optimal_gpu_ratio,
                result.optimal_cpu_ratio,
            )
        except Exception as e:
            logger.error(f"Profiling failed: {e}, using GPU only")
            self.scheduler = ParallelBatchScheduler(1.0, 0.0)

    def execute_model(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> Any:
        """동기 모델 실행 (CPU/GPU 병렬)."""
        if self.scheduler is None:
            self.initialize()

        seq_groups = execute_model_req.seq_group_metadata_list
        if not seq_groups:
            return None

        # 배치 분할
        gpu_groups, cpu_groups = self.scheduler.partition_requests(seq_groups)

        results = []

        # GPU 실행
        if gpu_groups and self.gpu_executor:
            gpu_req = ExecuteModelRequest(
                seq_group_metadata_list=gpu_groups,
                num_steps=execute_model_req.num_steps,
            )
            gpu_result = self.gpu_executor.execute_model(gpu_req)
            if gpu_result:
                results.extend(gpu_result if isinstance(gpu_result, list) else [gpu_result])
            self.stats["gpu_requests"] += len(gpu_groups)

        # CPU 실행 (병렬)
        if cpu_groups:
            cpu_req = ExecuteModelRequest(
                seq_group_metadata_list=cpu_groups,
                num_steps=execute_model_req.num_steps,
            )
            cpu_result = self.cpu_worker.execute_model(cpu_req)
            if cpu_result:
                results.extend(cpu_result if isinstance(cpu_result, list) else [cpu_result])
            self.stats["cpu_requests"] += len(cpu_groups)

        return results

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> Any:
        """비동기 모델 실행 (CPU/GPU 병렬)."""
        if self.scheduler is None:
            self.initialize()

        seq_groups = execute_model_req.seq_group_metadata_list
        if not seq_groups:
            return None

        # 배치 분할
        gpu_groups, cpu_groups = self.scheduler.partition_requests(seq_groups)

        tasks = []

        # GPU 태스크
        if gpu_groups and self.gpu_executor:
            gpu_req = ExecuteModelRequest(
                seq_group_metadata_list=gpu_groups,
                num_steps=execute_model_req.num_steps,
            )
            gpu_task = asyncio.create_task(
                self._execute_gpu_async(gpu_req)
            )
            tasks.append(("gpu", gpu_task, len(gpu_groups)))

        # CPU 태스크
        if cpu_groups:
            cpu_req = ExecuteModelRequest(
                seq_group_metadata_list=cpu_groups,
                num_steps=execute_model_req.num_steps,
            )
            cpu_task = asyncio.create_task(
                self._execute_cpu_async(cpu_req)
            )
            tasks.append(("cpu", cpu_task, len(cpu_groups)))

        # 결과 수집
        results = []
        for name, task, count in tasks:
            try:
                result = await task
                if result:
                    results.extend(result if isinstance(result, list) else [result])
                self.stats[f"{name}_requests"] += count
            except Exception as e:
                logger.error(f"{name} execution failed: {e}")

        return results

    async def _execute_gpu_async(self, request: ExecuteModelRequest) -> Any:
        """GPU 비동기 실행."""
        if hasattr(self.gpu_executor, 'execute_model_async'):
            return await self.gpu_executor.execute_model_async(request)
        else:
            # 동기 실행을 스레드풀에서
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.gpu_executor.execute_model,
                request,
            )

    async def _execute_cpu_async(self, request: ExecuteModelRequest) -> Any:
        """CPU 비동기 실행 (별도 스레드)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.cpu_thread_pool,
            self.cpu_worker.execute_model,
            request,
        )

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환."""
        total = self.stats["gpu_requests"] + self.stats["cpu_requests"]
        return {
            **self.stats,
            "total_requests": total,
            "gpu_ratio": self.stats["gpu_requests"] / total if total > 0 else 0,
            "cpu_ratio": self.stats["cpu_requests"] / total if total > 0 else 0,
            "scheduler_gpu_ratio": self.scheduler.gpu_ratio if self.scheduler else 0,
            "scheduler_cpu_ratio": self.scheduler.cpu_ratio if self.scheduler else 0,
        }

    def shutdown(self):
        """종료."""
        logger.info(f"ParallelBatchExecutor shutting down. Stats: {self.get_stats()}")
        self.cpu_thread_pool.shutdown(wait=True)
        logger.info("ParallelBatchExecutor shutdown complete")


# =============================================================================
# Dummy MoE Hybrid Executor (추후 구현)
# =============================================================================

class MoEHybridExecutor:
    """
    MoE Hybrid Executor (추후 구현).

    - MoE Expert Offload: GPU(Attention) + CPU(Experts)
    - N-gram Speculative Decoding
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        gpu_executor: Any,
    ):
        self.vllm_config = vllm_config
        self.gpu_executor = gpu_executor

        logger.warning(
            "MoEHybridExecutor is not yet implemented. "
            "Falling back to GPU-only execution."
        )

    async def execute_model_async(
        self,
        requests: List[Any],
    ) -> List[Any]:
        """더미 실행 (GPU만 사용)."""
        return self.gpu_executor.execute_model(requests)


# =============================================================================
# Factory
# =============================================================================

def create_hybrid_executor(
    vllm_config: VllmConfig,
    gpu_executor: Any,
) -> Optional[Any]:
    """하이브리드 executor 생성."""
    hybrid_config = vllm_config.hybrid_config

    if not hybrid_config.is_enabled():
        return None

    if hybrid_config.mode == "parallel-batch":
        return ParallelBatchExecutor(vllm_config, gpu_executor)
    elif hybrid_config.mode == "moe-hybrid":
        return MoEHybridExecutor(vllm_config, gpu_executor)
    else:
        return None
