# SPDX-License-Identifier: Apache-2.0
"""
HybridEngineCore: GPU + CPU 듀얼 추론 엔진.

GPU 경로: 기존 v1 EngineCore 100% 유지 (변경 없음)
CPU 경로: 별도 프로세스에서 독립 추론 (자체 모델, KV cache)

요청 라우팅: add_request() 시점에서 cpu_ratio 기반 분배
결과 병합: step() 시점에서 EngineCoreOutputs 병합

사용법:
    vllm serve model --hybrid-mode parallel-batch --hybrid-cpu-ratio 0.05
"""

import multiprocessing
import os
import queue
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.engine import (EngineCoreOutput, EngineCoreOutputs,
                             EngineCoreRequest, FinishReason)
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.v1.request import Request

logger = init_logger(__name__)


# ============================================================================
# Request Router
# ============================================================================

class RequestRouter:
    """cpu_ratio 기반 요청 분배기.

    단순 라운드로빈 방식으로 cpu_ratio 비율에 따라 요청을 CPU로 라우팅.
    """

    def __init__(self, cpu_ratio: float):
        self.cpu_ratio = max(0.0, min(1.0, cpu_ratio))
        self.request_counter = 0
        self.gpu_count = 0
        self.cpu_count = 0

        if self.cpu_ratio > 0:
            self.interval = max(1, int(1.0 / self.cpu_ratio))
        else:
            self.interval = 0

        logger.info(
            f"RequestRouter initialized: cpu_ratio={self.cpu_ratio:.2%}, "
            f"interval={self.interval}"
        )

    def route(self, request_id: str) -> str:
        """요청을 GPU 또는 CPU로 라우팅.

        Returns:
            "gpu" or "cpu"
        """
        if self.cpu_ratio <= 0 or self.interval <= 0:
            self.gpu_count += 1
            return "gpu"

        self.request_counter += 1
        if self.request_counter % self.interval == 0:
            self.cpu_count += 1
            return "cpu"
        else:
            self.gpu_count += 1
            return "gpu"

    def get_stats(self) -> dict:
        total = self.gpu_count + self.cpu_count
        return {
            "gpu_requests": self.gpu_count,
            "cpu_requests": self.cpu_count,
            "total_requests": total,
            "actual_cpu_ratio": (
                self.cpu_count / total if total > 0 else 0.0),
        }


# ============================================================================
# CPU Request/Response containers
# ============================================================================

@dataclass
class CPUInferenceRequest:
    """CPU 추론 요청."""
    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stop_token_ids: list[int]


@dataclass
class CPUInferenceResponse:
    """CPU 추론 응답."""
    request_id: str
    generated_token_ids: list[int]
    finished: bool
    finish_reason: Optional[str]  # "stop", "length", None


# ============================================================================
# CPU Inference Process
# ============================================================================

_SHUTDOWN_SENTINEL = "SHUTDOWN"


class CPUInferenceProcess:
    """별도 프로세스에서 실행되는 CPU 추론 엔진.

    - multiprocessing.Process + Queue 기반 통신
    - transformers.AutoModelForCausalLM으로 모델 로드
    - IPEX/NUMA 최적화 적용
    - 요청별 token-by-token 생성 후 결과 반환
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.hybrid_config = vllm_config.hybrid_config
        self.model_name = vllm_config.model_config.model
        self.trust_remote_code = vllm_config.model_config.trust_remote_code

        self.request_queue: multiprocessing.Queue = multiprocessing.Queue()
        self.result_queue: multiprocessing.Queue = multiprocessing.Queue()
        self.process: Optional[multiprocessing.Process] = None
        self.alive = False

        # 추적 중인 CPU 요청
        self.pending_requests: set[str] = set()

    def start(self):
        """CPU 추론 프로세스 시작."""
        ctx = multiprocessing.get_context("spawn")
        self.request_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.process = ctx.Process(
            target=self._cpu_worker_loop,
            args=(
                self.model_name,
                self.trust_remote_code,
                self.hybrid_config.cpu_num_threads,
                self.hybrid_config.cpu_dtype,
                self.hybrid_config.numa_aware,
                self.hybrid_config.numa_bind_node,
                self.hybrid_config.cpu_max_num_seqs,
                self.request_queue,
                self.result_queue,
            ),
            daemon=True,
            name="cpu-inference",
        )
        self.process.start()
        self.alive = True
        logger.info(
            f"CPU inference process started (PID={self.process.pid}): "
            f"model={self.model_name}, threads={self.hybrid_config.cpu_num_threads}, "
            f"dtype={self.hybrid_config.cpu_dtype}"
        )

    def submit_request(self, request: EngineCoreRequest):
        """CPU에 추론 요청 제출."""
        if not self.alive:
            logger.warning("CPU process not alive, cannot submit request")
            return

        # SamplingParams에서 필요한 정보 추출
        max_tokens = 128  # 기본값
        temperature = 1.0
        top_p = 1.0
        top_k = -1
        stop_token_ids = []

        if request.sampling_params is not None:
            sp = request.sampling_params
            max_tokens = sp.max_tokens if sp.max_tokens is not None else 128
            temperature = sp.temperature if sp.temperature is not None else 1.0
            top_p = sp.top_p if sp.top_p is not None else 1.0
            top_k = sp.top_k if sp.top_k is not None else -1
            if request.eos_token_id is not None:
                stop_token_ids = [request.eos_token_id]

        cpu_req = CPUInferenceRequest(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_token_ids=stop_token_ids,
        )

        self.request_queue.put(cpu_req)
        self.pending_requests.add(request.request_id)
        logger.debug(f"Request {request.request_id} submitted to CPU "
                     f"(prompt_len={len(request.prompt_token_ids)}, "
                     f"max_tokens={max_tokens})")

    def collect_results(self) -> list[EngineCoreOutput]:
        """CPU 추론 결과 수집.

        Non-blocking: 결과 큐에서 사용 가능한 결과만 수집.
        """
        outputs: list[EngineCoreOutput] = []

        while True:
            try:
                resp: CPUInferenceResponse = self.result_queue.get_nowait()
            except queue.Empty:
                break

            # EngineCoreOutput 생성
            finish_reason = None
            if resp.finished:
                if resp.finish_reason == "stop":
                    finish_reason = FinishReason.STOP
                elif resp.finish_reason == "length":
                    finish_reason = FinishReason.LENGTH
                else:
                    finish_reason = FinishReason.STOP

                # 완료된 요청 추적에서 제거
                self.pending_requests.discard(resp.request_id)

            output = EngineCoreOutput(
                request_id=resp.request_id,
                new_token_ids=resp.generated_token_ids,
                finish_reason=finish_reason,
            )
            outputs.append(output)

        return outputs

    def abort_request(self, request_id: str):
        """CPU 요청 취소. (현재 단순히 무시)"""
        self.pending_requests.discard(request_id)

    def has_pending_requests(self) -> bool:
        return len(self.pending_requests) > 0

    def is_alive(self) -> bool:
        if self.process is None:
            return False
        if not self.process.is_alive():
            self.alive = False
            return False
        return True

    def shutdown(self):
        """CPU 프로세스 종료."""
        if self.process is not None and self.process.is_alive():
            try:
                self.request_queue.put(_SHUTDOWN_SENTINEL)
                self.process.join(timeout=10)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=5)
            except Exception as e:
                logger.warning(f"Error shutting down CPU process: {e}")
        self.alive = False
        logger.info("CPU inference process shut down")

    @staticmethod
    def _cpu_worker_loop(
        model_name: str,
        trust_remote_code: bool,
        num_threads: int,
        dtype_str: str,
        numa_aware: bool,
        numa_bind_node: Optional[int],
        max_concurrent: int,
        request_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
    ):
        """CPU 추론 워커 루프 (별도 프로세스에서 실행).

        이 함수는 새로운 프로세스에서 실행되므로 모든 import를 내부에서 수행.
        """
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        from vllm.logger import init_logger as _init_logger
        _logger = _init_logger("vllm.cpu_inference")

        _logger.info(
            f"CPU worker starting: model={model_name}, "
            f"threads={num_threads}, dtype={dtype_str}"
        )

        # 1. Intel CPU 환경 설정
        try:
            from vllm.platforms.intel_cpu_utils import (
                setup_intel_cpu_environment,
                is_ipex_available,
            )
            env_config = setup_intel_cpu_environment(
                rank=0,
                world_size=1,
                enable_numa=numa_aware,
                enable_avx_optimization=True,
                enable_ipex=True,
            )
            _logger.info(f"Intel CPU environment configured: {env_config}")
        except Exception as e:
            _logger.warning(f"Intel CPU setup failed (non-fatal): {e}")

        # 2. 스레드 설정
        torch.set_num_threads(max(1, num_threads))
        os.environ["OMP_NUM_THREADS"] = str(max(1, num_threads))
        _logger.info(f"Threads set to {torch.get_num_threads()}")

        # 3. 모델 로드
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if dtype_str == "bfloat16":
                dtype = torch.bfloat16
            elif dtype_str == "float16":
                dtype = torch.float16
            else:
                dtype = torch.float32

            _logger.info(f"Loading model {model_name} (dtype={dtype})...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
            model = model.to("cpu")
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )

            # 모델 파라미터 크기 로깅
            total_params = sum(p.numel() for p in model.parameters())
            param_gb = total_params * (2 if dtype == torch.bfloat16 else 4) / (1024**3)
            _logger.info(
                f"Model loaded: {total_params/1e9:.1f}B params, "
                f"~{param_gb:.1f}GB memory"
            )

            # 4. IPEX 최적화
            try:
                if is_ipex_available():
                    import intel_extension_for_pytorch as ipex
                    model = ipex.optimize(
                        model,
                        dtype=dtype,
                        auto_kernel_selection=True,
                    )
                    _logger.info("IPEX optimization applied")
            except Exception as e:
                _logger.warning(f"IPEX optimization failed: {e}")

        except Exception as e:
            _logger.error(f"Failed to load model: {e}")
            return

        # 5. 메인 추론 루프
        _logger.info("CPU inference worker ready, entering main loop")

        while True:
            try:
                # 요청 대기 (blocking with timeout)
                try:
                    req = request_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # 종료 신호
                if req == _SHUTDOWN_SENTINEL:
                    _logger.info("Shutdown signal received")
                    break

                if not isinstance(req, CPUInferenceRequest):
                    _logger.warning(f"Invalid request type: {type(req)}")
                    continue

                # 추론 실행
                _logger.debug(
                    f"Processing request {req.request_id} "
                    f"(prompt_len={len(req.prompt_token_ids)}, "
                    f"max_tokens={req.max_tokens})"
                )

                start_time = time.perf_counter()
                input_ids = torch.tensor(
                    [req.prompt_token_ids], dtype=torch.long, device="cpu"
                )

                all_generated_ids: list[int] = []
                pending_ids: list[int] = []
                finished = False
                finish_reason = None

                use_autocast = dtype_str == "bfloat16"

                with torch.no_grad():
                    for step in range(req.max_tokens):
                        with torch.amp.autocast(
                            "cpu", enabled=use_autocast
                        ):
                            outputs = model(input_ids)

                        # Greedy 또는 temperature sampling
                        logits = outputs.logits[:, -1, :]

                        if req.temperature <= 0 or req.temperature < 1e-6:
                            next_token = logits.argmax(dim=-1).item()
                        else:
                            logits = logits / req.temperature
                            if req.top_k > 0:
                                top_k_logits, top_k_indices = torch.topk(
                                    logits, min(req.top_k, logits.size(-1))
                                )
                                logits = torch.full_like(
                                    logits, float('-inf')
                                )
                                logits.scatter_(1, top_k_indices, top_k_logits)

                            probs = torch.softmax(logits, dim=-1)
                            next_token = torch.multinomial(
                                probs, num_samples=1
                            ).item()

                        all_generated_ids.append(next_token)
                        pending_ids.append(next_token)

                        # 종료 조건 확인
                        if next_token in req.stop_token_ids:
                            finished = True
                            finish_reason = "stop"
                            break

                        # 다음 스텝 입력 생성
                        next_token_tensor = torch.tensor(
                            [[next_token]], dtype=torch.long, device="cpu"
                        )
                        input_ids = torch.cat(
                            [input_ids, next_token_tensor], dim=1
                        )

                        # 중간 결과 스트리밍 (10토큰마다)
                        if (step + 1) % 10 == 0 and pending_ids:
                            partial_resp = CPUInferenceResponse(
                                request_id=req.request_id,
                                generated_token_ids=list(pending_ids),
                                finished=False,
                                finish_reason=None,
                            )
                            result_queue.put(partial_resp)
                            pending_ids = []

                # max_tokens 도달
                if not finished:
                    finished = True
                    finish_reason = "length"

                # 최종 결과 전송 (남은 토큰 + 완료 신호)
                resp = CPUInferenceResponse(
                    request_id=req.request_id,
                    generated_token_ids=pending_ids,
                    finished=True,
                    finish_reason=finish_reason,
                )
                result_queue.put(resp)

                elapsed = time.perf_counter() - start_time
                total_generated = len(all_generated_ids)
                tok_per_sec = (
                    total_generated / elapsed if elapsed > 0 else 0
                )
                _logger.info(
                    f"Request {req.request_id} completed: "
                    f"generated {total_generated} tokens in {elapsed:.2f}s "
                    f"({tok_per_sec:.1f} tok/s)"
                )

            except Exception as e:
                _logger.error(f"Error in CPU worker loop: {e}")
                import traceback
                traceback.print_exc()

        _logger.info("CPU inference worker exiting")


# ============================================================================
# Hybrid Engine Core
# ============================================================================

class HybridEngineCore:
    """GPU + CPU 듀얼 추론 엔진.

    GPU 경로: 기존 v1 EngineCore (변경 없음)
    CPU 경로: CPUInferenceProcess (별도 프로세스)
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Optional[Callable] = None,
    ):
        self.vllm_config = vllm_config
        hybrid_config = vllm_config.hybrid_config

        logger.info("Initializing HybridEngineCore (parallel-batch mode)")

        # === GPU 경로: 기존 EngineCore 그대로 ===
        self.gpu_engine = EngineCore(
            vllm_config, executor_class, log_stats,
            executor_fail_callback,
        )

        # === CPU 경로 ===
        cpu_ratio = hybrid_config.cpu_ratio
        if cpu_ratio is None:
            # 자동 결정: 보수적으로 5%
            cpu_ratio = 0.05
            logger.info(f"Auto cpu_ratio: {cpu_ratio:.2%}")

        self.router = RequestRouter(cpu_ratio)
        self.request_to_path: dict[str, str] = {}
        # EngineCoreRequest를 보관 (CPU 경로용)
        self.cpu_pending_requests: dict[str, EngineCoreRequest] = {}

        # CPU 추론 프로세스 시작
        self.cpu_process: Optional[CPUInferenceProcess] = None
        self.cpu_available = False
        try:
            self.cpu_process = CPUInferenceProcess(vllm_config)
            self.cpu_process.start()
            self.cpu_available = True
            logger.info("CPU inference process started successfully")
        except Exception as e:
            logger.warning(
                f"CPU inference process failed to start: {e}. "
                f"Falling back to GPU-only mode."
            )
            self.router.cpu_ratio = 0.0

        logger.info(
            f"HybridEngineCore initialized: "
            f"cpu_available={self.cpu_available}, "
            f"cpu_ratio={self.router.cpu_ratio:.2%}"
        )

    # ------------------------------------------------------------------
    # EngineCore 호환 인터페이스
    # ------------------------------------------------------------------

    def get_supported_tasks(self, *args, **kwargs):
        return self.gpu_engine.get_supported_tasks(*args, **kwargs)

    @property
    def model_executor(self):
        return self.gpu_engine.model_executor

    @property
    def scheduler(self):
        return self.gpu_engine.scheduler

    @property
    def mm_input_cache_server(self):
        return self.gpu_engine.mm_input_cache_server

    @property
    def structured_output_manager(self):
        return self.gpu_engine.structured_output_manager

    @property
    def batch_queue_size(self):
        return self.gpu_engine.batch_queue_size

    @property
    def batch_queue(self):
        return self.gpu_engine.batch_queue

    @property
    def log_stats(self):
        return self.gpu_engine.log_stats

    @property
    def available_gpu_memory_for_kv_cache(self):
        return self.gpu_engine.available_gpu_memory_for_kv_cache

    def preprocess_add_request(self, request: EngineCoreRequest):
        """요청 전처리 (InprocClient에서 호출)."""
        # CPU 경로 요청인지 확인
        path = self.router.route(request.request_id)
        self.request_to_path[request.request_id] = path

        if path == "cpu" and self.cpu_available:
            # CPU 요청은 전처리 없이 직접 CPU 프로세스로 전송
            self.cpu_pending_requests[request.request_id] = request
            logger.debug(
                f"Request {request.request_id} routed to CPU "
                f"(prompt_len={len(request.prompt_token_ids)})"
            )
            # GPU 엔진의 preprocess 형식에 맞춰 dummy 반환
            # (실제 GPU scheduler에는 추가하지 않음)
            return None, request.current_wave
        else:
            # GPU 경로: 정상 전처리
            if path == "cpu":
                # CPU 불가 → GPU로 fallback
                self.request_to_path[request.request_id] = "gpu"
            return self.gpu_engine.preprocess_add_request(request)

    def add_request(self, request: Request, request_wave: int = 0):
        """스케줄러에 요청 추가 (GPU 경로만)."""
        # CPU 요청은 preprocess_add_request에서 이미 처리됨
        # 이 메서드는 GPU 요청에 대해서만 호출됨
        self.gpu_engine.add_request(request, request_wave)

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """GPU + CPU 실행 후 결과 병합."""
        # 1. CPU 요청 제출 (아직 보내지 않은 것들)
        self._submit_pending_cpu_requests()

        # 2. GPU step
        gpu_outputs: dict[int, EngineCoreOutputs] = {}
        gpu_executed = False
        if self.gpu_engine.scheduler.has_requests():
            gpu_outputs, gpu_executed = self.gpu_engine.step()

        # 3. CPU 결과 수집
        cpu_outputs: list[EngineCoreOutput] = []
        if self.cpu_available and self.cpu_process is not None:
            if not self.cpu_process.is_alive():
                logger.error("CPU process died! Disabling CPU path.")
                self.cpu_available = False
                self.router.cpu_ratio = 0.0
            else:
                cpu_outputs = self.cpu_process.collect_results()

        # 4. 결과 병합
        if cpu_outputs:
            gpu_outputs = self._merge_cpu_outputs(gpu_outputs, cpu_outputs)

        return gpu_outputs, gpu_executed or bool(cpu_outputs)

    def step_with_batch_queue(self):
        """배치 큐 모드의 step (GPU의 batch_queue + CPU 결과)."""
        # GPU step_with_batch_queue
        gpu_result = self.gpu_engine.step_with_batch_queue()

        # CPU 결과 수집 및 병합
        self._submit_pending_cpu_requests()

        if self.cpu_available and self.cpu_process is not None:
            if self.cpu_process.is_alive():
                cpu_outputs = self.cpu_process.collect_results()
                if cpu_outputs and gpu_result[0] is not None:
                    gpu_result = (
                        self._merge_cpu_outputs(gpu_result[0], cpu_outputs),
                        gpu_result[1],
                    )

        return gpu_result

    def _submit_pending_cpu_requests(self):
        """보류 중인 CPU 요청을 프로세스에 제출."""
        if not self.cpu_available or self.cpu_process is None:
            return

        submitted = []
        for req_id, req in self.cpu_pending_requests.items():
            self.cpu_process.submit_request(req)
            submitted.append(req_id)

        for req_id in submitted:
            del self.cpu_pending_requests[req_id]

    def _merge_cpu_outputs(
        self,
        gpu_outputs: dict[int, EngineCoreOutputs],
        cpu_outputs: list[EngineCoreOutput],
    ) -> dict[int, EngineCoreOutputs]:
        """CPU 결과를 GPU 결과에 병합."""
        if not cpu_outputs:
            return gpu_outputs

        # client_index 0에 CPU 결과 추가
        if 0 in gpu_outputs:
            gpu_outputs[0].outputs.extend(cpu_outputs)
        else:
            gpu_outputs[0] = EngineCoreOutputs(
                outputs=cpu_outputs,
            )

        # 완료된 CPU 요청 정리
        for output in cpu_outputs:
            if output.finished:
                self.request_to_path.pop(output.request_id, None)

        logger.debug(
            f"Merged {len(cpu_outputs)} CPU outputs "
            f"({sum(1 for o in cpu_outputs if o.finished)} finished)"
        )
        return gpu_outputs

    def abort_requests(self, request_ids: list[str]):
        """요청 취소."""
        gpu_aborts = []
        for req_id in request_ids:
            path = self.request_to_path.pop(req_id, "gpu")
            if path == "cpu":
                if self.cpu_process is not None:
                    self.cpu_process.abort_request(req_id)
                self.cpu_pending_requests.pop(req_id, None)
            else:
                gpu_aborts.append(req_id)

        if gpu_aborts:
            self.gpu_engine.abort_requests(gpu_aborts)

    def shutdown(self):
        """엔진 종료."""
        logger.info(
            f"HybridEngineCore shutting down. "
            f"Router stats: {self.router.get_stats()}"
        )
        if self.cpu_process is not None:
            self.cpu_process.shutdown()
        self.gpu_engine.shutdown()

    def execute_model_with_error_logging(self, *args, **kwargs):
        return self.gpu_engine.execute_model_with_error_logging(*args, **kwargs)

    def profile(self, is_start: bool = True):
        self.gpu_engine.profile(is_start)

    def reset_mm_cache(self):
        self.gpu_engine.reset_mm_cache()

    def reset_prefix_cache(self):
        self.gpu_engine.reset_prefix_cache()

    def sleep(self, level: int = 1):
        self.gpu_engine.sleep(level)

    def wake_up(self, tags=None):
        self.gpu_engine.wake_up(tags)

    def is_sleeping(self):
        return self.gpu_engine.is_sleeping()

    def execute_dummy_batch(self):
        self.gpu_engine.execute_dummy_batch()

    def add_lora(self, lora_request):
        return self.gpu_engine.add_lora(lora_request)

    def remove_lora(self, lora_id):
        return self.gpu_engine.remove_lora(lora_id)

    def list_loras(self):
        return self.gpu_engine.list_loras()

    def pin_lora(self, lora_id):
        return self.gpu_engine.pin_lora(lora_id)

    def save_sharded_state(self, *args, **kwargs):
        return self.gpu_engine.save_sharded_state(*args, **kwargs)

    def collective_rpc(self, *args, **kwargs):
        return self.gpu_engine.collective_rpc(*args, **kwargs)

    def __getattr__(self, name):
        """알 수 없는 속성은 GPU 엔진으로 위임."""
        return getattr(self.gpu_engine, name)
