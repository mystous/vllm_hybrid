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
from logging import DEBUG
from typing import Any, Callable, Optional

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.engine import (EngineCoreOutput, EngineCoreOutputs,
                             EngineCoreRequest, EngineCoreRequestType,
                             FinishReason)

logger = init_logger(__name__)


# ============================================================================
# Auto CPU Ratio Profiling
# ============================================================================

def estimate_cpu_throughput(
    model_name: str,
    trust_remote_code: bool,
    num_threads: int,
    dtype_str: str,
    num_warmup: int = 2,
    num_measure: int = 5,
    prompt_len: int = 32,
    gen_tokens: int = 10,
) -> float:
    """CPU 추론 처리량을 측정하여 tok/s 반환.

    더미 입력으로 짧은 추론을 실행해 CPU 속도를 측정합니다.
    GPU 속도는 vLLM 벤치마크 기반 휴리스틱을 사용합니다.

    Returns:
        추정된 최적 cpu_ratio (0.0 ~ 1.0)
    """
    import torch as _torch

    _torch.set_num_threads(max(1, num_threads))

    try:
        from transformers import AutoModelForCausalLM

        if dtype_str == "bfloat16":
            dtype = _torch.bfloat16
        elif dtype_str == "float16":
            dtype = _torch.float16
        else:
            dtype = _torch.float32

        logger.info(
            f"Profiling CPU throughput: model={model_name}, "
            f"threads={num_threads}, dtype={dtype_str}"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        model = model.to("cpu")
        model.eval()

        # 더미 입력 생성
        dummy_input = _torch.randint(
            100, 10000, (1, prompt_len), dtype=_torch.long, device="cpu"
        )

        use_autocast = dtype_str == "bfloat16"

        # Warmup
        with _torch.no_grad():
            for _ in range(num_warmup):
                ids = dummy_input.clone()
                for _ in range(gen_tokens):
                    with _torch.amp.autocast("cpu", enabled=use_autocast):
                        out = model(ids)
                    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    ids = _torch.cat([ids, next_tok], dim=1)

        # 측정
        import time as _time
        total_tokens = 0
        start = _time.perf_counter()

        with _torch.no_grad():
            for _ in range(num_measure):
                ids = dummy_input.clone()
                for _ in range(gen_tokens):
                    with _torch.amp.autocast("cpu", enabled=use_autocast):
                        out = model(ids)
                    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    ids = _torch.cat([ids, next_tok], dim=1)
                    total_tokens += 1

        elapsed = _time.perf_counter() - start
        cpu_tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0

        del model
        if hasattr(_torch, 'cuda'):
            pass  # CPU only, no cuda cleanup needed

        logger.info(
            f"CPU profiling result: {cpu_tok_per_sec:.1f} tok/s "
            f"({total_tokens} tokens in {elapsed:.2f}s)"
        )
        return cpu_tok_per_sec

    except Exception as e:
        logger.warning(f"CPU profiling failed: {e}")
        return 0.0


def compute_auto_cpu_ratio(
    cpu_tok_per_sec: float,
    gpu_tok_per_sec: float = 100.0,
) -> float:
    """CPU/GPU 처리량 기반 최적 cpu_ratio 계산.

    R_cpu = T_cpu / (T_gpu + T_cpu)

    Args:
        cpu_tok_per_sec: CPU 처리량 (tok/s)
        gpu_tok_per_sec: GPU 처리량 (tok/s), 기본 100 tok/s 휴리스틱

    Returns:
        cpu_ratio (0.01 ~ 0.5 범위로 클램핑)
    """
    if cpu_tok_per_sec <= 0:
        return 0.0

    total = gpu_tok_per_sec + cpu_tok_per_sec
    ratio = cpu_tok_per_sec / total

    # 최소 1%, 최대 50%로 클램핑
    ratio = max(0.01, min(0.5, ratio))

    logger.info(
        f"Auto cpu_ratio: {ratio:.2%} "
        f"(cpu={cpu_tok_per_sec:.1f}, gpu={gpu_tok_per_sec:.1f} tok/s)"
    )
    return ratio


# ============================================================================
# Utility: CPU Memory Check
# ============================================================================

def _check_cpu_memory_available(model_name: str, dtype_str: str) -> bool:
    """CPU 추론에 충분한 시스템 메모리가 있는지 확인."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        available_gb = int(line.split()[1]) / (1024 ** 2)
                        break
                else:
                    return True
        except Exception:
            return True

    bytes_per_param = 2 if dtype_str == "bfloat16" else 4
    estimated_gb = 0.0
    model_lower = model_name.lower()
    for marker in ["70b", "65b", "72b"]:
        if marker in model_lower:
            estimated_gb = 70 * bytes_per_param
            break
    for marker in ["13b", "14b"]:
        if marker in model_lower:
            estimated_gb = 14 * bytes_per_param
            break
    for marker in ["7b", "8b"]:
        if marker in model_lower:
            estimated_gb = 8 * bytes_per_param
            break
    for marker in ["3b", "2b", "1b", "1.5b"]:
        if marker in model_lower:
            estimated_gb = 3 * bytes_per_param
            break

    if estimated_gb == 0:
        estimated_gb = 16

    required_gb = estimated_gb + 4

    logger.info(
        f"CPU memory check: available={available_gb:.1f}GB, "
        f"estimated_model={estimated_gb:.1f}GB, "
        f"required={required_gb:.1f}GB"
    )

    if available_gb < required_gb:
        logger.warning(
            f"Insufficient memory for CPU inference: "
            f"available={available_gb:.1f}GB < "
            f"required={required_gb:.1f}GB"
        )
        return False
    return True


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
        self.abort_queue: multiprocessing.Queue = multiprocessing.Queue()
        self.process: Optional[multiprocessing.Process] = None
        self.alive = False
        self._shutdown_event: Optional[multiprocessing.Event] = None

        # 추적 중인 CPU 요청
        self.pending_requests: set[str] = set()

    def start(self):
        """CPU 추론 프로세스 시작."""
        ctx = multiprocessing.get_context("spawn")
        self.request_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.abort_queue = ctx.Queue()
        self._shutdown_event = ctx.Event()
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
                self.abort_queue,
                self._shutdown_event,
            ),
            daemon=False,
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
                if resp.finish_reason == "abort":
                    # abort된 요청 — 추적에서 제거만, 출력 안 함
                    self.pending_requests.discard(resp.request_id)
                    continue
                elif resp.finish_reason == "stop":
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
        """CPU 요청 취소 (워커에 abort 신호 전달)."""
        self.pending_requests.discard(request_id)
        try:
            self.abort_queue.put_nowait(request_id)
        except Exception:
            pass

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
        """CPU 프로세스 종료.

        종료 순서:
        1. shutdown_event 설정 (워커 루프가 즉시 감지)
        2. sentinel 전송 (Queue.get() 블로킹 해제용)
        3. result_queue drain (multiprocessing Queue 데드락 방지)
        4. process.join() 시도
        5. 실패 시 terminate + kill
        6. Queue 리소스 정리
        """
        if self.process is None or not self.process.is_alive():
            self.alive = False
            return

        try:
            # Step 1: Event 시그널 (가장 빠른 종료 경로)
            if self._shutdown_event is not None:
                self._shutdown_event.set()

            # Step 2: Sentinel 전송 (Queue.get 블로킹 해제)
            try:
                self.request_queue.put_nowait(_SHUTDOWN_SENTINEL)
            except Exception:
                pass  # Queue가 full이어도 event로 종료 가능

            # Step 3: result_queue drain (데드락 방지 -
            # multiprocessing Queue는 내부 버퍼 스레드가 있어서
            # drain하지 않으면 자식 프로세스가 join 불가)
            self._drain_queue(self.result_queue)
            self._drain_queue(self.abort_queue)

            # Step 4: 정상 종료 대기
            self.process.join(timeout=10)

            # Step 5: 강제 종료
            if self.process.is_alive():
                logger.warning(
                    "CPU process did not exit gracefully, terminating..."
                )
                self.process.terminate()
                # terminate 후에도 Queue drain 필요
                self._drain_queue(self.result_queue)
                self.process.join(timeout=5)

            if self.process.is_alive():
                logger.warning("CPU process still alive after terminate, "
                               "sending SIGKILL")
                self.process.kill()
                self.process.join(timeout=3)

        except Exception as e:
            logger.warning(f"Error shutting down CPU process: {e}")
        finally:
            self.alive = False
            # Step 6: Queue 리소스 정리
            self._cleanup_queues()
            logger.info("CPU inference process shut down")

    @staticmethod
    def _drain_queue(q: multiprocessing.Queue):
        """Queue의 모든 아이템을 읽어서 비움 (데드락 방지).

        Note: Queue.empty()는 race condition으로 신뢰할 수 없으므로
        get_nowait() + Empty 예외 방식으로 drain합니다.
        """
        try:
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
                except Exception:
                    break
        except Exception:
            pass

    def _cleanup_queues(self):
        """Queue 리소스 안전하게 정리.

        cancel_join_thread()를 먼저 호출하여 close() 후
        feeder 스레드가 hang하는 것을 방지합니다.
        """
        for q in [self.request_queue, self.result_queue, self.abort_queue]:
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass

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
        abort_queue: Optional[multiprocessing.Queue] = None,
        shutdown_event: Optional[multiprocessing.Event] = None,
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
            from transformers import AutoModelForCausalLM

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
            # 부모 프로세스에 에러 통보 (shutdown_event 설정)
            # 부모가 is_alive()=False로 감지하여 GPU-only 폴백
            if shutdown_event is not None:
                shutdown_event.set()
            return

        # 5. abort 추적용 set
        aborted_ids: set[str] = set()

        def _drain_abort_queue():
            """abort_queue에서 abort 신호를 모두 읽어 aborted_ids에 추가."""
            if abort_queue is None:
                return
            while True:
                try:
                    rid = abort_queue.get_nowait()
                    aborted_ids.add(rid)
                except queue.Empty:
                    break

        # 6. 메인 추론 루프
        _logger.info("CPU inference worker ready, entering main loop")

        _parent_pid = os.getppid()

        def _should_shutdown():
            """shutdown_event, sentinel, 또는 부모 사망 기반 종료 확인."""
            if (shutdown_event is not None
                    and shutdown_event.is_set()):
                return True
            # 부모 프로세스가 죽으면 고아가 되므로 자동 종료
            # (ppid가 1(init) 또는 변경되면 부모 사망)
            if os.getppid() != _parent_pid:
                _logger.warning("Parent process died, shutting down")
                return True
            return False

        while not _should_shutdown():
            try:
                # abort 신호 확인
                _drain_abort_queue()

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

                # 이미 abort된 요청은 즉시 스킵
                if req.request_id in aborted_ids:
                    aborted_ids.discard(req.request_id)
                    _logger.debug(
                        f"Request {req.request_id} already aborted, "
                        f"skipping"
                    )
                    resp = CPUInferenceResponse(
                        request_id=req.request_id,
                        generated_token_ids=[],
                        finished=True,
                        finish_reason="abort",
                    )
                    result_queue.put(resp)
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
                aborted = False

                use_autocast = dtype_str == "bfloat16"

                with torch.no_grad():
                    for step in range(req.max_tokens):
                        # 매 10스텝마다 abort/shutdown 확인
                        if step % 10 == 0:
                            if _should_shutdown():
                                _logger.info(
                                    f"Shutdown during request "
                                    f"{req.request_id} at step {step}"
                                )
                                aborted = True
                                break
                            _drain_abort_queue()
                            if req.request_id in aborted_ids:
                                aborted_ids.discard(req.request_id)
                                aborted = True
                                _logger.debug(
                                    f"Request {req.request_id} aborted "
                                    f"at step {step}"
                                )
                                break

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

                # abort된 요청 처리
                if aborted:
                    resp = CPUInferenceResponse(
                        request_id=req.request_id,
                        generated_token_ids=pending_ids,
                        finished=True,
                        finish_reason="abort",
                    )
                    result_queue.put(resp)
                    elapsed = time.perf_counter() - start_time
                    _logger.info(
                        f"Request {req.request_id} aborted after "
                        f"{len(all_generated_ids)} tokens in {elapsed:.2f}s"
                    )
                    continue

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
# Hybrid CPU Init (shared logic for HybridEngineCore / HybridEngineCoreProc)
# ============================================================================

def _init_hybrid_cpu_process(
    vllm_config: VllmConfig,
) -> tuple[bool, Optional[RequestRouter], Optional[CPUInferenceProcess]]:
    """Initialize CPU inference process for hybrid mode.

    Returns:
        (enabled, router, cpu_process)
    """
    try:
        hybrid_config = vllm_config.hybrid_config
        model_name = vllm_config.model_config.model
        dtype_str = hybrid_config.cpu_dtype

        if not _check_cpu_memory_available(model_name, dtype_str):
            logger.warning(
                "Insufficient memory for CPU inference. "
                "Falling back to GPU-only mode."
            )
            return False, None, None

        cpu_ratio = hybrid_config.cpu_ratio
        if cpu_ratio is None:
            try:
                cpu_tps = estimate_cpu_throughput(
                    model_name=model_name,
                    trust_remote_code=(
                        vllm_config.model_config.trust_remote_code),
                    num_threads=hybrid_config.cpu_num_threads,
                    dtype_str=dtype_str,
                )
                if cpu_tps > 0:
                    cpu_ratio = compute_auto_cpu_ratio(cpu_tps)
                else:
                    cpu_ratio = 0.05
                    logger.info(
                        "CPU profiling returned 0, "
                        f"using default cpu_ratio={cpu_ratio:.2%}")
            except Exception as e:
                cpu_ratio = 0.05
                logger.warning(
                    f"Auto cpu_ratio profiling failed: {e}. "
                    f"Using default {cpu_ratio:.2%}"
                )

        # cpu_ratio가 0이면 CPU 프로세스를 시작하지 않음
        # (불필요한 모델 로드로 수십 GB 메모리 낭비 방지)
        if cpu_ratio <= 0:
            logger.info(
                "Hybrid mode: cpu_ratio=0, CPU inference disabled"
            )
            return False, None, None

        router = RequestRouter(cpu_ratio)
        cpu_process = CPUInferenceProcess(vllm_config)
        cpu_process.start()

        logger.info(
            "Hybrid parallel-batch mode enabled: "
            f"cpu_ratio={cpu_ratio:.2%}, "
            f"cpu_pid={cpu_process.process.pid}"
        )
        return True, router, cpu_process

    except Exception as e:
        logger.warning(
            f"Hybrid CPU initialization failed: {e}. "
            f"Falling back to GPU-only mode."
        )
        return False, None, None


# ============================================================================
# Hybrid Mixin: shared methods for HybridEngineCore / HybridEngineCoreProc
# ============================================================================

class _HybridMixin:
    """Shared hybrid CPU logic used by both HybridEngineCore and
    HybridEngineCoreProc."""

    _hybrid_enabled: bool
    _hybrid_cpu_process: Optional[CPUInferenceProcess]
    _hybrid_router: Optional[RequestRouter]
    _hybrid_request_to_path: dict[str, str]

    def _check_hybrid_cpu_health(self):
        """CPU 프로세스 생존 여부 확인, 죽었으면 GPU-only로 자동 전환.

        사망한 CPU 프로세스의 pending 요청에 대해 abort 응답을 생성하여
        클라이언트가 무한 대기하지 않도록 합니다.
        """
        if not self._hybrid_enabled or self._hybrid_cpu_process is None:
            return

        if not self._hybrid_cpu_process.is_alive():
            logger.error(
                "CPU inference process died unexpectedly! "
                "Switching to GPU-only mode."
            )
            pending = list(self._hybrid_cpu_process.pending_requests)
            # abort 응답 생성하여 _hybrid_dead_outputs에 저장
            # → 다음 _collect_hybrid_cpu_outputs()에서 클라이언트에 전달
            if pending:
                abort_outputs = []
                for req_id in pending:
                    self._hybrid_request_to_path.pop(req_id, None)
                    abort_outputs.append(EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=[],
                        finish_reason=FinishReason.ABORT,
                    ))
                self._hybrid_dead_outputs = abort_outputs
                logger.warning(
                    f"Aborted {len(pending)} pending CPU requests: "
                    f"{pending[:5]}{'...' if len(pending) > 5 else ''}"
                )

            self._hybrid_cpu_process = None
            self._hybrid_enabled = False
            self._hybrid_router = None

    def _collect_hybrid_cpu_outputs(
        self,
        outputs: Optional[dict[int, EngineCoreOutputs]],
    ) -> dict[int, EngineCoreOutputs]:
        """Collect CPU inference results and merge into outputs."""
        # 먼저 dead outputs 처리 (CPU 사망 시 abort 응답)
        dead_outputs = getattr(self, '_hybrid_dead_outputs', None)
        if dead_outputs:
            self._hybrid_dead_outputs = None
            if outputs is None:
                outputs = {}
            if 0 in outputs:
                outputs[0].outputs.extend(dead_outputs)
            else:
                outputs[0] = EngineCoreOutputs(outputs=dead_outputs)

        if not self._hybrid_enabled or self._hybrid_cpu_process is None:
            return outputs or {}

        cpu_results = self._hybrid_cpu_process.collect_results()
        if not cpu_results:
            return outputs or {}

        if outputs is None:
            outputs = {}

        # Merge CPU results into engine_index=0 outputs.
        if 0 in outputs:
            outputs[0].outputs.extend(cpu_results)
        else:
            outputs[0] = EngineCoreOutputs(outputs=cpu_results)

        # Clean up finished CPU requests from tracking.
        for result in cpu_results:
            if result.finish_reason is not None:
                self._hybrid_request_to_path.pop(result.request_id, None)

        return outputs

    def _has_hybrid_cpu_pending(self) -> bool:
        """Check if there are pending CPU requests."""
        return (self._hybrid_enabled
                and self._hybrid_cpu_process is not None
                and self._hybrid_cpu_process.has_pending_requests())

    def _route_request_to_cpu(
        self, request: EngineCoreRequest,
    ) -> bool:
        """Route a request to CPU if applicable.

        Returns True if routed to CPU, False if should go to GPU.
        """
        if not self._hybrid_enabled or self._hybrid_cpu_process is None:
            return False

        path = self._hybrid_router.route(request.request_id)
        if path == "cpu":
            self._hybrid_cpu_process.submit_request(request)
            self._hybrid_request_to_path[request.request_id] = "cpu"
            return True

        self._hybrid_request_to_path[request.request_id] = "gpu"
        return False

    def _abort_hybrid_cpu_requests(self, request_ids: list[str]):
        """Abort CPU requests in hybrid mode."""
        if not self._hybrid_enabled or self._hybrid_cpu_process is None:
            return
        for rid in request_ids:
            if self._hybrid_request_to_path.get(rid) == "cpu":
                self._hybrid_cpu_process.abort_request(rid)
                self._hybrid_request_to_path.pop(rid, None)

    def _shutdown_hybrid_cpu(self):
        """Shutdown the CPU inference process."""
        if self._hybrid_cpu_process is not None:
            self._hybrid_cpu_process.shutdown()
            self._hybrid_cpu_process = None
            self._hybrid_enabled = False


# ============================================================================
# HybridEngineCore: Composition wrapper for InprocClient
# ============================================================================

class HybridEngineCore(_HybridMixin):
    """GPU + CPU 듀얼 추론 엔진 (InprocClient 전용 composition wrapper).

    GPU 경로는 기존 EngineCore를 100% 위임.
    CPU 경로는 별도 CPUInferenceProcess로 처리.
    """

    def __init__(self,
                 vllm_config: VllmConfig,
                 executor_class: Any,
                 log_stats: bool,
                 executor_fail_callback: Optional[Callable] = None):
        from vllm.v1.engine.core import EngineCore

        # GPU 경로: 기존 EngineCore 그대로 생성
        self.gpu_engine = EngineCore(
            vllm_config, executor_class, log_stats, executor_fail_callback)

        # Hybrid CPU 경로 초기화
        self._hybrid_enabled = False
        self._hybrid_cpu_process = None
        self._hybrid_router = None
        self._hybrid_request_to_path: dict[str, str] = {}
        self._hybrid_dead_outputs: Optional[list] = None

        if (hasattr(vllm_config, 'hybrid_config')
                and vllm_config.hybrid_config is not None
                and vllm_config.hybrid_config.is_enabled()
                and vllm_config.hybrid_config.mode == "parallel-batch"):
            enabled, router, cpu_proc = _init_hybrid_cpu_process(vllm_config)
            self._hybrid_enabled = enabled
            self._hybrid_router = router
            self._hybrid_cpu_process = cpu_proc

    # -- Hybrid-aware methods --

    def preprocess_add_request(
            self,
            request: EngineCoreRequest,
    ) -> tuple[Optional[Any], int]:
        """Preprocess the request, routing to CPU if applicable."""
        if self._route_request_to_cpu(request):
            return None, request.current_wave
        return self.gpu_engine.preprocess_add_request(request)

    def add_request(self, request: Any, request_wave: int = 0):
        self.gpu_engine.add_request(request, request_wave)

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        self._check_hybrid_cpu_health()

        has_gpu = self.gpu_engine.scheduler.has_requests()
        has_cpu = self._has_hybrid_cpu_pending()

        if not has_gpu and not has_cpu:
            return {}, False

        engine_core_outputs: dict[int, EngineCoreOutputs] = {}
        model_executed = False

        if has_gpu:
            engine_core_outputs, model_executed = self.gpu_engine.step()

        # Merge CPU results.
        engine_core_outputs = self._collect_hybrid_cpu_outputs(
            engine_core_outputs)

        return engine_core_outputs, model_executed

    def abort_requests(self, request_ids: list[str]):
        self._abort_hybrid_cpu_requests(request_ids)
        self.gpu_engine.abort_requests(request_ids)

    def shutdown(self):
        self._shutdown_hybrid_cpu()
        self.gpu_engine.shutdown()

    # -- Delegated methods (pass through to gpu_engine) --

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the inner GPU EngineCore."""
        # __getattr__ is only called when normal attribute lookup fails,
        # so self.gpu_engine won't cause recursion.
        return getattr(self.gpu_engine, name)


# ============================================================================
# HybridEngineCoreProc: Inheritance wrapper for process mode
# ============================================================================

class HybridEngineCoreProc(_HybridMixin):
    """GPU + CPU 듀얼 추론 엔진 (프로세스 모드).

    EngineCoreProc를 상속하여 hybrid CPU 로직을 추가.
    이 클래스는 run_engine_core()에서 EngineCoreProc 대신 생성됨.

    Note: 실제 상속은 __init_subclass__ 시점이 아닌
    _create_hybrid_proc()에서 동적으로 수행.
    """

    @staticmethod
    def create(
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: Any,
        log_stats: bool,
        client_handshake_address: Optional[str] = None,
        engine_index: int = 0,
    ) -> Any:
        """Create a HybridEngineCoreProc instance.

        이 팩토리 메서드는 EngineCoreProc를 동적으로 확장하여
        hybrid CPU 로직을 추가합니다.
        """
        from vllm.v1.engine.core import EngineCoreProc

        # EngineCoreProc 인스턴스 생성 (GPU 경로 완전 초기화)
        proc = EngineCoreProc(
            vllm_config=vllm_config,
            local_client=local_client,
            handshake_address=handshake_address,
            executor_class=executor_class,
            log_stats=log_stats,
            client_handshake_address=client_handshake_address,
            engine_index=engine_index,
        )

        # Hybrid CPU 경로 초기화
        proc._hybrid_enabled = False
        proc._hybrid_cpu_process = None
        proc._hybrid_router = None
        proc._hybrid_request_to_path = {}
        proc._hybrid_dead_outputs = None

        enabled, router, cpu_proc = _init_hybrid_cpu_process(vllm_config)
        proc._hybrid_enabled = enabled
        proc._hybrid_router = router
        proc._hybrid_cpu_process = cpu_proc

        # 메서드 바인딩: proc 인스턴스의 메서드를 hybrid 버전으로 교체
        import types
        proc._check_hybrid_cpu_health = types.MethodType(
            _HybridMixin._check_hybrid_cpu_health, proc)
        proc._collect_hybrid_cpu_outputs = types.MethodType(
            _HybridMixin._collect_hybrid_cpu_outputs, proc)
        proc._has_hybrid_cpu_pending = types.MethodType(
            _HybridMixin._has_hybrid_cpu_pending, proc)
        proc._route_request_to_cpu = types.MethodType(
            _HybridMixin._route_request_to_cpu, proc)
        proc._abort_hybrid_cpu_requests = types.MethodType(
            _HybridMixin._abort_hybrid_cpu_requests, proc)
        proc._shutdown_hybrid_cpu = types.MethodType(
            _HybridMixin._shutdown_hybrid_cpu, proc)

        # Override key methods with hybrid-aware versions
        _original_step = proc.step
        _original_step_with_batch_queue = proc.step_with_batch_queue
        _original_abort_requests = proc.abort_requests
        _original_shutdown = proc.shutdown
        _original_preprocess_add_request = proc.preprocess_add_request
        _original_process_input_queue = proc._process_input_queue

        def _hybrid_preprocess_add_request(request):
            """Route to CPU or pass through to GPU preprocessing."""
            if proc._route_request_to_cpu(request):
                return None, request.current_wave
            return _original_preprocess_add_request(request)

        def _hybrid_step():
            """GPU step + CPU results merge."""
            proc._check_hybrid_cpu_health()

            has_gpu = proc.scheduler.has_requests()
            has_cpu = proc._has_hybrid_cpu_pending()

            if not has_gpu and not has_cpu:
                return {}, False

            engine_core_outputs = {}
            model_executed = False

            if has_gpu:
                engine_core_outputs, model_executed = _original_step()

            engine_core_outputs = proc._collect_hybrid_cpu_outputs(
                engine_core_outputs)

            return engine_core_outputs, model_executed

        def _hybrid_step_with_batch_queue():
            """Batch queue step + CPU results merge."""
            proc._check_hybrid_cpu_health()

            engine_core_outputs, scheduled_batch = \
                _original_step_with_batch_queue()

            engine_core_outputs = proc._collect_hybrid_cpu_outputs(
                engine_core_outputs)

            return engine_core_outputs, scheduled_batch

        def _hybrid_abort_requests(request_ids):
            proc._abort_hybrid_cpu_requests(request_ids)
            _original_abort_requests(request_ids)

        def _hybrid_shutdown():
            proc._shutdown_hybrid_cpu()
            _original_shutdown()

        def _hybrid_process_input_queue():
            """Process input queue with CPU pending awareness.

            항상 timeout 기반 get()을 사용하여 CPU 결과 도착 시
            즉시 루프를 깨뜨릴 수 있도록 함. 영구 블로킹 방지.
            """
            waited = False
            while (not proc.engines_running
                   and not proc.scheduler.has_requests()
                   and not proc._has_hybrid_cpu_pending()):
                if logger.isEnabledFor(DEBUG) and proc.input_queue.empty():
                    logger.debug("EngineCore waiting for work.")
                    waited = True
                try:
                    req = proc.input_queue.get(timeout=0.05)
                except queue.Empty:
                    if proc._has_hybrid_cpu_pending():
                        break
                    continue
                proc._handle_client_request(*req)

            if waited:
                logger.debug("EngineCore loop active.")

            while not proc.input_queue.empty():
                try:
                    req = proc.input_queue.get_nowait()
                except queue.Empty:
                    break
                proc._handle_client_request(*req)

        # Bind hybrid methods
        proc.preprocess_add_request = _hybrid_preprocess_add_request
        proc.step = _hybrid_step
        proc.step_with_batch_queue = _hybrid_step_with_batch_queue
        proc.abort_requests = _hybrid_abort_requests
        proc.shutdown = _hybrid_shutdown
        proc._process_input_queue = _hybrid_process_input_queue

        # Update step_fn to point to the hybrid version
        if proc.batch_queue is None:
            proc.step_fn = _hybrid_step
        else:
            proc.step_fn = _hybrid_step_with_batch_queue

        logger.info("HybridEngineCoreProc initialized successfully")
        return proc
