# SPDX-License-Identifier: Apache-2.0
"""
HybridEngineCore: GPU + CPU 진정한 병렬 추론 아키텍처.

GPU EngineCoreProc와 CPU EngineCoreProc를 별도 프로세스에서 실행하여
total_throughput = GPU_throughput + CPU_throughput을 달성합니다.

GPU 경로: 기존 EngineCoreProc (MultiprocExecutor, TP=8)
CPU 경로: 별도 EngineCoreProc (UniProcExecutor, CPUWorker, PagedAttention)

요청 라우팅: 클라이언트 레이어에서 cpu_ratio 기반 분배
결과 수집: ZMQ PUSH/PULL로 비동기 interleaved 수신

사용법:
    vllm serve model --hybrid-mode parallel-batch --hybrid-cpu-ratio 0.05
"""

import contextlib
import copy
import os
import signal
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Optional

from vllm.config import HybridConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils import get_mp_context, get_open_zmq_ipc_path
from vllm.v1.engine.utils import (CoreEngine, EngineZmqAddresses,
                                   wait_for_engine_startup)
from vllm.v1.executor.abstract import Executor
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

logger = init_logger(__name__)


# ============================================================================
# Helper: hybrid mode 판별
# ============================================================================

def is_hybrid_mode(vllm_config: VllmConfig) -> bool:
    """VllmConfig가 hybrid parallel-batch 모드인지 판별."""
    return (hasattr(vllm_config, 'hybrid_config')
            and vllm_config.hybrid_config is not None
            and vllm_config.hybrid_config.is_enabled()
            and vllm_config.hybrid_config.mode == "parallel-batch")


# ============================================================================
# Auto CPU Ratio
# ============================================================================

def compute_auto_cpu_ratio(
    cpu_tok_per_sec: float,
    gpu_tok_per_sec: float,
) -> float:
    """CPU/GPU 처리량 기반 최적 cpu_ratio 계산.

    R_cpu = T_cpu / (T_gpu + T_cpu)

    Args:
        cpu_tok_per_sec: CPU 처리량 (tok/s), 실측값 사용 권장
        gpu_tok_per_sec: GPU 처리량 (tok/s), 실측값 사용 권장

    Returns:
        cpu_ratio (0.01 ~ 0.5 범위로 클램핑)
    """
    if cpu_tok_per_sec <= 0 or gpu_tok_per_sec <= 0:
        return 0.0

    total = gpu_tok_per_sec + cpu_tok_per_sec
    ratio = cpu_tok_per_sec / total

    # 최소 1%, 최대 50%로 클램핑
    ratio = max(0.01, min(0.5, ratio))

    logger.info(
        "Auto cpu_ratio: %.2f%% (cpu=%.1f, gpu=%.1f tok/s)",
        ratio * 100, cpu_tok_per_sec, gpu_tok_per_sec,
    )
    return ratio


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
            "RequestRouter initialized: cpu_ratio=%.2f%%, interval=%d",
            self.cpu_ratio * 100, self.interval,
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
# Capacity-Aware Router (CPU 용량 기반 라우팅)
# ============================================================================

class CapacityAwareRouter:
    """CPU 용량 기반 요청 분배기.

    CPU에 여유가 있으면 항상 CPU로 라우팅하여 CPU 활용률 100% 유지.
    CPU가 가득 차면 GPU로 라우팅.

    라우팅 전략:
    - capacity: 단순 슬롯 기반 (기본값, 기존 동작)
    - length-aware: 프롬프트 길이 고려 — 긴 프롬프트는 GPU, 짧은 것만 CPU
    - throughput-adaptive: EMA 처리량 기반 동적 CPU 슬롯 조정

    실시간 처리량 모니터링: 요청 시작/완료 시간과 생성 토큰 수를 추적하여
    GPU/CPU 각각의 실측 처리량(tok/s)을 계산합니다.
    """

    def __init__(self, cpu_max_num_seqs: int,
                 gpu_max_num_seqs: int = 256,
                 num_cpu_engines: int = 1,
                 routing_strategy: str = "capacity",
                 routing_priority: str = "gpu-first",
                 cpu_prefill_threshold: int = 512,
                 warmup_requests: int = 10,
                 stats_log_interval: int = 50):
        self.cpu_max_num_seqs = cpu_max_num_seqs  # per-engine
        self.gpu_max_num_seqs = gpu_max_num_seqs
        self.num_cpu_engines = max(1, num_cpu_engines)
        self.routing_strategy = routing_strategy
        # round-robin이면 priority 무시
        self.routing_priority = (
            "gpu-first" if routing_strategy == "round-robin"
            else routing_priority)
        self.cpu_first = (self.routing_priority == "cpu-first")
        self.cpu_prefill_threshold = cpu_prefill_threshold

        # round-robin 카운터
        self._rr_counter: int = 0

        # Per-CPU-engine 상태 (in_flight, count)
        # 엔진 경로는 "cpu:0", "cpu:1", ... 형식
        self._cpu_states: list[dict] = [
            {"in_flight": 0, "count": 0,
             "total_tokens": 0, "total_elapsed": 0.0, "ema_throughput": 0.0}
            for _ in range(self.num_cpu_engines)
        ]

        # Aggregate (backward compat용 + 통계 로깅)
        self.cpu_in_flight: int = 0
        self.cpu_count: int = 0
        self.gpu_in_flight: int = 0
        self.gpu_count: int = 0

        # 실시간 처리량 모니터링
        self._request_start_times: dict[str, float] = {}
        self._gpu_total_tokens: int = 0
        self._cpu_total_tokens: int = 0
        self._gpu_total_elapsed: float = 0.0
        self._cpu_total_elapsed: float = 0.0

        # throughput-adaptive 전략용 EMA 필드
        self._gpu_ema_throughput: float = 0.0
        self._cpu_ema_throughput: float = 0.0
        self._ema_alpha: float = 0.3
        self._adaptive_cpu_max_seqs: int = cpu_max_num_seqs

        # 워밍업 프로파일링
        self._warmup_requests = warmup_requests
        self._warmup_complete: bool = (warmup_requests <= 0)
        self._warmup_gpu_finished: int = 0
        self._warmup_cpu_finished: int = 0
        self._warmup_gpu_tokens: int = 0
        self._warmup_cpu_tokens: int = 0
        self._warmup_gpu_elapsed: float = 0.0
        self._warmup_cpu_elapsed: float = 0.0

        # 주기적 로깅
        self._stats_log_interval = stats_log_interval
        self._total_finished: int = 0

        logger.info(
            "CapacityAwareRouter initialized: gpu_max_num_seqs=%d, "
            "cpu_max_num_seqs=%d (per-engine), num_cpu_engines=%d, "
            "strategy=%s, priority=%s, prefill_threshold=%d, "
            "warmup=%d, stats_interval=%d",
            self.gpu_max_num_seqs, self.cpu_max_num_seqs,
            self.num_cpu_engines, self.routing_strategy,
            self.routing_priority, self.cpu_prefill_threshold,
            self._warmup_requests, self._stats_log_interval,
        )
        if not self._warmup_complete:
            logger.info(
                "Warmup profiling enabled: collecting data from "
                "first %d requests per device", warmup_requests)

    def route(self, request_id: str, prompt_len: int = 0) -> str:
        """요청을 GPU 또는 CPU로 라우팅.

        Args:
            request_id: 요청 ID.
            prompt_len: 프롬프트 토큰 수 (length-aware/throughput-adaptive용).
        """
        self._request_start_times[request_id] = time.monotonic()

        if self.routing_strategy == "round-robin":
            return self._route_round_robin(request_id)
        elif self.routing_strategy == "length-aware":
            return self._route_length_aware(request_id, prompt_len)
        elif self.routing_strategy == "throughput-adaptive":
            return self._route_throughput_adaptive(request_id, prompt_len)
        else:  # "capacity" (기본)
            return self._route_capacity(request_id)

    def _find_available_cpu(self) -> int:
        """여유 슬롯이 가장 많은 CPU 엔진 인덱스 반환. 없으면 -1."""
        best_idx = -1
        best_free = 0
        for i, state in enumerate(self._cpu_states):
            free = self.cpu_max_num_seqs - state["in_flight"]
            if free > best_free:
                best_free = free
                best_idx = i
        return best_idx

    def _to_gpu(self) -> str:
        """GPU로 라우팅."""
        self.gpu_in_flight += 1
        self.gpu_count += 1
        return "gpu"

    def _to_cpu(self) -> Optional[str]:
        """여유 있는 CPU 엔진으로 라우팅. 없으면 None."""
        best = self._find_available_cpu()
        if best >= 0:
            self._cpu_states[best]["in_flight"] += 1
            self._cpu_states[best]["count"] += 1
            self.cpu_in_flight += 1
            self.cpu_count += 1
            return f"cpu:{best}"
        return None

    def _route_capacity(self, request_id: str) -> str:
        """슬롯 기반 라우팅. priority에 따라 primary/secondary 결정.

        반환값: "gpu" | "cpu:0" | "cpu:1" | ...
        """
        if self.cpu_first:
            # CPU-first: CPU 슬롯 여유 시 CPU, 가득차면 GPU
            result = self._to_cpu()
            if result is not None:
                logger.info("Route %s → %s (cpu_in_flight=%d/%d)",
                            request_id, result, self.cpu_in_flight,
                            self.cpu_max_num_seqs)
                return result
            dest = self._to_gpu()
            logger.info("Route %s → %s (cpu full, gpu_in_flight=%d)",
                        request_id, dest, self.gpu_in_flight)
            return dest
        else:
            # GPU-first (기본): GPU 포화 시에만 CPU
            if self.gpu_in_flight < self.gpu_max_num_seqs:
                return self._to_gpu()
            result = self._to_cpu()
            if result is not None:
                return result
            return self._to_gpu()

    def _route_round_robin(self, request_id: str) -> str:
        """교대로 GPU/CPU 분배. CPU가 가득차면 GPU, 그 반대도 마찬가지."""
        self._rr_counter += 1
        if self._rr_counter % 2 == 0:
            # GPU 차례
            if self.gpu_in_flight < self.gpu_max_num_seqs:
                return self._to_gpu()
            result = self._to_cpu()
            return result if result is not None else self._to_gpu()
        else:
            # CPU 차례
            result = self._to_cpu()
            if result is not None:
                return result
            return self._to_gpu()

    def _route_length_aware(self, request_id: str,
                            prompt_len: int) -> str:
        """priority 기반 + 길이 조건: 짧은 프롬프트만 CPU 허용."""
        if self.cpu_first:
            # CPU-first: 짧으면 CPU 우선, 길면 GPU
            if prompt_len <= self.cpu_prefill_threshold:
                result = self._to_cpu()
                if result is not None:
                    return result
            return self._to_gpu()
        else:
            # GPU-first: GPU 포화 AND 짧은 프롬프트일 때만 CPU
            if (self.gpu_in_flight >= self.gpu_max_num_seqs
                    and prompt_len <= self.cpu_prefill_threshold):
                result = self._to_cpu()
                if result is not None:
                    return result
            return self._to_gpu()

    def _route_throughput_adaptive(self, request_id: str,
                                   prompt_len: int) -> str:
        """priority 기반 + EMA 처리량 기반 동적 CPU 슬롯 조정."""
        effective_max = self._adaptive_cpu_max_seqs
        if self.cpu_first:
            # CPU-first: 짧으면 CPU 우선 (adaptive 슬롯 범위 내)
            if (prompt_len <= self.cpu_prefill_threshold
                    and self.cpu_in_flight
                    < effective_max * self.num_cpu_engines):
                result = self._to_cpu()
                if result is not None:
                    return result
            return self._to_gpu()
        else:
            # GPU-first: GPU 포화 시에만 CPU
            if (self.gpu_in_flight >= self.gpu_max_num_seqs
                    and prompt_len <= self.cpu_prefill_threshold
                    and self.cpu_in_flight
                    < effective_max * self.num_cpu_engines):
                result = self._to_cpu()
                if result is not None:
                    return result
            return self._to_gpu()

    def on_request_finished(self, request_id: str, engine_path: str,
                            num_tokens: int = 0):
        """요청 완료 시 호출. 슬롯 반환 및 처리량 업데이트.

        Args:
            engine_path: "gpu" | "cpu:0" | "cpu:1" | ...
        """
        logger.info("Request finished: %s on %s, tokens=%d "
                     "(cpu_count=%d, gpu_count=%d, total=%d)",
                     request_id, engine_path, num_tokens,
                     self.cpu_count, self.gpu_count,
                     self._total_finished + 1)
        was_cpu = engine_path.startswith("cpu")
        if was_cpu:
            try:
                cpu_idx = int(engine_path.split(":")[1])
            except (IndexError, ValueError):
                cpu_idx = 0
            if 0 <= cpu_idx < len(self._cpu_states):
                self._cpu_states[cpu_idx]["in_flight"] = max(
                    0, self._cpu_states[cpu_idx]["in_flight"] - 1)
            self.cpu_in_flight = max(0, self.cpu_in_flight - 1)
        else:
            self.gpu_in_flight = max(0, self.gpu_in_flight - 1)

        # 처리량 측정: 시작 시간이 있고 토큰이 생성된 경우에만
        start = self._request_start_times.pop(request_id, None)
        if start is not None and num_tokens > 0:
            elapsed = time.monotonic() - start
            if elapsed > 0:
                if was_cpu:
                    self._cpu_total_tokens += num_tokens
                    self._cpu_total_elapsed += elapsed
                else:
                    self._gpu_total_tokens += num_tokens
                    self._gpu_total_elapsed += elapsed

                # 워밍업 프로파일링 데이터 수집
                if not self._warmup_complete:
                    self._collect_warmup_data(
                        was_cpu, num_tokens, elapsed)

                # throughput-adaptive: EMA 업데이트 및 동적 슬롯 조정
                if self.routing_strategy == "throughput-adaptive":
                    instant_tps = num_tokens / elapsed
                    alpha = self._ema_alpha
                    if was_cpu:
                        self._cpu_ema_throughput = (
                            alpha * instant_tps
                            + (1 - alpha) * self._cpu_ema_throughput)
                    else:
                        self._gpu_ema_throughput = (
                            alpha * instant_tps
                            + (1 - alpha) * self._gpu_ema_throughput)
                    self._update_adaptive_slots()

        # 주기적 통계 로깅
        self._total_finished += 1
        if (self._stats_log_interval > 0
                and self._total_finished % self._stats_log_interval == 0):
            self._log_periodic_stats()

    def _update_adaptive_slots(self):
        """CPU/GPU 처리량 비율 기반 동적 슬롯 수 조정."""
        if self._cpu_ema_throughput > 0 and self._gpu_ema_throughput > 0:
            ratio = self._cpu_ema_throughput / self._gpu_ema_throughput
            # CPU 처리량이 높을수록 더 많은 슬롯 할당 (최소 2, 최대 2배)
            new_max = max(2, min(self.cpu_max_num_seqs * 2,
                                int(self.cpu_max_num_seqs * (1 + ratio))))
            self._adaptive_cpu_max_seqs = new_max

    def _collect_warmup_data(self, was_cpu: bool,
                             num_tokens: int, elapsed: float):
        """워밍업 페이즈 데이터 수집 및 완료 판정."""
        if was_cpu:
            self._warmup_cpu_finished += 1
            self._warmup_cpu_tokens += num_tokens
            self._warmup_cpu_elapsed += elapsed
        else:
            self._warmup_gpu_finished += 1
            self._warmup_gpu_tokens += num_tokens
            self._warmup_gpu_elapsed += elapsed

        gpu_ready = (self._warmup_gpu_finished
                     >= self._warmup_requests)
        cpu_ready = (self._warmup_cpu_finished
                     >= self._warmup_requests)

        if gpu_ready and cpu_ready:
            self._finalize_warmup()
        elif (gpu_ready
              and self._warmup_cpu_finished >= 1
              and self._warmup_gpu_finished
              >= self._warmup_requests * 2):
            # GPU가 2배 넘게 완료, CPU가 1개 이상 → 강제 완료
            self._finalize_warmup()

    def _finalize_warmup(self):
        """워밍업 완료: 측정된 처리량으로 EMA 초기화."""
        self._warmup_complete = True

        gpu_avg_tps = (self._warmup_gpu_tokens
                       / self._warmup_gpu_elapsed
                       if self._warmup_gpu_elapsed > 0 else 0.0)
        cpu_avg_tps = (self._warmup_cpu_tokens
                       / self._warmup_cpu_elapsed
                       if self._warmup_cpu_elapsed > 0 else 0.0)

        # throughput-adaptive: EMA 초기화 → 즉시 슬롯 조정
        if self.routing_strategy == "throughput-adaptive":
            self._gpu_ema_throughput = gpu_avg_tps
            self._cpu_ema_throughput = cpu_avg_tps
            self._update_adaptive_slots()

        logger.info(
            "=== Warmup profiling complete ===")
        logger.info(
            "  GPU: %.1f tok/s (avg over %d reqs, %d tokens)",
            gpu_avg_tps, self._warmup_gpu_finished,
            self._warmup_gpu_tokens)
        logger.info(
            "  CPU: %.1f tok/s (avg over %d reqs, %d tokens)",
            cpu_avg_tps, self._warmup_cpu_finished,
            self._warmup_cpu_tokens)
        if self.routing_strategy == "throughput-adaptive":
            logger.info(
                "  EMA initialized → adaptive_slots=%d (base=%d)",
                self._adaptive_cpu_max_seqs,
                self.cpu_max_num_seqs)

    def _log_periodic_stats(self):
        """주기적 처리량/라우팅 통계 로깅."""
        stats = self.get_stats()
        extra = ""
        if self.routing_strategy == "throughput-adaptive":
            extra = (f", adaptive_slots="
                     f"{stats.get('adaptive_cpu_max_seqs', 'N/A')}")
        logger.info(
            "Router stats [%d reqs]: "
            "GPU=%.1f tok/s (%d reqs), "
            "CPU=%.1f tok/s (%d reqs), "
            "cpu_ratio=%.1f%%, in_flight=%d/%d%s",
            stats["total_requests"],
            stats["gpu_throughput_tok_s"], stats["gpu_requests"],
            stats["cpu_throughput_tok_s"], stats["cpu_requests"],
            stats["actual_cpu_ratio"] * 100,
            stats["cpu_in_flight"], stats["cpu_max_num_seqs"],
            extra)

    @property
    def gpu_throughput(self) -> float:
        """GPU 실측 처리량 (tok/s). 데이터 없으면 0.0."""
        if self._gpu_total_elapsed > 0:
            return self._gpu_total_tokens / self._gpu_total_elapsed
        return 0.0

    @property
    def cpu_throughput(self) -> float:
        """CPU 실측 처리량 (tok/s). 데이터 없으면 0.0."""
        if self._cpu_total_elapsed > 0:
            return self._cpu_total_tokens / self._cpu_total_elapsed
        return 0.0

    def get_stats(self) -> dict:
        total = self.gpu_count + self.cpu_count
        stats = {
            "gpu_requests": self.gpu_count,
            "cpu_requests": self.cpu_count,
            "total_requests": total,
            "cpu_in_flight": self.cpu_in_flight,
            "cpu_max_num_seqs": self.cpu_max_num_seqs,
            "routing_strategy": self.routing_strategy,
            "routing_priority": self.routing_priority,
            "actual_cpu_ratio": (
                self.cpu_count / total if total > 0 else 0.0),
            "gpu_throughput_tok_s": self.gpu_throughput,
            "cpu_throughput_tok_s": self.cpu_throughput,
            "gpu_total_tokens": self._gpu_total_tokens,
            "cpu_total_tokens": self._cpu_total_tokens,
            "warmup_complete": self._warmup_complete,
        }
        if self.routing_strategy == "throughput-adaptive":
            stats["gpu_ema_throughput"] = self._gpu_ema_throughput
            stats["cpu_ema_throughput"] = self._cpu_ema_throughput
            stats["adaptive_cpu_max_seqs"] = self._adaptive_cpu_max_seqs
        return stats


# ============================================================================
# CPU 파라미터 자동 감지
# ============================================================================

@dataclass
class ResolvedCpuParams:
    """자동 감지된 CPU 파라미터."""
    cpu_max_num_seqs: int
    cpu_kvcache_space_gb: int
    cpu_max_num_batched_tokens: int
    cpu_num_threads: int


def _resolve_cpu_params(hybrid_config: HybridConfig) -> ResolvedCpuParams:
    """HybridConfig의 0(auto) 값을 시스템 감지값으로 해석.

    자동 감지 전략:
    - cpu_num_threads: NUMA 바인딩 시 해당 노드 코어 수, 아니면 전체 물리 코어
    - cpu_max_num_seqs: effective_cores // 4 (최소 4)
    - cpu_kvcache_space_gb: total_memory * 0.4 (최소 32GB, 최대 512GB)
    - cpu_max_num_batched_tokens: cpu_max_num_seqs * 256

    NUMA 토폴로지를 고려하여 바인딩된 노드의 코어 수를 기반으로
    스레드/시퀀스 수를 결정합니다.
    """
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False) or 8
        total_mem_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        physical_cores = os.cpu_count() or 8
        # /proc/meminfo fallback
        total_mem_gb = 64
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_mem_gb = int(line.split()[1]) / (1024**2)
                        break
        except (OSError, ValueError):
            pass

    # NUMA 토폴로지 감지 — 바인딩될 노드의 코어 수 파악
    numa_node_cores = None
    numa_node_mem_gb = None
    numa_num_nodes = 1
    target_numa_node = hybrid_config.numa_bind_node  # None or specific node

    try:
        from vllm.platforms.intel_cpu_utils import (
            NUMAAllocator, detect_intel_cpu_features)
        allocator = NUMAAllocator()
        if allocator.is_available and allocator.num_nodes > 1:
            numa_num_nodes = allocator.num_nodes
            # CPU 워커는 항상 local_rank=0 → NUMA 노드 0과 일치시킴
            # (init_device()의 _get_autobind_cpu_ids()가 local_rank=0 기반)
            if target_numa_node is None:
                target_numa_node = allocator.get_preferred_node_for_rank(
                    0, allocator.num_nodes)
            node_info = allocator.get_node_info(target_numa_node)
            if node_info and node_info.cpu_ids:
                logical_cpus_on_node = len(node_info.cpu_ids)
                numa_node_mem_gb = node_info.total_memory_bytes / (1024**3)
                # cpu_ids는 HT 포함 논리 CPU → 물리 코어로 변환
                # features가 감지되면 threads_per_core 사용, 아니면 추정
                try:
                    feats = detect_intel_cpu_features()
                    tpc = max(1, feats.threads_per_core)
                except Exception:
                    tpc = 2 if logical_cpus_on_node > physical_cores else 1
                numa_node_cores = logical_cpus_on_node // tpc
                logger.info(
                    "NUMA node %d detected: %d physical cores "
                    "(%d logical CPUs, %d threads/core), %.0fGB memory",
                    target_numa_node, numa_node_cores,
                    logical_cpus_on_node, tpc, numa_node_mem_gb,
                )
    except ImportError:
        pass
    except Exception as e:
        logger.debug("NUMA topology detection failed: %s", e)

    # effective_cores: NUMA 바인딩 시 해당 노드 코어, 아니면 전체
    if hybrid_config.numa_aware and numa_node_cores is not None:
        effective_cores = numa_node_cores
    else:
        effective_cores = physical_cores

    # effective_mem: NUMA 바인딩 시 해당 노드 메모리, 아니면 전체
    if hybrid_config.numa_aware and numa_node_mem_gb is not None:
        effective_mem_gb = numa_node_mem_gb
    else:
        effective_mem_gb = total_mem_gb

    # cpu_num_threads: 유효 코어 수 전부 사용 (코어 낭비 없음)
    if hybrid_config.cpu_num_threads > 0:
        cpu_num_threads = hybrid_config.cpu_num_threads
    else:
        cpu_num_threads = effective_cores

    # cpu_max_num_seqs: 유효 코어 기반 (코어 4개당 시퀀스 1개)
    if hybrid_config.cpu_max_num_seqs > 0:
        cpu_max_num_seqs = hybrid_config.cpu_max_num_seqs
    else:
        cpu_max_num_seqs = max(4, effective_cores // 4)

    # cpu_kvcache_space_gb: 유효 메모리 기반 (40%, 최소 32GB, 최대 512GB)
    if hybrid_config.cpu_kvcache_space_gb > 0:
        cpu_kvcache_space_gb = hybrid_config.cpu_kvcache_space_gb
    else:
        cpu_kvcache_space_gb = max(32, min(512, int(effective_mem_gb * 0.4)))

    # cpu_max_num_batched_tokens
    if hybrid_config.cpu_max_num_batched_tokens > 0:
        cpu_max_num_batched_tokens = hybrid_config.cpu_max_num_batched_tokens
    else:
        cpu_max_num_batched_tokens = cpu_max_num_seqs * 256

    resolved = ResolvedCpuParams(
        cpu_max_num_seqs=cpu_max_num_seqs,
        cpu_kvcache_space_gb=cpu_kvcache_space_gb,
        cpu_max_num_batched_tokens=cpu_max_num_batched_tokens,
        cpu_num_threads=cpu_num_threads,
    )

    logger.info(
        "Resolved CPU params: max_seqs=%d, kvcache=%dGB, "
        "max_batched_tokens=%d, threads=%d "
        "(effective_cores=%d, physical_cores=%d, "
        "numa_nodes=%d, effective_mem=%.0fGB, total_mem=%.0fGB)",
        resolved.cpu_max_num_seqs,
        resolved.cpu_kvcache_space_gb,
        resolved.cpu_max_num_batched_tokens,
        resolved.cpu_num_threads,
        effective_cores,
        physical_cores,
        numa_num_nodes,
        effective_mem_gb,
        total_mem_gb,
    )

    return resolved


# ============================================================================
# CPU 프로세스 환경 설정
# ============================================================================

def _setup_cpu_process_env(
    resolved: ResolvedCpuParams,
    hybrid_config: Optional[HybridConfig] = None,
):
    """CPU 프로세스 시작 시 Intel 최적화 환경변수 + NUMA + IPEX + AMX 설정.

    run_cpu_engine_core()에서 호출됩니다.

    설정 순서:
    1. VLLM_CPU_KVCACHE_SPACE, OMP_NUM_THREADS (resolved 기반)
    2. configure_intel_optimizations() → AVX-512/AMX 감지 기반 환경변수
       (KMP_AFFINITY, MKL, ONEDNN_MAX_CPU_ISA 등)
    3. configure_pytorch_for_intel() → PyTorch Inductor, AMX 타일, IPEX
    4. NUMA affinity → numa_bind_node 또는 자동 선택
    """
    # 1. KV cache 크기 설정 (CpuPlatform이 참조)
    os.environ["VLLM_CPU_KVCACHE_SPACE"] = str(resolved.cpu_kvcache_space_gb)

    # OpenMP 스레드 수 설정 (configure_intel_optimizations 전에 설정)
    os.environ["OMP_NUM_THREADS"] = str(resolved.cpu_num_threads)

    # 스레드 바인딩 모드 설정 (CPUWorker.init_device()에서 참조)
    # "auto"는 NUMA 토폴로지 기반 자동 바인딩
    os.environ.setdefault("VLLM_CPU_OMP_THREADS_BIND", "auto")

    # 예약 CPU 코어 없음 — 하이브리드 모드에서 CPU 프로세스가 모든 코어 사용
    os.environ.setdefault("VLLM_CPU_NUM_OF_RESERVED_CPU", "0")

    # 2. Intel 환경변수 설정 (CPU 기능 감지 → AMX/AVX-512 자동 판별)
    # configure_intel_optimizations()가 setdefault로 설정하므로,
    # 미리 하드코딩하지 않고 감지 결과에 위임합니다.
    features = None
    try:
        from vllm.platforms.intel_cpu_utils import (
            configure_intel_optimizations,
            detect_intel_cpu_features,
        )
        features = detect_intel_cpu_features()
        settings = configure_intel_optimizations(features)

        logger.info(
            "Intel optimizations configured: AMX=%s, AVX512=%s, "
            "VNNI=%s, ONEDNN_ISA=%s",
            features.amx_bf16 or features.amx_int8,
            features.avx512f,
            features.avx512_vnni,
            settings.get("ONEDNN_MAX_CPU_ISA", "N/A"),
        )
    except ImportError:
        logger.debug(
            "intel_cpu_utils not available, "
            "falling back to basic environment setup")
        # 최소한의 fallback 환경변수
        for key, value in {
            "KMP_AFFINITY": "granularity=fine,compact,1,0",
            "KMP_BLOCKTIME": "1",
            "KMP_TPAUSE": "0",
        }.items():
            os.environ.setdefault(key, value)

    # 3. PyTorch 설정 (Inductor simdlen, AMX 타일 활성화, IPEX)
    try:
        from vllm.platforms.intel_cpu_utils import configure_pytorch_for_intel
        configure_pytorch_for_intel(features)
        logger.info("PyTorch Intel optimizations applied (IPEX/AMX/Inductor)")
    except ImportError:
        logger.debug("configure_pytorch_for_intel not available")
    except Exception as e:
        logger.warning("PyTorch Intel optimization failed: %s", e)

    # 4. NUMA affinity 설정
    numa_aware = True
    numa_bind_node = None
    if hybrid_config is not None:
        numa_aware = hybrid_config.numa_aware
        numa_bind_node = hybrid_config.numa_bind_node

    if numa_aware:
        try:
            from vllm.platforms.intel_cpu_utils import NUMAAllocator
            allocator = NUMAAllocator()
            if allocator.is_available:
                if numa_bind_node is not None:
                    target_node = numa_bind_node
                    if target_node != 0 and allocator.num_nodes > 1:
                        logger.warning(
                            "NUMA bind node %d != 0. CPUWorker의 "
                            "스레드 바인딩은 local_rank=0 기반 (NUMA 노드 0). "
                            "일치시키려면 VLLM_CPU_OMP_THREADS_BIND를 "
                            "수동 설정하세요.", target_node)
                else:
                    # CPU 워커는 local_rank=0 → NUMA 노드 0과 일치
                    target_node = allocator.get_preferred_node_for_rank(
                        0, allocator.num_nodes)
                allocator.bind_to_node(target_node)
                logger.info(
                    "CPU process bound to NUMA node %d "
                    "(total %d nodes)",
                    target_node, allocator.num_nodes,
                )
            else:
                logger.debug("NUMA not available on this system")
        except ImportError:
            logger.debug("NUMAAllocator not available")
        except Exception as e:
            logger.warning("NUMA binding failed: %s", e)
    else:
        logger.info("NUMA affinity disabled by config")

    logger.info(
        "CPU process env configured: OMP_NUM_THREADS=%s, "
        "VLLM_CPU_KVCACHE_SPACE=%s, ONEDNN_MAX_CPU_ISA=%s",
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("VLLM_CPU_KVCACHE_SPACE"),
        os.environ.get("ONEDNN_MAX_CPU_ISA", "not set"),
    )


# ============================================================================
# CPU VllmConfig Factory
# ============================================================================

def _create_cpu_vllm_config(
    gpu_config: VllmConfig,
    resolved: Optional[ResolvedCpuParams] = None,
) -> VllmConfig:
    """GPU VllmConfig에서 CPU EngineCore용 VllmConfig를 파생 생성.

    핵심 변경:
    - DeviceConfig: "cpu" 명시
    - ParallelConfig: TP=1, PP=1, 단일 프로세스
    - CacheConfig: CPU KV cache 할당
    - SchedulerConfig: CPU 처리량에 맞는 제한
    - CompilationConfig: CUDA graph 비활성화
    - HybridConfig: 비활성화 (CPU 엔진 내부에서는 hybrid 미사용)
    """
    from vllm.config import DeviceConfig

    hybrid = gpu_config.hybrid_config

    # 자동 감지된 파라미터 사용 (없으면 새로 감지)
    if resolved is None:
        resolved = _resolve_cpu_params(hybrid)

    # 1. DeviceConfig: "cpu" 명시 → __post_init__에서 auto 감지 우회
    cpu_device = DeviceConfig(device="cpu")

    # 2. ParallelConfig: TP=1, PP=1
    cpu_parallel = copy.deepcopy(gpu_config.parallel_config)
    cpu_parallel.tensor_parallel_size = 1
    cpu_parallel.pipeline_parallel_size = 1
    cpu_parallel.world_size = 1
    cpu_parallel.data_parallel_size = 1
    cpu_parallel.data_parallel_size_local = 1
    cpu_parallel.distributed_executor_backend = "mp"
    cpu_parallel.worker_cls = "auto"  # CpuPlatform이 CPUWorker로 설정

    # 3. CacheConfig: CPU KV cache 할당 (자동 감지값 사용)
    cpu_cache = copy.deepcopy(gpu_config.cache_config)
    cpu_cache.cpu_kvcache_space_bytes = resolved.cpu_kvcache_space_gb * (
        1024**3)

    # 4. SchedulerConfig: CPU 처리량에 맞는 제한 (자동 감지값 사용)
    cpu_sched = copy.deepcopy(gpu_config.scheduler_config)
    cpu_sched.max_num_seqs = resolved.cpu_max_num_seqs
    # CPU에서 chunked prefill 비활성화: decode와 interleave되면
    # 매 step마다 prefill chunk 처리로 decode가 극심하게 느려짐
    cpu_sched.enable_chunked_prefill = False
    cpu_sched.chunked_prefill_enabled = False
    # chunked prefill 끄면 max_num_batched_tokens >= max_model_len 필수
    # CPU에서는 max_model_len을 제한하여 메모리와 지연 시간 관리
    cpu_max_model_len = min(
        gpu_config.model_config.max_model_len,
        resolved.cpu_max_num_batched_tokens * resolved.cpu_max_num_seqs
    )
    cpu_sched.max_num_batched_tokens = max(
        resolved.cpu_max_num_batched_tokens, cpu_max_model_len)
    cpu_sched.max_model_len = cpu_max_model_len

    # 5. ModelConfig: deepcopy (config_updated 플래그 리셋)
    cpu_model = copy.deepcopy(gpu_config.model_config)
    cpu_model.config_updated = False  # CPU용 재검증 허용
    cpu_model.enforce_eager = True  # CPU에서는 torch.compile 비활성화
    cpu_model.max_model_len = cpu_max_model_len

    # 6. CompilationConfig: CUDA graph 비활성화, 커스텀 ops 비활성화
    # vLLM 커스텀 ops (rms_norm, silu_and_mul 등)는 CUDA 전용이므로
    # CPU에서는 PyTorch 네이티브 구현을 사용해야 함
    cpu_compilation = copy.deepcopy(gpu_config.compilation_config)
    cpu_compilation.level = 0  # NO_COMPILATION — 커스텀 ops 우회
    cpu_compilation.custom_ops = ["none"]  # 모든 커스텀 ops 비활성화

    # 7. LoadConfig: CPU 호환 로드
    cpu_load = copy.deepcopy(gpu_config.load_config)

    # 8. VllmConfig 조립
    cpu_config = replace(
        gpu_config,
        model_config=cpu_model,
        device_config=cpu_device,
        parallel_config=cpu_parallel,
        cache_config=cpu_cache,
        scheduler_config=cpu_sched,
        compilation_config=cpu_compilation,
        load_config=cpu_load,
        hybrid_config=HybridConfig(),  # CPU 엔진은 hybrid 미사용
        # GPU 전용 기능 비활성화
        lora_config=None,
        speculative_config=None,
    )

    # 9. CpuPlatform 설정 명시 적용
    # (GPU 프로세스 내에서 current_platform이 CUDA를 가리키므로)
    try:
        from vllm.platforms.cpu import CpuPlatform
        # CpuPlatform.check_and_update_config()가 kv cache를
        # 시스템 총 메모리 기반으로 덮어쓰므로 사전 저장 후 복원
        _saved_kv_bytes = cpu_config.cache_config.cpu_kvcache_space_bytes
        CpuPlatform.check_and_update_config(cpu_config)
        cpu_config.cache_config.cpu_kvcache_space_bytes = _saved_kv_bytes
    except Exception as e:
        logger.warning("CpuPlatform.check_and_update_config failed: %s", e)
        # fallback: 최소한의 수동 설정
        cpu_config.compilation_config.cudagraph_capture_sizes = []

    # worker_cls 강제 설정 (CpuPlatform이 올바르게 설정하지 않을 수 있음)
    cpu_config.parallel_config.worker_cls = (
        "vllm.v1.worker.cpu_worker.CPUWorker")
    # device_type 강제 설정 (heterogeneous로 덮어써지는 것 방지)
    cpu_config.device_config = DeviceConfig(device="cpu")

    logger.info(
        "CPU VllmConfig created: device=%s, TP=%d, max_seqs=%d, "
        "kv_cache=%.1fGB, worker=%s",
        cpu_config.device_config.device_type,
        cpu_config.parallel_config.tensor_parallel_size,
        cpu_config.scheduler_config.max_num_seqs,
        cpu_config.cache_config.cpu_kvcache_space_bytes / (1024**3),
        cpu_config.parallel_config.worker_cls,
    )

    return cpu_config


# ============================================================================
# CPU 프로세스 진입점
# ============================================================================

def run_cpu_engine_core(*args,
                        dp_rank: int = 0,
                        local_dp_rank: int = 0,
                        **kwargs):
    """CPU EngineCoreProc를 별도 프로세스에서 실행.

    EngineCoreProc.run_engine_core()와 동일한 패턴을 따르되,
    GPU config에서 CPU config를 파생하여 CPU EngineCore를 생성합니다.
    CUDA_VISIBLE_DEVICES="" 설정으로 CUDA 초기화를 방지합니다.

    중요: OMP/KMP 환경변수는 torch import 전에 설정해야 합니다.
    OpenMP 런타임은 첫 torch import 시 초기화되므로, 그 전에
    환경변수가 설정되어야 스레드 바인딩이 적용됩니다.
    """
    # ===== CRITICAL: OMP/환경변수를 torch import 전에 설정 =====
    # torch import 시 OpenMP 런타임이 초기화되므로,
    # 반드시 모든 OMP/KMP 환경변수를 먼저 설정해야 합니다.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # CPU 프로세스에서 current_platform을 CpuPlatform으로 강제 설정.
    # 부모 프로세스에서 CUDA로 캐시된 _current_platform이 spawn으로 상속되어
    # __post_init__()에서 잘못된 platform의 check_and_update_config가 호출됨.
    import vllm.platforms as _platforms
    from vllm.platforms.cpu import CpuPlatform
    _platforms._current_platform = CpuPlatform()

    vllm_config: VllmConfig = kwargs["vllm_config"]
    hybrid_cfg = vllm_config.hybrid_config
    numa_node_override = kwargs.get("numa_node", None)
    if numa_node_override is not None and numa_node_override >= 0:
        import copy as _copy
        hybrid_cfg = _copy.replace(hybrid_cfg, numa_bind_node=numa_node_override)

    # CPU 파라미터 자동 감지 및 환경 설정 (NUMA/AMX/IPEX 포함)
    # torch import 전에 호출하여 OMP 환경변수가 반영되도록 함
    resolved = _resolve_cpu_params(hybrid_cfg)
    _setup_cpu_process_env(resolved, hybrid_cfg)

    # ===== 이제 torch를 안전하게 import =====
    from vllm.transformers_utils.config import (
        maybe_register_config_serialize_by_value)
    from vllm.utils import decorate_logs, set_process_title

    # Signal handler (run_engine_core 패턴 동일)
    shutdown_requested = False

    maybe_register_config_serialize_by_value()

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    from vllm.v1.engine.core import EngineCoreProc

    engine_core: Optional[EngineCoreProc] = None
    try:
        set_process_title("CPU_EngineCore")
        decorate_logs()

        # GPU config에서 CPU config 파생 (자동 감지값 사용)
        cpu_config = _create_cpu_vllm_config(vllm_config, resolved)

        # CPU 전용 executor_class: UniProcExecutor 강제
        # MultiprocExecutor는 추가 워커 프로세스를 spawn하여
        # OMP 스레드가 분산되고 CPU 성능이 극심하게 저하됨
        from vllm.v1.executor.abstract import UniProcExecutor
        cpu_executor_class = UniProcExecutor

        logger.info(
            "Starting CPU EngineCore (PID %d) with executor %s",
            os.getpid(), cpu_executor_class.__name__,
        )

        # EngineCoreProc 생성 (CPU config 사용)
        # engine_index: 호출자가 kwargs로 전달 (기본값 1)
        cpu_kwargs = dict(kwargs)
        cpu_kwargs["vllm_config"] = cpu_config
        cpu_kwargs["executor_class"] = cpu_executor_class
        cpu_kwargs["engine_index"] = kwargs.get("engine_index", 1)
        # numa_node는 _resolve_cpu_params/_setup_cpu_process_env에서 이미 처리됨

        engine_core = EngineCoreProc(**cpu_kwargs)
        engine_core.run_busy_loop()

    except SystemExit:
        logger.debug("CPU EngineCore exiting.")
        raise
    except Exception as e:
        if engine_core is None:
            logger.exception("CPU EngineCore failed to start.")
        else:
            logger.exception("CPU EngineCore encountered a fatal error.")
            engine_core._send_engine_dead()
        raise e
    finally:
        if engine_core is not None:
            engine_core.shutdown()


# ============================================================================
# Hybrid Engine 프로세스 관리자
# ============================================================================

class HybridEngineProcManager:
    """GPU + CPU 엔진 프로세스를 관리.

    CoreEngineProcManager와 유사하지만, GPU와 CPU 프로세스를 별도로 추적하여
    CPU 프로세스 사망 시 GPU-only fallback을 지원합니다.
    """

    def __init__(self, processes: list):
        self.processes = processes
        self._finalizer = weakref.finalize(self, shutdown, self.processes)

    def close(self):
        self._finalizer()

    def sentinels(self) -> list:
        return [proc.sentinel for proc in self.processes]

    def finished_procs(self) -> dict[str, int]:
        return {
            proc.name: proc.exitcode
            for proc in self.processes if proc.exitcode is not None
        }


# ============================================================================
# GPU+CPU 하이브리드 엔진 프로세스 스폰
# ============================================================================

@contextlib.contextmanager
def launch_hybrid_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
) -> Iterator[tuple[HybridEngineProcManager, None, EngineZmqAddresses]]:
    """GPU + CPU 하이브리드 엔진 프로세스를 스폰.

    GPU: EngineCoreProc.run_engine_core (engine_index=0)
    CPU: run_cpu_engine_core (engine_index=1)

    launch_core_engines() 패턴을 재활용하되, 두 엔진 프로세스를 스폰합니다.
    핸드셰이크 소켓은 엔진 시작 전에 바인딩됩니다.
    """
    import zmq

    from vllm.utils import zmq_socket_ctx
    from vllm.v1.engine.core import EngineCoreProc

    # ZMQ 주소 설정 (동일 머신, IPC 사용)
    addresses = EngineZmqAddresses(
        inputs=[get_engine_client_zmq_addr(True, "")],
        outputs=[get_engine_client_zmq_addr(True, "")],
    )

    handshake_address = get_open_zmq_ipc_path()

    # 핸드셰이크 소켓을 먼저 바인딩 (launch_core_engines 패턴 동일)
    with zmq_socket_ctx(handshake_address, zmq.ROUTER,
                        bind=True) as handshake_socket:

        common_kwargs = {
            "vllm_config": vllm_config,
            "local_client": True,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
        }

        context = get_mp_context()

        # GPU 프로세스 스폰 (engine_index=0, 기존 run_engine_core)
        gpu_proc = context.Process(
            target=EngineCoreProc.run_engine_core,
            name="GPU_EngineCore_0",
            kwargs=common_kwargs,
        )

        # CPU 엔진 수 결정 (기본 1)
        num_cpu_engines = max(1, vllm_config.hybrid_config.num_cpu_engines)

        # NUMA 노드 할당 (num_cpu_engines > 1이면 NUMA 노드별로 분배)
        cpu_numa_nodes: list[int] = []
        if num_cpu_engines > 1:
            try:
                from vllm.platforms.intel_cpu_utils import NUMAAllocator
                alloc = NUMAAllocator()
                if alloc.is_available and alloc.num_nodes >= num_cpu_engines:
                    cpu_numa_nodes = list(range(num_cpu_engines))
                    logger.info(
                        "Multi-CPU engines: assigning NUMA nodes %s",
                        cpu_numa_nodes)
                else:
                    cpu_numa_nodes = [-1] * num_cpu_engines
                    logger.warning(
                        "NUMA nodes (%d) < num_cpu_engines (%d), "
                        "disabling per-engine NUMA binding",
                        alloc.num_nodes if alloc.is_available else 0,
                        num_cpu_engines)
            except Exception as e:
                cpu_numa_nodes = [-1] * num_cpu_engines
                logger.debug("NUMA detection failed: %s", e)
        else:
            cpu_numa_nodes = [-1]  # single CPU engine: use hybrid_config.numa_bind_node

        # CPU 프로세스 스폰 (engine_index=1,2,..., 각 NUMA 노드)
        cpu_procs = []
        for i in range(num_cpu_engines):
            engine_index = i + 1
            numa_node = cpu_numa_nodes[i] if i < len(cpu_numa_nodes) else -1
            cpu_proc_kwargs = dict(common_kwargs)
            cpu_proc_kwargs["engine_index"] = engine_index
            if numa_node >= 0:
                cpu_proc_kwargs["numa_node"] = numa_node
            cpu_proc = context.Process(
                target=run_cpu_engine_core,
                name=f"CPU_EngineCore_{engine_index}",
                kwargs=cpu_proc_kwargs,
            )
            cpu_procs.append(cpu_proc)

        all_procs = [gpu_proc] + cpu_procs
        manager = HybridEngineProcManager(all_procs)

        try:
            gpu_proc.start()
            for cpu_proc in cpu_procs:
                cpu_proc.start()
        except Exception:
            manager.close()
            raise

        # finished check
        if manager.finished_procs():
            manager.close()
            raise RuntimeError(
                "Hybrid engine process(es) exited immediately: "
                f"{manager.finished_procs()}")

        yield manager, None, addresses

        # 핸드셰이크 대기 (GPU 1 + CPU N = N+1 엔진)
        total_engines = 1 + num_cpu_engines
        engines_to_handshake = [
            CoreEngine(index=i, local=True) for i in range(total_engines)
        ]

        hybrid_parallel = copy.copy(vllm_config.parallel_config)
        hybrid_parallel.data_parallel_size_local = total_engines

        wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            hybrid_parallel,
            vllm_config.cache_config,
            manager,
            None,  # coordinator process
        )
