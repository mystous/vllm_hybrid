# SPDX-License-Identifier: Apache-2.0
"""Unit tests for hybrid core components.

Tests CapacityAwareRouter routing logic, slot management, and strategies,
plus _resolve_cpu_params auto-detection formulas.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.engine.hybrid_core import (
    CapacityAwareRouter,
    ResolvedCpuParams,
    _resolve_cpu_params,
)


# ============================================================================
# CapacityAwareRouter Tests
# ============================================================================

class TestCapacityAwareRouterCapacityStrategy:
    """Test the default 'capacity' routing strategy."""

    def test_gpu_first_by_default(self):
        """Default: GPU-first, GPU 슬롯 여유 시 GPU로."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4, warmup_requests=0)
        result = router.route("req-1", prompt_len=100)
        assert result == "gpu"
        assert router.gpu_in_flight == 1

    def test_cpu_first_when_priority_set(self):
        """CPU-first: CPU 슬롯 여유 시 CPU로."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4, routing_priority="cpu-first",
            warmup_requests=0)
        result = router.route("req-1", prompt_len=100)
        assert result.startswith("cpu")
        assert router.cpu_in_flight == 1

    def test_gpu_first_overflow_to_cpu(self):
        """GPU-first: GPU 포화 시 CPU로 overflow."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4, gpu_max_num_seqs=2, warmup_requests=0)
        router.route("req-1")  # gpu
        router.route("req-2")  # gpu (full)
        result = router.route("req-3")
        assert result.startswith("cpu")
        assert router.gpu_in_flight == 2

    def test_cpu_first_overflow_to_gpu(self):
        """CPU-first: CPU 포화 시 GPU로 overflow."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=2, routing_priority="cpu-first",
            warmup_requests=0)
        router.route("req-1")  # cpu
        router.route("req-2")  # cpu (full)
        result = router.route("req-3")
        assert result == "gpu"
        assert router.cpu_in_flight == 2

    def test_slot_release_on_finish(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=1, gpu_max_num_seqs=1, warmup_requests=0)
        router.route("req-1")  # gpu
        assert router.gpu_in_flight == 1
        # GPU full → CPU
        result = router.route("req-2")
        assert result.startswith("cpu")
        # Release CPU slot
        router.on_request_finished("req-2", engine_path="cpu:0", num_tokens=10)
        assert router.cpu_in_flight == 0

    def test_all_gpu_slots_fill_then_overflow_to_cpu(self):
        N_gpu = 4
        router = CapacityAwareRouter(
            cpu_max_num_seqs=8, gpu_max_num_seqs=N_gpu, warmup_requests=0)
        for i in range(N_gpu):
            result = router.route(f"req-{i}")
            assert result == "gpu"
        assert router.gpu_in_flight == N_gpu
        # Next should go to CPU
        for i in range(3):
            result = router.route(f"overflow-{i}")
            assert result.startswith("cpu")
        assert router.gpu_count == N_gpu
        assert router.cpu_count == 3

    def test_gpu_finish_doesnt_change_cpu_slots(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=2, gpu_max_num_seqs=1,
            routing_priority="cpu-first", warmup_requests=0)
        router.route("req-1")  # cpu
        router.route("req-2")  # cpu
        router.route("req-3")  # gpu (cpu full)
        router.on_request_finished("req-3", engine_path="gpu", num_tokens=5)
        assert router.cpu_in_flight == 2  # unchanged

    def test_cpu_in_flight_never_negative(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=2, warmup_requests=0)
        # Finish without prior route (edge case)
        router.on_request_finished("phantom", engine_path="cpu:0", num_tokens=0)
        assert router.cpu_in_flight == 0  # clamped to 0


class TestCapacityAwareRouterRoundRobin:
    """Test the 'round-robin' routing strategy."""

    def test_alternates_cpu_gpu(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4, gpu_max_num_seqs=4,
            routing_strategy="round-robin", warmup_requests=0)
        results = [router.route(f"r{i}") for i in range(4)]
        # odd=CPU, even=GPU
        assert results[0].startswith("cpu")
        assert results[1] == "gpu"
        assert results[2].startswith("cpu")
        assert results[3] == "gpu"

    def test_cpu_full_falls_back_to_gpu(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=1, gpu_max_num_seqs=10,
            routing_strategy="round-robin", warmup_requests=0)
        router.route("r0")  # cpu (fills up)
        router.route("r1")  # gpu
        result = router.route("r2")  # cpu turn but full → gpu
        assert result == "gpu"

    def test_gpu_full_falls_back_to_cpu(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=10, gpu_max_num_seqs=1,
            routing_strategy="round-robin", warmup_requests=0)
        router.route("r0")  # cpu
        router.route("r1")  # gpu (fills up)
        router.route("r2")  # cpu
        result = router.route("r3")  # gpu turn but full → cpu
        assert result.startswith("cpu")

    def test_ignores_priority_setting(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="round-robin",
            routing_priority="cpu-first",
            warmup_requests=0)
        assert router.routing_priority == "gpu-first"


class TestCapacityAwareRouterLengthAware:
    """Test the 'length-aware' routing strategy."""

    def test_gpu_first_short_prompt_stays_gpu(self):
        """GPU-first: GPU 여유 있으면 짧은 프롬프트도 GPU."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="length-aware",
            cpu_prefill_threshold=512,
            warmup_requests=0)
        result = router.route("req-1", prompt_len=100)
        assert result == "gpu"

    def test_cpu_first_short_prompt_goes_to_cpu(self):
        """CPU-first: 짧은 프롬프트는 CPU 우선."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="length-aware",
            routing_priority="cpu-first",
            cpu_prefill_threshold=512,
            warmup_requests=0)
        result = router.route("req-1", prompt_len=100)
        assert result.startswith("cpu")

    def test_cpu_first_long_prompt_goes_to_gpu(self):
        """CPU-first여도 긴 프롬프트는 GPU로."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="length-aware",
            routing_priority="cpu-first",
            cpu_prefill_threshold=512,
            warmup_requests=0)
        result = router.route("req-1", prompt_len=1000)
        assert result == "gpu"

    def test_above_threshold_goes_to_gpu(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="length-aware",
            routing_priority="cpu-first",
            cpu_prefill_threshold=512,
            warmup_requests=0)
        result = router.route("req-1", prompt_len=513)
        assert result == "gpu"

    def test_short_but_cpu_full_goes_to_gpu(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=1,
            routing_strategy="length-aware",
            routing_priority="cpu-first",
            cpu_prefill_threshold=512,
            warmup_requests=0)
        router.route("req-1", prompt_len=100)  # cpu
        result = router.route("req-2", prompt_len=100)
        assert result == "gpu"  # CPU full


class TestCapacityAwareRouterThroughputAdaptive:
    """Test the 'throughput-adaptive' routing strategy."""

    def test_gpu_first_initial_routing(self):
        """GPU-first: 초기 요청은 GPU로."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="throughput-adaptive",
            warmup_requests=0)
        result = router.route("req-1", prompt_len=100)
        assert result == "gpu"

    def test_cpu_first_initial_routing(self):
        """CPU-first: 초기 요청은 CPU로."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="throughput-adaptive",
            routing_priority="cpu-first",
            warmup_requests=0)
        result = router.route("req-1", prompt_len=100)
        assert result.startswith("cpu")

    def test_length_threshold_applies(self):
        """throughput-adaptive also checks prompt_len <= threshold."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="throughput-adaptive",
            routing_priority="cpu-first",
            cpu_prefill_threshold=256,
            warmup_requests=0)
        result = router.route("req-1", prompt_len=500)
        assert result == "gpu"

    def test_adaptive_slot_adjustment(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="throughput-adaptive",
            warmup_requests=0)
        # Simulate: CPU throughput = GPU throughput → ratio=1.0
        # N = clamp(4 * (1 + 1.0), 2, 8) = 8
        router._cpu_ema_throughput = 10.0
        router._gpu_ema_throughput = 10.0
        router._update_adaptive_slots()
        assert router._adaptive_cpu_max_seqs == 8  # 4 * (1+1) = 8

    def test_adaptive_slot_clamp_minimum(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="throughput-adaptive",
            warmup_requests=0)
        # CPU extremely slow → ratio ≈ 0
        router._cpu_ema_throughput = 0.01
        router._gpu_ema_throughput = 100.0
        router._update_adaptive_slots()
        assert router._adaptive_cpu_max_seqs == max(
            2, int(4 * (1 + 0.01 / 100.0)))

    def test_adaptive_slot_clamp_maximum(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="throughput-adaptive",
            warmup_requests=0)
        # CPU much faster than GPU → clamped to 2*N_max = 8
        router._cpu_ema_throughput = 1000.0
        router._gpu_ema_throughput = 1.0
        router._update_adaptive_slots()
        assert router._adaptive_cpu_max_seqs == 8  # clamped to 2*4

    def test_ema_update_on_finish(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="throughput-adaptive",
            warmup_requests=0)
        router.route("req-1", prompt_len=100)
        # Simulate passage of time
        router._request_start_times["req-1"] = time.monotonic() - 1.0
        router.on_request_finished("req-1", engine_path="cpu:0", num_tokens=10)
        assert router._cpu_ema_throughput > 0


class TestCapacityAwareRouterWarmup:
    """Test warmup profiling logic."""

    def test_warmup_disabled_when_zero(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4, warmup_requests=0)
        assert router._warmup_complete is True

    def test_warmup_completes_normally(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4, warmup_requests=2)
        assert router._warmup_complete is False

        # Complete 2 GPU + 2 CPU requests
        for i in range(2):
            router.route(f"gpu-{i}")
            router._request_start_times[f"gpu-{i}"] = time.monotonic() - 0.1
            router.on_request_finished(f"gpu-{i}", engine_path="gpu",
                                       num_tokens=10)
        for i in range(2):
            router.route(f"cpu-{i}")
            router._request_start_times[f"cpu-{i}"] = time.monotonic() - 0.1
            router.on_request_finished(f"cpu-{i}", engine_path="cpu:0",
                                       num_tokens=10)
        assert router._warmup_complete is True

    def test_warmup_forced_completion(self):
        """GPU 2W + CPU ≥ 1 → early termination."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4, warmup_requests=5)
        assert router._warmup_complete is False

        # Complete 10 GPU requests (2W) + 1 CPU request
        for i in range(10):
            router.route(f"gpu-{i}")
            router._request_start_times[f"gpu-{i}"] = time.monotonic() - 0.1
            router.on_request_finished(f"gpu-{i}", engine_path="gpu",
                                       num_tokens=5)
        # 1 CPU request
        router.route("cpu-0")
        router._request_start_times["cpu-0"] = time.monotonic() - 0.1
        router.on_request_finished("cpu-0", engine_path="cpu:0", num_tokens=5)

        assert router._warmup_complete is True


class TestCapacityAwareRouterFaultTolerance:
    """Test fault tolerance edge cases."""

    def test_crash_when_cpu_full(self):
        """When C=N at crash time, all new requests go to GPU (cpu-first)."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=2, routing_priority="cpu-first",
            warmup_requests=0)
        router.route("req-1")  # cpu, C=1
        router.route("req-2")  # cpu, C=2=N
        # Simulate crash: no on_request_finished calls
        # All new requests should go to GPU
        for i in range(5):
            result = router.route(f"post-crash-{i}")
            assert result == "gpu"

    def test_crash_when_cpu_not_full(self):
        """When C<N at crash time, new requests still go to dead CPU."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4, routing_priority="cpu-first",
            warmup_requests=0)
        router.route("req-1")  # cpu, C=1
        # Simulate crash with C=1 < N=4
        # Without a watchdog, new requests go to CPU (known limitation)
        result = router.route("post-crash-1")
        assert result.startswith("cpu")  # Known: no health check


# ============================================================================
# _resolve_cpu_params Tests
# ============================================================================

class TestResolveCpuParams:
    """Test auto-detection formulas in _resolve_cpu_params."""

    def _make_hybrid_config(self, **kwargs):
        """Create a minimal HybridConfig-like mock."""
        config = MagicMock()
        config.cpu_num_threads = kwargs.get("cpu_num_threads", 0)
        config.cpu_max_num_seqs = kwargs.get("cpu_max_num_seqs", 0)
        config.cpu_kvcache_space_gb = kwargs.get("cpu_kvcache_space_gb", 0)
        config.cpu_max_num_batched_tokens = kwargs.get(
            "cpu_max_num_batched_tokens", 0)
        config.numa_aware = kwargs.get("numa_aware", True)
        config.numa_bind_node = kwargs.get("numa_bind_node", None)
        return config

    def _make_mock_psutil(self, cores=56, total_mem_gb=512):
        """Create a mock psutil module."""
        mock_psutil = MagicMock()
        mock_psutil.cpu_count.return_value = cores
        mock_psutil.virtual_memory.return_value = MagicMock(
            total=total_mem_gb * 1024**3)
        return mock_psutil

    def _run_resolve(self, config, mock_psutil):
        """Run _resolve_cpu_params with mocked psutil and no NUMA."""
        with patch.dict("sys.modules", {
            "psutil": mock_psutil,
            "vllm.platforms.intel_cpu_utils": None,
        }):
            return _resolve_cpu_params(config)

    def test_auto_threads_uses_physical_cores(self):
        """cpu_num_threads=0 → effective_cores (no NUMA)."""
        config = self._make_hybrid_config(numa_aware=False)
        result = self._run_resolve(config, self._make_mock_psutil(cores=56))
        assert result.cpu_num_threads == 56

    def test_auto_max_seqs_formula(self):
        """cpu_max_num_seqs=0 → max(4, cores//4)."""
        config = self._make_hybrid_config(numa_aware=False)
        result = self._run_resolve(config, self._make_mock_psutil(cores=56))
        assert result.cpu_max_num_seqs == max(4, 56 // 4)  # 14

    def test_auto_max_seqs_minimum_4(self):
        """With few cores, min is 4."""
        config = self._make_hybrid_config(numa_aware=False)
        result = self._run_resolve(
            config, self._make_mock_psutil(cores=8, total_mem_gb=64))
        assert result.cpu_max_num_seqs == 4  # max(4, 8//4=2) = 4

    def test_auto_kvcache_formula(self):
        """cpu_kvcache_space_gb=0 → clamp(mem*0.4, 32, 512)."""
        config = self._make_hybrid_config(numa_aware=False)
        result = self._run_resolve(
            config, self._make_mock_psutil(cores=56, total_mem_gb=2048))
        # clamp(2048*0.4, 32, 512) = clamp(819, 32, 512) = 512
        assert result.cpu_kvcache_space_gb == 512

    def test_auto_kvcache_small_memory(self):
        """Small memory → minimum 32GB."""
        config = self._make_hybrid_config(numa_aware=False)
        result = self._run_resolve(
            config, self._make_mock_psutil(cores=8, total_mem_gb=32))
        # clamp(32*0.4, 32, 512) = clamp(12, 32, 512) = 32
        assert result.cpu_kvcache_space_gb == 32

    def test_auto_batched_tokens_formula(self):
        """cpu_max_num_batched_tokens=0 → max_seqs * 256."""
        config = self._make_hybrid_config(numa_aware=False)
        result = self._run_resolve(config, self._make_mock_psutil(cores=56))
        assert result.cpu_max_num_batched_tokens == (
            result.cpu_max_num_seqs * 256)

    def test_manual_override(self):
        """Explicit values override auto-detection."""
        config = self._make_hybrid_config(
            cpu_num_threads=28,
            cpu_max_num_seqs=8,
            cpu_kvcache_space_gb=200,
            cpu_max_num_batched_tokens=4096,
            numa_aware=False,
        )
        result = self._run_resolve(config, self._make_mock_psutil(cores=56))
        assert result.cpu_num_threads == 28
        assert result.cpu_max_num_seqs == 8
        assert result.cpu_kvcache_space_gb == 200
        assert result.cpu_max_num_batched_tokens == 4096


# ============================================================================
# ResolvedCpuParams Tests
# ============================================================================

class TestResolvedCpuParams:
    """Test the ResolvedCpuParams dataclass."""

    def test_fields(self):
        params = ResolvedCpuParams(
            cpu_max_num_seqs=14,
            cpu_kvcache_space_gb=200,
            cpu_max_num_batched_tokens=3584,
            cpu_num_threads=56,
        )
        assert params.cpu_max_num_seqs == 14
        assert params.cpu_kvcache_space_gb == 200
        assert params.cpu_max_num_batched_tokens == 3584
        assert params.cpu_num_threads == 56
