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

    def test_cpu_first_when_slots_available(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4, warmup_requests=0)
        result = router.route("req-1", prompt_len=100)
        assert result == "cpu"
        assert router.cpu_in_flight == 1

    def test_gpu_when_cpu_full(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=2, warmup_requests=0)
        router.route("req-1")
        router.route("req-2")
        result = router.route("req-3")
        assert result == "gpu"
        assert router.cpu_in_flight == 2

    def test_slot_release_on_finish(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=1, warmup_requests=0)
        router.route("req-1")
        assert router.cpu_in_flight == 1
        # Fill up → GPU
        result = router.route("req-2")
        assert result == "gpu"
        # Release CPU slot
        router.on_request_finished("req-1", was_cpu=True, num_tokens=10)
        assert router.cpu_in_flight == 0
        # Now CPU should be available again
        result = router.route("req-3")
        assert result == "cpu"

    def test_all_slots_fill_then_overflow_to_gpu(self):
        N = 8
        router = CapacityAwareRouter(
            cpu_max_num_seqs=N, warmup_requests=0)
        for i in range(N):
            result = router.route(f"req-{i}")
            assert result == "cpu"
        assert router.cpu_in_flight == N
        # Next should go to GPU
        for i in range(5):
            result = router.route(f"overflow-{i}")
            assert result == "gpu"
        assert router.cpu_count == N
        assert router.gpu_count == 5

    def test_gpu_finish_doesnt_change_cpu_slots(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=2, warmup_requests=0)
        router.route("req-1")  # cpu
        router.route("req-2")  # cpu
        router.route("req-3")  # gpu
        router.on_request_finished("req-3", was_cpu=False, num_tokens=5)
        assert router.cpu_in_flight == 2  # unchanged

    def test_cpu_in_flight_never_negative(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=2, warmup_requests=0)
        # Finish without prior route (edge case)
        router.on_request_finished("phantom", was_cpu=True, num_tokens=0)
        assert router.cpu_in_flight == 0  # clamped to 0


class TestCapacityAwareRouterLengthAware:
    """Test the 'length-aware' routing strategy."""

    def test_short_prompt_goes_to_cpu(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="length-aware",
            cpu_prefill_threshold=512,
            warmup_requests=0)
        result = router.route("req-1", prompt_len=100)
        assert result == "cpu"

    def test_long_prompt_goes_to_gpu(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="length-aware",
            cpu_prefill_threshold=512,
            warmup_requests=0)
        result = router.route("req-1", prompt_len=1000)
        assert result == "gpu"

    def test_exact_threshold_goes_to_cpu(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="length-aware",
            cpu_prefill_threshold=512,
            warmup_requests=0)
        result = router.route("req-1", prompt_len=512)
        assert result == "cpu"

    def test_above_threshold_goes_to_gpu(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="length-aware",
            cpu_prefill_threshold=512,
            warmup_requests=0)
        result = router.route("req-1", prompt_len=513)
        assert result == "gpu"

    def test_short_but_cpu_full_goes_to_gpu(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=1,
            routing_strategy="length-aware",
            cpu_prefill_threshold=512,
            warmup_requests=0)
        router.route("req-1", prompt_len=100)  # cpu
        result = router.route("req-2", prompt_len=100)
        assert result == "gpu"  # CPU full


class TestCapacityAwareRouterThroughputAdaptive:
    """Test the 'throughput-adaptive' routing strategy."""

    def test_initial_routing_uses_base_max(self):
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="throughput-adaptive",
            warmup_requests=0)
        result = router.route("req-1", prompt_len=100)
        assert result == "cpu"

    def test_length_threshold_applies(self):
        """throughput-adaptive also checks prompt_len <= threshold."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=4,
            routing_strategy="throughput-adaptive",
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
        router.on_request_finished("req-1", was_cpu=True, num_tokens=10)
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
            router.on_request_finished(f"gpu-{i}", was_cpu=False,
                                       num_tokens=10)
        for i in range(2):
            router.route(f"cpu-{i}")
            router._request_start_times[f"cpu-{i}"] = time.monotonic() - 0.1
            router.on_request_finished(f"cpu-{i}", was_cpu=True,
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
            router.on_request_finished(f"gpu-{i}", was_cpu=False,
                                       num_tokens=5)
        # 1 CPU request
        router.route("cpu-0")
        router._request_start_times["cpu-0"] = time.monotonic() - 0.1
        router.on_request_finished("cpu-0", was_cpu=True, num_tokens=5)

        assert router._warmup_complete is True


class TestCapacityAwareRouterFaultTolerance:
    """Test fault tolerance edge cases."""

    def test_crash_when_cpu_full(self):
        """When C=N at crash time, all new requests go to GPU."""
        router = CapacityAwareRouter(
            cpu_max_num_seqs=2, warmup_requests=0)
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
            cpu_max_num_seqs=4, warmup_requests=0)
        router.route("req-1")  # cpu, C=1
        # Simulate crash with C=1 < N=4
        # Without a watchdog, new requests go to CPU (known limitation)
        result = router.route("post-crash-1")
        assert result == "cpu"  # Known: no health check


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
