# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TSK_002 §4.5c reload completion sync — `transfer_async` 의 default-stream
wait_event 가 *진짜로* race 를 막는지 검증.

검증 차원:

1. **단위** — load 인 경우 transfer_async 가 default stream 에 wait_event
   를 emit. store 인 경우는 emit 하지 않음. monkeypatch 로 호출 카운트.
2. **통합** — mock end_event 가 deliberately 늦게 record 되도록 만들고,
   default stream 이 그 event 를 wait 하므로 그 후 enqueue 된 kernel 이
   reload 데이터를 read. wait_event 없으면 stale read.
3. **race 시나리오** — swap_blocks_batch 자체를 monkeypatch 로 sleep 으로
   대체해 transfer 시간을 deliberately 길게. wait_event 적용 시 forward
   kernel 이 transfer 끝까지 wait. 미적용 시 stale.
"""

from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="reload sync 검증은 CUDA 필요"
)


def _make_handler():
    """Minimal CpuGpuOffloadingHandler instance for unit-level testing.

    실제 init 은 vLLM platform 의존성이 많으므로 instance 만들고
    필요한 필드만 manually 세팅한다.
    """
    from vllm.v1.kv_offload.worker.cpu_gpu import (
        SingleDirectionOffloadingHandler,
    )
    handler = SingleDirectionOffloadingHandler.__new__(
        SingleDirectionOffloadingHandler
    )
    return handler


@cuda_required
def test_load_transfer_emits_wait_event_on_default_stream(monkeypatch):
    """load 의 transfer_async 가 default stream 에 wait_event(end_event) 를 emit."""
    import numpy as np
    from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec

    device = torch.device("cuda")

    # 작은 GPU/CPU tensors 만들어서 minimal transfer 시나리오
    gpu_tensor = torch.zeros((4, 64), dtype=torch.int8, device=device)
    cpu_tensor = torch.zeros((4, 64), dtype=torch.int8, device="cpu").pin_memory()

    handler = _make_handler()
    handler.src_tensors = [cpu_tensor]
    handler.dst_tensors = [gpu_tensor]
    handler.gpu_to_cpu = False  # load
    handler.src_block_size_factor = 1
    handler.dst_block_size_factor = 1
    handler.tensor_block_size_in_bytes = [64]
    handler.group_block_size_in_bytes = [64]
    handler.transfer_type = ("CPU", "GPU")
    handler._transfer_events = {}
    handler._transfers = __import__("collections").deque()
    handler._stream_pool = []
    handler._event_pool = []
    handler._block_size_in_bytes_arr = np.array([64], dtype=np.int64)

    # default stream 의 wait_event 호출을 spy 하기 위해 current_stream
    # 자체를 monkeypatch — PyTorch 의 cuda.current_stream() 은 호출마다
    # 새 wrapper 를 반환할 수 있어서 인스턴스 attribute patch 는 안
    # 잡힌다. stream 객체를 wrapping 해서 wait_event 만 가로챈다.
    import vllm.v1.kv_offload.worker.cpu_gpu as cpu_gpu_mod

    real_current_stream = torch.cuda.current_stream
    real_default = real_current_stream(device)
    wait_event_calls = []

    class SpyStream:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def wait_event(self, ev):
            wait_event_calls.append(ev)
            return self._wrapped.wait_event(ev)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    spy = SpyStream(real_default)
    monkeypatch.setattr(
        cpu_gpu_mod.torch.cuda, "current_stream", lambda *a, **kw: spy
    )

    src_spec = CPULoadStoreSpec(block_ids=[0, 1])
    dst_spec = GPULoadStoreSpec(
        block_ids=[2, 3], group_sizes=(2,), block_indices=(0,),
    )

    success = handler.transfer_async(job_id=1, transfer_spec=(src_spec, dst_spec))
    torch.cuda.synchronize()

    assert success
    assert len(wait_event_calls) == 1, (
        "load transfer 후 default stream 에 wait_event 가 정확히 1 회 호출되어야"
    )
    # 호출된 event 가 transfer 의 end_event 와 같은 객체
    assert wait_event_calls[0] is handler._transfer_events[1]


@cuda_required
def test_store_transfer_does_not_emit_wait_event(monkeypatch):
    """store (gpu_to_cpu) 는 default stream wait_event 안 emit — 비대칭 contract."""
    import numpy as np
    from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec

    device = torch.device("cuda")
    gpu_tensor = torch.zeros((4, 64), dtype=torch.int8, device=device)
    cpu_tensor = torch.zeros((4, 64), dtype=torch.int8, device="cpu").pin_memory()

    handler = _make_handler()
    handler.src_tensors = [gpu_tensor]
    handler.dst_tensors = [cpu_tensor]
    handler.gpu_to_cpu = True  # store
    handler.src_block_size_factor = 1
    handler.dst_block_size_factor = 1
    handler.tensor_block_size_in_bytes = [64]
    handler.group_block_size_in_bytes = [64]
    handler.transfer_type = ("GPU", "CPU")
    handler._transfer_events = {}
    handler._transfers = __import__("collections").deque()
    handler._stream_pool = []
    handler._event_pool = []
    handler._block_size_in_bytes_arr = np.array([64], dtype=np.int64)

    import vllm.v1.kv_offload.worker.cpu_gpu as cpu_gpu_mod

    real_default = torch.cuda.current_stream(device)
    wait_event_calls = []

    class SpyStream:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def wait_event(self, ev):
            wait_event_calls.append(ev)
            return self._wrapped.wait_event(ev)

        def wait_stream(self, s):
            return self._wrapped.wait_stream(s)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    spy = SpyStream(real_default)
    monkeypatch.setattr(
        cpu_gpu_mod.torch.cuda, "current_stream", lambda *a, **kw: spy
    )

    src_spec = GPULoadStoreSpec(
        block_ids=[0, 1], group_sizes=(2,), block_indices=(0,),
    )
    dst_spec = CPULoadStoreSpec(block_ids=[2, 3])
    handler.transfer_async(job_id=2, transfer_spec=(src_spec, dst_spec))
    torch.cuda.synchronize()

    assert wait_event_calls == [], (
        "store transfer 는 default stream wait_event 를 emit 하지 않아야 — "
        "kernel 이 CPU dest 를 read 하지 않으므로 sync 불필요"
    )


@cuda_required
def test_default_stream_actually_waits_on_slow_load(monkeypatch):
    """deliberately 느린 load transfer 후, default stream 에 enqueue 된
    kernel 이 transfer 가 *끝난 후* 에야 실행되는지 검증.

    이게 'wait_event 가 진짜 작동' 의 본질 검증 — race window 가 열린
    상태에서 sync 가 forward 를 차단하는지 직접 확인.
    """
    import numpy as np
    from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
    import vllm.v1.kv_offload.worker.cpu_gpu as cpu_gpu_mod

    device = torch.device("cuda")

    # destination GPU block 에 sentinel 넣고, transfer 후 실제 데이터로 덮인다
    gpu_tensor = torch.full((4, 64), -1, dtype=torch.int8, device=device)
    cpu_tensor = torch.full((4, 64), 7, dtype=torch.int8, device="cpu").pin_memory()

    handler = _make_handler()
    handler.src_tensors = [cpu_tensor]
    handler.dst_tensors = [gpu_tensor]
    handler.gpu_to_cpu = False
    handler.src_block_size_factor = 1
    handler.dst_block_size_factor = 1
    handler.tensor_block_size_in_bytes = [64]
    handler.group_block_size_in_bytes = [64]
    handler.transfer_type = ("CPU", "GPU")
    handler._transfer_events = {}
    handler._transfers = __import__("collections").deque()
    handler._stream_pool = []
    handler._event_pool = []
    handler._block_size_in_bytes_arr = np.array([64], dtype=np.int64)

    # swap_blocks_batch 자체를 *느린* 변형으로 monkeypatch — transfer stream
    # 에 50M cycles 의 _sleep 을 inject 후 실제 swap. 이러면 end_event 가
    # 그 sleep 후에 record 됨.
    real_swap = cpu_gpu_mod.ops.swap_blocks_batch

    def slow_swap(*args, **kwargs):
        # 현재 stream (= transfer stream) 에 sleep 추가
        torch.cuda._sleep(50_000_000)
        return real_swap(*args, **kwargs)

    monkeypatch.setattr(cpu_gpu_mod.ops, "swap_blocks_batch", slow_swap)

    src_spec = CPULoadStoreSpec(block_ids=[0, 1])
    dst_spec = GPULoadStoreSpec(
        block_ids=[2, 3], group_sizes=(2,), block_indices=(0,),
    )

    # transfer 시작
    success = handler.transfer_async(
        job_id=42, transfer_spec=(src_spec, dst_spec)
    )
    assert success

    # default stream 에 *그 직후* read kernel 을 enqueue. wait_event 가
    # 적용되었으면 이 read 는 transfer 끝나고 실행되어야 함.
    # gpu_tensor[2:4] 가 transfer 의 destination — 즉 write 후 7 이어야.
    read_buf = gpu_tensor[2:4].clone()  # 즉시 read

    torch.cuda.synchronize()

    # transfer 끝났으면 destination 영역이 7. wait_event 가 작동하면
    # read_buf 도 (read kernel 이 transfer 후 실행되었으므로) 7 이어야.
    assert (read_buf == 7).all(), (
        "wait_event 가 default stream 에 적용되었으면 read_buf 가 transfer 후 "
        "값 (7) 을 봐야. -1 이 보이면 race condition — wait_event 가 작동 안 함."
    )


@cuda_required
def test_race_visible_when_wait_event_disabled(monkeypatch):
    """negative test — wait_event 를 no-op 으로 만들면 race 가 deterministic
    하게 보임. fix 의 효과를 입증하는 대조군.

    이 테스트는 *fail 하지 않고* race 를 *expect* 한다 — read_buf 가 -1
    (sentinel, transfer 전 GPU block 의 값) 으로 보임을 assert. 즉 sync
    없으면 forward 가 in-flight transfer 보다 먼저 read 하여 stale 데이터.
    """
    import numpy as np
    from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
    import vllm.v1.kv_offload.worker.cpu_gpu as cpu_gpu_mod

    device = torch.device("cuda")

    gpu_tensor = torch.full((4, 64), -1, dtype=torch.int8, device=device)
    cpu_tensor = torch.full((4, 64), 7, dtype=torch.int8, device="cpu").pin_memory()

    handler = _make_handler()
    handler.src_tensors = [cpu_tensor]
    handler.dst_tensors = [gpu_tensor]
    handler.gpu_to_cpu = False
    handler.src_block_size_factor = 1
    handler.dst_block_size_factor = 1
    handler.tensor_block_size_in_bytes = [64]
    handler.group_block_size_in_bytes = [64]
    handler.transfer_type = ("CPU", "GPU")
    handler._transfer_events = {}
    handler._transfers = __import__("collections").deque()
    handler._stream_pool = []
    handler._event_pool = []
    handler._block_size_in_bytes_arr = np.array([64], dtype=np.int64)

    # wait_event 를 no-op 으로 — fix 가 *없는* 상태 simulation
    real_default = torch.cuda.current_stream(device)

    class NoSyncStream:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def wait_event(self, ev):
            return  # NO-OP — fix 가 사라진 상태

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    no_sync = NoSyncStream(real_default)
    monkeypatch.setattr(
        cpu_gpu_mod.torch.cuda, "current_stream", lambda *a, **kw: no_sync
    )

    # 같은 slow swap monkeypatch
    real_swap = cpu_gpu_mod.ops.swap_blocks_batch

    def slow_swap(*args, **kwargs):
        torch.cuda._sleep(50_000_000)
        return real_swap(*args, **kwargs)

    monkeypatch.setattr(cpu_gpu_mod.ops, "swap_blocks_batch", slow_swap)

    src_spec = CPULoadStoreSpec(block_ids=[0, 1])
    dst_spec = GPULoadStoreSpec(
        block_ids=[2, 3], group_sizes=(2,), block_indices=(0,),
    )
    handler.transfer_async(job_id=99, transfer_spec=(src_spec, dst_spec))

    # default stream 에 즉시 read enqueue. wait_event no-op 이므로
    # transfer 끝나기 *전* 에 read 가 실행됨 → -1 (stale) 을 봄.
    read_buf = gpu_tensor[2:4].clone()
    torch.cuda.synchronize()

    assert (read_buf == -1).all(), (
        "wait_event 가 no-op 인 상태에서 race window 가 열렸으면 forward 가 "
        "transfer 전에 read 하여 sentinel(-1)을 봐야. 7 이 보이면 sync 가 다른 "
        "곳에서 보장된 것 — fix 의 contribution 이 불명확."
    )
