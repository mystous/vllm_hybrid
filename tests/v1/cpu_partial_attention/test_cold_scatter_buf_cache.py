# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TSK_002 §4.6 — _COLD_SCATTER_BUFS cache contract.

cache key 가 잘못 잡히면 한 layer 의 scatter buffer 를 다른 layer 가 그대로
재사용해 ``index_copy_`` 가 ``Source/destination tensor must have same slice
shapes`` 로 터지거나 (이전 회귀, 컨텍스트 압축 이전 incident), 더 나쁘게는
silent shape mismatch 로 잘못된 결과가 merge 단계로 흘러간다. 본 파일은:

* key tuple ``(dev_idx, num_q_heads, head_dim, output_dtype, lse_dtype)`` 의
  각 차원이 cache 분리에 실제로 기여하는지
* 동일 key 의 ``num_tokens`` 가 늘어나면 in-place 로 새 텐서를 잡고, 줄어들면
  기존 텐서를 그대로 쓰는지 (no-shrink 의도)
* ``_set_cold_scatter_dirty`` 가 마지막에 기록한 인덱스만 보존하고 buffer
  텐서들은 그대로 두는지

를 dispatcher 호출 없이 helper 만 직접 검증한다 (CPU device, CUDA 무관).
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.backends.flash_attn import (
    _COLD_SCATTER_BUFS,
    _get_cold_scatter_buffers,
    _set_cold_merge_event,
    _set_cold_scatter_dirty,
)


@pytest.fixture(autouse=True)
def _isolate_cache():
    """각 테스트가 깨끗한 cache 상태에서 시작하도록 격리."""
    saved = dict(_COLD_SCATTER_BUFS)
    _COLD_SCATTER_BUFS.clear()
    try:
        yield
    finally:
        _COLD_SCATTER_BUFS.clear()
        _COLD_SCATTER_BUFS.update(saved)


def test_cache_returns_same_object_on_identical_key():
    dev = torch.device("cpu")
    out1, lse1, dirty1, ev1 = _get_cold_scatter_buffers(
        dev, num_tokens=8, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    out2, lse2, dirty2, ev2 = _get_cold_scatter_buffers(
        dev, num_tokens=8, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    assert out1.data_ptr() == out2.data_ptr(), "동일 key + 동일 num_tokens 는 동일 텐서를 돌려줘야"
    assert lse1.data_ptr() == lse2.data_ptr()
    assert dirty1 is None and dirty2 is None
    assert ev1 is None and ev2 is None, "fresh entry 는 last_merge_event=None"


def test_cache_grows_on_larger_num_tokens():
    dev = torch.device("cpu")
    out1, *_ = _get_cold_scatter_buffers(
        dev, num_tokens=4, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    out2, *_ = _get_cold_scatter_buffers(
        dev, num_tokens=16, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    assert out2.size(0) == 16
    assert out2.data_ptr() != out1.data_ptr(), "더 큰 num_tokens 요청은 새 alloc"


def test_cache_no_shrink_on_smaller_num_tokens():
    dev = torch.device("cpu")
    out1, *_ = _get_cold_scatter_buffers(
        dev, num_tokens=64, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    out2, *_ = _get_cold_scatter_buffers(
        dev, num_tokens=8, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    assert out2.size(0) == 64, "더 작은 요청은 기존 텐서를 그대로 사용 (shrink 안 함)"
    assert out2.data_ptr() == out1.data_ptr()


@pytest.mark.parametrize(
    "differ_field",
    ["num_q_heads", "head_dim", "output_dtype", "lse_dtype"],
)
def test_cache_key_separation_per_field(differ_field):
    """cache key 의 각 필드가 다르면 cache entry 가 분리되어야."""
    dev = torch.device("cpu")
    base = dict(
        num_tokens=8,
        num_q_heads=32,
        head_dim=128,
        output_dtype=torch.bfloat16,
        lse_dtype=torch.float32,
    )
    out_a, lse_a, *_ = _get_cold_scatter_buffers(dev, **base)

    variant = dict(base)
    if differ_field == "num_q_heads":
        variant["num_q_heads"] = 16
    elif differ_field == "head_dim":
        variant["head_dim"] = 64
    elif differ_field == "output_dtype":
        variant["output_dtype"] = torch.float16
    elif differ_field == "lse_dtype":
        variant["lse_dtype"] = torch.float16

    out_b, lse_b, *_ = _get_cold_scatter_buffers(dev, **variant)

    # 다른 key → 다른 텐서 객체가 나와야. 그리고 두 entry 가 cache 에 공존.
    assert out_a.data_ptr() != out_b.data_ptr(), (
        f"{differ_field} 가 다른데 같은 buffer 가 재사용됨 — cache key 분리 실패"
    )
    assert lse_a.data_ptr() != lse_b.data_ptr()
    assert len(_COLD_SCATTER_BUFS) == 2


def test_multi_config_coexistence():
    """layer A → layer B → layer A 호출 패턴에서 두 entry 가 공존하고,
    A 의 텐서 객체가 보존되는지."""
    dev = torch.device("cpu")
    out_a1, *_ = _get_cold_scatter_buffers(
        dev, num_tokens=8, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    out_b1, *_ = _get_cold_scatter_buffers(
        dev, num_tokens=8, num_q_heads=8, head_dim=64,
        output_dtype=torch.float16, lse_dtype=torch.float16,
    )
    out_a2, *_ = _get_cold_scatter_buffers(
        dev, num_tokens=8, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    assert out_a1.data_ptr() == out_a2.data_ptr(), "A 의 buffer 가 B 호출 후에도 보존되어야"
    assert out_a1.data_ptr() != out_b1.data_ptr()
    assert len(_COLD_SCATTER_BUFS) == 2


def test_dirty_idx_records_last_write():
    dev = torch.device("cpu")
    out, lse, dirty0, _ev0 = _get_cold_scatter_buffers(
        dev, num_tokens=8, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    assert dirty0 is None

    idx_a = torch.tensor([0, 2, 5], dtype=torch.long)
    _set_cold_scatter_dirty(
        dev, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
        dirty_idx=idx_a,
    )
    out2, lse2, dirty1, _ = _get_cold_scatter_buffers(
        dev, num_tokens=8, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    assert dirty1 is idx_a
    assert out2.data_ptr() == out.data_ptr() and lse2.data_ptr() == lse.data_ptr()

    # 다음 호출이 새 dirty_idx 로 덮어쓰면 이전 idx 는 사라져야.
    idx_b = torch.tensor([1, 3], dtype=torch.long)
    _set_cold_scatter_dirty(
        dev, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
        dirty_idx=idx_b,
    )
    _, _, dirty2, _ = _get_cold_scatter_buffers(
        dev, num_tokens=8, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    assert dirty2 is idx_b


def test_initial_lse_buffer_is_neg_inf():
    """fresh alloc 시 cold_lse_buf 는 -inf 로 채워져 있어야 — merge 단계에서
    아직 작성 안 된 row 가 자연스럽게 cold side 에서 무시됨."""
    dev = torch.device("cpu")
    _, lse, *_ = _get_cold_scatter_buffers(
        dev, num_tokens=4, num_q_heads=8, head_dim=64,
        output_dtype=torch.float16, lse_dtype=torch.float32,
    )
    assert torch.all(torch.isinf(lse) & (lse < 0)), "fresh cold_lse_buf 는 모두 -inf"


def test_cache_grow_resets_dirty_idx_to_none():
    """buffer 가 새 alloc 으로 교체되면 이전 dirty_idx + last_merge_event 도
    의미 없으니 None 으로 reset."""
    dev = torch.device("cpu")
    _get_cold_scatter_buffers(
        dev, num_tokens=4, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    _set_cold_scatter_dirty(
        dev, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
        dirty_idx=torch.tensor([0, 1], dtype=torch.long),
    )
    # 더 큰 num_tokens 로 grow 호출 — 새 buffer + dirty_idx + event 모두 None
    _, _, dirty, event = _get_cold_scatter_buffers(
        dev, num_tokens=32, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    assert dirty is None, "grow 시 dirty_idx 가 reset 되어야 — 이전 idx 는 새 buffer 에 stale"
    assert event is None, "grow 시 last_merge_event 도 reset — 이전 event 는 stale buffer 의 것"


def test_set_merge_event_records_in_cache_entry():
    """``_set_cold_merge_event`` 로 저장된 event 가 다음 ``_get`` 의 4번째 슬롯
    으로 그대로 반환되는지 — race fix 의 contract 검증."""
    dev = torch.device("cpu")
    _get_cold_scatter_buffers(
        dev, num_tokens=4, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    sentinel = object()  # event 객체 자리 — 실제 cuda.Event 가 아니어도 dict 가 들고만 있음
    _set_cold_merge_event(
        dev, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
        event=sentinel,  # type: ignore[arg-type]
    )
    *_, last_event = _get_cold_scatter_buffers(
        dev, num_tokens=4, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    assert last_event is sentinel


def test_set_merge_event_preserves_dirty_idx():
    """dirty 와 event setter 가 서로의 슬롯을 덮지 않는지."""
    dev = torch.device("cpu")
    _get_cold_scatter_buffers(
        dev, num_tokens=4, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    idx = torch.tensor([0, 1], dtype=torch.long)
    _set_cold_scatter_dirty(
        dev, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
        dirty_idx=idx,
    )
    sentinel = object()
    _set_cold_merge_event(
        dev, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
        event=sentinel,  # type: ignore[arg-type]
    )
    _, _, dirty, last_event = _get_cold_scatter_buffers(
        dev, num_tokens=4, num_q_heads=32, head_dim=128,
        output_dtype=torch.bfloat16, lse_dtype=torch.float32,
    )
    assert dirty is idx, "_set_cold_merge_event 가 dirty 슬롯을 덮으면 안 됨"
    assert last_event is sentinel, "_set_cold_scatter_dirty 가 event 슬롯을 덮으면 안 됨"


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="device index 분리 검증은 multi-GPU (cuda:0 vs cuda:1) 필요",
)
def test_cache_key_separation_per_cuda_index():
    """동일 (heads, head_dim, dtype) 라도 cuda device index 가 다르면 entry 분리.

    참고: ``_get_cold_scatter_buffers`` 의 ``dev_idx = device.index if ...
    else 0`` 매핑은 ``cpu`` (index=None) 와 ``cuda:0`` (index=0) 을 모두 0
    으로 떨어뜨리므로 그 둘 사이는 의도적으로 cache 를 share 한다 — 실제
    hot path 는 GPU only 이므로 충돌 시나리오 없음. 본 테스트는 prod
    (H100×8) 같은 multi-GPU 환경에서 cuda:0 vs cuda:1 만 검증."""
    out_a, *_ = _get_cold_scatter_buffers(
        torch.device("cuda:0"), num_tokens=4, num_q_heads=8, head_dim=64,
        output_dtype=torch.float16, lse_dtype=torch.float32,
    )
    out_b, *_ = _get_cold_scatter_buffers(
        torch.device("cuda:1"), num_tokens=4, num_q_heads=8, head_dim=64,
        output_dtype=torch.float16, lse_dtype=torch.float32,
    )
    assert out_a.device.index == 0
    assert out_b.device.index == 1
    assert out_a.data_ptr() != out_b.data_ptr()
    assert len(_COLD_SCATTER_BUFS) == 2
