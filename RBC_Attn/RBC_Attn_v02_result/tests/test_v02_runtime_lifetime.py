"""v0.2/P5 — slot 재사용·generation·동적 mask (지시서 §8.1, §8.4, §10.1).

핵심 계약: **버퍼 재사용과 plan 재사용은 다르다.** mask 가 바뀌면 반드시 새로 계획한다.
buffer 만 재사용하고 이전 descriptor 를 재생하면 실패다.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA 없음", allow_module_level=True)

from rbc import data, slots  # noqa: E402

G, D, BK = 8, 64, 128
NQ, NK, SEL = 64, 4096, 8


def _limits(max_nq=NQ, max_tasks=4096, max_slots=8192, max_task_k=65536, max_red=8192):
    return slots.ShapeLimits(max_nq=max_nq, max_tasks=max_tasks, max_slots=max_slots,
                             max_task_k=max_task_k, max_red=max_red, g=G, d=D, bk=BK)


def _input(pattern="random", seed=0):
    s = data.make_synthetic(pattern, hkv=1, nq=NQ, nk=NK, g=G, d=D, bk=BK, sel=SEL,
                            seed=seed)
    q = torch.from_numpy(s["q"][0]).cuda().to(torch.bfloat16)
    k = torch.from_numpy(s["k"][0]).cuda().to(torch.bfloat16)
    v = torch.from_numpy(s["v"][0]).cuda().to(torch.bfloat16)
    qpos = torch.from_numpy(np.ascontiguousarray(s["q_positions"], np.int32)).cuda()
    return s, q, k, v, qpos


def _run(rt, slot, s, q, k, v, qpos, *, policy="q_outer", max_m=32):
    meta = rt.plan_into(s["ptr"][0], s["block_ids"][0], slot, policy=policy, max_m=max_m,
                        causal=bool(s["causal"]))
    out, lse = rt.submit(slot, q, k, v, qpos, scale=float(s["scale"]))
    t = rt.finish(slot)
    return meta, out.clone(), lse.clone(), t


def test_hot_path_makes_no_allocation_or_pin_requests():
    """§8.7: hot path 의 device allocation·host pin 요청 수 = 0."""
    rt = slots.Runtime(_limits(), n_slots=2)
    s, q, k, v, qpos = _input()
    for i in range(5):
        _run(rt, i % 2, s, q, k, v, qpos)
    info = rt.info()
    assert info["host_pin_calls_hotpath"] == 0
    assert info["device_allocation_requests_hotpath"] == 0
    assert info["host_pin_calls_setup"] > 0        # 초기화 때는 있었다는 사실을 구분해 기록


def test_generation_increments_per_plan():
    rt = slots.Runtime(_limits(), n_slots=2)
    s, q, k, v, qpos = _input()
    gens = []
    for i in range(4):
        meta, *_ = _run(rt, 0, s, q, k, v, qpos)
        gens.append(meta["generation"])
    assert gens == [1, 2, 3, 4]
    # slot 1 은 독립적인 generation 을 센다
    meta, *_ = _run(rt, 1, s, q, k, v, qpos)
    assert meta["generation"] == 1


def test_dynamic_mask_reuses_buffer_but_replans():
    """mask 를 바꿔 가며 같은 slot 을 재사용해도 결과가 각 mask 에 맞아야 한다."""
    rt = slots.Runtime(_limits(), n_slots=2)
    refs = {}
    for seed in (0, 1, 2):
        s, q, k, v, qpos = _input(seed=seed)
        ref_out, ref_lse = data.reference_attention(s, head=0)
        _, out, lse, _ = _run(rt, seed % 2, s, q, k, v, qpos)
        o = out.float().cpu().numpy()
        scale = max(1e-6, float(np.abs(ref_out[0]).max()))
        rel = float(np.abs(o - ref_out[0]).max() / scale)
        assert rel <= 5e-3, f"seed{seed} 상대오차 {rel}"
        refs[seed] = o
    # 서로 다른 mask 가 서로 다른 결과를 냈는지 (descriptor 재생이 아니었는지) 확인
    assert not np.allclose(refs[0], refs[1])
    assert not np.allclose(refs[1], refs[2])


def test_replaying_stale_descriptor_is_detected():
    """buffer 만 재사용하고 새 계획을 만들지 않으면 결과가 새 mask 와 다르다.

    이 검사는 '재계획 없이 재생하면 실패' 라는 계약을 실제로 성립시키는지 본다.
    """
    rt = slots.Runtime(_limits(), n_slots=1)
    s0, q0, k0, v0, qpos0 = _input(seed=0)
    s1, q1, k1, v1, qpos1 = _input(seed=1)
    _, out_correct0, _, _ = _run(rt, 0, s0, q0, k0, v0, qpos0)
    # 새 mask 의 Q/K/V 로 '이전 계획' 을 그대로 재생 (금지된 사용)
    stale, lse = rt.submit(0, q1, k1, v1, qpos1, scale=float(s1["scale"]))
    rt.finish(0)
    stale = stale.clone()
    # 올바른 경로: 다시 계획한다
    _, correct1, _, _ = _run(rt, 0, s1, q1, k1, v1, qpos1)
    ref1, _ = data.reference_attention(s1, head=0)
    scale = max(1e-6, float(np.abs(ref1[0]).max()))
    assert float(np.abs(correct1.float().cpu().numpy() - ref1[0]).max() / scale) <= 5e-3
    assert not torch.allclose(stale, correct1), "stale descriptor 재생이 같은 결과를 냈다"


def test_capacity_exceeded_is_explicit_not_silent():
    """용량 초과는 예외로 실패한다. 조용히 재할당하지 않는다."""
    rt = slots.Runtime(_limits(max_task_k=16), n_slots=1)
    s, q, k, v, qpos = _input()
    with pytest.raises(slots.CapacityExceeded):
        rt.plan_into(s["ptr"][0], s["block_ids"][0], 0, policy="q_outer", max_m=32,
                     causal=bool(s["causal"]))
    tiny = slots.Runtime(_limits(max_nq=8), n_slots=1)
    with pytest.raises(slots.CapacityExceeded):
        tiny.plan_into(s["ptr"][0], s["block_ids"][0], 0, policy="q_outer", max_m=32,
                       causal=bool(s["causal"]))


def test_single_h2d_when_full_membership():
    """FULL membership 이면 membership 배열을 보내지 않아 H2D 가 1회다 (§8.3)."""
    rt = slots.Runtime(_limits(), n_slots=1)
    s, q, k, v, qpos = _input("common_private")
    meta, _, _, t = _run(rt, 0, s, q, k, v, qpos, policy="q_outer")
    if meta["full_membership"]:
        assert meta["membership_sent"] is False
        assert t.h2d_copies == 1
    else:
        assert t.h2d_copies == 2
    assert t.h2d_bytes > 0
    # 분해 항목이 서로 다른 것을 센다
    assert t.pack_ms >= 0.0 and t.dma_ms >= 0.0 and t.submit_api_ms >= 0.0


def test_two_slots_do_not_corrupt_each_other():
    """두 slot 을 번갈아 쓰면서 각자 결과가 자기 mask 에 맞아야 한다."""
    rt = slots.Runtime(_limits(), n_slots=2)
    sa, qa, ka, va, pa = _input(seed=3)
    sb, qb, kb, vb, pb = _input(seed=4)
    ra, _ = data.reference_attention(sa, head=0)
    rb, _ = data.reference_attention(sb, head=0)
    rt.plan_into(sa["ptr"][0], sa["block_ids"][0], 0, policy="q_outer", max_m=32,
                 causal=bool(sa["causal"]))
    rt.plan_into(sb["ptr"][0], sb["block_ids"][0], 1, policy="signature_only", max_m=32,
                 causal=bool(sb["causal"]))
    oa, _ = rt.submit(0, qa, ka, va, pa, scale=float(sa["scale"]))
    ob, _ = rt.submit(1, qb, kb, vb, pb, scale=float(sb["scale"]))
    rt.finish(0)
    rt.finish(1)
    for o, ref in ((oa, ra[0]), (ob, rb[0])):
        sc = max(1e-6, float(np.abs(ref).max()))
        assert float(np.abs(o.float().cpu().numpy() - ref).max() / sc) <= 5e-3


def test_empty_rows_are_filled_after_replan():
    """빈 행은 out=0, LSE=-inf. 이전 계획의 값이 남아서는 안 된다."""
    rt = slots.Runtime(_limits(), n_slots=1)
    s, q, k, v, qpos = _input(seed=5)
    _run(rt, 0, s, q, k, v, qpos)
    ptr = np.array(s["ptr"][0], dtype=np.int64)
    ids = np.array(s["block_ids"][0], dtype=np.int32)
    keep = np.ones(len(ids), dtype=bool)
    keep[ptr[2]:ptr[3]] = False
    cnt = np.diff(ptr).copy()
    cnt[2] = 0
    new_ptr = np.zeros(len(ptr), dtype=np.int64)
    new_ptr[1:] = np.cumsum(cnt)
    meta = rt.plan_into(new_ptr, ids[keep], 0, policy="q_outer", max_m=32,
                        causal=bool(s["causal"]))
    out, lse = rt.submit(0, q, k, v, qpos, scale=float(s["scale"]))
    rt.finish(0)
    assert meta["n_empty_rows"] == 1
    assert torch.all(out[2] == 0), "빈 행의 출력이 0 이 아니다"
    assert torch.all(torch.isneginf(lse[2])), "빈 행의 LSE 가 -inf 가 아니다"
