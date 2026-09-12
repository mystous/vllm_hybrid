"""GPU 정확성 테스트 — FP64/FP32 참조와 대조.

지시서 §1-3: CUDA 없음으로 skip 된 것은 통과가 아니다.
지시서 §1-5: 수치 실패 시 속도 측정을 진행하지 않는다. tolerance 를 결과에 맞춰 넓히지 않는다.

허용오차 근거: 입력이 BF16 (가수 8비트, 상대오차 약 2^-8 = 3.9e-3) 이고 누산은 FP32 다.
따라서 출력의 상대오차 기준을 5e-3, LSE 절대오차 기준을 5e-3 으로 **미리** 정한다.
FP16 입력은 가수 11비트라 더 조인다 (1e-3).
"""

import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA 없음 — 이 파일의 skip 은 통과가 아니다", allow_module_level=True)

from rbc import data, kernels, planner, runtime  # noqa: E402

POLICIES = ["rbc", "signature_only", "q_outer", "kv_outer"]

# 사전에 정한 허용오차 (결과를 보고 넓히지 않는다)
TOL = {
    "bfloat16": dict(rel=5e-3, lse=5e-3),
    "float16": dict(rel=1e-3, lse=1e-3),
}
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _run(sample, head, policy, dtype="bfloat16", max_m=128, min_tasks=0,
         block_m=None, cost=None):
    ptr, ids = sample["ptr"][head], sample["block_ids"][head]
    g, d = sample["q"].shape[2], sample["q"].shape[3]
    p = planner.plan_head(ptr, ids, bk=sample["bk"], d=d, g=g, max_m=max_m,
                          policy=policy, min_tasks=min_tasks, cost=cost)
    ok, msg = planner.verify_exact_partition(p, ptr, ids)
    assert ok, f"계획이 edge 계약을 위반: {msg}"
    dp = runtime.upload_plan(p, block_m=block_m, nq=sample['q'].shape[1])
    td = DTYPES[dtype]
    q = torch.from_numpy(sample["q"][head]).cuda().to(td)
    k = torch.from_numpy(sample["k"][head]).cuda().to(td)
    v = torch.from_numpy(sample["v"][head]).cuda().to(td)
    qpos = torch.from_numpy(sample["q_positions"]).cuda().to(torch.int32)
    out, lse = kernels.run_plan(dp, q, k, v, qpos, bk=sample["bk"],
                                causal=sample["causal"], scale=float(sample["scale"]))
    return p, out.float().cpu().numpy(), lse.float().cpu().numpy()


def _check(out, lse, ref_out, ref_lse, dtype, label=""):
    assert np.isfinite(out).all(), f"{label}: 출력에 NaN/Inf 오염"
    finite = np.isfinite(ref_lse)
    assert np.isfinite(lse[finite]).all(), f"{label}: LSE 에 NaN/Inf 오염"
    # 빈 행은 out=0, lse=-inf 로 정의된다
    assert np.all(out[~finite] == 0.0), f"{label}: 빈 행의 출력이 0 이 아니다"
    assert np.all(np.isneginf(lse[~finite])), f"{label}: 빈 행의 LSE 가 -inf 가 아니다"

    scale = max(1e-6, float(np.abs(ref_out).max()))
    rel = float(np.abs(out - ref_out).max() / scale)
    lse_err = float(np.abs(lse[finite] - ref_lse[finite]).max()) if finite.any() else 0.0
    tol = TOL[dtype]
    assert rel <= tol["rel"], f"{label}: 출력 상대오차 {rel:.3e} > 허용 {tol['rel']:.0e}"
    assert lse_err <= tol["lse"], f"{label}: LSE 오차 {lse_err:.3e} > 허용 {tol['lse']:.0e}"
    return rel, lse_err


# --------------------------------------------------------------------------
# 1. 정책별 정확성 (4 패턴 × 4 정책)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("pattern,policy", list(itertools.product(
    ["common_private", "random", "clustered", "disjoint"], POLICIES)))
def test_policy_correctness(pattern, policy):
    sel = 8 if pattern == "disjoint" else 16
    nq = 8 if pattern in ("common_private", "disjoint") else 16
    s = data.make_synthetic(pattern, hkv=1, nq=nq, nk=65536, g=8, d=64, bk=128,
                            sel=sel, seed=17, causal=True)
    ref_out, ref_lse = data.reference_attention(s, head=0)
    _, out, lse = _run(s, 0, policy)
    _check(out, lse, ref_out[0], ref_lse[0], "bfloat16", f"{pattern}/{policy}")


# --------------------------------------------------------------------------
# 2. 정책 간 일치 — 서로 다른 분해가 같은 값을 내야 한다
# --------------------------------------------------------------------------
def test_policies_agree_with_each_other():
    s = data.make_synthetic("common_private", hkv=1, nq=8, nk=65536, g=16, d=128,
                            bk=128, sel=16, seed=19, causal=True)
    ref = {}
    for pol in POLICIES:
        _, out, lse = _run(s, 0, pol)
        ref[pol] = (out, lse)
    base = ref["kv_outer"]
    for pol in POLICIES:
        d = float(np.abs(ref[pol][0] - base[0]).max())
        assert d < 2e-3, f"{pol} 와 kv_outer 의 출력 차이 {d:.3e}"


# --------------------------------------------------------------------------
# 3. dtype
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_dtypes(dtype):
    s = data.make_synthetic("random", hkv=1, nq=16, nk=32768, g=8, d=64, bk=128,
                            sel=12, seed=23, causal=True)
    ref_out, ref_lse = data.reference_attention(s, head=0)
    _, out, lse = _run(s, 0, "rbc", dtype=dtype)
    _check(out, lse, ref_out[0], ref_lse[0], dtype, dtype)


# --------------------------------------------------------------------------
# 4. 경계 조건
# --------------------------------------------------------------------------
def test_empty_rows_gpu():
    """빈 선택 행: out=0, lse=-inf (설계 §2.2)."""
    nq, nk, g, d, bk = 6, 8192, 8, 64, 128
    rng = np.random.default_rng(29)
    ptr = np.array([0, 4, 4, 8, 8, 12, 12], dtype=np.int64)   # 1,3,5 는 빈 행
    ids = np.concatenate([np.sort(rng.choice(nk // bk, 4, replace=False)) for _ in range(3)])
    s = dict(q=rng.standard_normal((1, nq, g, d), dtype=np.float32),
             k=rng.standard_normal((1, nk, d), dtype=np.float32),
             v=rng.standard_normal((1, nk, d), dtype=np.float32),
             ptr=ptr[None, :], block_ids=[ids.astype(np.int32)], bk=bk,
             q_positions=np.full(nq, nk - 1, dtype=np.int64), causal=False,
             scale=np.float32(1.0 / np.sqrt(d)))
    ref_out, ref_lse = data.reference_attention(s, head=0)
    for pol in POLICIES:
        _, out, lse = _run(s, 0, pol)
        _check(out, lse, ref_out[0], ref_lse[0], "bfloat16", f"empty/{pol}")


def test_tail_kv_block():
    """마지막 KV block 이 BK 로 나눠지지 않는 경우 (설계 §0 범위)."""
    nk = 65536 + 37
    bk = 128
    nq, g, d = 8, 8, 64
    rng = np.random.default_rng(31)
    n_blocks = (nk + bk - 1) // bk
    rows = [np.sort(np.append(rng.choice(n_blocks - 1, 7, replace=False), n_blocks - 1))
            for _ in range(nq)]
    ptr = np.zeros(nq + 1, dtype=np.int64)
    for i, r in enumerate(rows):
        ptr[i + 1] = ptr[i] + r.size
    s = dict(q=rng.standard_normal((1, nq, g, d), dtype=np.float32),
             k=rng.standard_normal((1, nk, d), dtype=np.float32),
             v=rng.standard_normal((1, nk, d), dtype=np.float32),
             ptr=ptr[None, :], block_ids=[np.concatenate(rows).astype(np.int32)], bk=bk,
             q_positions=np.full(nq, nk - 1, dtype=np.int64), causal=True,
             scale=np.float32(1.0 / np.sqrt(d)))
    ref_out, ref_lse = data.reference_attention(s, head=0)
    for pol in POLICIES:
        _, out, lse = _run(s, 0, pol)
        _check(out, lse, ref_out[0], ref_lse[0], "bfloat16", f"tail/{pol}")


def test_causal_partial_block():
    """q_pos 가 block 중간에 있는 경우 원소 단위 masking."""
    nk, bk, nq, g, d = 4096, 128, 4, 8, 64
    rng = np.random.default_rng(37)
    qpos = np.array([100, 200, 1000, 1500], dtype=np.int64)   # 전부 block 중간
    rows = []
    for i in range(nq):
        hi = int(qpos[i] // bk) + 1
        rows.append(np.arange(0, hi, dtype=np.int32))
    ptr = np.zeros(nq + 1, dtype=np.int64)
    for i, r in enumerate(rows):
        ptr[i + 1] = ptr[i] + r.size
    s = dict(q=rng.standard_normal((1, nq, g, d), dtype=np.float32),
             k=rng.standard_normal((1, nk, d), dtype=np.float32),
             v=rng.standard_normal((1, nk, d), dtype=np.float32),
             ptr=ptr[None, :], block_ids=[np.concatenate(rows)], bk=bk,
             q_positions=qpos, causal=True, scale=np.float32(1.0 / np.sqrt(d)))
    ref_out, ref_lse = data.reference_attention(s, head=0)
    for pol in POLICIES:
        _, out, lse = _run(s, 0, pol)
        _check(out, lse, ref_out[0], ref_lse[0], "bfloat16", f"causal/{pol}")


# --------------------------------------------------------------------------
# 5. M 탐색과 min_tasks — 같은 탐색 기회를 주어도 값이 같아야 한다
# --------------------------------------------------------------------------
@pytest.mark.parametrize("max_m", [16, 32, 64, 128])
def test_max_m_search_same_result(max_m):
    s = data.make_synthetic("random", hkv=1, nq=32, nk=32768, g=8, d=64, bk=128,
                            sel=10, seed=41, causal=True)
    ref_out, ref_lse = data.reference_attention(s, head=0)
    _, out, lse = _run(s, 0, "rbc", max_m=max_m)
    _check(out, lse, ref_out[0], ref_lse[0], "bfloat16", f"max_m={max_m}")


@pytest.mark.parametrize("min_tasks", [0, 64, 256])
def test_min_tasks_same_result(min_tasks):
    s = data.make_synthetic("common_private", hkv=1, nq=8, nk=65536, g=16, d=128,
                            bk=128, sel=16, seed=43, causal=True)
    ref_out, ref_lse = data.reference_attention(s, head=0)
    _, out, lse = _run(s, 0, "rbc", min_tasks=min_tasks)
    _check(out, lse, ref_out[0], ref_lse[0], "bfloat16", f"min_tasks={min_tasks}")


@pytest.mark.parametrize("block_m", [1, 2, 4, 8])
def test_block_m_tiling_same_result(block_m):
    """큰 task 를 여러 CTA 로 쪼개도 (query 타일링) 값이 같아야 한다."""
    s = data.make_synthetic("common_private", hkv=1, nq=8, nk=65536, g=8, d=64,
                            bk=128, sel=16, seed=47, causal=True)
    ref_out, ref_lse = data.reference_attention(s, head=0)
    _, out, lse = _run(s, 0, "rbc", block_m=block_m)
    _check(out, lse, ref_out[0], ref_lse[0], "bfloat16", f"block_m={block_m}")


# --------------------------------------------------------------------------
# 6. 여러 KV head — head 별로 다른 mask
# --------------------------------------------------------------------------
def test_multiple_kv_heads_independent():
    s = data.make_synthetic("random", hkv=3, nq=12, nk=32768, g=8, d=64, bk=128,
                            sel=10, seed=53, causal=True)
    ref_out, ref_lse = data.reference_attention(s)
    for h in range(3):
        _, out, lse = _run(s, h, "rbc")
        _check(out, lse, ref_out[h], ref_lse[h], "bfloat16", f"head{h}")


# --------------------------------------------------------------------------
# 7. 비용 계수를 바꿔 다른 계획이 나와도 값은 같아야 한다
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cost", [
    dict(flop_weight=0.0), dict(flop_weight=1.0), dict(partial_weight=0.0),
])
def test_cost_variants_same_result(cost):
    s = data.make_synthetic("random", hkv=1, nq=16, nk=32768, g=8, d=64, bk=128,
                            sel=12, seed=59, causal=True)
    ref_out, ref_lse = data.reference_attention(s, head=0)
    _, out, lse = _run(s, 0, "rbc", cost=cost)
    _check(out, lse, ref_out[0], ref_lse[0], "bfloat16", str(cost))
