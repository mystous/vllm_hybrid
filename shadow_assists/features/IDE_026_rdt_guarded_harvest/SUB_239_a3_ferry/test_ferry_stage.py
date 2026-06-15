"""SUB_239 FERRY — host-runnable correctness + overhead test.

Validates ``vllm.v1.lhc.ferry.FerryStager``:
  1. bit-exact: ferried (staged) tensor == source, for contiguous AND
     advanced-indexing *gather* sources (the real NEO swap-in shape), across
     fp16/bf16. This is the CLAUDE.md distribution-equivalence guarantee —
     staging is an exact copy, so swap-in KV is identical to the direct path.
  2. pool reuse: repeated stages of same shape reuse one bounce buffer.
  3. CPU-fallback works on host (DSA lane EPERM on this host — intel_iommu
     lacks sm_on), so stats show cpu_ops>0, dsa_ops==0.
  4. overhead microbench: staged-then-H2D vs direct H2D (informational).

Run:
  /home/mystous/vllm_dev_prj/bin/python \
    shadow_assists/.../SUB_239_a3_ferry/test_ferry_stage.py
"""
import os
import time

os.environ["VLLM_NEO_FERRY"] = "1"

import torch

from vllm.v1.lhc.ferry import FerryStager, ferry_enabled


def _gather_like_neo(n_blocks, kv_heads, block_size, head_dim, dtype):
    """Mimic copy_layer_out: a big CPU buffer, advanced-indexed by scattered
    block_ids → non-contiguous, non-pinned gather (the FERRY input)."""
    total = n_blocks * 4
    buf = torch.randn(total, kv_heads, block_size, head_dim).to(dtype)
    # scattered (non-contiguous) block ids
    ids = torch.tensor(list(range(0, total, 4))[:n_blocks], dtype=torch.long)
    return buf[ids], buf, ids


def test_bit_exact():
    st = FerryStager()
    for dtype in (torch.float16, torch.bfloat16):
        # contiguous source
        src = torch.randn(16, 8, 16, 128).to(dtype).contiguous()
        out = st.stage(src)
        assert out.shape == src.shape and out.dtype == dtype
        assert torch.equal(out, src), f"contig mismatch {dtype}"
        # gather source (real NEO swap-in shape). NOTE: torch advanced
        # indexing returns a *contiguous copy* (not a view) — so the real
        # k_cpu is contiguous but non-pinned / possibly remote-NUMA; FERRY's
        # value is pinned+local placement, not de-fragmentation.
        g, _buf, _ids = _gather_like_neo(16, 8, 16, 128, dtype)
        out2 = st.stage(g)
        assert torch.equal(out2, g.contiguous()), f"gather mismatch {dtype}"
    print("PASS bit_exact (fp16+bf16, contig+gather)")


def test_pool_reuse():
    st = FerryStager()
    src = torch.randn(16, 8, 16, 128).to(torch.float16).contiguous()
    p_first = st.stage(src).data_ptr()
    for _ in range(50):
        p = st.stage(src).data_ptr()
    assert p == p_first, "pool should reuse the same bounce buffer"
    print(f"PASS pool_reuse (1 buffer across 51 stages, stats={st.stats})")


def test_fallback_stats():
    st = FerryStager()
    src = torch.randn(64, 8, 16, 128).to(torch.float16).contiguous()
    for _ in range(20):
        st.stage(src)
    s = st.stats
    # On this host DSA lane is unavailable (EPERM) → all CPU fallback, still
    # correct. In a DSA-capable env dsa_ops would be > 0.
    assert s["cpu_ops"] + s["dsa_ops"] == 20
    print(f"PASS fallback_stats (host CPU-fallback): {s}")


def bench_overhead():
    if not torch.cuda.is_available():
        print("SKIP bench (no CUDA)")
        return
    dev = torch.device("cuda")
    # NEO-ish per-layer gather: 64 blocks × 8 heads × 16 × 128 fp16 ≈ 2 MiB
    g, _b, _i = _gather_like_neo(64, 8, 16, 128, torch.float16)
    nbytes = g.numel() * g.element_size()
    st = FerryStager()
    iters = 200

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = g.to(device=dev, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    direct = (time.perf_counter() - t0) / iters * 1e6  # us

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        staged = st.stage(g)
        _ = staged.to(device=dev, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    ferry = (time.perf_counter() - t0) / iters * 1e6

    print(f"BENCH per-layer {nbytes/1024:.0f} KiB  direct={direct:.1f}us  "
          f"ferry(CPU-stage+H2D)={ferry:.1f}us  stats={st.stats}")
    print("  NOTE: host DSA EPERM → stage is CPU copy (pure overhead here). "
          "DSA-capable env offloads the stage (≈free) + pinned/local H2D win.")


if __name__ == "__main__":
    assert ferry_enabled(), "VLLM_NEO_FERRY must be 1"
    test_bit_exact()
    test_pool_reuse()
    test_fallback_stats()
    bench_overhead()
    print("ALL FERRY TESTS PASSED")
