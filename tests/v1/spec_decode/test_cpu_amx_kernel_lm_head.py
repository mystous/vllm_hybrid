# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase A3-real (SUB_198) — real Qwen-0.5B lm_head integration smoke.

Validates:
  • safetensors loading of Qwen-0.5B's tied embed_tokens.weight as the
    lm_head weight (tie_word_embeddings=true).
  • new C ABI `amx_draft_qwen05b_load_lm_head` is bound and accepts the
    zero-pad (151_936 → 152_064) of the real weight.
  • AMX kernel forward with real weight produces argmax token ids that
    match a PyTorch CPU reference at top-1 / top-3 thresholds (BF16
    precision tolerance, "distributional similarity" — see
    CLAUDE.md §Constraint operating reinterpretation).

Run (worktree-isolated):
  OMP_NUM_THREADS=16 /workspace/vllm_dev_prj/bin/python \\
    tests/v1/spec_decode/test_cpu_amx_kernel_lm_head.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

WORKTREE_ROOT = Path("/workspace/poc_worktrees/wt_a1_cpudraft")
WORKTREE_KERNEL_PATH = (
    WORKTREE_ROOT / "vllm" / "v1" / "spec_decode" / "cpu_amx_kernel.py"
)

# HF cache — Qwen2.5-0.5B-Instruct (config: tie_word_embeddings=true,
# hidden=896, vocab=151,936).
QWEN_SAFETENSORS = Path(
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors"
)
QWEN_HIDDEN = 896
QWEN_VOCAB = 151_936
KERNEL_VOCAB_EXPECTED = 152_064


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────

def _load_kernel_module():
    for mod_name in ("vllm", "vllm.v1", "vllm.v1.spec_decode"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    spec = importlib.util.spec_from_file_location(
        "wt_cpu_amx_kernel", WORKTREE_KERNEL_PATH
    )
    assert spec and spec.loader, f"cannot load {WORKTREE_KERNEL_PATH}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bf16_tensor_to_uint16(t: torch.Tensor) -> np.ndarray:
    """Convert BF16 torch tensor to uint16 numpy view (bit-pattern)."""
    assert t.dtype == torch.bfloat16
    # view as int16 then reinterpret as uint16 (no value change)
    return t.contiguous().view(torch.uint16).cpu().numpy()


def _load_qwen_lm_head_bf16() -> tuple[np.ndarray, dict]:
    """Return (weight_uint16[V,H], meta) for the tied lm_head weight."""
    assert QWEN_SAFETENSORS.exists(), (
        f"safetensors not found: {QWEN_SAFETENSORS}")
    with safe_open(str(QWEN_SAFETENSORS), framework="pt") as f:
        keys = list(f.keys())
        # tie_word_embeddings=true ⇒ no explicit lm_head.weight key
        has_lm_head = "lm_head.weight" in keys
        emb = f.get_tensor("model.embed_tokens.weight")
    meta = {
        "path": str(QWEN_SAFETENSORS),
        "key_used": ("lm_head.weight" if has_lm_head
                     else "model.embed_tokens.weight (tied)"),
        "tie_word_embeddings": not has_lm_head,
        "shape": tuple(emb.shape),
        "dtype": str(emb.dtype),
        "total_keys": len(keys),
    }
    assert emb.shape == (QWEN_VOCAB, QWEN_HIDDEN), (
        f"expected ({QWEN_VOCAB},{QWEN_HIDDEN}), got {tuple(emb.shape)}")
    assert emb.dtype == torch.bfloat16, f"expected bf16, got {emb.dtype}"
    return _bf16_tensor_to_uint16(emb), meta


def _make_inputs_bf16(B: int, H: int, seed: int) -> tuple[np.ndarray,
                                                           torch.Tensor]:
    """Random BF16 hidden state, returned both as numpy uint16 and torch."""
    rng = np.random.default_rng(seed)
    # Range matches realistic Qwen last-hidden post-norm magnitudes
    # (~unit variance after RMSNorm) — bigger than the [-0.05, 0.05]
    # init range used in shape/dtype smoke. Larger range gives clearly
    # separated argmax over real vocab.
    fp32 = rng.normal(0.0, 1.0, size=(B, H)).astype(np.float32)
    t_bf16 = torch.from_numpy(fp32).to(torch.bfloat16)
    return _bf16_tensor_to_uint16(t_bf16), t_bf16


def _topk_match_rates(amx_ids: np.ndarray, ref_topk: np.ndarray
                      ) -> tuple[float, float]:
    """Return (top1_rate, top3_rate). amx_ids:(N,), ref_topk:(N,3)."""
    n = amx_ids.shape[0]
    top1 = float(np.mean(amx_ids == ref_topk[:, 0]))
    top3 = float(np.mean(np.any(amx_ids[:, None] == ref_topk[:, :3],
                                axis=1)))
    return top1, top3


# ─────────────────────────────────────────────────────────────────────
# cases
# ─────────────────────────────────────────────────────────────────────

def case_load_lm_head_symbol_bound() -> None:
    print("=" * 72)
    print("CASE — load_lm_head symbol bound + signature set")
    print("=" * 72)
    sys.modules.pop("wt_cpu_amx_kernel", None)
    mod = _load_kernel_module()
    kern = mod.AmxDraftKernel.get()
    print(f"  lib: {kern.lib_path}")
    print(f"  hw_amx={kern.hw_amx} loaded={kern.loaded}")
    assert kern.loaded, f"dlopen failed: {kern.load_error}"
    assert kern._has_load_lm_head, (
        "load_lm_head symbol not bound — rebuild the .so?")
    sig = kern._lib.amx_draft_qwen05b_load_lm_head
    assert sig.restype is not None, "restype should be c_int"
    assert len(sig.argtypes) == 4, (
        f"load_lm_head should have 4 argtypes, got {len(sig.argtypes)}")
    print("  [PASS] load_lm_head symbol bound + 4 argtypes set")


def case_load_lm_head_real_qwen() -> None:
    print("=" * 72)
    print("CASE — load real Qwen-0.5B lm_head + forward vs PyTorch ref")
    print("=" * 72)
    sys.modules.pop("wt_cpu_amx_kernel", None)
    mod = _load_kernel_module()
    kern = mod.AmxDraftKernel.get()

    if not kern.is_available():
        print("  [SKIP] AMX not available on this CPU")
        return

    rc = kern.ensure_init()
    assert rc == 0, f"init rc={rc}"

    # ── 1. safetensors load ──────────────────────────────────────
    w_uint16, meta = _load_qwen_lm_head_bf16()
    print(f"  safetensors: {meta['path']}")
    print(f"    key={meta['key_used']}, "
          f"tied={meta['tie_word_embeddings']}, "
          f"shape={meta['shape']}, dtype={meta['dtype']}, "
          f"total_keys={meta['total_keys']}")
    assert w_uint16.shape == (QWEN_VOCAB, QWEN_HIDDEN)
    assert w_uint16.dtype == np.uint16

    # ── 2. load into kernel (zero-pad 151,936 → 152,064 inside) ──
    rc_l = kern.load_lm_head(w_uint16)
    print(f"  load_lm_head rc={rc_l} (V_valid={QWEN_VOCAB} → "
          f"padded={mod.KERNEL_VOCAB})")
    assert rc_l == 0

    # ── 3. build PyTorch reference lm_head ───────────────────────
    # We do the matmul in FP32 with BF16 inputs — same as the AMX
    # accumulator path. ref_topk via FP32 logits over [0, V_valid).
    w_torch_bf16 = torch.from_numpy(w_uint16.view(np.int16)).view(
        torch.bfloat16).clone()  # [V, H]

    # ── 4. drive 8 different inputs, B=1, K=1 each ───────────────
    n_inputs = 8
    B_test, K_test = 1, 1
    H = mod.KERNEL_HIDDEN
    Vc = mod.CONFIG_VOCAB_QWEN_05B

    all_amx_ids: list[int] = []
    all_ref_top3: list[np.ndarray] = []
    logprob_max_abs_diffs: list[float] = []

    for i in range(n_inputs):
        inp_uint16, inp_t_bf16 = _make_inputs_bf16(B_test, H, 0x1000 + i)

        # AMX forward
        logits_amx, ids_amx = kern.forward(inp_uint16, B_test, K_test)
        # logits_amx:(B,K,V_padded) uint16(BF16), ids_amx:(B,K) int32
        assert logits_amx.shape == (B_test, K_test, mod.KERNEL_VOCAB)
        assert ids_amx.shape == (B_test, K_test)

        # PyTorch ref: fp32(act_bf16) @ w_bf16.T → fp32 logits
        with torch.no_grad():
            ref_logits = torch.nn.functional.linear(
                inp_t_bf16.to(torch.float32),
                w_torch_bf16.to(torch.float32))  # [B, V]
        ref_logits_np = ref_logits.cpu().numpy()  # [B, V_valid]
        ref_top3 = np.argsort(-ref_logits_np, axis=1)[:, :3]  # [B, 3]

        # AMX logits — reinterpret BF16 → FP32, then slice to V_valid
        amx_logits_bf16 = logits_amx[:, 0, :Vc]  # [B, V_valid] uint16
        amx_logits_fp32 = (amx_logits_bf16.astype(np.uint32) << 16) \
            .view(np.float32)  # [B, V_valid]

        # Convert to logprobs (softmax) for distributional diff.
        # Use fp64 for stable softmax.
        def _logprobs(x: np.ndarray) -> np.ndarray:
            xs = x.astype(np.float64)
            xs -= xs.max(axis=1, keepdims=True)
            ex = np.exp(xs)
            denom = ex.sum(axis=1, keepdims=True)
            return np.log(ex / denom)

        lp_amx = _logprobs(amx_logits_fp32)
        lp_ref = _logprobs(ref_logits_np)
        lp_mad = float(np.max(np.abs(lp_amx - lp_ref)))
        logprob_max_abs_diffs.append(lp_mad)

        all_amx_ids.append(int(ids_amx[0, 0]))
        all_ref_top3.append(ref_top3[0])

        print(f"  input #{i}: amx_id={int(ids_amx[0,0])}, "
              f"ref_top3={ref_top3[0].tolist()}, "
              f"logprob_max_abs_diff={lp_mad:.4f}")

    # ── 5. aggregate metrics ────────────────────────────────────
    amx_ids_np = np.array(all_amx_ids, dtype=np.int64)
    ref_top3_np = np.stack(all_ref_top3, axis=0)
    top1_rate, top3_rate = _topk_match_rates(amx_ids_np, ref_top3_np)
    mean_lp_mad = float(np.mean(logprob_max_abs_diffs))
    max_lp_mad = float(np.max(logprob_max_abs_diffs))

    print("─" * 72)
    print(f"  AGGREGATE over {n_inputs} inputs:")
    print(f"    top-1 match rate: {top1_rate*100:.1f}%  "
          f"(threshold ≥ 90.0%)")
    print(f"    top-3 match rate: {top3_rate*100:.1f}%  "
          f"(threshold ≥ 95.0%)")
    print(f"    logprob max-abs-diff: mean={mean_lp_mad:.4f}, "
          f"max={max_lp_mad:.4f}")

    # PASS gates — operational reinterpretation per CLAUDE.md §Constraint
    # (distributional similarity, not bit-exact).
    assert top1_rate >= 0.90, (
        f"top-1 match {top1_rate*100:.1f}% < 90%")
    assert top3_rate >= 0.95, (
        f"top-3 match {top3_rate*100:.1f}% < 95%")
    print("  [PASS] real Qwen lm_head AMX forward matches PyTorch "
          "reference within BF16 tolerance")


# pytest entry points
def test_load_lm_head_symbol_bound():
    case_load_lm_head_symbol_bound()


def test_load_lm_head_real_qwen():
    case_load_lm_head_real_qwen()


def main() -> int:
    cases = [
        case_load_lm_head_symbol_bound,
        case_load_lm_head_real_qwen,
    ]
    failures: list[tuple[str, BaseException]] = []
    for fn in cases:
        try:
            fn()
        except BaseException as e:
            failures.append((fn.__name__, e))
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print("=" * 72)
    if failures:
        print(f"FAILED: {len(failures)} case(s)")
        return 1
    print(f"ALL {len(cases)} REAL-LM_HEAD CASES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
