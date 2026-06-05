#!/usr/bin/env python
"""Phase A1 (SUB_198 §p_24layer) — Qwen-0.5B 24-layer AMX CPU forward.

Validates the new C ABI:
  • amx_draft_qwen05b_init_model
  • amx_draft_qwen05b_load_layer_weights × 24
  • amx_draft_qwen05b_load_embed_tokens
  • amx_draft_qwen05b_load_final_norm
  • amx_draft_qwen05b_load_lm_head (reuses SUB_198 A3-real path)
  • amx_draft_qwen05b_layer_forward       (P2 single-layer test)
  • amx_draft_qwen05b_forward_full        (P3 end-to-end + lm_head + argmax)

Run:
  OMP_NUM_THREADS=16 /workspace/vllm_dev_prj/bin/python \\
    shadow_assists/features/IDE_019_multi_source_drafter/\\
    SUB_198_amx_real_integration/p_24layer/test_24layer.py
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

WORKTREE_ROOT = Path("/workspace/poc_worktrees/wt_a1_cpudraft")
LIB_PATH = (
    WORKTREE_ROOT
    / "shadow_assists/features/IDE_019_multi_source_drafter"
    / "SUB_187_amx_draft_head/build/libamx_draft_qwen05b.so"
)
QWEN_SAFETENSORS = Path(
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors"
)
HF_QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"

HIDDEN = 896
Q_DIM = 896
KV_DIM = 128
INTERMEDIATE = 4864
N_LAYERS = 24
VOCAB_VALID = 151936
VOCAB_PADDED = 152064


def _u16(t: torch.Tensor) -> np.ndarray:
    assert t.dtype == torch.bfloat16
    return t.contiguous().view(torch.uint16).cpu().numpy()


def _load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(LIB_PATH))

    lib.amx_draft_qwen05b_init.restype = ctypes.c_int
    lib.amx_draft_qwen05b_init.argtypes = []

    lib.amx_draft_qwen05b_load_lm_head.restype = ctypes.c_int
    lib.amx_draft_qwen05b_load_lm_head.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

    lib.amx_draft_qwen05b_init_model.restype = ctypes.c_int
    lib.amx_draft_qwen05b_init_model.argtypes = []

    lib.amx_draft_qwen05b_free_model.restype = ctypes.c_int
    lib.amx_draft_qwen05b_free_model.argtypes = []

    lib.amx_draft_qwen05b_load_layer_weights.restype = ctypes.c_int
    lib.amx_draft_qwen05b_load_layer_weights.argtypes = [
        ctypes.c_int,                       # layer_idx
        ctypes.POINTER(ctypes.c_uint16),    # ln1_w
        ctypes.POINTER(ctypes.c_uint16),    # q_w
        ctypes.POINTER(ctypes.c_uint16),    # q_b
        ctypes.POINTER(ctypes.c_uint16),    # k_w
        ctypes.POINTER(ctypes.c_uint16),    # k_b
        ctypes.POINTER(ctypes.c_uint16),    # v_w
        ctypes.POINTER(ctypes.c_uint16),    # v_b
        ctypes.POINTER(ctypes.c_uint16),    # o_w
        ctypes.POINTER(ctypes.c_uint16),    # ln2_w
        ctypes.POINTER(ctypes.c_uint16),    # gate_w
        ctypes.POINTER(ctypes.c_uint16),    # up_w
        ctypes.POINTER(ctypes.c_uint16),    # down_w
    ]

    lib.amx_draft_qwen05b_load_embed_tokens.restype = ctypes.c_int
    lib.amx_draft_qwen05b_load_embed_tokens.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int, ctypes.c_int,
    ]

    lib.amx_draft_qwen05b_load_final_norm.restype = ctypes.c_int
    lib.amx_draft_qwen05b_load_final_norm.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
    ]

    lib.amx_draft_qwen05b_layer_forward.restype = ctypes.c_int
    lib.amx_draft_qwen05b_layer_forward.argtypes = [
        ctypes.c_int,                       # layer_idx
        ctypes.POINTER(ctypes.c_uint16),    # h_in
        ctypes.c_int,                       # S
        ctypes.c_int,                       # pos0
        ctypes.POINTER(ctypes.c_uint16),    # h_out
    ]

    lib.amx_draft_qwen05b_forward_full.restype = ctypes.c_int
    lib.amx_draft_qwen05b_forward_full.argtypes = [
        ctypes.POINTER(ctypes.c_int32),     # input_ids
        ctypes.c_int,                       # S_prompt
        ctypes.POINTER(ctypes.c_int32),     # out_ids
        ctypes.c_int,                       # K
        ctypes.POINTER(ctypes.c_uint16),    # logits_last_bf16 (may be NULL)
    ]

    lib.amx_draft_qwen05b_reset_kv_cache.restype = None
    lib.amx_draft_qwen05b_reset_kv_cache.argtypes = []

    return lib


def _u16_ptr(arr: np.ndarray):
    assert arr.dtype == np.uint16
    assert arr.flags["C_CONTIGUOUS"]
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def _i32_ptr(arr: np.ndarray):
    assert arr.dtype == np.int32
    assert arr.flags["C_CONTIGUOUS"]
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


# ────────────────────────────────────────────────────────────────────
# Phase: load all Qwen-0.5B weights into kernel
# ────────────────────────────────────────────────────────────────────

def _load_all_weights(lib) -> dict:
    """Load every weight tensor needed by the kernel."""
    print(f"[load] safetensors={QWEN_SAFETENSORS}")
    weights = {}
    t0 = time.time()
    with safe_open(str(QWEN_SAFETENSORS), framework="pt") as f:
        weights["embed"] = f.get_tensor("model.embed_tokens.weight")
        weights["final_norm"] = f.get_tensor("model.norm.weight")
        for L in range(N_LAYERS):
            pre = f"model.layers.{L}"
            weights[f"L{L}.ln1"]   = f.get_tensor(f"{pre}.input_layernorm.weight")
            weights[f"L{L}.ln2"]   = f.get_tensor(f"{pre}.post_attention_layernorm.weight")
            weights[f"L{L}.q_w"]   = f.get_tensor(f"{pre}.self_attn.q_proj.weight")
            weights[f"L{L}.q_b"]   = f.get_tensor(f"{pre}.self_attn.q_proj.bias")
            weights[f"L{L}.k_w"]   = f.get_tensor(f"{pre}.self_attn.k_proj.weight")
            weights[f"L{L}.k_b"]   = f.get_tensor(f"{pre}.self_attn.k_proj.bias")
            weights[f"L{L}.v_w"]   = f.get_tensor(f"{pre}.self_attn.v_proj.weight")
            weights[f"L{L}.v_b"]   = f.get_tensor(f"{pre}.self_attn.v_proj.bias")
            weights[f"L{L}.o_w"]   = f.get_tensor(f"{pre}.self_attn.o_proj.weight")
            weights[f"L{L}.gate"]  = f.get_tensor(f"{pre}.mlp.gate_proj.weight")
            weights[f"L{L}.up"]    = f.get_tensor(f"{pre}.mlp.up_proj.weight")
            weights[f"L{L}.down"]  = f.get_tensor(f"{pre}.mlp.down_proj.weight")
    print(f"[load] read {len(weights)} tensors in {time.time()-t0:.2f}s")

    # Init kernel
    rc = lib.amx_draft_qwen05b_init()
    print(f"[init] amx_draft_qwen05b_init rc={rc}")
    assert rc == 0, f"init failed rc={rc}"

    rc = lib.amx_draft_qwen05b_init_model()
    print(f"[init] amx_draft_qwen05b_init_model rc={rc}")
    assert rc == 0, f"init_model failed rc={rc}"

    # Load embed_tokens (also serves as tied lm_head row source)
    emb_np = _u16(weights["embed"])
    assert emb_np.shape == (VOCAB_VALID, HIDDEN)
    rc = lib.amx_draft_qwen05b_load_embed_tokens(
        _u16_ptr(emb_np), VOCAB_VALID, HIDDEN)
    print(f"[load] embed_tokens rc={rc}")
    assert rc == 0

    # Load tied lm_head (zero-pad inside)
    rc = lib.amx_draft_qwen05b_load_lm_head(
        _u16_ptr(emb_np), VOCAB_VALID, HIDDEN, VOCAB_PADDED)
    print(f"[load] lm_head (tied) rc={rc}")
    assert rc == 0

    # Final norm
    fn_np = _u16(weights["final_norm"])
    rc = lib.amx_draft_qwen05b_load_final_norm(_u16_ptr(fn_np))
    print(f"[load] final_norm rc={rc}")
    assert rc == 0

    # Per-layer weights
    # Keep np arrays alive (ctypes pointers must not dangle).
    kept = [emb_np, fn_np]
    t0 = time.time()
    for L in range(N_LAYERS):
        ln1 = _u16(weights[f"L{L}.ln1"])
        q_w = _u16(weights[f"L{L}.q_w"])
        q_b = _u16(weights[f"L{L}.q_b"])
        k_w = _u16(weights[f"L{L}.k_w"])
        k_b = _u16(weights[f"L{L}.k_b"])
        v_w = _u16(weights[f"L{L}.v_w"])
        v_b = _u16(weights[f"L{L}.v_b"])
        o_w = _u16(weights[f"L{L}.o_w"])
        ln2 = _u16(weights[f"L{L}.ln2"])
        gate = _u16(weights[f"L{L}.gate"])
        up   = _u16(weights[f"L{L}.up"])
        down = _u16(weights[f"L{L}.down"])
        rc = lib.amx_draft_qwen05b_load_layer_weights(
            L,
            _u16_ptr(ln1),
            _u16_ptr(q_w), _u16_ptr(q_b),
            _u16_ptr(k_w), _u16_ptr(k_b),
            _u16_ptr(v_w), _u16_ptr(v_b),
            _u16_ptr(o_w),
            _u16_ptr(ln2),
            _u16_ptr(gate), _u16_ptr(up), _u16_ptr(down))
        assert rc == 0, f"layer {L} load rc={rc}"
        kept.extend([ln1, q_w, q_b, k_w, k_b, v_w, v_b, o_w, ln2, gate, up, down])
    print(f"[load] 24 layers loaded in {time.time()-t0:.2f}s")
    return {"weights": weights, "kept": kept}


# ────────────────────────────────────────────────────────────────────
# P2 — Single-layer test vs HF transformers reference
# ────────────────────────────────────────────────────────────────────

def case_p2_single_layer(lib) -> dict:
    print("=" * 72)
    print("P2 — Single-layer forward vs HF transformers reference (layer[0])")
    print("=" * 72)

    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(HF_QWEN_ID)
    print(f"  config: layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"q_heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
          f"intermediate={cfg.intermediate_size} rope_theta={cfg.rope_theta}")
    model = AutoModelForCausalLM.from_pretrained(
        HF_QWEN_ID, dtype=torch.bfloat16)
    model.eval()

    # We feed a real prompt embedding (no random) to get a meaningful single-layer
    # check, then extract HF's layer[0] output via output_hidden_states.
    S = 1
    # Use token id 1234 (arbitrary, in-vocab).
    tok_id = 1234
    emb_w = model.model.embed_tokens.weight.detach()  # [V, H] BF16
    h_in_bf16_t = emb_w[tok_id:tok_id + 1]            # [1, H] BF16
    h_in_u16 = _u16(h_in_bf16_t)

    # AMX reset KV cache, then layer[0] forward
    lib.amx_draft_qwen05b_reset_kv_cache()
    h_out_u16 = np.zeros((S, HIDDEN), dtype=np.uint16)
    rc = lib.amx_draft_qwen05b_layer_forward(
        0, _u16_ptr(h_in_u16), S, 0, _u16_ptr(h_out_u16))
    assert rc == 0, f"layer_forward rc={rc}"
    h_out_amx_fp32 = torch.from_numpy(h_out_u16.view(np.int16)).view(
        torch.bfloat16).to(torch.float32).numpy()

    # HF reference: full model forward with output_hidden_states=True,
    # then layer[0] is hidden_states[1] (input embed = [0], after L0 = [1], …).
    with torch.no_grad():
        ids = torch.tensor([[tok_id]], dtype=torch.long)
        out = model(ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple of len N_LAYERS+1
    h_out_ref_t = hs[1][0]   # [S, H], output AFTER layer[0]
    h_out_ref_fp32 = h_out_ref_t.to(torch.float32).cpu().numpy()

    # Compare
    diff = h_out_amx_fp32 - h_out_ref_fp32
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    # Cosine similarity per row, average
    def _cos(a, b):
        n = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / n)
    cos_sim = float(np.mean([_cos(h_out_amx_fp32[s], h_out_ref_fp32[s])
                             for s in range(S)]))
    print(f"  layer[0] output shape: amx={h_out_amx_fp32.shape}, "
          f"ref={h_out_ref_fp32.shape}")
    print(f"  max-abs-diff: {max_abs:.5f}")
    print(f"  mean-abs-diff: {mean_abs:.5f}")
    print(f"  cosine sim (mean over S rows): {cos_sim:.6f}")
    print(f"  ref output magnitude (L2): {np.linalg.norm(h_out_ref_fp32):.3f}")
    print(f"  amx output magnitude (L2): {np.linalg.norm(h_out_amx_fp32):.3f}")

    verdict = "PASS" if cos_sim > 0.95 else ("WARN" if cos_sim > 0.80 else "FAIL")
    print(f"  [P2 verdict] cosine sim {cos_sim:.4f} → {verdict}")
    return {
        "p2_max_abs": max_abs,
        "p2_mean_abs": mean_abs,
        "p2_cos_sim": cos_sim,
        "p2_verdict": verdict,
        "hf_model": model,
    }


# ────────────────────────────────────────────────────────────────────
# P3 — Full 24-layer + lm_head + argmax vs HF
# ────────────────────────────────────────────────────────────────────

PROMPTS = [
    "The capital of France is",
    "import numpy as np\n# Define a function that",
    "Hello, my name is",
    "The largest planet in our solar system is",
    "def fibonacci(n):",
    "Once upon a time,",
]


def case_p3_full_forward(lib, hf_model) -> dict:
    print("=" * 72)
    print("P3 — Full 24-layer forward + lm_head + next-token vs HF")
    print("=" * 72)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_QWEN_ID)

    n_prompts = len(PROMPTS)
    amx_top1 = []
    ref_top3_all = []
    logprob_max_abs_diffs = []
    ref_top1_all = []

    MAX_SEQ = 32  # kernel MAX_SEQ=64; keep prompts short

    for idx, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        if len(ids) > MAX_SEQ:
            ids = ids[-MAX_SEQ:]
        S = len(ids)
        ids_np = np.array(ids, dtype=np.int32)
        # AMX full forward: K=1 (we only check next-token, not 7-step draft chain)
        K = 1
        out_ids = np.zeros(K, dtype=np.int32)
        logits_buf = np.zeros((K, VOCAB_PADDED), dtype=np.uint16)
        rc = lib.amx_draft_qwen05b_forward_full(
            _i32_ptr(ids_np), S, _i32_ptr(out_ids), K, _u16_ptr(logits_buf))
        assert rc == 0, f"forward_full rc={rc}"
        amx_id = int(out_ids[0])

        # HF reference
        with torch.no_grad():
            input_t = torch.tensor([ids], dtype=torch.long)
            out = hf_model(input_t, use_cache=False)
            logits = out.logits[0, -1, :]  # [V_valid]
        ref_logits_fp32 = logits.to(torch.float32).cpu().numpy()
        ref_top3 = np.argsort(-ref_logits_fp32)[:3]
        ref_top1_all.append(int(ref_top3[0]))

        # AMX logits → FP32
        amx_logits_bf16 = logits_buf[0, :VOCAB_VALID]  # uint16
        amx_logits_fp32 = (amx_logits_bf16.astype(np.uint32) << 16).view(np.float32)

        # logprob max-abs-diff
        def _logp(x):
            xs = x.astype(np.float64)
            xs -= xs.max()
            ex = np.exp(xs)
            return np.log(ex / ex.sum())
        lp_a = _logp(amx_logits_fp32)
        lp_r = _logp(ref_logits_fp32)
        lp_mad = float(np.max(np.abs(lp_a - lp_r)))
        logprob_max_abs_diffs.append(lp_mad)

        amx_top1.append(amx_id)
        ref_top3_all.append(ref_top3)

        try:
            amx_tok_str = tok.decode([amx_id])
            ref_tok_str = tok.decode([int(ref_top3[0])])
        except Exception:
            amx_tok_str = "?"; ref_tok_str = "?"
        print(f"  [#{idx}] S={S:2d} prompt={prompt!r:50s}")
        print(f"       AMX id={amx_id:6d}({amx_tok_str!r:>15s})  "
              f"REF top1={int(ref_top3[0]):6d}({ref_tok_str!r:>15s})  "
              f"top3={ref_top3.tolist()}  lp_mad={lp_mad:.4f}")

    amx_top1_np = np.array(amx_top1, dtype=np.int64)
    ref_top3_np = np.stack(ref_top3_all, axis=0)
    top1_match = float(np.mean(amx_top1_np == ref_top3_np[:, 0]))
    top3_match = float(np.mean(
        np.any(amx_top1_np[:, None] == ref_top3_np[:, :3], axis=1)))
    mean_lp_mad = float(np.mean(logprob_max_abs_diffs))
    max_lp_mad = float(np.max(logprob_max_abs_diffs))

    print("─" * 72)
    print(f"  AGGREGATE over {n_prompts} prompts:")
    print(f"    top-1 match: {top1_match*100:.1f}% (threshold 90%)")
    print(f"    top-3 match: {top3_match*100:.1f}% (threshold 95%)")
    print(f"    logprob max-abs-diff: mean={mean_lp_mad:.4f}  max={max_lp_mad:.4f}  (threshold <0.1)")
    verdict = (
        "PASS" if (top1_match >= 0.90 and top3_match >= 0.95
                   and max_lp_mad < 0.1)
        else "WARN" if top3_match >= 0.50 else "FAIL"
    )
    print(f"  [P3 verdict] {verdict}")

    return {
        "p3_top1": top1_match,
        "p3_top3": top3_match,
        "p3_lp_mad_mean": mean_lp_mad,
        "p3_lp_mad_max": max_lp_mad,
        "p3_verdict": verdict,
    }


# ────────────────────────────────────────────────────────────────────
# P4 — microbench: per-step latency B=1, K=7
# ────────────────────────────────────────────────────────────────────

def case_p4_microbench(lib) -> dict:
    print("=" * 72)
    print("P4 — Full forward microbench (B=1, S_prompt=8, K=7)")
    print("=" * 72)
    S = 8
    ids = np.arange(1, S + 1, dtype=np.int32)  # arbitrary token ids
    K = 7
    out_ids = np.zeros(K, dtype=np.int32)

    # Warmup
    for _ in range(2):
        rc = lib.amx_draft_qwen05b_forward_full(
            _i32_ptr(ids), S, _i32_ptr(out_ids), K, None)
        assert rc == 0

    # Measure
    n_iter = 5
    lats_ms = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        rc = lib.amx_draft_qwen05b_forward_full(
            _i32_ptr(ids), S, _i32_ptr(out_ids), K, None)
        t1 = time.perf_counter()
        assert rc == 0
        lats_ms.append((t1 - t0) * 1000.0)
    p50 = float(np.median(lats_ms))
    mean = float(np.mean(lats_ms))
    p99 = float(np.percentile(lats_ms, 99))
    per_step_p50 = p50 / K
    print(f"  full forward ms over {n_iter} iters: {[f'{x:.2f}' for x in lats_ms]}")
    print(f"  total p50={p50:.2f} ms, mean={mean:.2f} ms, p99={p99:.2f} ms")
    print(f"  per-step p50 (K={K}): {per_step_p50:.2f} ms")
    # GPU verify budget for Llama-70B is 40 ms — we want
    # per-step < 40/K = 5.7 ms (per-step model) OR total < 40 ms (single batch chunk).
    gpu_verify_budget_ms = 40.0
    print(f"  GPU verify budget (Llama-70B): {gpu_verify_budget_ms:.1f} ms")
    print(f"  Per-K-step amortized: {per_step_p50:.2f} ms  "
          f"({'PASS' if per_step_p50 < gpu_verify_budget_ms else 'FAIL'})")
    print(f"  Total K={K} draft ms: {p50:.2f} ms  "
          f"({'NET POSITIVE' if p50 < gpu_verify_budget_ms else 'NET NEGATIVE'})")
    return {
        "p4_p50_ms": p50,
        "p4_mean_ms": mean,
        "p4_p99_ms": p99,
        "p4_per_step_p50_ms": per_step_p50,
        "p4_gpu_budget_ms": gpu_verify_budget_ms,
    }


# ────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────

def main():
    print("─" * 72)
    print(f"LIB: {LIB_PATH}")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','(unset)')}")
    print("─" * 72)
    lib = _load_lib()

    _load_all_weights(lib)
    results = {}
    p2 = case_p2_single_layer(lib)
    results.update(p2)
    hf_model = p2["hf_model"]
    p3 = case_p3_full_forward(lib, hf_model)
    results.update(p3)
    p4 = case_p4_microbench(lib)
    results.update(p4)

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  P2 cosine sim:         {results.get('p2_cos_sim', 0):.4f}  → {results.get('p2_verdict')}")
    print(f"  P3 top-1 match:        {results.get('p3_top1', 0)*100:.1f}%")
    print(f"  P3 top-3 match:        {results.get('p3_top3', 0)*100:.1f}%")
    print(f"  P3 lp_mad max:         {results.get('p3_lp_mad_max', 0):.4f} → {results.get('p3_verdict')}")
    print(f"  P4 per-step p50 ms:    {results.get('p4_per_step_p50_ms', 0):.2f}")
    print(f"  P4 total K=7 p50 ms:   {results.get('p4_p50_ms', 0):.2f}  (budget {results.get('p4_gpu_budget_ms',0)} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
