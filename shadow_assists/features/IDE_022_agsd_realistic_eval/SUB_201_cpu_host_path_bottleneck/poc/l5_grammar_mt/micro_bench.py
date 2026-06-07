"""SUB_201 L5 — xgrammar grammar.accept_token() multi-thread micro-bench.

End-to-end e2e measurement on a live vLLM with constrained decode is heavily
disturbed in this shared dev box (multiple concurrent vLLM instances cause
EngineCore crashes and 500s). To still answer the lever question (does
multi-threading xgrammar's per-request FSM advance help?), we measure the
xgrammar GrammarMatcher.accept_token() path **directly**, the same code that
sits behind vLLM's `XgrammarGrammar.accept_tokens` (`backend_xgrammar.py:178`).

Protocol:
  - Llama-3.1 tokenizer + Llama vocab (128 256).
  - JSON-schema spec identical to b2_constrained's 5-key schema.
  - Build N independent GrammarMatcher instances (= N concurrent vLLM
    structured-output requests in a batch).
  - For each step, pick a random token id that the matcher currently accepts,
    then call accept_token() to advance the FSM. Repeat for STEPS steps,
    once with single-thread sequential, once with a ThreadPoolExecutor
    (xgrammar's matcher.accept_token releases the GIL inside the C++ FSM
    update, so threads run in parallel).
  - Report per-batch latency for both modes and the speed-up.

Run:
  PYTHONPATH=/workspace/host_vllm_hybrid \
    /workspace/vllm_dev_prj/bin/python micro_bench.py \
      --batch 16 --steps 256 --workers 8
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import xgrammar as xgr
from transformers import AutoTokenizer


JSON_SCHEMA_5KEY = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
        "country": {"type": "string"},
        "score": {"type": "number"},
    },
    "required": ["name", "age", "email", "country", "score"],
}


def _pick_accepted_token(matcher, vocab_size: int, rng: random.Random) -> int:
    """Pick a token id the matcher will currently accept.

    Strategy: ask xgrammar for the next-token bitmask, then sample a 1-bit
    from it (= a token the FSM will accept). This mirrors what the sampler
    would do after applying the bitmask to logits.
    """
    import torch
    bm = xgr.allocate_token_bitmask(1, vocab_size)
    matcher.fill_next_token_bitmask(bm, 0)
    arr = bm[0]  # int32 view, ceil(vocab/32) entries
    # Find a set bit.
    nbits = arr.numel() * 32
    # Try a few random indices first (typical accepted-set is large for JSON).
    for _ in range(64):
        b = rng.randrange(nbits)
        if b >= vocab_size:
            continue
        word = arr[b >> 5].item()
        if word & (1 << (b & 31)):
            return b
    # Fall back: scan.
    for i, w in enumerate(arr.tolist()):
        if not w:
            continue
        for bit in range(32):
            if w & (1 << bit):
                tok = (i << 5) | bit
                if tok < vocab_size:
                    return tok
    raise RuntimeError("no accepted token (grammar terminated?)")


def _advance_one(matcher, vocab_size: int, steps: int, rng_seed: int) -> int:
    """Advance a single matcher by `steps` tokens. Returns # of advances done."""
    rng = random.Random(rng_seed)
    n = 0
    for _ in range(steps):
        if matcher.is_terminated():
            break
        tok = _pick_accepted_token(matcher, vocab_size, rng)
        if not matcher.accept_token(tok):
            break
        n += 1
    return n


def _build_matchers(tokenizer, schema_str: str, n: int):
    compiler = xgr.GrammarCompiler(
        xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size)
    )
    grammar = compiler.compile_json_schema(schema_str)
    return [xgr.GrammarMatcher(grammar) for _ in range(n)]


def _precompute_token_seqs(tok, schema_str: str, n: int, steps: int, vocab_size: int, base_seed: int):
    """Build N matchers and walk each forward `steps` tokens, recording the
    accepted token at every step. Returns (token_seqs, advances_total).
    """
    matchers = _build_matchers(tok, schema_str, n)
    seqs: list[list[int]] = []
    advances_total = 0
    for i, m in enumerate(matchers):
        rng = random.Random(base_seed + i)
        seq = []
        for _ in range(steps):
            if m.is_terminated():
                break
            tok_id = _pick_accepted_token(m, vocab_size, rng)
            if not m.accept_token(tok_id):
                break
            seq.append(tok_id)
            advances_total += 1
        seqs.append(seq)
    return seqs, advances_total


def _replay_accept_only(matchers, token_seqs):
    """Replay the pre-computed token sequence on a fresh matcher. Pure
    accept_token() — no bitmask fill, no FSM introspection."""
    total = 0
    for m, seq in zip(matchers, token_seqs):
        for tok_id in seq:
            if not m.accept_token(tok_id):
                break
            total += 1
    return total


def _replay_one_accept(matcher, seq):
    total = 0
    for tok_id in seq:
        if not matcher.accept_token(tok_id):
            break
        total += 1
    return total


def run(args):
    print(f"[micro] loading tokenizer {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    vocab_size = tok.vocab_size
    schema_str = json.dumps(JSON_SCHEMA_5KEY)

    print(f"[micro] vocab_size={vocab_size} batch={args.batch} steps={args.steps} workers={args.workers}", flush=True)

    # Warmup compile (once).
    _build_matchers(tok, schema_str, 1)

    # Pre-compute token sequences so both modes replay the SAME tokens and
    # the timing window only captures accept_token() C++ work.
    print("[micro] === pre-compute accepted token sequences ===", flush=True)
    token_seqs, advances_pre = _precompute_token_seqs(
        tok, schema_str, args.batch, args.steps, vocab_size, base_seed=1000
    )
    print(f"[micro] pre-computed {advances_pre} accepts total, "
          f"avg len={advances_pre/args.batch:.1f}/seq", flush=True)

    # ---- mode A: full path (bitmask + accept) ---------------------------
    print("[micro] === A: full (fill_next_token_bitmask + accept) ===", flush=True)
    matchers = _build_matchers(tok, schema_str, args.batch)
    t0 = time.perf_counter()
    total_advances_serial = 0
    for i, m in enumerate(matchers):
        n = _advance_one(m, vocab_size, args.steps, rng_seed=1000 + i)
        total_advances_serial += n
    t_serial = time.perf_counter() - t0
    print(f"[micro] A serial: wall={t_serial*1000:.1f} ms, advances={total_advances_serial}", flush=True)

    matchers = _build_matchers(tok, schema_str, args.batch)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(_advance_one, m, vocab_size, args.steps, 1000 + i)
            for i, m in enumerate(matchers)
        ]
        total_advances_mt = sum(f.result() for f in futs)
    t_mt = time.perf_counter() - t0
    print(f"[micro] A mt-{args.workers}: wall={t_mt*1000:.1f} ms, advances={total_advances_mt}", flush=True)

    # ---- mode B: accept-only replay (matches vLLM scheduler hot path) ---
    print("[micro] === B: accept-only replay (vLLM scheduler hot path) ===", flush=True)
    matchers = _build_matchers(tok, schema_str, args.batch)
    t0 = time.perf_counter()
    total_b_serial = _replay_accept_only(matchers, token_seqs)
    t_b_serial = time.perf_counter() - t0
    print(f"[micro] B serial: wall={t_b_serial*1000:.1f} ms, accepts={total_b_serial}", flush=True)

    matchers = _build_matchers(tok, schema_str, args.batch)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(_replay_one_accept, m, s) for m, s in zip(matchers, token_seqs)
        ]
        total_b_mt = sum(f.result() for f in futs)
    t_b_mt = time.perf_counter() - t0
    print(f"[micro] B mt-{args.workers}: wall={t_b_mt*1000:.1f} ms, accepts={total_b_mt}", flush=True)

    speedup = t_serial / t_mt if t_mt > 0 else 0
    speedup_b = t_b_serial / t_b_mt if t_b_mt > 0 else 0
    print(f"[micro] === SUMMARY ===", flush=True)
    print(f"[micro] A serial : {t_serial*1000:9.2f} ms  (full path)", flush=True)
    print(f"[micro] A mt-{args.workers:<3}: {t_mt*1000:9.2f} ms  speedup {speedup:.2f}x  Δ {(t_mt-t_serial)/t_serial*100:+.1f}%", flush=True)
    print(f"[micro] B serial : {t_b_serial*1000:9.2f} ms  (accept only)", flush=True)
    print(f"[micro] B mt-{args.workers:<3}: {t_b_mt*1000:9.2f} ms  speedup {speedup_b:.2f}x  Δ {(t_b_mt-t_b_serial)/t_b_serial*100:+.1f}%", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "model": args.model,
                "batch": args.batch,
                "steps": args.steps,
                "workers": args.workers,
                "A_wall_serial_ms": t_serial * 1000.0,
                "A_wall_mt_ms": t_mt * 1000.0,
                "A_speedup": speedup,
                "A_delta_pct": (t_mt - t_serial) / t_serial * 100.0,
                "B_wall_serial_ms": t_b_serial * 1000.0,
                "B_wall_mt_ms": t_b_mt * 1000.0,
                "B_speedup": speedup_b,
                "B_delta_pct": (t_b_mt - t_b_serial) / t_b_serial * 100.0,
                "total_accepts_precomputed": advances_pre,
            }, f, indent=2)
        print(f"[micro] wrote {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
