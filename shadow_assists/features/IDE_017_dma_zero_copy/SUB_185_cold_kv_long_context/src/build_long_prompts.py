#!/usr/bin/env python3
"""SUB_185 — long-context workload generator.

Builds N prompts targeting p50 input length >= 8000 tokens by repeating
sonnet lines from /workspace/vllm_hybrid/benchmarks/sonnet.txt.

Output: JSON list of {"prompt": str, "n_tokens": int} (estimate via Qwen tokenizer)
Pre-tokenization step done here to avoid runtime startup cost in the benchmark.

Usage:
    python build_long_prompts.py --num-prompts 500 --target-tokens 8000 \
        --out /tmp/sub185_prompts.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SONNET_PATH = Path("/workspace/vllm_hybrid/benchmarks/sonnet.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=500)
    ap.add_argument("--target-tokens", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--measure-fraction", type=float, default=0.1,
                    help="Fraction of prompts to actually tokenize for length verification")
    args = ap.parse_args()

    from transformers import AutoTokenizer  # local lazy

    tok = AutoTokenizer.from_pretrained(args.model)
    lines = SONNET_PATH.read_text().strip().split("\n")
    # Average tokens per sonnet line — measure once.
    sample = "\n".join(lines[:100])
    sample_toks = len(tok.encode(sample))
    tokens_per_line = sample_toks / 100.0
    n_lines_needed = int(args.target_tokens / tokens_per_line * 1.1)  # +10% safety
    print(f"[build] sonnet lines={len(lines)} tokens_per_line={tokens_per_line:.2f} -> "
          f"n_lines_needed={n_lines_needed} for target={args.target_tokens}")

    rng = random.Random(args.seed)
    prompts = []
    measured_lens = []
    measure_step = max(1, int(1.0 / args.measure_fraction))
    for i in range(args.num_prompts):
        # repeat-with-shuffle pattern to reach ≥ 8K tokens
        chosen = rng.choices(lines, k=n_lines_needed)
        prompt = "\n".join(chosen)
        n_tok = None
        if i % measure_step == 0:
            n_tok = len(tok.encode(prompt))
            measured_lens.append(n_tok)
        prompts.append({"prompt": prompt, "n_tokens": n_tok})

    # report
    if measured_lens:
        measured_lens.sort()
        n = len(measured_lens)
        p50 = measured_lens[n // 2]
        p_min = measured_lens[0]
        p_max = measured_lens[-1]
        print(f"[build] sampled {n}/{args.num_prompts} prompts: "
              f"min={p_min} p50={p50} max={p_max}")

    args.out.write_text(json.dumps({
        "num_prompts": args.num_prompts,
        "target_tokens": args.target_tokens,
        "tokens_per_line": tokens_per_line,
        "n_lines_per_prompt": n_lines_needed,
        "sampled_token_lens": measured_lens,
        "prompts": prompts,
    }))
    print(f"[build] saved {args.num_prompts} prompts -> {args.out}")


if __name__ == "__main__":
    main()
