"""SUB_029 — NEO vs vanilla 출력 일치성 검증.

작은 workload (3 prompts × 128 tokens) 로 actual generated text 를
확인 . token-by-token 비교 가 아닌, **출력이 정상적인 자연어인지** 점검 .

Usage:
    python verify_outputs.py --mode vanilla --output /tmp/outputs_vanilla.txt
    python verify_outputs.py --mode neo --output /tmp/outputs_neo.txt
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_SONNET = Path("/workspace/vllm_hybrid/benchmarks/sonnet.txt")


def build_prompts(num_prompts: int, target_input_len: int,
                  sonnet_path: Path, seed: int = 0) -> list[str]:
    """``run_neo_baseline.py`` 와 동일한 prompt 생성 (재현성 보장)."""
    rng = random.Random(seed)
    lines = [
        ln.strip() for ln in sonnet_path.read_text().splitlines()
        if ln.strip()
    ]
    if not lines:
        raise RuntimeError(f"No usable lines in {sonnet_path}")
    lines_per_prompt = max(1, target_input_len // 10)
    prompts = []
    for _ in range(num_prompts):
        sample = rng.sample(lines, k=min(lines_per_prompt, len(lines)))
        prompts.append(" ".join(sample))
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vanilla", "neo"], required=True)
    ap.add_argument("--num-prompts", type=int, default=3)
    ap.add_argument("--target-input-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sonnet", type=Path, default=DEFAULT_SONNET)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--tensor-parallel-size", type=int, default=8)
    args = ap.parse_args()

    if args.model == "llama-70b":
        args.model = DEFAULT_MODEL

    prompts = build_prompts(
        args.num_prompts, args.target_input_len, args.sonnet, args.seed,
    )
    print(f"[verify] mode={args.mode} num_prompts={len(prompts)} "
          f"max_tokens={args.max_tokens}", flush=True)
    print(f"[verify] prompt[0] preview: {prompts[0][:120]!r}...", flush=True)

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=16384,
        max_num_seqs=256,
        kv_cache_dtype="fp8",
        max_num_batched_tokens=8192,
        async_scheduling=True,
        seed=args.seed,
    )
    if args.mode == "neo":
        llm_kwargs["enable_neo_asymmetric"] = True

    t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    init_s = time.perf_counter() - t0
    print(f"[verify] init {init_s:.1f}s", flush=True)

    params = SamplingParams(
        temperature=0.0, top_p=1.0,
        max_tokens=args.max_tokens, seed=args.seed,
    )

    gen_t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    wall_s = time.perf_counter() - gen_t0
    print(f"[verify] generate wall={wall_s:.2f}s", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.write(f"# mode={args.mode} num_prompts={len(prompts)} "
                f"max_tokens={args.max_tokens} seed={args.seed}\n")
        f.write(f"# wall={wall_s:.2f}s init={init_s:.1f}s\n\n")
        for i, out in enumerate(outputs):
            gen_text = out.outputs[0].text
            gen_tokens = list(out.outputs[0].token_ids)
            f.write(f"=== prompt {i} ===\n")
            f.write(f"prompt_tokens={len(out.prompt_token_ids)}\n")
            f.write(f"output_tokens={len(gen_tokens)}\n")
            f.write(f"first_20_tokens={gen_tokens[:20]}\n")
            f.write(f"text:\n{gen_text}\n")
            f.write(f"\n--- end prompt {i} ---\n\n")
    print(f"[verify] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
