# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vanilla baseline measurement for NEO comparison (IDE_006 4 차 재정의).

Runs a single configurable workload through vLLM's *vanilla* path
(``enable_neo_asymmetric=False``) and records the metrics that NEO
is supposed to improve:

* wall_time_s — total time from generate() start to all-prompts done
* total_input_tokens / total_output_tokens
* prompt_throughput_per_s / output_throughput_per_s
* preempt_count (grep'd from the engine log)
* kv_cache_usage_peak (grep'd from the engine log)

Output is JSON for ingestion by NEO comparison runs and TSK
documentation.

Run with the wrapper:
    bash eval/run_neo_baseline.sh --prompts 100 --max-model-len 8192
"""

from __future__ import annotations

import torch  # noqa: F401  — torch must load before vllm
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

DEFAULT_LOG_FILE = "/tmp/neo_baseline.log"
DEFAULT_OUTPUT_FILE = "/tmp/neo_baseline.json"

MODEL_ALIASES = {
    "qwen-1.5b":   "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-72b":    "Qwen/Qwen2.5-72B-Instruct",
    "llama-70b":   "meta-llama/Llama-3.3-70B-Instruct",
}


def _resolve_model(name_or_path: str) -> str:
    return MODEL_ALIASES.get(name_or_path.lower(), name_or_path)


def _build_prompts(num_prompts: int, target_input_len: int,
                   sonnet_path: Path, seed: int = 0) -> list[str]:
    """Compose ``num_prompts`` prompts of approximately
    ``target_input_len`` tokens each by sampling lines from
    ``sonnet_path`` (Shakespeare). Each prompt is a unique random
    sample so that prefix caching cannot collapse the workload."""
    rng = random.Random(seed)
    lines = [
        ln.strip() for ln in sonnet_path.read_text().splitlines()
        if ln.strip()
    ]
    if not lines:
        raise RuntimeError(f"No usable lines in {sonnet_path}")
    # Heuristic: 1 line of sonnet ≈ 10 tokens.
    lines_per_prompt = max(1, target_input_len // 10)
    prompts = []
    for _ in range(num_prompts):
        sample = rng.sample(lines, k=min(lines_per_prompt, len(lines)))
        prompts.append(" ".join(sample))
    return prompts


def _grep_metrics(log_file: Path) -> dict:
    """Grep engine logs for NEO-relevant metrics.

    vLLM v1's logs include lines like:
        ENGINE INFO ... preempted X requests
        kv_cache_utils ... GPU KV cache size: N tokens
    The exact strings vary by version, so we scan with broad patterns
    and fall back to ``None`` when the metric is not visible.
    """
    metrics = {
        "preempt_count": None,
        "kv_cache_size_tokens": None,
        "max_concurrency": None,
    }
    if not log_file.exists():
        return metrics
    text = log_file.read_text()

    # Preempt count — vLLM emits "Preempted N requests" or similar.
    preempt_re = re.compile(
        r"[Pp]reempt(?:ed|ion[s]?|s)?[^\d]{0,20}(\d+)\s+request"
    )
    matches = preempt_re.findall(text)
    if matches:
        # Sum across all log lines (each one reports per-step or
        # cumulative preempts depending on the build).
        metrics["preempt_count"] = sum(int(m) for m in matches)

    # KV cache size at startup
    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
    if m:
        metrics["kv_cache_size_tokens"] = int(m.group(1).replace(",", ""))

    m = re.search(r"Maximum concurrency for[^:]+:\s*([\d.]+)x", text)
    if m:
        metrics["max_concurrency"] = float(m.group(1))

    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-1.5b",
                    help=f"alias or path. aliases={list(MODEL_ALIASES)}")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--num-prompts", type=int, default=100)
    ap.add_argument("--target-input-len", type=int, default=1024,
                    help="approximate per-prompt input length in tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sonnet",
                    type=Path,
                    default=Path("benchmarks/sonnet.txt"))
    ap.add_argument("--log-file", type=Path, default=Path(DEFAULT_LOG_FILE))
    ap.add_argument("--output-file", type=Path, default=Path(DEFAULT_OUTPUT_FILE))
    args = ap.parse_args()

    print(f"[baseline] model={_resolve_model(args.model)}", flush=True)
    print(f"[baseline] TP={args.tensor_parallel_size} "
          f"max_model_len={args.max_model_len} "
          f"max_num_seqs={args.max_num_seqs}", flush=True)
    print(f"[baseline] num_prompts={args.num_prompts} "
          f"target_input_len={args.target_input_len} "
          f"max_tokens={args.max_tokens}", flush=True)

    prompts = _build_prompts(
        args.num_prompts, args.target_input_len, args.sonnet, args.seed,
    )
    avg_prompt_chars = sum(len(p) for p in prompts) / max(1, len(prompts))
    print(f"[baseline] avg prompt chars: {avg_prompt_chars:.0f}",
          flush=True)

    from vllm import LLM, SamplingParams

    init_t0 = time.perf_counter()
    llm = LLM(
        model=_resolve_model(args.model),
        enable_neo_asymmetric=False,    # vanilla baseline
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=False,
        seed=args.seed,
    )
    init_s = time.perf_counter() - init_t0
    print(f"[baseline] init {init_s:.1f}s", flush=True)

    params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed,
    )

    gen_t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    wall_s = time.perf_counter() - gen_t0

    total_in = sum(len(out.prompt_token_ids) for out in outputs)
    total_out = sum(
        sum(len(o.token_ids) for o in out.outputs) for out in outputs
    )
    in_tps = total_in / max(wall_s, 1e-9)
    out_tps = total_out / max(wall_s, 1e-9)
    req_per_s = len(outputs) / max(wall_s, 1e-9)

    print(f"[baseline] generate wall={wall_s:.1f}s")
    print(f"[baseline] total_input_tokens={total_in} "
          f"total_output_tokens={total_out}")
    print(f"[baseline] prompt_tps={in_tps:.1f} output_tps={out_tps:.1f} "
          f"req_per_s={req_per_s:.2f}")

    log_metrics = _grep_metrics(args.log_file)
    print(f"[baseline] log metrics: {log_metrics}")

    record = {
        "model": _resolve_model(args.model),
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "num_prompts": args.num_prompts,
        "target_input_len": args.target_input_len,
        "max_tokens": args.max_tokens,
        "init_s": init_s,
        "generate_wall_s": wall_s,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "prompt_tps": in_tps,
        "output_tps": out_tps,
        "req_per_s": req_per_s,
        "log_metrics": log_metrics,
    }
    args.output_file.write_text(json.dumps(record, indent=2))
    print(f"[baseline] wrote {args.output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
