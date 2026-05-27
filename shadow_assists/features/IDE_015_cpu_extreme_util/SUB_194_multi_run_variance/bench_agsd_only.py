#!/usr/bin/env python3
"""SUB_194 — agsd-gated only benchmark (skip vanilla-only / trident-only individual cells).

Reuses /tmp/sub094_benchmark.py infrastructure but runs only the AGSD router path.
Output JSON has identical schema (scenarios list with just one element) so the
aggregate.py downstream can read it.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/tmp")
from sub094_benchmark import build_mix, run_scenario  # type: ignore

import httpx


async def main_async(args: argparse.Namespace) -> None:
    weights_map = {
        "balanced": {"sonnet": 0.34, "chat": 0.33, "code": 0.33},
        "sonnet-heavy": {"sonnet": 0.60, "chat": 0.20, "code": 0.20},
        "code-heavy": {"sonnet": 0.10, "chat": 0.20, "code": 0.70},
    }
    weights = weights_map[args.mix]
    prompts, mix_counts = build_mix(args.num_prompts, weights, args.seed)
    print(f"[mix={args.mix}] counts={mix_counts}")

    sc = await run_scenario(
        "agsd-gated",
        "http://127.0.0.1:8000/generate",
        prompts, args.max_tokens, args.concurrency, is_router=True,
    )

    gating_stats = {}
    try:
        async with httpx.AsyncClient() as c:
            gating_stats = (await c.get("http://127.0.0.1:8000/stats", timeout=5)).json()
    except Exception:
        pass

    result = {
        "mix": args.mix,
        "mix_counts": mix_counts,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "scenarios": [sc],
        "gating_stats": gating_stats,
        "agsd_only": True,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"benchmark_{args.mix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[saved] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-prompts", type=int, default=500)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--mix", type=str, default="balanced")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
