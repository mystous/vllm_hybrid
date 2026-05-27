"""Mixed-traffic AGSD benchmark client — SUB_094 reproduction.

3 mix scenario (balanced / sonnet-heavy / code-heavy) × 200 prompts × concurrency=32.
3 backend scenario (vanilla-only / trident-only / AGSD-gated) 비교.

사용법:
    .venv/bin/python benchmark_mixed.py --scenario AGSD --mix balanced --out result.json

scenario:
    vanilla    — 모든 traffic 을 http://127.0.0.1:8001 로
    trident    — 모든 traffic 을 http://127.0.0.1:8002 로
    AGSD       — 모든 traffic 을 router http://127.0.0.1:8000 으로

mix (200 prompt 분포):
    balanced     — 68 sonnet : 66 chat : 66 code
    sonnet-heavy — 120 sonnet : 40 chat : 40 code
    code-heavy   — 20 sonnet : 40 chat : 140 code

본 client 는 OpenAI-compat /v1/completions endpoint 사용.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from typing import Any

import httpx

# ---- prompt builders ----
SONNET_PROMPT = (
    "Shall I compare thee to a summer's day? Thou art more lovely and more temperate. "
    "Rough winds do shake the darling buds of May, and summer's lease hath all too short a date. "
    "Continue this sonnet for at least 100 lines, maintaining iambic pentameter."
)

CHAT_PROMPT = (
    "<|system|>\nYou are a helpful assistant.<|user|>\n"
    "Explain the difference between speculative decoding and beam search in transformer inference, "
    "with examples.\n<|assistant|>\n"
)

CODE_PROMPT_TEMPLATE = """import os
import sys
import json
import time
import re
# main entry
def process_batch(items, batch_size=32, max_workers=8):
    # logic line 1: validate input
    # logic line 2: chunk into batches
    # logic line 3: spawn workers
    # logic line 4: collect results
    # logic line 5: error handling
    # logic line 6: cleanup
    # logic line 7: emit metrics
    # logic line 8: return aggregate
    # logic line 9: log completion
    # logic line 10: shutdown
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        if batch:
            yield batch
class BatchProcessor:
    def __init__(self, config):
        self.config = config
    def run(self):
        try:
            return process_batch(self.config.items)
        except Exception as e:
            raise
# Complete the implementation above with full error handling, retry logic, and tests.
"""

MIX_DEFINITIONS: dict[str, dict[str, int]] = {
    "balanced": {"sonnet": 68, "chat": 66, "code": 66},
    "sonnet-heavy": {"sonnet": 120, "chat": 40, "code": 40},
    "code-heavy": {"sonnet": 20, "chat": 40, "code": 140},
}


def build_mix(mix_name: str, seed: int = 0) -> list[tuple[str, str]]:
    """mix scenario → list of (workload, prompt)."""
    spec = MIX_DEFINITIONS[mix_name]
    items: list[tuple[str, str]] = []
    for workload, n in spec.items():
        if workload == "sonnet":
            for i in range(n):
                items.append(("sonnet", f"{SONNET_PROMPT} (variant #{i})"))
        elif workload == "chat":
            for i in range(n):
                items.append(("chat", CHAT_PROMPT + f" (#{i})"))
        elif workload == "code":
            for i in range(n):
                items.append(("code", CODE_PROMPT_TEMPLATE + f"\n# variant #{i}\n"))
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


BACKEND_URL: dict[str, str] = {
    "vanilla": "http://127.0.0.1:8001/v1",
    "trident": "http://127.0.0.1:8002/v1",
    "AGSD": "http://127.0.0.1:8000/v1",
}


async def _request_one(
    client: httpx.AsyncClient,
    url: str,
    workload: str,
    prompt: str,
    max_tokens: int,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{url}/completions",
                json={
                    "model": "model",  # vLLM serve 영역 model name 영역 ignore
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stream": False,
                },
                timeout=httpx.Timeout(600.0, connect=10.0),
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            ok = resp.status_code == 200
            tokens = 0
            if ok:
                data = resp.json()
                # OpenAI completion format
                tokens = data.get("usage", {}).get("completion_tokens", 0)
            return {
                "workload": workload,
                "ok": ok,
                "wall_ms": wall_ms,
                "tokens": tokens,
            }
        except Exception as e:
            return {
                "workload": workload,
                "ok": False,
                "wall_ms": (time.perf_counter() - t0) * 1000.0,
                "tokens": 0,
                "error": str(e),
            }


async def run(
    scenario: str, mix_name: str, max_tokens: int, concurrency: int, out_path: str
) -> None:
    items = build_mix(mix_name)
    url = BACKEND_URL[scenario]
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=concurrency * 2)
    ) as client:
        t_start = time.perf_counter()
        results = await asyncio.gather(
            *(
                _request_one(client, url, w, p, max_tokens, sem)
                for w, p in items
            )
        )
        wall_total_s = time.perf_counter() - t_start

    ok_n = sum(r["ok"] for r in results)
    total_tokens = sum(r["tokens"] for r in results if r["ok"])
    tps = total_tokens / wall_total_s if wall_total_s > 0 else 0.0
    walls = sorted(r["wall_ms"] for r in results if r["ok"])
    p50 = walls[len(walls) // 2] if walls else 0.0
    p99 = walls[int(len(walls) * 0.99)] if walls else 0.0

    summary = {
        "scenario": scenario,
        "mix": mix_name,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "n_prompts": len(items),
        "n_ok": ok_n,
        "wall_total_s": wall_total_s,
        "tokens_total": total_tokens,
        "tps": tps,
        "p50_ms": p50,
        "p99_ms": p99,
        "by_workload": {
            w: sum(1 for r in results if r["workload"] == w and r["ok"])
            for w in ("sonnet", "chat", "code")
        },
    }

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"=== {scenario} × {mix_name} ===")
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scenario", choices=list(BACKEND_URL.keys()), required=True
    )
    p.add_argument("--mix", choices=list(MIX_DEFINITIONS.keys()), required=True)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--out", default="benchmark_result.json")
    args = p.parse_args()
    asyncio.run(
        run(args.scenario, args.mix, args.max_tokens, args.concurrency, args.out)
    )


if __name__ == "__main__":
    main()
