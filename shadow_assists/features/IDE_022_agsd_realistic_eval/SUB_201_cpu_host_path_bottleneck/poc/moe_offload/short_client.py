"""SHORT MoE-offload A/B client.

Spec (HANDOFF §3.1):
    - 20 prompts × max_tokens=256, decode-weighted, concurrency=8.
    - same seed/prompts across A (full-GPU baseline) and B (kt-kernel offload).
    - measures decode tps, total wall, p50/p99.

Usage:
    python short_client.py --url http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 --out result_A.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx


def build_short_prompts(n: int, target_in: int, seed: int = 0) -> list[str]:
    """짧지만 일관된 prompt 20개 — decode-heavy 워크로드용 (short input, long output)."""
    sonnet_path = Path("/workspace/host_vllm_hybrid/benchmarks/sonnet.txt")
    lines = [ln.strip() for ln in sonnet_path.read_text().splitlines() if ln.strip()]
    rng_master = random.Random(seed)
    prompts: list[str] = []
    for i in range(n):
        rng = random.Random(seed * 1000 + i)
        # short input ~200~300 chars
        head = (
            f"[doc {i:03d}] Continue the following passage in vivid prose for "
            f"several long paragraphs. Stay coherent and elaborate fully.\n\n"
        )
        body = "\n".join(rng.choice(lines) for _ in range(4))
        prompts.append(head + body)
    return prompts


async def _request_one(client, base_url, model, prompt, max_tokens, sem, idx):
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{base_url}/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stream": False,
                },
                timeout=httpx.Timeout(1800.0, connect=10.0),
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            ok = resp.status_code == 200
            if ok:
                usage = resp.json().get("usage", {})
                return {
                    "idx": idx, "ok": True, "wall_ms": wall_ms,
                    "ctok": usage.get("completion_tokens", 0),
                    "ptok": usage.get("prompt_tokens", 0),
                }
            return {"idx": idx, "ok": False, "wall_ms": wall_ms,
                    "ctok": 0, "ptok": 0,
                    "err": f"HTTP {resp.status_code}: {resp.text[:300]}"}
        except Exception as e:  # noqa: BLE001
            return {"idx": idx, "ok": False,
                    "wall_ms": (time.perf_counter() - t0) * 1000.0,
                    "ctok": 0, "ptok": 0, "err": repr(e)}


async def run(url, model, n, max_tokens, concurrency, seed, out_path):
    prompts = build_short_prompts(n, target_in=256, seed=seed)
    print(f"[build] n={n} sample_chars[0]={len(prompts[0])}", flush=True)

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=concurrency * 2,
                            max_keepalive_connections=concurrency * 2)
    ) as client:
        # Warm-up: 1 short request (not counted)
        warm = await _request_one(client, url, model, prompts[0][:200],
                                   16, sem, -1)
        print(f"[warm] ok={warm['ok']} wall_ms={warm['wall_ms']:.1f}",
              flush=True)
        t_start = time.perf_counter()
        results = await asyncio.gather(*(
            _request_one(client, url, model, p, max_tokens, sem, i)
            for i, p in enumerate(prompts)
        ))
        wall_total_s = time.perf_counter() - t_start

    ok = [r for r in results if r["ok"]]
    err = [r for r in results if not r["ok"]]
    total_out = sum(r["ctok"] for r in ok)
    total_in = sum(r["ptok"] for r in ok)
    walls = sorted(r["wall_ms"] for r in ok)
    p50 = walls[len(walls)//2] if walls else 0.0
    p99 = walls[min(len(walls)-1, int(len(walls)*0.99))] if walls else 0.0

    summary = {
        "url": url, "model": model,
        "n_prompts": n, "n_ok": len(ok), "n_err": len(err),
        "max_tokens": max_tokens, "concurrency": concurrency,
        "wall_total_s": wall_total_s,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "decode_tps": total_out / wall_total_s if wall_total_s > 0 else 0.0,
        "req_per_s": len(ok) / wall_total_s if wall_total_s > 0 else 0.0,
        "p50_ms": p50, "p99_ms": p99,
        "first_errs": [r.get("err") for r in err[:3]],
    }
    Path(out_path).write_text(json.dumps({"summary": summary,
                                          "raw": results}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    asyncio.run(run(args.url, args.model, args.n, args.max_tokens,
                    args.concurrency, args.seed, args.out))


if __name__ == "__main__":
    main()
