"""[SUB_201/L10] Burst-pattern client.

Simulates a workload that L10 specifically targets:
  * Bimodal max_tokens: 70% short (max=64) + 30% long (max=2048).
  * Bursty arrivals: groups of size 1..32 arriving according to a
    Poisson process — between bursts, the system briefly drains.
  * Measures TTFT and TPOT per request via OpenAI-compatible streaming.

We then compute p50 and p99 of TTFT and TPOT — the L10 hypothesis is
that burst-aware admission lowers p99 TTFT under bimodal workloads
because short jobs no longer get blocked behind long ones.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import httpx
import pyarrow.parquet as pq


SHORT_MAX = 64
LONG_MAX = 2048
LONG_RATIO = 0.30  # 30% requests are "long"


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return round(s[i], 1)


async def _one(client, base_url, model, rec, max_tokens, t_start, results):
    """Issue one streaming completion and record TTFT/TPOT."""
    payload = {
        "model": model,
        "prompt": rec["raw_text"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    usage = {}
    err = None
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/completions",
            json=payload,
            timeout=httpx.Timeout(3600.0, connect=10.0),
        ) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", "ignore")[:120]
                err = f"HTTP {r.status_code}: {body}"
            else:
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    ch = obj.get("choices") or []
                    if ttft is None and ch and ch[0].get("text"):
                        ttft = (time.perf_counter() - t0) * 1000.0
                    if obj.get("usage"):
                        usage = obj["usage"]
    except Exception as e:
        err = repr(e)[:120]
    wall = (time.perf_counter() - t0) * 1000.0
    c = usage.get("completion_tokens", 0)
    tpot = round((wall - ttft) / (c - 1), 2) if (ttft is not None and c > 1) else None
    results.append(
        {
            "arrival_offset_ms": round((t0 - t_start) * 1000.0, 2),
            "wall_ms": round(wall, 2),
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "completion_tokens": c,
            "max_tokens": max_tokens,
            "is_long": max_tokens >= LONG_MAX,
            "ok": err is None,
            "err": err,
        }
    )


async def run(args):
    rng = random.Random(args.seed)
    rows_all = pq.read_table(args.inp).to_pylist()
    rng.shuffle(rows_all)

    # Construct burst arrival schedule.
    # Each burst: size ~ Uniform[1, args.burst_max], then quiet for
    # ~Exp(mean = args.idle_mean_s).
    schedule = []
    t_cur = 0.0
    while len(schedule) < args.n_requests:
        burst_size = rng.randint(1, args.burst_max)
        # All requests in a burst arrive within burst_dur (instantaneous-ish).
        for _ in range(burst_size):
            jitter = rng.uniform(0, args.burst_dur_s)
            schedule.append(t_cur + jitter)
            if len(schedule) >= args.n_requests:
                break
        t_cur += args.burst_dur_s + rng.expovariate(1.0 / max(args.idle_mean_s, 1e-3))
    schedule = schedule[: args.n_requests]
    schedule.sort()

    # Pick prompt + assign max_tokens (bimodal).
    plan = []
    for i, off in enumerate(schedule):
        rec = rows_all[i % len(rows_all)]
        is_long = rng.random() < LONG_RATIO
        plan.append((off, rec, LONG_MAX if is_long else SHORT_MAX))

    base_url = f"http://{args.host}:{args.port}"
    # Warmup once to ensure the server is hot.
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=64)
    ) as cl:
        try:
            await cl.post(
                f"{base_url}/v1/completions",
                json={
                    "model": args.model,
                    "prompt": "Hello",
                    "max_tokens": 4,
                    "temperature": 0.0,
                },
                timeout=60.0,
            )
        except Exception:
            pass

        results: list[dict] = []
        t_start = time.perf_counter()
        tasks: list[asyncio.Task] = []
        for off, rec, max_tok in plan:
            wait = off - (time.perf_counter() - t_start)
            if wait > 0:
                await asyncio.sleep(wait)
            tasks.append(
                asyncio.create_task(
                    _one(cl, base_url, args.model, rec, max_tok, t_start, results)
                )
            )
        await asyncio.gather(*tasks)
        wall_total = time.perf_counter() - t_start

    ok = [r for r in results if r["ok"]]
    short_ok = [r for r in ok if not r["is_long"]]
    long_ok = [r for r in ok if r["is_long"]]

    def summarize(label, lst):
        ttfts = [r["ttft_ms"] for r in lst if r["ttft_ms"] is not None]
        tpots = [r["tpot_ms"] for r in lst if r["tpot_ms"] is not None]
        return {
            "label": label,
            "n": len(lst),
            "ttft_ms_p50": _pct(ttfts, 0.50),
            "ttft_ms_p90": _pct(ttfts, 0.90),
            "ttft_ms_p99": _pct(ttfts, 0.99),
            "tpot_ms_p50": _pct(tpots, 0.50),
            "tpot_ms_p99": _pct(tpots, 0.99),
        }

    summary = {
        "tag": args.tag,
        "n_requested": args.n_requests,
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "wall_total_s": round(wall_total, 1),
        "burst_max": args.burst_max,
        "burst_dur_s": args.burst_dur_s,
        "idle_mean_s": args.idle_mean_s,
        "short_max_tok": SHORT_MAX,
        "long_max_tok": LONG_MAX,
        "long_ratio": LONG_RATIO,
        "overall": summarize("overall", ok),
        "short": summarize("short", short_ok),
        "long": summarize("long", long_ok),
        "first_err": next((r["err"] for r in results if not r["ok"]), None),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    if args.raw:
        with open(args.raw, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--n-requests", type=int, default=300)
    ap.add_argument(
        "--burst-max", type=int, default=32, help="max requests per burst"
    )
    ap.add_argument(
        "--burst-dur-s",
        type=float,
        default=0.05,
        help="jitter window inside a burst",
    )
    ap.add_argument(
        "--idle-mean-s",
        type=float,
        default=0.8,
        help="mean idle gap between bursts (Exp)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", default=None)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
