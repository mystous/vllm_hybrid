"""L12 — burst-pattern TTFT bench (low → high concurrency pulse).

The standard throughput_runner.py fires all N prompts at once with a
fixed concurrency cap.  L12 cares about *transitions*:

  Phase 1 (warm)   – 4 concurrent reqs for ``warm_s`` seconds
  Phase 2 (burst)  – ramp up to ``peak_conc`` over ``ramp_s`` seconds
  Phase 3 (steady) – hold ``peak_conc`` for ``hold_s`` seconds
  Phase 4 (cool)   – drop back to 4

We record per-request {arrival_phase, ttft_ms, wall_ms, tokens} and emit:

  * TTFT p50/p90/p99 per phase (especially Phase 2 = "burst onset")
  * Top-N worst TTFT requests + their arrival timestamp & phase

Bench targets a single vLLM server (the caller starts it on GPU 5 with
TP=1 Qwen2.5-7B) with stream=True so TTFT is well-defined.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections import defaultdict

import httpx
import pyarrow.parquet as pq


PHASE_NAMES = ("warm", "burst", "steady", "cool")


async def _one(client, url, model, rec, max_tokens, arrival_t, phase):
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
    usage: dict = {}
    try:
        async with client.stream(
            "POST",
            f"{url}/completions",
            json=payload,
            timeout=httpx.Timeout(3600.0, connect=10.0),
        ) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", "ignore")[:120]
                return {
                    "ok": False,
                    "arrival_t": arrival_t,
                    "phase": phase,
                    "wall_ms": (time.perf_counter() - t0) * 1000.0,
                    "ttft_ms": None,
                    "completion_tokens": 0,
                    "error": f"HTTP {r.status_code}: {body}",
                }
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:  # noqa: BLE001
                    continue
                ch = obj.get("choices") or []
                if ttft is None and ch and ch[0].get("text"):
                    ttft = (time.perf_counter() - t0) * 1000.0
                if obj.get("usage"):
                    usage = obj["usage"]
        wall = (time.perf_counter() - t0) * 1000.0
        c = usage.get("completion_tokens", 0)
        return {
            "ok": True,
            "arrival_t": arrival_t,
            "phase": phase,
            "wall_ms": wall,
            "ttft_ms": ttft,
            "completion_tokens": c,
            "prompt_tokens": usage.get("prompt_tokens", 0),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "arrival_t": arrival_t,
            "phase": phase,
            "wall_ms": (time.perf_counter() - t0) * 1000.0,
            "ttft_ms": None,
            "completion_tokens": 0,
            "error": repr(e)[:120],
        }


async def driver(args):
    rows = pq.read_table(args.inp).to_pylist()
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    base = f"http://{args.host}:{args.port}"
    url = base + "/v1"

    # Build arrival timeline.  Total prompts = warm_rate*warm_s + ... but
    # we cap at len(rows). Phase boundaries are wall-clock.
    timeline: list[tuple[float, str]] = []  # (arrival_s, phase)
    t = 0.0
    # Phase 1: warm — constant low rate
    for i in range(int(args.warm_s * args.warm_rate)):
        timeline.append((t, "warm"))
        t += 1.0 / args.warm_rate
    # Phase 2: burst — ramp rate from warm_rate to peak_rate over ramp_s
    n_burst = int(args.ramp_s * (args.warm_rate + args.peak_rate) / 2)
    for i in range(n_burst):
        progress = i / max(1, n_burst - 1)
        rate = args.warm_rate + (args.peak_rate - args.warm_rate) * progress
        timeline.append((t, "burst"))
        t += 1.0 / max(0.01, rate)
    # Phase 3: steady — hold at peak_rate
    for i in range(int(args.hold_s * args.peak_rate)):
        timeline.append((t, "steady"))
        t += 1.0 / args.peak_rate
    # Phase 4: cool — back to warm_rate
    for i in range(int(args.cool_s * args.warm_rate)):
        timeline.append((t, "cool"))
        t += 1.0 / args.warm_rate

    # Truncate to dataset size
    if len(timeline) > len(rows):
        timeline = timeline[: len(rows)]
    n = len(timeline)
    duration_s = timeline[-1][0] if timeline else 0
    print(
        f"[bench] timeline: {n} reqs over {duration_s:.1f}s "
        f"(warm={int(args.warm_s)}s@{args.warm_rate}rps, "
        f"burst={int(args.ramp_s)}s ramp→{args.peak_rate}rps, "
        f"steady={int(args.hold_s)}s@{args.peak_rate}rps, "
        f"cool={int(args.cool_s)}s@{args.warm_rate}rps)"
    )

    pending: list[asyncio.Task] = []
    t0 = time.perf_counter()

    async with httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max(64, int(args.peak_rate * 4)),
            max_keepalive_connections=max(64, int(args.peak_rate * 4)),
        )
    ) as cl:
        for idx, (arrival_s, phase) in enumerate(timeline):
            wait_until = t0 + arrival_s
            now = time.perf_counter()
            if wait_until > now:
                await asyncio.sleep(wait_until - now)
            arrival_t = time.perf_counter() - t0
            rec = rows[idx % len(rows)]
            pending.append(
                asyncio.create_task(
                    _one(
                        cl,
                        url,
                        args.model,
                        rec,
                        args.max_tokens,
                        arrival_t,
                        phase,
                    )
                )
            )

        print(f"[bench] all {n} requests dispatched; awaiting completion…")
        results = await asyncio.gather(*pending)

    wall = time.perf_counter() - t0
    return results, wall, duration_s


def summarise(results, wall, duration_s, out_path: str):
    ok = [r for r in results if r["ok"]]
    by_phase: dict[str, list] = defaultdict(list)
    for r in ok:
        if r["ttft_ms"] is not None:
            by_phase[r["phase"]].append(r["ttft_ms"])

    def _pct(xs, q):
        if not xs:
            return None
        xs = sorted(xs)
        i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
        return round(xs[i], 1)

    phase_stats = {}
    for ph in PHASE_NAMES:
        xs = by_phase[ph]
        phase_stats[ph] = {
            "n": len(xs),
            "ttft_p50": _pct(xs, 0.5),
            "ttft_p90": _pct(xs, 0.9),
            "ttft_p99": _pct(xs, 0.99),
            "ttft_max": round(max(xs), 1) if xs else None,
            "ttft_mean": round(sum(xs) / len(xs), 1) if xs else None,
        }

    # Top-10 worst TTFT
    worst = sorted(
        (r for r in ok if r["ttft_ms"] is not None),
        key=lambda r: -r["ttft_ms"],
    )[:10]
    worst_brief = [
        {
            "arrival_t": round(r["arrival_t"], 2),
            "phase": r["phase"],
            "ttft_ms": round(r["ttft_ms"], 1),
            "wall_ms": round(r["wall_ms"], 1),
            "tok": r["completion_tokens"],
        }
        for r in worst
    ]

    total_tok = sum(r["completion_tokens"] for r in ok)

    summary = {
        "n": len(results),
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "wall_s": round(wall, 1),
        "scheduled_duration_s": round(duration_s, 1),
        "completion_tokens_total": total_tok,
        "output_tps": round(total_tok / wall, 1) if wall > 0 else 0.0,
        "phase_stats": phase_stats,
        "worst10_ttft": worst_brief,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    # Pretty stdout
    print(f"[bench] wall={wall:.1f}s ok={len(ok)}/{len(results)} "
          f"tps={summary['output_tps']}")
    for ph in PHASE_NAMES:
        ps = phase_stats[ph]
        if ps["n"]:
            print(
                f"  {ph:>6}: n={ps['n']:4d}  TTFT p50={ps['ttft_p50']:>6}  "
                f"p90={ps['ttft_p90']:>6}  p99={ps['ttft_p99']:>7}  "
                f"max={ps['ttft_max']:>7}"
            )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8005)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="kept small to fit many bursts in budget")
    ap.add_argument("--warm-s", type=int, default=20)
    ap.add_argument("--warm-rate", type=float, default=2.0,
                    help="reqs/sec during warm phase")
    ap.add_argument("--ramp-s", type=int, default=10)
    ap.add_argument("--hold-s", type=int, default=30)
    ap.add_argument("--peak-rate", type=float, default=20.0,
                    help="reqs/sec during burst peak / steady")
    ap.add_argument("--cool-s", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results, wall, duration_s = asyncio.run(driver(args))
    summarise(results, wall, duration_s, args.out)


if __name__ == "__main__":
    main()
