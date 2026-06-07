"""Stack L2+L10 on suffix+FaP — sharegpt 200p × conc=16 × max-tok 512.

streaming completions; reports tps + TTFT (p50/p90/p99) + TPOT (p50/p99)
+ spec accept α + GPU util + CPU%.

usage:
  bench.py --run A --port 8009 --out runs/A.json --raw runs/A.raw.jsonl
"""
from __future__ import annotations
import argparse
import asyncio
import json
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import httpx
import pyarrow.parquet as pq


def _cpu_busy():
    with open("/proc/stat") as f:
        v = list(map(int, f.readline().split()[1:]))
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    return idle, sum(v)


class UtilSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stopev = threading.Event()
        self.gpu_util, self.gpu_mem, self.cpu = [], [], []

    def run(self):
        i0, t0 = _cpu_busy()
        while True:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"], capture_output=True,
                    text=True, timeout=5).stdout.strip().splitlines()
                us, ms = [], 0.0
                for ln in out:
                    a, b = ln.split(",")
                    if float(b) > 1000:
                        us.append(float(a))
                        ms += float(b)
                if us:
                    self.gpu_util.append(sum(us) / len(us))
                    self.gpu_mem.append(ms)
            except Exception:
                pass
            i1, t1 = _cpu_busy()
            if t1 > t0:
                self.cpu.append(100.0 * (1 - (i1 - i0) / (t1 - t0)))
            i0, t0 = i1, t1
            if self._stopev.wait(1.0):
                break

    def stop_means(self):
        self._stopev.set()
        self.join(timeout=3)
        m = lambda x: round(sum(x) / len(x), 1) if x else None
        return (m(self.gpu_util),
                round(sum(self.gpu_mem) / len(self.gpu_mem)) if self.gpu_mem else None,
                m(self.cpu))


def _pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return round(sorted_vals[i], 1)


def _scrape_spec(base):
    try:
        txt = httpx.get(f"{base}/metrics", timeout=10.0).text
    except Exception:
        return None, None
    acc = draft = None
    for ln in txt.splitlines():
        if ln.startswith("#") or " " not in ln:
            continue
        if ln.startswith("vllm:spec_decode_num_accepted_tokens_total{"):
            acc = (acc or 0.0) + float(ln.rsplit(" ", 1)[-1])
        elif ln.startswith("vllm:spec_decode_num_draft_tokens_total{"):
            draft = (draft or 0.0) + float(ln.rsplit(" ", 1)[-1])
    return acc, draft


async def _one(client, url, model, rec, max_tokens, sem):
    async with sem:
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
        try:
            ttft = None
            usage = {}
            async with client.stream(
                "POST", f"{url}/completions", json=payload,
                timeout=httpx.Timeout(3600.0, connect=10.0),
            ) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode("utf-8", "ignore")[:120]
                    return {"corpus": rec["corpus"], "ok": False,
                            "wall_ms": (time.perf_counter()-t0)*1000.0,
                            "completion_tokens": 0,
                            "error": f"HTTP {r.status_code}: {body}"}
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
            wall = (time.perf_counter() - t0) * 1000.0
            c = usage.get("completion_tokens", 0)
            tpot = round((wall - ttft) / (c - 1), 3) if (ttft is not None and c > 1) else None
            return {"corpus": rec["corpus"], "ok": True, "wall_ms": wall,
                    "ttft_ms": ttft, "tpot_ms": tpot,
                    "completion_tokens": c,
                    "prompt_tokens": usage.get("prompt_tokens", 0)}
        except Exception as e:
            return {"corpus": rec["corpus"], "ok": False,
                    "wall_ms": (time.perf_counter()-t0)*1000.0,
                    "completion_tokens": 0, "error": repr(e)[:120]}


async def run_bench(args):
    rows = pq.read_table(args.parquet).to_pylist()
    if args.limit:
        rows = rows[: args.limit]
    base = f"http://127.0.0.1:{args.port}"
    url = base + "/v1"
    sem = asyncio.Semaphore(args.concurrency)

    acc0, draft0 = _scrape_spec(base)
    sampler = UtilSampler(); sampler.start()
    async with httpx.AsyncClient(limits=httpx.Limits(
            max_connections=args.concurrency*2,
            max_keepalive_connections=args.concurrency*2)) as cl:
        t0 = time.perf_counter()
        res = await asyncio.gather(*(
            _one(cl, url, args.model, r, args.max_tokens, sem) for r in rows
        ))
        wall = time.perf_counter() - t0
    gpu_util, gpu_mem, cpu_util = sampler.stop_means()
    acc1, draft1 = _scrape_spec(base)

    accept_rate = acc_tok = draft_tok = None
    if None not in (acc0, acc1, draft0, draft1):
        acc_tok, draft_tok = acc1 - acc0, draft1 - draft0
        accept_rate = round(acc_tok / draft_tok, 4) if draft_tok > 0 else None

    ok = [r for r in res if r["ok"]]
    tot_c = sum(r["completion_tokens"] for r in ok)
    ttfts = sorted(r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None)
    tpots = sorted(r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None)

    summary = {
        "run": args.run,
        "model": args.model,
        "n": len(rows),
        "n_ok": len(ok),
        "n_err": len(res) - len(ok),
        "first_err": next((r.get("error") for r in res if not r["ok"]), None),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "wall_total_s": round(wall, 2),
        "total_completion_tokens": tot_c,
        "output_tps": round(tot_c / wall, 1) if wall > 0 else 0.0,
        "ttft_ms_p50": _pct(ttfts, 0.5),
        "ttft_ms_p90": _pct(ttfts, 0.9),
        "ttft_ms_p99": _pct(ttfts, 0.99),
        "tpot_ms_p50": _pct(tpots, 0.5),
        "tpot_ms_p99": _pct(tpots, 0.99),
        "accept_rate": accept_rate,
        "accept_tokens": acc_tok,
        "draft_tokens": draft_tok,
        "gpu_util": gpu_util,
        "gpu_mem_mib": gpu_mem,
        "cpu_util": cpu_util,
    }
    print("[bench]", json.dumps(summary, indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    if args.raw:
        with open(args.raw, "w") as f:
            for r in res:
                r2 = dict(r); r2["run"] = args.run
                f.write(json.dumps(r2) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="A|B|C|D|E")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument(
        "--parquet",
        default="/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b3_8gpu_full/sharegpt200.parquet",
    )
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=None)
    args = ap.parse_args()
    asyncio.run(run_bench(args))


if __name__ == "__main__":
    main()
