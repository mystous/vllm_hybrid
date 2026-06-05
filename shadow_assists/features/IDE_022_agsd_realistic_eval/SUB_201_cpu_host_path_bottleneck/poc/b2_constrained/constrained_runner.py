"""B2 constrained-decode runner — baseline vs JSON schema constrained vs grammar.

Same shape as throughput_runner but with:
  - --mode {baseline, json_schema, grammar} → adds vLLM extra_body.guided_*
  - JSON schema = simple 5-key object (name, age, email, country, score)
  - grammar = simple EBNF: list of words separated by commas

Measures: output_tps, TTFT/TPOT, gpu_util, cpu_util.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import threading
import time
from collections import defaultdict

import httpx
import pyarrow.parquet as pq


# ---- guided spec presets ---------------------------------------------------

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

GRAMMAR_WORDLIST = r"""
root ::= word ("," " "? word)*
word ::= [a-zA-Z]+
"""

# JSON-friendly system prompt to make schema-constrained generation faster
JSON_SYS = (
    "You are a JSON generator. Given the user prompt, respond ONLY with a single JSON "
    "object that fits the user's request, using these 5 fields exactly: "
    "name (string), age (integer), email (string), country (string), score (number). "
    "Do not include any text outside the JSON object."
)
GRAMMAR_SYS = (
    "List comma-separated words (letters only). Up to 30 words."
)


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
                        us.append(float(a)); ms += float(b)
                if us:
                    self.gpu_util.append(sum(us) / len(us)); self.gpu_mem.append(ms)
            except Exception:
                pass
            i1, t1 = _cpu_busy()
            if t1 > t0:
                self.cpu.append(100.0 * (1 - (i1 - i0) / (t1 - t0)))
            i0, t0 = i1, t1
            if self._stopev.wait(1.0):
                break

    def stop_means(self):
        self._stopev.set(); self.join(timeout=3)
        m = lambda x: round(sum(x) / len(x), 1) if x else None
        return m(self.gpu_util), (round(sum(self.gpu_mem) / len(self.gpu_mem)) if self.gpu_mem else None), m(self.cpu)


def _pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return round(sorted_vals[i], 1)


def _build_payload(args, rec):
    if args.mode == "baseline":
        prompt = rec["raw_text"]
        payload = {
            "model": args.model, "prompt": prompt, "max_tokens": args.max_tokens,
            "temperature": 0.0, "top_p": 1.0, "stream": True,
        }
    elif args.mode == "json_schema":
        # Prepend a brief instruction (kept inside prompt to avoid changing
        # endpoint to chat — completions endpoint is fine).
        prompt = f"{JSON_SYS}\n\nUser prompt: {rec['raw_text']}\n\nJSON:"
        payload = {
            "model": args.model, "prompt": prompt, "max_tokens": args.max_tokens,
            "temperature": 0.0, "top_p": 1.0, "stream": True,
            "guided_json": JSON_SCHEMA_5KEY,
        }
    elif args.mode == "grammar":
        prompt = f"{GRAMMAR_SYS}\n\nUser prompt: {rec['raw_text']}\n\nWords:"
        payload = {
            "model": args.model, "prompt": prompt, "max_tokens": args.max_tokens,
            "temperature": 0.0, "top_p": 1.0, "stream": True,
            "guided_grammar": GRAMMAR_WORDLIST,
        }
    else:
        raise ValueError(args.mode)
    payload["stream_options"] = {"include_usage": True}
    return payload


async def _one(client, url, args, rec, sem):
    async with sem:
        payload = _build_payload(args, rec)
        t0 = time.perf_counter()
        try:
            ttft = None; usage = {}
            async with client.stream("POST", f"{url}/completions", json=payload,
                                     timeout=httpx.Timeout(3600.0, connect=10.0)) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode("utf-8", "ignore")[:200]
                    return {"corpus": rec["corpus"], "ok": False,
                            "wall_ms": (time.perf_counter()-t0)*1000.0,
                            "completion_tokens": 0, "error": f"HTTP {r.status_code}: {body}"}
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
            tpot = round((wall - ttft) / (c - 1), 2) if (ttft is not None and c > 1) else None
            return {"corpus": rec["corpus"], "ok": True, "wall_ms": wall, "ttft_ms": ttft,
                    "tpot_ms": tpot, "completion_tokens": c,
                    "prompt_tokens": usage.get("prompt_tokens", 0)}
        except Exception as e:
            return {"corpus": rec["corpus"], "ok": False,
                    "wall_ms": (time.perf_counter()-t0)*1000.0,
                    "completion_tokens": 0, "error": repr(e)[:200]}


async def run(args):
    rows = pq.read_table(args.inp).to_pylist()
    if args.corpus:
        rows = [r for r in rows if r["corpus"] == args.corpus]
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    base = f"http://{args.host}:{args.port}"
    url = base + "/v1"
    sem = asyncio.Semaphore(args.concurrency)
    sampler = UtilSampler(); sampler.start()
    async with httpx.AsyncClient(limits=httpx.Limits(
            max_connections=args.concurrency*2, max_keepalive_connections=args.concurrency*2)) as cl:
        t0 = time.perf_counter()
        res = await asyncio.gather(*(_one(cl, url, args, r, sem) for r in rows))
        wall = time.perf_counter() - t0
    gpu_util, gpu_mem_mib, cpu_util = sampler.stop_means()
    ok = [r for r in res if r["ok"]]
    tot_c = sum(r["completion_tokens"] for r in ok)
    ttfts = sorted(r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None)
    tpots = sorted(r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None)
    by = defaultdict(list)
    for r in ok:
        if r["wall_ms"] > 0:
            by[r["corpus"]].append(r["completion_tokens"] / (r["wall_ms"]/1000.0))
    per_corpus = {c: round(sum(v)/len(v), 1) for c, v in by.items()}
    summary = {
        "model": args.model_tag or args.model, "mode": args.mode,
        "n": len(rows), "n_ok": len(ok),
        "concurrency": args.concurrency, "max_tokens": args.max_tokens,
        "wall_total_s": round(wall, 1), "total_completion_tokens": tot_c,
        "output_tps": round(tot_c/wall, 1) if wall > 0 else 0.0,
        "ttft_ms_p50": _pct(ttfts, 0.5), "ttft_ms_p99": _pct(ttfts, 0.99),
        "tpot_ms_p50": _pct(tpots, 0.5), "tpot_ms_p99": _pct(tpots, 0.99),
        "gpu_util": gpu_util, "gpu_mem_mib": gpu_mem_mib, "cpu_util": cpu_util,
        "per_corpus_reqtps": per_corpus,
        "n_err": len(res)-len(ok),
        "first_err": next((r.get("error") for r in res if not r["ok"]), None),
    }
    if args.raw:
        with open(args.raw, "a") as f:
            for r in res:
                r2 = dict(r); r2["model"] = args.model_tag or args.model; r2["mode"] = args.mode
                f.write(json.dumps(r2)+"\n")
    print(f"[b2:{args.mode}] {summary['output_tps']} tps "
          f"({len(ok)}/{len(rows)} ok, {wall:.0f}s, gpu {gpu_util}% cpu {cpu_util}% "
          f"ttft_p50 {summary['ttft_ms_p50']}ms tpot_p50 {summary['tpot_ms_p50']}ms "
          f"n_err={summary['n_err']})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--mode", choices=["baseline", "json_schema", "grammar"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--corpus", default="sharegpt")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", default=None)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
