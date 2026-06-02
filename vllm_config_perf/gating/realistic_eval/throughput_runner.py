"""TSK_042 (B) canonical 처리량 runner — 실 prompt parquet → concurrency=N async.

per (model, method) 1회: 실 trace 500 prompt(자연 입력 길이) 를 conc=32 로 동시 발사,
max_tokens=8192. aggregate output_tps + per-corpus 평균(per-request tps) 집계.
(per-prompt oracle 아님 — sub097 과 동일 footing 의 처리량 측정.)

사용:
  python throughput_runner.py --in sampled.parquet --method suffix \
    --model Qwen/... --port 8002 --max-tokens 8192 --concurrency 32 \
    --limit 500 --shuffle --out summary.json --raw raw.jsonl
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


def _cpu_busy():
    with open("/proc/stat") as f:
        v = list(map(int, f.readline().split()[1:]))
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    return idle, sum(v)


class UtilSampler(threading.Thread):
    """측정 동안 GPU util/mem(active GPU만) + CPU% 를 1s 간격 샘플."""
    def __init__(self):
        super().__init__(daemon=True)
        self._stopev = threading.Event()   # NOTE: Thread._stop() 내부 메서드와 충돌 금지
        self.gpu_util, self.gpu_mem, self.cpu = [], [], []

    def run(self):
        i0, t0 = _cpu_busy()
        while True:                            # 시작 즉시 1회 샘플 → 짧은 런도 ≥1 샘플 보장
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"], capture_output=True,
                    text=True, timeout=5).stdout.strip().splitlines()
                us, ms = [], 0.0
                for ln in out:
                    a, b = ln.split(",")
                    if float(b) > 1000:        # active GPU(모델 적재)만
                        us.append(float(a)); ms += float(b)
                if us:
                    self.gpu_util.append(sum(us) / len(us)); self.gpu_mem.append(ms)
            except Exception:  # noqa: BLE001
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


def _scrape_spec(base):
    """vLLM /metrics 에서 spec accepted/draft 토큰 누적 카운터 합 (spec 미사용 시 None)."""
    try:
        txt = httpx.get(f"{base}/metrics", timeout=10.0).text
    except Exception:  # noqa: BLE001
        return None, None
    acc = draft = None
    for ln in txt.splitlines():
        if ln.startswith("#") or " " not in ln:
            continue
        # 정확한 누적 카운터만 (`_total{`). `_created`(타임스탬프)·`_per_pos_*` 제외.
        if ln.startswith("vllm:spec_decode_num_accepted_tokens_total{"):
            acc = (acc or 0.0) + float(ln.rsplit(" ", 1)[-1])
        elif ln.startswith("vllm:spec_decode_num_draft_tokens_total{"):
            draft = (draft or 0.0) + float(ln.rsplit(" ", 1)[-1])
    return acc, draft


async def _one(client, url, model, rec, max_tokens, sem, stream):
    async with sem:
        payload = {"model": model, "prompt": rec["raw_text"], "max_tokens": max_tokens,
                   "temperature": 0.0, "top_p": 1.0, "stream": stream}
        t0 = time.perf_counter()
        try:
            if not stream:
                r = await client.post(f"{url}/completions", json=payload,
                                      timeout=httpx.Timeout(3600.0, connect=10.0))
                wall = (time.perf_counter() - t0) * 1000.0
                if r.status_code != 200:
                    return {"corpus": rec["corpus"], "ok": False, "wall_ms": wall,
                            "completion_tokens": 0, "error": f"HTTP {r.status_code}: {r.text[:100]}"}
                u = r.json().get("usage", {})
                return {"corpus": rec["corpus"], "ok": True, "wall_ms": wall, "ttft_ms": None,
                        "tpot_ms": None, "completion_tokens": u.get("completion_tokens", 0),
                        "prompt_tokens": u.get("prompt_tokens", 0)}
            # streaming → TTFT / TPOT 분해
            payload["stream_options"] = {"include_usage": True}
            ttft = None; usage = {}
            async with client.stream("POST", f"{url}/completions", json=payload,
                                     timeout=httpx.Timeout(3600.0, connect=10.0)) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode("utf-8", "ignore")[:100]
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
                    except Exception:  # noqa: BLE001
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
                    "tpot_ms": tpot, "completion_tokens": c, "prompt_tokens": usage.get("prompt_tokens", 0)}
        except Exception as e:  # noqa: BLE001
            return {"corpus": rec["corpus"], "ok": False,
                    "wall_ms": (time.perf_counter()-t0)*1000.0,
                    "completion_tokens": 0, "error": repr(e)[:100]}


async def run(args):
    rows = pq.read_table(args.inp).to_pylist()
    if args.corpus:                       # 단일 corpus 격리
        rows = [r for r in rows if r["corpus"] == args.corpus]
        cond = args.corpus
    else:                                  # mix
        if args.shuffle:
            random.Random(args.seed).shuffle(rows)
        cond = "mix"
    if args.limit:
        rows = rows[: args.limit]
    base = f"http://{args.host}:{args.port}"
    url = base + "/v1"
    sem = asyncio.Semaphore(args.concurrency)
    tag = args.model_tag or args.model
    acc0, draft0 = _scrape_spec(base)              # spec α: 측정 전 카운터
    sampler = UtilSampler(); sampler.start()
    async with httpx.AsyncClient(limits=httpx.Limits(
            max_connections=args.concurrency*2, max_keepalive_connections=args.concurrency*2)) as cl:
        t0 = time.perf_counter()
        res = await asyncio.gather(*(_one(cl, url, args.model, r, args.max_tokens, sem, args.stream) for r in rows))
        wall = time.perf_counter() - t0
    gpu_util, gpu_mem_mib, cpu_util = sampler.stop_means()
    acc1, draft1 = _scrape_spec(base)              # spec α: 측정 후 카운터 → delta
    accept_rate = acc_tok = draft_tok = None
    if None not in (acc0, acc1, draft0, draft1):
        acc_tok, draft_tok = acc1 - acc0, draft1 - draft0
        accept_rate = round(acc_tok / draft_tok, 4) if draft_tok > 0 else None
    ok = [r for r in res if r["ok"]]
    tot_c = sum(r["completion_tokens"] for r in ok)
    # per-corpus 평균 per-request tps (conc 동일 → method 간 상대비교 valid)
    by = defaultdict(list)
    for r in ok:
        if r["wall_ms"] > 0:
            by[r["corpus"]].append(r["completion_tokens"] / (r["wall_ms"]/1000.0))
    per_corpus = {c: round(sum(v)/len(v), 1) for c, v in by.items()}
    ttfts = sorted(r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None)
    tpots = sorted(r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None)
    summary = {"model": tag, "method": args.method, "condition": cond,
               "n": len(rows), "n_ok": len(ok),
               "concurrency": args.concurrency, "max_tokens": args.max_tokens, "stream": args.stream,
               "wall_total_s": round(wall, 1), "total_completion_tokens": tot_c,
               "output_tps": round(tot_c/wall, 1) if wall > 0 else 0.0,
               "ttft_ms_p50": _pct(ttfts, 0.5), "ttft_ms_p99": _pct(ttfts, 0.99),
               "tpot_ms_p50": _pct(tpots, 0.5), "tpot_ms_p99": _pct(tpots, 0.99),
               "accept_rate": accept_rate, "accept_tokens": acc_tok, "draft_tokens": draft_tok,
               "gpu_util": gpu_util, "gpu_mem_mib": gpu_mem_mib, "cpu_util": cpu_util,
               "per_corpus_reqtps": per_corpus,
               "n_err": len(res)-len(ok), "first_err": next((r.get("error") for r in res if not r["ok"]), None)}
    if args.raw:
        with open(args.raw, "a") as f:
            for r in res:
                r2 = dict(r); r2["model"] = tag; r2["method"] = args.method
                r2["condition"] = cond          # 셀 단위 분리(mix vs 동일 corpus 격리분 구분)
                f.write(json.dumps(r2)+"\n")
    print(f"[tput] {tag} × {args.method} × {cond}: {summary['output_tps']} tps "
          f"({len(ok)}/{len(rows)} ok, {wall:.0f}s, gpu {gpu_util}% cpu {cpu_util}% "
          f"ttft_p50 {summary['ttft_ms_p50']}ms α {accept_rate})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--corpus", default=None, help="단일 corpus 격리 측정 (없으면 mix)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-stream", dest="stream", action="store_false",
                    help="비스트리밍 (TTFT/TPOT 미측정)")
    ap.set_defaults(stream=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", default=None)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
