#!/usr/bin/env python3
"""SUB_239 FERRY e2e A/B 부하 클라이언트.

OpenAI-호환 endpoint 에 high-concurrency + 긴 컨텍스트 요청을 쏘아 NEO swap-out/in 을
발화시키고 throughput(tps) + per-request 지연(TTFT/e2e) 분포를 측정한다.

사용: python ferry_e2e_load.py --base http://127.0.0.1:8200 --model <m> \
        --concurrency 48 --prompt-tokens 6000 --max-tokens 256 --requests 240 --tag A
"""
import argparse
import asyncio
import json
import time

import aiohttp


def make_prompt(approx_tokens, idx):
    # 결정적·긴 프롬프트 (토큰≈단어수). idx 로 약간 변주(캐시 prefix 공유 억제).
    head = f"[req {idx}] Summarize and continue the following technical log in detail. "
    body = ("The Data Streaming Accelerator processes descriptors through work queues "
            "while the memory controller arbitrates bandwidth across NUMA domains. ")
    n = max(1, approx_tokens // 24)
    return head + body * n


async def one(session, base, model, prompt, max_tokens, results, seed):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,           # greedy → 결정적
        "seed": seed,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(f"{base}/v1/completions", json=payload) as r:
            data = await r.json()
            dt = time.perf_counter() - t0
            if "usage" in data:
                ct = data["usage"].get("completion_tokens", 0)
                pt = data["usage"].get("prompt_tokens", 0)
            else:
                ct = pt = 0
            results.append({"dt": dt, "ct": ct, "pt": pt,
                            "text": data.get("choices", [{}])[0].get("text", "")[:80]})
    except Exception as e:  # noqa: BLE001
        results.append({"dt": time.perf_counter()-t0, "ct": 0, "pt": 0, "err": str(e)[:120]})


async def run(args):
    results = []
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def guarded(i):
            async with sem:
                await one(session, args.base, args.model,
                          make_prompt(args.prompt_tokens, i), args.max_tokens, results, i)
        t0 = time.perf_counter()
        await asyncio.gather(*[guarded(i) for i in range(args.requests)])
        wall = time.perf_counter() - t0
    return results, wall


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs)-1, int(p*len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8200")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--prompt-tokens", type=int, default=6000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--requests", type=int, default=240)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--tag", default="A")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    results, wall = asyncio.run(run(args))
    errs = [r for r in results if "err" in r]
    ok = [r for r in results if "err" not in r]
    gen = sum(r["ct"] for r in ok)
    prompt_tok = sum(r["pt"] for r in ok)
    dts = [r["dt"] for r in ok]
    tps_gen = gen / wall if wall else 0
    tps_total = (gen + prompt_tok) / wall if wall else 0
    print(f"[{args.tag}] requests={len(results)} ok={len(ok)} err={len(errs)} wall={wall:.2f}s")
    print(f"[{args.tag}] gen_tokens={gen} prompt_tokens={prompt_tok}")
    print(f"[{args.tag}] throughput: gen={tps_gen:.1f} tok/s  total(incl prompt)={tps_total:.1f} tok/s")
    print(f"[{args.tag}] req latency p50={pct(dts,0.5):.2f}s p95={pct(dts,0.95):.2f}s "
          f"p99={pct(dts,0.99):.2f}s max={max(dts) if dts else 0:.2f}s")
    if errs:
        print(f"[{args.tag}] first err: {errs[0]['err']}")
    line = (f"FERRY_E2E,tag={args.tag},ok={len(ok)},err={len(errs)},wall_s={wall:.2f},"
            f"gen_tps={tps_gen:.1f},total_tps={tps_total:.1f},"
            f"p50_s={pct(dts,0.5):.2f},p95_s={pct(dts,0.95):.2f},p99_s={pct(dts,0.99):.2f}")
    print(line)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")
            # 결정적 출력 스폿체크용 첫 3개 텍스트 저장
            for r in ok[:3]:
                f.write(f"# [{args.tag}] sample: {json.dumps(r['text'], ensure_ascii=False)}\n")


if __name__ == "__main__":
    main()
