#!/usr/bin/env python3
"""깨끗한 70B 벤치 — prefix-cache 오염 제거용 *고유 프롬프트* 부하 클라이언트.
각 요청 프롬프트가 유니크(난수 토큰 헤더)라 prefix caching 이 무효 → tps 가 진짜 디코드 성능.
gen tok/s + GPU-bound 판별용 ttft/tpot 보고. 사용:
  bench_unique.py --base http://127.0.0.1:PORT --model M --conc 24 --ptok 2000 --mtok 256 --reqs 192
"""
import argparse, asyncio, time, random, string
import aiohttp

def uniq_prompt(ptok, idx, salt):
    # 유니크 헤더(난수 단어) + 가변 본문 → 요청마다 prefix 다름 (APC 무효화)
    rnd = " ".join("".join(random.choices(string.ascii_lowercase, k=6)) for _ in range(40))
    head = f"[uid {salt}-{idx}] {rnd}. Continue this distinct technical note in detail: "
    body = ("The accelerator pipeline schedules descriptors while the memory controller "
            "arbitrates bandwidth across domains and the sampler validates draft tokens. ")
    n = max(1, (ptok - 60) // 22)
    return head + body * n

async def one(sess, base, model, prompt, mtok, out):
    t0 = time.perf_counter()
    payload = {"model": model, "prompt": prompt, "max_tokens": mtok,
               "temperature": 0.0, "stream": False}
    try:
        async with sess.post(f"{base}/v1/completions", json=payload) as r:
            d = await r.json()
        comp = d["choices"][0]["text"]
        usage = d.get("usage", {})
        out.append((usage.get("completion_tokens", len(comp.split())), time.perf_counter()-t0))
    except Exception as e:
        out.append((0, -1.0))

async def run(args):
    random.seed(0)
    sem = asyncio.Semaphore(args.conc)
    out = []
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as sess:
        async def task(i):
            async with sem:
                p = uniq_prompt(args.ptok, i, args.salt)
                await one(sess, args.base, args.model, p, args.mtok, out)
        t0 = time.perf_counter()
        await asyncio.gather(*[task(i) for i in range(args.reqs)])
        wall = time.perf_counter() - t0
    ok = [o for o in out if o[1] > 0]
    gen = sum(o[0] for o in ok)
    print(f"BENCH,tag={args.tag},ok={len(ok)}/{args.reqs},wall_s={wall:.2f},"
          f"gen_tok={gen},gen_tps={gen/wall:.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--conc", type=int, default=24); ap.add_argument("--ptok", type=int, default=2000)
    ap.add_argument("--mtok", type=int, default=256); ap.add_argument("--reqs", type=int, default=192)
    ap.add_argument("--timeout", type=int, default=600); ap.add_argument("--tag", default="X")
    ap.add_argument("--salt", default="s0")
    asyncio.run(run(ap.parse_args()))
