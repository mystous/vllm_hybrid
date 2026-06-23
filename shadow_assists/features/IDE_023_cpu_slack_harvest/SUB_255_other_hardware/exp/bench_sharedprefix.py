#!/usr/bin/env python3
"""공유-프리픽스 워크로드 벤치 (prefix KV 재사용 이득 실측).
각 요청 = [고정 공유 prefix(동일)] + [유니크 짧은 suffix]. prefix-cache ON이면 공유 prefix KV
재사용→prefill skip. OFF면 매 요청 prefix 재연산. wall/throughput 차이로 KV 재사용 win 실측.
NEO(활성KV swap, 용량)와 차별: 정적 prefix 재사용(연산절감).
"""
import argparse, asyncio, time, random, string

def shared_prefix(ptok):
    # 고정 시드 → 모든 요청·런 동일 토큰열 (APC 해시 일치). 현실의 긴 시스템프롬프트/RAG 문서 모사.
    random.seed(12345)
    words=["system","context","document","policy","analysis","procedure","framework","protocol",
           "the","accelerator","memory","bandwidth","scheduler","throughput","latency","pipeline",
           "request","response","token","sequence","attention","inference","optimization","workload"]
    n=max(1,(ptok)//1)
    txt=" ".join(random.choice(words) for _ in range(n))
    return "[SHARED DOC]\n"+txt+"\n[END DOC]\nBased strictly on the document above, answer concisely. "

async def one(sess, base, model, prompt, mtok, out):
    t0=time.perf_counter()
    payload={"model":model,"prompt":prompt,"max_tokens":mtok,"temperature":0.0,"stream":False}
    try:
        async with sess.post(f"{base}/v1/completions",json=payload) as r:
            d=await r.json()
        comp=d["choices"][0]["text"]; usage=d.get("usage",{})
        out.append((usage.get("completion_tokens",len(comp.split())), usage.get("prompt_tokens",0), time.perf_counter()-t0))
    except Exception:
        out.append((0,0,-1.0))

async def run(args):
    import aiohttp
    PRE=shared_prefix(args.ptok)
    sem=asyncio.Semaphore(args.conc); out=[]
    conn=aiohttp.TCPConnector(limit=0); timeout=aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(connector=conn,timeout=timeout) as sess:
        async def task(i):
            async with sem:
                # 유니크 짧은 suffix (요청마다 다름 → 공유부분만 캐시 적중)
                suf=f"Question {args.salt}-{i}: summarize point {i%50} in one sentence."
                await one(sess,args.base,args.model,PRE+suf,args.mtok,out)
        t0=time.perf_counter()
        await asyncio.gather(*[task(i) for i in range(args.reqs)])
        wall=time.perf_counter()-t0
    ok=[o for o in out if o[2]>0]
    gen=sum(o[0] for o in ok); pt=sum(o[1] for o in ok)
    reqps=len(ok)/wall if wall>0 else 0
    print(f"BENCH,tag={args.tag},ok={len(ok)}/{args.reqs},wall_s={wall:.2f},"
          f"gen_tok={gen},gen_tps={gen/wall:.1f},prompt_tok={pt},req_per_s={reqps:.2f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True); ap.add_argument("--model",required=True)
    ap.add_argument("--conc",type=int,default=24); ap.add_argument("--ptok",type=int,default=3500)
    ap.add_argument("--mtok",type=int,default=32); ap.add_argument("--reqs",type=int,default=192)
    ap.add_argument("--timeout",type=int,default=600); ap.add_argument("--tag",default="X")
    ap.add_argument("--salt",default="s0")
    asyncio.run(run(ap.parse_args()))
