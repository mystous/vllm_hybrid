"""KV-constrained 워크로드 벤치: 많은 distinct 긴 prefix 를 반복 재사용.
working-set(prefix KV) > GPU KV cache 일 때:
  - GPU-only(offload off): 재사용 시점에 prefix 블록이 evict 됨 → prefill 재계산(TTFT↑).
  - GPU+DRAM(offload on): evict 블록이 DRAM 으로 내려가 재사용 때 fetch → 재계산 회피(TTFT↓).
측정: 전체 throughput, mean/p50 TTFT, 총 wall. /metrics 의 prefix cache hit 도 수집.
"""
import argparse, asyncio, time, json, sys
import aiohttp

def make_prefixes(n, approx_tokens):
    # 토큰 동일성이 prefix-cache hit 의 전제 → 고정 시드 문자열 n개 생성.
    # ~4글자/토큰 가정. 각 prefix 는 고유 헤더 + 반복 문장(서로 다른 주제어)로 distinct.
    topics = ["astronomy","genetics","economics","networking","thermodynamics",
              "linguistics","cryptography","ecology","metallurgy","cartography",
              "immunology","seismology","robotics","archaeology","oceanography",
              "neuroscience","aerodynamics","virology","topology","hydrology",
              "petrology","entomology","mycology","cosmology"]
    sent = (" The study of {t} requires careful observation of repeated patterns, "
            "rigorous measurement of underlying quantities, and systematic comparison "
            "against established theoretical models derived from first principles.")
    out=[]
    for i in range(n):
        t=topics[i%len(topics)]
        head=f"Document #{i} — a comprehensive technical reference on {t}."
        body=(sent.format(t=t))*( (approx_tokens*4)//len(sent.format(t=t)) + 1 )
        out.append(head+body)
    return out

async def one(session, url, model, prompt, max_tokens, idx):
    payload={"model":model,"prompt":prompt,"max_tokens":max_tokens,
             "temperature":0.0,"stream":True}
    t0=time.perf_counter(); ttft=None; ntok=0
    async with session.post(url, json=payload) as r:
        async for line in r.content:
            if not line: continue
            s=line.decode().strip()
            if not s.startswith("data:"): continue
            s=s[5:].strip()
            if s=="[DONE]": break
            try: d=json.loads(s)
            except: continue
            txt=d.get("choices",[{}])[0].get("text","")
            if txt:
                if ttft is None: ttft=time.perf_counter()-t0
                ntok+=1
    dur=time.perf_counter()-t0
    if ttft is None: ttft=dur
    return {"idx":idx,"ttft":ttft,"dur":dur,"ntok":ntok}

async def run(args):
    base=make_prefixes(args.n_prefix, args.prefix_tokens)
    url=f"http://127.0.0.1:{args.port}/v1/completions"
    # 라운드별로 모든 prefix 재사용. 각 요청 끝에 고유 suffix(재사용이지만 새 디코드).
    reqs=[]
    for rnd in range(args.rounds):
        order=list(range(args.n_prefix))
        # 결정적 셔플(라운드마다 회전) — LRU evict 유도
        order=order[rnd%args.n_prefix:]+order[:rnd%args.n_prefix]
        for pi in order:
            prompt=base[pi]+f"\n\nQuestion {rnd}-{pi}: summarize the key idea in one sentence.\nAnswer:"
            reqs.append((pi,prompt))
    t0=time.perf_counter()
    conn=aiohttp.TCPConnector(limit=args.concurrency)
    results=[]
    async with aiohttp.ClientSession(connector=conn,
            timeout=aiohttp.ClientTimeout(total=args.timeout)) as sess:
        sem=asyncio.Semaphore(args.concurrency)
        async def guarded(i,p):
            async with sem:
                return await one(sess,url,args.model,p,args.max_tokens,i)
        tasks=[asyncio.create_task(guarded(i,p)) for i,(pi,p) in enumerate(reqs)]
        for fut in asyncio.as_completed(tasks):
            results.append(await fut)
    wall=time.perf_counter()-t0
    ttfts=sorted(r["ttft"] for r in results)
    tot_out=sum(r["ntok"] for r in results)
    p50=ttfts[len(ttfts)//2]; p90=ttfts[int(len(ttfts)*0.9)]
    mean=sum(ttfts)/len(ttfts)
    out={"label":args.label,"n_req":len(results),"wall_s":round(wall,2),
         "out_tok":tot_out,"out_tps":round(tot_out/wall,1),
         "req_per_s":round(len(results)/wall,2),
         "ttft_mean":round(mean,3),"ttft_p50":round(p50,3),"ttft_p90":round(p90,3)}
    print(json.dumps(out))
    if args.out:
        with open(args.out,"a") as f: f.write(json.dumps(out)+"\n")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--port",type=int,default=8000)
    ap.add_argument("--model",required=True)
    ap.add_argument("--n-prefix",type=int,default=24)
    ap.add_argument("--prefix-tokens",type=int,default=3000)
    ap.add_argument("--rounds",type=int,default=4)
    ap.add_argument("--max-tokens",type=int,default=32)
    ap.add_argument("--concurrency",type=int,default=8)
    ap.add_argument("--timeout",type=int,default=600)
    ap.add_argument("--label",default="run")
    ap.add_argument("--out",default="")
    args=ap.parse_args()
    asyncio.run(run(args))
