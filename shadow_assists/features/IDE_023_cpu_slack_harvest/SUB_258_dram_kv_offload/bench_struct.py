"""구조화출력 CPU-critical-path 측정: 고동시성 JSON-schema 생성 vs 무제약(plain).
구조화출력은 매 스텝 grammar FSM advance + vocab-크기 bitmask fill = 순수 CPU work.
GPU가 굶으면(util↓, structured tps ≪ plain tps) CPU 병목 = 헤드룸 존재.
"""
import argparse, asyncio, time, json, aiohttp

SCHEMA={"type":"object","properties":{
  "name":{"type":"string"},"age":{"type":"integer"},
  "email":{"type":"string"},"city":{"type":"string"},
  "tags":{"type":"array","items":{"type":"string"}},
  "active":{"type":"boolean"},"score":{"type":"number"}},
  "required":["name","age","email","city","tags","active","score"]}

async def one(sess,url,model,structured,mt,idx):
    body={"model":model,"max_tokens":120,"temperature":0.0,
          "messages":[{"role":"user","content":f"Generate a fictional user profile #{idx} as JSON."}]}
    if structured:
        body["response_format"]={"type":"json_schema",
            "json_schema":{"name":"profile","schema":SCHEMA}}
    t0=time.perf_counter(); ntok=0; ttft=None
    body["stream"]=True
    async with sess.post(url,json=body) as r:
        async for line in r.content:
            s=line.decode().strip()
            if not s.startswith("data:"): continue
            s=s[5:].strip()
            if s=="[DONE]": break
            try: d=json.loads(s)
            except: continue
            delta=d.get("choices",[{}])[0].get("delta",{}).get("content","")
            if delta:
                if ttft is None: ttft=time.perf_counter()-t0
                ntok+=1
    return {"ntok":ntok,"ttft":ttft or (time.perf_counter()-t0)}

async def run(args):
    url=f"http://127.0.0.1:{args.port}/v1/chat/completions"
    t0=time.perf_counter()
    conn=aiohttp.TCPConnector(limit=args.concurrency)
    res=[]
    async with aiohttp.ClientSession(connector=conn,timeout=aiohttp.ClientTimeout(total=600)) as sess:
        sem=asyncio.Semaphore(args.concurrency)
        async def g(i):
            async with sem: return await one(sess,url,args.model,args.structured,args.mt,i)
        tasks=[asyncio.create_task(g(i)) for i in range(args.n)]
        for f in asyncio.as_completed(tasks): res.append(await f)
    wall=time.perf_counter()-t0
    tot=sum(r["ntok"] for r in res)
    print(json.dumps({"label":args.label,"structured":args.structured,"n":args.n,
        "wall_s":round(wall,2),"out_tok":tot,"out_tps":round(tot/wall,1),
        "req_per_s":round(args.n/wall,2)}))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--port",type=int,default=8000)
    ap.add_argument("--model",required=True)
    ap.add_argument("--n",type=int,default=200)
    ap.add_argument("--concurrency",type=int,default=200)
    ap.add_argument("--structured",action="store_true")
    ap.add_argument("--mt",default="")
    ap.add_argument("--label",default="run")
    asyncio.run(run(ap.parse_args()))
