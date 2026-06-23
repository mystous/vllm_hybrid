#!/usr/bin/env python3
"""고정 프롬프트로 greedy 생성 + per-token logprob 수집 (정확도 게이트용). JSON 저장."""
import argparse, asyncio, json, aiohttp
PROMPTS=[
 "Explain how a CPU cache hierarchy works, step by step.",
 "Write a Python function to compute the nth Fibonacci number.",
 "Summarize the theory of relativity in three sentences.",
 "What are the trade-offs between TCP and UDP?",
 "Describe the process of photosynthesis.",
 "List five best practices for writing maintainable code.",
 "How does a hash table achieve average O(1) lookup?",
 "Explain the difference between supervised and unsupervised learning.",
]
PROMPTS = PROMPTS + [
 "Describe how garbage collection works in modern runtimes.",
 "What is the CAP theorem and why does it matter?",
 "Explain backpropagation in neural networks.",
 "Write a SQL query to find the second highest salary.",
 "How does HTTPS establish a secure connection?",
 "Summarize the causes of the 2008 financial crisis.",
 "Explain the difference between processes and threads.",
 "What are the principles of RESTful API design?",
 "Describe how a B-tree index speeds up database queries.",
 "Explain the concept of eventual consistency.",
 "What is dynamic programming and when is it used?",
 "How does a transformer attention mechanism work?",
 "Explain virtual memory and paging.",
 "What are the trade-offs of microservices vs monolith?",
 "Describe the OSI network model layers.",
 "Explain how public-key cryptography works."]

async def gen(sess, base, model, prompt, mtok):
    payload={"model":model,"prompt":prompt,"max_tokens":mtok,"temperature":0.0,"logprobs":1}
    async with sess.post(f"{base}/v1/completions",json=payload) as r:
        d=await r.json()
    ch=d["choices"][0]; lp=ch.get("logprobs",{}) or {}
    toks=lp.get("tokens",[]); tlp=lp.get("token_logprobs",[])
    return {"text":ch["text"],"tokens":toks,"logprobs":[x for x in tlp if x is not None]}
async def main(a):
    out=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as s:
        for p in PROMPTS:
            out.append(await gen(s,a.base,a.model,p,a.mtok))
    json.dump(out,open(a.out,"w"))
    print(f"saved {len(out)} prompts -> {a.out}")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True);ap.add_argument("--model",required=True)
    ap.add_argument("--mtok",type=int,default=64);ap.add_argument("--out",required=True)
    asyncio.run(main(ap.parse_args()))
