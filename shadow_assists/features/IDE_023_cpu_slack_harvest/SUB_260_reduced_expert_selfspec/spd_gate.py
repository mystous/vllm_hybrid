"""SPD calibration: teacher-force 고정 텍스트의 per-position top-k logprob 수집(저장),
또는 두 수집본 비교(baseline vs SPD-drop)로 게이트 지표 계산.
- collect: --out 에 [{token_id: logprob} per position] 저장.
- compare: --base A.json --cmp B.json → max_logprob_diff(=baseline argmax 토큰의 logprob diff 최대),
  ppl_rel(시퀀스 PPL 상대차), argmax_match(정보용). 게이트 PASS = max_logprob_diff≤0.5 AND ppl_rel≤0.1.
"""
import argparse, json, math, requests

TEXTS=[
 "The development of large language models has fundamentally transformed natural language processing across translation, summarization, and reasoning tasks.",
 "In distributed computing, coordinating computation across many processors while minimizing communication overhead is the central performance challenge for large-scale systems.",
 "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    return quicksort([x for x in arr if x<pivot]) + [pivot] + quicksort([x for x in arr if x>pivot])",
 "The consequences of monetary policy ripple through financial markets, influencing borrowing costs, investment behavior, and the rate of inflation across the economy.",
]

def collect(port, model, nlp):
    url=f"http://127.0.0.1:{port}/v1/completions"; out=[]
    for t in TEXTS:
        body={"model":model,"prompt":t,"max_tokens":1,"temperature":0.0,"echo":True,"prompt_logprobs":nlp}
        r=requests.post(url,json=body,timeout=300); r.raise_for_status()
        pl=r.json()["choices"][0].get("prompt_logprobs")
        seq=[]
        for pos in pl:
            if pos is None: seq.append(None); continue
            seq.append({tid:info["logprob"] for tid,info in pos.items()})
        out.append(seq)
    return out

def argmax_tok(d):  # rank1 token = max logprob
    return max(d.items(), key=lambda kv: kv[1])[0]

def compare(A, B):
    maxdiff=0.0; lp_b=[]; lp_s=[]; tot=0; match=0
    for sa,sb in zip(A,B):
        for da,db in zip(sa,sb):
            if da is None or db is None: continue
            ta=argmax_tok(da)  # baseline argmax token
            tot+=1; match += (ta==argmax_tok(db))
            la=da[ta]
            lb=db.get(ta)  # SPD logprob for baseline argmax token (top-k에 있으면)
            if lb is not None:
                maxdiff=max(maxdiff, abs(la-lb)); lp_b.append(la); lp_s.append(lb)
    ppl_b=math.exp(-sum(lp_b)/len(lp_b)); ppl_s=math.exp(-sum(lp_s)/len(lp_s))
    ppl_rel=abs(ppl_s-ppl_b)/ppl_b
    return {"positions":tot,"argmax_match":round(match/tot,4),
            "max_logprob_diff":round(maxdiff,4),"ppl_base":round(ppl_b,3),
            "ppl_spd":round(ppl_s,3),"ppl_rel":round(ppl_rel,4),
            "GATE":"PASS" if (maxdiff<=0.5 and ppl_rel<=0.1) else "FAIL"}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["collect","compare"],required=True)
    ap.add_argument("--port",type=int); ap.add_argument("--model")
    ap.add_argument("--n-logprobs",type=int,default=20); ap.add_argument("--out")
    ap.add_argument("--base"); ap.add_argument("--cmp")
    a=ap.parse_args()
    if a.mode=="collect":
        json.dump(collect(a.port,a.model,a.n_logprobs), open(a.out,"w"))
        print(f"saved {a.out}")
    else:
        A=json.load(open(a.base)); B=json.load(open(a.cmp))
        print(json.dumps(compare(A,B)))
