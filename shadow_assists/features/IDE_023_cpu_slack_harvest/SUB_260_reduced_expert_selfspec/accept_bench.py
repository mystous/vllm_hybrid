"""B0 acceptance 측정: 동일 텍스트를 teacher-force(prompt_logprobs)해 각 위치 argmax 수집.
top-8 서버와 top-1(VLLM_MOE_FORCE_TOPK=1) 서버에서 각각 실행 → per-position argmax 일치율 = acceptance a.
두 서버 모두 같은 컨텍스트에 조건부라 순수 per-position 비교. 결과를 JSON으로 저장, 두 파일 비교.
"""
import argparse, json, requests

TEXTS=[
 "The development of large language models has fundamentally transformed natural language processing. These systems learn statistical patterns from vast text corpora and can generate coherent, contextually appropriate responses across a wide range of tasks including translation, summarization, and question answering.",
 "In distributed computing, the challenge of coordinating computation across many processors while minimizing communication overhead remains central. Techniques such as data parallelism, model parallelism, and pipeline parallelism each make different trade-offs between memory, computation, and inter-device communication bandwidth.",
 "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
 "The economic consequences of monetary policy decisions ripple through financial markets in complex ways. When central banks adjust interest rates, they influence borrowing costs, investment behavior, currency exchange rates, and ultimately the rate of inflation across the broader economy.",
]

def collect(port, model, n_lp):
    url=f"http://127.0.0.1:{port}/v1/completions"
    out=[]
    for t in TEXTS:
        body={"model":model,"prompt":t,"max_tokens":1,"temperature":0.0,
              "echo":True,"prompt_logprobs":n_lp}
        r=requests.post(url,json=body,timeout=300); r.raise_for_status()
        pl=r.json()["choices"][0].get("prompt_logprobs")
        argmax=[]
        for pos in pl:
            if pos is None: argmax.append(None); continue
            # pos: {token_id_str: {logprob, rank, decoded_token}}
            best=min(pos.items(), key=lambda kv: kv[1]["rank"])  # rank 1 = argmax
            argmax.append(int(best[0]))
        out.append(argmax)
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--port",type=int,required=True)
    ap.add_argument("--model",required=True)
    ap.add_argument("--n-logprobs",type=int,default=20)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    res=collect(a.port,a.model,a.n_logprobs)
    with open(a.out,"w") as f: json.dump(res,f)
    print(f"saved {a.out}: {sum(len(x) for x in res)} positions")
