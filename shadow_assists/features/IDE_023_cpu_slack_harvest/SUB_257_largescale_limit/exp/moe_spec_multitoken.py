"""다중토큰 accept: draft(top-1)가 K토큰 autoregressive 생성 → target(top-2) greedy와 일치 run-length.
실제 spec acceptance. run-length 길면 MoE-spec 실효 win."""
import torch, os, numpy as np
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="mistralai/Mixtral-8x7B-Instruct-v0.1"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF,dtype=torch.bfloat16,device_map=dev).eval()
blocks=[m for m in model.modules() if m.__class__.__name__=="MixtralSparseMoeBlock"]
DK=[2]
for b in blocks:
    g=b.gate; of=g.forward
    def mk(of):
        def f(h):
            out=of(h)
            if DK[0]>=2: return out
            rl,w,idx=out; w1=w[...,:1]; w1=w1/w1.sum(-1,keepdim=True); return rl,w1,idx[...,:1]
        return f
    g.forward=mk(of)
@torch.no_grad()
def gen(ids,n,dk):
    DK[0]=dk; out=[]
    o=model(ids,use_cache=True); p=o.past_key_values; nx=o.logits[:,-1:].argmax(-1)
    for _ in range(n):
        out.append(nx.item()); o=model(nx,past_key_values=p,use_cache=True); p=o.past_key_values; nx=o.logits[:,-1:].argmax(-1)
    return out
prompts=["The capital of France is","def fibonacci(n):","Machine learning is a field that","한국의 수도는","To compute the integral we","The weather forecast for tomorrow predicts","In the beginning the universe","Recursion works by"]
runs=[]
G=24  # 생성 길이
for pr in prompts:
    ids=tok(pr,return_tensors="pt").input_ids.to(dev)
    tgt=gen(ids,G,2)   # target greedy
    drf=gen(ids,G,1)   # draft(top-1) greedy
    # 일치 run: 첫 불일치까지 (단순 비교; 실제 spec은 재동기화하나 평균 run-length 근사)
    m=0
    for a,b in zip(drf,tgt):
        if a==b: m+=1
        else: break
    # 전체 일치율도
    tot=sum(1 for a,b in zip(drf,tgt) if a==b)/G
    runs.append((m,tot))
mr=np.mean([r[0] for r in runs]); mt=np.mean([r[1] for r in runs])
print(f"prompts={len(prompts)}, 생성{G}토큰")
print(f"첫불일치까지 평균 run-length={mr:.2f} 토큰")
print(f"전체 토큰 일치율={mt:.3f}")
# spec 경제: 한 사이클당 (accepted+1) 토큰을 (draft K개 + verify 1회)로. c=0.445
c=0.445; K=4
acc_per_cycle=min(mr,K)  # 근사
spd = (acc_per_cycle+1)/(c*K + 1)
print(f"근사 spec 속도(K={K}, c={c}): (run+1)/(c*K+1) = {spd:.2f}×")
print("판정: run-length>2 AND 속도>1.1이면 MoE-spec 실효 win.")
