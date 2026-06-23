"""70B FP4 레이어-블록 민감도 probe — bump 대상 레이어 선정.
80레이어를 10레이어 블록 8개로 나눠 각 블록만 FP4(group16) → 나머지 bf16 → 출력분포 KL 측정.
가장 민감한 블록을 bump(고정밀 유지) 후보로 랭크. 결과 runs/probe70b.json."""
import torch, os, json, time
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="meta-llama/Llama-3.1-70B-Instruct"
tok=AutoTokenizer.from_pretrained(HF)
t0=time.time()
print(f"[{time.time()-t0:.0f}s] loading 70B bf16 (device_map=auto)...", flush=True)
model=AutoModelForCausalLM.from_pretrained(HF,dtype=torch.bfloat16,device_map="auto").eval()
print(f"[{time.time()-t0:.0f}s] loaded.", flush=True)
def quant(x,b=4,group=16):
    o=x.shape; xf=x.float().reshape(-1); n=xf.numel(); pad=(group-n%group)%group
    if pad: xf=torch.nn.functional.pad(xf,(0,pad))
    g=xf.reshape(-1,group); qmax=2**(b-1)-1; s=g.abs().amax(-1,keepdim=True).clamp_min(1e-8)/qmax
    return (torch.round(g/s).clamp(-qmax,qmax)*s).reshape(-1)[:n].reshape(o).to(x.dtype)
text=("The accelerator processes descriptors while the memory controller arbitrates bandwidth. ")*120
ids=tok(text,return_tensors="pt").input_ids[:,:1200]
dev0=next(model.parameters()).device
ids=ids.to(dev0)
with torch.no_grad():
    base=torch.log_softmax(model(ids).logits[0].float(),-1)
def kl(lp): return (base.exp()*(base-lp)).sum(-1).mean().item()
layers=model.model.layers; nl=len(layers)
BLK=10; blocks=[list(range(i,min(i+BLK,nl))) for i in range(0,nl,BLK)]
res={}
for bi,blk in enumerate(blocks):
    saved=[]
    for li in blk:
        for nm,m in layers[li].named_modules():
            if isinstance(m,torch.nn.Linear): saved.append((m,m.weight.data.clone())); m.weight.data=quant(m.weight.data)
    with torch.no_grad():
        d=kl(torch.log_softmax(model(ids).logits[0].float(),-1))
    for m,w in saved: m.weight.data=w
    res[f"L{blk[0]}-{blk[-1]}"]=d
    print(f"[{time.time()-t0:.0f}s] block L{blk[0]}-{blk[-1]} FP4 KL={d:.5f}", flush=True)
# all-FP4 reference
saved=[]
for li in range(nl):
    for nm,m in layers[li].named_modules():
        if isinstance(m,torch.nn.Linear): saved.append((m,m.weight.data.clone())); m.weight.data=quant(m.weight.data)
with torch.no_grad():
    d_all=kl(torch.log_softmax(model(ids).logits[0].float(),-1))
for m,w in saved: m.weight.data=w
res["ALL_FP4"]=d_all
print(f"[{time.time()-t0:.0f}s] ALL_FP4 KL={d_all:.5f}", flush=True)
order=sorted([k for k in res if k!="ALL_FP4"],key=lambda k:res[k],reverse=True)
print("민감 블록 순위:", [(k,round(res[k],5)) for k in order], flush=True)
json.dump({"res":res,"order":order},open("runs/probe70b.json","w"),indent=2)
print("saved runs/probe70b.json", flush=True)
