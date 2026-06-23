"""목표-정렬 변종: 2bit 미사용. uniform-4 + 민감 top-k 레이어만 8bit (avg 약간↑) → joint KL 변화.
민감도 순위는 D_i(4) 큰 순. 게이트 도움 vs 속도(avg비트) trade 확인."""
import torch, os
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="meta-llama/Llama-3.1-8B-Instruct"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF,dtype=torch.bfloat16,device_map=dev).eval()
def quant(x,b,group=16):
    if b>=16: return x
    o=x.shape; xf=x.float().reshape(-1); n=xf.numel(); pad=(group-n%group)%group
    if pad: xf=torch.nn.functional.pad(xf,(0,pad))
    g=xf.reshape(-1,group); qmax=2**(b-1)-1; s=g.abs().amax(-1,keepdim=True).clamp_min(1e-8)/qmax
    return (torch.round(g/s).clamp(-qmax,qmax)*s).reshape(-1)[:n].reshape(o).to(x.dtype)
text=("The accelerator processes descriptors while the memory controller arbitrates bandwidth. ")*70
ids=tok(text,return_tensors="pt").input_ids[:,:1200].to(dev)
import torch as T
base=T.log_softmax(model(ids).logits[0].float(),-1)
def kl(lp): return (base.exp()*(base-lp)).sum(-1).mean().item()
layers=model.model.layers; nl=len(layers)
def joint(bitmap):
    saved=[]
    for li in range(nl):
        for nm,m in layers[li].named_modules():
            if isinstance(m,T.nn.Linear): saved.append((m,m.weight.data.clone())); m.weight.data=quant(m.weight.data,bitmap[li])
    d=kl(T.log_softmax(model(ids).logits[0].float(),-1))
    for m,w in saved: m.weight.data=w
    return d
# 민감도 = 개별 4bit D_i (재측정 간단판: 4bit 한 레이어)
sens={}
for li in range(nl):
    s=[]
    for nm,m in layers[li].named_modules():
        if isinstance(m,T.nn.Linear): s.append((m,m.weight.data.clone())); m.weight.data=quant(m.weight.data,4)
    sens[li]=kl(T.log_softmax(model(ids).logits[0].float(),-1))
    for m,w in s: m.weight.data=w
order=sorted(range(nl),key=lambda li:sens[li],reverse=True)  # 민감 순
d4=joint({li:4 for li in range(nl)})
print(f"uniform-4bit joint KL={d4:.5f} (avg 4.00bit)")
for k in [4,8,12]:
    bm={li:4 for li in range(nl)}
    for li in order[:k]: bm[li]=8
    avg=sum(bm.values())/nl; d=joint(bm)
    print(f"  +top{k} 민감→8bit (avg {avg:.2f}bit): KL={d:.5f} ({(d/d4-1)*100:+.1f}% vs uniform4)")
print("→ KL 크게↓면 게이트 여유↑(W4A4+spec PASS 가능), 단 avg비트↑=속도손해. trade 평가.")
