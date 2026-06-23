# cascade-aware 시제: (prompt×nbits) 풀로 instruction-break 예측. logprob-gate vs cascade features.
import torch, glob, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kv_conservativeness import SPEC
MODEL=glob.glob("/raid/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*")[0]
dev="cuda:0"; tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map=dev).eval()
def quant(t,nb):
    qm=2**(nb-1)-1; s=t.abs().amax(-1,keepdim=True).clamp_min(1e-8)/qm
    return (torch.round(t/s).clamp(-qm-1,qm)*s).to(t.dtype)
class QC(DynamicCache):
    nb=0
    def update(self,k,v,i,*a,**kw): return super().update(quant(k,QC.nb),quant(v,QC.nb),i,*a,**kw)
@torch.no_grad()
def gen(ids,nb):
    QC.nb=nb; return model.generate(ids,max_new_tokens=100,do_sample=False,past_key_values=QC() if nb>0 else DynamicCache())[0][ids.shape[1]:]
@torch.no_grad()
def tfl(ids,seq,nb):
    QC.nb=nb; o=model(torch.cat([ids,seq.unsqueeze(0)],1),past_key_values=QC() if nb>0 else DynamicCache())
    return o.logits[0,ids.shape[1]-1:-1].float()
rows=[]
for i,(p,ck) in enumerate(SPEC):
    enc=tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,return_tensors="pt",return_dict=True)
    ids=enc["input_ids"].to(dev); seq=gen(ids,0)
    if not ck(tok.decode(seq,skip_special_tokens=True)): continue
    L=seq.shape[0]; lg0=tfl(ids,seq,0); lp0=torch.log_softmax(lg0,-1)[torch.arange(L),seq]
    for nb in [8,6,5,4]:
        lgc=tfl(ids,seq,nb); lpc=torch.log_softmax(lgc,-1)[torch.arange(L),seq]
        mad=(lp0-lpc).abs().max().item()
        flips=(lgc.argmax(-1)!=seq); nf=int(flips.sum())
        ff=int(flips.float().argmax()) if nf>0 else L
        cfrac=(L-ff)/L                       # 첫 flip 이후 비율(cascade 폭)
        broke = not ck(tok.decode(gen(ids,nb),skip_special_tokens=True))
        rows.append((mad, nf, cfrac, broke))
import numpy as np
A=np.array([(r[0],r[1],r[2],r[3]) for r in rows],float)
mad,nf,cf,br=A[:,0],A[:,1],A[:,2],A[:,3].astype(bool)
N=len(A); P=int(br.sum())
print(f"# 풀 {N} (prompt×nbits), 깨짐 {P}/{N}")
# 예측기별 최고 정확도(임계 sweep)
def best(score, name, higher_breaks=True):
    bestacc=0;bt=None
    for t in sorted(set(score)):
        pred=(score>t) if higher_breaks else (score<t)
        acc=(pred==br).mean()
        if acc>bestacc:bestacc=acc;bt=t
    print(f"{name:28s}: best acc {bestacc:.2f} (thr={bt:.3f})")
    return bestacc
best(mad,"logprob max-abs-diff")           # 게이트(높을수록 깨짐)
best(cf,"cascade_frac(첫flip이후비율)")
best(nf,"n_flips")
best(mad*cf,"madxcascade(곱)")
# logprob-gate 고정임계 0.5의 실제 정확도
print(f"logprob-gate(thr=0.5 고정) acc: {((mad>0.5)==br).mean():.2f}")
