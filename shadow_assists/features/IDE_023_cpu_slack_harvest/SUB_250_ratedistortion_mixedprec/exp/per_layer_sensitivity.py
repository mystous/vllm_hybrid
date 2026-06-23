"""SUB_250 — water-filling mixed-precision 이론의 경험적 토대:
레이어별 FP4 양자화 민감도 곡선 측정. 한 레이어만 FP4(나머지 BF16) → 출력 분포 왜곡(게이트지표).
민감도가 비균일하면 water-filling 비트배분 헤드룸 존재 = 새 이론 가능."""
import torch, math, statistics as st, os, copy
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="meta-llama/Llama-3.1-8B-Instruct"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, device_map=dev).eval()
E2M1=torch.tensor([0,0.5,1,1.5,2,3,4,6],device=dev)
def qfp4(x,group=16):
    o=x.shape; xf=x.float().reshape(-1)
    n=xf.numel(); pad=(group-n%group)%group
    if pad: xf=torch.nn.functional.pad(xf,(0,pad))
    g=xf.reshape(-1,group); s=g.abs().amax(-1,keepdim=True).clamp_min(1e-8)/6.0
    q=(g/s).abs().unsqueeze(-1); idx=(q-E2M1).abs().argmin(-1)
    dq=((g/s).sign()*E2M1[idx]*s).reshape(-1)[:n]
    return dq.reshape(o).to(x.dtype)
prompts=["Explain how a CPU cache works.","Write a Python function for quicksort.",
         "Summarize relativity in 3 sentences.","What are TCP vs UDP tradeoffs?",
         "Describe photosynthesis.","How does a hash table get O(1)?"]
def logprobs(m):
    out=[]
    for p in prompts:
        ids=tok(p,return_tensors="pt").input_ids.to(dev)
        with torch.no_grad(): g=m.generate(ids,max_new_tokens=48,do_sample=False,return_dict_in_generate=True,output_scores=True)
        sc=torch.stack(g.scores,1)[0]  # [newtok, vocab]
        lp=torch.log_softmax(sc.float(),-1)
        toks=g.sequences[0,ids.shape[1]:]
        out.append((toks.tolist(),[lp[i,toks[i]].item() for i in range(len(toks))]))
    return out
print("baseline logprobs..."); base=logprobs(model)
# 레이어별: down_proj+gate_proj+up_proj+o_proj+qkv 를 FP4로 (그 레이어만), 측정 후 복원
layers=model.model.layers; nl=len(layers)
def gate(b2):
    md=0; rels=[]
    for (t1,l1),(t2,l2) in zip(base,b2):
        n=min(len(t1),len(t2))
        for i in range(n):
            if t1[i]==t2[i]: md=max(md,abs(l1[i]-l2[i]))
            else: break
        import math
        p1=math.exp(-sum(l1)/len(l1)); p2=math.exp(-sum(l2)/len(l2))
        rels.append(abs(p2-p1)/p1)
    return md, st.mean(rels)
import collections; sens={}
probe=[0,1,2,4,8,12,16,20,24,28,30,31]
for li in probe:
    lyr=layers[li]; saved={}
    for name,mod in lyr.named_modules():
        if isinstance(mod,torch.nn.Linear):
            saved[name]=mod.weight.data.clone(); mod.weight.data=qfp4(mod.weight.data)
    md,rel=gate(logprobs(model))
    for name,mod in lyr.named_modules():
        if name in saved: mod.weight.data=saved[name]
    sens[li]=(md,rel); print(f"  L{li:2d}: max_logprob_diff={md:.4f} ppl_rel={rel:.4f}")
mds=[v[0] for v in sens.values()]
print(f"\n=== 레이어별 FP4 민감도 (max_logprob_diff) ===")
print(f"  min={min(mds):.4f} max={max(mds):.4f} 비율={max(mds)/max(min(mds),1e-4):.1f}x")
print(f"  → 비율 크면(>3x) water-filling 헤드룸 존재: 둔감 레이어 저비트/민감 레이어 고비트 배분 가능.")
