#!/usr/bin/env python
# 분포-bounded KV압축 kill-gate: 프로젝트 게이트(per-token logprob max-abs-diff)가
# KV압축의 "조용한 instruction 누락"(Pitfalls 2510.00231)을 잡는가?
# KV를 int(nbits) 대칭양자화(custom cache)하고 baseline 출력을 teacher-force 재채점.
import torch, sys, json
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
MODEL="/raid/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots"
import glob, os
MODEL=glob.glob(MODEL+"/*")[0]
NBITS=int(sys.argv[1]) if len(sys.argv)>1 else 0   # 0=full, else 양자화 비트수
dev="cuda:0"
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map=dev)
model.eval()

def quant(t,nbits):
    if nbits<=0: return t
    qmax=2**(nbits-1)-1
    s=t.abs().amax(dim=-1,keepdim=True).clamp_min(1e-8)/qmax
    return (torch.round(t/s).clamp(-qmax-1,qmax)*s).to(t.dtype)

class QCache(DynamicCache):
    nbits=0
    def update(self,k,v,idx,*a,**kw):
        k=quant(k,QCache.nbits); v=quant(v,QCache.nbits)
        return super().update(k,v,idx,*a,**kw)

# multi-instruction 프롬프트 (IFEval류: 형식·키워드·길이 제약)
PROMPTS=[
 "Write a short note about coffee. Constraints: use EXACTLY two bullet points starting with '-', include the word 'aroma', and end your entire response with the token DONE.",
 "Describe the ocean in 3 sentences. You MUST include the word 'tide' and you MUST NOT use the letter e in your last sentence. Finish with the word END.",
 "List two benefits of sleep. Each must be one sentence. Wrap the whole answer in <ans>...</ans> tags and include the number 8.",
 "Explain photosynthesis to a child. Use no more than 40 words, include the word 'sunlight', and respond entirely in lowercase.",
]
def gen(prompt,nbits):
    msgs=[{"role":"user","content":prompt}]
    enc=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt",return_dict=True)
    ids=enc["input_ids"].to(dev)
    QCache.nbits=nbits
    with torch.no_grad():
        out=model.generate(ids,max_new_tokens=120,do_sample=False,
            past_key_values=QCache() if nbits>0 else DynamicCache(),
            return_dict_in_generate=True,output_scores=True)
    seq=out.sequences[0][ids.shape[1]:]
    return ids, seq, tok.decode(seq,skip_special_tokens=True)

def score(ids, comp_ids, nbits):
    # baseline 출력(comp_ids)을 nbits KV로 teacher-force, per-token logprob
    full=torch.cat([ids, comp_ids.unsqueeze(0)],dim=1)
    QCache.nbits=nbits
    with torch.no_grad():
        o=model(full,past_key_values=QCache() if nbits>0 else DynamicCache())
    logits=o.logits[0, ids.shape[1]-1:-1]   # comp_ids 위치 예측
    lp=torch.log_softmax(logits.float(),dim=-1)
    return lp[torch.arange(comp_ids.shape[0]),comp_ids]   # per-token logprob

def check_instr(text,i):
    t=text.lower()
    if i==0: return ('done' in t) and (text.count('-')>=2) and ('aroma' in t)
    if i==1: return ('tide' in t) and ('end' in t)
    if i==2: return ('<ans>' in t and '</ans>' in t) and ('8' in t)
    if i==3: return ('sunlight' in t) and (text==text.lower()) and (len(text.split())<=45)
    return False

print(f"# Llama-3.1-8B  KV quant nbits={NBITS}  vs full")
for i,p in enumerate(PROMPTS):
    ids,seq,txt=gen(p,0)                     # baseline full-KV 생성
    lp_full=score(ids,seq,0)
    ok_full=check_instr(txt,i)
    # 압축 KV로: (a) baseline출력 재채점, (b) 자유생성 instruction 체크
    lp_c=score(ids,seq,NBITS)
    _,_,txt_c=gen(p,NBITS)
    ok_c=check_instr(txt_c,i)
    madiff=(lp_full-lp_c).abs().max().item()
    ppl_full=torch.exp(-lp_full.mean()).item(); ppl_c=torch.exp(-lp_c.mean()).item()
    pplrel=abs(ppl_c-ppl_full)/ppl_full
    gate_pass = (madiff<=0.5) and (pplrel<=0.1)
    print(f"P{i}: instr full={ok_full} comp={ok_c} | max-abs-diff={madiff:.3f} ppl_rel={pplrel:.3f} | GATE={'PASS' if gate_pass else 'FAIL'} | instr_broke={ok_full and not ok_c}")
