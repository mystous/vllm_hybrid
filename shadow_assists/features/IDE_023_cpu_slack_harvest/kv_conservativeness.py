#!/usr/bin/env python
# 과보수성 정량화: KV quant nbits sweep로 게이트 통과율 vs 실제 instruction 보존율 곡선.
# 헤드룸 = (instruction 보존하는데 게이트가 FAIL시키는 영역) = tighter bound가 노릴 공간.
import torch, glob, json
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
MODEL=glob.glob("/raid/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*")[0]
dev="cuda:0"
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map=dev).eval()

def quant(t,nb):
    if nb<=0: return t
    qm=2**(nb-1)-1; s=t.abs().amax(-1,keepdim=True).clamp_min(1e-8)/qm
    return (torch.round(t/s).clamp(-qm-1,qm)*s).to(t.dtype)
class QCache(DynamicCache):
    nb=0
    def update(self,k,v,i,*a,**kw): return super().update(quant(k,QCache.nb),quant(v,QCache.nb),i,*a,**kw)

# 16 multi-instruction 프롬프트: (prompt, checker)
SPEC=[
 ("Write about cats. Include the word 'whiskers' and end with EXACTLY: STOP", lambda t:'whiskers' in t.lower() and t.rstrip().endswith('STOP')),
 ("Name two fruits as a numbered list 1. 2. and include the word 'fresh'.", lambda t:'fresh' in t.lower() and '1.' in t and '2.' in t),
 ("Describe rain in lowercase only, max 30 words, include 'cloud'.", lambda t:'cloud' in t.lower() and t==t.lower() and len(t.split())<=35),
 ("Explain gravity. Use the word 'mass' and wrap answer in [[ ]] brackets.", lambda t:'mass' in t.lower() and '[[' in t and ']]' in t),
 ("List 3 colors with bullets '- '. Include the word 'bright'. End with DONE.", lambda t:'bright' in t.lower() and t.count('- ')>=3 and 'DONE' in t),
 ("Write one sentence about the moon containing the word 'crater' and ending in a question mark.", lambda t:'crater' in t.lower() and t.rstrip().endswith('?')),
 ("Give two tips for studying. Wrap in <tips></tips> and include the number 5.", lambda t:'<tips>' in t and '</tips>' in t and '5' in t),
 ("Describe fire using exactly the word 'heat' twice. End with the token END.", lambda t:t.lower().count('heat')>=2 and 'END' in t),
 ("Write about music in ALL CAPS, include the word 'RHYTHM'.", lambda t:'RHYTHM' in t and t.upper()==t),
 ("Summarize a day at the beach in 2 sentences, include 'sand' and 'wave'.", lambda t:'sand' in t.lower() and 'wave' in t.lower()),
 ("List two animals, each on its own line starting with '* '. Include 'wild'.", lambda t:'wild' in t.lower() and t.count('* ')>=2),
 ("Explain why sleep matters, include 'rest', and finish with the word FINISH.", lambda t:'rest' in t.lower() and 'FINISH' in t),
 ("Write a haiku about winter, include the word 'snow', three lines.", lambda t:'snow' in t.lower() and t.count(chr(10))>=2),
 ("Describe a forest, include 'green' and 'tall', no commas allowed.", lambda t:'green' in t.lower() and 'tall' in t.lower() and ',' not in t),
 ("Give one fact about the sun containing 'energy', wrapped in quotes \"...\".", lambda t:'energy' in t.lower() and t.count('\"')>=2),
 ("List three numbers as 1) 2) 3) and include the word 'count'.", lambda t:'count' in t.lower() and '1)' in t and '2)' in t and '3)' in t),
]
@torch.no_grad()
def gen(ids,nb):
    QCache.nb=nb
    o=model.generate(ids,max_new_tokens=100,do_sample=False,
        past_key_values=QCache() if nb>0 else DynamicCache())
    return o[0][ids.shape[1]:]
@torch.no_grad()
def score(ids,seq,nb):
    QCache.nb=nb; full=torch.cat([ids,seq.unsqueeze(0)],1)
    o=model(full,past_key_values=QCache() if nb>0 else DynamicCache())
    lg=o.logits[0,ids.shape[1]-1:-1]; lp=torch.log_softmax(lg.float(),-1)
    return lp[torch.arange(seq.shape[0]),seq]

# baseline 1회
base={}
for i,(p,ck) in enumerate(SPEC):
    enc=tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,return_tensors="pt",return_dict=True)
    ids=enc["input_ids"].to(dev); seq=gen(ids,0)
    txt=tok.decode(seq,skip_special_tokens=True)
    base[i]=(ids,seq,ck(txt),txt)
nbase=sum(1 for i in base if base[i][2])
print(f"# Llama-3.1-8B  baseline instr-ok {nbase}/16.  nbits sweep:")
print(f"{'nbits':>5} {'gate통과율':>9} {'instr보존율':>11} {'과보수(gateFAIL·instrOK)':>22} {'silent(gatePASS·broke)':>22}")
for nb in [8,6,5,4,3]:
    gp=ip=cons=silent=0
    for i,(ids,seq,ok_b,txt_b) in base.items():
        if not ok_b: continue
        lpf=score(ids,seq,0); lpc=score(ids,seq,nb)
        mad=(lpf-lpc).abs().max().item()
        pf=torch.exp(-lpf.mean()).item(); pc=torch.exp(-lpc.mean()).item(); pr=abs(pc-pf)/pf
        gate=(mad<=0.5 and pr<=0.1)
        seq_c=gen(ids,nb); ok_c=SPEC[i][1](tok.decode(seq_c,skip_special_tokens=True))
        gp+=gate; ip+=ok_c
        if (not gate) and ok_c: cons+=1
        if gate and (not ok_c): silent+=1
    print(f"{nb:>5} {gp}/{nbase:>7} {ip}/{nbase:>9} {cons:>22} {silent:>22}")
