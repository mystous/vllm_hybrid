#!/usr/bin/env python
# eviction 변형 kill-gate: StreamingLLM류(sink S + window W) eviction이 instruction 깨는가, 게이트가 잡는가?
# incremental decode 루프로 직접 제어(generate 우회). teacher-force 채점 + greedy 자유생성 둘 다.
import torch, sys, glob
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
MODEL=glob.glob("/raid/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*")[0]
W=int(sys.argv[1]) if len(sys.argv)>1 else 0   # window 크기(0=eviction 없음=full)
S=4                                            # sink 토큰
dev="cuda:0"
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map=dev).eval()

def evict(cache,S,W):
    if W<=0: return
    for lyr in cache.layers:
        L=lyr.keys.shape[2]
        if L>S+W:
            idx=torch.cat([torch.arange(S,device=dev),torch.arange(L-W,L,device=dev)])
            lyr.keys=lyr.keys.index_select(2,idx); lyr.values=lyr.values.index_select(2,idx)

@torch.no_grad()
def run(prompt_ids, teacher=None, S=4, W=0, maxnew=120):
    # teacher!=None: 그 시퀀스를 강제하며 per-token logprob; None: greedy 자유생성
    cache=DynamicCache()
    o=model(prompt_ids,past_key_values=cache,use_cache=True)
    logits=o.logits[:,-1].float()
    lps=[]; gen=[]
    n = len(teacher) if teacher is not None else maxnew
    for i in range(n):
        if teacher is not None: t=teacher[i]
        else: t=logits.argmax(-1)[0]
        lps.append(torch.log_softmax(logits,-1)[0,t])
        gen.append(int(t))
        if teacher is None and int(t)==tok.eos_token_id: break
        evict(cache,S,W)
        cp=torch.tensor([cache.get_seq_length()],device=dev)
        o=model(t.view(1,1),past_key_values=cache,use_cache=True,cache_position=cp)
        logits=o.logits[:,-1].float()
    return torch.stack(lps), gen

PROMPTS=[
 "Write a short note about coffee. Constraints: use EXACTLY two bullet points starting with '-', include the word 'aroma', and end your entire response with the token DONE.",
 "List two benefits of sleep. Each must be one sentence. Wrap the whole answer in <ans>...</ans> tags and include the number 8.",
 "Explain photosynthesis to a child. Use no more than 40 words, include the word 'sunlight', and respond entirely in lowercase.",
 "Summarize the water cycle. You MUST use the word 'evaporation' and format as a numbered list 1. 2. 3., then write THE END.",
]
def check(text,i):
    t=text.lower()
    if i==0: return ('done' in t) and text.count('-')>=2 and 'aroma' in t
    if i==1: return '<ans>' in t and '</ans>' in t and '8' in t
    if i==2: return 'sunlight' in t and text==text.lower() and len(text.split())<=45
    if i==3: return 'evaporation' in t and 'the end' in t and '1.' in t
    return False
print(f"# Llama-3.1-8B  StreamingLLM eviction sink={S} window={W} (0=full)")
for i,p in enumerate(PROMPTS):
    enc=tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,return_tensors="pt",return_dict=True)
    ids=enc["input_ids"].to(dev)
    lp_full,base=run(ids,W=0)                          # baseline 자유생성
    txt_full=tok.decode(base,skip_special_tokens=True); ok_full=check(txt_full,i)
    base_t=torch.tensor(base,device=dev)
    lp_ev,_=run(ids,teacher=base_t,S=S,W=W)            # baseline출력을 eviction으로 재채점
    _,gen_ev=run(ids,S=S,W=W)                          # eviction 자유생성
    txt_ev=tok.decode(gen_ev,skip_special_tokens=True); ok_ev=check(txt_ev,i)
    mad=(lp_full-lp_ev).abs().max().item()
    pf=torch.exp(-lp_full.mean()).item(); pe=torch.exp(-lp_ev.mean()).item(); pr=abs(pe-pf)/pf
    gate='PASS' if (mad<=0.5 and pr<=0.1) else 'FAIL'
    print(f"P{i}: instr full={ok_full} evict={ok_ev} | max-abs-diff={mad:.3f} ppl_rel={pr:.3f} | GATE={gate} | broke={ok_full and not ok_ev}")
