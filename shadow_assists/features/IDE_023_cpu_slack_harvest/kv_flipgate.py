#!/usr/bin/env python
# tighter bound 시제: logprob max-abs-diff 대신 "argmax FLIP" 지표가 P7류 silent failure를 잡는가?
# baseline 시퀀스를 압축KV로 teacher-force → 각 위치서 압축 argmax가 baseline 토큰과 다른가(flip).
# flip = cascade 시작점. flip-gate(flip 0개=PASS)가 logprob-gate보다 instruction 보존을 잘 예측하는지 비교.
import torch, glob
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kv_conservativeness import SPEC
MODEL=glob.glob("/raid/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*")[0]
dev="cuda:0"; tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map=dev).eval()
def quant(t,nb):
    qm=2**(nb-1)-1; s=t.abs().amax(-1,keepdim=True).clamp_min(1e-8)/qm
    return (torch.round(t/s).clamp(-qm-1,qm)*s).to(t.dtype)
class QCache(DynamicCache):
    nb=0
    def update(self,k,v,i,*a,**kw): return super().update(quant(k,QCache.nb),quant(v,QCache.nb),i,*a,**kw)
@torch.no_grad()
def gen(ids,nb):
    QCache.nb=nb
    return model.generate(ids,max_new_tokens=100,do_sample=False,past_key_values=QCache() if nb>0 else DynamicCache())[0][ids.shape[1]:]
@torch.no_grad()
def tf_logits(ids,seq,nb):
    QCache.nb=nb; full=torch.cat([ids,seq.unsqueeze(0)],1)
    o=model(full,past_key_values=QCache() if nb>0 else DynamicCache())
    return o.logits[0,ids.shape[1]-1:-1].float()   # [len, vocab] seq 위치 예측
import sys
NB=int(sys.argv[1]) if len(sys.argv)>1 else 8
print(f"# Llama-3.1-8B nbits={NB}: logprob-gate vs FLIP-gate, instruction 보존 예측력")
print(f"{'P':>2} {'instr보존':>8} {'logprob_gate':>12} {'max-abs':>8} {'flips':>6} {'1st_flip':>8} {'flip_margin':>11}")
n=0; lp_correct=0; flip_correct=0
for i,(p,ck) in enumerate(SPEC):
    enc=tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,return_tensors="pt",return_dict=True)
    ids=enc["input_ids"].to(dev); seq=gen(ids,0)
    if not ck(tok.decode(seq,skip_special_tokens=True)): continue
    n+=1
    lg0=tf_logits(ids,seq,0); lgc=tf_logits(ids,seq,NB)
    lp0=torch.log_softmax(lg0,-1)[torch.arange(seq.shape[0]),seq]
    lpc=torch.log_softmax(lgc,-1)[torch.arange(seq.shape[0]),seq]
    mad=(lp0-lpc).abs().max().item()
    lp_gate = mad<=0.5
    # flip: 압축 argmax != baseline 토큰
    camax=lgc.argmax(-1)
    flips=(camax!=seq)
    nflip=int(flips.sum()); first=int(flips.float().argmax()) if nflip>0 else -1
    # baseline margin at first flip (top1-top2 of baseline logits)
    if nflip>0:
        s,_=lg0[first].topk(2); fmarg=(s[0]-s[1]).item()
    else: fmarg=float('nan')
    flip_gate = nflip==0
    ok_c=ck(tok.decode(gen(ids,NB),skip_special_tokens=True))
    # 예측력: gate PASS면 instr 보존 예측. 맞으면 correct.
    lp_correct += (lp_gate==ok_c)
    flip_correct += (flip_gate==ok_c)
    print(f"{i:>2} {str(ok_c):>8} {str(lp_gate):>12} {mad:>8.2f} {nflip:>6} {first:>8} {fmarg:>11.3f}")
print(f"\n예측 정확도(gate==instr보존): logprob-gate {lp_correct}/{n}, FLIP-gate {flip_correct}/{n}")
