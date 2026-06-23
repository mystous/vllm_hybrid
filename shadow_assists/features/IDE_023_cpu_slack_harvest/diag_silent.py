import torch, glob
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
MODEL=glob.glob("/raid/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*")[0]
dev="cuda:0"; tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map=dev).eval()
def quant(t,nb):
    qm=2**(nb-1)-1; s=t.abs().amax(-1,keepdim=True).clamp_min(1e-8)/qm
    return (torch.round(t/s).clamp(-qm-1,qm)*s).to(t.dtype)
class QCache(DynamicCache):
    nb=0
    def update(self,k,v,i,*a,**kw): return super().update(quant(k,QCache.nb),quant(v,QCache.nb),i,*a,**kw)
from kv_conservativeness import SPEC
@torch.no_grad()
def gen(ids,nb):
    QCache.nb=nb
    return model.generate(ids,max_new_tokens=100,do_sample=False,past_key_values=QCache() if nb>0 else DynamicCache())[0][ids.shape[1]:]
for i,(p,ck) in enumerate(SPEC):
    enc=tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,return_tensors="pt",return_dict=True)
    ids=enc["input_ids"].to(dev)
    tb=tok.decode(gen(ids,0),skip_special_tokens=True); 
    if not ck(tb): continue
    t8=tok.decode(gen(ids,8),skip_special_tokens=True)
    if not ck(t8):
        print(f"=== SILENT FAILURE P{i}: {p}")
        print(f"--- baseline(OK):\n{tb}\n--- nbits=8(broke):\n{t8}\n")
