"""SUB_249 전제검증 — 실제 Llama-3.1-8B K/V 캐시의 저랭크성 측정.
긴 프롬프트 forward → past_key_values 캡처 → 레이어/헤드별 K,V [seq×head_dim] SVD →
99%/90% 에너지 실효 rank. head_dim=128 대비 실효 rank ≪ 128 이면 저랭크(압축 가능)."""
import torch, glob, statistics as st
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="meta-llama/Llama-3.1-8B-Instruct"
import os; os.environ["HF_HOME"]="/raid/hf_cache"
dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF, torch_dtype=torch.bfloat16, device_map=dev)
model.eval()
# 긴 프롬프트 (~3000 토큰)
base=("The data streaming accelerator processes descriptors through work queues while the memory "
      "controller arbitrates bandwidth across NUMA domains and the sampler validates draft tokens. ")
text=base*120
ids=tok(text, return_tensors="pt").input_ids[:, :3000].to(dev)
print(f"seq_len={ids.shape[1]}")
with torch.no_grad():
    out=model(ids, use_cache=True)
pkv=out.past_key_values   # layers × (K,V) each [b, heads, seq, hd]
def eff_rank(M, thr):  # M [seq, hd]
    s=torch.linalg.svdvals(M.float())
    e=(s**2); c=torch.cumsum(e,0)/e.sum()
    return int((c<thr).sum().item())+1
import collections; agg=collections.defaultdict(list)
# transformers 5.x: pkv may be a Cache object
# transformers 5.x DynamicCache 접근 (여러 패턴 시도)
def get_kv(cache, li):
    for attr in ["key_cache","value_cache"]:
        pass
    if hasattr(cache,"key_cache") and len(getattr(cache,"key_cache"))>li:
        return cache.key_cache[li], cache.value_cache[li]
    if hasattr(cache,"layers"):
        L=cache.layers[li]
        return getattr(L,"keys",getattr(L,"key_cache",None)), getattr(L,"values",getattr(L,"value_cache",None))
    raise RuntimeError("cache 접근 불가: "+str(type(cache))+" attrs="+str([a for a in dir(cache) if not a.startswith('__')][:20]))
# 레이어 수
nl=getattr(pkv,"__len__",lambda:0)()
if nl==0:
    nl=len(pkv.key_cache) if hasattr(pkv,"key_cache") else len(pkv.layers)
k0,v0=get_kv(pkv,0); hd=k0.shape[-1]; nh=k0.shape[1]
layers=None
print(f"layers={nl} heads={nh} head_dim={hd}")
for li in [0, nl//2, nl-1]:
    Kc,Vc=get_kv(pkv,li); K,V=Kc[0],Vc[0]   # [heads, seq, hd]
    for hi in range(0, nh, max(1,nh//4)):
        agg[f"K_L{li}"].append(eff_rank(K[hi], 0.99))
        agg[f"V_L{li}"].append(eff_rank(V[hi], 0.99))
print(f"\n=== K/V 실효 rank @99% 에너지 (head_dim={hd}) ===")
for k in sorted(agg): print(f"  {k:8s} 평균 rank99={st.mean(agg[k]):.1f} / {hd}  → 압축비 ~{hd/st.mean(agg[k]):.1f}x")
print("\n→ rank99 ≪ head_dim 이면 KV 저랭크 = 란초스 압축 가능. rank99≈head_dim 이면 full-rank=FP4처럼 dead.")
