"""PoC — rank-r KV 절단이 어텐션 출력에 주는 오차 (정확도/압축 trade). 캡처 KV + K를 쿼리프록시.
attn_full=softmax(QKᵀ/√d)V  vs  rank-r 절단 K,V. 상대오차 vs rank → 압축/정확도."""
import torch, statistics as st, os
os.environ["HF_HOME"]="/raid/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
HF="meta-llama/Llama-3.1-8B-Instruct"; dev="cuda"
tok=AutoTokenizer.from_pretrained(HF)
model=AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, device_map=dev).eval()
text=("The data streaming accelerator processes descriptors through work queues while the memory "
      "controller arbitrates bandwidth across NUMA domains. ")*120
ids=tok(text, return_tensors="pt").input_ids[:, :3000].to(dev)
with torch.no_grad(): out=model(ids, use_cache=True)
pkv=out.past_key_values
def get_kv(c,li):
    if hasattr(c,"key_cache"): return c.key_cache[li],c.value_cache[li]
    L=c.layers[li]; return L.keys,L.values
def lowrank(M,r):  # [seq,hd] → rank-r 재구성
    U,S,Vh=torch.linalg.svd(M.float(),full_matrices=False); return (U[:,:r]*S[:r])@Vh[:r]
def attn_err(K,V,r):  # K,V [seq,hd]; K를 쿼리프록시
    d=K.shape[-1]; Q=K.float()
    full=torch.softmax(Q@K.float().T/d**0.5,-1)@V.float()
    Kr,Vr=lowrank(K,r),lowrank(V,r)
    lr=torch.softmax(Q@Kr.T/d**0.5,-1)@Vr
    return ((full-lr).norm()/full.norm()).item()
nl=len(pkv.key_cache) if hasattr(pkv,"key_cache") else len(pkv.layers)
k0,_=get_kv(pkv,0); nh=k0.shape[1]; hd=k0.shape[-1]
import collections; agg=collections.defaultdict(list)
for li in [0,nl//2,nl-1]:
    K,V=get_kv(pkv,li); K,V=K[0],V[0]
    for hi in range(0,nh,2):
        for r in [16,32,64]:
            agg[r].append(attn_err(K[hi],V[hi],r))
print(f"=== rank-r KV 절단 어텐션 출력 상대오차 (head_dim={hd}, 압축비=hd/r) ===")
for r in [16,32,64]:
    m=st.mean(agg[r]); print(f"  rank{r:3d} (압축 {hd/r:.1f}x): 어텐션출력 relerr={m:.4f}")
print("\n→ 낮은 rank서 오차 작으면 압축 유리. Palu/Eigen(50~60%압축≈2~2.5x)와 비교.")
