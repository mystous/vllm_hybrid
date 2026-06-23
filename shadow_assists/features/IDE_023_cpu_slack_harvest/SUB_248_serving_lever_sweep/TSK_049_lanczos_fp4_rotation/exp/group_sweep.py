"""회전 이득 vs 양자화 granularity — group 크기를 16(NVFP4)→coarse 로 키우며 회전 이득 관찰.
란초스(whiten)·Hadamard 가 coarse 에서 살아나면 = 회전의 가치는 coarse 양자화에 있음을 입증."""
import torch, glob, collections
from safetensors import safe_open
HF="/raid/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct"
E2M1=torch.tensor([0,0.5,1,1.5,2,3,4,6])
def quant_fp4(x, group):
    *lead,n=x.shape
    if group<=0 or group>n: group=n
    pad=(group-n%group)%group
    if pad: x=torch.nn.functional.pad(x,(0,pad))
    g=x.reshape(*lead,-1,group); amax=g.abs().amax(-1,keepdim=True).clamp_min(1e-8); s=amax/6.0
    q=(g/s).abs().unsqueeze(-1); lv=E2M1.to(x.device); idx=(q-lv).abs().argmin(-1)
    dq=(g/s).sign()*lv[idx]*s; return dq.reshape(*lead,-1)[...,:n]
def relerr(W,R,group): Wp=W if R is None else W@R; Q=quant_fp4(Wp,group); return ((Wp-Q).norm()/Wp.norm()).item()
def rots(W,dev):
    from compressed_tensors.transform import random_hadamard_matrix
    n=W.shape[1]; G=(W.t()@W).double(); L,U=torch.linalg.eigh(G); L=L.clamp_min(1e-6)
    return {"none":None,"hadamard":random_hadamard_matrix(n).to(dev).float(),
            "whiten(란초스)":(U@torch.diag(L.rsqrt())).float().to(dev)}
dev="cuda"
sf=sorted(glob.glob(f"{HF}/snapshots/*/*.safetensors"))
tg=[(f,k) for f in sf for k in safe_open(f,"pt").keys() if ("down_proj.weight" in k or "o_proj.weight" in k) and any(f".layers.{i}." in k for i in [0,16,31])]
print(f"가중치 {len(tg)}개, group sweep")
for group in [16,32,128,512,-1]:
    agg=collections.defaultdict(list)
    for f,k in tg:
        with safe_open(f,"pt") as h: W=h.get_tensor(k).float().to(dev)
        R=rots(W,dev)
        for nm,r in R.items(): agg[nm].append(relerr(W,r,group))
    gl="per-channel" if group<0 else f"g={group}"
    b=sum(agg["none"])/len(agg["none"])
    line=f"  {gl:12s} none={b:.4f}"
    for nm in ["hadamard","whiten(란초스)"]:
        m=sum(agg[nm])/len(agg[nm]); line+=f"  {nm}={m:.4f}({(m/b-1)*100:+.0f}%)"
    print(line)
print("\n→ group↑(coarse)일수록 회전(hadamard/란초스)이 none 대비 오차↓면 = 회전의 가치는 coarse 양자화. NVFP4(g16)엔 여지 적음.")
