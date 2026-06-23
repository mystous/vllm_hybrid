"""신규 시도 — 란초스 저랭크 양자화-오차 보정: Ŵ=Q_fp4(W), E=W-Ŵ, top-k SVD(E)=Lₖ,
보정 W̃=Ŵ+Lₖ. NVFP4(g16) 그대로 + 작은 rank-k FP16 보정으로 정확도↑ 가능한지.
저장 비용: rank-k → k*(out+in)*2B vs 원 가중치. 오차↓ vs 비용 trade 측정."""
import torch, glob, collections
from safetensors import safe_open
HF="/raid/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct"
E2M1=torch.tensor([0,0.5,1,1.5,2,3,4,6])
def qfp4(x,group=16):
    *lead,n=x.shape; pad=(group-n%group)%group
    if pad: x=torch.nn.functional.pad(x,(0,pad))
    g=x.reshape(*lead,-1,group); s=g.abs().amax(-1,keepdim=True).clamp_min(1e-8)/6.0
    q=(g/s).abs().unsqueeze(-1); lv=E2M1.to(x.device); idx=(q-lv).abs().argmin(-1)
    return ((g/s).sign()*lv[idx]*s).reshape(*lead,-1)[...,:n]
dev="cuda"
sf=sorted(glob.glob(f"{HF}/snapshots/*/*.safetensors"))
tg=[(f,k) for f in sf for k in safe_open(f,"pt").keys() if ("down_proj.weight" in k or "o_proj.weight" in k or "gate_proj.weight" in k) and any(f".layers.{i}." in k for i in [0,16,31])]
print(f"가중치 {len(tg)}개")
agg=collections.defaultdict(list)
for f,k in tg:
    with safe_open(f,"pt") as h: W=h.get_tensor(k).float().to(dev)
    Wq=qfp4(W); E=W-Wq
    base=(E.norm()/W.norm()).item(); agg["fp4(보정無)"].append(base)
    # top-k SVD of E (란초스/randomized) — 여러 rank
    U,S,Vh=torch.linalg.svd(E, full_matrices=False)
    for k_ in [8,16,32,64]:
        Lk=(U[:,:k_]*S[:k_])@Vh[:k_]
        err=((W-(Wq+Lk)).norm()/W.norm()).item(); agg[f"+rank{k_}"].append(err)
print("\n=== FP4 상대오차 — 란초스 저랭크 보정 ===")
b=sum(agg["fp4(보정無)"])/len(agg["fp4(보정無)"])
for nm in ["fp4(보정無)","+rank8","+rank16","+rank32","+rank64"]:
    m=sum(agg[nm])/len(agg[nm]); print(f"  {nm:12s} relerr={m:.4f}  vs fp4={(m/b-1)*100:+.1f}%")
# rank64 의 추가 저장 비율 (예: down_proj 4096x14336)
print("\n저장 오버헤드(rank64, FP16 보정 / FP4 가중치): ~ k*(out+in)*16bit / (out*in*4.25bit)")
print("→ 오차↓ 폭 vs 저장↑ 폭으로 신규성/실용성 판정. (LQER/LoftQ 류와 중복 여부는 별도 검토.)")
