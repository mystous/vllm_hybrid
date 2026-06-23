"""TSK_049 통제 실험 — 회전별 FP4 양자화 오차 비교 (실제 Llama-3.1-8B 가중치).
회전: none / Hadamard / eigen-projection(Uᵀ) / whitening(Λ^-1/2 Uᵀ, 란초스 고유분해).
FP4 = NVFP4 근사(E2M1 4-bit, per-group=16 FP8 스케일). 오차 = ‖W'-Q(W')‖_F/‖W'‖_F (저=좋음).
"""
import torch, math, glob, json, os
from safetensors import safe_open

HF="/raid/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct"
def load_weights():
    sf=sorted(glob.glob(f"{HF}/snapshots/*/*.safetensors"))
    W={}
    for f in sf:
        with safe_open(f,"pt") as h:
            for k in h.keys():
                # 대표 선형층: 여러 레이어의 mlp.down_proj / self_attn.o_proj (outlier 잘 생기는 곳)
                if any(s in k for s in ["mlp.down_proj.weight","self_attn.o_proj.weight"]) and ".layers.1" in k.split("model.layers.")[-1][:3]:
                    pass
    return sf

# E2M1 4-bit float 표현값 (NVFP4): {0,±0.5,±1,±1.5,±2,±3,±4,±6}
E2M1=torch.tensor([0,0.5,1,1.5,2,3,4,6])
def quant_fp4(x, group=16):
    # per-group(마지막 차원 16) absmax 스케일 → E2M1 라운드 → 디퀀트
    *lead, n = x.shape
    pad=(group-n%group)%group
    if pad: x=torch.nn.functional.pad(x,(0,pad))
    g=x.reshape(*lead, -1, group)
    amax=g.abs().amax(-1,keepdim=True).clamp_min(1e-8)
    s=amax/6.0                                  # 스케일: 최대값을 E2M1 max(6)에 맞춤
    q=g/s
    lv=E2M1.to(x.device)
    # 가장 가까운 E2M1 값 (부호 보존)
    sign=q.sign(); aq=q.abs().unsqueeze(-1)
    idx=(aq-lv).abs().argmin(-1)
    dq=sign*lv[idx]*s.squeeze(-1).unsqueeze(-1).squeeze(-1) if False else sign*lv[idx]
    dq=dq*s
    out=dq.reshape(*lead,-1)[...,:n]
    return out

def relerr(W, R=None):
    # W: [out,in]. 회전 R(in×in) 적용 → W'=W@R → 양자화 → 오차
    Wp = W if R is None else W@R
    Q = quant_fp4(Wp)
    return (Wp-Q).norm()/Wp.norm()

def hadamard(n, device):
    from compressed_tensors.transform import random_hadamard_matrix
    return random_hadamard_matrix(n).to(device).float()

def eigen_rotations(W):
    # 입력채널 Gram: G = WᵀW (in×in). Lanczos(고유분해)로 U,Λ.
    G=(W.t()@W).double()
    # 전체 eigh (in=4096~14336, 가능). 란초스는 top-k용이나 여기선 풀 분해로 정확 비교.
    L,U=torch.linalg.eigh(G)        # 오름차순
    L=L.clamp_min(1e-6)
    Uf=U.float()
    eigen=Uf                         # eigen-projection: x→Uᵀx (회전 W@U)
    whiten=(U@torch.diag(L.rsqrt())).float()  # whitening: 분산 균등화(outlier 펴기)
    # whitening은 비직교(스케일 포함) → 출력 동치 위해선 역변환 필요하나, 여기선 "양자화 친화도"만 측정
    return eigen, whiten

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    sf=sorted(glob.glob(f"{HF}/snapshots/*/*.safetensors"))
    targets=[]
    for f in sf:
        with safe_open(f,"pt") as h:
            for k in h.keys():
                if ("mlp.down_proj.weight" in k or "self_attn.o_proj.weight" in k) and any(f".layers.{i}." in k for i in [0,8,16,24,31]):
                    targets.append((f,k))
    print(f"대상 가중치 {len(targets)}개 (down_proj/o_proj, layers 0/8/16/24/31)")
    import collections; agg=collections.defaultdict(list)
    for f,k in targets:
        with safe_open(f,"pt") as h: W=h.get_tensor(k).float().to(dev)
        n=W.shape[1]
        eg,wh=eigen_rotations(W)
        Had=hadamard(n,dev)
        rots={"none":None,"hadamard":Had,"eigen(Uᵀ)":eg.to(dev),"whiten(란초스)":wh.to(dev)}
        for name,R in rots.items():
            e=relerr(W,R).item(); agg[name].append(e)
    print("\n=== FP4 양자화 상대오차 (평균, 낮을수록 양자화-친화) ===")
    base=sum(agg["none"])/len(agg["none"])
    for name in ["none","hadamard","eigen(Uᵀ)","whiten(란초스)"]:
        m=sum(agg[name])/len(agg[name]); print(f"  {name:16s} relerr={m:.4f}  vs none={(m/base-1)*100:+.1f}%")
    print("\n판정: hadamard/whiten 이 none 보다 오차↓면 회전이 FP4 친화. 란초스(whiten)가 hadamard 이기면 가치.")

if __name__=="__main__": main()
