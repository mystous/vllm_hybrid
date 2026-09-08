#!/usr/bin/env python3
"""recorder 덤프 → phase 분리 (prefill >64 tok / decode ≤64) → 논리 빈도 → hotmap_{prompt,alpha<a>}.json + routing_phase_stats.json (임의 모델).
usage: build_hotmaps_generic.py <cur_map.json|identity> <out_dir> <alpha...>   (통계 H 격자: 32,48,64,80,96,112)"""
import torch, glob, json, statistics, sys, os
cur_arg, out = sys.argv[1], sys.argv[2]; alphas=[float(a) for a in sys.argv[3:]]; os.makedirs(out, exist_ok=True)
fs=sorted(glob.glob("/tmp/expert_distribution_recorder_*_0.pt")); d=torch.load(fs[0], map_location="cpu", weights_only=False)
tp=None; td=None; np_=nd=0
for r in d["records"]:
    c=r["global_physical_count"].double(); t=float(c[0].sum()/8)
    if t>64: tp=c.clone() if tp is None else tp+c; np_+=1
    else: td=c.clone() if td is None else td+c; nd+=1
L,E=td.shape; print(f"L={L} E={E} prefill passes {np_} tokens {float(tp[0].sum()/8):.0f} | decode passes {nd} tokens {float(td[0].sum()/8):.0f}")
cur = [list(range(E)) for _ in range(L)] if cur_arg=="identity" else json.load(open(cur_arg))["physical_to_logical_map"]
def to_logical(t):
    o=torch.zeros(L,E,dtype=torch.double)
    for l in range(L):
        for p in range(E): o[l][cur[l][p]]+=t[l][p]
    return o
lp=to_logical(tp); ld=to_logical(td); pp=lp/lp.sum(1,keepdim=True); pd=ld/ld.sum(1,keepdim=True)
maps={"prompt":[torch.argsort(pp[l],descending=True).tolist() for l in range(L)]}
for a in alphas: sc=a*pp+(1-a)*pd; maps[f"alpha{a}"]=[torch.argsort(sc[l],descending=True).tolist() for l in range(L)]
maps["decode_only"]=[torch.argsort(pd[l],descending=True).tolist() for l in range(L)]
for n,m in maps.items(): json.dump({"physical_to_logical_map":m}, open(f"{out}/hotmap_{n}.json","w"))
Hs=[h for h in (32,48,64,80,96,112) if h<E]; k=8
st={"L":L,"E":E,"tokens_prefill":float(tp[0].sum()/8),"tokens_decode":float(td[0].sum()/8),"maps":{}}
for n,m in maps.items():
    o={}
    for H in Hs:
        covp=statistics.mean(float(sum(pp[l][e] for e in m[l][:H])) for l in range(L)); covd=statistics.mean(float(sum(pd[l][e] for e in m[l][:H])) for l in range(L))
        Dc={}; Dh={}
        for B in (1,4,8,16,32,64,128):
            Dc[B]=round(statistics.mean(float(sum(1-(1-min(1.0,float(pd[l][e]*k)))**B for e in m[l][H:])) for l in range(L)),2)
            Dh[B]=round(statistics.mean(float(sum(1-(1-min(1.0,float(pd[l][e]*k)))**B for e in m[l][:H])) for l in range(L)),2)
        o[H]={"cov_prefill":round(covp,4),"cov_decode":round(covd,4),"cold_pairs_per_token_prefill":round(k*(1-covp),4),"cold_pairs_per_token_decode":round(k*(1-covd),4),"E_Dc_decode":Dc,"E_Dh_decode":Dh}
        print(f"{n:12s} H={H:3d}: cov_p {covp:.4f} cov_d {covd:.4f} E[Dc] B32/64 = {Dc[32]}/{Dc[64]}")
    st["maps"][n]=o
json.dump(st, open(f"{out}/routing_phase_stats.json","w"), indent=1); print("saved", out)
