#!/usr/bin/env python3
"""recorder 덤프 (physical 카운트, 현재 hotmap 기준) → 논리 expert 빈도 → 층별 빈도순 physical_to_logical_map (hotmap_decode.json).
또한 두 hotmap 의 H=96 커버리지·E[D_cold](B=32) 를 새 트레이스로 계산해 비교 출력."""
import torch, glob, json, statistics, sys
cur=json.load(open("/tmp/hotmap.json"))["physical_to_logical_map"]
fs=sorted(glob.glob("/tmp/expert_distribution_recorder_*_0.pt")); print("files", len(fs))
d=torch.load(fs[0], map_location="cpu", weights_only=False); tot=None
npass=0
for r in d["records"]:
    c=r["global_physical_count"]
    if float(c[0].sum()/8) > 64: continue   # decode 패스만 (프리필 제외)
    npass+=1; tot=c.clone().double() if tot is None else tot+c.double()
print("decode passes", npass)
L,E=tot.shape; tokens=float(tot[0].sum()/8); print("layers",L,"experts",E,"tokens",tokens)
logical=torch.zeros(L,E,dtype=torch.double)
for l in range(L):
    for p in range(E): logical[l][cur[l][p]]+=tot[l][p]
newmap=[]
for l in range(L):
    order=torch.argsort(logical[l], descending=True).tolist(); newmap.append(order)
json.dump({"physical_to_logical_map":newmap}, open("/tmp/hotmap_decode2.json","w"))
def stats(pmap, H=96, B=32):
    cov=[]; dc=[]
    for l in range(L):
        p=logical[l]/tokens; hot=pmap[l][:H]; cold=pmap[l][H:]
        cov.append(float(sum(logical[l][e] for e in hot)/logical[l].sum()))
        dc.append(float(sum(1-(1-min(1.0,float(p[e])))**B for e in cold)))
    return statistics.mean(cov), statistics.mean(dc)
for name,m in (("prompt-derived(current)",cur),("decode-derived(new)",newmap)):
    for H in (80,96):
        c,dcv=stats(m,H); print(f"{name} H={H}: coverage {c:.4f} E[D_cold](B=32) {dcv:.2f}")
ov=[len(set(cur[l][:96])&set(newmap[l][:96])) for l in range(L)]; print("hot-96 overlap per layer: mean %.1f min %d"%(statistics.mean(ov),min(ov)))
print("saved /tmp/hotmap_decode2.json")
