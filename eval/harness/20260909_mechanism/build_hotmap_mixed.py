#!/usr/bin/env python3
"""phase-가중 hot set: 같은 recorder 덤프에서 prefill 패스(>64 tok)와 decode 패스(≤64 tok)를 분리해 논리 expert 빈도 p_p(e), p_d(e) 를 구하고
score(e) = α·p_p(e) + (1−α)·p_d(e) 로 층별 정렬 → /tmp/hotmap_mixed_<α>.json. 각 α 의 H=96 prefill/decode 커버리지 출력.
usage: build_hotmap_mixed.py 0.25 0.5 0.75"""
import torch, glob, json, statistics, sys
cur=json.load(open("/tmp/hotmap.json"))["physical_to_logical_map"]
fs=sorted(glob.glob("/tmp/expert_distribution_recorder_*_0.pt")); d=torch.load(fs[0], map_location="cpu", weights_only=False)
tp=None; td=None; np_=nd=0
for r in d["records"]:
    c=r["global_physical_count"].double(); t=float(c[0].sum()/8)
    if t>64: tp=c.clone() if tp is None else tp+c; np_+=1
    else: td=c.clone() if td is None else td+c; nd+=1
L,E=td.shape; print(f"prefill passes {np_} tokens {float(tp[0].sum()/8):.0f} | decode passes {nd} tokens {float(td[0].sum()/8):.0f}")
def to_logical(t):
    out=torch.zeros(L,E,dtype=torch.double)
    for l in range(L):
        for p in range(E): out[l][cur[l][p]]+=t[l][p]
    return out
lp=to_logical(tp); ld=to_logical(td)
pp=lp/lp.sum(1,keepdim=True); pd=ld/ld.sum(1,keepdim=True)   # 층별 (token,expert) 쌍 분포
def cov(pmap,H,prob): return statistics.mean(float(sum(prob[l][e] for e in pmap[l][:H])) for l in range(L))
def edc(pmap,H,prob,B=32): return statistics.mean(float(sum(1-(1-min(1.0,float(prob[l][e]*8)))**B for e in pmap[l][H:])) for l in range(L))
print(f"current   H=96: cov_prefill {cov(cur,96,pp):.4f} cov_decode {cov(cur,96,pd):.4f} E[D_c]dec(B32) {edc(cur,96,pd):.2f}")
for a in [float(x) for x in sys.argv[1:]]:
    score=a*pp+(1-a)*pd; m=[torch.argsort(score[l],descending=True).tolist() for l in range(L)]
    json.dump({"physical_to_logical_map":m}, open(f"/tmp/hotmap_mixed_{a}.json","w"))
    ov=statistics.mean(len(set(cur[l][:96])&set(m[l][:96])) for l in range(L))
    print(f"alpha={a} H=96: cov_prefill {cov(m,96,pp):.4f} cov_decode {cov(m,96,pd):.4f} E[D_c]dec(B32) {edc(m,96,pd):.2f} overlap_cur {ov:.1f}")
