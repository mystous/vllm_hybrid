#!/usr/bin/env python3
"""정상 스텝의 GPU 유휴 갭 구조: 갭(>thr µs) 앞/뒤 커널 이름, 갭 크기 분포, 층당 갭 수 → 대기 지점 특정.
usage: analyze_gaps.py <dir with *TP-0.trace.json.gz> [thr_us=30]"""
import gzip, json, glob, sys, collections
d=sys.argv[1]; thr=float(sys.argv[2]) if len(sys.argv)>2 else 30.0
f=sorted(glob.glob(f"{d}/*TP-0.trace.json.gz"))[0]
ev=json.load(gzip.open(f))["traceEvents"]; X=[e for e in ev if e.get("ph")=="X"]
gl=sorted([e for e in X if e["name"]=="cudaGraphLaunch"], key=lambda e:e["ts"])
gpu=sorted([e for e in X if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")], key=lambda e:e["ts"])
def short(n): return n.replace("void ","")[:38]
for i in range(1, min(len(gl)-1, 3)):   # 정상 스텝 2개
    t0,t1=gl[i]["ts"],gl[i+1]["ts"]; w=[e for e in gpu if t0<=e["ts"]<t1]
    if not w: continue
    span=max(e["ts"]+e["dur"] for e in w)-w[0]["ts"]
    gaps=[]; cur=w[0]["ts"]+w[0]["dur"]; prev=w[0]
    for e in w[1:]:
        g=e["ts"]-cur
        if g>thr: gaps.append((g, short(prev["name"]), short(e["name"])))
        if e["ts"]+e["dur"]>cur: cur=e["ts"]+e["dur"]; prev=e
    tot=sum(g for g,_,_ in gaps)
    print(f"\n=== step{i}: span {span/1000:.1f} ms | gaps>{thr:.0f}us: {len(gaps)} 개, 합 {tot/1000:.1f} ms ===")
    hist=collections.Counter()
    for g,a,b in gaps: hist[(a,b)]+=g
    print("갭 위치 (앞 커널 → 뒤 커널) 별 합계 ms, 상위 8:")
    for (a,b),v in hist.most_common(8):
        n=sum(1 for g,x,y in gaps if (x,y)==(a,b)); print(f"  {v/1000:6.2f} ms  x{n:3d}  {a} → {b}")
    sizes=sorted(g for g,_,_ in gaps)
    print(f"갭 크기 분위: p50 {sizes[len(sizes)//2]:.0f}us p90 {sizes[int(len(sizes)*.9)]:.0f}us max {sizes[-1]:.0f}us")
