#!/usr/bin/env python3
"""random 토큰 워크로드의 실측 라우팅 트레이스(routing_stats_random.json)로 M1/v2 의 random 셀 15개를 재예측 (retrodiction, 참고용).
usage: repredict_random_cells.py <routing_stats_random.json>"""
import sys, json, os
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hybrid_step_model as M
R=json.load(open(sys.argv[1]))
# 모델은 per_H 키를 str 로 읽음 → 정규화
R={"per_H":{str(k):{**v,"E_distinct_hot":{str(b):x for b,x in v["E_distinct_hot"].items()},"E_distinct_cold":{str(b):x for b,x in v["E_distinct_cold"].items()}} for k,v in R["per_H"].items()}}
KVT={64:131072,80:131072,96:49152}
cells=[(64,8,2000,94.8),(64,64,2000,389.1),(80,32,2000,187.7),(64,32,2000,233.4),(96,64,2000,139.4),(80,16,2000,123.0),(64,16,2000,143.3),
       (80,16,1000,115.1),(80,32,1000,184.2),(80,16,3000,141.7),(80,32,3000,255.2),(96,16,1000,93.8),(96,32,1000,158.4),(96,16,3000,93.8),(96,32,3000,101.0)]
errs=[]; print(f"{'H':>3}{'C':>4}{'ctx':>6}{'실측라우팅예측':>12}{'실측':>8}{'오차%':>8}  D_c")
for H,C,ctx,meas in cells:
    p=M.tpot_ms(C,H,ctx=ctx,deferred_N=4,kv_loc="gpu",kv_tokens=KVT[H],routing=R)
    e=(p["tpot_ms"]-meas)/meas*100; errs.append(abs(e))
    print(f"{H:3d}{C:4d}{ctx:6d}{p['tpot_ms']:12.1f}{meas:8.1f}{e:8.1f}  {p['D_c']}")
errs.sort(); print(f"\nrandom 트레이스 재예측: 중앙값|오차| {errs[len(errs)//2]:.1f}%  max {errs[-1]:.1f}%  (n={len(errs)}, retrodiction)")
