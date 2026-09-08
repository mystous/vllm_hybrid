#!/usr/bin/env python3
"""235B 사전 예측 (모델 v3 구조, 파라미터는 480B in-situ 값을 모델 크기로 스케일 — 235B 마이크로벤치 없음, 가정 명시):
  expert 크기 비 s = (4096·1536)/(6144·2560) = 0.40 → CPU expert 당 T_E = 75·0.40 = 30µs (DDR 스트리밍 비례), T_0 = 80·0.6 (행/버퍼 준비), T_W 40 유지
  GPU expert 커널: fixed 39.7 + 4.26·0.40·D_h µs; 비-expert: 480B 값 × (hidden 비 0.667) × (층 비 94/62) — attention/other 항만
  층 94, k=8, E=128. D_c/D_h = routing_phase_stats_235b.json (hotmap 별). 고정비 2.0ms, 핸드오프 20µs, D_c 비율 0.88.
  GPU-only: 층당 gpu_expert(D_h=E[distinct 전체]) + 비-expert, CPU 항 0. prefill 몫: 480B 법칙 × 0.40 (expert 연산 비례) 가정.
셀: GPU-only C32/C64, hybrid H96 prompt C32/C64, H96 α.25 C32/C64, H64 α.25 C64. 출력: predictions_235b.json"""
import json, os, sys, hashlib
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import hybrid_step_model as V2, hybrid_step_model_v3 as M3
S=0.40; L=94
ST=json.load(open(os.path.join(HERE,"..","routing_phase_stats_235b.json")))
def interp(t,B): return M3._interp(t,B)
def cell(C,H,hm,kv=131072,gpu_only=False):
    V2.set_workload(640); W=V2.WORKLOAD; B=min(C,max(1,kv//(W["in_unique"]+W["out"]//2)))
    if gpu_only:
        r=ST["maps"]["prompt"][str(112) if "112" in ST["maps"]["prompt"] else "96"]; Dh=interp(r["E_Dh_decode"],B)+interp(r["E_Dc_decode"],B); Dc=0.0; frac=0.0
    else:
        r=ST["maps"][hm][str(H)]; Dc=M3.DC_RATIO*interp(r["E_Dc_decode"],B); Dh=interp(r["E_Dh_decode"],B); frac=r["cold_pairs_per_token_prefill"]/8
    cpu=(40+80*0.6+75*S*Dc)/1e3 if not gpu_only else 0.0
    gexp=(39.7+4.26*S*Dh)/1e3; gnon=(V2.gpu_nonexpert_ms(B,640)-V2.GPU_NONEXPERT["fixed_ms"])*0.667*(L/62)/L
    layer=max(cpu,gexp+gnon)+(0.02 if not gpu_only else 0.0); step=L*layer+2.0
    bp=min(C,32); pre_req=interp({str(k):v*S for k,v in M3.PREFILL3["per_req_ms"].items()},bp)
    coef=(1100+28000*frac)/(1100+28000*0.019) if not gpu_only else 1100/(1100+28000*0.019)
    pre=B*pre_req*coef/W["out"]; tpot=step+pre; ttft=2.95*bp*pre_req*coef
    return dict(C=C,H=H,hotmap=hm,gpu_only=gpu_only,B_eff=B,D_c=round(Dc,2),D_h=round(Dh,1),cpu_layer_ms=round(cpu,3),gpu_layer_ms=round(gexp+gnon,3),step_ms=round(step,1),tpot_ms=round(tpot,1),ttft_ms=round(ttft),tok_s=round(B*W["out"]/(ttft+(W["out"]-1)*tpot)*1e3,1),binding="gpu" if gpu_only or cpu<gexp+gnon else "cpu_ddr")
cells=[dict(C=32,H=128,hm="prompt",gpu_only=True,kv=65536),dict(C=64,H=128,hm="prompt",gpu_only=True,kv=65536),
       dict(C=32,H=96,hm="prompt"),dict(C=64,H=96,hm="prompt"),dict(C=32,H=96,hm="alpha0.25"),dict(C=64,H=96,hm="alpha0.25"),dict(C=64,H=64,hm="alpha0.25")]
preds=[]
for c in cells:
    r=cell(c["C"],c["H"],c["hm"],c.get("kv",131072),c.get("gpu_only",False)); preds.append(r)
    print(f"{'GPU-only' if r['gpu_only'] else 'H%d %s'%(r['H'],r['hotmap']):18s} C{r['C']:2d}: TPOT {r['tpot_ms']:6.1f} tok/s {r['tok_s']:6.1f} TTFT {r['ttft_ms']:5.0f} D_c {r['D_c']:5.2f} D_h {r['D_h']:5.1f} cpuL {r['cpu_layer_ms']:.3f} gpuL {r['gpu_layer_ms']:.3f} {r['binding']}")
json.dump(dict(model="v3 구조 + 480B→235B 스케일 가정 (expert 크기 비 0.40)",script_sha256=hashlib.sha256(open(__file__,"rb").read()).hexdigest(),gate="TPOT 순위 일치 ≥80% (절대값은 스케일 가정이라 보고만); 핵심 질문 = GPU-only vs hybrid 우열 방향",predictions=preds),open(os.path.join(HERE,"..","predictions_235b.json"),"w"),indent=1,ensure_ascii=False)
print("saved predictions_235b.json")
