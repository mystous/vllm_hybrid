#!/usr/bin/env python3
"""M1 사전 예측 생성: m1_cells.json 의 12셀에 대해 현 모델 (hybrid_step_model.py) 로 TPOT·decode 스텝·binding 을 예측해
predictions.json 으로 저장. 측정 전에 git 커밋해 해시를 고정한다 (PLN_006 규율).
- deferred N: 모든 셀 4 (최종 구성과 동일) — H=64 셀 포함.
- KV 토큰: H 별 최대 가능치 (mem-fraction 0.92 공통): H=64/80 → 131072, H=96 → 49152. KV 는 토큰당 63 KB/GPU (bf16) 로 작아 hot expert 와의 메모리 교환은 C×ctx 가 큰 셀에서만 binding.
- kv=hicache 셀: HiCache 64GB 부착, ctx=2000 은 멀티턴(공유 prefix 1500 + 신규 100, 출력 512) 워크로드로 host prefix 읽기 발생 →
  DDR 간섭 계수 (같은 소켓 가정: 0.73) 적용. ctx=640 hicache 셀은 prefix 100 만 공유 → 간섭 없음 (계수 1.0) 으로 예측.
"""
import json, os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hybrid_step_model as M

KV_TOKENS = {64: 131072, 80: 131072, 96: 49152}   # H 별 최대 가능 KV (mem-fraction 0.92; hot-96 49152 부팅 확인 12:45, avail 4.10 GB)
cells = json.load(open(os.path.join(HERE, "..", "m1_cells.json")))["cells"]
routing = M.load_routing()
preds = []
for c in cells:
    kv_loc = "hicache_same_socket" if (c["kv"] == "hicache" and c["ctx"] == 2000) else "gpu"
    r = M.tpot_ms(c["C"], c["H"], ctx=c["ctx"], deferred_N=4, kv_loc=kv_loc, kv_tokens=KV_TOKENS[c["H"]], routing=routing)
    preds.append(dict(cell=c, deferred_N=4, kv_tokens=KV_TOKENS[c["H"]], interf=kv_loc,
                      pred_tpot_ms=r["tpot_ms"], pred_step_ms=r["step_ms"], pred_prefill_ms=r["prefill_ms_per_out_tok"],
                      pred_B_eff=r["B_eff"], pred_binding=r["binding"], thrash_flag=bool(r["f_retract"] > 0),
                      gpu_expert_ms=r["gpu_expert_ms"], cpu_cold_ms=r["cpu_cold_ms"], cpu_exposed_ms=r["cpu_exposed_ms"]))
model_src = open(os.path.join(HERE, "hybrid_step_model.py"), "rb").read()
out = dict(model_sha256=hashlib.sha256(model_src).hexdigest(), gate="median |err| <= 20% and rank agreement >= 80% on pairs differing >= 10%",
           note="predictions registered BEFORE measurement; thrash_flag cells (B_eff < C) reported separately", predictions=preds)
p = os.path.join(HERE, "..", "predictions.json")
json.dump(out, open(p, "w"), indent=1)
for x in preds:
    c = x["cell"]; print(f"H={c['H']:3d} C={c['C']:2d} ctx={c['ctx']:4d} kv={c['kv']:8s} -> TPOT {x['pred_tpot_ms']:6.1f} ms (step {x['pred_step_ms']:5.1f}, B_eff {x['pred_B_eff']:2d}, {x['pred_binding']}{', THRASH' if x['thrash_flag'] else ''})")
print("saved", p, "model sha256", out["model_sha256"][:12])
