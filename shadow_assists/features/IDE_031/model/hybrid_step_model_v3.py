#!/usr/bin/env python3
"""PLN_008 기계 모델 v3 — 새 실행 메커니즘 (IDE_033: callback-free + 빈 immediate 생략 + HT 형제 회피, N=k 전량 deferral; IDE_034: phase-가중 hot set) 용.

구조 (계측 근거: TaskQueue 작업별 시간 → 큐 99% 바쁨, 단일 FIFO → sync(L) 은 def(L-1) 완료 대기; 프로파일 갭 → GPU 는 층마다 CPU 완료를 기다림):
  층 시간 = max( GPU_layer(B, D_h), CPU_layer(B, D_c) ) + t_handoff
  스텝(ms) = L · 층 시간 + fixed_nonlayer
  CPU_layer(µs) = T_W + T_0 + T_E · E[D_c]          (in-situ AMX phase 분해: 40 + 80 + 75·D_c, B=32·64 에서 기울기 74~78 — 행 항 무시)
  GPU_layer = gpu_expert(B, D_h) + (gpu_nonexpert(B, ctx) − fixed)/L      (v2 항 그대로)
  E[D_c], E[D_h] = decode 라우팅 트레이스 (routing_phase_stats.json, hotmap 별 · H 별 · B 별), B 보간
  TPOT = 스텝 + prefill 몫;  prefill 몫 = v2 법칙 × prefill cold-pair 계수 (TTFT 3점: 1100 + 28000·cold_frac, hot-96 prompt 기준 정규화)
파라미터 출처: v2 (GPU 격자·프로파일), in-situ 계측 (phase, [kt-wrap]), 라우팅 트레이스. TPOT 에 대한 적합 없음.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hybrid_step_model as V2
L = V2.MODEL["L"]
CPU3 = dict(T_W_us=40.0, T_0_us=80.0, T_E_us=75.0)          # [kt-wrap] 20+21 ≈ 40; phase 절편 70~84; 기울기 74~78
DC_RATIO = 0.88                                              # in-situ 층당 활성 cold expert 평균 / 트레이스 기대값 (hot-96 6.85/7.81=0.88, hot-80 12.7/14.5=0.876; expert 독립 가정 보정)
FIXED_NONLAYER_MS = 2.0                                      # 가정: CPU 가 임계 경로 (큐 99% 바쁨) 이라 GPU 고정 작업은 겹침; 스텝 경계 샘플링·스케줄러 갭만 남음 (프로파일 미확보 — 셀로 검증)
TTFT_WAVE_FACTOR = 2.95
PREFILL3 = dict(per_req_ms={1: 268.8, 8: 73.8, 32: 17.0}, ref_frac=0.019)   # 새 메커니즘 α=0.25 hot-96, 입력 512/출력 1 실측 (09-09 02:45), b_p = C (고정 출력 길이 워크로드는 파도식 admission)
T_HANDOFF_US = 20.0                                          # done 플래그 memop 대기 + 폴러 반응 (마이크로벤치 memop 9µs + 폴러 스핀)
PREFILL_COLD = dict(a_ms=1100.0, b_ms_per_frac=28000.0, ref_frac=0.0083)   # TTFT(prompt 1304 / α.25 1629 / decode-only 2560) vs cold frac (.0083/.019/.0522)
STATS = json.load(open(os.path.join(HERE, "..", "routing_phase_stats.json")))

def _interp(table, B):
    ks = sorted(int(k) for k in table); 
    if B <= ks[0]: return table[str(ks[0])] if str(ks[0]) in table else table[ks[0]]
    for a, b in zip(ks, ks[1:]):
        if a <= B <= b:
            va = table[str(a)] if str(a) in table else table[a]; vb = table[str(b)] if str(b) in table else table[b]
            return va + (vb - va) * (B - a) / (b - a)
    return table[str(ks[-1])] if str(ks[-1]) in table else table[ks[-1]]

def routing(hotmap, H):
    return STATS["maps"][hotmap][str(H)]

def step_ms(C, H, hotmap="prompt", ctx=640, kv_tokens=49152):
    V2.set_workload(ctx); W = V2.WORKLOAD
    B = min(C, max(1, kv_tokens // (W["in_unique"] + W["out"] // 2)))
    r = routing(hotmap, H)
    D_c = DC_RATIO * _interp(r["E_Dc_decode"], B); D_h = _interp(r["E_Dh_decode"], B)
    cpu_layer = (CPU3["T_W_us"] + CPU3["T_0_us"] + CPU3["T_E_us"] * D_c) / 1e3
    g_non_total = V2.gpu_nonexpert_ms(B, ctx); fixed = V2.GPU_NONEXPERT["fixed_ms"]
    gpu_layer = V2.gpu_expert_ms(B, D_h) + (g_non_total - fixed) / L
    layer = max(cpu_layer, gpu_layer) + T_HANDOFF_US / 1e3
    step = L * layer + FIXED_NONLAYER_MS
    return dict(C=C, B_eff=B, H=H, hotmap=hotmap, ctx=ctx, D_c=round(D_c, 2), D_h=round(D_h, 1),
                cpu_layer_ms=round(cpu_layer, 3), gpu_layer_ms=round(gpu_layer, 3), step_ms=round(step, 1),
                binding="cpu_ddr" if cpu_layer >= gpu_layer else "gpu")

def tpot_ms(C, H, hotmap="prompt", ctx=640, kv_tokens=49152):
    r = step_ms(C, H, hotmap, ctx, kv_tokens); W = V2.WORKLOAD
    b_p = min(C, 32)
    frac = routing(hotmap, H)["cold_pairs_per_token_prefill"] / 8.0
    coef = (PREFILL_COLD["a_ms"] + PREFILL_COLD["b_ms_per_frac"] * frac) / (PREFILL_COLD["a_ms"] + PREFILL_COLD["b_ms_per_frac"] * PREFILL3["ref_frac"])
    pre = r["B_eff"] * _interp({str(k): v for k, v in PREFILL3["per_req_ms"].items()}, b_p) * coef / W["out"]
    r["prefill_ms_per_out_tok"] = round(pre, 1); r["prefill_coef"] = round(coef, 3)
    r["tpot_ms"] = round(r["step_ms"] + pre, 1)
    # TTFT ≈ TTFT_WAVE_FACTOR × (파도 prefill 시간 = b_p × 요청당 prefill(b_p)) × cold 계수 — 3점 (prompt 1304 / α.25 1629 / decode-only 2560) 에서 인자 2.9~3.0
    ttft = TTFT_WAVE_FACTOR * b_p * _interp({str(k): v for k, v in PREFILL3["per_req_ms"].items()}, b_p) * coef
    r["ttft_ms"] = round(ttft, 0)
    r["tok_s"] = round(r["B_eff"] * W["out"] / (ttft + (W["out"] - 1) * r["tpot_ms"]) * 1e3, 1)   # 요청 수명 = TTFT + (out−1)·TPOT
    return r

# 기측정 (파라미터 미사용) 대조: 새 메커니즘 4셀
RETRO = [dict(name="prompt H96 C32", C=32, H=96, hotmap="prompt", tpot=43.5, tok_s=600.0),
         dict(name="prompt H96 C64", C=64, H=96, hotmap="prompt", tpot=68.0, tok_s=773.0),
         dict(name="alpha.25 H96 C32", C=32, H=96, hotmap="alpha0.25", tpot=35.9, tok_s=661.4),
         dict(name="alpha.25 H96 C64", C=64, H=96, hotmap="alpha0.25", tpot=54.2, tok_s=883.7),
         dict(name="prompt H80 C32 (KV131K)", C=32, H=80, hotmap="prompt", tpot=71.0, tok_s=408.0, kv=131072),
         dict(name="prompt H80 C64 (KV131K)", C=64, H=80, hotmap="prompt", tpot=106.4, tok_s=561.0, kv=131072)]
if __name__ == "__main__":
    print(f"{'cell':26s} {'predTPOT':>8s} {'meas':>6s} {'err%':>6s} {'predTok':>8s} {'meas':>6s} | {'Dc':>5s} {'cpuL':>6s} {'gpuL':>6s} bind")
    for t in RETRO:
        p = tpot_ms(t["C"], t["H"], t["hotmap"], kv_tokens=t.get("kv", 49152))
        print(f"{t['name']:26s} {p['tpot_ms']:8.1f} {t['tpot']:6.1f} {(p['tpot_ms']-t['tpot'])/t['tpot']*100:+6.1f} {p['tok_s']:8.1f} {t['tok_s']:6.1f} ({(p['tok_s']-t['tok_s'])/t['tok_s']*100:+5.1f}%) | {p['D_c']:5.2f} {p['cpu_layer_ms']:6.3f} {p['gpu_layer_ms']:6.3f} {p['binding']} ttft {p['ttft_ms']:.0f}")
    # 사전 등록: 미측정 12셀
    import hashlib
    cells = [(96,"prompt",8),(96,"prompt",16),(96,"alpha0.25",8),(96,"alpha0.25",16)] + [(80,hm,C) for hm in ("prompt","alpha0.25") for C in (8,16,32,64)]
    preds = []
    for H, hm, C in cells:
        kv = 49152 if H == 96 else 131072
        r = tpot_ms(C, H, hm, kv_tokens=kv); preds.append(dict(cell=dict(H=H, hotmap=hm, C=C, ctx=640, kv_tokens=kv), pred_tpot_ms=r["tpot_ms"], pred_tok_s=r["tok_s"], pred_ttft_ms=r["ttft_ms"], pred_step_ms=r["step_ms"], pred_D_c=r["D_c"], pred_binding=r["binding"]))
    out = dict(model="v3 (새 메커니즘: CF+skip+pin, N=8; 층당 max(GPU,CPU); in-situ CPU 법칙; phase 라우팅 통계)", model_sha256=hashlib.sha256(open(__file__,"rb").read()).hexdigest(),
               gate="TPOT 중앙값 |오차| ≤20% AND 순위 일치 ≥80% (tok/s 는 보조)", retro="기측정 6셀 TPOT +0.6~+6.0%", predictions=preds)
    json.dump(out, open(os.path.join(HERE, "..", "predictions_newmech.json"), "w"), indent=1, ensure_ascii=False)
    print("\n사전 등록 12셀:")
    for p in preds: print(f"  H{p['cell']['H']} {p['cell']['hotmap']:9s} C{p['cell']['C']:2d}: TPOT {p['pred_tpot_ms']:6.1f} tok/s {p['pred_tok_s']:6.1f} TTFT {p['pred_ttft_ms']:5.0f} D_c {p['pred_D_c']:5.2f} {p['pred_binding']}")
