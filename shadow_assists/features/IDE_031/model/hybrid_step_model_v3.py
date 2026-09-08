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
    D_c = _interp(r["E_Dc_decode"], B); D_h = _interp(r["E_Dh_decode"], B)
    cpu_layer = (CPU3["T_W_us"] + CPU3["T_0_us"] + CPU3["T_E_us"] * D_c) / 1e3
    g_non_total = V2.gpu_nonexpert_ms(B, ctx); fixed = V2.GPU_NONEXPERT["fixed_ms"]
    gpu_layer = V2.gpu_expert_ms(B, D_h) + (g_non_total - fixed) / L
    layer = max(cpu_layer, gpu_layer) + T_HANDOFF_US / 1e3
    step = L * layer + fixed
    return dict(C=C, B_eff=B, H=H, hotmap=hotmap, ctx=ctx, D_c=round(D_c, 2), D_h=round(D_h, 1),
                cpu_layer_ms=round(cpu_layer, 3), gpu_layer_ms=round(gpu_layer, 3), step_ms=round(step, 1),
                binding="cpu_ddr" if cpu_layer >= gpu_layer else "gpu")

def tpot_ms(C, H, hotmap="prompt", ctx=640, kv_tokens=49152):
    r = step_ms(C, H, hotmap, ctx, kv_tokens); W = V2.WORKLOAD
    b_p = V2.PREFILL["bp_rule"](C)
    frac = routing(hotmap, H)["cold_pairs_per_token_prefill"] / 8.0
    coef = (PREFILL_COLD["a_ms"] + PREFILL_COLD["b_ms_per_frac"] * frac) / (PREFILL_COLD["a_ms"] + PREFILL_COLD["b_ms_per_frac"] * PREFILL_COLD["ref_frac"])
    pre = r["B_eff"] * V2.prefill_ms_per_request(b_p) * coef / W["out"]
    r["prefill_ms_per_out_tok"] = round(pre, 1); r["prefill_coef"] = round(coef, 3)
    r["tpot_ms"] = round(r["step_ms"] + pre, 1); r["tok_s"] = round(r["B_eff"] / r["tpot_ms"] * 1e3, 1)
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
        print(f"{t['name']:26s} {p['tpot_ms']:8.1f} {t['tpot']:6.1f} {(p['tpot_ms']-t['tpot'])/t['tpot']*100:+6.1f} {p['tok_s']:8.1f} {t['tok_s']:6.1f} | {p['D_c']:5.2f} {p['cpu_layer_ms']:6.3f} {p['gpu_layer_ms']:6.3f} {p['binding']}")
