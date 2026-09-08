#!/usr/bin/env python3
"""PLN_008 M0 — 하이브리드 MoE decode 스텝 기계 모델 v2 (초안, 2026-09-08).

원칙 (PLN_006 §0 승계): 파라미터는 마이크로벤치·스펙·라우팅 트레이스에서만. end-to-end 처리량으로 맞추지 않는다.
GPU 비-expert 커널 시간은 프로파일의 kt-무관 커널 합 (attention/GEMM/AR/topk 등) 을 그대로 쓴다 (측정치, 적합 아님).

스텝(ms) = Σ_layer [ gpu_nonexpert(B, ctx) + gpu_expert(B, D_h) + cpu_exposed + integ ]
  (구조 근거: 두 프로파일 분해 — hot-80: GPU 26 + CPU 대기 40 = 66 / hot-96: 커널 27.7 + 유휴 9.9 + 복사 1.9 = 39.5 — 에서
   GPU hot-expert 계산과 CPU cold 대기가 겹치지 않고 직렬임이 관측됨. max() 가 아니라 합.)
  cpu_cold = t0 + D_c · t_s · (BW_ref / BW_eff) + rows_c · t_r          (mb480: t_s≈55µs/expert, t_r≈19µs/row)
  cpu_exposed = cpu_cold − hidden;  hidden = min(gpu_nonexpert_nextlayer, cpu_cold · p_def)   (deferred N: p_def ≈ N/8)
  BW_eff = BW_ref · f_interf(KV 위치)                                     (DDR 2-스트림 벤치: 같은 소켓 0.73, 다른 소켓 0.89)
  B_eff = min(C, KV_tokens / ctx)                                          (KV 용량 → 실효 배치)
처리량(tok/s) = B_eff / 스텝
"""
import json, math, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

# ---------------- 모델·하드웨어 상수 (스펙) ----------------
MODEL = dict(L=62, E=160, k=8, hidden=6144, inter=2560, tp=4,
             expert_int4_bytes=3*6144*2560*0.5,            # CPU 측 expert 1개 (INT4) = 23.6 MB
             expert_fp8_shard_bytes=3*6144*2560*1.0/4,     # GPU 측 expert 1개 TP 샤드 (FP8) = 11.8 MB
             kv_bytes_per_token_per_gpu=62*2*8*128*1/4)    # FP8 KV, 8 kv-head, d=128, TP4 → ~63 KB
HW = dict(hbm_bw=3.35e12, ddr_read_per_socket=180e9, sockets=2)

# ---------------- 마이크로벤치 파라미터 ----------------
# (1) CPU 층 호출 법칙 — eval/results/20260908_085300_ide030_mb480_cpu_layer (T=1 평평 구간·T=32 기울기에서)
CPU = dict(t_s_us=55.0, t_r_us=19.0, t0_us=50.0, bw_ref=430e9)
# (2) DDR 2-스트림 간섭 — eval/results/20260908_093500_pln008_m0_ddr
INTERF = {"gpu": 1.0, "hicache_same_socket": 0.73, "hicache_other_socket": 0.89}
# (3) GPU hot expert 커널 g(B, D_h) — eval/results/20260908_115519_pln008_m0_gpu_moe_grid (CUDA graph 재생 타이밍)
GPU_EXPERT = dict(fixed_us=39.7, per_expert_us=4.26, per_token_us=-0.05)   # v2 graph 재생 격자 (16셀) 최소자승, 최대 오차 3%
# (4) GPU 비-expert 커널 (프로파일 kt-무관 커널, hot-96 C≈32 정상 스텝, ms/step): dense_gemm 3.34, attention 1.07, AR 1.35, other 3.76, quant 0.78, norm 0.11, memcpy 1.9
GPU_NONEXPERT = dict(fixed_ms=3.34+1.35+0.78+0.11+1.9, attn_ms_at_B32_ctx640=1.07, other_ms_at_B32=3.76)
INTEG_MS = 0.0  # 통합 고정비는 memcpy 1.9 에 포함 (host 함수 지연은 CPU t0 에 포함)

def load_routing():
    p = os.path.join(ROOT, "eval/results/20260908_094000_pln008_m0_routing/routing_stats.json")
    return json.load(open(p))

def interp_B(table, B):
    ks = sorted(int(k) for k in table)
    if B <= ks[0]: return table[str(ks[0])]
    if B >= ks[-1]: return table[str(ks[-1])]
    for a, b in zip(ks, ks[1:]):
        if a <= B <= b:
            t = (B - a) / (b - a); return table[str(a)] * (1 - t) + table[str(b)] * t

def gpu_expert_ms(B, D_h):
    g = GPU_EXPERT
    if g["per_expert_us"] is None:
        # 자리표시 (격자 벤치 전): 프로파일 실측 hot-96 C≈32 fused_moe 17.33 ms/step, D_h≈61.5 → expert 당 4.55 µs/층
        return (D_h * 17.33 / 61.5 / MODEL["L"])
    return (g["fixed_us"] + D_h * g["per_expert_us"] + B * g["per_token_us"]) / 1e3

def gpu_nonexpert_ms(B, ctx):
    n = GPU_NONEXPERT
    attn = n["attn_ms_at_B32_ctx640"] * (B / 32) * (ctx / 640)
    other = n["other_ms_at_B32"] * (0.5 + 0.5 * B / 32)
    return n["fixed_ms"] + attn + other

def gpu_pre_moe_ms(B, ctx):
    # 프로파일 범주 (hot-96 C≈32): attention 1.07 + dense_gemm 3.34 중 qkv/o 몫 ≈ 0.6 + AR 1.35 의 절반 (attention 뒤 AR)
    n = GPU_NONEXPERT
    attn = n["attn_ms_at_B32_ctx640"] * (B / 32) * (ctx / 640)
    return attn + 3.34 * 0.6 + 1.35 * 0.5

def cpu_cold_ms(B, H, routing, kv_loc="gpu"):
    r = routing["per_H"][str(H)]
    D_c = interp_B(r["E_distinct_cold"], B)
    rows = B * r["cold_pairs_per_token"]
    bw_eff = CPU["bw_ref"] * INTERF[kv_loc]
    per_layer_us = CPU["t0_us"] + D_c * CPU["t_s_us"] * (CPU["bw_ref"] / bw_eff) + rows * CPU["t_r_us"]
    return per_layer_us * MODEL["L"] / 1e3, D_c

def step_ms(C, H, ctx=640, deferred_N=0, kv_loc="gpu", kv_tokens=32768, routing=None):
    routing = routing or load_routing()
    set_workload(ctx)
    # KV 용량 → 실효 배치. sonnet: prefix 100 은 radix 캐시 공유 → 요청당 고유 토큰 ≈ in_unique + out/2 (생성 중 평균)
    B = min(C, int(kv_tokens // max(1, WORKLOAD["in_unique"] + WORKLOAD["out"] // 2)))
    oversub = max(0.0, 1.0 - B / C)                     # 초과분 → retraction (재-prefill) 비율 가정: 절반이 1회 재시작 (M1 에서 검증)
    f_retract = 0.5 * oversub
    r = routing["per_H"][str(H)]
    D_h = interp_B(r["E_distinct_hot"], B)
    g_exp = gpu_expert_ms(B, D_h) * MODEL["L"]
    g_non = gpu_nonexpert_ms(B, ctx)
    cpu, D_c = cpu_cold_ms(B, H, routing, kv_loc)
    # deferred 숨김: 단일 CPU 큐에서 층 L 의 deferred 작업은 층 L+1 의 immediate 작업보다 먼저 실행되므로
    # 겹칠 수 있는 창은 다음 층의 MoE 이전 GPU 구간뿐 (attention + qkv/o GEMM 몫 + AR 절반). p_def = N/8 은 topk 덤프 실측
    # (cold 가 하위 N 에 들 확률 0.13/0.26/0.52 @N=1/2/4 — eval/results/20260908_123000_pln008_m0_topk_dump).
    p_def = min(1.0, deferred_N / MODEL["k"])
    pre_moe_window = gpu_pre_moe_ms(B, ctx)
    hidden = min(pre_moe_window, cpu * p_def)
    cpu_exposed = max(0.0, cpu - hidden)
    total = g_non + g_exp + cpu_exposed + INTEG_MS       # 직렬 (프로파일 분해 근거)
    binding = "gpu_expert" if g_exp >= cpu_exposed else "cpu_ddr"
    return dict(C=C, B_eff=B, f_retract=round(f_retract, 3), H=H, ctx=ctx, N=deferred_N, kv=kv_loc, step_ms=round(total, 1),
                gpu_nonexpert_ms=round(g_non, 1), gpu_expert_ms=round(g_exp, 1), cpu_cold_ms=round(cpu, 1),
                cpu_exposed_ms=round(cpu_exposed, 1), D_h=round(D_h, 1), D_c=round(D_c, 2), binding=binding,
                tok_s=round(B / total * 1e3, 1))

# ---------------- 재예측 대상 (기측정 TPOT ms, 파라미터 미사용) ----------------
# TPOT = decode 스텝 + prefill 혼입 (sonnet 512+100 in / 128 out → 출력 토큰당 prefill 토큰 ≈ 4.8)
# prefill 전용 벤치 (hot-80, out=1): admission 배치 b_p 별 요청당 prefill 시간 (TTFT/b_p). eval/results/*_pln008_m0_prefill_hot80
PREFILL = dict(per_req_ms={1: 322.0, 8: 71.1, 32: 19.3}, bp_rule=lambda C: max(1, C // 3))   # b_p ≈ C/3: 서버 로그 admission 배치 실측 (C32: 평균 9~14, C≤16: 3.4)
def prefill_ms_per_request(b_p):
    ks = sorted(PREFILL["per_req_ms"]); 
    if b_p <= ks[0]: return PREFILL["per_req_ms"][ks[0]]
    if b_p >= ks[-1]: return PREFILL["per_req_ms"][ks[-1]]
    for a, b in zip(ks, ks[1:]):
        if a <= b_p <= b:
            t = (math.log(b_p) - math.log(a)) / (math.log(b) - math.log(a))
            return PREFILL["per_req_ms"][a] * (1 - t) + PREFILL["per_req_ms"][b] * t
WORKLOADS = {640: dict(in_total=612, in_unique=512, out=128),      # sonnet 512 + prefix 100 (공유) / 128
             2000: dict(in_total=1600, in_unique=1500, out=512)}   # 긴 프롬프트 셀 (M1): 1500(+100 prefix) / 512
WORKLOAD = WORKLOADS[640]
def set_workload(ctx):
    global WORKLOAD
    WORKLOAD = WORKLOADS[ctx]

RETRO = [
    dict(name="hot-80 C16 (8-30)",   C=16, H=80, N=0, kv_tokens=32768, tpot=53.31),
    dict(name="hot-80 C32",          C=32, H=80, N=0, kv_tokens=32768, tpot=83.26),
    dict(name="hot-80 C64 (8-30)",   C=64, H=80, N=0, kv_tokens=32768, tpot=134.03),
    dict(name="hot-80+def2 C32",     C=32, H=80, N=2, kv_tokens=32768, tpot=80.13),
    dict(name="hot-80+def4 C32",     C=32, H=80, N=4, kv_tokens=32768, tpot=78.89),
    dict(name="hot-96 C32",          C=32, H=96, N=0, kv_tokens=24576, tpot=54.79),
    dict(name="hot-96+def4 C16",     C=16, H=96, N=4, kv_tokens=24576, tpot=35.94),
    dict(name="hot-96+def4 C32",     C=32, H=96, N=4, kv_tokens=24576, tpot=51.51),
    dict(name="hot-96+def4 C64",     C=64, H=96, N=4, kv_tokens=24576, tpot=86.73),
]
# decode 스텝 직접 실측 (프로파일): hot-96+def4 C≈32 정상 스텝 39.0 ms (커널 27.7 + 복사 1.9 + 유휴 9.9)
PROFILE_STEPS = [dict(name="hot-96+def4 C32 decode step", C=32, H=96, N=4, kv_tokens=24576, step_ms=39.0)]

def tpot_ms(C, H, ctx=640, deferred_N=0, kv_loc="gpu", kv_tokens=32768, routing=None):
    r = step_ms(C, H, ctx, deferred_N, kv_loc, kv_tokens, routing)
    # 스텝당 prefill 정지 = B × (요청당 prefill 시간 at b_p) / 출력 토큰 수, retraction 재-prefill 포함
    b_p = PREFILL["bp_rule"](C)
    pre = r["B_eff"] * prefill_ms_per_request(b_p) / WORKLOAD["out"] * (1.0 + r["f_retract"])
    r["prefill_ms_per_out_tok"] = round(pre, 1); r["tpot_ms"] = round(r["step_ms"] + pre, 1)
    return r

if __name__ == "__main__":
    routing = load_routing()
    print("== decode 스텝 재예측 (프로파일 실측) ==")
    for t in PROFILE_STEPS:
        p = step_ms(t["C"], t["H"], deferred_N=t["N"], kv_tokens=t["kv_tokens"], routing=routing)
        print(f"  {t['name']}: pred {p['step_ms']} ms vs meas {t['step_ms']} ms ({(p['step_ms']-t['step_ms'])/t['step_ms']*100:+.1f}%)  [gNon {p['gpu_nonexpert_ms']} gExp {p['gpu_expert_ms']} cpu {p['cpu_cold_ms']} exposed {p['cpu_exposed_ms']} {p['binding']}]")
    print("== TPOT 재예측 (prefill 항 포함: b_p=C/4 가정) ==")
    errs = []
    print(f"{'case':22s} {'pred':>7s} {'meas':>7s} {'err%':>6s}  step  pre  gNon  gExp  cpu  cpuExp  Dh   Dc  binding")
    for t in RETRO:
        p = tpot_ms(t["C"], t["H"], deferred_N=t["N"], kv_tokens=t["kv_tokens"], routing=routing)
        e = (p["tpot_ms"] - t["tpot"]) / t["tpot"] * 100; errs.append(abs(e))
        print(f"{t['name']:22s} {p['tpot_ms']:7.1f} {t['tpot']:7.1f} {e:6.1f}  {p['step_ms']:5.1f} {p['prefill_ms_per_out_tok']:4.1f} {p['gpu_nonexpert_ms']:5.1f} {p['gpu_expert_ms']:5.1f} {p['cpu_cold_ms']:5.1f} {p['cpu_exposed_ms']:5.1f} {p['D_h']:5.1f} {p['D_c']:4.1f} {p['binding']}")
    errs.sort(); print(f"median |err| = {errs[len(errs)//2]:.1f}%  max = {errs[-1]:.1f}%")
