#!/usr/bin/env python3
"""PLN_006 — placement bound calculator v3 (동결판).

v2 → v3: P0 측정 반영.
  - per-config 비용 법칙 (a, BW, C) — 마이크로벤치 3구성 최소자승 (microbench-only fit)
  - 라우팅 집중: P0a 실측 U(C=16)=51.1 (480B, sonnet C16 decode), per-layer top-x share
  - 기준 지표: decode 정상상태율 = C / TPOT (output tput 은 prefill 혼입으로 2차 지표)
동결 후 E1 예측까지 수정 금지.
"""
import json, math

# ---- 마이크로벤치 적합 법칙: t_call(us) = a + bytes/BW + flops/C  (low-batch regime, T<128) ----
LAWS = {  # (threads, pools): (a_us, BW_GB/s, C_TF)
    (96,2): (266.4, 400.8, 23.4),
    (48,1): (377.4, 214.9, 16.9),
    (24,1): (604.5, 136.4, 12.2),
}
# 미측정 구성 (48,2) = 24thr-pool × 2 합성: pool 당 절반 부하 + straggler
def t_layer_us(threads, pools, U, T, w_e, f_e_tok):
    imb = 1.0 + math.sqrt(2.0/(math.pi*max(T*8,1))) if pools==2 else 1.0
    if (threads,pools) in LAWS:
        a,BW,C = LAWS[(threads,pools)]
        if pools==2:  # 법칙이 2pool 전체 측정치면 imb 이미 내재 → 중복 적용 안 함
            imb = 1.0
        return (a + U*w_e/(BW*1e9)*1e6 + T*f_e_tok/(C*1e12)*1e6) * imb
    if (threads,pools)==(48,2):
        a,BW,C = LAWS[(24,1)]
        return (a + (U/2)*w_e/(BW*1e9)*1e6 + (T/2)*f_e_tok/(C*1e12)*1e6) * imb
    raise KeyError((threads,pools))

MODELS = dict(
    q480=dict(L=62, E=160, k=8, w_e=6144*2560*3*0.5, f_e_tok=8*2*3*6144*2560,
              attn_w=10.1e9, U16=51.1,            # P0a 실측 (C=16 decode)
              topx_share={0:0.0, 8:0.4262, 16:0.6008, 32:0.7907}),  # per-layer, P0a
    q30 =dict(L=48, E=128, k=8, w_e=2048*768*3*0.5, f_e_tok=8*2*3*2048*768,
              attn_w=2.5e9, U16=None, topx_share=None),
)
BW_HBM=3.35e12; GPU_EFF=0.6; LAYER_OVH=40e-6

def U_of(m, C, rho_mode='measured'):
    unif = m['E']*(1-(1-m['k']/m['E'])**C)
    if rho_mode=='uniform': return unif
    unif16 = m['E']*(1-(1-m['k']/m['E'])**16)
    if m['U16'] is not None: return unif*(m['U16']/unif16)
    return unif*0.570                            # q480like: 480B 실측 ρ(0.57) 이식 (30B ρ 미측정 근사)

def decode_rate(model, C, threads, pools, tp, x_gpu=0, ctx=600, rho_mode='measured'):
    m = MODELS[model]
    U = U_of(m, C, rho_mode)
    s = m['topx_share'].get(x_gpu,0.0) if m['topx_share'] else 0.0
    U_cpu = max(U - x_gpu*0.95, 1.0) if x_gpu else U   # hot-x 는 거의 항상 활성
    f_cpu = m['f_e_tok']*(1-s)
    t_cpu = t_layer_us(threads,pools,U_cpu,C,m['w_e'],f_cpu)*1e-6
    kv = 2*8*128*ctx*1.0*C                              # per layer FP8
    t_gpu = (m['attn_w']/m['L'] + kv)/(BW_HBM*tp*GPU_EFF) + LAYER_OVH
    t_gpu += x_gpu*(m['w_e']*2)* (s and 1 or 1)/(BW_HBM*tp*GPU_EFF)*min(C/16,1)  # GPU expert 읽기 (FP8=2×INT4 bytes)
    step = (t_cpu + t_gpu)*m['L']
    return C/step

if __name__=='__main__':
    P={}
    # P1: 480B TP4 C=16, decode rate (=C/TPOT 예측). (96,2) 는 기측정 앵커 (예측 아님)
    P['P1'] = {f"t{t}_p{p}": round(decode_rate('q480',16,t,p,4),1) for t,p in ((96,2),(48,1),(48,2),(24,1))}
    # P2: x∈{16,32} (thr96/2pool)
    P['P2'] = {f"x{x}": round(decode_rate('q480',16,96,2,4,x_gpu=x),1) for x in (16,32)}
    # P3: 30B TP1 thr48/1pool, C sweep — ρ 미측정이므로 bracket [uniform, 480B-ρ 이식]
    P['P3'] = {f"C{C}": [round(decode_rate('q30',C,48,1,1,rho_mode='uniform'),1),
                          round(decode_rate('q30',C,48,1,1,rho_mode='q480like'),1)] for C in (1,4,16,64)}
    # P3 q480like: U16 이식
    # P4: 480B×4 TP2 thr24/1pool, C=4/8 each → 합산
    P['P4'] = {f"C{C}each_total": round(4*decode_rate('q480',C,24,1,2),1) for C in (4,8)}
    P['_note'] = dict(
        metric='decode steady rate = C/TPOT_median (output tput 는 2차)',
        anchors_known=dict(t96_p2_measured_K3='16/0.292=54.8 tok/s (K3 S_C16 TPOT)'),
        limits=['U(C) shape 스케일 근사', '30B ρ 미측정 → bracket', 'P2 hot-x 정적 배치 가정', '(48,2) 합성 법칙'])
    print(json.dumps(P, indent=1, ensure_ascii=False))
