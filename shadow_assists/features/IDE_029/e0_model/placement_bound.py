#!/usr/bin/env python3
"""PLN_006 E0b — placement bound calculator (v1).

원칙: 파라미터는 스펙시트 + 마이크로벤치에서만 (end-to-end 수치 캘리브레이션 금지).
입력 출처:
  - STREAM triad (2026-08-30 실측): socket-local 195 GB/s, remote 112, interleave96 276
  - PCIe (실측): H2D 51.9 / D2H 55.1 GB/s
  - AMX per-token cost curve (E1 microbench, 30B expert h=2048 m=768 k=8, INT4, 96thr/2pool)
  - H100 spec: HBM 3.35 TB/s, FP8 dense ~1979 TFLOPS (uses ~50% eff w/o cudagraph, decode)
  - 모델 구조: config.json (자동 추출값 하드코드)
"""
import json, math, sys

HW = dict(
    bw_dram_local=195e9, bw_dram_remote=112e9, bw_dram_interleave96=276e9,
    bw_pcie_h2d=51.9e9, bw_pcie_d2h=55.1e9,
    bw_hbm=3.35e12,            # per GPU
    n_gpu_total=8,
    # E1 microbench INT4 conc curve: n_e(tokens/expert) -> us per token per LAYER expert call
    amx_curve_30b={1:157.1, 8:39.4, 32:21.7, 64:18.5, 128:4.9, 256:3.9, 512:3.6, 1024:3.8},
)

MODELS = dict(
    q480=dict(L=62, h=6144, m_exp=2560, E=160, k=8, n_kv=8, hd=128, n_q=96,
              w_dtype=1.0,      # FP8 attn weights (GPU)
              exp_int4_bytes=6144*2560*3*0.5,   # 23.6MB / expert / layer
              attn_w_bytes=(6144*96*128 + 6144*8*128*2 + 96*128*6144)*1.0),
    q30=dict(L=48, h=2048, m_exp=768, E=128, k=8, n_kv=4, hd=128, n_q=32,
             w_dtype=1.0,
             exp_int4_bytes=2048*768*3*0.5,     # 2.36MB
             attn_w_bytes=(2048*32*128 + 2048*4*128*2 + 32*128*2048)*1.0),
)

def amx_us_per_tok(n_e, scale_flops):
    """E1 곡선 보간 (log-log) 후 expert 크기비로 스케일. 곡선은 30B expert 기준."""
    c = HW['amx_curve_30b']; ks = sorted(c)
    n_e = max(min(n_e, ks[-1]), ks[0])
    for a, b in zip(ks, ks[1:]):
        if a <= n_e <= b:
            f = (math.log(n_e)-math.log(a))/(math.log(b)-math.log(a))
            v = math.exp(math.log(c[a])*(1-f)+math.log(c[b])*f)
            return v*scale_flops
    return c[ks[-1]]*scale_flops

def imbalance(total_expert_draws, pools):
    """bulk-sync expert dispatch 에서 pool 간 straggler 계수 (이항 분산, 파라미터 프리).
    pools=1 → 1.0. pools=2 → E[max(X, n-X)]/(n/2) ≈ 1 + sqrt(2/(pi*n))."""
    if pools == 1: return 1.0
    n = max(total_expert_draws, 1)
    return 1.0 + math.sqrt(2.0/(math.pi*n))

def unique_experts(C, E, k):
    """균등 라우팅 시 layer 당 활성 expert 수 기대값 (upper bracket: 집중 라우팅이면 더 작음)"""
    return E*(1.0-(1.0-k/E)**C)

KT_EFF = 0.35     # kt kernel BW 효율 vs STREAM (E1: BW_eff 70~136 vs interleave 276 -> 0.25~0.5, 기하평균)
C_EFF_96T = 21e12 # AMX 유효 연산율 (E1 plateau, 96thr)
CALL_OVH = 20e-6  # per-layer expert call 고정비 (E1 n_e=1 분해: 157us - 139us bandwidth)
RHO = 1.0         # 라우팅 집중 계수 (1.0=균등 worst case; P0 라우팅 트레이스로 측정 예정 — fitting 금지)

def step_time(model, C, n_gpu, cpu_pools, ctx=600, mode='hybrid', threads=96, rho=None):
    m = MODELS[model]
    # --- GPU part per decode step ---
    kv_bytes = 2*m['n_kv']*m['hd']*ctx*1.0*m['L']            # per token, FP8 KV
    attn_w = m['attn_w_bytes']*m['L']                        # read once per step
    gpu_bytes = attn_w + kv_bytes*C
    t_gpu = gpu_bytes/(HW['bw_hbm']*n_gpu*0.6)               # 0.6 = decode eff w/o cudagraph (spec-side derate)
    t_gpu += m['L']*40e-6                                    # per-layer launch/collective overhead (no cudagraph, TP)
    if mode == 'gpu_only':
        exp_w = unique_experts(C, m['E'], m['k'])*m['exp_int4_bytes']*2*m['L']  # FP8 experts resident HBM
        t_gpu += exp_w/(HW['bw_hbm']*n_gpu*0.6)
        return t_gpu
    # --- CPU expert part (roofline v2) ---
    r = RHO if rho is None else rho
    U = unique_experts(C, m['E'], m['k'])*r
    bw = HW['bw_dram_local']*cpu_pools*KT_EFF
    c_eff = C_EFF_96T*threads/96.0
    flops = C*m['k']*2*3*m['h']*m['m_exp']
    t_cpu_layer = max(U*m['exp_int4_bytes']/bw, flops/c_eff) + CALL_OVH
    t_cpu = t_cpu_layer*m['L']*imbalance(C*m['k'], cpu_pools)
    # activation PCIe (tiny, but include)
    act = 2*m['h']*C*2*m['L']                                # h2d+d2h bf16
    t_pcie = act/HW['bw_pcie_h2d']
    return t_gpu + t_cpu + t_pcie                            # 순차 가정 (kt 는 layer 내 동기)

def tput(model, C, n_gpu, pools, **kw):
    return C/step_time(model, C, n_gpu, pools, **kw)

def bracket(model, C, n_gpu, pools, threads):
    return (round(tput(model,C,n_gpu,pools,threads=threads,rho=1.0),1),
            round(tput(model,C,n_gpu,pools,threads=threads,rho=8.0/unique_experts(C,MODELS[model]['E'],MODELS[model]['k'])),1))

def lower_bound(model, C, ctx=600):
    """정책 무관 하한: 활성 expert 가중치는 저장소에서 최소 1회/step 읽혀야 한다.
    최선의 저장소·경로 (DRAM 양소켓 로컬 병렬) 를 가정한 낙관 하한."""
    m = MODELS[model]
    U = unique_experts(C, m['E'], m['k'])
    bytes_exp = U*m['exp_int4_bytes']*m['L']
    t = bytes_exp/(2*HW['bw_dram_local'])                    # 390 GB/s 상한
    return C/t

if __name__ == '__main__':
    out = {}
    # 재예측 1: 30B GPU-only vs hybrid @ dev-류 부하 (C=64 기준, 8-27 실측은 32/32)
    g30 = tput('q30', 32, 1, 1, mode='gpu_only'); h30 = tput('q30', 32, 1, 1)
    out['r1_30b'] = dict(gpu_only=round(g30,1), hybrid=round(h30,1), ratio=round(g30/h30,1), measured_ratio=15)
    # 재예측 2: 480B S (TP4+96thr/2pool) C=16
    out['r2_480b_S_C16'] = dict(bracket_uniform_to_concentrated=bracket('q480',16,4,2,96), measured=44.5)
    # 재예측 3: 480B D 인스턴스 (TP4+48thr/1pool) C=16
    out['r3_480b_D_C16'] = dict(bracket_uniform_to_concentrated=bracket('q480',16,4,1,48), measured=30.6)
    # 재예측 4: knee 위치 — E1 곡선 자체가 입력이므로 구성상 재현 (평탄화 n_e 128~256)
    out['r4_knee'] = dict(note='curve is input; plateau at n_e 128-256 == measured 75-129 knee band (±50% gate)')
    # 재예측 5: K3 곡선 부호 (S vs 2×D at matched load)
    k3 = {}
    for total in (4,16,32,64):
        s = tput('q480', total, 4, 2, threads=96)
        d = 2*tput('q480', total//2, 4, 1, threads=48)
        k3[total] = dict(S=round(s,1), D=round(d,1), D_win=bool(d>s), gap_pct=round((d/s-1)*100,1))
    out['r5_k3'] = {'pred': k3, 'measured_gap_pct': {4:9.7, 16:11.7, 32:3.5, 64:2.2}}
    # 재예측 6: 동거 무간섭 (kt hybrid + GPU-only 70B) — 자원 교집합 없음 → 구조상 0 간섭
    out['r6_coexist'] = dict(note='disjoint resources (GPU sets, sockets) -> 0 interference structurally; measured ~0')
    # 하한과 gap
    lb = lower_bound('q480', 16)
    out['bound_480b_C16'] = dict(lower_bound_tps=round(lb,1), S_measured=44.5,
                                 note='uniform-routing bound; measured>bound means routing concentration (측정 P0 필요)' if 44.5>lb else f'gap: measured/bound={44.5/lb:.0%}')
    print(json.dumps(out, indent=1, ensure_ascii=False))
