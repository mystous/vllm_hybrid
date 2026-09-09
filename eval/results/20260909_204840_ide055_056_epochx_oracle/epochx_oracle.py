#!/usr/bin/env python3
"""EPOCH-X 문서 1단계 오라클 (IDE_055 EHES / IDE_056 SER).
실측 라우팅 트레이스 + 오늘 측정 상수로, 코드를 쓰기 전에 두 기법의 기전 지표가 움직이는지 판정한다.

측정 상수 (2026-09-09, hot-96 + KT_AVX_RB/PF, 소켓 2개 96스레드 풀):
  expert 1개 CPU 호출 = FIXED + STREAM + ROW*m   (m=1 에서 66 µs, 스트리밍 비중 88%)
  층당 핸드오프 고정비 HANDOFF
usage: epochx_oracle.py <out.json>
"""
import torch, glob, json, sys, statistics as st, math

FIXED, STREAM, ROW = 8.0, 58.0, 9.5      # µs (expert 당)
HANDOFF = 130.0                           # µs (층당)
BYTES_EXPERT = 23.6                       # MB (INT4 gate+up+down, TP 합)
L, E, K = 62, 160, 8
HOT = 96
out_f = sys.argv[1]

fs = sorted(glob.glob('/tmp/expert_distribution_recorder_*.pt'))
d = torch.load(fs[0], map_location='cpu', weights_only=False)
recs = d['records']
passes = [r['global_physical_count'] for r in recs]
toks = [float(c.sum()) / (L * K) for c in passes]
decode = [(c, t) for c, t in zip(passes, toks) if t <= 256]      # 배치 <=256 = decode 패스
prefill = [(c, t) for c, t in zip(passes, toks) if t > 256]
print(f"trace: {len(passes)} 패스 (decode {len(decode)}, prefill {len(prefill)}), 토큰/패스 중앙값 {st.median(toks):.0f}")

def cold_rows(c):
    """층별 cold expert 행 수 리스트 (physical idx >= HOT 가 CPU)."""
    return [c[l][HOT:][c[l][HOT:] > 0].tolist() for l in range(L)]

# ---------------- IDE_056: SER 오라클 ----------------
# 현행: decode 는 lockstep 배치 → 배치 안의 모든 독립 요청 행이 이미 한 read 에 합쳐져 있다.
# SER 가 행을 더 모으려면 (a) 다음 스텝을 기다리거나 (b) cohort 를 쪼개 시차를 둔다.
ser = {}
rows_now = [m for c, t in decode for lst in cold_rows(c) for m in lst]
step_us = []
for c, t in decode:
    for lst in cold_rows(c):
        step_us.append(HANDOFF + sum(FIXED + STREAM + ROW * m for m in lst))
T_step_layer = st.mean(step_us)                    # 층당 CPU µs
T_step = T_step_layer * L / 1000.0                 # 스텝당 CPU ms (층 62개)
ser['now'] = dict(reads=len(rows_now), rows_mean=round(st.mean(rows_now), 3), rows_median=st.median(rows_now),
                  rows_p90=sorted(rows_now)[int(.9 * len(rows_now))], cpu_us_per_layer=round(T_step_layer, 1),
                  cpu_ms_per_step=round(T_step, 2))
# (a) W (µs) 만큼 기다려 다음 스텝 행을 합칠 수 있는가: 다음 스텝은 T_step(ms) 뒤에 온다.
ser['cross_step'] = dict(next_step_arrival_ms=round(T_step, 2),
                         W_max_us_in_doc=500,
                         feasible=bool(T_step * 1000 <= 500))
# (b) cohort skew S: 배치를 S+1 개 cohort 로 쪼개면 같은 expert read 가 여러 번 발생
for S in (0, 1, 2, 4):
    k = S + 1
    rr, us = [], []
    for c, t in decode:
        for lst in cold_rows(c):
            # 행을 k 개 cohort 로 균등 분할 (라우팅은 cohort 간 독립 가정 → 같은 expert 가 k 번 읽힘)
            sub = []
            for m in lst:
                q, r = divmod(int(m), k)
                sub += [q + 1] * r + [q] * (k - r)
            sub = [x for x in sub if x > 0]
            rr += sub
            us.append(HANDOFF * k + sum(FIXED + STREAM + ROW * x for x in sub))
    ser[f'skew_S{S}'] = dict(rows_mean=round(st.mean(rr), 3), reads=len(rr), cpu_us_per_layer=round(st.mean(us), 1))
# (c) 혼합 forward (prefill 행이 decode 와 같은 read 에 합류) — 이미 채택된 IDE_052 의 기전 확인
if prefill and decode:
    rr = []
    for (cd, _), (cp, _) in zip(decode, prefill[:len(decode)]):
        for l in range(L):
            merged = (cd[l][HOT:] + cp[l][HOT:])
            rr += merged[merged > 0].tolist()
    ser['mixed_forward'] = dict(rows_mean=round(st.mean(rr), 3), reads=len(rr))

# ---------------- IDE_055: EHES 오라클 ----------------
# 층별 expert 활성 확률 p_e 와 평균 행 수 E[m_e] 를 트레이스에서 추정.
# CPU 시간 (층당) = Σ_e p_e [ FIXED·1{ρ_e<1} + (STREAM + ROW·m_e)(1-ρ_e) ],  제약 Σ_e ρ_e ≤ M (expert 단위)
act = torch.zeros(L, E); rows = torch.zeros(L, E)
for c, t in decode:
    act += (c > 0).float(); rows += c.float()
n = len(decode)
p = act / n                                  # 스텝당 활성 확률
mbar = torch.where(act > 0, rows / act.clamp(min=1), torch.zeros_like(rows))   # 활성 시 평균 행

def cpu_layer_us(rho):
    """rho: [L,E] (GPU 채널 비율). 층당 CPU µs 기대값."""
    frac = (1.0 - rho)
    fixed = (p * FIXED * (rho < 1.0).float()).sum(1)
    var = (p * (STREAM + ROW * mbar) * frac).sum(1)
    return (fixed + var)

def greedy_full(M):
    """현행: 층별로 기대 트래픽 큰 순으로 M 개 expert 를 GPU 에 전부 배치 (마지막 1개는 분수 허용 안 함)."""
    rho = torch.zeros(L, E)
    score = p * (STREAM + ROW * mbar)
    for l in range(L):
        idx = torch.argsort(score[l], descending=True)[:int(M)]
        rho[l, idx] = 1.0
    return rho

def fractional_opt(M):
    """동일 예산에서 최적 분수 배치: 같은 점수 순서로 채우되 마지막 expert 는 분수."""
    rho = torch.zeros(L, E)
    score = p * (STREAM + ROW * mbar)
    for l in range(L):
        idx = torch.argsort(score[l], descending=True)
        full = int(math.floor(M)); rem = M - full
        rho[l, idx[:full]] = 1.0
        if rem > 0: rho[l, idx[full]] = rem
    return rho

def spread_frac(M, width):
    """비교군: 상위 width 개 expert 에 예산을 균등 분수 배치 (문서가 말하는 '더 많은 expert 에 조금씩')."""
    rho = torch.zeros(L, E)
    score = p * (STREAM + ROW * mbar)
    r = min(1.0, M / width)
    for l in range(L):
        idx = torch.argsort(score[l], descending=True)[:width]
        rho[l, idx] = r
    return rho

ehes = {}
base = cpu_layer_us(greedy_full(HOT))
ehes['greedy_full_96'] = dict(cpu_us_per_layer=round(float(base.mean()), 1),
                              cpu_MB_per_layer=round(float((p * BYTES_EXPERT * (1 - greedy_full(HOT))).sum(1).mean()), 2))
for name, rho in (('fractional_opt_96', fractional_opt(HOT)),
                  ('spread_112x0.857', spread_frac(HOT, 112)),
                  ('spread_128x0.75', spread_frac(HOT, 128)),
                  ('spread_160x0.6', spread_frac(HOT, 160))):
    v = cpu_layer_us(rho)
    ehes[name] = dict(cpu_us_per_layer=round(float(v.mean()), 1),
                      delta_vs_greedy=f"{(float(v.mean())/float(base.mean())-1)*100:+.1f}%",
                      cpu_MB_per_layer=round(float((p * BYTES_EXPERT * (1 - rho)).sum(1).mean()), 2))
# 예산 frontier: 이산(정수 expert) vs 연속 — cliff 가 실제로 있는지
front = []
for M in (80, 84, 88, 92, 96, 100):
    g = float(cpu_layer_us(greedy_full(M)).mean()); f = float(cpu_layer_us(fractional_opt(M + 0.5)).mean())
    front.append(dict(M=M, greedy_us=round(g, 1), frac_us_at_M_plus_half=round(f, 1)))
ehes['budget_frontier'] = front

res = dict(constants=dict(FIXED=FIXED, STREAM=STREAM, ROW=ROW, HANDOFF=HANDOFF, HOT=HOT, L=L, E=E),
           ser=ser, ehes=ehes)
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(out_f, 'w'), ensure_ascii=False, indent=1)
print("saved", out_f)
