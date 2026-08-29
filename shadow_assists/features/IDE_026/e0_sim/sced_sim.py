#!/usr/bin/env python3
"""PLN_004 E0 — SCED 비용모델 시뮬레이터.

모델 (PLN_004 §1):
  CPU-측 MoE layer 시간 (roofline, 집계 근사):
    D(T) = E * (1 - (1 - k/E)^T)                # 기대 touched expert 수
    t_mem  = (D(T) * W_e + n_sh * W_e) / BW     # weight 1회 읽기 (shared 포함)
    t_comp = 6*h*m*T*(k + n_sh) / C             # 3 GEMM * 2 FLOP
    t_layer = max(t_mem, t_comp)
  스텝 시간: L * t_layer + t_other  (t_other = attention/dense/GPU/框架 잔여, 캘리브레이션)
  수락 모델 (iid token-level α): E[gen/req/step] = (1 - α^(K+1)) / (1 - α)
  goodput = M * E[gen] / t_step
  GPU-only spec arm: t_step = t0 + beta * M*(K+1)  (검증 토큰 선형)

캘리브레이션: 2026-08-27 실측 decode TPOT 로 BW_eff 역산 (t_other 민감도 스윕).
출력: 예측표 (markdown + json) — 실측 전 고정 (사전 등록).
"""
import json
import math
import sys

GB = 1e9
MS = 1e-3

MODELS = {
    "qwen3-30b-a3b-int8": dict(L=48, E=128, k=8, n_sh=0, h=2048, m=768, bytes_per=1.0,
                               calib_M=16, calib_step_s=0.1577),   # 8-27 t2 TPOT p50
    "r1-0528-int4": dict(L=58, E=256, k=8, n_sh=1, h=7168, m=2048, bytes_per=0.5,
                         calib_M=8, calib_step_s=0.3055),          # 8-27 r1 TPOT p50
}

C_CPU = 70e12          # AMX 지속 (turbo off 하한 추정, 민감도에서 스윕)
DRAFT_MS_PER_TOK = 0.35  # 0.6B draft, GPU (모델별 대동소이 가정)


def w_expert(p):
    return 3 * p["h"] * p["m"] * p["bytes_per"]


def distinct(T, p):
    return p["E"] * (1 - (1 - p["k"] / p["E"]) ** T)


def cpu_layer_time(T, p, BW, C=C_CPU):
    We = w_expert(p)
    t_mem = (distinct(T, p) * We + p["n_sh"] * We) / BW
    t_comp = 6 * p["h"] * p["m"] * T * (p["k"] + p["n_sh"]) / C
    return max(t_mem, t_comp), t_mem, t_comp


def calibrate_bw(p, t_other):
    """실측 decode step 시간에서 BW_eff 역산 (K=0)."""
    T = p["calib_M"]
    residual = p["calib_step_s"] - t_other
    if residual <= 0:
        return None
    # memory-bound 가정으로 역산 후 검증
    We = w_expert(p)
    bw = p["L"] * (distinct(T, p) * We + p["n_sh"] * We) / residual
    t, tm, tc = cpu_layer_time(T, p, bw)
    assert tm >= tc * 0.5, "calibration point not memory-bound?"
    return bw


def expected_gen(alpha, K):
    if alpha >= 1.0:
        return K + 1
    if alpha <= 0:
        return 1.0
    return (1 - alpha ** (K + 1)) / (1 - alpha)


def hybrid_goodput(M, K, alpha, p, BW, t_other):
    T = M * (K + 1)
    t_layer, tm, tc = cpu_layer_time(T, p, BW)
    t_draft = DRAFT_MS_PER_TOK * MS * K          # per request chain, 병렬 배치 가정 → per step
    t_step = p["L"] * t_layer + t_other + t_draft
    return M * expected_gen(alpha, K) / t_step, dict(
        T=T, distinct=round(distinct(T, p), 1), t_layer_ms=t_layer / MS,
        mem_bound=tm >= tc, t_step_ms=t_step / MS)


def gpu_goodput(M, K, alpha, t0, beta):
    t_step = t0 + beta * M * (K + 1) + DRAFT_MS_PER_TOK * MS * K
    return M * expected_gen(alpha, K) / t_step


def kstar(fn, Ks):
    vals = {K: fn(K) for K in Ks}
    best = max(vals, key=lambda K: vals[K])
    return best, vals


def main():
    Ks = list(range(0, 17))
    alphas = [0.01, 0.2, 0.4, 0.6, 0.8]
    Ms = [8, 16, 32, 64]
    t_other_sweep = [0.0, 0.02, 0.04]  # 캘리브레이션 민감도 (s)

    out = {"config": dict(C_CPU=C_CPU, draft_ms=DRAFT_MS_PER_TOK), "models": {}}
    lines = ["# E0 — SCED 사전 예측표 (사전 등록, 실측 전 고정)", ""]

    for name, p in MODELS.items():
        lines.append(f"## {name}")
        mrec = {"calib": {}, "pred": []}
        for t_other in t_other_sweep:
            bw = calibrate_bw(p, t_other)
            if bw is None:
                continue
            mrec["calib"][str(t_other)] = bw / GB
            lines.append(f"- t_other={t_other*1e3:.0f}ms 가정 → **BW_eff = {bw/GB:.0f} GB/s** "
                         f"(이론 614 대비 {bw/GB/614*100:.0f}%)")
        # 대표 캘리브레이션: t_other=20ms
        bw = calibrate_bw(p, 0.02)
        # GPU-only arm 파라미터 (Qwen 기준 근사: t0=9.4ms, beta=16.5us — R1 은 GPU-only 불가라 arm 없음)
        gpu_arm = name.startswith("qwen")
        t0, beta = 9.4e-3, 16.5e-6

        lines.append("")
        lines.append("| M | α | K*_hybrid | goodput@K* (tok/s) | vs K=0 (배) | K*_gpu | Δ(K*_hyb−K*_gpu) |")
        lines.append("|---|---|---|---|---|---|---|")
        for M in Ms:
            for alpha in alphas:
                kh, hv = kstar(lambda K: hybrid_goodput(M, K, alpha, p, bw, 0.02)[0], Ks)
                g0 = hv[0]
                gk = hv[kh]
                if gpu_arm:
                    kg, gv = kstar(lambda K: gpu_goodput(M, K, alpha, t0, beta), Ks)
                    delta = kh - kg
                    lines.append(f"| {M} | {alpha} | {kh} | {gk:.0f} | {gk/g0:.2f}× | {kg} | **{delta:+d}** |")
                else:
                    lines.append(f"| {M} | {alpha} | {kh} | {gk:.0f} | {gk/g0:.2f}× | — | — |")
                mrec["pred"].append(dict(M=M, alpha=alpha, K_hyb=kh, goodput=gk,
                                         gain_vs_K0=gk / g0,
                                         K_gpu=(kg if gpu_arm else None)))
        # H2 예측: 스텝당 CPU 시간의 K-기울기 (포화 후)
        M = 32
        t_hi = hybrid_goodput(M, 12, 0.5, p, bw, 0.02)[1]["t_step_ms"]
        t_lo = hybrid_goodput(M, 6, 0.5, p, bw, 0.02)[1]["t_step_ms"]
        linear = t_lo * (13 / 7)
        frac = (t_hi - t_lo) / (linear - t_lo) if linear > t_lo else 0
        lines.append("")
        lines.append(f"- H2 예측 (M=32): K 6→12 에서 스텝시간 {t_lo:.0f}→{t_hi:.0f}ms — "
                     f"선형 대비 기울기 비율 **{frac:.2f}** (가설 게이트 ≤0.15 대비)")
        n_star = p["bytes_per"] * C_CPU / (2 * bw)
        lines.append(f"- H1 예측: knee n* = {n_star:.0f} tokens/expert (BW_eff {bw/GB:.0f}GB/s, C {C_CPU/1e12:.0f}TF 기준)")
        lines.append("")
        out["models"][name] = mrec

    md = "\n".join(lines)
    print(md)
    with open(sys.argv[1] + "/E0_PREDICTIONS.md", "w") as f:
        f.write(md + "\n")
    with open(sys.argv[1] + "/e0_predictions.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
