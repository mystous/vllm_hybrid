#!/usr/bin/env python3
"""EXP-E11 구성 선택 비용모델 (실행계획서 §17).

구성 x = (kv dtype, graph_max, bucket list, chunk, admission cap, mixed) 과
워크로드 w = (L_in, L_out, 동시성 또는 도착률, SLO) 를 입력받아
지속 가능한 출력 처리량을 예측하고, feasible 후보 중 최대를 고른다.

모델 구조 (측정으로 뒷받침되는 항만 둔다):

  1. KV 용량.  boot 로그 24건에서 KV 풀 토큰 수는 graph 버킷 수·chunk 와 무관하고
     dtype 으로만 결정됐다 (bf16 63,654 / fp8 127,309 @ mem-fraction-static 0.94).
     따라서 N_kv(q) 는 상수표, --max-total-tokens M 이 있으면 min(N_kv, M).

  2. 실행 가능 배치.  요청 하나의 평균 KV 점유는 L_in + L_out/2.
     B_cap = eta * N_kv / (L_in + L_out/2),  B_eff = min(C, admission, B_cap, )
     eta 는 스케줄러 여유분 (fit).

  3. 디코드 스텝 시간 (ms).
     T(B) = t0 + t1 * pad(B) + t2(q) * B * (L_in + L_out/2) / 1000
     pad(B) = B 이상인 가장 작은 버킷 (B <= graph_max 일 때). B > graph_max 면
     eager 실행이므로 t1 에 eager 벌점 rho 를 곱한다.

  4. prefill 몫.  요청당 prefill 토큰 L_in 을 처리량 P (tok/s) 로 처리한다고 보고,
     디코드 시간과 나눠 갖는다:
        share_decode = (L_out * T/1000) / (L_out * T/1000 + L_in / P)
     출력 처리량 = B_eff / T * 1000 * share_decode

  5. SLO.  예측 TPOT = T, 예측 TTFT ~ (대기 배치 / 처리율) — TTFT 는 open-loop 에서만
     의미가 있으므로 여기서는 TPOT 제약만 hard 로 두고 TTFT 는 보고만 한다.

fit 대상 파라미터: t0, t1, t2_bf16, t2_fp8, eta, P4096, P8192, rho, eps   (9개)
"""
import json, math, csv, sys, os

N_KV = {"bf16": 63654, "fp8": 127309}          # mem-fraction-static 0.94, hot-96, TP4 실측
DEFAULT = dict(t0=17.0, t1=0.50, t2_bf16=0.5, t2_fp8=0.5, eta=0.9, P4096=1185.0, P8192=1185.0, rho=1.6, eps=0.95)


def pad_batch(B, graph_max, buckets):
    if B > graph_max:
        return B, True
    for b in sorted(buckets):
        if b >= B:
            return b, False
    return graph_max, False


def predict(cfg, wl, p):
    """cfg: dict(kv, graph_max, buckets, chunk, admission, maxtot)
       wl : dict(Lin, Lout, C)
       p  : 파라미터 dict.  반환: dict(tok_s, tpot, B_eff, binding)"""
    q = cfg["kv"]
    nkv = N_KV[q]
    if cfg.get("maxtot"):
        nkv = min(nkv, cfg["maxtot"])
    Lavg = wl["Lin"] + wl["Lout"] / 2.0
    B_cap = p["eta"] * nkv / Lavg
    cands = [("C", wl["C"]), ("KV", B_cap)]
    if cfg.get("admission"):
        cands.append(("admission", cfg["admission"]))
    binding, B_eff = min(cands, key=lambda t: t[1])
    B_eff = max(1.0, B_eff)
    Bp, eager = pad_batch(B_eff, cfg["graph_max"], cfg["buckets"])
    t2 = p["t2_bf16"] if q == "bf16" else p["t2_fp8"]
    T = p["t0"] + p["t1"] * (Bp * (p["rho"] if eager else 1.0)) + t2 * B_eff * Lavg / 1000.0
    dec = B_eff / T * 1000.0
    P = p["P8192"] if cfg.get("chunk", 4096) >= 8192 else p["P4096"]
    share = (wl["Lout"] * T / 1000.0) / (wl["Lout"] * T / 1000.0 + wl["Lin"] / P)
    # KV 가 요청을 다 담지 못하면 스케줄러가 실행 중 요청을 회수(retract)하고 나중에 다시
    # prefill 한다 -> 낭비된 작업만큼 처리량이 깎인다. 초과분에 비례하는 벌점 theta.
    over = max(0.0, wl["C"] / max(B_cap, 1e-9) - 1.0)
    return dict(tok_s=dec * share * p["eps"], tpot=T, B_eff=B_eff, binding=binding,
                share_decode=share, oversub=over)


# ---------------- fitting (numpy/scipy 없이 Nelder-Mead) ----------------
def nelder_mead(f, x0, steps, iters=4000, tol=1e-10):
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        y = list(x0); y[i] += steps[i]; simplex.append(y)
    fv = [f(s) for s in simplex]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: fv[i])
        simplex = [simplex[i] for i in order]; fv = [fv[i] for i in order]
        if abs(fv[-1] - fv[0]) < tol: break
        cen = [sum(s[j] for s in simplex[:-1]) / n for j in range(n)]
        xr = [cen[j] + 1.0 * (cen[j] - simplex[-1][j]) for j in range(n)]; fr = f(xr)
        if fr < fv[0]:
            xe = [cen[j] + 2.0 * (cen[j] - simplex[-1][j]) for j in range(n)]; fe = f(xe)
            simplex[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[-2]:
            simplex[-1], fv[-1] = xr, fr
        else:
            xc = [cen[j] + 0.5 * (simplex[-1][j] - cen[j]) for j in range(n)]; fc = f(xc)
            if fc < fv[-1]:
                simplex[-1], fv[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    fv[i] = f(simplex[i])
    i = min(range(n + 1), key=lambda i: fv[i])
    return simplex[i], fv[i]


KEYS = ["t0", "t1", "t2_bf16", "t2_fp8", "eta", "P4096", "P8192", "rho", "eps"]
LO = dict(t0=1.0, t1=0.05, t2_bf16=0.0, t2_fp8=0.0, eta=0.3, P4096=300.0, P8192=300.0, rho=1.0, eps=0.5)
HI = dict(t0=60.0, t1=2.0, t2_bf16=3.0, t2_fp8=3.0, eta=1.2, P4096=40000.0, P8192=40000.0, rho=15.0, eps=1.0)


def fit(cells, x0=None):
    """cells: [(cfg, wl, tok_s_measured, tpot_measured)] — 로그 처리량 + 로그 TPOT 잔차 합"""
    def unpack(v):
        return {k: min(HI[k], max(LO[k], v[i])) for i, k in enumerate(KEYS)}

    def loss(v):
        p = unpack(v); s = 0.0
        for cfg, wl, y, tp in cells:
            r = predict(cfg, wl, p)
            s += (math.log(max(r["tok_s"], 1e-6)) - math.log(y)) ** 2
            if tp:
                s += 0.5 * (math.log(max(r["tpot"], 1e-6)) - math.log(tp)) ** 2
        return s

    start = [(x0 or DEFAULT)[k] for k in KEYS]
    steps = [max(abs(s) * 0.3, 0.05) for s in start]
    best, fv = nelder_mead(loss, start, steps)
    return unpack(best), fv


def load_cells(csv_path, dirs_prefix="20260910", only_inf=True):
    out = []
    for r in csv.DictReader(open(csv_path)):
        if not r["dir"].startswith(dirs_prefix) or not r["kv"]:
            continue
        if only_inf and r["rate"] != "inf":
            continue
        if not r["C"].isdigit() or not r["tok_s"]:
            continue
        buckets = {"auto": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64], "32,64": [32, 64], "32,64,96": [32, 64, 96],
                   "32..224/32": [32, 64, 96, 128, 160, 192, 224]}[r["buckets"]]
        cfg = dict(kv=r["kv"], graph_max=int(r["graph_max"]), buckets=buckets,
                   chunk=int(r["chunk"]), admission=None,
                   maxtot=int(r["maxtot"]) if r["maxtot"] else None,
                   mixed=int(r["mixed"]), cpuinfer=int(r["cpuinfer"]), cell=r["cell"], dir=r["dir"])
        wl = dict(Lin=int(r["inlen"]), Lout=int(r["outlen"]), C=int(r["C"]))
        wl["np"] = int(r["np"]) if r["np"] else 0
        out.append((cfg, wl, float(r["tok_s"]), float(r["tpot"]) if r["tpot"] else None))
    return out


if __name__ == "__main__":
    csv_path = sys.argv[1]
    cells = load_cells(csv_path)
    # calibration = 완전한 실행 패치 (cpuinfer 96) + mixed off + admission cap 없음
    # 모델 적용 범위: cpuinfer 96 + 전체 실행 패치 + mixed off + admission cap 없음.
    # 추가로 job 크기 < 3*C 인 셀은 램프 구간이 지배하므로 보정에서 제외한다.
    # 20260910_002116 은 flush 도입 전 실행이라 prefix cache 오염 (A00 C64 971.75) -> 제외
    cells = [c for c in cells if c[0].get("dir", "") != "20260910_002116_expE00_E01_E02"]
    cal = [c for c in cells if c[0]["cpuinfer"] == 96 and c[0]["mixed"] == 0
           and c[0]["cell"] not in ("C1", "C2", "F00", "F10", "A11m64")
           and c[1]["np"] >= 3 * c[1]["C"]]
    hold = [c for c in cells if c not in cal]
    p, fv = fit(cal)
    print("fit params:", {k: round(v, 4) for k, v in p.items()}, " loss", round(fv, 4))
    print(f"\n{'set':4} {'cell':8} {'wl':>9} {'C':>4} {'meas':>8} {'pred':>8} {'err%':>7} {'B_eff':>6} {'bind':>9} {'TPOTm':>7} {'TPOTp':>7}")
    for tag, S in (("cal", cal), ("out", hold)):
        for cfg, wl, y, tp in S:
            r = predict(cfg, wl, p)
            print(f"{tag:4} {cfg['cell']:8} {str(wl['Lin'])+'x'+str(wl['Lout']):>9} {wl['C']:>4} "
                  f"{y:8.1f} {r['tok_s']:8.1f} {100*(r['tok_s']-y)/y:7.1f} {r['B_eff']:6.1f} "
                  f"{r['binding']:>9} {(tp or 0):7.1f} {r['tpot']:7.1f}")
    errs = sorted(abs(100 * (predict(c[0], c[1], p)["tok_s"] - c[2]) / c[2]) for c in cal)
    print(f"\ncalibration 절대오차 중앙값 {errs[len(errs)//2]:.1f}% / p90 {errs[int(len(errs)*0.9)]:.1f}% / 최대 {errs[-1]:.1f}%")
    json.dump(p, open(os.path.join(os.path.dirname(csv_path), "selector_params.json"), "w"), indent=1)
