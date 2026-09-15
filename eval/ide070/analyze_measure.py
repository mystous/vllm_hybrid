#!/usr/bin/env python3
"""IDE_070 / TSK_054 — run_measure.sh 산출물 → 지시서 §1 지표 (markdown).
사용: ~/venv-bench/bin/python analyze_measure.py <result_dir> [hotmap.json]
"""
import csv, glob, json, os, re, statistics as st, sys
import torch

D = sys.argv[1]
HOT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.cache/huggingface/kt/ide069/hotmap.json")
OUT_TOK = 256 * 128
NODE0 = set(range(0, 56)) | set(range(112, 168))
lines = []
def P(s=""): lines.append(s); print(s)

def bench(rep):
    t = open(f"{D}/{rep}/bench_summary.txt").read()
    g = lambda k: float(re.search(rf"{k}[^:]*:\s*([\d.]+)", t).group(1))
    dur = g("Benchmark duration"); out = g("Output token throughput") * dur; tot = g("Total token throughput") * dur
    return dict(tput=g("Output token throughput"), tpot=g("Median TPOT"), tpot95=g("P95 TPOT"),
                ttft=g("Median TTFT"), ttft95=g("P95 TTFT"), dur=dur, in_tok=tot - out, out_tok=out)

# ---------- pcm.csv ----------
def pcm(rep):
    rows = list(csv.reader(open(f"{D}/{rep}/pcm.csv")))
    grp, name, data = rows[0], rows[1], rows[2:]
    idx = {}; occ = {}
    order = ["System", "Socket 0", "Socket 1"]
    for i, (g, n) in enumerate(zip(grp, name)):
        idx.setdefault((g, n), i)
        occ.setdefault(n, []).append(i)
    for g in order:                       # 그룹명이 빈 열 (topdown 4개) 은 등장 순서로 System/S0/S1 에 배정
        for n, ii in occ.items():
            if (g, n) not in idx and len(ii) >= 3:
                idx[(g, n)] = ii[order.index(g)]
    col = lambda g, n: [float(r[idx[(g, n)]].strip().rstrip("%")) for r in data if len(r) > idx[(g, n)] and r[idx[(g, n)]].strip() not in ("", "N/A")]
    # 부하 구간만: 시스템 EXEC(전체 스레드 기준 IPC) 가 유휴값보다 큰 행
    ex = col("System", "EXEC"); thr = max(ex) * 0.3
    busy = [i for i, v in enumerate(ex) if v > thr]
    sel = lambda g, n: [v for i, v in enumerate(col(g, n)) if i in busy]
    m = lambda g, n: st.mean(sel(g, n))
    r = {"n_all": len(data), "n_busy": len(busy)}
    for g, tag in (("System", "sys"), ("Socket 0", "s0"), ("Socket 1", "s1")):
        r[tag] = dict(IPC=m(g, "IPC"), EXEC=m(g, "EXEC"), FREQ=m(g, "FREQ"), AFREQ=m(g, "AFREQ"),
                      L3MPI=m(g, "L3MPI"), L2MPI=m(g, "L2MPI"), L3HIT=m(g, "L3HIT"),
                      FE=m(g, "Frontend_bound(%)"), BS=m(g, "Bad_Speculation(%)"), BB=m(g, "Backend_Bound(%)"), RET=m(g, "Retiring(%)"),
                      READ=m(g, "READ")/2.0, WRITE=m(g, "WRITE")/2.0, INST=sum(sel(g, "INST")), ACYC=sum(sel(g, "ACYC")),
                      L3MISS=sum(sel(g, "L3MISS")))
        if g != "System":
            r[tag].update(LMB=m(g, "LMB"), RMB=m(g, "RMB"), L3OCC=m(g, "L3OCC"))
    r["upi_in"] = m("System", "TotalUPIin"); r["upi_out"] = m("System", "TotalUPIout"); r["upi_mc"] = m("System", "UPItoMC")
    r["dram_J"] = sum(sel("System Pack C-States", "DRAM Energy (Joules)")); r["proc_J"] = sum(sel("System Pack C-States", "Proc Energy (Joules)"))
    r["upi0_in"] = sum(m("SKT0dataIn", f"UPI{i}") for i in range(4)); r["upi1_in"] = sum(m("SKT1dataIn", f"UPI{i}") for i in range(4))
    r["upi0_in_pct"] = st.mean(m("SKT0dataIn (percent)", f"UPI{i}") for i in range(4)); r["upi1_in_pct"] = st.mean(m("SKT1dataIn (percent)", f"UPI{i}") for i in range(4))
    r["busy_sec"] = len(busy) * 2
    return r

# ---------- pcm_numa.csv ----------
def pcm_numa(rep):
    tot = []; s0 = []; s1 = []
    cur = {0: [0, 0, 0, 0], 1: [0, 0, 0, 0]}
    for ln in open(f"{D}/{rep}/pcm_numa.csv"):
        p = ln.rstrip(",\n").split(",")
        if len(p) < 6 or p[0] in ("Core", ""): continue
        if p[0] == "*":
            tot.append([float(x) for x in p[1:6]]); s0.append(cur[0]); s1.append(cur[1]); cur = {0: [0, 0, 0, 0], 1: [0, 0, 0, 0]}
            continue
        try: c = int(p[0])
        except ValueError: continue
        v = cur[0 if c in NODE0 else 1]
        v[0] += float(p[2]); v[1] += float(p[3]); v[2] += float(p[4]); v[3] += float(p[5])
    # 부하 구간: 총 instructions 상위
    ins = [t[1] for t in tot]; thr = max(ins) * 0.3
    b = [i for i, v in enumerate(ins) if v > thr]
    agg = lambda arr, k: sum(arr[i][k] for i in b)
    r = {"n_busy": len(b)}
    r["sys"] = dict(ipc=agg(tot, 1) / agg(tot, 2), local=agg(tot, 3), remote=agg(tot, 4))
    r["s0"] = dict(ipc=agg(s0, 0) / max(agg(s0, 1), 1), local=agg(s0, 2), remote=agg(s0, 3))
    r["s1"] = dict(ipc=agg(s1, 0) / max(agg(s1, 1), 1), local=agg(s1, 2), remote=agg(s1, 3))
    for k in ("sys", "s0", "s1"):
        x = r[k]; x["remote_pct"] = 100 * x["remote"] / max(x["local"] + x["remote"], 1)
    return r

# ---------- turbostat ----------
def turbo(rep):
    f = f"{D}/{rep}/turbostat.txt"
    if not os.path.exists(f): return None
    hdr = None; rows = []
    for ln in open(f):
        p = ln.split()
        if not p: continue
        if p[0] == "Core": hdr = p; continue
        if p[0] == "-" and hdr and len(p) == len(hdr): rows.append(dict(zip(hdr, p)))
    if not rows: return None
    g = lambda k: [float(r[k]) for r in rows if k in r]
    bz = g("Busy%"); thr = max(bz) * 0.3; b = [i for i, v in enumerate(bz) if v > thr]
    m = lambda k: st.mean([g(k)[i] for i in b])
    return dict(n=len(rows), n_busy=len(b), busy=m("Busy%"), avg_mhz=m("Avg_MHz"), bzy_mhz=m("Bzy_MHz"), ipc=m("IPC"))

# ---------- mpstat ----------
def mpstat(rep):
    per = {}
    for ln in open(f"{D}/{rep}/mpstat.txt"):
        p = ln.split()
        if len(p) < 12 or p[0] == "Average:" or not re.match(r"\d+:\d+:\d+", p[0]): continue
        off = 1 if p[1] in ("AM", "PM") else 0
        cpu = p[1 + off]
        if cpu == "all" or cpu == "CPU": continue
        try: usr, sys_ = float(p[2 + off]), float(p[4 + off])
        except ValueError: continue
        per.setdefault(int(cpu), []).append(usr + sys_)
    if not per: return None
    # 부하 구간: 전체 평균이 유휴보다 큰 샘플
    n = min(len(v) for v in per.values())
    tot = [sum(per[c][i] for c in per) / len(per) for i in range(n)]
    thr = max(tot) * 0.3; b = [i for i, v in enumerate(tot) if v > thr]
    avg = {c: st.mean([per[c][i] for i in b]) for c in per}
    s0 = [avg[c] for c in avg if c in NODE0]; s1 = [avg[c] for c in avg if c not in NODE0]
    phys = [avg[c] for c in avg if c < 112]; ht = [avg[c] for c in avg if c >= 112]
    hist = {k: sum(1 for v in avg.values() if lo <= v < hi) for k, (lo, hi) in
            {"<10%": (0, 10), "10-50%": (10, 50), "50-90%": (50, 90), ">=90%": (90, 101)}.items()}
    top = sorted(avg.items(), key=lambda kv: -kv[1])[:8]
    return dict(n_busy=len(b), all=st.mean(avg.values()), s0=st.mean(s0), s1=st.mean(s1), phys=st.mean(phys), ht=st.mean(ht),
                s0_phys=st.mean([avg[c] for c in range(0, 56)]), s1_phys=st.mean([avg[c] for c in range(56, 112)]),
                s0_ht=st.mean([avg[c] for c in range(112, 168)]), s1_ht=st.mean([avg[c] for c in range(168, 224)]),
                hist=hist, top=top, n_over50=sum(1 for v in avg.values() if v >= 50))

# ---------- vmstat / pidstat ----------
def vmstat(rep):
    rows = []
    for ln in open(f"{D}/{rep}/vmstat.txt"):
        p = ln.split()
        if len(p) >= 17 and p[0].isdigit(): rows.append([int(x) for x in p[:17]])
    r = [x for x in rows if x[12] + x[13] > 20]  # us+sy > 20%
    if not r: return None
    return dict(n=len(r), intr=st.mean(x[10] for x in r), cs=st.mean(x[11] for x in r), us=st.mean(x[12] for x in r), sy=st.mean(x[13] for x in r), id=st.mean(x[14] for x in r), runq=st.mean(x[0] for x in r))

def pidstat(rep):
    per = {}  # interval -> [vol, invol]
    key = None
    for ln in open(f"{D}/{rep}/pidstat_ctx.txt"):
        p = ln.split()
        if len(p) < 7 or not re.match(r"\d+:\d+:\d+", p[0]): continue
        off = 1 if p[1] in ("AM", "PM") else 0
        if p[1 + off] == "UID": key = p[0]; continue
        try: v, iv = float(p[4 + off]), float(p[5 + off])
        except ValueError: continue
        per.setdefault(key, [0, 0]); per[key][0] += v; per[key][1] += iv
    vals = [v for v in per.values() if v[0] + v[1] > 0]
    if not vals: return None
    vals.sort(key=lambda x: -(x[0] + x[1])); top = vals[:max(1, len(vals) // 2)]
    return dict(n=len(vals), vol=st.mean(v[0] for v in top), invol=st.mean(v[1] for v in top))

def perf(rep, f):
    p = f"{D}/{rep}/{f}"
    if not os.path.exists(p) or os.path.getsize(p) == 0: return None
    return open(p).read()

# ---------- recorder → cold selections / token ----------
def recorder():
    fs = sorted(glob.glob(f"{D}/expert_distribution_recorder_*.pt"))
    if not fs: return None
    hm = json.load(open(HOT))["physical_to_logical_map"]
    res = []
    for f in fs:
        lc = torch.load(f, map_location="cpu", weights_only=False)["logical_count"].sum(0).to(torch.float64)  # [62,160]
        L, E = lc.shape
        cold = 0.0; tot = lc.sum().item(); per_layer = []
        for l in range(L):
            hot = set(hm[l][:96]); c = sum(lc[l, e].item() for e in range(E) if e not in hot)
            cold += c; per_layer.append(c / max(lc[l].sum().item(), 1))
        tokens = tot / (L * 8)
        res.append(dict(file=os.path.basename(f), tokens=tokens, cold_sel_per_token=cold / tokens, cold_frac=cold / tot,
                        layer_cold_max=max(per_layer), layer_cold_min=min(per_layer),
                        distinct_cold_per_layer_step=None))
    return res

# ================= 출력 =================
reps = [r for r in ("rep1_pcm", "rep2_pcmnuma", "rep3_perf") if os.path.isdir(f"{D}/{r}")]
P(f"# IDE_070 TSK_054 측정 요약 — `{os.path.basename(D)}`"); P()
P("launch: `" + open(f"{D}/launch_cmd.txt").read().strip()[:200] + " …`"); P()
P("## 벤치 (C=64, N=256, sonnet 512/128)"); P()
P("| rep | tok/s | TPOT p50/p95 ms | TTFT p50/p95 ms | dur s | in/out tok |"); P("|---|---|---|---|---|---|")
B = {}
for r in reps:
    b = bench(r); B[r] = b
    P(f"| {r} | {b['tput']:.2f} | {b['tpot']:.1f}/{b['tpot95']:.1f} | {b['ttft']:.0f}/{b['ttft95']:.0f} | {b['dur']:.1f} | {b['in_tok']:.0f}/{b['out_tok']:.0f} |")
P()
if os.path.exists(f"{D}/rep1_pcm/pcm.csv"):
    p = pcm("rep1_pcm"); b = B["rep1_pcm"]
    P(f"## pcm (rep1, 2 s 간격, 부하 구간 {p['n_busy']}/{p['n_all']} 샘플 ≈ {p['busy_sec']} s)"); P()
    P("| 범위 | IPC(코어) | EXEC(스레드) | FREQ | AFREQ | L3 MPI | L2 MPI | L3 hit | FE/BS/BB/RET % | DRAM R+W GB/s (GB/간격 2 s ÷2) | 로컬/원격 MB/s |"); P("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, nm in (("sys", "system"), ("s0", "socket0"), ("s1", "socket1")):
        x = p[k]; lr = f"{x['LMB']:.0f}/{x['RMB']:.0f}" if "LMB" in x else "—"
        P(f"| {nm} | {x['IPC']:.2f} | {x['EXEC']:.2f} | {x['FREQ']:.2f} | {x['AFREQ']:.2f} | {x['L3MPI']:.4f} | {x['L2MPI']:.4f} | {x['L3HIT']:.2f} | {x['FE']:.0f}/{x['BS']:.0f}/{x['BB']:.0f}/{x['RET']:.0f} | {x['READ']:.1f}+{x['WRITE']:.1f}={x['READ']+x['WRITE']:.1f} | {lr} |")
    P()
    cyc = p["sys"]["ACYC"] * 1e6; ins = p["sys"]["INST"] * 1e6
    P(f"- 부하 구간 합계: active cycles {cyc:.3e}, instructions {ins:.3e}, L3 miss {p['sys']['L3MISS']*1e6:.3e}")
    P(f"- **cycles / output token = {cyc/b['out_tok']:.3e}** (출력 {b['out_tok']:.0f} tok), cycles / (input+output) token = {cyc/(b['in_tok']+b['out_tok']):.3e}")
    P(f"- instructions / output token = {ins/b['out_tok']:.3e}, L3 miss / output token = {p['sys']['L3MISS']*1e6/b['out_tok']:.3e}")
    P(f"- UPI: 시스템 in {p['upi_in']:.1f} / out {p['upi_out']:.1f} (pcm 단위), UPItoMC {p['upi_mc']:.2f}; 소켓0 수신 {p['upi0_in']:.0f} (링크 이용률 {p['upi0_in_pct']:.1f}%), 소켓1 수신 {p['upi1_in']:.0f} ({p['upi1_in_pct']:.1f}%)")
    P(f"- 에너지 (부하 구간): package {p['proc_J']:.0f} J, DRAM {p['dram_J']:.0f} J → package 평균 {p['proc_J']/max(p['busy_sec'],1):.0f} W, DRAM {p['dram_J']/max(p['busy_sec'],1):.0f} W")
    P()
if os.path.exists(f"{D}/rep2_pcmnuma/pcm_numa.csv"):
    n = pcm_numa("rep2_pcmnuma")
    P(f"## pcm-numa (rep2, 부하 구간 {n['n_busy']} 샘플; 접근 수 = demand + L2 prefetch, code, RFO)"); P()
    P("| 범위 | IPC | local DRAM 접근 | remote DRAM 접근 | **remote %** |"); P("|---|---|---|---|---|")
    for k, nm in (("sys", "system"), ("s0", "socket0 코어"), ("s1", "socket1 코어")):
        x = n[k]; P(f"| {nm} | {x['ipc']:.2f} | {x['local']:.3e} | {x['remote']:.3e} | {x['remote_pct']:.1f} |")
    P()
t = turbo("rep3_perf")
if t:
    P(f"## turbostat (rep3, 5 s 간격, 부하 {t['n_busy']}/{t['n']} 샘플, 전체 요약행)"); P()
    P(f"- Busy% {t['busy']:.1f}, Avg_MHz {t['avg_mhz']:.0f}, **Bzy_MHz {t['bzy_mhz']:.0f}** (turbo OFF 확인), IPC {t['ipc']:.2f}"); P()
for r in reps:
    m = mpstat(r)
    if not m: continue
    P(f"## mpstat 코어별 사용률 (%usr+%sys, {r}, 부하 {m['n_busy']} 샘플)"); P()
    P(f"- 전체 {m['all']:.1f}% · socket0 {m['s0']:.1f}% (phys {m['s0_phys']:.1f} / HT {m['s0_ht']:.1f}) · socket1 {m['s1']:.1f}% (phys {m['s1_phys']:.1f} / HT {m['s1_ht']:.1f})")
    P(f"- 분포: {m['hist']} · ≥50% 코어 수 {m['n_over50']}/224 · 최상위: " + ", ".join(f"cpu{c}={v:.0f}%" for c, v in m["top"]))
    P()
    break
for r in reps:
    v = vmstat(r); pd = pidstat(r)
    if v: P(f"- vmstat ({r}, 부하 {v['n']} 샘플): 인터럽트 {v['intr']:.0f}/s, **컨텍스트 스위치 {v['cs']:.0f}/s**, us {v['us']:.0f} sy {v['sy']:.0f} id {v['id']:.0f}, runq {v['runq']:.0f}")
    if pd: P(f"- pidstat TP0 스레드 합 ({r}, 상위 절반 구간 평균): 자발 {pd['vol']:.0f}/s, 비자발 {pd['invol']:.0f}/s")
P()
for f in ("perf_stat_sys.txt", "perf_stat_tp0.txt"):
    s = perf("rep3_perf", f)
    if s:
        P(f"## {f} (주의: perf 창 400 s = 부하 ~60 s + 유휴; 비율 지표만 신뢰)"); P("```"); P(s.strip()); P("```"); P()
rc = recorder()
if rc:
    P("## recorder → cold expert (rep1, hotmap96 기준 physical ≥96 = CPU)"); P()
    P("| 파일 | 토큰 수 | cold 선택/토큰 (8 선택 중) | cold 비율 | 층별 cold 최소/최대 |"); P("|---|---|---|---|---|")
    for x in rc:
        P(f"| {x['file'][-20:]} | {x['tokens']:.0f} | **{x['cold_sel_per_token']:.3f}** | {100*x['cold_frac']:.2f}% | {100*x['layer_cold_min']:.2f}% / {100*x['layer_cold_max']:.2f}% |")
    P()
hb = open(f"{D}/hbm_after_boot.csv").read().strip().replace("\n", "; ")
P(f"- HBM after boot: {hb}")
P(f"- TP0 affinity: {open(f'{D}/tp0_affinity.txt').read().strip().replace(chr(10), ' · ')}")
open(f"{D}/MEASURE_SUMMARY.md", "w").write("\n".join(lines) + "\n")
