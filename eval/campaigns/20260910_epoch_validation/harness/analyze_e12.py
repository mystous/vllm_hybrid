#!/usr/bin/env python3
"""EXP-E12 결과 정리: seed 반복 통계 (평균/표준편차/변동계수/95% CI) 와 soak 3구간 비교."""
import re, sys, glob, os, math

B = sys.argv[1]
NUM = r"([0-9]+\.?[0-9]*)"
def g(txt, pat, d=None):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else d

def metrics(f):
    t = open(f, errors="ignore").read()
    return dict(tok=g(t, r"Output token throughput \(tok/s\):\s*"+NUM),
                dur=g(t, r"Benchmark duration \(s\):\s*"+NUM),
                req=g(t, r"Request throughput \(req/s\):\s*"+NUM),
                tpot=g(t, r"Mean TPOT \(ms\):\s*"+NUM),
                tpot99=g(t, r"P99 TPOT \(ms\):\s*"+NUM),
                ttft=g(t, r"Median TTFT \(ms\):\s*"+NUM),
                ttft99=g(t, r"P99 TTFT \(ms\):\s*"+NUM),
                ok=g(t, r"Successful requests:\s*"+NUM),
                errs=len(re.findall(r"^Error \d+:", t, re.M)),
                err_kinds=sorted(set(re.findall(r"(ServerDisconnectedError|ConnectionResetError|ClientOSError|TimeoutError|IncompleteRead)", t))))

print("## 반복 (seed 5) 통계\n")
print("| 구성 | C | 평균 tok/s | 표준편차 | 변동계수 | 95% CI | 최소~최대 |")
print("|---|---|---|---|---|---|---|")
cells = {}
for f in sorted(glob.glob(os.path.join(B, "rep_*.log"))):
    m = re.match(r"rep_(.+)_C(\d+)_s(\d+)\.log", os.path.basename(f))
    if not m: continue
    cells.setdefault((m.group(1), m.group(2)), []).append(metrics(f)["tok"])
for (cell, C), v in cells.items():
    v = [x for x in v if x]
    if len(v) < 2: continue
    n = len(v); mu = sum(v)/n; sd = (sum((x-mu)**2 for x in v)/(n-1))**0.5
    tcrit = {2:12.706, 3:4.303, 4:3.182, 5:2.776}.get(n, 2.0)
    ci = tcrit*sd/math.sqrt(n)
    print(f"| {cell} | {C} | {mu:.2f} | {sd:.2f} | {100*sd/mu:.2f}% | ±{ci:.2f} ({100*ci/mu:.2f}%) | {min(v):.2f}~{max(v):.2f} | (n={n})")

print("\n## soak 3구간 (저부하 -> 용량근접 -> 저부하)\n")
print("| 구성 | 구간 | 계획 λ | 완료 λ | tok/s | TPOT 평균/p99 | TTFT p50/p99 | 성공 | 실패 |")
print("|---|---|---|---|---|---|---|---|---|")
rows = {}
for f in sorted(glob.glob(os.path.join(B, "soak_*.log"))):
    m = re.match(r"soak_(.+?)_(P\d_\w+)\.log", os.path.basename(f))
    if not m: continue
    cell, ph = m.group(1), m.group(2)
    d = metrics(f)
    if d["tok"] is None:      # 서버 사망 등으로 요약부가 없는 run
        print(f"| {cell} | {ph} | - | - | 측정불가 | - | - | - | 서버 사망 |")
        continue
    rows.setdefault(cell, {})[ph] = d
    lam = {"P1_low": 1.5, "P3_low": 1.5}.get(ph, None)
    N = d["ok"] or 0
    lam_c = N/d["dur"] if d["dur"] else 0
    print(f"| {cell} | {ph} | {lam if lam else '용량근접'} | {lam_c:.2f} | {d['tok']:.2f} | "
          f"{d['tpot']:.1f}/{d['tpot99']:.1f} | {d['ttft']:.0f}/{d['ttft99']:.0f} | {d['ok']:.0f} | "
          f"{d['errs']}{' (' + ','.join(d['err_kinds']) + ')' if d['err_kinds'] else ''} |")

print("\n## 저부하 복귀 열화 (P3 대 P1)\n")
print("| 구성 | tok/s 변화 | TPOT 평균 변화 | TPOT p99 변화 | 판정 (±5% 이내) |")
print("|---|---|---|---|---|")
for cell, d in rows.items():
    if "P1_low" in d and "P3_low" in d:
        a, b = d["P1_low"], d["P3_low"]
        dt = 100*(b["tok"]-a["tok"])/a["tok"]; dp = 100*(b["tpot"]-a["tpot"])/a["tpot"]
        d9 = 100*(b["tpot99"]-a["tpot99"])/a["tpot99"]
        ok = "통과" if abs(dt) <= 5 and abs(dp) <= 5 else "미달"
        print(f"| {cell} | {dt:+.2f}% | {dp:+.2f}% | {d9:+.2f}% | {ok} |")

print("\n## 자원 샘플 / 이상로그\n")
for f in sorted(glob.glob(os.path.join(B, "soak_sample_*.txt"))):
    cell = os.path.basename(f)[12:-4]
    mem = []; run = []
    for line in open(f):
        m = re.search(r"gpu_mem=\[(.*?)\] run=(\d*)", line)
        if m:
            v = [int(x.split()[0]) for x in m.group(1).split("MiB") if x.strip()]
            if v: mem.append(max(v))
            if m.group(2): run.append(int(m.group(2)))
    if mem:
        print(f"- {cell}: GPU 메모리 최대 {max(mem)} MiB, 최초 {mem[0]} MiB, 마지막 {mem[-1]} MiB, "
              f"샘플 {len(mem)}개 / 실행중 요청 p50 {sorted(run)[len(run)//2] if run else '-'} 최대 {max(run) if run else '-'}")
for line in open(os.path.join(B, "RUN.log")):
    if "이상로그" in line: print("- " + line.strip())
