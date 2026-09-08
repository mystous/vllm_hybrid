#!/usr/bin/env python3
"""decode 스텝 GPU 타임라인 분해: usage: analyze_trace.py <dir with *TP-0.trace.json.gz>"""
import gzip, json, glob, sys, collections
d = sys.argv[1]
f = sorted(glob.glob(f"{d}/*TP-0.trace.json.gz"))[0]
ev = json.load(gzip.open(f))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X"]
gl = sorted([e for e in X if e["name"] == "cudaGraphLaunch"], key=lambda e: e["ts"])
gpu = sorted([e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
print(f"graph launches: {len(gl)}  host dur(ms): {[round(e['dur']/1000,1) for e in gl]}")
rows = []
for i in range(len(gl) - 1):
    t0, t1 = gl[i]["ts"], gl[i + 1]["ts"]
    w = [e for e in gpu if t0 <= e["ts"] < t1]
    if not w:
        continue
    gs, ge = w[0]["ts"], max(e["ts"] + e["dur"] for e in w)
    busy = collections.Counter(); cnt = collections.Counter()
    for e in w:
        busy[e["cat"]] += e["dur"]; cnt[e["cat"]] += 1
    gaps = []; cur = gs
    for e in w:
        if e["ts"] > cur: gaps.append(e["ts"] - cur)
        cur = max(cur, e["ts"] + e["dur"])
    big = [g for g in gaps if g > 50]
    kb = collections.Counter()
    for e in w:
        if e["cat"] == "kernel": kb[e["name"][:48]] += e["dur"]
    rows.append(((ge - gs) / 1000, busy["kernel"] / 1000, busy["gpu_memcpy"] / 1000, sum(gaps) / 1000, len(big), sum(big) / 1000))
    print(f"step{i}: span {rows[-1][0]:.1f} ms | kernel {rows[-1][1]:.1f} x{cnt['kernel']} | memcpy {rows[-1][2]:.1f} | idle {rows[-1][3]:.1f} (gaps>50us: {len(big)}, {rows[-1][5]:.1f} ms) | top: " + ", ".join(f"{n[:22]} {v/1000:.1f}" for n, v in kb.most_common(4)))
if len(rows) > 1:
    r = rows[1:]
    print(f"AVG(step1+): span {sum(x[0] for x in r)/len(r):.1f} ms | kernel {sum(x[1] for x in r)/len(r):.1f} | memcpy {sum(x[2] for x in r)/len(r):.1f} | idle {sum(x[3] for x in r)/len(r):.1f}")
