#!/usr/bin/env python3
"""M1 게이트 판정: predictions.json (측정 전 커밋) vs 측정 디렉토리의 cell*_r{1,2}.log.
게이트: 중앙값 |오차| ≤ 20% AND 예측 차이 ≥10% 인 셀 쌍의 순위 일치 ≥ 80%. thrash_flag 셀은 별도 표기 (게이트 포함 여부 둘 다 보고).
usage: judge_m1.py <measure_dir> [predictions.json]"""
import sys, os, re, json, glob, itertools, statistics
mdir = sys.argv[1]
pfile = sys.argv[2] if len(sys.argv) > 2 else os.path.join(mdir, "predictions_at_measure.json")
P = json.load(open(pfile))
rows = []
for i, x in enumerate(P["predictions"]):
    c = x["cell"]
    logs = sorted(glob.glob(f"{mdir}/cell{i}_C{c['C']}_ctx{c['ctx']}_r*.log"))
    tp = []
    for lg in logs:
        m = re.search(r"Mean TPOT \(ms\):\s+([0-9.]+)", open(lg, errors="ignore").read())
        if m: tp.append(float(m.group(1)))
    if not tp:
        rows.append(dict(i=i, cell=c, pred=x["pred_tpot_ms"], meas=None, err=None, thrash=x["thrash_flag"], binding=x["pred_binding"])); continue
    meas = statistics.median(tp)
    rows.append(dict(i=i, cell=c, pred=x["pred_tpot_ms"], meas=meas, reps=tp, err=(x["pred_tpot_ms"] - meas) / meas * 100,
                     thrash=x["thrash_flag"], binding=x["pred_binding"]))
print(f"{'#':>2s} {'H':>3s} {'C':>3s} {'ctx':>5s} {'kv':8s} {'pred':>7s} {'meas':>7s} {'reps':>16s} {'err%':>7s} flag")
for r in rows:
    c = r["cell"]; reps = "/".join(f"{v:.1f}" for v in r.get("reps", [])) if r["meas"] else "-"
    print(f"{r['i']:2d} {c['H']:3d} {c['C']:3d} {c['ctx']:5d} {c['kv']:8s} {r['pred']:7.1f} {(r['meas'] or 0):7.1f} {reps:>16s} {(r['err'] if r['err'] is not None else 0):7.1f} {'THRASH' if r['thrash'] else ''} {r['binding']}")
def gate(subset, label):
    done = [r for r in subset if r["meas"] is not None]
    if not done: print(f"[{label}] 측정 없음"); return
    errs = sorted(abs(r["err"]) for r in done)
    med = errs[len(errs) // 2]
    pairs = [(a, b) for a, b in itertools.combinations(done, 2) if abs(a["pred"] - b["pred"]) / max(a["pred"], b["pred"]) >= 0.10]
    agree = sum(1 for a, b in pairs if (a["pred"] - b["pred"]) * (a["meas"] - b["meas"]) > 0)
    rank = agree / len(pairs) * 100 if pairs else float("nan")
    ok = med <= 20 and (rank >= 80 if pairs else True)
    print(f"[{label}] n={len(done)} 중앙값 |오차| = {med:.1f}% (max {errs[-1]:.1f}%) | 순위 일치 {agree}/{len(pairs)} = {rank:.0f}% | 게이트 {'통과' if ok else '실패'}")
gate(rows, "전체")
gate([r for r in rows if not r["thrash"]], "thrash 제외")
