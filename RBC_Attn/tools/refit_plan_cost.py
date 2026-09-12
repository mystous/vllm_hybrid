#!/usr/bin/env python3
"""교정 raw 에서 시간 모델만 다시 적합한다 (GPU 재측정 없음).

`calibrate_plan_cost.py` 가 저장한 특징·실측 시간을 그대로 쓰므로 모델 형태를 바꿀 때
GPU 를 다시 점유하지 않는다. 측정값을 새로 만들지는 않는다.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rbc import select

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="results/v02/select/calibration_raw.json")
ap.add_argument("--table", default="configs/plan_cost_table.json")
ap.add_argument("--ridge", type=float, default=1e-3)
ap.add_argument("--remeasure-plan-cost", action="store_true",
                help="계획 생성 CPU 비용을 지금 다시 잰다 (플래너를 바꾼 뒤 필요. GPU 불필요)")
a = ap.parse_args()

d = json.load(open(a.raw))
meta = d["meta"]

# raw 가 현재 특징 정의를 모두 담고 있는지 먼저 확인한다.
# 누락 필드를 기본값으로 채우면 그 특징이 전부 0 인 표가 만들어지고, 평가 입력이
# 모두 '범위 밖' 으로 판정돼 선택기가 조용히 무력화된다 (실제로 한 번 발생했다).
REQUIRED = ("bm", "candidate", "direct_rows", "empty_rows", "g", "max_k", "max_m",
            "max_q", "measured_us", "multi_rows", "n_tasks", "padded", "slots",
            "sum_k", "unique_blocks", "k_p50", "k_p90", "full_ratio")
_first = next((x for x in d["raw"] if "error" not in x), None)
if _first is None:
    raise SystemExit("raw 에 유효 표본이 없다")
_missing = [k for k in REQUIRED if k not in _first]
if _missing:
    raise SystemExit(f"raw 에 특징이 빠졌다: {_missing}\n"
                     f"tools/calibrate_plan_cost.py 를 다시 실행해 raw 를 새로 만들어야 한다.")
rows, pairs, bestm = [], {}, {}
raw_by_key = {}
for r in d["raw"]:
    if "error" in r:
        continue
    f = select.CandidateFeatures(
        name=r["candidate"], bm=int(r["bm"]), n_tasks=int(r["n_tasks"]),
        sum_k=int(r["sum_k"]), max_k=int(r["max_k"]), max_q=int(r["max_q"]),
        padded=float(r["padded"]), slots=int(r["slots"]),
        direct_rows=int(r["direct_rows"]), multi_rows=int(r["multi_rows"]),
        empty_rows=int(r["empty_rows"]), unique_blocks=int(r["unique_blocks"]),
        g=int(r["g"]), d=int(meta["d"]), bk=int(meta["bk"]),
        k_p50=float(r["k_p50"]), k_p90=float(r["k_p90"]),
        full_ratio=float(r["full_ratio"]))
    y = float(r["measured_us"])
    rows.append((f, y))
    pk = (r["seed"], r["nq"], r["g"], r["sel"], r["pattern"])
    cur = pairs.setdefault(pk, {})
    if r["candidate"] not in cur or y < cur[r["candidate"]][1]:
        cur[r["candidate"]] = (f, y)
    raw_by_key.setdefault(pk, []).append(r)

# 형상 구간별 최적 max_m 관측 (후보 전체에서 가장 빠른 조합의 max_m)
for pk, items in raw_by_key.items():
    best = min(items, key=lambda x: x["measured_us"])
    key = select.CostTable.shape_key(int(best["g"]), int(best["nq"]))
    bestm.setdefault(key, []).append(int(best["max_m"]))

meta = dict(meta); meta["refit"] = "raw 에서 재적합 (GPU 재측정 없음)"
if a.remeasure_plan_cost:
    # GPU 실행시간은 플래너 구현과 무관하므로 raw 를 그대로 쓰고, 계획 CPU 비용만
    # 현재 구현으로 다시 잰다. 이 값이 선택기의 extra_prepare 근거다.
    import time as _t, os as _os
    from rbc import data as _d, planner as _pl
    POL = {"q_outer": "q_outer", "signature": "signature_only",
           "bounded_merge": "rbc"}
    acc = {k: [] for k in POL}
    th = int(_os.environ.get("RBC_THREADS", 1))
    for nq in meta["nq"]:
        for gg in (meta["g"] if isinstance(meta["g"], list) else [meta["g"]]):
            for sel in meta["sel"]:
                for pat in meta["patterns"]:
                    try:
                        ss = _d.make_synthetic(pat, hkv=1, nq=nq, nk=meta["nk"], g=gg,
                                               d=meta["d"], bk=meta["bk"], sel=sel, seed=0)
                    except Exception:
                        continue
                    pr, ii = ss["ptr"][0], ss["block_ids"][0]
                    for name, pol in POL.items():
                        ts = []
                        for _ in range(5):
                            t0 = _t.perf_counter()
                            _pl.plan_head(pr, ii, bk=meta["bk"], d=meta["d"], g=gg,
                                          max_m=16, policy=pol, min_tasks=0, threads=th)
                            ts.append((_t.perf_counter() - t0) * 1e6)
                        acc[name].append(float(np.median(ts)))
    meta["plan_cost_us"] = {k: float(np.median(v)) for k, v in acc.items() if v}
    meta["plan_cost_remeasured"] = True
    print(f"계획 CPU 재측정 (threads={th}): "
          f"{ {k: round(v) for k, v in meta['plan_cost_us'].items()} }")
t = select.fit_table(rows, int(meta["sm"]), ridge=a.ridge, meta=meta, pairs=pairs,
                     best_max_m=bestm)
t.save(a.table)
print(f"표본 {len(rows)}개, 입력 {len(pairs)}개 -> {a.table}")
for k in sorted(t.resid, key=lambda x: tuple(int(v) for v in x.split(":"))):
    r = t.resid[k]
    dm = t.diff_margin.get(k)
    print(f"  BM:G={k:>7} n={r['n']:>4} p50={r['rel_p50']:.3f} p90={r['rel_p90']:.3f} "
          f"max={r['rel_max']:.3f}  diff_margin={'%.3f'%dm if dm else '전역'}")
print(f"  최적 max_m lookup: {t.best_max_m}")
print(f"  전역 차이 잔차 p50={t.meta.get('diff_margin_rel_p50'):.3f} "
      f"p90={t.meta.get('diff_margin_rel_p90'):.3f} (n={t.meta.get('diff_margin_n')})")

# oracle 대조: raw 안에서 선택기 규칙을 그대로 적용해 regret 을 미리 본다
reg, act, unsup = [], 0, 0
for pk, cands in pairs.items():
    q = cands.get("q_outer")
    if q is None: continue
    fq, yq = q
    tq = t.predict(fq)
    best_y = min(y for _, y in cands.values())
    if not np.isfinite(tq):
        unsup += 1; reg.append((yq - best_y) / best_y); continue
    margin = t.diff_error(fq) * tq
    cand_pred = []
    for name, (f, y) in cands.items():
        if name == "q_outer": continue
        p = t.predict(f)
        if np.isfinite(p): cand_pred.append((p, name, y))
    sel_y = yq
    if cand_pred:
        p, name, y = min(cand_pred)
        if (tq - p) > margin:
            sel_y = y; act += 1
    reg.append((sel_y - best_y) / best_y)
print(f"\n교정 입력에서의 선택 규칙 점검 (in-sample): regret 중앙값 {100*np.median(reg):.2f}% "
      f"p90 {100*np.quantile(reg,0.9):.2f}% 최악 {100*max(reg):.2f}%")
print(f"  non-Q 선택 {act}/{len(pairs)}, 외삽 폴백 {unsup}")
