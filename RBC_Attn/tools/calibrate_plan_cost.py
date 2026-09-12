#!/usr/bin/env python3
"""P4 교정 — 교정 seed(0~4) 에서 후보 계획을 실제 실행해 시간 모델을 적합한다 (§7.5).

- NCU 로 적합하지 않는다. profiling 없는 CUDA graph timing 을 쓴다.
- 평가 seed(101~109) 는 절대 쓰지 않는다.
- 후보는 q_outer / signature / bounded_merge 셋만 (§7.2).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbc import data, kernels, planner, runtime, select  # noqa: E402

CAND = {"q_outer": "q_outer", "signature": "signature_only", "bounded_merge": "rbc"}


def graph_us(fn, replays):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    try:
        with torch.cuda.graph(gr):
            fn()
        rep = gr.replay
    except Exception:
        rep = fn
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(replays):
        rep()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / replays * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--nq", nargs="*", type=int, default=[512, 2048])
    ap.add_argument("--sel", nargs="*", type=int, default=[8, 16, 32])
    ap.add_argument("--patterns", nargs="*",
                    default=["common_private", "random", "clustered"])
    ap.add_argument("--max-m", nargs="*", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--nk", type=int, default=65536)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--gs", nargs="*", type=int, default=None,
                    help="G 축. 주면 --g 무시. 평가 전용으로 남길 G 는 넣지 않는다")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--table", default="configs/plan_cost_table.json")
    ap.add_argument("--raw", default="results/v02/select/calibration_raw.json")
    a = ap.parse_args()

    sm = torch.cuda.get_device_properties(0).multi_processor_count
    rows, raw, plan_us = [], [], {k: [] for k in CAND}
    # 입력별 후보 묶음 — 후보 '차이' 의 예측 잔차를 구해 error_margin 근거로 쓴다
    pairs = {}
    bestm_raw = {}

    GS = a.gs if a.gs else [a.g]
    for seed in a.seeds:
        for nq in a.nq:
          for gg in GS:
            for sel in a.sel:
                for pattern in a.patterns:
                    try:
                        s = data.make_synthetic(pattern, hkv=1, nq=nq, nk=a.nk, g=gg,
                                                d=a.d, bk=128, sel=sel, seed=seed)
                    except Exception as e:
                        raw.append(dict(seed=seed, nq=nq, g=gg, sel=sel, pattern=pattern,
                                        error=f"data: {type(e).__name__}: {e}"))
                        continue
                    ptr, ids = s["ptr"][0], s["block_ids"][0]
                    q = torch.from_numpy(s["q"][0]).cuda().to(torch.bfloat16)
                    k = torch.from_numpy(s["k"][0]).cuda().to(torch.bfloat16)
                    v = torch.from_numpy(s["v"][0]).cuda().to(torch.bfloat16)
                    qpos = torch.from_numpy(
                        np.ascontiguousarray(s["q_positions"], np.int32)).cuda()
                    kw = dict(bk=128, causal=bool(s["causal"]), scale=float(s["scale"]))
                    for name, pol in CAND.items():
                        for mm in a.max_m:
                            try:
                                t0 = time.perf_counter()
                                p = planner.plan_head(ptr, ids, bk=128, d=a.d, g=gg,
                                                      max_m=mm, policy=pol, min_tasks=0)
                                plan_us[name].append((time.perf_counter() - t0) * 1e6)
                                dp = runtime.upload_plan(p, nq=nq)
                                us = float(np.median([
                                    graph_us(lambda: kernels.run_plan(
                                        dp, q, k, v, qpos, **kw), a.replays)
                                    for _ in range(a.reps)]))
                                f = select.features_from_plan(p, name=name, d=a.d, bk=128)
                                rows.append((f, us))
                                pk = (seed, nq, gg, sel, pattern)
                                cur = pairs.setdefault(pk, {})
                                if name not in cur or us < cur[name][1]:
                                    cur[name] = (f, us)   # 후보별 최적 max_m
                                bl = bestm_raw.setdefault(pk, [])
                                bl.append((us, mm, gg, nq))
                                raw.append(dict(seed=seed, nq=nq, g=gg, sel=sel,
                                                pattern=pattern,
                                                candidate=name, max_m=mm, bm=f.bm,
                                                measured_us=us, n_tasks=f.n_tasks,
                                                sum_k=f.sum_k, max_k=f.max_k,
                                                padded=f.padded, multi_rows=f.multi_rows,
                                                unique_blocks=f.unique_blocks,
                                                empty_rows=f.empty_rows,
                                                slots=f.slots, max_q=f.max_q,
                                                direct_rows=f.direct_rows,
                                                k_p50=f.k_p50, k_p90=f.k_p90,
                                                full_ratio=f.full_ratio))
                            except Exception as e:
                                raw.append(dict(seed=seed, nq=nq, g=gg, sel=sel,
                                                pattern=pattern, candidate=name, max_m=mm,
                                                error=f"{type(e).__name__}: {e}"))
                    del q, k, v, qpos
                    torch.cuda.empty_cache()

    meta = dict(seeds=a.seeds, nq=a.nq, sel=a.sel, patterns=a.patterns, max_m=a.max_m,
                nk=a.nk, g=(a.gs if a.gs else [a.g]), d=a.d, bk=128,
                replays=a.replays, reps=a.reps,
                gpu=torch.cuda.get_device_name(0), sm=sm, n_rows=len(rows),
                timing="cuda graph replay, profiling 없음",
                plan_cost_us={kk: float(np.median(vv)) for kk, vv in plan_us.items() if vv},
                time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    bestm = {}
    for pk, items in bestm_raw.items():
        us, mm, gg, nq_ = min(items)
        bestm.setdefault(select.CostTable.shape_key(gg, nq_), []).append(mm)
    table = select.fit_table(rows, sm, meta=meta, pairs=pairs, best_max_m=bestm)
    table.save(a.table)
    os.makedirs(os.path.dirname(a.raw) or ".", exist_ok=True)
    json.dump(dict(meta=meta, raw=raw), open(a.raw, "w"), indent=1)

    print(f"교정 표본 {len(rows)}개 -> {a.table}")
    # 표 키는 "BM:G" 다 (G 가 프로그램당 row 수를 바꾸므로 함께 묶는다)
    for key in sorted(table.resid, key=lambda x: tuple(int(v) for v in x.split(":"))):
        r = table.resid[key]
        print(f"  BM:G={key:>7} n={r['n']:>4} 상대잔차 p50={r['rel_p50']:.3f} "
              f"p90={r['rel_p90']:.3f} max={r['rel_max']:.3f}")
    print(f"  계획 CPU 중앙값(us): {meta['plan_cost_us']}")
    print(f"  최적 max_m lookup: {table.best_max_m}")
    dm = table.meta.get("diff_margin_rel_p90")
    if dm is not None:
        print(f"  후보 차이 예측 잔차: p50={table.meta['diff_margin_rel_p50']:.3f} "
              f"p90={dm:.3f} (n={table.meta['diff_margin_n']})")


if __name__ == "__main__":
    main()
