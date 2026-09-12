#!/usr/bin/env python3
"""E3 — 교정된 선택기와 oracle 비교 (지시서 §11 E3, §7.7 진단 기준).

각 held-out 입력에서 기록한다:
  predicted winner / 실제 선택한 candidate / 모든 후보를 나중에 측정한 oracle winner /
  selected vs oracle regret / selected vs best Q-outer 시간 / RBC 고유 계획 사용 여부

oracle 은 결과 해석용이다. 온라인 선택 경로는 probe + 교정 표만 쓴다.
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
    ap.add_argument("--seeds", nargs="*", type=int,
                    default=[101, 102, 103, 104, 105, 106, 107, 108, 109])
    ap.add_argument("--shapes", nargs="*",
                    default=["512:8", "512:16", "2048:16", "256:16", "512:16:4", "512:16:16"],
                    help="nq:sel[:g] — g 생략 시 8")
    ap.add_argument("--patterns", nargs="*",
                    default=["common_private", "random", "clustered"])
    ap.add_argument("--max-m", nargs="*", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--nk", type=int, default=65536)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--enable-merge", action="store_true")
    ap.add_argument("--table", default="configs/plan_cost_table.json")
    ap.add_argument("--out", default="results/v02/select/e3_selector.json")
    a = ap.parse_args()

    table = select.CostTable.load(a.table)
    plan_cost = table.meta.get("plan_cost_us", {})
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    rec = dict(meta=dict(seeds=a.seeds, shapes=a.shapes, patterns=a.patterns,
                         max_m=a.max_m, nk=a.nk, d=a.d, replays=a.replays, reps=a.reps,
                         enable_merge=bool(a.enable_merge),
                         gpu=torch.cuda.get_device_name(0), sm=sm,
                         calibration_seeds=table.meta.get("seeds"),
                         table=a.table, time=time.strftime("%Y-%m-%dT%H:%M:%S")),
               cells=[], errors=[])

    for seed in a.seeds:
        for shape in a.shapes:
            parts = shape.split(":")
            nq, sel = int(parts[0]), int(parts[1])
            g = int(parts[2]) if len(parts) > 2 else 8
            for pattern in a.patterns:
                try:
                    s = data.make_synthetic(pattern, hkv=1, nq=nq, nk=a.nk, g=g, d=a.d,
                                            bk=128, sel=sel, seed=seed)
                except Exception as e:
                    rec["errors"].append(dict(seed=seed, shape=shape, pattern=pattern,
                                              stage="data",
                                              detail=f"{type(e).__name__}: {e}"))
                    continue
                ptr, ids = s["ptr"][0], s["block_ids"][0]
                q = torch.from_numpy(s["q"][0]).cuda().to(torch.bfloat16)
                k = torch.from_numpy(s["k"][0]).cuda().to(torch.bfloat16)
                v = torch.from_numpy(s["v"][0]).cuda().to(torch.bfloat16)
                qpos = torch.from_numpy(
                    np.ascontiguousarray(s["q_positions"], np.int32)).cuda()
                kw = dict(bk=128, causal=bool(s["causal"]), scale=float(s["scale"]))

                # --- 온라인 선택 (probe + 교정 표. 후보 전수 실행 없음) ---
                # max_m 은 교정 lookup 이 정한다. 서로 다른 max_m 은 BM 구간이 달라
                # 회귀 절편이 다르므로 예측값을 직접 비교하지 않는다.
                t0 = time.perf_counter()
                p_sel, dec_sel = select.choose_plan(
                    ptr, ids, bk=128, d=a.d, g=g, table=table,
                    enable_merge=a.enable_merge, plan_cost_us=plan_cost, nq=nq)
                sel_cpu_us = (time.perf_counter() - t0) * 1e6
                mm_sel = table.pick_max_m(g, nq)
                # §7.7 의 regret 기준은 GPU 시간으로 매겨지는데 같은 절의 선택 규칙은
                # extra_prepare (계획 CPU 비용 차이) 를 함께 본다. 두 지표가 서로 다른
                # 것을 재므로 GPU-only 규칙 (extra_prepare=0) 의 선택도 함께 기록한다.
                _, dec_gpu = select.choose_plan(
                    ptr, ids, bk=128, d=a.d, g=g, table=table,
                    enable_merge=a.enable_merge, plan_cost_us=None, nq=nq)

                # --- oracle: 모든 후보 x max_m 를 나중에 측정 ---
                oracle = []
                ref = None
                for name, pol in CAND.items():
                    for mm in a.max_m:
                        try:
                            pp = planner.plan_head(ptr, ids, bk=128, d=a.d, g=g, max_m=mm,
                                                   policy=pol, min_tasks=0)
                            dpp = runtime.upload_plan(pp, nq=nq)
                            out, _ = kernels.run_plan(dpp, q, k, v, qpos, **kw)
                            o = out.float()
                            if name == "q_outer" and mm == a.max_m[0]:
                                ref = o.clone()
                            rel = (float((o - ref).abs().max() / max(1e-6, float(ref.abs().max())))
                                   if ref is not None else -1.0)
                            if not torch.isfinite(o).all() or (ref is not None and rel > 5e-3):
                                rec["errors"].append(dict(seed=seed, shape=shape,
                                                          pattern=pattern, candidate=name,
                                                          max_m=mm, stage="numeric", rel=rel))
                                continue
                            us = float(np.median([
                                graph_us(lambda: kernels.run_plan(dpp, q, k, v, qpos, **kw),
                                         a.replays) for _ in range(a.reps)]))
                            oracle.append(dict(candidate=name, max_m=mm, us=us, rel=rel))
                        except Exception as e:
                            rec["errors"].append(dict(seed=seed, shape=shape, pattern=pattern,
                                                      candidate=name, max_m=mm, stage="run",
                                                      detail=f"{type(e).__name__}: {e}"))
                if not oracle:
                    del q, k, v, qpos
                    torch.cuda.empty_cache()
                    continue
                best = min(oracle, key=lambda x: x["us"])
                bestq = min((x for x in oracle if x["candidate"] == "q_outer"),
                            key=lambda x: x["us"])
                sel_name = dec_sel.selected
                sel_key = {"q_outer": "q_outer", "signature": "signature",
                           "bounded_merge": "bounded_merge"}[sel_name]
                got = [x for x in oracle if x["candidate"] == sel_key and x["max_m"] == mm_sel]
                sel_meas = got[0]["us"] if got else None
                rec["cells"].append(dict(
                    seed=seed, shape=shape, nq=nq, sel=sel, g=g, pattern=pattern,
                    predicted_winner=sel_name, selected_candidate=sel_key,
                    selected_max_m=mm_sel, selected_reason=dec_sel.reason,
                    predicted_us=dec_sel.predicted, selector_cpu_us=sel_cpu_us,
                    selector_cpu_breakdown=dec_sel.cpu_us,
                    error_margin_us=dec_sel.error_margin,
                    extra_prepare_us=dec_sel.extra_prepare_us,
                    selected_measured_us=sel_meas,
                    oracle_winner=best["candidate"], oracle_max_m=best["max_m"],
                    oracle_us=best["us"], best_q_outer_us=bestq["us"],
                    best_q_outer_max_m=bestq["max_m"],
                    regret=(None if sel_meas is None
                            else (sel_meas - best["us"]) / best["us"]),
                    vs_best_q_outer=(None if sel_meas is None
                                     else (sel_meas - bestq["us"]) / bestq["us"]),
                    rbc_plan_used=bool(sel_key != "q_outer"),
                    gpu_only_rule=dict(
                        selected=dec_gpu.selected, reason=dec_gpu.reason,
                        predicted=dec_gpu.predicted,
                        measured_us=next((x["us"] for x in oracle
                                          if x["candidate"] == {"q_outer": "q_outer",
                                                                "signature": "signature",
                                                                "bounded_merge": "bounded_merge"}
                                          [dec_gpu.selected] and x["max_m"] == mm_sel), None)),
                    oracle_table=oracle))
                del q, k, v, qpos
                torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    cells = rec["cells"]
    reg = [c["regret"] for c in cells if c["regret"] is not None]
    act = [c for c in cells if c["rbc_plan_used"]]
    print(f"셀 {len(cells)}개, 제외/오류 {len(rec['errors'])}건 -> {a.out}")
    gr = [(c["gpu_only_rule"]["measured_us"] - c["oracle_us"]) / c["oracle_us"]
          for c in cells if c["gpu_only_rule"]["measured_us"] is not None]
    ga = sum(1 for c in cells if c["gpu_only_rule"]["selected"] != "q_outer")
    if gr:
        print(f"[GPU-only 규칙 (§7.7 진단 기준)] regret 중앙값 {100*np.median(gr):.2f}% "
              f"/ p90 {100*np.quantile(gr,0.9):.2f}% / 최악 {100*max(gr):.2f}%")
        print(f"  RBC_active_fraction = {ga}/{len(cells)} = {ga/len(cells):.3f}")
    if reg:
        print(f"[production 규칙 (extra_prepare 포함)] regret 중앙값 {100*np.median(reg):.2f}% / p90 {100*np.quantile(reg,0.9):.2f}%"
              f" / 최악 {100*max(reg):.2f}%  (기준: 중앙값 3%, 최악 5%)")
        print(f"RBC_active_fraction = {len(act)}/{len(cells)} = {len(act)/len(cells):.3f}")
        cpu = [c["selector_cpu_us"] for c in cells]
        print(f"선택기 CPU 중앙값 {np.median(cpu):.0f}us (probe+선택된 계획 생성 포함)")


if __name__ == "__main__":
    main()
