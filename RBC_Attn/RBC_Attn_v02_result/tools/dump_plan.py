#!/usr/bin/env python3
"""분할 전후 descriptor·비용·선택 이유를 덤프한다 (지시서 §10.1, §4.4).

GPU 없이 동작한다. 선택기의 예측값은 교정 표가 있을 때만 표시한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbc import data, planner, select  # noqa: E402

CAND = {"q_outer": "q_outer", "signature": "signature_only", "bounded_merge": "rbc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="common_private")
    ap.add_argument("--capture")
    ap.add_argument("--nq", type=int, default=512)
    ap.add_argument("--nk", type=int, default=65536)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--bk", type=int, default=128)
    ap.add_argument("--sel", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-m", nargs="*", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--splits", nargs="*", default=["none", "one_wave", "two_waves"])
    ap.add_argument("--sm", type=int, default=132)
    ap.add_argument("--table", default="configs/plan_cost_table.json")
    ap.add_argument("--verify", action="store_true", help="정확 분할 계약을 검사한다")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.capture:
        s = data.load_capture(a.capture)
        src = f"capture:{os.path.basename(a.capture)}"
    else:
        s = data.make_synthetic(a.pattern, hkv=1, nq=a.nq, nk=a.nk, g=a.g, d=a.d,
                                bk=a.bk, sel=a.sel, seed=a.seed)
        src = f"synthetic:{a.pattern}:seed{a.seed}"
    ptr, ids = s["ptr"][0], s["block_ids"][0]
    nq = len(ptr) - 1
    split_map = {"none": 0, "one_wave": a.sm, "two_waves": 2 * a.sm}

    table = None
    if os.path.exists(a.table):
        try:
            table = select.CostTable.load(a.table)
        except Exception as e:
            print(f"교정 표 무시 ({type(e).__name__}: {e})")

    rows = []
    for name, pol in CAND.items():
        for mm in a.max_m:
            for sp in a.splits:
                p = planner.plan_head(ptr, ids, bk=a.bk, d=a.d, g=a.g, max_m=mm,
                                      policy=pol, min_tasks=split_map[sp])
                st = p.stats(n_edges=len(ids))
                f = select.features_from_plan(p, name=name, d=a.d, bk=a.bk)
                pred = None
                if table is not None and table.supports(f):
                    pred = table.predict(f)
                ok, why = None, None
                if a.verify:
                    ok, why = planner.verify_exact_partition(p, ptr, ids)
                rows.append(dict(candidate=name, max_m=mm, split=sp, BM=f.bm,
                                 n_tasks=st["n_tasks"], slots=st["query_task_incidences"],
                                 partial_slots=st["partial_slots"],
                                 direct_rows=st["direct_rows"],
                                 multi_owner_rows=st["multi_owner_rows"],
                                 empty_rows=st["empty_rows"],
                                 kv_sum_k=f.sum_k, kv_max_k=f.max_k,
                                 unique_blocks=f.unique_blocks,
                                 padded_units=st["padded_units"],
                                 padded_ratio=st.get("padded_ratio"),
                                 plan_ms=p.plan_ms, predicted_us=pred,
                                 exact_partition=ok, exact_partition_note=why))

    probe = planner.probe_head(ptr, ids, bk=a.bk, d=a.d, g=a.g, max_m=a.max_m[0])
    rec = dict(meta=dict(source=src, nq=nq, nk=s["k"].shape[1], g=a.g, d=a.d, bk=a.bk,
                         sel=a.sel, sm=a.sm, table=a.table if table else None,
                         edges=int(len(ids))),
               probe=probe, rows=rows)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rec, open(a.out, "w"), indent=1)

    print(f"입력 {src} Nq={nq} G={a.g} D={a.d} BK={a.bk} edges={len(ids)}")
    hdr = (f"{'cand':14}{'M':>4}{'split':11}{'BM':>4}{'tasks':>7}{'slots':>7}{'part':>6}"
           f"{'direct':>7}{'multi':>6}{'sum_k':>8}{'max_k':>6}{'uniq':>7}{'padded':>10}"
           f"{'plan_ms':>8}{'pred_us':>9}")
    print(hdr)
    for r in rows:
        pu = "        -" if r["predicted_us"] is None else f"{r['predicted_us']:9.1f}"
        print(f"{r['candidate']:14}{r['max_m']:>4}{r['split']:11}{r['BM']:>4}"
              f"{r['n_tasks']:>7}{r['slots']:>7}{r['partial_slots']:>6}"
              f"{r['direct_rows']:>7}{r['multi_owner_rows']:>6}{r['kv_sum_k']:>8}"
              f"{r['kv_max_k']:>6}{r['unique_blocks']:>7}{r['padded_units']:>10}"
              f"{r['plan_ms']:>8.2f}{pu}")
    if a.verify:
        bad = [r for r in rows if r["exact_partition"] is not True]
        print(f"\n정확 분할 검사: {len(rows)-len(bad)}/{len(rows)} 통과")
        for r in bad:
            print(f"  실패 {r['candidate']} M{r['max_m']} {r['split']}: {r['exact_partition_note']}")
    if a.out:
        print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
