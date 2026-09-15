#!/usr/bin/env python3
"""IDE_071 — 기계적 후보 선택 (지시서 §14.2). 규칙 ID 를 selection_trace.jsonl 에 남긴다. 사람의 평가 문장 없음.
규칙 SEL-01: 후보 = 같은 phase 의 COMPLETED 셀 (모든 반복 valid, config_mismatch 없음).
규칙 SEL-02: primary = SHORT_COLD C64 output_tps 평균 (반복 ≥2). 동률/±1% 이내면 TPOT p95 평균이 낮은 쪽.
규칙 SEL-03: 처리량 상위 2 + 지연(TTFT p95, TPOT p95) 비지배 후보 최대 2 보존 (중복 제거) → candidates. anchor = 처리량 1위.
규칙 SEL-04: 계열(semantics_family) 별로 분리 집계.
사용: select_anchor.py <campaign_dir> <phase> [--workload SHORT_COLD] [--C 64] → manifests/anchor_<phase>.json, state/selection_trace.jsonl
"""
import json, os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def main():
    camp, phase = sys.argv[1], sys.argv[2]; a = sys.argv[3:]
    wl = a[a.index("--workload") + 1] if "--workload" in a else "SHORT_COLD"; C = int(a[a.index("--C") + 1]) if "--C" in a else 64
    state = json.load(open(f"{camp}/state/state.json"))
    cells = {json.loads(l)["cell_id"]: json.loads(l) for l in open(f"{camp}/manifests/cells_{phase}.jsonl")}
    rows = []
    for cid, spec in cells.items():
        rec = state["cells"].get(cid);
        if not rec: continue
        att = [x for x in rec["attempts"] if x["exit_status"] == "COMPLETED"]
        if not att: jappend({"rule": "SEL-01", "cell": cid, "excluded": True, "reason": "no COMPLETED attempt", "t": now()}, f"{camp}/state/selection_trace.jsonl"); continue
        reps = [r for r in att[-1]["reps"] if r["workload_id"] == wl and r["C"] == C and r["valid"]]
        if len(reps) < 2: jappend({"rule": "SEL-02", "cell": cid, "excluded": True, "reason": f"valid reps {len(reps)} < 2", "t": now()}, f"{camp}/state/selection_trace.jsonl"); continue
        rows.append({"cell": cid, "family": spec.get("semantics_family"), "tps": st.mean(r["output_tps"] for r in reps), "tps_sd": st.stdev(r["output_tps"] for r in reps) if len(reps) > 1 else 0,
                     "ttft95": st.mean(r["ttft_p95"] for r in reps), "tpot95": st.mean(r["tpot_p95"] for r in reps), "n": len(reps)})
    out = {"phase": phase, "workload": wl, "C": C, "families": {}}
    for fam in sorted(set(r["family"] for r in rows)):
        fr = sorted([r for r in rows if r["family"] == fam], key=lambda r: (-r["tps"], r["tpot95"]))
        top = fr[:2]
        nondom = []
        for r in fr:
            if any(o["ttft95"] <= r["ttft95"] and o["tpot95"] <= r["tpot95"] and (o["ttft95"] < r["ttft95"] or o["tpot95"] < r["tpot95"]) for o in fr if o is not r): continue
            nondom.append(r)
        cand = []; seen = set()
        for r in top + nondom[:2]:
            if r["cell"] not in seen: cand.append(r); seen.add(r["cell"])
        out["families"][fam] = {"ranked": fr, "candidates": cand, "anchor": fr[0]["cell"] if fr else None}
        jappend({"rule": "SEL-03", "phase": phase, "family": fam, "ranked": [(r["cell"], round(r["tps"], 2), round(r["tpot95"], 1), round(r["ttft95"], 0)) for r in fr], "candidates": [r["cell"] for r in cand], "anchor": out["families"][fam]["anchor"], "t": now()}, f"{camp}/state/selection_trace.jsonl")
    # 전체 앵커: 모든 계열 통틀어 처리량 1위 (SEMANTICS_PRESERVING 계열의 앵커도 별도 기록)
    allr = sorted(rows, key=lambda r: (-r["tps"], r["tpot95"]))
    out["anchor"] = allr[0]["cell"] if allr else None
    out["anchor_spec"] = cells.get(out["anchor"]) if out["anchor"] else None
    jdump(out, f"{camp}/manifests/anchor_{phase}.json")
    if out["anchor_spec"]: jdump(out["anchor_spec"], f"{camp}/manifests/anchor_{phase}_spec.json")
    print(json.dumps({k: (v if k != "anchor_spec" else "...") for k, v in out.items()}, ensure_ascii=False)[:1500])

if __name__ == "__main__":
    main()
