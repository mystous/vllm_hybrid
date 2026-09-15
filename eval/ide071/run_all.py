#!/usr/bin/env python3
"""IDE_071 — 캠페인 드라이버. manifests/cells_<phase>.jsonl 의 셀을 순서대로 실행 (고정 seed 무작위화, 6셀 또는 90분마다 앵커 재측정).
state/state.json (atomic), state/journal.jsonl (append-only). 셀 실패는 기록 후 다음 셀. 같은 config 재시도 최대 2회 (FAILED_BOOT/FAILED_RUNTIME 만).
사용: run_all.py <campaign_dir> <phase> [--anchor-cell R00_r0_def4] [--no-shuffle] [--only id1,id2] [--retry]
"""
import json, os, random, subprocess, sys, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

HERE = os.path.dirname(os.path.abspath(__file__))

def load_state(camp):
    p = f"{camp}/state/state.json"
    if os.path.exists(p): return json.load(open(p))
    return {"campaign_id": os.path.basename(camp.rstrip("/")), "t_start": now(), "cells": {}, "current": None, "phase": None, "history": []}

def save_state(camp, st): jdump(st, f"{camp}/state/state.json")
def journal(camp, ev): ev["t"] = now(); jappend(ev, f"{camp}/state/journal.jsonl")

def run_one(camp, cell, attempt):
    cell_path = f"{camp}/manifests/specs/{cell['cell_id']}.json"; os.makedirs(os.path.dirname(cell_path), exist_ok=True); jdump(cell, cell_path)
    log_path = f"{camp}/{cell['cell_id']}/{attempt}.runner.log"; os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as lf:
        r = subprocess.run([sys.executable, f"{HERE}/run_cell.py", camp, cell_path, attempt], stdout=lf, stderr=subprocess.STDOUT, timeout=cell.get("cell_timeout", 4 * 3600))
    sp = f"{camp}/{cell['cell_id']}/{attempt}/status.json"
    return json.load(open(sp)) if os.path.exists(sp) else {"state": "FAILED_RUNTIME", "exit_status": "FAILED_RUNTIME", "reps": [], "errors": [{"msg": "no status.json"}]}

def main():
    camp = sys.argv[1]; phase = sys.argv[2]; a = sys.argv[3:]
    anchor_id = a[a.index("--anchor-cell") + 1] if "--anchor-cell" in a else None
    only = set(a[a.index("--only") + 1].split(",")) if "--only" in a else None
    os.makedirs(f"{camp}/state", exist_ok=True)
    cells = [json.loads(l) for l in open(f"{camp}/manifests/cells_{phase}.jsonl")]
    if only: cells = [c for c in cells if c["cell_id"] in only]
    st = load_state(camp); st["phase"] = phase
    # 이미 종료 상태인 셀은 건너뜀 (재개). --retry 면 FAILED_BOOT/FAILED_RUNTIME 을 attempt+1 로 재시도 (최대 a3)
    order = list(range(len(cells)))
    if "--no-shuffle" not in a:
        random.Random(20260916 + sum(map(ord, phase))).shuffle(order)
    anchor = next((c for c in cells if c["cell_id"] == anchor_id), None) if anchor_id else None
    journal(camp, {"event": "phase_start", "phase": phase, "n_cells": len(cells), "order": [cells[i]["cell_id"] for i in order]})
    n_since_anchor = 0; t_anchor = time.monotonic(); anchor_k = 0
    for idx in order:
        c = cells[idx]; cid = c["cell_id"]; rec = st["cells"].get(cid, {"attempts": []})
        done = [x for x in rec["attempts"] if x["exit_status"] in ("COMPLETED", "FAILED_REQUESTS", "INVALID_CONFIG", "UNSUPPORTED", "NO_EFFECTIVE_PATH") or x["exit_status"].startswith("BLOCKED")]
        fails = [x for x in rec["attempts"] if x["exit_status"] in ("FAILED_BOOT", "FAILED_RUNTIME", "TIMEOUT")]
        if done and "--retry" not in a: continue
        if fails and ("--retry" not in a or len(fails) >= 3): continue
        for rf in c.get("requires_files", []):
            if not os.path.exists(rf):
                rec["attempts"].append({"attempt": "a0", "exit_status": "BLOCKED_DEPENDENCY", "reason": f"missing {rf}", "t": now()}); st["cells"][cid] = rec; save_state(camp, st)
                journal(camp, {"event": "cell_blocked", "cell": cid, "reason": f"missing {rf}"}); break
        else:
            # 앵커 재측정 규칙: 6셀 또는 90분
            if anchor and (n_since_anchor >= 6 or time.monotonic() - t_anchor > 5400) and cid != anchor_id:
                anchor_k += 1; aid = f"{anchor_id}__anchor{anchor_k}"; ac = dict(anchor); ac["cell_id"] = aid; ac["note"] = "앵커 재측정 (6셀/90분 규칙)"; ac["workloads"] = [{"workload_id": "SHORT_COLD", "C": 64, "reps": 2}]
                st["current"] = {"cell": aid, "attempt": "a1", "t": now()}; save_state(camp, st); journal(camp, {"event": "anchor_start", "cell": aid})
                s2 = run_one(camp, ac, "a1"); st["cells"][aid] = {"attempts": [{"attempt": "a1", "exit_status": s2.get("exit_status", s2["state"]), "reps": s2.get("reps"), "t": now()}], "anchor_of": anchor_id}
                journal(camp, {"event": "anchor_end", "cell": aid, "status": s2.get("exit_status")}); n_since_anchor = 0; t_anchor = time.monotonic(); save_state(camp, st)
            attempt = f"a{len(rec['attempts']) + 1}"
            st["current"] = {"cell": cid, "attempt": attempt, "phase": phase, "t": now()}; save_state(camp, st); journal(camp, {"event": "cell_start", "cell": cid, "attempt": attempt})
            try: s2 = run_one(camp, c, attempt)
            except subprocess.TimeoutExpired: s2 = {"state": "TIMEOUT", "exit_status": "TIMEOUT", "reps": [], "errors": [{"msg": "cell_timeout"}]}; stop_server()
            ex = s2.get("exit_status", s2["state"])
            if ex in ("FAILED_BOOT",) and c.get("expected_effect") == "UNSUPPORTED": ex = "UNSUPPORTED"
            rec["attempts"].append({"attempt": attempt, "exit_status": ex, "reps": s2.get("reps"), "n_errors": len(s2.get("errors", [])), "t": now(), "config_mismatch": s2.get("config_mismatch")})
            st["cells"][cid] = rec; st["history"].append({"cell": cid, "attempt": attempt, "status": ex, "t": now()}); st["current"] = None; save_state(camp, st)
            journal(camp, {"event": "cell_end", "cell": cid, "attempt": attempt, "status": ex, "reps": [(r["rep_id"], r["output_tps"], r["valid"]) for r in s2.get("reps", [])]})
            n_since_anchor += 1
    journal(camp, {"event": "phase_end", "phase": phase}); st["phase_done"] = st.get("phase_done", []) + [phase]; save_state(camp, st)
    print("phase done", phase)

if __name__ == "__main__":
    main()
