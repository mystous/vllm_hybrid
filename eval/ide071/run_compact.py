#!/usr/bin/env python3
"""IDE_071 — compact 지시서 (cpu_offload_no_02_compact) 실행기. 기존 하네스(run_cell.py) 재사용, 실행량 상한 계수.
대조군 B3/B4 는 이미 같은 워크로드(SHORT_COLD C64, flush, ignore_eos)로 측정한 P2_cf1_skip1_pin1_def8 / R04_d4_epoch 를 재사용 (사용자 지시: 이미 수행한 것은 반복하지 않음).
순서: S1~S5 각 1회 → 조건부 S6 → 계열별 확인 대상 선정(규칙 §5) → 부모 재확인 1회(후보 있는 계열만) → 대상 새 프로세스 3회 + ALT 1회 + GSM40 → 연속 부하 1회 (대상B) → 종료.
상태: <camp>/compact/compact_state.json, selection_trace.jsonl
"""
import json, os, subprocess, sys, time, glob, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = f"{REPO}/eval/results/IDE_071_20260916"; CD = f"{CAMP}/compact"; os.makedirs(CD, exist_ok=True)
SP = f"{CD}/compact_state.json"; TR = f"{CD}/selection_trace.jsonl"

EPOCH_ENV = {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1", "KT_AVX_RB": "1", "KT_AVX_PF": "2", "KT_FUSE_QIN": "1", "KT_AMX_MIN_QLEN": "1000000", "KT_AMX_MIN_ROWS": "3", "KT_CALLBACK_FREE": "1", "KT_COLD_DEFER": "1", "KT_COLD_TAU": "0.25"}
EPOCH_ARGS = {"init-expert-location": "/models/kt/ide070/hotmap_mixed_0.25.json", "kt-max-deferred-experts-per-token": 8, "mem-fraction-static": 0.94, "chunked-prefill-size": 4096, "kv-cache-dtype": "fp8_e5m2", "cuda-graph-max-bs": 224, "cuda-graph-bs": [32, 64, 96, 128, 160, 192, 224], "max-total-tokens": 143360}
B3 = {"env": {"KT_CALLBACK_FREE": "1", "KT_CF_SKIP_EMPTY_IMM": "1"}, "server_args": {"kt-max-deferred-experts-per-token": 8}, "pin_nonkt": True, "precision_family": "GPU_FP8_CPU_INT4_KV_BF16", "semantics_family": "SEMANTICS_UNVERIFIED"}
B4 = {"env": dict(EPOCH_ENV), "server_args": dict(EPOCH_ARGS), "pin_nonkt": False, "precision_family": "GPU_FP8_CPU_INT4_KV_FP8E5M2", "semantics_family": "SEMANTICS_UNVERIFIED"}
REUSE = {"B3": ("P2_cf1_skip1_pin1_def8", "a1"), "B4": ("R04_d4_epoch", "a1")}

def load(): return json.load(open(SP)) if os.path.exists(SP) else {"t_start": now(), "budget": {"bench": 0, "bench_max": 20, "boot": 0, "boot_max": 14, "diag": 0, "load": 0, "retry": 0, "gsm": 0}, "cells": {}, "current": None, "status": "RUNNING", "reused": {}}
def save(s): jdump(s, SP)
def trace(ev): ev["t"] = now(); jappend(ev, TR)

def cellspec(cid, parent, base, env_add=None, sa_add=None, pin=None, patch=None, workloads=None, note=""):
    env = dict(base["env"]); env.update(env_add or {}); sa = dict(base["server_args"]); sa.update(sa_add or {})
    c = {"cell_id": cid, "phase_id": "COMPACT", "parent_cell_id": parent, "change_kind": "PATCH" if patch else "CONFIG", "semantics_family": base["semantics_family"], "precision_family": base["precision_family"],
         "workloads": workloads or [{"workload_id": "SHORT_COLD", "C": 64, "reps": 1}], "env": env, "server_args": sa, "gpus": "0,1,2,3", "pin_nonkt": base["pin_nonkt"] if pin is None else pin,
         "boot_timeout": 600, "bench_timeout": 900, "note": note}
    if patch: c["patch"] = patch; c["expected_slots"] = 5952
    c["config_hash"] = config_hash({"env": env, "sa": sa, "pin": c["pin_nonkt"], "patch": patch}); return c

def run(s, spec, attempt="a1", n_bench=None, kind="bench"):
    """셀 실행 + 예산 계수. 반환 status dict."""
    cid = spec["cell_id"]; s["current"] = {"cell": cid, "attempt": attempt, "t": now()}; save(s)
    nb = n_bench if n_bench is not None else sum(w.get("reps", 1) for w in spec["workloads"])
    s["budget"]["boot"] += 1; s["budget"]["bench" if kind == "bench" else kind] += nb; save(s)
    p = f"{CD}/specs/{cid}.json"; os.makedirs(os.path.dirname(p), exist_ok=True); jdump(spec, p)
    with open(f"{CD}/{cid}.{attempt}.runner.log", "w") as lf:
        subprocess.run([sys.executable, f"{HERE}/run_cell.py", CD, p, attempt], stdout=lf, stderr=subprocess.STDOUT, timeout=2 * 3600)
    sp = f"{CD}/{cid}/{attempt}/status.json"; stt = json.load(open(sp)) if os.path.exists(sp) else {"exit_status": "FAILED_RUNTIME", "reps": [], "errors": []}
    rec = s["cells"].setdefault(cid, {"attempts": [], "parent": spec.get("parent_cell_id")})
    rec["attempts"].append({"attempt": attempt, "exit_status": stt.get("exit_status"), "reps": stt.get("reps"), "boot_seconds": (stt.get("boot") or {}).get("boot_seconds"), "config_mismatch": stt.get("config_mismatch"), "t": now()})
    s["current"] = None; save(s); return stt

def reps_of(stt, wl="SHORT_COLD"): return [r for r in (stt.get("reps") or []) if r["valid"] and r["workload_id"] == wl]

def main():
    s = load(); save(s)
    # 0. 재사용 대조군 (원본 캠페인에서 같은 워크로드로 측정)
    for k, (cid, att) in REUSE.items():
        stt = json.load(open(f"{CAMP}/{cid}/{att}/status.json")); rr = reps_of(stt)
        s["reused"][k] = {"source_cell": cid, "attempt": att, "runs": [(r["rep_id"], r["output_tps"], r["ttft_p95"], r["tpot_p95"]) for r in rr], "mean": st.mean(r["output_tps"] for r in rr)}
    trace({"rule": "REUSE", "detail": s["reused"]}); save(s)
    # 1. S1~S5 각 1회
    S = {
      "S1_b4_pin": cellspec("S1_b4_pin", "B4", B4, pin=True, note="B4 + 비-kt 스레드 재배치 (pin_nonkt: kt 물리 코어 0-47,56-103 확인 후 남은 물리 48-55,104-111 + 그 HT 형제 160-167,216-223)"),
      "S2_b3_v2_nu5952": cellspec("S2_b3_v2_nu5952", "B3", B3, sa_add={"init-expert-location": "/models/kt/ide070/hotmap_v2.json"}, env_add={"KT_GPU_EXPERTS_PER_LAYER": "/models/kt/ide070/layer_budget_5952.json"}, patch="per_layer", note="B3 + hotmap_v2 + 층별 5,952 (복합 변경)"),
      "S3_b4_graph64": cellspec("S3_b4_graph64", "B4", B4, sa_add={"cuda-graph-max-bs": 64, "cuda-graph-bs": [32, 64]}, note="B4, graph max 64, 목록 [32,64]"),
      "S4_b4_kv40960": cellspec("S4_b4_kv40960", "B4", B4, sa_add={"max-total-tokens": 40960}, note="B4, max-total-tokens 40960 (fp8_e5m2 유지)"),
      "S5_b4_chunk2048": cellspec("S5_b4_chunk2048", "B4", B4, sa_add={"chunked-prefill-size": 2048}, note="B4, chunked-prefill 2048"),
    }
    for cid, spec in S.items():
        if cid in s["cells"]: continue
        stt = run(s, spec)
        if stt.get("exit_status") in ("FAILED_BOOT", "FAILED_RUNTIME") and s["budget"]["retry"] < 2:
            s["budget"]["retry"] += 1; save(s); trace({"rule": "RETRY", "cell": cid, "reason": stt.get("exit_status")}); run(s, spec, "a2")
    # 2. 선택 규칙 §5 (계열별)
    def last_ok(cid):
        rec = s["cells"].get(cid);
        if not rec: return None
        a = rec["attempts"][-1]; rr = [r for r in (a.get("reps") or []) if r["valid"] and r["workload_id"] == "SHORT_COLD"]
        return (a, rr) if a["exit_status"] == "COMPLETED" and rr and not a.get("config_mismatch") else None
    fam = {"B3": ["S2_b3_v2_nu5952"], "B4": ["S1_b4_pin", "S3_b4_graph64", "S4_b4_kv40960", "S5_b4_chunk2048"]}
    qualified = {}
    for par, kids in fam.items():
        pr = [x[1] for x in s["reused"][par]["runs"]]; pm = st.mean(pr); delta = max(0.05, abs(pr[0] - pr[1]) / st.mean(pr[:2]))
        cands = []
        for k in kids:
            lo = last_ok(k)
            if not lo: trace({"rule": "SEL-1", "family": par, "cell": k, "excluded": True, "reason": "not COMPLETED/valid/config"}); continue
            tps = lo[1][0]["output_tps"]; gain = tps / pm - 1
            trace({"rule": "SEL-3/4", "family": par, "cell": k, "tps": tps, "parent_mean": pm, "delta": delta, "gain": gain, "qualified": gain > delta, "ttft95": lo[1][0]["ttft_p95"], "tpot95": lo[1][0]["tpot_p95"]})
            if gain > delta: cands.append((tps, k))
        qualified[par] = sorted(cands, reverse=True)
    # 3. S6: B4 계열에서 조건 충족 변경 ≥2 이고 호환되면 결합 1회
    q4 = [k for _, k in qualified["B4"]]
    s6_changes = [k for k in q4 if k in ("S1_b4_pin", "S3_b4_graph64", "S4_b4_kv40960", "S5_b4_chunk2048")]
    if len(s6_changes) >= 2 and "S6_b4_combo" not in s["cells"]:
        sa = {}; pin = False
        if "S1_b4_pin" in s6_changes: pin = True
        if "S3_b4_graph64" in s6_changes: sa.update({"cuda-graph-max-bs": 64, "cuda-graph-bs": [32, 64]})
        if "S4_b4_kv40960" in s6_changes: sa.update({"max-total-tokens": 40960})
        if "S5_b4_chunk2048" in s6_changes: sa.update({"chunked-prefill-size": 2048})
        spec6 = cellspec("S6_b4_combo", "B4", B4, sa_add=sa, pin=pin, note="결합: " + ",".join(s6_changes)); trace({"rule": "S6", "changes": s6_changes})
        stt = run(s, spec6); lo = last_ok("S6_b4_combo")
        if lo:
            pm = s["reused"]["B4"]["mean"]; tps = lo[1][0]["output_tps"]; pr = [x[1] for x in s["reused"]["B4"]["runs"]]; delta = max(0.05, abs(pr[0] - pr[1]) / st.mean(pr[:2]))
            trace({"rule": "SEL-4(S6)", "cell": "S6_b4_combo", "tps": tps, "gain": tps / pm - 1, "delta": delta, "qualified": tps / pm - 1 > delta})
            if tps / pm - 1 > delta: qualified["B4"] = sorted(qualified["B4"] + [(tps, "S6_b4_combo")], reverse=True)
    else:
        trace({"rule": "S6", "state": "NOT_TRIGGERED", "qualified_B4": s6_changes})
    # 4. 계열별 확인 대상 ≤1
    targets = {par: (q[0][1] if q else None) for par, q in qualified.items()}
    trace({"rule": "SEL-6", "targets": targets}); s["targets"] = targets; save(s)
    # 5. 부모 재확인 1회 (후보 있는 계열만) + 대상 새 프로세스 3회 + ALT 1회 + GSM40
    for par, tgt in targets.items():
        if not tgt: continue
        base = B3 if par == "B3" else B4
        if f"{par}_recheck" not in s["cells"]:
            run(s, cellspec(f"{par}_recheck", None, base, note=f"부모 {par} 재확인 1회 (새 프로세스)"))
        spec = json.load(open(f"{CD}/specs/{tgt}.json")); cf = dict(spec); cf["cell_id"] = f"{tgt}__confirm"; cf["parent_cell_id"] = tgt
        cf["workloads"] = [{"workload_id": "SHORT_COLD", "C": 64, "reps": 3}, {"workload_id": "COMPACT_ALT", "C": 64, "reps": 1}]; cf["gsm40"] = True
        if cf["cell_id"] not in s["cells"]: run(s, cf, n_bench=4)
    # 6. 연속 부하 1회: 대상 B (B4 계열) 없으면 대상 A, 둘 다 없으면 B4 재사용 구성
    tgt = targets.get("B4") or targets.get("B3")
    ld_base = json.load(open(f"{CD}/specs/{tgt}.json")) if tgt else cellspec("B4_load_base", "B4", B4)
    ld = dict(ld_base); ld["cell_id"] = (tgt or "B4") + "__load1024"; ld["workloads"] = [{"workload_id": "COMPACT_LOAD", "C": 64, "n": 1024, "reps": 1}]; ld["bench_timeout"] = 1800; ld.pop("gsm40", None)
    if ld["cell_id"] not in s["cells"]: run(s, ld, n_bench=0, kind="load")
    s["status"] = "BOUNDED_SEARCH_COMPLETE" if s["budget"]["bench"] <= 20 and s["budget"]["boot"] <= 14 else "SEARCH_BUDGET_REACHED"
    s["t_end"] = now(); save(s); trace({"rule": "END", "status": s["status"], "budget": s["budget"]}); print(s["status"], s["budget"])

if __name__ == "__main__":
    main()
