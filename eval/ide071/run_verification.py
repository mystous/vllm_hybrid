#!/usr/bin/env python3
"""IDE_072 — cpu_offload_no_03_compact_verification 실행기. 기존 run_cell.py 재사용.
단일 실행량 원장 execution_events.jsonl (run_id, kind, cell, attempt, parent, config_hash, input_hash, 예약/종료 상태·전송 수), branch_trace.jsonl (조건부 분기 입력·규칙·결과).
순서 (§4.2): V0 첫 부팅(smoke·warmup·SHORT_COLD 1·REPLAY) → V1 → V2(+GSM20) → Q0(GSM20 만) → V0 확인 부팅(SHORT_COLD 3·ALT 1·GSM20) → V1 확인 부팅 → LOAD(대상 규칙 §3.2) → 조건부 D0.
예산: PERF 11 / REPLAY 3 / LOAD 1 / DIAG 1 / RETRY 2 / 세션 18 / 부팅 10 / GSM 80.
"""
import json, os, subprocess, sys, time, glob, hashlib
os.environ["IDE071_FEAT"] = os.path.expanduser("~/projects/vllm_hybrid/shadow_assists/features/IDE_072")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = f"{REPO}/eval/results/IDE_072_20260916"; os.makedirs(f"{CAMP}/state", exist_ok=True)
SP = f"{CAMP}/state/state.json"; LEDGER = f"{CAMP}/state/execution_events.jsonl"; BT = f"{CAMP}/state/branch_trace.jsonl"
BUDGET = {"PERF": 11, "REPLAY": 3, "LOAD": 1, "DIAG": 1, "RETRY": 2, "SESSIONS": 18, "BOOT": 10, "GSM_Q": 80}

BASE_SA = {"init-expert-location": "/models/kt/ide070/hotmap_v2.json", "kt-max-deferred-experts-per-token": 8}
V0 = {"env": {"KT_CALLBACK_FREE": "1", "KT_CF_SKIP_EMPTY_IMM": "1", "KT_GPU_EXPERTS_PER_LAYER": "/models/kt/ide070/layer_budget_5952.json"}, "sa": dict(BASE_SA), "pin": True, "patch": "per_layer"}
V1 = {"env": dict(V0["env"], KT_AVX_RB="0"), "sa": dict(BASE_SA), "pin": True, "patch": "per_layer"}
V2 = {"env": {"KT_CALLBACK_FREE": "1", "KT_GPU_EXPERTS_PER_LAYER": "/models/kt/ide070/layer_budget_5952.json"}, "sa": dict(BASE_SA), "pin": True, "patch": "per_layer"}
Q0 = {"env": dict(V2["env"]), "sa": dict(BASE_SA, **{"kt-max-deferred-experts-per-token": 0}), "pin": True, "patch": "per_layer"}
CFG = {"V0": V0, "V1": V1, "V2": V2, "Q0": Q0}

def load():
    if os.path.exists(SP): return json.load(open(SP))
    return {"campaign_id": "IDE_072_20260916", "t_start": now(), "status": "RUNNING", "cells": {}, "current": None, "usage": {k: 0 for k in BUDGET}, "budget": BUDGET, "reproduction": {}, "load_target": None, "d0": None}
def save(s): jdump(s, SP)
def ledger(ev): ev["t"] = now(); jappend(ev, LEDGER)
def branch(ev): ev["t"] = now(); jappend(ev, BT)
def usage_from_ledger():
    u = {k: 0 for k in BUDGET}; boots = 0; gsm = 0
    for l in open(LEDGER) if os.path.exists(LEDGER) else []:
        e = json.loads(l)
        if e.get("event") == "session_end":
            k = e["kind"]
            if k in ("PERF", "REPLAY", "LOAD", "DIAG", "RETRY"): u[k] += 1; u["SESSIONS"] += 1
        if e.get("event") == "boot_end": u["BOOT"] += 1
        if e.get("event") == "gsm_end": u["GSM_Q"] += e.get("questions", 0)
    return u

def spec(cid, cfg, parent, workloads, note="", **kw):
    c = {"cell_id": cid, "phase_id": "IDE_072", "parent_cell_id": parent, "change_kind": "CONFIG", "semantics_family": "SEMANTICS_UNVERIFIED", "precision_family": "GPU_FP8_CPU_INT4_KV_BF16",
         "workloads": workloads, "env": dict(cfg["env"]), "server_args": dict(cfg["sa"]), "gpus": "0,1,2,3", "pin_nonkt": cfg["pin"], "patch": cfg["patch"], "expected_slots": 5952,
         "boot_timeout": 600, "bench_timeout": 300, "replay_timeout": 300, "note": note}
    c.update(kw); c["config_hash"] = config_hash({"env": c["env"], "sa": c["server_args"], "pin": c["pin_nonkt"], "patch": c["patch"]}); return c

def run(s, c, attempt, kinds):
    """셀 실행. kinds = 이 부팅에서 수행하는 세션 종류 목록 (PERF×n, REPLAY, LOAD, DIAG, GSM). 원장에 예약→종료 기록."""
    cid = c["cell_id"]; rid = f"{cid}.{attempt}"
    s["current"] = {"cell": cid, "attempt": attempt, "t": now()}; save(s)
    ledger({"event": "boot_start", "run_id": rid, "cell": cid, "attempt": attempt, "parent": c.get("parent_cell_id"), "config_hash": c["config_hash"], "kinds_reserved": kinds})
    p = f"{CAMP}/specs/{cid}.json"; os.makedirs(os.path.dirname(p), exist_ok=True); jdump(c, p)
    with open(f"{CAMP}/{cid}.{attempt}.runner.log", "w") as lf:
        subprocess.run([sys.executable, f"{HERE}/run_cell.py", CAMP, p, attempt], stdout=lf, stderr=subprocess.STDOUT, timeout=3 * 3600, env=dict(os.environ, IDE071_FEAT=os.environ["IDE071_FEAT"]))
    sp = f"{CAMP}/{cid}/{attempt}/status.json"; stt = json.load(open(sp)) if os.path.exists(sp) else {"exit_status": "FAILED_RUNTIME", "reps": [], "errors": []}
    ledger({"event": "boot_end", "run_id": rid, "verdict": (stt.get("boot") or {}).get("verdict"), "boot_seconds": (stt.get("boot") or {}).get("boot_seconds"), "exit_status": stt.get("exit_status")})
    # 세션 종료 기록 (실제 시작된 부하만 소비)
    for r in stt.get("reps") or []:
        k = "LOAD" if r["workload_id"] == "COMPACT_LOAD" else "PERF"
        if c.get("kind_override"): k = c["kind_override"]
        ledger({"event": "session_end", "run_id": rid, "kind": k, "rep": r["rep_id"], "workload": r["workload_id"], "sent": r["n"], "completed": r.get("completed"), "failed": r.get("failed"), "valid": r["valid"], "output_tps": r.get("output_tps")})
    if stt.get("replay"): ledger({"event": "session_end", "run_id": rid, "kind": "REPLAY", "sent": 32, "completed": stt["replay"].get("completed"), "failed": stt["replay"].get("failed"), "healthy_after": stt["replay"].get("healthy_after")})
    g = f"{CAMP}/{cid}/{attempt}/gsm20_paired.json"
    if os.path.exists(g): j = json.load(open(g)); ledger({"event": "gsm_end", "run_id": rid, "questions": j["n"], "correct": j["correct"], "unscored": j["unscored"], "truncated": j["truncated"], "errors": j["errors"]})
    unsent = [w["workload_id"] for w in c["workloads"]] if not (stt.get("reps")) else []
    if unsent and stt.get("exit_status") != "COMPLETED": ledger({"event": "not_sent", "run_id": rid, "workloads": unsent, "reason": stt.get("exit_status")})
    rec = s["cells"].setdefault(cid, {"attempts": []}); rec["attempts"].append({"attempt": attempt, "exit_status": stt.get("exit_status"), "reps": stt.get("reps"), "replay": stt.get("replay"), "gsm20": stt.get("gsm20"), "config_mismatch": stt.get("config_mismatch"), "n_errors": len(stt.get("errors", [])), "t": now()})
    s["usage"] = usage_from_ledger(); s["current"] = None; save(s); return stt

def nonfinite_error(stt):
    """서버 로그 원문에서 NaN/Inf/negative probability/CUDA assert/illegal memory access 를 구분해 반환."""
    d = f"{CAMP}/{stt['cell_id']}/{stt['attempt_id']}/server.full.log.gz"
    if not os.path.exists(d): return None
    import gzip; txt = gzip.open(d, "rb").read().decode("utf-8", "replace")
    kinds = {k: txt.count(k) for k in ("probability tensor contains either", "illegal memory access", "CUBLAS_STATUS_EXECUTION_FAILED", "nan", "inf", "device-side assert")}
    first = None
    for i, line in enumerate(txt.splitlines()):
        if "probability tensor" in line or "illegal memory access" in line or "device-side assert" in line: first = (i + 1, line[:300]); break
    return {"counts": kinds, "first_line": first, "log": os.path.relpath(d, REPO)}

def ok_run(stt):
    return stt.get("exit_status") == "COMPLETED" and all(r["valid"] for r in stt.get("reps") or []) and not stt.get("config_mismatch") and (not stt.get("replay") or stt["replay"].get("healthy_after"))

def main():
    s = load(); save(s)
    if not os.path.exists(LEDGER): ledger({"event": "campaign_start", "campaign_id": s["campaign_id"], "budget": BUDGET})
    # V1 중복 판정: KT_AVX_RB 는 getenv!=NULL 이면 ON (amx_kernels.hpp:1769) → V0(미설정)=OFF, V1("0")=ON → 실효 경로 다름
    branch({"rule": "V1_DUP_CHECK", "source": "kt-kernel/operators/amx/la/amx_kernels.hpp:1769 avx_rb_on(): getenv(KT_AVX_RB)!=nullptr", "V0": "unset → false", "V1": "\"0\" → true", "result": "DIFFERENT_EFFECTIVE_PATH → V1 실행"})
    branch({"rule": "V2_SKIP_OFF_METHOD", "source": "kt_kernel/experts_base.py:853 bool(os.environ.get(KT_CF_SKIP_EMPTY_IMM))", "method": "변수 미설정(unset) — 빈 문자열/'0' 도 truthy 이므로 unset 만이 OFF", "result": "V2 env 에서 KT_CF_SKIP_EMPTY_IMM 제거"})
    branch({"rule": "Q0_DEF0_SEMANTICS", "source": "kt_kernel/experts_base.py select_deferred_experts: protected_k = topk - max_deferred; max_deferred 0 → protected_k=8 → deferred 없음(모두 immediate). KT_COLD_DEFER 미설정", "result": "SEMANTICS: deferral 경로 비활성 (immediate only); 정확성 기준선으로 부르지 않음"})
    S1 = [{"workload_id": "SHORT_COLD", "C": 64, "reps": 1}]
    first = {}
    for v in ("V0", "V1", "V2"):
        cid = f"{v}_first"
        if cid not in s["cells"]:
            stt = run(s, spec(cid, CFG[v], "S2" if v == "V0" else "V0", S1, note=f"{v} 첫 부팅: smoke·warmup·SHORT_COLD 1회·REPLAY" + ("·GSM20" if v == "V2" else ""), replay=True, gsm20=(v == "V2")), "a1", ["PERF", "REPLAY"] + (["GSM"] if v == "V2" else []))
            stt["cell_id"] = cid; stt["attempt_id"] = "a1"; first[v] = stt
            nf = nonfinite_error(stt); s["reproduction"][v] = {"exit_status": stt.get("exit_status"), "replay": stt.get("replay"), "nonfinite": nf, "state": ("REPRODUCED" if nf and nf["first_line"] else "NOT_REPRODUCED_WITHIN_BUDGET") if stt.get("boot", {}).get("verdict") == "HEALTH_OK" else "NOT_RUN_PRECONDITION_FAILED"}; save(s)
    # Q0: 품질만
    if "Q0_first" not in s["cells"]: run(s, spec("Q0_first", Q0, "V2", [], note="Q0: smoke·warmup·GSM20 만 (성능 sweep 없음)", gsm20=True), "a1", ["GSM"])
    # 확인 부팅 V0, V1 (첫 실행이 정상 완료된 경우만; §4.2)
    for v in ("V0", "V1"):
        f = s["cells"].get(f"{v}_first", {}).get("attempts", [{}])[-1]
        eligible = f.get("exit_status") == "COMPLETED" and not f.get("config_mismatch") and (f.get("replay") or {}).get("healthy_after")
        branch({"rule": "CONFIRM_ELIGIBLE", "cell": v, "first_status": f.get("exit_status"), "replay_healthy": (f.get("replay") or {}).get("healthy_after"), "eligible": bool(eligible)})
        if not eligible or f"{v}_confirm" in s["cells"]: continue
        run(s, spec(f"{v}_confirm", CFG[v], f"{v}_first", [{"workload_id": "SHORT_COLD", "C": 64, "reps": 3}, {"workload_id": "COMPACT_ALT", "C": 64, "reps": 1}], note=f"{v} 새 부팅 확인: SHORT_COLD 3회·COMPACT_ALT 1회·GSM20", gsm20=True), "a1", ["PERF"] * 4 + ["GSM"])
    # LOAD 대상 (§3.2): V1 이 실제로 다른 구성이고 정규 성능·재현·출력 세션에서 오류 없이 실행 → V1; 아니면 V0; 둘 다 아니면 NOT_TRIGGERED
    def clean(v):
        a = s["cells"].get(f"{v}_first", {}).get("attempts", [{}])[-1]; b = s["cells"].get(f"{v}_confirm", {}).get("attempts", [{}])[-1]
        return all(x.get("exit_status") == "COMPLETED" and (x.get("n_errors") or 0) == 0 for x in (a, b)) and (a.get("replay") or {}).get("healthy_after") and all(r["valid"] for x in (a, b) for r in (x.get("reps") or []))
    target = "V1" if clean("V1") else ("V0" if clean("V0") else None)
    branch({"rule": "LOAD_TARGET", "V1_clean": clean("V1"), "V0_clean": clean("V0"), "target": target or "NOT_TRIGGERED_NO_ELIGIBLE_TARGET"}); s["load_target"] = target; save(s)
    if target and f"{target}_load" not in s["cells"]:
        # 마지막 서버가 V1_confirm 이면 같은 서버에서 실행 가능하나 run_cell 은 셀 단위 부팅이므로 새 부팅 1회 (부하 대상 복귀 예산)
        run(s, spec(f"{target}_load", CFG[target], f"{target}_confirm", [{"workload_id": "COMPACT_LOAD", "C": 64, "n": 1024, "reps": 1}], note="연속 부하 1회 (C64·1,024 요청)", bench_timeout=600), "a1", ["LOAD"])
    # D0: V0→V1→V2 순으로 오류가 관측된 첫 구성의 graph OFF, 최대 32 요청
    err_parent = next((v for v in ("V0", "V1", "V2") if s["reproduction"].get(v, {}).get("state") == "REPRODUCED" or s["cells"].get(f"{v}_first", {}).get("attempts", [{}])[-1].get("exit_status") in ("FAILED_RUNTIME", "FAILED_REQUESTS")), None)
    branch({"rule": "D0_TRIGGER", "parent": err_parent, "result": "RUN" if err_parent else "NOT_TRIGGERED"})
    if err_parent and "D0_graph_off" not in s["cells"]:
        cfg = dict(CFG[err_parent]); cfg["sa"] = dict(cfg["sa"], **{"disable-cuda-graph": True})
        run(s, spec("D0_graph_off", cfg, f"{err_parent}_first", [{"workload_id": "SHORT_COLD", "C": 16, "n": 32, "reps": 1}], note="조건부 진단: 부모의 graph 만 OFF, 32 요청. 일반 성능에 합산하지 않음", kind_override="DIAG"), "a1", ["DIAG"])
        s["d0"] = {"parent": err_parent}
    u = usage_from_ledger(); s["usage"] = u
    s["status"] = "BOUNDED_VALIDATION_COMPLETE" if all(u[k] <= BUDGET[k] for k in BUDGET) else "VALIDATION_BUDGET_REACHED"
    s["t_end"] = now(); save(s); ledger({"event": "campaign_end", "status": s["status"], "usage": u}); print(s["status"], u)

if __name__ == "__main__":
    main()
