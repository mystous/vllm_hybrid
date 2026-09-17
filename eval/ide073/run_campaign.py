#!/usr/bin/env python3
"""IDE_073 — 2모델 × (GPU8 / KT-BASIC4 / OPT4) 성능 비교 실행기 (지시서 §5). 기존 run_cell.py 재사용.
셀(=부팅) 단위: 워크로드 세션 목록·품질 20문항·프로브 여부. 원장 state/execution_events.jsonl, 상태 state/RUN_STATE.json (보고 루프용 요약), RUN_MANIFEST.json.
사용: run_campaign.py <model: qwen|glm> [--only cell1,cell2]
"""
import json, os, subprocess, sys, time, glob
os.environ["IDE071_FEAT"] = os.path.expanduser("~/projects/vllm_hybrid/shadow_assists/features/IDE_073")
sys.path.insert(0, os.path.expanduser("~/projects/vllm_hybrid/eval/ide071"))
from common import *
HERE = os.path.dirname(os.path.abspath(__file__)); RC = f"{REPO}/eval/ide071/run_cell.py"
CAMP = f"{REPO}/eval/results/IDE_073_20260917"; os.makedirs(f"{CAMP}/state", exist_ok=True)
SP = f"{CAMP}/state/RUN_STATE.json"; LEDGER = f"{CAMP}/state/execution_events.jsonl"; MAN = f"{CAMP}/state/RUN_MANIFEST.json"
BUDGET = {"PERF": 30, "DIAG": 8, "EXTRA": 6, "SESSIONS": 44, "BOOT": 20, "QUALITY_Q": 120}
QWEN = MODEL_FP8_CN; QKT = "/models/kt/qwen3-480b-int4"
GLM = "/models/hub/models--zai-org--GLM-4.7-FP8/snapshots/7b3b5f81eee81be12a6f8da2710eac4bafb0166a"
UPSTREAM = "PYTHONPATH=/sgl-workspace/sglang-upstream/python"
COMMON4 = {"host": "127.0.0.1", "port": 30000, "tp": 4, "context-length": 32768, "trust-remote-code": True, "kv-cache-dtype": "auto", "max-total-tokens": 40960, "mem-fraction-static": 0.95, "max-running-requests": 64, "cuda-graph-max-bs": 64, "cuda-graph-backend-prefill": "disabled"}
COMMON8 = dict(COMMON4, tp=8, **{"mem-fraction-static": 0.90})

def cfg(model, profile):
    """(server_args(replace), env, gpus, pin, patch, quality_flags)"""
    if model == "qwen":
        if profile == "GPU8": return dict(COMMON8, **{"model-path": QWEN, "served-model-name": "qwen480", "ep-size": 8, "attention-backend": "triton", "chunked-prefill-size": 8192}), {}, "0,1,2,3,4,5,6,7", False, None
        if profile == "BASIC4": return dict(COMMON4, **{"model-path": QWEN, "served-model-name": "qwen480", "attention-backend": "triton", "kt-weight-path": QKT, "kt-method": "AMXINT4", "kt-cpuinfer": 112, "kt-threadpool-count": 2, "kt-num-gpu-experts": 96, "kt-max-deferred-experts-per-token": 2, "chunked-prefill-size": 8192}), {}, "0,1,2,3", False, None   # 공식 worktree 는 설치된 kt_kernel API(gpu_experts_mask) 와 불일치로 부팅 실패 → 로컬 tree(호환 패치) + 로컬 기능 env 전부 unset
        if profile == "OPT4": return dict(COMMON4, **{"model-path": QWEN, "served-model-name": "qwen480", "attention-backend": "triton", "kt-weight-path": QKT, "kt-method": "AMXINT4", "kt-cpuinfer": 96, "kt-threadpool-count": 2, "kt-num-gpu-experts": 96, "init-expert-location": "/models/kt/ide070/hotmap_v2.json", "kt-max-deferred-experts-per-token": 8, "ep-dispatch-algorithm": "dynamic", "chunked-prefill-size": 8192}), {"KT_CALLBACK_FREE": "1", "KT_CF_SKIP_EMPTY_IMM": "1", "KT_GPU_EXPERTS_PER_LAYER": "/models/kt/ide070/layer_budget_5952.json", "KT_AVX_RB": "0"}, "0,1,2,3", True, "per_layer"
    if model == "glm":
        g = {"model-path": GLM, "served-model-name": "glm47", "attention-backend": "flashinfer", "fp8-gemm-backend": "triton", "enable-p2p-check": True, "disable-shared-experts-fusion": True, "enable-mixed-chunk": True, "chunked-prefill-size": 4096}
        if profile == "GPU8": return dict(COMMON8, **g, **{"ep-size": 8}), {}, "0,1,2,3,4,5,6,7", False, None
        if profile == "BASIC4": return dict(COMMON4, **g, **{"kt-weight-path": GLM, "kt-method": "FP8_PERCHANNEL", "kt-cpuinfer": 100, "kt-threadpool-count": 2, "kt-num-gpu-experts": 80, "kt-max-deferred-experts-per-token": 2}), {}, "0,1,2,3", False, None
        if profile == "OPT4":
            p = json.load(open(f"{CAMP}/state/glm_opt_plan.json")) if os.path.exists(f"{CAMP}/state/glm_opt_plan.json") else None
            if not p or p.get("status") != "READY": return None   # OPT_TRANSFER_BLOCKED_FORMAT / CALIBRATION_FAILED → NOT_RUN_DEPENDENCY
            return dict(COMMON4, **g, **{"kt-weight-path": p["kt_weight_path"], "kt-method": p["kt_method"], "kt-cpuinfer": 96, "kt-threadpool-count": 2, "kt-num-gpu-experts": 80, "init-expert-location": p["hotmap"], "kt-max-deferred-experts-per-token": 8, "ep-dispatch-algorithm": "dynamic"}), {"KT_CALLBACK_FREE": "1", "KT_CF_SKIP_EMPTY_IMM": "1", "KT_GPU_EXPERTS_PER_LAYER": p["budget"], "KT_AVX_RB": "0"}, "0,1,2,3", True, "per_layer"
    return None

def load():
    st = json.load(open(SP)) if os.path.exists(SP) else {"campaign_id": "IDE_073_20260917"}
    st.setdefault("cells", {}); st.setdefault("counts", {}); return st
def save(s): jdump(s, SP)
def ledger(ev): ev["t"] = now(); jappend(ev, LEDGER)
def usage():
    u = {k: 0 for k in BUDGET}
    evs = [json.loads(l) for l in (open(LEDGER) if os.path.exists(LEDGER) else [])]
    void = {(e.get("cell"), e.get("attempt"), e.get("rep")) for e in evs if e.get("event") == "session_void"}
    for e in evs:
        if e.get("event") == "session_end" and (e.get("cell"), e.get("attempt"), e.get("rep")) not in void: u[e["kind"]] = u.get(e["kind"], 0) + 1; u["SESSIONS"] += 1
        if e.get("event") == "boot_end": u["BOOT"] += 1
        if e.get("event") == "quality_end": u["QUALITY_Q"] += e.get("questions", 0)
    return u

def run(s, cid, model, profile, workloads, quality=False, replay=False, extra_env=None, kinds=None, attempt="a1", note="", cell_timeout=3600):
    c = cfg(model, profile)
    if c is None:
        gp = f"{CAMP}/state/glm_opt_plan.json"; reason = (json.load(open(gp)).get("status") if os.path.exists(gp) else "plan missing")
        s["cells"][cid] = {"status": "NOT_RUN_DEPENDENCY", "reason": f"OPT4 plan {reason}", "model": model, "profile": profile}; save(s); ledger({"event": "cell_skipped", "cell": cid, "reason": "NOT_RUN_DEPENDENCY"}); return None
    sa, env, gpus, pin, patch = c; env = dict(env); env.update(extra_env or {})
    spec = {"cell_id": cid, "phase_id": "IDE_073", "parent_cell_id": None, "change_kind": "CONFIG", "semantics_family": "SEMANTICS_UNVERIFIED", "precision_family": {"qwen": "GPU_FP8block_CPU_INT4", "glm": "GPU_FP8perchannel_CPU_FP8perchannel"}[model] if profile != "GPU8" else "GPU_FP8_ALL_GPU",
            "workloads": workloads, "env": env, "server_args": sa, "replace_server_args": True, "gpus": gpus, "pin_nonkt": pin, "patch": patch, "expected_slots": 5952 if model == "qwen" else 7120, "boot_timeout": 900, "bench_timeout": 1200, "replay_timeout": 300,
            "quality20": {"questions": f"/home/mystous/.cache/huggingface/kt/ide073/inputs/{model}/quality_requests.jsonl", "thinking_off": model == "glm", "model": sa["served-model-name"]} if quality else None, "replay": replay, "note": note, "model": model, "profile": profile}
    spec["config_hash"] = config_hash({"env": env, "sa": sa, "gpus": gpus, "pin": pin, "patch": patch})
    p = f"{CAMP}/specs/{cid}.json"; os.makedirs(os.path.dirname(p), exist_ok=True); jdump(spec, p)
    s["current"] = {"cell": cid, "model": model, "profile": profile, "attempt": attempt, "t": now()["wall_kst"]}; s["phase"] = f"{model}/{profile}"; save(s)
    ledger({"event": "boot_start", "cell": cid, "attempt": attempt, "model": model, "profile": profile, "config_hash": spec["config_hash"], "kinds_reserved": kinds})
    t0 = time.time()
    with open(f"{CAMP}/{cid}.{attempt}.runner.log", "w") as lf:
        try: subprocess.run([sys.executable, RC, CAMP, p, attempt], stdout=lf, stderr=subprocess.STDOUT, timeout=cell_timeout, env=dict(os.environ, IDE071_FEAT=os.environ["IDE071_FEAT"]))
        except subprocess.TimeoutExpired: lf.write("\nCELL_TIMEOUT\n"); stop_server()
    sp = f"{CAMP}/{cid}/{attempt}/status.json"; stt = json.load(open(sp)) if os.path.exists(sp) else {"exit_status": "TIMED_OUT", "reps": [], "errors": []}
    ledger({"event": "boot_end", "cell": cid, "attempt": attempt, "verdict": (stt.get("boot") or {}).get("verdict"), "boot_seconds": (stt.get("boot") or {}).get("boot_seconds"), "exit_status": stt.get("exit_status")})
    for r in stt.get("reps") or []:
        ledger({"event": "session_end", "cell": cid, "attempt": attempt, "kind": "PERF", "rep": r["rep_id"], "workload": r["workload_id"], "sent": r["n"], "completed": r.get("completed"), "failed": r.get("failed"), "valid": r["valid"], "output_tps": r.get("output_tps"), "duration": r.get("duration")})
    if stt.get("replay"): ledger({"event": "session_end", "cell": cid, "attempt": attempt, "kind": "EXTRA", "rep": "REPLAY", "sent": 32, "completed": stt["replay"].get("completed"), "healthy_after": stt["replay"].get("healthy_after")})
    q = f"{CAMP}/{cid}/{attempt}/quality20.json"
    if os.path.exists(q): j = json.load(open(q)); ledger({"event": "quality_end", "cell": cid, "attempt": attempt, "questions": j["n"], "correct": j["correct"], "unscored": j["unscored"], "truncated": j["truncated"], "errors": j["errors"]})
    s["cells"][cid] = {"status": stt.get("exit_status"), "model": model, "profile": profile, "attempt": attempt, "boot_s": (stt.get("boot") or {}).get("boot_seconds"), "reps": [(r["rep_id"], r.get("output_tps"), r.get("completed"), r["valid"]) for r in stt.get("reps") or []], "quality": stt.get("quality20"), "mismatch": stt.get("config_mismatch"), "n_errors": len(stt.get("errors", [])), "wall_s": time.time() - t0}
    s["recent"] = (s.get("recent") or [])[-5:] + [f"{cid}:{stt.get('exit_status')} reps={[(r['rep_id'], round(r.get('output_tps') or 0, 1)) for r in stt.get('reps') or []]}"]
    s["current"] = None; s["usage"] = usage(); s["counts"] = {"completed": sum(1 for v in s["cells"].values() if v.get("status") == "COMPLETED"), "failed": sum(1 for v in s["cells"].values() if str(v.get("status", "")).startswith("FAILED") or v.get("status") == "TIMED_OUT"), "blocked": sum(1 for v in s["cells"].values() if str(v.get("status", "")).startswith(("BLOCKED", "NOT_RUN", "UNSUPPORTED"))), "registered": len(s["cells"])}
    if stt.get("errors"): s["errors_recent"] = [f"{cid}: {str(e.get('reason') or e.get('verdict') or e.get('msg'))[:120]}" for e in stt["errors"][-2:]]
    save(s); return stt

def plan(model):
    m = model.upper(); MAIN = f"M073_{m}_MAIN_SHORT"; LOW = f"M073_{m}_LOW_CONCURRENCY"; LONG = f"M073_{m}_LONGER_PREFILL"
    W3 = [{"workload_id": MAIN, "C": 64, "reps": 3}, {"workload_id": LOW, "C": 1, "n": 16, "reps": 1}, {"workload_id": LONG, "C": 8, "n": 32, "reps": 1}]
    W1 = [{"workload_id": MAIN, "C": 64, "reps": 1}, {"workload_id": LOW, "C": 1, "n": 16, "reps": 1}, {"workload_id": LONG, "C": 8, "n": 32, "reps": 1}]
    W2 = [{"workload_id": MAIN, "C": 64, "reps": 2}]
    P = "Q" if model == "qwen" else "G"
    return [
        (f"{P}-GPU8", "GPU8", W3, True, "GPU-only 8장 참조: 자격(smoke)·MAIN×3·LOW·LONG·품질20 (1 부팅)"),
        (f"{P}-KT-BASIC4", "BASIC4", W3, True, "공식 기본 4장: MAIN×3·LOW·LONG·품질20 (1 부팅, 공식 checkout PYTHONPATH)"),
        (f"{P}-OPT4-b1", "OPT4", W1, True, "최고 구성 4장 첫 부팅: MAIN×1·LOW·LONG·품질20·REPLAY"),
        (f"{P}-OPT4-b2", "OPT4", W2, False, "최고 구성 새 부팅: MAIN×2 (재기동 재현)"),
    ]

def main():
    model = sys.argv[1]; only = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "--all" else None
    s = load()
    man = json.load(open(MAN)) if os.path.exists(MAN) else {"campaign_id": "IDE_073_20260917", "budget": BUDGET, "cells": {}}
    for cid, prof, W, qual, note in plan(model):
        man["cells"][cid] = {"model": model, "profile": prof, "workloads": W, "quality": qual, "note": note, "status": man["cells"].get(cid, {}).get("status", "PLANNED")}
    jdump(man, MAN); save(s)
    if not os.path.exists(LEDGER): ledger({"event": "campaign_start", "budget": BUDGET})
    for cid, prof, W, qual, note in plan(model):
        if only and cid not in only: continue
        if cid in s["cells"] and s["cells"][cid].get("status") == "COMPLETED": continue
        replay = cid.endswith("-OPT4-b1")
        att = f"a{len(glob.glob(f'{CAMP}/{cid}/a*/')) + 1}"   # 기존 attempt 디렉터리 보존
        stt = run(s, cid, model, prof, W, quality=qual, replay=replay, kinds=["PERF"] * sum(w.get("reps", 1) for w in W) + (["QUALITY"] if qual else []), attempt=att, note=note)
        man["cells"][cid]["status"] = (stt or {}).get("exit_status", s["cells"].get(cid, {}).get("status")); jdump(man, MAN)
        if stt and stt.get("exit_status") in ("FAILED_BOOT", "FAILED_RUNTIME", "TIMED_OUT") and usage()["EXTRA"] < BUDGET["EXTRA"] and usage()["BOOT"] < BUDGET["BOOT"]:
            ledger({"event": "retry", "cell": cid, "reason": stt.get("exit_status")}); ledger({"event": "session_end", "cell": cid, "kind": "EXTRA", "rep": "RETRY_RESERVE"})
            stt2 = run(s, cid, model, prof, W, quality=qual, replay=replay, kinds=["RETRY"], attempt=f"a{len(glob.glob(f'{CAMP}/{cid}/a*/')) + 1}", note=note + " (재시도)")
            man["cells"][cid]["status"] = (stt2 or {}).get("exit_status"); jdump(man, MAN)
    s["usage"] = usage(); save(s); print(model, "done", s["usage"])

if __name__ == "__main__":
    main()
