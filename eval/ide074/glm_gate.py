#!/usr/bin/env python3
"""IDE_074 M8 — GLM-4.7-FP8 native BASIC4 (FP8_PERCHANNEL, cpuinfer 100, uniform 80, deferred 2) 정상성 게이트 → 통과 시 G 예산(OFF/ON 각 1회, PROBE128 GLM).
게이트: IDE_073 quality_requests.jsonl 앞 4문항, chat API, temperature 0, max_tokens 1024, enable_thinking=false, ignore_eos 없음 (성능용과 혼합 금지).
  통과 = 4문항 모두 http 200·error 없음·finish_reason != length(절단 아님)·응답 시간 < 300 s. 실패 → BLOCKED_NORMAL_OUTPUT (문항·오류 원문 보존), 프로브 미실행.
ON = torch profiler(CPU+GPU, start_step=1) 만 (KT_EVT 는 callback-free 경로 전용이라 GLM 기본 경로(cudaLaunchHostFunc)에서 기록 없음 → 미적용 사실 기록). 같은 부팅에서 OFF→ON.
사용: glm_gate.py"""
import json, os, sys, time, subprocess
os.environ["IDE071_FEAT"] = os.path.expanduser("~/projects/vllm_hybrid/shadow_assists/features/IDE_074")
sys.path.insert(0, os.path.expanduser("~/projects/vllm_hybrid/eval/ide071")); sys.path.insert(0, os.path.expanduser("~/projects/vllm_hybrid/eval/ide073")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from run_cell import build_args, WORKLOADS, smoke_greedy4
import run_campaign as rc73, harness as H
CAMP = H.CAMP; QF = f"{HOME}/.cache/huggingface/kt/ide073/inputs/glm/quality_requests.jsonl"


def main():
    boot_id = f"G_{time.strftime('%H%M%S')}"; out = f"{CAMP}/glm/BASIC4/{boot_id}"; os.makedirs(out, exist_ok=True); os.makedirs(f"{H.KT_HOST}/{boot_id}/profiles", exist_ok=True)
    u = H.usage()
    if u["BOOT"] >= H.BUDGET["BOOT"] or u["SESSIONS"] + 2 > H.BUDGET["SESSIONS"]: raise SystemExit(f"budget {u}")
    sa, env, gpus, pin, patch = rc73.cfg("glm", "BASIC4"); env = dict(env); env["SGLANG_TORCH_PROFILER_DIR"] = f"{H.KT_CN}/{boot_id}/profiles"
    stop_server(); sh(f"bash {REPO}/eval/ide070/patch_per_layer_experts.sh {CN} --revert", timeout=120)
    H.ledger({"event": "boot_start", "boot_id": boot_id, "mode": "GLM_GATE", "env": env})
    b = boot_server(build_args(sa), env, gpus, out, timeout=1500); jdump({"server_args": sa, "env": env, "gpus": gpus, "pin": pin, "patch": patch, "mode": "GLM_GATE"}, f"{out}/requested_config.json")
    H.ledger({"event": "boot_end", "boot_id": boot_id, "mode": "GLM_GATE", "verdict": b["verdict"], "boot_seconds": b.get("boot_seconds")})
    if b["verdict"] != "HEALTH_OK": save_server_log(out); jdump({"status": "BOOT_FAILED", "boot": b}, f"{out}/gate_result.json"); print("BOOT FAILED"); return
    pids = all_server_host_pids(); kt = kt_worker_cpus(pids); jdump(server_info(), f"{out}/server_info.json")
    # 게이트: 4문항
    q4 = f"{out}/quality_requests_4.jsonl"; open(q4, "w").writelines(open(QF).readlines()[:4])
    r = sh(f"{HOME}/venv-bench/bin/python {REPO}/eval/ide073/quality20.py {q4} glm47 {out}/gate_quality4.json --thinking-off", timeout=1500)
    H.ledger({"event": "quality_end", "boot_id": boot_id, "questions": 4, "rc": r.returncode})
    g = json.load(open(f"{out}/gate_quality4.json")) if os.path.exists(f"{out}/gate_quality4.json") else {}
    items = g.get("items") or g.get("results") or []
    bad = [{"qid": i.get("qid"), "http": i.get("http"), "error": i.get("error"), "finish_reason": i.get("finish_reason"), "truncated": i.get("truncated"), "seconds": i.get("seconds"), "text_head": (i.get("text") or "")[:120]} for i in items if (i.get("error") not in (None, "None")) or str(i.get("finish_reason")) == "length" or str(i.get("truncated")) == "True" or float(i.get("seconds") or 0) > 300]
    passed = bool(items) and len(items) == 4 and not bad
    jdump({"status": "GATE_PASS" if passed else "BLOCKED_NORMAL_OUTPUT", "n_items": len(items), "bad": bad, "correct": g.get("correct"), "note": "게이트 문항은 전체 품질 예산(≤80) 내 4문항 계수"}, f"{out}/gate_result.json")
    print("GATE", "PASS" if passed else "BLOCKED_NORMAL_OUTPUT", bad[:2])
    if passed:
        wl = "M073_GLM_PROBE128"; run_bench(dict(WORKLOADS[wl]), 16, 32, "glm47", f"{out}/warmup", 1, timeout=900)
        H.session("OFF", "DIAG", "G_OFF", wl, 64, 128, boot_id, out, pids, kt)
        H.session("P1", "DIAG", "G_P1", wl, 64, 128, boot_id, out, pids, kt)   # KT_EVT 미설정 → CPU 이벤트 없음(기록), torch profiler 만
    save_server_log(out); stop_server(); print("glm gate done", H.usage())


if __name__ == "__main__": main()
