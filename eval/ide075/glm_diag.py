#!/usr/bin/env python3
"""IDE_075 G00 — GLM 정상 출력 진단 (확장 단계, 사용자 승인). 모드:
  D1: 코드 무수정 + --kt-num-gpu-experts 160 (CPU 기여 0) + KT_DUMP_TOPK=1 → 4문항. H1 예측: prefill 정상·decode ×2.5 → 여전히 비정상 (또는 부분 회복)
  D2: glm_rsf_patch apply + 기본 80 → 4문항 게이트. 통과 시 warmup → G_OFF(PROBE128, vllm) → G_CORR(PROBE128, probe 클라이언트 + profiler)
  D3: (선택) 수정 + --kt-max-deferred-experts-per-token 0 → 4문항 (H3 잔여 열화)
결과: eval/results/IDE_075_20260917/glm/BASIC4/<boot>/ ; 원장 phase=extended, quality 문항 계수."""
import json, os, sys, time, subprocess
os.environ["IDE071_FEAT"] = os.path.expanduser("~/projects/vllm_hybrid/shadow_assists/features/IDE_075"); os.environ["IDE075_EXTENDED"] = "1"; os.environ["IDE075_MODEL"] = "glm47"   # harness 는 import 시점에 MODEL 을 읽음 (D6 G_OFF/G_CORR 는 qwen480 이름·토크나이저로 실행됨 → D8 재실행)
sys.path.insert(0, os.path.expanduser("~/projects/vllm_hybrid/eval/ide071")); sys.path.insert(0, os.path.expanduser("~/projects/vllm_hybrid/eval/ide073")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from run_cell import build_args, WORKLOADS, smoke_greedy4
import run_campaign as rc73, harness as H
CAMP = H.CAMP; QF = f"{HOME}/.cache/huggingface/kt/ide073/inputs/glm/quality_requests.jsonl"; PATCH = f"{REPO}/eval/ide075/glm_rsf_patch.py"


def gate(out, tag):
    q4 = f"{out}/quality_requests_4.jsonl"; open(q4, "w").writelines(open(QF).readlines()[:4])
    r = sh(f"{HOME}/venv-bench/bin/python {REPO}/eval/ide073/quality20.py {q4} glm47 {out}/gate_quality4_{tag}.json --thinking-off", timeout=1500)
    H.ledger({"event": "quality_end", "boot_id": os.path.basename(out), "questions": 4, "rc": r.returncode, "tag": tag})
    g = json.load(open(f"{out}/gate_quality4_{tag}.json")) if os.path.exists(f"{out}/gate_quality4_{tag}.json") else {}
    items = g.get("items") or g.get("results") or []
    bad = [{"qid": i.get("qid"), "finish_reason": i.get("finish_reason"), "truncated": i.get("truncated"), "seconds": i.get("seconds"), "error": i.get("error"), "text_head": (i.get("text") or "")[:160]} for i in items if (i.get("error") not in (None, "None")) or str(i.get("finish_reason")) == "length" or str(i.get("truncated")) == "True" or float(i.get("seconds") or 0) > 300]
    res = {"tag": tag, "status": "GATE_PASS" if (items and len(items) == 4 and not bad) else "BLOCKED_NORMAL_OUTPUT", "correct": g.get("correct"), "n": len(items), "bad": bad, "texts": [(i.get("text") or "")[:200] for i in items]}
    jdump(res, f"{out}/gate_result_{tag}.json"); print("GATE", tag, res["status"], "correct", res["correct"], [b["finish_reason"] for b in bad]); return res


def boot_glm(boot_id, extra_args=None, extra_env=None):
    sa, env, gpus, pin, patch = rc73.cfg("glm", "BASIC4"); sa = dict(sa); env = dict(env); env.update(extra_env or {}); sa.update(extra_args or {}); env["SGLANG_TORCH_PROFILER_DIR"] = f"{H.KT_CN}/{boot_id}/profiles"
    if os.environ.get("IDE075_GLM_KT_EVT", "1") == "1": env["KT_EVT"] = f"{H.KT_CN}/{boot_id}/kt_evt.csv"   # CPU 기록기 (D6 G_CORR 는 미설정이라 CPU 레코드 없음 → D8 부터 켬; 같은 부팅의 G_OFF2/G_OFF3 도 기록기 ON 상태)
    out = f"{CAMP}/glm/BASIC4/{boot_id}"; os.makedirs(out, exist_ok=True); subprocess.run(f"{DOCKER} exec {CN} sh -c 'mkdir -p {H.KT_CN}/{boot_id}/profiles && chmod -R 777 {H.KT_CN}/{boot_id}'", shell=True)
    stop_server(); sh(f"bash {REPO}/eval/ide070/patch_per_layer_experts.sh {CN} --revert", timeout=120)
    H.ledger({"event": "boot_start", "boot_id": boot_id, "mode": "GLM_DIAG", "env": env, "args_override": extra_args})
    b = boot_server(build_args(sa), env, gpus, out, timeout=1500); jdump({"server_args": sa, "env": env, "gpus": gpus, "mode": "GLM_DIAG", "rsf_patch": subprocess.run(f"{DOCKER} exec {CN} python3 /tmp/glm_rsf_patch.py status", shell=True, capture_output=True, text=True).stdout}, f"{out}/requested_config.json")
    H.ledger({"event": "boot_end", "boot_id": boot_id, "mode": "GLM_DIAG", "verdict": b["verdict"], "boot_seconds": b.get("boot_seconds")})
    if b["verdict"] != "HEALTH_OK": save_server_log(out); return None, None, None
    pids = all_server_host_pids(); kt = kt_worker_cpus(pids); jdump(server_info(), f"{out}/server_info.json"); H.thread_role_map(pids, out)
    return out, pids, kt


def main():
    mode = sys.argv[1]; subprocess.run(f"{DOCKER} cp {PATCH} {CN}:/tmp/glm_rsf_patch.py", shell=True)
    if mode == "D1":
        subprocess.run(f"{DOCKER} exec {CN} python3 /tmp/glm_rsf_patch.py revert", shell=True)
        out, pids, kt = boot_glm(f"GLM_D1_{time.strftime('%H%M%S')}", {"kt-num-gpu-experts": 160}, {"KT_DUMP_TOPK": "1"})
        if out: gate(out, "D1_allgpu_unfixed"); sh(f"{DOCKER} cp {CN}:/tmp/topk_dump.pt {out}/topk_dump.pt", timeout=120); save_server_log(out)
    elif mode == "D2":
        subprocess.run(f"{DOCKER} exec {CN} python3 /tmp/glm_rsf_patch.py apply", shell=True)
        out, pids, kt = boot_glm(f"GLM_D2_{time.strftime('%H%M%S')}")
        if out:
            g = gate(out, "D2_fixed")
            if g["status"] == "GATE_PASS":
                smoke_greedy4(out, "glm47"); H.ledger({"event": "smoke_end", "boot_id": os.path.basename(out), "requests": 4})
                wl = "M073_GLM_PROBE128"; run_bench(dict(WORKLOADS[wl]), 16, 32, "glm47", f"{out}/warmup", 1, timeout=900); H.ledger({"event": "warmup_end", "boot_id": os.path.basename(out), "requests": 32})
                os.environ["IDE075_MODEL"] = "glm47"
                H.session("OFF", "G_OFF", wl, 64, 128, os.path.basename(out), out, pids, kt, client="vllm")
                H.session("CORR", "G_CORR", wl, 64, 128, os.path.basename(out), out, pids, kt, client="probe")
            save_server_log(out)
    elif mode == "D9":   # BASIC4 인자 + callback-free env (Qwen OPT4 와 같은 소프트웨어 경로) → CPU 기록기 동작 → GLM 의존성 측정. 구성은 BASIC4 와 다름(SEMANTICS: 같은 커널·같은 deferred 2, 전달 경로만 callback-free)
        subprocess.run(f"{DOCKER} exec {CN} python3 /tmp/glm_rsf_patch.py apply", shell=True)
        out, pids, kt = boot_glm(f"GLM_D9_{time.strftime('%H%M%S')}", None, {"KT_CALLBACK_FREE": "1", "KT_CF_SKIP_EMPTY_IMM": "1"})
        if out:
            g = gate(out, "D9_dispatchfix_rsf_cbfree")
            if g["status"] == "GATE_PASS":
                smoke_greedy4(out, "glm47"); H.ledger({"event": "smoke_end", "boot_id": os.path.basename(out), "requests": 4})
                wl = "M073_GLM_PROBE128"; run_bench(dict(WORKLOADS[wl]), 16, 32, "glm47", f"{out}/warmup", 1, timeout=900); H.ledger({"event": "warmup_end", "boot_id": os.path.basename(out), "requests": 32})
                H.session("CORR", "G_CORR3_cbfree", wl, 64, 128, os.path.basename(out), out, pids, kt, client="vllmw")
                H.session("OFF", "G_OFF4_cbfree", wl, 64, 128, os.path.basename(out), out, pids, kt, client="vllm")
            save_server_log(out)
    elif mode == "D8":   # D6 재실행 (모델 이름·토크나이저 glm47, nerdctl 지연 없는 상태에서 G_OFF2/G_CORR2)
        subprocess.run(f"{DOCKER} exec {CN} python3 /tmp/glm_rsf_patch.py apply", shell=True)
        out, pids, kt = boot_glm(f"GLM_D8_{time.strftime('%H%M%S')}")
        if out:
            g = gate(out, "D8_dispatchfix_rsf")
            if g["status"] == "GATE_PASS":
                smoke_greedy4(out, "glm47"); H.ledger({"event": "smoke_end", "boot_id": os.path.basename(out), "requests": 4})
                wl = "M073_GLM_PROBE128"; run_bench(dict(WORKLOADS[wl]), 16, 32, "glm47", f"{out}/warmup", 1, timeout=900); H.ledger({"event": "warmup_end", "boot_id": os.path.basename(out), "requests": 32})
                H.session("OFF", "G_OFF2", wl, 64, 128, os.path.basename(out), out, pids, kt, client="vllm")
                H.session("CORR", "G_CORR2", wl, 64, 128, os.path.basename(out), out, pids, kt, client="vllmw")
                H.session("OFF", "G_OFF3", wl, 64, 128, os.path.basename(out), out, pids, kt, client="vllm")
            save_server_log(out)
    elif mode in ("D6", "D7"):   # 확장: kt-kernel FP8 dispatch 수정(.so 재빌드) 후. D6 = rsf 수정 적용 + 기본 구성 → 게이트 → G_OFF/G_CORR ; D7 = rsf 수정 미적용 → 게이트만 (rsf 효과 분리)
        subprocess.run(f"{DOCKER} exec {CN} python3 /tmp/glm_rsf_patch.py {'apply' if mode == 'D6' else 'revert'}", shell=True)
        out, pids, kt = boot_glm(f"GLM_{mode}_{time.strftime('%H%M%S')}")
        if out:
            g = gate(out, f"{mode}_dispatchfix_{'rsf' if mode == 'D6' else 'norsf'}")
            if mode == "D6" and g["status"] == "GATE_PASS":
                smoke_greedy4(out, "glm47"); H.ledger({"event": "smoke_end", "boot_id": os.path.basename(out), "requests": 4})
                wl = "M073_GLM_PROBE128"; run_bench(dict(WORKLOADS[wl]), 16, 32, "glm47", f"{out}/warmup", 1, timeout=900); H.ledger({"event": "warmup_end", "boot_id": os.path.basename(out), "requests": 32})
                os.environ["IDE075_MODEL"] = "glm47"
                H.session("OFF", "G_OFF", wl, 64, 128, os.path.basename(out), out, pids, kt, client="vllm")
                H.session("CORR", "G_CORR", wl, 64, 128, os.path.basename(out), out, pids, kt, client="vllmw")
            save_server_log(out)
    elif mode == "D4":
        subprocess.run(f"{DOCKER} exec {CN} python3 /tmp/glm_rsf_patch.py revert", shell=True); subprocess.run(f"{DOCKER} exec {CN} python3 /tmp/glm_rsf_patch.py apply", shell=True)
        out, pids, kt = boot_glm(f"GLM_D4_{time.strftime('%H%M%S')}", {"kt-num-gpu-experts": 0})
        if out: gate(out, "D4_fixed_allcpu"); save_server_log(out)
    elif mode == "D5":
        subprocess.run(f"{DOCKER} exec {CN} python3 /tmp/glm_rsf_patch.py revert", shell=True)
        out, pids, kt = boot_glm(f"GLM_D5_{time.strftime('%H%M%S')}", {"kt-num-gpu-experts": 0, "kt-max-deferred-experts-per-token": 0})
        if out: gate(out, "D5_unfixed_allcpu_deferred0"); save_server_log(out)
    elif mode == "D3":
        subprocess.run(f"{DOCKER} exec {CN} python3 /tmp/glm_rsf_patch.py apply", shell=True)
        out, pids, kt = boot_glm(f"GLM_D3_{time.strftime('%H%M%S')}", {"kt-max-deferred-experts-per-token": 0})
        if out: gate(out, "D3_fixed_deferred0"); save_server_log(out)
    stop_server(); print("glm diag done", mode, H.usage())


if __name__ == "__main__": main()
