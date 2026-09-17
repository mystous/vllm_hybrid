#!/usr/bin/env python3
"""IDE_073 — 최적화 구성 프로브 4세션/모델 (지시서 §7~§8). 일반 성능표에 넣지 않음.
 boot A (프로브 env 없음): D0 PROBE_OFF (PROBE128, 저빈도 계측만) → D2 GPU_TIMELINE (SGLang torch profiler /start_profile·/stop_profile, CPU+CUDA activity, 같은 128 요청) → D3 CPU_TIMELINE (perf record 99 Hz 20 s kt 워커·스케줄러, 이어서 pcm-memory 20 s 별도 창)
 boot B (KT_PHASE_PROF=1 KT_TQ_TIMING=1): D1 LIGHT_PROBE (PROBE128; 서버 stdout 의 'Profiling Results' 64회당 1회 표본 + [kt-tq] 집계)
산출: profiles/<model>/<session>/ (events, cpu_jobs.csv, gpu trace, perf report, pcm csv, observer_overhead)
사용: probes.py <model>"""
import json, os, sys, time, glob, gzip, re, subprocess, shutil
os.environ["IDE071_FEAT"] = os.path.expanduser("~/projects/vllm_hybrid/shadow_assists/features/IDE_073")
sys.path.insert(0, os.path.expanduser("~/projects/vllm_hybrid/eval/ide071")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from run_cell import build_args, WORKLOADS, smoke_greedy4
from run_campaign import cfg, ledger, load, save, usage
CAMP = f"{REPO}/eval/results/IDE_073_20260917"; PROF_HOST = f"{HOME}/.cache/huggingface/kt/ide073/profiles"; PROF_CN = "/models/kt/ide073/profiles"

def boot_opt(model, extra_env, out):
    sa, env, gpus, pin, patch = cfg(model, "OPT4"); env = dict(env); env.update(extra_env)
    stop_server()
    if patch: sh(f"bash {REPO}/eval/ide070/patch_per_layer_experts.sh {CN}", timeout=120)
    b = boot_server(build_args(sa), env, gpus, out, timeout=900); jdump({"server_args": sa, "env": env, "gpus": gpus, "pin": pin, "patch": patch}, f"{out}/requested_config.json")
    if b["verdict"] != "HEALTH_OK": return b, None, None
    pids = all_server_host_pids(); kt = kt_worker_cpus(pids)
    if pin:
        free = sorted(set(range(224)) - kt - {c + 112 for c in kt if c < 112} - {c - 112 for c in kt if c >= 112}); n = pin_nonkt(set(free), pids); b["pin_moved"] = n
    jdump(thread_affinity_snapshot(pids), f"{out}/thread_affinity.json"); jdump(server_info(), f"{out}/server_info.json")
    smoke_greedy4(out, sa["served-model-name"]); return b, pids, kt

def session(model, name, out, pids, kt, wl_id, extra=""):
    os.makedirs(out, exist_ok=True); wl = dict(WORKLOADS[wl_id]); flush_cache(); time.sleep(3)
    jdump(thread_affinity_snapshot(pids), f"{out}/thread_affinity_start.json"); coll = Collectors(out, pids); coll.start(kt_cpus=kt)
    t0 = now(); b = run_bench(wl, 64, 128, {"qwen": "qwen480", "glm": "glm47"}[model], out, wl["seed"], timeout=900, extra=extra); t1 = now(); coll.stop()
    m = {"session": name, "workload": wl_id, "rc": b["rc"], "wall_seconds": b["wall_seconds"], "summary": b["summary"], "t_start": t0, "t_end": t1, "healthy_after": health()}
    jdump(m, f"{out}/metrics.json"); ledger({"event": "session_end", "cell": f"{model}-{name}", "kind": "DIAG", "rep": name, "workload": wl_id, "sent": 128, "completed": (b["summary"] or {}).get("completed"), "failed": (b["summary"] or {}).get("failed"), "valid": bool(b["summary"]) and (b["summary"] or {}).get("failed") == 0, "output_tps": (b["summary"] or {}).get("output_throughput")})
    return m

def main():
    model = sys.argv[1]; s = load(); PW = f"M073_{model.upper()}_PROBE128"; base = f"{CAMP}/{model}-PROBES"; os.makedirs(f"{PROF_HOST}/{model}", exist_ok=True)
    only = sys.argv[2] if len(sys.argv) > 2 else "AB"; suffix = sys.argv[3] if len(sys.argv) > 3 else ""   # 재시도: probes.py qwen A _retry
    res = json.load(open(f"{base}/probes_summary.json")) if os.path.exists(f"{base}/probes_summary.json") and suffix else {"model": model, "sessions": {}}
    # ---- boot A
    outA = f"{base}/bootA{suffix}"
    if "A" not in only: b = {"verdict": "SKIPPED"}
    os.makedirs(outA, exist_ok=True); s["current"] = {"cell": f"{model}-PROBES-bootA{suffix}"}; save(s)
    if "A" in only:
        ledger({"event": "boot_start", "cell": f"{model}-PROBES-bootA{suffix}", "kinds_reserved": ["DIAG"] if "2" in only else ["DIAG", "DIAG", "DIAG"]})
        b, pids, kt = boot_opt(model, {"SGLANG_TORCH_PROFILER_DIR": f"{PROF_CN}/{model}"}, outA); ledger({"event": "boot_end", "cell": f"{model}-PROBES-bootA{suffix}", "verdict": b["verdict"], "boot_seconds": b["boot_seconds"]})
    if "A" in only and b["verdict"] == "HEALTH_OK":
        run_bench(dict(WORKLOADS[PW]), 16, 32, {"qwen": "qwen480", "glm": "glm47"}[model], f"{outA}/warmup", 1, timeout=600)
        d2only = "2" in only   # D2 재실행 전용 (probes.py qwen A2 _retry2): D0·D3 는 이전 결과 유지
        if not d2only: res["sessions"]["D0"] = session(model, "D0_PROBE_OFF", f"{outA}/D0", pids, kt, PW)
        if not d2only and not res["sessions"]["D0"].get("healthy_after"):   # 서버 종료 → 후속 세션 전송 금지 (§3.3/§9)
            for k in ("D2", "D3"): res["sessions"][k] = {"session": k, "rc": None, "summary": None, "status": "NOT_RUN_PRECONDITION_FAILED", "reason": "D0 중 서버 종료"}
            ledger({"event": "not_sent", "cell": f"{model}-PROBES-bootA{suffix}", "sessions": ["D2", "D3"], "reason": "NOT_RUN_PRECONDITION_FAILED"}); save_server_log(outA)
        else:
          # D2: torch profiler (CPU+CUDA), 같은 128 요청. 트레이스는 profile id 로 /models/kt/ide073/profiles/<model>/ 에 저장
          pid_tag = f"D2_{int(time.time())}"
          r = sh(f"curl -s -X POST http://127.0.0.1:{PORT}/start_profile -H 'Content-Type: application/json' -d '{json.dumps({'output_dir': f'{PROF_CN}/{model}', 'activities': ['CPU', 'GPU'], 'with_stack': False, 'record_shapes': False, 'profile_id': pid_tag})}'", timeout=60)
          d2 = session(model, "D2_GPU_TIMELINE", f"{outA}/D2", pids, kt, PW); d2["start_profile_resp"] = r.stdout[:300]
          r2 = sh(f"curl -s -X POST http://127.0.0.1:{PORT}/stop_profile", timeout=600); d2["stop_profile_resp"] = r2.stdout[:300]; time.sleep(20)
          traces = sorted(glob.glob(f"{PROF_HOST}/{model}/*{pid_tag}*")) or sorted(glob.glob(f"{PROF_HOST}/{model}/*.trace.json.gz"), key=os.path.getmtime)[-4:]
          d2["trace_files"] = [(os.path.basename(t), os.path.getsize(t)) for t in traces]; d2["profiler_activities"] = ["CPU", "GPU"]; jdump(d2, f"{outA}/D2/metrics.json"); res["sessions"]["D2"] = d2
          if d2only: save_server_log(outA); jdump(res, f"{base}/probes_summary.json"); stop_server(); sh(f"bash {REPO}/eval/ide070/patch_per_layer_experts.sh {CN} --revert", timeout=120); s["current"] = None; save(s); return
          # D3: perf record 99 Hz 20 s (스케줄러 4 프로세스 = kt 워커 포함), 이어서 pcm-memory 20 s (별도 창)
          sched = ",".join(str(p["pid"]) for p in sched_pids()); outD3 = f"{outA}/D3"; os.makedirs(outD3, exist_ok=True)
          subprocess.Popen(f"sleep 15; sudo perf record -F 99 -g -p {sched} -o {outD3}/perf.data -- sleep 20 > {outD3}/perf_record.log 2>&1; sudo perf report -f -i {outD3}/perf.data --stdio --sort comm,dso,sym 2>/dev/null | head -400 > {outD3}/perf_report_top.txt; sudo perf stat -p {sched} -e task-clock,cycles,instructions,context-switches,cpu-migrations -o {outD3}/perf_stat_20s.txt -- sleep 20 > /dev/null 2>&1", shell=True)
          subprocess.Popen(f"sleep 60; sudo pcm-memory 2 -csv={outD3}/pcm_memory.csv > {outD3}/pcm_memory.log 2>&1 & sleep 22; sudo pkill -INT -x pcm-memory", shell=True)
          res["sessions"]["D3"] = session(model, "D3_CPU_TIMELINE", f"{outA}/D3", pids, kt, PW); time.sleep(30)
          sh(f"sudo chown -R $(id -u):$(id -g) {outD3}")
          save_server_log(outA)
    stop_server(); sh(f"bash {REPO}/eval/ide070/patch_per_layer_experts.sh {CN} --revert", timeout=120)
    # ---- boot B: D1 LIGHT_PROBE
    outB = f"{base}/bootB{suffix}"; b2 = {"verdict": "SKIPPED"}
    if "B" in only:
        os.makedirs(outB, exist_ok=True); s["current"] = {"cell": f"{model}-PROBES-bootB{suffix}"}; save(s)
        ledger({"event": "boot_start", "cell": f"{model}-PROBES-bootB{suffix}", "kinds_reserved": ["DIAG"]})
        b2, pids, kt = boot_opt(model, {"KT_PHASE_PROF": "1", "KT_TQ_TIMING": "1"}, outB); ledger({"event": "boot_end", "cell": f"{model}-PROBES-bootB{suffix}", "verdict": b2["verdict"], "boot_seconds": b2["boot_seconds"]})
    if "B" in only and b2["verdict"] == "HEALTH_OK":
        run_bench(dict(WORKLOADS[PW]), 16, 32, {"qwen": "qwen480", "glm": "glm47"}[model], f"{outB}/warmup", 1, timeout=600)
        r0 = dexec(CN, f"wc -l < {LOG_IN_CN}"); n0 = int(r0.stdout.strip() or 0)
        res["sessions"]["D1"] = session(model, "D1_LIGHT_PROBE", f"{outB}/D1", pids, kt, PW)
        r = dexec(CN, f"tail -n +{n0+1} {LOG_IN_CN} | grep -E 'Profiling Results|kt-tq|kt-cf|kt-wrap' | head -20000"); open(f"{outB}/D1/cpu_jobs_raw.txt", "w").write(r.stdout)
        rows = []
        for l in r.stdout.splitlines():
            m = re.search(r"numa\[(\d)\]\): activated_expert: (\d+), prepare: (\d+) us, cpy_input: (\d+) us, q_input: (\d+) us, up_gate: (\d+) us, act: (\d+) us, q_down: (\d+) us, down: (\d+) us, weight: (\d+) us, total: (\d+) us, max_local_num: (\d+), qlen: (\d+)", l)
            if m: rows.append(",".join(m.groups()))
        open(f"{outB}/D1/cpu_jobs.csv", "w").write("numa,activated_expert,prepare_us,cpy_input_us,q_input_us,up_gate_us,act_us,q_down_us,down_us,weight_us,total_us,max_local_num,qlen\n" + "\n".join(rows) + "\n")
        res["sessions"]["D1"]["cpu_jobs_rows"] = len(rows); res["sessions"]["D1"]["tq_lines"] = sum(1 for l in r.stdout.splitlines() if "kt-tq" in l); jdump(res["sessions"]["D1"], f"{outB}/D1/metrics.json")
        save_server_log(outB)
    stop_server(); sh(f"bash {REPO}/eval/ide070/patch_per_layer_experts.sh {CN} --revert", timeout=120)
    # 간섭 비교
    d0 = (res["sessions"].get("D0") or {}).get("summary") or {}
    for k in ("D1", "D2", "D3"):
        m = res["sessions"].get(k)
        if m and m.get("summary") and d0.get("output_throughput"):
            m["throughput_ratio"] = m["summary"]["output_throughput"] / d0["output_throughput"]; m["elapsed_ratio"] = m["summary"]["duration"] / d0["duration"]
            m["ratio_note"] = "D0 이 완료 요청 128 미만이면 비교 무효" if (d0.get("completed") or 0) < 128 else None
    jdump(res, f"{base}/probes_summary.json"); s["current"] = None; s["usage"] = usage(); save(s); print(json.dumps({k: {kk: v.get(kk) for kk in ("rc", "throughput_ratio", "elapsed_ratio")} for k, v in res["sessions"].items()}))

if __name__ == "__main__":
    main()
