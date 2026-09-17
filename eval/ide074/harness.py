#!/usr/bin/env python3
"""IDE_074 M2/M4/M6 — 측정 하네스. Qwen OPT4 고정 구성, 계측 모드별 부팅, 세션별 수집기 arm → 요청 → drain → 수집기 종료(생성 PID 만, wait).
모드: OFF(B: MAIN_SHORT C64 n256 / P0: PROBE128 C64 n128; 새 계측 없음), P1(KT_EVT + torch profiler CPU+GPU), P2(KT_EVT + KT_TQ_TIMING + KT_PHASE_PROF + perf stat + pcm-memory)
예산: 세션 24 · 부팅 12 (지시서 §9.2). 사용: harness.py <plan_id>   plan: B1 | B2 | B3 | F1 | F2 | SMOKE
산출: eval/results/IDE_074_20260917/qwen/OPT4/<boot_id>/<session>/ (requested/effective config, launch_cmd, bench_cmd, timestamps, metrics, requests.jsonl, 시계열, 수집기 로그)
       profiles/observer_windows.csv (append)"""
import json, os, sys, time, glob, gzip, re, subprocess, shutil, signal, csv
os.environ["IDE071_FEAT"] = os.path.expanduser("~/projects/vllm_hybrid/shadow_assists/features/IDE_074")
sys.path.insert(0, os.path.expanduser("~/projects/vllm_hybrid/eval/ide071")); sys.path.insert(0, os.path.expanduser("~/projects/vllm_hybrid/eval/ide073")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from run_cell import build_args, WORKLOADS, smoke_greedy4
import run_campaign as rc73
CAMP = f"{REPO}/eval/results/IDE_074_20260917"; ST = f"{CAMP}/state"; LEDGER = f"{ST}/execution_events.jsonl"; SP = f"{ST}/RUN_STATE.json"
KT_HOST = f"{HOME}/.cache/huggingface/kt/ide074"; KT_CN = "/models/kt/ide074"   # 컨테이너 sgl-kt 는 ~/.cache/huggingface → /models 마운트
BUDGET = {"SESSIONS": 24, "BOOT": 12}
os.makedirs(ST, exist_ok=True); os.makedirs(KT_HOST, exist_ok=True)


def ledger(ev): ev["t"] = now(); jappend(ev, LEDGER)
def load(): s = json.load(open(SP)) if os.path.exists(SP) else {"campaign_id": "IDE_074_20260917", "boots": {}}; s.setdefault("boots", {}); return s
def save(s): jdump(s, SP)
def usage():
    u = {"SESSIONS": 0, "BOOT": 0}
    for e in (json.loads(l) for l in open(LEDGER)) if os.path.exists(LEDGER) else []:
        if e["event"] == "session_end": u["SESSIONS"] += 1
        if e["event"] == "boot_end": u["BOOT"] += 1
    return u


MODE_ENV = {"OFF": {}, "P1": {"KT_EVT": None}, "P2": {"KT_EVT": None, "KT_TQ_TIMING": "1", "KT_PHASE_PROF": "1"}}


def boot(mode, boot_id):
    u = usage()
    if u["BOOT"] >= BUDGET["BOOT"]: raise SystemExit(f"BOOT budget exhausted {u}")
    sa, env, gpus, pin, patch = rc73.cfg("qwen", "OPT4"); env = dict(env)
    out = f"{CAMP}/qwen/OPT4/{boot_id}"; os.makedirs(out, exist_ok=True); os.makedirs(f"{KT_HOST}/{boot_id}/profiles", exist_ok=True)
    for k, v in MODE_ENV[mode].items(): env[k] = v if v is not None else f"{KT_CN}/{boot_id}/kt_evt.csv"
    env["SGLANG_TORCH_PROFILER_DIR"] = f"{KT_CN}/{boot_id}/profiles"
    stop_server(); sh(f"bash {REPO}/eval/ide070/patch_per_layer_experts.sh {CN}", timeout=120)
    ledger({"event": "boot_start", "boot_id": boot_id, "mode": mode, "env": env})
    b = boot_server(build_args(sa), env, gpus, out, timeout=900)
    jdump({"server_args": sa, "env": env, "gpus": gpus, "pin": pin, "patch": patch, "mode": mode, "kt_kernel_so_sha256": sh(f"{DOCKER} exec {CN} sha256sum /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so")["out"] if False else subprocess.run(f"{DOCKER} exec {CN} sha256sum /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so", shell=True, capture_output=True, text=True).stdout.strip()}, f"{out}/requested_config.json")
    ledger({"event": "boot_end", "boot_id": boot_id, "mode": mode, "verdict": b["verdict"], "boot_seconds": b.get("boot_seconds")})
    if b["verdict"] != "HEALTH_OK": save_server_log(out); return b, None, None, out
    pids = all_server_host_pids(); kt = kt_worker_cpus(pids)
    if pin:
        free = sorted(set(range(224)) - kt - {c + 112 for c in kt if c < 112} - {c - 112 for c in kt if c >= 112}); b["pin_moved"] = pin_nonkt(set(free), pids)
    jdump(thread_affinity_snapshot(pids), f"{out}/thread_affinity.json"); jdump(server_info(), f"{out}/server_info.json")
    smoke_greedy4(out, sa["served-model-name"])                                     # 부팅당 smoke 1회 (M5: OFF/ON 출력 비교)
    wl = dict(WORKLOADS["M073_QWEN_PROBE128"]); run_bench(wl, 16, 32, "qwen480", f"{out}/warmup", 1, timeout=600)   # 부팅당 warmup 1회 (IDE_073 과 동일 C16·32요청)
    runtime_proofs(out)
    s = load(); s["boots"][boot_id] = {"mode": mode, "verdict": b["verdict"], "pids": pids, "kt_cpus": sorted(kt)}; save(s)
    return b, pids, kt, out


def runtime_proofs(out):
    """서버 로그에서 실효 증빙: per-layer 행 수, [kt-cf] ready, [kt-evt] ready, capture 로그."""
    save_server_log(out)   # server.full.log.gz (docker logs 는 exec 로 띄운 서버 출력을 담지 않음)
    import gzip as _gz
    log = "\n".join(l for l in _gz.open(f"{out}/server.full.log.gz", "rt", errors="replace") if re.search(r"per-layer gpu experts|\[kt-cf\] callback-free|\[kt-evt\]|Capture target decode CUDA graph begin|KV Cache is allocated", l))
    lines = log.splitlines(); per = [int(m.group(1)) for l in lines for m in [re.search(r"layer \d+ -> (\d+)", l)] if m]
    jdump({"per_layer_effective": per, "per_layer_sum": sum(per), "cf_ready_lines": sum("callback-free handoff ready" in l for l in lines), "kt_evt_lines": [l for l in lines if "[kt-evt]" in l][:4], "capture_lines": [l for l in lines if "Capture target" in l][:4], "kv_lines": [l for l in lines if "KV Cache" in l][:2]}, f"{out}/runtime_proofs.json")


def server_log_tail_count():
    r = subprocess.run(f"{DOCKER} exec {CN} sh -c 'wc -l < {LOG_IN_CN}'", shell=True, capture_output=True, text=True); return int(r.stdout.strip() or 0)


def first_request_in_log(since_lines):
    """세션 시작 이후 서버 로그의 첫 요청 관련 행 (Prefill batch / POST /v1/completions) 타임스탬프 (초 해상도)."""
    r = subprocess.run(f"{DOCKER} exec {CN} sh -c \"tail -n +{since_lines + 1} {LOG_IN_CN} | grep -m1 -E 'Prefill batch|POST /v1/completions'\"", shell=True, capture_output=True, text=True); l = r.stdout.strip()
    m = re.search(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", l); return {"line": l[:160], "wall_utc_s_res": m.group(1) if m else None}


def drain(timeout=30):
    """마지막 응답 뒤 GPU 가 유휴(사용률 0 이 2 s 연속)가 될 때까지 대기. 요청 처리시간에 더하지 않고 별도 기록."""
    t0 = now(); idle = 0
    while now()["monotonic"] - t0["monotonic"] < timeout:
        u = subprocess.run("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0,1,2,3", shell=True, capture_output=True, text=True).stdout.split()
        idle = idle + 1 if all(x.strip() == "0" for x in u) else 0
        if idle >= 2: break
        time.sleep(1)
    return {"t_drain_end": now(), "drain_seconds": now()["monotonic"] - t0["monotonic"], "gpu_idle_confirmed": idle >= 2}


def session(mode, kind, name, wl_id, C, n, boot_id, out_root, pids, kt):
    u = usage()
    if u["SESSIONS"] >= BUDGET["SESSIONS"]: raise SystemExit(f"SESSION budget exhausted {u}")
    out = f"{out_root}/{name}"; os.makedirs(out, exist_ok=True); wl = dict(WORKLOADS[wl_id]); flush_cache(); time.sleep(3)
    jdump(thread_affinity_snapshot(pids), f"{out}/thread_affinity_start.json")
    obs = {"mode": mode, "collectors": []}; procs = {}
    log_before = server_log_tail_count()
    evt_host = f"{KT_HOST}/{boot_id}/kt_evt.csv"; evt_lines0 = sum(1 for _ in open(evt_host)) if os.path.exists(evt_host) else None
    coll = Collectors(out, pids); coll.start(kt_cpus=kt); obs["collectors"].append({"collector_id": "timeseries(cpu/gpu/mem)", "actual_start": now(), "clock_domain": "host CLOCK_REALTIME+monotonic"})
    if mode == "P1":
        pid_tag = f"{name}_{int(time.time())}"
        r = sh(f"curl -s -X POST http://127.0.0.1:{PORT}/start_profile -H 'Content-Type: application/json' -d '{json.dumps({'output_dir': f'{KT_CN}/{boot_id}/profiles', 'activities': ['CPU', 'GPU'], 'with_stack': False, 'record_shapes': False, 'profile_id': pid_tag, 'start_step': 1})}'", timeout=60)
        obs["collectors"].append({"collector_id": "torch_profiler", "profile_id": pid_tag, "arm_time": now(), "arm_resp": r.stdout[:200], "note": "start_step=1 → arm 이후 첫 forward 에서 실제 시작 (트레이스 첫 이벤트로 확인)", "clock_domain": "kineto host CLOCK_REALTIME (baseTimeNanoseconds+ts)"})
    if mode == "P2":
        sched = ",".join(str(p["pid"]) for p in sched_pids())
        procs["perf_stat"] = subprocess.Popen(f"exec sudo perf stat -p {sched} -e task-clock,cycles,instructions,context-switches,cpu-migrations,cache-misses -o {out}/perf_stat.txt", shell=True, stdout=open(f"{out}/perf_stat.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        obs["collectors"].append({"collector_id": "perf_stat", "pid": procs["perf_stat"].pid, "target": sched, "actual_start": now(), "clock_domain": "host"})
        procs["pcm"] = subprocess.Popen(f"exec sudo pcm-memory 1 -csv={out}/pcm_memory.csv", shell=True, stdout=open(f"{out}/pcm_memory.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        obs["collectors"].append({"collector_id": "pcm_memory", "pid": procs["pcm"].pid, "sample_interval_us": 1000000, "actual_start": now(), "clock_domain": "host (csv 자체 타임스탬프)"})
        time.sleep(3)
    t0 = now(); b = run_bench(wl, C, n, "qwen480", out, wl["seed"], timeout=1200); t1 = now()
    d = drain()
    if mode == "P1":
        r2 = sh(f"curl -s -X POST http://127.0.0.1:{PORT}/stop_profile", timeout=600); obs["collectors"][-1]["stop_resp"] = r2.stdout[:200]; obs["collectors"][-1]["stop_time"] = now()
        for _ in range(120):
            tr = sorted(glob.glob(f"{KT_HOST}/{boot_id}/profiles/*{pid_tag}*"))
            if len(tr) >= 4 and all(time.time() - os.path.getmtime(t) > 5 for t in tr): break
            time.sleep(5)
        obs["collectors"][-1]["trace_files"] = [(os.path.basename(t), os.path.getsize(t)) for t in tr]; obs["collectors"][-1]["actual_end"] = now()
    for k, p in procs.items():
        try: os.killpg(os.getpgid(p.pid), signal.SIGINT)
        except Exception as e: obs.setdefault("errors", []).append(f"{k}: {e}")
        try: rcode = p.wait(timeout=60)
        except Exception: rcode = None
        for c in obs["collectors"]:
            if c.get("pid") == p.pid: c["actual_end"] = now(); c["exit_code"] = rcode
    coll.stop(); obs["collectors"][0]["actual_end"] = now()
    if mode == "P2": sh(f"sudo chown -R $(id -u):$(id -g) {out}")
    evt_lines1 = sum(1 for _ in open(evt_host)) if os.path.exists(evt_host) else None
    fr = first_request_in_log(log_before)
    m = {"session": name, "mode": mode, "kind": kind, "workload": wl_id, "C": C, "n": n, "rc": b["rc"], "wall_seconds": b["wall_seconds"], "summary": b["summary"], "t_start": t0, "t_end": t1, "drain": d, "healthy_after": health(),
         "first_request_server_log": fr, "kt_evt_lines": [evt_lines0, evt_lines1], "observer": obs, "boot_id": boot_id}
    jdump(m, f"{out}/metrics.json")
    valid = bool(b["summary"]) and (b["summary"] or {}).get("failed") == 0 and b["rc"] == 0
    ledger({"event": "session_end", "cell": f"qwen-OPT4-{mode}-{name}", "kind": kind, "rep": name, "boot_id": boot_id, "workload": wl_id, "C": C, "sent": n, "completed": (b["summary"] or {}).get("completed"), "failed": (b["summary"] or {}).get("failed"), "valid": valid, "output_tps": (b["summary"] or {}).get("output_throughput"), "duration": (b["summary"] or {}).get("duration")})
    with open(f"{os.environ['IDE071_FEAT']}/profiles/observer_windows.csv", "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0: w.writerow(["run_id", "boot_id", "session", "mode", "collector_id", "collector_pid", "clock_domain", "actual_start_epoch", "actual_end_epoch", "bench_t_start_epoch", "bench_t_end_epoch", "first_request_server_log_utc", "drain_end_epoch", "exit_code", "note"])
        for c in obs["collectors"]: w.writerow(["IDE_074_20260917", boot_id, name, mode, c["collector_id"], c.get("pid"), c.get("clock_domain"), (c.get("actual_start") or c.get("arm_time") or {}).get("epoch"), (c.get("actual_end") or {}).get("epoch"), t0["epoch"], t1["epoch"], fr.get("wall_utc_s_res"), d["t_drain_end"]["epoch"], c.get("exit_code"), c.get("note", "")])
    return m


PLANS = {
    "SMOKE": ("P1", [("DIAG", "S1_P1_PROBE128", "M073_QWEN_PROBE128", 64, 128)]),
    "B1": ("OFF", [("PERF", "B_r1", "M073_QWEN_MAIN_SHORT", 64, 256), ("DIAG", "P0_r1", "M073_QWEN_PROBE128", 64, 128), ("PERF", "B_r2", "M073_QWEN_MAIN_SHORT", 64, 256), ("DIAG", "P0_r2", "M073_QWEN_PROBE128", 64, 128), ("PERF", "B_r3", "M073_QWEN_MAIN_SHORT", 64, 256), ("DIAG", "P0_r3", "M073_QWEN_PROBE128", 64, 128)]),
    "B2": ("P1", [("DIAG", "P1_r1", "M073_QWEN_PROBE128", 64, 128), ("DIAG", "P1_r2", "M073_QWEN_PROBE128", 64, 128), ("DIAG", "P1_r3", "M073_QWEN_PROBE128", 64, 128)]),
    "B3": ("P2", [("DIAG", "P2_r1", "M073_QWEN_PROBE128", 64, 128), ("DIAG", "P2_r2", "M073_QWEN_PROBE128", 64, 128), ("DIAG", "P2_r3", "M073_QWEN_PROBE128", 64, 128)]),
    "F1_OFF": ("OFF", [("DIAG", "F1_C1_OFF", "M073_QWEN_LOW_CONCURRENCY", 1, 16), ("DIAG", "F1_LONG_OFF", "M073_QWEN_LONGER_PREFILL", 8, 32)]),
    "F1_ON": ("P1", [("DIAG", "F1_C1_P1", "M073_QWEN_LOW_CONCURRENCY", 1, 16), ("DIAG", "F1_LONG_P1", "M073_QWEN_LONGER_PREFILL", 8, 32)]),
}


def main():
    plan = sys.argv[1]; mode, sessions = PLANS[plan]; boot_id = f"{plan}_{time.strftime('%H%M%S')}"
    b, pids, kt, out = boot(mode, boot_id)
    if b["verdict"] != "HEALTH_OK": print("BOOT FAILED", b); ledger({"event": "plan_abort", "plan": plan, "boot_id": boot_id, "verdict": b["verdict"]}); return
    for kind, name, wl, C, n in sessions:
        m = session(mode, kind, name, wl, C, n, boot_id, out, pids, kt); print(name, m["rc"], (m["summary"] or {}).get("output_throughput"), flush=True)
        if not m["healthy_after"]: ledger({"event": "server_died", "boot_id": boot_id, "after": name}); break
    save_server_log(out); stop_server(); sh(f"bash {REPO}/eval/ide070/patch_per_layer_experts.sh {CN} --revert", timeout=120)
    print("plan done", plan, usage())


if __name__ == "__main__": main()
