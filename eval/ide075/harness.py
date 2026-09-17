#!/usr/bin/env python3
"""IDE_075 A04/A11 — 측정 하네스 v2. IDE_074 harness 를 재사용하되 모드 OFF/CORE/CORR/RESOURCE, 같은 부팅 내 세션 경계 전환, window_events.jsonl, observer_config.json,
thread_role_map.csv, perf stat -I 1000 (interval), pcm-memory 1 s, 예산 = IDE_074 잔여(세션 7·부팅 5). 사용: harness.py <plan>  plan: OFF_OPEN | CORE | OFF_CLOSE | RESERVE:<mode>:<workload>
결과: eval/results/IDE_075_20260917/qwen/OPT4/<boot>/<session>/ ; 서버 보존: ~/.cache/huggingface/kt/ide075/<boot>/"""
import json, os, sys, time, glob, gzip, re, subprocess, shutil, signal, csv
os.environ["IDE071_FEAT"] = os.path.expanduser("~/projects/vllm_hybrid/shadow_assists/features/IDE_075")
sys.path.insert(0, os.path.expanduser("~/projects/vllm_hybrid/eval/ide071")); sys.path.insert(0, os.path.expanduser("~/projects/vllm_hybrid/eval/ide073")); sys.path.insert(0, os.path.expanduser("~/projects/vllm_hybrid/eval/ide074")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from run_cell import build_args, WORKLOADS, smoke_greedy4
import run_campaign as rc73
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("harness074", os.path.expanduser("~/projects/vllm_hybrid/eval/ide074/harness.py")); H74 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(H74)   # 같은 모듈명 충돌 회피 (15:27 OFF_OPEN 부팅 낭비 원인)
CAMP = f"{REPO}/eval/results/IDE_075_20260917"; ST = f"{CAMP}/state"; LEDGER = f"{ST}/execution_events.jsonl"; SP = f"{ST}/RUN_STATE.json"; L74 = f"{REPO}/eval/results/IDE_074_20260917/state/execution_events.jsonl"
KT_HOST = f"{HOME}/.cache/huggingface/kt/ide075"; KT_CN = "/models/kt/ide075"
BUDGET_TOTAL = {"SESSIONS": 24, "BOOT": 12}
MODEL = os.environ.get("IDE075_MODEL", "qwen480")   # GLM 진단 시 glm47
os.makedirs(ST, exist_ok=True)


EXT = os.environ.get("IDE075_EXTENDED") == "1"   # 사용자 승인 확장 단계 (16:20): 상한 우회, 원장 phase=extended
def ledger(ev): ev["t"] = now(); ev["phase"] = "extended" if EXT else "base"; jappend(ev, LEDGER)
def load(): s = json.load(open(SP)) if os.path.exists(SP) else {"campaign_id": "IDE_075_20260917", "boots": {}}; s.setdefault("boots", {}); return s
def save(s): jdump(s, SP)
def usage():
    u = {"SESSIONS": 0, "BOOT": 0, "SESSIONS_074": 0, "BOOT_074": 0, "SMOKE": 0, "WARMUP": 0}
    for p, sfx in ((L74, "_074"), (LEDGER, "")):
        for e in (json.loads(l) for l in open(p)) if os.path.exists(p) else []:
            if e["event"] == "session_end": u["SESSIONS" + sfx] += 1
            if e["event"] == "boot_start": u["BOOT" + sfx] += 1   # 부팅 시도(실패·중단 포함) 계수 (지시서 §14.5)
            if e["event"] == "smoke_end" and not sfx: u["SMOKE"] += 1
            if e["event"] == "warmup_end" and not sfx: u["WARMUP"] += 1
    u["SESSIONS_REMAINING"] = BUDGET_TOTAL["SESSIONS"] - u["SESSIONS_074"] - u["SESSIONS"]; u["BOOT_REMAINING"] = BUDGET_TOTAL["BOOT"] - u["BOOT_074"] - u["BOOT"]
    return u


def wev(out, name, extra=None, clock="host CLOCK_REALTIME", precision_ns=1000):
    e = {"campaign_id": "IDE_075_20260917", "event_name": name, "clock_domain": clock, "timestamp_ns": time.time_ns(), "wall_time_iso8601": now()["wall_kst"], "process_id": os.getpid(), "source": "harness", "precision_ns": precision_ns, "validity": "OK"}
    if extra: e.update(extra)
    jappend(e, f"{out}/window_events.jsonl"); return e


def thread_role_map(pids, out):
    rows = []
    for p in pids:
        pid = p["pid"] if isinstance(p, dict) else p
        r = subprocess.run(f"ps -L -o pid,ppid,tid,comm,psr --no-headers -p {pid}", shell=True, capture_output=True, text=True)
        for l in r.stdout.splitlines():
            f = l.split(None, 4)
            if len(f) < 5: continue
            aff = subprocess.run(f"taskset -cp {f[2]} 2>/dev/null | awk -F: '{{print $2}}'", shell=True, capture_output=True, text=True).stdout.strip()
            comm = f[3]; role = "KT_NUMA_worker" if comm.startswith("kt") or "worker" in comm.lower() else ("poller" if "poll" in comm.lower() else ("scheduler" if "sched" in comm.lower() else ("tokenizer" if "token" in comm.lower() else "unknown")))
            rows.append({"pid": f[0], "ppid": f[1], "tid": f[2], "thread_name": comm, "role": role, "role_evidence": "comm 문자열 (확정 아님; affinity 로 보강)", "affinity_list": aff, "current_cpu": f[4], "perf_included": "process-level -p (스레드 상속)"})
    with open(f"{out}/thread_role_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["pid"]); w.writeheader(); w.writerows(rows)
    return len(rows)


MODE_ENV = {"OFF": {}, "CORE": {"KT_EVT": None}}
import threading
class TidSampler:
    """/proc/<pid>/task/<tid>/stat 의 utime+stime 을 1 s 마다 읽어 TID 별 CPU 시간(ticks) 시계열 저장 (comm·affinity·psr 포함). 역할은 사후 분류."""
    def __init__(self, pids, out): self.pids = [p["pid"] if isinstance(p, dict) else p for p in pids]; self.out = out; self.ev = threading.Event(); self.th = None
    def _run(self):
        f = open(f"{self.out}/cpu_time_by_tid.csv", "w"); f.write("epoch_ns,pid,tid,comm,psr,utime_ticks,stime_ticks\n"); hz = os.sysconf("SC_CLK_TCK")
        while not self.ev.wait(1.0):
            t = time.time_ns()
            for pid in self.pids:
                try: tids = os.listdir(f"/proc/{pid}/task")
                except Exception: continue
                for tid in tids:
                    try:
                        st_ = open(f"/proc/{pid}/task/{tid}/stat").read(); comm = st_[st_.index("(") + 1: st_.rindex(")")]; fld = st_[st_.rindex(")") + 2:].split()
                        f.write(f"{t},{pid},{tid},{comm},{fld[36]},{fld[11]},{fld[12]}\n")
                    except Exception: pass
            f.flush()
        f.write(f"#clk_tck={hz}\n"); f.close()
    def start(self): self.th = threading.Thread(target=self._run, daemon=True); self.th.start()
    def stop(self): self.ev.set(); self.th.join(timeout=5)


def attach(plan, boot_id):
    """하네스 크래시 후 살아 있는 서버 재사용 (새 부팅 아님). health 대기 → 핀 → smoke → warmup → 증빙. boot_end 는 verdict=HEALTH_OK(attached) 로 기록."""
    mode = "OFF" if plan.startswith("OFF") else "CORE"; out = f"{CAMP}/qwen/OPT4/{boot_id}"; os.makedirs(out, exist_ok=True)
    t0 = now()
    while not health() and now()["monotonic"] - t0["monotonic"] < 900: time.sleep(5)
    if not health(): ledger({"event": "boot_end", "boot_id": boot_id, "mode": mode, "verdict": "ATTACH_FAILED"}); return {"verdict": "ATTACH_FAILED"}, None, None, out
    so = subprocess.run(f"{DOCKER} exec {CN} sha256sum /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so", shell=True, capture_output=True, text=True).stdout.strip()
    sa, env, gpus, pin, patch = rc73.cfg("qwen", "OPT4")
    if not os.path.exists(f"{out}/requested_config.json"): jdump({"server_args": sa, "gpus": gpus, "pin": pin, "patch": patch, "mode": mode, "plan": plan, "kt_kernel_so_sha256": so, "note": "attached"}, f"{out}/requested_config.json")
    ledger({"event": "boot_end", "boot_id": boot_id, "mode": mode, "verdict": "HEALTH_OK", "note": "attached after harness crash (boot_start 는 기존 행)", "attach_wait_s": now()["monotonic"] - t0["monotonic"]})
    pids = all_server_host_pids(); kt = kt_worker_cpus(pids)
    if pin:
        free = sorted(set(range(224)) - kt - {c + 112 for c in kt if c < 112} - {c - 112 for c in kt if c >= 112}); pin_nonkt(set(free), pids)
    jdump(thread_affinity_snapshot(pids), f"{out}/thread_affinity.json"); jdump(server_info(), f"{out}/server_info.json"); thread_role_map(pids, out)
    smoke_greedy4(out, sa["served-model-name"]); ledger({"event": "smoke_end", "boot_id": boot_id, "requests": 4})
    wl = dict(WORKLOADS["M073_QWEN_PROBE128"]); run_bench(wl, 16, 32, "qwen480", f"{out}/warmup", 1, timeout=600); ledger({"event": "warmup_end", "boot_id": boot_id, "requests": 32})
    H74.runtime_proofs(out)
    s = load(); s["boots"][boot_id] = {"mode": mode, "plan": plan, "verdict": "HEALTH_OK(attached)", "pids": pids, "kt_cpus": sorted(kt), "so": so}; save(s)
    return {"verdict": "HEALTH_OK"}, pids, kt, out


def boot(plan, boot_id):
    u = usage()
    if u["BOOT_REMAINING"] <= 0 and not EXT: raise SystemExit(f"BOOT budget exhausted {u}")
    mode = "OFF" if plan.startswith("OFF") else "CORE"
    sa, env, gpus, pin, patch = rc73.cfg("qwen", "OPT4"); env = dict(env)
    out = f"{CAMP}/qwen/OPT4/{boot_id}"; os.makedirs(out, exist_ok=True); subprocess.run(f"{DOCKER} exec {CN} sh -c 'mkdir -p {KT_CN}/{boot_id}/profiles && chmod -R 777 {KT_CN}/{boot_id}'", shell=True)
    for k, v in MODE_ENV[mode].items(): env[k] = v if v is not None else f"{KT_CN}/{boot_id}/kt_evt.csv"
    env["SGLANG_TORCH_PROFILER_DIR"] = f"{KT_CN}/{boot_id}/profiles"
    stop_server(); sh(f"bash {REPO}/eval/ide070/patch_per_layer_experts.sh {CN}", timeout=120)
    ledger({"event": "boot_start", "boot_id": boot_id, "plan": plan, "mode": mode, "env": env})
    b = boot_server(build_args(sa), env, gpus, out, timeout=900)
    so = subprocess.run(f"{DOCKER} exec {CN} sha256sum /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so", shell=True, capture_output=True, text=True).stdout.strip()
    jdump({"server_args": sa, "env": env, "gpus": gpus, "pin": pin, "patch": patch, "mode": mode, "plan": plan, "kt_kernel_so_sha256": so}, f"{out}/requested_config.json")
    ledger({"event": "boot_end", "boot_id": boot_id, "mode": mode, "verdict": b["verdict"], "boot_seconds": b.get("boot_seconds")})
    if b["verdict"] != "HEALTH_OK": save_server_log(out); return b, None, None, out
    pids = all_server_host_pids(); kt = kt_worker_cpus(pids)
    if pin:
        free = sorted(set(range(224)) - kt - {c + 112 for c in kt if c < 112} - {c - 112 for c in kt if c >= 112}); b["pin_moved"] = pin_nonkt(set(free), pids)
    jdump(thread_affinity_snapshot(pids), f"{out}/thread_affinity.json"); jdump(server_info(), f"{out}/server_info.json"); thread_role_map(pids, out)
    smoke_greedy4(out, sa["served-model-name"]); ledger({"event": "smoke_end", "boot_id": boot_id, "requests": 4})
    wl = dict(WORKLOADS["M073_QWEN_PROBE128"]); run_bench(wl, 16, 32, "qwen480", f"{out}/warmup", 1, timeout=600); ledger({"event": "warmup_end", "boot_id": boot_id, "requests": 32})
    H74.runtime_proofs(out)
    s = load(); s["boots"][boot_id] = {"mode": mode, "plan": plan, "verdict": b["verdict"], "pids": pids, "kt_cpus": sorted(kt), "so": so}; save(s)
    return b, pids, kt, out


def session(mode, name, wl_id, C, n, boot_id, out_root, pids, kt, client="vllm"):
    u = usage()
    if u["SESSIONS_REMAINING"] <= 0 and not EXT: raise SystemExit(f"SESSION budget exhausted {u}")
    out = f"{out_root}/{name}"; os.makedirs(out, exist_ok=True); wl = dict(WORKLOADS[wl_id])
    wev(out, "session_begin", {"session_id": name, "boot_id": boot_id, "mode": mode}); flush_cache(); wev(out, "engine_cache_flushed"); time.sleep(3)
    jdump(thread_affinity_snapshot(pids), f"{out}/thread_affinity_start.json")
    obs = {"mode": mode, "collectors": [], "observer_config": {"kt_evt": mode != "OFF", "torch_profiler": mode == "CORR", "perf_stat_interval": mode == "RESOURCE", "pcm_memory": mode == "RESOURCE", "light_monitor": "cpu 1 s / gpu 2 s / mem 5 s (IDE_074 동일)"}}; procs = {}
    jdump(obs["observer_config"], f"{out}/observer_config.json")
    evt_host = f"{KT_HOST}/{boot_id}/kt_evt.csv"; evt_lines0 = sum(1 for _ in open(evt_host)) if os.path.exists(evt_host) else None
    coll = Collectors(out, pids); coll.start(kt_cpus=kt); obs["collectors"].append({"collector_id": "timeseries", "actual_start": now(), "clock_domain": "host"}); wev(out, "collector_arm", {"collector_id": "timeseries"})
    if mode == "CORR":
        pid_tag = f"{name}_{int(time.time())}"
        r = sh(f"curl -s -X POST http://127.0.0.1:{PORT}/start_profile -H 'Content-Type: application/json' -d '{json.dumps({'output_dir': f'{KT_CN}/{boot_id}/profiles', 'activities': ['CPU', 'GPU'], 'with_stack': False, 'record_shapes': False, 'profile_id': pid_tag, 'start_step': 1})}'", timeout=60)
        obs["collectors"].append({"collector_id": "torch_profiler", "profile_id": pid_tag, "arm_time": now(), "arm_resp": r.stdout[:200], "clock_domain": "kineto host"}); wev(out, "collector_arm", {"collector_id": "torch_profiler", "note": "start_step=1: 다음 forward 에서 실제 시작"})
    tids = None
    if mode in ("RESOURCE", "FOCUS"):
        tids = TidSampler(pids, out); tids.start(); obs["collectors"].append({"collector_id": "tid_cpu_time", "sample_interval_us": 1000000, "actual_start": now(), "clock_domain": "host", "target_scope": "서버 host PID 전체 스레드 (/proc stat)"}); wev(out, "collector_arm", {"collector_id": "tid_cpu_time"})
    if mode == "FOCUS":
        sched = ",".join(str(p["pid"]) for p in sched_pids())
        if os.environ.get("IDE075_FOCUS_SYSWIDE"):   # 확장 2차: -p 는 대상 스레드의 switch-out 만 기록(switch-in 누락) → 시스템 전체 sched_switch + CLOCK_MONOTONIC + 큰 버퍼
            cmd = f"exec sudo perf record -a -k CLOCK_MONOTONIC -m 4096 -e sched:sched_switch -o {out}/perf_sched.data"; note = "system-wide sched_switch (switch-in 포함), -k CLOCK_MONOTONIC → clock_anchors 직접 변환"
        else:
            cmd = f"exec sudo perf record -e sched:sched_switch -e sched:sched_wakeup -p {sched} -o {out}/perf_sched.data"; note = "off-CPU/runnable 귀속용 (-p: 대상 스레드 switch-out 만 기록됨 — 구간 길이 산출 불가)"
        procs["sched"] = subprocess.Popen(cmd, shell=True, stdout=open(f"{out}/perf_sched.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        obs["collectors"].append({"collector_id": "perf_sched", "pid": procs["sched"].pid, "target": sched, "cmd": cmd, "actual_start": now(), "clock_domain": "host (perf 시각 → clock_anchors 로 변환)", "note": note}); wev(out, "collector_arm", {"collector_id": "perf_sched", "cmd": cmd})
        anchors = {"realtime_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}; jappend({"campaign_id": "IDE_075_20260917", **anchors, "note": "perf 시각(CLOCK_MONOTONIC) ↔ REALTIME anchor"}, f"{out}/clock_anchors.jsonl")
        time.sleep(2)
    if mode == "RESOURCE":
        sched = ",".join(str(p["pid"]) for p in sched_pids())
        procs["perf_stat"] = subprocess.Popen(f"exec sudo perf stat -I 1000 -x , -p {sched} -e task-clock,cycles,instructions,context-switches,cpu-migrations,cache-misses -o {out}/cpu_counter_intervals.csv", shell=True, stdout=open(f"{out}/perf_stat.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        obs["collectors"].append({"collector_id": "perf_stat_interval", "pid": procs["perf_stat"].pid, "target": sched, "sample_interval_us": 1000000, "actual_start": now(), "clock_domain": "host (perf -I 상대 초; 시작 시각 = actual_start)", "target_scope": "스케줄러 4 PID 프로세스 전체(스레드 상속)"}); wev(out, "collector_arm", {"collector_id": "perf_stat_interval"})
        procs["pcm"] = subprocess.Popen(f"exec sudo pcm-memory 1 -csv={out}/pcm_memory.raw.csv", shell=True, stdout=open(f"{out}/pcm_memory.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        obs["collectors"].append({"collector_id": "pcm_memory", "pid": procs["pcm"].pid, "sample_interval_us": 1000000, "actual_start": now(), "clock_domain": "host local(Asia/Seoul) csv timestamp", "target_scope": "host 전체 소켓별/System"}); wev(out, "collector_arm", {"collector_id": "pcm_memory"})
        time.sleep(3)
    if client == "vllmw":
        # vllm bench serve + barrier 래퍼 (같은 요청 코드). ready → collectors 는 이미 arm → start 파일 → 결과는 run_bench 와 같은 raw json
        ctl_host = f"{KT_HOST}/{boot_id}/ctl_{name}"
        for x in (".ready", ".start", ".anchor.json"):
            try: os.remove(ctl_host + x)
            except FileNotFoundError: pass
        subprocess.run(f"{DOCKER} cp {REPO}/eval/ide075/vllm_bench_wrapper.py {BENCH_CN}:/tmp/vllm_bench_wrapper.py", shell=True)
        res_host_dir = f"{HOME}/.cache/huggingface/kt/ide071/bench"; tag = f"{int(time.time()*1000)}_{os.getpid()}"; res_cn = f"{res_host_dir}/{tag}.json"
        cmd = bench_cmd(wl, C, n, MODEL, res_cn, wl["seed"]).replace("vllm bench serve ", "python3 /tmp/vllm_bench_wrapper.py ", 1)
        open(f"{out}/bench_cmd.sh", "w").write("#!/usr/bin/env bash\n# container " + BENCH_CN + " (IDE075_CTL=" + ctl_host + ")\n" + cmd + "\n")
        wev(out, "client_init_begin"); cp = subprocess.Popen(f"{DOCKER} exec -e IDE075_CTL={ctl_host} {BENCH_CN} {cmd}", shell=True, stdout=open(f"{out}/bench.stdout.log", "w"), stderr=open(f"{out}/bench.stderr.log", "w"))
        t_w = now()
        while not os.path.exists(ctl_host + ".ready") and now()["monotonic"] - t_w["monotonic"] < 300: time.sleep(0.05)
        wev(out, "client_ready", {"ready_file": os.path.exists(ctl_host + ".ready")}); wev(out, "start_signal_send"); open(ctl_host + ".start", "w").write(str(time.time_ns())); t0 = now()
        try: rc = cp.wait(timeout=1200)
        except subprocess.TimeoutExpired: cp.kill(); rc = -9
        t1 = now(); raw = f"{res_host_dir}/{tag}.json"; summary = None
        if os.path.exists(raw):
            shutil.copy(raw, f"{out}/benchmark.raw.json")
            try: summary = summarize_raw(f"{out}/benchmark.raw.json", f"{out}/requests.jsonl")
            except Exception as e: summary = {"_error": str(e)}
        anc = json.load(open(ctl_host + ".anchor.json")) if os.path.exists(ctl_host + ".anchor.json") else None
        if anc and summary:
            d_ = json.load(open(f"{out}/benchmark.raw.json")); sts = d_.get("start_times") or []; ttfts = d_.get("ttfts") or []; itls = d_.get("itls") or []
            with open(f"{out}/request_results.jsonl", "w") as f:
                for i, st_ in enumerate(sts):
                    t_send = anc["realtime_ns"] + int(round((st_ - anc["perf_counter"]) * 1e9)); tt = ttfts[i] if i < len(ttfts) else None; il = itls[i] if i < len(itls) else []
                    f.write(json.dumps({"request_id": i, "t_send_ns": t_send, "t_first_token_ns": t_send + int(round(tt * 1e9)) if tt else None, "t_end_ns": t_send + int(round((tt + sum(il)) * 1e9)) if tt else None, "n_itl": len(il), "clock": "perf_counter→realtime via anchor"}) + "\n")
            if sts:
                _ends = [anc["realtime_ns"] + int(round((sts[k_] + (ttfts[k_] if k_ < len(ttfts) else 0) + sum(itls[k_] if k_ < len(itls) else []) - anc["perf_counter"]) * 1e9)) for k_ in range(len(sts))]
                wev(out, "first_request_sent", {"timestamp_ns": anc["realtime_ns"] + int(round((min(sts) - anc["perf_counter"]) * 1e9)), "source": "vllm bench start_times + anchor", "precision_ns": 1000})
                wev(out, "last_response_received", {"timestamp_ns": max(_ends), "source": "vllm bench start+ttft+Σitl", "precision_ns": 1000})
            jdump(anc, f"{out}/clock_anchors_client.json")
        b = {"rc": rc, "summary": summary, "wall_seconds": t1["monotonic"] - t0["monotonic"]}; wev(out, "client_process_end", {"rc": rc})
    elif client == "probe":
        # barrier 클라이언트 (probe_client.py, 컨테이너 vllm-h100): CLIENT_READY 확인 → START 신호 → 요청
        ctl_host = f"{KT_HOST}/{boot_id}/ctl_{name}"; ctl_cn = ctl_host; res_host = f"{KT_HOST}/{boot_id}/probe_{name}.json"; res_cn = res_host   # vllm-h100 은 ~/.cache/huggingface 를 host 와 같은 경로로 마운트 (/models 아님)
        for x in (".ready", ".start"):
            try: os.remove(ctl_host + x)
            except FileNotFoundError: pass
        subprocess.run(f"{DOCKER} cp {REPO}/eval/ide075/probe_client.py {BENCH_CN}:/tmp/probe_client.py", shell=True)
        wev(out, "client_init_begin"); _cpus = os.environ.get("IDE075_CLIENT_CPUS", ""); _ts = f"taskset -c {_cpus} " if _cpus else ""   # 클라이언트를 free 코어에 고정 (KT worker 코어 간섭 배제 시험)
        cp = subprocess.Popen(f"{DOCKER} exec {BENCH_CN} {_ts}python3 /tmp/probe_client.py {wl['path_cn']} {n} {C} {wl['out']} {MODEL} {ctl_cn} {res_cn}", shell=True, stdout=open(f"{out}/bench.stdout.log", "w"), stderr=open(f"{out}/bench.stderr.log", "w"))
        t_w = now()
        while not os.path.exists(ctl_host + ".ready") and now()["monotonic"] - t_w["monotonic"] < 300: time.sleep(0.05)
        wev(out, "client_ready", {"ready_file": os.path.exists(ctl_host + ".ready")})
        wev(out, "start_signal_send"); open(ctl_host + ".start", "w").write(str(time.time_ns())); t0 = now()
        try: rc = cp.wait(timeout=1200)
        except subprocess.TimeoutExpired: cp.kill(); rc = -9
        t1 = now(); pr = json.load(open(res_host)) if os.path.exists(res_host) else {"summary": None, "requests": []}
        b = {"rc": rc, "summary": pr.get("summary"), "wall_seconds": t1["monotonic"] - t0["monotonic"]}
        with open(f"{out}/request_results.jsonl", "w") as f:
            for r in pr.get("requests", []): f.write(json.dumps(r) + "\n")
        if b["summary"]: wev(out, "first_request_sent", {"timestamp_ns": b["summary"]["t_first_request_sent_ns"], "source": "probe_client", "precision_ns": 1000}); wev(out, "last_response_received", {"timestamp_ns": b["summary"]["t_last_response_ns"], "source": "probe_client", "precision_ns": 1000})
        open(f"{out}/bench_cmd.sh", "w").write(f"#!/usr/bin/env bash\n# container {BENCH_CN}\n{_ts}python3 /tmp/probe_client.py {wl['path_cn']} {n} {C} {wl['out']} {MODEL} {ctl_cn} {res_cn}\n")
        wev(out, "client_process_end", {"rc": rc})
    else:
        wev(out, "client_process_begin"); t0 = now(); b = run_bench(wl, C, n, MODEL, out, wl["seed"], timeout=1200); t1 = now(); wev(out, "client_process_end", {"rc": b["rc"]})
    d = H74.drain(); wev(out, "drain_confirmed", {"gpu_idle_confirmed": d["gpu_idle_confirmed"], "drain_seconds": d["drain_seconds"], "method": "nvidia-smi util 0 ×2 (보조 신호)"})
    if mode == "CORR":
        r2 = sh(f"curl -s -X POST http://127.0.0.1:{PORT}/stop_profile", timeout=600); obs["collectors"][-1]["stop_resp"] = r2.stdout[:200]; obs["collectors"][-1]["stop_time"] = now(); wev(out, "collector_stop", {"collector_id": "torch_profiler"})
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
            if c.get("pid") == p.pid: c["actual_end"] = now(); c["exit_code"] = rcode; wev(out, "collector_stop", {"collector_id": c["collector_id"], "exit_code": rcode})
    if tids: tids.stop(); wev(out, "collector_stop", {"collector_id": "tid_cpu_time"})
    coll.stop(); obs["collectors"][0]["actual_end"] = now(); wev(out, "collector_stop", {"collector_id": "timeseries"})
    if mode in ("RESOURCE", "FOCUS"): sh(f"sudo chown -R $(id -u):$(id -g) {out}")
    time.sleep(1); evt_lines1 = sum(1 for _ in open(evt_host)) if os.path.exists(evt_host) else None
    # 서버측 창 이벤트 (kt_evt 세션 슬라이스): first_go / last_deferred_end
    fg = ld = None
    if evt_lines0 is not None and evt_lines1:
        for r in csv.DictReader(l for l in open(evt_host) if not l.startswith("#")):
            g = int(r["t_go"])
            if g < t0["epoch"] * 1e9 or g > d["t_drain_end"]["epoch"] * 1e9: continue
            fg = g if fg is None else min(fg, g); ld = max(ld or 0, int(r["t_fwd_exit"] or 0), int(r["t_done_store_a"] or 0), g)
        if fg: wev(out, "first_go", {"timestamp_ns": fg, "source": "kt_evt", "precision_ns": 100}); wev(out, "last_related_deferred_end", {"timestamp_ns": ld, "source": "kt_evt", "precision_ns": 100})
    wev(out, "log_flush_end")
    m = {"session": name, "mode": mode, "client": client, "kind": "DIAG" if wl_id == "M073_QWEN_PROBE128" else "PERF", "workload": wl_id, "C": C, "n": n, "rc": b["rc"], "wall_seconds": b["wall_seconds"], "summary": b["summary"], "t_start": t0, "t_end": t1, "drain": d, "healthy_after": health(), "kt_evt_lines": [evt_lines0, evt_lines1], "first_go_ns": fg, "last_deferred_end_ns": ld, "observer": obs, "boot_id": boot_id}
    jdump(m, f"{out}/metrics.json")
    valid = bool(b["summary"]) and (b["summary"] or {}).get("failed") == 0 and b["rc"] == 0
    ledger({"event": "session_end", "cell": f"qwen-OPT4-{mode}-{name}", "kind": m["kind"], "rep": name, "boot_id": boot_id, "workload": wl_id, "C": C, "sent": n, "completed": (b["summary"] or {}).get("completed"), "failed": (b["summary"] or {}).get("failed"), "valid": valid, "output_tps": (b["summary"] or {}).get("output_throughput"), "duration": (b["summary"] or {}).get("duration")})
    return m


PLANS = {"OFF_OPEN2": [("OFF", "S8_OFF_OPEN2", "M073_QWEN_PROBE128", 64, 128, "vllm"), ("OFF", "S8b_OFF_OPEN2_probe", "M073_QWEN_PROBE128", 64, 128, "probe")],
         "CORE2": [("CORR", "S9_CORR", "M073_QWEN_PROBE128", 64, 128, "probe"), ("CORR", "S10_CORR", "M073_QWEN_PROBE128", 64, 128, "probe"), ("RESOURCE", "S11_RESOURCE", "M073_QWEN_PROBE128", 64, 128, "probe"), ("CORR", "S12_C1_CORR", "M073_QWEN_LOW_CONCURRENCY", 1, 16, "probe"), ("CORR", "S13_LONG_CORR", "M073_QWEN_LONGER_PREFILL", 8, 32, "probe"), ("FOCUS", "S14_FOCUS_sched", "M073_QWEN_PROBE128", 64, 128, "probe")],
         "OFF_CLOSE2": [("OFF", "S15_OFF_CLOSE2", "M073_QWEN_PROBE128", 64, 128, "vllm"), ("OFF", "S15b_OFF_CLOSE2_probe", "M073_QWEN_PROBE128", 64, 128, "probe")],
         "OFF_Y": [("OFF", "S21_OFF_Y_vllm", "M073_QWEN_PROBE128", 64, 128, "vllm"), ("OFF", "S21b_OFF_Y_vllmw", "M073_QWEN_PROBE128", 64, 128, "vllmw"), ("OFF", "S21c_OFF_Y_vllm2", "M073_QWEN_PROBE128", 64, 128, "vllm")],
         "FOCUS2": [("FOCUS", "S28_FOCUS2_vllmw_syswide", "M073_QWEN_PROBE128", 64, 128, "vllmw"), ("CORR", "S29_CORR_vllmw_fp8fixso", "M073_QWEN_PROBE128", 64, 128, "vllmw")],
         "CORE4": [("CORR", "S22_CORR_vllmw", "M073_QWEN_PROBE128", 64, 128, "vllmw"), ("CORR", "S23_CORR_vllmw", "M073_QWEN_PROBE128", 64, 128, "vllmw"), ("RESOURCE", "S24_RESOURCE_vllmw", "M073_QWEN_PROBE128", 64, 128, "vllmw"), ("FOCUS", "S25_FOCUS_vllmw", "M073_QWEN_PROBE128", 64, 128, "vllmw"), ("CORR", "S26_C1_CORR_vllmw", "M073_QWEN_LOW_CONCURRENCY", 1, 16, "vllmw"), ("CORR", "S27_LONG_CORR_vllmw", "M073_QWEN_LONGER_PREFILL", 8, 32, "vllmw")],
         "OFF_X": [("OFF", "S16_OFF_X_vllm", "M073_QWEN_PROBE128", 64, 128, "vllm"), ("OFF", "S16b_OFF_X_probe_pinned", "M073_QWEN_PROBE128", 64, 128, "probe"), ("OFF", "S16c_OFF_X_vllm2", "M073_QWEN_PROBE128", 64, 128, "vllm")],
         "CORE3": [("CORR", "S17_CORR_pinned", "M073_QWEN_PROBE128", 64, 128, "probe"), ("CORR", "S18_CORR_pinned", "M073_QWEN_PROBE128", 64, 128, "probe"), ("FOCUS", "S19_FOCUS_pinned", "M073_QWEN_PROBE128", 64, 128, "probe"), ("RESOURCE", "S20_RESOURCE_pinned", "M073_QWEN_PROBE128", 64, 128, "probe")],
         "OFF_OPEN": [("OFF", "S1_OFF_OPEN", "M073_QWEN_PROBE128", 64, 128)], "OFF_A": [("OFF", "S1_OFF_A", "M073_QWEN_PROBE128", 64, 128)], "OFF_B": [("OFF", "S6_OFF_B", "M073_QWEN_PROBE128", 64, 128)],
         "CORE": [("CORE", "S2_CORE", "M073_QWEN_PROBE128", 64, 128), ("CORR", "S3_CORR", "M073_QWEN_PROBE128", 64, 128), ("CORR", "S4_CORR", "M073_QWEN_PROBE128", 64, 128), ("RESOURCE", "S5_RESOURCE", "M073_QWEN_PROBE128", 64, 128)],
         "OFF_CLOSE": [("OFF", "S6_OFF_CLOSE", "M073_QWEN_PROBE128", 64, 128)]}


def main():
    plan = sys.argv[1]
    if plan.startswith("ATTACH:"): pass   # 아래에서 처리 (PLANS 조회 전)
    elif plan.startswith("RESERVE:"):
        _, mode, wl = plan.split(":"); C, n = {"M073_QWEN_PROBE128": (64, 128), "M073_QWEN_LOW_CONCURRENCY": (1, 16), "M073_QWEN_LONGER_PREFILL": (8, 32)}[wl]; sessions = [(mode, f"S7_{mode}_{wl.split('_')[-1]}", wl, C, n)]; plan_key = "RESERVE_" + mode
    else: sessions = PLANS[plan]; plan_key = plan
    if plan.startswith("ATTACH:"):
        _, plan_key, boot_id = plan.split(":"); sessions = PLANS[plan_key]; b, pids, kt, out = attach(plan_key, boot_id)
    else:
        boot_id = f"{plan_key}_{time.strftime('%H%M%S')}"; b, pids, kt, out = boot(plan_key if not plan.startswith("RESERVE") else ("OFF" if sessions[0][0] == "OFF" else "CORE"), boot_id)
    if b["verdict"] != "HEALTH_OK": print("BOOT FAILED", b); ledger({"event": "plan_abort", "plan": plan, "boot_id": boot_id, "verdict": b["verdict"]}); return
    for spec in sessions:
        mode, name, wl, C, n = spec[:5]; client = spec[5] if len(spec) > 5 else "vllm"
        m = session(mode, name, wl, C, n, boot_id, out, pids, kt, client=client); print(name, m["rc"], (m["summary"] or {}).get("output_throughput"), flush=True)
        if not m["healthy_after"]: ledger({"event": "server_died", "boot_id": boot_id, "after": name}); break
    save_server_log(out); stop_server(); sh(f"bash {REPO}/eval/ide070/patch_per_layer_experts.sh {CN} --revert", timeout=120)
    print("plan done", plan, usage())


if __name__ == "__main__": main()
