#!/usr/bin/env python3
"""IDE_071 공용 — 컨테이너 실행, 서버 부팅/종료, 수집기, 상태 파일.
지시서 cpu_offload_no_02 §4·§16·§18. 모든 시각은 wall(UTC/KST)+monotonic 을 함께 남긴다.
"""
import gzip, hashlib, json, os, re, shutil, signal, subprocess, sys, time, threading
from datetime import datetime, timezone, timedelta

HOME = os.path.expanduser("~")
REPO = f"{HOME}/projects/vllm_hybrid"
FEAT = os.environ.get("IDE071_FEAT", f"{REPO}/shadow_assists/features/IDE_071")
DOCKER = f"{HOME}/bin/docker"           # = sudo nerdctl 셔임
NERDCTL = "/bin/nerdctl"
CN = os.environ.get("IDE071_CN", "sgl-kt")
BENCH_CN = os.environ.get("IDE071_BENCH_CN", "vllm-h100")
PORT = int(os.environ.get("IDE071_PORT", "30000"))
LOG_IN_CN = "/tmp/ide071_server.log"
KST = timezone(timedelta(hours=9))
SNAP_FP8 = "003f183a92fbe5b9a8325aaa8b2ae797c91dd90f"
MODEL_FP8_CN = f"/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/{SNAP_FP8}"
TOK_HOST = f"{HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/{SNAP_FP8}"
SONNET_CN = "/tmp/sonnet.txt"

def now():
    t = time.time(); m = time.monotonic()
    return {"wall_utc": datetime.fromtimestamp(t, timezone.utc).isoformat(), "wall_kst": datetime.fromtimestamp(t, KST).isoformat(), "epoch": t, "monotonic": m}

def sh(cmd, timeout=600, check=False, capture=True):
    r = subprocess.run(cmd, shell=True, capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed rc={r.returncode}: {cmd}\n{r.stderr[-2000:] if capture else ''}")
    return r

def dexec(cn, cmd, timeout=600, detach=False, stdin=None, env=None):
    envs = " ".join(f"-e {k}={v}" for k, v in (env or {}).items())
    flag = "-d" if detach else ("-i" if stdin is not None else "")
    return subprocess.run(f"{DOCKER} exec {flag} {envs} {cn} bash -c {json.dumps(cmd)}", shell=True, capture_output=True, text=True, timeout=timeout, input=stdin)

def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, path)

def jappend(obj, path):
    with open(path, "a") as f: f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

def config_hash(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]

# ---------------- 서버 ----------------
def server_pids_in_cn(cn=CN):
    r = dexec(cn, "pgrep -f 'sglang.launch_serve[r]' || true"); return [int(x) for x in r.stdout.split()]

def stop_server(cn=CN):
    """이 하네스가 띄운 sglang 서버만 종료 (컨테이너 안 launch_server 프로세스 트리). 다른 컨테이너·GPU 는 건드리지 않음."""
    dexec(cn, "pkill -f 'sglang.launch_serve[r]' 2>/dev/null; sleep 5; pkill -9 -f 'sglang.launch_serve[r]' 2>/dev/null; true")
    for _ in range(30):
        used = [int(x) for x in sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").stdout.split()]
        if all(u < 1500 for u in used): break
        time.sleep(3)

def health(port=PORT):
    return sh(f"curl -sf -m 5 http://127.0.0.1:{port}/health >/dev/null", timeout=10).returncode == 0

def boot_server(args, env, gpus, out_dir, timeout=1200, cn=CN, port=PORT, prefix=""):
    """서버 기동 → health 대기. 반환: dict(verdict, seconds, ...). 전체 서버 로그는 stop 시 server.full.log.gz 로 복사."""
    os.makedirs(out_dir, exist_ok=True)
    envs = " ".join(f"{k}={v}" for k, v in env.items())
    cmd = f"{envs} CUDA_VISIBLE_DEVICES={gpus} {prefix + ' ' if prefix else ''}python3 -m sglang.launch_server {args}"
    with open(f"{out_dir}/launch_cmd.sh", "w") as f: f.write("#!/usr/bin/env bash\n# container " + cn + "\n" + cmd + "\n")
    t0 = now()
    dexec(cn, f"{cmd} > {LOG_IN_CN} 2>&1", detach=True)
    verdict = "TIMEOUT"; i = 0
    while i < timeout:
        if health(port): verdict = "HEALTH_OK"; break
        if not server_pids_in_cn(cn): verdict = "DIED"; break
        time.sleep(5); i += 5
    t1 = now()
    res = {"verdict": verdict, "boot_seconds": t1["monotonic"] - t0["monotonic"], "t_boot_start": t0, "t_boot_end": t1, "launch_cmd": cmd}
    if verdict != "HEALTH_OK":
        save_server_log(out_dir, cn)
        r = dexec(cn, f"grep -nE 'Error|error|OOM|out of memory|Traceback|illegal|failed' {LOG_IN_CN} | head -40")
        res["error_lines"] = r.stdout
    return res

def save_server_log(out_dir, cn=CN):
    r = subprocess.run(f"{DOCKER} exec {cn} cat {LOG_IN_CN}", shell=True, capture_output=True)
    with gzip.open(f"{out_dir}/server.full.log.gz", "wb") as f: f.write(r.stdout)
    return len(r.stdout)

def server_info(port=PORT):
    r = sh(f"curl -s -m 30 http://127.0.0.1:{port}/get_server_info", timeout=40)
    try: return json.loads(r.stdout)
    except Exception: return {"_raw": r.stdout[:2000]}

def flush_cache(port=PORT):
    r = sh(f"curl -s -m 60 -X POST http://127.0.0.1:{port}/flush_cache", timeout=70)
    return {"rc": r.returncode, "body": r.stdout[:500]}

def sched_pids():
    """호스트 PID: GPU compute 프로세스 (스케줄러 TP rank). GPU bus id 로 정렬."""
    r = sh("nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader")
    out = []
    for l in r.stdout.strip().splitlines():
        p = [x.strip() for x in l.split(",")]
        if len(p) >= 2: out.append({"pid": int(p[0]), "bus": p[1], "mem": p[2] if len(p) > 2 else None})
    return out

def all_server_host_pids():
    """컨테이너 안 서버 프로세스 트리의 호스트 PID (launch_server + 자식). comm 으로 역할 표기."""
    r = sh("pgrep -f 'sglang.launch_serve[r]' || true"); roots = [int(x) for x in r.stdout.split()]
    pids = set(roots)
    for root in roots:
        r2 = sh(f"pgrep -P {root} || true"); pids.update(int(x) for x in r2.stdout.split())
        for c in list(pids):
            r3 = sh(f"pgrep -P {c} || true"); pids.update(int(x) for x in r3.stdout.split())
    out = []
    for p in sorted(pids):
        try: comm = open(f"/proc/{p}/comm").read().strip()
        except Exception: continue
        out.append({"pid": p, "comm": comm})
    return out

def thread_affinity_snapshot(pids):
    """모든 서버 프로세스의 TID 별 comm·affinity. 요약 (comm 접두 별 코어 집합) 도 함께."""
    snap = {"taken": now(), "procs": []}
    for pr in pids:
        p = pr["pid"]; th = []
        try: tids = os.listdir(f"/proc/{p}/task")
        except Exception: continue
        for t in tids:
            try:
                comm = open(f"/proc/{p}/task/{t}/comm").read().strip()
                st = open(f"/proc/{p}/task/{t}/status").read()
                cpus = re.search(r"Cpus_allowed_list:\s*(\S+)", st).group(1)
                th.append({"tid": int(t), "comm": comm, "cpus": cpus})
            except Exception: pass
        snap["procs"].append({"pid": p, "comm": pr.get("comm"), "n_threads": len(th), "threads": th})
    return snap

def pin_nonkt(free_cpus, pids):
    """비-kt 스레드(numa_* 제외) 를 free_cpus 로 이동. 컨테이너 안(root) 에서 sched_setaffinity. 반환: 이동 수.
    (지시서 §6.1: 전 rank·tokenizer·detokenizer·통신 helper 포함. 호스트 비특권 사용자는 EPERM 이므로 docker exec 로 실행)"""
    spec = ",".join(str(c) for c in sorted(free_cpus))
    py = """
import os,sys
free=set(int(x) for x in sys.argv[1].split(","))
n=0;procs=0
for pid in os.listdir("/proc"):
    if not pid.isdigit(): continue
    try:
        comm=open(f"/proc/{pid}/comm").read().strip(); st=open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0]
    except Exception: continue
    if st=="Z" or not (comm.startswith("sglang::") or comm=="python3"): continue
    try: tids=os.listdir(f"/proc/{pid}/task")
    except Exception: continue
    if len(tids)<50: continue
    procs+=1
    for t in tids:
        try:
            tc=open(f"/proc/{pid}/task/{t}/comm").read().strip()
            if tc.startswith("numa_"): continue
            os.sched_setaffinity(int(t),free); n+=1
        except Exception: pass
print(n,procs)
"""
    r = dexec(CN, f"python3 - {spec}", stdin=py)
    try: return int(r.stdout.split()[0])
    except Exception: return -1

def kt_worker_cpus(pids):
    s = set()
    for pr in pids:
        p = pr["pid"]
        try: tids = os.listdir(f"/proc/{p}/task")
        except Exception: continue
        for t in tids:
            try:
                comm = open(f"/proc/{p}/task/{t}/comm").read().strip()
                if not comm.startswith("numa_"): continue
                st = open(f"/proc/{p}/task/{t}/status").read()
                cpus = re.search(r"Cpus_allowed_list:\s*(\S+)", st).group(1)
                for part in cpus.split(","):
                    a, b = (part.split("-") + [None])[:2]; s.update(range(int(a), int(b if b else a) + 1))
            except Exception: pass
    return s

# ---------------- 수집기 (저부하) ----------------
class Collectors:
    """cpu_timeseries.csv (/proc/stat 1 s, 코어별 busy%), gpu_timeseries.csv (nvidia-smi 2 s), memory_timeseries.csv (free/numastat 5 s)."""
    def __init__(self, out_dir, pids=None):
        self.out = out_dir; self.stop_ev = threading.Event(); self.threads = []; self.pids = pids or []
    def _cpu(self):
        def read():
            d = {}
            for l in open("/proc/stat"):
                if l.startswith("cpu") and l[3] != " ":
                    p = l.split(); v = list(map(int, p[1:])); idle = v[3] + v[4]; d[p[0]] = (sum(v), idle)
            return d
        prev = read(); f = open(f"{self.out}/cpu_timeseries.csv", "w")
        f.write("epoch,monotonic,all_busy,phys_busy,smt_busy,kt_cpus_busy," + ",".join(f"cpu{i}" for i in range(224)) + "\n")
        kt = self.kt_cpus if hasattr(self, "kt_cpus") else set()
        while not self.stop_ev.wait(1.0):
            cur = read(); vals = []
            for i in range(224):
                k = f"cpu{i}"
                if k in cur and k in prev:
                    dt = cur[k][0] - prev[k][0]; di = cur[k][1] - prev[k][1]; vals.append(100 * (1 - di / dt) if dt > 0 else 0.0)
                else: vals.append(0.0)
            prev = cur
            phys = [vals[i] for i in range(112)]; smt = [vals[i] for i in range(112, 224)]
            ktv = [vals[i] for i in kt] if kt else [0.0]
            t = now(); f.write(f"{t['epoch']:.3f},{t['monotonic']:.3f},{sum(vals)/224:.2f},{sum(phys)/112:.2f},{sum(smt)/112:.2f},{sum(ktv)/len(ktv):.2f}," + ",".join(f"{v:.1f}" for v in vals) + "\n"); f.flush()
        f.close()
    def _gpu(self):
        p = subprocess.Popen(f"nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active --format=csv -l 2 > {self.out}/gpu_timeseries.csv 2>&1", shell=True, preexec_fn=os.setsid)
        self.stop_ev.wait(); os.killpg(p.pid, signal.SIGTERM)
    def _mem(self):
        f = open(f"{self.out}/memory_timeseries.csv", "w"); f.write("epoch,monotonic,mem_used_gb,mem_free_gb,buff_cache_gb,node0_free_mb,node1_free_mb,rss_sched_gb\n")
        while not self.stop_ev.wait(5.0):
            t = now(); fr = sh("free -g | awk 'NR==2{print $3,$4,$6}'").stdout.split()
            nf = sh("numastat -m 2>/dev/null | awk '/MemFree/{print $2,$3}'").stdout.split()
            rss = 0
            for pr in self.pids:
                try:
                    for l in open(f"/proc/{pr['pid']}/status"):
                        if l.startswith("VmRSS"): rss += int(l.split()[1])
                except Exception: pass
            f.write(f"{t['epoch']:.3f},{t['monotonic']:.3f},{','.join(fr) if len(fr)==3 else ',,'},{','.join(nf) if len(nf)==2 else ','},{rss/1e6:.2f}\n"); f.flush()
        f.close()
    def start(self, kt_cpus=None):
        self.kt_cpus = kt_cpus or set()
        for fn in (self._cpu, self._gpu, self._mem):
            th = threading.Thread(target=fn, daemon=True); th.start(); self.threads.append(th)
    def stop(self):
        self.stop_ev.set()
        for th in self.threads: th.join(timeout=10)

# ---------------- 벤치 ----------------
def bench_cmd(workload, C, n, model, result_file_cn, seed, extra=""):
    """vllm bench serve 명령 (컨테이너 vllm-h100 안). 결과 JSON (요청별 상세 포함) 은 /models 마운트를 통해 호스트로."""
    base = (f"vllm bench serve --backend openai --base-url http://127.0.0.1:{PORT} --endpoint /v1/completions --model {model} "
            f"--tokenizer {TOK_HOST} --num-prompts {n} --request-rate {workload.get('request_rate','inf')} --seed {seed} "
            f"--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 --save-result --save-detailed "
            f"--result-dir {os.path.dirname(result_file_cn)} --result-filename {os.path.basename(result_file_cn)} {extra}")
    if workload.get("max_concurrency", True) and workload.get("request_rate", "inf") == "inf":
        base += f" --max-concurrency {C}"
    if workload.get("ignore_eos", True): base += " --ignore-eos"
    ds = workload["dataset"]
    if ds == "sonnet":
        base += (f" --dataset-name sonnet --dataset-path {SONNET_CN} --sonnet-input-len {workload['in']} --sonnet-output-len {workload['out']} "
                 f"--sonnet-prefix-len {workload.get('prefix', 0)}")
    elif ds == "random":
        base += (f" --dataset-name random --random-input-len {workload['in']} --random-output-len {workload['out']} --random-prefix-len {workload.get('prefix', 0)} "
                 f"--random-range-ratio {workload.get('range_ratio', 0.0)}")
    elif ds == "custom":
        base += f" --dataset-name custom --dataset-path {workload['path_cn']} --custom-output-len {workload['out']}"
    return base

def run_bench(workload, C, n, model, out_dir, seed, cn=BENCH_CN, timeout=1800, extra=""):
    """벤치 실행. 결과: benchmark.raw.json (vllm 저장본), bench.stdout/stderr.log, bench_cmd.sh, timestamps."""
    os.makedirs(out_dir, exist_ok=True)
    res_host_dir = f"{HOME}/.cache/huggingface/kt/ide071/bench"; os.makedirs(res_host_dir, exist_ok=True)
    tag = f"{int(time.time()*1000)}_{os.getpid()}"
    res_cn = f"{res_host_dir}/{tag}.json"   # vllm-h100 은 ~/.cache/huggingface 를 같은 경로로 마운트
    cmd = bench_cmd(workload, C, n, model, res_cn, seed, extra)
    with open(f"{out_dir}/bench_cmd.sh", "w") as f: f.write("#!/usr/bin/env bash\n# container " + cn + "\n" + cmd + "\n")
    t0 = now()
    try:
        r = subprocess.run(f"{DOCKER} exec {cn} {cmd}", shell=True, capture_output=True, text=True, timeout=timeout); rc = r.returncode; so, se = r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        rc = -9; so = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""); se = "TIMEOUT"
    t1 = now()
    open(f"{out_dir}/bench.stdout.log", "w").write(so); open(f"{out_dir}/bench.stderr.log", "w").write(se)
    raw = f"{res_host_dir}/{tag}.json"; summary = None
    if os.path.exists(raw):
        shutil.copy(raw, f"{out_dir}/benchmark.raw.json")
        try: summary = summarize_raw(f"{out_dir}/benchmark.raw.json", f"{out_dir}/requests.jsonl")
        except Exception as e: summary = {"_error": str(e)}
    return {"rc": rc, "t_start": t0, "t_end": t1, "wall_seconds": t1["monotonic"] - t0["monotonic"], "summary": summary, "raw": f"{out_dir}/benchmark.raw.json" if summary else None}

def summarize_raw(raw_path, requests_out):
    d = json.load(open(raw_path))
    keys = ["completed", "failed", "duration", "total_input_tokens", "total_output_tokens", "request_throughput", "output_throughput", "total_token_throughput",
            "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "max_ttft_ms",
            "mean_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p95_tpot_ms", "p99_tpot_ms", "max_tpot_ms",
            "mean_itl_ms", "median_itl_ms", "p95_itl_ms", "p99_itl_ms", "mean_e2el_ms", "median_e2el_ms", "p95_e2el_ms", "p99_e2el_ms", "max_concurrency", "request_rate", "num_prompts"]
    s = {k: d.get(k) for k in keys}
    # 요청별
    n = len(d.get("input_lens", []) or [])
    with open(requests_out, "w") as f:
        for i in range(n):
            itl = (d.get("itls") or [[]]*n)[i] or []
            ttft = (d.get("ttfts") or [None]*n)[i]
            e2 = (d.get("e2els") or [None]*n)[i]
            tp = (d.get("tpots") or [None]*n)[i]
            row = {"idx": i, "input_len": d["input_lens"][i], "output_len": (d.get("output_lens") or [None]*n)[i],
                   "ttft_ms": ttft * 1000 if ttft is not None else None,
                   "itl_count": len(itl), "itl_sum_ms": sum(itl) * 1000 if itl else None,
                   "tpot_ms": (tp * 1000 if tp is not None else (sum(itl) * 1000 / len(itl) if itl else None)),
                   "tpot_source": "vllm_tpots" if tp is not None else ("mean_itl" if itl else None),
                   "e2el_ms": (e2 * 1000 if e2 is not None else ((ttft + sum(itl)) * 1000 if ttft is not None and itl else None)),
                   "error": (d.get("errors") or [None]*n)[i]}
            f.write(json.dumps(row) + "\n")
    s["n_request_rows"] = n
    return s
