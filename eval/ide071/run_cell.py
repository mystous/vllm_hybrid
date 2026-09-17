#!/usr/bin/env python3
"""IDE_071 — 셀 1개 실행기 (지시서 §16.1 상태 머신).
PENDING→PREFLIGHT→BOOTING→VERIFY_EFFECTIVE_CONFIG→WARMUP→CACHE_PREPARE→MEASURING→DRAINING→PERSISTING→COMPLETED
오류 → CAPTURE_FAILURE → CLEANUP_OWN_PROCESSES → 종료 상태. 이 프로세스는 예외를 밖으로 던지지 않고 status 로 반환한다.
사용: run_cell.py <campaign_dir> <cell.json> [attempt_id]
"""
import gzip, json, os, re, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

WORKLOADS = json.load(open(f"{os.path.dirname(os.path.abspath(__file__))}/configs/workloads.json"))
R0 = json.load(open(f"{os.path.dirname(os.path.abspath(__file__))}/configs/r0.json"))

def build_args(sa):
    parts = []
    for k, v in sa.items():
        if v is None or v is False: continue
        if v is True: parts.append(f"--{k}")
        elif isinstance(v, list): parts.append(f"--{k} " + " ".join(str(x) for x in v))
        else: parts.append(f"--{k} {v}")
    return " ".join(parts)

def log(cell_dir, msg):
    line = f"{now()['wall_kst']} {msg}"
    print(line, flush=True)
    with open(f"{cell_dir}/cell.log", "a") as f: f.write(line + "\n")

def set_state(st, cell_dir, state, **kw):
    st["state"] = state; st["state_history"].append({"state": state, "t": now()}); st.update(kw)
    jdump(st, f"{cell_dir}/status.json")

def runtime_proofs(cell, sinfo):
    """서버 로그·server_info 에서 설정 적용 증거를 수집한다 (지시서 §3.2, §5 R06)."""
    r = dexec(CN, f"grep -c 'kt-cf\\] callback-free handoff ready' {LOG_IN_CN}; grep -c 'IDE_070 per-layer gpu experts' {LOG_IN_CN}; "
                  f"grep -oE 'KV Cache is allocated.*|#tokens: [0-9]+|max_total_num_tokens=[0-9]+' {LOG_IN_CN} | head -3; "
                  f"grep -oE 'Capture cuda graph.*bs.*' {LOG_IN_CN} | head -2; grep -oE 'attention backend.*|Attention backend.*' {LOG_IN_CN} | head -2; "
                  f"grep -c 'IDE_070 per-layer gpu experts: layer' {LOG_IN_CN}")
    lines = r.stdout.splitlines()
    proofs = {"cf_ready_lines": lines[0] if lines else None, "per_layer_log_lines": lines[1] if len(lines) > 1 else None, "raw_grep": lines[2:]}
    r3 = dexec(CN, f"grep -m1 'server_args=ServerArgs' {LOG_IN_CN}; echo ==CAPTURE==; grep -iE 'capture' {LOG_IN_CN} | head -20")
    sa_line, _, cap = r3.stdout.partition("==CAPTURE==")
    proofs["server_args_log_line"] = sa_line.strip()[:20000]; proofs["capture_log_lines"] = [l for l in cap.strip().splitlines()][:20]
    proofs["server_args_log_graph_fields"] = re.findall(r"(cuda_graph[a-z_]*|disable_cuda_graph[a-z_]*|chunked_prefill_size|page_size|kv_cache_dtype|max_running_requests|schedule_policy)=([^,]*)", sa_line)
    args = sinfo if isinstance(sinfo, dict) else {}
    keys = ["kt_num_gpu_experts", "kt_max_deferred_experts_per_token", "kt_cpuinfer", "kt_threadpool_count", "max_total_tokens", "mem_fraction_static",
            "kv_cache_dtype", "cuda_graph_max_bs", "cuda_graph_bs", "disable_cuda_graph", "attention_backend", "prefill_attention_backend", "decode_attention_backend",
            "chunked_prefill_size", "max_running_requests", "schedule_conservativeness", "schedule_policy", "num_continuous_decode_steps", "scheduler_recv_interval",
            "disable_overlap_schedule", "enable_mixed_chunk", "page_size", "moe_runner_backend", "disable_custom_all_reduce", "init_expert_location", "ep_dispatch_algorithm",
            "speculative_algorithm", "max_prefill_tokens", "tp_size", "ep_size", "cuda_graph_backend_prefill", "cuda_graph_backend_decode", "triton_attention_num_kv_splits",
            "tokenizer_worker_num", "detokenizer_worker_num", "enable_dp_attention", "dp_size", "prefill_max_requests", "disable_cuda_graph_padding"]
    proofs["effective_server_args"] = {k: args.get(k) for k in keys if k in args}
    proofs["effective_extra"] = {k: args.get(k) for k in ("max_total_num_tokens", "max_req_num", "version", "model_path", "kv_cache_dtype") if k in args}
    # per-layer 검증: 층별 expert 수 62개
    if cell.get("env", {}).get("KT_GPU_EXPERTS_PER_LAYER"):
        r2 = dexec(CN, f"grep -oE 'per-layer gpu experts: layer [0-9]+ -> [0-9]+' {LOG_IN_CN} | sort -u")
        per = {}
        for l in r2.stdout.splitlines():
            m = re.search(r"layer (\d+) -> (\d+)", l)
            if m: per[int(m.group(1))] = int(m.group(2))
        proofs["per_layer_effective"] = [per.get(i) for i in range(62)]; proofs["per_layer_sum"] = sum(v for v in per.values())
    return proofs


def smoke_greedy4(cell_dir, model):
    """부팅 smoke: IDE_068 lib.sh 와 같은 greedy 4문항 (completions 3 + chat 1), temperature 0."""
    import urllib.request
    out = []
    def post(path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=600).read())
    for pmt in ["The capital of France is", "def fibonacci(n):", "1+2+3+...+100 ="]:
        try: r = post("/v1/completions", {"model": model, "prompt": pmt, "max_tokens": 48, "temperature": 0}); out.append({"prompt": pmt, "text": r["choices"][0]["text"], "finish_reason": r["choices"][0].get("finish_reason"), "usage": r.get("usage")})
        except Exception as e: out.append({"prompt": pmt, "error": str(e)})
    try:
        r = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "Write a Python function that reverses a string. Just the code."}], "max_tokens": 96, "temperature": 0})
        out.append({"prompt": "chat:reverse_string", "text": r["choices"][0]["message"]["content"], "finish_reason": r["choices"][0].get("finish_reason"), "usage": r.get("usage")})
    except Exception as e: out.append({"prompt": "chat:reverse_string", "error": str(e)})
    jdump(out, f"{cell_dir}/smoke_greedy4.json"); return out

def gsm40(cell_dir, model):
    """기존 하네스 eval/harness/20260909_mechanism/gsm_eval.py (GSM8K test 앞 40문항, temperature 0, max_tokens 768) → gsm40.json"""
    r = sh(f"{HOME}/venv-bench/bin/python {REPO}/eval/harness/20260909_mechanism/gsm_eval.py {cell_dir}/gsm40.json 40 {model}", timeout=3600)
    open(f"{cell_dir}/gsm40.log", "w").write(r.stdout + r.stderr); return r.stdout.strip()[-80:]

def main():
    camp_dir, cell_path = sys.argv[1], sys.argv[2]; attempt = sys.argv[3] if len(sys.argv) > 3 else "a1"
    cell = json.load(open(cell_path)); cid = cell["cell_id"]
    cell_dir = f"{camp_dir}/{cid}/{attempt}"; os.makedirs(cell_dir, exist_ok=True)
    st = {"campaign_dir": camp_dir, "cell_id": cid, "attempt_id": attempt, "state": "PENDING", "state_history": [], "reps": [], "t_created": now(), "errors": []}
    set_state(st, cell_dir, "PENDING")
    jdump(cell, f"{cell_dir}/cell_spec.json")
    coll = None; booted = False
    try:
        # ---- PREFLIGHT
        set_state(st, cell_dir, "PREFLIGHT")
        for c in (CN, BENCH_CN):
            if sh(f"{DOCKER} ps --format '{{{{.Names}}}}' | grep -qx {c}").returncode != 0:
                raise RuntimeError(f"container {c} not running")
        stop_server()
        if cell.get("patch") == "per_layer":
            r = sh(f"bash {REPO}/eval/ide070/patch_per_layer_experts.sh {CN}", timeout=120); log(cell_dir, f"patch per_layer: {r.stdout.strip()[:80]}")
        # ---- BOOTING
        set_state(st, cell_dir, "BOOTING")
        sa = dict(cell["server_args"]) if cell.get("replace_server_args") else dict(R0["server_args"])
        if not cell.get("replace_server_args"): sa.update(cell.get("server_args", {}))
        for k in cell.get("server_args_remove", []): sa.pop(k, None)
        env = dict(R0.get("env", {})); env.update(cell.get("env", {}))
        env = {k: v for k, v in env.items() if v is not None}
        gpus = cell.get("gpus", R0["gpus"])
        req = {"server_args": sa, "env": env, "gpus": gpus}
        jdump(req, f"{cell_dir}/requested_config.json")
        b = boot_server(build_args(sa), env, gpus, cell_dir, timeout=cell.get("boot_timeout", 1200), prefix=cell.get("launch_prefix", ""))
        st["boot"] = b; booted = (b["verdict"] == "HEALTH_OK")
        if not booted:
            st["errors"].append({"t": now(), "stage": "BOOTING", "verdict": b["verdict"], "error_lines": b.get("error_lines")})
            set_state(st, cell_dir, "FAILED_BOOT", exit_status="FAILED_BOOT"); stop_server(); return
        # ---- VERIFY_EFFECTIVE_CONFIG
        set_state(st, cell_dir, "VERIFY_EFFECTIVE_CONFIG")
        sinfo = server_info(); jdump(sinfo, f"{cell_dir}/server_info.json")
        proofs = runtime_proofs(cell, sinfo); jdump(proofs, f"{cell_dir}/runtime_proofs.json")
        eff = {"requested": req, "effective_server_args": proofs.get("effective_server_args"), "effective_extra": proofs.get("effective_extra"), "config_hash": config_hash(req)}
        jdump(eff, f"{cell_dir}/effective_config.json")
        mismatch = []
        for k, v in sa.items():
            ek = k.replace("-", "_")
            if ek in (proofs.get("effective_server_args") or {}):
                ev = proofs["effective_server_args"][ek]
                if isinstance(v, (int, float)) and isinstance(ev, (int, float)) and abs(float(v) - float(ev)) > 1e-9: mismatch.append((k, v, ev))
                elif isinstance(v, str) and isinstance(ev, str) and v != ev: mismatch.append((k, v, ev))
        st["config_mismatch"] = mismatch
        if cell.get("env", {}).get("KT_CALLBACK_FREE") and (proofs.get("cf_ready_lines") in (None, "0")):
            st["errors"].append({"t": now(), "stage": "VERIFY", "msg": "KT_CALLBACK_FREE set but no [kt-cf] ready line"}); mismatch.append(("KT_CALLBACK_FREE", "1", "no proof"))
        if cell.get("env", {}).get("KT_GPU_EXPERTS_PER_LAYER") and proofs.get("per_layer_sum") != cell.get("expected_slots", 5952):
            mismatch.append(("per_layer_sum", cell.get("expected_slots", 5952), proofs.get("per_layer_sum")))
        if mismatch and cell.get("strict_config", True):
            set_state(st, cell_dir, "INVALID_CONFIG", exit_status="INVALID_CONFIG"); save_server_log(cell_dir); stop_server(); return
        pids = all_server_host_pids(); st["server_pids"] = pids; kt = kt_worker_cpus(pids); st["kt_worker_cpus"] = sorted(kt)
        # pinning
        if cell.get("pin_nonkt"):
            free = sorted(set(range(224)) - kt - {c + 112 for c in kt if c < 112} - {c - 112 for c in kt if c >= 112})
            # 지시서: 비-kt 스레드는 남은 물리 코어와 그 HT 형제에 배치 (kt 물리 코어의 HT 형제는 제외)
            n = pin_nonkt(set(free), pids); st["pin"] = {"free_cpus": free, "moved_threads": n}
            # 벤치 클라이언트 컨테이너 프로세스도 예비 코어로
            log(cell_dir, f"pin_nonkt moved {n} threads to {free[:4]}..{free[-1]}")
        jdump(thread_affinity_snapshot(pids), f"{cell_dir}/thread_affinity.json")
        # ---- SMOKE (greedy 4) + WARMUP
        model = sa["served-model-name"]
        set_state(st, cell_dir, "SMOKE"); st["smoke"] = [(x.get("prompt"), (x.get("text") or x.get("error") or "")[:60]) for x in smoke_greedy4(cell_dir, model)]
        set_state(st, cell_dir, "WARMUP")
        wl0 = WORKLOADS[cell["workloads"][0]["workload_id"]] if cell.get("workloads") else WORKLOADS["SHORT_COLD"]
        w = run_bench(dict(wl0), 16, 32, model, f"{cell_dir}/warmup", wl0["seed"], timeout=900)
        st["warmup"] = {"rc": w["rc"], "wall_seconds": w["wall_seconds"], "summary": w["summary"]}
        if w["rc"] != 0 or not w["summary"] or (w["summary"].get("completed") or 0) < 32:
            st["errors"].append({"t": now(), "stage": "WARMUP", "rc": w["rc"], "stderr_tail": open(f"{cell_dir}/warmup/bench.stderr.log").read()[-1500:]})
        # ---- MEASURING (workload × rep)
        for wspec in cell["workloads"]:
            wid = wspec["workload_id"]; wl = dict(WORKLOADS[wid]); C = wspec["C"]; n = wspec.get("n") or wl["n_per_C"] * C
            reps = wspec.get("reps", 3)
            if wspec.get("request_rate"): wl["request_rate"] = wspec["request_rate"]; wl["max_concurrency"] = False
            for r in range(1, reps + 1):
                rep_id = f"{wid}_C{C}_rep{r}"; rd = f"{cell_dir}/{rep_id}"; os.makedirs(rd, exist_ok=True)
                ts = {"rep_id": rep_id, "workload": wid, "C": C, "n": n}
                set_state(st, cell_dir, "CACHE_PREPARE", current_rep=rep_id)
                ts["cache_prepare_start"] = now()
                if wl["cache_policy"].startswith("engine_cache_flush") or wl["cache_policy"] == "flush_then_prefix_only_prepare":
                    fl = flush_cache(); ts["flush"] = fl; time.sleep(3)
                if wl["cache_policy"] == "flush_then_prefix_only_prepare":
                    # 공통 prefix 만 1회 요청 (측정 프롬프트는 요청하지 않음): 같은 seed·prefix 로 num-prompts 1, output 1
                    run_bench(dict(wl, out=1), 1, 1, model, f"{rd}/prefix_prepare", wl["seed"], timeout=300)
                ts["cache_prepare_end"] = now()
                jdump(thread_affinity_snapshot(pids), f"{rd}/thread_affinity_start.json")
                coll = Collectors(rd, pids); coll.start(kt_cpus=kt)
                set_state(st, cell_dir, "MEASURING", current_rep=rep_id)
                ts["measure_start"] = now()
                b = run_bench(wl, C, n, model, rd, wl["seed"], timeout=cell.get("bench_timeout", 1800))
                ts["measure_end"] = now()
                set_state(st, cell_dir, "DRAINING", current_rep=rep_id); time.sleep(2)
                coll.stop(); coll = None
                jdump(thread_affinity_snapshot(pids), f"{rd}/thread_affinity_end.json")
                ts["drain_end"] = now(); jdump(ts, f"{rd}/timestamps.json")
                s = b["summary"] or {}
                rep_rec = {"rep_id": rep_id, "workload_id": wid, "C": C, "n": n, "rc": b["rc"], "wall_seconds": b["wall_seconds"],
                           "completed": s.get("completed"), "failed": s.get("failed"), "duration": s.get("duration"),
                           "input_tokens": s.get("total_input_tokens"), "output_tokens": s.get("total_output_tokens"),
                           "output_tps": s.get("output_throughput"), "total_tps": s.get("total_token_throughput"),
                           "ttft_p50": s.get("median_ttft_ms"), "ttft_p95": s.get("p95_ttft_ms"), "ttft_p99": s.get("p99_ttft_ms"),
                           "tpot_p50": s.get("median_tpot_ms"), "tpot_p95": s.get("p95_tpot_ms"), "tpot_p99": s.get("p99_tpot_ms"),
                           "itl_p50": s.get("median_itl_ms"), "itl_p95": s.get("p95_itl_ms"), "e2el_p50": s.get("median_e2el_ms"), "e2el_p95": s.get("p95_e2el_ms"),
                           "valid": bool(s) and b["rc"] == 0 and (s.get("failed") or 0) == 0 and (s.get("completed") or 0) == n}
                if not rep_rec["valid"]:
                    rep_rec["invalid_reason"] = f"rc={b['rc']} completed={s.get('completed')} failed={s.get('failed')} n={n}"
                    st["errors"].append({"t": now(), "stage": "MEASURING", "rep": rep_id, "reason": rep_rec["invalid_reason"], "stderr_tail": open(f"{rd}/bench.stderr.log").read()[-1500:]})
                jdump(rep_rec, f"{rd}/metrics.json"); st["reps"].append(rep_rec); jdump(st, f"{cell_dir}/status.json")
                log(cell_dir, f"{rep_id}: tps={rep_rec['output_tps']} tpot50={rep_rec['tpot_p50']} ttft50={rep_rec['ttft_p50']} ok={rep_rec['completed']}/{n} valid={rep_rec['valid']}")
                if not health():
                    st["errors"].append({"t": now(), "stage": "MEASURING", "rep": rep_id, "reason": "server not healthy after rep"})
                    set_state(st, cell_dir, "FAILED_RUNTIME", exit_status="FAILED_RUNTIME"); save_server_log(cell_dir); stop_server(); return
        # ---- REPLAY (요청 시): 워밍업과 같은 생성 인자 (SHORT_COLD seed, num-prompts 32, C16) 를 같은 서버에서 재전송 = WARM_SERVER_REPLAY
        if cell.get("replay") and health():
            set_state(st, cell_dir, "REPLAY"); rd = f"{cell_dir}/REPLAY_C16"; os.makedirs(rd, exist_ok=True)
            jdump(thread_affinity_snapshot(pids), f"{rd}/thread_affinity_start.json"); coll = Collectors(rd, pids); coll.start(kt_cpus=kt)
            b = run_bench(dict(wl0), 16, 32, model, rd, wl0["seed"], timeout=cell.get("replay_timeout", 300)); coll.stop(); coll = None
            s2 = b["summary"] or {}
            st["replay"] = {"rc": b["rc"], "wall_seconds": b["wall_seconds"], "completed": s2.get("completed"), "failed": s2.get("failed"), "output_tps": s2.get("output_throughput"), "healthy_after": health(),
                            "kind": "WARM_SERVER_REPLAY", "input_note": "ORIGINAL_FAILURE_INPUT_UNAVAILABLE: IDE_071 load1024/a1 워밍업 프롬프트 원문 미보존 → 같은 생성 인자(sonnet seed, 32 요청, C16) 로 재생성; token 동일성은 seed 만으로 단정하지 않음"}
            jdump(st["replay"], f"{rd}/metrics.json"); log(cell_dir, f"replay: {st['replay']}")
            if not st["replay"]["healthy_after"]:
                st["errors"].append({"t": now(), "stage": "REPLAY", "reason": "server not healthy after replay"}); set_state(st, cell_dir, "FAILED_RUNTIME", exit_status="FAILED_RUNTIME"); save_server_log(cell_dir); stop_server(); return
        # ---- GSM20_PAIRED (요청 시)
        if cell.get("gsm20") and health():
            set_state(st, cell_dir, "GSM20"); r = sh(f"{HOME}/venv-bench/bin/python {REPO}/eval/ide071/gsm20_paired.py {cell_dir}/gsm20_paired.json {model}", timeout=1200)
            open(f"{cell_dir}/gsm20_paired.log", "w").write(r.stdout + r.stderr); st["gsm20"] = r.stdout.strip()[-120:]; log(cell_dir, f"gsm20: {st['gsm20']}")
        # ---- QUALITY20 (IDE_073: 동일 20문항, 모델별 questions.jsonl, GLM thinking off)
        if cell.get("quality20") and health():
            q = cell["quality20"]; set_state(st, cell_dir, "QUALITY20")
            r = sh(f"{HOME}/venv-bench/bin/python {REPO}/eval/ide073/quality20.py {q['questions']} {q.get('model', model)} {cell_dir}/quality20.json {'--thinking-off' if q.get('thinking_off') else ''}", timeout=1800)
            open(f"{cell_dir}/quality20.log", "w").write(r.stdout + r.stderr); st["quality20"] = r.stdout.strip()[-120:]; log(cell_dir, f"quality20: {st['quality20']}")
        # ---- GSM40 (요청 시)
        if cell.get("gsm40"):
            set_state(st, cell_dir, "GSM40"); st["gsm40"] = gsm40(cell_dir, model); log(cell_dir, f"gsm40: {st['gsm40']}")
        # ---- PERSISTING
        set_state(st, cell_dir, "PERSISTING")
        n_log = save_server_log(cell_dir); st["server_log_bytes"] = n_log
        jdump(thread_affinity_snapshot(pids), f"{cell_dir}/thread_affinity_end.json")
        st["hbm_end"] = sh("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader").stdout.strip()
        stop_server()
        valid = [r for r in st["reps"] if r["valid"]]
        if not cell.get("workloads"):   # 품질 전용 셀: 측정 반복 없음 → GSM 산출물 유무로 판정
            exit_status = "COMPLETED" if (os.path.exists(f"{cell_dir}/gsm20_paired.json") or os.path.exists(f"{cell_dir}/gsm40.json") or os.path.exists(f"{cell_dir}/quality20.json")) else "FAILED_RUNTIME"
        else:
            exit_status = "COMPLETED" if valid and len(valid) == len(st["reps"]) else "FAILED_REQUESTS"
        set_state(st, cell_dir, exit_status, exit_status=exit_status)
    except Exception as e:
        tb = traceback.format_exc()
        st["errors"].append({"t": now(), "stage": st.get("state"), "exception": str(e), "traceback": tb[-3000:]})
        set_state(st, cell_dir, "CAPTURE_FAILURE")
        try:
            if coll: coll.stop()
            if booted: save_server_log(cell_dir)
        except Exception: pass
        set_state(st, cell_dir, "CLEANUP_OWN_PROCESSES"); stop_server()
        set_state(st, cell_dir, "FAILED_RUNTIME", exit_status="FAILED_RUNTIME")
    finally:
        if cell.get("patch") == "per_layer":
            sh(f"bash {REPO}/eval/ide070/patch_per_layer_experts.sh {CN} --revert", timeout=120)
        st["t_finished"] = now(); jdump(st, f"{cell_dir}/status.json")
        # errors.jsonl
        for e in st["errors"]: jappend(e, f"{cell_dir}/errors.jsonl")

if __name__ == "__main__":
    main()
