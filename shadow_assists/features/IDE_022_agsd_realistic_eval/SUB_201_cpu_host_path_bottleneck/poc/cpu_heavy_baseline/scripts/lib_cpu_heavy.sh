#!/usr/bin/env bash
# Common library for cpu_heavy_* rounds (C-1..C-10).
# Extends hw_heavy lib_heavy.sh by adding mpstat-based CPU utilization sampling
# during each bench sweep, so we can verify the "CPU activation" criterion
# (baseline cpu_util ~5.4% -> target ~30%+).
#
# Caller exports:
#   ROOT, MODEL, PORT, TP, MAX_MODEL_LEN, CONC, NPROMPT, MAX_TOKENS, N_SWEEPS,
#   EXTRA_ENV (array), EXTRA_CLI (array), EXTRA_LOG_TAG.

RUNS=$ROOT/runs
LOGS=$ROOT/logs
mkdir -p "$RUNS" "$LOGS"

SAMPLED=${SAMPLED:-/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/ide023_round_1/sharegpt500.parquet}
RUNNER=${RUNNER:-/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/throughput_runner.py}
VPY=${VPY:-/workspace/vllm_dev_prj/bin/python}
VLLM=${VLLM:-/workspace/vllm_dev_prj/bin/vllm}
export LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib

wait_gpu_free() {
    for i in $(seq 1 90); do
        local busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500 {c++} END{print c+0}')
        [ "$busy" -eq 0 ] && return 0
        sleep 2
    done
    return 1
}

kill_pgroup() {
    local pid=$1
    [ -z "$pid" ] && return 0
    if [ -d "/proc/$pid" ]; then
        kill -TERM -- -"$pid" 2>/dev/null || true
        for i in $(seq 1 30); do [ -d "/proc/$pid" ] || break; sleep 1; done
        [ -d "/proc/$pid" ] && kill -KILL -- -"$pid" 2>/dev/null || true
    fi
    for op in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
        [ -d "/proc/$op" ] && kill -KILL "$op" 2>/dev/null || true
    done
    sleep 3
    wait_gpu_free || true
}

wait_ready() {
    local boot_log=$1
    for i in $(seq 1 900); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        if [ -n "${boot_log:-}" ] && [ -f "$boot_log" ]; then
            if grep -qE "vllm serve: error|Engine core initialization failed|Engine startup failed|Failed core proc|^RuntimeError:|core dumped|Aborted \(core dumped\)|set_mempolicy.*Operation not permitted|ValueError:.*not support|AssertionError" "$boot_log"; then
                echo "[wait_ready] fatal" | tee -a "$boot_log"
                return 1
            fi
        fi
        sleep 2
    done
    return 1
}

start_one() {
    local tag=$1
    local boot_log=$LOGS/${tag}_boot.log
    echo "[boot] tag=$tag model=$MODEL tp=$TP port=$PORT env=[${EXTRA_ENV[*]:-}] cli=[${EXTRA_CLI[*]:-}]" | tee "$boot_log"
    local args=( env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_WORKER_MULTIPROC_METHOD=spawn )
    [ ${#EXTRA_ENV[@]} -gt 0 ] && args+=( "${EXTRA_ENV[@]}" )
    args+=( setsid "$VLLM" serve "$MODEL" \
        --tensor-parallel-size "$TP" \
        --port "$PORT" \
        --gpu-memory-utilization 0.85 \
        --max-model-len "$MAX_MODEL_LEN" )
    [ ${#EXTRA_CLI[@]} -gt 0 ] && args+=( "${EXTRA_CLI[@]}" )
    printf '[boot] arg: %s\n' "${args[@]}" >> "$boot_log"
    "${args[@]}" >> "$boot_log" 2>&1 &
    local pid=$!
    echo "$pid" > "$RUNS/${tag}.pid"
    if ! wait_ready "$boot_log"; then
        kill_pgroup "$pid"
        return 1
    fi
    return 0
}

# Start mpstat in background, write per-second total CPU usage to a file.
# Returns the mpstat PID.
start_mpstat() {
    local out=$1
    # -P ALL -u 1: every 1s, all CPUs (we only parse "all" row)
    mpstat -u 1 > "$out" 2>&1 &
    echo $!
}

stop_mpstat() {
    local pid=$1
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
}

# Parse mpstat output to mean CPU utilization (100 - idle).
parse_mpstat_avg() {
    local f=$1
    # mpstat lines look like: "12:34:56 PM all 5.10 0.00 ..."  with %idle at the end.
    # Use python for robustness (locale-independent decimal).
    python3 -c "
import re,sys
vals=[]
for line in open('$f',errors='ignore'):
    parts=line.split()
    if len(parts) < 12: continue
    if parts[1].lower() != 'all' and parts[2].lower() != 'all': continue
    # find numeric tokens; %idle is last numeric column
    try:
        nums=[float(p) for p in parts if re.match(r'^-?\d+(\.\d+)?$', p)]
    except: continue
    if len(nums) < 9: continue
    idle = nums[-1]
    vals.append(100.0 - idle)
if vals:
    # drop first sample (warm-up artefact)
    vs = vals[1:] if len(vals) > 1 else vals
    print(f'{sum(vs)/len(vs):.2f}')
else:
    print('nan')
"
}

bench_one() {
    local tag=$1 sweep=${2:-1}
    local mpf=$LOGS/${tag}_s${sweep}.mpstat
    local mppid=$(start_mpstat "$mpf")
    "$VPY" "$RUNNER" \
        --in "$SAMPLED" \
        --method "${EXTRA_LOG_TAG:-cpuh}_${tag}_s${sweep}" \
        --model "$MODEL" \
        --port "$PORT" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONC" \
        --limit "$NPROMPT" \
        --shuffle --seed $((42 + sweep)) \
        --out "$RUNS/${tag}_s${sweep}.json" \
        --raw "$RUNS/${tag}_s${sweep}.raw.jsonl" \
        2>&1 | tee -a "$LOGS/${tag}_bench.log"
    stop_mpstat "$mppid"
    local cpu_avg=$(parse_mpstat_avg "$mpf")
    echo "$cpu_avg" > "$RUNS/${tag}_s${sweep}.cpu_util"
    echo "[mpstat] tag=$tag sweep=$sweep cpu_avg=$cpu_avg %" | tee -a "$LOGS/${tag}_bench.log"
}

stop_one() {
    local tag=$1
    local pid=$(cat "$RUNS/${tag}.pid" 2>/dev/null || echo "")
    [ -n "$pid" ] && kill_pgroup "$pid"
    rm -f "$RUNS/${tag}.pid"
}

do_case_nsweep() {
    local tag=$1 n=${2:-$N_SWEEPS}
    local all_done=1
    for s in $(seq 1 $n); do [ -f "$RUNS/${tag}_s${s}.json" ] || all_done=0; done
    if [ "$all_done" -eq 1 ]; then echo "[skip] $tag"; return 0; fi
    if ! start_one "$tag"; then
        stop_one "$tag"
        echo "{\"tag\":\"$tag\",\"status\":\"boot_fail\"}" > "$RUNS/${tag}_s1.json"
        return 1
    fi
    for s in $(seq 1 $n); do
        if [ ! -f "$RUNS/${tag}_s${s}.json" ]; then
            bench_one "$tag" "$s"
        fi
        sleep 2
    done
    stop_one "$tag"
}

# Aggregator: list of tags via stdin or args -> per-tag summary line.
# Usage: summarize_tags tag1 tag2 ...
summarize_tags() {
    local out=$ROOT/summary.json
    python3 - "$@" <<'PYEOF'
import json,os,sys,glob,statistics as st
tags = sys.argv[1:]
root = os.environ.get('ROOT')
runs = os.path.join(root, 'runs')
res = []
for tag in tags:
    files = sorted(glob.glob(f"{runs}/{tag}_s*.json"))
    tps=[]; wall=[]; ttft=[]; tpot=[]; acc=[]; gpu=[]; cpu_util=[]
    for f in files:
        try:
            j = json.load(open(f))
        except: continue
        if 'status' in j and j.get('status')=='boot_fail': continue
        tps.append(j.get('output_tps') or j.get('output_throughput') or j.get('tps') or 0)
        wall.append(j.get('wall_s') or j.get('wall_time_s') or 0)
        ttft.append(j.get('ttft_p50_ms') or j.get('ttft_p50') or 0)
        tpot.append(j.get('tpot_p50_ms') or j.get('tpot_p50') or 0)
        if j.get('accept_rate') is not None: acc.append(j['accept_rate'])
        if j.get('gpu_util') is not None: gpu.append(j['gpu_util'])
        # also read sidecar cpu_util file
        cu_f = f.replace('.json', '.cpu_util')
        if os.path.exists(cu_f):
            try:
                v = float(open(cu_f).read().strip())
                cpu_util.append(v)
            except: pass
    def stat(x):
        if not x: return (None, None)
        if len(x)==1: return (round(x[0],2), 0.0)
        return (round(st.mean(x),2), round(st.stdev(x),2))
    tm,ts = stat(tps); wm,ws = stat(wall); tfm,tfs = stat(ttft); tpm,tps_ = stat(tpot)
    am,_ = stat(acc); gm,_ = stat(gpu); cum,cus = stat(cpu_util)
    res.append({
        'tag': tag,
        'n_sweeps': len(tps),
        'output_tps_mean': tm,
        'output_tps_std': ts,
        'wall_s_mean': wm,
        'ttft_p50_ms_mean': tfm,
        'tpot_p50_ms_mean': tpm,
        'accept_rate_mean': am,
        'gpu_util_mean': gm,
        'cpu_util_mpstat_mean': cum,
        'cpu_util_mpstat_std': cus,
    })
out = os.path.join(root, 'summary.json')
json.dump(res, open(out, 'w'), indent=2)
print(json.dumps(res, indent=2))
PYEOF
}
