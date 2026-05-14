#!/usr/bin/env bash
# [Worker Thread Idle 분석] 14 OMP thread cumulative CPU time + duty cycle.
# v1.5.1 env (chain firing 95%) + 5분 launch + 60s window 측정.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
TAG="omp_idle_analysis"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === v1.5.1 env (chain firing 95%) ===
export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_LOAD_AWARE_MIN_RUNNING=32
export VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP=2
export VLLM_NEO_FORCE_SWAP_IN=1
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest
export VLLM_NEO_MIRROR_MIN_BUFFER=8
export VLLM_NEO_OPTION_K=1
export VLLM_NEO_OPTION_C=1
export VLLM_NEO_OPTION_L=1
export VLLM_NEO_OPTION_M2=1
export VLLM_NEO_OPTION_C_FULL_MIRROR=1
unset VLLM_NEO_OPTION_O2 VLLM_NEO_OPTION_A VLLM_NEO_DISABLE_CHAIN
unset VLLM_NEO_DISABLE_FORCE_PIPELINED VLLM_NEO_DISABLE_FUSED_RMSNORM
unset VLLM_NEO_DISABLE_SWAP_IN VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN VLLM_NEO_SWAP_COOLDOWN
unset VLLM_NEO_PROFILE VLLM_DEBUG_FAULTHANDLER

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[omp_idle] $(TZ=Asia/Seoul date -Iseconds) starting → ${OUT_DIR}"

taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[omp_idle] launcher PID=${LAUNCHER_PID}"

# Engine init 4분
sleep 240

# === 1. /proc/<worker>/task/<tid>/stat snapshot t0 (60s diff용) ===
PRIMARY_WORKER=$(pgrep -f "VLLM::Worker_TP0" 2>/dev/null | head -1)
echo "[omp_idle] primary worker: $PRIMARY_WORKER"
snapshot_threads() {
    local pid="$1"; local label="$2"
    local out="${OUT_DIR}/proc_threads_${label}.txt"
    echo "=== snapshot $(TZ=Asia/Seoul date -Iseconds) pid=$pid ===" > "$out"
    for tid_dir in /proc/$pid/task/[0-9]*; do
        tid=$(basename "$tid_dir")
        stat_line=$(cat "$tid_dir/stat" 2>/dev/null)
        comm=$(cat "$tid_dir/comm" 2>/dev/null)
        if [ -n "$stat_line" ]; then
            echo "tid=$tid comm=$comm stat=$stat_line" >> "$out"
        fi
    done
}
snapshot_threads "$PRIMARY_WORKER" "t0"

# /proc/stat per-CPU t0
cat /proc/stat | head -120 > "${OUT_DIR}/proc_stat_t0.txt"

# === 2. top -H -b -d 1 -n 60 — 60s sample (각 thread CPU%) ===
top -H -b -d 1 -n 60 -p "$PRIMARY_WORKER" > "${OUT_DIR}/top_H_worker.txt" 2>&1 &
TOP_PID=$!

# 60s measurement window
sleep 60

snapshot_threads "$PRIMARY_WORKER" "t1"
cat /proc/stat | head -120 > "${OUT_DIR}/proc_stat_t1.txt"
kill $TOP_PID 2>/dev/null
wait 2>/dev/null

# Cleanup
pgrep -f "run_neo_baseline\|VLLM::EngineCore\|VLLM::Worker" 2>/dev/null \
    | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::Worker\|VLLM::EngineCore" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[omp_idle] $(TZ=Asia/Seoul date -Iseconds) DONE"

# === Analysis ===
echo ""
echo "===== throughput ====="
grep -oE 'Avg generation throughput: *[0-9.]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE "[0-9]+\.[0-9]+" | tail -5
echo ""
echo "===== /proc thread CPU time diff (t1 - t0) — 60s window ====="
export OUT_DIR
python3 - <<'PYEOF'
import re, os
od = os.environ.get("OUT_DIR", ".")
def parse(file):
    out = {}
    for ln in open(f"{od}/{file}").read().splitlines():
        m = re.match(r'tid=(\d+) comm=(\S+) stat=(.+)', ln)
        if not m: continue
        tid = int(m.group(1))
        comm = m.group(2)
        parts = m.group(3).split()
        try:
            utime = int(parts[13])
            stime = int(parts[14])
            state = parts[2]
            out[tid] = (utime, stime, state, comm)
        except Exception:
            pass
    return out
t0 = parse("proc_threads_t0.txt")
t1 = parse("proc_threads_t1.txt")
results = []
for tid in t1:
    u1, s1, st1, comm = t1[tid]
    if tid in t0:
        u0, s0, _, _ = t0[tid]
        d_u = (u1 - u0) / 100.0
        d_s = (s1 - s0) / 100.0
    else:
        d_u = u1 / 100.0
        d_s = s1 / 100.0
    total = d_u + d_s
    cpu_pct = total / 60 * 100  # 60s window
    results.append((cpu_pct, tid, comm, d_u, d_s, st1))
results.sort(reverse=True)
print(f"{'rank':>4} {'tid':>10} {'comm':>20} {'state':>5} {'CPU%':>8} {'utime_s':>10} {'stime_s':>10}")
for i, (cpu, tid, comm, du, ds, st) in enumerate(results[:30]):
    if cpu > 0.5 or st in ('R', 'D'):
        print(f"{i:>4} {tid:>10} {comm:>20} {st:>5} {cpu:>8.1f}% {du:>10.2f} {ds:>10.2f}")
print(f"... 총 {len(results)} threads — 활동 thread (CPU>0.5%): {len([r for r in results if r[0]>0.5])}")
PYEOF
echo ""
echo "===== /proc/stat per-CPU usage (60s diff, top 30) ====="
python3 - <<'PYEOF'
import os
od = os.environ.get("OUT_DIR", ".")
def parse(f):
    out = {}
    for ln in open(f"{od}/{f}").read().splitlines():
        if ln.startswith("cpu") and not ln.startswith("cpu "):
            parts = ln.split()
            cpu = parts[0]
            user, nice, system, idle, iowait, irq, softirq, steal = [int(x) for x in parts[1:9]]
            busy = user + nice + system + irq + softirq
            total = busy + idle + iowait + steal
            out[cpu] = (busy, total)
    return out
t0 = parse("proc_stat_t0.txt")
t1 = parse("proc_stat_t1.txt")
res = []
for cpu in t1:
    if cpu in t0:
        db = t1[cpu][0] - t0[cpu][0]
        dt = t1[cpu][1] - t0[cpu][1]
        if dt > 0:
            usage = db / dt * 100
            res.append((usage, cpu))
res.sort(reverse=True)
print(f"{'rank':>4} {'cpu':>6} {'usage_%':>10}")
for i, (u, c) in enumerate(res[:30]):
    print(f"{i:>4} {c:>6} {u:>10.1f}")
busy_cores = len([r for r in res if r[0]>10.0])
print(f"... 총 {len(res)} cpus / busy (>10%) = {busy_cores} cores")
PYEOF
echo ""
echo "===== top -H summary (60s sample, 가장 hot thread top 20) ====="
awk '/^[0-9]+ root/ {if($9+0 > 1) {comm=$NF; if(!(comm in seen)) {seen[comm]=1; print $1, $9"%", $12}}}' "${OUT_DIR}/top_H_worker.txt" 2>/dev/null | sort -u | head -20

echo "[omp_idle] analysis done"
