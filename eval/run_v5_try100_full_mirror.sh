#!/usr/bin/env bash
# [try98] 다중 동적 분석 도구 동시 적용.
# 1. pacpu kernel 의 [OMP CPU] log — 각 tid 의 진짜 CPU time
# 2. pidstat -t — thread 별 CPU usage + context switch
# 3. mpstat -P ALL — per-CPU usage (어느 core 가 busy/idle)
# 4. perf stat (engine init 후) — IPC, cache miss, cycles
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try100_v5_full_mirror"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_LOAD_AWARE_MIN_RUNNING=32
export VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP=2
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest
export VLLM_NEO_FORCE_SWAP_IN=0
export VLLM_NEO_SWAP_COOLDOWN=20
export VLLM_NEO_MIRROR_MIN_BUFFER=4
export VLLM_NEO_OPTION_K=1
export VLLM_NEO_OPTION_C=1
export VLLM_NEO_OPTION_L=1
export VLLM_NEO_OPTION_M2=1
unset VLLM_NEO_OPTION_A VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN
unset VLLM_NEO_PROFILE

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[try98] starting → ${LOG_FILE}"
taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[try98] launcher PID=${LAUNCHER_PID}"

# Init 4 min
sleep 240

# Worker PIDs
WORKER_PIDS=$(ps -ef | grep "VLLM::Worker" | grep -v grep | awk '{print $2}')
PRIMARY_WORKER=$(echo "$WORKER_PIDS" | head -1)
echo "[try98] worker PIDs: $WORKER_PIDS"
echo "[try98] primary worker: $PRIMARY_WORKER"

# pidstat/mpstat/perf 미설치 — /proc 직접 측정.

# 1. /proc/<pid>/task/<tid>/stat 2회 snapshot (60s 차이) → thread 별 utime/stime diff
snapshot_threads() {
    local pid="$1"; local label="$2"
    local out="${OUT_DIR}/proc_threads_${label}.txt"
    echo "=== snapshot at $(date +%s) ===" > "$out"
    for tid_dir in /proc/$pid/task/[0-9]*; do
        tid=$(basename "$tid_dir")
        stat_line=$(cat "$tid_dir/stat" 2>/dev/null)
        if [ -n "$stat_line" ]; then
            # field: pid comm state ppid ... utime stime ...
            echo "tid=$tid stat=$stat_line" >> "$out"
        fi
    done
}
snapshot_threads "$PRIMARY_WORKER" "t0"

# 2. /proc/stat snapshot (per-CPU usage)
cat /proc/stat | head -120 > "${OUT_DIR}/proc_stat_t0.txt"

# 60s 측정 영역
sleep 60

snapshot_threads "$PRIMARY_WORKER" "t1"
cat /proc/stat | head -120 > "${OUT_DIR}/proc_stat_t1.txt"

# top -H -b 60s sample
top -H -b -d 1 -n 60 -p "$PRIMARY_WORKER" > "${OUT_DIR}/top_H.txt" 2>&1 &
TOP_PID=$!
sleep 65
kill $TOP_PID 2>/dev/null
wait 2>/dev/null

# 1 min 추가 진행 후 종료
sleep 30

pgrep -f "run_neo_baseline\|VLLM::EngineCore\|VLLM::Worker" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::Worker" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[try98] DONE $(date -Iseconds)"

# Analysis
echo ""
echo "===== [OMP CPU] log (각 tid 별 진짜 CPU time, last 30) ====="
grep '\[OMP CPU\]' "${LOG_FILE}.stdout" 2>/dev/null | tail -30
echo ""
echo "===== [OMP CPU] tid 별 unique log count ====="
grep '\[OMP CPU\]' "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE 'tid=[0-9]+' | sort | uniq -c | sort -k2.5n
echo ""
echo "===== /proc/<tid>/stat diff (utime + stime per thread, 60s) ====="
python3 - <<'PYEOF'
import re, os
od = os.environ.get("OUT_DIR", ".")
def parse(file):
    out = {}
    for ln in open(f"{od}/{file}").read().splitlines():
        m = re.match(r'tid=(\d+) stat=(.+)', ln)
        if not m: continue
        tid = int(m.group(1))
        parts = m.group(2).split()
        # stat format: pid comm state ppid pgrp session tty_nr ... [13]utime [14]stime
        try:
            utime = int(parts[13])
            stime = int(parts[14])
            state = parts[2]
            out[tid] = (utime, stime, state)
        except Exception as e:
            pass
    return out

t0 = parse("proc_threads_t0.txt")
t1 = parse("proc_threads_t1.txt")
clk_tck = 100  # default
results = []
for tid in t1:
    u1, s1, st1 = t1[tid]
    if tid in t0:
        u0, s0, _ = t0[tid]
        d_u = (u1 - u0) / clk_tck
        d_s = (s1 - s0) / clk_tck
    else:
        d_u = u1 / clk_tck
        d_s = s1 / clk_tck
    total = d_u + d_s
    results.append((total, tid, d_u, d_s, st1))
results.sort(reverse=True)
print(f"{'rank':>4} {'tid':>10} {'state':>5} {'CPU_sec_total':>14} {'utime_s':>10} {'stime_s':>10}")
for i, (tot, tid, du, ds, st) in enumerate(results[:20]):
    if tot > 0 or st == 'R':
        print(f"{i:>4} {tid:>10} {st:>5} {tot:>14.2f} {du:>10.2f} {ds:>10.2f}")
print(f"... 총 {len(results)} threads (>0 CPU 영역)")
PYEOF
echo ""
echo "===== /proc/stat per-CPU (60s diff, top 30) ====="
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
print(f"  ... 총 {len([r for r in res if r[0]>1.0])} cpus > 1% busy")
PYEOF
echo ""
echo "===== throughput ====="
grep -oE 'Avg generation throughput:[^,]+' "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo "[try98] analysis done"
