#!/usr/bin/env bash
# [try101] py-spy record 로 flamegraph + raw folded stacks 생성.
# 60s sampling, native, threads — 각 thread 의 시간 소비 함수 분포.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try101_v5_flamegraph"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
PYSPY=/workspace/vllm_dev_prj/bin/py-spy
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# try100 setup 그대로 (full_mirror=1)
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
export VLLM_NEO_OPTION_C_FULL_MIRROR=1
unset VLLM_NEO_OPTION_A VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN
unset VLLM_NEO_PROFILE
export OMP_NUM_THREADS=14
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "[try101] starting → ${LOG_FILE}"
taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.85 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts 500 \
    --target-input-len 8192 --max-tokens 8192 \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[try101] launcher PID=${LAUNCHER_PID}"

# Init 4 min — cdec 활성 영역 확보
sleep 240

# Primary worker PID
PRIMARY_WORKER=$(ps -ef | grep "VLLM::Worker_TP0" | grep -v grep | awk '{print $2}' | head -1)
echo "[try101] primary worker PID: $PRIMARY_WORKER"

# 동시 측정 — flamegraph + raw folded stacks (60s, native, threads)
"$PYSPY" record -p "$PRIMARY_WORKER" -d 60 -n -t -i \
    -o "${OUT_DIR}/flame.svg" -f flamegraph 2>&1 &
SPY_PID1=$!

"$PYSPY" record -p "$PRIMARY_WORKER" -d 60 -n -t -i \
    -o "${OUT_DIR}/raw.txt" -f raw 2>&1 &
SPY_PID2=$!

"$PYSPY" record -p "$PRIMARY_WORKER" -d 60 -n -t -i \
    -o "${OUT_DIR}/speedscope.json" -f speedscope 2>&1 &
SPY_PID3=$!

wait $SPY_PID1 $SPY_PID2 $SPY_PID3

# 추가 30s 후 종료
sleep 30
pgrep -f "run_neo_baseline\|VLLM::EngineCore\|VLLM::Worker" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 5
pgrep -f "VLLM::Worker" 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "[try101] DONE $(date -Iseconds)"

# Analysis — raw folded stacks 의 top 함수
echo ""
echo "===== raw stacks top 함수 (시간 점유 합계) ====="
if [ -f "${OUT_DIR}/raw.txt" ]; then
    # raw format: stack;function1;function2... <count>
    awk '
    {
        n = NF
        cnt = $n
        # 마지막 frame (top of stack = 시간 소비 함수)
        # frames = split($0 까지 SPACE-1)
        line = $0
        sub(/ [0-9]+$/, "", line)
        n_frames = split(line, frames, ";")
        if (n_frames > 0) {
            top_func = frames[n_frames]
            top[top_func] += cnt
            total += cnt
        }
    }
    END {
        # sort by count desc
        n = 0
        for (f in top) { arr[n++] = f }
        # simple sort
        for (i = 0; i < n; i++) {
            for (j = i+1; j < n; j++) {
                if (top[arr[i]] < top[arr[j]]) {
                    tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp
                }
            }
        }
        printf "%-7s %-60s %s\n", "count", "function (top of stack)", "% of total"
        for (i = 0; i < 25 && i < n; i++) {
            printf "%-7d %-60s %.1f%%\n", top[arr[i]], substr(arr[i], 1, 60), top[arr[i]]/total*100
        }
        printf "\n총 samples: %d\n", total
    }' "${OUT_DIR}/raw.txt"
fi

echo ""
echo "===== file sizes ====="
ls -la "${OUT_DIR}"/*.{svg,txt,json} 2>/dev/null
echo ""
echo "===== throughput ====="
grep -oE 'Avg generation throughput:[^,]+' "${LOG_FILE}.stdout" 2>/dev/null | tail -3
