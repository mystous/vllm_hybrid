#!/usr/bin/env bash
# [Plan v5 — profile] try88 회차.
# Goal: 동적 분석 — swap-out/in elapsed time + cycle 추적 + py-spy continuous.
# 본 회차 = 8분만. fact 수집 우선.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try88_v5_profile"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
PYSPY_DIR="${OUT_DIR}/pyspy_dumps"
mkdir -p "${PYSPY_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
PYSPY=/workspace/vllm_dev_prj/bin/py-spy
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# v1.5 적재 (Option O v2 코드) + v3 env
export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_LOAD_AWARE_MIN_RUNNING=32
export VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP=2
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=1
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest
export VLLM_NEO_FORCE_SWAP_IN=0
export VLLM_NEO_SWAP_COOLDOWN=50
export VLLM_NEO_MIRROR_MIN_BUFFER=4
export VLLM_NEO_OPTION_K=1
export VLLM_NEO_OPTION_C=1
export VLLM_NEO_OPTION_L=1
export VLLM_NEO_OPTION_M2=1
unset VLLM_NEO_OPTION_A
unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN

# 동적 분석 mode 활성
export VLLM_NEO_PROFILE=1

echo "[v5-profile] starting (instrumentation + py-spy) → ${LOG_FILE}"
"$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --max-num-seqs 256 \
    --num-prompts 500 \
    --target-input-len 8192 \
    --max-tokens 8192 \
    --enable-neo-asymmetric \
    --async-scheduling \
    --enforce-eager false \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" \
    --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[v5-profile] launcher PID=${LAUNCHER_PID}"

# Wait for engine init (5분 동안 worker proc 안정화 후 py-spy attach)
sleep 240

# Worker PIDs 추출 → py-spy continuous
WORKER_PIDS=$(ps -ef | grep "VLLM::Worker" | grep -v grep | awk '{print $2}' | head -2)
echo "[v5-profile] worker PIDs for py-spy: ${WORKER_PIDS}"

# 3 dumps × 60s interval
for i in 1 2 3; do
    sleep 60
    for pid in $WORKER_PIDS; do
        DUMP_FILE="${PYSPY_DIR}/dump_w${pid}_$(printf '%02d' "$i")_$(date +%H%M%S).txt"
        "$PYSPY" dump --pid "$pid" > "$DUMP_FILE" 2>&1 || echo "  py-spy dump failed for $pid"
    done
    echo "[v5-profile] dump $i done"
done

# 추가 1분 (총 ~8분) 후 mid-stop
sleep 60
kill -TERM $LAUNCHER_PID 2>/dev/null
sleep 5
kill -9 $LAUNCHER_PID 2>/dev/null
ps -ef | grep "EngineCore\|VLLM::Worker" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null

echo "[v5-profile] DONE $(date -Iseconds)"

# Quick analysis
echo ""
echo "===== PROFILE 분석 ====="
echo "--- SWAP_OUT elapsed (last 5) ---"
grep '\[PROFILE SWAP_OUT\]' "${LOG_FILE}.stdout" 2>/dev/null | tail -5
echo "--- SWAP_OUT elapsed_ms 통계 ---"
grep -oE 'PROFILE SWAP_OUT.*elapsed_ms=[0-9.]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE 'elapsed_ms=[0-9.]+' | cut -d= -f2 \
    | awk '{sum+=$1; sumsq+=$1*$1; cnt++; if($1>max)max=$1; if(min==""||$1<min)min=$1} END {if(cnt>0) printf "  count=%d avg=%.2f min=%.2f max=%.2f\n", cnt, sum/cnt, min, max}'
echo "--- SWAP_IN elapsed_ms 통계 ---"
grep -oE 'PROFILE SWAP_IN.*elapsed_ms=[0-9.]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | grep -oE 'elapsed_ms=[0-9.]+' | cut -d= -f2 \
    | awk '{sum+=$1; sumsq+=$1*$1; cnt++; if($1>max)max=$1; if(min==""||$1<min)min=$1} END {if(cnt>0) printf "  count=%d avg=%.2f min=%.2f max=%.2f\n", cnt, sum/cnt, min, max}'
echo "--- cycle_age_ms 통계 (swap-out → swap-in 사이 시간) ---"
grep -oE 'cycle_age_ms=[0-9.-]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | cut -d= -f2 \
    | awk '$1>=0 {sum+=$1; cnt++; if($1>max)max=$1; if(min==""||$1<min)min=$1} END {if(cnt>0) printf "  count=%d avg=%.1f min=%.1f max=%.1f\n", cnt, sum/cnt, min, max}'
echo "--- so_count / si_count 분포 (cycle 횟수 per req) ---"
grep -oE 'so_count=[0-9]+' "${LOG_FILE}.stdout" 2>/dev/null | sort | uniq -c | sort -rn | head -5
echo "--- throughput ---"
grep -oE 'Avg generation throughput:[^,]+' "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo "--- bytes_MiB 통계 (per swap) ---"
grep -oE 'bytes_MiB=[0-9.]+' "${LOG_FILE}.stdout" 2>/dev/null \
    | cut -d= -f2 \
    | awk '{sum+=$1; cnt++} END {if(cnt>0) printf "  count=%d avg=%.1f MiB/swap total=%.1f GiB\n", cnt, sum/cnt, sum/1024}'
echo "[v5-profile] analysis done"
