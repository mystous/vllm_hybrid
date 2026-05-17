#!/usr/bin/env bash
# 500p × 8192 NEO S1-S9 run + 112-core CPU utilization deep dive.
#
# 동시 수집:
#   - /proc sampler (per-core util + per-thread stat + wchan + NUMA placement + affinity)
#   - py-spy native record (8 worker × 60s mid-run)
#   - perf record (system-wide 60s mid-run, dwarf call-graph)
#   - perf stat (60s hardware counter — cycles/IPC/LLC miss)
#   - nvidia-smi periodic GPU state
#
# 산출: $OUTDIR/{env, timeseries, deep_dive_60s, analysis}
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

OUTDIR="${1:-}"
if [ -z "${OUTDIR}" ]; then
    TS="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
    OUTDIR="${ROOT_DIR}/eval/results/${TS}_cpu112_analysis_500p"
fi
mkdir -p "${OUTDIR}/env" "${OUTDIR}/timeseries" "${OUTDIR}/deep_dive_60s" "${OUTDIR}/analysis"

# === DURATION (full run ≈ 30 min, dry-run override 시 짧게) ===
RUN_DURATION="${RUN_DURATION:-2400}"      # sampler 최대 2400 s (vllm 끝나면 stop)
DEEP_DIVE_DELAY="${DEEP_DIVE_DELAY:-600}" # 10 min 후 perf/py-spy 60 s sample
NUM_PROMPTS="${NUM_PROMPTS:-500}"
TARGET_INPUT_LEN="${TARGET_INPUT_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-8192}"

LOG_FILE="${OUTDIR}/engine.log"

ulimit -c unlimited
PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# === NEO env (run_neo_standard.sh 와 동일, S1-S9 정합) ===
for v in $(env | grep -oE "^VLLM_NEO_[A-Z_0-9]+"); do unset "$v"; done
unset VLLM_DEBUG_FAULTHANDLER VLLM_DEBUG_CDEC_PATH ENABLE_NEO_INV

export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_CPU_RESIDENT_REQS=128
export VLLM_NEO_ASYNC_SWAP_BUFFERS=3
export VLLM_NEO_PROFILE=1
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
unset OMP_PLACES
export KMP_BLOCKTIME=50
export KMP_AFFINITY=verbose,scatter
export VLLM_NEO_CPU_PIN_PER_WORKER=1
export VLLM_NEO_CPU_PIN_CORES=12
export VLLM_NEO_NUMA_BIND=1

echo "[cpu112] $(TZ=Asia/Seoul date -Iseconds) start → ${OUTDIR}"
echo "[cpu112] config: 500p × 8192, RUN_DURATION=${RUN_DURATION} DEEP_DIVE_DELAY=${DEEP_DIVE_DELAY}"
env | grep -E "^VLLM_NEO_|^KMP|^OMP" | sort | tee "${OUTDIR}/env/runtime_env.txt"

# === vLLM launch (background) ===
taskset -c 0-111 "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b --tensor-parallel-size 8 --gpu-memory-utilization 0.92 \
    --max-model-len 16384 --max-num-seqs 256 --num-prompts "${NUM_PROMPTS}" \
    --target-input-len "${TARGET_INPUT_LEN}" --max-tokens "${MAX_TOKENS}" \
    --enable-neo-asymmetric --async-scheduling --enforce-eager false \
    --kv-cache-dtype fp8 --max-num-batched-tokens 8192 \
    --log-file "${LOG_FILE}" --output-file "${OUTDIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "[cpu112] launcher PID=${LAUNCHER_PID}"

# === warmup wait — vllm 가 워커 spawn + 메모리 alloc 끝낼 시간 ===
# Note: 첫 측정 (2026-05-17) 에서 60 s 부족 (워커가 82 s 후 spawn) → 120 s 로 변경.
# 또한 pgrep loop 으로 워커가 모두 보일 때까지 polling (max 180 s).
echo "[cpu112] waiting up to 180s for vLLM warmup + worker spawn..."
WARMUP_MAX=180
WARMUP_START=$(date +%s)
while [ $(($(date +%s) - WARMUP_START)) -lt ${WARMUP_MAX} ]; do
    WORKER_COUNT=$(pgrep -f "VLLM::Worker_TP[0-7]" 2>/dev/null | sort -u | wc -l)
    ENGINE_COUNT=$(pgrep -f "VLLM::EngineCor" 2>/dev/null | wc -l)
    if [ "$WORKER_COUNT" -eq 8 ] && [ "$ENGINE_COUNT" -ge 1 ]; then
        echo "[cpu112] warmup done: ${WORKER_COUNT} workers + ${ENGINE_COUNT} engine after $(($(date +%s) - WARMUP_START))s"
        break
    fi
    sleep 5
done

# === target PID 수집 (EngineCore + 8 TP worker) ===
ENGINE_PID=$(pgrep -f "VLLM::EngineCor" | head -1)
WORKER_PIDS=$(pgrep -f "VLLM::Worker_TP[0-7]" | sort -u | head -8)
ALL_TARGETS=$(echo "${ENGINE_PID} ${WORKER_PIDS}" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ',' | sed 's/,$//')
echo "[cpu112] EngineCore=${ENGINE_PID}, Workers=${WORKER_PIDS}"
echo "[cpu112] sampler targets: ${ALL_TARGETS}"
echo "${ALL_TARGETS}" > "${OUTDIR}/env/target_pids.txt"

# === Sampler 시작 (background, full duration) ===
"$PY" "${SCRIPT_DIR}/cpu112_sampler.py" "${OUTDIR}/timeseries" "${RUN_DURATION}" "${ALL_TARGETS}" \
    > "${OUTDIR}/timeseries/sampler.log" 2>&1 &
SAMPLER_PID=$!
echo "[cpu112] sampler PID=${SAMPLER_PID}"

# === nvidia-smi periodic dump (background, 5 s 간격) ===
( while kill -0 ${LAUNCHER_PID} 2>/dev/null; do
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw \
        --format=csv,noheader 2>/dev/null
    echo "---$(date +%s)"
    sleep 5
done ) > "${OUTDIR}/timeseries/nvidia_smi.log" 2>&1 &
NVSMI_PID=$!

# === Deep-dive: DEEP_DIVE_DELAY 후 60s perf + py-spy ===
( sleep "${DEEP_DIVE_DELAY}"
  if kill -0 ${LAUNCHER_PID} 2>/dev/null; then
      echo "[cpu112] deep-dive start at $(TZ=Asia/Seoul date -Iseconds)" > "${OUTDIR}/deep_dive_60s/start.log"

      # py-spy on 8 TP worker (병행)
      for wpid in ${WORKER_PIDS}; do
          tp_idx=$(ps -p ${wpid} -o cmd= | grep -oE "TP[0-7]" | head -1)
          [ -z "${tp_idx}" ] && tp_idx="UNK_${wpid}"
          py-spy record -p ${wpid} -d 60 -n -r 250 \
              -o "${OUTDIR}/deep_dive_60s/pyspy_${tp_idx}.svg" \
              > "${OUTDIR}/deep_dive_60s/pyspy_${tp_idx}.log" 2>&1 &
      done

      # perf system-wide record (60 s, dwarf call-graph)
      perf record -a -F 99 --call-graph dwarf -g \
          -o "${OUTDIR}/deep_dive_60s/perf.data" -- sleep 60 \
          > "${OUTDIR}/deep_dive_60s/perf_record.log" 2>&1 &

      # perf stat — hardware counter (60 s)
      perf stat -a -e cycles,instructions,cache-misses,cache-references,LLC-loads,LLC-load-misses,branch-misses,context-switches,cpu-migrations,page-faults \
          -- sleep 60 \
          > "${OUTDIR}/deep_dive_60s/perf_stat.txt" 2>&1

      # numa accounting snapshot at deep-dive time
      numastat > "${OUTDIR}/deep_dive_60s/numastat_global.txt" 2>&1
      for p in ${ENGINE_PID} ${WORKER_PIDS}; do
          numastat -p ${p} > "${OUTDIR}/deep_dive_60s/numastat_pid${p}.txt" 2>&1
          cat /proc/${p}/sched > "${OUTDIR}/deep_dive_60s/sched_pid${p}.txt" 2>&1
      done

      wait
      echo "[cpu112] deep-dive done at $(TZ=Asia/Seoul date -Iseconds)" >> "${OUTDIR}/deep_dive_60s/start.log"
  else
      echo "[cpu112] launcher dead before deep-dive could start" > "${OUTDIR}/deep_dive_60s/SKIPPED.log"
  fi
) &
DEEPDIVE_PID=$!

# === Wait for launcher (vllm run 끝) ===
START_TS=$(date +%s)
MAX_WAIT=5400
while kill -0 ${LAUNCHER_PID} 2>/dev/null; do
    sleep 15
    NOW_TS=$(date +%s)
    ELAPSED=$((NOW_TS - START_TS))
    if [ $((ELAPSED % 300)) -lt 16 ]; then
        echo "[cpu112] $(TZ=Asia/Seoul date -Iseconds) elapsed=${ELAPSED}s, launcher alive"
    fi
    if [ ${ELAPSED} -gt ${MAX_WAIT} ]; then
        echo "[cpu112] timeout — killing"
        kill -9 ${LAUNCHER_PID} 2>/dev/null
        break
    fi
done

# === Cleanup ===
echo "[cpu112] launcher exited after $(($(date +%s) - START_TS))s, cleaning up"
kill ${SAMPLER_PID} 2>/dev/null
kill ${NVSMI_PID} 2>/dev/null
kill ${DEEPDIVE_PID} 2>/dev/null
sleep 3
pgrep -f "VLLM::Worker\|VLLM::EngineCore" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 3

echo "[cpu112] $(TZ=Asia/Seoul date -Iseconds) DONE"
echo ""
echo "===== result =====" | tee "${OUTDIR}/RESULT_SUMMARY.txt"
cat "${OUTDIR}/result.json" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"output_tps={d.get('output_tps')} wall_s={d.get('generate_wall_s')}\")" 2>/dev/null | tee -a "${OUTDIR}/RESULT_SUMMARY.txt"
echo ""
echo "===== sampler files =====" | tee -a "${OUTDIR}/RESULT_SUMMARY.txt"
ls -la "${OUTDIR}/timeseries/" | tee -a "${OUTDIR}/RESULT_SUMMARY.txt"
echo ""
echo "===== deep_dive files =====" | tee -a "${OUTDIR}/RESULT_SUMMARY.txt"
ls -la "${OUTDIR}/deep_dive_60s/" | tee -a "${OUTDIR}/RESULT_SUMMARY.txt"
