#!/bin/bash
# Step 1 — 8-GPU concurrent run.
#  Scenario controlled by env STEP1_SCENARIO:
#    vanilla8   — 8x vanilla on GPU 0..7 (cluster baseline)
#    offload8   — 8x offload on GPU 0..7 (each kt with 28 threads, shared NUMA)
#    mix4v4o    — 4x vanilla GPU 0..3 + 4x offload GPU 4..7 (each kt with 56 threads)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lifecycle.sh"

SCENARIO=${STEP1_SCENARIO:-offload8}
N_PROMPTS=${N_PROMPTS:-100}
CONC=${CONC:-8}
MAX_TOK=${MAX_TOK:-256}
BASE_PORT=${BASE_PORT:-8011}
LOG_DIR="$HERE/logs/step1/${SCENARIO}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[step1] scenario=$SCENARIO log_dir=$LOG_DIR"

# Build per-instance profile/gpu/threads/pin.
declare -a PROFILES PORTS GPUS THREADS PINS

case "$SCENARIO" in
    vanilla8)
        for i in 0 1 2 3 4 5 6 7; do
            PROFILES+=("vanilla"); PORTS+=("$((BASE_PORT+i))")
            GPUS+=("$i"); THREADS+=("0"); PINS+=("")
        done
        ;;
    offload8)
        # 224 threads / 8 instances = 28 threads each (over-subscribe if 56)
        # NUMA-aware: GPU 0-3 -> NUMA 0 (CPU 0-55,112-167); GPU 4-7 -> NUMA 1 (56-111,168-223)
        # Each NUMA has 56 phys + 56 logic = 112 cores; 4 instances/NUMA -> 28 each.
        PINS_N0=("0-13,112-125" "14-27,126-139" "28-41,140-153" "42-55,154-167")
        PINS_N1=("56-69,168-181" "70-83,182-195" "84-97,196-209" "98-111,210-223")
        for i in 0 1 2 3; do
            PROFILES+=("offload"); PORTS+=("$((BASE_PORT+i))")
            GPUS+=("$i"); THREADS+=("28"); PINS+=("${PINS_N0[$i]}")
        done
        for i in 0 1 2 3; do
            GPU=$((i+4))
            PROFILES+=("offload"); PORTS+=("$((BASE_PORT+GPU))")
            GPUS+=("$GPU"); THREADS+=("28"); PINS+=("${PINS_N1[$i]}")
        done
        ;;
    mix4v4o)
        # 4 vanilla + 4 offload (each offload gets 56 threads — half NUMA each)
        PINS_OFF=("0-27,112-139" "28-55,140-167" "56-83,168-195" "84-111,196-223")
        for i in 0 1 2 3; do
            PROFILES+=("vanilla"); PORTS+=("$((BASE_PORT+i))")
            GPUS+=("$i"); THREADS+=("0"); PINS+=("")
        done
        for i in 0 1 2 3; do
            GPU=$((i+4))
            PROFILES+=("offload"); PORTS+=("$((BASE_PORT+GPU))")
            GPUS+=("$GPU"); THREADS+=("56"); PINS+=("${PINS_OFF[$i]}")
        done
        ;;
    offload2)
        PROFILES+=("offload"); PORTS+=("$((BASE_PORT+0))")
        GPUS+=("0"); THREADS+=("112"); PINS+=("0-55,112-167")
        PROFILES+=("offload"); PORTS+=("$((BASE_PORT+1))")
        GPUS+=("1"); THREADS+=("112"); PINS+=("56-111,168-223")
        ;;
    offload4)
        PINS_OFF=("0-27,112-139" "28-55,140-167" "56-83,168-195" "84-111,196-223")
        for i in 0 1 2 3; do
            PROFILES+=("offload"); PORTS+=("$((BASE_PORT+i))")
            GPUS+=("$i"); THREADS+=("56"); PINS+=("${PINS_OFF[$i]}")
        done
        ;;
    vanilla2)
        PROFILES+=("vanilla"); PORTS+=("$((BASE_PORT+0))"); GPUS+=("0"); THREADS+=("0"); PINS+=("")
        PROFILES+=("vanilla"); PORTS+=("$((BASE_PORT+1))"); GPUS+=("1"); THREADS+=("0"); PINS+=("")
        ;;
    vanilla4)
        for i in 0 1 2 3; do
            PROFILES+=("vanilla"); PORTS+=("$((BASE_PORT+i))")
            GPUS+=("$i"); THREADS+=("0"); PINS+=("")
        done
        ;;
    *)
        echo "Unknown SCENARIO=$SCENARIO"; exit 1;;
esac

N=${#PROFILES[@]}
echo "[step1] launching $N instances (sequential boot)"
for ((k=0; k<N; k++)); do
    P=${PROFILES[$k]}
    PORT=${PORTS[$k]}
    GPU=${GPUS[$k]}
    TH=${THREADS[$k]}
    PIN=${PINS[$k]}
    echo "[step1]  -> [$k] profile=$P port=$PORT gpu=$GPU thr=$TH pin=$PIN"
    bash "$HERE/start_server_step1.sh" "$P" "$PORT" "$GPU" "$TH" "$PIN"
done

# Wait for all of them to be ready.
echo "[step1] waiting for $N servers to become ready"
ALL_OK=1
for ((k=0; k<N; k++)); do
    PORT=${PORTS[$k]}
    GPU=${GPUS[$k]}
    if wait_ready "$PORT" 900; then
        echo "[step1] ready: port=$PORT gpu=$GPU"
    else
        echo "[step1] FAIL ready: port=$PORT gpu=$GPU"
        ALL_OK=0
    fi
done

if [ "$ALL_OK" != "1" ]; then
    echo "[step1] some servers failed; collecting partial pid info..."
fi

# Capture CPU stats once before run.
top -bn1 | head -5 > "$LOG_DIR/cpu_before.txt" 2>&1 || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$LOG_DIR/gpu_mem_before.txt" 2>&1 || true

# Run measurement.
PORT_LIST=$(IFS=' '; echo "${PORTS[*]}")
echo "[step1] running measurement against ports: $PORT_LIST"
LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib \
    /workspace/vllm_dev_prj/bin/python "$HERE/measure_concurrent.py" \
    --endpoints $PORT_LIST \
    --n "$N_PROMPTS" --max-tokens "$MAX_TOK" --conc "$CONC" \
    --out-dir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/measure.log"

# Capture stats after.
top -bn1 | head -5 > "$LOG_DIR/cpu_after.txt" 2>&1 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > "$LOG_DIR/gpu_after.txt" 2>&1 || true

# Tear down all servers.
echo "[step1] tearing down servers"
LOGS="$HERE/logs/step1"
for pf in "$LOGS"/${PROFILES[0]}_gpu*_p*.pid "$LOGS"/vanilla_gpu*_p*.pid "$LOGS"/offload_gpu*_p*.pid; do
    [ -e "$pf" ] || continue
    kill_pgroup "$pf"
done
for GPU in 0 1 2 3 4 5 6 7; do
    kill_gpu_orphans "$GPU"
done
for GPU in 0 1 2 3 4 5 6 7; do
    wait_gpu_free "$GPU" 30 || true
done
echo "[step1] done. results in $LOG_DIR"
