#!/usr/bin/env bash
# Measure a single LHC config end-to-end: boot vllm serve, run 1 warmup
# + 3 sweep benches against sonnet, then kill. Used by run_full_sweep.sh
# for vanilla / lhc_dsa / lhc_dsa_fp8kv.
set -u

CFG=${1:?need cfg}
ROOT=/workspace/host_vllm_hybrid/lhc_phase2
RUNS=$ROOT/runs_sweep
mkdir -p "$RUNS"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8001
TP=8
MAX_MODEL_LEN=16384
SONNET=/workspace/host_vllm_hybrid/benchmarks/sonnet.txt
HARNESS=/workspace/host_vllm_hybrid/vllm_config_perf/gating/benchmark_workloads.py
VPY=/workspace/vllm_dev_prj/bin/python
VLLM=/workspace/vllm_dev_prj/bin/vllm

NPROMPTS=200
MAX_TOKENS=512
TARGET_IN=1024
CONC=64

EXTRA_ENV=()
EXTRA_CLI=()
case "$CFG" in
    vanilla) ;;
    lhc_dsa)
        EXTRA_ENV=(VLLM_LHC_DSA=1 VLLM_LEVER_N9=1 VLLM_LHC_DSA_MIN=65536)
        ;;
    lhc_dsa_fp8kv)
        EXTRA_ENV=(VLLM_LHC_DSA=1 VLLM_LEVER_N9=1 VLLM_LHC_DSA_MIN=65536)
        EXTRA_CLI=(--kv-cache-dtype fp8)
        ;;
    *) echo "unknown cfg=$CFG" >&2; exit 2 ;;
esac

BOOT_LOG=$RUNS/${CFG}_boot.log
PID_FILE=$RUNS/${CFG}.pid

echo "[boot] cfg=$CFG at $(date -u +%FT%TZ)" | tee "$BOOT_LOG"
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${EXTRA_ENV[@]}" \
    setsid "$VLLM" serve "$MODEL" \
    --tensor-parallel-size "$TP" --port "$PORT" \
    --gpu-memory-utilization 0.85 --max-model-len "$MAX_MODEL_LEN" \
    "${EXTRA_CLI[@]}" >> "$BOOT_LOG" 2>&1 &
SRV=$!
echo "$SRV" > "$PID_FILE"
echo "[boot] pid=$SRV"

# wait ready
for i in $(seq 1 270); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "[ready] cfg=$CFG after ${i}*2s"; break
    fi
    sleep 2
done
if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "[boot] FAIL"; tail -50 "$BOOT_LOG" >&2; exit 1
fi

# warmup
echo "[warmup] cfg=$CFG"
"$VPY" "$HARNESS" --scenario vanilla --workload sonnet \
    --num-prompts 50 --target-input-len "$TARGET_IN" \
    --max-tokens "$MAX_TOKENS" --concurrency "$CONC" \
    --model "$MODEL" --sonnet "$SONNET" --seed 1 \
    --out "$RUNS/${CFG}_sonnet_warmup.json" 2>&1 | tail -5

# 3 sweep
for sw in 1 2 3; do
    echo "[bench] cfg=$CFG sweep=$sw"
    "$VPY" "$HARNESS" --scenario vanilla --workload sonnet \
        --num-prompts "$NPROMPTS" --target-input-len "$TARGET_IN" \
        --max-tokens "$MAX_TOKENS" --concurrency "$CONC" \
        --model "$MODEL" --sonnet "$SONNET" --seed $((42 + sw)) \
        --out "$RUNS/${CFG}_sonnet_s${sw}.json" 2>&1 | tail -5
done

# kill
echo "[kill] cfg=$CFG pid=$SRV"
kill -TERM -- -$SRV 2>/dev/null || true
for i in $(seq 1 30); do
    [ -d "/proc/$SRV" ] || break
    sleep 1
done
[ -d "/proc/$SRV" ] && kill -KILL -- -$SRV 2>/dev/null
for op in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
    [ -d "/proc/$op" ] && kill -KILL "$op" 2>/dev/null
done
sleep 5
# wait gpu free
for i in $(seq 1 30); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500 {c++} END{print c+0}')
    [ "$busy" -eq 0 ] && { echo "[gpu] free"; break; }
    sleep 2
done
rm -f "$PID_FILE"
echo "[done] cfg=$CFG at $(date -u +%FT%TZ)"
