#!/usr/bin/env bash
# LHC Phase 4 misuse anti-pattern sweep.
#
# 5 anti-patterns × {baseline, misuse} × {sonnet, sharegpt-equiv chat} × 3 sweeps.
# Total cells: 5 × 2 × 2 × 3 = 60.
#
# Anti-patterns:
#   ap1 — DSA_MIN=64 (too-small transfers via DSA): vs DSA_MIN=65536
#   ap2 — AMX C3 FORCE_EVERY_STEP: vs prefix-hit-only (default off)
#   ap3 — DSA FORCE_REMOTE_NUMA (cross-socket): vs NUMA-local
#   ap4 — WQ_PER_RANK=0 (PASID contention): vs WQ_PER_RANK=1
#   ap5 — REGIME_ADAPTIVE=0 (always-on, Option A): vs REGIME_ADAPTIVE=1
#
# Output: lhc_phase4/misuse/runs/<ap>_<config>_<wl>_s<n>_bench.{json,log}
set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4/misuse
RUNS=${BASE}/runs
mkdir -p "${RUNS}"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8500
TP=8
GPU_MEM=0.92
MAX_LEN=16384
DATA="/workspace/host_vllm_hybrid/benchmarks/sonnet.txt"

# baseline workload — sharegpt 500p conc=64 max-tok=2048
# We use sonnet harness (same as optionC) with input 512 + output 2048 to match
# the chat-style hot baseline. Plus a sonnet workload to broaden the surface.
WORKLOADS="${WORKLOADS:-chat sonnet}"

APS="${APS:-ap1 ap2 ap3 ap4 ap5}"
SWEEPS="${SWEEPS:-3}"

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

cleanup_orphans() {
    pgrep -f "vllm serve" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | xargs -r kill -9 2>/dev/null || true
    sleep 5
}
cleanup_orphans

# Wait for VLLM ready, return 0 ok / 1 timeout.
wait_ready() {
    local p=$1
    for i in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:${p}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    return 1
}

# Workload params:
#   chat:  input=256 output=512 prompts=500 conc=64 (hot baseline)
#   sonnet: input=512 output=512 prompts=500 conc=64
get_workload() {
    case "$1" in
        chat)   echo "256 512 0 500 64" ;;
        sonnet) echo "512 512 0 500 64" ;;
        *) echo "256 512 0 500 64" ;;
    esac
}

# Returns "ENV_PRE|FLAGS" for the given ap+config.
# config: baseline (proper LHC use) | misuse (anti-pattern)
build_env() {
    local ap=$1
    local cfg=$2
    case "$ap" in
        ap1)
            # DSA_MIN: misuse=64 (per-call overhead dominates), baseline=65536.
            # All other params equal: WQ-per-rank ON, adaptive ON (Option C).
            if [[ "$cfg" == "misuse" ]]; then
                echo "VLLM_LHC_REGIME_ADAPTIVE=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=64 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_INTERVAL=20"
            else
                echo "VLLM_LHC_REGIME_ADAPTIVE=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=65536 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_INTERVAL=20"
            fi
            ;;
        ap2)
            # AMX C3 force every step vs prefix-hit-only (regime gate).
            if [[ "$cfg" == "misuse" ]]; then
                echo "VLLM_LHC_REGIME_ADAPTIVE=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1 VLLM_LHC_AMX_C3_FORCE_EVERY_STEP=1 VLLM_LHC_REGIME_INTERVAL=20"
            else
                echo "VLLM_LHC_REGIME_ADAPTIVE=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_INTERVAL=20"
            fi
            ;;
        ap3)
            # NUMA cross-socket forced vs local.
            if [[ "$cfg" == "misuse" ]]; then
                echo "VLLM_LHC_REGIME_ADAPTIVE=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_FORCE_REMOTE_NUMA=1 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_INTERVAL=20"
            else
                echo "VLLM_LHC_REGIME_ADAPTIVE=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_INTERVAL=20"
            fi
            ;;
        ap4)
            # WQ-per-rank OFF (all 8 ranks share wq0.0).
            if [[ "$cfg" == "misuse" ]]; then
                echo "VLLM_LHC_REGIME_ADAPTIVE=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=0 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_INTERVAL=20"
            else
                echo "VLLM_LHC_REGIME_ADAPTIVE=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_INTERVAL=20"
            fi
            ;;
        ap5)
            # Regime detector OFF (Option A always-on) vs Option C adaptive.
            if [[ "$cfg" == "misuse" ]]; then
                echo "VLLM_LHC_REGIME_ADAPTIVE=0 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1"
            else
                echo "VLLM_LHC_REGIME_ADAPTIVE=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1 VLLM_LHC_REGIME_INTERVAL=20"
            fi
            ;;
    esac
}

for AP in $APS; do
    for WL in $WORKLOADS; do
        read INPUT_LEN OUTPUT_LEN PREFIX_LEN NPROMPTS CONC <<<"$(get_workload "$WL")"
        for CFG in baseline misuse; do
            ENV_PRE="$(build_env "$AP" "$CFG")"
            for S in $(seq 1 "$SWEEPS"); do
                TAG="${AP}_${CFG}_${WL}_s${S}"
                LOG="${RUNS}/${TAG}_boot.log"
                BENCH="${RUNS}/${TAG}_bench"

                if [[ -s "${BENCH}.json" ]]; then
                    echo "[$(ts)] skip existing $TAG"
                    continue
                fi

                echo "[$(ts)] === ${TAG} === env: ${ENV_PRE}" | tee "${LOG}"
                cleanup_orphans

                eval "${ENV_PRE} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                    nohup /workspace/vllm_dev_prj/bin/vllm serve ${MODEL} \
                      --port ${PORT} --host 127.0.0.1 \
                      --tensor-parallel-size ${TP} \
                      --gpu-memory-utilization ${GPU_MEM} \
                      --max-model-len ${MAX_LEN} \
                      --max-num-seqs ${CONC} \
                      --enable-prefix-caching \
                      >> ${LOG} 2>&1 &"
                SERVE_PID=$!
                echo "${SERVE_PID}" > "${RUNS}/${TAG}.pid"

                if ! wait_ready ${PORT}; then
                    echo "[$(ts)] $TAG: vllm NOT READY — skip" | tee -a "${LOG}"
                    cleanup_orphans
                    continue
                fi
                echo "[$(ts)] $TAG: ready" | tee -a "${LOG}"

                /workspace/vllm_dev_prj/bin/vllm bench serve \
                    --model "${MODEL}" \
                    --dataset-name sonnet \
                    --dataset-path "${DATA}" \
                    --sonnet-input-len ${INPUT_LEN} \
                    --sonnet-output-len ${OUTPUT_LEN} \
                    --sonnet-prefix-len ${PREFIX_LEN} \
                    --num-prompts ${NPROMPTS} \
                    --max-concurrency ${CONC} \
                    --port ${PORT} \
                    --save-result --result-dir "${RUNS}" \
                    --result-filename "${TAG}_bench.json" \
                    2>&1 | tee "${BENCH}.log"

                cleanup_orphans
            done
        done
    done
done

echo "[$(ts)] misuse sweep complete."
