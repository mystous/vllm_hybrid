#!/usr/bin/env bash
# LHC Phase 4 — METRONOME-LHC integration sweep (LHC_P4_005).
#
# 9 workloads × 7 configs × 3 sweeps. Produces:
#   lhc_phase4/runs/<workload>_<config>_<sweep>.{json,log,hookstats}
#
# After all 189 sweep cells, aggregate via:
#   python3 lhc_phase4/aggregate.py
#
# Workloads:
#   sonnet, chat, code, balanced, sonnet-heavy, code-heavy, wd1, wd2, wd3
#
# Configs:
#   vanilla
#   dsa             — VLLM_LHC_DSA=1
#   amx_c3          — VLLM_LHC_AMX_C3=1 + standalone hash hook
#   dsa_amx         — DSA + AMX C3, no orchestrator
#   metronome       — VLLM_LHC_METRONOME=1 + DSA + AMX C3 + 5-stage
#   metronome_sfx   — METRONOME-LHC + Suffix (--speculative-config)
#   metronome_full  — METRONOME-LHC + Suffix + fp8 KV

set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase4
mkdir -p "${BASE}/runs"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT=8500
TP=8
GPU_MEM=0.92

WORKLOADS="${WORKLOADS:-sonnet chat code balanced sonnet-heavy code-heavy wd1 wd2 wd3}"
CONFIGS="${CONFIGS:-vanilla dsa amx_c3 dsa_amx metronome metronome_sfx metronome_full}"
SWEEPS="${SWEEPS:-3}"

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

# Pre-clean orphan vllm.
pgrep -f "vllm serve" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2

for WORKLOAD in $WORKLOADS; do
    case "$WORKLOAD" in
        sonnet)        INPUT_LEN=512;  OUTPUT_LEN=512; PREFIX_LEN=0;    NPROMPTS=500; CONC=64;  MAX_LEN=4096  ;;
        chat)          INPUT_LEN=256;  OUTPUT_LEN=512; PREFIX_LEN=0;    NPROMPTS=500; CONC=64;  MAX_LEN=4096  ;;
        code)          INPUT_LEN=1024; OUTPUT_LEN=512; PREFIX_LEN=0;    NPROMPTS=500; CONC=64;  MAX_LEN=4096  ;;
        balanced)      INPUT_LEN=768;  OUTPUT_LEN=768; PREFIX_LEN=0;    NPROMPTS=500; CONC=64;  MAX_LEN=4096  ;;
        sonnet-heavy)  INPUT_LEN=2048; OUTPUT_LEN=2048;PREFIX_LEN=0;    NPROMPTS=200; CONC=64;  MAX_LEN=8192  ;;
        code-heavy)    INPUT_LEN=4096; OUTPUT_LEN=1024;PREFIX_LEN=0;    NPROMPTS=200; CONC=64;  MAX_LEN=8192  ;;
        wd1)           INPUT_LEN=24000;OUTPUT_LEN=4096;PREFIX_LEN=200;  NPROMPTS=64;  CONC=32;  MAX_LEN=32768 ;;
        wd2)           INPUT_LEN=512;  OUTPUT_LEN=512; PREFIX_LEN=0;    NPROMPTS=500; CONC=64;  MAX_LEN=4096  ;;
        wd3)           INPUT_LEN=12000;OUTPUT_LEN=512; PREFIX_LEN=8000; NPROMPTS=128; CONC=32;  MAX_LEN=16384 ;;
        *) echo "unknown workload: $WORKLOAD"; continue ;;
    esac

    for CONFIG in $CONFIGS; do
        FLAGS=""
        ENV_PRE=""
        case "$CONFIG" in
            vanilla)        ENV_PRE="" ;;
            dsa)            ENV_PRE="VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=4096" ;;
            amx_c3)         ENV_PRE="VLLM_LHC_AMX_C3=1" ;;
            dsa_amx)        ENV_PRE="VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1" ;;
            metronome)      ENV_PRE="VLLM_LHC_METRONOME=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1"; FLAGS="--enable-neo-asymmetric" ;;
            metronome_sfx)  ENV_PRE="VLLM_LHC_METRONOME=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1"; FLAGS="--enable-neo-asymmetric --speculative-config '{\"method\":\"suffix\",\"num_speculative_tokens\":4}'" ;;
            metronome_full) ENV_PRE="VLLM_LHC_METRONOME=1 VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_AMX_C3=1"; FLAGS="--enable-neo-asymmetric --speculative-config '{\"method\":\"suffix\",\"num_speculative_tokens\":4}' --kv-cache-dtype fp8" ;;
            *) echo "unknown config: $CONFIG"; continue ;;
        esac

        for SWEEP in $(seq 1 $SWEEPS); do
            TAG="${WORKLOAD}_${CONFIG}_${SWEEP}"
            LOG="${BASE}/runs/${TAG}_boot.log"
            BENCH="${BASE}/runs/${TAG}_bench"

            if [[ -s "${BENCH}.json" ]]; then
                echo "[$(ts)] skip existing $TAG"
                continue
            fi

            echo "[$(ts)] === ${TAG} ===" | tee "${LOG}"

            pgrep -f "vllm serve" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
            sleep 2

            eval "${ENV_PRE} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
                nohup /workspace/vllm_dev_prj/bin/vllm serve ${MODEL} \
                  --port ${PORT} --host 127.0.0.1 \
                  --tensor-parallel-size ${TP} \
                  --gpu-memory-utilization ${GPU_MEM} \
                  --max-model-len ${MAX_LEN} \
                  --max-num-seqs ${CONC} \
                  --enable-prefix-caching \
                  ${FLAGS} \
                  >> ${LOG} 2>&1 &"
            SERVE_PID=$!
            echo "${SERVE_PID}" > "${BASE}/runs/${TAG}.pid"

            # Wait for ready (max 600s).
            READY=0
            for i in $(seq 1 120); do
                if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
                    READY=1; break
                fi
                sleep 5
            done
            if [[ $READY -eq 0 ]]; then
                echo "[$(ts)] $TAG: vllm not ready, skip" | tee -a "${LOG}"
                pgrep -f "vllm serve" 2>/dev/null | xargs -r kill -9 || true
                continue
            fi

            /workspace/vllm_dev_prj/bin/vllm bench serve \
                --model "${MODEL}" \
                --dataset-name sonnet \
                --sonnet-input-len ${INPUT_LEN} \
                --sonnet-output-len ${OUTPUT_LEN} \
                --sonnet-prefix-len ${PREFIX_LEN} \
                --num-prompts ${NPROMPTS} \
                --max-concurrency ${CONC} \
                --port ${PORT} \
                --save-result --result-dir "${BASE}/runs" \
                --result-filename "${TAG}_bench.json" \
                2>&1 | tee "${BENCH}.log"

            pgrep -f "vllm serve" 2>/dev/null | xargs -r kill -9 || true
            sleep 3
        done
    done
done

echo "[$(ts)] sweep complete. aggregate with: python3 ${BASE}/aggregate.py"
