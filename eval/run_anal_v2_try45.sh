#!/usr/bin/env bash
# [ANAL.v2] Phase 2-B — try45 = v1.2 + probe + try22 skip 제거 (분석 patch).
# 시나리오 B: assertion fire reproducer.
# Workload: 동일 minimal repro (10 reqs × 16384 max_tok, THRESHOLD=0.5).
# py-spy continuous sampling on assertion fire trigger.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try45_anal_v2_phase2B"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"
DUMP_DIR="${OUT_DIR}/pyspy_dumps"
mkdir -p "${DUMP_DIR}"

PY=/workspace/vllm_dev_prj/bin/python
PYSPY=/workspace/vllm_dev_prj/bin/py-spy
export VLLM_NEO_PREDICTIVE_THRESHOLD=0.5
export VLLM_NEO_SWAP_OUT_RATIO=0.5
# Phase 2-B 핵심: try22 skip 제거 → assertion fire reproducer.
export VLLM_ANAL_DISABLE_TRY22_SKIP=1
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

echo "[try45] starting launcher (background) → ${LOG_FILE}"
# 500p × 50:50 production workload — try22 skip 제거 + 자연 swap_out fire.
"$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b \
    --tensor-parallel-size 8 \
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
echo "[try45] launcher PID=${LAUNCHER_PID}" | tee "${OUT_DIR}/launcher.pid"
echo "${LAUNCHER_PID}" > "${OUT_DIR}/launcher.pid"

# Wait for EngineCore spawn.
echo "[try45] waiting for EngineCore spawn …"
ENGINE_PID=""
for _ in $(seq 1 120); do
    sleep 5
    ENGINE_PID="$(pgrep -af 'EngineCore|engine.*core|VllmWorker' \
        | awk '{print $1}' | head -1)"
    if [ -n "${ENGINE_PID}" ]; then
        echo "[try45] EngineCore PID=${ENGINE_PID}"
        echo "${ENGINE_PID}" > "${OUT_DIR}/engine_core.pid"
        break
    fi
done
if [ -z "${ENGINE_PID}" ]; then
    ENGINE_PID="${LAUNCHER_PID}"
fi

# Continuous py-spy sampling — every 30s; expect assertion fire mid-run.
# Loop bounded by 30 samples (15 min) — assertion fire 시 즉시 break.
echo "[try45] py-spy continuous sample (every 30s, up to 30 samples)"
for i in $(seq 1 30); do
    sleep 30
    DUMP_FILE="${DUMP_DIR}/dump_$(printf '%02d' "$i")_$(date +%H%M%S).txt"
    echo "--- dump $i @ $(date -Iseconds) ---" > "${DUMP_FILE}"
    "${PYSPY}" dump --pid "${ENGINE_PID}" --locals \
        >> "${DUMP_FILE}" 2>&1 || \
        echo "[try45] dump $i failed (process gone?)" >> "${DUMP_FILE}"
    if ! kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
        echo "[try45] launcher dead → assertion fire likely. Exiting samples."
        break
    fi
done

# Wait for launcher (or already dead).
wait "${LAUNCHER_PID}" 2>/dev/null
LAUNCHER_RC=$?
echo "[try45] launcher exit=${LAUNCHER_RC}"

# Summary.
echo "[try45] result dir: ${OUT_DIR}"
echo "[try45] AssertionError occurrences:"
grep -nE 'AssertionError|Traceback' "${LOG_FILE}.stdout" 2>/dev/null | head -10
echo "[try45] [ANAL.P*] count by probe:"
grep -oE '\[ANAL\.P[0-9]+\]' "${LOG_FILE}.stdout" 2>/dev/null | sort | uniq -c
echo "[try45] DONE $(date -Iseconds)"
