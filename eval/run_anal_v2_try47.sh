#!/usr/bin/env bash
# [ANAL.v2] Phase 4 — try47 = v1.2 + probe + try22 skip 제거 + 강제 swap_out.
# Plan: 5 reqs × 8192 max_tok + KV pool 작게 (gpu-memory-utilization 0.40).
# probe P1~P8 활성. assertion fire req lifecycle 시계열 추적.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try47_anal_v2_phase4_repro"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}/pyspy_dumps"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
PYSPY=/workspace/vllm_dev_prj/bin/py-spy
export VLLM_NEO_PREDICTIVE_THRESHOLD=0.5
export VLLM_NEO_SWAP_OUT_RATIO=0.5
export VLLM_ANAL_DISABLE_TRY22_SKIP=1
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

echo "[try47] Phase 4 — minimal repro (5 reqs, tight KV pool)"
echo "[try47] launching … → ${LOG_FILE}"
"$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.40 \
    --max-model-len 9216 \
    --max-num-seqs 5 \
    --num-prompts 5 \
    --target-input-len 1024 \
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
echo "${LAUNCHER_PID}" > "${OUT_DIR}/launcher.pid"

echo "[try47] waiting for EngineCore spawn …"
ENGINE_PID=""
for _ in $(seq 1 120); do
    sleep 5
    ENGINE_PID="$(pgrep -af 'EngineCore|engine.*core|VllmWorker' \
        | awk '{print $1}' | head -1)"
    if [ -n "${ENGINE_PID}" ]; then
        echo "[try47] EngineCore PID=${ENGINE_PID}"
        echo "${ENGINE_PID}" > "${OUT_DIR}/engine_core.pid"
        break
    fi
done
[ -z "${ENGINE_PID}" ] && ENGINE_PID="${LAUNCHER_PID}"

echo "[try47] py-spy continuous (every 30s, up to 20 samples)"
for i in $(seq 1 20); do
    sleep 30
    DUMP_FILE="${OUT_DIR}/pyspy_dumps/dump_$(printf '%02d' "$i")_$(date +%H%M%S).txt"
    echo "--- dump $i @ $(date -Iseconds) ---" > "${DUMP_FILE}"
    "${PYSPY}" dump --pid "${ENGINE_PID}" --locals \
        >> "${DUMP_FILE}" 2>&1 || \
        echo "[try47] dump $i failed" >> "${DUMP_FILE}"
    if ! kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
        echo "[try47] launcher dead — exit sampling loop"
        break
    fi
done

wait "${LAUNCHER_PID}" 2>/dev/null
LAUNCHER_RC=$?
echo "[try47] launcher exit=${LAUNCHER_RC}"

# Summary — focus on assertion fire + swap lifecycle.
echo "[try47] result dir: ${OUT_DIR}"
echo "[try47] AssertionError occurrences:"
grep -nE 'AssertionError|Traceback' "${LOG_FILE}.stdout" 2>/dev/null | head -20
echo "[try47] [ANAL.P*] count by probe:"
grep -oE '\[ANAL\.P[0-9]+\]' "${LOG_FILE}.stdout" 2>/dev/null | sort | uniq -c
echo "[try47] swap activity:"
grep -cE 'new_swap_(in|out)=[1-9]|deferred_count=[1-9]|cdec_ids=[1-9]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[try47] DONE $(date -Iseconds)"
