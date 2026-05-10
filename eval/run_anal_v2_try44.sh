#!/usr/bin/env bash
# [ANAL.v2] Phase 2-A — try44 = v1.2 + probe + minimal repro env.
# 시나리오 A: try22 skip 유지 (assertion fire X), v1.2 그대로.
# Workload: 10 reqs × 16384 max_tok, VLLM_NEO_PREDICTIVE_THRESHOLD=0.5.
# py-spy 5회 sample on /[ANAL.P2|ANAL.P8]/ events (swap fire trigger).
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try44_anal_v2_phase2A"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"
DUMP_DIR="${OUT_DIR}/pyspy_dumps"
mkdir -p "${DUMP_DIR}"

# 1) launch baseline (background)
PY=/workspace/vllm_dev_prj/bin/python
export VLLM_NEO_PREDICTIVE_THRESHOLD=0.5
export VLLM_NEO_SWAP_OUT_RATIO=0.5
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

echo "[try44] starting launcher (background) → ${LOG_FILE}"
# 500p × 50:50 production workload (PLN_001 §5.6 baseline 정합) — KV pool
# 자연스럽게 90%+ 도달 → swap_out 자연 발화 + chain dynamics 캡처.
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
echo "[try44] launcher PID=${LAUNCHER_PID}"
echo "${LAUNCHER_PID}" > "${OUT_DIR}/launcher.pid"

# 2) wait for engine core to spawn (vllm splits into LLM driver + EngineCore subproc)
echo "[try44] waiting for EngineCore spawn …"
ENGINE_PID=""
for _ in $(seq 1 120); do
    sleep 5
    # find python child of launcher containing "EngineCore" in cmdline or related
    ENGINE_PID="$(pgrep -af 'EngineCore|engine.*core|VllmWorker' \
        | awk '{print $1}' | head -1)"
    if [ -n "${ENGINE_PID}" ]; then
        echo "[try44] EngineCore PID=${ENGINE_PID}"
        echo "${ENGINE_PID}" > "${OUT_DIR}/engine_core.pid"
        break
    fi
done
if [ -z "${ENGINE_PID}" ]; then
    echo "[try44] WARN: EngineCore PID not found via pgrep — use launcher PID"
    ENGINE_PID="${LAUNCHER_PID}"
fi

# 3) py-spy: 5x dumps spaced 60s (production workload ~10-15 min) — captures
# steady-state stack distribution including swap_out fire windows.
PYSPY=/workspace/vllm_dev_prj/bin/py-spy
echo "[try44] py-spy dump x5 (spacing 60s)"
for i in 1 2 3 4 5; do
    sleep 60
    DUMP_FILE="${DUMP_DIR}/dump_$(printf '%02d' "$i")_$(date +%H%M%S).txt"
    echo "--- dump $i @ $(date -Iseconds) ---" > "${DUMP_FILE}"
    "${PYSPY}" dump --pid "${ENGINE_PID}" --locals \
        >> "${DUMP_FILE}" 2>&1 || \
        echo "[try44] dump $i failed (process gone?)" >> "${DUMP_FILE}"
    if ! kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
        echo "[try44] launcher dead — exit sampling loop"
        break
    fi
done

# 4) wait for launcher to complete (or timeout)
wait "${LAUNCHER_PID}" 2>/dev/null || true
echo "[try44] launcher done. exit=$?"

# 5) summary — vllm Python logging goes to stderr → ${LOG_FILE}.stdout
echo "[try44] result dir: ${OUT_DIR}"
echo "[try44] stdout log lines: $(wc -l < "${LOG_FILE}.stdout" 2>/dev/null || echo 0)"
echo "[try44] [ANAL.P*] count by probe:"
grep -oE '\[ANAL\.P[0-9]+\]' "${LOG_FILE}.stdout" 2>/dev/null | sort | uniq -c
echo "[try44] swap activity check:"
grep -cE 'new_swap_(in|out)=[1-9]|deferred_count=[1-9]|cdec_ids=[1-9]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[try44] DONE $(date -Iseconds)"
