#!/usr/bin/env bash
# [ANAL.v2] Phase 3 — try46 = v38 (eeed0d46fc) checkout + 동일 workload.
# v38 의 cdec firing 93% path 의 실 runtime trace 확보 (vs v1.2 미발화).
# 운영: 1) git stash (probes 보존) → 2) checkout v38 → 3) launch → 4) checkout
#       feat → 5) stash pop. C++ src 일부 변경 있으나 Python 측 chain 분석에는
#       영향 없으므로 rebuild skip (.so HEAD 버전 사용).
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "[try46] current branch: ${CURRENT_BRANCH}"
if [ "${CURRENT_BRANCH}" != "feat/ide006-tsk019-neo-performance-max" ]; then
    echo "[try46] ABORT: must run from feat/ide006-tsk019-neo-performance-max"
    exit 1
fi

# 1) stash probes (tracked-modified만; untracked eval/ 보존)
echo "[try46] git stash (probes 보존)"
STASH_REF=""
if [ -n "$(git diff --name-only HEAD)" ]; then
    git stash push -m "anal_v2_try46_pre_checkout" || {
        echo "[try46] ABORT: stash failed"; exit 1;
    }
    STASH_REF="$(git rev-parse --quiet --verify stash@{0})"
    echo "[try46] stash ref: ${STASH_REF}"
fi

# 2) checkout v38 — *vllm/ 만* 으로 한정. eval/ 은 HEAD 의 launcher 유지.
echo "[try46] checkout eeed0d46fc -- vllm/"
if ! git checkout eeed0d46fc -- vllm/ 2>&1 | head -3; then
    echo "[try46] ABORT: checkout failed"
    [ -n "${STASH_REF}" ] && git stash pop
    exit 1
fi

# 3) launch
TS="$(date +%Y%m%d_%H%M%S)"
TAG="try46_anal_v2_phase3_v38"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}/pyspy_dumps"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
PYSPY=/workspace/vllm_dev_prj/bin/py-spy
export VLLM_NEO_PREDICTIVE_THRESHOLD=0.5
export VLLM_NEO_SWAP_OUT_RATIO=0.5
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

echo "[try46] launching v38 baseline (background)"
"$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
    --model llama-70b \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.75 \
    --max-model-len 16384 \
    --max-num-seqs 256 \
    --num-prompts 500 \
    --target-input-len 8192 \
    --max-tokens 8192 \
    --enable-neo-asymmetric \
    --log-file "${LOG_FILE}" \
    --output-file "${OUT_DIR}/result.json" \
    > "${LOG_FILE}.stdout" 2>&1 &
LAUNCHER_PID=$!
echo "${LAUNCHER_PID}" > "${OUT_DIR}/launcher.pid"

# 4) py-spy 5x dumps spaced 60s
echo "[try46] waiting for EngineCore spawn …"
ENGINE_PID=""
for _ in $(seq 1 120); do
    sleep 5
    ENGINE_PID="$(pgrep -af 'EngineCore|engine.*core|VllmWorker' \
        | awk '{print $1}' | head -1)"
    if [ -n "${ENGINE_PID}" ]; then
        echo "[try46] EngineCore PID=${ENGINE_PID}"
        echo "${ENGINE_PID}" > "${OUT_DIR}/engine_core.pid"
        break
    fi
done
[ -z "${ENGINE_PID}" ] && ENGINE_PID="${LAUNCHER_PID}"

echo "[try46] py-spy dump x5 (spacing 60s)"
for i in 1 2 3 4 5; do
    sleep 60
    DUMP_FILE="${OUT_DIR}/pyspy_dumps/dump_$(printf '%02d' "$i")_$(date +%H%M%S).txt"
    echo "--- dump $i @ $(date -Iseconds) ---" > "${DUMP_FILE}"
    "${PYSPY}" dump --pid "${ENGINE_PID}" --locals \
        >> "${DUMP_FILE}" 2>&1 || \
        echo "[try46] dump $i failed (process gone?)" >> "${DUMP_FILE}"
    if ! kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
        echo "[try46] launcher dead — exit sampling loop"
        break
    fi
done

# 5) wait + cleanup
wait "${LAUNCHER_PID}" 2>/dev/null
LAUNCHER_RC=$?
echo "[try46] launcher exit=${LAUNCHER_RC}"

# 6) revert checkout + restore probes
echo "[try46] revert vllm/ to HEAD"
git checkout HEAD -- vllm/ 2>&1 | head -3
if [ -n "${STASH_REF}" ]; then
    echo "[try46] git stash pop"
    git stash pop 2>&1 | head -5
fi

# 7) summary
echo "[try46] result dir: ${OUT_DIR}"
grep -cE 'cdec_future|CDEC|FORK|swap_in|swap_out|preempt' \
    "${LOG_FILE}.stdout" 2>/dev/null \
    | head -5
echo "[try46] DONE $(date -Iseconds)"
