#!/usr/bin/env bash
# [Plan v4 / Phase K — D6+D7 verify] try62 회차.
# Goal: D7 (swap_in/cdec race guard) 적용 후 try61 의 SEGV 회피
#       + run 정상 완주 + swap_in 반복 발화.
#
# Stop condition (PASS):
#   - run 정상 종료 (EngineDeadError/segfault 0)
#   - [Plan v4 D7] swap_in/cdec race guard first fire 출현
#   - [Plan v4 D4] swap_in attach 반복 발화 (count > 1)
#   - 19 항목 #14 ✅ 진입 (SWAP_IN_DONE > 0 sustained)
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try62_v4_K_D7_verify_race_guard"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

export VLLM_NEO_FORCE_SWAP_IN=1
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=2
export VLLM_NEO_SWAP_IN_ORDER=oldest

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO

echo "[K-D7] starting Plan v4 D7 race-guard verify → ${LOG_FILE}"
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
    > "${LOG_FILE}.stdout" 2>&1
LAUNCHER_RC=$?
echo "[K-D7] launcher exit=${LAUNCHER_RC}"

echo "[K-D7] result dir: ${OUT_DIR}"
echo "[K-D7] === Phase K-D7 gates ==="
echo "[K-D7] swap_out worker fire (NEO SWAP_OUT CALL):"
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D7] swap_in attach total counts (D4/D6 path):"
grep -c '\[Plan v4 D4\] swap_in attach' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D7] D7 race guard first fire:"
grep '\[Plan v4 D7\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[K-D7] D7 race guard total fire counts:"
grep -c '\[Plan v4 D7\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D7] CPU-resident skip log:"
grep '\[Plan v4 H\]' "${LOG_FILE}.stdout" 2>/dev/null | head -3
echo "[K-D7] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[K-D7] crash counts (incl. segfault):"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error|Segfault encountered|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-D7] DONE $(date -Iseconds)"
