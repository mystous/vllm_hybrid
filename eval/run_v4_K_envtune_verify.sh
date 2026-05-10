#!/usr/bin/env bash
# [Plan v4 / Phase K — env tune verify] try63 회차.
# Goal: D6+D7 코드 그대로 + env tuning 으로 mirror cycle 단축 →
#       SEGV 회피 (block_pos OOB margin 확보) + NEO 발화 유지.
#
# SEGV 안전 조건 (수학):
#   cycle = mirror_max / cap < BLOCK_SIZE (16)
#   본 회차: cycle = 64/8 = 8 step (margin 7 step)
#
# Stop condition (PASS):
#   - run 정상 종료 (EngineDeadError/segfault 0)
#   - swap_in done > 100 (반복 fire)
#   - SWAP_IN_DONE > 0 sustained → #14 KV migration LRU ✅
#   - chain firing 발화 (cdec dispatch 활성)
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try63_v4_K_envtune_cap8_mirror64"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

# [env tune] cap 4× ↑, mirror_max default 유지 → cycle 8 step (안전 margin 7).
export VLLM_NEO_FORCE_SWAP_IN=1
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=8
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO

echo "[K-envtune] starting cap=8 mirror=64 cycle=8 → ${LOG_FILE}"
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
echo "[K-envtune] launcher exit=${LAUNCHER_RC}"

echo "[K-envtune] result dir: ${OUT_DIR}"
echo "[K-envtune] === Phase K-envtune gates ==="
echo "[K-envtune] swap_out worker fire (NEO SWAP_OUT CALL):"
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-envtune] swap_in attach total log fires:"
grep -c '\[Plan v4 D4\] swap_in attach' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-envtune] swap_in worker done count (per-worker):"
echo "$(($(grep -c "swap-in: req" "${LOG_FILE}.stdout" 2>/dev/null) / 8))"
echo "[K-envtune] D7 race guard fire counts:"
grep -c '\[Plan v4 D7\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-envtune] CPU-resident skip (mirror_size 추세):"
grep '\[Plan v4 H\]' "${LOG_FILE}.stdout" 2>/dev/null | tail -3
echo "[K-envtune] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[K-envtune] crash counts (incl. segfault):"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error|Segfault encountered|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null
echo "[K-envtune] DONE $(date -Iseconds)"
