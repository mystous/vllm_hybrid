#!/usr/bin/env bash
# [Plan v4 / Phase K — Option I + Option K + Option C] try80 회차.
# Goal: Option I (mirror MIN_BUFFER 누적) + Option K (D10 가드 완화 — mirror
#       의 SWAPPED_OUT reqs 의 num_new_tokens=1 부여) + Option C (decide_mode
#       load-balanced cdec 배포). 진짜 chain firing 활성화 fix.
# Stop condition: FORK STAT active > 0 + crash 0 + run 완주
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="try80_v4_K_OptIKC"
OUT_DIR="${ROOT_DIR}/eval/results/${TS}_${TAG}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/engine.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_LOAD_AWARE_MIN_RUNNING=32
export VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP=2
export VLLM_NEO_FORCE_SWAP_IN=1
export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
export VLLM_NEO_CPU_RESIDENT_REQS=64
export VLLM_NEO_SWAP_IN_ORDER=oldest

# Option I + K + C
export VLLM_NEO_MIRROR_MIN_BUFFER=8
export VLLM_NEO_OPTION_K=1
export VLLM_NEO_OPTION_C=1
unset VLLM_NEO_OPTION_A

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN

echo "[OptIKC] starting I+K+C → ${LOG_FILE}"
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
echo "[OptIKC] launcher exit=${LAUNCHER_RC}"

echo "[OptIKC] === Phase K-OptIKC gates ==="
echo "[OptIKC] Option K first fire (D10 num_new_tokens=1 부여):"
grep -c '\[Option K\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptIKC] Option I first fire:"
grep '\[Option I\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[OptIKC] Option C / D17C first fire:"
grep '\[Option C / D17C\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[OptIKC] Option C exception count:"
grep -c '\[Option C / D17C\] exception' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptIKC] D15+D16 first fire:"
grep '\[Plan v4 D15+D16\]' "${LOG_FILE}.stdout" 2>/dev/null | head -1
echo "[OptIKC] swap_out worker fire:"
grep -c '\[NEO SWAP_OUT CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptIKC] swap_in worker done (per-worker):"
echo "$(($(grep -c "swap-in: req" "${LOG_FILE}.stdout" 2>/dev/null) / 8))"
echo "[OptIKC] D11 OOB precheck:"
grep -c '\[NEO CDEC D11 OOB PRECHECK\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptIKC] NEO FORK STAT (last):"
grep 'NEO FORK STAT' "${LOG_FILE}.stdout" 2>/dev/null | tail -1
echo "[OptIKC] FORK active>0 count (chain firing fire 횟수):"
grep -E 'NEO FORK STAT.*active=[1-9]' "${LOG_FILE}.stdout" 2>/dev/null | wc -l
echo "[OptIKC] NEO CDEC CALL counts:"
grep -c '\[NEO CDEC CALL\]' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptIKC] mirror size top5:"
grep 'mirror_set_size' "${LOG_FILE}.stdout" 2>/dev/null | grep -oE 'mirror_set_size=[0-9]+' | sort | uniq -c | sort -rn | head -5
echo "[OptIKC] crash counts:"
grep -cE 'AssertionError|OutOfMemoryError|EngineDeadError|CUDA error|Segfault encountered|brute::store_kv' "${LOG_FILE}.stdout" 2>/dev/null
echo "[OptIKC] result.json:"
ls -la "${OUT_DIR}/result.json" 2>/dev/null || echo "(미생성)"
echo "[OptIKC] DONE $(date -Iseconds)"
