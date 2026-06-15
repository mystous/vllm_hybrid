#!/usr/bin/env bash
# Orchestrate 9-model TSK_042 LHC Path 1 validation (Llama-8B handled by agent #72).
# Sequential per model. Each model: vanilla + lhc_path1, 7 corpus + mix, N sweeps.
# Big models (70B+) default to 1 sweep to fit time budget.
set -uo pipefail
cd /workspace/host_vllm_hybrid

ROOT=lhc_phase4/tsk042_10model_validation
RUNNER="$ROOT/run_model.sh"
GLOG="$ROOT/_orchestrate.log"
mkdir -p "$ROOT"

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$GLOG"; }

# Wait until GPU memory across all visible GPUs is mostly free (below threshold MiB total).
# Threshold: 8000 MiB total across 8 GPUs (= ~1GB/GPU; allows residual driver state).
wait_for_gpu_free(){
  local max_min="${1:-720}"  # default 12h max wait — orchestrate may be queued behind #72
  local elapsed=0 step=30
  while [ $elapsed -lt $((max_min * 60)) ]; do
    local u
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
    if [ "${u:-1}" -lt 8000 ]; then
      log "GPU is free (total used=${u} MiB)"
      return 0
    fi
    if [ $((elapsed % 300)) -eq 0 ]; then
      log "Waiting for GPU free … total used=${u} MiB (elapsed ${elapsed}s)"
    fi
    sleep $step
    elapsed=$((elapsed + step))
  done
  log "GPU wait timeout after ${max_min} min — proceeding anyway"
  return 1
}

# Model matrix: priority|tag|hf-id|TP|sweeps|gpu_mem_util
# DeepSeek-R1 671B (MoE) → TP=8 (single node). Note: original spec said 2×TP4 but for
# single-node we use TP=8 EP=1.
MODELS=(
  "1|Qwen2.5-7B-Instruct|Qwen/Qwen2.5-7B-Instruct|4|3|0.92"
  "2|DeepSeek-R1-Distill-Qwen-7B|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|4|3|0.92"
  "3|Qwen2.5-32B-Instruct|Qwen/Qwen2.5-32B-Instruct|8|3|0.92"
  "4|DeepSeek-R1-Distill-Qwen-32B|deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|8|3|0.92"
  "5|Llama-3.1-70B-Instruct|meta-llama/Llama-3.1-70B-Instruct|8|1|0.92"
  "6|DeepSeek-R1-Distill-Llama-70B|deepseek-ai/DeepSeek-R1-Distill-Llama-70B|8|1|0.92"
  "7|Qwen2.5-72B-Instruct|Qwen/Qwen2.5-72B-Instruct|8|1|0.92"
  "8|Llama-3.1-405B-Instruct-FP8|meta-llama/Llama-3.1-405B-Instruct-FP8|8|1|0.92"
  "9|DeepSeek-R1|deepseek-ai/DeepSeek-R1|8|1|0.92"
)

# Allow user filter via PRIORITY env (e.g. "1 2 3" to only run first three).
PRIORITY_FILTER="${PRIORITY:-1 2 3 4 5 6 7 8 9}"

# Per-model run with a budget-aware corpus/sweep schedule.
run_one(){
  local pri=$1 tag=$2 model=$3 tp=$4 sweeps=$5 gmu=$6
  log "=== priority=$pri tag=$tag model=$model TP=$tp sweeps=$sweeps gmu=$gmu ==="

  # For 405B / 671B: sanity-only — 1 corpus (humaneval) × 1 sweep, both configs.
  local corpora="sharegpt swebench humaneval mbpp wildchat lmsys"
  if [ "$pri" = "8" ] || [ "$pri" = "9" ]; then
    corpora="humaneval"
    log "[$tag] sanity-only schedule (1 corpus × 1 sweep × vanilla+lhc_path1)"
  fi

  # Skip if completion marker exists.
  local marker="$ROOT/${tag}/.done"
  if [ -f "$marker" ]; then
    log "[$tag] already done — skipping"
    return 0
  fi

  wait_for_gpu_free 720 || true

  MODEL="$model" TAG="$tag" TP="$tp" SWEEPS="$sweeps" \
    CORPORA="$corpora" GPU_MEM_UTIL="$gmu" \
    bash "$RUNNER" 2>&1 | tee -a "$GLOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    touch "$marker"
    log "[$tag] OK (marker written)"
  else
    log "[$tag] runner exited rc=$rc"
  fi
}

log "=== TSK_042 9-model orchestration start (filter='${PRIORITY_FILTER}') ==="
for entry in "${MODELS[@]}"; do
  IFS='|' read -r pri tag model tp sweeps gmu <<< "$entry"
  # check filter
  if ! echo " $PRIORITY_FILTER " | grep -q " $pri "; then
    log "[skip pri=$pri tag=$tag] not in filter"
    continue
  fi
  run_one "$pri" "$tag" "$model" "$tp" "$sweeps" "$gmu"
done
log "=== TSK_042 9-model orchestration DONE ==="
