#!/usr/bin/env bash
# Unified TSK_042 10-model orchestration. Sequential per model.
# Spec (user-given):
#   1 Llama-3.1-8B-Instruct          TP=8 sweep=3 (all 7 corpus)
#   2 Qwen/Qwen2.5-7B-Instruct       TP=4 sweep=3
#   3 DS-R1-Distill-Qwen-7B          TP=4 sweep=3
#   4 Llama-3.1-70B-Instruct         TP=8 sweep=2
#   5 Qwen/Qwen2.5-32B-Instruct      TP=8 sweep=3
#   6 DS-R1-Distill-Qwen-32B         TP=8 sweep=3
#   7 DS-R1-Distill-Llama-70B        TP=8 sweep=2
#   8 Qwen/Qwen2.5-72B-Instruct      TP=8 sweep=2
#   9 Llama-3.1-405B-FP8             TP=8 sweep=1
#  10 DeepSeek-R1                    TP=8 sweep=1 corpus=mix only
set -uo pipefail
cd /workspace/host_vllm_hybrid

ROOT=lhc_phase4/tsk042_10model_unified
RUNNER="$ROOT/run_model.sh"
GLOG="$ROOT/_orchestrate.log"
mkdir -p "$ROOT"

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$GLOG"; }

wait_for_gpu_free(){
  local max_min="${1:-30}"
  local elapsed=0 step=15
  while [ $elapsed -lt $((max_min * 60)) ]; do
    local u
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
    if [ "${u:-1}" -lt 8000 ]; then
      log "GPU is free (total used=${u} MiB)"
      return 0
    fi
    if [ $((elapsed % 120)) -eq 0 ]; then
      log "Waiting for GPU free … total used=${u} MiB (elapsed ${elapsed}s)"
    fi
    sleep $step
    elapsed=$((elapsed + step))
  done
  log "GPU wait timeout after ${max_min} min — kill orphans + proceeding"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u); do
    kill -9 "$p" 2>/dev/null || true
  done
  sleep 5
  return 1
}

# priority|tag|hf-id|TP|sweeps|gpu_mem_util|corpora_override|sweeps_for_lhc
# corpora_override empty → use 7 corpus (6 + mix).
# corpora_override = "mix_only" → only mix.
MODELS=(
  "1|Llama-3.1-8B-Instruct|meta-llama/Llama-3.1-8B-Instruct|8|3|0.92|"
  "2|Qwen2.5-7B-Instruct|Qwen/Qwen2.5-7B-Instruct|4|3|0.92|"
  "3|DeepSeek-R1-Distill-Qwen-7B|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|4|3|0.92|"
  "4|Llama-3.1-70B-Instruct|meta-llama/Llama-3.1-70B-Instruct|8|2|0.92|"
  "5|Qwen2.5-32B-Instruct|Qwen/Qwen2.5-32B-Instruct|8|3|0.92|"
  "6|DeepSeek-R1-Distill-Qwen-32B|deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|8|3|0.92|"
  "7|DeepSeek-R1-Distill-Llama-70B|deepseek-ai/DeepSeek-R1-Distill-Llama-70B|8|2|0.92|"
  "8|Qwen2.5-72B-Instruct|Qwen/Qwen2.5-72B-Instruct|8|2|0.92|"
  "9|Llama-3.1-405B-Instruct-FP8|meta-llama/Llama-3.1-405B-Instruct-FP8|8|1|0.92|"
  "10|DeepSeek-R1|deepseek-ai/DeepSeek-R1|8|1|0.92|mix_only"
)

PRIORITY_FILTER="${PRIORITY:-1 2 3 4 5 6 7 8 9 10}"

run_one(){
  local pri=$1 tag=$2 model=$3 tp=$4 sweeps=$5 gmu=$6 corpora_override=$7
  log "=== priority=$pri tag=$tag model=$model TP=$tp sweeps=$sweeps gmu=$gmu corpora_override='${corpora_override}' ==="

  local corpora="sharegpt swebench humaneval mbpp wildchat lmsys"
  local corpora_exact=0
  if [ "$corpora_override" = "mix_only" ]; then
    corpora=""  # only mix corpus (handled by mix block in runner)
    corpora_exact=0
  fi

  local sweep_list=""
  for i in $(seq 1 "$sweeps"); do sweep_list="$sweep_list $i"; done

  local marker="$ROOT/${tag}/.done"
  if [ -f "$marker" ]; then
    log "[$tag] already done — skipping"
    return 0
  fi

  wait_for_gpu_free 30 || true

  MODEL="$model" TAG="$tag" TP="$tp" SWEEPS="${sweep_list# }" \
    CORPORA="$corpora" CORPORA_EXACT="$corpora_exact" \
    GPU_MEM_UTIL="$gmu" \
    bash "$RUNNER" 2>&1 | tee -a "$GLOG"

  log "=== priority=$pri tag=$tag finished ==="
  return 0
}

log "=== TSK_042 10-model unified orchestration start (filter='$PRIORITY_FILTER') ==="
for entry in "${MODELS[@]}"; do
  IFS='|' read -r pri tag model tp sweeps gmu corpora_override <<<"$entry"
  if [[ " $PRIORITY_FILTER " == *" $pri "* ]]; then
    run_one "$pri" "$tag" "$model" "$tp" "$sweeps" "$gmu" "$corpora_override"
  else
    log "[$tag] not in PRIORITY_FILTER='$PRIORITY_FILTER' — skipping"
  fi
done
log "=== TSK_042 10-model unified orchestration finish ==="
