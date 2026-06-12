#!/usr/bin/env bash
# SUB_201 Optimal Config vs MWOR — 10 model × 7 corpus, **TP=8 강제** (B200 8 GPU).
#
# 사용자 명령:
#   - TP=8 가능하면 모두 TP=8 (이전 B3 8GPU FaP 측정에서 작은 모델 +41% finding)
#   - TP=8 불가 모델 (heads=28): TP=4 fallback (Qwen-2.5-7B, DS-R1-Distill-Qwen-7B)
#   - spec: 200p × conc=16 × max-tok=512 (TSK_042 protocol 일치)
#   - 결과: runs_tp8/
#   - Optimal: vanilla + FaP + L2 + L10
#   - MWOR: per (family, corpus) oracle winner + FaP + L2 + L10
set -uo pipefail
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/optimal_vs_mwor
RUNS="$ROOT/runs_tp8"
LOGD="$ROOT/logs_tp8"
mkdir -p "$RUNS" "$LOGD"
RAW="$RUNS/per_request_raw.jsonl"
[ -f "$RAW" ] || : > "$RAW"

SAMPLED=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet
RE=vllm_config_perf/gating/realistic_eval

# TSK_042 protocol
CONC=16
MAXTOK=512
LIMIT=200
MML=16384

# common env (L2 + L10 always-on)
export LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib
export ARCTIC_INFERENCE_ENABLED=0
export VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8
export VLLM_NGRAM_DIVIDE_BY_TP=0
export VLLM_PREFETCH_TOKENIZE=1
export VLLM_PREFETCH_TOKENIZE_WORKERS=2
export VLLM_BURST_AWARE_ADMISSION=1

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

wait_ready(){ local url=$1 pid=$2 timeout_s=${3:-1800}
  local end=$(( $(date +%s) + timeout_s ))
  while [ "$(date +%s)" -lt "$end" ]; do
    curl -sf "$url/v1/models" >/dev/null 2>&1 && { log "READY $url"; return 0; }
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      log "DEAD backend $url (pid $pid)"; return 1; fi
    sleep 5
  done
  log "TIMEOUT $url"; return 1
}

kill_pgroup(){ local pid=$1
  [ -z "$pid" ] && return 0
  local pg; pg=$(ps -o pgid= -p "$pid" 2>/dev/null|tr -d ' ')
  [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null
  kill -9 "$pid" 2>/dev/null
  sleep 3
  # orphan VLLM::Worker compute-apps PID 직접 kill (사용자 protocol)
  local cpids; cpids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  for cp in $cpids; do [ -n "$cp" ] && kill -9 "$cp" 2>/dev/null; done
  sleep 2
}

wait_gpu_free(){ for i in $(seq 1 48); do
  local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}')
  [ "${u:-1}" -lt 4000 ] && { log "GPU freed"; return 0; }
  sleep 5
done; log "GPU not fully freed (계속)"; return 0; }

# (model_id, tag, family, tp, gpus, optimal_method, boot_timeout_s, mwor_unique_methods_joined)
# TP=8 강제 (Qwen-7B, DS-Qwen-7B 는 heads=28 → TP=4 fallback)
# mwor_methods: oracle_table 의 unique winner set (vanilla 제외, llm-d 제외 — llm-d 는 별 backend → estimation)
#   = suffix (대부분), ngram (없음), vanilla (= Optimal 재사용)
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct|Qwen2.5-7B-Instruct|Qwen-7B|4|0,1,2,3|vanilla|900|suffix"
  "Qwen/Qwen2.5-32B-Instruct|Qwen2.5-32B-Instruct|Qwen-32B|8|0,1,2,3,4,5,6,7|vanilla|1500|suffix"
  "Qwen/Qwen2.5-72B-Instruct|Qwen2.5-72B-Instruct|Qwen-72B|8|0,1,2,3,4,5,6,7|vanilla|1800|suffix"
  "meta-llama/Llama-3.1-8B-Instruct|Llama-3.1-8B-Instruct|Llama-8B|8|0,1,2,3,4,5,6,7|vanilla|1200|suffix"
  "meta-llama/Llama-3.1-70B-Instruct|Llama-3.1-70B-Instruct|Llama-70B|8|0,1,2,3,4,5,6,7|vanilla|1800|suffix"
  "meta-llama/Llama-3.1-405B-Instruct-FP8|Llama-3.1-405B-Instruct-FP8|Llama-405B|8|0,1,2,3,4,5,6,7|vanilla|3000|suffix"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|DeepSeek-R1-Distill-Qwen-7B|DS-Qwen-7B|4|0,1,2,3|vanilla|900|suffix"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|DeepSeek-R1-Distill-Qwen-32B|DS-Qwen-32B|8|0,1,2,3,4,5,6,7|vanilla|1500|suffix"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B|DeepSeek-R1-Distill-Llama-70B|DS-Llama-70B|8|0,1,2,3,4,5,6,7|vanilla|1800|suffix"
  "deepseek-ai/DeepSeek-R1|DeepSeek-R1|DS-R1-671B|8|0,1,2,3,4,5,6,7|vanilla|3600|"
)

CORPORA=(humaneval mbpp swebench sharegpt lmsys wildchat mix)

spec_arg(){ case "$1" in
  vanilla) echo "";;
  suffix)  echo '--speculative-config {"method":"suffix","num_speculative_tokens":7}';;
  ngram)   echo '--speculative-config {"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":4}';;
  *) echo "";; esac; }

run_method_boot(){
  # $1=model $2=tag $3=tp $4=gpus $5=method $6=boot_timeout $7=conf_label
  local model=$1 tag=$2 tp=$3 gpus=$4 method=$5 timeout=$6 conf=$7

  local need_run=0
  for corp in "${CORPORA[@]}"; do
    local out="$RUNS/summ_${tag}_${conf}_${method}_${corp}.json"
    if [ ! -f "$out" ]; then need_run=1; break; fi
  done
  if [ "$need_run" -eq 0 ]; then
    log "SKIP all corpus exists $tag × $conf × $method"
    return 0
  fi

  local boot_log="$LOGD/${tag}_${conf}_${method}_boot.log"
  local sa; sa=$(spec_arg "$method")
  log "==== boot $tag conf=$conf method=$method tp=$tp gpus=$gpus ===="
  CUDA_VISIBLE_DEVICES=$gpus setsid "$VBIN" serve "$model" \
    --tensor-parallel-size $tp --port 8001 --gpu-memory-utilization 0.85 \
    --max-model-len $MML --allow-deprecated-quantization \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    $sa > "$boot_log" 2>&1 < /dev/null &
  local pid=$!
  log "  PID=$pid boot_log=$boot_log timeout=${timeout}s"
  if ! wait_ready http://127.0.0.1:8001 "$pid" "$timeout"; then
    log "  BOOT FAIL $tag × $conf × $method"
    for corp in "${CORPORA[@]}"; do
      local out="$RUNS/summ_${tag}_${conf}_${method}_${corp}.json"
      [ ! -f "$out" ] && : > "$out.FAIL"
    done
    kill_pgroup "$pid"
    wait_gpu_free
    return 1
  fi

  for corp in "${CORPORA[@]}"; do
    local out="$RUNS/summ_${tag}_${conf}_${method}_${corp}.json"
    if [ -f "$out" ]; then
      log "  SKIP corpus=$corp (exists)"; continue
    fi
    local tput_log="$LOGD/${tag}_${conf}_${method}_${corp}_tput.log"
    log "  bench corp=$corp ${LIMIT}p × conc=$CONC × max=$MAXTOK"
    local corpus_flag=""
    if [ "$corp" != "mix" ]; then
      corpus_flag="--corpus $corp"
    fi
    PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "$method" \
      --model "$model" --model-tag "$tag" --port 8001 --max-tokens "$MAXTOK" \
      --concurrency "$CONC" --limit "$LIMIT" --shuffle \
      $corpus_flag \
      --out "$out" --raw "$RAW" >> "$tput_log" 2>&1 || log "  bench fail corp=$corp"
    local tps; tps=$("$PY" -c "import json;d=json.load(open('$out'));print('tps',d['output_tps'],'n_ok',d['n_ok'],'wall',d['wall_total_s'])" 2>/dev/null || echo ERR)
    log "  → $corp: $tps"
  done
  kill_pgroup "$pid"
  wait_gpu_free
}

log "=== SUB_201 Optimal vs MWOR TP=8 sweep start → $RUNS ==="
log "common env: VLLM_PREFETCH_TOKENIZE=$VLLM_PREFETCH_TOKENIZE WORKERS=$VLLM_PREFETCH_TOKENIZE_WORKERS BURST=$VLLM_BURST_AWARE_ADMISSION"
log "common args: FaP cudagraph_mode (always on), conc=$CONC max_tok=$MAXTOK limit=$LIMIT mml=$MML"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r model tag fam tp gpus optm timeout mwor_methods <<< "$entry"
  log ""
  log "############## $tag (fam=$fam tp=$tp gpus=$gpus) ##############"

  # Optimal Config: vanilla method, 7 corpus all in 1 boot
  run_method_boot "$model" "$tag" "$tp" "$gpus" "$optm" "$timeout" "optimal"

  # MWOR: each unique winner method (vanilla 제외) 별 1 boot, 7 corpus 측정
  IFS=',' read -ra mwor_list <<< "$mwor_methods"
  for mw in "${mwor_list[@]}"; do
    [ -z "$mw" ] && continue
    if [ "$mw" = "$optm" ]; then
      log "  MWOR method '$mw' == Optimal — skip (reuse optimal)"; continue
    fi
    run_method_boot "$model" "$tag" "$tp" "$gpus" "$mw" "$timeout" "mwor"
  done
done

log "=== sweep DONE → $RUNS ==="
