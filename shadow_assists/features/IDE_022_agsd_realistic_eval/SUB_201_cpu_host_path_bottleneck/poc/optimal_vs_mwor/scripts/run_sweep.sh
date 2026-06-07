#!/usr/bin/env bash
# SUB_201 Optimal Config vs MWOR — 10 model × mix corpus, B200 8 GPU.
# 비교 4 config (cell 별):
#   - Vanilla / llm-d : TSK_042 (runs/tput_t1t3_20260602) 재사용 (별도 run 안 함)
#   - Optimal Config  : vanilla method + FaP + L2 + L10  (신규 측정)
#   - MWOR (Oracle)   : oracle_table.csv 의 (model_family, mix) winner method + FaP + L2 + L10  (신규 측정)
#
# 본 sweep 은 신규 측정만 수행:
#   conf = optimal  (vanilla method)
#   conf = mwor     (winner method per model — vanilla → optimal 와 동일이라 skip 처리)
#
# 동일 setting 으로 TSK_042 와 비교 가능하도록: conc=32, max_tok=8192, 500p mix, MML=16384
set -uo pipefail
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
ROOT=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/optimal_vs_mwor
RUNS="$ROOT/runs"
LOGD="$ROOT/logs"
mkdir -p "$RUNS" "$LOGD"
RAW="$RUNS/per_request_raw.jsonl"
[ -f "$RAW" ] || : > "$RAW"

SAMPLED=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet
RE=vllm_config_perf/gating/realistic_eval

# common args
CONC=32
MAXTOK=8192
LIMIT=500
MML=16384

# common env (L2 + L10 always-on for both optimal + mwor)
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

wait_ready(){ local url=$1 pid=$2 timeout_s=${3:-600}
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
  # orphan VLLM::Worker compute-apps kill
  sleep 2
  local cpids; cpids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  for cp in $cpids; do [ -n "$cp" ] && kill -9 "$cp" 2>/dev/null; done
}

wait_gpu_free(){ for i in $(seq 1 36); do
  local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}')
  [ "${u:-1}" -lt 4000 ] && { log "GPU freed"; return 0; }
  sleep 5
done; log "GPU not fully freed (계속)"; return 0; }

# (model_id, tag, tp, gpus, optimal_method, mwor_method, boot_timeout_s)
# mwor_method 는 oracle_table.csv 의 mix winner. vanilla winner 이면 MWOR == Optimal Config → skip 처리.
# llm-d winner 인 모델은 oracle_table mix 컬럼상 없음 (전부 vanilla/suffix).
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct|Qwen2.5-7B-Instruct|1|0|vanilla|suffix|900"
  "Qwen/Qwen2.5-32B-Instruct|Qwen2.5-32B-Instruct|2|0,1|vanilla|suffix|1200"
  "Qwen/Qwen2.5-72B-Instruct|Qwen2.5-72B-Instruct|4|0,1,2,3|vanilla|suffix|1800"
  "meta-llama/Llama-3.1-8B-Instruct|Llama-3.1-8B-Instruct|1|0|vanilla|suffix|900"
  "meta-llama/Llama-3.1-70B-Instruct|Llama-3.1-70B-Instruct|4|0,1,2,3|vanilla|suffix|1800"
  "meta-llama/Llama-3.1-405B-Instruct-FP8|Llama-3.1-405B-Instruct-FP8|8|0,1,2,3,4,5,6,7|vanilla|suffix|2400"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|DeepSeek-R1-Distill-Qwen-7B|1|0|vanilla|suffix|900"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|DeepSeek-R1-Distill-Qwen-32B|2|0,1|vanilla|suffix|1200"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B|DeepSeek-R1-Distill-Llama-70B|4|0,1,2,3|vanilla|suffix|1800"
  "deepseek-ai/DeepSeek-R1|DeepSeek-R1|8|0,1,2,3,4,5,6,7|vanilla|vanilla|3600"
)

spec_arg(){ case "$1" in
  vanilla) echo "";;
  suffix)  echo '--speculative-config {"method":"suffix","num_speculative_tokens":7}';;
  ngram)   echo '--speculative-config {"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":4}';;
  *) echo "";; esac; }

run_one_config(){ # $1=model $2=tag $3=tp $4=gpus $5=method $6=boot_timeout $7=conf_label
  local model=$1 tag=$2 tp=$3 gpus=$4 method=$5 timeout=$6 conf=$7
  local out="$RUNS/summ_${tag}_${conf}_${method}_mix.json"
  if [ -f "$out" ]; then log "SKIP $tag × $conf × $method (exists)"; return 0; fi
  local boot_log="$LOGD/${tag}_${conf}_${method}_boot.log"
  local tput_log="$LOGD/${tag}_${conf}_${method}_tput.log"
  local sa; sa=$(spec_arg "$method")
  log "==== boot $tag conf=$conf method=$method tp=$tp gpus=$gpus ===="
  CUDA_VISIBLE_DEVICES=$gpus setsid "$VBIN" serve "$model" \
    --tensor-parallel-size $tp --port 8001 --gpu-memory-utilization 0.85 \
    --max-model-len $MML --allow-deprecated-quantization \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    $sa > "$boot_log" 2>&1 < /dev/null &
  local pid=$!
  log "  PID=$pid (boot_log=$boot_log timeout=${timeout}s)"
  if wait_ready http://127.0.0.1:8001 "$pid" "$timeout"; then
    log "  bench mix 500p × conc=$CONC × max=$MAXTOK"
    PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "$method" \
      --model "$model" --model-tag "$tag" --port 8001 --max-tokens "$MAXTOK" \
      --concurrency "$CONC" --limit "$LIMIT" --shuffle \
      --out "$out" --raw "$RAW" >> "$tput_log" 2>&1
    local tps; tps=$("$PY" -c "import json;d=json.load(open('$out'));print('tps',d['output_tps'],'n_ok',d['n_ok'],'wall',d['wall_total_s'])" 2>/dev/null || echo ERR)
    log "  → $tps"
  else
    log "  BOOT FAIL $tag × $conf × $method"
    : > "$out.FAIL"
  fi
  kill_pgroup "$pid"
  wait_gpu_free
}

log "=== SUB_201 Optimal vs MWOR sweep start → $RUNS ==="
log "common env: VLLM_PREFETCH_TOKENIZE=$VLLM_PREFETCH_TOKENIZE WORKERS=$VLLM_PREFETCH_TOKENIZE_WORKERS BURST=$VLLM_BURST_AWARE_ADMISSION"
log "common args: FaP cudagraph_mode (always on)"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r model tag tp gpus optm mwor timeout <<< "$entry"
  log ""
  log "############## $tag (tp=$tp gpus=$gpus optm=$optm mwor=$mwor) ##############"
  # Optimal Config (vanilla method)
  run_one_config "$model" "$tag" "$tp" "$gpus" "$optm" "$timeout" "optimal"
  # MWOR (winner method) — if same as optimal, skip; else run
  if [ "$mwor" = "$optm" ]; then
    log "  MWOR == Optimal Config method (both '$optm') → MWOR 측정 결과는 optimal 동일 (per-cell winner 이 vanilla)"
  else
    run_one_config "$model" "$tag" "$tp" "$gpus" "$mwor" "$timeout" "mwor"
  fi
done

log "=== sweep DONE → $RUNS ==="
