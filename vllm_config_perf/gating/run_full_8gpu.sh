#!/usr/bin/env bash
# 수정 재실험 (모든 config 가 8 GPU 사용):
#   Phase 1 vanilla-only : TP=8 단일 (GPU0-7, :8001, spec OFF)
#   Phase 2 trident-only : TP=8 단일 (GPU0-7, :8002, suffix K=32 + FULL_AND_PIECEWISE)  # B200 default (b3_sched DECISION)
#   Phase 3 AGSD         : TP=4×2 (vanilla GPU0-3 :8001 + trident GPU4-7 :8002) + router :8000
# 공통: 500p × 8192in × 8192out × 1-run, max-model-len 20480 (400 에러 회피)
set -uo pipefail
cd /workspace/host_vllm_hybrid
PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
RD=/workspace/host_vllm_hybrid/vllm_config_perf/gating/runs/agsd_32b_b200_v2_20260601
LOGD=/tmp/agsd_v2_logs
mkdir -p "$RD" "$LOGD"
MODEL=Qwen/Qwen2.5-32B-Instruct
MML=20480
NP=500; TIN=8192; TOUT=8192; CONC=32
WORKLOADS="sonnet chat code balanced sonnet-heavy code-heavy"

export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

wait_ready(){ # $1=url
  for i in $(seq 1 240); do
    curl -sf "$1/v1/models" >/dev/null 2>&1 && { log "READY $1 (${i}x5s)"; return 0; }
    sleep 5
  done
  log "TIMEOUT $1"; return 1
}

kill_pgroup(){ # $1=leader pid
  local pid=$1
  [ -z "$pid" ] && return 0
  local pgid; pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ -n "$pgid" ] && kill -9 -"$pgid" 2>/dev/null
  kill -9 "$pid" 2>/dev/null
}

wait_gpu_free(){ # GPU mem 다 빠질 때까지 (최대 ~2분)
  for i in $(seq 1 24); do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
    [ "${used:-1}" -lt 4000 ] && { log "GPU freed"; return 0; }
    sleep 5
  done
  log "GPU not fully freed (계속)"; return 0
}

run_cells(){ # $1=scenario
  local sc=$1
  for wl in $WORKLOADS; do
    local out="$RD/${sc}__${wl}.json"
    [ -f "$out" ] && { log "skip $sc×$wl (exists)"; continue; }
    log "==== $sc × $wl ===="
    PYTHONPATH=/workspace/host_vllm_hybrid "$PY" vllm_config_perf/gating/benchmark_workloads.py \
      --scenario "$sc" --workload "$wl" --num-prompts "$NP" \
      --target-input-len "$TIN" --max-tokens "$TOUT" --concurrency "$CONC" \
      --out "$out" >> "$RD/sweep.log" 2>&1
    local tps; tps=$("$PY" -c "import json;d=json.load(open('$out'))['summary'];print('tps',round(d['output_tps'],1),'n_ok',d['n_ok'])" 2>/dev/null || echo ERR)
    log "     → $tps"
  done
}

start_one(){ # $1=port $2=gpus $3=gmu $4=extra-args  → echo pid
  local port=$1 gpus=$2 gmu=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpus setsid "$VBIN" serve "$MODEL" \
    --tensor-parallel-size $(echo "$gpus"|tr ',' '\n'|wc -l) --port "$port" \
    --gpu-memory-utilization "$gmu" --max-model-len "$MML" \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' "$@" \
    > "$LOGD/p${port}.log" 2>&1 < /dev/null &
  echo $!
}

log "=== FULL 8-GPU re-run start → $RD ==="

# ---------- Phase 1: vanilla-only TP=8 ----------
log "### PHASE 1 vanilla-only TP=8 (GPU0-7 :8001) ###"
PID_V=$(start_one 8001 0,1,2,3,4,5,6,7 0.85)
wait_ready http://127.0.0.1:8001 && run_cells vanilla
kill_pgroup "$PID_V"; wait_gpu_free

# ---------- Phase 2: trident-only TP=8 ----------
log "### PHASE 2 trident-only TP=8 (GPU0-7 :8002, suffix) ###"
PID_T=$(start_one 8002 0,1,2,3,4,5,6,7 0.80 --speculative-config '{"method":"suffix","num_speculative_tokens":32}')
wait_ready http://127.0.0.1:8002 && run_cells trident
kill_pgroup "$PID_T"; wait_gpu_free

# ---------- Phase 3: AGSD TP=4×2 + router ----------
log "### PHASE 3 AGSD TP=4×2 (vanilla GPU0-3 :8001 + trident GPU4-7 :8002) + router :8000 ###"
PID_V2=$(start_one 8001 0,1,2,3 0.85)
PID_T2=$(start_one 8002 4,5,6,7 0.80 --speculative-config '{"method":"suffix","num_speculative_tokens":32}')
wait_ready http://127.0.0.1:8001; wait_ready http://127.0.0.1:8002
AGSD_VANILLA_URL="http://127.0.0.1:8001/v1" AGSD_TRIDENT_URL="http://127.0.0.1:8002/v1" \
  AGSD_CHAT_BACKEND=vanilla PYTHONPATH=/workspace/host_vllm_hybrid \
  setsid "$PY" -m uvicorn vllm_config_perf.gating.agsd_router:app \
  --host 0.0.0.0 --port 8000 --loop uvloop --workers 1 > "$LOGD/router.log" 2>&1 < /dev/null &
PID_R=$!
sleep 5
run_cells AGSD
kill_pgroup "$PID_V2"; kill_pgroup "$PID_T2"; kill_pgroup "$PID_R"; wait_gpu_free

log "=== ALL DONE → $RD ==="
