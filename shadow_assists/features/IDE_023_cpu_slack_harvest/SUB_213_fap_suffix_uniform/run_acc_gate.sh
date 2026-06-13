#!/usr/bin/env bash
set -u
cd /home/mystous/vllm_hybrid
PY=/home/mystous/vllm_dev_prj/bin/python
VBIN=/home/mystous/vllm_dev_prj/bin/vllm
B=shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_213_fap_suffix_uniform
OUT=$B/runs_acc_gate; mkdir -p $OUT
MODEL="meta-llama/Llama-3.1-70B-Instruct"; PORT=8011
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export HF_HOME=/raid/hf_cache HF_HUB_OFFLINE=1
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
boot(){
    env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $1 setsid taskset -c 0-47,56-103 "$VBIN" serve "$MODEL" \
        --tensor-parallel-size 8 --port $PORT --gpu-memory-utilization 0.85 --max-model-len 16384 \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
        --speculative-config '{"method":"suffix","num_speculative_tokens":6}' \
        > "$OUT/boot_$2.log" 2>&1 < /dev/null &
    PID=$!
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
        kill -0 $PID 2>/dev/null || return 1; sleep 5
    done; return 1
}
kill_all(){ pg=$(ps -o pgid= -p ${PID:-0} 2>/dev/null | tr -d ' '); [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 "$p" 2>/dev/null; done
    sleep 8; }

log "=== 정확도 게이트: A (no-pad) ==="
boot "" A || { log "BOOT A FAIL"; exit 1; }
PYTHONPATH=. $PY $B/accuracy_gate.py collect --port $PORT --model $MODEL --n 32 --max-tokens 128 --out $OUT/A_nopad.jsonl
kill_all
log "=== B (PAD_UNIFORM) ==="
boot "VLLM_SUFFIX_PAD_UNIFORM=1" B || { log "BOOT B FAIL"; exit 1; }
PYTHONPATH=. $PY $B/accuracy_gate.py collect --port $PORT --model $MODEL --n 32 --max-tokens 128 --out $OUT/B_pad.jsonl
kill_all
log "=== compare ==="
$PY $B/accuracy_gate.py compare $OUT/A_nopad.jsonl $OUT/B_pad.jsonl | tee $OUT/verdict.txt
log "=== acc gate done ==="
