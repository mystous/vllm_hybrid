#!/usr/bin/env bash
# B0 측정#1 (DP-attention+EP config): TP無 → attention DP(all-reduce 없음), experts EP(all-to-all 지배).
# 축소 top-k가 지배 통신(EP all-to-all)을 줄이는가. 인자: <force_topk: 0|N> <out_tag>
set -uo pipefail; set +B
DIR=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_260_reduced_expert_selfspec
BENCH=$DIR/../SUB_258_dram_kv_offload/bench_struct.py
PY=/home/mystous/vllm_dev_prj/bin/python; NSYS=/usr/local/cuda/bin/nsys
MODEL=deepseek-ai/DeepSeek-R1; LOGD=$DIR/runs; mkdir -p $LOGD
FORCE=${1:-0}; TAG=${2:-dp_top8}; PORT=$((8120+FORCE))
export HF_HOME=/raid/hf_cache VLLM_USE_DEEP_GEMM=1
[ "$FORCE" -gt 0 ] && export VLLM_MOE_FORCE_TOPK=$FORCE || true
# 부하 워처: 높은 동시성으로 8 DP replica 채움
( for i in $(seq 1 60); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 5; done
  echo "[load] ready $(date -u +%H:%M:%S)"
  for r in $(seq 1 6); do
    $PY $BENCH --port $PORT --model $MODEL --n 128 --concurrency 64 --label load >/dev/null 2>&1
  done
) &
LOADER=$!
echo "[nsys-DP] FORCE_TOPK=$FORCE tag=$TAG port=$PORT $(date -u +%H:%M:%S)"
$NSYS profile --trace=cuda,nvtx --delay=140 --duration=25 --sample=none --cpuctxsw=none \
  --output=$LOGD/$TAG --force-overwrite=true \
  $PY -m vllm.entrypoints.openai.api_server --model $MODEL \
    --data-parallel-size 8 --enable-expert-parallel --enforce-eager \
    --gpu-memory-utilization 0.90 --max-model-len 4096 --port $PORT \
    > $LOGD/nsys_$TAG.log 2>&1 &
prev=-1
for i in $(seq 1 70); do
  if [ -f $LOGD/$TAG.nsys-rep ]; then sz=$(stat -c %s $LOGD/$TAG.nsys-rep 2>/dev/null||echo 0); [ "$sz" = "$prev" ] && [ "$sz" -gt 1000 ] && { echo "[nsys] rep 안정 $sz"; break; }; prev=$sz; fi
  sleep 6
done
sleep 3
for p in $(grep -ohE "pid=[0-9]+" $LOGD/nsys_$TAG.log 2>/dev/null | sed 's/pid=//' | sort -un); do kill $p 2>/dev/null; done
kill $LOADER 2>/dev/null; sleep 8
echo "===== GPU 커널 요약 ($TAG) ====="
$NSYS stats --report cuda_gpu_kern_sum --format table $LOGD/$TAG.nsys-rep 2>/dev/null | head -38
echo "[done $TAG]"