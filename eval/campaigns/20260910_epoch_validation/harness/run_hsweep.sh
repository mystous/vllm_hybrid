#!/usr/bin/env bash
# IDE_061: hot expert 수 H 재탐색. H 를 줄이면 (a) GPU 가중치 메모리가 줄어 KV 풀이 커지고
# (b) CPU 가 더 많은 expert 를 맡는다. 물리 DRAM 여유가 60% 남아 있으므로 (b) 가 가능한지 본다.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide061_hsweep; mkdir -p $B
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
ENVC="SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 KT_AVX_RB=1 KT_AVX_PF=2 KT_FUSE_QIN=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25"
COMMON="--model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --init-expert-location /tmp/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --kv-cache-dtype fp8_e5m2 --cuda-graph-max-bs 224 --cuda-graph-bs 32 64 96 128 160 192 224"

boot() { local H=$1
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
  docker exec -d sgl-kt bash -c "$ENVC CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server $COMMON --kt-num-gpu-experts $H --max-total-tokens 143360 > /tmp/sgl_H_$H.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && break
    docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || { echo "H$H BOOT_FAIL $(docker exec sgl-kt grep -m1 -o 'OOM on device.*\|torch.OutOfMemoryError.*' /tmp/sgl_H_$H.log | cut -c1-100)" | tee -a $B/RUN.log; return 1; }; sleep 10; i=$((i+10)); done
  ((i>=1500)) && { echo "H$H BOOT_TIMEOUT" | tee -a $B/RUN.log; return 1; }
  docker exec sgl-kt bash /tmp/pin_nonkt.sh >/dev/null 2>&1
  docker cp sgl-kt:/tmp/sgl_H_$H.log $B/server_H$H.log 2>/dev/null
  echo "H$H BOOT_OK ${i}s kv=$(docker exec sgl-kt grep -m1 -o '#tokens: [0-9]*' /tmp/sgl_H_$H.log | grep -o '[0-9]*') gpu_mem=[$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')]" | tee -a $B/RUN.log; return 0; }

bench() { local H=$1 C=$2 NP=$3 dram=$4; local rf=/tmp/h_${H}_C${C}.log
  curl -s -X POST http://127.0.0.1:30000/flush_cache -o /dev/null; sleep 3
  docker exec -d vllm-h100 bash -c "timeout 1800 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $NP --max-concurrency $C --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95,99 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  [ "$dram" = "1" ] && { sleep 25; W=60 bash $SP/dram_sample.sh 60 1 $B/dram_H${H}_C${C}.txt; }
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>1830)) && break; done
  local out=$B/h${H}_C${C}.log; docker cp vllm-h100:$rf $out 2>/dev/null
  echo "H$H C$C: tok/s $(grep -h 'Output token throughput' $out|awk '{print $NF}'), dur $(grep -h 'Benchmark duration' $out|awk '{print $NF}')s, TPOT $(grep -h 'Mean TPOT' $out|awk '{print $NF}')/p99 $(grep -h 'P99 TPOT' $out|awk '{print $NF}'), TTFT p50 $(grep -h 'Median TTFT' $out|awk '{print $NF}'), ok $(grep -h 'Successful requests' $out|awk '{print $NF}')  $([ -f $B/dram_H${H}_C${C}.txt ] && tail -1 $B/dram_H${H}_C${C}.txt)" | tee -a $B/RUN.log; }

echo "== IDE_061 hot expert 수 H 재탐색 $(date +%H:%M) ==" | tee -a $B/RUN.log
for H in 96 80 64 112; do
  boot $H && { bench $H 64 512 0; bench $H 224 512 1; }
done
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "HSWEEP_DONE $B"
