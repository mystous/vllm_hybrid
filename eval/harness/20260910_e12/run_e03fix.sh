#!/usr/bin/env bash
# EXP-E03 보정: 고정 동시성에서의 물리 DRAM 을 창이 벤치 안쪽에 들어가도록 다시 잰다.
# 각 구성 자기 최적 동시성에서 2048요청 (250초 이상) 을 돌리고 60~120초 구간만 계측.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_e03fix_dram_fixedC; mkdir -p $B
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
ENVC="SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 KT_AVX_RB=1 KT_AVX_PF=2 KT_FUSE_QIN=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25"
COMMON="--model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --mem-fraction-static 0.94 --chunked-prefill-size 4096"
TUN="--cuda-graph-max-bs 96 --cuda-graph-bs 32 64 96"
EPB="--kv-cache-dtype fp8_e5m2 --max-total-tokens 143360 --cuda-graph-max-bs 224 --cuda-graph-bs 32 64 96 128 160 192 224"

boot() { local tag=$1; shift
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
  docker exec -d sgl-kt bash -c "$ENVC CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server $COMMON $* > /tmp/sgl_F2_$tag.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && break
    docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || { echo "$tag BOOT_FAIL" | tee -a $B/RUN.log; return 1; }; sleep 10; i=$((i+10)); done
  ((i>=1500)) && { echo "$tag BOOT_TIMEOUT" | tee -a $B/RUN.log; return 1; }
  docker exec sgl-kt bash /tmp/pin_nonkt.sh >/dev/null 2>&1
  echo "$tag BOOT_OK ${i}s kv=$(docker exec sgl-kt grep -m1 -o '#tokens: [0-9]*' /tmp/sgl_F2_$tag.log | grep -o '[0-9]*')" | tee -a $B/RUN.log; return 0; }

meas() { local tag=$1 C=$2 NP=$3; local rf=/tmp/f2_${tag}_C${C}.log
  curl -s -X POST http://127.0.0.1:30000/flush_cache -o /dev/null; sleep 3
  docker exec -d vllm-h100 bash -c "timeout 1800 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $NP --max-concurrency $C --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95,99 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  sleep 60; W=60 bash $SP/dram_sample.sh 60 1 $B/dram_${tag}_C${C}.txt
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>1830)) && break; done
  local out=$B/bench_${tag}_C${C}.log; docker cp vllm-h100:$rf $out 2>/dev/null
  local dur=$(grep -h 'Benchmark duration' $out|awk '{print $NF}')
  echo "$tag C$C: tok/s $(grep -h 'Output token throughput' $out|awk '{print $NF}'), dur ${dur}s (창 60~120s), TPOT $(grep -h 'Mean TPOT' $out|awk '{print $NF}'), ok $(grep -h 'Successful requests' $out|awk '{print $NF}')" | tee -a $B/RUN.log
  echo "   DRAM $(tail -1 $B/dram_${tag}_C${C}.txt)" | tee -a $B/RUN.log; }

echo "== EXP-E03 보정: 고정 동시성 물리 DRAM (창을 벤치 안쪽에) $(date +%H:%M) ==" | tee -a $B/RUN.log
boot TUNOPS $TUN --enable-mixed-chunk && { meas TUNOPS 96 2048; meas TUNOPS 64 2048; }
boot EPCORE $EPB                      && { meas EPCORE 224 2048; meas EPCORE 64 2048; }
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "E03FIX_DONE $B"
