#!/usr/bin/env bash
# IDE_062: 저부하 (도착률 1.5) 토큰 지연이 기준선의 1.75배인 원인 특정.
# 가설: attention 비용이 할당된 KV 풀 크기에 비례하고, 소배치에서만 드러난다.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide062_lowload_tpot; mkdir -p $B
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
ENVC="SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 KT_AVX_RB=1 KT_AVX_PF=2 KT_FUSE_QIN=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25"
COMMON="--model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --mem-fraction-static 0.94 --chunked-prefill-size 4096"
G224="--cuda-graph-max-bs 224 --cuda-graph-bs 32 64 96 128 160 192 224"
G96="--cuda-graph-max-bs 96 --cuda-graph-bs 32 64 96"

boot() { local tag=$1; shift
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
  docker exec -d sgl-kt bash -c "$ENVC CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server $COMMON $* > /tmp/sgl_L_$tag.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && break
    docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || { echo "$tag BOOT_FAIL" | tee -a $B/RUN.log; return 1; }; sleep 10; i=$((i+10)); done
  ((i>=1500)) && { echo "$tag BOOT_TIMEOUT" | tee -a $B/RUN.log; return 1; }
  docker exec sgl-kt bash /tmp/pin_nonkt.sh >/dev/null 2>&1
  echo "$tag BOOT_OK ${i}s kv=$(docker exec sgl-kt grep -m1 -o '#tokens: [0-9]*' /tmp/sgl_L_$tag.log | grep -o '[0-9]*')" | tee -a $B/RUN.log; return 0; }

low() { local tag=$1; local rf=/tmp/l_${tag}.log
  curl -s -X POST http://127.0.0.1:30000/flush_cache -o /dev/null; sleep 3
  docker exec -d vllm-h100 bash -c "timeout 900 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 450 --request-rate 1.5 --seed 42 --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  sleep 60; W=90 bash $SP/dram_sample.sh 90 1 $B/dram_${tag}.txt
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>930)) && break; done
  local out=$B/low_${tag}.log; docker cp vllm-h100:$rf $out 2>/dev/null
  echo "$tag lam1.5: tok/s $(grep -h 'Output token throughput' $out|awk '{print $NF}'), TPOT $(grep -h 'Mean TPOT' $out|awk '{print $NF}')/p99 $(grep -h 'P99 TPOT' $out|awk '{print $NF}'), ITL p50 $(grep -h 'Median ITL' $out|awk '{print $NF}'), TTFT p50 $(grep -h 'Median TTFT' $out|awk '{print $NF}'), ok $(grep -h 'Successful requests' $out|awk '{print $NF}')  $(tail -1 $B/dram_${tag}.txt)" | tee -a $B/RUN.log; }

echo "== IDE_062 저부하 TPOT 원인 특정 $(date +%H:%M) ==" | tee -a $B/RUN.log
echo "# 기준: TUNOPS(bf16,graph96,풀63654) 61.96 ms / EPCORE(fp8,graph224,풀127309) 123.31 ms" | tee -a $B/RUN.log
boot D0 --kv-cache-dtype fp8_e5m2 --max-total-tokens 143360 $G224                  && low D0   # EPOCH 재확인 (mixed off)
boot D1 --kv-cache-dtype fp8_e5m2 --max-total-tokens 63654  $G224                  && low D1   # 풀만 기준선 수준
boot D2 --max-total-tokens 63654 $G224                                             && low D2   # bf16 + graph224
boot D3 --kv-cache-dtype fp8_e5m2 --max-total-tokens 143360 $G96                   && low D3   # fp8 + graph96
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "IDE062_DONE $B"
