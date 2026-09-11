#!/usr/bin/env bash
# EXP-E11 1단계: chunk 항 + eager 벌점(rho) 보정. 후보 공간 (graph_max 160/192/224) 밖인 graph_max 96 에서만 수행해
# held-out 오염을 피한다. chunk 4096 vs 8192 를 같은 조건에서 비교해 P(chunk) 를 만든다.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
REPO=$HOME/projects/vllm_hybrid
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_e11_stage1_chunkcal; mkdir -p $B
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
ENVC="SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 KT_AVX_RB=1 KT_AVX_PF=2 KT_FUSE_QIN=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25"
COMMON="--model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --mem-fraction-static 0.94 --kv-cache-dtype fp8_e5m2 --cuda-graph-max-bs 96 --cuda-graph-bs 32 64 96"

boot() { local tag=$1; shift
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
  docker exec -d sgl-kt bash -c "$ENVC CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server $COMMON $* > /tmp/sgl_C1_$tag.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && break
    docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || { echo "$tag BOOT_FAIL" | tee -a $B/RUN.log; return 1; }; sleep 10; i=$((i+10)); done
  ((i>=1500)) && { echo "$tag BOOT_TIMEOUT" | tee -a $B/RUN.log; return 1; }
  docker exec sgl-kt bash /tmp/pin_nonkt.sh >/dev/null 2>&1
  echo "$tag BOOT_OK ${i}s kv=$(docker exec sgl-kt grep -m1 -o '#tokens: [0-9]*' /tmp/sgl_C1_$tag.log | grep -o '[0-9]*')" | tee -a $B/RUN.log; return 0; }

bench() { local tag=$1 C=$2 inlen=$3 outlen=$4 NP=$5; local rf=/tmp/c1_${tag}_C${C}_${inlen}x${outlen}.log
  curl -s -X POST http://127.0.0.1:30000/flush_cache -o /dev/null; sleep 3
  docker exec -d vllm-h100 bash -c "timeout 1800 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len $inlen --sonnet-output-len $outlen --sonnet-prefix-len 100 --num-prompts $NP --max-concurrency $C --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95,99 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>1830)) && break; done
  local out=$B/$(basename $rf .log).log; docker cp vllm-h100:$rf $out 2>/dev/null
  echo "$tag C$C ${inlen}x${outlen}: tok/s $(grep -h 'Output token throughput' $out|awk '{print $NF}'), dur $(grep -h 'Benchmark duration' $out|awk '{print $NF}')s, TPOT $(grep -h 'Mean TPOT' $out|awk '{print $NF}')/p99 $(grep -h 'P99 TPOT' $out|awk '{print $NF}'), TTFT p50 $(grep -h 'Median TTFT' $out|awk '{print $NF}'), ok $(grep -h 'Successful requests' $out|awk '{print $NF}')" | tee -a $B/RUN.log; }

echo "== E11 1단계 chunk 보정 $(date +%H:%M) ==" | tee -a $B/RUN.log
for CH in 4096 8192; do
  boot CH$CH --chunked-prefill-size $CH && {
    bench CH$CH 64 512 128 512
    bench CH$CH 96 512 128 512
    bench CH$CH 128 512 128 512   # B > graph_max(96) -> eager 벌점 rho 보정
    bench CH$CH 32 4096 128 128
    bench CH$CH 64 4096 128 192
  }
done
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "E11_S1_DONE $B"
