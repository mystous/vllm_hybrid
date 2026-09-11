#!/usr/bin/env bash
# IDE_063: EPOCH-OPS 의 NaN 사망 원인 분리. 각 셀 도착률 5.0 으로 3,000요청 (약 10분) 지속.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide063_nan_isolation; mkdir -p $B
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
BASE="SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 KT_AVX_RB=1 KT_AVX_PF=2 KT_FUSE_QIN=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1"
COMMON="--model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --cuda-graph-max-bs 224 --cuda-graph-bs 32 64 96 128 160 192 224"

boot() { local tag=$1 env=$2; shift 2
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
  docker exec -d sgl-kt bash -c "$BASE $env CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server $COMMON $* > /tmp/sgl_N_$tag.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && break
    docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || { echo "$tag BOOT_FAIL" | tee -a $B/RUN.log; return 1; }; sleep 10; i=$((i+10)); done
  ((i>=1500)) && { echo "$tag BOOT_TIMEOUT" | tee -a $B/RUN.log; return 1; }
  docker exec sgl-kt bash /tmp/pin_nonkt.sh >/dev/null 2>&1
  echo "$tag BOOT_OK ${i}s kv=$(docker exec sgl-kt grep -m1 -o '#tokens: [0-9]*' /tmp/sgl_N_$tag.log | grep -o '[0-9]*')  env=[$env] args=[$*]" | tee -a $B/RUN.log; return 0; }

stress() { local tag=$1; local rf=/tmp/n_${tag}.log
  curl -s -X POST http://127.0.0.1:30000/flush_cache -o /dev/null; sleep 3
  docker exec -d vllm-h100 bash -c "timeout 1200 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 3000 --request-rate 5.0 --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95,99 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0 dead=0
  until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do
    sleep 15; w=$((w+15))
    if ! curl -sf -m 5 http://127.0.0.1:30000/health >/dev/null 2>&1; then
      docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null 2>&1 || { dead=$w; echo "$tag 서버 사망 감지 ${w}s 지점" | tee -a $B/RUN.log; break; }
    fi
    ((w>1230)) && break
  done
  sleep 5
  docker cp vllm-h100:$rf $B/stress_${tag}.log 2>/dev/null
  docker cp sgl-kt:/tmp/sgl_N_$tag.log $B/server_${tag}.log 2>/dev/null
  local nan=$(grep -c "probability tensor contains" $B/server_${tag}.log 2>/dev/null || echo 0)
  local ok=$(grep -h 'Successful requests' $B/stress_${tag}.log 2>/dev/null | awk '{print $NF}')
  local tps=$(grep -h 'Output token throughput' $B/stress_${tag}.log 2>/dev/null | awk '{print $NF}')
  local lastrun=$(grep -o '#running-req: [0-9]*' $B/server_${tag}.log 2>/dev/null | tail -1 | awk '{print $2}')
  echo "$tag 결과: NaN assert ${nan}건, 사망시점 ${dead}s (0=생존), 성공 ${ok:-?}/3000, tok/s ${tps:-?}, 마지막 실행중요청 ${lastrun:-?}" | tee -a $B/RUN.log; }

echo "== IDE_063 NaN 원인 분리 $(date +%H:%M) — 각 셀 λ5.0 x 3000요청 ==" | tee -a $B/RUN.log
boot N1 "KT_COLD_DEFER=1 KT_COLD_TAU=0.25" --kv-cache-dtype fp8_e5m2 --max-total-tokens 143360 --enable-mixed-chunk && stress N1
boot N2 ""                                 --kv-cache-dtype fp8_e5m2 --max-total-tokens 143360 --enable-mixed-chunk && stress N2
boot N3 "KT_COLD_DEFER=1 KT_COLD_TAU=0.25"                                                     --enable-mixed-chunk && stress N3
boot N4 "KT_COLD_DEFER=1 KT_COLD_TAU=0.25" --kv-cache-dtype fp8_e5m2 --max-total-tokens 143360                      && stress N4
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "IDE063_DONE $B"
