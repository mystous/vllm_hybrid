#!/usr/bin/env bash
# EXP-E11 3단계: held-out 워크로드에서 12개 후보 전수 측정 (oracle) -> selection_regret 산정.
# 후보 = graph_max {160,192,224} x 버킷 간격 {16,32} x chunk {4096,8192}. 나머지는 고정
# (hot-96 a0.25, fp8 KV, Dtau, mixed off, cpuinfer 96, 전체 실행 패치).
# held-out 워크로드 (보정에 없는 조건: random 데이터셋 + 새 길이 조합):
#   WH1 = random 512/512,  C=168  (B_cap~180. 168 은 어느 간격에서도 버킷이 아니라 padding 이 갈리고,
#                                  graph_max 160 후보에서는 graph 를 벗어나 eager 가 된다)
#   WH2 = random 3072/256, C=40   (B_cap~43. prefill 이 토큰의 92% -> chunk 가 드러난다)
# 예측은 이 스크립트 실행 전에 predictions.json 으로 커밋되어 있어야 한다.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
REPO=$HOME/projects/vllm_hybrid
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_e11_stage3_oracle; mkdir -p $B
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
ENVC="SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 KT_AVX_RB=1 KT_AVX_PF=2 KT_FUSE_QIN=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25"
COMMON="--model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --mem-fraction-static 0.94 --kv-cache-dtype fp8_e5m2 --max-total-tokens 143360"

buckets() { local G=$1 s=$2 out=""; local b=$s
  while ((b<=G)); do out="$out $b"; b=$((b+s)); done
  [[ " $out " == *" $G "* ]] || out="$out $G"; echo $out; }

boot() { local tag=$1; shift
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
  docker exec -d sgl-kt bash -c "$ENVC CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server $COMMON $* > /tmp/sgl_O_$tag.log 2>&1"
  local i=0; while ((i<1800)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && break
    docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || { echo "$tag BOOT_FAIL $(docker exec sgl-kt grep -m1 -o 'OOM on device.*\|torch.OutOfMemoryError.*' /tmp/sgl_O_$tag.log | cut -c1-100)" | tee -a $B/RUN.log; return 1; }; sleep 10; i=$((i+10)); done
  ((i>=1800)) && { echo "$tag BOOT_TIMEOUT" | tee -a $B/RUN.log; return 1; }
  docker exec sgl-kt bash /tmp/pin_nonkt.sh >/dev/null 2>&1
  echo "$tag BOOT_OK ${i}s kv=$(docker exec sgl-kt grep -m1 -o '#tokens: [0-9]*' /tmp/sgl_O_$tag.log | grep -o '[0-9]*')" | tee -a $B/RUN.log; return 0; }

bench() { local tag=$1 wl=$2 C=$3 inlen=$4 outlen=$5 NP=$6; local rf=/tmp/o_${tag}_${wl}.log
  curl -s -X POST http://127.0.0.1:30000/flush_cache -o /dev/null; sleep 3
  docker exec -d vllm-h100 bash -c "timeout 2400 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name random --random-input-len $inlen --random-output-len $outlen --random-range-ratio 0.0 --ignore-eos --num-prompts $NP --max-concurrency $C --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95,99 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 15; w=$((w+15)); ((w>2430)) && break; done
  local out=$B/o_${tag}_${wl}.log; docker cp vllm-h100:$rf $out 2>/dev/null
  echo "ORACLE $tag $wl (C$C ${inlen}x${outlen}): tok/s $(grep -h 'Output token throughput' $out|awk '{print $NF}'), dur $(grep -h 'Benchmark duration' $out|awk '{print $NF}')s, TPOT $(grep -h 'Mean TPOT' $out|awk '{print $NF}')/p99 $(grep -h 'P99 TPOT' $out|awk '{print $NF}'), TTFT p50 $(grep -h 'Median TTFT' $out|awk '{print $NF}')/p99 $(grep -h 'P99 TTFT' $out|awk '{print $NF}'), ok $(grep -h 'Successful requests' $out|awk '{print $NF}')" | tee -a $B/RUN.log; }

echo "== E11 3단계 oracle (12 후보 x held-out 2) $(date +%H:%M) ==" | tee -a $B/RUN.log
for G in 160 192 224; do for S in 32 16; do for CH in 4096 8192; do
  TAG="G${G}s${S}c${CH}"
  BK=$(buckets $G $S)
  echo "--- 후보 $TAG : buckets [$BK] chunk $CH ---" | tee -a $B/RUN.log
  boot $TAG --cuda-graph-max-bs $G --cuda-graph-bs $BK --chunked-prefill-size $CH && {
    bench $TAG WH1 168 512 512 504
    bench $TAG WH2  40 3072 256 120
  }
done; done; done
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "E11_S3_DONE $B"
