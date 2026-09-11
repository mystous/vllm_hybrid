#!/usr/bin/env bash
# EXP-E03 물리 계측 (실행계획서 §9.4 미달 항목) + §14.1 동일작업 대조.
# perf uncore_imc 로 소켓별 물리 DRAM 바이트를 재고, 출력 토큰당 바이트를 구성별로 비교한다.
# 각 구성마다: (a) 고정 job C64 512요청 (동일 동시성·동일 작업) (b) open-loop lam=1.5 300초 (동일 도착열)
# 두 경우 모두 정상상태 안쪽 120초 창에서만 카운터를 읽는다.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_e03_physical_dram; mkdir -p $B
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
ENVC="SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 KT_AVX_RB=1 KT_AVX_PF=2 KT_FUSE_QIN=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25"
COMMON="--model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --mem-fraction-static 0.94 --chunked-prefill-size 4096"
TUN="--cuda-graph-max-bs 96 --cuda-graph-bs 32 64 96"
EPB="--kv-cache-dtype fp8_e5m2 --max-total-tokens 143360 --cuda-graph-max-bs 224 --cuda-graph-bs 32 64 96 128 160 192 224"

boot() { local tag=$1; shift
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
  docker exec -d sgl-kt bash -c "$ENVC CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server $COMMON $* > /tmp/sgl_D_$tag.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && break
    docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || { echo "$tag BOOT_FAIL" | tee -a $B/RUN.log; return 1; }; sleep 10; i=$((i+10)); done
  ((i>=1500)) && { echo "$tag BOOT_TIMEOUT" | tee -a $B/RUN.log; return 1; }
  docker exec sgl-kt bash /tmp/pin_nonkt.sh >/dev/null 2>&1
  echo "$tag BOOT_OK ${i}s kv=$(docker exec sgl-kt grep -m1 -o '#tokens: [0-9]*' /tmp/sgl_D_$tag.log | grep -o '[0-9]*')" | tee -a $B/RUN.log; return 0; }

# 유휴 기준선 (모델 상주 상태에서 요청 없음) — 배경 트래픽을 빼기 위해
idle() { local tag=$1; W=20 bash $SP/dram_sample.sh 20 1 $B/dram_${tag}_idle.txt
  echo "$tag IDLE $(tail -1 $B/dram_${tag}_idle.txt)" | tee -a $B/RUN.log; }

measure() { local tag=$1 kind=$2 C=$3 NP=$4 RATE=$5; local rf=/tmp/d_${tag}_${kind}.log
  curl -s -X POST http://127.0.0.1:30000/flush_cache -o /dev/null; sleep 3
  local CARG="--max-concurrency $C"; [ "$C" = "none" ] && CARG=""
  docker exec -d vllm-h100 bash -c "timeout 1800 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $NP $CARG --request-rate $RATE --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95,99 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  sleep 45                                  # 램프 통과
  W=120 bash $SP/dram_sample.sh 120 1 $B/dram_${tag}_${kind}.txt
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>1830)) && break; done
  local out=$B/bench_${tag}_${kind}.log; docker cp vllm-h100:$rf $out 2>/dev/null
  echo "$tag $kind (C=$C rate=$RATE NP=$NP): tok/s $(grep -h 'Output token throughput' $out|awk '{print $NF}'), dur $(grep -h 'Benchmark duration' $out|awk '{print $NF}')s, TPOT $(grep -h 'Mean TPOT' $out|awk '{print $NF}'), ok $(grep -h 'Successful requests' $out|awk '{print $NF}')" | tee -a $B/RUN.log
  echo "   DRAM $(tail -1 $B/dram_${tag}_${kind}.txt)" | tee -a $B/RUN.log; }

echo "== EXP-E03 물리 DRAM 계측 $(date +%H:%M) ==" | tee -a $B/RUN.log
echo "# perf $(sudo perf --version 2>&1 | head -1), uncore_imc 8채널 x 2소켓, cas_count_read/write x 64B" | tee -a $B/RUN.log

run_cfg() { local tag=$1; shift
  boot $tag "$@" || return 1
  idle $tag
  measure $tag fixedC64 64 512 inf
  measure $tag arrival15 none 450 1.5; }

run_cfg TUNOPS $TUN --enable-mixed-chunk
run_cfg EPCORE $EPB
run_cfg EPOPS  $EPB --enable-mixed-chunk

docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "E03PHYS_DONE $B"
