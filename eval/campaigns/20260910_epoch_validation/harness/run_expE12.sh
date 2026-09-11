#!/usr/bin/env bash
# EXP-E12 (장시간 안정성 + 부하 변화 시나리오) + 반복/신뢰구간 (5 seed).
# 구성 3종: TUNOPS(강한 기준선, bf16 KV + graph96 + mixed), EPCORE(EPOCH-CORE: fp8 KV + sparse buckets, mixed off), EPOPS(EPOCH + mixed).
# 각 구성: (a) 최적 C 에서 고정 512 job 을 seed 5개로 반복 -> 평균/표준편차/95% CI
#          (b) soak: open-loop 저부하(1.5) -> 용량 근접(구성별) -> 저부하(1.5), 각 1200s = 60분
#          soak 중 30초 간격으로 GPU 메모리 / #running-req / 서버 이상로그 샘플링
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_expE12_soak_reps; mkdir -p $B
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
ENVC="SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 KT_AVX_RB=1 KT_AVX_PF=2 KT_FUSE_QIN=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25"
COMMON="--model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --mem-fraction-static 0.94 --chunked-prefill-size 4096"
TUN="--cuda-graph-max-bs 96 --cuda-graph-bs 32 64 96"
EPB="--kv-cache-dtype fp8_e5m2 --max-total-tokens 143360 --cuda-graph-max-bs 224 --cuda-graph-bs 32 64 96 128 160 192 224"

boot() { local tag=$1; shift
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
  docker exec -d sgl-kt bash -c "$ENVC CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server $COMMON $* > /tmp/sgl_S_$tag.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && break
    docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || { echo "$tag BOOT_FAIL" | tee -a $B/RUN.log; return 1; }; sleep 10; i=$((i+10)); done
  ((i>=1500)) && { echo "$tag BOOT_TIMEOUT" | tee -a $B/RUN.log; return 1; }
  docker exec sgl-kt bash /tmp/pin_nonkt.sh >/dev/null 2>&1
  echo "$tag BOOT_OK ${i}s kv=$(docker exec sgl-kt grep -m1 -o '#tokens: [0-9]*' /tmp/sgl_S_$tag.log | grep -o '[0-9]*')" | tee -a $B/RUN.log; return 0; }

rep() { local f=$1 lab=$2
  echo "$lab tok/s $(grep -h 'Output token throughput' $f|awk '{print $NF}'), dur $(grep -h 'Benchmark duration' $f|awk '{print $NF}')s, req/s $(grep -h 'Request throughput' $f|awk '{print $NF}'), TPOT $(grep -h 'Mean TPOT' $f|awk '{print $NF}')/p99 $(grep -h 'P99 TPOT' $f|awk '{print $NF}'), TTFT p50 $(grep -h 'Median TTFT' $f|awk '{print $NF}')/p99 $(grep -h 'P99 TTFT' $f|awk '{print $NF}'), ok $(grep -h 'Successful requests' $f|awk '{print $NF}')" | tee -a $B/RUN.log; }

# (a) 반복: 고정 512 job, seed 5개
reps() { local cell=$1 C=$2
  for S in 42 7 1234 99 20260910; do local rf=/tmp/s_rep_${cell}_s${S}.log
    curl -s -X POST http://127.0.0.1:30000/flush_cache -o /dev/null; sleep 3
    docker exec -d vllm-h100 bash -c "timeout 1800 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 512 --max-concurrency $C --request-rate inf --seed $S --percentile-metrics ttft,tpot --metric-percentiles 50,95,99 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
    local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>1830)) && break; done
    docker cp vllm-h100:$rf $B/rep_${cell}_C${C}_s${S}.log 2>/dev/null; rep $B/rep_${cell}_C${C}_s${S}.log "REP ${cell} C${C} seed${S}:"
  done; }

# soak 중 자원 샘플러
sampler_start() { local cell=$1; ( while :; do
    printf "%s gpu_mem=[%s] run=%s\n" "$(date +%H:%M:%S)" \
      "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')" \
      "$(docker exec sgl-kt bash -c "grep -o '#running-req: [0-9]*' /tmp/sgl_S_${cell}.log | tail -1 | awk '{print \$2}'" 2>/dev/null)"
    sleep 30; done ) >> $B/soak_sample_${cell}.txt 2>&1 &
  echo $! > /tmp/sampler_pid_$cell; }
sampler_stop() { local cell=$1; local p=$(cat /tmp/sampler_pid_$cell 2>/dev/null); [ -n "$p" ] && kill $p 2>/dev/null; }

# (b) soak 한 구간: open-loop λ 를 1200초 분량
phase() { local cell=$1 ph=$2 lam=$3; local N=$(python3 -c "print(int(round($lam*1200)))"); local rf=/tmp/s_soak_${cell}_${ph}.log
  docker exec -d vllm-h100 bash -c "timeout 2400 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $N --request-rate $lam --seed 42 --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 20; w=$((w+20)); ((w>2430)) && break; done
  docker cp vllm-h100:$rf $B/soak_${cell}_${ph}.log 2>/dev/null
  local d=$(grep -h 'Benchmark duration' $B/soak_${cell}_${ph}.log|awk '{print $NF}')
  rep $B/soak_${cell}_${ph}.log "SOAK ${cell} ${ph} lam${lam} (N=$N):"
  echo "   lambda_completed $(python3 -c "print(round($N/max($d,1e-9),2))" 2>/dev/null)" | tee -a $B/RUN.log; }

soak() { local cell=$1 lam_hi=$2
  curl -s -X POST http://127.0.0.1:30000/flush_cache -o /dev/null; sleep 3
  sampler_start $cell
  phase $cell P1_low 1.5
  phase $cell P2_cap $lam_hi
  phase $cell P3_low 1.5
  sampler_stop $cell
  docker cp sgl-kt:/tmp/sgl_S_${cell}.log $B/server_${cell}.log 2>/dev/null
  echo "SOAK ${cell} 이상로그: OOM=$(grep -ci 'out of memory' $B/server_${cell}.log) retract=$(grep -c -i 'retract' $B/server_${cell}.log) err=$(grep -c -iE '^\[.*\] (ERROR|CRITICAL)' $B/server_${cell}.log)" | tee -a $B/RUN.log; }

echo "== EXP-E12 soak + 반복 $(date +%H:%M) ==" | tee -a $B/RUN.log

boot TUNOPS $TUN --enable-mixed-chunk && { reps TUNOPS 96;  soak TUNOPS 3.0; }
boot EPCORE $EPB                      && { reps EPCORE 224; soak EPCORE 4.5; }
boot EPOPS  $EPB --enable-mixed-chunk && { reps EPOPS 224;  soak EPOPS 5.0; }

docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "E12_DONE $B"
