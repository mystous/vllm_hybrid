#!/usr/bin/env bash
# IDE_047: EAGLE3 투기 디코딩 × 하이브리드 (hot-96, CF+τ, AMX 선택, FP8 KV). 메모리 단계적 후퇴 (graph 160/KV110k → 128/98k → 96/80k). 순서: C160 (fresh) → C64 → GSM40
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
until grep -q "IDE051_DONE" $SP/ide051.log 2>/dev/null; do sleep 20; done
grep -q "BUILD_FAIL" $SP/ide051.log && { echo "skip"; echo "IDE051S_DONE"; exit 0; }
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1); M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP"); TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$(ls -d $REPO/eval/results/*_ide051_fuse_qin | tail -1); echo "== serving A/B FUSE_QIN on/off (운영점 A) $(date +%H:%M)" | tee -a $B/RUN.log | tee -a $B/RUN.log
COMMON="--model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8"
boot() { docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 8
  docker exec -d sgl-kt bash -c "$3 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server $COMMON $2 > /tmp/sgl_051_$1.log 2>&1"
  local i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "$1 boot ok ${i}s $(docker exec sgl-kt grep -m1 -o 'KV Cache is allocated.*' /tmp/sgl_051_$1.log | cut -c1-90)" | tee -a $B/RUN.log; docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log; return 0; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  echo "$1 BOOT_FAIL $(docker exec sgl-kt grep -m1 -o 'OOM on device.*' /tmp/sgl_051_$1.log | cut -c1-120)" | tee -a $B/RUN.log; docker exec sgl-kt grep -m2 -B1 -A6 "Traceback\|Error" /tmp/sgl_051_$1.log | tail -30 | cut -c1-200 | tee -a $B/RUN.log; return 1; }
bench() { local rf=/tmp/i51_$1.log; local NP=$(( $2*3 ))
  docker exec -d vllm-h100 bash -c "timeout 600 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $NP --max-concurrency $2 --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  local w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>660)) && break; done; docker cp vllm-h100:$rf $B/$1.log 2>/dev/null
  echo "$1 C$2: $(grep -h 'Output token throughput' $B/$1.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/$1.log|awk '{print $NF}'), TTFT $(grep -h 'Mean TTFT' $B/$1.log|awk '{print $NF}')  $3" | tee -a $B/RUN.log; }
ok=0
DRAFT=$(ls -d /models/hub/models--lmsys--SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI/snapshots/*/ 2>/dev/null | head -1)
DRAFT=/models/hub/models--lmsys--SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI/snapshots/b2a87592e335783be6c617c3be93ceeab99334a2
SPEC="--speculative-algorithm EAGLE3 --speculative-draft-model-path $DRAFT --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 160"
ENV="SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 KT_AVX_RB=1 KT_AVX_PF=2"
run_set() { local tag=$1
  bench ${tag}_c160 160 "(기준 fresh 965~985 / TPOT 120 / TTFT 5.6~5.9s)"
  grep -q "C160: 0.00" $B/RUN.log 2>/dev/null && tail -1 $B/RUN.log | grep -q "${tag}_c160 C160: 0.00" && { echo "$tag RUNTIME_FAIL $(docker exec sgl-kt grep -m1 -o 'torch.OutOfMemoryError.*' /tmp/sgl_051_$tag.log | cut -c1-120)" | tee -a $B/RUN.log; return 1; }
  docker exec sgl-kt grep -o "accept len: [0-9.]*\|accept_length: [0-9.]*\|acc_len[=: ]*[0-9.]*" /tmp/sgl_051_$tag.log | tail -3 | tr "\n" " " | sed "s/^/$tag accept(C160): /" | tee -a $B/RUN.log; echo | tee -a $B/RUN.log
  bench ${tag}_c64 64 "(기준 728~766 / 60~63 / 3.0~3.2s)"
  docker exec sgl-kt grep -o "accept len: [0-9.]*\|accept_length: [0-9.]*\|acc_len[=: ]*[0-9.]*" /tmp/sgl_051_$tag.log | tail -3 | tr "\n" " " | sed "s/^/$tag accept(C64): /" | tee -a $B/RUN.log; echo | tee -a $B/RUN.log
  python3 $SP/gsm_eval.py $B/gsm40_$tag.json 40 2>&1 | tail -1 | sed "s/^/$tag GSM40: /" | tee -a $B/RUN.log; return 0; }
try_cfg() { local tag=$1; shift; boot $tag "$* $SPEC" "$ENV" || return 1; docker exec sgl-kt bash -c "nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '" | sed "s/^/$tag gpu mem after boot: /" | tee -a $B/RUN.log; echo | tee -a $B/RUN.log; run_set $tag; }
SPEC=""
ARGS="--kv-cache-dtype fp8_e5m2 --mem-fraction-static 0.94 --chunked-prefill-size 8192 --max-total-tokens 122880 --cuda-graph-max-bs 192 --cuda-graph-bs 32 64 96 128 160 192"
boot fuse_on "$ARGS" "$ENV KT_FUSE_QIN=1" && { ok=1; bench on_c192 192 "(기준 1048.7 / 131.1)"; bench on_c160 160 "(기준 1012.2 / 114.0)"; python3 $SP/gsm_eval.py $B/gsm40_on.json 40 2>&1 | tail -1 | sed "s/^/fuse_on GSM40: /" | tee -a $B/RUN.log; }
boot fuse_off "$ARGS" "$ENV" && { ok=1; bench off_c192 192 "(대조)"; bench off_c160 160 "(대조)"; }
[ $ok = 1 ] || echo "ALL BOOT_FAIL" | tee -a $B/RUN.log
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "IDE051S_DONE $B"
