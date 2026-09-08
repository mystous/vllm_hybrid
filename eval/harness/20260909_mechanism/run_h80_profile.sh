#!/usr/bin/env bash
# N=8(cold 전량 deferral) 정상 상태 프로파일: 스텝 분해 (커널/복사/유휴) — 숨김 캡의 대기 지점 특정
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
until grep -q "N8_PROFILE_DONE\|BOOT_FAIL" $SP/n8prof.log 2>/dev/null; do sleep 30; done
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$HOME/projects/vllm_hybrid/eval/results/$(date +%Y%m%d_%H%M%S)_ide032_h80_n8_profile_b; mkdir -p $B
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; rm -rf /tmp/sgl_prof4; mkdir -p /tmp/sgl_prof4; true'; sleep 5
docker exec -d sgl-kt bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 131072 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 80 --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 8 > /tmp/sgl_h80prof.log 2>&1"
i=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo HEALTH | tee $B/RUN.log; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || { echo BOOT_FAIL | tee $B/RUN.log; exit 1; }; sleep 10; i=$((i+10)); done
rf=/tmp/h80prof_bench.log; docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts 160 --max-concurrency 32 --request-rate inf --seed 42 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
sleep 45; curl -s -X POST http://127.0.0.1:30000/start_profile -H 'Content-Type: application/json' -d '{"output_dir":"/tmp/sgl_prof4","num_steps":6,"activities":["CPU","GPU"]}' >/dev/null; sleep 15; curl -s -X POST http://127.0.0.1:30000/stop_profile >/dev/null
w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>500)) && break; done
docker cp sgl-kt:/tmp/sgl_prof4/. $B/ 2>/dev/null; ls $B | grep -c trace.json.gz | tee -a $B/RUN.log
python3 $SP/analyze_trace.py $B 2>&1 | tail -6 | tee -a $B/RUN.log
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; echo "H80_PROFILE_DONE $B"
