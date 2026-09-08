#!/usr/bin/env bash
# CF + τ deferral (마스크 캐시 수정 후 재시도): hot-96 α=0.25, callback-free (빈 immediate 생략 OFF, 빈 deferred 생략 ON if built), 코어 재배치. τ 는 스윕 최선값 (GSM100 ≥96 중 최대 tok/s). C32 + GSM100 + C64. IDE_036 종료 후.
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
until grep -q "IDE036_DONE" $SP/ide036.log 2>/dev/null; do sleep 30; done
TD=$(ls -d $REPO/eval/results/*_ide035_tau_sweep_hostcb | tail -1)
TAU=$(python3 - "$TD/RUN.log" <<'PY'
import re,sys
t=open(sys.argv[1]).read(); best=None
for m in re.finditer(r"tau([0-9.]+) C32: ([0-9.]+) tok/s", t):
    tau,tok=m.group(1),float(m.group(2)); g=re.search(r"tau%s GSM100: ACC ([0-9.]+)"%re.escape(tau), t)
    if g and float(g.group(1))>=0.96 and (best is None or tok>best[1]): best=(tau,tok)
print(best[0] if best else "0.25")
PY
)
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide035_cf_tau; mkdir -p $B; echo "tau=$TAU" | tee $B/RUN.log
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
docker exec -d sgl-kt bash -c "KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_DEF=1 KT_COLD_DEFER=1 KT_COLD_TAU=$TAU CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 49152 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 8 > /tmp/sgl_cftau.log 2>&1"
i=0; ok=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
[ $ok = 1 ] || { echo "CFTAU BOOT_FAIL" | tee -a $B/RUN.log; docker exec sgl-kt grep -m2 -B3 -A12 -E "Traceback|Error" /tmp/sgl_cftau.log | tail -30 >> $B/RUN.log; echo "CFTAU_DONE"; exit 1; }
echo "boot ok ${i}s" | tee -a $B/RUN.log; docker exec sgl-kt bash /tmp/pin_nonkt.sh | tee -a $B/RUN.log
for C in 32 64; do rf=/tmp/cftau_c$C.log; NP=$(( C==32 ? 128 : 192 ))
  docker exec -d vllm-h100 bash -c "timeout 500 vllm bench serve --backend openai --base-url http://127.0.0.1:30000 --endpoint /v1/completions --model q480 --tokenizer $TOK --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --num-prompts $NP --max-concurrency $C --request-rate inf --seed 42 --percentile-metrics ttft,tpot --metric-percentiles 50,95 > $rf 2>&1; echo DONE_EXIT=\$? >> $rf"
  w=0; until docker exec vllm-h100 bash -c "grep -q DONE_EXIT $rf" 2>/dev/null; do sleep 10; w=$((w+10)); ((w>560)) && break; done; docker cp vllm-h100:$rf $B/c$C.log 2>/dev/null
  echo "CF+tau$TAU C$C: $(grep -h 'Output token throughput' $B/c$C.log|awk '{print $NF}') tok/s, TPOT $(grep -h 'Mean TPOT' $B/c$C.log|awk '{print $NF}')  (N=8 skip: C32 659/36.5, C64 884/54.2)" | tee -a $B/RUN.log; done
python3 $SP/gsm_eval.py $B/gsm100.json 100 2>&1 | tail -1 | sed "s/^/CF+tau$TAU GSM100: /" | tee -a $B/RUN.log
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "CFTAU_DONE $B"
