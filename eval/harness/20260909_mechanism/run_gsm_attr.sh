#!/usr/bin/env bash
# GSM100 귀속: (a) N=4 host 콜백 (기존 최종, 97.0 재측정 = 잡음 추정) (b) N=8 host 콜백 (IDE_032 구성) — 둘 다 prompt map, hot-96
set -uo pipefail
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
SP=/tmp/claude-1019/-home-mystous-projects-vllm-hybrid/dfe3454a-8b67-497c-afa5-34e0ac76691f/scratchpad
REPO=$HOME/projects/vllm_hybrid
until grep -q "GSM100B_DONE" $SP/gsm100b.log 2>/dev/null; do sleep 30; done
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
M=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
B=$REPO/eval/results/$(date +%Y%m%d_%H%M%S)_ide034_gsm100_attribution; mkdir -p $B
for N in 4 8; do
  docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'; sleep 5
  docker exec -d sgl-kt bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --mem-fraction-static 0.92 --max-total-tokens 49152 --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /tmp/hotmap.json --ep-dispatch-algorithm dynamic --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token $N > /tmp/sgl_attr$N.log 2>&1"
  i=0; ok=0; while ((i<1500)); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; docker exec sgl-kt pgrep -f "sglang.launch_serve[r]" >/dev/null || break; sleep 10; i=$((i+10)); done
  [ $ok = 1 ] || { echo "N$N BOOT_FAIL" | tee -a $B/RUN.log; continue; }
  echo "N$N boot ok ${i}s" | tee -a $B/RUN.log
  python3 $SP/gsm_eval.py $B/gsm100_N$N.json 100 2>&1 | tail -1 | sed "s/^/HOSTCB N=$N GSM100: /" | tee -a $B/RUN.log
done
docker exec sgl-kt bash -c 'pkill -9 -f sglang.launch_serve[r] 2>/dev/null; true'
echo "GSMATTR_DONE $B"
