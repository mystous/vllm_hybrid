#!/usr/bin/env bash
set -uo pipefail; set +B; cd "$(dirname "$0")"
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
M=meta-llama/Llama-3.1-70B-Instruct; PORT=8021; LOGD=runs
echo "name,conc,gen_tps,p_ttft_proxy" > runs/async_results.csv
wgf(){ for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
boot(){ wgf; CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HOME=/raid/hf_cache setsid $VBIN serve $M --tensor-parallel-size 4 --port $PORT --gpu-memory-utilization 0.85 --max-model-len 4096 $1 > $LOGD/serve_async_$2.log 2>&1 & echo $!; }
rdy(){ for i in $(seq 1 90); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && return 0; grep -qiE "Traceback \(most|AssertionError|RuntimeError|unrecognized arg" $1 && return 1; sleep 5; done; return 1; }
cell(){ # name flags
  local name=$1 flags=$2; local L=$(boot "$flags" "$name")
  if ! rdy $LOGD/serve_async_$name.log; then echo "$name,BOOT_FAIL,," >>runs/async_results.csv; kill -KILL -- -$L 2>/dev/null; wgf; return; fi
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $M --conc 24 --ptok 2000 --mtok 128 --reqs 32 --tag W --salt w >/dev/null 2>&1
  for C in 24 4; do
    local t=$($PY bench_unique.py --base http://127.0.0.1:$PORT --model $M --conc $C --ptok 1500 --mtok 256 --reqs $((C*6)) --tag $name --salt c$C 2>&1 | grep -oE "gen_tps=[0-9.]+"|cut -d= -f2)
    echo "$name,$C,$t," >> runs/async_results.csv; echo "  [$name conc=$C] tps=$t"
  done
  kill -KILL -- -$L 2>/dev/null; wgf
}
echo "=== async OFF ==="; cell async_off ""
echo "=== async ON  ==="; cell async_on "--async-scheduling"
echo "=== 결과 ==="; cat runs/async_results.csv
