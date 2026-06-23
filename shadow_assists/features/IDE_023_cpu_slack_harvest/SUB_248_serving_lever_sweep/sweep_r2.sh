#!/usr/bin/env bash
# SUB_248 라운드2 — 완전히 새로운 레버 10종 (연산축소·커널·병렬도, round1/축적과 구별).
# env 확장 버그 수정(`env $env setsid`). 정밀 pgid 종료. 깨끗한 벤치.
set -uo pipefail
set +B
cd "$(dirname "$0")"
PY=/home/mystous/vllm_dev_prj/bin/python
VBIN=/home/mystous/vllm_dev_prj/bin/vllm
MODEL=meta-llama/Llama-3.1-70B-Instruct
PORT=8021
CSV=runs/sweep_r2_results.csv
LOGD=runs; mkdir -p $LOGD
echo "name,status,gpu_util_mid,gen_tps_r1,gen_tps_r2,best_tps" > $CSV

CONFIGS=(
  "fp8_weight|0,1,2,3||--quantization fp8"
  "fp8_weight_spec|0,1,2,3||--quantization fp8 --speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":5,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"
  "attn_flashinfer|0,1,2,3|VLLM_ATTENTION_BACKEND=FLASHINFER|"
  "attn_flash_attn|0,1,2,3|VLLM_ATTENTION_BACKEND=FLASH_ATTN|"
  "attn_triton|0,1,2,3|VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1|"
  "tp2|0,1||"
  "tp2_spec|0,1||--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":5,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"
  "kv_fp8_e5m2|0,1,2,3||--kv-cache-dtype fp8_e5m2"
  "enforce_eager|0,1,2,3||--enforce-eager"
  "fp8_weight_tp8|0,1,2,3,4,5,6,7||--quantization fp8"
)

wait_gpu_free(){ for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }

run_cfg(){
  local name=$1 gpus=$2 env=$3 flags=$4
  local tp=$(echo $gpus | tr ',' '\n' | wc -l)
  local slog=$LOGD/serve_r2_${name}.log
  wait_gpu_free
  echo "[boot] $name (tp=$tp) env=$env flags=$flags"
  CUDA_VISIBLE_DEVICES=$gpus HF_HOME=/raid/hf_cache \
    env $env setsid $VBIN serve $MODEL --tensor-parallel-size $tp --port $PORT \
      --gpu-memory-utilization 0.85 --max-model-len 4096 $flags > $slog 2>&1 &
  local LEAD=$!
  local ready=0
  for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then ready=1; break; fi
    if grep -qiE "Traceback|Error:|unrecognized|out of memory|died unexpectedly|ValueError|RuntimeError|No such" $slog 2>/dev/null; then break; fi
    sleep 5
  done
  if [ $ready -ne 1 ]; then
    echo "$name,BOOT_FAIL,,,," >> $CSV
    echo "  [FAIL] $name — $(grep -iE 'error|unrecognized|valueerror|runtimeerror|no such|not support' $slog | tail -1 | cut -c1-90)"
    kill -TERM -- -$LEAD 2>/dev/null; sleep 4; kill -KILL -- -$LEAD 2>/dev/null; wait_gpu_free; return
  fi
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 128 --reqs 32 --tag W --salt w >/dev/null 2>&1
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r1 > $LOGD/bench_r2_${name}_1.txt 2>&1 &
  local BPID=$!; sleep 6
  local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')
  wait $BPID
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r2 > $LOGD/bench_r2_${name}_2.txt 2>&1
  local t1=$(grep -oE "gen_tps=[0-9.]+" $LOGD/bench_r2_${name}_1.txt | cut -d= -f2)
  local t2=$(grep -oE "gen_tps=[0-9.]+" $LOGD/bench_r2_${name}_2.txt | cut -d= -f2)
  local best=$(echo -e "${t1:-0}\n${t2:-0}" | sort -rn | head -1)
  echo "$name,OK,$util,$t1,$t2,$best" >> $CSV
  echo "  [OK] $name util=$util% tps=$t1/$t2 best=$best"
  kill -TERM -- -$LEAD 2>/dev/null; sleep 4; kill -KILL -- -$LEAD 2>/dev/null
  wait_gpu_free
}

echo "===== SUB_248 라운드2 (신규 10종) 시작 ====="
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r name gpus env flags <<< "$c"
  run_cfg "$name" "$gpus" "$env" "$flags"
done
echo "===== 라운드2 완료 ====="
$PY - <<'PYEOF'
import csv
rows=list(csv.DictReader(open("runs/sweep_r2_results.csv")))
BASE=1436.9  # round1 baseline
for r in rows:
    if r["status"]!="OK" or not r["best_tps"]:
        print(f"  {r['name']:22s} {r['status']}"); continue
    b=float(r["best_tps"]); print(f"  {r['name']:22s} util={r['gpu_util_mid']:>3}% best={b:8.1f}  vs_base(1436.9)={(b/BASE-1)*100:+.1f}%")
PYEOF
