#!/usr/bin/env bash
# SUB_255 iter2: NVLS/SHARP(NVSwitch in-network all-reduce) A/B. custom AR(현행) vs NCCL-NVLS.
# reduction을 GPU SM이 아닌 NVSwitch 하드웨어로 오프로드 시 tps 변화.
set -uo pipefail; set +B
DIR=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_255_other_hardware
SWEEP=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_248_serving_lever_sweep
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
MODEL=/raid/hf_cache/awqgptq_nvfp4_70b; PORT=8034; LOGD=$DIR/runs; CSV=$LOGD/nvls_results.csv
mkdir -p $LOGD; echo "name,gpu_util,tps_r1,tps_r2,best_tps" > $CSV
# name|extra_env|extra_flags
CONFIGS=(
  "custom_ar|VLLM_X=0|"
  "nccl_nvls|NCCL_NVLS_ENABLE=1 NCCL_ALGO=NVLS|--disable-custom-all-reduce"
)
wgf(){ for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
run(){ local name=$1 env=$2 flags=$3; local slog=$LOGD/serve_${name}.log
  wgf; echo "[boot] $name env=$env flags=$flags"
  env $env VLLM_SUFFIX_PAD_UNIFORM=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/raid/hf_cache setsid $VBIN serve $MODEL \
    --tensor-parallel-size 8 --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --speculative-config '{"method":"suffix","num_speculative_tokens":6}' $flags \
    --port $PORT > $slog 2>&1 &
  local L=$!; local ok=0
  for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }; grep -qiE "Traceback \(most|AssertionError|RuntimeError|out of memory|ValueError" $slog && break; sleep 5; done
  if [ $ok -ne 1 ]; then echo "$name,BOOT_FAIL,,," >>$CSV; echo "  [FAIL] $(grep -iE 'error|valueerror|assert' $slog|tail -1|cut -c1-90)"; kill -KILL -- -$L 2>/dev/null; wgf; return; fi
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 128 --reqs 32 --tag W --salt w >/dev/null 2>&1
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r1 > $LOGD/bn_${name}_1.txt 2>&1 &
  local B=$!; sleep 6; local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0|tr -d ' '); wait $B
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r2 > $LOGD/bn_${name}_2.txt 2>&1
  local t1=$(grep -oE "gen_tps=[0-9.]+" $LOGD/bn_${name}_1.txt|cut -d= -f2); local t2=$(grep -oE "gen_tps=[0-9.]+" $LOGD/bn_${name}_2.txt|cut -d= -f2)
  local best=$(echo -e "${t1:-0}\n${t2:-0}"|sort -rn|head -1)
  echo "$name,$util,$t1,$t2,$best" >>$CSV; echo "  [OK] $name util=$util% best=$best"
  kill -KILL -- -$L 2>/dev/null; wgf
}
echo "===== SUB_255 iter2 NVLS A/B ====="
for c in "${CONFIGS[@]}"; do IFS='|' read -r n e f <<< "$c"; run "$n" "$e" "$f"; done
echo "===== 완료 ====="; cat $CSV
