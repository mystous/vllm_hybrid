#!/usr/bin/env bash
# SUB_256 iter8 ★: TP8(통신 50%) vs DP8(통신 0, 40GB 모델이 1 GPU에 들어감) aggregate throughput.
# NVFP4 70B=40GB < B200 183GB → DP8(8 독립복제본) 가능. 통신 제거가 throughput win 만드는지.
set -uo pipefail; set +B
DIR=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_256_comm_bottleneck
SWEEP=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_248_serving_lever_sweep
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
MODEL=/raid/hf_cache/awqgptq_nvfp4_70b; PORT=8040; LOGD=$DIR/runs; CSV=$LOGD/dp_results.csv
mkdir -p $LOGD; echo "name,parallel,gpu_util,gen_tps_r1,gen_tps_r2,best_tps" > $CSV
SPEC='--speculative-config {"method":"suffix","num_speculative_tokens":6}'
CC='--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE"}'
# name|parallel flags
CONFIGS=(
  "tp8|--tensor-parallel-size 8"
  "dp8|--data-parallel-size 8 --tensor-parallel-size 1"
)
wgf(){ for i in $(seq 1 80); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
run(){ local name=$1 pflags=$2; local slog=$LOGD/serve_dp_${name}.log
  wgf; echo "[boot] $name $pflags"
  VLLM_SUFFIX_PAD_UNIFORM=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/raid/hf_cache setsid $VBIN serve $MODEL \
    $pflags --gpu-memory-utilization 0.85 --max-model-len 16384 $CC $SPEC \
    --port $PORT > $slog 2>&1 &
  local L=$!; local ok=0
  for i in $(seq 1 220); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }; grep -qiE "Traceback \(most|AssertionError|RuntimeError|out of memory|ValueError" $slog && break; sleep 5; done
  if [ $ok -ne 1 ]; then echo "$name,$pflags,BOOT_FAIL,,," >>$CSV; echo "  [FAIL] $(grep -iE 'error|valueerror|assert|out of mem' $slog|tail -1|cut -c1-100)"; kill -KILL -- -$L 2>/dev/null; wgf; return; fi
  # 고동시성으로 8 복제본/8 GPU 포화
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 96 --ptok 2000 --mtok 128 --reqs 96 --tag W --salt w >/dev/null 2>&1
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 96 --ptok 2000 --mtok 256 --reqs 768 --tag $name --salt r1 > $LOGD/bdp_${name}_1.txt 2>&1 &
  local B=$!; sleep 8; local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0|tr -d ' '); wait $B
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 96 --ptok 2000 --mtok 256 --reqs 768 --tag $name --salt r2 > $LOGD/bdp_${name}_2.txt 2>&1
  local t1=$(grep -oE "gen_tps=[0-9.]+" $LOGD/bdp_${name}_1.txt|cut -d= -f2); local t2=$(grep -oE "gen_tps=[0-9.]+" $LOGD/bdp_${name}_2.txt|cut -d= -f2)
  local best=$(echo -e "${t1:-0}\n${t2:-0}"|sort -rn|head -1)
  echo "$name,$pflags,$util,$t1,$t2,$best" >>$CSV; echo "  [OK] $name util=$util% best=$best (r1 $t1/r2 $t2)"
  kill -KILL -- -$L 2>/dev/null; wgf
}
echo "===== iter8 TP8 vs DP8 aggregate throughput ====="
for c in "${CONFIGS[@]}"; do IFS='|' read -r n p <<< "$c"; run "$n" "$p"; done
echo "===== 완료 ====="; cat $CSV
$PY - <<'PYEOF'
import csv
r={x["name"]:x for x in csv.DictReader(open("/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_256_comm_bottleneck/runs/dp_results.csv")) if x.get("best_tps") not in (None,"","BOOT_FAIL")}
if "tp8" in r and "dp8" in r:
    a=float(r["tp8"]["best_tps"]); b=float(r["dp8"]["best_tps"])
    print(f"aggregate gen_tps: TP8={a:.0f} vs DP8={b:.0f} → {(b/a-1)*100:+.1f}% (통신 제거 효과)")
PYEOF
