#!/usr/bin/env bash
# SUB_254 (c): TP8 best 구성의 all_reduce(6%) 감소 레버 A/B.
# baseline(현행: trtllm AR+rms fusion) vs +sequence_parallel vs +gemm_comms(async-TP) vs +both.
set -uo pipefail; set +B
DIR=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_254_flamegraph_bottleneck
SWEEP=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_248_serving_lever_sweep
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
MODEL=/raid/hf_cache/awqgptq_nvfp4_70b; PORT=8032; LOGD=$DIR/runs; CSV=$LOGD/comm_results.csv
mkdir -p $LOGD
echo "name,gpu_util,tps_r1,tps_r2,best_tps" > $CSV
# name|compilation-config json
BASE='"cudagraph_mode":"FULL_AND_PIECEWISE"'
CONFIGS=(
  "baseline|{$BASE}"
  "sp|{$BASE,\"pass_config\":{\"enable_sp\":true}}"
  "gemm_comms|{$BASE,\"pass_config\":{\"fuse_gemm_comms\":true}}"
  "sp_gemm|{$BASE,\"pass_config\":{\"enable_sp\":true,\"fuse_gemm_comms\":true}}"
)
wgf(){ for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
run(){ local name=$1 cc=$2; local slog=$LOGD/serve_comm_${name}.log
  wgf; echo "[boot] $name cc=$cc"
  VLLM_SUFFIX_PAD_UNIFORM=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/raid/hf_cache setsid $VBIN serve $MODEL \
    --tensor-parallel-size 8 --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --compilation-config "$cc" \
    --speculative-config '{"method":"suffix","num_speculative_tokens":6}' \
    --port $PORT > $slog 2>&1 &
  local L=$!; local ok=0
  for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }; grep -qiE "Traceback \(most|AssertionError|RuntimeError|out of memory|ValueError" $slog && break; sleep 5; done
  if [ $ok -ne 1 ]; then echo "$name,BOOT_FAIL,,," >>$CSV; echo "  [FAIL] $(grep -iE 'error|assert|valueerror' $slog|tail -1|cut -c1-90)"; kill -KILL -- -$L 2>/dev/null; wgf; return; fi
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 128 --reqs 32 --tag W --salt w >/dev/null 2>&1
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r1 > $LOGD/bc_${name}_1.txt 2>&1 &
  local B=$!; sleep 6; local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0|tr -d ' '); wait $B
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r2 > $LOGD/bc_${name}_2.txt 2>&1
  local t1=$(grep -oE "gen_tps=[0-9.]+" $LOGD/bc_${name}_1.txt|cut -d= -f2); local t2=$(grep -oE "gen_tps=[0-9.]+" $LOGD/bc_${name}_2.txt|cut -d= -f2)
  local best=$(echo -e "${t1:-0}\n${t2:-0}"|sort -rn|head -1)
  echo "$name,$util,$t1,$t2,$best" >>$CSV
  echo "  [OK] $name util=$util% best=$best"
  kill -KILL -- -$L 2>/dev/null; wgf
}
echo "===== SUB_254 (c) TP8 통신 레버 A/B ====="
for c in "${CONFIGS[@]}"; do IFS='|' read -r n cc <<< "$c"; run "$n" "$cc"; done
echo "===== 완료 ====="; cat $CSV
$PY - <<'PYEOF'
import csv
r={x["name"]:x for x in csv.DictReader(open("/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_254_flamegraph_bottleneck/runs/comm_results.csv")) if x.get("best_tps") and x["best_tps"] not in("","BOOT_FAIL")}
b=float(r["baseline"]["best_tps"]) if "baseline" in r else None
for n in ["baseline","sp","gemm_comms","sp_gemm"]:
    if n in r:
        t=float(r[n]["best_tps"]); print(f"  {n:12s} {t:7.1f} tps  {('+%.1f%%'%((t/b-1)*100)) if b else ''} util={r[n]['gpu_util']}%")
print("baseline(trtllm AR+rms fusion) 대비 sp/gemm_comms 가 통신 줄여 tps↑면 채택.")
PYEOF
