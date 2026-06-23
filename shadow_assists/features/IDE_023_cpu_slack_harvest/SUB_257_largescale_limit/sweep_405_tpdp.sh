#!/usr/bin/env bash
# SUB_257 iter6 ★: 405B-FP8 통신 해결책 측정 — TP8(통신많음) vs TP4×DP2(통신↓+DP throughput).
# 405GB→최소 TP4. "최소 TP로 쪼개고 나머지 DP" = 70B DP win의 대형 dense 일반화 = 진짜 해결책.
set -uo pipefail; set +B
DIR=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_257_largescale_limit
SWEEP=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_248_serving_lever_sweep
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
MODEL=meta-llama/Llama-3.1-405B-Instruct-FP8; PORT=8053; LOGD=$DIR/runs; CSV=$LOGD/r405_tpdp.csv
mkdir -p $LOGD; echo "name,parallel,gpu_util,gen_tps_r1,gen_tps_r2,best_tps" > $CSV
CC='--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE"}'
CONFIGS=(
  "tp8|--tensor-parallel-size 8"
  "tp4dp2|--data-parallel-size 2 --tensor-parallel-size 4"
)
wgf(){ for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
run(){ local name=$1 pflags=$2; local slog=$LOGD/serve_405_${name}.log
  wgf; echo "[boot] $name $pflags ($(date -u +%H:%M))"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/raid/hf_cache HF_MODULES_CACHE=/home/mystous/hf_mods setsid $VBIN serve $MODEL \
    $pflags --gpu-memory-utilization 0.90 --max-model-len 8192 $CC \
    --port $PORT > $slog 2>&1 &
  local L=$!; local ok=0
  for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }; grep -qiE "Traceback \(most|AssertionError|RuntimeError|out of memory|ValueError" $slog && break; sleep 5; done
  if [ $ok -ne 1 ]; then echo "$name,$pflags,BOOT_FAIL,,," >>$CSV; echo "  [FAIL] $(grep -iE 'error|valueerror|assert|out of mem' $slog|tail -1|cut -c1-100)"; kill -KILL -- -$L 2>/dev/null; wgf; return; fi
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 64 --ptok 1000 --mtok 128 --reqs 64 --tag W --salt w >/dev/null 2>&1
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 64 --ptok 1000 --mtok 256 --reqs 512 --tag $name --salt r1 > $LOGD/b405_${name}_1.txt 2>&1 &
  local B=$!; sleep 8; local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0|tr -d ' '); wait $B
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 64 --ptok 1000 --mtok 256 --reqs 512 --tag $name --salt r2 > $LOGD/b405_${name}_2.txt 2>&1
  local t1=$(grep -oE "gen_tps=[0-9.]+" $LOGD/b405_${name}_1.txt|cut -d= -f2); local t2=$(grep -oE "gen_tps=[0-9.]+" $LOGD/b405_${name}_2.txt|cut -d= -f2)
  local best=$(echo -e "${t1:-0}\n${t2:-0}"|sort -rn|head -1)
  echo "$name,$pflags,$util,$t1,$t2,$best" >>$CSV; echo "  [OK] $name util=$util% best=$best"
  kill -KILL -- -$L 2>/dev/null; wgf
}
echo "===== 405B-FP8 TP8 vs TP4×DP2 (통신 해결책) ====="
for c in "${CONFIGS[@]}"; do IFS='|' read -r n p <<< "$c"; run "$n" "$p"; done
echo "===== 완료 ====="; cat $CSV
$PY - <<'PYEOF'
import csv
r={x["name"]:x for x in csv.DictReader(open("/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_257_largescale_limit/runs/r405_tpdp.csv")) if x.get("best_tps") not in (None,"","BOOT_FAIL")}
if "tp8" in r and "tp4dp2" in r:
    a=float(r["tp8"]["best_tps"]); b=float(r["tp4dp2"]["best_tps"])
    print(f"405B aggregate gen_tps: TP8={a:.0f} vs TP4xDP2={b:.0f} → {(b/a-1)*100:+.1f}% (최소TP+DP 통신 해결책)")
PYEOF
