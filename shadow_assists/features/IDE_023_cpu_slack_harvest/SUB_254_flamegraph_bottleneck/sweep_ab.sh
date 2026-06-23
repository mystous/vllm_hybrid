#!/usr/bin/env bash
# SUB_254 (a)+(b): best 구성(NVFP4 awqgptq + suffix K6 + FaP + pad)
# (a) TP4 vs TP8 tps 비교 (TP8 통신비 실증). (b) suffix propose 지연 주입 sweep로 critical-path 검정.
set -uo pipefail; set +B
DIR=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_254_flamegraph_bottleneck
SWEEP=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_248_serving_lever_sweep
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
MODEL=/raid/hf_cache/awqgptq_nvfp4_70b; PORT=8031; LOGD=$DIR/runs; CSV=$LOGD/ab_results.csv
mkdir -p $LOGD
echo "name,tp,delay_us,gpu_util,tps_r1,tps_r2,best_tps" > $CSV
# name|tp|gpus|delay_us
CONFIGS=(
  "tp8_d0|8|0,1,2,3,4,5,6,7|0"
  "tp4_d0|4|0,1,2,3|0"
  "tp4_d300|4|0,1,2,3|300"
  "tp4_d800|4|0,1,2,3|800"
)
wgf(){ for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
run(){ local name=$1 tp=$2 gpus=$3 delay=$4; local slog=$LOGD/serve_${name}.log
  wgf; echo "[boot] $name (tp=$tp delay=${delay}us)"
  VLLM_SUFFIX_PAD_UNIFORM=1 VLLM_SUFFIX_PROBE_DELAY_US=$delay CUDA_VISIBLE_DEVICES=$gpus HF_HOME=/raid/hf_cache setsid $VBIN serve $MODEL \
    --tensor-parallel-size $tp --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --speculative-config '{"method":"suffix","num_speculative_tokens":6}' \
    --port $PORT > $slog 2>&1 &
  local L=$!; local ok=0
  for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }; grep -qiE "Traceback \(most|AssertionError|RuntimeError|out of memory" $slog && break; sleep 5; done
  if [ $ok -ne 1 ]; then echo "$name,$tp,$delay,BOOT_FAIL,,," >>$CSV; echo "  [FAIL]"; tail -5 $slog; kill -KILL -- -$L 2>/dev/null; wgf; return; fi
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 128 --reqs 32 --tag W --salt w >/dev/null 2>&1
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r1 > $LOGD/b_${name}_1.txt 2>&1 &
  local B=$!; sleep 6; local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0|tr -d ' '); wait $B
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r2 > $LOGD/b_${name}_2.txt 2>&1
  local t1=$(grep -oE "gen_tps=[0-9.]+" $LOGD/b_${name}_1.txt|cut -d= -f2); local t2=$(grep -oE "gen_tps=[0-9.]+" $LOGD/b_${name}_2.txt|cut -d= -f2)
  local best=$(echo -e "${t1:-0}\n${t2:-0}"|sort -rn|head -1)
  echo "$name,$tp,$delay,$util,$t1,$t2,$best" >>$CSV
  echo "  [OK] $name util=$util% best=$best"
  kill -KILL -- -$L 2>/dev/null; wgf
}
echo "===== SUB_254 (a)TP4vsTP8 + (b)지연 sweep ====="
for c in "${CONFIGS[@]}"; do IFS='|' read -r n t g d <<< "$c"; run "$n" "$t" "$g" "$d"; done
echo "===== 완료 ====="; cat $CSV
$PY - <<'PYEOF'
import csv
r={x["name"]:x for x in csv.DictReader(open("/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_254_flamegraph_bottleneck/runs/ab_results.csv")) if x.get("best_tps") and x["best_tps"] not in("","BOOT_FAIL")}
def g(n): return float(r[n]["best_tps"]) if n in r else None
t8,t4=g("tp8_d0"),g("tp4_d0")
if t8 and t4: print(f"(a) TP8={t8:.0f} vs TP4={t4:.0f} → TP4 {'+' if t4>t8 else ''}{(t4/t8-1)*100:.1f}% (TP8 통신비 {'실증' if t4>t8 else '미확인'})")
d0,d3,d8=g("tp4_d0"),g("tp4_d300"),g("tp4_d800")
if d0 and d8: print(f"(b) TP4 지연0={d0:.0f} / 300us={d3:.0f} / 800us={d8:.0f} → 800us시 {(1-d8/d0)*100:.1f}% 하락 = {'critical-path' if (1-d8/d0)>0.15 else 'GPU오버랩(SUB_225 천장)'}")
PYEOF
