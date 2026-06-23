#!/usr/bin/env bash
# SUB_256 iter6: 큰 prefill AR을 느린 NCCL Ring에서 빠른 경로로 — VLLM_USE_NCCL_SYMM_MEM A/B.
# prefill-heavy 부하(ptok 2000, mtok 128)로 prefill-AR 비중 노출. tps 직접 측정.
set -uo pipefail; set +B
DIR=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_256_comm_bottleneck
SWEEP=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_248_serving_lever_sweep
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
MODEL=/raid/hf_cache/awqgptq_nvfp4_70b; PORT=8038; LOGD=$DIR/runs; CSV=$LOGD/symm512_results.csv
mkdir -p $LOGD; echo "name,env,gpu_util,prefill_tps_r1,prefill_tps_r2,best" > $CSV
# name|extra_env
CONFIGS=(
  "symm512|VLLM_SYMM_MEM_MAX_SIZE_MB=512"
  "symm1024|VLLM_SYMM_MEM_MAX_SIZE_MB=1024"
)
wgf(){ for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
run(){ local name=$1 envv=$2; local slog=$LOGD/serve_ar_${name}.log
  wgf; echo "[boot] $name env=$envv"
  env $envv VLLM_SUFFIX_PAD_UNIFORM=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/raid/hf_cache setsid $VBIN serve $MODEL \
    --tensor-parallel-size 8 --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --speculative-config '{"method":"suffix","num_speculative_tokens":6}' \
    --port $PORT > $slog 2>&1 &
  local L=$!; local ok=0
  for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }; grep -qiE "Traceback \(most|AssertionError|RuntimeError|out of memory|ValueError" $slog && break; sleep 5; done
  if [ $ok -ne 1 ]; then echo "$name,$envv,BOOT_FAIL,,," >>$CSV; echo "  [FAIL] $(grep -iE 'error|valueerror|assert' $slog|tail -1|cut -c1-90)"; kill -KILL -- -$L 2>/dev/null; wgf; return; fi
  # prefill-heavy: 짧은 gen, 긴 prompt → prefill AR 비중↑
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 4000 --mtok 8 --reqs 48 --tag W --salt w >/dev/null 2>&1
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 4000 --mtok 8 --reqs 384 --tag $name --salt r1 > $LOGD/bar_${name}_1.txt 2>&1 &
  local B=$!; sleep 6; local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0|tr -d ' '); wait $B
  $PY $SWEEP/bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 4000 --mtok 8 --reqs 384 --tag $name --salt r2 > $LOGD/bar_${name}_2.txt 2>&1
  # prefill 처리량 = prompt_tok 기준 (bench는 gen_tps; prefill 비중 위해 wall 비교)
  local w1=$(grep -oE "wall_s=[0-9.]+" $LOGD/bar_${name}_1.txt|cut -d= -f2); local w2=$(grep -oE "wall_s=[0-9.]+" $LOGD/bar_${name}_2.txt|cut -d= -f2)
  local t1=$(grep -oE "gen_tps=[0-9.]+" $LOGD/bar_${name}_1.txt|cut -d= -f2); local t2=$(grep -oE "gen_tps=[0-9.]+" $LOGD/bar_${name}_2.txt|cut -d= -f2)
  # prefill 처리 속도 ∝ reqs/wall (짧은 gen이라 wall≈prefill 지배)
  local p1=$(awk "BEGIN{print 384/$w1}" 2>/dev/null); local p2=$(awk "BEGIN{print 384/$w2}" 2>/dev/null)
  local best=$(echo -e "${p1:-0}\n${p2:-0}"|sort -rn|head -1)
  echo "$name,$envv,$util,$p1,$p2,$best" >>$CSV
  echo "  [OK] $name util=$util% prefill_req/s(best)=$best (wall $w1/$w2, gen_tps $t1/$t2)"
  kill -KILL -- -$L 2>/dev/null; wgf
}
echo "===== iter6 AR-path A/B (prefill-heavy) ====="
for c in "${CONFIGS[@]}"; do IFS='|' read -r n e <<< "$c"; run "$n" "$e"; done
echo "===== 완료 ====="; cat $CSV
$PY - <<'PYEOF'
import csv
r={x["name"]:x for x in csv.DictReader(open("/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_256_comm_bottleneck/runs/symm512_results.csv")) if x.get("best") not in (None,"","BOOT_FAIL")}
if "baseline" in r and "nccl_symm" in r:
    b=float(r["baseline"]["best"]); n=float(r["nccl_symm"]["best"])
    print(f"prefill req/s: baseline={b:.2f} vs nccl_symm={n:.2f} → {(n/b-1)*100:+.1f}% (NCCL Ring→NVLS 라우팅 효과)")
PYEOF
