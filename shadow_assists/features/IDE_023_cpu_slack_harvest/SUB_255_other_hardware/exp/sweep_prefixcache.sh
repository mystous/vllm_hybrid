#!/usr/bin/env bash
# SUB_255 iter3: 공유-프리픽스 워크로드서 prefix KV 재사용(APC) win 실측. best TP8.
# prefix-cache ON(KV 재사용→prefill skip) vs OFF(매 요청 재연산). NEO(활성KV swap)와 차별=정적 prefix 재사용.
set -uo pipefail; set +B
DIR=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_255_other_hardware
EXP=$DIR/exp
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
MODEL=/raid/hf_cache/awqgptq_nvfp4_70b; PORT=8035; LOGD=$DIR/runs; CSV=$LOGD/prefixcache_results.csv
mkdir -p $LOGD; echo "name,apc,wall_s,gen_tps,req_per_s" > $CSV
# name|apc_flag
CONFIGS=(
  "apc_off|--no-enable-prefix-caching"
  "apc_on|--enable-prefix-caching"
)
wgf(){ for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
run(){ local name=$1 flag=$2; local slog=$LOGD/serve_${name}.log
  wgf; echo "[boot] $name $flag"
  VLLM_SUFFIX_PAD_UNIFORM=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/raid/hf_cache setsid $VBIN serve $MODEL \
    --tensor-parallel-size 8 --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --speculative-config '{"method":"suffix","num_speculative_tokens":6}' $flag \
    --port $PORT > $slog 2>&1 &
  local L=$!; local ok=0
  for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }; grep -qiE "Traceback \(most|AssertionError|RuntimeError|out of memory|ValueError" $slog && break; sleep 5; done
  if [ $ok -ne 1 ]; then echo "$name,$flag,BOOT_FAIL,," >>$CSV; echo "  [FAIL] $(grep -iE 'error|valueerror|assert' $slog|tail -1|cut -c1-90)"; kill -KILL -- -$L 2>/dev/null; wgf; return; fi
  # warmup (공유 prefix 1회 적재; OFF면 무효)
  $PY $EXP/bench_sharedprefix.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 3500 --mtok 32 --reqs 24 --tag W --salt w >/dev/null 2>&1
  # 측정 (동일 공유 prefix, 유니크 짧은 suffix)
  local r=$($PY $EXP/bench_sharedprefix.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 3500 --mtok 32 --reqs 192 --tag $name --salt r1 2>&1 | grep "^BENCH")
  echo "  $r"
  local wall=$(echo "$r"|grep -oE "wall_s=[0-9.]+"|cut -d= -f2); local tps=$(echo "$r"|grep -oE "gen_tps=[0-9.]+"|cut -d= -f2); local rps=$(echo "$r"|grep -oE "req_per_s=[0-9.]+"|cut -d= -f2)
  echo "$name,$flag,$wall,$tps,$rps" >>$CSV
  kill -KILL -- -$L 2>/dev/null; wgf
}
echo "===== SUB_255 iter3 prefix-cache 재사용 win 실측 ====="
for c in "${CONFIGS[@]}"; do IFS='|' read -r n f <<< "$c"; run "$n" "$f"; done
echo "===== 완료 ====="; cat $CSV
$PY - <<'PYEOF'
import csv
r={x["name"]:x for x in csv.DictReader(open("/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_255_other_hardware/runs/prefixcache_results.csv")) if x.get("wall_s") not in (None,"","BOOT_FAIL")}
if "apc_off" in r and "apc_on" in r:
    wo,wn=float(r["apc_off"]["wall_s"]),float(r["apc_on"]["wall_s"])
    ro,rn=float(r["apc_off"]["req_per_s"]),float(r["apc_on"]["req_per_s"])
    print(f"공유-프리픽스 워크로드: APC_off wall={wo:.1f}s({ro:.2f}req/s) vs APC_on wall={wn:.1f}s({rn:.2f}req/s)")
    print(f"→ prefix KV 재사용 throughput +{(rn/ro-1)*100:.0f}% (wall -{(1-wn/wo)*100:.0f}%) = prefill 회피 win")
PYEOF
