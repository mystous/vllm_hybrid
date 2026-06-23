#!/usr/bin/env bash
# SUB_248 — 70B 서빙 성능 레버 10종 자동 sweep (GPU-side, 축적 CPU/NUMA/harvest와 구별).
# 각 설정: 부팅 → ready → 깨끗한 벤치(고유 프롬프트, APC 무효) → GPU util 샘플 → pgid 정밀종료.
# 실패 시 skip. 결과 CSV. broad pkill 절대 금지 (setsid pgid 종료만).
set -uo pipefail
set +B   # brace expansion 비활성 — --speculative-config JSON {...} 보호 (unquoted 확장)
cd "$(dirname "$0")"
PY=/home/mystous/vllm_dev_prj/bin/python
VBIN=/home/mystous/vllm_dev_prj/bin/vllm
MODEL=meta-llama/Llama-3.1-70B-Instruct
PORT=8021
CSV=runs/sweep_results.csv
LOGD=runs; mkdir -p $LOGD
echo "name,status,gpu_util_mid,gen_tps_r1,gen_tps_r2,best_tps,vs_baseline" > $CSV

# 설정: name | gpus | env | serve_flags     (베이스 공통: TP, gmu, mml)
CONFIGS=(
  "baseline|0,1,2,3||"
  "spec_ngram_k5|0,1,2,3||--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":5,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"
  "fp8_kv|0,1,2,3||--kv-cache-dtype fp8"
  "max_num_seqs_512|0,1,2,3|| --max-num-seqs 512"
  "batched_tokens_8192|0,1,2,3||--max-num-batched-tokens 8192"
  "attn_flashinfer|0,1,2,3|VLLM_ATTENTION_BACKEND=FLASHINFER|"
  "attn_flash_attn|0,1,2,3|VLLM_ATTENTION_BACKEND=FLASH_ATTN|"
  "cudagraph_full|0,1,2,3||--compilation-config {\"cudagraph_mode\":\"FULL\"}"
  "tp8|0,1,2,3,4,5,6,7||"
  "fp8kv_spec_ngram|0,1,2,3||--kv-cache-dtype fp8 --speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":5,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"
  "spec_ngram_k8|0,1,2,3||--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":8,\"prompt_lookup_max\":5,\"prompt_lookup_min\":2}"
)

wait_gpu_free(){ for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }

run_cfg(){
  local name=$1 gpus=$2 env=$3 flags=$4
  local tp=$(echo $gpus | tr ',' '\n' | wc -l)
  local slog=$LOGD/serve_${name}.log
  wait_gpu_free
  echo "[boot] $name (gpus=$gpus tp=$tp) flags=$flags"
  CUDA_VISIBLE_DEVICES=$gpus HF_HOME=/raid/hf_cache $env \
    setsid $VBIN serve $MODEL --tensor-parallel-size $tp --port $PORT \
      --gpu-memory-utilization 0.85 --max-model-len 4096 $flags > $slog 2>&1 &
  local LEAD=$!
  # ready 대기 (최대 ~400s)
  local ready=0
  for i in $(seq 1 80); do
    if curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then ready=1; break; fi
    if grep -qiE "Traceback|Error:|unrecognized|out of memory|died unexpectedly|ValueError|RuntimeError" $slog 2>/dev/null; then break; fi
    sleep 5
  done
  if [ $ready -ne 1 ]; then
    echo "$name,BOOT_FAIL,,,,," >> $CSV
    echo "  [FAIL] $name boot — $(grep -iE 'error|unrecognized|valueerror|runtimeerror' $slog | tail -1 | cut -c1-80)"
    kill -TERM -- -$LEAD 2>/dev/null; sleep 4; kill -KILL -- -$LEAD 2>/dev/null; wait_gpu_free; return
  fi
  # warmup
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 128 --reqs 32 --tag W --salt w >/dev/null 2>&1
  # 벤치 2회 + util 샘플
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r1 > $LOGD/bench_${name}_1.txt 2>&1 &
  local BPID=$!; sleep 6
  local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')
  wait $BPID
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r2 > $LOGD/bench_${name}_2.txt 2>&1
  local t1=$(grep -oE "gen_tps=[0-9.]+" $LOGD/bench_${name}_1.txt | cut -d= -f2)
  local t2=$(grep -oE "gen_tps=[0-9.]+" $LOGD/bench_${name}_2.txt | cut -d= -f2)
  local best=$(echo -e "$t1\n$t2" | sort -rn | head -1)
  echo "$name,OK,$util,$t1,$t2,$best," >> $CSV
  echo "  [OK] $name util=$util% tps=$t1/$t2 best=$best"
  # 정밀 종료 (pgid)
  kill -TERM -- -$LEAD 2>/dev/null; sleep 4; kill -KILL -- -$LEAD 2>/dev/null
  wait_gpu_free
}

echo "===== SUB_248 10-lever sweep 시작 ====="
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r name gpus env flags <<< "$c"
  run_cfg "$name" "$gpus" "$env" "$flags"
done
echo "===== 완료. baseline 대비 계산 ====="
$PY - <<'PYEOF'
import csv
rows=list(csv.DictReader(open("runs/sweep_results.csv")))
base=next((float(r["best_tps"]) for r in rows if r["name"]=="baseline" and r["best_tps"]), None)
print(f"baseline best_tps = {base}")
for r in rows:
    if r["status"]!="OK" or not r["best_tps"]:
        print(f"  {r['name']:24s} {r['status']}"); continue
    b=float(r["best_tps"]); vs=(b/base-1)*100 if base else 0
    print(f"  {r['name']:24s} util={r['gpu_util_mid']:>3}% best={b:8.1f}  vs_base={vs:+.1f}%")
PYEOF
