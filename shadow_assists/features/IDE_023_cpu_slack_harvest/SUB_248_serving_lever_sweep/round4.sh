#!/usr/bin/env bash
# 라운드4 — 코드(compute-graph) 구조 변경 10종: 커널 fusion/그래프분할 패스로 연산그래프 재작성.
# FP8 베이스(검증된 win) 위. fusion=수학적 동치 재작성 → 정확도 FP8 그대로. ref=fp8 1817.3.
set -uo pipefail; set +B
cd "$(dirname "$0")"
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
MODEL=meta-llama/Llama-3.1-70B-Instruct; PORT=8021; LOGD=runs; mkdir -p $LOGD
CSV=runs/sweep_r4_results.csv
echo "name,status,gpu_util_mid,gen_tps_r1,gen_tps_r2,best_tps" > $CSV
SPEC='--speculative-config {"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'

CONFIGS=(
  "fp8_base|0,1,2,3|--quantization fp8"
  "fuse_norm_quant|0,1,2,3|--quantization fp8 --compilation-config {\"pass_config\":{\"fuse_norm_quant\":true}}"
  "fuse_act_quant|0,1,2,3|--quantization fp8 --compilation-config {\"pass_config\":{\"fuse_act_quant\":true}}"
  "fuse_attn_quant|0,1,2,3|--quantization fp8 --compilation-config {\"pass_config\":{\"fuse_attn_quant\":true}}"
  "fuse_rope_kvcache|0,1,2,3|--quantization fp8 --compilation-config {\"pass_config\":{\"fuse_rope_kvcache\":true}}"
  "qk_norm_rope_fusion|0,1,2,3|--quantization fp8 --compilation-config {\"pass_config\":{\"enable_qk_norm_rope_fusion\":true}}"
  "fuse_gemm_comms|0,1,2,3|--quantization fp8 --compilation-config {\"pass_config\":{\"fuse_gemm_comms\":true}}"
  "enable_sp|0,1,2,3|--quantization fp8 --compilation-config {\"pass_config\":{\"enable_sp\":true}}"
  "all_fuse|0,1,2,3|--quantization fp8 --compilation-config {\"pass_config\":{\"fuse_norm_quant\":true,\"fuse_act_quant\":true,\"fuse_attn_quant\":true,\"fuse_rope_kvcache\":true}}"
  "all_fuse_gemmcomm|0,1,2,3|--quantization fp8 --compilation-config {\"pass_config\":{\"fuse_norm_quant\":true,\"fuse_act_quant\":true,\"fuse_attn_quant\":true,\"fuse_gemm_comms\":true,\"enable_sp\":true}}"
)

wait_gpu_free(){ for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
run_cfg(){
  local name=$1 gpus=$2 flags=$3; local tp=$(echo $gpus|tr ',' '\n'|wc -l); local slog=$LOGD/serve_r4_${name}.log
  wait_gpu_free; echo "[boot] $name flags=$flags"
  CUDA_VISIBLE_DEVICES=$gpus HF_HOME=/raid/hf_cache \
    setsid $VBIN serve $MODEL --tensor-parallel-size $tp --port $PORT \
      --gpu-memory-utilization 0.85 --max-model-len 4096 $flags > $slog 2>&1 &
  local LEAD=$!; local ready=0
  for i in $(seq 1 100); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ready=1; break; }; grep -qiE "Traceback \(most|raise [A-Z]|AssertionError|CUDA out of memory|died unexpectedly|unrecognized arguments" $slog 2>/dev/null && break; sleep 5; done
  if [ $ready -ne 1 ]; then echo "$name,BOOT_FAIL,,,," >> $CSV; echo "  [FAIL] $(grep -iE 'error|valueerror|runtimeerror|assert|no such' $slog|tail -1|cut -c1-90)"; kill -KILL -- -$LEAD 2>/dev/null; wait_gpu_free; return; fi
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 128 --reqs 32 --tag W --salt w >/dev/null 2>&1
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r1 > $LOGD/bench_r4_${name}_1.txt 2>&1 &
  local BPID=$!; sleep 6; local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0|tr -d ' '); wait $BPID
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $MODEL --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r2 > $LOGD/bench_r4_${name}_2.txt 2>&1
  local t1=$(grep -oE "gen_tps=[0-9.]+" $LOGD/bench_r4_${name}_1.txt|cut -d= -f2); local t2=$(grep -oE "gen_tps=[0-9.]+" $LOGD/bench_r4_${name}_2.txt|cut -d= -f2)
  local best=$(echo -e "${t1:-0}\n${t2:-0}"|sort -rn|head -1)
  echo "$name,OK,$util,$t1,$t2,$best" >> $CSV; echo "  [OK] $name util=$util% tps=$t1/$t2 best=$best"
  kill -KILL -- -$LEAD 2>/dev/null; wait_gpu_free
}
echo "===== 라운드4 (compute-graph 구조 fusion 10종, FP8 베이스) ====="
for c in "${CONFIGS[@]}"; do IFS='|' read -r name gpus flags <<< "$c"; run_cfg "$name" "$gpus" "$flags"; done
echo "===== 완료 ====="
$PY - <<'PYEOF'
import csv
rows=list(csv.DictReader(open("runs/sweep_r4_results.csv")))
fp8=next((float(r["best_tps"]) for r in rows if r["name"]=="fp8_base" and r["best_tps"]),1817.3)
print(f"fp8_base(ref)={fp8}  bf16_baseline=1436.9")
for r in rows:
    if r["status"]!="OK" or not r["best_tps"]: print(f"  {r['name']:22s} {r['status']}"); continue
    b=float(r["best_tps"]); print(f"  {r['name']:22s} util={r['gpu_util_mid']:>3}% best={b:8.1f} vs_fp8={(b/fp8-1)*100:+.1f}% vs_bf16={(b/1436.9-1)*100:+.1f}%")
PYEOF
