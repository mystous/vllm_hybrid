#!/usr/bin/env bash
# Round1 측정: GPTQ-NVFP4 W4A4 ±spec 70B vs RedHat RTN W4A4 — tps + 분포동등 게이트.
# 핵심: GPTQ(오차역전파)가 W4A4+spec 게이트(0.128 FAIL)를 PASS로 구제하는가.
set -uo pipefail; set +B
SWEEP=/home/mystous/vllm_hybrid/shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_248_serving_lever_sweep
cd "$SWEEP"
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
PORT=8022; LOGD=../SUB_250_ratedistortion_mixedprec/step4/runs; CSV=$LOGD/gptq_results.csv
GPUS=${GPUS:-4,5,6,7}
mkdir -p $LOGD
echo "name,model,status,gpu_util,tps_r1,tps_r2,best_tps,token_match,max_logprob_diff,ppl_rel,gate" > $CSV
GPTQ=/raid/hf_cache/gptq_nvfp4_70b
SPEC='--speculative-config {"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
CONFIGS=(
  "gptq|$GPTQ|$GPUS|"
  "gptq_spec|$GPTQ|$GPUS|$SPEC"
)
g0=$(echo $GPUS|cut -d, -f1)
wgf(){ for i in $(seq 1 50); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
gate(){ $PY - "$1" <<'PYEOF'
import json,math,sys
a=json.load(open("runs/lp_bf16.json"))
try: b=json.load(open(sys.argv[1]))
except: print("NA NA NA NA"); sys.exit()
tot=m=0; md=0; rels=[]
for x,y in zip(a,b):
    tx,ty,lx,ly=x["tokens"],y["tokens"],x["logprobs"],y["logprobs"]; n=min(len(tx),len(ty)); tot+=n
    for i in range(n):
        if tx[i]==ty[i]:
            m+=1
            if i<len(lx) and i<len(ly): md=max(md,abs(lx[i]-ly[i]))
        else: break
    def ppl(l): return math.exp(-sum(l)/len(l)) if l else float('nan')
    pa,pb=ppl(lx),ppl(ly)
    if pa==pa and pb==pb and pa>0: rels.append(abs(pb-pa)/pa)
mr=max(rels) if rels else 9
g="PASS" if (md<=0.5 and mr<=0.1) else "FAIL"
print(f"{m/tot*100:.1f} {md:.4f} {mr:.4f} {g}")
PYEOF
}
run(){ local name=$1 model=$2 gpus=$3 flags=$4; local tp=$(echo $gpus|tr ',' '\n'|wc -l); local slog=$LOGD/serve_${name}.log
  wgf; echo "[boot] $name ($model)"
  CUDA_VISIBLE_DEVICES=$gpus HF_HOME=/raid/hf_cache setsid $VBIN serve $model --tensor-parallel-size $tp --port $PORT --gpu-memory-utilization 0.85 --max-model-len 4096 $flags > $slog 2>&1 &
  local L=$!; local ok=0
  for i in $(seq 1 130); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }; grep -qiE "Traceback \(most|AssertionError|RuntimeError|unrecognized arg|ValueError|out of memory" $slog && break; sleep 5; done
  if [ $ok -ne 1 ]; then echo "$name,$model,BOOT_FAIL,,,,,,,," >>$CSV; echo "  [FAIL] $(grep -iE 'error|valueerror|runtime|no such' $slog|tail -1|cut -c1-90)"; kill -KILL -- -$L 2>/dev/null; wgf; return; fi
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $model --conc 24 --ptok 2000 --mtok 128 --reqs 32 --tag W --salt w >/dev/null 2>&1
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $model --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r1 > $LOGD/b_${name}_1.txt 2>&1 &
  local B=$!; sleep 6; local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g0|tr -d ' '); wait $B
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $model --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r2 > $LOGD/b_${name}_2.txt 2>&1
  local t1=$(grep -oE "gen_tps=[0-9.]+" $LOGD/b_${name}_1.txt|cut -d= -f2); local t2=$(grep -oE "gen_tps=[0-9.]+" $LOGD/b_${name}_2.txt|cut -d= -f2)
  local best=$(echo -e "${t1:-0}\n${t2:-0}"|sort -rn|head -1)
  $PY collect_logprobs.py --base http://127.0.0.1:$PORT --model $model --mtok 64 --out $LOGD/lp_${name}.json >/dev/null 2>&1
  read mm md mr gg <<< "$(gate $LOGD/lp_${name}.json)"
  echo "$name,$model,OK,$util,$t1,$t2,$best,$mm,$md,$mr,$gg" >>$CSV
  echo "  [OK] $name util=$util% best=$best gate=$gg (match=$mm% diff=$md rel=$mr)"
  kill -KILL -- -$L 2>/dev/null; wgf
}
echo "===== Round1 GPTQ-NVFP4 sweep (GPU $GPUS) ====="
for c in "${CONFIGS[@]}"; do IFS='|' read -r n m g f <<< "$c"; run "$n" "$m" "$g" "$f"; done
echo "===== 완료 ====="
cat $CSV