#!/usr/bin/env bash
# FP4 오프라인양자화 10-method sweep — 사전양자화 NVFP4 체크포인트 + 회전(SpinQuant=란초스②) + 조합.
# 각 방법: boot → tps(깨끗) → logprob → 분포동등 게이트(vs bf16) → pgid 정밀종료.
set -uo pipefail; set +B; cd "$(dirname "$0")"
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
PORT=8021; LOGD=runs; CSV=runs/fp4_results.csv
echo "name,model,status,gpu_util,tps_r1,tps_r2,best_tps,token_match,max_logprob_diff,ppl_rel,gate" > $CSV
REDFP4=RedHatAI/Llama-3.1-70B-Instruct-NVFP4
REDA16=RedHatAI/Llama-3.1-70B-Instruct-NVFP4A16
SPIN=daslab-testing/Llama-3.1-70B-Instruct-spinquantR1R2R4-nvfp4a16
SPEC='--speculative-config {"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
# name|model|gpus|flags
CONFIGS=(
  "w4a4|$REDFP4|0,1,2,3|"
  "w4a16|$REDA16|0,1,2,3|"
  "spinquant_rot|$SPIN|0,1,2,3|"
  "w4a4_kvfp8|$REDFP4|0,1,2,3|--kv-cache-dtype fp8"
  "w4a4_spec|$REDFP4|0,1,2,3|$SPEC"
  "w4a16_spec|$REDA16|0,1,2,3|$SPEC"
  "spinquant_spec|$SPIN|0,1,2,3|$SPEC"
  "w4a4_tp8|$REDFP4|0,1,2,3,4,5,6,7|"
  "fp8_ref|meta-llama/Llama-3.1-70B-Instruct|0,1,2,3|--quantization fp8"
  "bf16_ref|meta-llama/Llama-3.1-70B-Instruct|0,1,2,3|"
)
wgf(){ for i in $(seq 1 50); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
gate(){ # $1=lp_file -> echo "match diff rel pass"
  $PY - "$1" <<'PYEOF'
import json,math,sys,statistics as st
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
run(){ local name=$1 model=$2 gpus=$3 flags=$4; local tp=$(echo $gpus|tr ',' '\n'|wc -l); local slog=$LOGD/serve_fp4_${name}.log
  wgf; echo "[boot] $name ($model)"
  CUDA_VISIBLE_DEVICES=$gpus HF_HOME=/raid/hf_cache setsid $VBIN serve $model --tensor-parallel-size $tp --port $PORT --gpu-memory-utilization 0.85 --max-model-len 4096 $flags > $slog 2>&1 &
  local L=$!; local ok=0
  for i in $(seq 1 110); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }; grep -qiE "Traceback \(most|AssertionError|RuntimeError|unrecognized arg|ValueError|out of memory" $slog && break; sleep 5; done
  if [ $ok -ne 1 ]; then echo "$name,$model,BOOT_FAIL,,,,,,,," >>$CSV; echo "  [FAIL] $(grep -iE 'error|valueerror|runtime|no such' $slog|tail -1|cut -c1-80)"; kill -KILL -- -$L 2>/dev/null; wgf; return; fi
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $model --conc 24 --ptok 2000 --mtok 128 --reqs 32 --tag W --salt w >/dev/null 2>&1
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $model --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r1 > $LOGD/b_${name}_1.txt 2>&1 &
  local B=$!; sleep 6; local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0|tr -d ' '); wait $B
  $PY bench_unique.py --base http://127.0.0.1:$PORT --model $model --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag $name --salt r2 > $LOGD/b_${name}_2.txt 2>&1
  local t1=$(grep -oE "gen_tps=[0-9.]+" $LOGD/b_${name}_1.txt|cut -d= -f2); local t2=$(grep -oE "gen_tps=[0-9.]+" $LOGD/b_${name}_2.txt|cut -d= -f2)
  local best=$(echo -e "${t1:-0}\n${t2:-0}"|sort -rn|head -1)
  $PY collect_logprobs.py --base http://127.0.0.1:$PORT --model $model --mtok 64 --out $LOGD/lp_${name}.json >/dev/null 2>&1
  read mm md mr gg <<< "$(gate $LOGD/lp_${name}.json)"
  echo "$name,$model,OK,$util,$t1,$t2,$best,$mm,$md,$mr,$gg" >>$CSV
  echo "  [OK] $name util=$util% best=$best gate=$gg (match=$mm% diff=$md rel=$mr)"
  kill -KILL -- -$L 2>/dev/null; wgf
}
echo "===== FP4 10-method sweep ====="
for c in "${CONFIGS[@]}"; do IFS='|' read -r n m g f <<< "$c"; run "$n" "$m" "$g" "$f"; done
echo "===== 완료 ====="
$PY - <<'PYEOF'
import csv
rows=list(csv.DictReader(open("runs/fp4_results.csv")))
bf=next((float(r["best_tps"]) for r in rows if r["name"]=="bf16_ref" and r["best_tps"]),1436.9)
fp8=next((float(r["best_tps"]) for r in rows if r["name"]=="fp8_ref" and r["best_tps"]),1817.3)
print(f"bf16={bf} fp8={fp8}")
for r in rows:
    if r["status"]!="OK" or not r["best_tps"]: print(f"  {r['name']:16s} {r['status']}"); continue
    b=float(r["best_tps"]); print(f"  {r['name']:16s} best={b:7.1f} vs_fp8={(b/fp8-1)*100:+5.1f}% vs_bf16={(b/bf-1)*100:+6.1f}% gate={r['gate']}(match{r['token_match']}% diff{r['max_logprob_diff']} rel{r['ppl_rel']})")
PYEOF
