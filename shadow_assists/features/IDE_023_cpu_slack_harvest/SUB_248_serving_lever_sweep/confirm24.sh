#!/usr/bin/env bash
set -uo pipefail; set +B; cd "$(dirname "$0")"
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
PORT=8024; LOGD=runs
SPEC='--speculative-config {"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
wgf(){ for i in $(seq 1 50); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
collect(){ # name model flags
  local name=$1 model=$2 flags=$3; wgf; echo "[boot] $name"
  CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HOME=/raid/hf_cache setsid $VBIN serve $model --tensor-parallel-size 4 --port $PORT --gpu-memory-utilization 0.85 --max-model-len 4096 $flags > $LOGD/serve_c24_${name}.log 2>&1 &
  local L=$!
  for i in $(seq 1 110); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; grep -qiE "Traceback \(most|AssertionError|RuntimeError|out of memory" $LOGD/serve_c24_${name}.log && { echo "BOOT_FAIL $name"; kill -KILL -- -$L 2>/dev/null; wgf; return 1; }; sleep 5; done
  $PY collect_logprobs24.py --base http://127.0.0.1:$PORT --model $model --mtok 80 --out $LOGD/lp24_${name}.json
  kill -KILL -- -$L 2>/dev/null; wgf
}
collect bf16 meta-llama/Llama-3.1-70B-Instruct ""
collect w4a4 RedHatAI/Llama-3.1-70B-Instruct-NVFP4 ""
collect w4a4spec RedHatAI/Llama-3.1-70B-Instruct-NVFP4 "$SPEC"
echo "=== 24-프롬프트 게이트 (vs bf16, mean+max ppl_rel) ==="
$PY - <<'PYEOF'
import json,math,statistics as st
a=json.load(open("runs/lp24_bf16.json"))
def gate(f):
    b=json.load(open(f)); tot=m=0; md=0; rels=[]
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
    return m/tot*100, md, st.mean(rels), max(rels)
for n,f in [("W4A4","runs/lp24_w4a4.json"),("W4A4+spec","runs/lp24_w4a4spec.json")]:
    mm,md,rmean,rmax=gate(f)
    g="PASS" if (md<=0.5 and rmax<=0.1) else ("PASS(mean)" if (md<=0.5 and rmean<=0.1) else "FAIL")
    print(f"  {n:12s} match={mm:.1f}% max_logprob_diff={md:.4f} ppl_rel mean={rmean:.4f} max={rmax:.4f} → {g}")
PYEOF
