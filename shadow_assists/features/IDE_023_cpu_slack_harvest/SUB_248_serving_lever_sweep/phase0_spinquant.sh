#!/usr/bin/env bash
set -uo pipefail; set +B; cd "$(dirname "$0")"
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
SPIN=daslab-testing/Llama-3.1-70B-Instruct-spinquantR1R2R4-nvfp4a16
PORT=8023; LOGD=runs
wgf(){ for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
wgf; echo "[boot] SpinQuant TP=1 (GPU0)"
CUDA_VISIBLE_DEVICES=0 HF_HOME=/raid/hf_cache setsid $VBIN serve $SPIN --tensor-parallel-size 1 --port $PORT --gpu-memory-utilization 0.85 --max-model-len 4096 > $LOGD/serve_spin_tp1.log 2>&1 &
L=$!
for i in $(seq 1 110); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo READY; break; }; grep -qiE "Traceback \(most|AssertionError|RuntimeError|unrecognized arg|ValueError|out of memory|NotImplementedError" $LOGD/serve_spin_tp1.log && { echo "BOOT_FAIL"; grep -iE "error|notimpl|valueerror|runtime" $LOGD/serve_spin_tp1.log|tail -2; kill -KILL -- -$L 2>/dev/null; exit 1; }; sleep 5; done
echo "[tps(TP=1, 정보용)]"; $PY bench_unique.py --base http://127.0.0.1:$PORT --model $SPIN --conc 8 --ptok 1500 --mtok 128 --reqs 16 --tag W --salt w >/dev/null 2>&1
$PY bench_unique.py --base http://127.0.0.1:$PORT --model $SPIN --conc 8 --ptok 1500 --mtok 256 --reqs 48 --tag spin --salt r 2>&1 | grep -oE "gen_tps=[0-9.]+"
echo "[정확도 logprob]"; $PY collect_logprobs.py --base http://127.0.0.1:$PORT --model $SPIN --mtok 64 --out $LOGD/lp_spinquant.json
kill -KILL -- -$L 2>/dev/null; wgf
echo "[게이트 — SpinQuant(회전) vs bf16, plain-W4A4 와 비교]"
$PY - <<'PYEOF'
import json,math,statistics as st
a=json.load(open("runs/lp_bf16.json"))
def cmp(f):
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
    return m/tot*100, md, max(rels) if rels else 9
import os
for name,f in [("SpinQuant(회전)","runs/lp_spinquant.json"),("plain-W4A4","runs/lp_w4a4.json"),("W4A4+spec","runs/lp_w4a4_spec.json"),("FP8","runs/lp_fp8.json")]:
    if os.path.exists(f):
        mm,md,mr=cmp(f); g="PASS" if (md<=0.5 and mr<=0.1) else "FAIL"
        print(f"  {name:16s} match={mm:5.1f}% max_logprob_diff={md:.4f} ppl_rel={mr:.4f} → {g}")
PYEOF
