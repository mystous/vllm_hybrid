#!/usr/bin/env bash
set -uo pipefail; set +B; cd "$(dirname "$0")"
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
M=meta-llama/Llama-3.1-70B-Instruct; PORT=8021; LOGD=runs
wgf(){ for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
wgf
echo "[boot] mxfp4 70B"
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HOME=/raid/hf_cache setsid $VBIN serve $M --tensor-parallel-size 4 --port $PORT --gpu-memory-utilization 0.85 --max-model-len 4096 --quantization mxfp4 > $LOGD/serve_fp4.log 2>&1 &
L=$!
for i in $(seq 1 100); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo READY; break; }; grep -qiE "Traceback \(most|AssertionError|RuntimeError|unrecognized arg|ValueError" $LOGD/serve_fp4.log && { echo "BOOT_FAIL"; grep -iE "error|valueerror|runtime" $LOGD/serve_fp4.log|tail -2; kill -KILL -- -$L 2>/dev/null; exit 1; }; sleep 5; done
# tps
$PY bench_unique.py --base http://127.0.0.1:$PORT --model $M --conc 24 --ptok 2000 --mtok 128 --reqs 32 --tag W --salt w >/dev/null 2>&1
echo "[tps]"; for r in 1 2; do $PY bench_unique.py --base http://127.0.0.1:$PORT --model $M --conc 24 --ptok 2000 --mtok 256 --reqs 192 --tag fp4 --salt r$r 2>&1 | grep -oE "gen_tps=[0-9.]+"; done
# 정확도 logprob
$PY collect_logprobs.py --base http://127.0.0.1:$PORT --model $M --mtok 64 --out $LOGD/lp_fp4.json
kill -KILL -- -$L 2>/dev/null; wgf
echo "[정확도 게이트 — FP4 vs bf16(70B)]"
$PY - <<'PYEOF'
import json,math,statistics as st
a=json.load(open("runs/lp_bf16.json")); b=json.load(open("runs/lp_fp4.json"))
tot=match=0; maxdiff=0; rels=[]
for x,y in zip(a,b):
    tx,ty,lx,ly=x["tokens"],y["tokens"],x["logprobs"],y["logprobs"]; n=min(len(tx),len(ty)); tot+=n
    for i in range(n):
        if tx[i]==ty[i]: match+=1; 
        if tx[i]==ty[i] and i<len(lx) and i<len(ly): maxdiff=max(maxdiff,abs(lx[i]-ly[i]))
        else: break
    def ppl(l): return math.exp(-sum(l)/len(l)) if l else float('nan')
    pa,pb=ppl(lx),ppl(ly)
    if pa==pa and pb==pb and pa>0: rels.append(abs(pb-pa)/pa)
g=(maxdiff<=0.5) and (max(rels)<=0.1)
print(f"token_match={match/tot*100:.1f}% max_abs_logprob_diff={maxdiff:.4f}(<=0.5) ppl_rel max={max(rels):.4f}(<=0.1)")
print(f"==> FP4 분포동등 게이트: {'PASS ✅ (rotation 불요)' if g else 'FAIL ❌ → eigen-rotation 필요'}")
PYEOF
