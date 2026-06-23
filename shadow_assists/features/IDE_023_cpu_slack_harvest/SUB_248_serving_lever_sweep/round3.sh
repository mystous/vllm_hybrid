#!/usr/bin/env bash
# 라운드3 — (A) FP4 online 부팅 프로브 (B) FP8 가중치 정확도 게이트 (bf16 vs fp8 logprob 분포동등).
set -uo pipefail; set +B
cd "$(dirname "$0")"
PY=/home/mystous/vllm_dev_prj/bin/python; VBIN=/home/mystous/vllm_dev_prj/bin/vllm
MODEL=meta-llama/Llama-3.1-70B-Instruct; PORT=8021; LOGD=runs; mkdir -p $LOGD
wait_gpu_free(){ for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0|tr -d ' '); [ "${u:-0}" -lt 2000 ] && return 0; sleep 3; done; }
boot(){ # $1 name $2 flags ; echo LEAD pid
  local name=$1 flags=$2; local slog=$LOGD/serve_r3_${name}.log
  wait_gpu_free
  CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HOME=/raid/hf_cache \
    setsid $VBIN serve $MODEL --tensor-parallel-size 4 --port $PORT \
      --gpu-memory-utilization 0.85 --max-model-len 4096 $flags > $slog 2>&1 &
  echo $!
}
ready(){ for i in $(seq 1 90); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && return 0; grep -qiE "Traceback|Error:|unrecognized|out of memory|ValueError|RuntimeError|No such|not support" $1 2>/dev/null && return 1; sleep 5; done; return 1; }

echo "=== (A) FP4 online 부팅 프로브 (mxfp4) ==="
L=$(boot fp4_mxfp4 "--quantization mxfp4")
if ready $LOGD/serve_r3_fp4_mxfp4.log; then echo "FP4 BOOT OK (online 지원!)"; else echo "FP4 BOOT FAIL: $(grep -iE 'error|not support|valueerror|no such' $LOGD/serve_r3_fp4_mxfp4.log|tail -1|cut -c1-90)"; fi
kill -KILL -- -$L 2>/dev/null; wait_gpu_free

echo "=== (B) 정확도 게이트: bf16 logprob 수집 ==="
L=$(boot bf16 "")
if ready $LOGD/serve_r3_bf16.log; then
  $PY collect_logprobs.py --base http://127.0.0.1:$PORT --model $MODEL --mtok 64 --out $LOGD/lp_bf16.json
else echo "bf16 boot fail"; fi
kill -KILL -- -$L 2>/dev/null; wait_gpu_free

echo "=== (B) fp8 가중치 logprob 수집 ==="
L=$(boot fp8 "--quantization fp8")
if ready $LOGD/serve_r3_fp8.log; then
  $PY collect_logprobs.py --base http://127.0.0.1:$PORT --model $MODEL --mtok 64 --out $LOGD/lp_fp8.json
else echo "fp8 boot fail"; fi
kill -KILL -- -$L 2>/dev/null; wait_gpu_free

echo "=== 분포동등 비교 (D-ii) ==="
$PY - <<'PYEOF'
import json, math
a=json.load(open("runs/lp_bf16.json")); b=json.load(open("runs/lp_fp8.json"))
tot_match=tot=0; maxdiff=0; ppl_rels=[]
for x,y in zip(a,b):
    tx,ty=x["tokens"],y["tokens"]; lx,ly=x["logprobs"],y["logprobs"]
    n=min(len(tx),len(ty));
    m=sum(1 for i in range(n) if tx[i]==ty[i]); tot_match+=m; tot+=n
    # 공통 prefix(토큰 일치 구간)에서 logprob diff
    for i in range(n):
        if tx[i]==ty[i] and i<len(lx) and i<len(ly):
            maxdiff=max(maxdiff,abs(lx[i]-ly[i]))
        else: break
    def ppl(l): return math.exp(-sum(l)/len(l)) if l else float('nan')
    pa,pb=ppl(lx),ppl(ly)
    if pa==pa and pb==pb and pa>0: ppl_rels.append(abs(pb-pa)/pa)
import statistics as st
print(f"token_match_rate = {tot_match/tot*100:.1f}%  (informational)")
print(f"max_abs_logprob_diff(공통prefix) = {maxdiff:.4f}  (게이트 ≤ 0.5)")
print(f"ppl_rel_diff mean = {st.mean(ppl_rels):.4f} max = {max(ppl_rels):.4f}  (게이트 ≤ 0.1)")
gate = (maxdiff<=0.5) and (max(ppl_rels)<=0.1)
print(f"==> 분포동등 게이트: {'PASS ✅ — FP8 win 유효' if gate else 'FAIL ❌ — 정확도 저하 큼'}")
PYEOF
echo "===== 라운드3 완료 ====="
