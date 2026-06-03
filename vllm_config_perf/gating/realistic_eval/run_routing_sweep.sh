#!/usr/bin/env bash
# 실험2: llm-d 부하 스윕. 한 모델(이미 배포됨)에 대해 CONCS의 각 conc × 7 condition 측정.
# method 태그 = llm-d-c<conc>. backfill(accept α + 라우팅 분배) 적용.
set -uo pipefail
export KUBECONFIG=/home/mystous/.kube/config
MODEL="${MODEL:?}"; TAG="${TAG:?}"; CONCS="${CONCS:-8 64}"
NS=llm-d; CONT=heuristic_zhukovsky
RE_C=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval
RE_H=/home/mystous/vllm_hybrid/vllm_config_perf/gating/realistic_eval
OUTC=$RE_C/runs/routing_llmd_20260603; OUTH=$RE_H/runs/routing_llmd_20260603
IN=$RE_C/runs/tput_t1t3_20260602/sampled_prompts.parquet
P=$RE_H/pod_metrics.py; LOG=/home/mystous/k8s_llmd/routing_run.log
snap(){ local pod=$(kubectl -n $NS get pod -l method=$1 -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  kubectl -n $NS exec "$pod" -c vllm -- bash -lc 'curl -s localhost:8000/metrics' 2>/dev/null | python3 "$P" 2>/dev/null || echo "0 0 0"; }
cell(){ local cond=$1 conc=$2; shift 2; local m="llm-d-c${conc}"
  read sa sd sr < <(snap suffix); read va vd vr < <(snap vanilla)
  sudo docker exec $CONT bash -c "cd $RE_C && PYTHONPATH=. /workspace/vllm_dev_prj/bin/python throughput_runner.py --in $IN --method $m --model '$MODEL' --model-tag '$TAG' --host 172.20.0.2 --port 30080 --max-tokens 8192 --concurrency $conc $* --out $OUTC/summ_${TAG}_${m}_${cond}.json --raw $OUTC/per_request_raw.jsonl" >>"$LOG" 2>&1
  read sa2 sd2 sr2 < <(snap suffix); read va2 vd2 vr2 < <(snap vanilla)
  sudo python3 "$P" --patch "$OUTH/summ_${TAG}_${m}_${cond}.json" $sa $sd $sr $va $vd $vr $sa2 $sd2 $sr2 $va2 $vd2 $vr2 >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] cell $cond c$conc done"; }
echo "[sweep] $TAG concs=[$CONCS] start $(date +%H:%M:%S)"
for conc in $CONCS; do
  for C in sharegpt swebench humaneval mbpp wildchat lmsys; do cell $C $conc --corpus $C; done
  cell mix $conc --limit 500 --shuffle
  echo "[sweep] $TAG conc=$conc DONE"
done
echo "[sweep] $TAG ALL DONE $(date +%H:%M:%S)"
