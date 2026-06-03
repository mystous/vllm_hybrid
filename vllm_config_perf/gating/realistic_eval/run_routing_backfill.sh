#!/usr/bin/env bash
# host 실행. 셀마다 백엔드 /metrics 스냅샷 → 셀 측정 → 스냅샷 → summ에 accept α + 라우팅 분배 patch.
set -uo pipefail
export KUBECONFIG=/home/mystous/.kube/config
MODEL="${MODEL:?}"; TAG="${TAG:?}"
NS=llm-d; CONT=heuristic_zhukovsky
RE_C=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval
RE_H=/home/mystous/vllm_hybrid/vllm_config_perf/gating/realistic_eval
OUTC=$RE_C/runs/routing_llmd_20260603; OUTH=$RE_H/runs/routing_llmd_20260603
IN=$RE_C/runs/tput_t1t3_20260602/sampled_prompts.parquet
P=$RE_H/pod_metrics.py
LOG=/home/mystous/k8s_llmd/routing_run.log
snap(){ local pod=$(kubectl -n $NS get pod -l method=$1 -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  kubectl -n $NS exec "$pod" -c vllm -- bash -lc 'curl -s localhost:8000/metrics' 2>/dev/null | python3 "$P" 2>/dev/null || echo "0 0 0"; }
cell(){ local cond=$1; shift
  read sa sd sr < <(snap suffix); read va vd vr < <(snap vanilla)
  sudo docker exec $CONT bash -c "cd $RE_C && PYTHONPATH=. /workspace/vllm_dev_prj/bin/python throughput_runner.py --in $IN --method llm-d --model '$MODEL' --model-tag '$TAG' --host 172.20.0.2 --port 30080 --max-tokens 8192 --concurrency 32 $* --out $OUTC/summ_${TAG}_llm-d_${cond}.json --raw $OUTC/per_request_raw.jsonl" >>"$LOG" 2>&1
  read sa2 sd2 sr2 < <(snap suffix); read va2 vd2 vr2 < <(snap vanilla)
  sudo python3 "$P" --patch "$OUTH/summ_${TAG}_llm-d_${cond}.json" $sa $sd $sr $va $vd $vr $sa2 $sd2 $sr2 $va2 $vd2 $vr2 >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] cell $cond done"; }
echo "[backfill] $TAG start $(date +%H:%M:%S)"
for C in sharegpt swebench humaneval mbpp wildchat lmsys; do cell $C --corpus $C; done
cell mix --limit 500 --shuffle
echo "[backfill] $TAG ALL DONE $(date +%H:%M:%S)"
