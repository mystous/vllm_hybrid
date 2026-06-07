#!/usr/bin/env bash
# Usage: run_measure.sh A|B <port> <gpu_count_for_report>
set -euo pipefail
LABEL=$1
PORT=$2
NGPU=$3
DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/moe_offload
MODEL=$(grep -oE "MODEL_NAME=.*" $DIR/MODEL_NAME.env | cut -d= -f2-)
OUT=$DIR/result_${LABEL}.json

# Health check
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then break; fi
  sleep 2
done
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "[$LABEL] /v1/models not ready after 60s"; exit 1
fi

echo "[$LABEL] /v1/models OK at port $PORT — start measurement"
echo "[$LABEL] $(date)"
echo "[$LABEL] gpu before:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# CPU/GPU snapshot start
GPUSNAP=$DIR/${LABEL}_gpu_snap.csv
CPUSNAP=$DIR/${LABEL}_cpu_snap.csv
(
  echo "ts,gpu_idx,mem_used_MiB,util_pct"
  while true; do
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      | awk -v ts="$(date +%s)" '{print ts","$0}'
    sleep 2
  done
) > $GPUSNAP 2>/dev/null &
GPID=$!
(
  echo "ts,cpu_pct,mem_kb"
  while true; do
    top -bn1 | awk -v ts="$(date +%s)" '/^%Cpu/ {cpu=$2+$4} /MiB Mem/ {mem=$8} END {print ts","cpu","mem}'
    sleep 2
  done
) > $CPUSNAP 2>/dev/null &
CPID=$!

# Actual measurement
/workspace/sglang_kt_prj/bin/python $DIR/short_client.py \
  --url http://127.0.0.1:$PORT/v1 \
  --model "$MODEL" \
  --n 20 --max-tokens 256 --concurrency 8 \
  --seed 0 \
  --out $OUT

# Stop snapshots
kill $GPID $CPID 2>/dev/null || true

echo "[$LABEL] gpu after:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Inject NGPU into result
/workspace/sglang_kt_prj/bin/python - <<PY
import json
p = "$OUT"
d = json.load(open(p))
d["summary"]["gpu_count"] = $NGPU
d["summary"]["tps_per_gpu"] = d["summary"]["decode_tps"] / $NGPU if $NGPU > 0 else 0
d["summary"]["label"] = "$LABEL"
json.dump(d, open(p, "w"), indent=2)
print("RESULT", json.dumps(d["summary"], indent=2))
PY

echo "[$LABEL] done at $(date)"
