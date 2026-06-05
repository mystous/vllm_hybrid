#!/usr/bin/env bash
# Quick re-run: boot only, no bench — just the 10-case sample for the chosen mode.
# Used to validate that intra-mode (baseline vs baseline_re) divergence exists,
# i.e. that any mismatch observed across modes is BF16/TP nondeterminism rather
# than an EXCLUSIVE wire-in correctness regression.
set -uo pipefail
MODE="${1:?usage: run_v3_sample_only.sh baseline|double|exclusive}"
TAG="${2:-re}"   # output filename suffix: ..._${TAG}.jsonl
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT=8002
GPUS="4,5"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
PARQ=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet

POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b1_e2e
LOGD="$POC_DIR/_logs"
mkdir -p "$LOGD"

case "$MODE" in
  baseline)
    NAT_FLAG=0; EXC_FLAG=0 ;;
  double)
    NAT_FLAG=1; EXC_FLAG=0 ;;
  exclusive)
    NAT_FLAG=0; EXC_FLAG=1 ;;
  *)
    echo "unknown mode: $MODE"; exit 2 ;;
esac

BOOT_LOG="$LOGD/boot_${MODE}_${TAG}.log"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

USED45=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F',' '$1==4||$1==5 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
if [ "${USED45:-0}" -gt 4000 ]; then
  log "ABORT — GPU 4,5 already busy: used=${USED45} MiB > 4000"
  exit 1
fi
log "pre-check OK — GPU 4,5 free (used=${USED45} MiB)"

log "=== boot $MODE ($TAG) (port=$PORT, gpus=$GPUS, NATIVE=$NAT_FLAG, EXCLUSIVE=$EXC_FLAG) ==="
CUDA_VISIBLE_DEVICES=$GPUS \
  VLLM_USE_AVX512_DETOK_INC=0 \
  VLLM_USE_AVX512_DETOK_NATIVE=$NAT_FLAG \
  VLLM_USE_AVX512_DETOK_EXCLUSIVE=$EXC_FLAG \
  VLLM_AVX512_DETOK_VERIFY=0 \
  ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid "$VBIN" serve "$MODEL" \
    --tensor-parallel-size 2 --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --allow-deprecated-quantization \
    > "$BOOT_LOG" 2>&1 < /dev/null &
PID=$!
log "PID=$PID  log=$BOOT_LOG"

WAIT_READY_MAX=240
T_START=$(date +%s)
READY=0
for i in $(seq 1 $WAIT_READY_MAX); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    BOOT_SEC=$(( $(date +%s) - T_START ))
    log "READY in ${BOOT_SEC}s"
    break
  fi
  sleep 1
done
if [ "$READY" != "1" ]; then
  log "TIMEOUT after ${WAIT_READY_MAX}s — abort"
  tail -50 "$BOOT_LOG"; exit 1
fi

SAMPLE_OUT="$POC_DIR/llama8b_${MODE}_v3_${TAG}.sample10.jsonl"
log "=== sample10 ($TAG): → $SAMPLE_OUT ==="
PYTHONPATH=/workspace/host_vllm_hybrid \
  "$PY" - <<PYEOF
import json, time
import httpx
import pyarrow.parquet as pq

rows = pq.read_table("$PARQ").to_pylist()
SAMPLE_IDX = [0, 7, 13, 22, 31, 47, 58, 79, 100, 153]
rows = [rows[i] for i in SAMPLE_IDX]
out = []
with httpx.Client(timeout=600.0) as cl:
    for idx, rec in zip(SAMPLE_IDX, rows):
        t0 = time.perf_counter()
        payload = {"model": "$MODEL", "prompt": rec["raw_text"],
                   "max_tokens": 256, "temperature": 0.0, "top_p": 1.0,
                   "seed": 1234, "stream": False}
        r = cl.post("http://127.0.0.1:$PORT/v1/completions", json=payload)
        wall = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            out.append({"idx": idx, "ok": False, "error": r.text[:200]})
            continue
        j = r.json()
        text = j["choices"][0]["text"]
        out.append({"idx": idx, "ok": True, "wall_ms": round(wall, 2),
                    "completion_tokens": j.get("usage", {}).get("completion_tokens"),
                    "text_bytes_len": len(text.encode("utf-8")),
                    "text_sha": __import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
                    "text": text})
        print(f"  idx={idx} sha={out[-1]['text_sha'][:16]}")
with open("$SAMPLE_OUT", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(out)} to $SAMPLE_OUT")
PYEOF

# kill backend
log "=== kill backend pid=$PID ==="
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
if [ -n "$PGID" ]; then kill -9 -"$PGID" 2>/dev/null; fi
kill -9 "$PID" 2>/dev/null
sleep 3
for orphan_pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u); do
  if [ -n "$orphan_pid" ]; then
    cmd=$(cat /proc/$orphan_pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmd" | grep -qE "VLLM|vllm.*serve|EngineCore"; then
      log "kill orphan $orphan_pid"; kill -9 "$orphan_pid" 2>/dev/null
    fi
  fi
done
for i in $(seq 1 30); do
  used45=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$1==4||$1==5 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
  if [ "${used45:-1}" -lt 1000 ]; then log "GPU 4,5 freed (used=${used45} MiB)"; break; fi
  sleep 3
done
log "=== DONE $MODE $TAG ==="
