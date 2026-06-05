#!/usr/bin/env bash
# B1 Phase A4-prod sample10 — boot a fresh backend in the requested mode
# and capture full generation text for 10 deterministic prompts → byte-equal
# cross-check across baseline vs avx512_prod.
set -uo pipefail
MODE="${1:?usage: run_sample10.sh baseline|avx512_prod}"
cd /workspace/host_vllm_hybrid

PY=/workspace/vllm_dev_prj/bin/python
VBIN=/workspace/vllm_dev_prj/bin/vllm
PORT=8002
GPUS="4,5"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
PARQ=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet

POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b1_e2e
LOGD="$POC_DIR/_logs"

if [ "$MODE" = "baseline" ]; then
  NAT_FLAG=0
elif [ "$MODE" = "avx512_prod" ]; then
  NAT_FLAG=1
else
  echo "unknown mode: $MODE"; exit 2
fi

SAMPLE_OUT="$POC_DIR/llama8b_${MODE}_prod.sample10.jsonl"
BOOT_LOG="$LOGD/boot_${MODE}_sample10.log"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

USED45=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F',' '$1==4||$1==5 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
if [ "${USED45:-0}" -gt 4000 ]; then
  log "ABORT — GPU 4,5 busy: ${USED45} MiB"; exit 1
fi
log "pre-check OK — GPU 4,5 free (used=${USED45} MiB)"

log "=== sample10 $MODE boot (NATIVE=$NAT_FLAG) ==="
CUDA_VISIBLE_DEVICES=$GPUS \
  VLLM_USE_AVX512_DETOK_INC=0 \
  VLLM_USE_AVX512_DETOK_NATIVE=$NAT_FLAG \
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
echo $PID > "$LOGD/${MODE}_sample10.pid"
log "PID=$PID  log=$BOOT_LOG"

# wait_ready
WAIT_READY_MAX=240
T_START=$(date +%s)
READY=0
for i in $(seq 1 $WAIT_READY_MAX); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1; BOOT_SEC=$(( $(date +%s) - T_START ))
    log "READY in ${BOOT_SEC}s"; break
  fi
  sleep 1
done
[ "$READY" = "1" ] || { log "TIMEOUT"; tail -50 "$BOOT_LOG"; exit 1; }

INIT_FAILED=$(grep -c "init failed" "$BOOT_LOG" || true)
log "init-failed warnings: $INIT_FAILED"

log "=== capture sample10 → $SAMPLE_OUT ==="
PYTHONPATH=/workspace/host_vllm_hybrid \
  "$PY" - <<PYEOF
import json, time, hashlib
import httpx
import pyarrow.parquet as pq

rows = pq.read_table("$PARQ").to_pylist()
SAMPLE_IDX = [0, 7, 13, 22, 31, 47, 58, 79, 100, 153]
selected = [rows[i] for i in SAMPLE_IDX]
out = []
with httpx.Client(timeout=600.0) as cl:
    for idx, rec in zip(SAMPLE_IDX, selected):
        t0 = time.perf_counter()
        payload = {
            "model": "$MODEL", "prompt": rec["raw_text"],
            "max_tokens": 256, "temperature": 0.0, "top_p": 1.0,
            "seed": 1234, "stream": False,
        }
        r = cl.post("http://127.0.0.1:$PORT/v1/completions", json=payload)
        wall = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            out.append({"idx": idx, "ok": False, "error": r.text[:200]})
            continue
        j = r.json()
        text = j["choices"][0]["text"]
        rec_out = {
            "idx": idx, "ok": True,
            "wall_ms": round(wall, 2),
            "prompt_chars": len(rec["raw_text"]),
            "completion_tokens": j.get("usage", {}).get("completion_tokens"),
            "text_bytes_len": len(text.encode("utf-8")),
            "text_sha": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
        }
        out.append(rec_out)
        print(f"  idx={idx} ok wall={round(wall,1)}ms tok={rec_out['completion_tokens']} sha={rec_out['text_sha'][:16]}")
with open("$SAMPLE_OUT", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(out)} records → $SAMPLE_OUT")
PYEOF

# kill backend
log "=== kill backend pid=$PID ==="
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
[ -n "$PGID" ] && kill -9 -"$PGID" 2>/dev/null
kill -9 "$PID" 2>/dev/null
sleep 3
for orphan in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{print $1}' | sort -u); do
  if [ -n "$orphan" ]; then
    cmd=$(cat /proc/$orphan/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmd" | grep -qE "VLLM|vllm.*serve|EngineCore"; then
      log "kill orphan $orphan"; kill -9 "$orphan" 2>/dev/null
    fi
  fi
done

for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F',' '$1==4||$1==5 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
  if [ "${used:-1}" -lt 1000 ]; then
    log "GPU 4,5 freed (used=${used} MiB)"; break
  fi
  sleep 3
done
log "=== sample10 $MODE DONE ==="
