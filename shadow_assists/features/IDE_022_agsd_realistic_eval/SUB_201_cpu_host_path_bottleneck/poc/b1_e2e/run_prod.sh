#!/usr/bin/env bash
# B1 AVX-512 detok lever — Phase A4-prod e2e (production wire-in)
# Llama-3.1-8B TP=2, GPU 4,5
# run1: baseline (VLLM_USE_AVX512_DETOK_NATIVE=0; INC=0) — current production
# run2: avx512 prod (VLLM_USE_AVX512_DETOK_NATIVE=1; INC=0 to avoid dual-call)
set -uo pipefail
MODE="${1:?usage: run_prod.sh baseline|avx512_prod}"
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

if [ "$MODE" = "baseline" ]; then
  NAT_FLAG=0
  OUT_JSON="$POC_DIR/llama8b_baseline_prod.json"
  OUT_RAW="$POC_DIR/llama8b_baseline_prod.raw.jsonl"
  BOOT_LOG="$LOGD/boot_baseline_prod.log"
elif [ "$MODE" = "avx512_prod" ]; then
  NAT_FLAG=1
  OUT_JSON="$POC_DIR/llama8b_avx512_prod.json"
  OUT_RAW="$POC_DIR/llama8b_avx512_prod.raw.jsonl"
  BOOT_LOG="$LOGD/boot_avx512_prod.log"
else
  echo "unknown mode: $MODE"; exit 2
fi

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# --- safety: GPU 4,5 must be free at start ---
USED45=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F',' '$1==4||$1==5 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
if [ "${USED45:-0}" -gt 4000 ]; then
  log "ABORT — GPU 4,5 already busy: used=${USED45} MiB > 4000"
  exit 1
fi
log "pre-check OK — GPU 4,5 free (used=${USED45} MiB)"

# --- boot ---
log "=== B1 e2e prod $MODE boot (port=$PORT, gpus=$GPUS, NATIVE=$NAT_FLAG) ==="
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
echo $PID > "$LOGD/${MODE}_prod.pid"
log "PID=$PID  log=$BOOT_LOG"

# --- wait_ready ---
WAIT_READY_MAX=240
T_START=$(date +%s)
READY=0
for i in $(seq 1 $WAIT_READY_MAX); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    BOOT_SEC=$(( $(date +%s) - T_START ))
    log "READY in ${BOOT_SEC}s"
    echo "$BOOT_SEC" > "$LOGD/${MODE}_prod.boot_sec"
    break
  fi
  sleep 1
done
if [ "$READY" != "1" ]; then
  log "TIMEOUT after ${WAIT_READY_MAX}s — abort"
  tail -50 "$BOOT_LOG"
  exit 1
fi

# check no init failed warnings
INIT_FAILED=$(grep -c "init failed" "$BOOT_LOG" || true)
log "init-failed warnings in boot log: $INIT_FAILED"

# --- benchmark: sharegpt 200 prompt × conc=16 × vanilla, stream ---
log "=== bench: sharegpt 200p × conc=16 × vanilla (stream) ==="
PYTHONPATH=/workspace/host_vllm_hybrid \
  "$PY" vllm_config_perf/gating/realistic_eval/throughput_runner.py \
    --in "$PARQ" \
    --method vanilla \
    --model "$MODEL" \
    --model-tag "Llama-3.1-8B-Instruct" \
    --port "$PORT" \
    --max-tokens 8192 \
    --concurrency 16 \
    --limit 200 \
    --corpus sharegpt \
    --out "$OUT_JSON" \
    --raw "$OUT_RAW" \
  2>&1 | tee -a "$LOGD/bench_${MODE}_prod.log"

log "=== bench done → $OUT_JSON ==="

# --- byte-equal sample: 10 deterministic prompts, full text capture ---
SAMPLE_OUT="$POC_DIR/llama8b_${MODE}_prod.sample10.jsonl"
log "=== sample10: capture full text for byte-equal verify → $SAMPLE_OUT ==="
PYTHONPATH=/workspace/host_vllm_hybrid \
  "$PY" - <<PYEOF | tee -a "$LOGD/sample_${MODE}_prod.log"
import json, time
import httpx
import pyarrow.parquet as pq

rows = pq.read_table("$PARQ").to_pylist()
# stable, deterministic 10 prompt selection — same indices for both modes
SAMPLE_IDX = [0, 7, 13, 22, 31, 47, 58, 79, 100, 153]
rows = [rows[i] for i in SAMPLE_IDX]
out = []
with httpx.Client(timeout=600.0) as cl:
    for idx, rec in zip(SAMPLE_IDX, rows):
        t0 = time.perf_counter()
        payload = {
            "model": "$MODEL",
            "prompt": rec["raw_text"],
            "max_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 1234,
            "stream": False,
        }
        r = cl.post("http://127.0.0.1:$PORT/v1/completions", json=payload)
        wall = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            out.append({"idx": idx, "ok": False, "error": r.text[:200]})
            continue
        j = r.json()
        text = j["choices"][0]["text"]
        out.append({
            "idx": idx,
            "ok": True,
            "wall_ms": round(wall, 2),
            "prompt_chars": len(rec["raw_text"]),
            "completion_tokens": j.get("usage", {}).get("completion_tokens"),
            "text_bytes_len": len(text.encode("utf-8")),
            "text_sha": __import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
        })
        print(f"  idx={idx} ok wall={round(wall,1)}ms tokens={out[-1]['completion_tokens']} sha={out[-1]['text_sha'][:16]}")
with open("$SAMPLE_OUT", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(out)} sample records to $SAMPLE_OUT")
PYEOF

# --- kill backend ---
log "=== kill backend pid=$PID ==="
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
if [ -n "$PGID" ]; then
  kill -9 -"$PGID" 2>/dev/null
fi
kill -9 "$PID" 2>/dev/null

# orphan VLLM::Worker on GPU 4,5
sleep 3
for orphan_pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null \
                    | awk -F',' '{print $1}' | sort -u); do
  if [ -n "$orphan_pid" ]; then
    cmd=$(cat /proc/$orphan_pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmd" | grep -qE "VLLM|vllm.*serve|EngineCore"; then
      log "kill orphan $orphan_pid: $cmd"
      kill -9 "$orphan_pid" 2>/dev/null
    fi
  fi
done

# --- wait GPU 4,5 free ---
for i in $(seq 1 30); do
  used45=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | awk -F',' '$1==4||$1==5 {gsub(/ /,"",$2); s+=$2} END{print s+0}')
  if [ "${used45:-1}" -lt 1000 ]; then
    log "GPU 4,5 freed (used=${used45} MiB)"
    break
  fi
  sleep 3
done

nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits \
  | awk -F',' '$1==4||$1==5' | tee "$LOGD/${MODE}_prod.gpu_after.txt"

log "=== DONE prod $MODE ==="
