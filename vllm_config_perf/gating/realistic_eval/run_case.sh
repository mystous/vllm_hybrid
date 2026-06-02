#!/usr/bin/env bash
# TSK_042 단일 케이스 — (MODEL, METHOD) 백엔드 1회 부팅 → CONDITIONS 연속 측정 → kill.
# 캡처 메트릭: output_tps / TTFT·TPOT p50,p99 / accept α / GPU util,mem / CPU% / per-corpus reqtps.
#
# 설정값은 전부 env 로 주입(복붙용). 예:
#   MODEL=Qwen/Qwen2.5-7B-Instruct METHOD=suffix \
#   SAMPLED=$PWD/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet \
#   OUTDIR=$PWD/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602 \
#   bash vllm_config_perf/gating/realistic_eval/run_case.sh
set -uo pipefail
cd /workspace/host_vllm_hybrid
PY=${PY:-/workspace/vllm_dev_prj/bin/python}
VBIN=${VBIN:-/workspace/vllm_dev_prj/bin/vllm}
RE=vllm_config_perf/gating/realistic_eval

# ── 설정값 (env override 가능) ──────────────────────────────────────────────
MODEL="${MODEL:?MODEL 필요 (예: Qwen/Qwen2.5-7B-Instruct)}"
METHOD="${METHOD:-vanilla}"                 # vanilla | suffix | ngram
CONDITIONS="${CONDITIONS:-sharegpt swebench humaneval mbpp wildchat lmsys mix}"
SAMPLED="${SAMPLED:?SAMPLED parquet 경로 필요}"
OUTDIR="${OUTDIR:?OUTDIR 필요}"
CONC="${CONC:-32}"                          # 동시성
MAXTOK="${MAXTOK:-8192}"                     # 출력 토큰 상한 (vanilla/suffix)
NGRAM_MAXTOK="${NGRAM_MAXTOK:-$MAXTOK}"      # ngram 전용 출력 상한 (느려서 축소 권장, 예 2048)
MML="${MML:-16384}"                          # max-model-len (실입력≤4096 + 출력)
SKIP_EXISTING="${SKIP_EXISTING:-0}"          # 1=summ 이미 있으면 해당 조건/phase 건너뜀(resume)
LIMIT="${LIMIT:-500}"                        # mix 조건 샘플 상한 (corpus 격리는 전량)
PORT="${PORT:-8001}"
GMU="${GMU:-0.85}"                           # gpu-memory-utilization
STREAM="${STREAM:-1}"                        # 1=TTFT/TPOT 측정(stream) / 0=비스트리밍
TP="${TP:-auto}"                             # auto=헤드수%8 판정 / 4 / 8
TAG="${TAG:-$(basename "$MODEL")}"
RESULTS_MD="${RESULTS_MD:-shadow_assists/features/IDE_022_agsd_realistic_eval/TSK_042_realistic_workload_oracle/RESULTS.md}"
AGG="${AGG:-1}"                               # 1=셀-당 build_throughput_table 호출(cells/+parquet+SUMMARY+RESULTS 갱신) / 0=skip
# 엔진 lifecycle:
#   BOOT=1 (기본) 스크립트가 엔진 부팅 / BOOT=0 이미 PORT 에 떠 있는 엔진에 attach(부팅 안 함)
#   KEEP=1 측정 후 엔진 유지(kill 안 함) / KEEP=0 측정 후 kill
#   기본: BOOT=1→KEEP=0(부팅 후 kill=스위치) , BOOT=0→KEEP=1(남의 엔진 유지)
BOOT="${BOOT:-1}"
if [ -z "${KEEP+x}" ]; then [ "$BOOT" = 1 ] && KEEP=0 || KEEP=1; fi
# ────────────────────────────────────────────────────────────────────────────

LOGD="$OUTDIR/_logs"; mkdir -p "$LOGD"
RAW="${RAW:-$OUTDIR/per_request_raw.jsonl}"; touch "$RAW"

export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0

log(){ echo "[$(date '+%H:%M:%S')] $*"; }
wait_ready(){ for i in $(seq 1 180); do curl -sf "$1/v1/models" >/dev/null 2>&1 && { log "READY $1"; return 0; }; [ -n "${2:-}" ] && ! kill -0 "$2" 2>/dev/null && { log "DEAD backend (boot 실패) $1"; return 1; }; sleep 5; done; log "TIMEOUT $1"; return 1; }
kill_pgroup(){ local pid=$1; [ -z "$pid" ] && return 0; local pg; pg=$(ps -o pgid= -p "$pid" 2>/dev/null|tr -d ' '); [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null; kill -9 "$pid" 2>/dev/null; }
# ss/lsof/fuser 부재 환경 → /proc cmdline 스캔으로 'vllm serve --port <PORT>' 프로세스 PID 탐색
find_engine_pid(){ local port=$1 pid cl; for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do cl=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null); case "$cl" in *"vllm serve"*"--port $port"*|*"vllm serve"*"--port=$port"*) echo "$pid"; return 0;; esac; done; }
# backstop: 남은 vLLM GPU 워커를 compute-apps PID 로 직접 reap (cmdline 확인 → 타 테넌트 보호)
reap_vllm_gpu(){ local g; for g in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' < "/proc/$g/cmdline" 2>/dev/null | grep -qiE 'vllm|enginecore' && kill -9 "$g" 2>/dev/null; done; }
kill_engine(){ local pid="$1"; [ -z "$pid" ] && pid=$(find_engine_pid "$PORT"); log "  KILL — 엔진 종료 (pid=${pid:-none}, PORT $PORT)"; kill_pgroup "$pid"; wait_gpu_free && return 0; log "  GPU 미해제 → vllm compute-apps backstop reap"; reap_vllm_gpu; wait_gpu_free; }
wait_gpu_free(){ for i in $(seq 1 24); do local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}'); [ "${u:-1}" -lt 4000 ] && { log "GPU freed"; return 0; }; sleep 5; done; log "GPU 미해제(timeout)"; return 1; }
spec_args(){ case "$1" in
  vanilla) echo "";;
  suffix)  echo "--speculative-config {\"method\":\"suffix\",\"num_speculative_tokens\":32}";;
  ngram)   echo "--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":8,\"prompt_lookup_max\":4}";;
  *) echo "";; esac; }
pick_tp(){ local h; h=$("$PY" -c "import glob,json,os,sys;c=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--'+sys.argv[1].replace('/','--')+'/snapshots/*/config.json'));print(json.load(open(c[0])).get('num_attention_heads',0) if c else 0)" "$1" 2>/dev/null); if [ "${h:-0}" -gt 0 ] && [ $((h % 8)) -eq 0 ]; then echo 8; else echo 4; fi; }

[ "$TP" = auto ] && TP=$(pick_tp "$MODEL")
if [ "$TP" = 8 ]; then GPUS="${GPUS:-0,1,2,3,4,5,6,7}"; else GPUS="${GPUS:-0,1,2,3}"; fi
SA=$(spec_args "$METHOD")
[ "$STREAM" = 0 ] && SFLAG="--no-stream" || SFLAG=""
EMAXTOK="$MAXTOK"; [ "$METHOD" = ngram ] && EMAXTOK="$NGRAM_MAXTOK"   # ngram 출력 상한 별도

# resume: 모든 조건 summ 이미 있으면 부팅 없이 phase 통째 skip
if [ "$SKIP_EXISTING" = 1 ]; then
  allskip=1; for C in $CONDITIONS; do [ -s "$OUTDIR/summ_${TAG}_${METHOD}_${C}.json" ] || allskip=0; done
  [ "$allskip" = 1 ] && { log "=== CASE skip-all $TAG × $METHOD (전 조건 existing) ==="; exit 0; }
fi

log "=== CASE $TAG × $METHOD (TP=$TP gpus=$GPUS conc=$CONC max=$EMAXTOK stream=$STREAM boot=$BOOT keep=$KEEP skip=$SKIP_EXISTING) → $OUTDIR ==="
PID=""
if [ "$BOOT" = 1 ]; then
  CUDA_VISIBLE_DEVICES=$GPUS setsid "$VBIN" serve "$MODEL" \
    --tensor-parallel-size "$TP" --port "$PORT" --gpu-memory-utilization "$GMU" \
    --max-model-len "$MML" --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    $SA > "$LOGD/${TAG}_${METHOD}.log" 2>&1 < /dev/null &
  PID=$!
else
  log "  ATTACH 모드 — PORT $PORT 에 떠 있는 엔진 사용 (부팅 생략)"
fi
rc=0
if wait_ready "http://127.0.0.1:$PORT" "$PID"; then
  for C in $CONDITIONS; do
    OJ="$OUTDIR/summ_${TAG}_${METHOD}_${C}.json"
    if [ "$SKIP_EXISTING" = 1 ] && [ -s "$OJ" ]; then log "  skip(existing) $C"; continue; fi
    if [ "$C" = mix ]; then ARG="--limit $LIMIT --shuffle"; else ARG="--corpus $C"; fi
    PYTHONPATH=. "$PY" "$RE/throughput_runner.py" --in "$SAMPLED" --method "$METHOD" \
      --model "$MODEL" --model-tag "$TAG" --port "$PORT" --max-tokens "$EMAXTOK" \
      --concurrency "$CONC" $ARG $SFLAG \
      --out "$OJ" --raw "$RAW" >> "$LOGD/tput.log" 2>&1 \
      && { log "  done $C"; \
           if [ "$AGG" = 1 ]; then \
             PYTHONPATH=. "$PY" "$RE/build_throughput_table.py" --dir "$OUTDIR" \
               --parquet "$OUTDIR/metrics_table.parquet" --cells-dir "$OUTDIR/cells" \
               --raw-file "$RAW" --results-md "$RESULTS_MD" > "$OUTDIR/SUMMARY.md" 2>> "$LOGD/aggregate.log" \
               && log "  aggregated (cells/+parquet+SUMMARY+RESULTS)" \
               || log "  aggregate FAIL — 다음 셀 진행"; \
           fi; } \
      || { log "  FAIL $C"; rc=1; }
  done
else
  log "  SKIP $TAG × $METHOD (backend 미준비)"; rc=1
fi
if [ "$KEEP" = 1 ]; then
  log "  KEEP — 엔진 유지 (pid=${PID:-attached}, PORT $PORT)"
else
  # BOOT=1 이면 보유 PID, attach(BOOT=0)면 /proc 스캔으로 엔진 PID 발견 → pgroup kill + backstop
  kill_engine "$PID"
fi
log "=== CASE done $TAG × $METHOD (rc=$rc) ==="
exit $rc
