#!/usr/bin/env bash
# IDE_070 / TSK_058 — D 시리즈: 비동기 파이프라인 (이 빌드에 이미 구현된 IDE_033 callback-free 핸드오프 스택을 켜서 측정).
#   d1_cf        : KT_CALLBACK_FREE=1                          (hot96 v1, N=4, KV40k — 기준선과 동일 나머지)
#   d2_cf_skip8  : + KT_CF_SKIP_EMPTY_IMM=1, N=8              (빈 immediate 생략은 N ≥ k 필요)
#   d3_cf_skip8_pin : d2 + 비-kt 스레드를 빈 코어로 재배치 (eval/harness/20260909_mechanism/pin_nonkt.sh)
#   d4_epoch     : 9-10 캠페인 최적 스택 (ENVC + hotmap_mixed_0.25 + N=8 + fp8 KV + graph224 + KV 143,360) — C64 3회 + C224 1회
# 사용: run_dseries.sh <cells>   REPS=3
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
CELLS=${1:-d1_cf,d2_cf_skip8,d3_cf_skip8_pin,d4_epoch}; REPS=${REPS:-3}; C=64; N=$((C*4))
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide070_dseries
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480; PIN=$REPO/eval/harness/20260909_mechanism/pin_nonkt.sh
base_args() {  # base_args <deferred N>
  echo "--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token $1 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic"
}
EPOCH_ENV="SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 KT_AVX_RB=1 KT_AVX_PF=2 KT_FUSE_QIN=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25"
epoch_args() {
  echo "--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --init-expert-location /models/kt/ide070/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --kv-cache-dtype fp8_e5m2 --cuda-graph-max-bs 224 --cuda-graph-bs 32 64 96 128 160 192 224 --kt-num-gpu-experts 96 --max-total-tokens 143360"
}
pin_nonkt() {  # 워커(numa_*) 가 쓰지 않는 코어 집합을 계산해 그곳으로 비-kt 스레드 이동 (컨테이너 안에서 실행)
  local TP0=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1)
  local used=$(for t in /proc/$TP0/task/*; do c=$(cat $t/comm 2>/dev/null); case $c in numa_*) grep Cpus_allowed_list $t/status | cut -f2;; esac; done | sort -un | tr '\n' ',' | sed 's/,$//')
  local free=$(python3 -c "
u=set()
for p in '$used'.split(','):
    if not p: continue
    a,b=(p.split('-')+[None])[:2]; u.update(range(int(a),int(b if b else a)+1))
f=[c for c in range(224) if c not in u]
# 구간 표기
out=[]; s=None
for c in f:
    if s is None: s=e=c
    elif c==e+1: e=c
    else: out.append(f'{s}-{e}' if s!=e else str(s)); s=e=c
if s is not None: out.append(f'{s}-{e}' if s!=e else str(s))
print(','.join(out))")
  log "pin_nonkt: worker cpus=$used → free=$free"
  docker exec -i -e FREE="$free" "$CN" bash -s < "$PIN" | tee -a "$RUN_LOG"
}
run_cell() {  # run_cell <name> <env> <args...>
  local name=$1; export BOOT_ENV=$2; shift 2
  local out=$BASE/$name
  log "-- cell $name env='$BOOT_ENV' --"
  boot "$out" 0,1,2,3 2400 "$@"
  echo "$BOOT_VERDICT $BOOT_SEC" > "$out/verdict.txt"
  if [ "$BOOT_VERDICT" = HEALTH_OK ]; then
    docker exec "$CN" bash -c "grep -c 'kt-cf' $LOG_IN_CN" > "$out/cf_lines.txt" 2>&1
    [ "$name" = d3_cf_skip8_pin ] && pin_nonkt
    smoke "$out" "$MODEL" | tee -a "$RUN_LOG"
    bench "$out/warm" "$MODEL" "$TOK" 16 32 >/dev/null 2>&1
    for r in $(seq 1 $REPS); do
      bench "$out/rep$r" "$MODEL" "$TOK" $C $N | tee -a "$RUN_LOG"
      cpu_summary "$out/rep$r/cpu_util.txt" | tee -a "$RUN_LOG"
    done
    if [ "$name" = d4_epoch ]; then
      curl -s -X POST "http://127.0.0.1:$PORT/flush_cache" >/dev/null; sleep 3
      bench "$out/c224" "$MODEL" "$TOK" 224 896 | tee -a "$RUN_LOG"; cpu_summary "$out/c224/cpu_util.txt" | tee -a "$RUN_LOG"
    fi
    docker exec "$CN" bash -c "tail -60 $LOG_IN_CN" > "$out/server_tail.log" 2>/dev/null
  else
    log "cell $name boot failed: $(tail -3 "$out/error_lines.log")"
  fi
  stop_server; unset BOOT_ENV
}
log "== IDE_070 D-series $TS cells=$CELLS reps=$REPS =="
stop_server
for c in ${CELLS//,/ }; do
  case $c in
    d1_cf)           run_cell d1_cf "KT_CALLBACK_FREE=1" $(base_args 4) ;;
    d2_cf_skip8)     run_cell d2_cf_skip8 "KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1" $(base_args 8) ;;
    d3_cf_skip8_pin) run_cell d3_cf_skip8_pin "KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1" $(base_args 8) ;;
    d4_epoch)        run_cell d4_epoch "$EPOCH_ENV" $(epoch_args) ;;
    *) log "unknown cell $c" ;;
  esac
done
sudo chown -R "$(id -u):$(id -g)" "$BASE" 2>/dev/null
for d in $BASE/*/; do n=$(basename $d); for r in $d/rep*/ $d/c224/; do [ -f $r/bench_summary.txt ] && log "$n $(basename $r): $(grep -E 'Output token|Median TPOT|P95 TPOT|Median TTFT' $r/bench_summary.txt | tr '\n' ' ')"; done; done
log "== D-series done =="
echo "$BASE"
