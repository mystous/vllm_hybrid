#!/usr/bin/env bash
# IDE_069 2단계 — TP4 hot expert 셀 sweep. hotmap.json 이 있어야 한다.
# 각 셀 = 부팅 → greedy 4문항 → 지정한 동시성들로 sonnet 512/128 벤치 (C 당 요청 수 = 4×C).
# 사용: run_sweep.sh <cell_spec_file>   (줄 형식: name|gpu_experts|deferral|graph(on/off)|mem_frac|max_tokens|concurrencies(,)|extra flags)
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
SPEC=${1:?cell spec file}
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide069_sweep_$(basename "$SPEC" .txt)
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480
HOTMAP=/models/kt/ide069/hotmap.json
COMMON="--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2"
cp "$SPEC" "$BASE/cells.txt"
log "== IDE_069 sweep $TS spec=$(basename $SPEC) =="
log "turbo no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
stop_server
while IFS='|' read -r name gx def graph mf mt concs extra; do
  [ -z "$name" ] && continue; case "$name" in \#*) continue;; esac
  out=$BASE/$name
  if [ "$graph" = on ]; then G="--cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64"; else G="--disable-cuda-graph"; fi
  HM=""; [ "$gx" != 0 ] && HM="--init-expert-location $HOTMAP"
  DEF=""; [ "$def" != 0 ] && DEF="--kt-max-deferred-experts-per-token $def"
  log "-- cell $name (gx=$gx def=$def graph=$graph mf=$mf mt=$mt C=$concs $extra) --"
  boot "$out" 0,1,2,3 2400 $COMMON --kt-num-gpu-experts "$gx" $HM $DEF $G --mem-fraction-static "$mf" --max-total-tokens "$mt" $extra
  echo "$BOOT_VERDICT $BOOT_SEC" > "$out/verdict.txt"
  if [ "$BOOT_VERDICT" = HEALTH_OK ]; then
    log "smoke:"; smoke "$out" "$MODEL" | tee -a "$RUN_LOG"
    for c in ${concs//,/ }; do
      n=$((c*4)); [ $n -lt 64 ] && n=64
      mkdir -p "$out/c$c"
      log "bench C$c n=$n:"; bench "$out/c$c" "$MODEL" "$TOK" "$c" "$n" | tee -a "$RUN_LOG"
      cpu_summary "$out/c$c/cpu_util.txt" | tee -a "$RUN_LOG"
    done
    docker exec "$CN" bash -c "tail -150 $LOG_IN_CN" > "$out/server_tail.log" 2>/dev/null
  else
    grep -vE "Ignore import" "$out/error_lines.log" | tail -3 | cut -c1-200 | tee -a "$RUN_LOG"
  fi
  stop_server
done < "$SPEC"
log "== sweep done $(date +%H:%M:%S) =="
echo "$BASE"
