#!/usr/bin/env bash
# IDE_070 / TSK_054 — perf stat 창을 벤치 명령 자체로 고정 (2차의 SIGINT 종료가 출력 없이 끝난 결함 보정).
#   rep1: perf stat -a --per-socket -- <bench>   (시스템)
#   rep2: perf stat -p <스케줄러 4개> -- <bench>   (프로세스)
#   rep3: perf stat -p <TP0> -e 상세 이벤트 -- <bench>
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide070_measure3
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480; C=${C:-64}; N=$((C*4)); GPUS=0,1,2,3
ARGS="--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic"
log "== IDE_070 measure3 $TS C=$C =="
stop_server
boot "$BASE" "$GPUS" 2400 $ARGS
[ "$BOOT_VERDICT" = HEALTH_OK ] || { log "boot fail"; stop_server; exit 1; }
GPU0_BUS=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' '$1==0{print $2}')
TP0=$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | awk -F', ' -v b="$GPU0_BUS" '$2==b{print $1}' | head -1)
SCHED=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
log "TP0=$TP0 sched=$SCHED"
smoke "$BASE" "$MODEL" | tee -a "$RUN_LOG"
bench "$BASE/warm" "$MODEL" "$TOK" 16 32 >/dev/null 2>&1

# bench_perf <dir> <perf args...> : perf stat 이 벤치 명령을 감싸 창이 정확히 벤치 구간이 된다
bench_perf() {
  local out=$1; shift; mkdir -p "$out"
  ( bash "$CPU_SAMPLER" "$out/cpu_util.txt" & echo $! > "$out/.cpu_mon" )
  ( vmstat 2 > "$out/vmstat.txt" 2>&1 & echo $! > "$out/.vmstat" )
  date +%s > "$out/bench_t0.txt"
  # sudo 아래에서는 ~/bin/docker 셔임이 PATH 에 없어 /usr/bin/docker(실제 dockerd, 컨테이너 없음) 가 잡힘 → nerdctl 직접 호출
  sudo perf stat "$@" -o "$out/perf_stat.txt" -- "$(sudo which nerdctl)" exec "$BENCH_CN" vllm bench serve --backend openai --base-url "http://127.0.0.1:$PORT" --endpoint /v1/completions \
    --model "$MODEL" --tokenizer "$TOK" --dataset-name sonnet --dataset-path /tmp/sonnet.txt \
    --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 \
    --num-prompts "$N" --max-concurrency "$C" --request-rate inf --seed 42 \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95 > "$out/bench.log" 2>&1
  date +%s > "$out/bench_t1.txt"
  kill "$(cat "$out/.cpu_mon")" "$(cat "$out/.vmstat")" 2>/dev/null
  grep -E "Successful requests|Failed requests|Benchmark duration|Output token throughput|Total token throughput|Median TTFT|P95 TTFT|Median TPOT|P95 TPOT" "$out/bench.log" | sed 's/  */ /g' | tee "$out/bench_summary.txt"
  cpu_summary "$out/cpu_util.txt" | tee -a "$RUN_LOG"
}
bench_perf "$BASE/rep1_sys" -a --per-socket -e task-clock,cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations,page-faults,minor-faults,major-faults | tee -a "$RUN_LOG"
sleep 5
bench_perf "$BASE/rep2_sched" -p "$SCHED" -e task-clock,cycles,instructions,cache-references,cache-misses,context-switches,cpu-migrations,page-faults | tee -a "$RUN_LOG"
sleep 5
bench_perf "$BASE/rep3_tp0" -p "$TP0" -e task-clock,cycles,instructions,LLC-loads,LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,dTLB-loads,dTLB-load-misses,context-switches | tee -a "$RUN_LOG"
sudo chown -R "$(id -u):$(id -g)" "$BASE" 2>/dev/null
stop_server
for r in rep1_sys rep2_sched rep3_tp0; do log "$r: $(grep -E 'Output token|Median TPOT' $BASE/$r/bench_summary.txt | tr '\n' ' ')"; done
log "== measure3 done =="
echo "$BASE"
