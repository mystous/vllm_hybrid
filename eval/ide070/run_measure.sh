#!/usr/bin/env bash
# IDE_070 / TSK_054 — 기준선(TP4 hot96 def4 KV40k, C64) 측정 묶음. 코드 변경 없음.
#   rep1: pcm (코어 IPC·주파수·L3 miss + 소켓별 DRAM GB/s)  + recorder 로 cold expert 호출/토큰
#   rep2: pcm-numa (local/remote DRAM 접근)
#   rep3: perf stat -a --per-socket (cycles/instructions/cache/ctx-switch/faults) + turbostat + py-spy(TP0)
#   매 rep: mpstat -P ALL (코어별 사용률), numastat -p (전후), iostat/vmstat, nvidia-smi dmon
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide070_measure_baseline
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480; C=${C:-64}; N=$((C*4))
GPUS=${GPUS:-0,1,2,3}
REC_CN=/models/kt/ide069/rec_measure_$TS; REC_HOST=$HOME/.cache/huggingface/kt/ide069/rec_measure_$TS
ARGS="--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer ${CPUINFER:-96} --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic --expert-distribution-recorder-mode stat ${EXTRA:-}"
log "== IDE_070 measure $TS C=$C GPUS=$GPUS =="
log "turbo no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo) max_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"
stop_server
echo "CUDA_VISIBLE_DEVICES=$GPUS python3 -m sglang.launch_server $ARGS" > "$BASE/launch_cmd.txt"
docker exec -d "$CN" bash -c "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=$REC_CN CUDA_VISIBLE_DEVICES=$GPUS python3 -m sglang.launch_server $ARGS > $LOG_IN_CN 2>&1"
i=0; ok=0; while ((i<2400)); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }; docker exec "$CN" pgrep -f "sglang.launch_serve[r]" >/dev/null 2>&1 || break; sleep 10; i=$((i+10)); done
log "boot ok=$ok ${i}s"; [ $ok = 1 ] || { docker exec "$CN" bash -c "tail -60 $LOG_IN_CN" > "$BASE/boot_fail.log"; stop_server; exit 1; }
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$BASE/hbm_after_boot.csv"; free -g > "$BASE/free_after_boot.txt"
smoke "$BASE" "$MODEL" | tee -a "$RUN_LOG"
# 서버 프로세스 (호스트 PID). TP0 = 가장 많은 스레드를 가진 python
PIDS=$(pgrep -f "sglang.launch_serve[r]" | tr '\n' ',' | sed 's/,$//')
TP0=$(for p in $(pgrep -f "sglang.launch_serve[r]"); do echo "$(ls /proc/$p/task 2>/dev/null | wc -l) $p"; done | sort -rn | head -1 | awk '{print $2}')
log "server pids=$PIDS tp0(threads-max)=$TP0 threads=$(ls /proc/$TP0/task | wc -l)"
grep -E "Cpus_allowed_list|Mems_allowed_list" /proc/$TP0/status > "$BASE/tp0_affinity.txt"
# 워밍업 1회 (짧게)
bench "$BASE/warm" "$MODEL" "$TOK" 16 32 >/dev/null 2>&1

common_on() {  # common_on <dir>
  mkdir -p "$1"
  numastat -p $TP0 > "$1/numastat_before.txt" 2>&1; numastat > "$1/numastat_sys_before.txt" 2>&1
  ( mpstat -P ALL 2 > "$1/mpstat.txt" 2>&1 & echo $! > "$1/.mpstat" )
  ( iostat -xz 5 > "$1/iostat.txt" 2>&1 & echo $! > "$1/.iostat" )
  ( vmstat 2 > "$1/vmstat.txt" 2>&1 & echo $! > "$1/.vmstat" )
  ( nvidia-smi dmon -s pucm -d 2 > "$1/nvsmi_dmon.txt" 2>&1 & echo $! > "$1/.dmon" )
  ( pidstat -w -t -p $TP0 5 > "$1/pidstat_ctx.txt" 2>&1 & echo $! > "$1/.pidstat" )
}
common_off() { for f in .mpstat .iostat .vmstat .dmon .pidstat; do kill "$(cat "$1/$f" 2>/dev/null)" 2>/dev/null; done; numastat -p $TP0 > "$1/numastat_after.txt" 2>&1; numastat > "$1/numastat_sys_after.txt" 2>&1; }

# ---- rep1: pcm (코어+메모리) + recorder ----
d=$BASE/rep1_pcm; common_on "$d"
curl -s -X POST "http://127.0.0.1:$PORT/start_expert_distribution_record" >/dev/null
( sudo pcm 2 -csv="$d/pcm.csv" > "$d/pcm.log" 2>&1 & echo $! > "$d/.pcm" )
sleep 4; bench "$d" "$MODEL" "$TOK" $C $N | tee -a "$RUN_LOG"
sudo pkill -INT -f "pcm 2 -csv" 2>/dev/null; sleep 2; sudo pkill -f "pcm 2 -csv" 2>/dev/null
curl -s -X POST "http://127.0.0.1:$PORT/dump_expert_distribution_record" >/dev/null; curl -s -X POST "http://127.0.0.1:$PORT/stop_expert_distribution_record" >/dev/null
common_off "$d"; cpu_summary "$d/cpu_util.txt" | tee -a "$RUN_LOG"; sleep 5

# ---- rep2: pcm-numa ----
d=$BASE/rep2_pcmnuma; common_on "$d"
( sudo pcm-numa 2 -csv="$d/pcm_numa.csv" > "$d/pcm_numa.log" 2>&1 & echo $! > "$d/.pcm" )
sleep 4; bench "$d" "$MODEL" "$TOK" $C $N | tee -a "$RUN_LOG"
sudo pkill -INT -f "pcm-numa 2" 2>/dev/null; sleep 2; sudo pkill -f "pcm-numa 2" 2>/dev/null
common_off "$d"; cpu_summary "$d/cpu_util.txt" | tee -a "$RUN_LOG"; sleep 5

# ---- rep3: perf stat + turbostat + py-spy ----
d=$BASE/rep3_perf; common_on "$d"
( sudo turbostat --interval 5 --quiet --show Core,CPU,Avg_MHz,Busy%,Bzy_MHz,IPC > "$d/turbostat.txt" 2>&1 & echo $! > "$d/.turbo" )
( sudo perf stat -a --per-socket -e task-clock,cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations,page-faults,minor-faults,major-faults -o "$d/perf_stat_sys.txt" sleep 400 & echo $! > "$d/.perf" )
( sudo perf stat -p $TP0 -e task-clock,cycles,instructions,cache-references,cache-misses,context-switches,cpu-migrations,page-faults,major-faults -o "$d/perf_stat_tp0.txt" sleep 400 & echo $! > "$d/.perf2" )
sleep 4; ( sleep 20; sudo "$HOME/.local/bin/py-spy" record --pid $TP0 --duration 30 --native --format raw --output "$d/pyspy_raw.txt" > "$d/pyspy.log" 2>&1; sudo "$HOME/.local/bin/py-spy" dump --pid $TP0 > "$d/pyspy_dump.txt" 2>&1 ) & PS=$!
bench "$d" "$MODEL" "$TOK" $C $N | tee -a "$RUN_LOG"
wait $PS 2>/dev/null
sudo pkill -INT -f "perf stat" 2>/dev/null; sudo pkill -f "turbostat" 2>/dev/null; sleep 3
common_off "$d"; cpu_summary "$d/cpu_util.txt" | tee -a "$RUN_LOG"
sudo chown -R "$(id -u):$(id -g)" "$BASE" 2>/dev/null
ls -la "$REC_HOST" > "$BASE/recorder_files.txt" 2>&1; cp "$REC_HOST"/*.pt "$BASE/" 2>/dev/null
docker exec "$CN" bash -c "tail -100 $LOG_IN_CN" > "$BASE/server_tail.log" 2>/dev/null
stop_server
for r in rep1_pcm rep2_pcmnuma rep3_perf; do log "$r: $(grep -E 'Output token|Median TPOT|P95 TPOT' $BASE/$r/bench_summary.txt | tr '\n' ' ')"; done
log "== measure done =="
echo "$BASE"
