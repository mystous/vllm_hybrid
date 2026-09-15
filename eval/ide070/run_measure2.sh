#!/usr/bin/env bash
# IDE_070 / TSK_054 — 1차 측정(run_measure.sh) 결함 보정 재측정.
#   결함 1: perf stat -p 대상이 HTTP 서버 프로세스(스레드 최다)였음 → TP0 = GPU0 에 올라간 compute 프로세스 (nvidia-smi)
#   결함 2: perf 창 400 s 가 부하(~60 s)+유휴를 섞음 → perf 는 bench 직전 시작·직후 SIGINT (실제 perf pid 에)
#   결함 3: py-spy --native 실패(UNW_EBADREG) → --native 없이; numastat -p 는 sudo
#   추가: perf record (TP0, 30 s, -g 없음) 로 hot symbol 상위 / perf stat 을 스케줄러 4개 (TP0..3) 전체에 대해
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide070_measure2
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480; C=${C:-64}; N=$((C*4)); GPUS=${GPUS:-0,1,2,3}
ARGS="--model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer ${CPUINFER:-96} --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic ${EXTRA:-}"
log "== IDE_070 measure2 $TS C=$C GPUS=$GPUS =="
stop_server
boot "$BASE" "$GPUS" 2400 $ARGS
[ "$BOOT_VERDICT" = HEALTH_OK ] || { log "boot fail $BOOT_VERDICT"; stop_server; exit 1; }
smoke "$BASE" "$MODEL" | tee -a "$RUN_LOG"
# 스케줄러 pid: GPU 별 compute 프로세스. GPU index 는 nvidia-smi 순서 = CUDA_VISIBLE_DEVICES 순서(0,1,2,3)
nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv > "$BASE/compute_apps.csv"
GPU0_BUS=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v g="${GPUS%%,*}" '$1==g{print $2}')
TP0=$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | awk -F', ' -v b="$GPU0_BUS" '$2==b{print $1}' | head -1)
SCHED=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
log "TP0 pid=$TP0 comm=$(cat /proc/$TP0/comm) threads=$(ls /proc/$TP0/task | wc -l)  all sched pids=$SCHED"
grep -E "Cpus_allowed_list|Mems_allowed_list|Threads" /proc/$TP0/status > "$BASE/tp0_status.txt"
# 스레드별 affinity 요약 (kt 스레드풀이 어느 코어에 고정됐는지)
for t in /proc/$TP0/task/*; do echo "$(cat $t/comm) $(grep Cpus_allowed_list $t/status | cut -f2)"; done | sort | uniq -c | sort -rn > "$BASE/tp0_thread_affinity.txt"
bench "$BASE/warm" "$MODEL" "$TOK" 16 32 >/dev/null 2>&1

common_on() { mkdir -p "$1"; sudo numastat -p $TP0 > "$1/numastat_tp0_before.txt" 2>&1; numastat > "$1/numastat_sys_before.txt" 2>&1
  ( mpstat -P ALL 2 > "$1/mpstat.txt" 2>&1 & echo $! > "$1/.mpstat" ); ( vmstat 2 > "$1/vmstat.txt" 2>&1 & echo $! > "$1/.vmstat" )
  ( nvidia-smi dmon -s pucm -d 2 > "$1/nvsmi_dmon.txt" 2>&1 & echo $! > "$1/.dmon" ); ( pidstat -w -t -p $TP0 5 > "$1/pidstat_ctx.txt" 2>&1 & echo $! > "$1/.pidstat" ); }
common_off() { for f in .mpstat .vmstat .dmon .pidstat; do kill "$(cat "$1/$f" 2>/dev/null)" 2>/dev/null; done; sudo numastat -p $TP0 > "$1/numastat_tp0_after.txt" 2>&1; numastat > "$1/numastat_sys_after.txt" 2>&1; }
perf_pid() { pgrep -x perf | tr '\n' ' '; }   # sudo 래퍼가 아닌 실제 perf 프로세스들

# ---- rep1: perf stat (system per-socket + 스케줄러 4개) 창 = bench 구간 ----
d=$BASE/rep1_perfstat; common_on "$d"
( sudo perf stat -a --per-socket -e task-clock,cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations,page-faults,minor-faults,major-faults -o "$d/perf_stat_sys.txt" sleep 3600 >/dev/null 2>&1 & )
( sudo perf stat -p $SCHED -e task-clock,cycles,instructions,cache-references,cache-misses,context-switches,cpu-migrations,page-faults,major-faults -o "$d/perf_stat_sched.txt" sleep 3600 >/dev/null 2>&1 & )
( sudo perf stat -p $TP0 -e task-clock,cycles,instructions,cache-references,cache-misses,context-switches,cpu-migrations,page-faults -o "$d/perf_stat_tp0.txt" sleep 3600 >/dev/null 2>&1 & )
sleep 3; log "perf pids: $(perf_pid)"
bench "$d" "$MODEL" "$TOK" $C $N | tee -a "$RUN_LOG"
for p in $(perf_pid); do sudo kill -INT $p; done; sleep 3
common_off "$d"; cpu_summary "$d/cpu_util.txt" | tee -a "$RUN_LOG"; sleep 5

# ---- rep2: py-spy (TP0, python 수준) + perf record (TP0, native 심볼) ----
d=$BASE/rep2_profile; common_on "$d"
( sleep 15; sudo "$HOME/.local/bin/py-spy" record --pid $TP0 --duration 30 --rate 200 --format raw --output "$d/pyspy_raw.txt" > "$d/pyspy.log" 2>&1
  sudo "$HOME/.local/bin/py-spy" record --pid $TP0 --duration 10 --rate 200 --format speedscope --output "$d/pyspy.speedscope.json" >> "$d/pyspy.log" 2>&1
  sudo "$HOME/.local/bin/py-spy" dump --pid $TP0 > "$d/pyspy_dump.txt" 2>&1 ) & PS=$!
( sleep 20; sudo perf record -F 499 -p $TP0 -o "$d/perf.data" -- sleep 20 > "$d/perf_record.log" 2>&1
  sudo perf report -i "$d/perf.data" --stdio --sort dso,symbol 2>/dev/null | head -80 > "$d/perf_report_top.txt"
  sudo perf report -i "$d/perf.data" --stdio --sort comm 2>/dev/null | head -40 > "$d/perf_report_comm.txt" ) & PR=$!
bench "$d" "$MODEL" "$TOK" $C $N | tee -a "$RUN_LOG"
wait $PS $PR 2>/dev/null
common_off "$d"; cpu_summary "$d/cpu_util.txt" | tee -a "$RUN_LOG"; sleep 5

# ---- rep3: pcm-memory (채널별 대역폭) + turbostat ----
d=$BASE/rep3_pcmmem; common_on "$d"
( sudo pcm-memory 2 -csv="$d/pcm_memory.csv" > "$d/pcm_memory.log" 2>&1 & )
( sudo turbostat --interval 5 --quiet --show Core,CPU,Avg_MHz,Busy%,Bzy_MHz,IPC > "$d/turbostat.txt" 2>&1 & )
sleep 4; bench "$d" "$MODEL" "$TOK" $C $N | tee -a "$RUN_LOG"
sudo pkill -INT -x pcm-memory; sudo pkill -x turbostat; sleep 3
common_off "$d"; cpu_summary "$d/cpu_util.txt" | tee -a "$RUN_LOG"
sudo chown -R "$(id -u):$(id -g)" "$BASE" 2>/dev/null
docker exec "$CN" bash -c "tail -100 $LOG_IN_CN" > "$BASE/server_tail.log" 2>/dev/null
stop_server
for r in rep1_perfstat rep2_profile rep3_pcmmem; do log "$r: $(grep -E 'Output token|Median TPOT' $BASE/$r/bench_summary.txt | tr '\n' ' ')"; done
log "== measure2 done =="
echo "$BASE"
