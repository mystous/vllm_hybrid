#!/usr/bin/env bash
# SUB_217 [D4] — cldemote 협조적 aggressor (polite harvest) vLLM 70B 실측판
# 질문: HW 파티션 없이 (무가드) harvest 커널 작성법만으로 간섭을 줄이는가
# 변형: basic(일반 store) / cldemote(자발 강등) / nt(streaming store, RFO 제거)
#
# 측정 설정 = SUB_212 canonical 준수: Llama-3.1-70B-Instruct TP=8, suffix K=32,
#   FaP, gmu 0.85, MML 16384, conc 32, maxtok 8192, 7 corpora (6 + mix(500,shuffle))
# 변경점 (실험 통제): vLLM 을 primary HT {0-47,56-103} 에 taskset 고정 —
#   sibling {112-159,168-215} 과 free 물리코어 {48-55,104-111} 를 실험 변인으로 확보.
#   절대 tps 는 canonical 과 다를 수 있으나 셀 간 비교는 동일 조건.
#
# 셀 (cell-major, 단일 부팅):
#   C0_base   : aggressor 없음
#   C1_sep20  : aggressor 16T @ free 물리코어 {48-55,104-111},  harvest MBA 20%
#   C2_sib100 : aggressor 16T @ vLLM sibling {112-119,168-175}, harvest MBA 100%
#   C3_sib20  : aggressor 16T @ vLLM sibling {112-119,168-175}, harvest MBA 20%
# 판정: 연좌 = C3 < C2 (aggressor 를 조였더니 victim 이 나빠짐 = max-throttle 연좌)
set -u
cd /home/mystous/vllm_hybrid

PY=/home/mystous/vllm_dev_prj/bin/python
VBIN=/home/mystous/vllm_dev_prj/bin/vllm
RE=vllm_config_perf/gating/realistic_eval
SUBD=shadow_assists/features/IDE_026_rdt_guarded_harvest
SRC=$SUBD/src
OUTDIR=$SUBD/SUB_217_d4_cldemote_polite/runs
LOGD="$OUTDIR/_logs"
mkdir -p "$LOGD"
RAW="$OUTDIR/per_request_raw.jsonl"

MODEL="meta-llama/Llama-3.1-70B-Instruct"
TAG="Llama-3.1-70B-Instruct"
TP=8 GPUS=0,1,2,3,4,5,6,7 PORT=8011 MML=16384 CONC=32 MAXTOK=8192 LIMIT=500
SAMPLED="$RE/runs/tput_t1t3_20260602/sampled_prompts.parquet"
CORPORA=(sharegpt swebench humaneval mbpp wildchat lmsys)
VLLM_CPUS="0-47,56-103"

# canonical env (TSK_042 harness) + 호스트 실행 env
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export HF_HOME=/raid/hf_cache HF_HUB_OFFLINE=1

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

wait_ready(){
    local i
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { log "READY"; return 0; }
        ! kill -0 "$1" 2>/dev/null && { log "DEAD backend"; return 1; }
        sleep 5
    done
    log "TIMEOUT"; return 1
}

kill_pgroup(){
    local pid=$1; [ -z "$pid" ] && return 0
    local pg; pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null
    kill -9 "$pid" 2>/dev/null
}

AGGR_PID=""
MON_PID=""

start_cell(){
    local name=$1 aggr_cpus=$2 mb=$3 amode=${4:-basic}
    if [ -n "$aggr_cpus" ]; then
        sudo python3 "$SRC/rdt_ctl.py" setup --group harvest --mb "$mb" >> "$LOGD/rdt.log" 2>&1
        sudo python3 "$SRC/rdt_ctl.py" assign --group harvest --cpus "$aggr_cpus" >> "$LOGD/rdt.log" 2>&1
        sudo python3 "$SRC/rdt_ctl.py" mon --groups harvest --interval 2 --duration 100000 \
            --csv "$OUTDIR/mon_${name}.csv" --quiet >> "$LOGD/rdt.log" 2>&1 &
        MON_PID=$!
        "$SRC/victim_aggressor" --role aggressor --cpus "$aggr_cpus" --secs 1000000 --array-mb 32 --aggr-mode "$amode" \
            > "$LOGD/aggr_${name}.log" 2>&1 &
        AGGR_PID=$!
        log "cell $name: aggressor pid=$AGGR_PID cpus=$aggr_cpus MBA=$mb%"
        sleep 5
    else
        log "cell $name: baseline (aggressor 없음)"
    fi
}

end_cell(){
    [ -n "$AGGR_PID" ] && kill -9 "$AGGR_PID" 2>/dev/null; AGGR_PID=""
    [ -n "$MON_PID" ] && sudo kill "$MON_PID" 2>/dev/null; MON_PID=""
    sudo python3 "$SRC/rdt_ctl.py" teardown --group harvest >> "$LOGD/rdt.log" 2>&1 || true
    sleep 2
}

bench_all(){
    local cell=$1 C OUT
    for C in "${CORPORA[@]}"; do
        OUT="$OUTDIR/summ_${TAG}_${cell}_${C}.json"
        [ -s "$OUT" ] && { log "  skip $cell x $C"; continue; }
        log "  bench $cell x $C"
        PYTHONPATH=. "$PY" "$RE/throughput_runner.py" \
            --in "$SAMPLED" --method "suffix_$cell" \
            --model "$MODEL" --model-tag "$TAG" \
            --port $PORT --max-tokens "$MAXTOK" \
            --concurrency "$CONC" --corpus "$C" \
            --out "$OUT" --raw "$RAW" \
            >> "$LOGD/bench_${cell}.log" 2>&1 || log "    bench FAIL $cell x $C"
    done
    OUT="$OUTDIR/summ_${TAG}_${cell}_mix.json"
    if [ ! -s "$OUT" ]; then
        log "  bench $cell x mix"
        PYTHONPATH=. "$PY" "$RE/throughput_runner.py" \
            --in "$SAMPLED" --method "suffix_$cell" \
            --model "$MODEL" --model-tag "$TAG" \
            --port $PORT --max-tokens "$MAXTOK" \
            --concurrency "$CONC" --limit "$LIMIT" --shuffle \
            --out "$OUT" --raw "$RAW" \
            >> "$LOGD/bench_${cell}.log" 2>&1 || log "    bench FAIL $cell x mix"
    fi
}

trap 'echo "[trap] interrupted"; end_cell; kill_pgroup ${PID:-}; exit 130' INT TERM

# v2 (2026-06-12): 셀마다 fresh boot — v1 단일부팅에서 suffix global tree 누적학습
# (acc 0.72→0.84 드리프트, tps +47% 오염) 발견. SUB_212 도 config 당 새 부팅이므로
# per-cell boot 가 canonical 에 더 충실. 셀 내 corpus 순서는 동일 (웜업 위상 동등).

boot_server(){
    BOOT_LOG="$LOGD/boot_$1.log"; : > "$BOOT_LOG"
    env CUDA_VISIBLE_DEVICES=$GPUS setsid taskset -c "$VLLM_CPUS" "$VBIN" serve "$MODEL" \
        --tensor-parallel-size $TP --port $PORT \
        --gpu-memory-utilization 0.85 \
        --max-model-len "$MML" \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
        --speculative-config '{"method":"suffix","num_speculative_tokens":32}' \
        > "$BOOT_LOG" 2>&1 < /dev/null &
    PID=$!
    log "boot pid=$PID cell=$1 (taskset $VLLM_CPUS)"
    wait_ready "$PID"
}

wait_gpu_free(){
    local i u
    for i in $(seq 1 60); do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1}END{print s+0}')
        [ "${u:-1}" -lt 4000 ] && { log "GPU freed"; return 0; }
        sleep 5
    done
    log "GPU NOT FREED"; return 0
}

run_one_cell(){
    local name=$1 aggr=$2 mb=$3 amode=${4:-basic}
    if ! boot_server "$name"; then
        log "BOOT FAIL cell=$name"; tail -30 "$BOOT_LOG"; kill_pgroup "$PID"; return 1
    fi
    start_cell "$name" "$aggr" "$mb" "$amode"
    bench_all "$name"
    end_cell
    kill_pgroup "$PID"
    wait_gpu_free
}

log "=== SUB_217 sweep start: $TAG suffix K=32, 4 cells x 7 corpora, per-cell boot ==="
run_one_cell A0_base     ""                ""    ""
run_one_cell A1_basic    "112-119,168-175" 100   basic
run_one_cell A2_cldemote "112-119,168-175" 100   cldemote
run_one_cell A3_nt       "112-119,168-175" 100   nt
log "=== SUB_217 sweep done ==="
