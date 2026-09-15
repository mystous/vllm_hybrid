#!/usr/bin/env bash
# IDE_069 1단계 — 라우팅 빈도 기록 → hotmap.json 재생성.
# TP4 하이브리드(expert 전량 CPU) 로 띄우고 stat recorder 를 켠 뒤 대표 워크로드(sonnet 512/128 C16 64req)를 흘린다.
set -uo pipefail
source "$HOME/projects/vllm_hybrid/eval/ide068/lib.sh"
TS=$(date +%Y%m%d_%H%M%S)
BASE=$REPO/eval/results/${TS}_ide069_hotmap
mkdir -p "$BASE"; RUN_LOG=$BASE/RUN.log
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/*/ | head -1)
TOK=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
M_CN=/models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/$(basename "$SNAP")
MODEL=q480
REC_CN=/models/kt/ide069/rec_$TS; REC_HOST=$HOME/.cache/huggingface/kt/ide069/rec_$TS
mkdir -p "$REC_HOST"
log "== IDE_069 hotmap start $TS =="
stop_server
out=$BASE/rec; mkdir -p "$out"
docker exec -d "$CN" bash -c "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=$REC_CN CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path $M_CN --served-model-name $MODEL --host 127.0.0.1 --port $PORT --tp 4 --attention-backend triton --trust-remote-code --disable-cuda-graph --mem-fraction-static 0.80 --max-total-tokens 65536 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 0 --expert-distribution-recorder-mode stat > $LOG_IN_CN 2>&1"
i=0; ok=0
while ((i<1200)); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }; docker exec "$CN" pgrep -f "sglang.launch_serve[r]" >/dev/null 2>&1 || break; sleep 10; i=$((i+10)); done
log "boot ok=$ok ${i}s"
[ $ok = 1 ] || { docker exec "$CN" bash -c "tail -60 $LOG_IN_CN" > "$out/boot_fail.log"; stop_server; exit 1; }
curl -s -X POST "http://127.0.0.1:$PORT/start_expert_distribution_record" | tee -a "$RUN_LOG"; echo
log "workload (sonnet 512/128 C16 64req):"
bench "$out" "$MODEL" "$TOK" 16 64 | tee -a "$RUN_LOG"
curl -s -X POST "http://127.0.0.1:$PORT/dump_expert_distribution_record" | tee -a "$RUN_LOG"; echo
curl -s -X POST "http://127.0.0.1:$PORT/stop_expert_distribution_record" >/dev/null
sleep 3; ls -la "$REC_HOST" | tee -a "$RUN_LOG"
stop_server
python3 "$REPO/eval/ide069/build_hotmap.py" "$REC_HOST" "$HOME/.cache/huggingface/kt/ide069/hotmap.json" | tee -a "$RUN_LOG"
cp "$HOME/.cache/huggingface/kt/ide069/hotmap.json" "$HOME/.cache/huggingface/kt/ide069/hotmap_stats.json" "$BASE/" 2>/dev/null
log "== hotmap done =="
