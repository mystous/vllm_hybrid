#!/usr/bin/env bash
# SUB_239 FERRY e2e A/B — NEO swap-in 경로에서 VLLM_NEO_FERRY 0 vs 1 (DSA 가속).
#
# 한 셀 = serve 부팅 → 부하(swap 발화) → tps/지연 측정 → 종료. FERRY env 만 토글.
# GPU 블록을 --num-gpu-blocks-override 로 캡해 swap-out 강제(2×B200 KV 과다 회피).
set -uo pipefail
cd "$(dirname "$0")"

PY=/workspace/vllm_dev_prj/bin/python
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
GPUS=${GPUS:-0,1}
PORT=${PORT:-8200}
GPU_BLOCKS=${GPU_BLOCKS:-1024}        # 캡 → swap 강제 (1024 blk × 16 = 16K tok 총용량)
MAXLEN=${MAXLEN:-16384}
CONC=${CONC:-48}
PROMPT_TOK=${PROMPT_TOK:-6000}
MAXTOK=${MAXTOK:-256}
REQS=${REQS:-240}
OUT=${OUT:-./e2e_results}
mkdir -p "$OUT"

common_env() {
  export VLLM_LHC_DSA=1 VLLM_LHC_DSA_DEV=/dev/dsa/wq1.0
  export VLLM_NEO_NUMA_BIND=1 VLLM_NEO_CPU_PIN_PER_WORKER=1
  export CUDA_VISIBLE_DEVICES=$GPUS
}

serve_up() {  # $1 = FERRY (0/1), $2 = logfile
  common_env
  export VLLM_NEO_FERRY=$1
  echo "[serve] FERRY=$1 GPUS=$GPUS blocks=$GPU_BLOCKS → $2"
  $PY -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name ferry \
      --tensor-parallel-size 2 \
      --enable-neo-asymmetric --kv-cache-policy exclusive \
      --gpu-memory-utilization 0.85 --max-model-len "$MAXLEN" \
      --num-gpu-blocks-override "$GPU_BLOCKS" \
      --port "$PORT" --disable-log-requests \
      > "$2" 2>&1 &
  echo $!
}

wait_ready() {  # $1 = logfile, $2 = pid
  for i in $(seq 1 120); do
    if grep -q "Application startup complete\|Uvicorn running" "$1" 2>/dev/null; then return 0; fi
    if ! kill -0 "$2" 2>/dev/null; then echo "[ERR] 서버 프로세스 사망"; tail -30 "$1"; return 1; fi
    sleep 3
  done
  echo "[ERR] 부팅 타임아웃"; tail -30 "$1"; return 1
}

run_cell() {  # $1 = tag(A/B), $2 = FERRY
  local tag=$1 ferry=$2
  local slog="$OUT/serve_${tag}.log"
  local pid; pid=$(serve_up "$ferry" "$slog")
  if ! wait_ready "$slog" "$pid"; then kill "$pid" 2>/dev/null; return 1; fi
  echo "[serve] ready (pid=$pid) — 부하 시작"
  $PY ferry_e2e_load.py --base "http://127.0.0.1:$PORT" --model ferry \
      --concurrency "$CONC" --prompt-tokens "$PROMPT_TOK" --max-tokens "$MAXTOK" \
      --requests "$REQS" --tag "$tag" --out "$OUT/e2e_summary.txt" 2>&1 | tee "$OUT/load_${tag}.log"
  # swap 발화·DSA 증거 추출
  echo "[markers $tag] NEO BUF/SWAP:"; grep -c "NEO BUF ALLOC\|NEO SWAP\|swap-in\|swap_out" "$slog" 2>/dev/null || true
  grep -m3 "NEO BUF ALLOC\|NEO SWAP" "$slog" 2>/dev/null | sed "s/^/  [$tag] /" || true
  echo "[dsa $tag] dsa_lane/FERRY 로그:"; grep -i "dsa\|ferry" "$slog" 2>/dev/null | grep -iv "warn" | tail -4 | sed "s/^/  [$tag] /" || true
  kill "$pid" 2>/dev/null; sleep 8
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 4
}

echo "=== FERRY e2e A/B: model=$MODEL conc=$CONC prompt=$PROMPT_TOK maxtok=$MAXTOK reqs=$REQS blocks=$GPU_BLOCKS ==="
: > "$OUT/e2e_summary.txt"
run_cell A 0   # FERRY off
run_cell B 1   # FERRY on (DSA)
echo ""; echo "===== 요약 ====="; cat "$OUT/e2e_summary.txt"
