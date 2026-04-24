#!/usr/bin/env bash
# =============================================================================
# g0_strategy_compare.sh — 3 router 전략 (Algorithm 1/2/3) 순차 실행
#
# Usage:
#   ./g0_strategy_compare.sh <env_prefix>
#
# env_prefix 에 대해 다음 3개 env 가 있다고 가정:
#   <env_prefix>_strat_capacity.env          (Algorithm 1)
#   <env_prefix>_strat_length_aware.env      (Algorithm 2)
#   <env_prefix>_strat_throughput_adaptive.env (Algorithm 3)
#
# Example:
#   ./g0_strategy_compare.sh eval/envs/g0_dev_rtx3090_qwen7b
#   ./g0_strategy_compare.sh eval/envs/g0_h100x8_qwen7b
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    cat <<EOF >&2
Usage: $0 <env_prefix>

  env_prefix   e.g. eval/envs/g0_dev_rtx3090_qwen7b
               → <prefix>_strat_{capacity,length_aware,throughput_adaptive}.env
EOF
    exit 2
fi

PREFIX="$1"
STRATS=(capacity length_aware throughput_adaptive)

# env 존재 체크
for s in "${STRATS[@]}"; do
    f="${PREFIX}_strat_${s}.env"
    [[ -f "$f" ]] || { echo "[ERROR] missing env: $f" >&2; exit 2; }
done

echo "=== g0_strategy_compare ==="
echo "  env_prefix: $PREFIX"
echo "  strategies: ${STRATS[*]}"
echo ""

for s in "${STRATS[@]}"; do
    ENV_FILE="${PREFIX}_strat_${s}.env"
    echo "─── strategy=$s  ($(basename $ENV_FILE)) ───"

    PORT=$(grep -E '^PORT=' "$ENV_FILE" | head -1 | cut -d= -f2 | tr -d '"')
    PORT="${PORT:-8000}"

    # 기존 서버 정리
    pkill -f 'api_server' 2>/dev/null || true
    pkill -f 'serve\.sh' 2>/dev/null || true
    sleep 3

    SERVE_LOG="/tmp/g0_strat_${s}_serve.log"
    bash "$SCRIPT_DIR/serve.sh" hybrid "$ENV_FILE" > "$SERVE_LOG" 2>&1 &
    SERVE_PID=$!

    echo "  waiting server ready..."
    START=$(date +%s)
    while true; do
        if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            echo "  server ready ($(($(date +%s) - START))s)"
            break
        fi
        if ! kill -0 $SERVE_PID 2>/dev/null; then
            echo "[ERROR] server died. tail:" >&2
            tail -20 "$SERVE_LOG" >&2
            exit 3
        fi
        if (( $(date +%s) - START > 300 )); then
            echo "[ERROR] server ready timeout" >&2
            kill $SERVE_PID 2>/dev/null || true
            exit 4
        fi
        sleep 5
    done

    bash "$SCRIPT_DIR/bench.sh" hybrid "$ENV_FILE" || echo "[WARN] bench rc=$?"

    kill -TERM $SERVE_PID 2>/dev/null || true
    wait $SERVE_PID 2>/dev/null || true
    sleep 3
    echo ""
done

echo "=== strategy 비교 완료 ==="
cat <<EOF
수동 mv 예 (각 run 을 전략별 폴더로):
  mv eval/results/<ts1>_..._seqs1 measurement_results/RTX3090/g0_02_strat_capacity/seqs1
  mv eval/results/<ts2>_..._seqs1 measurement_results/RTX3090/g0_02_strat_length_aware/seqs1
  mv eval/results/<ts3>_..._seqs1 measurement_results/RTX3090/g0_02_strat_throughput_adaptive/seqs1

분석: notebook eval/RTX3090/analysis_g0_compare.ipynb
EOF
