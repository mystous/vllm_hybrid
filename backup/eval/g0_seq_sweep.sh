#!/usr/bin/env bash
# =============================================================================
# g0_seq_sweep.sh — 단일 env 로 HYBRID_CPU_MAX_SEQS 를 바꿔가며 serve+bench 반복
#
# Usage:
#   ./g0_seq_sweep.sh <env_file> <seq1> [seq2 ...]
#
# Example:
#   ./g0_seq_sweep.sh eval/envs/g0_dev_rtx3090_qwen7b.env 1 2 4 8 16
#
# 각 iter:
#   1. env 복사 후 HYBRID_CPU_MAX_SEQS=N 으로 override
#   2. serve.sh 백그라운드 실행
#   3. /v1/models 응답까지 대기
#   4. bench.sh 실행 (결과는 eval/results/<ts>_.../_seqs<N>/)
#   5. 서버 종료
#
# 사용자가 수동으로 eval/results/<ts>_.../ → measurement_results/<HW>/<TODO_NN>/seqs<N>/
# 로 mv.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
    cat <<EOF >&2
Usage: $0 <env_file> <seqs...>

  env_file   base env (hybrid mode 전제)
  seqs       공백 구분 HYBRID_CPU_MAX_SEQS 값

Example:
  $0 eval/envs/g0_dev_rtx3090_qwen7b.env 1 2 4 8 16
EOF
    exit 2
fi

ENV_FILE="$1"
shift
SEQS=("$@")

[[ -f "$ENV_FILE" ]] || { echo "[ERROR] env file not found: $ENV_FILE" >&2; exit 2; }

PORT=$(grep -E '^PORT=' "$ENV_FILE" | head -1 | cut -d= -f2 | tr -d '"')
PORT="${PORT:-8000}"

echo "=== g0_seq_sweep ==="
echo "  env_file: $ENV_FILE"
echo "  port:     $PORT"
echo "  seqs:     ${SEQS[*]}"
echo ""

for S in "${SEQS[@]}"; do
    echo "─── seqs=$S ───"
    TMP_ENV="$(mktemp -t "g0_sweep_seqs${S}.XXXXXX.env")"
    grep -v '^[[:space:]]*HYBRID_CPU_MAX_SEQS=' "$ENV_FILE" > "$TMP_ENV"
    echo "HYBRID_CPU_MAX_SEQS=$S" >> "$TMP_ENV"

    # 기존 서버 정리
    pkill -f 'api_server' 2>/dev/null || true
    pkill -f 'serve\.sh' 2>/dev/null || true
    sleep 2

    # serve 실행
    SERVE_LOG="/tmp/g0_sweep_seqs${S}_serve.log"
    bash "$SCRIPT_DIR/serve.sh" hybrid "$TMP_ENV" > "$SERVE_LOG" 2>&1 &
    SERVE_PID=$!

    # ready 대기 (최대 300s)
    echo "  waiting server ready..."
    START=$(date +%s)
    while true; do
        if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            echo "  server ready ($(($(date +%s) - START))s)"
            break
        fi
        if ! kill -0 $SERVE_PID 2>/dev/null; then
            echo "[ERROR] server process died. tail of log:" >&2
            tail -20 "$SERVE_LOG" >&2
            rm -f "$TMP_ENV"
            exit 3
        fi
        if (( $(date +%s) - START > 300 )); then
            echo "[ERROR] server ready timeout (300s)" >&2
            kill $SERVE_PID 2>/dev/null || true
            rm -f "$TMP_ENV"
            exit 4
        fi
        sleep 5
    done

    # bench 실행
    bash "$SCRIPT_DIR/bench.sh" hybrid "$TMP_ENV" || echo "[WARN] bench rc=$?"

    # 서버 종료
    kill -TERM $SERVE_PID 2>/dev/null || true
    wait $SERVE_PID 2>/dev/null || true
    sleep 3
    rm -f "$TMP_ENV"
    echo ""
done

echo "=== sweep 완료 ==="
echo "결과: eval/results/<ts>_.../ (_seqs<N> suffix 로 구분됨)"
echo "수동 mv 예: mv eval/results/<ts>_..._seqs1 measurement_results/RTX3090/g0_00_7b/seqs1"
