#!/usr/bin/env bash
# [Plan v4 / D5] Worker side py-spy attach — pacpu thread active 측정 (#16).
# Goal: NEO chain firing 시 worker 측 cdec executor thread (pacpu) 가 진짜
# active 인지 확인. EngineCore-only py-spy (plan v2) 의 한계 보완.
set -uo pipefail

PYSPY=/workspace/vllm_dev_prj/bin/py-spy

# 가장 최근 NEO ON 회차 dir 찾기.
DIR=$(ls -td /workspace/vllm_hybrid/eval/results/*try60_v4_J_verify_neo_full_migration 2>/dev/null | head -1)
if [ -z "$DIR" ]; then
    echo "[D5] no try60 dir found"; exit 1
fi
DUMP_DIR="${DIR}/worker_pyspy_dumps"
mkdir -p "${DUMP_DIR}"

echo "[D5] looking for VllmWorker_TP* procs …"
WORKER_PIDS=$(pgrep -af 'VllmWorker' 2>/dev/null | awk '{print $1}' | head -8)
echo "[D5] found workers: $WORKER_PIDS"

if [ -z "$WORKER_PIDS" ]; then
    echo "[D5] no Worker_TP* procs alive — D3 must be running"
    exit 1
fi

# 첫 worker (TP0) 에 5 회 dump (30s 간격).
W0=$(echo "$WORKER_PIDS" | head -1)
echo "[D5] sampling Worker_TP0 (PID $W0) — 5 dumps × 30s"
for i in 1 2 3 4 5; do
    DUMP_FILE="${DUMP_DIR}/worker_tp0_dump_${i}.txt"
    echo "--- dump $i @ $(TZ=Asia/Seoul date -Iseconds) ---" > "${DUMP_FILE}"
    "${PYSPY}" dump --pid "${W0}" --locals \
        >> "${DUMP_FILE}" 2>&1 || \
        echo "[D5] dump $i failed" >> "${DUMP_FILE}"
    [ "$i" -lt 5 ] && sleep 30
done

# 분석: thread 별 active/idle 상태 + cdec executor 발견.
echo "[D5] === thread state summary ==="
for dump in "${DUMP_DIR}"/worker_tp0_dump_*.txt; do
    echo "--- $(basename $dump) ---"
    grep -E '^Thread.*\((active|idle)\):' "$dump" | head -10
    echo ""
done

# pacpu / cdec executor thread 식별.
echo "[D5] === cdec executor / pacpu thread search ==="
for dump in "${DUMP_DIR}"/worker_tp0_dump_*.txt; do
    if grep -qE 'pacpu|cdec_executor|paged_attention_cpu' "$dump"; then
        echo "$(basename $dump): pacpu/cdec found ✓"
    else
        echo "$(basename $dump): no pacpu/cdec thread"
    fi
done

echo "[D5] DONE — dumps in: ${DUMP_DIR}"
