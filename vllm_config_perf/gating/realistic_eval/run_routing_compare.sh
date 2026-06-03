#!/usr/bin/env bash
# TSK_042 §6 — llm-d 라우팅 측정 (단일 전략=llm-d). 백엔드(vanilla+suffix) 기동·Gateway 연결 가정.
# 컨테이너 내부 실행. condition별 summ_<TAG>_llm-d_<cond>.json (기존 스키마) 생성.
set -uo pipefail
RE=/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval
PY=/workspace/vllm_dev_prj/bin/python
GW_HOST="${GW_HOST:-172.20.0.2}"; GW_PORT="${GW_PORT:-30080}"
MODEL="${MODEL:?set MODEL}"; TAG="${TAG:?set TAG}"
MAXTOK="${MAXTOK:-8192}"; CONC="${CONC:-32}"; LIMIT="${LIMIT:-500}"
IN="${IN:-$RE/runs/tput_t1t3_20260602/sampled_prompts.parquet}"
OUTDIR="${OUTDIR:-$RE/runs/routing_llmd_20260603}"
CORPORA="${CORPORA:-sharegpt swebench humaneval mbpp wildchat lmsys}"
RAW="$OUTDIR/per_request_raw.jsonl"
mkdir -p "$OUTDIR"; cd "$RE"
echo "[routing] $TAG via llm-d $GW_HOST:$GW_PORT conc=$CONC max=$MAXTOK $(date +%H:%M:%S)"
for Cc in $CORPORA; do
  PYTHONPATH=. "$PY" throughput_runner.py --in "$IN" --method llm-d \
    --model "$MODEL" --model-tag "$TAG" --host "$GW_HOST" --port "$GW_PORT" \
    --max-tokens "$MAXTOK" --concurrency "$CONC" --corpus "$Cc" \
    --out "$OUTDIR/summ_${TAG}_llm-d_${Cc}.json" --raw "$RAW" >>"$OUTDIR/run.log" 2>&1
  echo "  done $Cc $(date +%H:%M:%S)"
done
PYTHONPATH=. "$PY" throughput_runner.py --in "$IN" --method llm-d \
  --model "$MODEL" --model-tag "$TAG" --host "$GW_HOST" --port "$GW_PORT" \
  --max-tokens "$MAXTOK" --concurrency "$CONC" --limit "$LIMIT" --shuffle \
  --out "$OUTDIR/summ_${TAG}_llm-d_mix.json" --raw "$RAW" >>"$OUTDIR/run.log" 2>&1
echo "  done mix → $OUTDIR $(date +%H:%M:%S)"
