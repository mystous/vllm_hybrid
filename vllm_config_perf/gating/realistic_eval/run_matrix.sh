#!/usr/bin/env bash
# TSK_042 전체 매트릭스 — MODELS × METHODS 각각을 run_case.sh 로 직렬 실행 후 집계.
# run_case.sh 가 케이스(백엔드 phase)당 부팅·측정·kill 을 담당. 본 스크립트는 루프 + 집계.
#
# 설정값은 전부 env 로 주입(복붙용). 예 (T1~T3 전체):
#   MODELS="Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct \
#           deepseek-ai/DeepSeek-R1-Distill-Qwen-7B Qwen/Qwen2.5-32B-Instruct \
#           deepseek-ai/DeepSeek-R1-Distill-Qwen-32B Qwen/Qwen2.5-72B-Instruct \
#           meta-llama/Llama-3.1-70B-Instruct deepseek-ai/DeepSeek-R1-Distill-Llama-70B" \
#   METHODS="vanilla suffix ngram" \
#   SAMPLED=$PWD/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/sampled_prompts.parquet \
#   OUTDIR=$PWD/vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602 \
#   bash vllm_config_perf/gating/realistic_eval/run_matrix.sh
#
# 옵션 env (run_case.sh 로 전달): CONDITIONS CONC MAXTOK MML LIMIT PORT GMU STREAM TP GPUS
set -uo pipefail
cd /workspace/host_vllm_hybrid
PY=${PY:-/workspace/vllm_dev_prj/bin/python}
RE=vllm_config_perf/gating/realistic_eval

MODELS="${MODELS:?MODELS 필요 (공백구분 HF id 목록)}"
METHODS="${METHODS:-vanilla suffix ngram}"
SAMPLED="${SAMPLED:?SAMPLED parquet 경로 필요}"
OUTDIR="${OUTDIR:?OUTDIR 필요}"
mkdir -p "$OUTDIR/_logs"
RAW="$OUTDIR/per_request_raw.jsonl"
[ "${RESUME:-0}" = 1 ] || : > "$RAW"        # RESUME=1 이면 기존 raw 보존(완료셀 유지)
export RAW SAMPLED OUTDIR

# RESULTS_MD 를 case 에서도 동일 경로 사용 (셀-당 build_throughput_table.py 호출 시 필요)
RESULTS_MD="${RESULTS_MD:-shadow_assists/features/IDE_022_agsd_realistic_eval/TSK_042_realistic_workload_oracle/RESULTS.md}"
export RESULTS_MD

# 사용자가 설정한 옵션만 run_case.sh 로 export (미설정 시 run_case.sh 기본값 사용)
for v in CONDITIONS CONC MAXTOK NGRAM_MAXTOK MML LIMIT PORT GMU STREAM TP GPUS SKIP_EXISTING AGG WAIT_READY_MAX PY VBIN; do
  eval "[ -n \"\${$v+x}\" ] && export $v"
done

log(){ echo "[$(date '+%H:%M:%S')] $*"; }
i=0; tot=$(( $(echo $MODELS|wc -w) * $(echo $METHODS|wc -w) ))
log "=== MATRIX models=$(echo $MODELS|wc -w) × methods=$(echo $METHODS|wc -w) = $tot phase → $OUTDIR ==="
for MODEL in $MODELS; do
  for M in $METHODS; do
    i=$((i+1)); log "### phase $i/$tot: $(basename "$MODEL") × $M ###"
    MODEL="$MODEL" METHOD="$M" bash "$RE/run_case.sh" || log "  (phase $i rc!=0, 계속)"
  done
done
log "=== aggregate (final) ==="
PYTHONPATH=. "$PY" "$RE/build_throughput_table.py" --dir "$OUTDIR" \
  --parquet "$OUTDIR/metrics_table.parquet" --cells-dir "$OUTDIR/cells" \
  --raw-file "$RAW" --results-md "$RESULTS_MD" | tee "$OUTDIR/SUMMARY.md"
log "=== MATRIX DONE → $OUTDIR (parquet+csv, RESULTS.md=$RESULTS_MD) ==="
echo "$OUTDIR" > /tmp/tput_run_dir.txt
