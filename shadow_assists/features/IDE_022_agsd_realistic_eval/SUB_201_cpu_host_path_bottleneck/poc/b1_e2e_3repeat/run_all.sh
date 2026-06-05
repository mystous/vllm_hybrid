#!/usr/bin/env bash
# 12 run sequential driver — 3 repeat × 4 mode (random shuffle per repeat).
#
# Repeat 1: A_baseline, B_b3, C_b1, D_b1b3   (canonical)
# Repeat 2: C_b1, A_baseline, D_b1b3, B_b3    (shuffled)
# Repeat 3: B_b3, D_b1b3, A_baseline, C_b1    (shuffled)
#
# Stops at first failure (set -e). Each run produces llama8b_r<N>_<MODE>.json.
set -uo pipefail
cd /workspace/host_vllm_hybrid

POC_DIR=/workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/b1_e2e_3repeat
RUN_SH="$POC_DIR/run.sh"
LOGD="$POC_DIR/_logs"
mkdir -p "$LOGD"
DRIVER_LOG="$LOGD/driver.log"

# 12-run sequence: "<rep> <mode>" per line
SEQ=(
  "1 A_baseline"
  "1 B_b3"
  "1 C_b1"
  "1 D_b1b3"
  "2 C_b1"
  "2 A_baseline"
  "2 D_b1b3"
  "2 B_b3"
  "3 B_b3"
  "3 D_b1b3"
  "3 A_baseline"
  "3 C_b1"
)

T_DRIVE_START=$(date +%s)
{
echo "==========================================================="
echo "B1 EXCLUSIVE 3-repeat × 4-mode sweep — driver start"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Sequence: ${#SEQ[@]} runs"
echo "==========================================================="
} | tee -a "$DRIVER_LOG"

IDX=0
for item in "${SEQ[@]}"; do
  IDX=$((IDX+1))
  rep=$(echo "$item" | awk '{print $1}')
  mode=$(echo "$item" | awk '{print $2}')
  TAG="r${rep}_${mode}"
  JSON="$POC_DIR/llama8b_${TAG}.json"

  if [ -f "$JSON" ]; then
    {
    echo ""
    echo "[$(date '+%H:%M:%S')] === [${IDX}/${#SEQ[@]}] SKIP $TAG — exists ==="
    } | tee -a "$DRIVER_LOG"
    continue
  fi

  {
  echo ""
  echo "[$(date '+%H:%M:%S')] === [${IDX}/${#SEQ[@]}] START $TAG ==="
  } | tee -a "$DRIVER_LOG"
  T_RUN_START=$(date +%s)

  bash "$RUN_SH" "$rep" "$mode" 2>&1 | tee -a "$DRIVER_LOG"
  rc=${PIPESTATUS[0]}
  T_RUN_END=$(date +%s)
  RUN_DUR=$((T_RUN_END - T_RUN_START))

  if [ "$rc" != "0" ]; then
    echo "[$(date '+%H:%M:%S')] === FAIL $TAG (rc=$rc, dur=${RUN_DUR}s) — abort driver ===" | tee -a "$DRIVER_LOG"
    exit "$rc"
  fi
  echo "[$(date '+%H:%M:%S')] === DONE $TAG (dur=${RUN_DUR}s) ===" | tee -a "$DRIVER_LOG"
  # extra cooldown between runs
  sleep 5
done

T_DRIVE_END=$(date +%s)
TOTAL=$((T_DRIVE_END - T_DRIVE_START))
{
echo ""
echo "==========================================================="
echo "ALL 12 RUNS DONE — total ${TOTAL}s ($(($TOTAL/60))m)"
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==========================================================="
} | tee -a "$DRIVER_LOG"

# final GPU 4,5 check
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits \
  | awk -F',' '$1==4||$1==5' | tee "$LOGD/final_gpu_check.txt"
