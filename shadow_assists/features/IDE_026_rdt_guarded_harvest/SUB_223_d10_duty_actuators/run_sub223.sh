#!/usr/bin/env bash
# SUB_223 [D10] — SIGSTOP duty actuator, 10ms period, duty sweep (feasible subset).
set -u
IDE=shadow_assists/features/IDE_026_rdt_guarded_harvest
SRC=$IDE/src; FD=$IDE/SUB_223_d10_duty_actuators; OUT=$FD/runs; mkdir -p "$OUT"
BIN="$SRC/victim_aggressor"; PY=/home/mystous/vllm_dev_prj/bin/python
VIC=0-7; AGG=8-23; VSEC=10; PERIOD=10
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
run_victim(){ taskset -c "$VIC" "$BIN" --role victim --cpus "$VIC" --ws-mb 64 --copy-kb 4096 --chase-steps 200000 --secs "$VSEC" 2>/dev/null | grep VICTIM_RESULT; }
echo "duty_pct,iters,mean_ms,p50_ms,p95_ms,p99_ms,p999_ms,ns_per_load" > "$OUT/results.csv"
for DUTY in 0 25 50 75 100; do
    log "duty=${DUTY}% (SIGSTOP 10ms)"
    setsid "$PY" "$FD/duty_ctl.py" "$DUTY" "$PERIOD" "$AGG" "$((VSEC+3))" >/dev/null 2>&1 &
    CTL=$!; sleep 2
    R=$(run_victim)
    kill -9 -"$CTL" 2>/dev/null; kill -9 "$CTL" 2>/dev/null; sleep 1
    pkill -9 -f 'victim_aggressor --role aggressor' 2>/dev/null; sleep 0.5
    echo "$DUTY,${R#VICTIM_RESULT,}" >> "$OUT/results.csv"; echo "  duty=$DUTY: $R"
done
log "=== done ==="
