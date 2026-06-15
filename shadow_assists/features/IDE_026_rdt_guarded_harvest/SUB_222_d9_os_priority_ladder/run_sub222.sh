#!/usr/bin/env bash
# SUB_222 [D9] — OS 우선순위 사다리 (RDT-無). victim(0-7) + aggressor(8-23 동소켓).
set -u
IDE=shadow_assists/features/IDE_026_rdt_guarded_harvest
SRC=$IDE/src; OUT=$IDE/SUB_222_d9_os_priority_ladder/runs; mkdir -p "$OUT"
BIN="$SRC/victim_aggressor"
VIC=0-7; AGG=8-23; VSEC=10; ABYTES=32
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
run_victim(){ taskset -c "$VIC" "$BIN" --role victim --cpus "$VIC" --ws-mb 64 --copy-kb 4096 --chase-steps 200000 --secs "$VSEC" 2>/dev/null | grep VICTIM_RESULT; }
start_aggr(){ # $1 = priority launcher prefix (may be empty)
    setsid bash -c "exec $1 taskset -c $AGG \"$BIN\" --role aggressor --cpus $AGG --array-mb $ABYTES --aggr-mode basic --secs 100000" >/dev/null 2>&1 &
    echo $!; }
stop_aggr(){ local pid=$1; [ -z "$pid" ] && return; kill -9 -"$pid" 2>/dev/null; kill -9 "$pid" 2>/dev/null; sleep 1; }
CG=/sys/fs/cgroup/sub222_harvest
echo "cell,iters,mean_ms,p50_ms,p95_ms,p99_ms,p999_ms,ns_per_load" > "$OUT/results.csv"
emit(){ echo "$1,${2#VICTIM_RESULT,}" >> "$OUT/results.csv"; echo "  $1: $2"; }

log C0_baseline;   R=$(run_victim); emit C0_baseline "$R"
log C1_default;    A=$(start_aggr ""); sleep 2; R=$(run_victim); stop_aggr "$A"; emit C1_default "$R"
log C2_nice19;     A=$(start_aggr "nice -n 19"); sleep 2; R=$(run_victim); stop_aggr "$A"; emit C2_nice19 "$R"
log C3_sched_idle; A=$(start_aggr "chrt --idle 0"); sleep 2; R=$(run_victim); stop_aggr "$A"; emit C3_sched_idle "$R"
log C4_cpumax50
sudo mkdir -p "$CG" 2>/dev/null; echo "+cpu" | sudo tee /sys/fs/cgroup/cgroup.subtree_control >/dev/null 2>&1
echo "50000 100000" | sudo tee "$CG/cpu.max" >/dev/null 2>&1
A=$(start_aggr ""); sleep 1
# setsid 자식의 실제 aggressor TID 들을 cgroup 으로 이동
for t in $(pgrep -g "$A" 2>/dev/null); do echo "$t" | sudo tee "$CG/cgroup.procs" >/dev/null 2>&1; done
sleep 2; R=$(run_victim); stop_aggr "$A"; emit C4_cpumax50 "$R"
sudo rmdir "$CG" 2>/dev/null
log "=== done ==="
