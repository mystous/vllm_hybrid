#!/bin/bash
# Server lifecycle helpers — wait_ready, kill_pgroup, kill_gpu_orphans.
HERE_LC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

wait_ready() {
    local PORT=$1
    local TIMEOUT=${2:-600}
    local START=$SECONDS
    while true; do
        if curl -fs "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        if (( SECONDS - START > TIMEOUT )); then
            echo "[wait_ready] TIMEOUT port=$PORT after ${TIMEOUT}s"
            return 1
        fi
        sleep 5
    done
}

kill_pgroup() {
    local PIDFILE=$1
    if [ ! -f "$PIDFILE" ]; then return; fi
    local PID=$(cat "$PIDFILE")
    if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then return; fi
    local PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
    if [ -n "$PGID" ]; then
        kill -TERM -"$PGID" 2>/dev/null
        sleep 3
        kill -KILL -"$PGID" 2>/dev/null
    fi
    rm -f "$PIDFILE"
}

kill_gpu_orphans() {
    local GPU=$1
    # nvidia-smi only knows the visible UUIDs; we map GPU index via --query-gpu first
    local UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null)
    [ -z "$UUID" ] && return
    nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null \
        | awk -F', ' -v U="$UUID" '$2==U {print $1}' \
        | xargs -r -n1 kill -9 2>/dev/null
}

kill_all_servers() {
    # Kill anything left for our profile dir.
    local LOG_DIR=$1
    for pf in "$LOG_DIR"/*.pid; do
        [ -e "$pf" ] || continue
        kill_pgroup "$pf"
    done
}

wait_gpu_free() {
    local GPU=$1
    local TIMEOUT=${2:-90}
    local START=$SECONDS
    while true; do
        local UM=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [ -z "$UM" ] || [ "$UM" -lt 1000 ]; then return 0; fi
        if (( SECONDS - START > TIMEOUT )); then
            echo "[wait_gpu_free] TIMEOUT gpu=$GPU used=${UM}MiB"
            return 1
        fi
        sleep 2
    done
}
