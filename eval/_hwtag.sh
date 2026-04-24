#!/usr/bin/env bash
# =============================================================================
# _hwtag.sh — Compute HW_TAG="<cpu>x<sockets>_<gpu>x<count>" for dir naming.
# Sourced by serve/bench/run scripts. Sets $HW_TAG.
# =============================================================================

_sanitize() {
    # Compact a string for safe use in filenames: collapse whitespace to '_',
    # drop common marketing noise, strip leading/trailing separators.
    echo "$1" | sed -E '
        s/\(R\)//g; s/\(TM\)//g; s/\(tm\)//g;
        s/CPU @ [0-9.]+GHz//g;
        s/Processor//g;
        s/[0-9]+-Core//g;
        s/@ [0-9.]+GHz//g;
        s/[[:space:]]+/ /g;
        s/^ +//; s/ +$//;
        s/ /_/g;
    ' | tr -s '_'
}

_cpu_raw=""
_cpu_sockets="1"
if command -v lscpu &>/dev/null; then
    _cpu_raw=$(lscpu | awk -F: '/Model name/{gsub(/^[ \t]+/,"",$2); print $2; exit}')
    _cpu_sockets=$(lscpu | awk -F: '/^Socket\(s\)/{gsub(/^[ \t]+/,"",$2); print $2; exit}')
fi
_cpu_sockets="${_cpu_sockets:-1}"
_cpu_short=$(_sanitize "${_cpu_raw:-unknownCPU}")
CPU_TAG="${_cpu_short}x${_cpu_sockets}"

if command -v nvidia-smi &>/dev/null; then
    _gpu_raw=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    _gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    _gpu_short=$(_sanitize "${_gpu_raw/NVIDIA /}")
    GPU_TAG="${_gpu_short:-unknownGPU}x${_gpu_count:-0}"
else
    GPU_TAG="noGPU"
fi

HW_TAG="${CPU_TAG}_${GPU_TAG}"
export HW_TAG CPU_TAG GPU_TAG
