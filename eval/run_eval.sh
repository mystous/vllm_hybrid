#!/usr/bin/env bash
# =============================================================================
# run_eval.sh — GPU-only / Hybrid full evaluation pipeline
#
# Execution order:
#   1. Start GPU-only server
#   2. Start monitor (GPU/CPU utilization)
#   3. Wait for server to be ready
#   4. Run benchmark
#   5. Stop monitor, stop server
#   6. Start Hybrid server
#   7. Start monitor
#   8. Wait for server to be ready
#   9. Run benchmark
#  10. Stop monitor, stop server
#  11. Generate comparison report
#
# Usage:
#   ./run_eval.sh <env_file> [mode]
#
#   env_file: path to .env config (e.g. env/h100x8.env)
#   mode    : all | gpu_only | hybrid | compare  (default: all)
#
# Examples:
#   ./run_eval.sh env/dev_rtx3090.env
#   ./run_eval.sh env/h100x8.env hybrid
#   ./run_eval.sh env/h100x8.env compare
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <env_file> [mode]" >&2
    echo "  env_file: path to .env config (e.g. env/h100x8.env)" >&2
    echo "  mode    : all | gpu_only | hybrid | compare (default: all)" >&2
    echo "" >&2
    echo "Available env files:" >&2
    for f in "${SCRIPT_DIR}"/envs/*.env; do
        echo "  ${f##"${SCRIPT_DIR}/"}" >&2
    done
    exit 1
fi

ENV_FILE="$1"
# Resolve relative path against script dir
if [[ ! "${ENV_FILE}" = /* ]]; then
    ENV_FILE="${SCRIPT_DIR}/${ENV_FILE}"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[ERROR] env file not found: ${ENV_FILE}" >&2
    exit 1
fi

# Export so sub-scripts (serve.sh, benchmark.sh) can inherit
export EVAL_ENV_FILE="${ENV_FILE}"

# shellcheck disable=SC1090
set -a          # auto-export all variables from env file
source "${ENV_FILE}"
set +a

# ---------------------------------------------------------------------------
# Pre-flight: check model is downloaded
# ---------------------------------------------------------------------------
_model_check() {
    python3 - "${MODEL}" <<'PYEOF'
import sys
try:
    from huggingface_hub import try_to_load_from_cache, scan_cache_dir
    model_id = sys.argv[1]
    # Check if config.json exists in cache (reliable indicator)
    result = try_to_load_from_cache(model_id, "config.json")
    if result is None or isinstance(result, type(None)):
        sys.exit(1)
except Exception:
    sys.exit(1)
PYEOF
}

if ! _model_check 2>/dev/null; then
    echo "" >&2
    echo "[ERROR] Model not found in local cache: ${MODEL}" >&2
    echo "" >&2
    echo "  Please download the model first:" >&2
    echo "" >&2
    echo "    huggingface-cli download ${MODEL}" >&2
    echo "" >&2
    echo "  Or in Python:" >&2
    echo "" >&2
    echo "    from huggingface_hub import snapshot_download" >&2
    echo "    snapshot_download(\"${MODEL}\")" >&2
    echo "" >&2
    exit 1
fi

RESULTS_BASE="${SCRIPT_DIR}/${RESULTS_DIR:-results}"

# Run timestamp + hardware/model tag
# Format: YYYYMMDD_HHMMSS_<gputype>_x<gpucount>_<modelname>
_TS="$(TZ=Asia/Seoul date '+%Y%m%d_%H%M%S')"
_GPU_TYPE=""
_GPU_COUNT="0"
if command -v nvidia-smi &>/dev/null; then
    _GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 \
        | sed 's/NVIDIA //;s/ /_/g')
    _GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi
_GPU_TYPE="${_GPU_TYPE:-unknown}"
_MODEL_SHORT="${MODEL##*/}"  # "Qwen/Qwen2.5-72B-Instruct" → "Qwen2.5-72B-Instruct"

RUN_TS="${RUN_TS:-${_TS}_${_GPU_TYPE}_x${_GPU_COUNT}_${_MODEL_SHORT}}"
RESULTS_DIR="${RESULTS_BASE}/${RUN_TS}"
export EVAL_RUN_DIR="${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"


MODE="${2:-all}"  # all / gpu_only / hybrid / compare

SERVER_PID=""
MONITOR_PID=""

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

log() { echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

wait_for_server() {
    local url="http://localhost:${PORT}/health"
    local timeout="${SERVER_READY_TIMEOUT:-300}"
    local poll="${SERVER_READY_POLL:-3}"
    local elapsed=0

    log "Waiting for server to be ready... (up to ${timeout}s)"
    while ! curl -sf "$url" > /dev/null 2>&1; do
        if [[ $elapsed -ge $timeout ]]; then
            log "[ERROR] Server startup timed out (exceeded ${timeout}s)"
            return 1
        fi
        sleep "$poll"
        elapsed=$((elapsed + poll))
        log "  Waiting... ${elapsed}/${timeout}s"
    done
    log "Server ready (took ${elapsed}s)"
}

start_server() {
    local server_mode="$1"
    log "=== Starting server: MODE=${server_mode} ==="
    bash "${SCRIPT_DIR}/serve.sh" "${server_mode}" \
        > "${RESULTS_DIR}/${server_mode}_serve.log" 2>&1 &
    SERVER_PID=$!
    log "Server PID: ${SERVER_PID}"
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "Stopping server (PID=${SERVER_PID})"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
    # Clean up any remaining vLLM processes
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3
}

start_monitor() {
    local prefix="$1"
    local interval="${MONITOR_INTERVAL:-1}"
    log "Starting monitor: prefix=${prefix}"
    python3 "${SCRIPT_DIR}/monitor.py" "${prefix}" --interval "${interval}" \
        > "${RESULTS_DIR}/monitor_${prefix##*/}.log" 2>&1 &
    MONITOR_PID=$!
    log "Monitor PID: ${MONITOR_PID}"
}

stop_monitor() {
    if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        log "Stopping monitor (PID=${MONITOR_PID})"
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
        MONITOR_PID=""
    fi
}

cleanup() {
    log "Cleaning up..."
    stop_monitor
    stop_server
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# System information collection
# ---------------------------------------------------------------------------

collect_system_info() {
    local outfile="${RESULTS_DIR}/system_info.json"
    log "Collecting system information → ${outfile}"

    python3 - "${outfile}" <<'PYEOF'
import json, os, sys, subprocess, platform, re, shutil
from pathlib import Path
from datetime import datetime, timezone

def run(cmd, **kwargs):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10, **kwargs)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""

def parse_lscpu():
    raw = run(["lscpu"])
    if not raw:
        return {}
    d = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d

info = {}

# --- Timestamp & environment ---
info["timestamp"] = datetime.now(timezone.utc).isoformat()
info["hostname"] = platform.node()

# --- OS / Kernel ---
info["os"] = {
    "system": platform.system(),
    "release": platform.release(),
    "version": platform.version(),
    "distro": run(["lsb_release", "-ds"]) or run(["cat", "/etc/os-release"]).split("\n")[0] if Path("/etc/os-release").exists() else "",
}

# --- CPU ---
lscpu = parse_lscpu()
info["cpu"] = {
    "model_name": lscpu.get("Model name", ""),
    "architecture": lscpu.get("Architecture", platform.machine()),
    "sockets": int(lscpu.get("Socket(s)", 1)),
    "cores_per_socket": int(lscpu.get("Core(s) per socket", 0)),
    "threads_per_core": int(lscpu.get("Thread(s) per core", 0)),
    "total_cpus": int(lscpu.get("CPU(s)", os.cpu_count() or 0)),
    "cpu_max_mhz": lscpu.get("CPU max MHz", ""),
    "cpu_min_mhz": lscpu.get("CPU min MHz", ""),
    "l1d_cache": lscpu.get("L1d cache", ""),
    "l1i_cache": lscpu.get("L1i cache", ""),
    "l2_cache": lscpu.get("L2 cache", ""),
    "l3_cache": lscpu.get("L3 cache", ""),
    "flags": [],
}
# ISA flags relevant to vLLM hybrid
cpu_flags_raw = lscpu.get("Flags", "")
if not cpu_flags_raw:
    cpuinfo = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    m = re.search(r"^flags\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    if m:
        cpu_flags_raw = m.group(1)
important_flags = [
    "avx", "avx2", "avx512f", "avx512bw", "avx512vl", "avx512_vnni",
    "amx_bf16", "amx_int8", "amx_tile", "sse4_1", "sse4_2", "fma",
]
all_flags = cpu_flags_raw.lower().split()
info["cpu"]["flags"] = [f for f in important_flags if f in all_flags]

# --- NUMA ---
numa_info = run(["numactl", "--hardware"])
if numa_info:
    info["numa"] = {"raw": numa_info}
    nodes = re.findall(r"node (\d+) size: (\d+) MB", numa_info)
    info["numa"]["nodes"] = [{"node": int(n), "size_mb": int(s)} for n, s in nodes]
    info["numa"]["num_nodes"] = len(nodes)
else:
    # Fallback: count from /sys
    numa_path = Path("/sys/devices/system/node")
    if numa_path.exists():
        nodes = sorted(p.name for p in numa_path.glob("node*"))
        info["numa"] = {"num_nodes": len(nodes), "nodes": [{"node": int(n.replace("node",""))} for n in nodes]}
    else:
        info["numa"] = {"num_nodes": 1}

# --- Memory ---
meminfo_path = Path("/proc/meminfo")
if meminfo_path.exists():
    memraw = meminfo_path.read_text()
    mem = {}
    for line in memraw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            mem[k.strip()] = v.strip()
    info["memory"] = {
        "total": mem.get("MemTotal", ""),
        "available": mem.get("MemAvailable", ""),
        "swap_total": mem.get("SwapTotal", ""),
        "hugepages_total": mem.get("HugePages_Total", ""),
        "hugepage_size": mem.get("Hugepagesize", ""),
    }

# --- GPU (nvidia-smi) ---
if shutil.which("nvidia-smi"):
    # CSV query for structured data
    gpu_csv = run([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version,pci.bus_id,power.limit,clocks.max.sm,clocks.max.mem,temperature.gpu,compute_mode",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    if gpu_csv:
        for line in gpu_csv.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 12:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "memory_total_mib": parts[3],
                    "memory_free_mib": parts[4],
                    "driver_version": parts[5],
                    "pci_bus_id": parts[6],
                    "power_limit_w": parts[7],
                    "max_sm_clock_mhz": parts[8],
                    "max_mem_clock_mhz": parts[9],
                    "temperature_c": parts[10],
                    "compute_mode": parts[11],
                })
    info["gpu"] = {
        "count": len(gpus),
        "devices": gpus,
    }
    # CUDA version
    cuda_ver = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    nvcc_ver = run(["nvcc", "--version"])
    cuda_match = re.search(r"release (\S+),", nvcc_ver) if nvcc_ver else None
    info["gpu"]["cuda_version"] = cuda_match.group(1) if cuda_match else ""
    info["gpu"]["driver_version"] = gpus[0]["driver_version"] if gpus else ""
    # GPU topology (NVLink etc.)
    topo = run(["nvidia-smi", "topo", "-m"])
    if topo:
        info["gpu"]["topology"] = topo
else:
    info["gpu"] = {"count": 0, "devices": [], "note": "nvidia-smi not found"}

# --- Python / vLLM / torch ---
info["software"] = {
    "python_version": platform.python_version(),
}
try:
    import torch
    info["software"]["torch_version"] = torch.__version__
    info["software"]["torch_cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["software"]["torch_cuda_version"] = torch.version.cuda or ""
        info["software"]["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else ""
except ImportError:
    pass
try:
    import vllm
    info["software"]["vllm_version"] = getattr(vllm, "__version__", "unknown")
except ImportError:
    pass
try:
    import intel_extension_for_pytorch as ipex
    info["software"]["ipex_version"] = ipex.__version__
except ImportError:
    info["software"]["ipex_version"] = "not installed"

# --- Disk I/O (workspace) ---
disk_usage = run(["df", "-h", "."])
if disk_usage:
    lines = disk_usage.splitlines()
    if len(lines) >= 2:
        parts = lines[1].split()
        info["disk"] = {
            "filesystem": parts[0] if len(parts) > 0 else "",
            "size": parts[1] if len(parts) > 1 else "",
            "used": parts[2] if len(parts) > 2 else "",
            "available": parts[3] if len(parts) > 3 else "",
        }

# --- Kernel parameters (relevant to performance) ---
info["kernel"] = {}
sysctl_keys = [
    "vm.swappiness", "vm.nr_hugepages", "vm.overcommit_memory",
    "kernel.numa_balancing", "kernel.sched_min_granularity_ns",
]
for key in sysctl_keys:
    val = run(["sysctl", "-n", key])
    if val:
        info["kernel"][key] = val

# --- Env file contents ---
env_file = os.environ.get("EVAL_ENV_FILE", "")
if env_file and Path(env_file).exists():
    info["eval_env"] = Path(env_file).read_text()

# --- Hybrid config (structured) ---
info["hybrid_config"] = {
    "routing_strategy": os.environ.get("HYBRID_ROUTING_STRATEGY", "capacity"),
    "routing_priority": os.environ.get("HYBRID_ROUTING_PRIORITY", "gpu-first"),
    "cpu_max_seqs": os.environ.get("HYBRID_CPU_MAX_SEQS", "0"),
    "cpu_kvcache_gb": os.environ.get("HYBRID_CPU_KVCACHE_GB", "0"),
    "cpu_threads": os.environ.get("HYBRID_CPU_THREADS", "0"),
    "numa_aware": os.environ.get("HYBRID_NUMA_AWARE", "true"),
    "num_cpu_engines": os.environ.get("HYBRID_NUM_CPU_ENGINES", "1"),
    "stats_log_interval": os.environ.get("HYBRID_STATS_LOG_INTERVAL", "50"),
}

# Write JSON
outfile = sys.argv[1]
with open(outfile, "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print(f"System info saved: {outfile}")
PYEOF
}

# ---------------------------------------------------------------------------
# GPU-only evaluation
# ---------------------------------------------------------------------------

run_gpu_only() {
    log "=========================================="
    log " [1/2] Starting GPU-only evaluation"
    log "=========================================="

    stop_server   # Clean up any leftover server

    start_server "gpu_only"
    start_monitor "${RESULTS_DIR}/gpu_only_monitor"

    # Stream key events to console
    local serve_log="${RESULTS_DIR}/gpu_only_serve.log"
    tail -f "${serve_log}" 2>/dev/null | grep --line-buffered -E \
        "Initializing.*engine|Loading model|loaded model|CUDA graphs|ERROR|FATAL|Traceback" | \
        while IFS= read -r line; do
            log "[gpu] ${line##*] }"
        done &
    TAIL_PID=$!

    wait_for_server

    log "--- Running GPU-only benchmark ---"
    bash "${SCRIPT_DIR}/benchmark.sh" "gpu_only"
    stop_monitor

    kill "${TAIL_PID}" 2>/dev/null; wait "${TAIL_PID}" 2>/dev/null || true

    stop_server

    log "GPU-only evaluation complete."
}

# ---------------------------------------------------------------------------
# Hybrid evaluation
# ---------------------------------------------------------------------------

run_hybrid() {
    log "=========================================="
    log " [2/2] Starting Hybrid evaluation"
    log "=========================================="

    stop_server

    start_server "hybrid"
    start_monitor "${RESULTS_DIR}/hybrid_monitor"

    # Stream key hybrid events to console in background
    local serve_log="${RESULTS_DIR}/hybrid_serve.log"
    tail -f "${serve_log}" 2>/dev/null | grep --line-buffered -E \
        "CapacityAwareRouter init|Warmup profiling|Router stats|CPU_EngineCore.*Starting|CPU_EngineCore.*Resolved|GPU_EngineCore.*Initializing|Loading model|loaded model|CUDA graphs|cpu_max_num_seqs|ERROR|FATAL|Traceback" | \
        while IFS= read -r line; do
            log "[hybrid] ${line##*] }"
        done &
    TAIL_PID=$!

    wait_for_server

    log "--- Running Hybrid benchmark ---"
    bash "${SCRIPT_DIR}/benchmark.sh" "hybrid"
    stop_monitor

    # Stop log tail
    kill "${TAIL_PID}" 2>/dev/null; wait "${TAIL_PID}" 2>/dev/null || true

    stop_server

    log "Hybrid evaluation complete."
}

# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

run_compare() {
    log "=========================================="
    log " Generating comparison report"
    log "=========================================="
    python3 "${SCRIPT_DIR}/compare.py" \
        --results-dir "${RESULTS_DIR}" \
        --gpu-label gpu_only \
        --hybrid-label hybrid
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

log "Eval starting: MODE=${MODE}"
log "ENV_FILE: ${ENV_FILE}"
log "RUN_TS: ${RUN_TS}"
log "Results path: ${RESULTS_DIR}"
log "------------------------------------------------------------"
log "  Model          : ${MODEL}"
log "  Port           : ${PORT}"
log "  TP size        : ${TENSOR_PARALLEL_SIZE:-1}"
log "  GPU mem util   : ${GPU_MEMORY_UTIL}"
log "  Num prompts    : ${NUM_PROMPTS}"
log "  Input len      : ${INPUT_LEN}"
log "  Output len     : ${OUTPUT_LEN}"
log "  Request rate   : ${REQUEST_RATE}"
log "  CPU engines    : ${HYBRID_NUM_CPU_ENGINES:-1}"
log "  NUMA aware     : ${HYBRID_NUMA_AWARE:-true}"
log "  Routing        : ${HYBRID_ROUTING_STRATEGY:-capacity} (${HYBRID_ROUTING_PRIORITY:-gpu-first})"
log "  CPU max seqs   : ${HYBRID_CPU_MAX_SEQS:-0 (auto)}"
log "  CPU kvcache GB : ${HYBRID_CPU_KVCACHE_GB:-0 (auto)}"
log "  CPU threads    : ${HYBRID_CPU_THREADS:-0 (auto)}"
log "  Monitor intv   : ${MONITOR_INTERVAL:-1}s"
log "------------------------------------------------------------"

# Collect system info before any benchmark
collect_system_info

case "${MODE}" in
    all)
        run_gpu_only
        run_hybrid
        run_compare
        ;;
    gpu_only)
        run_gpu_only
        ;;
    hybrid)
        run_hybrid
        ;;
    compare)
        run_compare
        ;;
    *)
        echo "Usage: $0 <env_file> [all|gpu_only|hybrid|compare]" >&2
        exit 1
        ;;
esac

log "Done."
