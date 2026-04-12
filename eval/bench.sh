#!/usr/bin/env bash
# =============================================================================
# bench.sh — Benchmark runner (서버가 이미 실행 중인 상태에서 사용)
#
# Usage:
#   ./bench.sh <mode> <env_file>
#   mode     : gpu_only | hybrid
#   env_file : path to .env config
#
# Examples:
#   ./bench.sh gpu_only envs/dev_rtx3090.env
#   ./bench.sh hybrid  envs/h100x4_cpu_first.env
#
# 서버를 먼저 띄운 뒤 실행:
#   Terminal 1: ./serve.sh hybrid envs/h100x4_cpu_first.env
#   Terminal 2: ./bench.sh hybrid envs/h100x4_cpu_first.env
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <gpu_only|hybrid> <env_file>" >&2
    exit 1
fi

MODE="$1"
if [[ "${MODE}" != "gpu_only" && "${MODE}" != "hybrid" ]]; then
    echo "[ERROR] MODE must be 'gpu_only' or 'hybrid'." >&2
    exit 1
fi

ENV_ARG="$2"
if [[ ! "${ENV_ARG}" = /* ]]; then
    ENV_ARG="${SCRIPT_DIR}/${ENV_ARG}"
fi
ENV_FILE="${ENV_ARG}"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[ERROR] env file not found: ${ENV_FILE}" >&2
    exit 1
fi

# Source env (auto-export for sub-processes)
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

log() { echo "[$(TZ=Asia/Seoul date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Model pre-check
# ---------------------------------------------------------------------------
_model_check() {
    python3 - "${MODEL}" <<'PYEOF'
import sys
try:
    from huggingface_hub import try_to_load_from_cache
    model_id = sys.argv[1]
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
    echo "  Please download first: huggingface-cli download ${MODEL}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Server health check
# ---------------------------------------------------------------------------
_check_server() {
    curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1
}

if ! _check_server; then
    echo "" >&2
    echo "[ERROR] Server not running on port ${PORT}" >&2
    echo "  Start the server first:" >&2
    echo "    ./serve.sh ${MODE} ${ENV_FILE}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------
RESULTS_BASE="${SCRIPT_DIR}/${RESULTS_DIR:-results}"

_TS="$(TZ=Asia/Seoul date '+%Y%m%d_%H%M%S')"
_GPU_TYPE=""
_GPU_COUNT="0"
if command -v nvidia-smi &>/dev/null; then
    _GPU_LINES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
    if [[ -n "${_GPU_LINES}" ]]; then
        _GPU_TYPE=$(echo "${_GPU_LINES}" | head -1 | sed 's/NVIDIA //;s/ /_/g')
        _GPU_COUNT=$(echo "${_GPU_LINES}" | wc -l)
    fi
fi
_GPU_TYPE="${_GPU_TYPE:-unknown}"
_MODEL_SHORT="${MODEL##*/}"

# Mode tag: _G_ = gpu_only, _H_ = hybrid
if [[ "${MODE}" == "gpu_only" ]]; then
    _MODE_TAG="G"
else
    _MODE_TAG="H"
fi

# Priority tag: _C_ = cpu-first, _G_ = gpu-first, _R_ = round-robin
_STRATEGY="${HYBRID_ROUTING_STRATEGY:-capacity}"
_PRIORITY="${HYBRID_ROUTING_PRIORITY:-gpu-first}"
if [[ "${MODE}" == "gpu_only" ]]; then
    _PRI_TAG=""
elif [[ "${_STRATEGY}" == "round-robin" ]]; then
    _PRI_TAG="R"
elif [[ "${_PRIORITY}" == "cpu-first" ]]; then
    _PRI_TAG="C"
else
    _PRI_TAG="G"
fi

# Directory name: TS_MODE_PRI_GPU_MODEL
# e.g. 20260407_180242_H_C_H100_80GB_HBM3_x4_Qwen2.5-7B-Instruct
if [[ -n "${_PRI_TAG}" ]]; then
    _TAG="${_MODE_TAG}_${_PRI_TAG}"
else
    _TAG="${_MODE_TAG}"
fi

RUN_DIR="${RESULTS_BASE}/${_TS}_${_TAG}_${_GPU_TYPE}_x${_GPU_COUNT}_${_MODEL_SHORT}"
mkdir -p "${RUN_DIR}"

log "============================================================"
log " Benchmark: MODE=${MODE}"
log " Model          : ${MODEL}"
log " Port           : ${PORT}"
log " Routing        : ${HYBRID_ROUTING_STRATEGY:-capacity} (${HYBRID_ROUTING_PRIORITY:-gpu-first})"
log " CPU max seqs   : ${HYBRID_CPU_MAX_SEQS:-0 (auto)}"
log " Num prompts    : ${NUM_PROMPTS}"
log " Input/Output   : ${INPUT_LEN}/${OUTPUT_LEN}"
log " Request rate   : ${REQUEST_RATE}"
log " ENV_FILE       : ${ENV_FILE}"
log " Results        : ${RUN_DIR}"
log "============================================================"

# ---------------------------------------------------------------------------
# Collect system info
# ---------------------------------------------------------------------------
log "Collecting system info..."
python3 - "${RUN_DIR}/system_info.json" <<'PYEOF'
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
info["timestamp"] = datetime.now(timezone.utc).isoformat()
info["hostname"] = platform.node()

info["os"] = {
    "system": platform.system(),
    "release": platform.release(),
    "version": platform.version(),
    "distro": run(["lsb_release", "-ds"]) or "",
}

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

numa_info = run(["numactl", "--hardware"])
if numa_info:
    info["numa"] = {"raw": numa_info}
    nodes = re.findall(r"node (\d+) size: (\d+) MB", numa_info)
    info["numa"]["nodes"] = [{"node": int(n), "size_mb": int(s)} for n, s in nodes]
    info["numa"]["num_nodes"] = len(nodes)
else:
    numa_path = Path("/sys/devices/system/node")
    if numa_path.exists():
        nodes = sorted(p.name for p in numa_path.glob("node*"))
        info["numa"] = {"num_nodes": len(nodes), "nodes": [{"node": int(n.replace("node",""))} for n in nodes]}
    else:
        info["numa"] = {"num_nodes": 1}

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

if shutil.which("nvidia-smi"):
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
                    "index": int(parts[0]), "name": parts[1], "uuid": parts[2],
                    "memory_total_mib": parts[3], "memory_free_mib": parts[4],
                    "driver_version": parts[5], "pci_bus_id": parts[6],
                    "power_limit_w": parts[7], "max_sm_clock_mhz": parts[8],
                    "max_mem_clock_mhz": parts[9], "temperature_c": parts[10],
                    "compute_mode": parts[11],
                })
    info["gpu"] = {"count": len(gpus), "devices": gpus}
    nvcc_ver = run(["nvcc", "--version"])
    cuda_match = re.search(r"release (\S+),", nvcc_ver) if nvcc_ver else None
    info["gpu"]["cuda_version"] = cuda_match.group(1) if cuda_match else ""
    info["gpu"]["driver_version"] = gpus[0]["driver_version"] if gpus else ""
    topo = run(["nvidia-smi", "topo", "-m"])
    if topo:
        info["gpu"]["topology"] = topo
else:
    info["gpu"] = {"count": 0, "devices": []}

info["software"] = {"python_version": platform.python_version()}
try:
    import torch
    info["software"]["torch_version"] = torch.__version__
    info["software"]["torch_cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["software"]["torch_cuda_version"] = torch.version.cuda or ""
except ImportError:
    pass
try:
    import vllm
    info["software"]["vllm_version"] = getattr(vllm, "__version__", "unknown")
except ImportError:
    pass

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

info["benchmark_config"] = {
    "mode": os.environ.get("_BENCH_MODE", ""),
    "num_prompts": os.environ.get("NUM_PROMPTS", ""),
    "input_len": os.environ.get("INPUT_LEN", ""),
    "output_len": os.environ.get("OUTPUT_LEN", ""),
    "request_rate": os.environ.get("REQUEST_RATE", ""),
}

outfile = sys.argv[1]
with open(outfile, "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print(f"System info saved: {outfile}")
PYEOF

# ---------------------------------------------------------------------------
# Start monitor
# ---------------------------------------------------------------------------
MONITOR_PID=""
start_monitor() {
    local interval="${MONITOR_INTERVAL:-1}"
    log "Starting monitor (interval=${interval}s)"
    python3 "${SCRIPT_DIR}/monitor.py" "${RUN_DIR}/${MODE}_monitor" \
        --interval "${interval}" \
        > "${RUN_DIR}/monitor_${MODE}.log" 2>&1 &
    MONITOR_PID=$!
}

stop_monitor() {
    if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        log "Stopping monitor"
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
        MONITOR_PID=""
    fi
}

trap stop_monitor EXIT INT TERM

# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------
export _BENCH_MODE="${MODE}"

RESULT_FILE="${RUN_DIR}/${MODE}.json"
LOG_FILE="${RUN_DIR}/${MODE}_bench.log"

# Wall time measurement
WALL_START=$(date +%s.%N)

start_monitor

log "--- Running benchmark ---"
python -u "${VLLM_ROOT}/benchmarks/benchmark_serving.py" \
    --backend vllm \
    --base-url "http://localhost:${PORT}" \
    --model "${MODEL}" \
    --dataset-name random \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate "${REQUEST_RATE}" \
    --save-result \
    --result-filename "${RESULT_FILE}" \
    2>&1 | tee "${LOG_FILE}"

stop_monitor

WALL_END=$(date +%s.%N)
WALL_SECS=$(python3 -c "print(f'{${WALL_END} - ${WALL_START}:.2f}')")

# Append wall_time to result JSON
if [[ -f "${RESULT_FILE}" ]]; then
    python3 -c "
import json
with open('${RESULT_FILE}') as f:
    d = json.load(f)
d['wall_time_s'] = ${WALL_SECS}
with open('${RESULT_FILE}', 'w') as f:
    json.dump(d, f, indent=2)
"
fi

log "============================================================"
log " Done!  (wall time: ${WALL_SECS}s)"
log " Results : ${RUN_DIR}"
log " Benchmark: ${RESULT_FILE}"
log " Monitor  : ${RUN_DIR}/${MODE}_monitor_gpu.csv"
log "            ${RUN_DIR}/${MODE}_monitor_cpu.csv"
log "============================================================"
