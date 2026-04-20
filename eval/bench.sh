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

# ─────────────────────────────────────────────────────────────────────
# 서버 로그 byte offset 기록 — 이 시점 이후 기록된 로그만 이번 run 에
# 속함. bench 완료 후 이 offset 부터 tail 로 잘라 RUN_DIR 에 저장.
# 여러 bench 를 같은 서버에 연속으로 돌려도 구간 분리 정확.
# ─────────────────────────────────────────────────────────────────────
SERVER_LOG_SRC="${SCRIPT_DIR}/serve_logs/server_latest.log"
if [[ -f "${SERVER_LOG_SRC}" ]] || [[ -L "${SERVER_LOG_SRC}" ]]; then
    SERVER_LOG_START_BYTES="$(wc -c < "${SERVER_LOG_SRC}" 2>/dev/null || echo 0)"
else
    SERVER_LOG_START_BYTES="0"
fi

# 서버에 명시적 marker 주입도 시도 — 런 구분용.
# /health 같은 엔드포인트가 서버 로그에 찍히지 않을 수 있어서 실패해도 무시.
# 서버 로그에 [BENCH-START] 마커를 남기려면 아래 주석 풀 것:
# curl -sf -X POST "http://localhost:${PORT}/v1/completions" \
#     -H "Content-Type: application/json" \
#     -d "{\"model\":\"${MODEL}\",\"prompt\":\"__BENCH_MARKER_${_TS:-x}__\",\"max_tokens\":1}" \
#     > /dev/null 2>&1 || true

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

# Directory name: TS_MODE_PRI_GPU_MODEL_seqsN
# e.g. 20260407_180242_H_C_H100_80GB_HBM3_x4_Qwen2.5-7B-Instruct_seqs1
if [[ -n "${_PRI_TAG}" ]]; then
    _TAG="${_MODE_TAG}_${_PRI_TAG}"
else
    _TAG="${_MODE_TAG}"
fi

# max_num_seqs 정보 suffix — G0 sweep 에서 1/2/4/8/16 구분용
# hybrid: HYBRID_CPU_MAX_SEQS, gpu_only: 생략
if [[ "${MODE}" == "hybrid" ]]; then
    _SEQS_TAG="_seqs${HYBRID_CPU_MAX_SEQS:-auto}"
else
    _SEQS_TAG=""
fi

RUN_DIR="${RESULTS_BASE}/${_TS}_${_TAG}_${_GPU_TYPE}_x${_GPU_COUNT}_${_MODEL_SHORT}${_SEQS_TAG}"
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

# ─────────────────────────────────────────────────────────────────────
# 서버 로그 구간 추출: bench 시작 시점 (SERVER_LOG_START_BYTES) 부터
# 현재까지 기록된 부분만 RUN_DIR 로 잘라 저장. 같은 서버에서 bench 를
# 여러 번 돌려도 각 run 에 해당하는 로그만 분리됨.
#
# 저장되는 marker 예:
#   [HYBRID-LAUNCH] num_cpu_engines=N ...         ← 서버 부팅 시만 한 번
#     (첫 번째 run 에만 찍힘. 2번째 이후 run 로그엔 이미 찍힌 뒤라 안 포함)
#   [HYBRID-CPU-WORKER] init_cpu_threads_env ...  ← 동일
#   [HYBRID-ROUTER-INIT] strategy=... max_seqs=... ← 첫 route 호출 시
#   [HYBRID-WAVE-DISPATCH] req=... → cpu:N ...    ← 매 CPU dispatch
#   [HYBRID-WAVE] engine=N wave closed/drained    ← wave state 전이
#   [HYBRID-CPU-PROFILE] step=... attn=... mlp=.. ← PROFILE=1 시 매 step
#   [HYBRID-CPU-ATTN-IPEX] call=... num_seqs=...  ← PROFILE=1 시 매 IPEX call
#   [HYBRID-ROUTER-STATS] finished=... CPU=...    ← stats_log_interval 마다
#
# ⚠ 첫 run 과 이후 run 의 차이:
#   - 첫 run: 서버 부팅 로그 (LAUNCH/RESOLVE/CPU-WORKER init) 포함
#   - 이후 run: route/dispatch/profile 만 포함 (부팅은 앞 run 에 속함)
#   - 부팅 정보가 필요하면 첫 run 의 server.log 를 참조
# ─────────────────────────────────────────────────────────────────────
if [[ -f "${SERVER_LOG_SRC}" ]] || [[ -L "${SERVER_LOG_SRC}" ]]; then
    SERVER_LOG_END_BYTES="$(wc -c < "${SERVER_LOG_SRC}" 2>/dev/null || echo 0)"
    LOG_DELTA=$((SERVER_LOG_END_BYTES - SERVER_LOG_START_BYTES))

    # (1) boot.log: 전체 파일에서 부팅 markers 추출 (NUMA/engine 설정 확인용)
    #     서버가 한 번만 부팅하므로 매 run 에 동일 내용. grep 기반이라 언제든 추출 가능.
    #     KERNEL / APPLIED-FEATURES 는 §06+ feature patch/manifest 마커.
    grep -E '\[HYBRID-(LAUNCH|RESOLVE|CPU-ENV|CPU-PROC|CPU-WORKER|ROUTER-INIT|KERNEL|KERNEL-Q8_0|APPLIED-FEATURES|BATCH-AWARE-ATTN)\]|auto thread-binding list|CPU VllmConfig created|Model loading took' \
        "${SERVER_LOG_SRC}" 2>/dev/null > "${RUN_DIR}/${MODE}_server_boot.log" || true
    BOOT_LINES=$(wc -l < "${RUN_DIR}/${MODE}_server_boot.log" 2>/dev/null || echo 0)

    # (2) run.log: 이번 bench 구간의 dispatch/profile/stats
    if [[ "${LOG_DELTA}" -gt 0 ]]; then
        tail -c +$((SERVER_LOG_START_BYTES + 1)) "${SERVER_LOG_SRC}" \
            > "${RUN_DIR}/${MODE}_server_run.log" 2>/dev/null \
            && log "Server run log: ${LOG_DELTA} bytes → ${MODE}_server_run.log" \
            || log "WARN: server run log slice 실패"
    else
        log "WARN: server log 에 새로 기록된 내용 없음 (delta=${LOG_DELTA})"
    fi

    log "Server boot log: ${BOOT_LINES} lines → ${MODE}_server_boot.log"
else
    log "WARN: server log not found at ${SERVER_LOG_SRC}"
    log "      serve.sh 이 tee 지원 버전인지 확인 필요"
fi

# ─────────────────────────────────────────────────────────────────────
# PROFILE 모드 manifest (applied_features.json / env_snapshot.txt /
# git_sha.txt) 를 RUN_DIR 로 복사. serve.sh 가 VLLM_HYBRID_RESULT_DIR
# 에 이미 생성해둠.
# ─────────────────────────────────────────────────────────────────────
PROFILE_DIR="${VLLM_HYBRID_RESULT_DIR:-${SCRIPT_DIR}/serve_logs/profile_latest}"
if [[ -d "${PROFILE_DIR}" ]]; then
    for f in applied_features.json env_snapshot.txt git_sha.txt; do
        if [[ -f "${PROFILE_DIR}/${f}" ]]; then
            cp "${PROFILE_DIR}/${f}" "${RUN_DIR}/${f}" && \
                log "Profile manifest: ${f} → ${RUN_DIR}/" || true
        fi
    done
fi

log "============================================================"
log " Done!  (wall time: ${WALL_SECS}s)"
log " Results : ${RUN_DIR}"
log " Benchmark: ${RESULT_FILE}"
log " Monitor  : ${RUN_DIR}/${MODE}_monitor_gpu.csv"
log "            ${RUN_DIR}/${MODE}_monitor_cpu.csv"
log " Server   : ${RUN_DIR}/${MODE}_server.log"
log "============================================================"
