#!/usr/bin/env bash
# [perf-compare] vanilla vs v1.3 — 각 3 회 sequential + metric 별 avg/min/max 집계.
# Goal: v1.3 (try68 stack — D6+D7+D8+D10+D11+D12 default 0) 의 실측 throughput
#       이 vanilla (NEO OFF, 현 codebase) 와 비교. bistability 영향 측정.
#
# 6 회 × ~25 min = 약 150 분 소요.
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT_DIR"

TS_BASE="$(date +%Y%m%d_%H%M%S)"
SUITE_DIR="${ROOT_DIR}/eval/results/${TS_BASE}_perf_compare_v13_vs_vanilla"
mkdir -p "${SUITE_DIR}"
SUITE_LOG="${SUITE_DIR}/suite.log"

PY=/workspace/vllm_dev_prj/bin/python
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

log() {
    local msg="$1"
    echo "[$(TZ=Asia/Seoul date -Iseconds)] $msg" | tee -a "${SUITE_LOG}"
}

clear_neo_env() {
    unset VLLM_NEO_FORCE_SWAP_IN VLLM_NEO_MAX_SWAP_IN_PER_STEP
    unset VLLM_NEO_CPU_RESIDENT_REQS VLLM_NEO_SWAP_IN_ORDER
    unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN
    unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
    unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
    unset VLLM_NEO_LRU_FALLBACK_FIFO
    unset VLLM_NEO_PREDICTIVE_THRESHOLD VLLM_NEO_SWAP_COOLDOWN
}

setup_v13_env() {
    # try68 env (v1.3 reproducibility 시도 — bistability 의 active 평형 진입은
    # 보장 안 됨)
    export VLLM_NEO_FORCE_SWAP_IN=1
    export VLLM_NEO_MAX_SWAP_IN_PER_STEP=4
    export VLLM_NEO_CPU_RESIDENT_REQS=64
    export VLLM_NEO_SWAP_IN_ORDER=oldest
}

run_one() {
    local config="$1"
    local idx="$2"
    local extra_flags="$3"
    local sub_dir="${SUITE_DIR}/${config}_run${idx}"
    mkdir -p "${sub_dir}"
    local log_file="${sub_dir}/engine.log"

    log "[$config run $idx/3] starting (extra: ${extra_flags:-none})"
    "$PY" -u "${SCRIPT_DIR}/run_neo_baseline.py" \
        --model llama-70b \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 16384 \
        --max-num-seqs 256 \
        --num-prompts 500 \
        --target-input-len 8192 \
        --max-tokens 8192 \
        ${extra_flags} \
        --async-scheduling \
        --enforce-eager false \
        --kv-cache-dtype fp8 \
        --max-num-batched-tokens 8192 \
        --log-file "${log_file}" \
        --output-file "${sub_dir}/result.json" \
        > "${log_file}.stdout" 2>&1
    local rc=$?
    log "[$config run $idx/3] exit=${rc} result=$(test -f ${sub_dir}/result.json && echo OK || echo FAIL)"
    sleep 5
}

log "===== suite start — vanilla 3 + v1.3 3 ====="

# Vanilla: NEO OFF (flag 미부여)
clear_neo_env
for i in 1 2 3; do
    run_one "vanilla" "${i}" ""
done

# v1.3: NEO ON + try68 env
clear_neo_env
setup_v13_env
for i in 1 2 3; do
    run_one "v13" "${i}" "--enable-neo-asymmetric"
done

log "===== suite runs done — 집계 시작 ====="

# 집계
"$PY" - <<EOF | tee -a "${SUITE_LOG}"
import json, glob, os, statistics

suite = '${SUITE_DIR}'
configs = ['vanilla', 'v13']
metrics = [
    ('output_tps', 'output tokens/s', '%.2f', 'higher_better'),
    ('prompt_tps', 'prompt tokens/s', '%.2f', 'higher_better'),
    ('generate_wall_s', 'generate wall (s)', '%.2f', 'lower_better'),
    ('req_per_s', 'req/s', '%.4f', 'higher_better'),
    ('init_s', 'init (s)', '%.2f', 'lower_better'),
    ('total_output_tokens', 'total out tokens', '%d', 'higher_better'),
]

all_data = {}
for config in configs:
    paths = sorted(glob.glob(os.path.join(suite, f'{config}_run*', 'result.json')))
    rows = []
    for p in paths:
        try:
            with open(p) as f:
                rows.append(json.load(f))
        except Exception as e:
            print(f'  load fail: {p} ({e})')
    all_data[config] = rows
    print(f'\n[loaded] {config}: {len(rows)} runs')

print('\n' + '=' * 92)
print(f'{"metric":<22} | {"config":<8} | {"avg":>12} | {"min":>12} | {"max":>12} | n |')
print('-' * 92)
for key, label, fmt, _ in metrics:
    for config in configs:
        rows = all_data[config]
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if vals:
            avg = statistics.mean(vals)
            mn, mx = min(vals), max(vals)
            line = f'{label:<22} | {config:<8} | ' \
                   f'{(fmt % avg):>12} | {(fmt % mn):>12} | {(fmt % mx):>12} | {len(vals)} |'
            print(line)
        else:
            print(f'{label:<22} | {config:<8} | (no data)')
    print()

# v1.3 vs vanilla 비교 (avg 기준)
print('=' * 92)
print('v1.3 vs vanilla (avg 기준, % 차이):')
print('-' * 92)
for key, label, fmt, polarity in metrics:
    vv = [r.get(key) for r in all_data['vanilla'] if r.get(key) is not None]
    nn = [r.get(key) for r in all_data['v13'] if r.get(key) is not None]
    if vv and nn:
        v_avg = statistics.mean(vv)
        n_avg = statistics.mean(nn)
        if v_avg != 0:
            pct = (n_avg - v_avg) / v_avg * 100
            sign = '+' if pct >= 0 else ''
            arrow = '↑' if pct >= 0 else '↓'
            note = ''
            if polarity == 'higher_better':
                note = '(better)' if pct > 0 else '(worse)'
            else:
                note = '(better)' if pct < 0 else '(worse)'
            print(f'{label:<22} : {sign}{pct:>7.2f}% {arrow}  {note}')

print('=' * 92)
print(f'suite dir: {suite}')
EOF

log "===== suite DONE ====="
echo "[perf-compare] suite dir: ${SUITE_DIR}"
echo "[perf-compare] summary log: ${SUITE_LOG}"
