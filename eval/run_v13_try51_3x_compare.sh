#!/usr/bin/env bash
# v1.3 (= try51 stack, commit f2678c2f4) 3 회 + vanilla 3 회 (이전 측정 재사용) 비교.
# 본 script 는 *detached HEAD f2678c2f4* 시점에서 실행.
# 결과는 /workspace/vllm_hybrid/eval/results/ 에 저장 (untracked, checkout 무관).
set -uo pipefail

ORIG_BRANCH="feat/ide006-tsk019-neo-performance-max"
TRY51_COMMIT="f2678c2f4"
ROOT_DIR="/workspace/vllm_hybrid"
cd "$ROOT_DIR"

PY=/workspace/vllm_dev_prj/bin/python

TS_BASE="$(date +%Y%m%d_%H%M%S)"
SUITE_DIR="${ROOT_DIR}/eval/results/${TS_BASE}_v13_try51_3x_compare"
mkdir -p "${SUITE_DIR}"
SUITE_LOG="${SUITE_DIR}/suite.log"

# Vanilla 3 회 결과 source (이전 측정)
VANILLA_SUITE="${ROOT_DIR}/eval/results/20260510_081620_perf_compare_v13_vs_vanilla"

log() { echo "[$(TZ=Asia/Seoul date -Iseconds)] $1" | tee -a "${SUITE_LOG}"; }

log "===== suite start — v1.3 (=try51) 3 회 측정 ====="
log "v1.3 reference commit: ${TRY51_COMMIT}"
log "vanilla 3 회 source (재사용): ${VANILLA_SUITE}"
log "suite dir: ${SUITE_DIR}"

# === detached checkout v1.3 stack ===
log "[checkout] git checkout ${TRY51_COMMIT}"
git checkout "${TRY51_COMMIT}" 2>&1 | tail -3 | tee -a "${SUITE_LOG}"
log "[checkout] HEAD now at: $(git rev-parse --short HEAD)"

# === try51 default env (Phase A+B) ===
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1

unset VLLM_NEO_DISABLE_CHAIN VLLM_NEO_DISABLE_FORCE_PIPELINED
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_LRU_FALLBACK_FIFO

# v4 D6~D12 env 도 제거 (try51 시점에 없음 — 영향 X 지만 명시)
unset VLLM_NEO_FORCE_SWAP_IN VLLM_NEO_MAX_SWAP_IN_PER_STEP
unset VLLM_NEO_CPU_RESIDENT_REQS VLLM_NEO_SWAP_IN_ORDER
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_D12_TOKEN_MARGIN
unset VLLM_NEO_PREDICTIVE_THRESHOLD VLLM_NEO_SWAP_COOLDOWN

export VLLM_NEO_FORCE_PIPELINED=1

run_one() {
    local idx="$1"
    local sub_dir="${SUITE_DIR}/v13_run${idx}"
    mkdir -p "${sub_dir}"
    local log_file="${sub_dir}/engine.log"

    log "[v13 run $idx/3] starting"
    "$PY" -u "${ROOT_DIR}/eval/run_neo_baseline.py" \
        --model llama-70b \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 16384 \
        --max-num-seqs 256 \
        --num-prompts 500 \
        --target-input-len 8192 \
        --max-tokens 8192 \
        --enable-neo-asymmetric \
        --async-scheduling \
        --enforce-eager false \
        --kv-cache-dtype fp8 \
        --max-num-batched-tokens 8192 \
        --log-file "${log_file}" \
        --output-file "${sub_dir}/result.json" \
        > "${log_file}.stdout" 2>&1
    local rc=$?
    log "[v13 run $idx/3] exit=${rc} result=$(test -f ${sub_dir}/result.json && echo OK || echo FAIL)"
    sleep 5
}

for i in 1 2 3; do
    run_one "${i}"
done

log "===== v13 3 회 완료 — branch 복귀 + 집계 ====="

# === branch 복귀 ===
git checkout "${ORIG_BRANCH}" 2>&1 | tail -2 | tee -a "${SUITE_LOG}"
log "[checkout] HEAD restored to: $(git rev-parse --short HEAD) ($(git branch --show-current))"

# === 집계 (vanilla 재사용 + v13 신규) ===
"$PY" - <<EOF | tee -a "${SUITE_LOG}"
import json, glob, os, statistics

vanilla_suite = '${VANILLA_SUITE}'
v13_suite = '${SUITE_DIR}'

def load_runs(suite, prefix):
    paths = sorted(glob.glob(os.path.join(suite, f'{prefix}_run*', 'result.json')))
    rows = []
    for p in paths:
        try:
            with open(p) as f:
                rows.append(json.load(f))
        except Exception:
            pass
    return rows

vanilla_runs = load_runs(vanilla_suite, 'vanilla')
v13_runs = load_runs(v13_suite, 'v13')

print(f'\nvanilla runs: {len(vanilla_runs)}  (source: {vanilla_suite})')
print(f'v13 runs    : {len(v13_runs)}  (source: {v13_suite})')

metrics = [
    ('output_tps', 'output tokens/s', '%.2f', 'higher_better'),
    ('prompt_tps', 'prompt tokens/s', '%.2f', 'higher_better'),
    ('generate_wall_s', 'generate wall (s)', '%.2f', 'lower_better'),
    ('req_per_s', 'req/s', '%.4f', 'higher_better'),
    ('init_s', 'init (s)', '%.2f', 'lower_better'),
    ('total_output_tokens', 'total out tokens', '%d', 'higher_better'),
]

all_data = {'vanilla': vanilla_runs, 'v1.3': v13_runs}

print('\n' + '=' * 100)
print(f'{"metric":<22} | {"config":<8} | {"avg":>13} | {"min":>13} | {"max":>13} | n |')
print('-' * 100)
for key, label, fmt, _ in metrics:
    for config in ['vanilla', 'v1.3']:
        rows = all_data[config]
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if vals:
            avg = statistics.mean(vals)
            mn, mx = min(vals), max(vals)
            print(f'{label:<22} | {config:<8} | {(fmt % avg):>13} | {(fmt % mn):>13} | {(fmt % mx):>13} | {len(vals)} |')
        else:
            print(f'{label:<22} | {config:<8} | (no data)')
    print()

print('=' * 100)
print('v1.3 vs vanilla (avg 기준, % 차이):')
print('-' * 100)
for key, label, fmt, polarity in metrics:
    vv = [r.get(key) for r in all_data['vanilla'] if r.get(key) is not None]
    nn = [r.get(key) for r in all_data['v1.3'] if r.get(key) is not None]
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

print('=' * 100)
print(f'v1.3 suite dir   : {v13_suite}')
print(f'vanilla suite dir: {vanilla_suite}')
EOF

log "===== suite DONE ====="
