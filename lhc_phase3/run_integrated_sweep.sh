#!/usr/bin/env bash
# LHC Phase 3 — Task G: integrated sweep (baseline vs LHC components).
#
# Configs:
#   vanilla       — no LHC
#   lhc_dsa       — enable_neo_asymmetric + DSA WQ-per-rank
#   lhc_amx_c3    — AMX C3 prefix scan path (env VLLM_LHC_AMX_C3=1)
#   lhc_full      — DSA + AMX C3
#   lhc_full_suf  — full + suffix spec decode
#   lhc_full_fp8  — full + fp8 kv-cache
#   lhc_full_suf_fp8 — full + suffix + fp8 kv
#
# Workloads (TSK_042 6 + W-D1/W-D3 KV-heavy → 8 selected by time budget):
#   sonnet/chat/code/balanced/sonnet-heavy/code-heavy + W-D1 + W-D3
#
# Runs 3 sweeps per (cfg, workload). Time budget: ~3 hours.

set -uo pipefail
BASE=/workspace/host_vllm_hybrid/lhc_phase3
mkdir -p "${BASE}/runs_G"
RESULTS="${BASE}/runs_G/results.csv"
echo "config,workload,sweep_idx,tput_tok_s,p50_e2e_ms,p99_e2e_ms,total_input_tok,total_output_tok,neo_swap_out_count,neo_swap_in_count,dsa_workers_enabled" > "${RESULTS}"

export HF_HUB_OFFLINE=1
MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8500

ts() { TZ=Asia/Seoul date '+%H:%M:%S KST'; }

start_serve() {
    local cfg=$1 boot_log=$2
    local neo_flag=""
    local dsa_env="VLLM_LHC_DSA=0"
    local amx_env="VLLM_LHC_AMX_C3=0"
    local extra=""
    local kvdt="auto"
    case "${cfg}" in
        vanilla) ;;
        lhc_dsa)
            neo_flag="--enable-neo-asymmetric"
            dsa_env="VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=4096"
            ;;
        lhc_amx_c3)
            amx_env="VLLM_LHC_AMX_C3=1"
            ;;
        lhc_full)
            neo_flag="--enable-neo-asymmetric"
            dsa_env="VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=4096"
            amx_env="VLLM_LHC_AMX_C3=1"
            ;;
        lhc_full_suf)
            neo_flag="--enable-neo-asymmetric --speculative-config {\"method\":\"suffix\",\"num_speculative_tokens\":8}"
            dsa_env="VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=4096"
            amx_env="VLLM_LHC_AMX_C3=1"
            ;;
        lhc_full_fp8)
            neo_flag="--enable-neo-asymmetric"
            dsa_env="VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=4096"
            amx_env="VLLM_LHC_AMX_C3=1"
            kvdt="fp8"
            ;;
        lhc_full_suf_fp8)
            neo_flag="--enable-neo-asymmetric --speculative-config {\"method\":\"suffix\",\"num_speculative_tokens\":8}"
            dsa_env="VLLM_LHC_DSA=1 VLLM_LHC_DSA_WQ_PER_RANK=1 VLLM_LHC_DSA_MIN=4096"
            amx_env="VLLM_LHC_AMX_C3=1"
            kvdt="fp8"
            ;;
    esac

    pgrep -f "vllm serve" 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 3

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        eval ${dsa_env} ${amx_env} \
        nohup /workspace/vllm_dev_prj/bin/vllm serve "${MODEL}" \
            --port ${PORT} --host 127.0.0.1 \
            --tensor-parallel-size 8 \
            --gpu-memory-utilization 0.70 \
            --max-model-len 32768 \
            --max-num-seqs 256 \
            --enable-prefix-caching \
            --kv-cache-dtype ${kvdt} \
            ${neo_flag} \
            > "${boot_log}" 2>&1 &
    echo $!
}

wait_ready() {
    local pid=$1 boot_log=$2
    for i in $(seq 1 60); do
        sleep 10
        if curl -sf -m 3 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 ${pid} 2>/dev/null; then
            echo "[$(ts)] serve died" >> "${boot_log}"
            return 1
        fi
    done
    return 1
}

kill_serve() {
    local pid=$1
    local pgid=$(ps -o pgid= -p ${pid} 2>/dev/null | tr -d ' ' || echo "")
    [ -n "${pgid}" ] && kill -9 -${pgid} 2>/dev/null
    sleep 3
    pgrep -f "VLLM::" 2>/dev/null | xargs -r kill -9 2>/dev/null
    pgrep -f "vllm serve" 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 5
}

bench_workload() {
    local workload=$1 cfg=$2 sweep_idx=$3
    local bench_dir="${BASE}/runs_G/${cfg}/${workload}"
    mkdir -p "${bench_dir}"
    local bench_log="${bench_dir}/s${sweep_idx}.log"
    local result_json="s${sweep_idx}.json"

    # workload params
    local sonnet_input=550 sonnet_output=150 sonnet_prefix=200 nprompts=200 conc=32
    case "${workload}" in
        sonnet)        sonnet_input=550;  sonnet_output=150; sonnet_prefix=200; nprompts=200; conc=32 ;;
        chat)          sonnet_input=300;  sonnet_output=300; sonnet_prefix=100; nprompts=200; conc=32 ;;
        code)          sonnet_input=800;  sonnet_output=400; sonnet_prefix=200; nprompts=200; conc=32 ;;
        balanced)      sonnet_input=500;  sonnet_output=250; sonnet_prefix=150; nprompts=200; conc=32 ;;
        sonnet-heavy)  sonnet_input=1200; sonnet_output=150; sonnet_prefix=400; nprompts=150; conc=32 ;;
        code-heavy)    sonnet_input=1000; sonnet_output=600; sonnet_prefix=200; nprompts=120; conc=24 ;;
        wd1)           sonnet_input=24000;sonnet_output=4096;sonnet_prefix=200; nprompts=16;  conc=8 ;;
        wd3)           sonnet_input=12000;sonnet_output=512; sonnet_prefix=8000;nprompts=64;  conc=32 ;;
    esac

    /workspace/vllm_dev_prj/bin/python -m vllm.entrypoints.cli.main bench serve \
        --backend vllm --model "${MODEL}" \
        --host 127.0.0.1 --port ${PORT} \
        --dataset-name sonnet --dataset-path /workspace/host_vllm_hybrid/benchmarks/sonnet.txt \
        --sonnet-input-len ${sonnet_input} \
        --sonnet-output-len ${sonnet_output} \
        --sonnet-prefix-len ${sonnet_prefix} \
        --num-prompts ${nprompts} \
        --max-concurrency ${conc} \
        --save-result --result-dir "${bench_dir}/" \
        --result-filename "${result_json}" \
        --seed $((40 + sweep_idx)) \
        > "${bench_log}" 2>&1

    # parse result
    local tput=$(python3 -c "import json; d=json.load(open('${bench_dir}/${result_json}')); print(d.get('output_throughput', d.get('total_token_throughput', 0)))" 2>/dev/null || echo 0)
    local p50=$(python3 -c "import json; d=json.load(open('${bench_dir}/${result_json}')); print(d.get('median_e2el_ms', 0))" 2>/dev/null || echo 0)
    local p99=$(python3 -c "import json; d=json.load(open('${bench_dir}/${result_json}')); print(d.get('p99_e2el_ms', 0))" 2>/dev/null || echo 0)
    local tin=$(python3 -c "import json; d=json.load(open('${bench_dir}/${result_json}')); print(d.get('total_input_tokens', 0))" 2>/dev/null || echo 0)
    local tout=$(python3 -c "import json; d=json.load(open('${bench_dir}/${result_json}')); print(d.get('total_output_tokens', 0))" 2>/dev/null || echo 0)
    echo "${tput},${p50},${p99},${tin},${tout}"
}

# Time-budget selection: 7 configs × 8 workloads × 3 sweeps × ~30s ≈ 28 min nominal,
# but boot is the bottleneck (~3 min × 7 = 21 min). Total budget ~3 hr.

CONFIGS=(vanilla lhc_dsa lhc_amx_c3 lhc_full lhc_full_suf lhc_full_fp8 lhc_full_suf_fp8)
WORKLOADS=(sonnet chat code balanced sonnet-heavy code-heavy wd1 wd3)

for cfg in "${CONFIGS[@]}"; do
    boot_log="${BASE}/runs_G/${cfg}/boot.log"
    mkdir -p "${BASE}/runs_G/${cfg}"
    echo "[$(ts)] === config=${cfg} === booting" | tee -a "${boot_log}"
    spid=$(start_serve "${cfg}" "${boot_log}")
    if ! wait_ready ${spid} "${boot_log}"; then
        echo "[$(ts)] cfg=${cfg} boot FAILED — skip" | tee -a "${boot_log}"
        kill_serve ${spid}
        continue
    fi
    echo "[$(ts)] cfg=${cfg} ready" | tee -a "${boot_log}"

    for w in "${WORKLOADS[@]}"; do
        for s in 1 2 3; do
            echo "[$(ts)] cfg=${cfg} w=${w} s=${s}" | tee -a "${boot_log}"
            metrics=$(bench_workload "${w}" "${cfg}" "${s}")
            neo_so=$(grep -c "\[NEO\] swap-out:" "${boot_log}" 2>/dev/null || echo 0)
            neo_si=$(grep -c "\[NEO\] swap-in:" "${boot_log}" 2>/dev/null || echo 0)
            dsa_w=$(grep -c "\[LHC DSA\] lane ENABLED" "${boot_log}" 2>/dev/null || echo 0)
            echo "${cfg},${w},${s},${metrics},${neo_so},${neo_si},${dsa_w}" >> "${RESULTS}"
        done
    done

    kill_serve ${spid}
done

echo "[$(ts)] sweep done — results in ${RESULTS}" | tee -a "${BASE}/runs_G/done.flag"
