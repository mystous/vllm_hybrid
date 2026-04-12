#!/usr/bin/env bash
# =============================================================================
# cpu_profile.sh — Many-Core CPU 병목 종합 진단
#
# 용도: H100x4 (96 core Xeon 8480+) 에서 왜 76 thread 가 24 thread 보다
#       느린지 한 번에 전부 측정. 서버 비용이 비싸므로 재실행 없이 완결.
#
# 실행: bash cpu_profile.sh 2>&1 | tee /tmp/cpu_profile_$(date +%Y%m%d_%H%M%S).log
# 소요 시간: 약 10-15 분
# 사전 조건: vLLM 설치 완료, GPU 서버 안 돌고 있는 상태
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${SCRIPT_DIR}/analysis_log/${TIMESTAMP}_cpu_profile"
mkdir -p "$OUTDIR"
echo "=== CPU Profile Output: $OUTDIR ==="
echo "=== Start: $(date) ==="

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: 시스템 토폴로지 — SNC, NUMA, SMT, LLC 구조
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "SECTION 1: System Topology"
echo "================================================================"

echo "--- lscpu ---"
lscpu | tee "$OUTDIR/lscpu.txt"

echo ""
echo "--- NUMA nodes ---"
ls -d /sys/devices/system/node/node* 2>/dev/null | tee "$OUTDIR/numa_nodes.txt"
for node in /sys/devices/system/node/node*/cpulist; do
    echo "$(dirname $node | xargs basename): $(cat $node)"
done | tee "$OUTDIR/numa_cpulist.txt"

echo ""
echo "--- LLC (L3 cache) topology ---"
# 각 core 가 어느 LLC 를 공유하는지
if [ -d /sys/devices/system/cpu/cpu0/cache ]; then
    for cpu in /sys/devices/system/cpu/cpu[0-9]*/cache/index3; do
        cpuid=$(echo $cpu | grep -oP 'cpu\K[0-9]+')
        shared=$(cat "$cpu/shared_cpu_list" 2>/dev/null || echo "N/A")
        size=$(cat "$cpu/size" 2>/dev/null || echo "N/A")
        echo "cpu$cpuid: L3 shared_with=[$shared] size=$size"
    done | sort -t'u' -k2 -n | head -20 | tee "$OUTDIR/llc_topology.txt"
    echo "(showing first 20 cores)"
else
    echo "No cache topology info available"
fi

echo ""
echo "--- Physical core mapping (detect SMT) ---"
python3 -c "
import os, json
topo = {}
for cpu_dir in sorted(os.listdir('/sys/devices/system/cpu/')):
    if not cpu_dir.startswith('cpu') or not cpu_dir[3:].isdigit():
        continue
    cpu_id = int(cpu_dir[3:])
    try:
        with open(f'/sys/devices/system/cpu/{cpu_dir}/topology/core_id') as f:
            core_id = int(f.read().strip())
        with open(f'/sys/devices/system/cpu/{cpu_dir}/topology/physical_package_id') as f:
            pkg_id = int(f.read().strip())
        topo[cpu_id] = {'core_id': core_id, 'package': pkg_id}
    except:
        pass

# Detect SMT: multiple cpu_ids per core_id
from collections import defaultdict
core_map = defaultdict(list)
for cpu_id, info in topo.items():
    core_map[(info['package'], info['core_id'])].append(cpu_id)

smt_pairs = {k: v for k, v in core_map.items() if len(v) > 1}
print(f'Total logical CPUs: {len(topo)}')
print(f'Physical cores: {len(core_map)}')
print(f'SMT pairs (>1 logical per physical): {len(smt_pairs)}')
if smt_pairs:
    print('First 5 SMT pairs:')
    for k, v in list(smt_pairs.items())[:5]:
        print(f'  pkg{k[0]}.core{k[1]}: logical CPUs {v}')
else:
    print('No SMT detected (1 thread per core)')

with open('$OUTDIR/core_topology.json', 'w') as f:
    json.dump({'cores': dict(core_map), 'smt_pairs': len(smt_pairs),
               'logical_cpus': len(topo), 'physical_cores': len(core_map)}, f, indent=2, default=str)
" | tee "$OUTDIR/smt_detection.txt"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: GEMM Scaling — thread 수 별 matmul 성능
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "SECTION 2: GEMM Scaling (thread count sweep)"
echo "================================================================"

python3 << 'PYGEMM' | tee "$OUTDIR/gemm_scaling.txt"
import torch, time, json, os

results = []

# 7B Qwen2.5 의 주요 GEMM 크기들
gemm_configs = [
    ("decode_qkv",   16, 3584, 3584),    # QKV proj, batch=16
    ("decode_ffn_up", 16, 3584, 9728),   # Gate+Up proj, batch=16
    ("decode_ffn_dn", 16, 9728, 3584),   # Down proj, batch=16
    ("decode_single", 1, 3584, 9728),    # Single token (batch=1)
    ("prefill_128",  128, 3584, 9728),   # Prefill 128 tokens
]

thread_counts = [4, 8, 12, 16, 24, 32, 48, 64, 76, 96]

for name, M, K, N in gemm_configs:
    print(f"\n--- GEMM: {name} [{M}x{K}] × [{K}x{N}] ---")
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)

    for nthreads in thread_counts:
        try:
            torch.set_num_threads(nthreads)
            # warmup
            for _ in range(20):
                c = a @ b
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            iters = 500 if M <= 16 else 100
            t0 = time.perf_counter()
            for _ in range(iters):
                c = a @ b
            t1 = time.perf_counter()
            ms = (t1 - t0) / iters * 1000
            gflops = 2 * M * K * N / ms / 1e6
            print(f"  threads={nthreads:3d}: {ms:.3f} ms  ({gflops:.1f} GFLOPS)")
            results.append({"name": name, "M": M, "K": K, "N": N,
                           "threads": nthreads, "ms": ms, "gflops": gflops})
        except Exception as e:
            print(f"  threads={nthreads}: ERROR {e}")

with open(os.path.join("$OUTDIR", "gemm_scaling.json"), "w") as f:
    json.dump(results, f, indent=2)
PYGEMM


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Attention Kernel Scaling — IPEX vs 순수 SDPA
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "SECTION 3: Attention Kernel Scaling"
echo "================================================================"

python3 << 'PYATTN' | tee "$OUTDIR/attention_scaling.txt"
import torch, time, json, os

results = []
# 7B Qwen2.5: num_heads=28, num_kv_heads=4, head_dim=128
num_kv_heads = 4
head_dim = 128
seq_len = 256  # typical decode position
batch_sizes = [1, 4, 8, 16]
thread_counts = [4, 8, 16, 24, 48, 76, 96]

print("=== Attention: batched Q×K^T + score×V (simulated) ===")
print("(Pure torch SDPA, not IPEX — measures compute+BW scaling)")

for batch in batch_sizes:
    for nthreads in thread_counts:
        try:
            torch.set_num_threads(nthreads)
            # Simulate multi-head attention for batch sequences
            q = torch.randn(batch, num_kv_heads, 1, head_dim, dtype=torch.bfloat16)
            k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)
            v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)

            # warmup
            for _ in range(20):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

            iters = 200
            t0 = time.perf_counter()
            for _ in range(iters):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            t1 = time.perf_counter()
            ms = (t1 - t0) / iters * 1000

            print(f"  batch={batch:2d} threads={nthreads:3d}: {ms:.3f} ms")
            results.append({"batch": batch, "threads": nthreads, "ms": ms,
                           "seq_len": seq_len, "kv_heads": num_kv_heads})
        except Exception as e:
            print(f"  batch={batch} threads={nthreads}: ERROR {e}")

with open(os.path.join("$OUTDIR", "attention_scaling.json"), "w") as f:
    json.dump(results, f, indent=2)
PYATTN


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Memory Bandwidth — STREAM-like 측정
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "SECTION 4: Memory Bandwidth (STREAM-like)"
echo "================================================================"

python3 << 'PYSTREAM' | tee "$OUTDIR/memory_bw.txt"
import torch, time, json, os

results = []
sizes_mb = [64, 256, 1024]  # 64MB, 256MB, 1GB
thread_counts = [1, 4, 8, 16, 24, 48, 76, 96]

for size_mb in sizes_mb:
    n = size_mb * 1024 * 1024 // 4  # float32 elements
    a = torch.randn(n, dtype=torch.float32)
    b = torch.randn(n, dtype=torch.float32)

    print(f"\n--- STREAM Copy: {size_mb} MB ---")
    for nthreads in thread_counts:
        try:
            torch.set_num_threads(nthreads)
            # warmup
            for _ in range(3):
                c = a + b
            iters = 20
            t0 = time.perf_counter()
            for _ in range(iters):
                c = a + b
            t1 = time.perf_counter()
            bw_gb = 3 * size_mb / 1024 * iters / (t1 - t0)  # read a, read b, write c
            print(f"  threads={nthreads:3d}: {bw_gb:.1f} GB/s")
            results.append({"size_mb": size_mb, "threads": nthreads, "bw_gbs": bw_gb})
        except Exception as e:
            print(f"  threads={nthreads}: ERROR {e}")

with open(os.path.join("$OUTDIR", "memory_bw.json"), "w") as f:
    json.dump(results, f, indent=2)
PYSTREAM


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: 실제 모델 per-layer 시간 분해
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "SECTION 5: Per-Layer Time Breakdown (actual 7B model)"
echo "================================================================"
echo "(이 섹션은 모델 로드 필요, 3-5분 소요)"

python3 << 'PYLAYER' | tee "$OUTDIR/layer_breakdown.txt"
import torch, time, json, os, sys

# Check if model is available
model_name = "Qwen/Qwen2.5-7B-Instruct"
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("transformers not available, skipping layer breakdown")
    sys.exit(0)

thread_counts_to_test = [24, 76]
results = []

for nthreads in thread_counts_to_test:
    print(f"\n=== Layer Breakdown: {nthreads} threads ===")
    torch.set_num_threads(nthreads)

    try:
        print(f"Loading model with {nthreads} threads...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map="cpu"
        )
        model.eval()

        # Prepare input (simulate decode: batch=16, seq_len=1, past_kv exists)
        input_ids = torch.randint(0, 32000, (1, 128))

        # Warmup
        print("Warmup...")
        with torch.no_grad():
            _ = model(input_ids)

        # Hook into each layer to measure time
        layer_times = {}

        def make_hook(name):
            def hook(module, input, output):
                if name not in layer_times:
                    layer_times[name] = []
                layer_times[name].append(time.perf_counter())
            return hook

        hooks = []
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.self_attn.register_forward_hook(make_hook(f"layer{i}_attn")))
            hooks.append(layer.mlp.register_forward_hook(make_hook(f"layer{i}_mlp")))

        # Also hook the start of each layer
        pre_hooks = []
        def make_pre_hook(name):
            def hook(module, input):
                if name not in layer_times:
                    layer_times[name] = []
                layer_times[name].append(time.perf_counter())
            return hook

        for i, layer in enumerate(model.model.layers):
            pre_hooks.append(layer.self_attn.register_forward_pre_hook(make_pre_hook(f"layer{i}_attn_pre")))
            pre_hooks.append(layer.mlp.register_forward_pre_hook(make_pre_hook(f"layer{i}_mlp_pre")))

        # Run 5 iterations
        print("Measuring...")
        all_runs = []
        for run in range(5):
            layer_times = {}
            with torch.no_grad():
                t_start = time.perf_counter()
                _ = model(input_ids)
                t_end = time.perf_counter()

            total_ms = (t_end - t_start) * 1000

            # Compute per-layer attention and MLP times
            attn_total = 0
            mlp_total = 0
            num_layers = len(model.model.layers)

            for i in range(num_layers):
                attn_pre_key = f"layer{i}_attn_pre"
                attn_post_key = f"layer{i}_attn"
                mlp_pre_key = f"layer{i}_mlp_pre"
                mlp_post_key = f"layer{i}_mlp"

                if all(k in layer_times for k in [attn_pre_key, attn_post_key, mlp_pre_key, mlp_post_key]):
                    attn_ms = (layer_times[attn_post_key][0] - layer_times[attn_pre_key][0]) * 1000
                    mlp_ms = (layer_times[mlp_post_key][0] - layer_times[mlp_pre_key][0]) * 1000
                    attn_total += attn_ms
                    mlp_total += mlp_ms

            other_ms = total_ms - attn_total - mlp_total
            all_runs.append({"total": total_ms, "attn": attn_total, "mlp": mlp_total, "other": other_ms})

        # Remove hooks
        for h in hooks + pre_hooks:
            h.remove()

        # Average over runs (skip first)
        avg = {k: sum(r[k] for r in all_runs[1:]) / len(all_runs[1:]) for k in ["total", "attn", "mlp", "other"]}

        print(f"  Total:     {avg['total']:.1f} ms")
        print(f"  Attention: {avg['attn']:.1f} ms ({avg['attn']/avg['total']*100:.1f}%)")
        print(f"  MLP (FFN): {avg['mlp']:.1f} ms ({avg['mlp']/avg['total']*100:.1f}%)")
        print(f"  Other:     {avg['other']:.1f} ms ({avg['other']/avg['total']*100:.1f}%)")

        results.append({"threads": nthreads, **avg})

        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc; gc.collect()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

if results:
    with open(os.path.join("$OUTDIR", "layer_breakdown.json"), "w") as f:
        json.dump(results, f, indent=2)

    if len(results) >= 2:
        r24 = results[0]
        r76 = results[1]
        print(f"\n=== Comparison: 24 vs 76 threads ===")
        print(f"  Total:  {r24['total']:.1f} → {r76['total']:.1f} ms ({r76['total']/r24['total']:.2f}×)")
        print(f"  Attn:   {r24['attn']:.1f} → {r76['attn']:.1f} ms ({r76['attn']/r24['attn']:.2f}×)")
        print(f"  MLP:    {r24['mlp']:.1f} → {r76['mlp']:.1f} ms ({r76['mlp']/r24['mlp']:.2f}×)")
        print(f"  Other:  {r24['other']:.1f} → {r76['other']:.1f} ms")
        print(f"\n  → Bottleneck: {'ATTENTION' if r76['attn']/r76['total'] > 0.5 else 'MLP/FFN' if r76['mlp']/r76['total'] > 0.5 else 'OTHER/DISPATCH'}")
PYLAYER


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: 실제 vLLM CPU 추론 thread 수 sweep (1 request)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "SECTION 6: vLLM Single-Request Thread Sweep"
echo "================================================================"
echo "(vLLM import 필요, 1 req 씩 다른 thread 수로 측정)"

python3 << 'PYVLLM' | tee "$OUTDIR/vllm_thread_sweep.txt"
import os, sys, json, time

# Only run if vLLM is available
try:
    # Quick test: can we import vllm?
    import vllm
except ImportError:
    print("vLLM not importable, skipping")
    sys.exit(0)

# Use offline/simple completion to avoid server overhead
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM LLM class not available, skipping")
    sys.exit(0)

results = []
thread_counts = [8, 16, 24, 32, 48, 76]
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B for speed (7B too slow for sweep)

for nthreads in thread_counts:
    print(f"\n--- vLLM CPU decode: {nthreads} threads, 1.5B, 1 request ---")
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)

    try:
        import torch
        torch.set_num_threads(nthreads)

        llm = LLM(model=model_name, device="cpu", dtype="bfloat16",
                   max_model_len=512, enforce_eager=True,
                   tensor_parallel_size=1)

        params = SamplingParams(max_tokens=64, temperature=0.0)
        prompt = "The meaning of life is"

        # Warmup
        _ = llm.generate([prompt], params)

        # Measure
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], params)
        t1 = time.perf_counter()

        n_tokens = len(outputs[0].outputs[0].token_ids)
        elapsed = t1 - t0
        tps = n_tokens / elapsed

        print(f"  tokens={n_tokens} elapsed={elapsed:.2f}s tps={tps:.2f}")
        results.append({"threads": nthreads, "tokens": n_tokens,
                        "elapsed_s": elapsed, "tps": tps})

        del llm
        import gc; gc.collect()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

if results:
    with open(os.path.join("$OUTDIR", "vllm_thread_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)
PYVLLM


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Intel 진단 — AMX 상태, oneDNN 설정, IPEX 버전
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "SECTION 7: Intel Environment Diagnostics"
echo "================================================================"

python3 << 'PYINTEL' | tee "$OUTDIR/intel_diag.txt"
import os, sys

print("=== PyTorch ===")
import torch
print(f"torch: {torch.__version__}")
print(f"mkldnn available: {torch.backends.mkldnn.is_available()}")
print(f"num_threads: {torch.get_num_threads()}")
print(f"num_interop_threads: {torch.get_num_interop_threads()}")

print("\n=== IPEX ===")
try:
    import intel_extension_for_pytorch as ipex
    print(f"ipex: {ipex.__version__}")
except ImportError:
    print("IPEX not installed")

print("\n=== oneDNN ===")
print(f"ONEDNN_MAX_CPU_ISA: {os.environ.get('ONEDNN_MAX_CPU_ISA', 'not set')}")
print(f"MKL_ENABLE_INSTRUCTIONS: {os.environ.get('MKL_ENABLE_INSTRUCTIONS', 'not set')}")
print(f"KMP_AFFINITY: {os.environ.get('KMP_AFFINITY', 'not set')}")
print(f"KMP_BLOCKTIME: {os.environ.get('KMP_BLOCKTIME', 'not set')}")

print("\n=== CPU Features ===")
try:
    from vllm.platforms.intel_cpu_utils import detect_intel_cpu_features
    f = detect_intel_cpu_features()
    print(f"Model: {f.model_name}")
    print(f"Sockets: {f.num_sockets}, Cores/Socket: {f.cores_per_socket}, Threads/Core: {f.threads_per_core}")
    print(f"AVX-512F: {f.avx512f}, VNNI: {f.avx512_vnni}, BF16: {f.avx512_bf16}")
    print(f"AMX-BF16: {f.amx_bf16}, AMX-INT8: {f.amx_int8}")
except Exception as e:
    print(f"detect failed: {e}")

print("\n=== vLLM CPU ops ===")
try:
    import vllm._custom_ops as ops
    print(f"HAS_CPU_OPS: {ops.HAS_CPU_OPS}")
    print(f"HAS_CPU_UTILS: {ops.HAS_CPU_UTILS}")
except Exception as e:
    print(f"custom_ops: {e}")
PYINTEL


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: perf stat (hardware counters) — GEMM 24 vs 76 threads
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "SECTION 8: Hardware Counters (perf stat)"
echo "================================================================"

if command -v perf &>/dev/null; then
    echo "--- 24 threads ---"
    perf stat -e cache-references,cache-misses,LLC-loads,LLC-load-misses,instructions,cycles \
        python3 -c "
import torch; torch.set_num_threads(24)
a=torch.randn(16,3584,dtype=torch.bfloat16); b=torch.randn(3584,9728,dtype=torch.bfloat16)
for _ in range(500): c=a@b
" 2>&1 | tee "$OUTDIR/perf_stat_24.txt"

    echo ""
    echo "--- 76 threads ---"
    perf stat -e cache-references,cache-misses,LLC-loads,LLC-load-misses,instructions,cycles \
        python3 -c "
import torch; torch.set_num_threads(76)
a=torch.randn(16,3584,dtype=torch.bfloat16); b=torch.randn(3584,9728,dtype=torch.bfloat16)
for _ in range(500): c=a@b
" 2>&1 | tee "$OUTDIR/perf_stat_76.txt"
else
    echo "perf not available (try: apt install linux-tools-generic)"
fi


# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "=== Complete: $(date) ==="
echo "=== Results saved to: $OUTDIR ==="
echo "=== Log: tee output above ==="
echo "================================================================"
echo ""
echo "결과 요약 보려면:"
echo "  cat $OUTDIR/gemm_scaling.txt       # GEMM thread scaling"
echo "  cat $OUTDIR/attention_scaling.txt   # Attention thread scaling"
echo "  cat $OUTDIR/memory_bw.txt          # Memory bandwidth"
echo "  cat $OUTDIR/layer_breakdown.txt    # Attention vs MLP 비율"
echo "  cat $OUTDIR/smt_detection.txt      # SMT / 토폴로지"
echo ""
echo "git add & push 하려면:"
echo "  cd $(dirname $OUTDIR) && git add $(basename $OUTDIR)/"
