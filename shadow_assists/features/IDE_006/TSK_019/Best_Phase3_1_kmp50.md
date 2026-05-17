# Best — Phase 3.1 + KMP_BLOCKTIME=50 (2,038.7 tps, 2026-05-15)

> Phase 3.1 (Persistent OMP team) merged + env KMP_BLOCKTIME=50.
> base (Phase 3.4, 2,157 tps) 영역의 직접 비교 X — Phase 3.1 merged HEAD 위 영역.
> 본 file = Phase 3 측정 chain 의 best.

---

## fact (KST 2026-05-15 20:31)

### Identity

| 항목 | 값 |
|---|---|
| **base commit** | `099d23e54` (Merge Phase 3.1 → feat/neo-amx-apply) |
| Phase 3.1 commit | `63695e224` (csrc/cpu/pacpu/core.h OMP persistent) |
| 측정 dir | `eval/results/20260515_203107_phase3_1_kmp50/` |
| 측정 시각 (KST) | 2026-05-15 20:31:07 → 21:14 |
| launch script | `eval/run_neo_phase3_1_kmp50.sh` |

### 측정 결과

| 지표 | 값 |
|---|---|
| **output_tps** | **2,038.7** |
| wall | 1,591s (26.5 분) |
| prompt_tps | (자료) |
| **400p 완주** | **✅** |
| crash (Timeout/Engine/segv) | 0 / 0 / 0 |
| **shape_mismatch** | **0** ★ |
| **22 strict** | **19/19** ✅ |
| cdec_wait_avg | **2.38 ms** (v1.6 best 2.68ms 영역 −11.2%) |
| ratio (cdec/gpu) | 18.16× |
| b1_avg | 5 |

### Workload

| | |
|---|---|
| model | meta-llama/Llama-3.3-70B-Instruct |
| tensor_parallel_size | 8 (H100 80GB × 8) |
| max_model_len | 16,384 |
| max_num_seqs | 256 |
| max_num_batched_tokens | 8,192 |
| **num_prompts** | **400** (v1.6 best 500 영역의 80%) |
| target_input_len | 8,192 |
| max_tokens | 8,192 |
| kv_cache_dtype | fp8 |
| gpu_memory_utilization | 0.92 |

### env (eval/run_neo_phase3_1_kmp50.sh 적재)

```bash
# Hardware tuning + Phase 3.2 KMP 변경
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
unset OMP_PLACES
export KMP_BLOCKTIME=50    # ★ Phase 3.1+KMP=50 — 50ms spin (default 200 → 50)

# NEO 표준 (Phase 3.4 baseline 동일)
export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_CPU_RESIDENT_REQS=128
export VLLM_NEO_ASYNC_SWAP_BUFFERS=3
export VLLM_NEO_PROFILE=1

# CPU pin + NUMA
export VLLM_NEO_CPU_PIN_PER_WORKER=1
export VLLM_NEO_CPU_PIN_CORES=12
export VLLM_NEO_NUMA_BIND=1

# 실험적 NEO env 영역 unset (run_neo_standard.sh 와 동일)
unset VLLM_NEO_OPTION_A VLLM_NEO_OPTION_C VLLM_NEO_OPTION_K
unset VLLM_NEO_OPTION_L VLLM_NEO_OPTION_M2 VLLM_NEO_OPTION_O2
...
```

### 코드 변경 (Phase 3.1)

- `csrc/cpu/pacpu/core.h:ispc_attention_tasks` 영역 — `omp_set_dynamic(0)` + `omp_set_max_active_levels(1)` (thread_local flag 1회 적용)
- pacpu rebuild 필요

### KMP sweep fact (chronology)

| KMP_BLOCKTIME | output_tps | wall (s) | cdec_wait | 결과 |
|---|---:|---:|---:|---|
| 0 (Phase 3.2) | crash | — | — | EngineDeadError (CPU saturation) |
| 10 | 2,008.4 | 1,619 | 3.56 ms | over-spin (b1_avg=10 큰 영역) |
| **50** | **2,038.7** | **1,591** | **2.38 ms** ✓ | **sweet spot** |
| default 200 (Phase 3.1) | 2,044 avg (±181, 3-run) | 1,602 | 2.93 ms | variance 큼 |

→ **KMP=50** = throughput 영역 안정 + cdec_wait 가장 짧음.

### 비교 (Phase 3 chain)

| measurement | output_tps | wall | cdec_wait | Δ vs Phase 3.4 |
|---|---:|---:|---:|---:|
| Phase 3.4 baseline (KMP default) | 1,930.5 | 1,688 | 2.68 ms | — |
| Phase 3.1 (KMP default, 3-run avg) | 2,044.0 (±181) | 1,602 | 2.93 ms | +5.88% |
| Phase 3.3 (CUDA Stream Priority) | 1,886.3 | 1,728 | 3.30 ms | −2.29% |
| Phase 3.1+3.3 combined | 1,977.2 | 1,641 | 2.98 ms | +2.42% |
| **Phase 3.1+KMP=50 (현 best)** | **2,038.7** | 1,591 | **2.38 ms** | **+5.61%** |

### 재현 절차

```bash
git checkout 099d23e54  # 또는 feat/neo-amx-apply HEAD
CXX=/tmp/gcc12/usr/bin/g++-12 bash csrc/cpu/pacpu/build.sh llama3_3_70b 8
bash eval/run_neo_phase3_1_kmp50.sh
```

### 한계

- 1회 측정 (variance 영역 확인 영역 미달, 3-run avg 영역 필요)
- 400p workload (v1.6 best 500p 영역 보다 영역 작음 — 직접 throughput 비교 영역 부정확)
- v1.6 best 2,157 tps 영역 보다 영역 절대값 영역 ↓ (그러나 500p × 8192 = 4M tokens 영역 vs 400p × 8192 = 3.2M tokens 영역 의 다른 workload)
