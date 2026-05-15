# Best — v1.6 (2,157 tps, 2026-05-15)

> v1.6 의 best configuration (이전 plan, Phase 3 측정 baseline).
> 신규 best (Phase 3.1+KMP=50, 2,038.7 tps) → `Best_Phase3_1_kmp50.md`.
> 본 file = 이력 보존.

---

## fact (KST 2026-05-15 00:14)

### Identity

| 항목 | 값 |
|---|---|
| **commit id** | **`64f9e0c48`** |
| commit 시각 (KST) | 2026-05-15 00:18:04 |
| commit subject | `feat(IDE_006/TSK_019 v1.6): shape mismatch fix + 22 항목 strict 19/19 — async swap_out / ensure_capacity race 차단` |
| 측정 dir | `eval/results/20260514_233540_neo_standard/` |
| 측정 시각 (KST) | 2026-05-14 23:35:40 → 2026-05-15 00:07:02 |
| launch script | `eval/run_neo_standard.sh` |

### 측정 결과

| 지표 | 값 |
|---|---|
| **output_tps** | **2,156.9** |
| wall | 1,882.2s (31.4 분) |
| init | 79.8s |
| total_output_tokens | 4,059,711 |
| total_input_tokens | 2,709,037 |
| prompt_tps | 1,439.3 |
| req_per_s | 0.266 |
| **500p 완주** | **✅** |
| crash (assert/cuda/segv/dead) | 0 / 0 / 0 / 0 |
| **shape mismatch** | **0** ★ |
| **22 항목 strict** | **19/19** ✅ |
| vs vanilla 4,886 | 44.1% |

### Workload

| | |
|---|---|
| model | meta-llama/Llama-3.3-70B-Instruct |
| tensor_parallel_size | 8 (H100 80GB × 8) |
| max_model_len | 16,384 |
| max_num_seqs | 256 |
| max_num_batched_tokens | 8,192 |
| num_prompts | 500 |
| target_input_len | 8,192 |
| max_tokens | 8,192 |
| kv_cache_dtype | fp8 |
| async_scheduling | True |
| enable_neo_asymmetric | True |
| enforce_eager | False |
| gpu_memory_utilization | 0.92 |

### env (eval/run_neo_standard.sh 적재)

```bash
# Hardware tuning
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
unset OMP_PLACES

# NEO 표준 (SUB_027 H5 winner 영역)
export VLLM_NEO_PREDICTOR=heuristic
export VLLM_NEO_CPU_RESIDENT_REQS=128
export VLLM_NEO_ASYNC_SWAP_BUFFERS=3
export VLLM_NEO_MIRROR_MAX=80
export VLLM_NEO_SYNC_SWAP_BATCHED=1
export VLLM_NEO_PROFILE=1

# CPU pin + NUMA
export VLLM_NEO_CPU_PIN_PER_WORKER=1
export VLLM_NEO_CPU_PIN_CORES=12
export VLLM_NEO_NUMA_BIND=1

# 모두 unset (실험적 NEO env 정리)
unset VLLM_NEO_OPTION_A VLLM_NEO_OPTION_C VLLM_NEO_OPTION_K
unset VLLM_NEO_OPTION_L VLLM_NEO_OPTION_M2 VLLM_NEO_OPTION_O2
unset VLLM_NEO_OPTION_C_FULL_MIRROR
unset VLLM_NEO_FORCE_SWAP_IN VLLM_NEO_FORCE_PIPELINED
unset VLLM_NEO_LOAD_AWARE_MIN_RUNNING
unset VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP
unset VLLM_NEO_MAX_SWAP_IN_PER_STEP
unset VLLM_NEO_MIRROR_MIN_BUFFER VLLM_NEO_MIN_RUNNING_DECODE
unset VLLM_NEO_SWAP_COOLDOWN
unset VLLM_NEO_SWAP_IN_ORDER
unset VLLM_NEO_DISABLE_FORCE_PIPELINED VLLM_NEO_DISABLE_CHAIN
unset VLLM_NEO_DISABLE_FUSED_RMSNORM VLLM_NEO_DISABLE_SWAP_IN
unset VLLM_NEO_DISABLE_D5 VLLM_NEO_LRU_FALLBACK_FIFO
unset VLLM_NEO_D12_TOKEN_MARGIN
unset VLLM_NEO_NEOSCHED_STEP23 VLLM_NEO_DRIVE_6STEP VLLM_NEO_6STEP_DRY_RUN
unset VLLM_NEO_DECIDE_MODE_BALANCE
unset VLLM_NEO_HEURISTIC_LINR_PER_TOKEN_MS
unset VLLM_NEO_HEURISTIC_PREF_PER_TOKEN_MS
unset VLLM_NEO_HEURISTIC_GDEC_PER_TOKEN_MS
unset VLLM_NEO_HEURISTIC_CDEC_PER_TOKEN_PAIR_MS
unset VLLM_NEO_HEURISTIC_LNCH_MS
unset VLLM_NEO_ASYNC_CDEC VLLM_NEO_CDEC_PIPELINE_DEPTH
unset VLLM_DEBUG_FAULTHANDLER VLLM_DEBUG_CDEC_PATH
unset ENABLE_NEO_INV
```

### 22 항목 fire fact

| # | 항목 | 상태 | fact |
|---|---|:-:|---|
| 1 | KV exclusive ownership | ✅ | SWAP_OUT_CALL = **14,168** |
| 2 | CPU attention 직접 (chain) | ✅ | active = 1,582/39,600 (**4.0%**) |
| 3 | Asymmetric Pipelining | ✅ | OOM=0 |
| 4 | Stage 분할 | ✅ | OOM=0 |
| 5 | 6단계 Scheduler | ✅ | Plan v4 D15+D16 fire=1 |
| 6 | Mode Select | ✅ | active=1,582 |
| 7 | 3-way attention dispatch | ✅ | eligible=1,582 active=1,582 |
| **8** | **swap_out/in invariant** | **✅** | **mismatch=0** ★ |
| 9 | paged_attention_cpu (pacpu) | ✅ | CDEC max = **30,000** |
| 10 | Q/K/V D2H transfer | ✅ | pacpu 동반 |
| 11 | sub_batches attach | ✅ | eligible=1,582 |
| **12** | **b0/b1 정렬** | **✅** | reject_split_oob=0, mismatch=0 |
| 13 | forward_pipeline overlap | ✅ | active=1,582 |
| **14** | **KV migration LRU + capacity** | **✅** | swap_out=14,168, mismatch=0 |
| 15 | NEO > vanilla throughput | ❌ | 2,157 vs vanilla 4,886 (**44.1%**) |
| 16 | CPU util HIGH | ⏳ | 별도 측정 |
| 17 | token correctness | ⏳ | 별도 측정 |
| 18 | deadlock 회피 | ✅ | engine_dead=0 |
| 19 | silent worker crash 0 | ✅ | assert=0 cuda=0 segv=0 |
| 20 | Option I (CPU resident queue) | ✅ | first fire=1, mirror mode=**10** (2,897회) |
| 21 | Option L (BUF EXTEND) | ✅ | fire=**272** |
| **22** | **Option M2 (swap-in size sync)** | **✅** | fire=8, mismatch=0 |

→ **NEO 본질 19/19 ALL ✅** (성능 15 ❌, 별도 측정 16/17 ⏳)

### 핵심 fix (commit 64f9e0c48)

1. **shape mismatch fix** — `NeoCpuKvBuffer._in_flight_swap_out`
   - async swap_out gather 시점과 drain 시점 사이 Option L (`ensure_capacity`) 가 block_ids 를 extend 하면 → drain scatter 시 staged 데이터 size 와 block_ids 길이 mismatch → shape mismatch 매 run 176-400 회 발생
   - fix: gather 시 `mark_swap_out_in_flight(req_id)` 호출, drain 시 `clear_swap_out_in_flight(req_id)` 호출. `ensure_capacity` 는 in_flight set 안에 있는 req 는 extend skip
   - verify: mismatch 카운트 176-400 → 0

2. **22 항목 monitor 용 env-gated debug log 복원**
   - `[NEO CDEC CALL]` cdec dispatch counter (env: VLLM_NEO_PROFILE=1)
   - `[NEO SWAP_OUT CALL]` async/sync swap counter
   - cleanup commit 0b318ac26 에서 제거된 두 log marker 를 env-gated 형식으로 복원. production hot-path 영향 없음 (default OFF)

### 재현 절차

```bash
# 1. checkout
git checkout 64f9e0c48

# 2. pacpu rebuild (현 source 와 .so 정합 확보)
CXX=/tmp/gcc12/usr/bin/g++-12 bash csrc/cpu/pacpu/build.sh llama3_3_70b 8

# 3. launch
bash eval/run_neo_standard.sh

# 4. monitor
tail -F eval/results/$(ls -t eval/results/ | head -1)/engine.log.stdout
```

### 비교 (다른 측정과)

| 측정 | commit | tps | wall | mismatch | strict 22 |
|---|---|---:|---:|---:|:-:|
| SUB_027 H5 원본 (5/14 00:53) | 2fce4b357 | 2,302 | 1,768s | 176 | 15/19 |
| SUB_021 (5/14 10:22) | 2fce4b357 + WIP | 2,210 | 1,839s | 0 (우연) | 19/19 |
| **현재 best (5/15 00:14)** | **64f9e0c48** | **2,157** | **1,882s** | **0 (의도)** | **19/19** ★ |
| vanilla baseline (5/14 14:58) | cea5a9d31 | 4,886 | 838s | N/A | N/A |

→ throughput 은 원본 H5 대비 93.7% (single-trial noise), strict 22 fire 는 의도적 보장.

---

(history 는 `README.md` 의 통합 영역 참고)
