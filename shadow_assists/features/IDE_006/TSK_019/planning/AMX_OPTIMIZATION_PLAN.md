# AMX 최적화 + 전체 최적화 통합 plan

> **작성**: 2026-05-21 KST (turn 8, turn 11 정리 영역에서 별도 doc 으로 추출).
> **출처**: 사용자 명시 (turn 8 verification): "이걸 기반으로 AMX 최적화 및 전체 최적화 작업 진행할거야. 아주 꼼꼼하고 자세해야해".
> **본 plan** = 측정 fact ([`../measurements/timeline_neo_amx_apply_20260520/README.md`](../measurements/timeline_neo_amx_apply_20260520/README.md) §16-18) 통합 + [`../analysis/M_sub015_phase3_hpc_optimization.md`](../analysis/M_sub015_phase3_hpc_optimization.md) §5 의 lever priority + [`../analysis/N_cdec_leftover_elimination_ideas.md`](../analysis/N_cdec_leftover_elimination_ideas.md) 의 외부 idea 영역.
>
> **★ 본 doc 영역 = turn 11 reorganization 영역에서 timeline README 영역의 § "AMX 최적화 + 전체 최적화 통합 plan" 영역 영구 추출**.

---


### A.1 출발점 fact (turn 8 측정 backing)

| Baseline | tps | crash | mechanism | 근거 |
|---|---:|---|---|---|
| **v1.6 best (env-OFF, gmu=0.85, 3-run avg)** | **1,833.0** | 0 | S1-S9 mechanism + default 환경 | `p3_compare_3run_085_20260520` §2.2.1 |
| **★ env-ON combined (HEAD `0776086f5`, 1-run)** | **1,833.95 (+0.05% noise)** | 0 | S1-S9 mechanism + P3 (K BF16+AMX) + P4 (async cdec) + D + OOB-silent + OOB G/H rate-limit | §18 본 turn fact |
| **vanilla (NEO off, gmu=0.85, 3-run avg)** | **4,680.2** | 0 | 영역 reference 분모 | §2.2.1 |

**핵심 fact**:
- env-ON combined = v1.6 best 와 essentially equal (+0.05% noise, 1-run)
- 이전 P3 단독 -2.5% 회귀 → 본 turn env-ON combined +0.05% = **swap_out dtype fix + P4 async cdec + OOB stability** cumulative 의 효과
- **long workload stability 영구 확정** (이전 P4 단독 EngineDeadError 해소)
- AMX 본체 의 진정한 net loss 영역 = **±2% noise 안** (3-run avg 영역 statistical confidence 필요)

### A.2 ② cdec leftover +18 ms 의 cycle 분해 (perf record 2026-05-17, S1-S9 base)

| Layer | cycle % (perf) | wall 환산 (② 의 비율) | 의미 |
|---|---:|---:|---|
| **libgomp (OMP barrier wait)** | **43.75%** | **+11.2 ms (62% of ②)** | barrier #1/#2/implicit 의 thread imbalance wait → ★★★ TRUE TOP BOTTLENECK |
| **libpacpu (ISPC compute)** | 26.38% | +6.8 ms (38% of ②) | actual cdec 영역 work |
| ⊳ softmax (transcendental) | 9.73% of total | +2.5 ms | cycle/FLOP worst — ★ fast_exp 후보 |
| ⊳ qk_product (matmul) | 8.75% of total | +2.3 ms | AMX 본체 영역 가속 candidate |
| ⊳ av_product (matmul) | 7.90% of total | +2.0 ms | AMX 본체 영역 가속 candidate |
| libtorch_cpu (swap path) | 10.24% | ⚪ async wall hidden | parallel work, wall block X |
| python interpreter | 1.84% | (① 의 일부) | |

### A.2.1 ★ 전체 workflow phase 영역 distribution (turn 9 보강 — py-spy stack 분석 backing)

**측정 환경**: env-ON 100p × 8192 × 60s py-spy 9-process 병렬 sampling, Worker_TP × 8 aggregate (EngineCore 제외).
**Total samples**: 13,999. **Active (non-wait) 영역**: **41.6% (5,824 samples)** — 나머지 58.4% = WAIT_dequeue (ep_poll) + futex_wait.

**Active 영역 100% 기준 phase 분포**:

| # | Workflow phase | active % | 영역 시간 추정 (per-step) | step 영역 위치 | py-spy stack source |
|---|---|---:|---:|---|---|
| 1 | **FWD_double_attention** (b0/b1 attn + cdec direct) | 24.67% | ~17 ms | layer loop 영역 (6~54 ms) | `forward_double` / `neo_attention` / `unified_attention_with_output` |
| 2 | **STEP_prepare_inputs** ★ MISSING in old timeline | **21.91%** | **~10 ms** | **step start (0~3 ms area)** | `_prepare_inputs` / `_prepare_input_ids` / `_update_states` |
| 3 | **SWAP_in** (copy_layer_out × 80 layer) | 17.03% | ~12 ms | forward loop 영역 distributed | `copy_layer_out` / `_neo_handle_kv_swap` |
| 4 | (other / unclassified) | 14.18% | ~10 ms | 분포 | execute_model entry / all_gather / _get_slot_mappings / flash_attn.py:build / copy.py |
| 5 | **SAMPLE** (logits sampling) ★ MISSING breakdown | **10.54%** | **~9 ms** | **③.1 영역 (84~93 ms)** | `sampler` / `multinomial` / `_get_next_token` |
| 6 | **SWAP_out_async** (gather/dma launch) | 6.35% | ~4 ms | swap_stream 영역 | `_neo_swap_out_gather` / `_neo_swap_out_dma` |
| 7 | ATEN_index/copy/reshape | 4.79% | ~3 ms | swap path 의 CPU spillover | `torch::utils::recursive_store` / `index_kernel` |
| 8 | FWD_first_stage (Q/K/V transfer launch) | 0.17% | <0.5 ms | step start | `forward_first_stage` |
| 9 | CUDA_event_record/sync | 0.14% | <0.5 ms | ③.3 cudaEventRecord | `record_event` / `synchronize` |
| 10 | GEMM_linear (Q/K/V/out proj) | 0.09% | <0.5 ms | layer loop (분포) | `default_unquantized_gemm` |
| 11 | FWD_postproj | 0.05% | <0.5 ms | layer loop (분포) | `postproj` / `neo_postproj` |
| 12 | EMIT_async_output | 0.03% | <0.5 ms | ③.4 영역 (99~109 ms) | `async_output` / `_bookkeeping_sync` |
| 13 | NCCL_AllReduce | 0.02% | <0.5 ms | layer loop (분포) — main path 영역 외 | `all_reduce` |

→ **wait 영역 58.4%** 의 분해: WAIT_dequeue (next-step IPC) = 58.40% — Worker_TP 가 다음 step instruction 영역 기다리는 영역 (main path wait 영역 의 dominant).

### A.2.2 ★ NEO 1-step workflow 영역 전체 sequence (turn 9 정리)

| 순서 | Phase | 시간 (ms) | wall 영역 | 영역 |
|---|---|---:|---|---|
| 0 | dequeue (다음 step instruction 영역 IPC wait) | (idle) | ⊳ workers waiting | EngineCore 영역 의 다음 step ready signal 영역 |
| **1** | **prepare_inputs** ★ | ~3 ms | main thread | `_prepare_input_ids` / `_update_states` |
| 2 | metadata build, slot_mappings | ~2 ms | main thread | `flash_attn.py:build` / `_get_slot_mappings` |
| 3 | forward_first_stage (Q/K/V transfer launch) | <1 ms | main thread | `forward_first_stage` |
| **4** | **forward_double × 80 layer** ★ | ~48 ms | main thread (대부분) + GPU stream concurrent | per-layer 영역: preproj + attn (b0=GPU full, b1=cdec direct) + postproj + MLP + AllReduce + swap_in copy_layer_out |
| 5 | forward_last_stage | <1 ms | main thread | `forward_last_stage` |
| **6** | **① Python overhead** | +12 ms | main thread | skip_gpu check, cdec submit/launch, cudaStream sync |
| **7** | **② b1 cdec leftover** | +18 ms | main thread (cdec compute on CPU) | NEO §4.4 cdec 의 마지막 leftover 영역 (S5 direct), libgomp barrier 영역 62% + libpacpu compute 38% |
| **8** | **③.1 sample + NCCL all_gather** ★ | +9 ms | main thread (GPU sample + main wait) | logits softmax + multinomial/argmax + NCCL all_gather (TP=8 영역) |
| **9** | **③.2 NEO scheduler admit/swap** ★ | +4 ms | main thread (Python) | 다음 step 영역 admit/swap-in 결정 (NEO scheduler) |
| **10** | **③.3 cudaEventRecord** ★ | +2 ms | main thread (GPU event signal) | step ready signal → async_output wake |
| **11** | **③.4 emit + bookkeeping** | +10 ms | async_output thread | cudaEventSync + token D2H + ZMQ socket |
| **합 (wall)** | | **~109 ms** | | vanilla 54 ms + NEO 추가 55 ms |

**핵심 insight**:
- step start 영역 의 prepare_inputs + metadata build = **~5 ms** (이전 timeline 영역에 missing)
- ③ +25 ms 영역 정량 분해 = ③.1 sample (9 ms) + ③.2 admit (4 ms) + ③.3 cudaEvt (2 ms) + ③.4 emit (10 ms) ✓
- **84~99 ms gap (③.1+③.2+③.3 = 15 ms)** 의 진정한 source = sample + admit + cudaEvt (이전 turn 5 "TBD" → turn 9 정량 확정)
- async_output 의 cudaEventSync 영역 wake = ③.4 영역 만 (10 ms)
- swap_in copy_layer_out (17.03%) = forward 영역에 분포 (per-layer × 80) — 이미 forward 영역 시간 안에 포함

### A.3 ★ AMX 최적화 영역 — 7 sub-task 영역 plan

#### Sub-task A1: AMX setup overhead 단축 (M packing M=8 → 16)
- **현재 상태**: AMX tile 16×16 영역 의 M=8 만 사용 → 50% occupancy. setup overhead 650-900 cycle/block > work 64 cycle × 10배
- **방안**: 2 sequence 의 Q vector 영역 packing M=8+8=16 → AMX tile full occupancy (100% utilization)
- **예상 효과**: -3-5 ms wall (이론 +3-5%)
- **risk**: Q vector 2 sequence packing 영역 의 alignment + boundary handling 영역 복잡성
- **effort**: 2-3 주 (architectural)
- **prerequisite**: env-ON path 활성 (HOST_K_BF16=1 + USE_AMX=1)

#### Sub-task A2: AMX K^T pre-pack 영역 outer-loop hoist (이미 turn 4 의 Step 2 영역에서 시도, 다른 방식)
- **현재 상태**: K^T transpose + BF16 cast 가 block 당 200-400 cycle
- **방안**: K^T pre-pack 영역 을 layer 진입 직후 1 회 batch 영역 cache (SUB_015 Phase 3 Step 2 영역과 유사 — 단 cache scope 영역 확장)
- **예상 효과**: -1-2 ms wall (이론 +1-2%)
- **risk**: cache invalidation 영역 (layer 진입 후 K cache 변경 X 영역 만 적용 가능)
- **effort**: 1 주

#### Sub-task A3: AMX softmax + av 영역 cumulative (현재 softmax/av = ISPC fallback)
- **현재 상태**: AMX 영역 qk 만 적용. softmax + av = ISPC fallback
- **방안**: softmax 의 polynomial 영역 + av 의 matmul 영역 모두 AMX path 적재 (single-pass online softmax + AMX av)
- **예상 효과**: -2 ms (softmax) + -1 ms (av) = -3 ms (이론 +3%)
- **risk**: 정확도 영향 (polynomial degree + AMX accumulator precision) — TST_003 verdict 영역 영구 확인 필요
- **effort**: 1-2 주

#### Sub-task A4: F4 TP=4 영역 (M=8 → 16 — different from A1 packing)
- **현재 상태**: TP=8, NUM_Q_HEADS=8 (model 영역 의 64/TP)
- **방안**: TP=4, NUM_Q_HEADS=16 → AMX tile 100% occupancy 영역 native
- **예상 효과**: -3-5 ms (이론 +3-5%) **단 GPU 측 parallelism -50% trade-off**
- **risk**: GPU 영역 의 throughput 영역 영향 (별도 측정 필요)
- **effort**: 2-4 주 (architectural, NEO scheduler 영역 영향)

#### Sub-task A5: F5 BLOCK_SIZE 16 → 32 영역
- **현재 상태**: BLOCK_SIZE=16
- **방안**: BLOCK_SIZE=32 → per-block work 2× 영역, AMX setup overhead amortize 영역 비율 ↑
- **예상 효과**: -2-4 ms (이론 +2-4%)
- **risk**: NEO scheduler 영역 의 page granularity 변화 영향, KV cache memory 분배 영역 변화
- **effort**: 2-4 주

#### Sub-task A6: AMX BF16 → INT8 quantization 영역 (장기)
- **현재 상태**: AMX BF16 path
- **방안**: K cache INT8 quantization + AMX INT8 path (tile_dpbssd) → 2048 ops/cycle/core (BF16 의 2×)
- **예상 효과**: -3-6 ms (이론 +3-6%) **단 정확도 영향 영구 검증 필요**
- **risk**: K cache 영역 정확도 영향 (per-block dequant scale 영역 보존 필요)
- **effort**: 4-6 주

#### Sub-task A7: AMX KV cache 영역 prefetch (cache miss 영역 단축)
- **현재 상태**: KV cache 영역 sequential access (cache miss 영역 빈도 ↑)
- **방안**: SW prefetch (PREFETCH instruction) 영역 + cache line alignment 영역 영역 정합
- **예상 효과**: -1 ms (이론 +1%)
- **risk**: prefetch hint 영역 의 effectiveness 영역 hardware-dependent
- **effort**: 1 주

### A.4 ★ 전체 최적화 영역 — Phase 별 plan (cumulative)

#### Phase α (즉시, 1-2 일, env-only sweep) — turn 8 instrumentation 활성 + KMP tuning

| Step | Lever | env | effort | 예상 효과 |
|---|---|---|:-:|---:|
| α-1 | OMP barrier instrumentation 활성 + pacpu rebuild | `VLLM_NEO_OMP_PROFILE=1` | 30 min | (측정 only — barrier #1/#2 정량 확보) |
| α-2 | async cdec PROFILE instrumentation 활성 (env-ON 측정) | `VLLM_NEO_PROFILE=1` + env-ON | 30 min | (측정 only — async cdec drain wait 정량) |
| α-3 | KMP_BLOCKTIME=INF + KMP_AFFINITY 명시 sweep | `KMP_BLOCKTIME=INF` + `KMP_AFFINITY=granularity=fine,compact,1,0` | 1 시간 | **-2~-4 ms (-2~4%)** |
| α-4 | env-ON 3-run avg (statistical confidence) | env-ON 모두 활성 | 90 min | (측정 only — +0.05% 영역 confidence 확정) |

#### Phase β (1-2 주, surgical changes)

| Step | Lever | 코드 변경 | effort | 예상 효과 |
|---|---|---|:-:|---:|
| β-1 | softmax fast_exp (ISPC polynomial) | `pacpu.ispc` softmax 영역 3-pass → single-pass | 3-5 일 | **-2 ms (-2%)** |
| β-2 | Sub-task A2 AMX K^T pre-pack outer hoist | `amx_kernel.cpp` | 1 주 | -1-2 ms |
| β-3 | F3 K BF16 host store 단독 (USE_AMX=0) | env-only + 정확도 검증 | 1 일 | -1 ms |

#### Phase γ (2-4 주, AMX 본체 최적화)

| Step | Lever | 코드 변경 | effort | 예상 효과 |
|---|---|---|:-:|---:|
| γ-1 | **Sub-task A1 AMX M packing M=8→16** | `amx_kernel.cpp` 의 Q vector packing 영역 + boundary | 2-3 주 | **-3-5 ms** |
| γ-2 | **Sub-task A3 AMX softmax + av 영역 cumulative** | `pacpu.ispc` 의 softmax/av → AMX path | 1-2 주 | -3 ms |
| γ-3 | Sub-task A7 KV cache prefetch | `core.h` SW prefetch hint | 1 주 | -1 ms |
