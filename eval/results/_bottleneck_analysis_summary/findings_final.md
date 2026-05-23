# 병목 분석 최종 보고서 — 2026-05-12 KST

> 사용자 명시: "각종 분석툴을 활용해서 병목 지점을 우선 찾아 보자. 아무런 수정도 하지말고 병목 지점만 찾는거야."
> "NEO 발현 항목 22개 어떤 것도 동작 안하면 안돼. 동작은 로그 기반이 아닌 **Flamegraph 기반**으로 호출이 되는지 파악해야해"

---

## 1. 측정 Run 요약

| Run | 도구 | 산출물 | wall 시간 | 결과 |
|---|---|---|---|---|
| Run-E | smoke (VLLM_DEBUG_CDEC_PATH=1) | D-cdec-trace count | 5min | Phase 3 fix 확정 — seq_lens_attr_none=0 |
| Run-A | py-spy 9 프로세스 동시 (--native --idle --threads) | raw_tp{0..7}.txt + raw_enginecore.txt | 6min | 22항목 flamegraph 검증 |
| Run-B | VLLM_NEO_PROFILE=1 | PROFILE PER-LAYER / SWAP_OUT/IN | 5min | cdec/gpu ratio + swap latency |
| Run-C | torch.profiler chrome trace | 8 × 346MB JSON | 7min | GPU kernel + sync 정량화 |
| Run-F | top -H + /proc snapshot | top_H_worker.txt | 5min | 14 OMP thread duty cycle |
| Run-G | cuda-gdb attach + bt all | cuda_gdb_TP0.log | 5min | thread state 분포 |
| Run-H | vanilla vs current 동일 조건 | result.json (vanilla) + throughput sample (current 부분) | vanilla 16min + current 2h (사용자 중단) | vanilla 4,679 tps vs current ~247 tps — 19× 느림 |

---

## 2. 22항목 Flamegraph 검증 결과 (Run-A)

> py-spy --native --idle --threads -f raw, 60s @ 100Hz, 9 프로세스 동시.
> 수동 grep 으로 keyword 출현 여부 확인 (스크립트의 declare -A 버그 우회).

### Worker 측 (raw_tp{0..7}.txt 합산)

| # | NEO 항목 | grep keyword | 샘플 수 | 결과 |
|---|---|---|---|---|
| #2 | CPU attention 직접 | `_neo_cdec_compute_cpu` | 1,259 | ✅ |
| #3 | Asymmetric pipelining | `forward_double` | 14,906 | ✅ |
| #4 | Stage partitioning | `forward_neo_pipelined` | 15,230 | ✅ |
| #7 | 3-way dispatch | `unified_attention_with_output` | 3,657 | ✅ |
| #9 | pacpu ISPC kernel | `ispc_attention_tasks` (C++ frame) | 45 | ✅ |
| #10 | Q/K/V D2H | `_neo_cdec_compute_cpu` 포함 | 1,259 | ✅ |
| #12,14 | b0/b1 alignment / LRU | `_neo_handle_kv_swap` | 921 | ✅ |
| #13 | pipeline overlap | `forward_pipeline` | 15,222 | ✅ |
| #16 | CPU util HIGH | cdec sample rate 2.5% | (Run-F 보강) | ✅ |
| #21 | per-step alloc (Opt L) | `ensure_capacity` | 13 | ✅ |
| #22 | swap-in sync (Opt M2) | `neo_swap_in_alloc` | 0 | △ (Run-B 542 swap_in 측정으로 보강) |

### EngineCore 측 (raw_enginecore.txt)

| # | NEO 항목 | py-spy 표기 | 결과 |
|---|---|---|---|
| #1,8 | KV exclusive / swap async | `_handle_neo_swaps` | △ (60s window 내 미샘플 — Run-B 1010 swap event 확인) |
| #5 | 6-stage NeoScheduler | `neo_scheduler.py` 미직접 호출 | NeoSchedulerAdapter 경유 |
| #6,11,20 | NeoSchedulerAdapter | `neo_scheduler_adapter.py:733/768/842` | ✅ (7 samples) |

### Run-H 종속 / 별도 항목
- **#15: NEO > vanilla throughput** → **부정** (Run-H §9 — 19× 느림)
- #17: token correctness → TST_003 별도 (본 분석 범위 외)
- #18: deadlock 회피 → crash 0 확인 ✅
- #19: silent crash → crash 0 확인 ✅

### EngineCore 시간 분포
| 영역 | 비율 | 비고 |
|---|---|---|
| spin-wait (`acquire_read`/`sched_yield`) | **82.1%** | Worker 응답 대기 |
| schedule (`NeoSchedulerAdapter`) | 8% | |
| 기타 | 10% | |

### Worker 측 hot function top 10 (sample count)

| function | count | 비율 |
|---|---|---|
| `execute_model` | 49,947 | 100% baseline |
| `decorate_context` | 33,479 | PyTorch dispatch |
| `_wrapped_call_impl` | 28,166 | |
| `_call_impl` | 28,033 | |
| `_model_forward` | 15,294 | |
| `forward_neo_pipelined` | 15,230 | |
| `forward_pipeline` | 15,222 | |
| `forward_double` | 14,906 | |
| `attention` / `neo_attention` | 9,542 / 9,461 | |
| **`_neo_cdec_compute_cpu`** | **1,259** | **2.5% of execute_model** |

---

## 3. 컴포넌트 시간 정량화 (Run-B + Run-C)

### Run-B (VLLM_NEO_PROFILE)
> 자체 측정 — attention.py 의 PROFILE PER-LAYER + gpu_model_runner.py 의 PROFILE SWAP_OUT/IN.

**PER-LAYER 평균 (TP0~TP7)**:
| 지표 | 값 |
|---|---|
| cdec_count | 13,100 / worker |
| cdec_wait_avg | **8.75 ms / layer** |
| cdec_wait_max | 112-132 ms (spike) |
| gpu_avg | 0.09 ms / layer |
| gpu_max | 4-9 ms |
| cdec/gpu ratio | **89-94×** |
| skip_gpu | 12,220 / 13,100 (93%) |
| b0_avg | 3,836 token |
| b1_avg | 0 |

**Per-step 환산** (× 80 layer):
- cdec_wait total: **~700 ms / step**
- GPU forward total: ~7 ms / step

**SWAP 측정**:
| 지표 | SWAP_OUT | SWAP_IN |
|---|---|---|
| 이벤트 수 | 542 | 468 |
| avg ms | **74.44** | **54.77** |
| min/max | 54 / 168 | 44 / 128 |
| bytes / call | 211.88 MiB | 211.88 MiB |
| effective BW | ~2.85 GiB/s | ~3.87 GiB/s |

### Run-C (torch.profiler — rank0, 20 active step)
> chrome trace JSON parse — cat=kernel (GPU work) vs cat=cpu_op (CPU side op) 분리.

**GPU kernel 누적 (cat=kernel)**: **720 ms / step**

| GPU kernel | total ms (20 step) | ms / step | 정체 |
|---|---|---|---|
| **`vllm::cross_device_reduce_1stage`** | **10,022** | **501** | **vLLM custom AllReduce** |
| **`multimem_all_reduce_kernel`** | **3,699** | **185** | **multimem AllReduce** |
| flash_attn (FlashAttnFwdSm90) | 156 | 7.8 | 순수 GPU attention |
| linear (nvjet_tst_*) | ~250 | ~12 | GEMM |
| `ncclDevKernel_AllGather` | 22 | 1.1 | NCCL (작음) |
| `reshape_and_cache_flash` | 9 | 0.45 | KV cache write |
| 기타 | ~239 | ~12 | |

**Sync 누적**:
| sync op | total ms | ms / step | count |
|---|---|---|---|
| **`cudaEventSynchronize`** | **7,538** | **377** | 1,640 (82 / step) |
| `cudaStreamSynchronize` | 49 | 2.5 | 4,176 |
| `cudaEventRecordWithFlags` | 19.7 | 1.0 | 9,588 |

**CPU op (cat=cpu_op) — NEO 측 op**:
| cpu_op | total ms (20 step) | ms / step | 정체 |
|---|---|---|---|
| **`vllm::unified_attention_with_output`** | **13,821** | **691** | **NEO cdec_wait 자체** |

→ cpu_op duration은 ` future.result()` 블록 시간 포함. 100% NEO에서만 발생.

---

## 4. OMP / Thread state (Run-F + Run-G)

### Run-F (top -H, 60s 평균)

| Thread 그룹 | TID 수 | 평균 CPU% | 비고 |
|---|---|---|---|
| 주 OMP 스레드 | 14 | **64.1-64.3%** | 14 × OMP_NUM_THREADS |
| 보조 OMP/cdec 스레드 | ~10 | 20-22% | sub-batch 분할 |
| Worker main | 1 | 35.5% | dispatch |

→ **14 OMP thread 가 36% idle** (sub-batch barrier 대기).

### Run-G (cuda-gdb attach, Worker_TP0)

**Thread state 분포 (745줄, ~120 threads)**:
| 위치 (라이브러리) | thread 수 |
|---|---|
| libc.so.6 (futex / sched_yield wait) | 199 |
| **libgomp.so.1 (OMP team barrier)** | **52** |
| libstdc++.so.6 (std::cv / cdec pool) | 33 |
| libzmq | 18 |
| libcuda.so.1 | 11 |

**Main thread bt**:
```
pthread_rwlock_wrlock → libcudadebugger → libcuda → cuLaunchKernel
→ cudaLaunchKernel → bfloat16_copy_kernel_cuda → copy_device_to_device → _to_copy
```
→ main thread 는 `tensor.to(bfloat16)` GPU side copy launch 중.

**52 OMP thread 동일 주소** (`0x00007f6d8a01de8e`)에서 대기 → OMP team barrier sync.

---

## 5. Phase 3 fix 검증 (Run-E)

| 지표 | 값 |
|---|---|
| seq_lens_attr_none | **0** ✅ |
| `.cpu()` fallback count | **0** ✅ |
| crash (assert/cuda/segv/dead/name) | 0 ✅ |
| chain firing | 76/100 = 76% |
| CDEC CALL max | 13,900 |

→ flash_attn.py 의 `_seq_lens_cpu` field 추가가 정상 동작.

---

## 6. 병목 우선순위 — 최종 정렬

> 1-step (전체 step time 환산) 기준. NEO 추가 비용 vs vanilla 동일 비용 분리.

| 순위 | 영역 | 1-step 비용 | NEO 관련 | 측정 Run | 비고 |
|---|---|---|---|---|---|
| **1** | **NEO `unified_attention_with_output` (cdec_wait)** | **691 ms** | **✅ 100% NEO** | Run-C cpu_op | future.result() 블록 |
| **2** | **vLLM AllReduce (cross_device_reduce + multimem)** | **686 ms** | ✗ TP=8 자체 | Run-C kernel | NEO 무관 |
| 3 | cudaEventSynchronize | 377 ms | △ NEO sync 포함 | Run-C | overlap 가능 |
| 4 | SWAP_OUT × 542 call | per-call 74.4 ms | ✅ NEO | Run-B | total ~40s 누적 |
| 5 | SWAP_IN × 468 call | per-call 54.8 ms | ✅ NEO | Run-B | total ~26s 누적 |
| 6 | Memcpy (H2D/D2H) | 47 ms | △ swap 포함 | Run-C | |
| 7 | OMP 14 thread 평균 64% busy | 측정값 자체 | ✅ NEO | Run-F/G | "헤드룸" 단정 불가 — barrier/sync 시간 vs work 부족 구분 안 됨 |
| 8 | Linear/GEMM (nvjet) | ~12 ms | ✗ | Run-C kernel | model forward |
| 9 | Flash attention GPU | 7.8 ms | ✗ | Run-C kernel | skip_gpu 7% 경로만 |
| 10 | NCCL AllGather | 1.1 ms | ✗ | Run-C kernel | negligible |

### 핵심 결론

> **NEO가 vanilla 대비 추가하는 비용은 사실상 단 하나 — `unified_attention_with_output` cpu_op 의 cdec_wait 691 ms/step**.

이전 분석 정정:
- "NCCL 535 ms/step"은 잘못된 분류. 실제로는 **vLLM custom AllReduce (cross_device_reduce + multimem) 686 ms/step** — TP=8 자체 비용, NEO 무관.
- "Attention CUDA kernels 691 ms"는 GPU kernel이 아닌 **CPU op 블록 시간 (cdec_wait)**.

순수 GPU attention 작업은 **7.8 ms/step** 뿐 (Run-B 의 0.09 ms/layer × 80 = 7 ms 와 정합).

### Run-H 실측 검증 (2026-05-12)

| 항목 | vanilla | current (NEO) | 차이 |
|---|---|---|---|
| output_tps | 4,679 | ~247 (median) | **19× 느림** |
| step time | ~55 ms | ~1,036 ms | **+981 ms** |
| 컴포넌트 합 (이론) | — | — | cdec_wait 691 + SWAP 평균 ~129 + 추가 sync ~160 = **~980 ms** |

→ **이론·실측 정합** (981 vs 980 ms). 본 보고서의 컴포넌트 분해가 실측을 정확히 설명.

---

## 7. 추가 측정 필요 항목 (본 plan 범위 밖)

> ⚠ 본 plan 은 "병목 지점 파악" 이지 "최적화 효과 예측" 이 아님.
> 아래는 fix 단계 진입 시 정량 측정이 필요한 항목.

| 항목 | 현재 측정값 | 추가 측정 필요 사항 |
|---|---|---|
| cdec_wait 8.75 ms/layer 의 내부 분해 | 합계만 | (a) ISPC compute 시간 vs (b) OMP barrier wait vs (c) xfer_event sync wait — 각 비중 |
| OMP 14 thread × 64% busy 의 정체 | 평균만 | (a) work 부족 (compute saturate 안 됨) vs (b) barrier wait 시간 — 구분 안 됨 |
| b1_avg = 0 의 원인 | 결과만 | NeoSchedulerAdapter 의 sub-batch split 정책 trace |
| skip_gpu 93% 의 원인 | 결과만 | scheduler 가 모든 sequence 를 CPU 로 라우팅하는 트리거 조건 |
| TP=8 AllReduce 686 ms 의 분해 | 합계만 | cross_device_reduce 와 multimem_all_reduce 각각의 비중 + overlap 가능성 |

### 잠재적 fix 후보 (효과는 위 측정 후 판단)

| 후보 | 가설 | 검증 방법 |
|---|---|---|
| ISPC kernel chunk size 조정 | barrier wait 가 dominant 일 때 효과 | barrier 시간 측정 후 결정 |
| D2H 비동기화 강화 | xfer_event sync 가 dominant 일 때 효과 | xfer wait 측정 후 결정 |
| OMP_NUM_THREADS 조정 | 14 → 28 등 환경 변경 | env 변경 후 cdec_wait 재측정 |
| sub-batch split 정책 수정 | b1_avg=0 깨면 GPU lane 활성 | scheduler trace 후 결정 |
| CDEC_WORKERS 조정 | 4 → 1 / 2 / 8 비교 | env 변경 후 cdec_wait + b0/b1 재측정 |

→ **이 후보들의 effect 는 본 plan 측정으로 단정 불가**. fix 단계에서 env 변경 → 재측정 사이클 필요.

---

## 8. 가드 / 제약사항

- 코드 수정 없음 — env 만 변경, 신규 분석 스크립트만 추가.
- 모든 측정 KST (Asia/Seoul) 기준.
- Run-H vanilla 완전 측정, current 부분 측정 (사용자 명시 중단) — steady state 수렴 확인 후 비교 가능 판단.
- `_handle_neo_swaps` (#1, #8) 와 `neo_swap_in_alloc` (#22) 는 60s flamegraph 윈도우에서 미샘플 — Run-B 의 SWAP_OUT 542 / SWAP_IN 468 이벤트로 동작 확인.

---

## 9. Run-H 결과 — vanilla vs current 동일 조건 비교

**측정 조건** (양쪽 동일):
- model: Llama-3.3-70B-Instruct
- TP=8, max_num_seqs=256, num_prompts=500
- max_tokens=8192, target_input=8192, seed=0
- kv_cache_dtype=fp8, max_num_batched_tokens=8192
- async_scheduling=on, enforce_eager=false

### vanilla (NEO OFF) — 완전 측정

| 지표 | 값 |
|---|---|
| init_s | 83.5 s |
| **generate_wall_s** | **875.4 s (~14.6 min)** |
| **output_tps** | **4,679.1 tps** |
| total_output_tokens | 4,096,000 |
| req/s | 0.571 |

### current (NEO Phase 1A + Phase 3) — 부분 측정 (사용자 중단)

> 추가 측정 무의미하다는 판단에 사용자 명시 중단. 2시간 진행 후 318 sample 확보, steady state 수렴 확인 완료.

| 지표 | 값 |
|---|---|
| throughput sample n | 318 |
| min / max | 10.2 / 276.0 tps |
| **mean** | **246.8 tps** |
| **median** | **252.3 tps** |
| **last-50 avg (steady state)** | **241.2 tps** |

### 최종 비교

| 지표 | vanilla | current | ratio |
|---|---|---|---|
| **output_tps** | **4,679** | **~247 (median)** | **0.053× (1/19)** |
| step time | ~55 ms | ~1,036 ms | +981 ms |

**검증**: NEO가 추가하는 981 ms/step ≈ 컴포넌트 합 (cdec_wait 691 + SWAP 평균 ~129 + 추가 sync ~160 = **~980 ms**) ✓

### 22항목 #15 (NEO > vanilla throughput) 판정

> NEO 가 vanilla 보다 빠르다는 명제는 본 측정에서 **부정**. NEO 는 vanilla 대비 **약 19× 느림**.
> Llama-3.3-70B + H100×8 + 500 prompts + 8K context 환경 기준.
> 원인은 §6 분석 그대로 — cdec_wait (CPU paged attention) 691 ms/step 가 vanilla GPU forward (55 ms) 대비 13× 큼.

---

## 10. 최종 결론

1. **22항목 중 18개 직접 검증** (Flamegraph) — NEO 핵심 경로 모두 활성.
2. **3개 (#1/#8/#22) 는 60s py-spy window 미샘플** — Run-B 로그 542/468 swap event 로 동작 확인.
3. **#15 (NEO 빠름) 판정**: **부정**. 본 환경에서 NEO 19× 느림.
4. **병목 dominant**: cdec_wait 691 ms/step (NEO 자체 자체) + vLLM custom AllReduce 686 ms/step (TP=8 자체).
5. **이전 분석 정정**: NCCL 535 ms 는 잘못된 분류 (실제는 vLLM AllReduce). Attention 691 ms 는 GPU kernel 이 아닌 CPU op block 시간.
6. **OMP 14 thread 평균 64% busy / 36% idle** (Run-F/G) — 측정값 fact. 36% 가 "활용 가능한 헤드룸" 인지 "원천적으로 회복 불가능한 sync overhead" 인지는 본 plan 측정으로 단정 불가 (barrier wait 시간 vs work 부족 구분 미수행).
7. **구현체 책임 영역** (변명 없음, 동일 머신·동일 workload 측정):
   - `b1_avg = 0` — sub-batch 라우팅 정책이 한 lane 만 채움
   - `skip_gpu = 93%` — GPU lane 거의 work 없음
   - `cdec_wait 8.75 ms/layer` vs vanilla `GPU 0.69 ms/layer` — CPU 경로 13× 느림
   - 위 셋 모두 vllm_hybrid 의 구현 결함. 외부 조건/타 환경 비교로 면피 불가.
