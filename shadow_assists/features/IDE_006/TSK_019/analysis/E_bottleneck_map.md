# Phase E — Bottleneck map (최종 표 1)

> 분석 시각: KST 2026-05-15 ~
> 산출 목적: NEO v1.6 운영 환경에서 CPU 측 시간 소모가 발생하는 **모든 위치를 함수 / line / kernel 단위로 분해** 한 표.
> 입력: A_kernel_signature_map.md, A_neo_upstream_audit.md, C_existing_paths_inventory.md, C_pacpu_vs_cpu_attn_amx_gap.md, D_bottleneck_table.md, D_roofline_notes.md, D_candidate_long_list.md
> 측정 dir: `eval/results/20260514_141511_v16_flamegraph/flamegraphs/` (NEO v1.6, 8 worker × 90s, 61,618 samples), `eval/results/20260514_233540_neo_standard/` (commit `64f9e0c48`, 2,157 tps, 1882s wall)
> **본 문서는 결정 문서 아님**. 위치 + 정량 정보만 제공.

---

## E.1 Bottleneck map — 표 형식

각 row 는 한 개의 CPU 시간 소모 위치. 영역 분류:
- **P** = pacpu kernel 내부 (CPU paged_attention, NEO cherry-pick)
- **S** = swap path (NEO swap-in/out)
- **F** = forward / attention dispatch (Python + ATen)
- **R** = RPC / wait (engine ↔ worker)
- **G** = GPU sync (output thread + libcuda)
- **O** = OMP / thread overhead
- **T** = TP collective (NCCL)

| ID | 영역 | 위치 (file:line 또는 symbol) | 호출 path | 영역 % | ms/call 추정 | 호출 빈도 | 현재 ISA 상태 |
|---|---|---|---|---:|---:|---|---|
| BM01 | P | `csrc/cpu/pacpu/pacpu.ispc:5` (`qk_product`) | `attn_one_seq` → ispc | 40% of cdec_wait | ~3.5 ms / layer | 80 layer × cdec freq | ISPC `avx512spr-x16` FP16 |
| BM02 | P | `csrc/cpu/pacpu/pacpu.ispc:71` (`av_product`) | `attn_one_seq` → ispc | 40% of cdec_wait | ~3.5 ms / layer | 동상 | ISPC `avx512spr-x16` FP16 |
| BM03 | P | `csrc/cpu/pacpu/pacpu.ispc:109` (`softmax`) | `attn_one_seq` 내부 | 10% of cdec_wait | ~1.0 ms / layer | 동상 | ISPC built-in exp |
| BM04 | P | `csrc/cpu/pacpu/pacpu.ispc:142-160` (`attn_one_seq`) | host `core.h:ispc_attention_tasks` | 5% of cdec_wait | ~0.4 ms / layer | per seq × layer | ISPC composition |
| BM05 | O | `csrc/cpu/pacpu/core.h:296` (`#pragma omp parallel`) | `ispc_attention_tasks` 진입 | 5% of cdec_wait (kernel) + 8.26% (worker via index_kernel) | ~0.5 ms / layer | per layer | libgomp default |
| BM06 | O | `csrc/cpu/pacpu/core.h:314,333` (omp barrier) | OMP team 내부 | 3% of cdec_wait | ~0.3 ms / layer | × 2 per layer | libgomp barrier |
| BM07 | S | `vllm/v1/core/sched/neo_cpu_kv_buffer.py:464` (`copy_layer_out`) | `_neo_swap_in_one_req` → `_wrapped` | swap path 232 samples (TP0) | (swap 영역 8.26% omp_pool 의 source) | per swap-in req | ATen `index_kernel` + AVX-512 (libtorch 내부) |
| BM08 | S | `vllm/v1/worker/gpu_model_runner.py:6735` (`_neo_handle_kv_swap`) | `execute_model` → swap orchestration | 1,187 samples (TP0) = ~12% of worker | (swap-out + swap-in) | per step | Python + ATen dispatch |
| BM09 | S | `vllm/v1/worker/gpu_model_runner.py:696x` (`_neo_swap_in_one_req`) | `_neo_handle_kv_swap` 내부 | 462 samples | (swap-in 본문) | per req | Python + tensor ops |
| BM10 | F | `vllm/model_executor/layers/attention/attention.py:1014` (`cdec_future.submit`) | model forward → pacpu dispatch | (executor 영역, dispatch only) | (Python overhead 작음) | per layer per req | Python ThreadPoolExecutor |
| BM11 | F | `vllm/model_executor/models/llama.py:636` (`attention`) | `forward_neo_pipelined` → `_forward_pipeline_inner` | 1,677 samples (TP0) = ~16% of worker | (Python dispatch + ATen) | per layer per step | Python |
| BM12 | F | `vllm/model_executor/models/llama.py:422` (`neo_attention`) | NEO sub-batch attention path | 1,675 samples (TP0) | (cdec dispatch + GPU 합) | per layer per step | Python |
| BM13 | F | `vllm/v1/attention/sub_batch_executor.py:23x` (`forward_double`) | NEO sub_batch pipeline | 1,688 samples (TP0) | (CUDA stream sync 영역) | per step | Python + CUDA |
| BM14 | G | `torch._C._cuda.THPEvent_synchronize` (`cudaEventSynchronize`) | async output thread + sub_batch sync | 1,749 samples × 8 = ~18% (worker total) | (GPU 대기) | per event | CUDA driver |
| BM15 | R | `vllm/v1/utils/shm_broadcast.py:755` (`dequeue`) | worker_busy_loop input wait | 1,689 samples (TP0) = ~16% | (RPC wait) | per step | Python + sched_yield |
| BM16 | R | `vllm/v1/utils/shm_broadcast.py:674` (`acquire_read`) | dequeue 내부 lock | 1,407 samples (TP0) = ~14% | (idle wait) | per dequeue | sched_yield |
| BM17 | F | `vllm/v1/worker/gpu_model_runner.py:4587-4595` (`execute_model`, `_model_forward`) | worker_busy_loop → forward | 4,196-4,216 samples (TP0) = ~40% | (forward orchestration + GPU launch) | per step | Python |
| BM18 | F | `vllm/model_executor/models/llama.py:662` (`forward_neo_pipelined`) | execute_model 진입 | 4,190 samples (TP0) | (pipeline orchestration) | per step | Python |
| BM19 | F | `vllm/v1/attention/sub_batch_executor.py` (`_forward_pipeline_inner`) | forward_neo_pipelined 내부 | 4,110 samples (TP0) | (sub-batch dispatch) | per step | Python |
| BM20 | T | NCCL all_reduce | TP communication | 2.86% (worker total) | ~50 us / layer | per layer × 80 | NCCL (자체 AVX-512 가능) |
| BM21 | G | `torch.ops.aten.copy_` + `index_kernel` (libtorch) | swap path advanced indexing | 8.26% omp_pool | (BM07 의 OMP source) | per swap-in | OMP + AVX (ATen) |
| BM22 | F | Python closures / dict / layer dispatch | llama.py:600-680 영역 | ~3% (worker total) | (interpreter overhead) | per layer | scalar |

---

## E.2 정량 root totals (검증)

| 영역 | samples | % | 산정 방식 |
|---|---:|---:|---|
| **NEO cdec 영역** (pacpu 자체 시간) | < 600 | **< 1%** | flamegraph 에서 `libpacpu-*.so` 의 symbol 0건 |
| **NEO swap 영역** (BM07-BM09 + BM21) | ~7,500 | **~12%** | `_neo_handle_kv_swap` self+children |
| **GPU forward (F + G)** (BM10-BM19) | ~25,000 | **~40%** | model forward + GPU sync |
| **Engine ↔ Worker RPC** (BM15, BM16) | ~16,500 | **~27%** | idle wait |
| **async output thread** (BM14 일부) | ~11,000 | **~18%** | cudaEventSync + libc threads |
| **libtorch / aten 일반** | ~11,000 | **~18%** | model layer ops (overlap 있음, 영역 합 100% 초과) |
| **TP all_reduce** (BM20) | ~1,800 | **2.86%** | NCCL |
| **Python overhead** (BM22) | ~3,000 | **~5%** | Python bookkeeping |

→ 합 100% 초과 (영역 중복 포함). flamegraph 의 multi-frame sample 한 sample 이 여러 영역에 잡힘.

---

## E.3 핵심 fact 정리

1. **pacpu kernel (BM01-BM04) self-time < 1%** — flamegraph 의 hot frame 이 아니다. cdec dispatch 빈도가 mirror=10/running=222 (4.5%) 이라 sample 잡힐 만큼 안 큼
2. **8.26% omp_pool 은 pacpu 가 아니라 ATen `index_kernel`** (BM07 + BM21). NEO swap-in path 의 advanced indexing 에서 OMP 발생
3. **AMX/AVX 적용 후보 영역 (compute-bound)**:
   - BM01 `qk_product` (compute-bound, AI=30-50, AVX-512 ridge 근처)
   - BM03 `softmax` (scalar exp bottleneck)
4. **NEO swap path (BM07-BM09)** 가 worker 의 12% — pacpu 가 아닌 영역. AMX 가속 무관
5. **GPU sync + RPC wait** 가 worker 의 45% — ISA 가속 미관련

---

## E.4 측정 미달 영역 (open)

| 항목 | 현재 상태 | 필요 측정 |
|---|---|---|
| pacpu 3 kernel 별 ms/call 실측 | 추정만 (영역 % × cdec_wait) | PROFILE 로그 (VLLM_NEO_PROFILE=1) 활성 후 추출 |
| chain firing 80-99% 영역 의 pacpu sample | flamegraph 안 잡힘 | Option C/L/M2 ON + flamegraph 재측정 |
| AMX BF16 변환 cost | 미측정 | dev 머신에서 conversion bench |
| `cdec_executor max_workers` 의 layer 의존성 cap | SUB_023 settled (+4 시 -52%) | AMX 적용 후 재측정 필요 |
| OMP team launch overhead 영역 % 정확 분해 | 추정 (BM05 8.26%) | OMP `OMP_TEAM_OVERHEAD` profile |
| libpacpu-*.so symbol 가시화 | symbol 없음 (gcc -g 미사용) | `-g -fno-omit-frame-pointer` 재빌드 후 flamegraph 재측정 |

---

## E.5 cross-ref

- 각 row 의 file:line 은 `git grep` 또는 직접 `Read` 로 검증 가능
- 영역 % 와 samples 는 `eval/results/20260514_141511_v16_flamegraph/flamegraphs/flame_tp0.svg` 기준
- pacpu 영역 (BM01-BM06) 는 sample 직접 없음 — `cdec_wait 8.75 ms / layer` 와 산술적 % 분해
- 본 표는 Phase E 의 두 번째 표 (`E_amx_avx_applicability.md`) 의 입력

---

## E.6 측정 보강 (2026-05-15 KST, 정정 v2) — swap path 의 wall vs CPU duty 분리

`E_open_questions.md` 의 OQ11 측정 결과 반영.

### swap path 의 async hidden vs critical path 분리 (BUFFERS sweep 포함)

| 측정 | BUFFERS | wall (s) | output_tps | async swap (TP0) | sync swap (TP0) |
|---|---:|---:|---:|---:|---:|
| ASYNC=1, B=3 (baseline) | 3 | 995.3 | 1,638.3 | **12,440** | **0** |
| ASYNC=1, B=6 | 6 | 995.5 | 1,645.9 | **12,440** | **0** |
| ASYNC=0 (sync only) | (off) | 1,216.8 | 1,346.5 | 0 | **24,496** |

→ 측정 dir: `eval/results/20260515_074040_async1_base/` (B=3), `..083247_async1_b6/` (B=6), `..075914_async0_sync/` (sync), script `eval/run_neo_async_sweep.sh`, 200p × 8192 in/out

### 정정 fact (이전 분석의 오류 정정)

이전 분석 (sync fallback 8:1 dominant) 은 shell script 의 `grep -c 'sync'` 가 **"Asynchronous"** 단어 안의 "sync" 에 매치된 카운팅 오류. 정정:

1. **ASYNC=1 시 모든 swap-out 이 async path 처리** (sync fallback 0)
2. **BUFFERS=3 vs 6 거의 동일** — 한 step 당 평균 swap-out 수가 3 이하
3. **ASYNC=0 시 swap-out 횟수 ~2× 증가** (12,440 → 24,496) — sync 가 느려 GPU KV backpressure → swap 더 빈번

### BM07/BM08/BM09/BM21 (swap path) 영역의 wall impact 재분류

| ID | flamegraph CPU duty | wall impact (실측 정정) | 비고 |
|---|---:|---|---|
| BM08 `_neo_handle_kv_swap` (main thread) | 11-12% | **wall hidden 영역 대부분** (async 가 forward 와 overlap) | ASYNC=1 시 sync fallback 0 |
| BM07 `copy_layer_out` (TP0 의 drain target) | 232 samples (TP0) | wall critical (drain blocks forward) | TP0 의 `_neo_drain_pending_swap_dma` 7.28% — drain wait 영역, async 라도 forward 후 drain 동기 wait |
| BM09 `_neo_swap_in_one_req` | 462 samples (TP0) | **wall critical** (GPU 가 데이터 받아야 forward) | swap-in 은 sync path, async 무관 |
| BM21 `at::native::index_kernel` OMP 8.26% | 5,100 samples | **wall hidden 영역** (OMP team 별도 thread + async stream 의 gather 영역) | async swap-out gather 의 OMP 는 wall hidden |

### swap path 의 wall vs CPU duty 비율 (정정)

- flamegraph 의 NEO swap 영역 = ~12% (worker CPU duty)
- **ASYNC=1 baseline 에서 wall 에 실제 contribut 하는 영역**: 주로 swap-in (BM09) + drain wait (BM07) — 약 **5-7% 의 wall impact** 추정
- **ASYNC=0 비활성 시 wall +22.3% 회귀** — async hidden 효과 손실 + swap-out 횟수 2× 증가의 합산
- async swap-out 의 hidden 효과는 main thread CPU duty 의 상당 부분 (gather + DMA + OMP) 을 wall 에서 가림

→ 사용자의 가설 "swap path 가 wall 에서 hidden 처리됨" 은 **정합** — ASYNC=1 시 swap-out 영역의 CPU duty 가 wall 에서 대부분 hidden. AMX/AVX 가속 검토 시 **swap path 의 우선순위는 낮음** (async 가 hidden 이미 처리 + swap-in/drain wait 만 wall critical).

---

## E.7 모든 overhead 항목 thread 위치 + wall critical 검증 (2026-05-15 KST)

TP0 worker (10,407 samples = 100%) 의 thread 구조 + 각 항목의 wall 영향 분류.

### Thread 구조

| Thread | sample | % | 영역 |
|---|---:|---:|---|
| main worker thread (`worker_main` multiproc_executor.py:876) | 8,558 | **82.23%** | model forward + swap + execute + RPC wait |
| async output thread (`_bootstrap` threading.py:1032) | 1,798 | **17.28%** | GPU output event wait (cudaEventSync) |
| 합 | 10,356 | 99.51% | (나머지 0.49% = OMP team / Triton background) |

### Wall critical / hidden / idle 분류 표 (BM ID 와 cross-ref)

| BM | 항목 | sample | % | thread | wall 분류 |
|---|---|---:|---:|---|---|
| BM18 | forward_neo_pipelined (llama.py:662) | 4,190 | 40.26% | main | **✓ critical** |
| BM19 | _forward_pipeline_inner (sub_batch_executor.py:338) | 4,110 | 39.49% | main | **✓ critical** |
| BM11/BM12 | forward_double + attention (gdec/cdec) | 1,688+1,265 | 28.38% | main | **✓ critical** |
| BM08 | _neo_handle_kv_swap (gpu_model_runner.py:6735) | 1,187 | 11.41% | main | **✗ hidden** (ASYNC=1, OQ11) |
| BM07 | _neo_drain_pending_swap_dma | 758 | 7.28% | main | **△ partial** (drain sync wait 의 tail) |
| BM15 | dequeue (shm_broadcast:755) | 1,689 | 16.23% | main | **△ idle wait** (engine dispatch latency) |
| BM16 | acquire_read (shm_broadcast:674) | 1,407 | 13.52% | main | **△ idle wait** |
| BM15 | wait (shm_broadcast:184) | 1,112 | 10.69% | main | **△ idle wait** |
| (RPC) | sched_yield (utils.py:48 + libc.so) | 2,049 | 19.69% | main | **△ idle wait** (dequeue 내부) |
| BM14 | cudaEventSynchronize | 1,746 | 16.78% | **별도 output thread** | **✗ wall critical 아님** (병렬) |
| BM21 | OMP pool (ATen index_kernel via swap-in) | ~5,100 | 8.26% (avg) | **별도 OMP thread** | **△ partial** (main wait 중이면 critical) |
| BM20 | NCCL all_reduce | ~298 | 2.86% | main | **✓ critical** (TP block) |

### 정정된 분류 결론

| 분류 | 합산 % | 영역 |
|---|---:|---|
| **wall critical (main thread, model forward + NCCL + drain)** | **~50%** | 진짜 model 일 |
| **wall hidden (별도 thread / async stream)** | **~36%** | output event + swap async + OMP |
| **idle wait (main thread, dispatch latency)** | **~30-40%** | RPC wait, 영역 중복 |

### 이전 "overhead" 분류 정정

이전 분석에서 "wall critical" 처럼 표기된 항목 중 실제로는 hidden / 별도 thread 인 것들:

1. **cudaEventSync 18%**: 별도 async output thread, main thread 와 병렬 — wall 의 tail 영역에서만 critical
2. **NEO swap path 12%**: ASYNC=1 시 async stream hidden (OQ11 측정으로 확정)
3. **OMP pool 8.26%**: ATen `index_kernel` 의 OMP team thread — main thread wait 중이면 critical, async path 안이면 hidden
4. **RPC wait 27%**: idle wait 영역 — worker 자체 일이 아니라 engine dispatch latency 의 결과

### Phase E 표 (AMX/AVX 가속) 우선순위 영향

- 진짜 wall critical = model forward (BM18-BM19) + GPU attention (BM11 gdec) + CPU attention (BM12 cdec) + NCCL (BM20) + drain tail (BM07)
- BM01-BM03 (pacpu kernel) 의 영역은 BM12 (cdec sub-batch) 의 일부 — 약 **12% of main thread = critical path 의 12% 안의 일부**
- AMX 가 BM01 가속 시 cdec sub-batch wall 축소 → 그러나 sub-batch[0] (gdec, GPU) 가 dominant 이면 wall 한계 도달

→ **AMX 가 NEO net-win 영역 도달의 필요 조건은 cdec dispatch 빈도 증가** (chain firing 80-99% 영역 도달) — workload + Option C/L/M2 조정 동반 필요.
