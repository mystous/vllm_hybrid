# Phase D — Bottleneck table (flamegraph 재분석)

> 분석 시각: KST 2026-05-15 ~
> 1차 input: `eval/results/20260514_141511_v16_flamegraph/flamegraphs/` (NEO v1.6, 8 worker × 90s)
> 2차 input: `eval/results/20260514_233540_neo_standard/` (commit `64f9e0c48` 측정)

---

## D.1 핵심 발견 — pacpu kernel 자체는 flamegraph 의 hot frame 이 아니다

py-spy native unwind 으로 잡힌 NEO 운영 시점 (90s × 8 worker = 61,618 samples) 의 CPU stack 분석:

1. **`libpacpu-llama3_3_70b-tp8.so` 의 symbol 0건** — pacpu 의 qk_product/av_product/softmax 함수 frame **flamegraph 에 안 보임**
2. **libgomp.so.1 의 OMP pool 8.26%** 는 **pacpu 가 아니라 ATen `index_kernel`** 에서 호출됨

### libgomp call stack (TP0 worker, 1 sample 추적)

```
worker_main
└─ worker_busy_loop
   └─ execute_model (worker_base)
      └─ execute_model (gpu_worker)
         └─ execute_model (gpu_model_runner:4587)
            └─ _neo_handle_kv_swap (gpu_model_runner:6735)        ← 1,187 samples
               └─ _neo_swap_in_one_req (gpu_model_runner:696x)    ← 462 samples
                  └─ _wrapped (neo_cpu_kv_buffer:128)              ← 461
                     └─ copy_layer_out (neo_cpu_kv_buffer:464)     ← 232
                        └─ THPVariable_getitem (libtorch_python)
                           └─ at::indexing::dispatch_index         ← advanced indexing
                              └─ at::native::index_kernel          ← ★ OMP parallel
                                 └─ at::TensorIteratorBase::for_each
                                    └─ GOMP_parallel (libgomp.so) ← **이게 omp_pool 의 source**
```

→ **omp_pool 8.26%** 가 **NEO swap-in path 의 ATen advanced indexing** 에서 OMP 발생. **pacpu kernel 의 OMP 가 아님**.

### 의미

- 우리 측정 환경의 NEO v1.6 에서 cdec dispatch 가 mirror=10 (running 222 의 4.5%) 영역으로 **pacpu kernel 자체 시간이 sample 잡힐 만큼 안 큼**
- flamegraph 의 "CPU bottleneck" 은 **pacpu 가 아니라 ATen indexing + swap-in/out path**
- AMX/AVX 가속이 효과 보려면 **cdec 발화가 충분한 영역 (Option C/L/M2 ON, mirror 60-80)** 에서 측정해야 함 — 본 flamegraph 는 그 영역 측정 아님

---

## D.2 NEO v1.6 worker 의 CPU self-time top frames (TP0, 10,407 samples)

| samples | frame | call stack 짧음 |
|---:|---|---|
| 8,558 | worker_main (multiproc_executor.py:876) | root |
| 6,798 | worker_busy_loop (executor:972) | |
| 4,216 | execute_model (gpu_model_runner:4595) | model forward 진입 |
| 4,196 | _model_forward (gpu_model_runner:3797) | |
| 4,190 | forward_neo_pipelined (llama.py:662) | NEO 진입 |
| 4,110 | _forward_pipeline_inner (sub_batch_executor) | |
| 1,798 | _bootstrap (threading.py:1032) | output thread |
| 1,757 | worker_busy_loop (line 963) | input wait |
| 1,751 | async_output_busy_loop | |
| 1,749 | THPEvent_synchronize | GPU event sync |
| 1,749 | cudaEventSynchronize (libcudart.so) | |
| 1,746 | cuEventSynchronize (libcuda.so) | |
| 1,689 | dequeue (shm_broadcast.py:755) | RPC queue |
| 1,688 | forward_double (sub_batch_executor:23x) | NEO chain firing 영역 |
| 1,677 | attention (llama.py:636) | |
| 1,675 | neo_attention (llama.py:422) | |
| 1,407 | acquire_read (shm_broadcast:674) | |
| 1,381 | 0x...libcuda.so | CUDA driver |
| 1,285 | execute_model (gpu_model_runner:4587) | NEO swap path |
| 1,187 | _neo_handle_kv_swap | ★ NEO swap-out + swap-in |
|  462 | _neo_swap_in_one_req | ★ swap-in 본문 |
|  232 | copy_layer_out (neo_cpu_kv_buffer:464) | CPU → tensor copy |
|  227 | at::native::index_kernel | ★ advanced indexing |
|  220 | at::TensorIteratorBase::for_each | OMP parallel 호출 |
|   85 | GOMP_parallel (libgomp.so) | OMP entry |

---

## D.3 영역별 % 분류 (TP0, 8 worker 합산 61,618 samples)

| 카테고리 | 누적 samples | % | 비고 |
|---|---:|---:|---|
| GPU forward (forward_neo_pipelined + libcuda sync) | ~25,000 | ~40% | NEO forward path |
| Engine ↔ Worker RPC (shm dequeue + acquire_read + sched_yield) | ~16,500 | ~27% | idle wait |
| async output thread (cudaEventSync + libc threads) | ~11,000 | ~18% | output dispatch |
| **NEO swap path** (`_neo_handle_kv_swap` + 자식) | **~7,500** | **~12%** | **swap-out + swap-in** |
| **OMP pool (libgomp via ATen index_kernel)** | **~5,100** | **8.26%** | **ATen advanced indexing** |
| libtorch / aten 일반 | ~11,000 | ~18% | model layer ops |
| TP all_reduce | ~1,800 | 2.86% | NCCL collective |
| 기타 (Python bookkeeping) | ~3,000 | ~5% | |

→ **pacpu kernel 자체 시간 < 1% (sample 안 잡힘)**. 우리 측정 환경에서 cdec dispatch 가 너무 적어 flamegraph 로 못 봄.

---

## D.4 bottleneck location map (pacpu kernel 단위)

| # | 위치 | file:line | 호출 path | 측정 ms/call (추정) | % of cdec_wait | 호출 빈도 | 현재 ISA |
|---|---|---|---|---:|---:|---|---|
| 1 | `qk_product` | `csrc/cpu/pacpu/pacpu.ispc:5` | attn_one_seq → ispc | ~3.5 ms / layer (추정) | ~40% | ~10 reqs × 80 layer / step | ISPC `avx512spr-x16` |
| 2 | `av_product` | `csrc/cpu/pacpu/pacpu.ispc:71` | attn_one_seq → ispc | ~3.5 ms / layer | ~40% | 동상 | ISPC `avx512spr-x16` |
| 3 | `softmax` | `csrc/cpu/pacpu/pacpu.ispc:109` | attn_one_seq → host | ~1.0 ms / layer | ~10% | 동상 | scalar / ISPC `foreach` |
| 4 | OMP team launch | `core.h:296` `omp parallel` | ispc_attention_tasks | ~0.5 ms / layer | ~5% | per layer | libgomp |
| 5 | OMP barrier × 2 | `core.h:314,333` | core.h | ~0.3 ms / layer | ~3% | per layer | libgomp |

→ cdec_wait 8.75 ms / layer × 80 layer × cdec freq 의 단순 budget. 실측 PROFILE 로그 (없음) 확인 필요.

---

## D.5 추가 CPU bottleneck (pacpu 외)

| # | 위치 | file:line | % of total CPU | 현재 ISA |
|---|---|---|---:|---|
| 6 | `_neo_handle_kv_swap` (swap-out + swap-in) | gpu_model_runner:6735 | 12% (worker) | Python + ATen |
| 7 | `at::native::index_kernel` (advanced indexing) | libtorch_cpu.so | 8% (omp_pool) | OMP + AVX (libtorch 내부) |
| 8 | `cudaEventSynchronize` (output sync) | libcudart.so | 18% (async output) | (GPU sync) |
| 9 | shm_broadcast dequeue + acquire_read | shm_broadcast.py | 27% (RPC) | Python + sched_yield |
| 10 | `forward_double` stream sync | sub_batch_executor.py:23x | (interleaved) | CUDA driver |
| 11 | TP all_reduce (NCCL) | NCCL | 2.86% | NCCL |
| 12 | Python overhead (closures, dict, layer dispatch) | llama.py:600-680 | ~3% | scalar |

---

## D.6 산술 강도 (FLOPs / byte) — pacpu 3 kernel

(Phase A.7 의 kernel signature map 인용)

| kernel | FLOPs / token | bytes / token | AI (FLOPs/byte) | roofline 적합 ISA |
|---|---:|---:|---:|---|
| `qk_product` | 32,768 | ~288 | **114 → effective 30-50** | AVX-512 / AMX BF16 |
| `av_product` | 2,048 | ~288 | **7** | memory-bound, AMX 큰 효과 X |
| `softmax` | ~30 (exp 비싸지만) | ~64 | **~0.5** | memory-bound, AVX-512 exp helper 유효 |

→ **qk_product**: compute-bound (AI 30-50, AMX/AVX-512 가속 효과 큼)
→ **av_product**: memory-BW bound (AI 7, AMX 효과 작음 — V cache 재사용 없음)
→ **softmax**: scalar bottleneck (exp), AVX-512 fast_exp 가속 유효

---

## D.7 Amdahl cap — cdec_executor `max_workers=2`

NEO 의 cdec executor 는 `max_workers=2` (SUB_023 settled, +4 시 -52% regression — layer dependency chain).

| 시나리오 | cdec layer 처리량 | NEO sub-batch[1] wall |
|---|---|---|
| 현재 (max_workers=2) | 2 cdec future 동시 처리 | bounded by 가장 느린 chain |
| AMX 시 (5× speedup, cap 동일) | wall time 4-5× 단축 | 그러나 GPU sub-batch[0] 의 wall 이 bottleneck 으로 전환 |

→ **AMX 가 cdec kernel 5× 가속 시**: cdec_wait 8.75 → 1.75 ms/layer (예상). GPU paged_attention 0.09 ms/layer 와 19× 격차 잔존. **여전히 GPU 가 빠름** → AMX 만으로 NEO sub-batch pipeline 의 GPU/CPU 비등 영역 도달 못 함.

---

## D.8 결론 — Phase D bottleneck mapping

### 핵심 fact

1. **pacpu kernel 자체 시간 < 1% (flamegraph 기준)** — cdec dispatch 빈도가 너무 낮음 (mirror=10/running=222, 4.5%)
2. **omp_pool 8.26% = ATen index_kernel 의 OMP** (swap-in path), pacpu 가 아님
3. **AMX/AVX 가속 후보 영역**:
   - **qk_product** (compute-bound, AMX BF16 가능 5-10× speedup 잠재)
   - **softmax** (AVX-512 fast_exp 가능, 작은 영역)
   - **swap path (`copy_layer_out` 의 ATen index_kernel)** — vllm 의 cpu_attn_amx 아닌 별도 영역 (8.26% 가 여기)
4. **AMX 만 도입 시 cdec_wait 8.75 → 1-2 ms/layer 추정** — GPU 0.09 ms 와 여전히 큰 격차. 단독으로 NEO net-win 영역 도달 불가
5. **NEO 의 sweet spot 도달은 workload 변경 (작은 batch, 긴 seq, mirror cap 60+) 동반 필요**

### 측정 미달 영역

- `cdec_wait` 의 내부 분해 (`qk_product` vs `av_product` vs `softmax` 별 시간) — PROFILE 로그 추가 필요
- AMX 가속의 정확한 speedup (BF16 cast cost 포함)
- Option C/L/M2 ON 시 (chain 99% 영역) 의 flamegraph 재측정

### Phase E 입력

위 fact + `A_kernel_signature_map.md` + `C_existing_paths_inventory.md` 결합하여 최종 bottleneck map + 적용 가능성 표 작성.
