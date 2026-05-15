# Phase E — AMX / AVX-512 적용 가능성 표 (최종 표 2)

> 분석 시각: KST 2026-05-15 ~
> 산출 목적: `E_bottleneck_map.md` 의 모든 row (BM01-BM22) 에 대해 — **AMX 적용 가능 (Y / N / 조건부) + 근거**, **AVX-512 intrinsic 적용 가능 (Y / N / 조건부) + 근거**, **재사용 가능 자산**, **예상 speedup × wall 절감**, **위험 요소** 를 정량/정성 정리.
> 입력: D_bottleneck_table.md, D_roofline_notes.md, D_candidate_long_list.md, C_existing_paths_inventory.md, C_pacpu_vs_cpu_attn_amx_gap.md
> **본 문서는 결정 문서 아님**. 각 위치의 적용 가능성 + 근거만 명시.

---

## E.6 가능성 표

각 row 의 의미:
- **AMX**: Y = 직접 적용 가능 / 조건부 = layout·dtype 변환 cost 동반 / N = 적용 불가
- **AVX-512**: Y = intrinsic 직접 적용 가능 / 조건부 = ISPC 가 이미 lower / N = ISA 미관련 영역
- **재사용**: 기존 vllm 자산 활용 가능 (file:line + 명시)
- **위험**: dtype precision drop, build complexity, dev 검증 불가 등

| ID | 위치 | AMX 가능 | AMX 근거 | AVX-512 가능 | AVX-512 근거 | 재사용 자산 | 예상 speedup | wall 절감 (ms/step 추정) | 위험 |
|---|---|:-:|---|:-:|---|---|---:|---:|---|
| BM01 | `qk_product` (`pacpu.ispc:5`) | **Y** | M=8 (tile rows 16 의 1/2 — 패딩으로 fit), N=16 (BLOCK_SIZE), K=128 (HEAD_DIM) — AMX-TMUL 의 tile dim 정합. BF16 변환 필요 | **Y** | AVX-512 BF16 `vdpbf16ps` 직접 적용 가능. 현 ISPC 가 FP16 FMA 로 lower 중 — BF16 intrinsic 명시 시 + FMA throughput 2× | `csrc/cpu/micro_gemm/cpu_micro_gemm_amx.hpp` `TileGemm224` template ◎ | AMX 4-7× (실효), AVX-512 BF16 1.5-2× | 2.5-3.0 ms / layer × 80 = **200-240 ms / step** | FP16 → BF16 변환 정확도 검증 (v1.1 SUB_006 v42 가 BF16 manual kernel 로 -3.16% 회귀). dev (i9-12900KF) AMX 미지원 → cross-compile + sim |
| BM02 | `av_product` (`pacpu.ispc:71`) | **조건부** | M=8, N=128 (HEAD_DIM), K=seq_len — K dim 큼. memory-bound (AI=7) → AMX 효과 작음 | **Y** | BF16 FMA 적용 가능. memory BW 한계로 1.2-1.5× | `cpu_micro_gemm_amx.hpp` (조건부) | AMX 1.5×, AVX-512 BF16 1.2× | 0.6-1.2 ms / layer × 80 = **50-100 ms / step** | BW-bound 영역 — 효과 작음. 단독 적용 가치 낮음 |
| BM03 | `softmax` (`pacpu.ispc:109`) | **N** | GEMM 아님. AMX 의 적용 영역 외 | **Y** | `cpu_arch_macros.h` 의 `_mm512_fast_exp_ps` 직접 ports | `csrc/cpu/cpu_arch_macros.h` `fast_exp` ◎ | 2-3× (exp 단축) | 0.4-0.5 ms / layer × 80 = **30-40 ms / step** | polynomial 5-degree 정확도 (FP32 ~1e-6) — 분포 유사성 영향 작음 |
| BM04 | `attn_one_seq` (`pacpu.ispc:142`) | N | composition only, compute 영역 외 | N | ISA 미관련 (setup) | - | - | ~0 | - |
| BM05 | OMP team (`core.h:296`) | N | ISA 영역 외 — thread 관리 영역 | N | OMP runtime 영역 | persistent OMP team 도입 (libgomp `omp_set_dynamic(0)`) | OMP overhead 30-50% 감소 | 0.2 ms / layer × 80 = **15 ms / step** | OMP team persistent 화는 thread affinity 충돌 가능 |
| BM06 | omp barrier (`core.h:314,333`) | N | 동상 | N | 동상 | barrier 제거 또는 last-iter overlap | barrier wait 절감 | 0.1 ms / layer × 80 = **8 ms / step** | algorithm 변경 필요 (loop fusion) |
| BM07 | swap `copy_layer_out` (`neo_cpu_kv_buffer.py:464`) | N | ATen advanced indexing 영역, AMX 미관련 | **조건부** | ATen 의 `index_kernel` 이 이미 AVX-512 use. **직접 copy** 로 ATen 우회하면 AVX-512 NT-store 명시 가능 | 직접 `memcpy` 또는 cpu_types_x86 의 VEC store | 1.2-1.5× (NT-store + alignment) | (swap path 영역) **80-120 ms / step** | NT-store cache pollution 우려, alignment 요구 |
| BM08 | `_neo_handle_kv_swap` (`gpu_model_runner.py:6735`) | N | Python orchestration | N | Python 영역 | batched swap (per-step) | Python overhead 절감 | (orchestration 영역) | swap 알고리즘 변경 영향 |
| BM09 | `_neo_swap_in_one_req` | N | Python | N | Python | 동상 | 동상 | 동상 | 동상 |
| BM10 | `cdec_future.submit` (`attention.py:1014`) | N | Python ThreadPoolExecutor | N | Python | ThreadPool batched submit | dispatch overhead 절감 | 작음 | - |
| BM11 | `attention` (`llama.py:636`) | N | Python | N | Python | - | - | - | - |
| BM12 | `neo_attention` (`llama.py:422`) | N | Python | N | Python | - | - | - | - |
| BM13 | `forward_double` (`sub_batch_executor.py`) | N | CUDA stream sync | N | CUDA 영역 | - | - | - | - |
| BM14 | `cudaEventSynchronize` | N | GPU wait | N | GPU wait | - | - | - | - |
| BM15 | `dequeue` (`shm_broadcast.py:755`) | N | sched_yield | N | RPC wait | busy-spin 가능 (CPU 과다) | (조건부) | 작음 | CPU 과부하 |
| BM16 | `acquire_read` (`shm_broadcast.py:674`) | N | 동상 | N | 동상 | 동상 | 동상 | 동상 | 동상 |
| BM17 | `execute_model` (`gpu_model_runner.py:4587`) | N | Python | N | Python | - | - | - | - |
| BM18 | `forward_neo_pipelined` (`llama.py:662`) | N | Python | N | Python | - | - | - | - |
| BM19 | `_forward_pipeline_inner` (`sub_batch_executor.py`) | N | Python | N | Python | - | - | - | - |
| BM20 | NCCL all_reduce | N | NCCL 외부 | N | NCCL 자체 AVX-512 자동 | NCCL 자체 | - | - | NCCL 영역 |
| BM21 | ATen `index_kernel` (libtorch) | N | libtorch 영역 | (이미 적용) | libtorch 가 이미 AVX-512 use | - | 직접 copy 도입 시 BM07 와 합산 | (BM07 와 합산) | libtorch 우회 영역 |
| BM22 | Python closures / dict | N | interpreter | N | interpreter | Cython / native compile | 작음 | - | 큰 PR |

---

## E.7 가능성 표 — 통계 요약

### AMX 적용 가능 영역

| 분류 | row 수 | 합산 wall 절감 (ms/step 추정) |
|---|---:|---:|
| **AMX Y (직접)** | **1** (BM01) | 200-240 ms |
| **AMX 조건부** | **1** (BM02) | 50-100 ms |
| **AMX N** | 20 | - |

### AVX-512 적용 가능 영역

| 분류 | row 수 | 합산 wall 절감 (ms/step 추정) |
|---|---:|---:|
| **AVX-512 Y (직접)** | **3** (BM01, BM02, BM03) | 280-380 ms |
| **AVX-512 조건부** | **1** (BM07) | 80-120 ms |
| **AVX-512 N** | 18 | - |

### 둘 다 불가능 영역

- Python orchestration (BM08-BM12, BM17-BM19, BM22): **9 row**
- GPU/RPC wait (BM13-BM16, BM20): **5 row**
- 영역 외 또는 의미 없음 (BM04, BM21): **2 row**

---

## E.8 합산 wall 절감 추정 (총)

`AMX (BM01) + AVX-512 (BM02 + BM03)` 조합 적용 시:
- BM01 AMX: 200-240 ms
- BM02 AVX-512: 50-100 ms
- BM03 AVX-512 fast_exp: 30-40 ms
- BM05 OMP persistent: 15 ms
- 합: **~300-400 ms / step 절감**

NEO v1.6 현재 step wall ≈ 1882s / step count. step count = 8 worker × (500p × 8192 token / batch). 정확한 step 수 미산정 — 추정으로 **throughput 4-10% 향상 영역**.

`AMX (BM01) + 추가 swap copy (BM07) + AVX-512 fast_exp (BM03)` 조합 시:
- 위 합 + BM07 = **~380-520 ms / step 절감**
- **throughput 5-12% 향상 영역**

→ **paper claim H100 14% gain 의 60-80% 영역 도달 가능** (workload 동일). workload 조정 (작은 batch + 긴 seq + mirror cap 60-80) 동반 시 14% 이상 도달 가능.

---

## E.9 위험 요소 종합

| 위험 | 영향 영역 | 완화 방안 |
|---|---|---|
| **FP16 → BF16 정확도 drop** | BM01, BM02 (pacpu 의 K/V cache dtype 변환) | v1.1 SUB_006 v42 의 -3.16% 회귀 이미 관찰. 분포 유사성 게이트 (per-token logprob max abs diff) 로 검증. BF16 + FP32 accumulator 유지 |
| **AMX dev 검증 불가** | BM01 | i9-12900KF 미지원. Intel SDE simulator + cross-compile 후 prod 직접 측정 |
| **build complexity** | BM01-BM03 | `-mamx-tile -mamx-bf16 -mavx512bf16` flag. TSK_003 의 precedent (`partial_attention_amx.cpp`) 활용 가능 |
| **OMP team persistent 충돌** | BM05 | thread affinity (taskset) 와 OMP team 의 상호작용 검증 필요 |
| **Polynomial fast_exp 정확도** | BM03 | 5-degree polynomial FP32 ~1e-6, softmax 의 attention score 영역에서 영향 미미 |
| **NT-store cache pollution** | BM07 | swap-in 직후 다른 영역에서 cache miss 가능. alignment 강제 |
| **cdec_executor cap (max_workers=2)** | BM01-BM03 가속 시 Amdahl cap | AMX 적용 후 layer wall 짧아짐 → cap 영향 변화 측정 필요 |

---

## E.10 재사용 자산 매핑 (Phase C 의 inventory 와 cross-ref)

| 자산 | 위치 | 적용 row | 재사용 난이도 |
|---|---|---|---|
| `csrc/cpu/micro_gemm/cpu_micro_gemm_amx.hpp` (`TileGemm224`) | `csrc/cpu/micro_gemm/` | BM01, BM02 | ◎ (BF16 변환만) |
| `csrc/cpu/cpu_arch_macros.h` (`_mm512_fast_exp_ps`) | `csrc/cpu/` | BM03 | ◎ (직접 include + call) |
| `csrc/cpu/cpu_attn_amx.hpp` `TileGemm224` | `csrc/cpu/` | BM01 (alternative) | △ (layout + API 재포장) |
| `csrc/cpu/dnnl_kernels.cpp` `onednn_mm` | `csrc/cpu/` | BM01 (alternative) | △ (layout 변환 cost) |
| `csrc/cpu/cpu_types_x86.hpp` (VEC type abstraction) | `csrc/cpu/` | BM07 (직접 copy) | ○ |
| TSK_003 `partial_attention_amx.cpp` | (위치 미상, prod 152 PASS) | BM01-BM03 | ⏳ (위치 확인 필요) |

---

## E.11 결론 — 적용 가능성 표 정리

### Category A — AMX 직접 적용 가능 (단일 위치)

- **BM01 `qk_product`**: AMX-BF16 직접 적용. 실효 4-7× speedup. wall 절감 200-240 ms/step. 재사용 `cpu_micro_gemm_amx.hpp::TileGemm224`. 위험: FP16→BF16 정확도.

### Category B — AVX-512 intrinsic 직접 적용 (단일 위치)

- **BM01 `qk_product`**: AVX-512 BF16 `vdpbf16ps` 적용. 1.5-2× speedup (AMX 대안). 영역 동일.
- **BM02 `av_product`**: AVX-512 BF16 적용. 1.2-1.5× (BW-bound 영역).
- **BM03 `softmax`**: AVX-512 `fast_exp` 적용. 2-3× speedup. wall 절감 30-40 ms/step. 재사용 `cpu_arch_macros.h`.

### Category C — layout / dispatch 변경 동반 (조건부)

- **BM07 swap copy_layer_out**: AVX-512 NT-store 도입 시 1.2-1.5×. swap path 영역.
- **BM05 OMP team persistent**: ISA 가속 후 overhead 비중 증가 대응.

### Category D — 둘 다 불가능 (16 row)

- Python orchestration / GPU wait / RPC wait — ISA 가속 영역 외.

### 종합 수치

- **AMX 적용 가능 위치 = 1** (BM01, 직접) + **1** (BM02, 조건부) = **2 row**
- **AVX-512 적용 가능 위치 = 3** (BM01-BM03, 직접) + **1** (BM07, 조건부) = **4 row**
- **둘 다 불가능 위치 = 16 row** (Python / GPU / RPC / NCCL)

→ 본 표는 "어디에 적용 가능한가" 의 fact 만 제공. **적용 결정은 후속 작업**.
