# Phase D — ISA 가속 후보 long-list

> 분석 시각: KST 2026-05-15 ~
> 산출 목적: bottleneck mapping (D_bottleneck_table.md) + roofline (D_roofline_notes.md) 으로 식별된 모든 ISA 가속 가능 후보 위치를 우선순위 미부여 상태로 long-list 화. Phase E 의 입력.
> 우선순위 / apply 결정은 본 문서가 아님.

---

## D.16 long-list — 15 entries

| # | 영역 | 위치 (file:line) | 종류 | ISA 후보 | 현재 ISA | 기대 영역 |
|---|---|---|---|---|---|---|
| L01 | pacpu | `csrc/cpu/pacpu/pacpu.ispc:5` (`qk_product`) | GEMM (Q · K^T) | AMX-BF16 / AVX-512 BF16 | ISPC `avx512spr-x16` FP16 | qk_product 의 inner loop |
| L02 | pacpu | `csrc/cpu/pacpu/pacpu.ispc:71` (`av_product`) | GEMV (A · V) | AVX-512 BF16 (AMX 효과 적음) | ISPC `avx512spr-x16` FP16 | av_product 의 inner loop |
| L03 | pacpu | `csrc/cpu/pacpu/pacpu.ispc:109` (`softmax`) | exp + reduce | AVX-512 `fast_exp` | ISPC built-in exp | softmax exp 부분 |
| L04 | pacpu | `csrc/cpu/pacpu/pacpu.ispc:140-160` (`attn_one_seq`) | composition | (no compute, 그러나 layout setup) | scalar | dispatch / setup |
| L05 | pacpu | `csrc/cpu/pacpu/core.h:296-333` (OMP team) | OMP team launch + barrier | (ISA 미관련) — persistent OMP 도입 | libgomp default | OMP overhead 절감 |
| L06 | swap path | `vllm/v1/core/sched/neo_cpu_kv_buffer.py:464` (`copy_layer_out`) | tensor advanced indexing | ATen native (이미 AVX-512), 또는 NEO-side direct copy 로 ATen 우회 | ATen `index_kernel` AVX-512 | swap-in path 의 8.26% omp_pool |
| L07 | swap path | `gpu_model_runner.py:6735` (`_neo_handle_kv_swap`) | Python overhead + ATen dispatch | (ISA 미관련) — Python overhead 절감, batched copy | Python | swap orchestration |
| L08 | model forward | `vllm/model_executor/models/llama.py:636,422` (attention / neo_attention) | Python overhead | (ISA 미관련) | Python | attention dispatch hot path |
| L09 | model forward | `vllm/v1/attention/sub_batch_executor.py:23x` (`forward_double`) | CUDA stream sync | (ISA 미관련) | CUDA | sub_batch overlap |
| L10 | TP collective | NCCL all_reduce | NCCL | (NCCL 자체 AVX-512 가능) | NCCL | 2.86% |
| L11 | RPC | `vllm/v1/utils/shm_broadcast.py:674` (`acquire_read`), `:755` (`dequeue`) | sched_yield + spin | (ISA 미관련) | libc | 27% idle wait (낮은 가치) |
| L12 | output thread | `gpu_worker.py` async output (`cudaEventSynchronize`) | GPU sync | (ISA 미관련) | CUDA | 18% — GPU 대기로 줄이기 어려움 |
| L13 | linear (MoE 외) | `vllm/model_executor/layers/linear.py` (gate/up/down proj) | GEMM | AMX-BF16 (vllm dnnl backend) | cuBLAS GPU | GPU 영역 (CPU 가속 무관) |
| L14 | KV cache 변환 | `vllm/v1/worker/gpu_model_runner.py` cache fp8 quant | INT8 quant | AVX-512 VNNI | torch fp8 op | 적용 영역 작음 |
| L15 | pacpu | `csrc/cpu/pacpu/pacpu.ispc:5-100` (qk + av merged) | fused QK·softmax·AV | AMX + AVX-512 fused kernel | scalar split | end-to-end attention fusion |

---

## D.17 long-list 분류 (대분류)

### Category A — ISA 가속이 직접 적용 가능 (compute 영역)

- **L01** `qk_product` → **AMX-BF16 + AVX-512 BF16** (roofline 상 4-7× 실효, 영역 40% of cdec_wait)
- **L02** `av_product` → AVX-512 BF16 (memory-bound, AMX 효과 작음, 1.5-2×)
- **L03** `softmax` → AVX-512 `fast_exp` (2-6× 영역, scalar bottleneck)
- **L15** fused kernel → AMX + AVX-512 (큰 PR, 정확도 검증 필요)

### Category B — Layout / dispatch 변경으로 ISA 가속 가능

- **L04** `attn_one_seq` setup — kernel 외 시간을 줄여 AMX 효과 더 크게
- **L05** OMP team persistent — AMX 적용 후 OMP overhead 비중 증가 대응
- **L14** KV cache fp8 quant 영역 — AVX-512 VNNI (작은 영역)

### Category C — ISA 가속 무관 (Python / orchestration / wait)

- **L06** swap path `copy_layer_out` — ATen 이 이미 AVX-512 (8.26%), pacpu 가 아닌 이 영역 줄이는 것은 swap 알고리즘 변경 필요
- **L07** `_neo_handle_kv_swap` Python overhead — batched copy / 비동기 화
- **L08** llama.py attention dispatch — Python closure / dict 영역
- **L09** sub_batch_executor stream sync — overlap 알고리즘 변경
- **L11** RPC idle wait — sched_yield → busy spin 가능하나 CPU 과다
- **L12** async output thread — GPU 대기 영역, ISA 미관련

### Category D — 영역 외 (다른 PR 영역)

- **L10** NCCL — 외부 library
- **L13** linear layer (gate/up/down proj) — GPU 영역, CPU 가속 무관

---

## D.18 후보 → kernel 영역별 정량 추정 (long-list 기준)

| # | 위치 | 영역 % | 적용 ISA | 이론 speedup | 실효 speedup | wall 절감 (ms/layer 추정) |
|---|---|---:|---|---:|---:|---:|
| L01 | `qk_product` | 40% of cdec_wait (= 3.5 ms) | AMX-BF16 | 13-25× | 4-7× | 2.5-3.0 ms |
| L02 | `av_product` | 40% (3.5 ms) | AVX-512 BF16 | 1.5-2× (BW limit) | 1.2-1.5× | 0.6-1.2 ms |
| L03 | `softmax` | 10% (0.9 ms) | AVX-512 `fast_exp` | 2-6× | 2-3× | 0.4-0.5 ms |
| L04 | setup | 5% (0.4 ms) | - | - | - | ~0 |
| L05 | OMP overhead | 5% (0.4 ms) | persistent OMP | - | - | 0.2 ms |
| L06 | swap copy_layer_out | 8.26% (worker total) | layout 변경 | 1.5-2× | 1.2-1.5× | (1.0-2.0 ms swap path) |
| L15 | fused | (L01-L03 합) | AMX + AVX-512 | - | 5-8× | 5-6 ms (cdec_wait 8.75 → 2-3) |

→ pacpu 영역 적용 cap: cdec_wait **8.75 ms → 1.5-3 ms / layer** (AMX + AVX-512 fast_exp 합)

---

## D.19 추가 검토 항목 (gate 미충족)

본 long-list 의 정량 추정은 측정 보조 없이 산출. 추정의 신뢰도를 높이려면:

1. **PROFILE 로그 (`VLLM_NEO_PROFILE=1`) ON 상태 측정** — 3 kernel 별 ms/call 실측 (D.18 의 영역 % 검증)
2. **Option C / L / M2 ON + chain firing 80-99% 영역 flamegraph 재측정** — pacpu kernel 의 실측 시간 sample
3. **AMX BF16 dtype 변환 cost 측정** — pacpu FP16 → BF16 변환의 throughput / 정확도 영향 dev 검증
4. **OMP persistent overhead 측정** — `omp_set_dynamic(0)` + `omp_set_num_threads(N)` 도입 시 launch cost 감소량
5. **cdec_executor cap (max_workers=2) 의 layer 의존성 분석** — AMX 시 layer wall time 짧아지면 cap 의 영향 변화 정량

---

## D.20 long-list 결론

- **AMX 직접 적용 후보**: L01, L02 (L02 는 효과 작음), L15 (fused, 큰 PR)
- **AVX-512 적용 후보**: L01 (AMX 대안), L02 (memory-bound 영역), L03 (scalar exp 영역)
- **layout / dispatch 영역**: L04, L05, L07 (지원 작업)
- **ISA 미관련**: L08-L13 (대부분 큰 영역이나 ISA 가속 불가)

→ Phase E 에서 위 long-list 의 각 entry 를 (file:line + 영역 % + 가속 가능성 + 위험 요소) 표로 정리.
