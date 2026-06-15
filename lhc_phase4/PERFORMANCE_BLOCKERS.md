# LHC 성능 향상을 차단한 기술적 요소 — 8 Conversation Cross-Evidence

> **결론**: 본 환경 (DGX B200 + Llama-3.1-70B TP=8 + real trace + conc=32 + max_tok=8192) 에서 vLLM 1.7 의 기존 최적화가 새 LHC algorithm 의 hot path 진입 가능 영역을 모두 선점하여, 추가 양수 lever 발굴 path 가 구조적으로 차단됨.

## 작성 일자
2026-06-09

## 시도 범위 (8 conversation 누적)
- 100+ host-side lever (env flag, NUMA, AMX, DSA, NEO, AGSD, MoE offload, MWOR 등)
- LHC 4 phase (옵션 A 정적, 옵션 C adaptive, misuse anti-pattern, conc=256 검증)
- TSK_042 5 model × 7 corpus real-trace 검증 (Llama-8B, Qwen-7B, DS-R1-Distill-Qwen-7B, Llama-70B, Qwen-32B)
- D-1 async pipelining
- B-1~B-7 sub-ideas (Python C API SIMD, adaptive cudagraph, prefill chunk, GPU stream priority, shared mem IPC, AVX-512 detok, adaptive max_num_seqs / KV SIMD / sampling SIMD / DSA prefetch)
- 새 algorithm 발명 (X: Workload-Adaptive Request Bundling, Y: Speculative Pre-prefill, Z: KV Block Coalescing)

---

## 1. 차단 요소 카테고리

### 1.1. GPU saturation 한계
**측정 결과**: 본 baseline 의 gpu_util **99.6%**, cpu_util 5.8%

| 차단 메커니즘 | 영향 |
|---|---|
| GPU 가 매 step 99.6% 사용 → host critical path 비중 **< 1%** | 어떤 host-side 가속도 wall-clock 안에서 차지 비중 작아 end-to-end 효과 noise floor 미만 |
| host gap 자체 짧음 (~ms 단위) → kernel 가속해도 amortize 안 됨 | C kernel 10× 가속해도 step 전체에서 < 1% 향상 (Path 1, B-1 reject 근거) |

### 1.2. vLLM-native 기능 중복 (이미 차지함)

| 시도 algorithm | vLLM-native 중복 메커니즘 | 코드 위치 |
|---|---|---|
| **D-1 async pipelining** | `async_scheduling` default True + `max_concurrent_batches=2` + `step_with_batch_queue` (prev update + next schedule overlap) | `vllm/config/vllm.py:789-832`<br>`vllm/v1/executor/multiproc_executor.py:476-487`<br>`vllm/v1/engine/core.py:240-271` |
| **X Workload-Adaptive Request Bundling** | **Cascade attention** 자동 활성화 (`num_common_prefix_blocks > 0` 시 모든 main backend) + prefix cache hit 자동 dedup + L10 burst-aware SJF reorder | `vllm/v1/core/sched/scheduler.py:1155-1163, 2087-2130`<br>`vllm/v1/attention/backends/flash_attn.py:482`<br>`vllm/v1/attention/backends/flashinfer.py:901`<br>`vllm/v1/attention/backends/triton_attn.py:226`<br>`vllm/v1/core/block_pool.py` BlockHashToBlockMap |
| **B-5 Multi-instance shared memory IPC** | `shm_broadcast.py::ShmRingBuffer` + `MessageQueue` + `SingleWriterShmRingBuffer` 정확히 동일 메커니즘 | `vllm/distributed/device_communicators/shm_broadcast.py` |
| **B-6 AVX-512 detokenize batched** | HF tokenizers 가 native Rust 구현 + asyncio output_handler 분리 | `tokenizers/` PyO3 binding |
| **B-4 GPU stream priority** | `_hwc1_make_stream` 등 보조 stream 분리 + priority 인프라 이미 native | `vllm/v1/worker/gpu_model_runner.py` |
| **B-2 Adaptive cudagraph mode** | `FULL_AND_PIECEWISE` 가 이미 batch 별 dispatch | `cudagraph_dispatcher.py:142` |

### 1.3. vLLM 구조적 제약 (architectural blockers)

| 시도 algorithm | 구조적 차단 사유 |
|---|---|
| **B-2 Adaptive cudagraph mode** | cuda graph capture 는 **startup-only** — runtime 동적 재캡처 불가 (모든 capture key (max_num_seqs, max_num_batched_tokens, cudagraph_mode) 변경 불가) |
| **B-3 Prefill chunk adaptive** | `max_num_batched_tokens` 가 cuda graph capture key — runtime 변경 시 graph mismatch |
| **B-7a Adaptive max_num_seqs** | 동일 사유 (graph capture size) |
| **Z KV Block Coalescing** | paged attention 정의상 fragmentation **부재** — block_id 는 단순 인덱스, attention 은 block_table indirection (random gather), 물리적 인접성 무의미 |
| **Z KV Block Coalescing (DSA path)** | DSA 는 host-DRAM 영역 copy engine 만 — GPU HBM 직접 조작 불가. block move + remap 시도 시 모든 running request block_table race → correctness 위험 |

### 1.4. Workload 통계 한계

| 시도 algorithm | workload 통계 한계 |
|---|---|
| **Y Speculative Future-Request Pre-prefill** | 실 workload `sampled_prompts.parquet` (2159 prompts × 6 corpora) prefix 200자 unique 비율 **sharegpt 99.8% / swebench 100% / humaneval 100% / mbpp 100% / wildchat 95% / lmsys 94%** — system prompt 반복 거의 없음 → proactive prefill hit rate ≤ 5%, 잘못 prefill 시 KV pool 낭비 음수 가능 |
| **간이 vs 전체 실험 차이** | conc=256 + synthetic rust/json (반복 boilerplate) 에서 LHC Path 1 +19% 발견 → conc=32 + real trace (high prompt diversity) 에서 net-neutral. **prefix-cache hit + hash chain hot path 비중** 이 host 가속 효과를 결정 |

### 1.5. ctypes / Python ↔ C dispatch overhead

| 측정 진단 | 영향 |
|---|---|
| Python ↔ C ctypes dispatch ~**1.5μs/call** | hash chain payload 자체 (~0.2μs) 보다 dispatch 가 큼 → C kernel 5× 가속해도 ctypes dispatch 가 dominate |
| **B-1 Python C API direct extension** 으로 우회 가능하나 hot path 비중 < 1% 면 의미 없음 | gpu_util 99.6% 환경의 hot path 비중 자체가 noise floor 미만 |

---

## 2. 통합 결론

본 환경 (DGX B200 + Llama-3.1-70B TP=8 + real trace + conc=32 + max_tok=8192) 에서 LHC 기반 새 algorithm 양수 path 가 차단된 진짜 원인:

1. **vLLM 1.7 이 이미 host-side 양수 lever 전부 차지** — async scheduling, cascade attention, prefix-caching dedup, ShmRingBuffer IPC, HF Rust tokenizers, FULL_AND_PIECEWISE dispatch, burst-aware SJF reorder, B3 FaP, async_scheduling, suffix decode, fp8 KV 등
2. **GPU saturated 99.6%** → host hot path 비중 < 1%, host 가속의 end-to-end 효과 noise floor 미만
3. **cuda graph startup-only** → runtime adaptive 변경 불가 (cudagraph_mode, max_num_seqs, max_num_batched_tokens 모두 capture key)
4. **paged attention 의 random-gather 모델** → KV 의 물리적 인접성 무의미, fragmentation 개념 부재 → DSA defrag 불가
5. **DSA 가 host-DRAM copy only** → GPU HBM 영역 직접 조작 불가
6. **ctypes overhead 1.5μs** → hash chain 같은 작은 payload 의 host hook 효과 차단
7. **real workload prefix entropy 매우 높음** (≥94% unique) → proactive prefill 무효
8. **vLLM-native cascade attention** 이 동일 prefix sequence 의 attention compute 공유 자동 처리 → bundling 알고리즘 무효

## 3. cross-evidence 결정 (8 conversation 통합)

| 시도 영역 | 결과 |
|---|---|
| LHC infrastructure (DSA + AMX + WQ-per-rank + NEO OOB) | 인프라 자체는 성공 (DSA aggregate 56.88 GB/s, AMX C3 prefix scan 2.04× GPU latency) |
| LHC Path 1 conc=256 narrow regime | rust/json corpus 에서 **+19% 양수** (간이 환경) |
| LHC Path 1 real-trace 5 model × 7 corpus | **net-neutral cross-model 일관** (Llama-8B, Qwen-7B, DS-R1-Distill-Qwen-7B, Llama-70B, Qwen-32B 모두 |Δ| ≤ 1% mean) |
| D-1 async pipelining | vLLM-native 와 중복 (실측 mean -0.47%) |
| B-1~B-7 sub-ideas | Phase 1 사전 분석 전부 reject (vLLM-native 중복 또는 구조적 차단) |
| 새 algorithm X/Y/Z | Phase 1 사전 분석 전부 reject (cascade attention / workload entropy / paged 구조) |

## 4. 결과의 paper 가치

본 결과는 **measured-negative finding** 으로서 paper §discussion 의 contribution 가능:
1. vLLM 1.7 의 host-side 최적화가 이미 매우 잘 진행되어 새 host algorithm 진입 path 가 좁음을 정량적으로 입증
2. LHC infrastructure (DSA + AMX) 자체는 정상 작동 (literature 0건 → 최초 적용)
3. LHC 가 가치 발현 가능한 narrow regime 식별 (conc=256 + 짧은 prompt + prefix-cache hit ≥ 60% + boilerplate-heavy corpus)
4. 다른 환경 (smaller GPU, multi-tenant, multi-instance) 에 LHC 가 가치 가능성

## 5. 권장 다음 path

| 옵션 | 내용 | 가능성 |
|---|---|---|
| paper 최종 정리 + commit | 8 conversation cross-evidence 통합 + honest discussion | 가장 honest 마무리 |
| smaller GPU regime | A100 40GB / T4 24GB → KV pool 자연 압박 | hw 변경 필요 |
| multi-tenant / multi-instance | KV churn 발생 | 시스템 architecture 변경 |
| GPU-side 최적화 (LHC 외) | sm_100 native FP4, custom flashinfer fork | non-LHC 방향 |
| LHC 종결 선언 | 명시 종료 | — |

---

## 6. 산출물 위치
- `/workspace/host_vllm_hybrid/lhc_phase4/` — 모든 LHC 측정 데이터
- `/workspace/host_vllm_hybrid/vllm/v1/lhc/` — LHC 인프라 코드 (production C kernel + Python wrapper + regime detector)
- `/workspace/host_vllm_hybrid/paper/` — paper draft (LHC contribution + measured-negative finding 통합)
- `/workspace/host_vllm_hybrid/lhc_phase4/tsk042_10model_unified/` — TSK_042 real-trace 5 model × 7 corpus baseline + lhc_path1 측정 데이터

## 7. 본 문서 작성 후 task 종료
8 conversation cross-evidence 가 본 환경에서 LHC 양수 path 부재를 결정적으로 증명. 본 문서로 기술적 차단 요소를 정리하여 paper §discussion 통합 자료로 활용 가능. **LHC algorithm 탐색 task 종료**.
