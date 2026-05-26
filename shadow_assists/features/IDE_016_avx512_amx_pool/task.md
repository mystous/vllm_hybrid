# task.md — IDE_016 단계별 구현

## TSK_024 — AVX-512 batch tokenizer / detokenizer

### Step 1: profile current tokenizer (SUB_161 이미 완료)
- vLLM 의 detokenize p50 측정 → 본 RESULTS 참조
- GIL contention 정량 (multi-thread 일 때 thread blocking 시간)

### Step 2: AVX-512 batch BPE search 구현
- src/avx512_sampling/batch_bpe.cpp
- 32 sequence × 152K vocab → 16-way SIMD search
- target: p50 −40% vs current Python tokenizer

### Step 3: integration on canonical
- vllm/v1/engine/processor.py 의 detokenize 경로에 hook
- ENV `VLLM_USE_AVX512_TOKENIZER=1` 으로 enable

## TSK_025 — AVX-512 sampling + logit processor ★★★

### Step 1: top-k/top-p AVX-512 vectorize
- src/avx512_sampling/topk_topp_kernel.cpp
- partial sort with `_mm512_mask_compress_ps` + threshold scan
- vocab 152K × batch 32 → 32-wide SIMD reduction
- correctness: greedy choice + sampled distribution 비교 vs PyTorch baseline

### Step 2: logit bias + temperature + repetition penalty vectorize
- src/avx512_sampling/logit_processor.cpp
- vectorize chain: logits → bias add → temp divide → penalty multiply
- in-place operation, no intermediate tensor

### Step 3: integration vs vLLM sampler.py
- 패치 위치: `vllm/v1/sample/sampler.py:3521 _sample`
- `if envs.VLLM_USE_AVX512_SAMPLING` → C++ kernel call
- correctness: per-token logprob max abs diff < 1e-3
- latency target: 3-5 ms → 1.5 ms / step (target 60-70% 감소)

### Step 4: measurement on canonical
- 500p × balanced × 3 mix
- target: AGSD-gated +5-10% throughput Δ (TSK_025 alone)
- util 캡처 (monitor.py)

## TSK_026 — AMX tile-based draft head matmul ★★

### Step 1: AMX tile config 설계
- src/amx_matmul/tile_config.cpp
- Qwen 0.5B (hidden 896, intermediate 4864) → tile descriptor
- Qwen 1.5B (hidden 1536, intermediate 8960) → tile descriptor

### Step 2: AMX kernel intrinsic 구현
- src/amx_matmul/amx_qwen_draft.cpp
- `_tile_loadd`, `_tile_dpbf16ps`, `_tile_stored`
- BF16 matmul: A[M,K] × B[K,N] → C[M,N]
- libxsmm 참고 (reference implementation)

### Step 3: PyTorch CPU matmul baseline 대비 측정
- microbench: tile size sweep
- target: ≥3× speedup vs torch.matmul(cpu)

### Step 4: canonical integration (placeholder — IDE_019 의 TSK_036 에서 사용)

## TSK_027 — AMX medium-context CPU prefill assist

### Step 1: theory + microbench
- 512-2K context 의 GPU prefill compute-bound 정량
- CPU AMX prefill 의 theoretical bound

### Step 2: async CPU AMX prefill thread 구현
- src/amx_matmul/async_prefill_thread.cpp
- producer/consumer ring buffer
- H2D pipeline (DMA push from IDE_017)

### Step 3: TTFT measurement
- target: −15% on 1K context
- canonical: SUB_098 protocol + medium-context workload mix
