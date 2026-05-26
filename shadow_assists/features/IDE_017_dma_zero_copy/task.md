# task.md — IDE_017 단계별 구현

## TSK_028 — Pinned memory pool + DMA push primitive

### Step 1: size-class allocator 설계
- src/dma_pool/pinned_pool.cpp
- size classes: 4KB / 64KB / 1MB / 16MB / 64MB (SUB_166 의 측정 region 매핑)
- free-list per class + 전체 max pool size (e.g., 4 GB)
- thread-safe (lockless ring buffer 권장)

### Step 2: DMA push API
- `cudaMemcpyAsync` wrapper + per-stream event
- batching: 작은 chunk 여러 개를 같은 stream 에 묶어 fixed overhead 분산

### Step 3: vLLM integration
- vllm/v1/worker/kv_cache_manager.py 의 GPU↔CPU swap 경로에 hook
- ENV `VLLM_USE_DMA_POOL=1` 으로 activate

### Step 4: measurement
- SUB_166 protocol 재현 (block size sweep)
- canonical AGSD-gated 통합 후 +1-2% throughput lift (가설)

## TSK_029 — Zero-copy CPU compute path

### Step 1: candidate identification
- spec candidate IDs (~2-8 KB per batch)
- attention bias (~1-100 KB)
- draft logits (~1-2 MB)
→ small data 들은 zero-copy 적합 candidate

### Step 2: pinned buffer dual-access 구현
- `cudaHostAllocMapped` flag
- GPU side: device pointer via `cudaHostGetDevicePointer`
- CPU side: direct read/write
- coherence: explicit `cuMemHostSync` 또는 weakly-coherent (paper 시 측정)

### Step 3: measurement vs cudaMemcpy round-trip
- microbench: zero-copy read latency vs DMA push + GPU read
- target: small data (< 256 KB) 영역에서 zero-copy 가 net positive

## TSK_030 — Cold-KV decompress + DMA push (IDE_006 재정의)

### Step 1: cold KV detection threshold
- KV cache 의 token age + access frequency tracking
- threshold rule: 50+ step ago && access < 5 in last 100 steps

### Step 2: CPU AVX-512 decompress kernel
- src/dma_pool/cold_kv_decompress.cpp
- INT8/INT4 → BF16 dequant (AVX-512 VNNI + cast)
- AVX-512 batch decompress: 32 vocab × 1024 tokens / cycle

### Step 3: DMA push integration
- decompress → pinned buffer → DMA push to GPU
- pipeline: stage 1 (decompress) overlap stage 2 (DMA)

### Step 4: TTFT impact measurement
- canonical: SUB_098 protocol + long context workload (e.g., 32K context)
- target: TTFT −5-10% (저빈도 cold-KV 영역만 quantize, frequent KV 는 BF16 유지)
