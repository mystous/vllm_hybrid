# CLAUDE.md — IDE_017 구현 시 알아야 할 것

## 0. 핵심 규칙
- C++ + CUDA, 빌드 시 `nvcc -arch=sm_90` (H100) 또는 호환
- SUB_166 측정 결과를 input 으로 사용 (35 μs overhead, 1 MB crossover, 54 GB/s asymptotic)
- physical core 0-111, HT 시블링 금지, 최대 100 core
- 시간 KST 표시
- 측정 1-run default
- commit/push 명시 지시 시만

## 1. SUB_166 측정의 의미

| 데이터 | 본 IDE 의 lever |
|---|---|
| 4-256 KB overhead-bound | DMA batching 필요 — 작은 데이터는 cudaMemcpy 와 차이 없음 |
| 1-4 MB crossover | DMA 시작 의미 있는 region |
| 16-64 MB asymptotic | cold-KV chunk 등 large data 의 main target |
| 35 μs fixed overhead | per-transfer cost — DMA call 의 lower bound latency |

## 2. NUMA affinity 주의 (SUB_113)

- GPU 0-3 ↔ NUMA 0 / GPU 4-7 ↔ NUMA 1
- cross-NUMA DMA 가능하지만 추가 latency 발생 가능
- pinned alloc 시 `numa_alloc_onnode` 또는 `cudaMemAdvise(MEM_LOCATION_PREFERRED)` 권장

## 3. 통합 위치 (vLLM)

- KV cache: `vllm/v1/worker/kv_cache_manager.py` 의 GPU/CPU swap 경로
- speculative path: `vllm/v1/spec_decode/` 의 candidate transfer
- LMCache 와 충돌 가능 — `VLLM_USE_DMA_ZERO_COPY=1` ENV 로 plugin 활성화

## 4. 알려진 risk + fallback

| risk | fallback |
|---|---|
| pinned memory 가 host RAM 압박 (typically 8-32 GB 영역) | size-class pool + total limit (예: 4 GB pool) |
| zero-copy 가 GPU L2 cache 무력화 | 작은 hot data 만 zero-copy, 큰 cold data 는 DMA push |
| TSK_030 cold-KV decompress 정확도 위반 | AVX-512 dequant kernel 의 numerical validation 필요 |
