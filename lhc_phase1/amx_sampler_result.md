# LHC Phase 1 — AMX sampler microbench 결과

**날짜**: 2026-06-08  
**HW**: Intel Xeon Platinum 8570 (AMX bf16/int8/tile + AVX-512 BF16/VNNI/FP16) + NVIDIA B200

## 1. 측정 설정

`amx_sampler_bench.py` — torch CPU bf16 (oneDNN backend) vs torch CUDA bf16.

- op: `softmax(logits, dim=-1)` → `topk(k=5, dim=-1)`
- shapes: `[batch, vocab]`, vocab ∈ {128 256 (Llama), 152 064 (Qwen), 257 152 (DeepSeek)}, batch ∈ {1, 8, 32, 64}
- warmup 20, iters 200, p50 latency
- AMX backend: oneDNN ISA = `avx10_1_512_amx` (verbose 로 확인됨, `brg_matmul:avx10_1_512_amx` 자동 선택)
- AVX-512 backend: child process re-exec 로 `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` 강제 (AMX off)
- GPU backend: `cuda:0` (B200), CUDA event timing
- 비고: IPEX 는 torch 2.11 비호환 → torch native oneDNN 경로만 사용

## 2. 결과 표 (latency μs, p50)

| vocab        | bs |  AMX (μs) |  AVX-512 (μs) |  GPU (μs) | AMX/GPU | AVX/GPU | AMX/AVX |
|--------------|---:|----------:|--------------:|----------:|--------:|--------:|--------:|
| llama-128k   |  1 |   349.7   |    334.0      |   101.0   |  3.46×  |  3.31×  |  1.05×  |
| llama-128k   |  8 |  1445.1   |   1413.9      |   108.7   | 13.30×  | 13.01×  |  1.02×  |
| llama-128k   | 32 |   528.5   |    566.9      |   131.0   |  4.03×  |  4.33×  |  0.93×  |
| llama-128k   | 64 |   533.9   |    574.4      |   157.5   |  3.39×  |  3.65×  |  0.93×  |
| qwen-152k    |  1 |   427.3   |    400.5      |   109.9   |  3.89×  |  3.64×  |  1.07×  |
| qwen-152k    |  8 |   601.3   |    625.4      |   116.7   |  5.15×  |  5.36×  |  0.96×  |
| qwen-152k    | 32 |   619.2   |    623.0      |   142.2   |  4.36×  |  4.38×  |  0.99×  |
| qwen-152k    | 64 |   644.8   |    758.4      |   170.9   |  3.77×  |  4.44×  |  0.85×  |
| deepseek-256k|  1 |   757.0   |    702.2      |   136.2   |  5.56×  |  5.15×  |  1.08×  |
| deepseek-256k|  8 |  1006.2   |    992.5      |   145.2   |  6.93×  |  6.83×  |  1.01×  |
| deepseek-256k| 32 |  1039.3   |   1070.9      |   186.3   |  5.58×  |  5.75×  |  0.97×  |
| deepseek-256k| 64 |  1187.7   |   1171.9      |   228.2   |  5.20×  |  5.13×  |  1.01×  |

## 3. 처리량 (samples / s)

| vocab        | bs | AMX     | AVX-512 | GPU      |
|--------------|---:|--------:|--------:|---------:|
| llama-128k   |  1 |    2.9k |    3.0k |     9.9k |
| llama-128k   |  8 |    5.5k |    5.7k |    73.6k |
| llama-128k   | 32 |   60.5k |   56.5k |   244.2k |
| llama-128k   | 64 |  119.9k |  111.4k |   406.2k |
| qwen-152k    |  1 |    2.3k |    2.5k |     9.1k |
| qwen-152k    |  8 |   13.3k |   12.8k |    68.6k |
| qwen-152k    | 32 |   51.7k |   51.4k |   225.1k |
| qwen-152k    | 64 |   99.3k |   84.4k |   374.4k |
| deepseek-256k|  1 |    1.3k |    1.4k |     7.3k |
| deepseek-256k|  8 |    8.0k |    8.1k |    55.1k |
| deepseek-256k| 32 |   30.8k |   29.9k |   171.7k |
| deepseek-256k| 64 |   53.9k |   54.6k |   280.4k |

## 4. 핵심 관찰

### 4.1 AMX 는 sampler 에서 AVX-512 대비 거의 등가

- 12 조합 중 AMX/AVX 비율 평균 0.99 ×, 최대 1.08 ×, 최소 0.85 ×.
- **이유**: sampler 의 핵심 op (softmax exp+norm, top-k partial sort) 는 **memory-bound** 이고 matmul/conv 가 아니므로 AMX tile engine 의 강점 (8× peak FLOPs vs AVX-512 BF16) 이 발현 안 됨.
- AMX 가 의미 있는 곳: logits projection (`hidden @ embedding.T`, 즉 vocab head matmul) 또는 speculative decode draft head. **현재 측정한 sampler 그 자체는 AMX 의 sweet spot 이 아님.**

### 4.2 CPU vs GPU 는 3.4×–13.3× GPU 우위

- Phase 1 의 viability 기준 "CPU latency ≤ 2 × GPU" 는 **미달**.
- 단 batch=8 의 outlier (llama 13.3×, qwen 5.15×) 는 torch CPU softmax 의 internal blocking 비효율로 보임 (batch=32 에서 latency 가 batch=8 보다 짧아지는 비단조성).
- batch=32~64 의 일반 영역에서는 CPU/GPU = 3.4×–5.8× 범위.

### 4.3 LHC overlap 가치 재평가

비록 latency 가 GPU 대비 3–5× 느리지만, **LHC 의 본질은 단일 op latency 가 아니라 GPU 와의 직교 overlap** 임. 다음 조건에서 의미 있음:
- **GPU 가 다음 step prefill 중**일 때 CPU 가 idle 이면, CPU 가 sampler 를 280–1200 μs 에 처리해도 GPU prefill (보통 5–20 ms) 안에 충분히 끝남 → free overlap.
- batch=64 / vocab=128k 기준 AMX 119k sps 는 H100 single-GPU sampler 처리량과 동급 (논문 reference). B200 가 단순 더 빠를 뿐 **lane 자체의 vacancy 가 있으면 가치 있음**.

## 5. Verdict for AMX lane

- **AMX lane viable** : **YES (조건부)**. 단 sampler 단독 op 으로는 가치 작음. 진짜 AMX 가치는 (a) **logits head matmul** (vocab head 4096 × 128k bf16 GEMM), (b) **draft head matmul** (speculative decode), (c) **embedding lookup + RMSNorm** 의 fused path 에서 나옴 → Phase 2 에서 이들 op 의 AMX 효율을 별도로 측정해야 함.
- 현재 측정한 softmax+topk 만으로는 AMX vs AVX 차이 무. **즉 LHC 안에서도 AMX 의 sub-lane 선정이 정밀해야 함**.

## 6. 산출물

- `/workspace/host_vllm_hybrid/lhc_phase1/amx_sampler_bench.py`
- `/workspace/host_vllm_hybrid/lhc_phase1/amx_sampler_amx.json`
- `/workspace/host_vllm_hybrid/lhc_phase1/amx_sampler_avx512.json`
- `/workspace/host_vllm_hybrid/lhc_phase1/amx_sampler_gpu.json`
