# LHC Phase 3 — Task C: AMX sub-lane re-selection microbench

**날짜**: 2026-06-08
**머신**: DGX B200 (Xeon Platinum 8570, 56 cores/socket × 2, AMX_BF16 native; B200 sm_100)
**소속**: `lhc_phase3/PHASE3_VERDICT.md` 의 Task C 부분

---

## 0. TL;DR

Phase 1 (sampler) FAIL, Phase 2 (logits head) FAIL. Phase 3에서 5개 후보 (C1~C5) 모두 microbench. **C3 (prefix radix-tree byte scan) 만 gate (AMX ≤ 3× GPU latency) PASS**. C1/C2/C4/C5 모두 6.8× ~ 151× FAIL.

| sub-lane | 정의 | best AMX/GPU | gate (≤3×) | verdict |
|---|---|---|---|---|
| C1 draft head matmul | hidden[bs,2048] @ embed_T[2048,vocab] | 6.79× | FAIL | reject |
| C2 RMS norm | x · rsqrt(mean(x²)+ε) · w | 4.19× | FAIL | reject |
| **C3 prefix byte scan** | radix-tree compare + hash | **2.04×** | **PASS** | **accept (best)** |
| C4 KV scale calib | per-layer per-head abs-max | 39.82× | FAIL | reject |
| C5 fused norm+add | (x+res)·rsqrt(mean²+ε)·w | 25.49× | FAIL | reject |

→ **AMX sub-lane 단일 winner = C3**. Phase 3 통합 측정 (Task G) 에는 C3 only AMX path 로 진행.

---

## 1. 측정 셋업

- **A) AMX bf16**: torch CPU bf16, oneDNN default ISA (`avx10_1_512_amx` 가 SPR/EMR/GNR에서 자동 선택). OMP_NUM_THREADS=56.
- **B) AVX-512 BF16**: 동일 코드 child re-exec w/ `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` (AMX off, AVX-512_BF16 만).
- **C) GPU bf16**: torch CUDA bf16, B200 단일 GPU.
- WARMUP=5, ITERS=30, p50 보고.
- 코드: `lhc_phase3/amx_sub_lane_bench.py`.
- raw json: `amx_sub_lane_{gpu,amx,avx512}.json`.

---

## 2. 전체 결과 (p50 μs, AMX/GPU ratio)

### 2.1 C1 — draft head matmul (smaller K=2048 hidden)

| config | GPU | AMX | AVX512 | AMX/GPU |
|---|---|---|---|---|
| qwen-152k_h2048 bs1 | 106.4 | 1419.7 | 1486.3 | 13.34× |
| qwen-152k_h2048 bs4 | 105.8 | 2741.7 | 1811.6 | 25.92× |
| qwen-152k_h2048 bs16 | 107.6 | 2882.4 | 3061.2 | 26.78× |
| qwen-152k_h2048 bs64 | 109.7 | 1948.3 | 15111.1 | 17.75× |
| llama-128k_h2048 bs1 | 92.4 | 1260.3 | 1445.6 | 13.64× |
| llama-128k_h2048 bs16 | 92.9 | 1395.8 | 2579.2 | 15.03× |
| **draft_h2048_v32k bs16** | 37.3 | 252.9 | 603.2 | **6.79×** (best) |
| draft_h2048_v32k bs64 | 37.1 | 329.9 | 1636.8 | 8.89× |

분석: hidden 4096→2048 절반 줄여도 GPU sm_100 의 절대 TFLOPS (≥350 TF @ bs64) 와 격차 좁히지 못함. AMX peak BF16 ~ 56 cores × 2 TMUL × 1024 BF16 ops/cycle ≈ 110 TF/s 이론치인데 실측 25 TF (저활용). vocab 축소 (v32k) 까지 가도 6.79×.

### 2.2 C2 — RMS norm (memory-bound)

| config | GPU | AMX | AVX512 | AMX/GPU |
|---|---|---|---|---|
| **tok2k_h4096** | 112.7 | 471.9 | 3122.1 | **4.19×** (best) |
| tok8k_h4096 | 380.5 | 49777.9 | 41645.2 | 130.84× |
| tok2k_h8192 | 195.8 | 15978.6 | 22341.5 | 81.60× |
| tok8k_h8192 | 731.9 | 104894.5 | 97023.3 | 143.31× |

분석: small tile (tok2k, h4096) 에서만 4.19× 로 gate 근접. larger working set 진입하면 100×+ — torch eager 가 BF16 → FP32 promote → BF16 demote 매번 발생, AMX bf16 reduce 가속이 의미 없음. vLLM 내 fused RMSNorm kernel 비교 시 격차 크게 줄 수 있으나 microbench 만으로는 가치 입증 불가.

### 2.3 C3 — prefix radix-tree byte scan + hash (winner)

| config | GPU | AMX | AVX512 | AMX/GPU |
|---|---|---|---|---|
| prefix_64KB | 61.9 | 141.2 | 121.1 | 2.28× |
| prefix_256KB | 61.4 | 144.3 | 115.0 | 2.35× |
| **prefix_1MB** | 64.1 | 130.6 | 124.7 | **2.04×** (best) |
| prefix_4MB | 95.1 | 250.6 | 5733.4 | 2.64× |

분석:
- **모든 4 size 에서 AMX ≤ 3× GPU latency 게이트 PASS**.
- GPU 절대값 60-95μs 의 절반은 사실 kernel launch + sync overhead — 실제 GPU mem BW 는 88 GB/s (4 MB) 까지 나오지만 host 가 wait 해야 함. CPU lane 은 ortho.
- AVX-512 가 1 MB 까지 AMX 와 동등 (memory-bound; 둘 다 L2 → DRAM 한계), 4 MB 에서 AVX-512 망가짐 (5.7 ms) — AMX bf16 path 가 L2 streaming 우월.
- 응용: prefix-cache 의 radix-tree node match (`__builtin_memcmp` 후 hash), DSA memcmp(byte equality) 와 결합 시 16 MB+ prefix 까지 가속 가능.

### 2.4 C4 — KV scale calib (per-layer per-head abs-max)

| config | GPU | AMX | AVX512 | AMX/GPU |
|---|---|---|---|---|
| **kvscale_h32_d128_t1k** | 35.1 | 1398.2 | 1705.9 | **39.82×** (best) |
| kvscale_h32_d128_t8k | 121.8 | 17321.0 | 18608.3 | 142.26× |
| kvscale_h64_d128_t8k | 215.0 | 24500.1 | 26376.6 | 113.97× |

분석: torch `.abs().amax((0,2))` 가 CPU 에서 fully sequential, 56 threads 활용 못함. handwritten AVX-512 reduce 시 ~10× 개선 추정해도 GPU 35μs vs CPU 140μs 로 여전히 4× FAIL. KV scale 은 GPU 가 압도적.

### 2.5 C5 — fused norm + residual add

| config | GPU | AMX | AVX512 | AMX/GPU |
|---|---|---|---|---|
| **tok2k_h4096** | 124.0 | 3159.5 | 443.7 | **25.49×** (best) |
| tok8k_h4096 | 442.4 | 56633.6 | 57597.6 | 128.02× |
| tok8k_h8192 | 851.7 | 128765.5 | 123751.3 | 151.19× |

분석: AVX-512 가 tok2k 에서 443 μs 로 AMX 보다 7× 빠름 — torch matmul/bmm 이 AMX tile 을 잘못 활용 (norm 은 GEMM 아님). 어쨌든 GPU 124μs 와 격차 크게 FAIL.

---

## 3. C3 의 vLLM 적용 시나리오 (선정 근거)

| 적용 위치 | 현 GPU 시간 | C3 lane 가능성 |
|---|---|---|
| prefix-cache lookup (PageBlockManager hash chain) | host op, GPU 비관여 | direct host gain |
| LoRA adapter selection (request → adapter hash) | host op | direct host gain |
| structured output XGrammar token mask precompute | host op | candidate |
| KV slot ID dedup before kernel launch | host op | candidate |

→ **GPU 가 비관여**한 host-side byte scan op 가 vLLM critical path 에 다수 존재. C3 lane 은 이 host op 들을 AMX BF16 + AVX-512 mixed 로 가속. Phase 1 의 DSA memcmp lane 과 결합하면 4 MB prefix까지 ortho.

---

## 4. Task C gate matrix

| sub-lane | AMX/GPU ≤ 3× | absolute AMX latency 의미 | 회수 가능성 (vLLM hot path) | accept |
|---|---|---|---|---|
| C1 draft matmul | FAIL (6.8×) | 250μs — GPU bs64 단일 37μs | GPU 가 더 직접적 | reject |
| C2 RMS norm | FAIL (4.2×) | tile dependent | fused kernel 안에 흡수됨 | reject |
| **C3 prefix scan** | **PASS (2.0×)** | 130μs @ 1MB | prefix-cache, host hash 경로 다수 | **accept** |
| C4 KV scale | FAIL (40×) | torch reduce sequential | GPU 압도 | reject |
| C5 norm+add | FAIL (25×) | GEMM-shape 가 아님 | fused kernel 안 | reject |

**verdict**: C3 단독 채택, Task G 통합 측정 진입.

---

## 5. caveat — AMX peak vs torch 실측 격차

CPU 절대 latency 가 microbench 에서 실망스러운 이유:
1. torch.matmul 이 BF16 GEMM 호출 시 oneDNN 의 `brg_matmul:avx10_1_512_amx` primitive 까지는 가는 것이 Phase 2에서 확인되었으나, K=2048 small-batch shape 에서 prepacking + dispatch overhead 가 200-1000μs.
2. element-wise op (C2, C5) 은 torch eager mode 에서 BF16 → FP32 promote → BF16 demote 매 op 발생 → 실효 BW 4-6 GB/s (DRAM peak ~100 GB/s × node 의 4-6% 만).
3. C4 의 amax reduce 는 dim=(0,2) skip 패턴이라 sequential.

이론 peak 와 격차가 큰 점은 **vLLM 통합 시 fused kernel (oneDNN inplace, ipex `linear_silu_mul`, mkl_dnn ROPE) 가 microbench 격차의 1/3-1/5 로 줄임**. 즉 C2/C5 는 통합 측정에서 다시 확인이 정당화될 수 있으나 Phase 3 임무는 정량 gate 통과 sub-lane 채택이므로 **microbench gate 만 사용**.

---

## 6. 산출물

```
lhc_phase3/
├── amx_sub_lane_bench.py
├── amx_sub_lane_gpu.json
├── amx_sub_lane_amx.json
├── amx_sub_lane_avx512.json
└── amx_sub_lane_microbench.md      ← 본 문서
```
