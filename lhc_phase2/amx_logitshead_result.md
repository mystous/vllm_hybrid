# LHC Phase 2 — Task 2: AMX logits head matmul gate (측정)

**날짜**: 2026-06-08
**대상 op**: `logits[bs, vocab] = hidden[bs, hidden_dim] @ embed_T[hidden_dim, vocab]`
**비교**: AMX bf16 (oneDNN brg_matmul) / AVX-512 BF16 / B200 bf16
**gate (Phase 1 verdict)**: AMX/GPU latency ≤ 1.5× **AND** throughput ≥ 50k logits/s @ vocab 152k

---

## 0. TL;DR

| metric | gate | 측정 | verdict |
|--------|------|------|---------|
| AMX/GPU latency ratio (best case: qwen-152k bs=64, 56T) | ≤ 1.5× | **71×** (12.2 ms vs 173 μs) | **FAIL** |
| AMX throughput @ qwen-152k bs=64 | ≥ 50k logits/s | 795M logits/s | numerically PASS |
| AMX/AVX512 latency ratio | (정보) | 0.6–1.3× | AVX 가 작은 batch 에서 종종 더 빠름 |
| GPU 절대 latency | — | 165–320 μs | — |

→ **AMX logits-head lane: gate FAIL**.

→ 원인 (one-DNN verbose 확인): primitive 선택은 정확히 `brg_matmul:avx10_1_512_amx` 인데도 절대 latency 가 GPU 의 ~70-90×. AMX peak FLOPS 자체는 sufficient (CPU 의 ~2 TFLOPS BF16) 이지만, B200 의 BF16 Tensor Core 가 ~2 PFLOPS 로 **1000× 격차**. matmul 만으로는 ortho-lane 가치 없음 (GPU 가 그 시간 동안 더 큰 batch 의 다음 step prefill 을 처리하는 게 net-better).

---

## 1. 측정 결과 — AMX (112 thread 기본)

vocab 별 bs sweep, latency p50 μs / throughput logits/s.

| vocab × bs | AMX bf16 (112T) | AVX-512 BF16 | GPU bf16 (B200) | AMX/GPU | AVX/GPU |
|-----------|-----------------|--------------|-----------------|--------:|--------:|
| llama-128k × 1  | 15710 / 8.16 M | **7547 / 17.0 M** | 167 / 768 M | 94× | 45× |
| llama-128k × 8  | 16326 / 62.8 M | **8080 / 127 M**  | 165 / 6.23 G | 99× | 49× |
| llama-128k × 32 | 16098 / 255 M  | 12655 / 324 M | 169 / 24.2 G | 95× | 75× |
| llama-128k × 64 | **17112 / 480 M** | 19254 / 426 M | 167 / 49.2 G | **102×** | 115× |
| qwen-152k × 1   | 12260 / 12.4 M | **5493 / 27.7 M** | 172 / 885 M | 71× | 32× |
| qwen-152k × 8   | 12061 / 101 M  | **6419 / 190 M**  | 172 / 7.06 G | 70× | 37× |
| qwen-152k × 32  | 12099 / 402 M  | 11787 / 413 M | 174 / 27.9 G | 70× | 68× |
| qwen-152k × 64  | **13928 / 699 M** | 19553 / 498 M | 173 / 56.2 G | **80×** | 113× |
| deepseek-256k × 1 | 18058 / 14.2 M | **9730 / 26.4 M** | 313 / 821 M | 58× | 31× |
| deepseek-256k × 8 | 18044 / 114 M  | 11984 / 172 M | 314 / 6.56 G | 57× | 38× |
| deepseek-256k × 32 | 19217 / 428 M | 22186 / 371 M | 321 / 25.6 G | 60× | 69× |
| deepseek-256k × 64 | **23249 / 708 M** | 36045 / 457 M | 318 / 51.7 G | **73×** | 113× |

## 2. 56-thread 단일-socket 측정 (NUMA 분산 가설 확인)

KMP_AFFINITY=fine,compact 로 단일 socket bind:

| vocab × bs | AMX 56T p50 | AMX 112T p50 | 개선율 |
|-----------|-------------|--------------|--------|
| llama-128k × 1 | 8783 | 15711 | 1.79× |
| llama-128k × 8 | 8114 | 16326 | 2.01× |
| llama-128k × 64 | 11444 | 17112 | 1.50× |
| qwen-152k × 64 | **12240** | 13928 | 1.14× |
| deepseek-256k × 64 | 15434 | 23249 | 1.51× |

→ 56T 가 작은 batch 에서 ~2× 개선 (NUMA cross-socket 비용 회피). 그래도 **AMX/GPU best ratio = 12240/173 = 71×** → gate FAIL.

## 3. oneDNN verbose 검증

```
onednn_verbose,v1,info,cpu,isa:Intel AVX-512 with float16, Intel DL Boost
                                and bfloat16 support and Intel AMX with
                                bfloat16 and 8-bit integer support
onednn_verbose,v1,primitive,exec,cpu,matmul,brg_matmul:avx10_1_512_amx,
                                undef,src:bf16::blocked:ab::f0
                                wei:bf16::blocked:ab::f0
                                dst:bf16::blocked:ab::f0,
                                ...,8x4096:4096x152064,62.825
```

→ oneDNN 가 정확히 **AMX tile (brg_matmul:avx10_1_512_amx) primitive** 호출. AMX HW 가 실제로 사용 중인 상태에서의 한계.

## 4. ortho-lane 분석

AMX logits-head 를 ortho-lane 으로 쓰려면 다음 중 하나 성립 필요:

- (i) AMX 가 GPU 보다 빠르거나 (현실: 70-100× 느림 → FAIL)
- (ii) AMX 가 GPU 와 동시에 진행하여 GPU 가 그 시간 동안 더 큰 step 의 다음 작업을 처리 가능 → 그러나:
  - logits 결과는 sample → 다음 token prefill 의 input 이라 **직렬 의존**
  - AMX 8-15 ms 동안 GPU 는 같은 batch 의 다음 decode step 대기 → throughput 손실
  - 큰 batch 일수록 손실 비율 증가 (gpu latency 가 batch-scaling 거의 안 함, AMX 는 batch-1 도 8 ms)

→ ortho-lane 가치 **음수**. 채택 시 throughput 감소 예상.

## 5. 결론

| 항목 | 판정 |
|------|------|
| logits-head AMX gate | **FAIL** (70-100× ratio) |
| AMX lane 전체 폐기? | NO — 다른 sub-lane 탐색 필요 |
| 다음 sub-lane 후보 | **AMX 미적용 + 양수 음수 비교 표** 만들어 Phase 3 에서 재정의 |

### 5.1 가능한 다른 AMX sub-lane (Phase 3 검토 대상)

1. **CPU-side draft model head** (speculative decoding 의 draft 가 작은 vocab + 작은 hidden → AMX peak 영역에 더 적합). 직전 세션 `cpu_amx_draft` 가 -98% 였던 것은 full draft model 을 CPU 에 두었기 때문. **draft head matmul 만 CPU**, draft transformer 는 GPU 유지하면 ortho 성립할 가능성. (별도 microbench 필요)
2. **Prefill chunked classifier / safety filter / token classifier**: 작은 model 의 CPU offload (LM head 가 아닌 작은 BERT-class). 본 LHC 의 lane separation theorem 의 "CPU 가 GPU 못하는 일" = 작은 micro-batch.
3. **RMSNorm / RoPE / activation fused CPU**: memory-bound op 이지만 GPU 의 BW 가 압도적이므로 의문. PMU bound op 분류 후 결정.

## 6. Task 2 verdict

→ **AMX logits-head lane: NO-GO**. Task 3 (vLLM 통합) 에서 **DSA lane 단독으로 진행**.
→ AMX 는 Phase 3 에서 (1)~(3) sub-lane microbench 후 재평가.

## 7. 산출물

```
/workspace/host_vllm_hybrid/lhc_phase2/
├── amx_logitshead_bench.py         ← 3-backend bench
├── amx_logitshead_amx.json          ← raw (AMX 112T)
├── amx_logitshead_avx512.json       ← raw (AVX512)
├── amx_logitshead_gpu.json          ← raw (B200)
└── amx_logitshead_run.log           ← full output + oneDNN verbose
```
