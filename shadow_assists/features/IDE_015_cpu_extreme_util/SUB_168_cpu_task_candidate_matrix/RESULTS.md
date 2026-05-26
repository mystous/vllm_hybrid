# SUB_168 — per-phase CPU task candidate matrix (paper Table 1b)

> **parent**: IDE_015 / TSK_023 (원 SUB_105) — 분석 SUB
> **scope**: 2026-05-26 (analysis only)
> **status**: ✅ 완료 — SUB_162 (CPU thread state) + SUB_117 (per-worker util) + SUB_167 (GPU idle phase matrix) 결합 → CPU task candidate matrix
> **purpose**: IDE_018 phase-burst scheduler 의 task pool 정의 — attention-phase vs linear-phase 별 CPU task candidate

---

## 0. 두괄식 — paper Table 1b: CPU task candidate × phase matrix

| Phase | GPU 상태 | CPU 가용 자원 | CPU task candidate | latency target | source SUB |
|---|---|---|---|---:|---|
| **attention** (memory-bound) | HBM bandwidth-bound, ALU idle | 10.24 TFLOPS AMX (SUB_117), idle 96% (SUB_162) | (A) schedule next batch / (B) detokenize / (C) grammar check / (D) classify next request | < 10 ms | SUB_117, SUB_162, SUB_148 |
| **linear** (compute-bound) | tensor cores 100%, HBM available | 10.24 TFLOPS AMX + DMA 54 GB/s (SUB_166) | (E) KV prefetch / (F) AMX draft head / (G) cold-KV decompress | < 5 ms | SUB_117, SUB_166 |
| **sampling** | low GPU util (~15-25%) | sampler 44.3% CPU 시간 (SUB_161) | (H) AVX-512 sampling rewrite (TSK_025) / (I) logit pre-process | < 3 ms | SUB_161 |
| **TP all-reduce** | network-bound | communication_op 19% (SUB_161), AVX-512 idle | (J) logit pre-compute on next layer | < 2 ms | SUB_161 |
| **inter-step idle** | 0% GPU, 100% CPU available | full 10.24 TFLOPS, 96% S threads | **all tasks possible** | < 5 ms | SUB_162 |

→ 본 matrix 가 **IDE_018 phase-burst scheduler 의 task pool 정의** + **paper §3 method 의 Figure 1** 의 입력.

---

## 1. Task candidate breakdown (10 tasks × 5 phases)

### 1.1 Task A — schedule next batch (attention-phase)

- **input**: 다음 batch 의 prompt list + KV cache state
- **work**: assemble metadata (position ids, attention mask, rope cache)
- **CPU cost**: ~1-3 ms (AVX-512 vectorize 가능)
- **dependency**: TSK_024 (AVX-512 vectorized scheduler)
- **GPU phase 적합도**: attention 시 GPU 가 HBM bandwidth-bound → ALU 자유 → CPU 가 안전하게 메타데이터 빌드

### 1.2 Task B — detokenize previous step output (attention-phase)

- **input**: token IDs from prev step (vocab 152K, batch 32-128)
- **work**: token id → text token (vocab lookup + AVX-512 batch BPE/SentencePiece)
- **CPU cost**: ~1-5 ms / batch
- **dependency**: TSK_024 (AVX-512 tokenizer)
- **GPU phase 적합도**: attention (memory-bound), 또는 inter-step idle

### 1.3 Task C — grammar / constraint check (attention-phase)

- **input**: candidate tokens (top-k) + constraint state machine
- **work**: XGrammar 또는 outlines style verification
- **CPU cost**: ~2-10 ms (workload-dependent)
- **dependency**: 기존 XGrammar 통합
- **GPU phase 적합도**: attention

### 1.4 Task D — request classifier (attention-phase)

- **input**: new incoming requests
- **work**: workload classification (SUB_076 의 classifier PoC)
- **CPU cost**: ~1 ms
- **dependency**: IDE_012 (완료 — SUB_076 classifier)
- **GPU phase 적합도**: attention (CPU 비 GPU 자원)

### 1.5 Task E — KV cache prefetch via DMA pull (linear-phase)

- **input**: predicted next KV chunks (CPU-side staging buffer)
- **work**: DMA pull from CPU memory to GPU
- **CPU cost**: minimal (DMA initiation, 1 μs)
- **DMA cost**: 60 μs / 1 MB chunk (SUB_166)
- **dependency**: TSK_028 (pinned pool)
- **GPU phase 적합도**: linear (GPU 가 compute 중이면 HBM 일부 자유)

### 1.6 Task F — AMX draft head inference (linear-phase)

- **input**: input embeddings, hidden states
- **work**: Qwen 0.5B/1.5B forward on CPU AMX
- **CPU cost**: ~5-10 ms / batch (target ≤ 5 ms)
- **dependency**: TSK_026 AMX kernel + TSK_036 draft head
- **GPU phase 적합도**: linear (overlap with target GPU forward)

### 1.7 Task G — cold-KV decompress + DMA push (linear-phase)

- **input**: quantized cold KV blocks (host RAM)
- **work**: AVX-512 dequant (Q8/INT4 → BF16) + DMA push to GPU
- **CPU cost**: ~5-20 ms / chunk
- **DMA cost**: 113 μs / 4 MB chunk (SUB_166)
- **dependency**: TSK_030 (IDE_006 재정의)
- **GPU phase 적합도**: linear

### 1.8 Task H — AVX-512 sampling rewrite (sampling-phase)

- **input**: logits tensor (vocab × batch BF16)
- **work**: top-k/top-p + greedy choice on CPU
- **CPU cost**: 본 SUB_161 의 44.3% sampler.py 시간을 ~50% 단축 가능 (target 1.5 ms / step)
- **dependency**: TSK_025 (AVX-512 sampling kernel)
- **GPU phase 적합도**: sampling (GPU 거의 idle)

### 1.9 Task I — logit pre-processing (sampling-phase)

- **input**: pre-sampling logit tensor
- **work**: bias / temperature / penalty application (AVX-512)
- **CPU cost**: ~0.5-1 ms
- **dependency**: TSK_025
- **GPU phase 적합도**: sampling

### 1.10 Task J — logit pre-compute on next layer (TP all-reduce phase)

- **input**: hidden state intermediate
- **work**: AVX-512 vectorize layer init for next forward
- **CPU cost**: ~1-2 ms
- **dependency**: 새 TSK 필요 (현 plan 에 없음)
- **GPU phase 적합도**: NCCL all-reduce (network-bound)

---

## 2. paper §3 main figure 후보 — task scheduling diagram

```
GPU timeline:   |  attention  | linear  | TP-ar | sample | post |
                ────────────────────────────────────────────────
CPU pool A:     | A schedule  | E prefetch     |        |       |
CPU pool B:     | B detokenize| F AMX draft    | H AVX  |       |
CPU pool C:     | C grammar   | G cold-KV     |        | tokenize |
CPU pool D:     |             | D classify    |  J     |       |
                ────────────────────────────────────────────────
N=20 worker (사용자 100 core max constraint 반영):
  4 worker × 5 task type (A-J 시간별 dispatch)
```

→ phase-burst scheduler 가 GPU phase signal 받아 task pool 에서 task dispatch.
→ 10 tasks / 5 phase 의 dispatch 정책 = TSK_036 의 phase-burst scheduler design 입력.

---

## 3. expected lift breakdown (per-task)

| Task | expected per-step latency 감소 | expected throughput lift |
|---|---:|---:|
| H AVX-512 sampling rewrite | −1.5~3 ms / step | +5-10% |
| B AVX-512 detokenize | −0.5-2 ms / step | +1-3% |
| A schedule + metadata | −0.3-1 ms / step | +1-2% |
| F AMX draft head (extra speculation) | acceptance rate +5-15% | +5-15% |
| E + G KV prefetch + cold KV | TTFT −10-20% | +2-5% |
| C grammar offload | latency tail 감소 | +1-2% (p99) |
| **Total integrated (IDE_018)** | — | **+15-30%** |

→ Phase 2 (kernel dev) + Phase 3 (IDE_018) 완료 시 throughput 추가 lift 예상.

---

## 4. 다음 step

- **(별도 turn — kernel dev)** TSK_024-027, TSK_028-030, TSK_031-034, TSK_035-037 — 본 matrix 의 task 별 구현
- (paper 작성 시) 본 SUB 의 matrix 를 §3 Figure 1 의 input 으로 사용

---

## 5. source data 참조

- [SUB_161](../SUB_161_ncu_pyspy_profile/RESULTS.md) — py-spy sampler 44.3% / logits 27% / penalties 23%
- [SUB_162](../SUB_162_cpu_idle_gap/RESULTS.md) — VLLM Worker_TP 96% S / EngineCore 99.8% S
- [SUB_166](../SUB_166_dma_microbench/RESULTS.md) — DMA 35 μs overhead + 54 GB/s asymptotic
- [SUB_117](../SUB_117_per_worker_util/RESULTS.md) — 10.24 TFLOPS available CPU compute
- [SUB_167](../SUB_167_gpu_idle_phase_matrix/RESULTS.md) — GPU idle window phase matrix
- [SUB_148](../SUB_148_trident_thread_placement/RESULTS.md) — VLLM 자체 thread pin 없음 → OS scheduling 협조 가능
