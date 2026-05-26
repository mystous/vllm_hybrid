# task.md — IDE_019 단계별 구현

## TSK_035 — Jacobi lookahead correctness + AVX-512 kernel

### Step 1: theory + lossless guarantee proof
- design/jacobi_lossless_proof.md
- Jacobi iteration 의 candidate token 이 rejection sampling 과 정합하는지 수식 증명
- reference: lookahead decoding (USENIX OSDI'24)

### Step 2: CPU Jacobi iteration kernel (AVX-512 vectorize)
- src/jacobi_avx512/jacobi_kernel.cpp
- K parallel hypothesis lane (typically K=5-7)
- AVX-512 16-wide BF16 SIMD: K=7 candidate × 16 vocab chunk parallel
- iteration termination: convergence detection (changes < threshold)

### Step 3: candidate quality measurement vs ngram/suffix
- acceptance rate per workload (sonnet / chat / code)
- target: ≥ ngram 60% (chat workload), ≥ suffix code 40%

## TSK_036 — AMX draft head on small model

### Step 1: Qwen 0.5B model load to CPU
- src/amx_draft_head/qwen_model_load.cpp
- ALL weights to CPU pinned memory (~1 GB for 0.5B)
- weight repack to AMX-friendly K-major (TSK_026 amx_repack_b_bf16)

### Step 2: full forward pipeline
- src/amx_draft_head/qwen_forward.cpp
- embed → 24 × (RMSNorm + Attn + MLP) → RMSNorm → lm_head
- Attn: RoPE + GQA + flash attention (AVX-512)
- MLP: TSK_026 amx_matmul_bf16
- target: ≤ 5 ms / batch (B=32)

### Step 3: draft step latency benchmark
- microbench: per-step latency vs Qwen 32B target step
- target: draft step ≤ 5 ms / target step ~10-15 ms → draft amortizable

### Step 4: K acceptance rate vs ngram K (R/K balance)
- per workload (sonnet / chat / code) measurement
- target: chat α ≥ 60% with K=5-7 candidates

## TSK_037 — AGSD multi-source integration on canonical

### Step 1: router 4-method 분기 확장
- /tmp/sub094_router.py 수정 (또는 새 wrapper)
- method list: vanilla, ngram, suffix, **cpu_amx_draft**
- classifier (SUB_076 PoC) 가 workload 별 method 선택

### Step 2: per-workload best-source selection rule
- decision rule (from SUB_011 + 새 측정):
  - chat: cpu_amx_draft (α 가장 큼 가설)
  - sonnet: suffix (K 작아 GPU 우월)
  - code: ngram (K=7 이미 lever)

### Step 3: e2e on canonical 3 mix
- vs single-source baseline (suffix only / ngram only)
- target: +5-15% throughput (per-workload best vs single)
- accuracy gate: per-token logprob max abs diff < 1e-3
