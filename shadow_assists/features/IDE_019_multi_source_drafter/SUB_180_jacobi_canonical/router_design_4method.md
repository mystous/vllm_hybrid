# 4-method AGSD router design (cpu_jacobi 분기)

> SUB_180 design output. 실 통합은 SUB_181 (TSK_037).
> base = `/tmp/sub094_router.py` (3-method: vanilla / ngram / suffix)
> add  = `cpu_jacobi` 4th method.

---

## 1. 현재 router 구조 요약

```python
# /tmp/sub094_router.py
BACKENDS = {
    "vanilla": ".../v1/completions",       # GPU 0-3, no spec
    "trident": ".../v1/completions",       # GPU 4-7, spec (ngram or suffix)
}
# decide(workload) returns one of: "vanilla" | "trident"
# classifier (regex) -> workload in {"sonnet", "chat", "code"}
```

## 2. 4-method 확장 design

### 2.1 새 backend (논리적)

```python
BACKENDS = {
    "vanilla":   "http://127.0.0.1:8001/v1/completions",   # 기존
    "trident":   "http://127.0.0.1:8002/v1/completions",   # 기존 (ngram/suffix on GPU)
    "cpu_jacobi": "http://127.0.0.1:8003/v1/completions",  # NEW: CPU draft + GPU verify
}
```

- `cpu_jacobi` endpoint 는 별도 vllm process. spec_config 에 `method="cpu_jacobi"`
  추가, draft 는 CPU AVX-512 Jacobi kernel (libjacobi_avx512.so) 에서 수행
  후 logits 또는 token sequence 를 verify GPU 에 enqueue.

### 2.2 decide rule 확장

```python
def decide(workload: str, model_size: str, prefix_len: int) -> str:
    """
    classifier output (workload) + 추가 signal 로 4-way 분기.
    """
    if workload == "code":
        # n-gram suffix repetition 강함 -> ngram 우월
        return "trident"           # ngram K=7 (SUB_044 10,778 tps)
    if workload == "sonnet":
        # creative, low repetition -> suffix (or vanilla) 우월
        return "trident"           # suffix
    if workload == "chat":
        # multi-turn, semantic continuity -> Jacobi self-draft 가 acceptance 높음 가설
        #   IDE_011: chat α=81.2% (K=6.69 measured)
        return "cpu_jacobi"        # NEW lever — 본 SUB feasibility 후 SUB_181 측정
    return "vanilla"
```

### 2.3 fast-path / fallback

```python
# fast-path: prefix_len < 16 token -> classification 우회, vanilla 직행
if prefix_len < 16:
    return "vanilla"

# fallback: cpu_jacobi unhealthy -> degrade to trident
if not health_ok("cpu_jacobi"):
    return "trident"
```

### 2.4 cpu_jacobi backend 의 내부 pipeline

```
[request] -> classifier ok -> POST /v1/completions
  vllm process (cpu_jacobi-enabled build) {
    on each decode step:
      1) CPU AVX-512 Jacobi kernel produces K candidate tokens
         (input: target model's last hidden state copied to host)
         - LM head matmul on CPU (libjacobi_avx512.so)
         - argmax per K position
         - Jacobi fixed-point iteration (max 8 iters)
      2) DMA push candidate token IDs to GPU (small, ZC region)
      3) GPU target model 1-pass verify across K+1 positions
      4) accept prefix that matches argmax of target logits
  }
```

## 3. e2e overhead budget (SUB_180 측정 기반)

| stage | latency (Sapphire Rapids, T=64) |
|---|---:|
| CPU LM-head argmax K=7 B=4 (BK=28) | **~1,713 ms** (실측 본 SUB) |
| CPU Jacobi iter loop (5 avg iters) | × 5 = **~8.5 sec** |
| DMA push K token IDs (≤ 32 B) | < 50 μs |
| GPU verify (1 forward) | ~35-44 ms (canonical) |

→ **현재 kernel 의 throughput 으로는 net loss**. GPU verify 35 ms 대비 CPU
draft 1,700 ms 가 ~50× 큰 cost. **본 kernel 의 vocab argmax 가
bottleneck** (152K wide × 5120 hidden × BF16 BW-bound).

## 4. net-win 조건 (SUB_181 측정 target)

본 SUB 의 verdict 는 "drop-in lever 아님 / NEW workload 후보" 인데, net-win 을
위해서는 다음 중 하나 필요:

1. **vocab tile sharing across K**: 동일 W matrix 를 K row 가 공유하므로
   bandwidth amortize. 본 SUB kernel 은 이미 outer-vocab tile, but per-row
   inner loop 가 scalar gather (W column stride = vocab) — true 16-wide
   tile-major repack 필요 (W transpose to [vocab, hidden] before main loop).
2. **partial vocab (top-N)**: full vocab 대신 top-1000 most-frequent token 만
   evaluate. quality 영향 검증 필요.
3. **smaller draft model**: Qwen 0.5B (hidden=896) 로 draft 시 vocab/hidden
   factor 4-6× 감소 → 5120/896 ≈ 5.7× 빨라짐. Jacobi self-draft 대신 별도
   small model + draft head 패턴.
4. **K parallelism native (AMX)**: BK=28 × hidden=5120 × vocab=152064 matmul 을
   AMX tile 로 한번에. SUB_175 의 0.046 TFLOPS limit 으로는 부적합.

→ 본 SUB 의 **결론**: SUB_181 측정 전제는 (1) + (3). 즉 Jacobi kernel 자체
보단 small draft model + W-major repack 이 lever.

## 5. spec_config 통합 hook (vllm side)

```python
# vllm/config/speculative.py 확장 (스케치)
SPEC_METHODS = ("ngram", "eagle", "medusa", "suffix", "cpu_jacobi")  # NEW

# vllm/v1/spec_decode/ 아래 새 module:
#   cpu_jacobi_proposer.py
#       class CpuJacobiProposer(SpecDecodeProposer):
#           def __init__(self, K, max_iters, lm_head_weight_path):
#               self.lib = ctypes.CDLL(".../libjacobi_avx512.so")
#               self.W = load_lm_head_bf16(lm_head_weight_path)  # mmap
#           def propose(self, hidden_state, **kwargs):
#               BK = batch * self.K
#               candidates = np.zeros(BK, dtype=np.int32)
#               self.lib.jacobi_run(hidden_state, self.W, candidates,
#                                   batch, self.K, hidden, vocab,
#                                   self.max_iters, n_threads, iters_used_ptr)
#               return candidates
```

## 6. accuracy gate strategy (SUB_181)

- `jacobi_lossless_proof.md` §1.3 의 fixed-point theorem 에 의거,
  greedy mode 에서 token-level bit-exact 가 기대됨.
- verify stage 가 항상 강제되므로 distribution-level lossless 는 보장 (rejection
  sampler).
- gate metric (SUB_181 측정 시): per-token logprob max abs diff < 1e-3,
  token agreement ≥ 99.5% (greedy seed=42, 8 prompts × 32 tokens).
