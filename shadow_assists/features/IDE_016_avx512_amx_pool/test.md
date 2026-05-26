# test.md — IDE_016 테스트 계획

## 1. correctness tests

### 1.1 AVX-512 sampling vs PyTorch baseline (TSK_025)

```python
# tests/test_correctness_vs_gpu.py
import torch
from vllm_hybrid_kernels import avx512_sampling

@pytest.mark.parametrize("k,p", [(20, 0.95), (50, 0.9), (1, 1.0)])
def test_topk_topp_matches_torch(k, p):
    logits = torch.randn(32, 152064, dtype=torch.bfloat16)  # batch=32, Qwen vocab
    # Reference: pure PyTorch
    ref = vllm_sampler_reference(logits, k=k, p=p)
    # Test: AVX-512 kernel
    out = avx512_sampling.topk_topp(logits, k=k, p=p)
    # 분포·의도 수준 검증 (CLAUDE.md 운영 해석)
    logprob_ref = torch.log_softmax(ref, dim=-1)
    logprob_out = torch.log_softmax(out, dim=-1)
    max_abs_diff = (logprob_ref - logprob_out).abs().max().item()
    assert max_abs_diff < 1e-3, f"logprob max abs diff {max_abs_diff} exceeds 1e-3"
```

### 1.2 AMX matmul vs PyTorch CPU matmul (TSK_026)

```python
# tests/test_amx_matmul.py
@pytest.mark.parametrize("M,K,N", [
    (128, 896, 4864),     # Qwen 0.5B MLP
    (128, 1536, 8960),    # Qwen 1.5B MLP
    (128, 5120, 27648),   # Qwen 32B MLP (SUB_106 reference)
])
def test_amx_matmul_correctness(M, K, N):
    A = torch.randn(M, K, dtype=torch.bfloat16)
    W = torch.randn(K, N, dtype=torch.bfloat16)
    ref = (A.float() @ W.float())  # FP32 reference
    out = amx_matmul.matmul_bf16(A, W)
    rel_err = ((ref - out.float()).abs() / ref.abs().clamp_min(1e-3)).max().item()
    assert rel_err < 0.01, f"AMX matmul rel err {rel_err} exceeds 1%"
```

## 2. latency benchmark

### 2.1 sampling latency (TSK_025)

```python
# tests/bench_sampling_latency.py
import time
N_ITER = 1000
logits = torch.randn(32, 152064, dtype=torch.bfloat16)

# baseline (PyTorch)
torch.cuda.synchronize()  # if on GPU
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = pytorch_sample(logits, k=20, p=0.95)
t_baseline = (time.perf_counter() - t0) * 1000 / N_ITER  # ms

# AVX-512
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = avx512_sampling.topk_topp(logits, k=20, p=0.95)
t_avx512 = (time.perf_counter() - t0) * 1000 / N_ITER

print(f"baseline {t_baseline:.2f} ms / avx512 {t_avx512:.2f} ms / speedup {t_baseline/t_avx512:.2f}x")
assert t_avx512 < t_baseline * 0.5, "expect >=2x speedup"
```

### 2.2 AMX matmul latency (TSK_026)

target: ≥3× vs `torch.matmul(cpu)` for Qwen 0.5B/1.5B shapes.

## 3. end-to-end throughput (canonical)

### 3.1 TSK_025 alone on AGSD-gated balanced 500p

```bash
# enable env var
export VLLM_USE_AVX512_SAMPLING=1
# canonical AGSD launch (SUB_160 protocol)
bash /tmp/run_canonical_agsd_500p.sh

# expected:
#   AGSD-gated balanced 500p tps:
#     SUB_160 baseline 5,474 → IDE_016/TSK_025 target ≥5,747 (+5%)
```

### 3.2 TSK_026 alone (AMX draft head via IDE_019)
- deferred (depends on IDE_019)

## 4. util capture (필수)

모든 measurement 시 `eval/monitor.py` background attach:
```bash
python eval/monitor.py /path/to/out --interval 0.5 &
```

기록: per-test CPU% / GPU% / 전후 비교.

## 5. accuracy gate (CLAUDE.md 운영 해석)

- per-token logprob max abs diff < 1e-3
- sequence PPL relative diff < 1%
- token-level 일치율은 informational (regression 추적 metric)

## 6. CI / regression

- pre-commit hook: `pytest tests/test_correctness_*.py`
- per-PR: bench latency + correctness 모두 pass
- nightly: e2e throughput on canonical
