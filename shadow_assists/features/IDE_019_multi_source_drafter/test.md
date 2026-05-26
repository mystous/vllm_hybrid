# test.md — IDE_019 테스트 계획

## 1. correctness

### 1.1 Jacobi lossless guarantee (TSK_035)

```python
def test_jacobi_matches_autoregressive():
    """Jacobi 의 K-token output 이 K times autoregressive decode 와 동등."""
    seed = 42
    prompt = "..."
    ar_output = autoregressive_decode(prompt, max_tokens=64, seed=seed)
    jacobi_output = jacobi_decode(prompt, max_tokens=64, K=7, seed=seed)
    # CLAUDE.md 운영 해석: 분포·의도 수준
    # Jacobi 는 lossless guarantee 가 strict — token-level 일치 기대
    assert ar_output == jacobi_output, "Jacobi must be lossless"
```

### 1.2 AMX draft head correctness (TSK_036)

```python
def test_amx_qwen05b_matches_gpu():
    """AMX CPU forward 가 GPU Qwen 0.5B 와 일관 (분포 수준)."""
    inputs = torch.randint(0, 152064, (32, 128))  # B=32, seq=128
    gpu_logits = gpu_qwen05b_forward(inputs)  # FP16/BF16 GPU
    cpu_logits = amx_qwen05b_forward(inputs)  # AMX BF16 CPU
    logprob_gpu = torch.log_softmax(gpu_logits, dim=-1)
    logprob_cpu = torch.log_softmax(cpu_logits, dim=-1)
    max_abs = (logprob_gpu - logprob_cpu).abs().max().item()
    assert max_abs < 1e-3, f"max abs diff {max_abs} > 1e-3"
```

## 2. latency benchmarks

### 2.1 Jacobi vs ngram (TSK_035)
- per-step draft latency comparison
- target: Jacobi K=7 within ~120% of ngram K=7 (CPU 비용 인정)

### 2.2 AMX draft head per-step latency (TSK_036)
- target: ≤ 5 ms / batch (B=32)
- microbench on prod (Sapphire Rapids)

## 3. acceptance rate (TSK_037)

### 3.1 per-workload acceptance rate

| workload | suffix α | ngram α | cpu_amx α (target) |
|---|---:|---:|---:|
| sonnet | (high) | (medium) | TBD |
| chat | 81.2% (SUB_011) | medium | **≥ 60%** |
| code | high (K=7) | high | TBD |

### 3.2 per-workload best-source rule validation
- compare: (a) 4-source AGSD with auto-select, (b) suffix only, (c) ngram only, (d) cpu_amx only
- target: 4-source AGSD ≥ best of (b,c,d) for each workload

## 4. e2e — TSK_037 measurement

```bash
bash /tmp/run_canonical_agsd_500p_multi_source.sh

# expected outputs:
#   chat: cpu_amx selected, α ~60%+, tps +10-15%
#   sonnet: suffix selected, no regression
#   code: ngram selected, no regression
#   accuracy gate: per-token logprob max abs diff < 1e-3
```

## 5. accuracy gate (CLAUDE.md 운영 해석)
- per-token logprob max abs diff < 1e-3
- token-level 일치 informational
- sequence PPL relative diff < 1%

## 6. util capture
- monitor.py background, CPU-side draft 의 thread placement check
- cpu_amx 활성 시: 0.5B forward 가 cpu 80-99 의 worker pool 에서 실행
