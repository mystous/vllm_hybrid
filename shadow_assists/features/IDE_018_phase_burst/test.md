# test.md — IDE_018 테스트 계획

## 1. correctness

### 1.1 phase detection accuracy (TSK_031)

```python
# tests/test_phase_detection.py
import torch
import pytest

def test_phase_signal_emitted_per_layer():
    """Run a known model, verify cuda event hooks fire for each layer."""
    # Set up known workload: Qwen 1.5B × 10 steps
    # Hook capture: phase boundary timestamps
    signals = capture_phase_signals(model="Qwen/Qwen2.5-1.5B", steps=10)
    # Expect: 10 steps × (attn + linear + sample + tp_ar) signals
    expected_phases = 10 * 4
    assert len(signals) >= expected_phases * 0.9  # tolerate some skip

def test_phase_signal_latency():
    """Phase signal latency < 50 μs (target from task.md)."""
    latencies = measure_signal_dispatch_latency(n=1000)
    p50 = sorted(latencies)[500]
    p99 = sorted(latencies)[990]
    print(f"phase signal latency: p50={p50:.1f} μs, p99={p99:.1f} μs")
    assert p50 < 50, f"p50 {p50} μs > target 50 μs"
```

### 1.2 task pool correctness (TSK_032/033)

각 task 의 output 이 baseline (phase-burst OFF) 과 동일해야 함:

```python
def test_task_B_detokenize_matches_baseline():
    """Detokenize task output ≡ vLLM 의 default detokenize."""
    output_baseline = run_canonical(use_phase_burst=False, mix="balanced", n_prompts=10)
    output_burst = run_canonical(use_phase_burst=True, mix="balanced", n_prompts=10)
    # CLAUDE.md 운영 해석: 분포·의도 수준
    # 본 case 는 detokenize 만이므로 token-level 동일성 기대
    for prompt_idx, (b, p) in enumerate(zip(output_baseline, output_burst)):
        assert b == p, f"prompt {prompt_idx} detokenize mismatch"
```

## 2. latency benchmarks

### 2.1 CUDA event hook overhead (TSK_031)
- target: hook overhead < 1% step time
- microbench: vLLM forward 10 steps with/without hooks

```python
def test_hook_overhead():
    t_no_hook = bench_forward(use_hooks=False, n_steps=100)
    t_with_hook = bench_forward(use_hooks=True, n_steps=100)
    overhead = (t_with_hook - t_no_hook) / t_no_hook
    assert overhead < 0.01, f"hook overhead {100*overhead:.2f}% > 1%"
```

## 3. e2e — paper main result (TSK_034)

### 3.1 CPU util 4.1% → 30%+ measurement

```bash
# canonical setup
bash /tmp/run_canonical_agsd_500p_phase_burst.sh

# expected outputs:
#   CPU util avg: 30%+ ★
#   Throughput vs SUB_098 baseline: +10-20% ★
#   GPU util delta: +5-10pp (sustained activity)
```

### 3.2 paper Figure 5 input

| metric | baseline (SUB_098) | IDE_018 target | source |
|---|---:|---:|---|
| CPU util avg | 4.1% | **30%+** | monitor.py |
| Throughput (AGSD balanced 500p) | 5,474 | **6,021+** (+10%) | benchmark |
| GPU util avg (8 GPU) | 27.7% | 35-40% | monitor.py |
| per-step phase duration | (TBD) | (TBD) | TSK_031 hook |

## 4. ablation (paper §4)

| config | CPU util | tps | source |
|---|---:|---:|---|
| baseline (no fill) | 4.1% | 5,474 | SUB_098 |
| + SUB_112 task-level pin (no phase) | 16% | 5,700 | SUB_117/160 |
| + TSK_025 alone (AVX-512 sample) | 16-20% | 5,750-5,900 | est. |
| + TSK_034 full phase-burst | **30%+** | **6,000-6,500** | IDE_018 measurement |

## 5. accuracy gate
- CLAUDE.md 운영 해석: per-token logprob max abs diff < 1e-3
- token-level 일치 informational
- sequence PPL relative diff < 1%

## 6. util capture (필수)
- monitor.py background, 0.2s interval (고해상도)
- per-CPU 분포 capture (어느 core 가 task pool 의 어떤 task 를 실행하는지)
- per-phase signal trace (post-run analysis)
