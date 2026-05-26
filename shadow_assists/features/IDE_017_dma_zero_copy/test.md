# test.md — IDE_017 테스트 계획

## 1. correctness

### 1.1 Pinned pool consistency (TSK_028)

```python
# tests/test_pinned_pool.py
import torch
from vllm_hybrid_dma import PinnedPool

def test_alloc_free_lifecycle():
    pool = PinnedPool(total_size=1024*1024*1024)  # 1 GB pool
    ptrs = []
    for sz in [4096, 65536, 1048576]:
        for _ in range(10):
            ptrs.append(pool.alloc(sz))
    for p in ptrs:
        pool.free(p)
    assert pool.in_use_bytes() == 0

def test_dma_push_roundtrip():
    src = torch.randn(1024*1024, dtype=torch.bfloat16).pin_memory()
    dst = torch.empty_like(src, device="cuda:0")
    event = PinnedPool.push_async(src, dst)
    event.synchronize()
    assert torch.allclose(src.cuda(), dst)
```

### 1.2 Zero-copy coherence (TSK_029)

```python
def test_zero_copy_cpu_write_gpu_read():
    buf = ZeroCopyBuffer(size=4096)
    buf.cpu_view()[:] = 42.0  # CPU write
    # explicit sync (paper 측정 시 또는 fence)
    buf.sync()
    val = buf.gpu_view().mean().item()  # GPU read
    assert val == 42.0
```

### 1.3 Cold-KV decompress correctness (TSK_030)

```python
def test_cold_kv_dequant():
    bf16_orig = torch.randn(128, 5120, dtype=torch.bfloat16)
    int8_quant, scales = cold_kv_quantize(bf16_orig)  # INT8 quantize
    bf16_dequant = cold_kv_dequant_avx512(int8_quant, scales)
    rel_err = ((bf16_orig - bf16_dequant).abs() / bf16_orig.abs().clamp_min(1e-3)).max()
    assert rel_err < 0.05  # INT8 quant 의 reasonable bound
```

## 2. latency benchmarks

### 2.1 DMA push latency reproduce SUB_166

```bash
# verify: 4 KB → ~35 μs, 1 MB → ~60 μs, 64 MB → ~1251 μs
python tests/bench_dma_latency.py --reproduce-sub166
```

### 2.2 Zero-copy vs DMA crossover

```python
# small data (4 KB ~ 64 KB)
def bench_small_data_compare():
    sizes = [4*1024, 16*1024, 64*1024, 256*1024]
    for sz in sizes:
        t_dma = bench_dma_push(sz)
        t_zerocopy = bench_zerocopy_read(sz)
        print(f"size={sz}: dma={t_dma:.1f}us, zerocopy={t_zerocopy:.1f}us")
        # 4 KB ~ 256 KB 영역에서 zerocopy 가 net positive 예상
```

## 3. e2e throughput

### 3.1 TSK_028 alone (DMA pool)
- canonical AGSD-gated 통합
- vllm KV swap 경로에 적용
- target: 0~+2% (small lift expected)

### 3.2 TSK_030 (cold-KV decompress + DMA)
- canonical + long context workload
- target: TTFT −5-10% / steady throughput +1-3%

## 4. util capture
모든 measurement 에 monitor.py background attach. GPU SM util / PCIe bandwidth 캡처.

## 5. accuracy gate
- TSK_029: zero-copy coherence — explicit sync 후 GPU read 값 ≡ CPU written 값
- TSK_030: per-token logprob max abs diff < 1e-3 (INT8 cold-KV vs BF16 original)
