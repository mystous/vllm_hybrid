# SUB_062 — GPU Direct + lockfree queue CPU/GPU

> **parent**: TSK_020 / 카테고리 D (HPC classic)
> **status**: 대기 (plan only)
> **effort**: medium-large (3-5 일)
> **CPU% target**: marginal (latency 영역 효과)
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §4

---

## 1. Mechanism

CPU/GPU 영역 data exchange 영역 lock-free + zero-copy 영역 영역 최적화:

1. **GPU Direct Storage** — pinned mem 영역 zero-copy H2D/D2H (영역 cudaMemcpyAsync 영역 영역 영역 영역)
2. **SPSC (Single-Producer Single-Consumer) lockfree queue** — CPU draft / spec result 영역 GPU thread 영역 영역 영역 영역 lock 영역 없는 영역 ring buffer
3. **RDMA-style direct memory** — host-side pinned + GPU-side memory mapping (UVA)

```
[CPU producer thread] → [SPSC lockfree ring buffer (pinned mem)]
                                                    ↓ (GPU-side mmap, no copy)
                                          [GPU consumer thread]
```

영역 SUB_050~053 (Eagle/Medusa/Lookahead CPU draft 영역) 영역 SUB_058 (radix prefix) 영역 CPU→GPU transfer overhead 영역 큰 lever 영역 영역 영역 영역 가속 효과 영역 영역 영역.

## 2. 출처

| 자료 | 위치 |
|---|---|
| NVIDIA GPU Direct | https://developer.nvidia.com/gpudirect |
| Boost.Lockfree | https://www.boost.org/doc/libs/release/doc/html/lockfree.html |
| folly ProducerConsumerQueue | GitHub `facebook/folly` |
| UVA / pinned mem | CUDA C Programming Guide §3.2.5 |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `csrc/cpu/lockfree_queue.h` (신규) | SPSC lockfree ring buffer (atomic 영역 영역 fence) |
| `csrc/cpu/gpu_direct.cpp` (신규) | pinned mem + cudaHostAlloc + cudaHostGetDevicePointer |
| `vllm/v1/spec_decode/utils.py` | Python wrapper (pybind11) |
| `vllm/v1/worker/gpu_model_runner.py` | spec draft transfer path 영역 본 영역 사용 |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | GPU Direct 영역 지원 여부 확인 (CUDA version, GPU 영역 capability) | 0.5 일 |
| Phase 1 | SPSC lockfree ring buffer (boost.lockfree 또는 자체) | 1 일 |
| Phase 2 | pinned mem + UVA mapping | 1 일 |
| Phase 3 | pybind11 wrapper + Python integration | 0.5 일 |
| Phase 4 | benchmark (transfer latency vs cudaMemcpyAsync) | 1 일 |
| 총 | | **4 일** |

## 5. CPU% target / throughput 가설

- CPU%: marginal (lock-free queue 자체 영역 영역 영역 영역)
- latency 영역 효과: cudaMemcpyAsync 영역 ~10-50μs/transfer 영역 영역 영역 lock-free + UVA 영역 ~1-5μs
- throughput: spec decode 영역 transfer overhead 영역 큰 lever (SUB_050/051/058) 영역 결합 시 +5-10%
- 단독 측정 시 throughput 영향 ≈ 0

## 6. Risk

| 위험 | 완화 |
|---|---|
| GPU Direct Storage 영역 specific HW 영역 (Hopper 영역 일부 영역 지원 영역 변동) | 본 환경 (H100) 영역 지원 영역 영역 확인 |
| UVA 영역 mem footprint 영역 host mem 영역 차지 | 영역 lock-free queue size 영역 작게 |
| pybind11 영역 build 영역 추가 영역 영역 | 본 영역 영역 csrc 영역 build 영역 영역 영역 |

## 7. Dependencies

- 본 영역 영역 SUB_050~053/058 영역 lever 영역 land 영역 후 영역 본 영역 의미 영역 (standalone benefit 영역 영역)
- CUDA Toolkit + GPU Direct enable

## 8. Acceptance criteria

- [ ] lock-free queue throughput ≥ 10M op/sec/thread
- [ ] H2D transfer latency ≤ 5μs (small payload < 64 KB)
- [ ] SUB_050/058 영역 결합 시 throughput +3% 이상
- [ ] CPU% impact ≤ +2%
