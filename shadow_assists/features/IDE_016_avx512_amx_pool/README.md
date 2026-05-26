# IDE_016 — AVX-512 + AMX CPU SIMD Acceleration Pool

> **scope**: tokenizer / sampling / draft head matmul / prefill assist 의 CPU-side acceleration.
> **paper angle**: SIMD native ISA × spec decode 의 CPU-side hot path. AMX tile matmul + spec draft head 의 첫 production-grade 측정 — 직접 대응 논문 없음.
> **parent**: TSK_020 / IDE_015 Phase A 결과 기반.
> **status**: ✅ design + skeleton 작성 완료 / ⚠ build + 검증 별도 turn 필요.

---

## 1. 이론적 배경

### 1.1 Phase A 측정의 lever 정량 (SUB_161 input)

| Phase A finding | implication for IDE_016 |
|---|---|
| **trident TP0 의 44.3% CPU 시간 = sampler.py** (SUB_161) | TSK_025 (AVX-512 sampling) 의 가장 큰 lift potential |
| logits_processor 27% + penalties 23% | TSK_025 의 logit chain 통합 lever |
| update_async_output_token_ids 17.6% | TSK_024 tokenizer/detokenizer 의 적용 영역 |
| **AMX BF16 22.05 TFLOPS peak** (SUB_106) | TSK_026 AMX draft head 의 lower bound throughput |
| **10.24 TFLOPS available (N=32 pinned)** (SUB_117) | total CPU compute budget for IDE_016 |

### 1.2 4 sub-task 분류

| TSK | 영역 | scope | priority |
|---|---|---|---|
| TSK_024 | AVX-512 batch tokenizer / detokenizer | BPE / SentencePiece search vectorize | ★★ |
| **TSK_025** | **AVX-512 sampling + logit processor** | top-k / top-p / temp / penalty vectorize | **★★★ (SUB_161 의 44% lever)** |
| TSK_026 | AMX tile-based draft head matmul | Qwen 0.5B/1.5B forward on CPU AMX | ★★ |
| TSK_027 | AMX medium-context CPU prefill assist | 512-2K context 의 GPU prefill 보조 | ★ |

→ TSK_025 가 본 IDE 의 첫 진입점 (SUB_161 의 가장 큰 CPU phase share).

---

## 2. 구현 방향 (skeleton overview)

### 2.1 AVX-512 sampling kernel (TSK_025)

```cpp
// src/avx512_sampling/topk_topp_kernel.cpp
#include <immintrin.h>

// top-k via partial sort with AVX-512 gather + max reduction
void topk_avx512_bf16(const __bf16* logits, int vocab, int batch, int K,
                     int32_t* indices_out, float* values_out);

// top-p (nucleus) via prefix sum + threshold scan
void topp_avx512_bf16(const __bf16* sorted_probs, int vocab, int batch, float p,
                     int32_t* cutoff_out);
```

### 2.2 AMX draft head matmul (TSK_026)

```cpp
// src/amx_matmul/amx_qwen_draft.cpp
#include <immintrin.h>

// AMX tile config (rows/cols/strides) for Qwen 0.5B/1.5B draft model
struct AmxTileCfg {
    uint16_t bytes_per_row;   // 64 (BF16 × 32 cols)
    uint16_t rows;            // 16
    // ... up to 8 tile descriptors
};

// AMX BF16 matmul: A[M,K] × B[K,N] -> C[M,N]
void amx_matmul_bf16(const __bf16* A, const __bf16* B, float* C,
                    int M, int K, int N);
```

### 2.3 Python binding layer

```python
# src/_python/avx512_sampling.py
import torch
from . import _avx512_sampling_cpp  # C++ extension

def topk_topp(logits: torch.Tensor, k: int, p: float) -> torch.Tensor:
    """AVX-512 accelerated top-k/top-p sampling. CPU tensor expected."""
    assert logits.device.type == "cpu"
    assert logits.dtype == torch.bfloat16
    return _avx512_sampling_cpp.topk_topp(logits, k, p)
```

### 2.4 vLLM integration hook

vLLM `vllm/v1/sample/sampler.py` 의 `_sample` 함수 (SUB_161 의 27.9% CPU 시간 hotspot) 에 fallback path 삽입.

```python
# vllm/v1/sample/sampler.py:3521 _sample patch (design only)
if envs.VLLM_USE_AVX512_SAMPLING and logits.device.type == "cpu":
    from vllm_hybrid_kernels import avx512_sampling
    return avx512_sampling.topk_topp(logits, k=top_k, p=top_p)
# else: existing GPU/CPU path
```

---

## 3. Hardware target + Build

| target | requirement |
|---|---|
| AVX-512 | Intel/AMD CPUs from 2016+ — 개발 머신 (Alder Lake i9-12900KF) AVX-512 fuse-off 주의 (CLAUDE.md), prod (Sapphire Rapids 8480+) full support |
| AMX (BF16 / INT8 / TILE) | **Sapphire Rapids 이상만** (개발 머신 미지원, prod 머신 native) |
| compiler | g++-12 이상 (`_Float16` + AMX intrinsic 지원), 또는 clang 14+ |

Build:
```bash
cd shadow_assists/features/IDE_016_avx512_amx_pool/src
mkdir build && cd build
cmake .. -DCMAKE_CXX_FLAGS="-mavx512f -mavx512bw -mavx512vl -mavx512bf16 -mamx-bf16 -mamx-int8 -mamx-tile -march=sapphirerapids -O3"
make -j 32
```

---

## 4. 검증 (test.md 참조)

- correctness: GPU baseline 과 logits/sampling 결과 비교 (per-token logprob max abs diff < 1e-3)
- latency: per-step sampling latency 측정 (target 1.5 ms vs current 3-5 ms)
- throughput: canonical AGSD-gated 측정 (target +5-10% via TSK_025 alone)

---

## 5. 다음 step (별도 turn 권장)

1. C++ kernel intrinsic 구현 (현재는 stub) → build + unit test
2. Python binding 구현 (pybind11)
3. vLLM patch 작성 + 통합 test
4. canonical 측정 (SUB_160 protocol)

---

## 6. 파일 구조

```
IDE_016_avx512_amx_pool/
├── README.md         # 이 파일
├── CLAUDE.md         # Claude 가 본 feature 구현 시 알아야 할 것
├── task.md           # TSK_024~027 단계별 구현 내용
├── test.md           # 테스트 코드 / 방법 / 예상 결과
├── design/
│   ├── avx512_sampling_design.md
│   ├── amx_matmul_design.md
│   └── vllm_integration_design.md
├── src/
│   ├── avx512_sampling/
│   │   ├── topk_topp_kernel.cpp
│   │   ├── logit_processor.cpp
│   │   └── penalty_ops.cpp
│   ├── amx_matmul/
│   │   ├── amx_qwen_draft.cpp
│   │   └── tile_config.cpp
│   └── _python/
│       └── avx512_sampling.py
└── tests/
    ├── test_topk_topp.py
    ├── test_amx_matmul.py
    └── test_correctness_vs_gpu.py
```
