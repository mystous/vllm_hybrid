# AVX-512 / AMX CPU 최적화 구현 계획서

> **목표**: AVX-512 및 AMX를 활용하여 vLLM CPU 백엔드 성능을 극대화
> **환경**: Intel Xeon (AVX-512F, AVX-512BW, AVX-512DQ, AVX-512VL, AVX-512VNNI, AMX-BF16, AMX-INT8)
> **작성일**: 2026-02-03
> **업데이트**: 2026-02-03 - AMX 지원 추가

---

## 0. AMX (Advanced Matrix Extensions) 지원 상태

### 구현 완료 (2026-02-03)

| 파일 | 변경 내용 |
|------|-----------|
| `vllm/platforms/intel_cpu_utils.py` | AMX 타일 권한 요청, oneDNN ISA 설정 |
| `vllm/executor/parallel_batch_executor.py` | IPEX AMX 모드 활성화 |

### AMX 동작 방식

```
CPU 감지 → amx_bf16/amx_int8 플래그 확인
    ↓
AMX 타일 권한 요청 (ARCH_REQ_XCOMP_PERM syscall)
    ↓
oneDNN ISA 설정 (ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX)
    ↓
IPEX FP32MathMode.BF16 설정
    ↓
모델 최적화 시 AMX 커널 자동 선택
```

### 환경변수 설정

```bash
# AMX 활성화 (Sapphire Rapids+)
export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

# AVX-512 전용 (AMX 미지원 시)
export ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI
```

---

## 목차

1. [현재 상태 분석](#1-현재-상태-분석)
2. [성능 병목점 식별](#2-성능-병목점-식별)
3. [구현 단계 개요](#3-구현-단계-개요)
4. [Phase 1: AVX-512 VNNI INT8 GEMM 커널](#4-phase-1-avx-512-vnni-int8-gemm-커널)
5. [Phase 2: Q8_0 양자화 지원](#5-phase-2-q8_0-양자화-지원)
6. [Phase 3: Decode GEMV 최적화](#6-phase-3-decode-gemv-최적화)
7. [Phase 4: 배치 처리 최적화](#7-phase-4-배치-처리-최적화)
8. [Phase 5: 메모리 대역폭 최적화](#8-phase-5-메모리-대역폭-최적화)
9. [테스트 및 벤치마크](#9-테스트-및-벤치마크)
10. [예상 성능 향상](#10-예상-성능-향상)

---

## 1. 현재 상태 분석

### 1.1 현재 CPU 백엔드 구조

```
vllm/
├── v1/
│   ├── attention/backends/
│   │   └── cpu_attn.py          # PyTorch SDPA 기반 Attention
│   └── worker/
│       ├── cpu_worker.py        # CPU 워커 (Intel 최적화 설정)
│       └── cpu_model_runner.py  # NUMA-aware KVCache 할당
├── platforms/
│   ├── cpu.py                   # CPU 플랫폼 감지
│   └── intel_cpu_utils.py       # Intel 유틸리티
└── csrc/cpu/
    ├── torch_bindings.cpp       # PyTorch 바인딩
    ├── attention.cpp            # PagedAttention 커널
    ├── quant.cpp                # 양자화 커널
    └── cpu_types_x86.hpp        # AVX-512 SIMD 타입
```

### 1.2 현재 SIMD 활용 현황

| 연산 | 현재 구현 | AVX-512 활용 |
|------|----------|-------------|
| Attention (Prefill) | IPEX PagedAttention / PyTorch SDPA | 부분적 |
| Attention (Decode) | PyTorch SDPA (batched) | 미활용 |
| Linear (GEMM) | oneDNN/MKL | 자동 |
| Activation | PyTorch ops | Inductor 의존 |
| LayerNorm | PyTorch ops | Inductor 의존 |

### 1.3 문제점

1. **Decode 단계**: PyTorch SDPA는 AVX-512 VNNI 미활용
2. **양자화 미지원**: BF16만 지원, INT8 양자화 없음
3. **GEMV 최적화 부족**: Decode의 [1, hidden] × [hidden, vocab] 비효율
4. **배치 처리**: 소규모 배치에서 SIMD 활용도 저하

---

## 2. 성능 병목점 식별

### 2.1 Decode 단계 프로파일링 (예상)

```
Decode Token Generation (1 token):
├── Attention: 35%
│   ├── Q×K^T:    15%  ← GEMV, 메모리 바운드
│   ├── Softmax:  5%   ← 벡터화 가능
│   └── Score×V: 15%  ← GEMV, 메모리 바운드
├── FFN: 45%
│   ├── Linear1:  20%  ← GEMV (4x expansion)
│   ├── Activation: 5% ← 벡터화 가능
│   └── Linear2:  20%  ← GEMV
├── LayerNorm: 10%
└── Other: 10%
```

### 2.2 주요 병목

| 병목 | 원인 | 해결책 |
|------|------|--------|
| **GEMV 메모리 바운드** | Weight 로드가 지배적 | INT8 양자화로 대역폭 2배 절약 |
| **Softmax 직렬화** | 루프 기반 구현 | AVX-512 벡터화 |
| **배치 비효율** | 작은 배치에서 SIMD 미활용 | 다중 시퀀스 병렬 처리 |

---

## 3. 구현 단계 개요

```
Phase 1: AVX-512 VNNI INT8 GEMM 커널 (1주)
    ↓
Phase 2: Q8_0 양자화 지원 (3일)
    ↓
Phase 3: Decode GEMV 최적화 (1주)
    ↓
Phase 4: 배치 처리 최적화 (3일)
    ↓
Phase 5: 메모리 대역폭 최적화 (3일)
    ↓
테스트 및 벤치마크 (2일)
```

---

## 4. Phase 1: AVX-512 VNNI INT8 GEMM 커널

### 4.1 목표

AVX-512 VNNI의 `vpdpbusd` 명령어를 활용한 INT8 GEMM 구현

### 4.2 VNNI 명령어 이해

```cpp
// vpdpbusd: 4개의 INT8 곱셈 + INT32 누적을 1사이클에
// dst[i] += src1[4i:4i+3] · src2[4i:4i+3]
__m512i _mm512_dpbusd_epi32(__m512i src, __m512i a, __m512i b);

// 512비트 레지스터 = 64개 INT8 = 16개 INT32 결과
// → 64 MACs/cycle per register
```

### 4.3 파일 구조

```
csrc/cpu/
├── gemm_vnni.hpp          # [신규] VNNI GEMM 헤더
├── gemm_vnni.cpp          # [신규] VNNI GEMM 구현
├── torch_bindings.cpp     # [수정] 바인딩 추가
└── cpu_types_x86.hpp      # [수정] INT8 벡터 타입 추가
```

### 4.4 구현: cpu_types_x86.hpp 확장

```cpp
// csrc/cpu/cpu_types_x86.hpp에 추가

#ifdef __AVX512VNNI__

// INT8 벡터 타입 (64개 INT8 = 512비트)
struct INT8Vec64 {
    __m512i reg;

    INT8Vec64() = default;
    explicit INT8Vec64(__m512i r) : reg(r) {}

    // 메모리에서 로드
    explicit INT8Vec64(const int8_t* ptr) {
        reg = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
    }

    // 메모리에 저장
    void save(int8_t* ptr) const {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), reg);
    }

    // 비템포럴 저장 (캐시 오염 방지)
    void stream(int8_t* ptr) const {
        _mm512_stream_si512(reinterpret_cast<__m512i*>(ptr), reg);
    }

    static constexpr int get_elem_num() { return 64; }
};

// UINT8 벡터 (가중치용 - unsigned)
struct UINT8Vec64 {
    __m512i reg;

    UINT8Vec64() = default;
    explicit UINT8Vec64(__m512i r) : reg(r) {}
    explicit UINT8Vec64(const uint8_t* ptr) {
        reg = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
    }

    static constexpr int get_elem_num() { return 64; }
};

// INT32 누적 벡터 (16개 INT32 = 512비트)
struct INT32Vec16 {
    __m512i reg;

    INT32Vec16() : reg(_mm512_setzero_si512()) {}
    explicit INT32Vec16(__m512i r) : reg(r) {}

    // 수평 합산 (16개 INT32 → 1개 INT32)
    int32_t reduce_add() const {
        // 256비트로 축소
        __m256i low = _mm512_castsi512_si256(reg);
        __m256i high = _mm512_extracti64x4_epi64(reg, 1);
        __m256i sum256 = _mm256_add_epi32(low, high);

        // 128비트로 축소
        __m128i low128 = _mm256_castsi256_si128(sum256);
        __m128i high128 = _mm256_extracti128_si256(sum256, 1);
        __m128i sum128 = _mm_add_epi32(low128, high128);

        // 최종 합산
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        return _mm_cvtsi128_si32(sum128);
    }

    // VNNI dot product: this += a * b (4개씩 묶어서 계산)
    void dpbusd(const UINT8Vec64& a, const INT8Vec64& b) {
        reg = _mm512_dpbusd_epi32(reg, a.reg, b.reg);
    }

    static constexpr int get_elem_num() { return 16; }
};

#endif // __AVX512VNNI__
```

### 4.5 구현: gemm_vnni.hpp

```cpp
// csrc/cpu/gemm_vnni.hpp
#pragma once

#include <torch/torch.h>
#include "cpu_types.hpp"

#ifdef __AVX512VNNI__

namespace vllm {
namespace cpu {

// INT8 GEMM: C[M,N] = A[M,K] × B[K,N]
// A: INT8 (activation, signed)
// B: UINT8 (weight, unsigned, pretransposed to [N,K])
// C: INT32 (output, 나중에 스케일링)
void int8_gemm_vnni(
    const int8_t* A,      // [M, K] row-major
    const uint8_t* B,     // [N, K] row-major (transposed)
    int32_t* C,           // [M, N] row-major
    int M, int N, int K,
    int lda, int ldb, int ldc
);

// INT8 GEMV: y[N] = x[K] × W[K,N]
// 디코드 단계용 최적화 (M=1)
void int8_gemv_vnni(
    const int8_t* x,      // [K]
    const uint8_t* W,     // [N, K] row-major (transposed)
    int32_t* y,           // [N]
    int N, int K
);

// 스케일링 및 BF16 변환
void dequantize_int32_to_bf16(
    const int32_t* input,
    at::BFloat16* output,
    float scale_a,
    float scale_w,
    int size
);

}  // namespace cpu
}  // namespace vllm

#endif // __AVX512VNNI__
```

### 4.6 구현: gemm_vnni.cpp

```cpp
// csrc/cpu/gemm_vnni.cpp
#include "gemm_vnni.hpp"
#include <immintrin.h>
#include <omp.h>

#ifdef __AVX512VNNI__

namespace vllm {
namespace cpu {

// 마이크로커널: 6x16 타일 (6행 × 16열)
// 레지스터 사용: 6개 누적 + 2개 로드 = 8개 ZMM
static inline void gemm_6x16_vnni_kernel(
    const int8_t* A,      // [6, K]
    const uint8_t* B,     // [16, K]
    int32_t* C,           // [6, 16]
    int K,
    int lda, int ldb, int ldc
) {
    // 누적 레지스터 초기화 (6행 × 1열 of 16-wide)
    __m512i c0 = _mm512_setzero_si512();
    __m512i c1 = _mm512_setzero_si512();
    __m512i c2 = _mm512_setzero_si512();
    __m512i c3 = _mm512_setzero_si512();
    __m512i c4 = _mm512_setzero_si512();
    __m512i c5 = _mm512_setzero_si512();

    // K를 64 단위로 처리 (VNNI는 4개씩 묶음)
    int k = 0;
    for (; k + 63 < K; k += 64) {
        // A의 6행 로드 (각 64 INT8)
        __m512i a0 = _mm512_loadu_si512(A + 0*lda + k);
        __m512i a1 = _mm512_loadu_si512(A + 1*lda + k);
        __m512i a2 = _mm512_loadu_si512(A + 2*lda + k);
        __m512i a3 = _mm512_loadu_si512(A + 3*lda + k);
        __m512i a4 = _mm512_loadu_si512(A + 4*lda + k);
        __m512i a5 = _mm512_loadu_si512(A + 5*lda + k);

        // B의 열들과 dot product
        // 각 열에 대해 broadcast + dpbusd
        for (int n = 0; n < 16; ++n) {
            __m512i bn = _mm512_loadu_si512(B + n*ldb + k);

            // VNNI: c[row][n] += sum(a[row][k:k+64] * b[n][k:k+64])
            // 실제로는 16개의 4-way dot product
            __m512i bn_bcast = _mm512_set1_epi32(
                *reinterpret_cast<const int32_t*>(B + n*ldb + k)
            );

            // 더 효율적인 구현: outer product 형태
            // 여기서는 간략화된 버전
        }
    }

    // 잔여 K 처리 (생략, 실제 구현 필요)

    // 결과 저장
    _mm512_storeu_si512(C + 0*ldc, c0);
    _mm512_storeu_si512(C + 1*ldc, c1);
    _mm512_storeu_si512(C + 2*ldc, c2);
    _mm512_storeu_si512(C + 3*ldc, c3);
    _mm512_storeu_si512(C + 4*ldc, c4);
    _mm512_storeu_si512(C + 5*ldc, c5);
}

void int8_gemm_vnni(
    const int8_t* A,
    const uint8_t* B,
    int32_t* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    constexpr int MR = 6;   // 마이크로커널 행
    constexpr int NR = 16;  // 마이크로커널 열

    // 캐시 블로킹 파라미터
    constexpr int MC = 72;   // L2 캐시용 M 블록
    constexpr int NC = 256;  // L2 캐시용 N 블록
    constexpr int KC = 256;  // L1 캐시용 K 블록

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i += MC) {
        for (int j = 0; j < N; j += NC) {
            int mb = std::min(MC, M - i);
            int nb = std::min(NC, N - j);

            // K 방향 블로킹
            for (int p = 0; p < K; p += KC) {
                int kb = std::min(KC, K - p);

                // 마이크로커널 호출
                for (int ii = 0; ii < mb; ii += MR) {
                    for (int jj = 0; jj < nb; jj += NR) {
                        int mr = std::min(MR, mb - ii);
                        int nr = std::min(NR, nb - jj);

                        if (mr == MR && nr == NR) {
                            gemm_6x16_vnni_kernel(
                                A + (i+ii)*lda + p,
                                B + (j+jj)*ldb + p,
                                C + (i+ii)*ldc + (j+jj),
                                kb, lda, ldb, ldc
                            );
                        } else {
                            // 경계 처리 (간략화)
                        }
                    }
                }
            }
        }
    }
}

// GEMV 최적화 (M=1)
void int8_gemv_vnni(
    const int8_t* x,
    const uint8_t* W,
    int32_t* y,
    int N, int K
) {
    // K를 64 단위로 패딩
    int K_aligned = (K + 63) & ~63;

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        __m512i acc = _mm512_setzero_si512();

        // 64 INT8씩 처리
        int k = 0;
        for (; k + 63 < K; k += 64) {
            __m512i vx = _mm512_loadu_si512(x + k);
            __m512i vw = _mm512_loadu_si512(W + n*K + k);

            // VNNI dot product
            acc = _mm512_dpbusd_epi32(acc, vw, vx);
        }

        // 잔여 처리 (마스크 사용)
        if (k < K) {
            __mmask64 mask = (1ULL << (K - k)) - 1;
            __m512i vx = _mm512_maskz_loadu_epi8(mask, x + k);
            __m512i vw = _mm512_maskz_loadu_epi8(mask, W + n*K + k);
            acc = _mm512_dpbusd_epi32(acc, vw, vx);
        }

        // 수평 합산
        y[n] = _mm512_reduce_add_epi32(acc);
    }
}

// Dequantization
void dequantize_int32_to_bf16(
    const int32_t* input,
    at::BFloat16* output,
    float scale_a,
    float scale_w,
    int size
) {
    float combined_scale = scale_a * scale_w;
    __m512 vscale = _mm512_set1_ps(combined_scale);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i += 16) {
        // INT32 → FP32
        __m512i vi = _mm512_loadu_si512(input + i);
        __m512 vf = _mm512_cvtepi32_ps(vi);

        // 스케일 적용
        vf = _mm512_mul_ps(vf, vscale);

        // FP32 → BF16
        __m256i vbf16 = _mm512_cvtneps_pbh(vf);  // AVX512_BF16 필요

        // 저장
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(output + i),
            vbf16
        );
    }
}

}  // namespace cpu
}  // namespace vllm

#endif // __AVX512VNNI__
```

### 4.7 PyTorch 바인딩: torch_bindings.cpp 수정

```cpp
// csrc/cpu/torch_bindings.cpp에 추가

#include "gemm_vnni.hpp"

void int8_gemm_vnni_torch(
    torch::Tensor& C,           // [M, N] int32
    const torch::Tensor& A,     // [M, K] int8
    const torch::Tensor& B,     // [N, K] uint8 (transposed)
    double scale_a,
    double scale_w
) {
#ifdef __AVX512VNNI__
    TORCH_CHECK(A.scalar_type() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.scalar_type() == torch::kUInt8, "B must be uint8");
    TORCH_CHECK(C.scalar_type() == torch::kInt32, "C must be int32");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    vllm::cpu::int8_gemm_vnni(
        A.data_ptr<int8_t>(),
        B.data_ptr<uint8_t>(),
        C.data_ptr<int32_t>(),
        M, N, K,
        K, K, N
    );
#else
    TORCH_CHECK(false, "AVX512-VNNI not available");
#endif
}

// 바인딩 등록
TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cpu), ops) {
    // 기존 바인딩...

    // INT8 GEMM (VNNI)
    ops.def("int8_gemm_vnni(Tensor! C, Tensor A, Tensor B, float scale_a, float scale_w) -> ()");
    ops.impl("int8_gemm_vnni", torch::kCPU, &int8_gemm_vnni_torch);
}
```

### 4.8 CMakeLists.txt 수정

```cmake
# csrc/cpu/CMakeLists.txt에 추가

# AVX-512 VNNI 컴파일 플래그
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(AVX512_VNNI_FLAGS "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni")
endif()

# VNNI 소스 파일
set(VNNI_SOURCES
    gemm_vnni.cpp
)

# 컴파일
foreach(src ${VNNI_SOURCES})
    set_source_files_properties(${src} PROPERTIES
        COMPILE_FLAGS "${AVX512_VNNI_FLAGS}"
    )
endforeach()
```

---

## 5. Phase 2: Q8_0 양자화 지원

### 5.1 목표

llama.cpp의 Q8_0 양자화 포맷을 vLLM에서 지원

### 5.2 Q8_0 포맷 이해

```
Q8_0 블록 구조 (34 바이트):
┌─────────────────────────────────────┐
│ scale (float16) │ quants (int8[32]) │
│     2 bytes     │     32 bytes      │
└─────────────────────────────────────┘

dequantize: value = quant * scale
```

### 5.3 파일 구조

```
vllm/
├── model_executor/
│   └── layers/
│       └── quantization/
│           └── q8_0.py           # [신규] Q8_0 양자화 클래스
└── csrc/cpu/
    └── quant_q8_0.cpp            # [신규] Q8_0 커널
```

### 5.4 구현: q8_0.py

```python
# vllm/model_executor/layers/quantization/q8_0.py

import torch
from torch import nn
from typing import Optional, Tuple
from vllm import _custom_ops as ops

class Q8_0Config:
    """Q8_0 양자화 설정"""

    BLOCK_SIZE = 32  # llama.cpp Q8_0 블록 크기

    @classmethod
    def get_name(cls) -> str:
        return "q8_0"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # CPU용


class Q8_0LinearMethod:
    """Q8_0 양자화 Linear 레이어"""

    def __init__(self, quant_config: Q8_0Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict:
        # 블록 수 계산
        num_blocks = (input_size + Q8_0Config.BLOCK_SIZE - 1) // Q8_0Config.BLOCK_SIZE

        # 양자화된 가중치
        qweight = torch.empty(
            output_size, num_blocks, Q8_0Config.BLOCK_SIZE,
            dtype=torch.int8
        )

        # 스케일 (블록당 1개)
        scales = torch.empty(
            output_size, num_blocks,
            dtype=torch.float16
        )

        return {
            "qweight": qweight,
            "scales": scales,
        }

    def apply_weights(
        self,
        weights: dict,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [batch, seq_len, input_size] 또는 [tokens, input_size]
        output: [batch, seq_len, output_size] 또는 [tokens, output_size]
        """
        qweight = weights["qweight"]  # [out, blocks, 32]
        scales = weights["scales"]    # [out, blocks]

        # 입력 양자화
        x_quant, x_scale = self._quantize_activation(x)

        # INT8 GEMM
        output = ops.q8_0_linear(
            x_quant,      # [*, input_size] int8
            qweight,      # [output_size, blocks, 32] int8
            scales,       # [output_size, blocks] fp16
            x_scale,      # [*] fp32
        )

        if bias is not None:
            output = output + bias

        return output

    def _quantize_activation(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """동적 INT8 양자화"""
        # 절대값 최대
        x_abs_max = x.abs().max(dim=-1, keepdim=True).values
        x_abs_max = x_abs_max.clamp(min=1e-5)

        # 스케일 계산
        scale = 127.0 / x_abs_max

        # 양자화
        x_quant = (x * scale).round().clamp(-128, 127).to(torch.int8)

        # 역스케일 (dequant용)
        inv_scale = x_abs_max / 127.0

        return x_quant, inv_scale.squeeze(-1)


class Q8_0Linear(nn.Module):
    """Q8_0 양자화 Linear 모듈"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # 양자화된 파라미터
        num_blocks = (input_size + 31) // 32

        self.register_buffer(
            "qweight",
            torch.zeros(output_size, num_blocks, 32, dtype=torch.int8)
        )
        self.register_buffer(
            "scales",
            torch.zeros(output_size, num_blocks, dtype=torch.float16)
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(output_size, dtype=torch.bfloat16)
            )
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "Q8_0Linear":
        """FP32/BF16 Linear를 Q8_0으로 변환"""
        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None

        q8_linear = cls(in_features, out_features, has_bias)

        # 가중치 양자화
        weight = linear.weight.data.float()  # [out, in]

        num_blocks = (in_features + 31) // 32

        for i in range(num_blocks):
            start = i * 32
            end = min(start + 32, in_features)
            block = weight[:, start:end]

            # 블록별 스케일
            block_max = block.abs().max(dim=1).values.clamp(min=1e-5)
            scale = block_max / 127.0

            # 양자화
            block_quant = (block / scale.unsqueeze(1)).round().clamp(-128, 127)

            q8_linear.qweight[:, i, :end-start] = block_quant.to(torch.int8)
            q8_linear.scales[:, i] = scale.to(torch.float16)

        if has_bias:
            q8_linear.bias = linear.bias.data.to(torch.bfloat16)

        return q8_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ops.q8_0_linear(
            x, self.qweight, self.scales, self.bias
        )
```

### 5.5 구현: quant_q8_0.cpp

```cpp
// csrc/cpu/quant_q8_0.cpp

#include <torch/torch.h>
#include <immintrin.h>
#include <omp.h>

#ifdef __AVX512VNNI__

namespace {

// Q8_0 Linear: y = x @ W^T
// x: [M, K] bfloat16/float
// W: [N, num_blocks, 32] int8
// scales: [N, num_blocks] float16
// y: [M, N] bfloat16
void q8_0_linear_impl(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const c10::optional<torch::Tensor>& bias
) {
    int M = input.size(0);
    int K = input.size(1);
    int N = qweight.size(0);
    int num_blocks = qweight.size(1);

    // 입력을 INT8로 동적 양자화
    auto input_float = input.to(torch::kFloat32);
    auto input_max = input_float.abs().max(-1, true).values.clamp_min(1e-5f);
    auto input_scale = 127.0f / input_max;
    auto input_quant = (input_float * input_scale)
        .round()
        .clamp(-128, 127)
        .to(torch::kInt8);
    auto input_inv_scale = input_max / 127.0f;

    // 출력 버퍼
    auto output_int32 = torch::zeros({M, N}, torch::kInt32);

    const int8_t* x_ptr = input_quant.data_ptr<int8_t>();
    const int8_t* w_ptr = qweight.data_ptr<int8_t>();
    const at::Half* s_ptr = scales.data_ptr<at::Half>();
    int32_t* out_ptr = output_int32.data_ptr<int32_t>();

    // VNNI GEMM
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            __m512i acc = _mm512_setzero_si512();

            for (int b = 0; b < num_blocks; ++b) {
                // 입력 블록 (32 INT8)
                __m256i vx = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(x_ptr + m*K + b*32)
                );
                __m512i vx_ext = _mm512_cvtepi8_epi16(vx);

                // 가중치 블록 (32 INT8)
                __m256i vw = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(w_ptr + (n*num_blocks + b)*32)
                );
                __m512i vw_ext = _mm512_cvtepi8_epi16(vw);

                // 16-bit 곱셈 후 32-bit 누적
                __m512i prod = _mm512_madd_epi16(vx_ext, vw_ext);
                acc = _mm512_add_epi32(acc, prod);
            }

            // 수평 합산
            out_ptr[m*N + n] = _mm512_reduce_add_epi32(acc);
        }
    }

    // Dequantization: output = int32_result * input_scale * weight_scales
    // 복잡한 스케일 적용 (블록별 스케일 고려)
    // 여기서는 간략화: 전체 스케일 적용
    auto output_float = output_int32.to(torch::kFloat32);

    // 입력 스케일 적용
    output_float = output_float * input_inv_scale;

    // 가중치 스케일 적용 (행별)
    auto weight_scale_sum = scales.sum(-1).to(torch::kFloat32);  // [N]
    output_float = output_float * weight_scale_sum.unsqueeze(0);

    // Bias 추가
    if (bias.has_value()) {
        output_float = output_float + bias.value().to(torch::kFloat32);
    }

    // BF16으로 변환
    output.copy_(output_float.to(torch::kBFloat16));
}

}  // namespace

// PyTorch 바인딩
void q8_0_linear(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const c10::optional<torch::Tensor>& bias
) {
    TORCH_CHECK(input.device().is_cpu(), "Input must be on CPU");
    TORCH_CHECK(qweight.scalar_type() == torch::kInt8, "Weight must be int8");

    q8_0_linear_impl(output, input, qweight, scales, bias);
}

#endif // __AVX512VNNI__
```

---

## 6. Phase 3: Decode GEMV 최적화

### 6.1 목표

디코드 단계의 [1, K] × [K, N] GEMV 연산 최적화

### 6.2 현재 문제점

```python
# 현재 cpu_attn.py의 디코드 (PyTorch SDPA)
output = F.scaled_dot_product_attention(query, key, value)
# → 내부적으로 일반 GEMM 호출, SIMD 활용 미흡
```

### 6.3 구현: decode_gemv.cpp

```cpp
// csrc/cpu/decode_gemv.cpp

#include <torch/torch.h>
#include <immintrin.h>
#include <omp.h>

namespace vllm {
namespace cpu {

// BF16 GEMV: y[N] = x[K] × W[K,N]
// 메모리 바운드 → 프리페치 + 비템포럴 로드 최적화
void bf16_gemv_avx512(
    const at::BFloat16* x,    // [K]
    const at::BFloat16* W,    // [N, K] row-major (transposed)
    at::BFloat16* y,          // [N]
    int N, int K
) {
    // K를 32 단위로 처리 (BF16 × 32 = 512비트)
    int K_aligned = K & ~31;

    #pragma omp parallel for schedule(static, 64)
    for (int n = 0; n < N; ++n) {
        __m512 acc = _mm512_setzero_ps();

        // 프리페치 힌트 (다음 행)
        if (n + 1 < N) {
            _mm_prefetch(W + (n+1)*K, _MM_HINT_T0);
        }

        int k = 0;
        for (; k < K_aligned; k += 32) {
            // BF16 로드 (32개)
            __m512i vx_raw = _mm512_loadu_si512(x + k);
            __m512i vw_raw = _mm512_loadu_si512(W + n*K + k);

            // BF16 → FP32 변환 (상위 16개)
            __m512 vx_hi = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(
                    _mm512_extracti64x4_epi64(vx_raw, 1)), 16)
            );
            __m512 vw_hi = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(
                    _mm512_extracti64x4_epi64(vw_raw, 1)), 16)
            );

            // BF16 → FP32 변환 (하위 16개)
            __m512 vx_lo = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(
                    _mm512_castsi512_si256(vx_raw)), 16)
            );
            __m512 vw_lo = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(
                    _mm512_castsi512_si256(vw_raw)), 16)
            );

            // FMA
            acc = _mm512_fmadd_ps(vx_lo, vw_lo, acc);
            acc = _mm512_fmadd_ps(vx_hi, vw_hi, acc);
        }

        // 잔여 처리
        float sum = _mm512_reduce_add_ps(acc);
        for (; k < K; ++k) {
            sum += static_cast<float>(x[k]) * static_cast<float>(W[n*K + k]);
        }

        // BF16으로 변환하여 저장
        y[n] = static_cast<at::BFloat16>(sum);
    }
}

// 배치 GEMV: Y[B,N] = X[B,K] × W[K,N]
// B개의 GEMV를 병렬 처리
void bf16_batch_gemv_avx512(
    const at::BFloat16* X,    // [B, K]
    const at::BFloat16* W,    // [N, K]
    at::BFloat16* Y,          // [B, N]
    int B, int N, int K
) {
    // 외부: 배치 병렬화
    // 내부: N 방향 벡터화
    #pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < B; ++b) {
        bf16_gemv_avx512(
            X + b*K,
            W,
            Y + b*N,
            N, K
        );
    }
}

}  // namespace cpu
}  // namespace vllm
```

### 6.4 cpu_attn.py 수정

```python
# vllm/v1/attention/backends/cpu_attn.py 수정

def forward_decode(
    self,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    ...
) -> torch.Tensor:
    """최적화된 디코드 어텐션"""

    num_tokens, num_heads, head_size = query.shape

    # AVX-512 최적화 경로
    if hasattr(ops, 'decode_attention_avx512') and query.dtype == torch.bfloat16:
        return ops.decode_attention_avx512(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            self.scale,
        )

    # 기존 PyTorch SDPA 경로 (폴백)
    return self._forward_decode_sdpa(...)
```

---

## 7. Phase 4: 배치 처리 최적화

### 7.1 목표

소규모 배치에서도 SIMD 효율 극대화

### 7.2 전략: 다중 시퀀스 인터리빙

```
기존 (시퀀스별 처리):
  Seq0: [Q0] × [K0,V0] → Out0
  Seq1: [Q1] × [K1,V1] → Out1
  ...
  → 각 시퀀스가 SIMD 폭 미달

최적화 (인터리빙):
  [Q0,Q1,...,Q15] × [K_all,V_all] → [Out0,...,Out15]
  → 16개 시퀀스를 512비트로 병렬 처리
```

### 7.3 구현: batch_attention.cpp

```cpp
// csrc/cpu/batch_attention.cpp

#include <torch/torch.h>
#include <immintrin.h>

namespace vllm {
namespace cpu {

// 16개 시퀀스 병렬 디코드 어텐션
// Query: [16, num_heads, head_size]
// Key/Value: PagedAttention 형식
void batch16_decode_attention_avx512(
    const at::BFloat16* query,      // [16, H, D]
    const at::BFloat16* key_cache,  // [num_blocks, H, D/x, B, x]
    const at::BFloat16* value_cache,
    const int32_t* block_tables,    // [16, max_blocks]
    const int32_t* context_lens,    // [16]
    at::BFloat16* output,           // [16, H, D]
    int num_heads,
    int head_size,
    int block_size,
    int max_blocks_per_seq,
    float scale
) {
    const int num_seqs = 16;  // AVX-512 = 16 × 32비트

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < num_heads; ++h) {
        for (int d = 0; d < head_size; d += 16) {
            // 16개 시퀀스의 Query 로드 (인터리빙)
            __m512 q_vec[16];
            for (int s = 0; s < num_seqs; ++s) {
                // 각 시퀀스의 query[h,d:d+16] 로드
                q_vec[s] = load_bf16_to_fp32(
                    query + s*num_heads*head_size + h*head_size + d
                );
            }

            // 어텐션 스코어 계산 (softmax 전)
            __m512 scores[16];  // 각 시퀀스별 max_context 스코어
            for (int s = 0; s < num_seqs; ++s) {
                scores[s] = _mm512_setzero_ps();
            }

            // KV 캐시 순회
            int max_ctx = 0;
            for (int s = 0; s < num_seqs; ++s) {
                max_ctx = std::max(max_ctx, context_lens[s]);
            }

            for (int ctx = 0; ctx < max_ctx; ++ctx) {
                // 16개 시퀀스에서 해당 context의 key 로드
                __m512 k_vec[16];
                for (int s = 0; s < num_seqs; ++s) {
                    if (ctx < context_lens[s]) {
                        int block_idx = ctx / block_size;
                        int block_offset = ctx % block_size;
                        int physical_block = block_tables[s*max_blocks_per_seq + block_idx];

                        k_vec[s] = load_key_from_cache(
                            key_cache, physical_block, h, d, block_offset
                        );
                    } else {
                        k_vec[s] = _mm512_setzero_ps();  // 패딩
                    }
                }

                // Q·K dot product (16개 시퀀스 병렬)
                for (int s = 0; s < num_seqs; ++s) {
                    __m512 dot = _mm512_mul_ps(q_vec[s], k_vec[s]);
                    // scores[s][ctx] = reduce_add(dot) * scale
                }
            }

            // Softmax (생략, 각 시퀀스별로)

            // Score × V (생략)

            // 결과 저장
        }
    }
}

}  // namespace cpu
}  // namespace vllm
```

### 7.4 Python 인터페이스

```python
# vllm/v1/attention/backends/cpu_attn.py

def forward_decode_batched(
    self,
    query: torch.Tensor,          # [B, H, D]
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,   # [B, max_blocks]
    context_lens: torch.Tensor,   # [B]
) -> torch.Tensor:
    """배치 최적화 디코드"""

    batch_size = query.size(0)

    # 16개씩 배치 처리
    if batch_size >= 16 and hasattr(ops, 'batch16_decode_attention_avx512'):
        outputs = []
        for i in range(0, batch_size, 16):
            end = min(i + 16, batch_size)
            if end - i == 16:
                # 전체 16개 배치
                out = ops.batch16_decode_attention_avx512(
                    query[i:end],
                    key_cache,
                    value_cache,
                    block_tables[i:end],
                    context_lens[i:end],
                    self.scale,
                )
            else:
                # 잔여 (패딩 후 처리)
                out = self._forward_decode_padded(
                    query[i:end], key_cache, value_cache,
                    block_tables[i:end], context_lens[i:end]
                )
            outputs.append(out)
        return torch.cat(outputs, dim=0)

    # 폴백
    return self._forward_decode_sequential(...)
```

---

## 8. Phase 5: 메모리 대역폭 최적화

### 8.1 목표

메모리 바운드 연산에서 대역폭 활용 극대화

### 8.2 기법

#### 8.2.1 비템포럴 로드/스토어

```cpp
// 캐시를 오염시키지 않는 스트리밍 연산

// 비템포럴 로드 (NT load)
__m512i data = _mm512_stream_load_si512(ptr);

// 비템포럴 스토어 (NT store)
_mm512_stream_si512(ptr, data);

// 사용 시점:
// - 대용량 데이터 순차 접근
// - 재사용되지 않는 데이터
// - KV 캐시 순회
```

#### 8.2.2 소프트웨어 프리페치

```cpp
// L1 캐시로 프리페치 (가장 빠름)
_mm_prefetch(ptr + PREFETCH_DISTANCE, _MM_HINT_T0);

// L2 캐시로 프리페치
_mm_prefetch(ptr + PREFETCH_DISTANCE * 2, _MM_HINT_T1);

// 비템포럴 프리페치 (streaming)
_mm_prefetch(ptr + PREFETCH_DISTANCE, _MM_HINT_NTA);
```

#### 8.2.3 NUMA-aware 메모리 접근

```cpp
// csrc/cpu/numa_utils.hpp

#include <numa.h>
#include <numaif.h>

class NUMAContext {
public:
    // 현재 스레드를 NUMA 노드에 바인딩
    static void bind_to_node(int node) {
        numa_run_on_node(node);
        numa_set_preferred(node);
    }

    // 메모리를 특정 노드에 할당
    static void* alloc_on_node(size_t size, int node) {
        return numa_alloc_onnode(size, node);
    }

    // 인터리브 할당 (모든 노드에 분산)
    static void* alloc_interleaved(size_t size) {
        return numa_alloc_interleaved(size);
    }
};
```

### 8.3 KV 캐시 메모리 레이아웃 최적화

```cpp
// 현재 레이아웃: [num_blocks, H, D/x, B, x]
// → 비연속적 메모리 접근

// 최적화 레이아웃: [num_blocks, B, H, D]
// → 블록 내 연속 접근

// 레이아웃 변환 커널
void optimize_kv_cache_layout(
    const at::BFloat16* src,  // [num_blocks, H, D/x, B, x]
    at::BFloat16* dst,        // [num_blocks, B, H, D]
    int num_blocks, int num_heads, int head_size, int block_size
) {
    #pragma omp parallel for
    for (int b = 0; b < num_blocks; ++b) {
        for (int t = 0; t < block_size; ++t) {
            for (int h = 0; h < num_heads; ++h) {
                for (int d = 0; d < head_size; ++d) {
                    // 인덱스 계산 (생략)
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}
```

---

## 9. 테스트 및 벤치마크

### 9.1 단위 테스트

```python
# tests/cpu/test_avx512_kernels.py

import torch
import pytest
from vllm import _custom_ops as ops

class TestAVX512Kernels:

    @pytest.mark.parametrize("M,N,K", [
        (1, 4096, 4096),      # GEMV
        (32, 4096, 4096),     # 소규모 GEMM
        (128, 4096, 4096),    # 중규모 GEMM
    ])
    def test_int8_gemm_vnni(self, M, N, K):
        """INT8 GEMM 정확도 테스트"""
        # 입력 생성
        A = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        B = torch.randint(0, 255, (N, K), dtype=torch.uint8)
        C = torch.zeros(M, N, dtype=torch.int32)

        # VNNI 커널
        ops.int8_gemm_vnni(C, A, B, 1.0, 1.0)

        # 참조 구현
        A_float = A.float()
        B_float = B.float()
        C_ref = torch.mm(A_float, B_float.T).int()

        # 검증
        assert torch.allclose(C, C_ref, atol=1)

    def test_q8_0_linear(self):
        """Q8_0 Linear 정확도 테스트"""
        # FP32 Linear
        linear_fp32 = torch.nn.Linear(4096, 4096, bias=False)

        # Q8_0 변환
        from vllm.model_executor.layers.quantization.q8_0 import Q8_0Linear
        linear_q8 = Q8_0Linear.from_float(linear_fp32)

        # 입력
        x = torch.randn(32, 4096, dtype=torch.bfloat16)

        # 출력 비교
        y_fp32 = linear_fp32(x.float())
        y_q8 = linear_q8(x)

        # 상대 오차 < 1%
        rel_error = (y_fp32.float() - y_q8.float()).abs() / y_fp32.abs().clamp(min=1e-5)
        assert rel_error.mean() < 0.01

    def test_decode_attention_avx512(self):
        """디코드 어텐션 정확도 테스트"""
        B, H, D = 16, 32, 128
        max_ctx = 1024

        query = torch.randn(B, H, D, dtype=torch.bfloat16)
        # ... KV 캐시 설정 ...

        # AVX-512 커널
        out_avx = ops.batch16_decode_attention_avx512(...)

        # 참조 (PyTorch)
        out_ref = F.scaled_dot_product_attention(...)

        assert torch.allclose(out_avx, out_ref, atol=1e-2)
```

### 9.2 성능 벤치마크

```python
# benchmarks/bench_avx512.py

import torch
import time
from vllm import _custom_ops as ops

def benchmark_gemv(N, K, iterations=1000):
    """GEMV 벤치마크"""
    x = torch.randn(K, dtype=torch.bfloat16)
    W = torch.randn(N, K, dtype=torch.bfloat16)
    y = torch.empty(N, dtype=torch.bfloat16)

    # 워밍업
    for _ in range(10):
        ops.bf16_gemv_avx512(y, x, W)

    # 벤치마크
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    for _ in range(iterations):
        ops.bf16_gemv_avx512(y, x, W)

    elapsed = time.perf_counter() - start

    # 성능 계산
    flops = 2 * N * K * iterations
    gflops = flops / elapsed / 1e9

    # 대역폭 (읽기: x + W, 쓰기: y)
    bytes_accessed = (K + N*K + N) * 2 * iterations  # BF16 = 2 bytes
    bandwidth = bytes_accessed / elapsed / 1e9  # GB/s

    print(f"GEMV [{N}x{K}]:")
    print(f"  Time: {elapsed/iterations*1000:.3f} ms")
    print(f"  GFLOPS: {gflops:.2f}")
    print(f"  Bandwidth: {bandwidth:.2f} GB/s")

    return gflops, bandwidth

def benchmark_int8_gemm(M, N, K, iterations=100):
    """INT8 GEMM 벤치마크"""
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8)
    B = torch.randint(0, 255, (N, K), dtype=torch.uint8)
    C = torch.zeros(M, N, dtype=torch.int32)

    # 워밍업
    for _ in range(10):
        ops.int8_gemm_vnni(C, A, B, 1.0, 1.0)

    # 벤치마크
    start = time.perf_counter()

    for _ in range(iterations):
        ops.int8_gemm_vnni(C, A, B, 1.0, 1.0)

    elapsed = time.perf_counter() - start

    # INT8 OPS (2*M*N*K per iteration)
    ops_count = 2 * M * N * K * iterations
    tops = ops_count / elapsed / 1e12

    print(f"INT8 GEMM [{M}x{N}x{K}]:")
    print(f"  Time: {elapsed/iterations*1000:.3f} ms")
    print(f"  TOPS: {tops:.2f}")

    return tops

if __name__ == "__main__":
    print("=" * 50)
    print("AVX-512 Kernel Benchmarks")
    print("=" * 50)

    # GEMV (디코드 단계)
    for N in [4096, 8192, 16384]:
        benchmark_gemv(N, 4096)

    print()

    # INT8 GEMM
    for M in [1, 32, 128]:
        benchmark_int8_gemm(M, 4096, 4096)
```

### 9.3 통합 테스트

```bash
#!/bin/bash
# scripts/test_avx512_integration.sh

set -e

echo "=== AVX-512 Integration Test ==="

# CPU 기능 확인
echo "1. CPU Features:"
python -c "
from vllm.platforms.intel_cpu_utils import detect_intel_cpu_features
f = detect_intel_cpu_features()
print(f'  AVX-512F: {f.avx512f}')
print(f'  AVX-512 VNNI: {f.avx512_vnni}')
print(f'  AVX-512 BF16: {f.avx512_bf16}')
"

# 커널 테스트
echo "2. Unit Tests:"
pytest tests/cpu/test_avx512_kernels.py -v

# 벤치마크
echo "3. Benchmarks:"
python benchmarks/bench_avx512.py

# 모델 추론 테스트
echo "4. Model Inference Test:"
python -c "
from vllm import LLM

llm = LLM(
    model='facebook/opt-125m',
    device='cpu',
    dtype='bfloat16',
    enforce_eager=True,
)

output = llm.generate('Hello, world!', max_tokens=10)
print(f'  Generated: {output[0].outputs[0].text}')
"

echo "=== All Tests Passed ==="
```

---

## 10. 예상 성능 향상

### 10.1 연산별 예상 향상

| 연산 | 현재 | AVX-512 최적화 | AMX 최적화 | 향상 |
|------|------|---------------|-----------|------|
| **Decode GEMV (BF16)** | 1x | 1.5-2x | 2-3x | AMX-BF16 |
| **Decode GEMV (INT8)** | 1x | 2-3x | 3-4x | AMX-INT8 |
| **Prefill GEMM (BF16)** | 1x | 1.5x | 2-3x | AMX-BF16 타일 |
| **Prefill GEMM (INT8)** | 1x | 2x | 3-4x | AMX-INT8 타일 |
| **Attention Softmax** | 1x | 1.5x | 1.5x | AVX-512 벡터화 |
| **배치 어텐션** | 1x | 1.3x | 1.5x | 인터리빙 |

### 10.2 AMX vs AVX-512 성능 비교

| 연산 | AVX-512 VNNI | AMX-INT8 | AMX 향상 |
|------|-------------|----------|----------|
| INT8 GEMM (M=1) | ~100 GOP/s | ~150 GOP/s | 1.5x |
| INT8 GEMM (M=32) | ~400 GOP/s | ~800 GOP/s | 2x |
| BF16 GEMM (M=32) | ~200 GF/s | ~500 GF/s | 2.5x |

*GOP/s = Giga Operations per second, GF/s = Giga FLOPS*

### 10.2 전체 추론 성능 예상

**70B 모델 (Q8_0 양자화):**

| 단계 | 현재 | 최적화 후 |
|------|------|----------|
| Decode | 2-3 tok/s | 4-6 tok/s |
| Prefill | 느림 | 1.5-2x 향상 |

**13B 모델 (Q8_0 양자화):**

| 단계 | 현재 | 최적화 후 |
|------|------|----------|
| Decode | 10-15 tok/s | 20-30 tok/s |
| Prefill | 중간 | 1.5-2x 향상 |

### 10.3 제한 사항

1. **메모리 바운드**: GEMV는 계산보다 메모리 대역폭이 병목
   - AVX-512로 계산은 빨라지지만, 메모리 로드 시간은 동일
   - INT8 양자화가 실질적 해결책 (대역폭 2배 절감)

2. **Prefill vs Decode 차이**:
   - Prefill: 계산 바운드 → SIMD 최적화 효과 큼
   - Decode: 메모리 바운드 → 양자화 + 캐시 최적화 필요

3. **70B 모델 한계**:
   - CPU만으로는 여전히 GPU 대비 10-20배 느림
   - 하이브리드 환경에서 CPU는 보조 역할이 적합

---

## 부록 A: 빌드 가이드

### A.1 의존성

```bash
# 필수
sudo apt install cmake ninja-build

# Intel 도구 (선택)
# oneAPI Base Toolkit: MKL, oneDNN
wget https://registrationcenter-download.intel.com/...
sudo sh l_BaseKit_p_*.sh

# libnuma
sudo apt install libnuma-dev
```

### A.2 빌드 명령

```bash
cd vllm_hybrid

# AVX-512 VNNI 활성화 빌드
CMAKE_ARGS="-DVLLM_CPU_AVX512_VNNI=ON" pip install -e .

# 또는 직접 CMake
mkdir build && cd build
cmake .. \
  -DVLLM_CPU_AVX512_VNNI=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -GNinja
ninja
```

### A.3 검증

```bash
# AVX-512 VNNI 활성화 확인
python -c "
import vllm._C as C
print('AVX512-VNNI kernels available:', hasattr(C, 'int8_gemm_vnni'))
"
```

---

## 부록 B: 참고 자료

1. [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
2. [oneDNN Developers Guide](https://oneapi-src.github.io/oneDNN/)
3. [llama.cpp AVX-512 Implementation](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu-aarch64.c)
4. [PyTorch CPU Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

*작성일: 2026-02-03*
*작성자: Claude*
*버전: 1.0*
