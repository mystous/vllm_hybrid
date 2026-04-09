#pragma once

// Compatibility shim for CUB API changes in CUDA 13.0+
// In CUDA 13.0, cub::Sum and cub::Max were removed.
// Use cuda::std::plus<> and cuda::std::maximum<> instead.

#include <cuda_runtime.h>

#if CUDART_VERSION >= 13000
  #include <cuda/std/functional>
  #include <cuda/functional>
  #define VLLM_CUB_SUM ::cuda::std::plus<>{}
  #define VLLM_CUB_MAX ::cuda::maximum<>{}
#else
  #define VLLM_CUB_SUM cub::Sum{}
  #define VLLM_CUB_MAX cub::Max{}
#endif
