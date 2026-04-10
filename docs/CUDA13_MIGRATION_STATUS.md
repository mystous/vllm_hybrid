# CUDA 13.0 Migration Status

**Branch**: `h100_cu13`
**Last commit**: `f0b527824`
**Date**: 2026-04-10

---

## Environment

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 13.0 (V13.0.88) |
| PyTorch | 2.9.0+cu130 |
| torchvision | 0.24.0+cu130 |
| torchaudio | 2.9.0+cu130 |
| xformers | 0.0.33 |
| IPEX | 2.8.0+gitcb81bf2 (소스 빌드, torch 2.9 대응) |
| transformers | 5.5.0 |
| NCCL | 2.27.7 |
| Docker | mystous/vllm_hybrid:cu13_v0.6_h100x4 |
| Hardware | H100 80GB x4, Xeon Platinum 8480+ (96 cores, 1 socket KVM) |

---

## Completed Work

### 1. CUDA 12.8 -> 13.0 Build Migration
- `setup.py`: MAIN_CUDA_VERSION 12.8 -> 13.0
- `CMakeLists.txt`: CUDA version checks, `-Xcompiler=-fvisibility=default`
- All `cu128` references -> `cu130` (Dockerfiles, requirements, CI, tests)
- torch 2.7.1 -> 2.9.0 (pyproject.toml, CMakeLists.txt, requirements/)

### 2. CUB API Changes (CUDA 13.0)
- **File**: `csrc/cuda_compat_cub.cuh` (new)
- `cub::Sum{}` -> `::cuda::std::plus<>{}` via `VLLM_CUB_SUM` macro
- `cub::Max{}` -> `::cuda::maximum<>{}` via `VLLM_CUB_MAX` macro
- `CUDART_VERSION >= 13000` ifdef guard
- Applied to: layernorm_kernels.cu, layernorm_quant_kernels.cu, layernorm_utils.cuh, topk_softmax_kernels.cu, int8_quant_kernels.cu, fp8/common.cu

### 3. Marlin Cross-TU Template Linking (CUDA 13.0)
- **Problem**: `__global__` template instantiations become static in CUDA 13.0
- **Fix**: `-rdc=true` for marlin kernel files + `CUDA_SEPARABLE_COMPILATION ON` + `CUDA_RESOLVE_DEVICE_SYMBOLS ON` on `_C` and `_moe_C` targets
- `MARLIN_GLOBAL_VISIBLE` / `MARLIN_MOE_GLOBAL_VISIBLE` macros with `__attribute__((visibility("default")))` for `__CUDACC_VER_MAJOR__ >= 13`

### 4. Flash Attention CUTLASS Warnings
- **File**: `cmake/external_projects/vllm_flash_attn.cmake`
- Suppress `-Wdeprecated-declarations` for `long4`, `double4` etc. (CUDA 13.0)

### 5. FlashMLA CCCL Include Path
- **File**: `cmake/external_projects/flashmla.cmake`
- CUDA 13.0: `cuda/std/*` headers moved to `cccl/` subdirectory
- Added `${_CUDA_ROOT}/include/cccl` to include path

### 6. transformers 5.x Compatibility
- `vllm/transformers_utils/configs/deepseek_vl2.py`: `vision_config`, `projector_config` default `None` (dataclass field ordering)
- `vllm/transformers_utils/tokenizer.py`: `all_special_tokens_extended` hasattr fallback

### 7. IPEX Source Build for torch 2.9
- IPEX 2.8.0 source from GitHub, patched for torch 2.9:
  - Python version check: `os.exit(127)` -> `warnings.warn()`
  - C++ version check (`csrc/jit/initialization.cpp`): `exit(127)` -> `fprintf(stderr, WARNING)`
  - CMake version check: `FATAL_ERROR` -> `WARNING`, `find_package(Torch)` without version constraint
  - `setup.py`: `getattr(torch._C, f"_PYBIND11_{pname}")` -> `getattr(..., None)`
- Build: `USE_CUDA=0 python setup.py bdist_wheel`
- Install: `pip install dist/intel_extension_for_pytorch-*.whl --force-reinstall --no-deps`
- **IPEX source patches are in `/tmp/ipex_build/`** (NOT committed, rebuild needed on new machine)

### 8. CPU Hybrid Engine Fixes
- **OMP thread binding**: `_resolve_cpu_params()` + `_setup_cpu_process_env()` moved before torch import in `run_cpu_engine_core()`
- **Lazy torch import**: `intel_cpu_utils.py` uses `_get_torch()` instead of module-level `import torch`
- **CpuPlatform forced**: `_platforms._current_platform = CpuPlatform()` in CPU process
- **device_type forced**: `cpu.py` check_and_update_config forces `device_type="cpu"` instead of assert
- **UniProcExecutor**: CPU engine uses UniProcExecutor (not MultiprocExecutor)
- **Chunked prefill disabled**: CPU engine sets `enable_chunked_prefill=False`, `chunked_prefill_enabled=False`
- **max_model_len**: CPU `max_model_len = min(gpu_max_model_len, batched_tokens * max_seqs)`
- **KV cache protection**: Save/restore `cpu_kvcache_space_bytes` around `CpuPlatform.check_and_update_config()`

---

## Current Issue: CPU Engine 500 Error

### Symptom
- Server starts, GPU + CPU engines both initialize successfully
- CapacityAwareRouter routes requests to `cpu:0` correctly (confirmed in logs)
- CPU engine returns **500 Internal Server Error** with empty message
- No traceback in server logs (error swallowed internally)

### Confirmed Working
- Routing: `cpu-first` + `capacity` strategy -> `cpu:0` (logged)
- CPU engine: UniProcExecutor, IPEX loaded, KV cache allocated, warmup completed
- OMP threads: 8 (reduced from 96 to avoid 17ms fork/join overhead)

### Root Cause Investigation
1. **OMP overhead discovered**: 96 threads = 17ms/op overhead (vs 0.02ms actual compute). Reduced to 8 threads.
2. **IPEX decode timing**: 34ms/call regardless of context length (ctx=2 or ctx=512 both 34ms). This is due to OMP thread sync overhead, NOT IPEX bug.
3. **With 8 threads**: IPEX attention should be sub-ms, but **500 error prevents testing**.

### Next Steps
1. **Debug 500 error**: The CPU engine receives the request but fails during execution. Need to:
   - Add try/except with traceback logging in `EngineCore._handle_client_request()` or CPU model runner's `execute_model()`
   - Or enable `VLLM_LOGGING_LEVEL=DEBUG` to get more detail
   - Check if the error is in prefill or decode step
   - Possible causes: tensor shape mismatch, missing custom op fallback, KV cache layout incompatibility

2. **After 500 fix**: Re-benchmark with 8 threads to measure actual CPU throughput improvement

3. **IPEX rebuild**: On new machine, need to rebuild IPEX from source with same patches (see Section 7)

---

## Key Files Modified

| File | Changes |
|------|---------|
| `setup.py` | MAIN_CUDA_VERSION 13.0 |
| `CMakeLists.txt` | CUDA 13.0 visibility, separable compilation, version checks |
| `csrc/cuda_compat_cub.cuh` | NEW: CUB compat macros |
| `csrc/layernorm_kernels.cu` | VLLM_CUB_SUM |
| `csrc/layernorm_quant_kernels.cu` | VLLM_CUB_SUM |
| `csrc/quantization/fused_kernels/layernorm_utils.cuh` | VLLM_CUB_SUM/MAX |
| `csrc/moe/topk_softmax_kernels.cu` | VLLM_CUB_MAX |
| `csrc/quantization/compressed_tensors/int8_quant_kernels.cu` | VLLM_CUB_MAX |
| `csrc/quantization/fp8/common.cu` | VLLM_CUB_MAX |
| `csrc/quantization/gptq_marlin/kernel.h` | MARLIN_GLOBAL_VISIBLE |
| `csrc/quantization/gptq_marlin/marlin_template.h` | MARLIN_GLOBAL_VISIBLE |
| `csrc/moe/marlin_moe_wna16/kernel.h` | MARLIN_MOE_GLOBAL_VISIBLE |
| `csrc/moe/marlin_moe_wna16/marlin_template.h` | MARLIN_MOE_GLOBAL_VISIBLE |
| `cmake/external_projects/vllm_flash_attn.cmake` | -Wno-deprecated-declarations |
| `cmake/external_projects/flashmla.cmake` | CCCL include path |
| `vllm/platforms/cpu.py` | device_type force instead of assert |
| `vllm/platforms/intel_cpu_utils.py` | Lazy torch import via _get_torch() |
| `vllm/v1/engine/hybrid_core.py` | OMP before torch, UniProcExecutor, CpuPlatform force, chunked prefill off, KV cache protection |
| `vllm/v1/engine/core_client.py` | Route logging in CapacityAwareRouter |
| `vllm/v1/attention/backends/cpu_attn.py` | IPEX decode timing instrumentation |
| `vllm/transformers_utils/configs/deepseek_vl2.py` | dataclass field defaults |
| `vllm/transformers_utils/tokenizer.py` | all_special_tokens_extended fallback |
| `docker/Dockerfile` | CUDA 13.0.0, cu130 |
| `docker/Dockerfile.nightly_torch` | CUDA 13.0.0, cu130 |
| `requirements/*.txt` | torch 2.9.0, cu130 |
| `pyproject.toml` | torch 2.9.0 |

---

## Eval Environment Files

| File | Description |
|------|-------------|
| `eval/envs/h100x4_Qwen2.5-1.5B_cpu_first_thro.env` | H100x4 + throughput-adaptive, CPU threads=8, KV=8GB, max_seqs=2 |

---

## Build Commands

```bash
# vLLM build (CUDA + CPU hybrid)
python tools/generate_cmake_presets.py
cmake --preset release
cmake --build --preset release --target install

# IPEX build (from /tmp/ipex_build or fresh clone)
git clone --depth 1 --branch v2.8.0+cpu https://github.com/intel/intel-extension-for-pytorch.git ipex_build
cd ipex_build
git submodule update --init --recursive

# Apply patches:
# 1. CMakeLists.txt: find_package(Torch) without version, WARNING instead of FATAL_ERROR
# 2. intel_extension_for_pytorch/__init__.py: warnings.warn instead of os.exit
# 3. csrc/jit/initialization.cpp: fprintf WARNING instead of exit(127)
# 4. setup.py: getattr with None default for _PYBIND11_* attrs

USE_CUDA=0 python setup.py bdist_wheel
pip install dist/intel_extension_for_pytorch-*.whl --force-reinstall --no-deps
```

---

*Last updated: 2026-04-10 11:30 UTC*
