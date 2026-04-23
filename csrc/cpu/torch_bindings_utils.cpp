// SPDX-License-Identifier: Apache-2.0
//
// Standalone _C_utils torch extension that exposes init_cpu_threads_env.
//
// Background
// ----------
// The original cpu_extension.cmake build (CPU-only target) compiles
// csrc/cpu/torch_bindings.cpp which registers init_cpu_threads_env into the
// `_C_utils` torch namespace. However, when vLLM is built for CUDA/ROCm
// (`VLLM_TARGET_DEVICE=cuda`), cpu_extension.cmake is *not* included, so the
// utils namespace is left empty. This makes CPUWorker fall back to
// Python-level sched_setaffinity, which only sets a process-wide affinity
// mask and cannot pin individual OMP threads to single cores. The result is
// poor cache locality (and on multi-NUMA hosts, cross-node DRAM traffic),
// which directly degrades hybrid CPU inference throughput.
//
// This file builds the C++ implementation into a tiny dedicated extension
// (`_C_utils`) that is *always* compiled regardless of the primary target
// device. The extension only depends on libnuma + OpenMP, has no SIMD
// requirements, and can be loaded on any x86_64 system.

// Use the same include pattern as csrc/cpu/torch_bindings_hybrid.cpp so that
// the build is compatible with stable ABI (USE_SABI=3) — torch/extension.h
// drags in pybind11 macros that conflict with Py_LIMITED_API.
#include <torch/all.h>
#include <torch/library.h>
#include "core/registration.h"

// Forward declaration — actual implementation lives in csrc/cpu/utils.cpp,
// which is compiled into the same extension.
std::string init_cpu_threads_env(const std::string& cpu_ids);

// TORCH_EXTENSION_NAME is set by setuptools/cmake to the loader's module name
// (here: `_C_utils`). The TORCH_LIBRARY_EXPAND macro produces a TORCH_LIBRARY
// block whose namespace matches that name, giving us
// `torch.ops._C_utils.init_cpu_threads_env` exactly like the CPU-only build.
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("init_cpu_threads_env(str cpu_ids) -> str", &init_cpu_threads_env);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
