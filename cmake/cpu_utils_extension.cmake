#
# cpu_utils_extension.cmake
#
# Build a small `_C_utils` torch extension that exposes init_cpu_threads_env
# for the hybrid CPU+GPU inference path. See csrc/cpu/torch_bindings_utils.cpp
# for the rationale (CUDA builds otherwise leave the namespace empty, forcing
# CPUWorker into a Python sched_setaffinity fallback that cannot pin OMP
# threads 1:1 to cores).
#
# This file is included from CMakeLists.txt only when VLLM_TARGET_DEVICE is
# `cuda` or `rocm` so it does NOT conflict with cpu_extension.cmake which
# already registers the same symbol on CPU-only builds.
#
# Dependencies:
#   - libnuma  (optional — falls back to a warning-only stub if absent)
#   - OpenMP   (required for the parallel-for affinity loop in utils.cpp)
#
# No SIMD requirements: utils.cpp deliberately avoids cpu_types.hpp so this
# target compiles on any x86_64 host (AVX2, AVX-512, AMX — all OK).
#

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    message(STATUS "CPU utils extension: skipping (not x86_64)")
    return()
endif()

set(CPU_UTILS_FLAGS
    -O2
    -DVLLM_CPU_EXTENSION)

# csrc/ root for "core/registration.h" include
include_directories("${CMAKE_SOURCE_DIR}/csrc")

# OpenMP — required (utils.cpp uses #pragma omp parallel for to pin threads)
find_package(OpenMP COMPONENTS CXX QUIET)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "CPU utils extension: OpenMP not found, skipping. "
                    "init_cpu_threads_env will not be available; CPUWorker "
                    "will fall back to Python sched_setaffinity.")
    return()
endif()

set(CPU_UTILS_LIBS OpenMP::OpenMP_CXX)

# libnuma — optional
find_library(CPU_UTILS_NUMA_LIBRARY numa)
if(CPU_UTILS_NUMA_LIBRARY)
    list(APPEND CPU_UTILS_LIBS numa)
    message(STATUS "CPU utils extension: libnuma found at "
                   "${CPU_UTILS_NUMA_LIBRARY}")
else()
    list(APPEND CPU_UTILS_FLAGS -DVLLM_NUMA_DISABLED)
    message(STATUS "CPU utils extension: libnuma not found, "
                   "VLLM_NUMA_DISABLED stub will be used")
endif()

set(CPU_UTILS_SRC
    "csrc/cpu/utils.cpp"
    "csrc/cpu/torch_bindings_utils.cpp")

message(STATUS "CPU utils extension: building _C_utils target")

define_gpu_extension_target(
    _C_utils
    DESTINATION vllm
    LANGUAGE CXX
    SOURCES ${CPU_UTILS_SRC}
    LIBRARIES ${CPU_UTILS_LIBS}
    COMPILE_FLAGS ${CPU_UTILS_FLAGS}
    USE_SABI 3
    WITH_SOABI)
