#
# cpu_hybrid_extension.cmake
#
# Build Phase 1-5 CPU kernels as a separate _C_cpu_ops target
# when CUDA/ROCm is the primary target device.
#
# This allows hybrid CPU+GPU inference without modifying the main
# _C CUDA extension or cpu_extension.cmake.
#
# Dependencies: OpenMP (optional), libnuma (optional)
# No oneDNN required (Phase 1-5 kernels use raw AVX-512 intrinsics)
#

# Only build on x86_64
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    message(STATUS "CPU hybrid extension: skipping (not x86_64)")
    return()
endif()

#
# Detect ISA from /proc/cpuinfo
#
execute_process(COMMAND cat /proc/cpuinfo
                RESULT_VARIABLE _HYB_CPUINFO_RET
                OUTPUT_VARIABLE _HYB_CPUINFO)

if(NOT _HYB_CPUINFO_RET EQUAL 0)
    message(STATUS "CPU hybrid extension: skipping (cannot read /proc/cpuinfo)")
    return()
endif()

function(find_isa_hybrid CPUINFO TARGET OUT)
    string(FIND ${CPUINFO} ${TARGET} ISA_FOUND)
    if(NOT ISA_FOUND EQUAL -1)
        set(${OUT} ON PARENT_SCOPE)
    else()
        set(${OUT} OFF PARENT_SCOPE)
    endif()
endfunction()

# Check AVX-512 support (required)
find_isa_hybrid("${_HYB_CPUINFO}" "avx512f" HYB_AVX512_FOUND)
if(NOT HYB_AVX512_FOUND)
    message(STATUS "CPU hybrid extension: skipping (no AVX-512 support)")
    return()
endif()

# Check AVX-512 VNNI support (optional, enables Phase 1-2)
find_isa_hybrid("${_HYB_CPUINFO}" "avx512_vnni" HYB_AVX512VNNI_FOUND)

# Allow env override for cross-compilation
if(DEFINED ENV{VLLM_CPU_AVX512VNNI} AND "$ENV{VLLM_CPU_AVX512VNNI}")
    set(HYB_AVX512VNNI_FOUND ON)
endif()

#
# Compile flags (CXX only, no NVCC)
#
set(CPU_HYB_FLAGS
    -mavx512f
    -mavx512vl
    -mavx512bw
    -mavx512dq
    -mf16c
    -O3
    -funroll-loops
    -ffast-math
    -DVLLM_CPU_EXTENSION)

if(HYB_AVX512VNNI_FOUND)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
       CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.3)
        list(APPEND CPU_HYB_FLAGS -mavx512vnni)
    else()
        set(HYB_AVX512VNNI_FOUND OFF)
        message(WARNING "CPU hybrid extension: disabling VNNI (requires gcc >= 12.3)")
    endif()
endif()

#
# Libraries
#
set(CPU_HYB_LIBS "")

# OpenMP (optional, recommended for performance)
find_package(OpenMP COMPONENTS CXX QUIET)
if(OpenMP_CXX_FOUND)
    list(APPEND CPU_HYB_LIBS OpenMP::OpenMP_CXX)
endif()

# libnuma (optional, for NUMA-aware memory allocation)
find_library(HYB_NUMA_LIBRARY numa)
if(HYB_NUMA_LIBRARY)
    list(APPEND CPU_HYB_LIBS numa)
else()
    list(APPEND CPU_HYB_FLAGS -DVLLM_NUMA_DISABLED)
endif()

#
# Source files: Phase 1-5 CPU kernels + hybrid torch bindings
#
set(CPU_HYB_SRC
    "csrc/cpu/torch_bindings_hybrid.cpp"
    "csrc/cpu/decode_gemv.cpp"
    "csrc/cpu/batch_attention.cpp"
    "csrc/cpu/mem_opt.cpp")

if(HYB_AVX512VNNI_FOUND)
    list(APPEND CPU_HYB_SRC
        "csrc/cpu/gemm_vnni.cpp"
        "csrc/cpu/quant_q8_0.cpp")
endif()

message(STATUS "CPU hybrid extension: building _C_cpu_ops"
    " (AVX512=${HYB_AVX512_FOUND}, VNNI=${HYB_AVX512VNNI_FOUND})")
message(STATUS "CPU hybrid extension sources: ${CPU_HYB_SRC}")

#
# Define the _C_cpu_ops extension target
#
define_gpu_extension_target(
    _C_cpu_ops
    DESTINATION vllm
    LANGUAGE CXX
    SOURCES ${CPU_HYB_SRC}
    LIBRARIES ${CPU_HYB_LIBS}
    COMPILE_FLAGS ${CPU_HYB_FLAGS}
    USE_SABI 3
    WITH_SOABI)
