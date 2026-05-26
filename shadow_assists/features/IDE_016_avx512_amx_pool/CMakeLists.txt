# IDE_016 — AVX-512 sampling + AMX matmul — build
#
# Targets:
#   - avx512_amx_pool_core (static lib): kernels (sampling + AMX matmul)
#   - avx512_amx_pool._core (pybind11 module): torch-compatible Python binding
#   - avx512_amx_pool_bench (executable): standalone microbench
#
# Usage:
#   cd shadow_assists/features/IDE_016_avx512_amx_pool
#   mkdir -p build && cd build
#   cmake -DCMAKE_BUILD_TYPE=Release \
#         -DPython_EXECUTABLE=$(which python) \
#         ..
#   cmake --build . -j

cmake_minimum_required(VERSION 3.18)
project(avx512_amx_pool LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Sapphire Rapids native + AVX-512/AMX intrinsics.
# (개발 머신 Alder Lake 에선 AVX-512 fuse-off 가능 — runtime cpuid 게이트가
# python_bindings.cpp 에서 분기. AMX 는 hw 미지원이라 runtime skip.)
add_compile_options(
    -Wall -Wextra
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=sapphirerapids>
    $<$<CONFIG:Release>:-mavx512f>
    $<$<CONFIG:Release>:-mavx512bw>
    $<$<CONFIG:Release>:-mavx512vl>
    $<$<CONFIG:Release>:-mavx512dq>
    $<$<CONFIG:Release>:-mavx512cd>
    $<$<CONFIG:Release>:-mavx512bf16>
    $<$<CONFIG:Release>:-mamx-tile>
    $<$<CONFIG:Release>:-mamx-bf16>
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Debug>:-g>
)

# ── Core static library ───────────────────────────────────────────
add_library(avx512_amx_pool_core STATIC
    src/avx512_sampling/topk_topp_kernel.cpp
    src/avx512_sampling/logit_processor.cpp
    src/avx512_sampling/penalty_ops.cpp
    src/amx_matmul/amx_qwen_draft.cpp
    src/avx512_tokenizer/batch_bpe_kernel.cpp
)
target_include_directories(avx512_amx_pool_core PUBLIC src)
target_link_libraries(avx512_amx_pool_core PUBLIC pthread)

# ── Standalone microbench ─────────────────────────────────────────
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/bench_microbench.cpp)
    add_executable(avx512_amx_pool_bench tests/bench_microbench.cpp)
    target_link_libraries(avx512_amx_pool_bench PRIVATE avx512_amx_pool_core)
endif()

# ── Unit test (C++ only — no python required) ─────────────────────
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_kernels_cpp.cpp)
    add_executable(avx512_amx_pool_test tests/test_kernels_cpp.cpp)
    target_link_libraries(avx512_amx_pool_test PRIVATE avx512_amx_pool_core)
endif()

# ── pybind11 / torch Python module ────────────────────────────────
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_rc
)

if(_pybind11_rc EQUAL 0 AND EXISTS "${PYBIND11_CMAKE_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_core MODULE src/python_bindings.cpp)
    target_link_libraries(_core PRIVATE avx512_amx_pool_core)
    set_target_properties(_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/avx512_amx_pool
    )
else()
    message(WARNING "pybind11 not found — skipping Python module")
endif()
