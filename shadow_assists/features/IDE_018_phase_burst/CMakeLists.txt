# IDE_018 — Sub-Layer Phase-Aware CPU Burst — build
#
# Targets:
#   - phase_burst_core (static lib): phase_detector + scheduler + task_pool_*
#   - phase_burst._core (pybind11 module): Python binding
#   - phase_burst_bench (executable): standalone microbench (TSK_031 latency)
#
# Usage:
#   cd shadow_assists/features/IDE_018_phase_burst
#   mkdir -p build && cd build
#   cmake -DCMAKE_BUILD_TYPE=Release \
#         -DENABLE_CUDA=OFF \                # ON only on prod machine
#         -DPython_EXECUTABLE=$(which python) \
#         ..
#   cmake --build . -j

cmake_minimum_required(VERSION 3.18)
project(phase_burst LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build flags — release-only optimization, x86 PAUSE intrinsics.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
    -Wall -Wextra
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>     # AVX-512 / AMX hint passthrough
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Debug>:-g>
)

# ── Optional CUDA support ─────────────────────────────────────────
option(ENABLE_CUDA "Enable CUDA event hook path" OFF)
if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(ENABLE_CUDA=1)
endif()

# ── Core static library ───────────────────────────────────────────
add_library(phase_burst_core STATIC
    src/phase_detector.cpp
    src/scheduler.cpp
    src/task_pool_attention.cpp
    src/task_pool_linear.cpp
    src/task_pool_dummy_fill.cpp
)
target_include_directories(phase_burst_core PUBLIC src)
target_link_libraries(phase_burst_core PUBLIC pthread)
if(ENABLE_CUDA)
    target_link_libraries(phase_burst_core PUBLIC CUDA::cudart)
endif()

# ── Standalone microbench ─────────────────────────────────────────
add_executable(phase_burst_bench tests/bench_phase_latency.cpp)
target_link_libraries(phase_burst_bench PRIVATE phase_burst_core)

# ── Scheduler unit test (no pybind, no CUDA) ──────────────────────
add_executable(phase_burst_scheduler_test tests/test_scheduler_cpp.cpp)
target_link_libraries(phase_burst_scheduler_test PRIVATE phase_burst_core)

# ── pybind11 Python module ────────────────────────────────────────
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
# pybind11 vendored or via pip: prefer pip-installed pybind11.
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
    target_link_libraries(_core PRIVATE phase_burst_core)
    # Place the built .so next to phase_burst/__init__.py so
    # `import phase_burst` finds `phase_burst._core` without PYTHONPATH gymnastics.
    set_target_properties(_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/phase_burst
    )
else()
    message(WARNING "pybind11 not found via pip — skipping Python module")
endif()
