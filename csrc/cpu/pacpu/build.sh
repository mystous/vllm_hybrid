#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# pacpu build script — TSK_018 Phase 3.3 (env auto-detect + fallback).
#
# Usage:
#   bash csrc/cpu/pacpu/build.sh <MODEL> <TP>
#
# Args:
#   MODEL  one of:  llama3_3_70b | qwen2_5_1_5b   (case-insensitive)
#   TP     positive integer  (e.g. 1, 2, 4, 8)
#
# Env (optional overrides — each falls back to auto-detect):
#   ISPC_BIN          path to ispc binary
#   CXX               C++ compiler (must support _Float16; g++ ≥ 12)
#   CUDA_HOST_COMPILER  C++ host compiler for CUDA detection (g++-11 is fine)
#
# Output:
#   csrc/cpu/pacpu/build/libpacpu-<model>-tp<TP>.so

set -euo pipefail

# ---------- arg parsing ----------
if [[ $# -ne 2 ]]; then
    echo "usage: $0 <MODEL> <TP>" >&2
    echo "  MODEL ∈ { llama3_3_70b, qwen2_5_1_5b }" >&2
    echo "  TP    positive integer" >&2
    exit 2
fi
MODEL="${1,,}"          # lower-case
TP="$2"

case "$MODEL" in
    llama3_3_70b|qwen2_5_1_5b) ;;
    *)
        echo "error: unknown MODEL '$MODEL'." >&2
        echo "       supported: llama3_3_70b, qwen2_5_1_5b" >&2
        echo "       (add a new macro to dtype.h then update this list)" >&2
        exit 2
        ;;
esac

if ! [[ "$TP" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: TP must be a positive integer (got '$TP')" >&2
    exit 2
fi

# ---------- ISPC ----------
if [[ -z "${ISPC_BIN:-}" ]]; then
    if command -v ispc >/dev/null 2>&1; then
        ISPC_BIN="$(command -v ispc)"
    else
        # Search common dev-machine layouts.
        for cand in /workspace/ispc-*/bin/ispc /opt/ispc-*/bin/ispc /usr/local/bin/ispc; do
            if [[ -x "$cand" ]]; then
                ISPC_BIN="$cand"
                break
            fi
        done
    fi
fi
if [[ -z "${ISPC_BIN:-}" || ! -x "$ISPC_BIN" ]]; then
    echo "error: ispc binary not found." >&2
    echo "       install ispc 1.23+ then either prepend its bin/ to PATH" >&2
    echo "       or pass ISPC_BIN=/abs/path/to/ispc" >&2
    exit 3
fi
echo "  ISPC_BIN  = $ISPC_BIN"

# ---------- CXX (g++ ≥ 12 — _Float16 required) ----------
if [[ -z "${CXX:-}" ]]; then
    for cand in g++-13 g++-12; do
        if command -v "$cand" >/dev/null 2>&1; then
            CXX="$(command -v "$cand")"
            break
        fi
    done
fi
if [[ -z "${CXX:-}" || ! -x "$CXX" ]]; then
    echo "error: g++ ≥ 12 not found (g++-11 lacks _Float16 support)." >&2
    echo "       install g++-12 or g++-13, or pass CXX=/abs/path" >&2
    exit 3
fi
echo "  CXX       = $CXX"

# ---------- CUDA host compiler (any g++ works for stub detection) ----------
if [[ -z "${CUDA_HOST_COMPILER:-}" ]]; then
    for cand in g++-11 g++-12 g++-13; do
        if command -v "$cand" >/dev/null 2>&1; then
            CUDA_HOST_COMPILER="$(command -v "$cand")"
            break
        fi
    done
fi
echo "  CUDA host = ${CUDA_HOST_COMPILER:-<not set — cmake default>}"

# ---------- Torch (Python-side cmake_prefix_path) ----------
if ! command -v python >/dev/null 2>&1; then
    echo "error: 'python' not on PATH; activate the vLLM venv first" >&2
    exit 3
fi
TORCH_DIR="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')/Torch"
if [[ ! -d "$TORCH_DIR" ]]; then
    echo "error: TORCH_DIR not a directory: $TORCH_DIR" >&2
    echo "       (likely the python interpreter has no torch installed)" >&2
    exit 3
fi
echo "  TORCH_DIR = $TORCH_DIR"

# ---------- build ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

cmake_args=(
    -B "$BUILD_DIR"
    -S "$SCRIPT_DIR"
    -DTorch_DIR="$TORCH_DIR"
    -DModel="$MODEL"
    -DTP="$TP"
    -DCMAKE_CXX_COMPILER="$CXX"
    -DCMAKE_ISPC_COMPILER="$ISPC_BIN"
)
if [[ -n "${CUDA_HOST_COMPILER:-}" ]]; then
    cmake_args+=(-DCMAKE_CUDA_HOST_COMPILER="$CUDA_HOST_COMPILER")
fi

echo "→ cmake configure"
cmake "${cmake_args[@]}"
echo "→ cmake build"
cmake --build "$BUILD_DIR" -j"$(nproc)"

# ---------- verify ----------
EXPECTED="$BUILD_DIR/libpacpu-${MODEL}-tp${TP}.so"
if [[ ! -f "$EXPECTED" ]]; then
    echo "error: expected output not found: $EXPECTED" >&2
    exit 4
fi
SIZE=$(stat -c%s "$EXPECTED")
echo "✓ built: $EXPECTED ($SIZE bytes)"
