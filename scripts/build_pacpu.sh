#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# build_pacpu.sh — alias-friendly wrapper around csrc/cpu/pacpu/build.sh.
# TSK_018 Phase 3.3.
#
# Usage:
#   bash scripts/build_pacpu.sh                 # build dev default (qwen-1.5b TP=1)
#   bash scripts/build_pacpu.sh qwen-1.5b 1     # explicit alias + TP
#   bash scripts/build_pacpu.sh llama-70b 8     # prod target
#   bash scripts/build_pacpu.sh all             # build all known (alias, TP) pairs
#
# Aliases map to the macros defined in csrc/cpu/pacpu/dtype.h.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILDER="$REPO_ROOT/csrc/cpu/pacpu/build.sh"

declare -A ALIAS_MODEL=(
    [qwen-1.5b]=qwen2_5_1_5b
    [qwen2_5_1_5b]=qwen2_5_1_5b
    [llama-70b]=llama3_3_70b
    [llama3_3_70b]=llama3_3_70b
)

# Default (dev, RTX 3090) target — used when no args.
DEFAULT_ALIAS="qwen-1.5b"
DEFAULT_TP="1"

# (alias, TP) pairs for `all`. Add / remove as new dev or prod targets land.
ALL_TARGETS=(
    "qwen-1.5b 1"
    "llama-70b 8"
)

build_one() {
    local alias_in="$1"
    local tp="$2"
    local model="${ALIAS_MODEL[$alias_in]:-}"
    if [[ -z "$model" ]]; then
        echo "error: unknown alias '$alias_in'." >&2
        echo "       known: ${!ALIAS_MODEL[*]}" >&2
        exit 2
    fi
    echo "═══════════════════════════════════════════════════════════"
    echo " Building pacpu: $alias_in → $model  (TP=$tp)"
    echo "═══════════════════════════════════════════════════════════"
    bash "$BUILDER" "$model" "$tp"
    echo
}

if [[ $# -eq 0 ]]; then
    build_one "$DEFAULT_ALIAS" "$DEFAULT_TP"
elif [[ "$1" == "all" ]]; then
    for pair in "${ALL_TARGETS[@]}"; do
        # shellcheck disable=SC2086
        build_one $pair
    done
else
    if [[ $# -lt 2 ]]; then
        echo "usage: $0 [<alias> <TP> | all]" >&2
        echo "       known aliases: ${!ALIAS_MODEL[*]}" >&2
        exit 2
    fi
    build_one "$1" "$2"
fi
