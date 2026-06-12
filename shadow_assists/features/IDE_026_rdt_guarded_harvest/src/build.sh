#!/usr/bin/env bash
# IDE_026 T1 마이크로벤치 빌드.
# -march=native 우선 (EMR 호스트: AVX-512 포함). 실패 시 -mavx512f, 그 다음 scalar.
set -euo pipefail
cd "$(dirname "$0")"

SRC=victim_aggressor.c
OUT=victim_aggressor
COMMON="-O3 -pthread -Wall -Wextra"

if gcc $COMMON -march=native -o "$OUT" "$SRC" 2>/dev/null; then
    echo "[build] -march=native OK"
elif gcc $COMMON -mavx512f -o "$OUT" "$SRC" 2>/dev/null; then
    echo "[build] -mavx512f OK"
else
    gcc $COMMON -o "$OUT" "$SRC"
    echo "[build] scalar fallback (AVX-512 미사용 — aggressor 압박 약함 주의)"
fi
# AVX-512 코드 포함 여부 확인
if objdump -d "$OUT" | grep -q zmm; then
    echo "[build] zmm 명령 포함 확인 (AVX-512 경로 활성)"
else
    echo "[build] 경고: zmm 명령 없음 — scalar triad 로 빌드됨"
fi
