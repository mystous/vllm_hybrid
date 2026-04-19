#!/usr/bin/env bash
# =============================================================================
# verify_hugetlb_1g.sh — NinjaGap §03 Phase 2 현장 검증 스크립트
# =============================================================================
#
# 1GB hugetlbfs 경로가 실제로 vLLM 안에서 동작하는지를 end-to-end 로 확인.
# "env 는 =1 인데 실제로 allocator 가 켜지는지" / "alloc 시도 시 실패 사유가
# 무엇인지" 를 분리해서 찍는다.
#
# 사용법:
#   # 방법 1: 현재 shell env 기반 확인
#   bash setup/verify_hugetlb_1g.sh
#
#   # 방법 2: env 파일 source 후 확인 (serve.sh 와 동일한 동작 재현)
#   bash setup/verify_hugetlb_1g.sh eval/envs/g0_h100x8_qwen32b_hugetlb_1g_full.env
#
# 출력 섹션:
#   [1] git / 코드 반영 여부
#   [2] vllm 패키지 로드 경로
#   [3] env 변수 상태 (serve.sh 와 동일한 해석 순서)
#   [4] hugetlbfs mount 상태
#   [5] hugetlb_allocator 모듈 import 가능 여부
#   [6] cpu_model_runner 가 allocator 를 import 하는지 (_HUGETLB_1G_AVAILABLE)
#   [7] 실제 1GB region alloc + release 시도
#   [8] 작은 nn.Linear 로 bind_params_to_hugetlb 동작 확인
#
# 모든 단계는 독립. 앞 단계 실패해도 뒤 단계 계속 시도.
# =============================================================================
set -u

PYTHON="${PYTHON:-python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    if [ -x /vllm_dev_prj/bin/python ]; then
        PYTHON=/vllm_dev_prj/bin/python
    else
        PYTHON=python3
    fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_FILE="${1:-}"
if [ -n "${ENV_FILE}" ]; then
    if [ ! -f "${ENV_FILE}" ]; then
        if [ -f "${REPO_ROOT}/${ENV_FILE}" ]; then
            ENV_FILE="${REPO_ROOT}/${ENV_FILE}"
        else
            echo "[ERROR] env 파일 없음: ${ENV_FILE}" >&2
            exit 1
        fi
    fi
    # serve.sh 와 동일 — source 후 3개 키 export (없으면 default 로 0)
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi
export HYBRID_HUGETLB_1G_ENABLE="${HYBRID_HUGETLB_1G_ENABLE:-0}"
export HYBRID_HUGETLB_1G_PATH="${HYBRID_HUGETLB_1G_PATH:-/mnt/hugetlb_1g}"
export HYBRID_HUGETLB_1G_BIND_WEIGHTS="${HYBRID_HUGETLB_1G_BIND_WEIGHTS:-0}"

line() { printf '=%.0s' {1..72}; echo; }

line
echo "[1] git / 코드 반영"
line
(
    cd "${REPO_ROOT}" || exit
    echo "branch: $(git branch --show-current 2>/dev/null || echo '(not a repo)')"
    echo "recent commits:"
    git log --oneline -5 2>/dev/null | sed 's/^/  /'
    echo ""
    echo "expected Phase 2 commits (feat/g0-03-phase2-1gb-hugetlb):"
    git log --oneline feat/g0-03-phase2-1gb-hugetlb 2>/dev/null | grep -E 'hugetlb' | head -5 | sed 's/^/  /' || \
        echo "  (해당 branch 가 로컬에 없거나 관련 commit 없음)"
)
echo ""

line
echo "[2] vllm 패키지 로드 경로"
line
"${PYTHON}" - <<'PYEOF'
import importlib.util, sys
for mod in ("vllm", "vllm.platforms", "vllm.platforms.hugetlb_allocator",
            "vllm.v1.worker.cpu_model_runner"):
    spec = importlib.util.find_spec(mod)
    if spec is None:
        print(f"  MISSING: {mod}")
    else:
        print(f"  OK     : {mod:45s} -> {spec.origin}")
PYEOF
echo ""

line
echo "[3] env 변수 상태 (serve.sh 해석 순서 적용 후)"
line
echo "  HYBRID_HUGETLB_1G_ENABLE        = ${HYBRID_HUGETLB_1G_ENABLE}"
echo "  HYBRID_HUGETLB_1G_PATH          = ${HYBRID_HUGETLB_1G_PATH}"
echo "  HYBRID_HUGETLB_1G_BIND_WEIGHTS  = ${HYBRID_HUGETLB_1G_BIND_WEIGHTS}"
if [ -n "${ENV_FILE}" ]; then
    echo "  (env 파일: ${ENV_FILE})"
fi
echo ""

line
echo "[4] hugetlbfs mount"
line
if [ -d "${HYBRID_HUGETLB_1G_PATH}" ]; then
    echo "  path exists: ${HYBRID_HUGETLB_1G_PATH}"
    ls -ld "${HYBRID_HUGETLB_1G_PATH}" | sed 's/^/  /'
    if mount | grep -q " on ${HYBRID_HUGETLB_1G_PATH} type hugetlbfs"; then
        mount | grep " on ${HYBRID_HUGETLB_1G_PATH} type hugetlbfs" | sed 's/^/  /'
    else
        echo "  [WARN] mount 에 hugetlbfs 항목 없음"
    fi
    echo "  stat -f:"
    stat -f "${HYBRID_HUGETLB_1G_PATH}" 2>/dev/null | sed 's/^/    /' || \
        echo "    stat -f 실패"
else
    echo "  [MISSING] path not found: ${HYBRID_HUGETLB_1G_PATH}"
fi
echo "  /proc/meminfo HugePages_1048576:"
grep -E "HugePages_1048576|Hugetlb|Hugepagesize" /proc/meminfo | sed 's/^/    /'
echo ""

line
echo "[5] hugetlb_allocator import / is_configured()"
line
"${PYTHON}" - <<'PYEOF'
import os, sys
try:
    from vllm.platforms.hugetlb_allocator import (
        is_configured, bind_weights_enabled, HugeTLB1GAllocator,
        _mount_path,
    )
    print(f"  import OK")
    print(f"  is_configured()        = {is_configured()}")
    print(f"  bind_weights_enabled() = {bind_weights_enabled()}")
    print(f"  _mount_path()          = {_mount_path()}")
    a = HugeTLB1GAllocator.get()
    print(f"  HugeTLB1GAllocator.get() = {a}")
    if a is not None:
        print(f"    path={a.path}")
        print(f"    total_bytes={getattr(a, '_total_bytes', '?')}")
except Exception as e:
    print(f"  [FAIL] import / init 실패: {type(e).__name__}: {e}")
    sys.exit(0)
PYEOF
echo ""

line
echo "[6] cpu_model_runner 에서 _HUGETLB_1G_AVAILABLE"
line
"${PYTHON}" - <<'PYEOF'
try:
    import vllm.v1.worker.cpu_model_runner as m
    print(f"  module file: {m.__file__}")
    print(f"  _HUGETLB_1G_AVAILABLE  = {getattr(m, '_HUGETLB_1G_AVAILABLE', 'MISSING')}")
    print(f"  _hugetlb_is_configured = {getattr(m, '_hugetlb_is_configured', 'MISSING')}")
    if hasattr(m, "_hugetlb_is_configured"):
        print(f"  _hugetlb_is_configured() = {m._hugetlb_is_configured()}")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")
PYEOF
echo ""

line
echo "[7] alloc_region 실제 시도 (1GB)"
line
"${PYTHON}" - <<'PYEOF'
import logging, os
logging.basicConfig(level=logging.INFO, format='  %(levelname)s %(message)s')
from vllm.platforms.hugetlb_allocator import HugeTLB1GAllocator, is_configured
if not is_configured():
    print("  skip: HYBRID_HUGETLB_1G_ENABLE != 1")
else:
    a = HugeTLB1GAllocator.get()
    if a is None:
        print("  [FAIL] allocator init failed (see [5] warnings)")
    else:
        mm = a.alloc_region(1 << 30, numa_node=-1, tag="verify")
        if mm is None:
            print("  [FAIL] alloc_region returned None — hugetlb pool 부족 or path 문제")
        else:
            print(f"  [OK] 1GB region allocated, len(mm)={len(mm)}")
            # write/read sanity
            mm[0] = 0x42
            mm[len(mm)-1] = 0x43
            print(f"    mm[0]=0x{mm[0]:x}, mm[-1]=0x{mm[len(mm)-1]:x}")
        # cleanup
        a.release_all()
        print("  release_all() done")
PYEOF
echo ""

line
echo "[8] bind_params_to_hugetlb (작은 Linear)"
line
"${PYTHON}" - <<'PYEOF'
import logging, os
logging.basicConfig(level=logging.INFO, format='  %(levelname)s %(message)s')
from vllm.platforms.hugetlb_allocator import (
    bind_params_to_hugetlb, bind_weights_enabled, HugeTLB1GAllocator,
)
if not bind_weights_enabled():
    print("  skip: HYBRID_HUGETLB_1G_BIND_WEIGHTS != 1")
else:
    import torch
    m = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.Linear(32, 8),
    )
    sentinel = m[0].weight.data_ptr()
    print(f"  before bind: m[0].weight.data_ptr()=0x{sentinel:x}")
    migrated, n_bytes = bind_params_to_hugetlb(m, numa_node=-1)
    print(f"  bind result: migrated={migrated}, bytes={n_bytes}")
    after = m[0].weight.data_ptr()
    print(f"  after bind : m[0].weight.data_ptr()=0x{after:x}  (changed: {sentinel != after})")
    # allocator 상태
    a = HugeTLB1GAllocator.get()
    if a is not None:
        print(f"  allocator _mmaps count={len(a._mmaps)}, total_bytes={a._total_bytes}")
        a.release_all()
PYEOF
echo ""

line
echo "DONE"
line
