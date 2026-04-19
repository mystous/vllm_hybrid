#!/usr/bin/env bash
# =============================================================================
# setup_hugetlb_1g_rhel.sh — 1GB hugetlbfs 관리 (NinjaGap §03 Phase 2)
# =============================================================================
#
# 사용법:
#   sudo bash setup/setup_hugetlb_1g_rhel.sh enable [N]   # N개 1GB 페이지 확보 (기본 64)
#   bash       setup/setup_hugetlb_1g_rhel.sh verify      # 현재 상태 조회
#   sudo bash setup/setup_hugetlb_1g_rhel.sh disable      # 완전 원복 (umount + pool 0)
#
# enable:
#   1) 전체 풀 nr_hugepages=N 설정
#   2) NUMA 노드별로 N/num_nodes 씩 분배 (2 NUMA 면 각 32)
#   3) /mnt/hugetlb_1g 에 mount
#   4) verify 출력
#
# disable:
#   1) /mnt/hugetlb_1g umount (쓰는 프로세스 있으면 경고)
#   2) 전체 풀 + NUMA 노드별 풀 모두 0
#   3) mount point rmdir
#   4) verify 출력
#
# 주의:
#   - 호스트 OS 에서만 동작. 컨테이너 안에서는 /sys 가 read-only.
#   - 컨테이너가 /mnt/hugetlb_1g 를 bind mount 중이면 disable 전에 컨테이너 먼저 stop.
#   - 1GB 페이지는 물리 연속 메모리 필요 — 단편화 심하면 enable 실패 가능.
# =============================================================================
set -u

MOUNT_POINT=/mnt/hugetlb_1g
DEFAULT_N=64
SYS_HUGE=/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
NUMA_HUGE_GLOB='/sys/devices/system/node/node*/hugepages/hugepages-1048576kB/nr_hugepages'

need_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "[ERROR] '$1' 는 root 권한 필요 (sudo 사용)" >&2
        exit 1
    fi
}

usage() {
    grep -E '^#' "$0" | sed 's/^# \?//' | head -30
}

verify() {
    echo "=== hugetlbfs 1GB 상태 ==="
    if [[ -r "${SYS_HUGE}" ]]; then
        echo "total nr_hugepages = $(cat ${SYS_HUGE})"
    else
        echo "[WARN] ${SYS_HUGE} 읽기 불가 (컨테이너 내부일 수 있음)"
    fi

    echo ""
    echo "per-NUMA nr_hugepages:"
    for f in ${NUMA_HUGE_GLOB}; do
        [[ -r "$f" ]] || continue
        node=$(echo "$f" | grep -oE 'node[0-9]+')
        echo "  ${node}: $(cat "$f")"
    done

    echo ""
    echo "/proc/meminfo:"
    grep -E "^HugePages_1048576|^Hugetlb:|^Hugepagesize:" /proc/meminfo | sed 's/^/  /'

    echo ""
    if mount | grep -q " on ${MOUNT_POINT} type hugetlbfs"; then
        echo "mount: ACTIVE"
        mount | grep " on ${MOUNT_POINT} type hugetlbfs" | sed 's/^/  /'
        if [[ -d "${MOUNT_POINT}" ]]; then
            echo ""
            echo "mount point 내용:"
            ls -la "${MOUNT_POINT}" 2>/dev/null | sed 's/^/  /'
            echo ""
            echo "stat -f:"
            stat -f "${MOUNT_POINT}" 2>/dev/null | grep -E "Type|Blocks|Free|Available" | sed 's/^/  /'
        fi
    else
        echo "mount: INACTIVE (not mounted on ${MOUNT_POINT})"
    fi
}

enable_pool() {
    need_root enable
    local n="${1:-${DEFAULT_N}}"

    # NUMA 노드 개수 감지
    local nodes=(${NUMA_HUGE_GLOB})
    local num_nodes=${#nodes[@]}

    echo "[INFO] target pool size: ${n} × 1GB  (num NUMA nodes detected: ${num_nodes})"

    # 선제 defrag (1GB 연속 블록 확보 가능성 ↑)
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo 1 > /proc/sys/vm/compact_memory 2>/dev/null || true

    # 전체 풀
    echo "${n}" > "${SYS_HUGE}"

    # NUMA 분배
    if (( num_nodes > 1 )); then
        local per_node=$(( n / num_nodes ))
        local remainder=$(( n - per_node * num_nodes ))
        local i=0
        for f in "${nodes[@]}"; do
            local val="${per_node}"
            if (( i == 0 )); then
                val=$(( per_node + remainder ))
            fi
            echo "${val}" > "${f}" 2>/dev/null || true
            i=$(( i + 1 ))
        done
    fi

    # 실제 확보된 개수 확인
    local got
    got=$(cat "${SYS_HUGE}")
    local got_free
    got_free=$(grep "^HugePages_1048576_Free:" /proc/meminfo 2>/dev/null | awk '{print $2}')
    if [[ -z "${got_free}" ]]; then
        # fallback: Hugetlb 크기 계산
        got_free=$(awk '/^HugePages_1048576:/ {print $2}' /proc/meminfo 2>/dev/null || echo "unknown")
    fi

    if (( got < n )); then
        echo "[WARN] requested ${n}, allocated ${got} — 단편화로 부분 확보. drop_caches + compact_memory 재시도 가능"
    fi

    # mount point 준비 + mount
    mkdir -p "${MOUNT_POINT}"
    if mount | grep -q " on ${MOUNT_POINT} type hugetlbfs"; then
        echo "[INFO] already mounted on ${MOUNT_POINT}, skipping mount"
    else
        local total_bytes=$(( got * 1024 * 1024 * 1024 ))
        if ! mount -t hugetlbfs -o "pagesize=1G,size=${total_bytes}" none "${MOUNT_POINT}"; then
            echo "[ERROR] mount 실패" >&2
            exit 1
        fi
        echo "[OK] mounted ${MOUNT_POINT} (pagesize=1G, size=${total_bytes} bytes)"
    fi

    echo ""
    verify
}

disable_pool() {
    need_root disable

    # umount 시도 (이미 안 되어 있어도 OK)
    if mount | grep -q " on ${MOUNT_POINT} type hugetlbfs"; then
        if fuser -v "${MOUNT_POINT}" 2>/dev/null | grep -q '[0-9]'; then
            echo "[WARN] ${MOUNT_POINT} 사용 중인 프로세스 있음:"
            fuser -v "${MOUNT_POINT}" 2>&1 | sed 's/^/  /'
            echo "       컨테이너가 이 경로를 bind mount 했으면 먼저 'podman stop' 하세요."
        fi
        if umount "${MOUNT_POINT}" 2>/dev/null; then
            echo "[OK] umount ${MOUNT_POINT}"
        else
            echo "[WARN] umount 실패 (사용 중) — 프로세스 정리 후 재시도 필요"
        fi
    else
        echo "[INFO] ${MOUNT_POINT} 는 이미 mount 되지 않은 상태"
    fi

    # 전체 풀 0 으로
    if [[ -w "${SYS_HUGE}" ]]; then
        echo 0 > "${SYS_HUGE}"
        echo "[OK] ${SYS_HUGE} <- 0"
    fi

    # NUMA 노드별 풀 0 으로
    for f in ${NUMA_HUGE_GLOB}; do
        [[ -w "$f" ]] || continue
        echo 0 > "${f}" 2>/dev/null && \
            echo "[OK] $(echo "$f" | grep -oE 'node[0-9]+') <- 0"
    done

    # 빈 mount point 제거 (실패 무시)
    if [[ -d "${MOUNT_POINT}" ]]; then
        rmdir "${MOUNT_POINT}" 2>/dev/null && echo "[OK] rmdir ${MOUNT_POINT}" || \
            echo "[INFO] ${MOUNT_POINT} 디렉토리 남음 (비어있지 않거나 다른 mount)"
    fi

    echo ""
    verify
    echo ""
    echo "[DONE] 1GB hugetlb 풀 원복 완료. 호스트 메모리 회수 확인:"
    free -h 2>/dev/null | head -3 | sed 's/^/  /'
}

case "${1:-}" in
    enable)    enable_pool "${2:-${DEFAULT_N}}" ;;
    disable)   disable_pool ;;
    verify|status) verify ;;
    *)         usage; exit 1 ;;
esac
