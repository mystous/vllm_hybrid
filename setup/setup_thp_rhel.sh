#!/usr/bin/env bash
# =============================================================================
# setup_thp_rhel.sh — RHEL 계열 (RHEL/Rocky/Alma 8+) THP 관리 스크립트
# =============================================================================
#
# NinjaGap §03 Phase 1: 2MB Transparent Huge Pages (재부팅 불필요)
#
# 사용법:
#   sudo bash setup/setup_thp_rhel.sh enable          # 현재 세션에 THP always
#   bash       setup/setup_thp_rhel.sh verify         # 상태 조회 (sudo 불필요)
#   sudo bash setup/setup_thp_rhel.sh persist         # 재부팅 후에도 유지 (systemd)
#   sudo bash setup/setup_thp_rhel.sh disable         # 원복 (madvise 로 복귀)
#   sudo bash setup/setup_thp_rhel.sh status_detailed # AnonHugePages / per-PID 통계
#
# 권장 순서:
#   1) verify   — 현재 상태 확인
#   2) enable   — 실험 세션만 적용 (재부팅 원복)
#   3) 측정 후 효과 확인되면 persist 로 영구화, 아니면 disable 로 원복
#
# 주의:
#   - 컨테이너 안에서 /sys/kernel/mm 은 보통 read-only. 호스트에서 실행 필요.
#   - RHEL 은 부팅 시 tuned profile 이 THP 상태를 되돌릴 수 있음. persist 필요.
# =============================================================================
set -euo pipefail

THP_ENABLED=/sys/kernel/mm/transparent_hugepage/enabled
THP_DEFRAG=/sys/kernel/mm/transparent_hugepage/defrag
KHUGEPAGED_DEFRAG=/sys/kernel/mm/transparent_hugepage/khugepaged/defrag
SYSTEMD_UNIT=/etc/systemd/system/thp-always.service

usage() {
    grep -E '^#' "$0" | head -30
}

need_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "[ERROR] '$1' 명령은 root 권한이 필요합니다 (sudo 사용)." >&2
        exit 1
    fi
}

need_host() {
    if [[ ! -w "${THP_ENABLED}" ]]; then
        echo "[ERROR] ${THP_ENABLED} 에 쓸 수 없습니다." >&2
        echo "        컨테이너 안이면 호스트에서 실행하세요." >&2
        exit 1
    fi
}

read_active() {
    awk '{
        for (i=1;i<=NF;i++) {
            if ($i ~ /^\[/) { gsub(/[][]/, "", $i); print $i; exit }
        }
    }' "$1"
}

verify() {
    if [[ ! -r "${THP_ENABLED}" ]]; then
        echo "[WARN] ${THP_ENABLED} 읽기 불가. 컨테이너 안에서는 호스트 상태 확인 불가."
        return 0
    fi
    local en df kh
    en=$(read_active "${THP_ENABLED}")
    df=$(read_active "${THP_DEFRAG}")
    kh=$(read_active "${KHUGEPAGED_DEFRAG}" 2>/dev/null || echo "n/a")
    echo "[THP] enabled = ${en}"
    echo "[THP] defrag  = ${df}"
    echo "[THP] khugepaged/defrag = ${kh}"
    echo "[THP] /proc/meminfo:"
    grep -E "^(AnonHugePages|ShmemHugePages|HugePages_Total|HugePages_Free|Hugepagesize):" /proc/meminfo | sed 's/^/       /'
    if [[ "${en}" == "always" ]]; then
        echo "[OK] THP=always — §03 측정 조건 충족."
    else
        echo "[NOTICE] THP!=always. 측정 전에 'enable' 필요."
    fi
}

enable_runtime() {
    need_root enable
    need_host
    echo always > "${THP_ENABLED}"
    echo always > "${THP_DEFRAG}"
    # khugepaged defrag 도 active 로 — 이미 할당된 4KB anon 페이지도 promotion
    if [[ -w "${KHUGEPAGED_DEFRAG}" ]]; then
        echo 1 > "${KHUGEPAGED_DEFRAG}"
    fi
    echo "[OK] THP=always applied (runtime only)."
    verify
}

persist_systemd() {
    need_root persist
    cat > "${SYSTEMD_UNIT}" <<'EOF'
[Unit]
Description=Set Transparent Huge Pages to always for vLLM hybrid §03
After=sysinit.target local-fs.target
Before=tuned.service

[Service]
Type=oneshot
ExecStart=/bin/sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled && echo always > /sys/kernel/mm/transparent_hugepage/defrag && echo 1 > /sys/kernel/mm/transparent_hugepage/khugepaged/defrag"
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now thp-always.service
    echo "[OK] ${SYSTEMD_UNIT} 등록 + 활성."
    echo "     재부팅 후에도 THP=always 유지됨."
    echo "     원복: sudo bash $0 disable"
    verify
}

disable_thp() {
    need_root disable
    need_host
    # systemd unit 있으면 내리고 삭제
    if systemctl list-unit-files 2>/dev/null | grep -q '^thp-always.service'; then
        systemctl disable --now thp-always.service || true
        rm -f "${SYSTEMD_UNIT}"
        systemctl daemon-reload
        echo "[OK] thp-always.service 제거."
    fi
    echo madvise > "${THP_ENABLED}"
    echo madvise > "${THP_DEFRAG}"
    echo "[OK] THP=madvise 로 원복."
    verify
}

status_detailed() {
    verify
    echo ""
    echo "[THP] vLLM 관련 프로세스 AnonHugePages:"
    # api_server / EngineCore 프로세스 targeting
    local pids
    pids=$(pgrep -f 'api_server|EngineCore' || true)
    if [[ -z "${pids}" ]]; then
        echo "       (vLLM 프로세스 실행 중 아님)"
        return
    fi
    echo "PID      COMM                    AnonHugePages   RSS"
    for pid in ${pids}; do
        if [[ -r "/proc/${pid}/status" ]]; then
            local comm ahp rss
            comm=$(awk '/^Name:/ {print $2}' "/proc/${pid}/status")
            ahp=$(awk '/^AnonHugePages:/ {printf "%s %s", $2, $3}' "/proc/${pid}/status")
            rss=$(awk '/^VmRSS:/ {printf "%s %s", $2, $3}' "/proc/${pid}/status")
            printf "%-8s %-22s %-15s %s\n" "${pid}" "${comm}" "${ahp:-n/a}" "${rss:-n/a}"
        fi
    done
}

case "${1:-}" in
    enable)           enable_runtime ;;
    verify|status)    verify ;;
    status_detailed)  status_detailed ;;
    persist)          persist_systemd ;;
    disable)          disable_thp ;;
    *)                usage; exit 1 ;;
esac
