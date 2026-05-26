#!/usr/bin/env bash
# IRQ smp_affinity reroute — nvidia GPU IRQs → cpu 6-23 (NUMA 0 free area)
# 사용자 constraint: kernel 여유 0-5, 56-61 / vllm vanilla 6-30 / etc.
# nvidia IRQ 가 vllm 영역 (6-30) 에 떨어져도 PCIe affinity 활용 가능, 단 cpu_fill 영역 (80-99) 침범은 차단

set -euo pipefail

# Mask for cpu 6-23 (18 cpus on NUMA 0, in vllm vanilla range)
# Bit positions 6..23 → hex
TARGET_MASK="00ffffc0"   # bits 6..23 set, lower 6 bits cleared
TARGET_LIST="6-23"

echo "=== nvidia IRQ identification ==="
NVIDIA_IRQS=$(grep -i nvidia /proc/interrupts | awk -F: '{print $1}' | tr -d ' ')
echo "found $(echo $NVIDIA_IRQS | wc -w) nvidia IRQs"

echo ""
echo "=== rerouting to cpu ${TARGET_LIST} (mask 0x${TARGET_MASK}) ==="
fail=0
for IRQ in $NVIDIA_IRQS; do
    if echo "${TARGET_MASK}" > /proc/irq/${IRQ}/smp_affinity 2>/dev/null; then
        : # success
    elif echo "${TARGET_LIST}" > /proc/irq/${IRQ}/smp_affinity_list 2>/dev/null; then
        :
    else
        echo "WARN: IRQ ${IRQ} affinity 변경 실패 (root + irqbalance disable 필요)"
        fail=$((fail+1))
    fi
done
echo "  failed: ${fail}"

echo ""
echo "=== disable irqbalance (자동 재배치 차단) ==="
sudo systemctl stop irqbalance 2>/dev/null && echo "  stopped" || echo "  already stopped or N/A"
sudo systemctl disable irqbalance 2>/dev/null && echo "  disabled" || echo "  already disabled or N/A"

echo ""
echo "=== verification ==="
for IRQ in $NVIDIA_IRQS; do
    L=$(cat /proc/irq/${IRQ}/smp_affinity_list 2>/dev/null)
    printf "IRQ %5s → cpu %s\n" "$IRQ" "$L"
done | head -10

echo ""
echo "=== expected: cpu_fill 영역 (80-99) nvidia IRQ landing ==="
echo "  (다음 측정 시 /proc/interrupts grep nvidia 확인)"
