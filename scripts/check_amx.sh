#!/bin/bash
# AMX 감지 문제 디버깅 스크립트
# H100 서버 (Intel Xeon 8480+)에서 실행하세요

echo "========================================"
echo "AMX Detection Debug Script v2"
echo "========================================"

echo ""
echo "1. CPU Model:"
grep "model name" /proc/cpuinfo | head -1

echo ""
echo "2. Kernel Version (AMX needs 5.16+):"
KERNEL_VERSION=$(uname -r)
echo "  $KERNEL_VERSION"

# Parse kernel version
MAJOR=$(echo $KERNEL_VERSION | cut -d. -f1)
MINOR=$(echo $KERNEL_VERSION | cut -d. -f2 | cut -d- -f1)
if [ "$MAJOR" -lt 5 ] || ([ "$MAJOR" -eq 5 ] && [ "$MINOR" -lt 16 ]); then
    echo "  ⚠️  WARNING: Kernel $MAJOR.$MINOR is too old for AMX (need 5.16+)"
else
    echo "  ✓ Kernel $MAJOR.$MINOR supports AMX"
fi

echo ""
echo "3. CPU Flags (AMX related) - checking both formats:"
echo "  Underscore format (amx_*):"
grep -o -E "amx_[a-z0-9]+" /proc/cpuinfo | sort -u | sed 's/^/    /'
echo "  Dash format (amx-*):"
grep -o -E "amx-[a-z0-9]+" /proc/cpuinfo | sort -u | sed 's/^/    /'

echo ""
echo "4. Full flags line search:"
grep "^flags" /proc/cpuinfo | head -1 | tr ' ' '\n' | grep -i amx | sed 's/^/    /'

echo ""
echo "5. Check specific AMX flags:"
for FLAG in amx_bf16 amx_int8 amx_tile amx-bf16 amx-int8 amx-tile; do
    if grep -q "$FLAG" /proc/cpuinfo; then
        echo "  ✓ $FLAG found"
    else
        echo "  ✗ $FLAG NOT found"
    fi
done

echo ""
echo "6. Kernel config for AMX (if available):"
if [ -f /proc/config.gz ]; then
    zcat /proc/config.gz 2>/dev/null | grep -i -E "(AMX|XSTATE)" | sed 's/^/    /' || echo "  No AMX config found"
elif [ -f /boot/config-$(uname -r) ]; then
    grep -i -E "(AMX|XSTATE)" /boot/config-$(uname -r) | sed 's/^/    /' || echo "  No AMX config found"
else
    echo "  Kernel config not accessible"
fi

echo ""
echo "7. lscpu flags (AMX):"
lscpu | grep -i "Flags" | tr ' ' '\n' | grep -i amx | sed 's/^/    /' || echo "  No AMX in lscpu flags"

echo ""
echo "8. XSTATE (AMX requires XSAVE support):"
if [ -d /sys/devices/system/cpu/cpu0/cpuid ]; then
    echo "  CPUID available"
else
    echo "  CPUID not directly accessible"
fi

# Check dmesg for AMX-related messages
echo ""
echo "9. dmesg AMX messages (may require sudo):"
dmesg 2>/dev/null | grep -i amx | tail -5 | sed 's/^/    /' || echo "  No AMX messages in dmesg (or no permission)"

echo ""
echo "10. Python vLLM detection test:"
python3 -c "
import sys
sys.path.insert(0, '$(pwd)')
try:
    from vllm.platforms.intel_cpu_utils import detect_intel_cpu_features
    features = detect_intel_cpu_features()
    print(f'  CPU: {features.model_name}')
    print(f'  AVX2: {features.avx2}')
    print(f'  AVX-512: {features.avx512f}')
    print(f'  AMX-BF16: {features.amx_bf16}')
    print(f'  AMX-INT8: {features.amx_int8}')
except Exception as e:
    print(f'  Error: {e}')
" 2>&1 | grep -v "^$"

echo ""
echo "11. Raw cpuinfo amx check:"
python3 -c "
with open('/proc/cpuinfo', 'r') as f:
    content = f.read()

# Check both raw and lowercase
raw_bf16 = 'amx_bf16' in content or 'amx-bf16' in content
raw_int8 = 'amx_int8' in content or 'amx-int8' in content
raw_tile = 'amx_tile' in content or 'amx-tile' in content

print(f'  amx_bf16/amx-bf16 in raw: {raw_bf16}')
print(f'  amx_int8/amx-int8 in raw: {raw_int8}')
print(f'  amx_tile/amx-tile in raw: {raw_tile}')

# Extract flags line
for line in content.split('\n'):
    if line.startswith('flags'):
        flags = line.split(':')[1].strip() if ':' in line else ''
        print(f'  Flags line length: {len(flags)} chars')
        amx_flags = [f for f in flags.split() if 'amx' in f.lower()]
        if amx_flags:
            print(f'  AMX flags found: {amx_flags}')
        else:
            print('  No AMX flags in flags line')
        break
"

echo ""
echo "========================================"
echo "진단 요약:"
echo "========================================"
echo ""
echo "AMX가 검출되지 않는 경우 확인할 사항:"
echo "  1. 커널 버전이 5.16 이상인지 확인"
echo "  2. 커널에 CONFIG_X86_INTEL_AMX=y 설정이 있는지 확인"
echo "  3. 컨테이너/VM 환경이라면 호스트에서 AMX 지원 확인"
echo "  4. BIOS에서 AMX가 비활성화되어 있지 않은지 확인"
echo ""
echo "========================================"
