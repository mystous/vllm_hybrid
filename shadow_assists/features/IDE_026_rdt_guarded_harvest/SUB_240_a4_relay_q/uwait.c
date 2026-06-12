/* SUB_240 [A4 RELAY-Q 프리미티브] — tpause/umwait 대기 .so (ctypes 용)
 *
 * u_tpause(cycles): TSC cycles 만큼 C0.2 에서 대기 (syscall 0, 트래픽 0, 전력 절감)
 *   umwait max (실측 100k TSC) 초과분은 체인.
 * 빌드: gcc -O2 -march=native -fPIC -shared -o libuwait.so uwait.c
 */
#include <stdint.h>
#include <immintrin.h>

static inline uint64_t rdtsc(void) { return __rdtsc(); }

void u_tpause(uint64_t cycles) {
    uint64_t deadline = rdtsc() + cycles;
    /* C0.2 (state=0) — 더 깊은 절전, wake 지연 수백 ns */
    while (rdtsc() < deadline) {
        _tpause(0, deadline);
    }
}

/* 향후 umonitor/umwait (주소 감시) 확장 자리 */
void u_umwait_addr(volatile void *addr, uint64_t cycles) {
    uint64_t deadline = rdtsc() + cycles;
    _umonitor((void *)addr);
    _umwait(0, deadline);
}
