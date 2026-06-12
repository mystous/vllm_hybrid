/* SUB_232 [D19] — 크기-적응 memcpy 디스패처 (LD_PRELOAD)
 *
 * 경로 선택 (실측 ISA: erms+fsrm+avx512):
 *   size < θ_NT : rep movsb (FSRM µcode fast path — glibc 동급, 레지스터 압박 0)
 *   size ≥ θ_NT : AVX-512 NT-store (vmovntdq) — RFO 생략 + LLC 비오염
 *                 (head/tail 비정렬 구간은 rep movsb)
 *
 * θ_NT 기본 8 MiB, env TUNED_MEMCPY_NT_BYTES 로 조정.
 * TUNED_MEMCPY_STATS=1 이면 exit 시 호출 통계 stderr 출력.
 *
 * 빌드: gcc -O2 -march=native -fPIC -shared -o libtunedmemcpy.so tuned_memcpy.c
 * 사용: LD_PRELOAD=$PWD/libtunedmemcpy.so <prog>
 */
#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

static size_t g_nt_thresh = 8u << 20;   /* 8 MiB */
static _Atomic unsigned long g_n_small = 0, g_n_nt = 0;
static _Atomic unsigned long long g_b_small = 0, g_b_nt = 0;
static int g_stats = 0;

__attribute__((constructor))
static void init_thresh(void) {
    const char *e = getenv("TUNED_MEMCPY_NT_BYTES");
    if (e) { size_t v = strtoull(e, NULL, 10); if (v >= 4096) g_nt_thresh = v; }
    g_stats = getenv("TUNED_MEMCPY_STATS") != NULL;
}

__attribute__((destructor))
static void dump_stats(void) {
    if (g_stats)
        fprintf(stderr, "[tuned_memcpy] small(rep movsb): n=%lu bytes=%llu | nt: n=%lu bytes=%llu (thresh=%zu)\n",
                g_n_small, g_b_small, g_n_nt, g_b_nt, g_nt_thresh);
}

static inline void rep_movsb(void *d, const void *s, size_t n) {
    __asm__ volatile("rep movsb"
                     : "+D"(d), "+S"(s), "+c"(n)
                     :
                     : "memory");
}

static void nt_copy(char *d, const char *s, size_t n) {
    /* head: dst 를 64 B 정렬까지 */
    size_t head = (64 - ((uintptr_t)d & 63)) & 63;
    if (head) { rep_movsb(d, s, head); d += head; s += head; n -= head; }
    /* body: 64 B 단위 NT store (src 는 비정렬 load 허용) */
    size_t blocks = n >> 6;
    for (size_t i = 0; i < blocks; i++) {
        __m512i v = _mm512_loadu_si512((const void *)(s + (i << 6)));
        _mm512_stream_si512((void *)(d + (i << 6)), v);
    }
    _mm_sfence();
    size_t done = blocks << 6;
    if (n - done) rep_movsb(d + done, s + done, n - done);
}

void *memcpy(void *dst, const void *src, size_t n) {
    if (n >= g_nt_thresh) {
        if (g_stats) { g_n_nt++; g_b_nt += n; }
        nt_copy((char *)dst, (const char *)src, n);
    } else {
        if (g_stats) { g_n_small++; g_b_small += n; }
        rep_movsb(dst, src, n);
    }
    return dst;
}
