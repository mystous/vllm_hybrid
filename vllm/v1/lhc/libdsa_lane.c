/*
 * LHC Phase 2/3 — DSA lane shared library.
 *
 * Exports four C entry points consumable from Python ctypes:
 *   int  dsa_lane_init(const char *dev_path);
 *   int  dsa_lane_memcpy(void *dst, const void *src, size_t n);
 *   void dsa_lane_stats(uint64_t *ops, uint64_t *bytes, uint64_t *fails);
 *   void dsa_lane_close(void);
 *
 * Mechanism:
 *   - Dedicated WQ (mode=dedicated, type=user): MOVDIR64B to the mmap'd
 *     portal. No PASID required.
 *   - Shared WQ (mode=shared, type=user): request PASID-XSAVE perm via
 *     arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILE_DATA=18 / PASID=17),
 *     then submit via ENQCMD. ENQCMD writes the PASID-tagged descriptor
 *     into the shared work queue and reports retry via ZF.
 *
 *   The library auto-detects WQ mode by reading /sys/bus/dsa/devices/
 *   <wq_name>/mode at init time. Phase 3 host config exposes wq{0..1}.{0..3}
 *   in shared mode (host-owned, multi-engine ratio 4-engine/4-WQ).
 *
 *   Poll the completion record. Pure CPU-side memcpy from the caller's
 *   viewpoint — no PCIe-GPU traffic.
 *
 * Build:
 *   gcc -O3 -march=native -mmovdir64b -menqcmd -fPIC -shared \
 *       -o libdsa_lane.so libdsa_lane.c -pthread
 *
 * Caveats:
 *  * The dedicated WQ binds to *one* mm via cdev open; SWQ binds to many
 *    via PASID. Either way the lib opens the fd once at init and reuses
 *    it from any caller thread.
 *  * Single descriptor + completion record is used (sync submit). The
 *    caller's chunk is what determines bandwidth.
 *  * Failures (timeout, bad status) fall back to nothing — caller is
 *    expected to use plain memcpy after dsa_lane_memcpy < 0.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <linux/idxd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

#ifndef ARCH_GET_XCOMP_PERM
#define ARCH_GET_XCOMP_PERM 0x1022
#endif
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_PASID
#define XFEATURE_PASID 10  /* XSAVE bit 10 = PASID MSR component */
#endif

static int g_fd = -1;
static volatile void *g_portal = NULL;
static int g_is_shared = 0;  /* 1 if SWQ → use ENQCMD; 0 if DWQ → MOVDIR64B */
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static atomic_uint_fast64_t g_ops = 0;
static atomic_uint_fast64_t g_bytes = 0;
static atomic_uint_fast64_t g_fail = 0;

static inline void movdir64b(volatile void *p, const void *d) {
    _movdir64b((void *)p, d);
}

/* ENQCMD: submit 64B descriptor to shared WQ. ZF=1 → retry. */
static inline int enqcmd_submit(volatile void *p, const void *d) {
    uint8_t retry;
    asm volatile (
        ".byte 0xf2, 0x0f, 0x38, 0xf8, 0x02\n\t"  /* enqcmd (%rdx),%rax */
        "setz %0"
        : "=r"(retry)
        : "a"(p), "d"(d)
        : "memory", "cc");
    return retry ? -1 : 0;
}

/* Read WQ mode from /sys/bus/dsa/devices/<wq_name>/mode.
   Returns 1 if "shared", 0 if "dedicated" or unknown. */
static int detect_shared_mode(const char *dev_path) {
    /* dev_path = "/dev/dsa/wqX.Y" — strip prefix to get wq name. */
    const char *p = strrchr(dev_path, '/');
    if (!p) return 0;
    p++;
    char sysfs_path[256];
    snprintf(sysfs_path, sizeof(sysfs_path),
             "/sys/bus/dsa/devices/%s/mode", p);
    FILE *f = fopen(sysfs_path, "r");
    if (!f) return 0;
    char buf[32] = {0};
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) return 0;
    return (strncmp(buf, "shared", 6) == 0) ? 1 : 0;
}

/* Request the kernel grant XSAVE-PASID permission for this thread. Some
 * kernels gate ENQCMD on this. Idempotent. */
static int request_pasid_perm(void) {
    /* arch_prctl is syscall 158 on x86_64. */
    long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_PASID);
    /* EINVAL on kernels without XFEATURE_PASID gate → treat as benign. */
    if (rc != 0 && errno != EINVAL && errno != ENOSYS) {
        fprintf(stderr, "[libdsa_lane] arch_prctl(PASID) rc=%ld errno=%d\n",
                rc, errno);
    }
    return 0;
}

static inline double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int dsa_lane_init(const char *dev_path) {
    pthread_mutex_lock(&g_mu);
    if (g_fd >= 0) {
        pthread_mutex_unlock(&g_mu);
        return 0; /* already open */
    }
    const char *dev = (dev_path && *dev_path) ? dev_path : "/dev/dsa/wq0.0";

    g_is_shared = detect_shared_mode(dev);
    if (g_is_shared) {
        /* Request XSAVE-PASID component permission (harmless if N/A). */
        request_pasid_perm();
    }

    int fd = open(dev, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "[libdsa_lane] open(%s) failed: %s\n",
                dev, strerror(errno));
        pthread_mutex_unlock(&g_mu);
        return -1;
    }
    void *p = mmap(NULL, 4096, PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "[libdsa_lane] mmap portal failed: %s\n", strerror(errno));
        close(fd);
        pthread_mutex_unlock(&g_mu);
        return -2;
    }
    g_fd = fd;
    g_portal = p;
    pthread_mutex_unlock(&g_mu);
    return 0;
}

int dsa_lane_memcpy(void *dst, const void *src, size_t n) {
    if (g_portal == NULL) return -1;
    if (n == 0) return 0;
    /* DSA descriptors are CRAV/RCR/BOF, opcode MEMMOVE. */
    struct dsa_hw_desc desc __attribute__((aligned(64)));
    struct dsa_completion_record comp __attribute__((aligned(32)));
    memset(&desc, 0, sizeof(desc));
    memset(&comp, 0, sizeof(comp));
    desc.flags = IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_BOF;
    desc.opcode = DSA_OPCODE_MEMMOVE;
    desc.completion_addr = (uint64_t)&comp;
    desc.xfer_size = (uint32_t)n; /* max 2 GB per descriptor */
    desc.src_addr = (uint64_t)src;
    desc.dst_addr = (uint64_t)dst;

    if (g_is_shared) {
        /* Shared WQ: ENQCMD with bounded retry. ZF=1 → SWQ full. */
        int retries = 0;
        while (enqcmd_submit(g_portal, &desc) != 0) {
            _mm_pause();
            if (++retries > 100000) {
                atomic_fetch_add(&g_fail, 1);
                return -4;
            }
        }
    } else {
        movdir64b(g_portal, &desc);
    }

    /* poll completion with a 1-second timeout (generous) */
    double t0 = now_us();
    while (comp.status == DSA_COMP_NONE) {
        _mm_pause();
        if ((now_us() - t0) > 1e6) {
            atomic_fetch_add(&g_fail, 1);
            return -3;
        }
    }
    if ((comp.status & DSA_COMP_STATUS_MASK) != DSA_COMP_SUCCESS) {
        atomic_fetch_add(&g_fail, 1);
        return -(int)comp.status;
    }
    atomic_fetch_add(&g_ops, 1);
    atomic_fetch_add(&g_bytes, n);
    return 0;
}

/* Returns triple (ops, bytes, failures) packed via OUT params. */
void dsa_lane_stats(uint64_t *ops, uint64_t *bytes, uint64_t *fails) {
    if (ops)   *ops   = atomic_load(&g_ops);
    if (bytes) *bytes = atomic_load(&g_bytes);
    if (fails) *fails = atomic_load(&g_fail);
}

void dsa_lane_close(void) {
    pthread_mutex_lock(&g_mu);
    if (g_portal) { munmap((void *)g_portal, 4096); g_portal = NULL; }
    if (g_fd >= 0) { close(g_fd); g_fd = -1; }
    pthread_mutex_unlock(&g_mu);
}
