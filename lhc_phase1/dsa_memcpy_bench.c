/*
 * LHC Phase 1 — DSA microbench (C, userspace MOVDIR64B / dedicated-WQ path).
 *
 * Why MOVDIR64B + dedicated WQ:
 *   The earlier ENQCMD/shared-WQ path hung on PASID handshake. A *dedicated*
 *   WQ of type=user has its WQCFG.PASID bound to the opening process's mm by
 *   the kernel idxd cdev driver (iommu_sva_bind on open), so per-descriptor
 *   PASID is unnecessary and MOVDIR64B (which does not carry a PASID) is the
 *   correct, deterministic submission instruction.
 *
 * Prereq (host root):
 *   accel-config disable-wq dsa0/wq0.0 ; accel-config disable-device dsa0
 *   accel-config config-wq dsa0/wq0.0 --mode=dedicated --type=user --name=lhc \
 *       --priority=10 --block-on-fault=1 --wq-size=64 \
 *       --max-transfer-size=2147483648 --max-batch-size=1024 --group-id=0
 *   accel-config config-engine dsa0/engine0.0 --group-id=0
 *   accel-config enable-device dsa0 ; accel-config enable-wq dsa0/wq0.0
 *
 * Build & run (device node is root-only -> run via sudo):
 *   gcc -O3 -march=native -mmovdir64b -o dsa_memcpy_bench dsa_memcpy_bench.c
 *   sudo ./dsa_memcpy_bench            # all ops
 *   sudo ./dsa_memcpy_bench memcpy     # single op: memcpy|memfill|compare
 *
 * Emits one JSON line per (op,size) to stdout.
 */

#define _GNU_SOURCE
#include <fcntl.h>
#include <immintrin.h>
#include <linux/idxd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* MOVDIR64B: atomically write a 64-byte descriptor to the WQ portal. */
static inline void movdir64b(volatile void *portal, const void *desc) {
    _movdir64b((void *)portal, desc);
}

static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

/* poll completion record status with a wall-clock timeout (us). returns status. */
static uint8_t poll_completion(volatile uint8_t *status, double timeout_us) {
    double t0 = now_us();
    while (*status == DSA_COMP_NONE) {
        _mm_pause();
        if ((now_us() - t0) > timeout_us) return 0xFF; /* sentinel: timeout */
    }
    return *status;
}

#define OP_MEMCPY  0
#define OP_MEMFILL 1
#define OP_COMPARE 2
static const char *op_name[] = {"memcpy", "memfill", "compare"};

static int bench_one(volatile void *portal, int op, size_t size, int iters) {
    void *src = aligned_alloc(64, size);
    void *dst = aligned_alloc(64, size);
    if (!src || !dst) { fprintf(stderr, "alloc fail %zu\n", size); return -1; }
    /* pre-fault BOTH buffers so we measure DMA, not minor page faults */
    memset(src, 0xA5, size);
    memset(dst, 0x00, size);

    struct dsa_hw_desc desc __attribute__((aligned(64)));
    struct dsa_completion_record comp __attribute__((aligned(32)));
    memset(&desc, 0, sizeof(desc));
    memset(&comp, 0, sizeof(comp));

    /* CRAV: completion-record-addr valid, RCR: request completion record,
     * BOF: block on (page) fault rather than erroring out. */
    desc.flags = IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_BOF;
    desc.completion_addr = (uint64_t)&comp;
    desc.xfer_size = size;

    switch (op) {
    case OP_MEMCPY:
        desc.opcode = DSA_OPCODE_MEMMOVE;
        desc.src_addr = (uint64_t)src;
        desc.dst_addr = (uint64_t)dst;
        break;
    case OP_MEMFILL:
        desc.opcode = DSA_OPCODE_MEMFILL;
        desc.pattern = 0xA5A5A5A5A5A5A5A5ULL; /* src_addr union == pattern */
        desc.dst_addr = (uint64_t)dst;
        break;
    case OP_COMPARE:
        desc.opcode = DSA_OPCODE_COMPARE;
        memset(dst, 0xA5, size); /* make them equal */
        desc.src_addr  = (uint64_t)src;
        desc.src2_addr = (uint64_t)dst; /* dst_addr union == src2_addr */
        break;
    }

    /* warmup + correctness check */
    for (int i = 0; i < 5; i++) {
        comp.status = DSA_COMP_NONE;
        movdir64b(portal, &desc);
        uint8_t st = poll_completion(&comp.status, 5e6);
        if (DSA_COMP_STATUS(st) != DSA_COMP_SUCCESS) {
            fprintf(stderr, "[%s %zu] warmup status=0x%02x (0xff=timeout)\n",
                    op_name[op], size, st);
            free(src); free(dst);
            return -1;
        }
    }
    if (op == OP_MEMCPY && memcmp(src, dst, size) != 0) {
        fprintf(stderr, "[memcpy %zu] DATA MISMATCH\n", size);
        free(src); free(dst); return -1;
    }

    double *samples = malloc(sizeof(double) * iters);
    for (int i = 0; i < iters; i++) {
        comp.status = DSA_COMP_NONE;
        double t0 = now_us();
        movdir64b(portal, &desc);
        poll_completion(&comp.status, 5e6);
        double t1 = now_us();
        samples[i] = t1 - t0;
    }
    qsort(samples, iters, sizeof(double), cmp_double);
    double p50 = samples[iters/2];
    double p99 = samples[(int)(iters*0.99)];
    double bw_gbs = (double)size / 1000.0 / p50;   /* bytes/us == GB/s (1e9) */
    printf("{\"backend\":\"dsa_movdir64b_dedicated\",\"op\":\"%s\",\"size_bytes\":%zu,"
           "\"lat_us_p50\":%.3f,\"lat_us_p99\":%.3f,\"bw_GBs_p50\":%.2f,\"iters\":%d}\n",
           op_name[op], size, p50, p99, bw_gbs, iters);
    fflush(stdout);
    free(samples); free(src); free(dst);
    return 0;
}

int main(int argc, char **argv) {
    const char *dev = "/dev/dsa/wq0.0";
    int fd = open(dev, O_RDWR);
    if (fd < 0) { perror(dev); return 1; }
    /* dedicated-WQ portal: 4KB, mapped WRITE-only at offset 0 */
    void *portal = mmap(NULL, 4096, PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
    if (portal == MAP_FAILED) { perror("mmap portal"); return 1; }

    size_t sizes[] = {4096, 65536, 1<<20, 16<<20, 256<<20};
    int    iters[] = {5000,  5000,  1000,  300,    40};
    int    nsz = sizeof(sizes)/sizeof(sizes[0]);

    int only = -1;
    if (argc > 1) {
        for (int o = 0; o < 3; o++) if (!strcmp(argv[1], op_name[o])) only = o;
        if (only < 0) { fprintf(stderr, "unknown op '%s'\n", argv[1]); return 2; }
    }

    for (int o = 0; o < 3; o++) {
        if (only >= 0 && o != only) continue;
        for (int i = 0; i < nsz; i++)
            if (bench_one(portal, o, sizes[i], iters[i]) < 0) {
                fprintf(stderr, "abort op=%s size=%zu\n", op_name[o], sizes[i]);
                break;
            }
    }

    munmap(portal, 4096); close(fd);
    return 0;
}
