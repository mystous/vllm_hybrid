/*
 * LHC Phase 2 — DSA multi-descriptor pipelined bench.
 *
 * Phase 1 single-descriptor sync submission topped out at ~31 GB/s on wq0.0
 * (single DSA engine). In Phase 2 we pipeline N in-flight descriptors against
 * the same dedicated WQ (wq0.0 has size=64, max-batch=1024) and let the DSA
 * scheduler dispatch them across the 4 engines bound to dsa0/group0.
 *
 * Two strategies, both implemented:
 *   (A) descriptor pipelining: split a single transfer into K chunks, submit
 *       all K with MOVDIR64B back-to-back, then poll the completion array.
 *       Single host thread; the WQ HW handles distribution to engines.
 *   (B) per-chunk batch descriptor (DSA batch op). Same effect, lower portal
 *       traffic — but requires batch-descriptor support which adds setup.
 *       Phase 2.1 implements (A) only; (B) is a Phase 2.2 optimization.
 *
 * NOTE: dsa1 is disabled and container sysfs is read-only, so cross-socket
 * measurement is skipped here. The Phase 2 gate is "single-device aggregate
 * BW >= 0.8 * cudaMemcpy" which is the within-node usage pattern in vLLM
 * (KV swap NUMA-local).
 *
 * Build:
 *   gcc -O3 -march=native -mmovdir64b -pthread \
 *       -o dsa_multi_engine_bench dsa_multi_engine_bench.c
 * Run:
 *   ./dsa_multi_engine_bench
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

/* Run one (total_size, depth) configuration. Returns 0 on success.
 * depth = number of pipelined chunks; each chunk = total_size / depth.
 * iters = number of repetitions for percentile stats.
 */
static int bench_pipeline(volatile void *portal,
                          size_t total_size,
                          int depth,
                          int iters)
{
    if (depth < 1) depth = 1;
    if (depth > 64) depth = 64; /* WQ size */
    size_t chunk = total_size / depth;
    if (chunk * depth != total_size) {
        /* require evenly divisible to keep math simple */
        chunk = (total_size + depth - 1) / depth;
    }
    /* Single src/dst pair, divided into `depth` regions. */
    void *src = aligned_alloc(64, chunk * depth);
    void *dst = aligned_alloc(64, chunk * depth);
    if (!src || !dst) { fprintf(stderr, "alloc fail %zu\n", chunk * depth); return -1; }
    memset(src, 0xA5, chunk * depth);
    memset(dst, 0x00, chunk * depth);

    /* One descriptor + completion record per in-flight slot. */
    struct dsa_hw_desc *descs = aligned_alloc(64, sizeof(struct dsa_hw_desc) * depth);
    struct dsa_completion_record *comps =
        aligned_alloc(32, sizeof(struct dsa_completion_record) * depth);
    if (!descs || !comps) { fprintf(stderr, "desc alloc fail\n"); return -1; }
    memset(descs, 0, sizeof(struct dsa_hw_desc) * depth);
    memset(comps, 0, sizeof(struct dsa_completion_record) * depth);

    for (int k = 0; k < depth; k++) {
        descs[k].flags = IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_BOF;
        descs[k].opcode = DSA_OPCODE_MEMMOVE;
        descs[k].completion_addr = (uint64_t)&comps[k];
        descs[k].xfer_size = chunk;
        descs[k].src_addr = (uint64_t)src + k * chunk;
        descs[k].dst_addr = (uint64_t)dst + k * chunk;
    }

    /* warmup */
    for (int w = 0; w < 3; w++) {
        for (int k = 0; k < depth; k++) comps[k].status = DSA_COMP_NONE;
        for (int k = 0; k < depth; k++) movdir64b(portal, &descs[k]);
        for (int k = 0; k < depth; k++) {
            volatile uint8_t *s = &comps[k].status;
            double t0 = now_us();
            while (*s == DSA_COMP_NONE) {
                _mm_pause();
                if ((now_us() - t0) > 5e6) {
                    fprintf(stderr, "warmup timeout depth=%d k=%d size=%zu\n",
                            depth, k, total_size);
                    return -1;
                }
            }
            if ((*s & DSA_COMP_STATUS_MASK) != DSA_COMP_SUCCESS) {
                fprintf(stderr, "warmup status=0x%02x depth=%d k=%d\n",
                        (unsigned)*s, depth, k);
                return -1;
            }
        }
    }
    if (memcmp(src, dst, chunk * depth) != 0) {
        fprintf(stderr, "depth=%d size=%zu DATA MISMATCH\n", depth, total_size);
        return -1;
    }

    /* timed runs */
    double *samples = malloc(sizeof(double) * iters);
    for (int it = 0; it < iters; it++) {
        for (int k = 0; k < depth; k++) comps[k].status = DSA_COMP_NONE;
        double t0 = now_us();
        for (int k = 0; k < depth; k++) movdir64b(portal, &descs[k]);
        /* poll all completions */
        for (int k = 0; k < depth; k++) {
            volatile uint8_t *s = &comps[k].status;
            while (*s == DSA_COMP_NONE) _mm_pause();
        }
        double t1 = now_us();
        samples[it] = t1 - t0;
    }
    qsort(samples, iters, sizeof(double), cmp_double);
    double p50 = samples[iters / 2];
    double p99 = samples[(int)(iters * 0.99)];
    double bw_p50 = (double)(chunk * depth) / 1000.0 / p50; /* GB/s */
    double bw_p99 = (double)(chunk * depth) / 1000.0 / p99;
    printf("{\"backend\":\"dsa_pipeline\",\"op\":\"memcpy\","
           "\"total_bytes\":%zu,\"depth\":%d,\"chunk_bytes\":%zu,"
           "\"lat_us_p50\":%.2f,\"lat_us_p99\":%.2f,"
           "\"bw_GBs_p50\":%.2f,\"bw_GBs_p99\":%.2f,\"iters\":%d}\n",
           chunk * depth, depth, chunk, p50, p99, bw_p50, bw_p99, iters);
    fflush(stdout);
    free(samples); free(comps); free(descs); free(src); free(dst);
    return 0;
}

int main(int argc, char **argv) {
    const char *dev = "/dev/dsa/wq0.0";
    int fd = open(dev, O_RDWR);
    if (fd < 0) { perror(dev); return 1; }
    void *portal = mmap(NULL, 4096, PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
    if (portal == MAP_FAILED) { perror("mmap portal"); return 1; }

    /* sweep: total_size x depth */
    size_t sizes[] = {1ULL<<20, 16ULL<<20, 64ULL<<20, 256ULL<<20};
    int depths[]   = {1, 2, 4, 8, 16, 32};
    int iters[]    = {1000, 200, 100, 40};

    int nsz = sizeof(sizes)/sizeof(sizes[0]);
    int nd  = sizeof(depths)/sizeof(depths[0]);

    for (int si = 0; si < nsz; si++) {
        for (int di = 0; di < nd; di++) {
            /* skip depths that would create <64KB chunks (descriptor cost dominates) */
            if (sizes[si] / depths[di] < 65536) continue;
            if (bench_pipeline(portal, sizes[si], depths[di], iters[si]) < 0) {
                fprintf(stderr, "abort size=%zu depth=%d\n", sizes[si], depths[di]);
            }
        }
    }

    munmap(portal, 4096); close(fd);
    return 0;
}
