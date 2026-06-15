/*
 * LHC Phase 2 — DSA multi-thread submit bench.
 *
 * Single dedicated WQ (wq0.0) opened once by the process. N pthreads each
 * mmap the portal (same fd, same PASID) and submit descriptors against it
 * concurrently. Each thread owns its own descriptor + completion record so
 * we just stress the WQ pipeline.
 *
 * Goal: confirm that the wq0.0 single-engine cap (~31 GB/s observed in Phase 1
 * + descriptor pipelining at depth=32) is truly a single-engine HW limit,
 * not a submit-side serialization. If N threads also stall at 31 GB/s, we
 * have a hard HW gate that only multi-engine binding (host root) can lift.
 *
 * Build:
 *   gcc -O3 -march=native -mmovdir64b -pthread -o dsa_mt_bench dsa_mt_bench.c
 */

#define _GNU_SOURCE
#include <fcntl.h>
#include <immintrin.h>
#include <linux/idxd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static inline void movdir64b(volatile void *portal, const void *desc) {
    _movdir64b((void *)portal, desc);
}
static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

typedef struct {
    volatile void *portal;
    size_t chunk;
    int iters;
    atomic_int *go;
    double t0_us;
    double bytes_done;
    void *src, *dst;
    int tid;
} thr_arg_t;

static void *thread_fn(void *p) {
    thr_arg_t *a = (thr_arg_t *)p;
    struct dsa_hw_desc desc __attribute__((aligned(64)));
    struct dsa_completion_record comp __attribute__((aligned(32)));
    memset(&desc, 0, sizeof(desc));
    memset(&comp, 0, sizeof(comp));
    desc.flags = IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_BOF;
    desc.opcode = DSA_OPCODE_MEMMOVE;
    desc.completion_addr = (uint64_t)&comp;
    desc.xfer_size = a->chunk;
    desc.src_addr = (uint64_t)a->src;
    desc.dst_addr = (uint64_t)a->dst;

    /* wait for start signal */
    while (atomic_load_explicit(a->go, memory_order_acquire) == 0) _mm_pause();

    for (int i = 0; i < a->iters; i++) {
        comp.status = DSA_COMP_NONE;
        movdir64b(a->portal, &desc);
        volatile uint8_t *s = &comp.status;
        while (*s == DSA_COMP_NONE) _mm_pause();
        if ((*s & DSA_COMP_STATUS_MASK) != DSA_COMP_SUCCESS) {
            fprintf(stderr, "tid=%d status=0x%02x\n", a->tid, (unsigned)*s);
            return NULL;
        }
    }
    a->bytes_done = (double)a->chunk * a->iters;
    return NULL;
}

static void bench_mt(volatile void *portal, size_t chunk, int nthr, int iters) {
    thr_arg_t args[64];
    pthread_t ths[64];
    atomic_int go; atomic_init(&go, 0);
    for (int t = 0; t < nthr; t++) {
        args[t].portal = portal;
        args[t].chunk = chunk;
        args[t].iters = iters;
        args[t].go = &go;
        args[t].src = aligned_alloc(64, chunk);
        args[t].dst = aligned_alloc(64, chunk);
        memset(args[t].src, 0xA5, chunk);
        memset(args[t].dst, 0x00, chunk);
        args[t].tid = t;
        pthread_create(&ths[t], NULL, thread_fn, &args[t]);
    }
    /* short stagger to let threads spin up to the gate */
    struct timespec ts = {0, 5 * 1000 * 1000};
    nanosleep(&ts, NULL);
    double t0 = now_us();
    atomic_store_explicit(&go, 1, memory_order_release);
    for (int t = 0; t < nthr; t++) pthread_join(ths[t], NULL);
    double t1 = now_us();
    double total_bytes = 0;
    for (int t = 0; t < nthr; t++) {
        total_bytes += args[t].bytes_done;
        free(args[t].src); free(args[t].dst);
    }
    double elapsed_us = t1 - t0;
    double agg_gbs = total_bytes / 1000.0 / elapsed_us;
    printf("{\"backend\":\"dsa_mt\",\"chunk_bytes\":%zu,\"threads\":%d,"
           "\"iters_per_thread\":%d,\"elapsed_us\":%.1f,"
           "\"total_bytes\":%.0f,\"aggregate_GBs\":%.2f}\n",
           chunk, nthr, iters, elapsed_us, total_bytes, agg_gbs);
    fflush(stdout);
}

int main(int argc, char **argv) {
    const char *dev = "/dev/dsa/wq0.0";
    int fd = open(dev, O_RDWR);
    if (fd < 0) { perror(dev); return 1; }
    void *portal = mmap(NULL, 4096, PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
    if (portal == MAP_FAILED) { perror("mmap"); return 1; }

    size_t chunks[] = {1ULL<<20, 16ULL<<20};
    int thrs[]      = {1, 2, 4, 8, 16};
    int iters[]     = {200, 50}; /* per thread */

    for (int ci = 0; ci < (int)(sizeof(chunks)/sizeof(chunks[0])); ci++) {
        for (int ti = 0; ti < (int)(sizeof(thrs)/sizeof(thrs[0])); ti++) {
            bench_mt(portal, chunks[ci], thrs[ti], iters[ci]);
        }
    }
    munmap(portal, 4096); close(fd);
    return 0;
}
