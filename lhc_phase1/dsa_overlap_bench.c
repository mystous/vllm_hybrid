/*
 * LHC Phase 1 — DSA async-overlap proof.
 *
 * Question: while DSA performs a large copy, how much *useful CPU work* can the
 * submitting core do? This is the core LHC thesis — DSA is an orthogonal lane:
 * the copy consumes a DMA engine, not CPU compute/load-store bandwidth.
 *
 * Method:
 *   1. Calibrate a pure integer-accumulate kernel: ops/us with NO copy running.
 *   2. DSA path: submit one large MEMMOVE, then run the kernel, checking the
 *      completion record only every CHUNK ops; record ops finished during copy.
 *   3. glibc path: the CPU IS the copy — 0 overlap ops are possible by
 *      construction (memcpy fully occupies the core for its whole duration).
 *
 *   overlap_efficiency = ops_during_dsa_copy / (rate * dsa_copy_us)
 *   -> fraction of the copy window the core stayed free for other work.
 *
 * Build: gcc -O3 -march=native -mmovdir64b -o dsa_overlap_bench dsa_overlap_bench.c
 * Run:   sudo ./dsa_overlap_bench
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
#include <time.h>
#include <unistd.h>

static double now_us(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}

/* opaque integer kernel the optimizer cannot elide; ~1 dependent add/iter */
static volatile uint64_t g_sink;
static uint64_t work(uint64_t iters, uint64_t seed){
    uint64_t a=seed;
    for(uint64_t i=0;i<iters;i++){ a = a*6364136223846793005ULL + 1442695040888963407ULL; a ^= a>>31; }
    return a;
}

int main(void){
    int fd = open("/dev/dsa/wq0.0", O_RDWR);
    if(fd<0){perror("open wq");return 1;}
    void *portal = mmap(NULL,4096,PROT_WRITE,MAP_SHARED|MAP_POPULATE,fd,0);
    if(portal==MAP_FAILED){perror("mmap");return 1;}

    /* 1. calibrate kernel rate (ops/us) */
    uint64_t cal = 2000000000ULL;
    double c0=now_us(); g_sink=work(cal,12345); double c1=now_us();
    double rate = cal/(c1-c0);          /* ops per us */
    printf("{\"phase\":\"calibrate\",\"ops_per_us\":%.1f}\n", rate); fflush(stdout);

    size_t sizes[]={16<<20, 256<<20};
    for(int s=0;s<2;s++){
        size_t size=sizes[s];
        void*src=aligned_alloc(64,size),*dst=aligned_alloc(64,size);
        memset(src,0xA5,size); memset(dst,0,size);
        struct dsa_hw_desc desc __attribute__((aligned(64)));
        struct dsa_completion_record comp __attribute__((aligned(32)));
        memset(&desc,0,sizeof desc); memset(&comp,0,sizeof comp);
        desc.flags=IDXD_OP_FLAG_CRAV|IDXD_OP_FLAG_RCR|IDXD_OP_FLAG_BOF;
        desc.opcode=DSA_OPCODE_MEMMOVE;
        desc.completion_addr=(uint64_t)&comp;
        desc.src_addr=(uint64_t)src; desc.dst_addr=(uint64_t)dst; desc.xfer_size=size;

        /* warmup */
        comp.status=DSA_COMP_NONE; _movdir64b(portal,&desc);
        while(comp.status==DSA_COMP_NONE) _mm_pause();

        /* 2. submit then do useful work, checking completion every CHUNK ops */
        const uint64_t CHUNK=200000;
        comp.status=DSA_COMP_NONE;
        double t0=now_us();
        _movdir64b(portal,&desc);
        uint64_t done=0, a=999;
        while(comp.status==DSA_COMP_NONE){ a=work(CHUNK,a); done+=CHUNK; }
        double t1=now_us();
        g_sink=a;
        double copy_us=t1-t0;
        double overlap_ops_us = done/copy_us;
        double eff = (done)/(rate*copy_us);   /* fraction of window core was free */
        printf("{\"phase\":\"overlap\",\"size_bytes\":%zu,\"copy_us\":%.1f,"
               "\"overlap_ops\":%llu,\"overlap_ops_per_us\":%.1f,"
               "\"core_free_fraction\":%.4f,\"bw_GBs\":%.2f}\n",
               size, copy_us, (unsigned long long)done, overlap_ops_us, eff,
               (double)size/1000.0/copy_us);
        fflush(stdout);
        free(src); free(dst);
    }
    munmap(portal,4096); close(fd);
    return 0;
}
