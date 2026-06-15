/*
 * LHC Phase 1 — DSA correctness VERIFICATION (not a perf bench).
 * Proves the DSA engine genuinely moves/fills/compares bytes, with a unique
 * per-byte pattern, a pre-copy inequality assertion, exact completion status
 * codes, and post-op byte-exact checks. No timing — purely "is it real?".
 *
 * Build: gcc -O2 -march=native -mmovdir64b -o dsa_verify dsa_verify.c
 * Run:   sudo ./dsa_verify
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

static uint8_t submit(volatile void*portal, struct dsa_hw_desc*d, struct dsa_completion_record*c){
    c->status=DSA_COMP_NONE;
    _movdir64b((void*)portal,d);
    double t0=now_us();
    while(c->status==DSA_COMP_NONE){ _mm_pause(); if(now_us()-t0>5e6) return 0xFF; }
    return c->status;
}

int main(void){
    int fd=open("/dev/dsa/wq0.0",O_RDWR);
    if(fd<0){perror("open");return 1;}
    void*portal=mmap(NULL,4096,PROT_WRITE,MAP_SHARED|MAP_POPULATE,fd,0);
    if(portal==MAP_FAILED){perror("mmap");return 1;}

    const size_t N=4*1024*1024;   /* 4 MB */
    uint8_t*src=aligned_alloc(64,N),*dst=aligned_alloc(64,N);
    struct dsa_hw_desc desc __attribute__((aligned(64)));
    struct dsa_completion_record comp __attribute__((aligned(32)));
    int fail=0;

    /* unique per-byte pattern so a stale/zeroed dst cannot masquerade as success */
    for(size_t i=0;i<N;i++){ src[i]=(uint8_t)(i*131u+7u); dst[i]=(uint8_t)(~src[i]); }

    /* ---- TEST 1: MEMMOVE ---- */
    if(memcmp(src,dst,N)==0){ printf("PRE-CHECK FAIL: dst already == src\n"); return 2; }
    memset(&desc,0,sizeof desc); memset(&comp,0,sizeof comp);
    desc.flags=IDXD_OP_FLAG_CRAV|IDXD_OP_FLAG_RCR|IDXD_OP_FLAG_BOF;
    desc.opcode=DSA_OPCODE_MEMMOVE; desc.completion_addr=(uint64_t)&comp;
    desc.src_addr=(uint64_t)src; desc.dst_addr=(uint64_t)dst; desc.xfer_size=N;
    uint8_t st=submit(portal,&desc,&comp);
    int bytes_ok = (memcmp(src,dst,N)==0);
    printf("TEST memmove : status=0x%02x (expect 0x01 SUCCESS) | bytes_completed=%u/%zu | dst==src? %s\n",
           st, comp.bytes_completed, N, bytes_ok?"YES":"NO");
    /* tamper one byte and confirm our own check can detect mismatch (anti-false-positive) */
    dst[N/2]^=0xFF; printf("  self-test (flip 1 byte) -> dst==src? %s (expect NO)\n", memcmp(src,dst,N)==0?"YES":"NO"); dst[N/2]^=0xFF;
    if(DSA_COMP_STATUS(st)!=DSA_COMP_SUCCESS || !bytes_ok) fail=1;

    /* ---- TEST 2: MEMFILL ---- */
    uint64_t pat=0x1122334455667788ULL;
    for(size_t i=0;i<N;i++) dst[i]=0;
    memset(&desc,0,sizeof desc); memset(&comp,0,sizeof comp);
    desc.flags=IDXD_OP_FLAG_CRAV|IDXD_OP_FLAG_RCR|IDXD_OP_FLAG_BOF;
    desc.opcode=DSA_OPCODE_MEMFILL; desc.completion_addr=(uint64_t)&comp;
    desc.pattern=pat; desc.dst_addr=(uint64_t)dst; desc.xfer_size=N;
    st=submit(portal,&desc,&comp);
    int fill_ok=1; for(size_t i=0;i<N;i++) if(dst[i]!=((uint8_t*)&pat)[i&7]){fill_ok=0;break;}
    printf("TEST memfill : status=0x%02x | pattern check (8-byte LE repeat) %s\n", st, fill_ok?"OK":"BAD");
    if(DSA_COMP_STATUS(st)!=DSA_COMP_SUCCESS || !fill_ok) fail=1;

    /* ---- TEST 3: COMPARE (equal then unequal) ---- */
    for(size_t i=0;i<N;i++){src[i]=(uint8_t)i; dst[i]=(uint8_t)i;}
    memset(&desc,0,sizeof desc); memset(&comp,0,sizeof comp);
    desc.flags=IDXD_OP_FLAG_CRAV|IDXD_OP_FLAG_RCR|IDXD_OP_FLAG_BOF;
    desc.opcode=DSA_OPCODE_COMPARE; desc.completion_addr=(uint64_t)&comp;
    desc.src_addr=(uint64_t)src; desc.src2_addr=(uint64_t)dst; desc.xfer_size=N;
    st=submit(portal,&desc,&comp);
    /* compare result: comp.result==0 means equal, !=0 means differ (offset in bytes_completed) */
    printf("TEST compare(equal)   : status=0x%02x result=%u (expect result=0 equal)\n", st, comp.result);
    int eq_ok = (DSA_COMP_STATUS(st)==DSA_COMP_SUCCESS && comp.result==0);
    dst[N-100]^=0xFF;  /* introduce a diff */
    memset(&comp,0,sizeof comp); desc.completion_addr=(uint64_t)&comp;
    st=submit(portal,&desc,&comp);
    printf("TEST compare(unequal) : status=0x%02x result=%u diff_offset=%u (expect result!=0, offset~%zu)\n",
           st, comp.result, comp.bytes_completed, N-100);
    int neq_ok = (DSA_COMP_STATUS(st)==DSA_COMP_SUCCESS && comp.result!=0);
    if(!eq_ok||!neq_ok) fail=1;

    printf("\n==== DSA VERIFICATION: %s ====\n", fail?"FAIL":"ALL PASS");
    munmap(portal,4096); close(fd);
    return fail;
}
