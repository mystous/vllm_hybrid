/* SUB_236 DSA memcpy 트래픽 생성기 — shared WQ(ENQCMD) 로 memcpy descriptor 연속 제출.
 * RMID 미태깅 DSA 발 메모리 트래픽 생성 → victim 간섭 측정용. sudo 불요(portal world-rw).
 * 인자: wq_path cpu copy_mb secs */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <sys/mman.h>
#include <x86intrin.h>

/* idxd descriptor (64B) — uapi/linux/idxd.h 요약 */
struct dsa_hw_desc {
    uint32_t pasid;            /* [0] */
    uint32_t flags_opcode;     /* [4] flags:24 | opcode:8 */
    uint64_t completion_addr;  /* [8] */
    uint64_t src_addr;         /* [16] */
    uint64_t dst_addr;         /* [24] */
    uint32_t xfer_size;        /* [32] */
    uint16_t int_handle;       /* [36] */
    uint16_t rsvd;             /* [38] */
    uint8_t  rest[24];         /* [40..63] */
} __attribute__((aligned(64)));

#define DSA_OPCODE_MEMMOVE 0x03
#define IDXD_OP_FLAG_CRAV  0x0004  /* completion record address valid */
#define IDXD_OP_FLAG_RCR   0x0008  /* request completion record */
#define IDXD_OP_FLAG_CC    0x0100  /* cache control (write dst to cache) */

static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void pin(int c){cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);sched_setaffinity(0,sizeof(s),&s);}

/* ENQCMD: portal 에 64B descriptor 제출. ZF=0 성공, ZF=1 retry(큐 full) */
static inline void movdir64b(volatile void *portal, const void *desc){
    asm volatile(".byte 0x66,0x0f,0x38,0xf8,0x02" : : "a"(portal), "d"(desc) : "memory");
}
static inline int enqcmd(volatile void *portal, const void *desc){
    uint8_t ret;
    asm volatile(".byte 0xf2,0x0f,0x38,0xf8,0x02\t\n setz %0"
                 : "=r"(ret) : "a"(portal), "d"(desc) : "memory");
    return ret; /* 0=성공 */
}

int main(int argc,char**argv){
    const char*wq=argv[1]; int cpu=atoi(argv[2]); size_t cmb=atol(argv[3]); double secs=atof(argv[4]);
    pin(cpu);
    int fd=open(wq,O_RDWR);
    if(fd<0){perror("open wq");return 1;}
    volatile void*portal=mmap(NULL,4096,PROT_WRITE,MAP_SHARED|MAP_POPULATE,fd,0);
    if(portal==MAP_FAILED){perror("mmap portal");return 1;}
    size_t sz=cmb<<20;
    char*src=aligned_alloc(4096,sz),*dst=aligned_alloc(4096,sz);
    memset(src,0x5a,sz); memset(dst,0,sz);
    /* completion record (32B aligned) */
    volatile uint8_t comp[32] __attribute__((aligned(32)))={0};
    struct dsa_hw_desc d; memset(&d,0,sizeof d);
    d.flags_opcode=(DSA_OPCODE_MEMMOVE<<24)|IDXD_OP_FLAG_CRAV|IDXD_OP_FLAG_RCR|IDXD_OP_FLAG_CC;
    d.completion_addr=(uint64_t)comp; d.src_addr=(uint64_t)src; d.dst_addr=(uint64_t)dst;
    d.xfer_size=(uint32_t)sz;
    double tend=now()+secs; uint64_t ops=0, retry=0;
    while(now()<tend){
        comp[0]=0;
        int r=0;
        if(getenv("DSA_DEDICATED")){ movdir64b(portal,&d); }
        else { int tries=0; while((r=enqcmd(portal,&d))!=0){ retry++; if(++tries>100000){break;} _mm_pause(); } }
        if(r==0){ /* 완료 대기 (comp[0] != 0) */
            int w=0; while(comp[0]==0 && w++<10000000) _mm_pause();
            ops++;
        }
    }
    double el=now()-(tend-secs);
    printf("DSA_RESULT,ops=%lu,retry=%lu,GBps=%.1f\n",ops,retry,ops*(double)sz/el/1e9);
    return 0;
}
