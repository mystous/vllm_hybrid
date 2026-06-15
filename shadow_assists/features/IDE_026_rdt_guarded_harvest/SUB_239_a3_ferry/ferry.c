/* SUB_239 FERRY — "컴퓨트는 로컬, 운반은 DSA".
 *
 * 원격 NUMA(node1) 데이터셋을 worker(node0 코어)가:
 *   REMOTE: 직접 cross-UPI 로 읽어 sum-reduce  (SUB_235: remote = +43% 지연)
 *   FERRY : DSA(shared WQ, ENQCMD)로 node1→node0 스테이징 후 node0-로컬 sum-reduce
 *
 * ⚠ NUMA 배치: 이 컨테이너는 mbind/set_mempolicy 가 EPERM(seccomp) →
 *    명시적 바인딩 불가. 대신 **first-touch 정책**: 대상 노드의 코어에 핀한
 *    헬퍼 스레드가 버퍼를 memset 으로 fault-in → 페이지가 그 노드에 앉는다.
 *    배치 검증은 별도 access-latency 프로빙(원격이 ~2.1x 느림)으로 사후 확인.
 *
 * 측정: (1) CPU-busy sum 시간 (= CPU 점유; FERRY 의 copy 는 DSA 오프로드)
 *       (2) end-to-end 시간 (FERRY = ferry_copy + local_sum)
 *       (3) 유효 처리량 GB/s
 * 인자: <mode remote|ferry> <cpu> <ws_mb> <iters> [wq_path] [touch_cpu_node1]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <errno.h>
#include <sys/mman.h>
#include <x86intrin.h>

static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void pin(int c){cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);sched_setaffinity(0,sizeof(s),&s);}

/* ---- first-touch 헬퍼: 지정 코어에서 버퍼를 fault-in ---- */
struct touch_arg { int cpu; char *buf; size_t sz; };
static void *touch_fn(void *p){
    struct touch_arg *a=p; pin(a->cpu);
    /* page 단위 first-touch (memset 은 컴파일러가 통째로 처리 가능하나 OK) */
    for(size_t i=0;i<a->sz;i+=4096) a->buf[i]=1;
    memset(a->buf,1,a->sz);
    return NULL;
}
/* mmap + 지정 코어 스레드로 first-touch */
static char *alloc_touched(size_t sz, int touch_cpu){
    char *p=mmap(NULL,sz,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(p==MAP_FAILED){perror("mmap");exit(2);}
    struct touch_arg a={touch_cpu,p,sz};
    pthread_t t; pthread_create(&t,NULL,touch_fn,&a); pthread_join(t,NULL);
    return p;
}

/* ---- 배치 검증: node0 코어에서 1회 sum 의 평균 ns/elem 측정 ---- */
static double probe_latency(const volatile uint64_t *a, size_t n){
    double t0=now(); volatile uint64_t s=0;
    for(size_t i=0;i<n;i++) s+=a[i];
    double dt=now()-t0; if(s==0xdeadbeef) printf(".");
    return dt/n*1e9; /* ns per 8B elem (streaming, prefetch 포함) */
}

/* ---- DSA ---- */
struct dsa_hw_desc{uint32_t pasid;uint32_t flags_opcode;uint64_t completion_addr;
 uint64_t src_addr;uint64_t dst_addr;uint32_t xfer_size;uint16_t int_handle;uint16_t rsvd;uint8_t rest[24];}__attribute__((aligned(64)));
#define OP_MEMMOVE 0x03
#define CRAV 0x0004
#define RCR  0x0008
#define CC   0x0100
#define DSA_MAX_XFER (2u<<20)
static inline int enqcmd(volatile void*p,const void*d){uint8_t r;
 asm volatile(".byte 0xf2,0x0f,0x38,0xf8,0x02\n setz %0":"=r"(r):"a"(p),"d"(d):"memory");return r;}

static size_t dsa_ferry(volatile void *portal, char *dst, const char *src, size_t sz){
    static volatile uint8_t comp[32] __attribute__((aligned(32)));
    struct dsa_hw_desc d; memset(&d,0,sizeof d);
    d.flags_opcode=(OP_MEMMOVE<<24)|CRAV|RCR|CC; d.completion_addr=(uint64_t)comp;
    size_t done=0;
    while(done<sz){
        size_t chunk=sz-done<DSA_MAX_XFER?sz-done:DSA_MAX_XFER;
        comp[0]=0; d.src_addr=(uint64_t)(src+done); d.dst_addr=(uint64_t)(dst+done); d.xfer_size=(uint32_t)chunk;
        int tries=0; while(enqcmd(portal,&d)!=0){ if(++tries>1000000){fprintf(stderr,"enqcmd stuck\n");return done;} _mm_pause(); }
        int w=0; while(comp[0]==0 && w++<200000000) _mm_pause();
        if(comp[0]!=0x01) fprintf(stderr,"comp status=0x%02x\n",comp[0]);
        done+=chunk;
    }
    return done;
}
static uint64_t sum_reduce(const volatile uint64_t *a, size_t n){uint64_t s=0;for(size_t i=0;i<n;i++)s+=a[i];return s;}

/* ---- offset 기반 pointer-chase: 인덱스 사이클 (절대주소 아님 → DSA 복사 후에도 유지) ----
 * cacheline(8 elem) 간격 Sattolo 사이클. 의존 load 라 prefetch 무력 → 진짜 NUMA 지연 노출. */
static void build_chase(uint64_t *a, size_t n){
    size_t lines=n/8; if(lines<2) lines=2;
    size_t *perm=malloc(lines*sizeof(size_t));
    for(size_t i=0;i<lines;i++) perm[i]=i;
    srandom(12345);
    for(size_t i=lines-1;i>0;i--){ size_t j=(size_t)(random()%i); size_t t=perm[i];perm[i]=perm[j];perm[j]=t; }
    for(size_t i=0;i<lines;i++) a[perm[i]*8] = (uint64_t)(perm[(i+1)%lines]*8);
    free(perm);
}
/* steps 번 의존 chase. 반환: 최종 인덱스(최적화 방지) */
static uint64_t chase_run(const volatile uint64_t *a, size_t steps){
    uint64_t idx=0; for(size_t i=0;i<steps;i++) idx=a[idx]; return idx;
}

int main(int argc,char**argv){
    if(argc<5){fprintf(stderr,"usage: %s remote|ferry cpu ws_mb iters [wq] [touch_cpu_node1]\n",argv[0]);return 1;}
    const char*mode=argv[1]; int cpu=atoi(argv[2]); size_t ws=(size_t)atol(argv[3])<<20; long iters=atol(argv[4]);
    const char*wq=argc>5?argv[5]:"/dev/dsa/wq1.0";
    int touch1=argc>6?atoi(argv[6]):56;  /* node1 코어 (56-111) */
    pin(cpu);
    int ferry=!strcmp(mode,"ferry");

    int chase = getenv("FERRY_CHASE") != NULL;   /* latency-bound 워크로드 */
    size_t steps = ws/64;                          /* chase step 수 ≈ cacheline 수 */

    /* src 는 항상 원격(node1): node1 코어에서 first-touch */
    char *src=alloc_touched(ws, touch1);
    size_t n=ws/sizeof(uint64_t);
    if(chase) build_chase((uint64_t*)src, n);
    /* 배치 검증 프로브: node0(cpu)에서 src 접근 지연 */
    double src_ns=probe_latency((uint64_t*)src,n);

    char *stage=NULL; volatile void *portal=NULL; double stage_ns=0;
    if(ferry){
        stage=alloc_touched(ws, cpu);     /* 로컬(node0) staging: worker 코어에서 first-touch */
        stage_ns=probe_latency((uint64_t*)stage,n);
        int fd=open(wq,O_RDWR); if(fd<0){perror("open wq");return 1;}
        portal=mmap(NULL,4096,PROT_WRITE,MAP_SHARED|MAP_POPULATE,fd,0);
        if(portal==MAP_FAILED){perror("mmap portal");return 1;}
    }

    volatile uint64_t sink=0; double t_ferry=0,t_sum=0,t0=now();
    for(long it=0;it<iters;it++){
        if(ferry){
            double a=now(); dsa_ferry(portal,stage,src,ws); double b=now(); t_ferry+=b-a;
            sink += chase ? chase_run((uint64_t*)stage,steps) : sum_reduce((uint64_t*)stage,n);
            t_sum+=now()-b;
        }else{
            double a=now();
            sink += chase ? chase_run((uint64_t*)src,steps) : sum_reduce((uint64_t*)src,n);
            t_sum+=now()-a;
        }
    }
    double tot=now()-t0, bytes=(double)ws*iters;
    /* chase: 유효 단위 = step(의존 load), ns/step 이 핵심. sum: GB/s. */
    double ns_per_step = chase ? t_sum/( (double)steps*iters )*1e9 : 0;
    printf("FERRY_RESULT,mode=%s,workload=%s,ws_mb=%zu,iters=%ld,src_probe_ns=%.3f,stage_probe_ns=%.3f,"
           "cpu_busy_sum_s=%.4f,ferry_s=%.4f,e2e_s=%.4f,sum_GBps=%.2f,e2e_GBps=%.2f,ns_per_step=%.2f,sink=%llu\n",
           mode, chase?"chase":"sum", ws>>20, iters, src_ns, stage_ns, t_sum, t_ferry, tot,
           bytes/t_sum/1e9, bytes/tot/1e9, ns_per_step, (unsigned long long)sink);
    return 0;
}
