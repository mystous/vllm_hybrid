/* SUB_239 보강 — co-located victim 간섭 (호스트, DSA 불요).
 *
 * FERRY 의 미해결 우려: "데이터를 node0 로 끌어오면 node0 iMC 부하↑ → 같은 node0 에
 * 동거하는 victim(serving) 이 오히려 손해" — REMOTE(node1 직접접근, node0 iMC 가벼움)
 * 대비 정량화. DSA 운반 자체는 무관하므로 aggressor 의 *버퍼 NUMA 배치*로 시뮬레이션:
 *   - LOCAL  배치 = FERRY 후 staging 버퍼를 node0 에서 스트리밍-read (node0 iMC 부하)
 *   - REMOTE 배치 = REMOTE 시나리오로 worker 가 node1 버퍼 직접 read (node0 iMC 가벼움)
 * victim 은 node0 의 작은 워킹셋 pointer-chase 지연(p50/p99) — 동거 코어에서 동시 실행.
 *
 * 인자: victim_cpu agg_cpu0 agg_node ws_mb secs [nagg] [vws_mb]
 *   agg_node: 0=local(ferry),1=remote, -1=off ; nagg=aggressor 코어수(기본1, acpu0부터 연속)
 *   vws_mb=victim 워킹셋 MB (기본128 — L3 초과 = 메모리-bound 라야 iMC 압력 노출)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void pin(int c){cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);sched_setaffinity(0,sizeof(s),&s);}
static int cmpd(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return x<y?-1:x>y?1:0;}

/* aggressor: 지정 코어/노드에서 STREAM-read (sum) 로 iMC BW 점유 (코어당 1 스레드) */
static volatile int g_run=1, g_agg_on=0;
#define MAXAGG 32
static uint64_t *g_abuf[MAXAGG]; static size_t g_an; static int g_acpu[MAXAGG];
static volatile uint64_t g_sink;
struct af{int slot;};
static void *agg_fn(void*arg){ struct af*f=arg; int slot=f->slot; pin(g_acpu[slot]);
    uint64_t*buf=g_abuf[slot];
    while(g_run){ if(!g_agg_on){ for(volatile int k=0;k<1000;k++); continue; }
        uint64_t s=0; for(size_t i=0;i<g_an;i+=8) s+=buf[i]; g_sink+=s; }
    return NULL; }

/* first-touch: 지정 코어(=노드)에서 fault-in */
struct ta{int cpu; uint64_t*b; size_t n;};
static void*touch(void*p){struct ta*a=p;pin(a->cpu);for(size_t i=0;i<a->n;i++)a->b[i]=i;return NULL;}

int main(int argc,char**argv){
    if(argc<6){fprintf(stderr,"usage: %s vcpu acpu0 anode ws_mb secs [nagg] [vws_mb]\n",argv[0]);return 1;}
    int vcpu=atoi(argv[1]), acpu0=atoi(argv[2]), anode=atoi(argv[3]);
    size_t ws=(size_t)atol(argv[4])<<20; double secs=atof(argv[5]);
    int nagg=argc>6?atoi(argv[6]):1; if(nagg<1)nagg=1; if(nagg>MAXAGG)nagg=MAXAGG;
    size_t VWS=(size_t)(argc>7?atol(argv[7]):128)<<20;
    g_an=ws/sizeof(uint64_t);
    /* aggressor 버퍼 N개: anode==1 → node1 코어들(56+)로, 아니면 각 acpu(node0)로 first-touch */
    for(int s=0;s<nagg;s++){
        g_acpu[s]=acpu0+s; g_abuf[s]=malloc(ws);
        int touch_cpu=(anode==1)? 56+s : g_acpu[s];
        struct ta a={touch_cpu,g_abuf[s],g_an}; pthread_t t; pthread_create(&t,NULL,touch,&a); pthread_join(t,NULL);
    }
    g_agg_on = (anode>=0);
    pthread_t ath[MAXAGG]; struct af af[MAXAGG];
    for(int s=0;s<nagg;s++){ af[s].slot=s; pthread_create(&ath[s],NULL,agg_fn,&af[s]); }

    /* victim: node0 의 워킹셋(기본128MB = L3 초과 → 메모리-bound) chain pointer-chase 지연 */
    pin(vcpu);
    size_t VN=VWS/sizeof(void*); void**vb=malloc(VWS);
    size_t*idx=malloc(VN*sizeof(size_t)); for(size_t i=0;i<VN;i++)idx[i]=i;
    unsigned seed=999; for(size_t i=VN-1;i>0;i--){size_t j=(seed=seed*1103515245+12345)%(i+1);size_t t=idx[i];idx[i]=idx[j];idx[j]=t;}
    for(size_t i=0;i<VN;i++) vb[idx[i]]=(void*)&vb[idx[(i+1)%VN]];
    free(idx);

    size_t NS=300000; double*samp=malloc(NS*sizeof(double)); size_t ns=0;
    double tend=now()+secs; void**p=(void**)vb[0];
    while(now()<tend){
        struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
        for(int k=0;k<256;k++){ p=(void**)*p; asm volatile("":"+r"(p)::); }
        clock_gettime(CLOCK_MONOTONIC,&b);
        g_sink+=(uint64_t)(uintptr_t)p;
        double dt=((b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec))/256.0;
        if(ns<NS)samp[ns++]=dt;
    }
    g_run=0; for(int s=0;s<nagg;s++) pthread_join(ath[s],NULL);
    qsort(samp,ns,sizeof(double),cmpd);
    double mean=0; for(size_t i=0;i<ns;i++)mean+=samp[i]; mean/=ns;
    const char*lbl = anode<0?"agg_off":(anode==0?"LOCAL(ferry)":"REMOTE");
    printf("COLOC,agg=%s,anode=%d,nagg=%d,aggws_mb=%zu,vws_mb=%zu,n=%zu,p50=%.2f,p99=%.2f,mean=%.2f\n",
        lbl,anode,nagg,ws>>20,VWS>>20,ns,samp[ns/2],samp[(size_t)(ns*0.99)],mean);
    return 0;
}
