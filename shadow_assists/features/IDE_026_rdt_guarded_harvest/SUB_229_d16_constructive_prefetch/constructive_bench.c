/* SUB_229 [D16] constructive interference 마이크로벤치.
 * 공유 포인터체이스 버퍼에서 victim 이 latency 측정; helper(별 코어)가 같은 체인을
 * 앞서 prefetch. helper ON/OFF 로 victim p99 비교. 인자: victim_cpu helper_cpu helper_on mb secs */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void pin(int c){cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}
static void **buf; static size_t NL; static volatile int run=1, helper_on=0;
static volatile void *g_sink;
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return x<y?-1:x>y?1:0;}
static void build(unsigned seed){ /* 무작위 순환 체인 */
    size_t *idx=malloc(NL*sizeof(size_t)); for(size_t i=0;i<NL;i++)idx[i]=i;
    for(size_t i=NL-1;i>0;i--){size_t j=(seed=seed*1103515245+12345)%(i+1);size_t t=idx[i];idx[i]=idx[j];idx[j]=t;}
    for(size_t i=0;i<NL;i++) buf[idx[i]]=(void*)&buf[idx[(i+1)%NL]];
    free(idx);
}
static void *helper_fn(void*arg){ int c=*(int*)arg; pin(c);
    while(run){ if(!helper_on){ for(volatile int k=0;k<1000;k++); continue; }
        void **p=(void**)buf[0]; for(size_t i=0;i<NL;i++){ __builtin_prefetch(p,0,1); p=(void**)*p; } g_sink=p; }
    return NULL;
}
int main(int argc,char**argv){
    int vcpu=atoi(argv[1]), hcpu=atoi(argv[2]); helper_on=atoi(argv[3]);
    size_t mb=atol(argv[4]); double secs=atof(argv[5]);
    NL=(mb<<20)/sizeof(void*); buf=malloc(NL*sizeof(void*)); build(12345);
    pthread_t h; pthread_create(&h,NULL,helper_fn,&hcpu);
    pin(vcpu);
    size_t NS=200000; double *samp=malloc(NS*sizeof(double)); size_t ns=0;
    double tend=now()+secs; void**p=(void**)buf[0];
    while(now()<tend){
        struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
        for(int k=0;k<256;k++){ p=(void**)*p; asm volatile("":"+r"(p)::); }
        clock_gettime(CLOCK_MONOTONIC,&b);
        g_sink=p;            /* escape: DCE 방지 */
        double dt=((b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec))/256.0;
        if(ns<NS)samp[ns++]=dt;
    }
    run=0; pthread_join(h,NULL);
    qsort(samp,ns,sizeof(double),cmp);
    printf("RESULT,helper=%d,n=%zu,p50=%.1f,p99=%.1f,mean=%.1f\n",
        helper_on,ns,samp[ns/2],samp[(size_t)(ns*0.99)],({double s=0;for(size_t i=0;i<ns;i++)s+=samp[i];s/ns;}));
    return 0;
}
