// SUB_198 — AMX draft head firer (proxy concurrent fire)
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <atomic>
#include <signal.h>
#include <sched.h>
#include <unistd.h>
#include <omp.h>
extern "C" {
    int    amx_draft_qwen05b_init();
    void   amx_draft_qwen05b_free();
    double amx_draft_qwen05b_step_ms(int B, int K);
    int    amx_draft_qwen05b_hw_amx();
}
static std::atomic<bool> g_stop{false};
void on_signal(int){ g_stop.store(true); }
int main(){
    signal(SIGTERM,on_signal); signal(SIGINT,on_signal);
    cpu_set_t mask; CPU_ZERO(&mask);
    for(int c=80;c<=95;++c) CPU_SET(c,&mask);
    sched_setaffinity(0,sizeof(mask),&mask);
    omp_set_num_threads(64);
    int hw = amx_draft_qwen05b_hw_amx();
    int rc = amx_draft_qwen05b_init();
    if (rc!=0){ std::fprintf(stderr,"[amx_firer] init failed rc=%d\n",rc); return 1; }
    constexpr int B=1, K=7, kCycleMs=1000;
    std::fprintf(stderr,"[amx_firer] start workers=64 cores=80-95 B=%d K=%d cycle=%dms amx_hw=%d pid=%d\n",
                 B,K,kCycleMs,hw,getpid());
    uint64_t cycles=0; double total_ms=0;
    while(!g_stop.load()){
        auto t0 = std::chrono::steady_clock::now();
        total_ms += amx_draft_qwen05b_step_ms(B,K);
        ++cycles;
        if (cycles%100==0)
            std::fprintf(stderr,"[amx_firer] cycles=%lu avg=%.3f ms/cycle\n",(unsigned long)cycles,total_ms/cycles);
        auto t1=std::chrono::steady_clock::now();
        auto el=std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
        auto sl=kCycleMs-el;
        if (sl>0) std::this_thread::sleep_for(std::chrono::milliseconds(sl));
    }
    std::fprintf(stderr,"[amx_firer] stop cycles=%lu avg=%.3f ms/cycle\n",
                 (unsigned long)cycles, cycles?total_ms/cycles:0.0);
    amx_draft_qwen05b_free();
    return 0;
}
