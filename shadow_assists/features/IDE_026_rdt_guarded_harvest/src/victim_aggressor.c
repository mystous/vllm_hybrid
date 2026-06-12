/*
 * IDE_026 / TSK_048 T1 — CPU-only 간섭 재현 마이크로벤치 (단일 파일)
 *
 *   victim   : serving host path 모사 — pointer-chase (의존 load, LLC 상주
 *              working set) + memcpy burst (detok-like 버퍼 복사).
 *              반복(iteration)당 latency 샘플 → p50/p95/p99/p99.9 보고.
 *   aggressor: harvest 모사 — AVX-512 STREAM-triad (a = b + s*c),
 *              스레드당 LLC 초과 배열, 합산 GB/s 보고.
 *
 * 빌드:  build.sh 참조 (gcc -O3 -march=native -pthread)
 * 실행:  ./victim_aggressor --role victim   --cpus 0-7   --secs 30
 *        ./victim_aggressor --role aggressor --cpus 16-55 --secs 40
 *
 * 결과 줄 (파싱용):
 *   VICTIM_RESULT,iters,mean_ms,p50_ms,p95_ms,p99_ms,p999_ms,ns_per_load
 *   AGGR_RESULT,threads,secs,total_GBps
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

/* ---------------- 공통 ---------------- */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* "0-7,16,20-23" → cpu 배열 */
static int g_aggr_mode = 0; /* 0=basic 1=cldemote 2=nt */

static int parse_cpulist(const char *s, int *cpus, int max) {
    int n = 0;
    while (*s && n < max) {
        char *end;
        long a = strtol(s, &end, 10), b = a;
        if (*end == '-') { b = strtol(end + 1, &end, 10); }
        for (long c = a; c <= b && n < max; c++) cpus[n++] = (int)c;
        s = (*end == ',') ? end + 1 : end;
    }
    return n;
}

static void pin_to(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
        fprintf(stderr, "warn: setaffinity(cpu %d) 실패\n", cpu);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* ---------------- victim ---------------- */

#define CACHELINE 64

typedef struct {
    int cpu;
    size_t ws_bytes;     /* pointer-chase working set */
    size_t copy_bytes;   /* memcpy burst 크기 */
    size_t chase_steps;  /* iteration 당 의존 load 수 */
    double secs;
    /* out */
    double *samples;     /* iteration latency (sec) */
    size_t n_samples, cap_samples;
} victim_arg_t;

/* Sattolo: 전 노드 단일 사이클 (cacheline 단위) */
static void build_chain(void **buf, size_t n_lines, unsigned seed) {
    size_t *perm = malloc(n_lines * sizeof(size_t));
    for (size_t i = 0; i < n_lines; i++) perm[i] = i;
    srandom(seed);
    for (size_t i = n_lines - 1; i > 0; i--) {
        size_t j = (size_t)(random() % i); /* Sattolo: j < i */
        size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    size_t stride = CACHELINE / sizeof(void *);
    for (size_t i = 0; i < n_lines; i++) {
        size_t next = perm[(i + 1) % n_lines];
        buf[perm[i] * stride] = &buf[next * stride];
    }
    free(perm);
}

static void *victim_thread(void *p) {
    victim_arg_t *a = (victim_arg_t *)p;
    pin_to(a->cpu);

    size_t n_lines = a->ws_bytes / CACHELINE;
    void **chase = aligned_alloc(CACHELINE, n_lines * CACHELINE);
    char *src = aligned_alloc(CACHELINE, a->copy_bytes);
    char *dst = aligned_alloc(CACHELINE, a->copy_bytes);
    if (!chase || !src || !dst) { fprintf(stderr, "alloc 실패\n"); exit(2); }
    build_chain(chase, n_lines, (unsigned)(a->cpu * 7919 + 1));
    memset(src, 0x5a, a->copy_bytes);
    memset(dst, 0, a->copy_bytes);

    /* warmup: 체인 1회 순회 */
    volatile void **pp = (volatile void **)chase;
    for (size_t i = 0; i < n_lines; i++) pp = (volatile void **)*pp;

    double t_end = now_sec() + a->secs;
    while (now_sec() < t_end) {
        double t0 = now_sec();
        /* (1) 의존 load 체인 */
        for (size_t i = 0; i < a->chase_steps; i++)
            pp = (volatile void **)*pp;
        /* (2) memcpy burst (host path 의 detok/serialize 류 복사 모사) */
        memcpy(dst, src, a->copy_bytes);
        double dt = now_sec() - t0;
        if (a->n_samples == a->cap_samples) {
            a->cap_samples = a->cap_samples ? a->cap_samples * 2 : 4096;
            a->samples = realloc(a->samples, a->cap_samples * sizeof(double));
        }
        a->samples[a->n_samples++] = dt;
    }
    /* pp 가 최적화로 제거되지 않도록 */
    if ((uintptr_t)pp == 1) fprintf(stderr, ".");
    free(chase); free(src); free(dst);
    return NULL;
}

static int run_victim(int *cpus, int n_cpus, size_t ws_mb, size_t copy_kb,
                      size_t chase_steps, double secs) {
    victim_arg_t *args = calloc(n_cpus, sizeof(victim_arg_t));
    pthread_t *th = calloc(n_cpus, sizeof(pthread_t));
    for (int i = 0; i < n_cpus; i++) {
        args[i].cpu = cpus[i];
        args[i].ws_bytes = ws_mb << 20;
        args[i].copy_bytes = copy_kb << 10;
        args[i].chase_steps = chase_steps;
        args[i].secs = secs;
        pthread_create(&th[i], NULL, victim_thread, &args[i]);
    }
    size_t total = 0;
    for (int i = 0; i < n_cpus; i++) { pthread_join(th[i], NULL); total += args[i].n_samples; }

    double *all = malloc(total * sizeof(double));
    size_t k = 0;
    double sum = 0;
    for (int i = 0; i < n_cpus; i++)
        for (size_t j = 0; j < args[i].n_samples; j++) { all[k++] = args[i].samples[j]; sum += args[i].samples[j]; }
    qsort(all, total, sizeof(double), cmp_double);
    #define PCT(p) (all[(size_t)((p) * (total - 1))] * 1e3)
    double mean_ms = sum / total * 1e3;
    double ns_per_load = PCT(0.50) * 1e6 / (double)chase_steps; /* p50 기준, memcpy 포함이라 상한 */
    printf("victim: threads=%d ws=%zuMB copy=%zuKB chase=%zu iters=%zu\n",
           n_cpus, ws_mb, copy_kb, chase_steps, total);
    printf("  mean=%.3fms p50=%.3fms p95=%.3fms p99=%.3fms p99.9=%.3fms\n",
           mean_ms, PCT(0.50), PCT(0.95), PCT(0.99), PCT(0.999));
    printf("VICTIM_RESULT,%zu,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f\n",
           total, mean_ms, PCT(0.50), PCT(0.95), PCT(0.99), PCT(0.999), ns_per_load);
    #undef PCT
    free(all);
    for (int i = 0; i < n_cpus; i++) free(args[i].samples);
    free(args); free(th);
    return 0;
}

/* ---------------- aggressor ---------------- */

typedef struct {
    int cpu;
    size_t elems;        /* 배열당 double 수 */
    double secs;
    /* out */
    double bytes_moved;
} aggr_arg_t;

static void *aggr_thread(void *p) {
    aggr_arg_t *a = (aggr_arg_t *)p;
    pin_to(a->cpu);
    size_t n = a->elems;
    double *A = aligned_alloc(64, n * sizeof(double));
    double *B = aligned_alloc(64, n * sizeof(double));
    double *C = aligned_alloc(64, n * sizeof(double));
    if (!A || !B || !C) { fprintf(stderr, "alloc 실패\n"); exit(2); }
    for (size_t i = 0; i < n; i++) { B[i] = 1.0; C[i] = 2.0; A[i] = 0.0; }

    const double s = 3.0;
    double t_end = now_sec() + a->secs;
    size_t iters = 0;
    while (now_sec() < t_end) {
#ifdef __AVX512F__
        __m512d vs = _mm512_set1_pd(s);
        if (g_aggr_mode == 2) {                 /* nt: RFO 없는 streaming store */
            for (size_t i = 0; i + 8 <= n; i += 8) {
                __m512d vb = _mm512_load_pd(B + i);
                __m512d vc = _mm512_load_pd(C + i);
                _mm512_stream_pd(A + i, _mm512_fmadd_pd(vs, vc, vb));
            }
            _mm_sfence();
        } else if (g_aggr_mode == 1) {          /* cldemote: 소비 완료 라인 자발 강등 */
            for (size_t i = 0; i + 8 <= n; i += 8) {
                __m512d vb = _mm512_load_pd(B + i);
                __m512d vc = _mm512_load_pd(C + i);
                _mm512_store_pd(A + i, _mm512_fmadd_pd(vs, vc, vb));
                _cldemote((void *)(A + i));
                _cldemote((void *)(B + i));
                _cldemote((void *)(C + i));
            }
        } else {                                /* basic: 일반 store (LLC 압박) */
            for (size_t i = 0; i + 8 <= n; i += 8) {
                __m512d vb = _mm512_load_pd(B + i);
                __m512d vc = _mm512_load_pd(C + i);
                _mm512_store_pd(A + i, _mm512_fmadd_pd(vs, vc, vb));
            }
        }
#else
        for (size_t i = 0; i < n; i++) A[i] = B[i] + s * C[i];
#endif
        iters++;
    }
    /* STREAM triad 관례: 3 * 8 bytes/elem */
    a->bytes_moved = (double)iters * (double)n * 24.0;
    if (A[1] < 0) fprintf(stderr, "."); /* 최적화 제거 방지 */
    free(A); free(B); free(C);
    return NULL;
}

static int run_aggressor(int *cpus, int n_cpus, size_t array_mb, double secs) {
    aggr_arg_t *args = calloc(n_cpus, sizeof(aggr_arg_t));
    pthread_t *th = calloc(n_cpus, sizeof(pthread_t));
    size_t elems = (array_mb << 20) / sizeof(double);
    double t0 = now_sec();
    for (int i = 0; i < n_cpus; i++) {
        args[i].cpu = cpus[i];
        args[i].elems = elems;
        args[i].secs = secs;
        pthread_create(&th[i], NULL, aggr_thread, &args[i]);
    }
    double total_bytes = 0;
    for (int i = 0; i < n_cpus; i++) { pthread_join(th[i], NULL); total_bytes += args[i].bytes_moved; }
    double elapsed = now_sec() - t0;
    double gbps = total_bytes / elapsed / 1e9;
#ifdef __AVX512F__
    const char *isa = "avx512";
#else
    const char *isa = "scalar";
#endif
    printf("aggressor: threads=%d array=%zuMBx3/thread isa=%s elapsed=%.1fs\n",
           n_cpus, array_mb, isa, elapsed);
    printf("AGGR_RESULT,mode=%d,%d,%.1f,%.2f\n", g_aggr_mode, n_cpus, elapsed, gbps);
    free(args); free(th);
    return 0;
}

/* ---------------- main ---------------- */

static void usage(const char *argv0) {
    fprintf(stderr,
        "사용법: %s --role victim|aggressor --cpus <list> [--secs N]\n"
        "  victim 전용 : --ws-mb N (기본 64)  --copy-kb N (기본 4096)  --chase-steps N (기본 200000)\n"
        "  aggressor   : --array-mb N (스레드당 배열, 기본 32)  --aggr-mode basic|cldemote|nt\n", argv0);
}

int main(int argc, char **argv) {
    const char *role = NULL, *cpulist = NULL;
    double secs = 30;
    size_t ws_mb = 64, copy_kb = 4096, chase_steps = 200000, array_mb = 32;

    for (int i = 1; i < argc; i++) {
        #define ARG(name) (!strcmp(argv[i], name) && i + 1 < argc)
        if (ARG("--role")) role = argv[++i];
        else if (ARG("--cpus")) cpulist = argv[++i];
        else if (ARG("--secs")) secs = atof(argv[++i]);
        else if (ARG("--aggr-mode")) { const char *m = argv[++i]; g_aggr_mode = !strcmp(m,"cldemote") ? 1 : !strcmp(m,"nt") ? 2 : 0; }
        else if (ARG("--ws-mb")) ws_mb = (size_t)atol(argv[++i]);
        else if (ARG("--copy-kb")) copy_kb = (size_t)atol(argv[++i]);
        else if (ARG("--chase-steps")) chase_steps = (size_t)atol(argv[++i]);
        else if (ARG("--array-mb")) array_mb = (size_t)atol(argv[++i]);
        else { usage(argv[0]); return 1; }
        #undef ARG
    }
    if (!role || !cpulist) { usage(argv[0]); return 1; }

    int cpus[256];
    int n = parse_cpulist(cpulist, cpus, 256);
    if (n <= 0) { fprintf(stderr, "cpus 파싱 실패: %s\n", cpulist); return 1; }

    if (!strcmp(role, "victim"))
        return run_victim(cpus, n, ws_mb, copy_kb, chase_steps, secs);
    if (!strcmp(role, "aggressor"))
        return run_aggressor(cpus, n, array_mb, secs);
    usage(argv[0]);
    return 1;
}
