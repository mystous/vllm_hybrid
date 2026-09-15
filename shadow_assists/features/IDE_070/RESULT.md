# IDE_070 결과 — CPU offloading 지시서(cpu_offload_no_01) 전 항목 실행 결과

측정값만 기록한다. 평가·해석·의견은 포함하지 않는다. 노드 violet-h100-016, 브랜치 feat/cpu-offload-diag, 2026-09-15.

## 0. 공통 조건

- 모델: Qwen3-Coder-480B-A35B-Instruct-FP8 (GPU: attention·dense·hot expert), CPU expert: kt INT4 (`/models/kt/qwen3-480b-int4`, AMXINT4). E 시리즈만 GPU 측 체크포인트 QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ (4-bit g128).
- 서빙: SGLang 0.5.18 (+로컬 수정 10 파일) + kt-kernel 0.7.0.post1 (컨테이너 `sgl-kt`), TP4 (GPU 0,1,2,3; A6 는 0,1,4,5), H100 80GB ×4, Xeon 8480+ ×2 (turbo OFF 2.0 GHz, BIOS 잠금), DDR5-4400 8ch×2DPC/소켓 (2 TB).
- 기준 구성: hot expert 96 (IDE_069 hotmap), deferred 4, cuda graph bs≤64 (prefill graph 비활성), mem-fraction 0.95, max-total-tokens 40,960, `--ep-dispatch-algorithm dynamic`, cpuinfer 96, threadpool 2.
- 벤치: `vllm bench serve` (컨테이너 vllm-h100), sonnet 512/128 prefix 100, seed 42, C=64 고정 동시성, 256 요청, 셀당 워밍업 1회 (C16 32요청) 후 3회 반복. 반복 간 `/flush_cache` 없음 (같은 seed 프롬프트 재사용). d4 의 C224 셀만 flush 후 896 요청.
- CPU busy = /proc/stat 2 s 샘플 전체 코어 평균. HBM = 부팅 직후 `nvidia-smi` memory.used.
- 하드웨어·소프트웨어 상세 프로브: `shadow_assists/features/IDE_069/hwsw/` (lscpu, dmidecode, nvidia_topo, 컨테이너 패키지 목록).

## 1. 측정 (TSK_054)

### 1.1 1차 측정 (`eval/results/20260915_194755_ide070_measure_baseline/MEASURE_SUMMARY.md` 전문)


launch: `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model …`

#### 벤치 (C=64, N=256, sonnet 512/128)

| rep | tok/s | TPOT p50/p95 ms | TTFT p50/p95 ms | dur s | in/out tok |
|---|---|---|---|---|---|
| rep1_pcm | 633.18 | 77.4/86.6 | 3075/4355 | 51.8 | 128730/32767 |
| rep2_pcmnuma | 634.48 | 78.1/86.5 | 3002/3741 | 51.6 | 128723/32739 |
| rep3_perf | 638.97 | 77.8/86.1 | 2996/3653 | 51.3 | 128727/32766 |

#### pcm (rep1, 2 s 간격, 부하 구간 27/34 샘플 ≈ 54 s)

| 범위 | IPC(코어) | EXEC(스레드) | FREQ | AFREQ | L3 MPI | L2 MPI | L3 hit | FE/BS/BB/RET % | DRAM R+W GB/s (GB/간격 2 s ÷2) | 로컬/원격 MB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| system | 1.25 | 0.56 | 0.45 | 1.00 | 0.0018 | 0.0025 | 0.25 | 7/1/67/24 | 183.1+30.7=213.8 | — |
| socket0 | 1.21 | 0.56 | 0.46 | 1.00 | 0.0027 | 0.0036 | 0.24 | 7/1/68/23 | 92.4+15.3=107.7 | 195483/16974 |
| socket1 | 1.29 | 0.56 | 0.44 | 1.00 | 0.0010 | 0.0016 | 0.29 | 7/1/67/24 | 90.7+15.5=106.2 | 181365/1656 |

- 부하 구간 합계: active cycles 1.090e+13, instructions 1.355e+13, L3 miss 2.280e+10
- **cycles / output token = 3.327e+08** (출력 32767 tok), cycles / (input+output) token = 6.750e+07
- instructions / output token = 4.137e+08, L3 miss / output token = 6.959e+05
- UPI: 시스템 in 20841.0 / out 75890.0 (pcm 단위), UPItoMC 0.05; 소켓0 수신 17610 (링크 이용률 5.8%), 소켓1 수신 3231 (0.7%)
- 에너지 (부하 구간): package 36134 J, DRAM 7021 J → package 평균 669 W, DRAM 130 W

#### pcm-numa (rep2, 부하 구간 26 샘플; 접근 수 = demand + L2 prefetch, code, RFO)

| 범위 | IPC | local DRAM 접근 | remote DRAM 접근 | **remote %** |
|---|---|---|---|---|
| system | 1.25 | 1.524e+11 | 3.139e+09 | 2.0 |
| socket0 코어 | 1.22 | 7.789e+10 | 2.827e+09 | 3.5 |
| socket1 코어 | 1.29 | 7.451e+10 | 3.118e+08 | 0.4 |

#### turbostat (rep3, 5 s 간격, 부하 11/14 샘플, 전체 요약행)

- Busy% 44.1, Avg_MHz 878, **Bzy_MHz 1990** (turbo OFF 확인), IPC 1.23

#### mpstat 코어별 사용률 (%usr+%sys, rep1_pcm, 부하 27 샘플)

- 전체 43.4% · socket0 44.7% (phys 81.8 / HT 7.6) · socket1 42.0% (phys 81.5 / HT 2.6)
- 분포: {'<10%': 112, '10-50%': 15, '50-90%': 2, '>=90%': 95} · ≥50% 코어 수 97/224 · 최상위: cpu32=96%, cpu12=95%, cpu33=95%, cpu30=95%, cpu7=94%, cpu17=94%, cpu1=94%, cpu37=94%

- vmstat (rep1_pcm, 부하 26 샘플): 인터럽트 265286/s, **컨텍스트 스위치 242368/s**, us 44 sy 2 id 54, runq 105
- pidstat TP0 스레드 합 (rep1_pcm, 상위 절반 구간 평균): 자발 683/s, 비자발 7/s
- vmstat (rep2_pcmnuma, 부하 26 샘플): 인터럽트 264012/s, **컨텍스트 스위치 240853/s**, us 44 sy 2 id 54, runq 104
- pidstat TP0 스레드 합 (rep2_pcmnuma, 상위 절반 구간 평균): 자발 699/s, 비자발 8/s
- vmstat (rep3_perf, 부하 26 샘플): 인터럽트 262082/s, **컨텍스트 스위치 238601/s**, us 44 sy 2 id 54, runq 105
- pidstat TP0 스레드 합 (rep3_perf, 상위 절반 구간 평균): 자발 3700/s, 비자발 9/s

#### perf_stat_sys.txt (주의: perf 창 400 s = 부하 ~60 s + 유휴; 비율 지표만 신뢰)
```
# started on Tue Sep 15 19:52:49 2026


 Performance counter stats for 'system wide':

S0      112      44,802,562.79 msec task-clock                       #  112.001 CPUs utilized             
S0      112  6,887,216,716,185      cycles                           #    0.154 GHz                       
S0      112  8,197,017,132,747      instructions                     #    1.19  insn per cycle            
S0      112    139,392,879,809      cache-references                 #    3.111 M/sec                     
S0      112     88,691,507,617      cache-misses                     #   63.63% of all cache refs         
S0      112  1,135,938,563,056      branches                         #   25.354 M/sec                     
S0      112      5,142,561,449      branch-misses                    #    0.45% of all branches           
S0      112         40,968,179      context-switches                 #  914.416 /sec                      
S0      112            433,155      cpu-migrations                   #    9.668 /sec                      
S0      112          2,934,407      page-faults                      #   65.496 /sec                      
S0      112          2,930,536      minor-faults                     #   65.410 /sec                      
S0      112                 65      major-faults                     #    0.001 /sec                      
S1      112      44,802,438.01 msec task-clock                       #  112.001 CPUs utilized             
S1      112  6,663,897,418,596      cycles                           #    0.149 GHz                       
S1      112  7,860,064,097,764      instructions                     #    1.18  insn per cycle            
S1      112    123,038,522,456      cache-references                 #    2.746 M/sec                     
S1      112     81,158,042,228      cache-misses                     #   65.96% of all cache refs         
S1      112  1,103,616,803,875      branches                         #   24.633 M/sec                     
S1      112      5,627,707,316      branch-misses                    #    0.51% of all branches           
S1      112         50,412,443      context-switches                 #    1.125 K/sec                     
S1      112            838,427      cpu-migrations                   #   18.714 /sec                      
S1      112          4,117,185      page-faults                      #   91.896 /sec                      
S1      112          4,115,642      minor-faults                     #   91.862 /sec                      
S1      112                 78      major-faults                     #    0.002 /sec                      

     400.018603371 seconds time elapsed
```

#### perf_stat_tp0.txt (주의: perf 창 400 s = 부하 ~60 s + 유휴; 비율 지표만 신뢰)
```
# started on Tue Sep 15 19:52:49 2026


 Performance counter stats for process id '1884396':

          6,924.46 msec task-clock                       #    0.017 CPUs utilized             
    13,304,072,756      cycles                           #    1.921 GHz                         (56.74%)
    16,574,737,010      instructions                     #    1.25  insn per cycle              (57.64%)
        89,239,174      cache-references                 #   12.888 M/sec                       (54.75%)
        37,648,789      cache-misses                     #   42.19% of all cache refs           (53.42%)
            17,561      context-switches                 #    2.536 K/sec                     
             1,850      cpu-migrations                   #  267.169 /sec                      
            15,802      page-faults                      #    2.282 K/sec                     
                 0      major-faults                     #    0.000 /sec                      

     400.004093380 seconds time elapsed
```

#### recorder → cold expert (rep1, hotmap96 기준 physical ≥96 = CPU)

| 파일 | 토큰 수 | cold 선택/토큰 (8 선택 중) | cold 비율 | 층별 cold 최소/최대 |
|---|---|---|---|---|
| 789469485.6312454.pt | 565424 | **7.302** | 1.47% | 0.43% / 8.44% |
| 789469485.6312456.pt | 565424 | **7.302** | 1.47% | 0.43% / 8.44% |
| 789469485.6312597.pt | 565424 | **7.302** | 1.47% | 0.43% / 8.44% |

- HBM after boot: 0, 77402 MiB; 1, 77464 MiB; 2, 77464 MiB; 3, 76984 MiB; 4, 0 MiB; 5, 0 MiB; 6, 0 MiB; 7, 0 MiB
- TP0 affinity: Cpus_allowed_list:	0-223 · Mems_allowed_list:	0-1

- 주: pcm 표의 DRAM 열은 pcm CSV 의 READ/WRITE (2 s 간격당 GB) 를 2 로 나눈 GB/s. rep3 의 perf stat 은 창 400 s (부하 52 s + 유휴) 이며 `-p` 대상은 HTTP 서버 프로세스 (스케줄러 아님).

### 1.2 2차 측정 (`eval/results/20260915_200733_ide070_measure2/`)

- rep1_perfstat: 633.06 tok/s, TPOT p50/p95 76.2/87.5 ms, TTFT p50/p95 3066/4319 ms
- rep2_profile: 517.21 tok/s, TPOT p50/p95 96.9/107.5 ms, TTFT p50/p95 3586/4662 ms (py-spy 200 Hz + perf record 동시 실행 중)
- rep3_pcmmem: 632.81 tok/s, TPOT p50/p95 77.5/87.5 ms, TTFT p50/p95 3013/3645 ms
- TP0 (`sglang::scheduler_TP0`, GPU0 compute 프로세스): Threads: 277 · Cpus_allowed_list: 0-55,112-167 · Mems_allowed_list: 0-1
- kt 워커 스레드 고정: numa_0_t ×47 → cpu 1–47, numa_1_t ×47 → cpu 57–103; 마스터 numa_0_m_0 → cpu 0, numa_1_m_0 → cpu 56.
- numastat TP0 Private: node0 338562.27 MB, node1 113343.17 MB, 합계 451905.44 MB
- numastat TP0 Total: node0 340193.65 MB, node1 113343.17 MB, 합계 453536.82 MB
- perf record (TP0, `-F 499`, 20 s, 298,502 샘플; 파싱된 행 합 85.4 %) 분류:

| 분류 | 샘플 % |
|---|---|
| kt_kernel_ext 0x75xxx–0x78xxx (역어셈블: vpdpbusd/vpsignb/vmovdqa32 = AVX-512 VNNI) | 36.05 |
| [vdso]/clock_gettime/chrono::now | 29.19 |
| kernel perf_adjust_freq_unthr_context (perf 자체) | 7.02 |
| kt_kernel_ext 0x443xxx–0x446xxx (lock xadd 루프, unique_lock::unlock 인접) | 6.07 |
| kt_kernel_ext 기타 | 3.04 |
| kernel 기타 | 1.91 |
| kt_kernel_ext std 동기화 심볼 | 1.86 |
| libc/libstdc++ 기타 | 0.22 |
| python | 0.03 |
| 기타 dso | 0.00 |

| 스레드(comm) 군 | 샘플 % |
|---|---|
| numa_*_t (kt 워커) | 79.34 |
| python3 | 2.62 |
| numa_*_m (kt 마스터) | 1.57 |
| sglang::schedul | 1.13 |
| pt_gloo_runloop | 0.19 |
| gloo_tcp_loop | 0.18 |
| pt_nccl_heartbt | 0.11 |
| pt_nccl_watchdg | 0.11 |

- pcm-memory (rep3, 2 s 간격, 부하 26/34 샘플): 시스템 읽기 192.6 GB/s, 쓰기 29.5 GB/s, 읽기 최대 샘플 244.6 GB/s
  - SKT0: 읽기 97.3 + 쓰기 14.9 GB/s; 채널별 읽기 MB/s [12159, 0, 12156, 12167, 0, 12165, 12157, 0, 12153, 12163, 0, 12164]
  - SKT1: 읽기 95.4 + 쓰기 14.6 GB/s; 채널별 읽기 MB/s [11908, 0, 11914, 11929, 0, 11926, 11911, 0, 11914, 11929, 0, 11928]

### 1.3 3차 측정 perf stat (`eval/results/20260915_203526_ide070_measure3/`)

- rep1_sys: 벤치 결과 없음 (bench.log: Error response from daemon: No such container: vllm-h100)
```
 Performance counter stats for 'system wide':
S0      112           8,390.01 msec task-clock                       #  140.806 CPUs utilized             
S0      112      1,236,701,654      cycles                           #    0.147 GHz                         (75.43%)
S0      112      1,056,189,812      instructions                     #    0.85  insn per cycle              (76.06%)
S0      112          9,359,883      cache-references                 #    1.116 M/sec                       (76.66%)
S0      112          2,777,331      cache-misses                     #   29.67% of all cache refs           (76.49%)
S0      112        214,788,982      branches                         #   25.601 M/sec                       (75.50%)
S0      112          3,850,951      branch-misses                    #    1.79% of all branches             (74.46%)
S0      112             17,526      context-switches                 #    2.089 K/sec                     
S0      112                310      cpu-migrations                   #   36.949 /sec                      
S0      112                502      page-faults                      #   59.833 /sec                      
S0      112                501      minor-faults                     #   59.714 /sec                      
S0      112                  0      major-faults                     #    0.000 /sec                      
S1      112           7,714.21 msec task-clock                       #  129.465 CPUs utilized             
S1      112        892,898,143      cycles                           #    0.116 GHz                         (75.83%)
S1      112        767,448,944      instructions                     #    0.86  insn per cycle              (75.96%)
S1      112          8,217,919      cache-references                 #    1.065 M/sec                       (76.07%)
S1      112          2,469,698      cache-misses                     #   30.05% of all cache refs           (75.59%)
S1      112        151,222,683      branches                         #   19.603 M/sec                       (74.62%)
S1      112          2,123,253      branch-misses                    #    1.40% of all branches             (73.87%)
S1      112             11,118      context-switches                 #    1.441 K/sec                     
S1      112                365      cpu-migrations                   #   47.315 /sec                      
S1      112              4,519      page-faults                      #  585.802 /sec                      
S1      112              4,508      minor-faults                     #  584.376 /sec                      
S1      112                 12      major-faults                     #    1.556 /sec                      
       0.059585458 seconds time elapsed
```
- rep2_sched: 벤치 결과 없음 (bench.log: Error response from daemon: No such container: vllm-h100)
```
 Performance counter stats for process id '2398552,2398960,2399033,2399337':
            126.82 msec task-clock                       #    2.789 CPUs utilized             
       234,917,024      cycles                           #    1.852 GHz                         (74.70%)
       307,528,723      instructions                     #    1.31  insn per cycle              (77.16%)
         1,033,236      cache-references                 #    8.147 M/sec                       (84.47%)
            36,671      cache-misses                     #    3.55% of all cache refs           (76.72%)
             2,994      context-switches                 #   23.608 K/sec                     
                 5      cpu-migrations                   #   39.426 /sec                      
                 0      page-faults                      #    0.000 /sec                      
       0.045475636 seconds time elapsed
```
- rep3_tp0: 벤치 결과 없음 (bench.log: Error response from daemon: No such container: vllm-h100)
```
 Performance counter stats for process id '2398552':
             38.37 msec task-clock                       #    1.073 CPUs utilized             
        72,258,075      cycles                           #    1.883 GHz                         (44.23%)
        99,931,303      instructions                     #    1.38  insn per cycle              (46.26%)
           104,399      LLC-loads                        #    2.721 M/sec                       (46.29%)
             5,924      LLC-load-misses                  #    5.67% of all LL-cache accesses    (29.90%)
        28,259,334      L1-dcache-loads                  #  736.452 M/sec                       (40.92%)
         1,874,756      L1-dcache-load-misses            #    6.63% of all L1-dcache accesses   (50.17%)
        28,630,772      dTLB-loads                       #  746.131 M/sec                       (48.81%)
           131,946      dTLB-load-misses                 #    0.46% of all dTLB cache accesses  (36.01%)
               503      context-switches                 #   13.108 K/sec                     
       0.035766569 seconds time elapsed
```

### 1.3 3차 측정 perf stat (`eval/results/20260915_213804_ide070_measure3/`)

- rep1_sys: 664.93 tok/s, TPOT p50/p95 72.6/85.3 ms
```
 Performance counter stats for 'system wide':
S0      112       7,001,001.52 msec task-clock                       #  112.016 CPUs utilized             
S0      112  5,383,097,979,418      cycles                           #    0.769 GHz                       
S0      112  6,560,929,398,811      instructions                     #    1.22  insn per cycle            
S0      112    119,220,986,079      cache-references                 #   17.029 M/sec                     
S0      112     81,254,406,012      cache-misses                     #   68.15% of all cache refs         
S0      112    806,198,569,918      branches                         #  115.155 M/sec                     
S0      112      1,715,271,771      branch-misses                    #    0.21% of all branches           
S0      112          7,492,731      context-switches                 #    1.070 K/sec                     
S0      112            112,618      cpu-migrations                   #   16.086 /sec                      
S0      112            363,544      page-faults                      #   51.927 /sec                      
S0      112            362,982      minor-faults                     #   51.847 /sec                      
S0      112                  0      major-faults                     #    0.000 /sec                      
S1      112       7,000,632.34 msec task-clock                       #  112.010 CPUs utilized             
S1      112  5,202,397,445,100      cycles                           #    0.743 GHz                       
S1      112  6,718,015,928,705      instructions                     #    1.29  insn per cycle            
S1      112    106,027,939,081      cache-references                 #   15.145 M/sec                     
S1      112     72,784,566,099      cache-misses                     #   68.65% of all cache refs         
S1      112    866,541,033,598      branches                         #  123.780 M/sec                     
S1      112      1,458,502,128      branch-misses                    #    0.17% of all branches           
S1      112          9,502,237      context-switches                 #    1.357 K/sec                     
S1      112            331,028      cpu-migrations                   #   47.285 /sec                      
S1      112            868,121      page-faults                      #  124.006 /sec                      
S1      112            867,832      minor-faults                     #  123.965 /sec                      
S1      112                 31      major-faults                     #    0.004 /sec                      
      62.500231624 seconds time elapsed
```
- rep2_sched: 648.13 tok/s, TPOT p50/p95 75.6/85.0 ms
```
 Performance counter stats for process id '3133894,3133968,3134212,3134311':
      5,040,448.82 msec task-clock                       #   79.172 CPUs utilized             
10,005,916,873,302      cycles                           #    1.985 GHz                       
12,952,696,579,658      instructions                     #    1.29  insn per cycle            
   223,627,183,962      cache-references                 #   44.367 M/sec                     
   156,947,133,328      cache-misses                     #   70.18% of all cache refs         
         1,392,203      context-switches                 #  276.206 /sec                      
            12,437      cpu-migrations                   #    2.467 /sec                      
             1,989      page-faults                      #    0.395 /sec                      
      63.664317030 seconds time elapsed
```
- rep3_tp0: 636.48 tok/s, TPOT p50/p95 77.2/87.5 ms
```
 Performance counter stats for process id '3133894':
      5,030,306.54 msec task-clock                       #   77.682 CPUs utilized             
 9,913,979,847,244      cycles                           #    1.971 GHz                         (62.50%)
12,520,891,751,534      instructions                     #    1.26  insn per cycle              (75.00%)
    36,865,543,526      LLC-loads                        #    7.329 M/sec                       (75.00%)
    24,164,053,599      LLC-load-misses                  #   65.55% of all LL-cache accesses    (75.00%)
 2,720,114,722,740      L1-dcache-loads                  #  540.745 M/sec                       (75.00%)
    88,116,956,916      L1-dcache-load-misses            #    3.24% of all L1-dcache accesses   (75.00%)
 2,728,836,825,895      dTLB-loads                       #  542.479 M/sec                       (50.00%)
       612,610,856      dTLB-load-misses                 #    0.02% of all dTLB cache accesses  (50.00%)
           526,649      context-switches                 #  104.695 /sec                      
      64.755423339 seconds time elapsed
```

## 2. A 시리즈 — 코드 무변경 (TSK_055)

A1 (turbo ON): `/sys/devices/system/cpu/intel_pstate/no_turbo` 쓰기가 root 로도 `Operation not permitted` (BIOS 잠금, IDE_030 2026-08-30 기록과 동일) → 미실행. A2 (물리 코어 전용): 2차 측정에서 kt 워커 96 스레드가 물리 코어 0–47, 56–103 에 고정되어 있음이 확인되어 별도 셀 없음.

### A 시리즈 셀

디렉터리 `eval/results/20260915_203843_ide070_aseries/`. a0 = 기준선 (cpuinfer 96, GPU 0,1,2,3); a4_cpu80/112 = cpuinfer 80/112; a6_gpu0145 = CUDA_VISIBLE_DEVICES=0,1,4,5. 각 rep 의 turbostat Bzy_MHz 는 셀 디렉터리 `rep*.turbostat.txt`.

| 셀 | 부팅 | rep | tok/s | 총 tok/s | TPOT p50/p95 ms | TTFT p50/p95 ms | 요청 성공/실패 | CPU busy % | HBM MiB (GPU0-3) |
|---|---|---|---|---|---|---|---|---|---|
| a0_base_turboOFF | HEALTH_OK 110 | rep1 | 642.14 | 3164.9 | 75.7/88.0 | 2944/3851 | 256/0 | 38.5 | 77352/77422/77422/76942 |
|  |  | rep2 | 620.00 | 3055.8 | 80.3/88.9 | 3002/3748 | 256/0 | 38.8 |  |
|  |  | rep3 | 627.63 | 3093.4 | 79.0/88.7 | 3003/3691 | 256/0 | 38.9 |  |
| | | 평균±표준편차 | **629.9 ± 9.2** | | 78.3 ± 1.9 / 88.5 ± 0.4 | 2983.2 ± 27.7 / 3763.4 ± 66.4 | | | |
| a4_cpu112 | HEALTH_OK 100 | rep1 | 652.82 | 3217.5 | 73.4/86.9 | 2943/3757 | 256/0 | 44.0 | 77352/77422/77422/76942 |
|  |  | rep2 | 629.89 | 3104.5 | 78.1/87.5 | 3003/3813 | 256/0 | 44.8 |  |
|  |  | rep3 | 639.51 | 3151.9 | 76.2/88.2 | 2986/3644 | 256/0 | 44.7 |  |
| | | 평균±표준편차 | **640.7 ± 9.4** | | 75.9 ± 1.9 / 87.5 ± 0.5 | 2977.3 ± 25.3 / 3738.2 ± 69.9 | | | |
| a4_cpu80 | HEALTH_OK 110 | rep1 | 617.99 | 3045.8 | 79.0/92.3 | 2576/4039 | 256/0 | 32.6 | 77352/77422/77422/76942 |
|  |  | rep2 | 588.19 | 2899.0 | 83.5/93.2 | 3295/4091 | 256/0 | 33.4 |  |
|  |  | rep3 | 598.14 | 2948.0 | 82.6/95.8 | 3311/3957 | 256/0 | 33.3 |  |
| | | 평균±표준편차 | **601.4 ± 12.4** | | 81.7 ± 2.0 / 93.8 ± 1.5 | 3060.9 ± 342.7 / 4029.2 ± 55.3 | | | |
| a6_gpu0145 | HEALTH_OK 110 | rep1 | 635.54 | 3132.4 | 77.2/89.3 | 2943/3899 | 256/0 | 38.1 | 77352/77422/0/0 |
|  |  | rep2 | 617.94 | 3045.6 | 80.3/89.1 | 3049/3880 | 256/0 | 39.2 |  |
|  |  | rep3 | 634.16 | 3125.5 | 78.3/86.9 | 2997/3637 | 256/0 | 38.9 |  |
| | | 평균±표준편차 | **629.2 ± 8.0** | | 78.6 ± 1.3 / 88.4 ± 1.1 | 2996.4 ± 43.0 / 3805.0 ± 119.4 | | | |

<details><summary>launch 명령</summary>

- `a0_base_turboOFF`: `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`
- `a4_cpu112`: `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 112 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`
- `a4_cpu80`: `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 80 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`
- `a6_gpu0145`: `CUDA_VISIBLE_DEVICES=0,1,4,5 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`

</details>

## 3. B 시리즈 — NUMA (TSK_057)

### B 시리즈 셀

디렉터리 `eval/results/20260915_210401_ide070_bseries/`. b1 = `numactl --interleave=all`; b2 = `numactl --membind=0`; b3 = `--kt-threadpool-count 1`; b4 = `numactl --cpunodebind=0 --membind=0` + cpuinfer 56, threadpool 1. 셀 디렉터리에 numastat_tp0.txt, tp0_thread_affinity.txt, rep*.pcm_numa.csv.

| 셀 | 부팅 | rep | tok/s | 총 tok/s | TPOT p50/p95 ms | TTFT p50/p95 ms | 요청 성공/실패 | CPU busy % | HBM MiB (GPU0-3) |
|---|---|---|---|---|---|---|---|---|---|
| b1_interleave | HEALTH_OK 110 | rep1 | 666.52 | 3285.0 | 72.1/84.6 | 2962/3774 | 256/0 | 37.0 | 77352/77422/77422/76942 |
|  |  | rep2 | 654.57 | 3226.2 | 74.4/85.5 | 2999/3630 | 256/0 | 37.5 |  |
|  |  | rep3 | 647.99 | 3193.7 | 75.7/85.1 | 2982/3634 | 256/0 | 37.5 |  |
| | | 평균±표준편차 | **656.4 ± 7.7** | | 74.0 ± 1.5 / 85.1 ± 0.4 | 2981.0 ± 15.1 / 3679.4 ± 67.2 | | | |
| b2_membind0 | HEALTH_OK 100 | rep1 | 663.90 | 3272.1 | 72.8/86.2 | 2946/3781 | 256/0 | 37.3 | 77352/77422/77422/76942 |
|  |  | rep2 | 641.00 | 3159.3 | 77.3/85.9 | 3011/3658 | 256/0 | 37.7 |  |
|  |  | rep3 | 644.12 | 3174.6 | 75.9/86.0 | 2991/3630 | 256/0 | 37.7 |  |
| | | 평균±표준편차 | **649.7 ± 10.1** | | 75.3 ± 1.9 / 86.0 ± 0.1 | 2982.4 ± 27.0 / 3689.8 ± 65.4 | | | |
| b3_pool1 | HEALTH_OK 110 | rep1 | 446.56 | 2200.9 | 119.3/136.0 | 2812/4053 | 256/0 | 39.5 | 77352/77422/77422/76942 |
|  |  | rep2 | 451.79 | 2226.7 | 119.7/129.7 | 2850/4020 | 256/0 | 39.5 |  |
|  |  | rep3 | 452.14 | 2228.4 | 119.7/129.8 | 2844/3896 | 256/0 | 39.4 |  |
| | | 평균±표준편차 | **450.2 ± 2.6** | | 119.6 ± 0.2 / 131.8 ± 2.9 | 2835.1 ± 16.4 / 3989.8 ± 67.7 | | | |
| b4_local | HEALTH_OK 100 | rep1 | 444.10 | 2188.8 | 122.4/133.8 | 2825/4126 | 256/0 | 24.2 | 77352/77422/77422/76942 |
|  |  | rep2 | 426.02 | 2100.9 | 126.8/142.1 | 3532/4071 | 256/0 | 24.7 |  |
|  |  | rep3 | 418.69 | 2063.6 | 127.3/142.0 | 3517/4015 | 256/0 | 25.0 |  |
| | | 평균±표준편차 | **429.6 ± 10.7** | | 125.5 ± 2.2 / 139.3 ± 3.9 | 3291.1 ± 329.8 / 4070.6 ± 45.2 | | | |

<details><summary>launch 명령</summary>

- `b1_interleave`: `numactl --interleave=all env CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`
- `b2_membind0`: `numactl --membind=0 env CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`
- `b3_pool1`: `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 1 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`
- `b4_local`: `numactl --cpunodebind=0 --membind=0 env CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 56 --kt-threadpool-count 1 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`

</details>

## 4. C 시리즈 — hot expert 배치 (TSK_056)

- 층별 예산 (`build_layer_budget.py`, 슬롯 5952 고정, 입력 = 1차 측정 recorder 트레이스): per_layer 최소 75 / 최대 142. 같은 트레이스 기준 coverage 평균 uniform 98.581 % → 비균일 98.799 %, coverage 최소 91.58 % → 98.11 %, cold 선택/토큰 7.039 → 5.955.
- per_layer: `[142, 117, 107, 116, 106, 82, 113, 96, 89, 98, 88, 98, 95, 98, 97, 89, 88, 90, 95, 107, 89, 89, 86, 81, 85, 89, 88, 85, 85, 90, 79, 95, 93, 75, 80, 87, 87, 90, 89, 105, 97, 99, 91, 100, 90, 97, 104, 87, 105, 102, 97, 102, 104, 100, 102, 93, 98, 101, 106, 104, 113, 102]`
- hotmap_v2 (1차 측정 트레이스로 재생성, calls 280,450,304): top-N coverage 평균/최소 = 64: 93.8/77.1 %, 80: 96.9/85.6 %, 96: 98.6/91.6 %, 112: 99.5/95.6 %, 128: 99.9/98.1 %
- 구현: `patch_per_layer_experts.sh` 가 컨테이너의 `kt_ep_wrapper.py` `create_kt_config_from_server_args` 에서 `num_gpu_experts` 를 환경변수 `KT_GPU_EXPERTS_PER_LAYER` (json `per_layer[layer_idx]`) 로 치환. 실험 후 원복. 팩토리 단위 검증: layer0 → 142, layer33 → 75, layer61 → 102, 환경변수 없음 → 96.

### C 시리즈 셀 (20260915_202341_ide070_tsk056)

디렉터리 `eval/results/20260915_202341_ide070_tsk056/`. u96v2 = uniform 96 + hotmap_v2; nu5952 = 층별 예산 + hotmap_v2 + 패치. **이 실행의 nu5952 는 패치 미적용 (patch 스크립트 `docker exec` 에 `-i` 누락, 서버 로그 per-layer 라인 0) — uniform 96 과 동일 구성.**

| 셀 | 부팅 | rep | tok/s | 총 tok/s | TPOT p50/p95 ms | TTFT p50/p95 ms | 요청 성공/실패 | CPU busy % | HBM MiB (GPU0-3) |
|---|---|---|---|---|---|---|---|---|---|
| nu5952 | HEALTH_OK 100 | rep1 | 682.15 | 3362.1 | 69.5/82.6 | 2924/3733 | 256/0 | 37.0 | 77352/77422/77422/76942 |
|  |  | rep2 | 661.49 | 3260.3 | 73.4/82.5 | 3018/3770 | 256/0 | 37.7 |  |
|  |  | rep3 | 662.35 | 3264.5 | 73.6/82.4 | 2998/3647 | 256/0 | 37.7 |  |
| | | 평균±표준편차 | **668.7 ± 9.5** | | 72.2 ± 1.9 / 82.5 ± 0.1 | 2980.2 ± 40.8 / 3716.8 ± 51.4 | | | |
| u96v2 | HEALTH_OK 100 | rep1 | 677.58 | 3339.6 | 70.5/83.1 | 2988/3813 | 256/0 | 36.9 | 77352/77422/77422/76942 |
|  |  | rep2 | 645.81 | 3191.6 | 75.2/90.0 | 2999/3752 | 256/0 | 37.9 |  |
|  |  | rep3 | 666.87 | 3286.8 | 72.5/81.6 | 3001/3633 | 256/0 | 37.5 |  |
| | | 평균±표준편차 | **663.4 ± 13.2** | | 72.7 ± 1.9 / 84.9 ± 3.7 | 2995.9 ± 5.4 / 3732.8 ± 74.8 | | | |

<details><summary>launch 명령</summary>

- `nu5952`: `KT_GPU_EXPERTS_PER_LAYER=/models/kt/ide070/layer_budget_5952.json CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide070/hotmap_v2.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`
- `u96v2`: `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide070/hotmap_v2.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`

</details>

### C 시리즈 셀 (20260915_213137_ide070_tsk056)

디렉터리 `eval/results/20260915_213137_ide070_tsk056/`. u96v2 = uniform 96 + hotmap_v2; nu5952 = 층별 예산 + hotmap_v2 + 패치.

| 셀 | 부팅 | rep | tok/s | 총 tok/s | TPOT p50/p95 ms | TTFT p50/p95 ms | 요청 성공/실패 | CPU busy % | HBM MiB (GPU0-3) |
|---|---|---|---|---|---|---|---|---|---|
| nu5952 | HEALTH_OK 110 | rep1 | 717.70 | 3537.3 | 65.8/78.7 | 2889/3711 | 256/0 | 36.9 | 77444/77514/77514/77034 |
|  |  | rep2 | 700.56 | 3452.8 | 67.8/78.0 | 2961/3593 | 256/0 | 37.3 |  |
|  |  | rep3 | 671.91 | 3314.9 | 71.8/80.6 | 3031/3702 | 256/0 | 37.1 |  |
| | | 평균±표준편차 | **696.7 ± 18.9** | | 68.4 ± 2.5 / 79.1 ± 1.1 | 2960.4 ± 57.9 / 3668.9 ± 53.6 | | | |

<details><summary>launch 명령</summary>

- `nu5952`: `KT_GPU_EXPERTS_PER_LAYER=/models/kt/ide070/layer_budget_5952.json CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide070/hotmap_v2.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`

</details>

## 5. D 시리즈 — 비동기 파이프라인 (TSK_058)

이 빌드의 kt-kernel 에는 IDE_033 (2026-09-09) 의 callback-free 핸드오프 (mapped go/done 플래그 + 폴러 스레드 + 스트림 memop), pinned 슬롯 링 버퍼 (`KExpertsCPUBuffer`, pin_memory=True, non_blocking 복사) 가 이미 구현되어 있고 환경변수로 켠다. 셀: d1 = `KT_CALLBACK_FREE=1`; d2 = + `KT_CF_SKIP_EMPTY_IMM=1`, deferred 8; d3 = d2 + 비-kt 스레드 코어 재배치 (`pin_nonkt.sh`, 빈 코어 48-55,104-111 + HT 형제); d4 = 2026-09-10 캠페인 스택 (ENVC: KT_AVX_RB=1 KT_AVX_PF=2 KT_FUSE_QIN=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25; hotmap_mixed_0.25; deferred 8; fp8_e5m2 KV; graph bs 32–224; max-total-tokens 143,360; chunked-prefill 4,096; mem-fraction 0.94) C64 3회 + C224 1회.

### D 시리즈 셀

디렉터리 `eval/results/20260915_214429_ide070_dseries/`. cf_lines.txt = 서버 로그의 `[kt-cf]` 라인 수. d2_cf_skip8: rep1 도중 요청 128/256 실패, 이후 스케줄러 프로세스 종료 (server_tail.log 에 Python fatal 덤프와 py-spy 실패 메시지만 남음), rep2·rep3 은 서버 부재로 0.

| 셀 | 부팅 | rep | tok/s | 총 tok/s | TPOT p50/p95 ms | TTFT p50/p95 ms | 요청 성공/실패 | CPU busy % | HBM MiB (GPU0-3) |
|---|---|---|---|---|---|---|---|---|---|
| d1_cf | HEALTH_OK 100 | rep1 | 678.39 | 3343.6 | 73.6/88.5 | 2503/3605 | 256/0 | 38.3 | 77352/77422/77422/76942 |
|  |  | rep2 | 660.30 | 3254.4 | 73.3/87.3 | 3139/3774 | 256/0 | 38.6 |  |
|  |  | rep3 | 669.98 | 3302.1 | 72.5/85.7 | 3082/3649 | 256/0 | 38.6 |  |
| | | 평균±표준편차 | **669.6 ± 7.4** | | 73.1 ± 0.5 / 87.2 ± 1.1 | 2907.7 ± 287.2 / 3675.9 ± 71.4 | | | |
| d2_cf_skip8 | HEALTH_OK 100 | rep1 | 52.67 | 259.5 | 63.0/70.7 | 1896/2602 | 128/128 | 5.2 | 77334/77422/77422/76942 |
|  |  | rep2 | 0.00 | 0.0 | 0.0/0.0 | 0/0 | 0/256 | 2.4 |  |
|  |  | rep3 | 0.00 | 0.0 | 0.0/0.0 | 0/0 | 0/256 | 2.5 |  |
| | | 평균±표준편차 | **17.6 ± 24.8** | | 21.0 ± 29.7 / 23.6 ± 33.3 | 631.9 ± 893.6 / 867.3 ± 1226.6 | | | |
| d3_cf_skip8_pin | HEALTH_OK 100 | rep1 | 792.22 | 3904.6 | 64.1/73.5 | 1874/2888 | 256/0 | 37.4 | 77334/77422/77422/76942 |
|  |  | rep2 | 778.11 | 3835.0 | 65.5/75.6 | 2182/2742 | 256/0 | 37.3 |  |
|  |  | rep3 | 773.44 | 3812.0 | 66.3/77.0 | 2144/2658 | 256/0 | 37.2 |  |
| | | 평균±표준편차 | **781.3 ± 8.0** | | 65.3 ± 0.9 / 75.4 ± 1.4 | 2066.5 ± 137.3 / 2762.7 ± 94.9 | | | |
| d4_epoch | HEALTH_OK 100 | rep1 | 813.62 | 4010.0 | 57.7/70.9 | 2563/4122 | 256/0 | 37.1 | 78932/78946/78946/78466 |
|  |  | rep2 | 852.97 | 4204.0 | 56.1/67.8 | 2347/3393 | 256/0 | 37.1 |  |
|  |  | rep3 | 781.52 | 3851.9 | 58.4/71.2 | 3114/4038 | 256/0 | 37.3 |  |
|  |  | c224 | 1089.19 | 5367.9 | 148.6/191.9 | 7371/12541 | 896/0 | 43.0 |  |
| | | 평균±표준편차 | **816.0 ± 29.2** | | 57.4 ± 0.9 / 70.0 ± 1.5 | 2674.6 ± 323.0 / 3851.0 ± 325.6 | | | |

<details><summary>launch 명령</summary>

- `d1_cf`: `KT_CALLBACK_FREE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`
- `d2_cf_skip8`: `KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 8 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`
- `d3_cf_skip8_pin`: `KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 8 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`
- `d4_epoch`: `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 KT_AVX_RB=1 KT_AVX_PF=2 KT_FUSE_QIN=1 KT_AMX_MIN_QLEN=1000000 KT_AMX_MIN_ROWS=3 KT_CALLBACK_FREE=1 KT_COLD_DEFER=1 KT_COLD_TAU=0.25 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --cuda-graph-backend-prefill disabled --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --init-expert-location /models/kt/ide070/hotmap_mixed_0.25.json --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 8 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --kv-cache-dtype fp8_e5m2 --cuda-graph-max-bs 224 --cuda-graph-bs 32 64 96 128 160 192 224 --kt-num-gpu-experts 96 --max-total-tokens 143360`

</details>

## 6. E 시리즈 — GPU expert W4 (TSK_059)

이 모델의 compressed-tensors W4A16 체크포인트는 없음 (검색: AWQ 4종, GPTQ-Int4-Int8Mix, AutoRound int4-mixed, NVFP4 6종). QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ (awq 4-bit, group 128, 252 GB) 를 사용. CPU expert 는 FP8 스냅샷에서 변환한 kt INT4 그대로. 셀: hot 96 / 128 / 144, 나머지 기준 구성.

### E 시리즈 셀

디렉터리 `eval/results/20260915_221142_ide070_eseries/`. 부팅 실패 셀은 `server_boot_fail.log` 에 서버 로그 400행. *_nograph = `--disable-cuda-graph` 재시도.

| 셀 | 부팅 | rep | tok/s | 총 tok/s | TPOT p50/p95 ms | TTFT p50/p95 ms | 요청 성공/실패 | CPU busy % | HBM MiB (GPU0-3) |
|---|---|---|---|---|---|---|---|---|---|
| e1_awq_h96 | DIED 700 | e1_awq_h96 | (없음) | | | | | | — |
| e2_awq_h128 | DIED 100 | e2_awq_h128 | (없음) | | | | | | — |
| e3_awq_h144 | DIED 110 | e3_awq_h144 | (없음) | | | | | | — |

<details><summary>launch 명령</summary>

- `e1_awq_h96`: `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--QuantTrio--Qwen3-Coder-480B-A35B-Instruct-AWQ/snapshots/9ce3eaa67fe88609afec235117e97eb03d9b3cda --served-model-name q480awq --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`
  - error_lines: `[2026-09-15 13:12:28 TP1] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
RuntimeError: Failed at /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/elementwise/activation.cuh:208: CUDA error: an illegal memory access was encountered
Exc`
- `e2_awq_h128`: `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--QuantTrio--Qwen3-Coder-480B-A35B-Instruct-AWQ/snapshots/9ce3eaa67fe88609afec235117e97eb03d9b3cda --served-model-name q480awq --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 128 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`
  - error_lines: `RuntimeError: moe_sum_reduce CUDA kernel (small-token) launch failed
RuntimeError: moe_sum_reduce CUDA kernel (small-token) launch failed
RuntimeError: moe_sum_reduce CUDA kernel (small-token) launch failed
RuntimeError: Failed at /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/elementwise/activation.cuh:208: CUDA error: an illegal memory access was encountered
Exception: Capture cuda graph f`
- `e3_awq_h144`: `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--QuantTrio--Qwen3-Coder-480B-A35B-Instruct-AWQ/snapshots/9ce3eaa67fe88609afec235117e97eb03d9b3cda --served-model-name q480awq --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 144 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic`
  - error_lines: `Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
Exception: Capture cuda graph failed: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nv`

</details>

### E 시리즈 셀

디렉터리 `eval/results/20260915_222941_ide070_eseries/`. 부팅 실패 셀은 `server_boot_fail.log` 에 서버 로그 400행. *_nograph = `--disable-cuda-graph` 재시도.

| 셀 | 부팅 | rep | tok/s | 총 tok/s | TPOT p50/p95 ms | TTFT p50/p95 ms | 요청 성공/실패 | CPU busy % | HBM MiB (GPU0-3) |
|---|---|---|---|---|---|---|---|---|---|
| e1_awq_h96_nograph | DIED 100 | e1_awq_h96_nograph | (없음) | | | | | | — |
| e2_awq_h128_nograph | DIED 100 | e2_awq_h128_nograph | (없음) | | | | | | — |

<details><summary>launch 명령</summary>

- `e1_awq_h96_nograph`: `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--QuantTrio--Qwen3-Coder-480B-A35B-Instruct-AWQ/snapshots/9ce3eaa67fe88609afec235117e97eb03d9b3cda --served-model-name q480awq --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic --disable-cuda-graph`
  - error_lines: `[2026-09-15 13:31:39] Pyspy failed (py-spy dump --native --pid 774658). Error: Error: No such process (os error 3)
    No such process (os error 3)
[2026-09-15 13:31:39] Pyspy failed (py-spy dump  --pid 774658). Error: Error: Failed to get process executable name. Check that the process is running.
    0: No such file or directory (os error 2)
    1: No such file or directory (os error 2)`
- `e2_awq_h128_nograph`: `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--QuantTrio--Qwen3-Coder-480B-A35B-Instruct-AWQ/snapshots/9ce3eaa67fe88609afec235117e97eb03d9b3cda --served-model-name q480awq --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 128 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic --disable-cuda-graph`
  - error_lines: `[2026-09-15 13:33:25 TP3] Ignore import error when loading sglang.srt.models.sarashina2_vision: cannot import name 'MultimodalDataItem' from 'sglang.srt.managers.mm_utils' (/sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py)
RuntimeError: moe_sum_reduce CUDA kernel (small-token) launch failed
RuntimeError: moe_sum_reduce CUDA kernel (small-token) launch failed
RuntimeError: Failed at /sg`

</details>

## 7. 파일 색인

- 하네스: `eval/ide070/` (run_measure.sh, run_measure2.sh, run_measure3.sh, analyze_measure.py, build_layer_budget.py, patch_per_layer_experts.sh, run_aseries.sh, run_bseries.sh, run_tsk056.sh, run_dseries.sh, run_eseries.sh, make_result.py), 공용 `eval/ide068/lib.sh`.
- 결과: `eval/results/*_ide070_*` (대용량 .pt / perf.data / speedscope 는 git 제외). 진행 로그: `shadow_assists/features/IDE_070/PROGRESS.md`. 지시서 사본: `cpu_offload_no_01_지시서.md`.
- hotmap·예산: `~/.cache/huggingface/kt/ide069/hotmap.json` (v1), `~/.cache/huggingface/kt/ide070/{hotmap_v2.json, layer_budget_5952.json, hotmap_mixed_0.25.json}` (컨테이너 `/models/kt/...`).
