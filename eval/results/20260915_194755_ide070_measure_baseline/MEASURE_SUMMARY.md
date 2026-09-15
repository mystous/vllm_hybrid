# IDE_070 TSK_054 측정 요약 — `20260915_194755_ide070_measure_baseline`

launch: `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model …`

## 벤치 (C=64, N=256, sonnet 512/128)

| rep | tok/s | TPOT p50/p95 ms | TTFT p50/p95 ms | dur s | in/out tok |
|---|---|---|---|---|---|
| rep1_pcm | 633.18 | 77.4/86.6 | 3075/4355 | 51.8 | 128730/32767 |
| rep2_pcmnuma | 634.48 | 78.1/86.5 | 3002/3741 | 51.6 | 128723/32739 |
| rep3_perf | 638.97 | 77.8/86.1 | 2996/3653 | 51.3 | 128727/32766 |

## pcm (rep1, 2 s 간격, 부하 구간 27/34 샘플 ≈ 54 s)

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

## pcm-numa (rep2, 부하 구간 26 샘플; 접근 수 = demand + L2 prefetch, code, RFO)

| 범위 | IPC | local DRAM 접근 | remote DRAM 접근 | **remote %** |
|---|---|---|---|---|
| system | 1.25 | 1.524e+11 | 3.139e+09 | 2.0 |
| socket0 코어 | 1.22 | 7.789e+10 | 2.827e+09 | 3.5 |
| socket1 코어 | 1.29 | 7.451e+10 | 3.118e+08 | 0.4 |

## turbostat (rep3, 5 s 간격, 부하 11/14 샘플, 전체 요약행)

- Busy% 44.1, Avg_MHz 878, **Bzy_MHz 1990** (turbo OFF 확인), IPC 1.23

## mpstat 코어별 사용률 (%usr+%sys, rep1_pcm, 부하 27 샘플)

- 전체 43.4% · socket0 44.7% (phys 81.8 / HT 7.6) · socket1 42.0% (phys 81.5 / HT 2.6)
- 분포: {'<10%': 112, '10-50%': 15, '50-90%': 2, '>=90%': 95} · ≥50% 코어 수 97/224 · 최상위: cpu32=96%, cpu12=95%, cpu33=95%, cpu30=95%, cpu7=94%, cpu17=94%, cpu1=94%, cpu37=94%

- vmstat (rep1_pcm, 부하 26 샘플): 인터럽트 265286/s, **컨텍스트 스위치 242368/s**, us 44 sy 2 id 54, runq 105
- pidstat TP0 스레드 합 (rep1_pcm, 상위 절반 구간 평균): 자발 683/s, 비자발 7/s
- vmstat (rep2_pcmnuma, 부하 26 샘플): 인터럽트 264012/s, **컨텍스트 스위치 240853/s**, us 44 sy 2 id 54, runq 104
- pidstat TP0 스레드 합 (rep2_pcmnuma, 상위 절반 구간 평균): 자발 699/s, 비자발 8/s
- vmstat (rep3_perf, 부하 26 샘플): 인터럽트 262082/s, **컨텍스트 스위치 238601/s**, us 44 sy 2 id 54, runq 105
- pidstat TP0 스레드 합 (rep3_perf, 상위 절반 구간 평균): 자발 3700/s, 비자발 9/s

## perf_stat_sys.txt (주의: perf 창 400 s = 부하 ~60 s + 유휴; 비율 지표만 신뢰)
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

## perf_stat_tp0.txt (주의: perf 창 400 s = 부하 ~60 s + 유휴; 비율 지표만 신뢰)
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

## recorder → cold expert (rep1, hotmap96 기준 physical ≥96 = CPU)

| 파일 | 토큰 수 | cold 선택/토큰 (8 선택 중) | cold 비율 | 층별 cold 최소/최대 |
|---|---|---|---|---|
| 789469485.6312454.pt | 565424 | **7.302** | 1.47% | 0.43% / 8.44% |
| 789469485.6312456.pt | 565424 | **7.302** | 1.47% | 0.43% / 8.44% |
| 789469485.6312597.pt | 565424 | **7.302** | 1.47% | 0.43% / 8.44% |

- HBM after boot: 0, 77402 MiB; 1, 77464 MiB; 2, 77464 MiB; 3, 76984 MiB; 4, 0 MiB; 5, 0 MiB; 6, 0 MiB; 7, 0 MiB
- TP0 affinity: Cpus_allowed_list:	0-223 · Mems_allowed_list:	0-1
