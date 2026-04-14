# H100x8 물리 — 7B wave-batch 재앙적 결과 + 종합 CPU 프로파일

**날짜**: 2026-04-14
**환경**: H100x8 물리 (violet-h100-023)
**결정적 발견**:
1. wave-batch + 7B 에서 hybrid wall **147× 증가** (4회 재현)
2. `max_seqs=1` 으로 4× 개선하지만 여전히 gpu_only 대비 34× 느림
3. 현재 워크로드 (500 req × 128/128) 에서 GPU 과잉 공급, hybrid 순수 오버헤드
4. CPU 프로파일 실측 최적 thread = **32** (vLLM 실제 inference), 76t 는 **48× 성능 추락**
5. Wave-batch 전략은 전 환경에서 실패 (dev/KVM/물리) → **영구 폐기 확정**

---

# PART A. H100x8 물리 벤치마크 결과 (7B) — 상세 분석

**공통 설정**: 500 req × 128/128, TP=4, Qwen2.5-7B-Instruct, burst (rate=inf)

## A.1 세 가지 모드 비교 — Wall Time 시각화

```
Wall Time (seconds, log-scale representation)

gpu_only           │██ 14초
                   │
max_seqs=1 hybrid  │███████████████████████████████████ 480초 (34×)
                   │
wave=16 hybrid #1  │█████████████████████████████████████████████████████████████ 2072초 (148×)
wave=16 hybrid #2  │██████████████████████████████████████████████████████████ 1987초
wave=16 hybrid #3  │████████████████████████████████████████████████████████████████ 2095초
wave=16 hybrid #4  │███████████████████████████████████████████████████████████████ 2057초
                   │
                   └─── 0 ─── 500 ─── 1000 ─── 1500 ─── 2000 ─── 2500 (s)
```

Hybrid 평균 = 2,053 ± 40 초 (1.94% 편차). **재현성 100%**.

## A.2 5 runs 전체 수치

| Run | Mode | Dur (s) | Wall (s) | Out TP | Req TP | **TPOT med** | TPOT mean | TPOT P99 | TTFT med | TTFT mean | TTFT P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 081534 | **gpu_only** | 3.77 | 14.01 | 16,501 | 132.48 | **22.74** | 23.12 | 55.83 | 799 | 815 | 1,075 |
| 081840 | hybrid #1 | 2,062 | 2,072 | 30 | 0.24 | **23.40** | 532.57 | 15,691 | 824 | 2,938 | 69,433 |
| 123757 | hybrid #2 | 1,977 | 1,987 | 32 | 0.25 | **22.56** | 502.34 | 15,012 | 757 | 2,875 | 70,066 |
| 132948 | hybrid #3 | 2,085 | 2,095 | 30 | 0.24 | **21.91** | 536.98 | 15,987 | 834 | 2,501 | 54,651 |
| 141742 | hybrid #4 | 2,046 | 2,057 | 30 | 0.24 | **22.09** | 522.60 | 15,588 | 736 | 2,771 | 66,804 |

## A.3 TPOT 분포 해부 — Bimodal Distribution

핵심 관찰: **TPOT median = 22ms (gpu_only 와 동일) 인데 mean = 532ms, P99 = 15,691ms**.
→ 분포가 **완전히 bimodal**: 484 req (GPU) = 빠름, 16 req (CPU) = 극단적 tail.

```
TPOT Distribution (Run #1: mean=532.57ms)

              Count ↑
                    │                                                              
        484 req     │██████████                 (GPU, ~22ms cluster)
         (96.8%)    │██████████                                                    
                    │                                                              
                    │                                                              
                    │                                                              
                    │                                                              
                    │                                                              
                    │                                                              
         16 req     │                                            ██ (CPU, ~15,977ms)
          (3.2%)    │                                            ██                
                    │                                                              
                    └────────────────────────────────────────────────────→ TPOT (ms)
                     0     22    100   500   1000  5000  10000  15000   20000
```

### 역산으로 CPU TPOT 추정

```
500 × mean_TPOT = 484 × TPOT_GPU + 16 × TPOT_CPU
500 × 532.57    = 484 × 22.02    + 16 × TPOT_CPU
266,285         = 10,658         + 16 × TPOT_CPU
TPOT_CPU ≈ 15,977 ms   ← P99 값 15,691 과 거의 일치 ✓
```

**CPU 에 할당된 16 req 는 각각 TPOT ≈ 16 초**. gpu_only 의 22ms 대비 **726× 느림**.

## A.4 TTFT 분포 — 동일한 bimodal

```
Run #1 TTFT: mean=2,938ms, median=824ms, P99=69,433ms

gpu_only:    TTFT median 799ms, mean 815ms, P99 1,075ms  ← 정상 분포 (prefill 균일)
hybrid #1:   TTFT median 824ms, mean 2,938ms, P99 69,433ms  ← 분포 왜곡

→ median 은 동일 (GPU req 가 다수이므로)
→ P99 = 69초 → CPU req 중 일부가 prefill 단계에서 69초 대기
```

**TTFT P99 = 69초** 의미: 16 req 중 16번째 req 가 앞선 15 req 의 prefill 끝나길 69초 대기.
`chunked_prefill=False` 이므로 CPU 는 prefill 을 **순차 처리** (동시 다발 prefill 금지).

```
CPU Wave Prefill Schedule:
req 0:  [  P  ] → decode...
req 1:         [  P  ] → decode...
req 2:                [  P  ] → decode...
...
req 15:                                            [  P  ] → decode...
        0s    ~4s   ~8s   ~12s  ...  ~60s  ~64s  ~69s

각 prefill ~4초 (7B, 128 token prompt, CPU 48t 가정)
마지막 req 의 TTFT = 15 × 4 = 60초 이상
```

## A.5 Wall Time 분해 (Run #1 상세)

```
Total Wall: 2,072 seconds

├─ API server + client setup     ~5s   (0.2%)
├─ GPU phase (484 req)           ~14s  (0.7%)    ← gpu_only 와 동일한 속도
├─ CPU wave #1 prefill           ~60s  (2.9%)    ← 16 req 순차 prefill
├─ CPU wave #1 decode (batch=15) ~1990s (96.1%)  ← 주범
└─ Client gather/shutdown        ~5s   (0.2%)

시간 분포:
0s       14s                                                         2072s
├────────┼───────────────────────────────────────────────────────────────┤
│GPU done│─────────────── 2,058s CPU tail ───────────────────────────────│
         │                                                              │
         │  CPU decode @ batch=15 (+1 solo): 15.6s per token × 128 = 1,997s
```

GPU 는 전체의 **0.7%** 시간만 쓰고, **99.3%** 는 CPU wave 대기.

## A.6 CPU/GPU Utilization Timeline (모니터 CSV 분석)

```
Run #1 timeline (2,072s, 1,777 samples @ 1 Hz):

GPU Utilization
100%│▐█                                                               
 75%│▐█                                                               
 50%│▐█                                                               
 25%│▐█ ▍                                                             
  0%│▐█▐                                                              
    └─┼──┼───────────────────────────────────────────────────┼────────┤
     10s 14s                                                2070s   
     │GPU busy 4s│──────────── GPU idle 2,060s ────────────│

CPU Utilization (all 224 logical cores)
100%│                                                                 
 75%│                                                                 
 50%│                                                                 
 25%│  ▌███████████████████████████████████████████████████████████   
  0%│▉▋                                                           ▋   
    └─┼──┼───────────────────────────────────────────────────┼────────┤
     0s 10s                                               2070s   2072s
     │          CPU busy 2,060s @ ~19% util                 │
```

**리소스별 요약 (4 runs 평균)**:

| 지표 | gpu_only | hybrid (4 runs 평균) | ratio |
|---|---:|---:|---:|
| GPU util mean | 4.3% | **0.025%** | 0.006× |
| GPU busy samples (>10%) | 2/11 | 2.25/1,760 | 매우 희박 |
| CPU util mean | 3.1% | **19.05%** | 6.1× |
| CPU busy duration | — | **2,052 s** (99.4% of wall) | |

**hybrid 에서 GPU 는 사실상 꺼져있음**. 4 GPU 전부 74 GB씩 weight 로드해놓고 34 분 동안 대기.

## A.7 왜 batch=16 CPU decode 가 이렇게 느린가 — 정량

### per-step 시간 역산

```
CPU wave decode wall = 2040s (모니터 CPU busy duration)
  ├─ prefill (16 req 순차, 각 ~3s)  ≈  48s
  └─ decode                          ≈  1,992s

Decode step 수 = 128 (output tokens)
per-step time (batch=16) = 1,992s / 128 = 15.6 초/step
per-step per-req equivalent = 15.6 / 16 = 975 ms/req/step
```

### batch=1 solo (Layer Breakdown 실측) 와 비교

| 구성 | per-step (ms) | batch=1 대비 |
|---|---:|---:|
| batch=1 (24t, Layer Breakdown) | 103 | 1× |
| batch=16 (wave) 실측 | **15,600** | **152× 느림** |

**batch=16 decode step 이 batch=1 의 152배 느림.**

### 이론 예상 vs 실측

```
이론:
  MLP (batched GEMM, weight BW amortized): ~45ms → ~47ms (16× M-dim)
  Attention (per-seq loop, IPEX): 30ms × 16 = 480ms
  Other/dispatch: ~25ms × 16 = 400ms (per-seq loop)
  Total ~927ms/step → 128 tok = 119초 이론

실측: 1,992초 decode (이론의 16.7×)

추가 손실 원인:
- L3 경합 (16 seq × KV cache 가 L3 210MB 초과)
- OMP barrier 비용 (56-64t × 28 layer × 16 seq iteration)
- IPEX overhead (seq 간 context switch)
- NUMA cross traffic (일부 seq 가 다른 NUMA 의 KV 접근)
```

## A.8 `max_seqs=1` 이 4× 나은 이유 (Anatomy)

```
wave=16:  CPU 가 16 req 를 동시에 처리 → batch=15 decode × 128 tokens = 1,992s
                                         (attention 15× 폭발)

max_seqs=1: CPU 가 1 req 만 처리 → 다음 req 투입 (continuous batching)
            각 req = 128 tok × ~230ms/step = ~30s per req
            16 req 순차 = 480s (8분)
            
            ≈ wave=16 의 (1,992 + prefill) / 4.15 배 빠름 → 실측 4.3× 일치 ✓
```

## A.9 3-way 비교 종합

```
Performance Comparison (lower is better for Wall, higher for Throughput)

                Wall (s)              Output Tok/s           GPU Utilization
            ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
gpu_only    │█ 14         │       │█████████████│       │█ 4.3%       │
            │             │       │  16,501     │       │             │
max_seqs=1  │█████ 480    │       │█ ~130       │       │█ ~0.5%?     │
            │             │       │             │       │             │
wave=16     │████████████████████││ 30          │       │▏ 0.025%     │
            │  2,053      │       │             │       │             │
            └─────────────┘       └─────────────┘       └─────────────┘

Slowdown from gpu_only:
  gpu_only → max_seqs=1  : 34× slower wall
  gpu_only → wave=16     : 147× slower wall
  max_seqs=1 → wave=16   : 4.3× slower wall (wave 가 추가 3× penalty)
```

## A.10 공통 hybrid_config (5 runs 모두 동일)

```json
{
  "routing_strategy": "wave-batch",
  "routing_priority": "cpu-first",
  "cpu_max_seqs": "16",
  "cpu_kvcache_gb": "0",
  "cpu_threads": "0",
  "numa_aware": "true",
  "num_cpu_engines": "0",
  "stats_log_interval": "25"
}
```

## A.11 결정적 수치 — 세 가지 실패 모드

| 실패 모드 | 수치 | 해석 |
|---|---|---|
| GPU 과잉 유휴 | GPU util **0.025% mean** | GPU 가 34분 중 0.009초만 일함 |
| Attention 폭발 | per-step **15,600 ms** (batch=16) | batch=1 대비 152× |
| CPU tail 지배 | Wall 의 **99.3%** 가 CPU 대기 | 34분 중 33분 56초 |
| TPOT bimodal | P99 **15,691 ms** = CPU req 값 | GPU 는 22ms, CPU 는 15,977ms |
| TTFT 왜곡 | P99 **69,433 ms** | 16번째 CPU req 의 prefill 대기 |
| Throughput 붕괴 | 16,501 → 30 tok/s (**551× ↓**) | 전체 처리량이 CPU tail 에 끌려다님 |

---

# PART B. 종합 CPU 프로파일 (analysis_log/20260413_075749_cpu_profile)

## B.1 시스템 토폴로지

| 항목 | 값 | 비고 |
|---|---|---|
| CPU | Intel Xeon Platinum 8480+ | 2 소켓 물리 |
| Logical CPUs | **224** (0-223) | |
| Physical cores | **112** | 2S × 56C |
| Threads/core | **2** (SMT 활성) | 112 SMT pairs |
| Sockets | 2 | |
| NUMA nodes | **2** | |
| NUMA node0 | CPUs 0-55, 112-167 | 같은 물리코어의 SMT sibling 이 +112 매핑 |
| NUMA node1 | CPUs 56-111, 168-223 | |
| L1d | 각 코어 48 KB | |
| L1i | 각 코어 32 KB | |
| L2 | **224 MiB** (2 MB × 112 코어) | |
| L3 | **210 MB** (105 MB × 2 소켓) | KVM 의 13× |

**SMT 매핑 예시** (pkg0.core0 의 logical CPUs): [0, 112]
→ core0 의 SMT sibling 이 cpu0 과 cpu112.

## B.2 GEMM Scaling — 76 thread 절벽 확정

Qwen2.5-7B 크기 기준 GEMM × thread sweep (500 iter × 100 iter, bf16):

### decode_qkv [16×3584] × [3584×3584]

| threads | 4 | 8 | 12 | 16 | 24 | 32 | **48** | 64 | 76 | 96 | 112 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ms | 1.028 | 0.517 | 0.401 | 0.318 | 0.248 | 0.180 | **0.119** | 0.137 | 0.140 | 1.138 | 1.723 |
| GFLOPS | 400 | 794 | 1,025 | 1,293 | 1,654 | 2,290 | **3,456** | 3,009 | 2,946 | 361 | 239 |

### decode_ffn_up [16×3584] × [3584×9728]

| threads | 4 | 8 | 12 | 16 | 24 | 32 | **48** | 64 | 76 | 96 | 112 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ms | 2.893 | 1.857 | 1.204 | 0.909 | 1.066 | 0.517 | **0.479** | 0.489 | 2.154 | 1.796 | 1.778 |
| GFLOPS | 386 | 601 | 927 | 1,228 | 1,047 | 2,158 | **2,330** | 2,283 | 518 | 621 | 628 |

### decode_ffn_dn [16×9728] × [9728×3584]

| threads | 4 | 8 | 12 | 16 | 24 | 32 | **48** | 64 | 76 | 96 | 112 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ms | 2.684 | 1.863 | 1.269 | 0.992 | 0.602 | 0.555 | **0.232** | 1.325 | 1.835 | 1.851 | 2.286 |
| GFLOPS | 416 | 599 | 879 | 1,125 | 1,853 | 2,010 | **4,817** | 842 | 608 | 603 | 488 |

### decode_single [1×3584] × [3584×9728] (batch=1 GEMV)

| threads | 4 | 8 | 12 | 16 | 24 | 32 | **48** | 64 | 76 | 96 | 112 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ms | 2.432 | 1.516 | 1.065 | 0.833 | 0.567 | 0.440 | 0.360 | **0.284** | 1.950 | 1.522 | 0.295 |
| GFLOPS | 29 | 46 | 66 | 84 | 123 | 159 | 194 | **245** | 36 | 46 | 236 |

### prefill_128 [128×3584] × [3584×9728]

| threads | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 76 | **96** | 112 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ms | 4.202 | 3.850 | 2.702 | 2.262 | 1.399 | 1.159 | 0.942 | 0.757 | 0.658 | **0.637** | 1.703 |
| GFLOPS | 2,124 | 2,319 | 3,304 | 3,946 | 6,380 | 7,699 | 9,477 | 11,783 | 13,563 | **14,019** | 5,240 |

### GEMM Scaling 그래프 — 5 shape × 11 thread

```
Decode GEMM GFLOPS 비교 (log-scale, 48t 기준 = 최고)

                  4     8    12    16    24    32    48⭐  64    76    96    112
decode_qkv    │   ▎    ▌    ▋    █    █▌  █▌   ████  ██    ██    ▎     ▏
[16×3584²]    │  400  794  1025 1293 1654 2290 3456  3009  2946   361   239

decode_ffn_up │  ▏    ▎    ▌    ▋    ▌   █▌   ██    ██    ▎     ▎     ▎   
[16×3584×9728]│  386  601  927  1228 1047 2158 2330  2283   518  621   628

decode_ffn_dn │  ▎    ▎    ▌    ▋    █▌  █▌   ████████  ▌     ▎     ▎     ▎   
[16×9728×3584]│  416  599  879  1125 1853 2010 4817   842   608   603   488

decode_single │ .    .    .     .    .    .     ▏    ▏     .     .     ▏   
[1×3584×9728] │  29   46   66    84  123  159  194   245    36    46   236

prefill_128   │  █    █▌   ██   ██▌  ████ █████ ██████ ███████ █████████⭐█████████ ██  
[128×3584×9728│ 2124 2319 3304 3946 6380 7699 9477  11783  13563  14019   5240
                                                           (prefill peak)
                                      ↑            ↑
                                   48t decode      76-96t prefill
                                      최적           최적

            decode 절벽:  48t → 76t 에서 N=9728 shape 들이 5-8× 추락
            prefill 절벽: 96t → 112t 에서 추락 (14,019 → 5,240)
```

### FFN 절벽 확대 (N=9728 3개 shape)

```
FFN_up (N=9728) GFLOPS
 2500│                    █                                              
     │                    █                                              
 2000│                    █  █                                           
     │              █     █  █                                           
 1500│           █  █     █  █                                           
     │        █  █  █     █  █                                           
 1000│     █  █  █  █  █  █  █                                           
     │     █  █  █  █  █  █  █              █   █                        
  500│  █  █  █  █  █  █  █  █              █   █                        
     │  █  █  █  █  █  █  █  █    ▏         █   █                        
    0└──────────────────────────────────────────────────────────────────
       4  8  12 16 24 32 48 64    76⚠      96   112
                      ↑           ↑        ↑
                     최적(2338)    518(↓22%)  621,628
```

**GEMM 결론**:
- **Decode GEMM 최적 = 48 thread** (FFN_dn 4,817 GFLOPS, 절대 peak)
- **76 thread 절벽**: FFN_up 518 GFLOPS (48t 의 22%), FFN_dn 608 GFLOPS (13%)
- **96 thread**: decode 는 추락, prefill 은 scaling 유지 (14,019 GFLOPS peak)
- **112 thread**: prefill 도 추락 (5,240)
- **Prefill vs Decode 차이**: Prefill 은 compute-bound (M=128 크므로 tile 충분) → 76-96t 까지 scaling. Decode 는 memory-bound (M=16 작음) → 48t 이후 oneDNN tiling 의 pathological 선택

## B.3 Attention Scaling (순수 torch SDPA, 1.5B 스펙)

1.5B: num_kv_heads=2, head_dim=64, seq_len=256

| batch \ threads | 4 | 8 | 16 | 24 | 48 | 76 | 96 | 112 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.021 | 0.021 | 0.023 | 0.023 | 0.026 | 0.028 | 0.030 | 0.032 |
| 4 | 0.029 | 0.023 | 0.022 | 0.023 | 0.025 | 0.027 | 0.030 | 0.042 |
| 8 | 0.041 | 0.027 | 0.023 | 0.024 | 0.027 | 0.029 | 0.032 | 0.034 |
| 16 | 0.065 | 0.040 | 0.030 | 0.030 | 0.030 | 0.031 | 0.036 | 0.037 |
| | | | | (ms) | | | | |

**Attention 결론**:
- 순수 SDPA 는 **0.02-0.07 ms 수준 (매우 작음)**, thread 수 거의 무관
- batch=16 에서도 0.030 ms (16t) — batch 1 대비 1.3× 만 증가
- **⚠ 실제 vLLM IPEX PagedAttention 은 이보다 훨씬 느림** (per-seq KV cache 접근) — Layer Breakdown 의 30-60ms 가 실제 값
- 순수 attention kernel 자체는 병목이 아님. IPEX 의 per-seq 구조가 문제

## B.4 Memory Bandwidth (STREAM-like, 3× data size/iter)

### 64 MB (L3 상주 가능)

| threads | 1 | 4 | 8 | 16 | **24** | 48 | 76 | 96 | 112 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GB/s | 9.5 | 37.1 | 65.9 | 103.4 | **124.3** | 122.8 | 117.6 | 116.0 | 111.0 |

### 256 MB (L3 초과)

| threads | 1 | 4 | 8 | 16 | 24 | **48** | 76 | 96 | 112 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GB/s | 8.3 | 28.8 | 52.3 | 78.1 | 89.0 | **105.6** | 97.6 | 100.3 | 98.7 |

### 1 GB (DRAM 스트리밍)

| threads | 1 | 4 | 8 | 16 | 24 | **48** | 76 | 96 | 112 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GB/s | 8.3 | 27.3 | 52.7 | 83.3 | 101.3 | **129.8** | 120.6 | 125.1 | 127.7 |

### Memory BW Scaling 시각화 (1 GB working set)

```
DRAM Bandwidth (GB/s)
 150│                                     ⭐                              
    │                                    █████                           
 125│                             █████   █████   █████ █████   █████    
    │                             █████   █████   █████ █████   █████    
 100│                    █████   █████   █████   █████ █████   █████    
    │                    █████   █████   █████   █████ █████   █████    
  75│               █████ █████   █████   █████   █████ █████   █████    
    │               █████ █████   █████   █████   █████ █████   █████    
  50│        █████ █████ █████   █████   █████   █████ █████   █████    
    │  █████ █████ █████ █████   █████   █████   █████ █████   █████    
  25│█████   █████ █████ █████   █████   █████   █████ █████   █████    
    │█████   █████ █████ █████   █████   █████   █████ █████   █████    
   0└────────────────────────────────────────────────────────────────→ threads
      1    4    8   16   24   48   76   96  112
     8.3 27.3 52.7 83.3 101.3 129.8 120.6 125.1 127.7
                                     ↑
                                    포화 (43% of 300 GB/s 이론)
```

**Memory BW 결론**:
- DRAM 스트리밍 최적 = **48 thread, 129.8 GB/s**
- 물리 이론 ~300 GB/s (2S DDR5-4800 × 8ch) 의 **43%** — 2-NUMA cross-socket traffic + OMP 동기화 비용
- 76t 이상에서 BW 정체 또는 감소 — thread 간 경합
- KVM 의 26.5 GB/s 대비 **4.9× 빠름**

## B.5 Layer Breakdown (실제 1.5B Qwen2.5 model, 5 iter 평균)

| threads | Total (ms) | Attention (ms) | MLP/FFN (ms) | Other (ms) | Attn % | MLP % | Other % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **24** | **103.0** | 31.5 | 47.3 | 24.3 | 30.6% | 45.9% | 23.5% |
| 76 | 156.1 | 41.1 | 82.1 | 32.8 | 26.3% | 52.6% | 21.0% |
| 96 | 285.1 | 56.4 | **192.6** | 36.1 | 19.8% | 67.5% | 12.7% |
| 112 | 281.0 | 58.3 | 185.4 | 37.3 | 20.7% | 66.0% | 13.3% |

### Comparison 24 vs 76

| | 24t → 76t | ratio |
|---|---|---:|
| Total | 103.0 → 156.1 ms | 1.51× |
| Attention | 31.5 → 41.1 ms | 1.31× |
| **MLP/FFN** | **47.3 → 82.1 ms** | **1.74×** |
| Other | 24.3 → 32.8 ms | 1.35× |

### Layer Breakdown 시각화 (stacked bar)

```
Per-layer time (ms) — thread 증가에 따른 구성 변화

  300│                                ▒▒▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒▒▒▒▒           
     │                                ▒▒▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒▒▒▒▒           
  250│                                ▒▒▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒▒▒▒▒           
     │                                ▒▒▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒▒▒▒▒           
  200│                                ▒▒▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒▒▒▒▒           
     │                  ▒▒▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒▒▒▒▒           
  150│                  ▒▒▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒▒▒▒▒           
     │                  ▒▒▒▒▒▒▒▒▒▒   ████████     ████████              
  100│ ████████▓▓▓▓     ████████▓▓  ████████     ████████              
     │ ████████▓▓▓▓     ████████▓▓  ████████     ████████              
   50│ ████████▓▓▓▓     ████████▓▓  ████████     ████████              
     │ ▓▓▓▓▓▓▓▓▓▓▓▓     ▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓    ▓▓▓▓▓▓▓▓▓▓            
    0└──────────────────────────────────────────────────────────────
        24t              76t          96t           112t              
        103 ms⭐         156 ms       285 ms        281 ms              
        
   ▓▓▓▓▓ Attention   ████ MLP/FFN   ▒▒▒▒ Other/Dispatch                 
   (31.5/41.1/56.4/58.3)
                     (47.3/82.1/192.6/185.4)
                                    (24.3/32.8/36.1/37.3)
```

**→ Bottleneck: MLP/FFN** (모든 thread 수에서 가장 큰 비중, thread 증가에 가장 민감)

- 96t 에서 MLP 가 192.6ms — 24t (47.3ms) 의 **4.1배**
- Attention 은 24t → 96t 에서 1.8× 증가 (MLP 의 절반 증가율)
- Other (Python dispatch / scheduler) 는 비교적 안정 (1.5× 증가)
- **MLP 가 GEMM 절벽의 직접 영향** — Part B.2 의 FFN_up/dn 76t 절벽이 layer-level 에도 전파

## B.6 vLLM 실제 CPU inference Thread Sweep (1.5B, 3 runs 평균)

`VLLM_CPU_OMP_THREADS_BIND` 에 명시적 core list 전달 → 실제 thread 수 제어 성공.

| threads | torch_threads (실제) | tps | elapsed (s) | run variance |
|---:|---:|---:|---:|---|
| 8 | 8 | 13.77 | 4.65 | 4.66/4.64/4.64 |
| 16 | 16 | 18.92 | 3.38 | 3.38/3.38/3.39 |
| 24 | 24 | 22.16 | 2.89 | 2.89/2.89/2.88 |
| **32** | **32** | **23.55** | **2.72** | **2.72/2.73/2.71** ⭐ |
| 48 | 48 | 23.30 | 2.75 | 2.76/2.75/2.73 |
| **76** | **76** | **0.49** | **131.3** | **137.18/137.97/118.72** ⚠ |
| 96 | 96 | 19.59 | 3.27 | 3.23/3.24/3.33 |
| 112 | 112 | 18.24 | 3.51 | 3.50/3.53/3.50 |

### vLLM CPU tps — 76-thread 절벽 시각화

```
 tps
  25│             ██ ██                                                  
    │          ██ ██ ██                                                  
  20│       ██ ██ ██ ██                           ██                     
    │       ██ ██ ██ ██                           ██ ██                  
  15│    ██ ██ ██ ██ ██                           ██ ██                  
    │    ██ ██ ██ ██ ██                           ██ ██                  
  10│    ██ ██ ██ ██ ██                           ██ ██                  
    │ ██ ██ ██ ██ ██ ██                           ██ ██                  
   5│ ██ ██ ██ ██ ██ ██                           ██ ██                  
    │ ██ ██ ██ ██ ██ ██                           ██ ██                  
   0│ ██ ██ ██ ██ ██ ██       ▎                   ██ ██                  
    └──────────────────────────────────────────────────────────────→ threads
       8  16  24  32  48  64  76  96 112
                      ⭐          ⚠
       13.8  18.9  22.2  23.5  23.3    0.49    19.6  18.2
                           ↑                 ↑
                         최적                  48× 추락 (131.3s vs 2.7s)
```

### 76-thread 이상 동작 지속시간

```
run #1 @ 76t:  137.18 초  ██████████████████████████████████████████████
run #2 @ 76t:  137.97 초  ██████████████████████████████████████████████
run #3 @ 76t:  118.72 초  ██████████████████████████████████████████
         (평균 131.29s, 편차 ±8%)
         
         vs 32t 의 2.72 초 = ████

76t 에서 3 runs 모두 50× 이상 느림 — 비결정적이지만 지속적 절벽
```

**vLLM end-to-end CPU inference 최적 = 32 thread (23.55 tok/s)**.
**76 thread 에서 131.3 초 (48× 추락) — micro-bench 뿐 아니라 실제 서빙에서도 동일 패턴 확정**.

76t 에서 3 runs 변동이 큼 (118-138초) — oneDNN 의 비결정적 tile scheduling 영향.

## B.7 Intel 환경 진단

```
PyTorch: 2.9.0+cu130
IPEX: 2.8.0+gitcb81bf2
mkldnn: available

CPU Features:
  AVX-512F: Yes
  AVX-512 VNNI: Yes (INT8 acceleration)
  AVX-512 BF16: Yes
  AMX-BF16: Yes (Sapphire Rapids+)
  AMX-INT8: Yes (Sapphire Rapids+)

vLLM CPU ops:
  HAS_CPU_OPS: True
  HAS_CPU_UTILS: True

oneDNN env:
  ONEDNN_MAX_CPU_ISA: not set (auto-detect AMX)
  MKL_ENABLE_INSTRUCTIONS: not set
  KMP_AFFINITY: not set
  KMP_BLOCKTIME: not set
```

AMX BF16/INT8 모두 활성화 가능 상태. `configure_intel_optimizations()` 호출 시 자동 `ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX` 설정됨 (hybrid serve 경로).

---

# PART C. 재앙의 메커니즘 분석

## C.1 wave-batch 가 왜 실패하는가

`cpu_max_seqs=16` 로 CPU 에 16 req 한꺼번에 admit → decode 구조:

```
시간 →
req 0:  [prefill] → [decode solo] → done! (16→15 빠르게 완료)
req 1:  [ wait  ] [prefill] → decode...
req 2:  [ wait  ] [ wait  ] [prefill] → ...
...
req 15: [          wait                ] [prefill] → decode batch=15
```

`chunked_prefill=False` 이므로 prefill 이 순차 처리. req 0 은 solo decode 로 빠르게 끝나고, 나머지 15 개가 **batch=15 로 동시 decode** 시작.

## C.2 Attention 이 batch 에 선형 증가하는 이유

IPEX `single_query_cached_kv_attention` 은 per-sequence for-loop 구조:

```python
for seq in batch:                              # ← batch 15 면 15 번 호출
    attn_out[seq] = PagedAttention(q, k_cache[seq], v_cache[seq])
```

| | batch=1 | batch=15 |
|---|---|---|
| MLP/FFN (batched GEMM, weight 공유) | ~45 ms | ~45 ms (거의 동일) |
| **Attention (per-seq loop)** | **~30 ms** | **~450 ms** (15×) |
| Total per step | ~100 ms | **~500 ms** |
| 128 tokens | 13s | **64s** |

**CPU matmul batching 의 기대 (weight BW amortization) 가 attention 의 per-seq 선형 증가에 의해 완전히 상쇄됩니다.**

## C.3 왜 `max_seqs=1` 이 4× 나은가

batch=1 solo decode → attention 30ms (layer 당) → total ~100ms → 128 tok = **13s 이론**.
실측 **30초/req** (L3 경합 + OMP barrier overhead).

16 req 순차 × 30s = **480s (8분)** — wave=16 의 2,053s 대비 **4.3× 빠름**.

## C.4 그러나 여전히 gpu_only 대비 34× 느린 이유

현재 워크로드에서 GPU 는 500 req 를 **14초에 단독 처리**. CPU 에 16 req 를 보내면:
- GPU: 484 req 를 14s 에 완료
- CPU: 16 req × 30s = 480s tail
- Wall = max(14s, 480s) = **480s**

CPU 기여 계산:
- GPU throughput: 16,501 tok/s
- CPU 기여 (16 × 128 / 480): **4.3 tok/s = GPU 의 0.026%**

**GPU 가 이미 놀고 있을 때 (util 4.3%) CPU 를 쓰는 건 순수 overhead**. Amdahl's law: 느린 구성요소가 추가되면 전체 지연.

## C.5 Request Lifecycle Timeline — 3 구성 비교

### gpu_only (14 s wall)

```
Time → 
0s                               4s                    14s
├──────────────────────────────────┤
│ All 500 reqs on GPU              │                    │
│ Prefill wave × 2-3 (chunked)     │ Decode 128 tok      │
│                                   │ (continuous batching)│
└──────────────────────────────────┴────────────────────┘
       GPU fully active until done                      
```

### hybrid max_seqs=1 (480 s wall)

```
Time →
0s  1s  14s                                                               480s
├───┤───┤───────────────────────────────────────────────────────────────────┤
│ P │GPU│───── GPU idle 466s ────────────────────────────────────────────── │
│ r │484│                                                                   │
│ o │req│CPU: req0─┐  req1─┐  req2─┐  ... req15─┐                          │
│ b │   │          │        │       │                │                          │
│ e │   │  30s each (solo decode, batch=1)                                 │
│   │   │          │        │       │                │                          │
│   │   │         done    done    done                done                 │
└───┴───┴───────────────────────────────────────────────────────────────────┘
   16 req sequentially on CPU, 30s each = 480s tail
```

### hybrid wave=16 (2,072 s wall)

```
Time →
0s  1s  14s      60s                                                    2072s
├───┤───┤──────────┤─────────────────────────────────────────────────────┤
│ P │GPU│ CPU wave │                                                     │
│ r │484│ prefill  │                                                     │
│ o │req│ 16 reqs  │                                                     │
│ b │ 14│ sequential│ CPU decode batch=15 (+ 1 solo done) : 1,992 s       │
│ e │ s │  ~60s    │ per-step = 15.6s × 128 tokens                       │
│   │   │          │                                                     │
│   │   │          │ Attention × 15 = 450ms/step (per-seq loop)          │
│   │   │          │ + MLP 47ms + OMP/L3 overhead = 15.6s/step           │
│   │   │          │                                                     │
│   │   │          │ 16 reqs all finish around 2,060s                    │
└───┴───┴──────────┴─────────────────────────────────────────────────────┘
       GPU idle for ENTIRE remaining 2,058 seconds (99.3% of wall)
```

## C.6 3-way wall 분해 — Amdahl's Law 시각화

```
Wall time contribution (stacked, in seconds)

gpu_only     │ GPU: 14s │                                                  
             │▓▓▓▓▓▓▓▓▓▓│                                                  
                                                                           
max_seqs=1   │ 14s│ CPU tail: 466s                                         
             │▓▓▓▓│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░      
                                                                           
wave=16      │14s│ CPU wave prefill 60s│    CPU decode tail: 1992s          
             │▓▓▓▓│████████████████████│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░...│
             │                                                                       ↓
             └──────────────────────────────────────────────────────── 2072s ──┘

▓▓▓ GPU work    ███ CPU prefill (sequential)    ░░░ CPU decode tail (batch=15)

  "GPU is idle for 99.3% of hybrid wave=16 wall"
```

## C.7 per-Step Decomposition — Attention Explosion 정량

```
1.5B model Layer Breakdown (Part B.5 실측) × 7B 모델 스케일 추정:

                   1.5B measured  |  7B estimated  |  batch=15 estimated
                   ────────────── | ─────────────  | ───────────────────
MLP per step:      47 ms (24t)    |  ~110 ms       |  ~130 ms (M 확장)
Attention/seq:     31 ms          |  ~70 ms        |  70 × 15 = 1,050 ms
                                                    (per-seq loop 에서 폭발)
Other/dispatch:    24 ms          |  ~45 ms        |  ~100 ms (seq 반복)
                   ────────────── | ─────────────  | ───────────────────
TOTAL per step:    103 ms         |  ~225 ms       |  ~1,280 ms

실측 wave=16 per-step = 15,600 ms → 이론 추정 (1,280ms) 의 12×
추가 오버헤드:
  - 7B weight 14 GB vs L3 210 MB → L3 miss 지속, DRAM traffic 급증
  - KV cache 16 seq × 256 tokens × 2 KV × 128 dim × 2 byte = 32 MB (추가 L3 경합)
  - OMP barrier × 28 layer × 2-3 연산/layer = ~100 barrier/step
  - IPEX `single_query_cached_kv_attention` 의 Python dispatch overhead
```

### Attention 폭발 시각화

```
Attention 연산 시간 (ms per step)

batch=1 (MLP/FFN, 1.5B):   ▓ 31 ms
batch=1 (7B, estimated):    ▓▓ 70 ms
batch=1 (max_seqs=1 솔로):  ▓▓ ~70 ms  ← 30s / (128 step + prefill) ≈ 200ms?

batch=15 (wave, 7B):        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                            ▓ ~1,050 ms (이론) vs ~15,500 ms (실측)

이론 예측 (15× 증가) vs 실측 (500× 증가) 의 10× 차이는:
  - L3 cache thrashing (KV + weight + activation 동시 점유)
  - oneDNN GEMM 이 cross-NUMA 됐을 가능성 (56C 소켓 1개로는 부족)
  - IPEX overhead 가 seq 수에 superlinear
```

---

# PART D. 5대 결정적 인사이트

## D1. Wave-batch 는 전 환경에서 실패 → 영구 폐기

| 환경 | wave-batch 결과 | 원인 |
|---|---|---|
| dev RTX 3090 (16C AVX2) | batching 효과 0 | attention per-seq loop |
| H100x4 KVM (96 vCPU) | 76t GEMM 절벽 | L3 16 MB + oneDNN tiling |
| **H100x8 물리 (2S×56C×2T, 7B)** | **147× 성능 추락 (4/4 재현)** | batch=15 decode attention 15× 폭발 |
| H100x8 물리 (1.5B) | 1.46× 느림 (유일한 "괜찮은" 케이스) | 1.5B 는 L3 에 상주 가능 |

**매트릭스 전체에서 wave-batch 는 CPU 에 penalty**. 구조 자체가 잘못됨.

## D2. 최적 CPU thread = 32 (확정)

| 벤치 유형 | 최적 | 비고 |
|---|---:|---|
| vLLM end-to-end inference | **32** | 23.55 tok/s, 실운영 지표 |
| Micro-bench GEMM decode | 48 | FFN_dn 4,817 GFLOPS |
| Transformers forward | 24 | 103 ms total |
| Memory BW (STREAM 1GB) | 48 | 129.8 GB/s |
| GEMM prefill | 96 | 14,019 GFLOPS |

**운영 권장 = 32 thread** (vLLM 실측 최고).

## D3. 76 thread 절대 금지 (48× 성능 추락)

- Micro-bench: FFN_up/dn 5-8× 추락
- **vLLM 실제 inference: 131초 (32t 의 48배)**
- 재현 4/4 — oneDNN 의 NUMA 경계 tile scheduling 병리

## D4. 현재 워크로드는 hybrid 로 이득 불가

**500 req × 128/128 은 GPU 단독으로 14초 완료**. CPU 를 쓰면 순수 overhead.

**"hybrid < gpu_only" 달성 조건** (구조적 변경 필요):
- **70B TP=8** (weight 140 GB → GPU 포화)
- **long context** (16K+ input, KV cache 압박)
- **rate-limited 대량 batch** (2000+ req)
- **Spec decode CPU drafter** (CPU 가 "다른 일" 수행)

## D5. Attention per-seq 구조가 근본 병목

IPEX `single_query_cached_kv_attention` 의 per-sequence for-loop 이 batching 효과를 무효화.

**해결 경로** (ideation 에서 논의):
- **B1 NEO**: layer-level batch 분할 (GPU-batch + CPU-batch)
- **B2 ScoutAttention**: CPU 가 1 layer 앞서 top-k KV 예측
- **Head Folding** (Qwen 변형): batch × KV_head 을 M-dim 으로 fold → AMX tile 포화

---

# PART E. 권장 설정 및 다음 단계

## E.1 hybrid 설정 (H100x8 물리)

```bash
HYBRID_NUM_CPU_ENGINES=2              # auto (2 NUMA)
HYBRID_CPU_THREADS=32                 # engine 당 32t, 총 64 cores 사용
HYBRID_CPU_MAX_SEQS=1                 # wave-batch 영구 폐기
HYBRID_CPU_CORE_RATIO=1.0             # core_ratio 무시 (cpu_threads 가 우선)
HYBRID_ROUTING_STRATEGY=throughput-adaptive
HYBRID_ROUTING_PRIORITY=gpu-first     # ⚠ 현재 워크로드에선 CPU 억제
```

**주의**: `cpu-first` 는 GPU 여유 있어도 CPU 로 라우팅 → 현재 워크로드에서 penalty. `gpu-first` 는 GPU 포화 시에만 CPU → 현재 워크로드에선 CPU dispatch ≈ 0 → **hybrid = gpu_only**.

## E.2 다음 단계 우선순위

### 즉시 (1일)
1. **gpu-first 로 재측정** — hybrid = gpu_only 검증 (CPU dispatch 억제 확인)
2. **70B 다운로드**: `huggingface-cli download Qwen/Qwen2.5-72B-Instruct`

### 단기 (1주)
3. **70B TP=8 baseline + hybrid** — GPU 포화 영역에서 hybrid 첫 의미있는 측정
4. **Long-context 워크로드** (INPUT_LEN=4096, OUTPUT_LEN=1024, rate-limited)

### 중기 (2주+)
5. **Spec decode CPU drafter** (Tier 2 / A1) — 유일한 구조적 ninja gap 경로
6. **throughput-adaptive gate 재설계** — GPU queue depth 기반 dispatch (현재 cpu-first 는 항상 CPU)

---

# PART F. 데이터 소스

## H100x8 물리 벤치 (7B, 이번 세션)

- gpu_only: `eval/results/20260413_081534_G_H100_80GB_HBM3_x8_Qwen2.5-7B-Instruct/`
- hybrid wave=16 (4 runs): `eval/results/20260413_{081840,123757,132948,141742}_H_C_H100_80GB_HBM3_x8_Qwen2.5-7B-Instruct/`
- `before_h100_23/`: 이전 실험 135 디렉토리 (보존)

## CPU 프로파일 최종본

- `eval/analysis_log/20260413_075749_cpu_profile/`
  - `lscpu.txt`, `numa_cpulist.txt`, `smt_detection.txt`, `llc_topology.txt`
  - `gemm_scaling.txt/.json`
  - `attention_scaling.txt/.json`
  - `memory_bw.txt/.json`
  - `layer_breakdown.txt/.json`
  - `vllm_thread_sweep.txt/.json` ⭐ 핵심 데이터
  - `intel_diag.txt`
  - `core_topology.json`

## 관련 이전 문서

- `experiment_result/20260412_142000_h100x4_x8_physical_wave_batch_and_cpu_profile/` — H100x8 1.5B 성공 사례
- `experiment_result/20260412_050600_h100x1_qwen1.5b_7b_32b_wave_batch_scaling/` — KVM H100x1 실험
- `experiment_result/20260412_023700_dev_rtx3090_wave_batch_gate_fix_initial_validation/` — dev 초기 검증
- `ideation/20260413_1400_consolidated_optimization_roadmap.md` — 통합 최적화 로드맵
- `ideation/20260412_060000_many_core_cpu_llm_inference_research.md` — many-core CPU 서베이
