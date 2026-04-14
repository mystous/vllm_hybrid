# H100x8 물리 — 7B wave-batch 실패 원인 재분석 + CPU 프로파일

**날짜**: 2026-04-14 (재작성)
**환경**: H100x8 물리 (violet-h100-023)

## ⚠ 이전 버전 정정 사항

이전 초안의 주요 오류를 다음과 같이 정정합니다:

| 오류 | 정정 |
|---|---|
| "wave 가 단일 engine 에 16 req 몰아서 batch=15 decode" | 실제로는 `num_cpu_engines=2` 이면 **2 engine 에 alternating 분배** 의도이지만, **버그로 `hybrid_num_cpu_engines` argparse default=1 이 적용**되어 engine 1 개만 spawn 된 것으로 추정 (2026-04-14 fix 이전). 정확한 dispatch 는 서버 로그 필요 |
| "attention 이 per-seq for-loop 로 batch 에 선형 증가" | **틀린 분석**. IPEX `single_query_cached_kv_attention` 은 파이썬 레벨에서 `query.shape[0]=num_seqs` 전체를 C++ 커널에 batched 로 전달. Python for-loop 없음. 실제 커널 내부 동작은 IPEX 소스 확인 필요 |
| "profile 의 76t 절벽이 hybrid 에 적용" | profile 은 standalone `torch.set_num_threads(76)` 테스트. 실제 hybrid 엔진은 NUMA 에 bind 되어 56 cores 상한. **76t cliff 는 hybrid 에서 발생 불가** |
| "현재 hybrid 가 96 cores 만 사용 중" | `HYBRID_CPU_THREADS=0` (auto) 이면 각 engine 이 자기 NUMA physical cores (56) 전부 사용. 단 num_cpu_engines 가 1 로 resolve 되면 engine 1 개 × 56t = 56 cores 만 사용 |
| "max_seqs=16 이 batch 크기 문제" | batch 크기 자체는 MLP/attention 모두 IPEX/oneDNN 내부에서 batched 처리. **실제 느린 원인은 아직 확정 불가** (IPEX kernel 내부 동작 / KV cache paged access / L3 contention 중 어느 것인지) |

## 확정된 버그 (별도 commit 으로 수정)

**`vllm/engine/arg_utils.py:1048`** — `--hybrid-num-cpu-engines` argparse `default=1` → `default=0` 로 수정 (auto sentinel). 이전: env 에서 `HYBRID_NUM_CPU_ENGINES=0` (auto) 설정해도 argparse 가 1 을 덮어씌워서 NUMA auto-detect 가 스킵됐음.

**`eval/serve.sh`** — `HYBRID_NUM_CPU_ENGINES` 가 env 에 설정되면 값이 무엇이든 CLI 로 전달하도록 조건 수정 (`-gt 1` → `-n`).

---

# PART A. 측정 결과

## A.1 설정

- Hardware: Xeon Platinum 8480+ 2S × 56C × 2T = 224 logical, 2 NUMA, L3 105 MB × 2 socket, NUMA mem 1 TB × 2
- Model: Qwen2.5-7B-Instruct, TP=4 (H100 x 4 active), BF16
- Workload: 500 req × 128 in / 128 out, burst (rate=inf)
- Code revision: `50b3bc035` 이후, wave-batch + cold-start probe-to-GPU gate 포함

### 실행된 hybrid_config (5 runs 모두 동일, system_info.json 기준)

```json
{
  "routing_strategy": "wave-batch",
  "routing_priority": "cpu-first",
  "cpu_max_seqs": "16",
  "cpu_kvcache_gb": "0",
  "cpu_threads": "0",
  "numa_aware": "true",
  "num_cpu_engines": "0",         // auto → NUMAAllocator.num_nodes = 2
  "stats_log_interval": "25"
}
```

### 실제 resolve 결과 (코드 `_resolve_num_cpu_engines` 기준)

- `num_cpu_engines = 2` (NUMA 2 노드)
- `cpu_num_threads = 56` (`_resolve_cpu_params` 가 `numa_node_cores` 반환, physical/NUMA)
- 각 CPU engine process 가 자기 NUMA 에 bind, 56 threads 로 동작
- Router 의 총 CPU 용량: `cpu_max_num_seqs × num_cpu_engines = 16 × 2 = 32` slots

## A.2 5 runs 전체 결과

| Run | Mode | Duration (s) | Wall (s) | Output TP | Req TP | TPOT med | TPOT mean | TPOT P99 | TTFT med | TTFT mean | TTFT P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 081534 | **gpu_only** | 3.77 | 14.01 | 16,501 | 132.48 | 22.74 | 23.12 | 55.83 | 799 | 815 | 1,075 |
| 081840 | hybrid #1 | 2,062 | 2,072 | 30 | 0.24 | 23.40 | 532.57 | 15,691 | 824 | 2,938 | 69,433 |
| 123757 | hybrid #2 | 1,977 | 1,987 | 32 | 0.25 | 22.56 | 502.34 | 15,012 | 757 | 2,875 | 70,066 |
| 132948 | hybrid #3 | 2,085 | 2,095 | 30 | 0.24 | 21.91 | 536.98 | 15,987 | 834 | 2,501 | 54,651 |
| 141742 | hybrid #4 | 2,046 | 2,057 | 30 | 0.24 | 22.09 | 522.60 | 15,588 | 736 | 2,771 | 66,804 |

### Wall Time 비교 (log-scale)

```
gpu_only      │██ 14 s
max_seqs=1    │███████████████████████████████████ 480 s (사용자 실측)
wave=16 #1    │█████████████████████████████████████████████████████████████ 2,072 s
wave=16 #2    │██████████████████████████████████████████████████████████ 1,987 s
wave=16 #3    │████████████████████████████████████████████████████████████████ 2,095 s
wave=16 #4    │███████████████████████████████████████████████████████████████ 2,057 s
              └──────────────────────────────────────────────────────────────────────
              0           500         1000        1500        2000       (seconds)
```

평균 wave=16 wall = **2,053 ± 40 s** (1.94% 편차, 4회 재현).

## A.3 TPOT 분포가 bimodal 인 이유 — 어느 routing 이 일어났는가

**코드 기반 routing 흐름** (`_route_wave_batch`, line 464-506):

```
req 1   : cold-start gate (line 486: gpu_count+cpu_count == 0) → GPU
req 2+  : _find_wave_open_cpu 로 가장 accepted 수 적은 engine 선택
```

`_find_wave_open_cpu` (line 451-461) 의 **strict `<` 비교** 시뮬레이션:

```
초기: engine0=(0, open), engine1=(0, open)

req 2: i=0 (0<17 → best=0, best_idx=0); i=1 (0<0 False) → pick engine0 → (1,0)
req 3: i=0 (1<17 → best=1, idx=0); i=1 (0<1 True → best=0, idx=1) → engine1 → (1,1)
req 4: i=0 (1<17 → best=1, idx=0); i=1 (1<1 False) → engine0 → (2,1)
req 5: i=0 (2<17 → best=2, idx=0); i=1 (1<2 True → best=1, idx=1) → engine1 → (2,2)
...
```

**alternating pattern**: 2 req 당 (k,k) → k+1 씩 증가. 16 CPU req 후 각 engine 에 8.

계속 alternating 으로:
```
CPU req 1~16:  각 engine 에 8 씩 (아직 open, max=16)
CPU req 17~32: 각 engine 이 accepted=16 도달 → wave closed
CPU req 33+:   둘 다 closed → _find_wave_open_cpu 가 -1 반환 → GPU 로
```

즉 **최대 32 req 까지 CPU 로 dispatch 가능** (각 engine 16 씩).

### 실제 dispatch 개수 — TPOT 역산

```
500 × mean_TPOT = N_GPU × 22.74 + N_CPU × TPOT_CPU
```

**Case A: CPU = 16 req**
```
500 × 532.57 = 484 × 22.74 + 16 × X
X = (266,285 - 11,006) / 16 = 15,955 ms ≈ P99 TPOT (15,691)  ← ✓
```

**Case B: CPU = 32 req**
```
500 × 532.57 = 468 × 22.74 + 32 × X
X = (266,285 - 10,642) / 32 = 7,989 ms  ← P99 TPOT (15,691) 과 불일치 ✗
```

**→ 실제 CPU dispatch = 16 req** (Case A 가 P99 값과 일치)

### 왜 32 가 아니고 16 인가 (가설)

bench 의 500 req 가 **매우 빠르게 burst 도착** → CPU engine 0 가 16 accept 하는 동안 engine 1 에도 간간이 dispatch. 하지만 **engine 0 가 먼저 accepted=16 에 도달하여 close → engine 1 도 곧 close 또는 GPU 로 fallback**. 16 vs 32 의 정확한 구분은 서버 측 `[HYBRID-WAVE]` 로그 필요 (현재 결과 디렉토리에는 없음).

**중요**: bench.log 는 client 측만 저장됨. 서버 측 stats (`[HYBRID-ROUTER-STATS]` + `[HYBRID-WAVE]` markers) 는 이번 run 에 수집되지 않아 정확한 per-engine dispatch 개수는 미확정.

### TPOT Bimodal Distribution (Run #1, CPU=16 가정)

```
Count ↑
484 req │██████████           (GPU, TPOT ~22 ms cluster)
        │██████████
        │
        │
        │
        │
 16 req │                                            ██ (CPU, TPOT ~15,955 ms)
        │                                            ██
        └─────────────────────────────────────────────→ TPOT (ms)
         0    22   100  500  1,000  5,000  10,000  15,000
```

## A.4 TTFT 왜곡 — CPU engine 내 prefill 순차화

`_create_cpu_vllm_config` (line 1124 근처):
```python
cpu_sched.enable_chunked_prefill = False
cpu_sched.chunked_prefill_enabled = False
```

CPU engine 은 **한 번에 1 req 만 prefill**. 각 engine 이 최대 16 req 받으면:

```
Engine 0 prefill schedule (각 prefill ~3s 가정):
req 1st:  [P] → decode ...
req 2nd:      [P] → decode ...
...
req 16th:                                  [P] → decode ...
          0s  3s  6s  ...              45s 48s
```

16번째 req 의 TTFT = 앞선 15 req 의 prefill 대기 ≈ 45s ~ 69s (실측 P99 = 69,433 ms).

**Note**: 2 engines 병렬 prefill 이므로 **전체 TTFT worst = 각 engine 내 15번째 위치 prefill 대기** (32 개가 단일 queue 가 아님).

## A.5 CPU/GPU Utilization Timeline

모니터 CSV 기준 (1 Hz sampling):

| Run | samples | GPU avg | GPU busy (>10%) | CPU avg | CPU busy duration |
|---|---:|---:|---:|---:|---:|
| gpu_only | 11 | 4.3% | 2 samples | 3.1% | — |
| hybrid #1 | 1,777 | 0.0% | 2 samples | 18.9% | 2,060 s (10s~2,070s) |
| hybrid #2 | 1,704 | 0.1% | 3 samples | 19.3% | 1,982 s (3s~1,985s) |
| hybrid #3 | 1,796 | 0.0% | 2 samples | 18.9% | 2,083 s (10s~2,093s) |
| hybrid #4 | 1,764 | 0.0% | 2 samples | 19.1% | 2,045 s (10s~2,055s) |

### Timeline (Run #1)

```
GPU utilization over 2,072 s
100%│▐█                                                                           
 75%│▐█                                                                           
 50%│▐█                                                                           
 25%│▐█                                                                           
  0%│▐█▐░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 
    └──────────────────────────────────────────────────────────────────────────→ 
     0  14s                                                                 2072s
     │GPU busy│──────── GPU idle ≈ 2,058 s (99.3%) ────────────────────────────│

CPU utilization (all 224 logical cores)
100%│                                                                           
 75%│                                                                           
 50%│                                                                           
 25%│  ▌██████████████████████████████████████████████████████████████████      
  0%│▉▋                                                                       ▋ 
    └──────────────────────────────────────────────────────────────────────────→ 
     0s 10s                                                          2070s  2072s
        │          CPU busy ~2,060 s @ mean 19% util                 │

19% of 224 logical cores ≈ 43 cores @ 100% equivalent.
실제: 2 engines × 56 thread pin = 112 logical (50% 에 가까움)
CPU util 19% 는 "사용된 thread 들이 BW-bound 로 ~40% 이용률" 을 의미
(thread 는 pin 되어 있지만 DRAM 기다리는 시간이 많아 전체 평균 낮음)
```

---

# PART B. wave-batch decode 가 왜 2,053 s 걸렸는가 — 코드 기반 분석

## B.1 두 engine 병렬 실행 구조

```
         GPU (TP=4) ────────────┐
                                │
                                │  468~484 req (bench 도착 순서대로)
                                │
                                ▼
         ┌──────────────────────────────────────┐
         │ _route_wave_batch (hybrid_core.py)   │
         │                                      │
         │ ┌────────────────────┐               │
         │ │ Cold-start gate    │ → probe GPU    │
         │ │ cpu_count==0 ?     │                │
         │ └────────────────────┘                │
         │         │ No                          │
         │         ▼                             │
         │ _find_wave_open_cpu()                 │
         │  alternating to engine 0/1           │
         │         │                             │
         │         ▼                             │
         └─────────┼─────────────────────────────┘
                   │
           ┌───────┴───────┐
           ▼               ▼
    CPU engine 0     CPU engine 1
    (NUMA 0)         (NUMA 1)
    56 cores         56 cores
    max 16 req       max 16 req
    
    batch decode     batch decode
    (attention per-seq loop inside each engine)
```

두 engine 은 **독립 process**, **독립 NUMA**, **독립 DRAM 채널**. 병렬 실행. 각 engine 내부는 `chunked_prefill=False` + `max_num_seqs=16` 로 단일 scheduler 가 batch decode.

## B.2 각 engine 내부 — 왜 느린가

### 프로파일 실측 (Part C.1 참조) 에서

1.5B Qwen2.5, 24 threads, transformers forward (batch=1 equivalent):
```
Total per step: 103 ms
Attention:       31.5 ms  (30.6%)
MLP/FFN:         47.3 ms  (45.9%)
Other:           24.3 ms  (23.5%)
```

56 threads 에서는 (76 보간):
```
Total: 추정 ~140 ms (1.5B, batch=1)
```

7B 는 weight ~4.7× → per step 추정 **400-500 ms** (batch=1, 56 threads).

### Batch decode 에서 attention — Python 레벨은 batched 호출

`cpu_attn.py::_IPEXPagedAttention.forward_decode` (line 1234-1264) 실제 코드:

```python
# query.shape = [num_seqs, num_heads*head_size]
# context_lens.shape = [num_seqs]
ipex_modules.PagedAttention.single_query_cached_kv_attention(
    output, query.contiguous(),      # ← num_seqs 전체를 한 번에 전달
    key_cache, value_cache, head_mapping,
    scale, block_tables, context_lens,  # ← context_lens 도 num_seqs 전체
    block_size, max_context_len, alibi_slopes)
```

**Python 레벨에서 for-loop 없음**. IPEX C++ 커널에 batch 전체를 한 번에 전달. 이전 분석 ("16번 호출") 은 **소스 코드 확인 없이 추측** 한 오류.

**⚠ 미확정**: IPEX C++ 커널 (`ipex_modules.PagedAttention.single_query_cached_kv_attention`) 내부가 실제로 batched 병렬 처리인지 아니면 내부에서 for-loop 인지 **IPEX 소스 코드 확인 필요**. 이 섹션의 원래 목적 ("batch=16 이 왜 느린가") 에 대한 답은 **현재 확정 불가**.

### 실측만 가능한 사실

```
per step batch=N decode 실측 (2,053s / 128 tokens):

wave=16 (사용자 실행):  per-step ≈ 16 s (이론 batch=1 의 ~32×, 원인 미확정)
max_seqs=1 (사용자 실측): per-step ≈ 3.75 s (480s / 128, 2 engines 병렬 가정)
```

이론/예상 per-step time 은 **IPEX 커널 내부 확인 전까지 정확 계산 불가**. 이전 보고서의 "MLP 500ms + Attention 16×200ms = 3,800ms" 계산은 per-seq loop 가정 기반이라 무효.

### 실측 (각 engine 2,053 s ≒ CPU busy duration)

```
실측 per step: 2,053 s / 128 tokens = ~16 s/step
이론 대비: 16,000 / 3,800 = 4.2× 느림

추가 손실 원인 (확정 불가, 가설):
- L3 경합: 각 engine 의 16 seq × KV cache + 7B weight 7 GB 가 NUMA-local L3 105 MB 에 들어가지 않음
- OMP barrier 비용: 56 threads × 28 layer × 여러 sync = barrier overhead
- IPEX `single_query_cached_kv_attention` 의 Python dispatch overhead (seq 당 호출)
- DRAM 실효 BW 가 thread 56 에서 NUMA-local BW 포화 안 됨 (Part C.4 참조)
```

## B.3 max_seqs=1 이 4× 개선되는 이유

`max_seqs=1` 이면 각 engine 은 batch=1 (solo decode) 로만 동작:

```
per step: MLP 500 ms + Attention 200 ms × 1 + Other 50 ms ≈ 750 ms
128 tokens: ~96 s per req solo

16 req per engine sequential × 2 engines parallel:
16 × 96 s / 2 (engines) = 768 s
실측 사용자 보고 ~480 s (8 분)
```

실측 480s 가 추정 768s 보다 빠른 것은:
- 16 req 가 한꺼번에 engine 당 할당되지 않음 (router 는 항상 1 개만 dispatch)
- GPU 가 빠르게 처리해서 GPU 완료 시점 이후 CPU 에 남은 req 수가 16 보다 적음
- 실제 CPU 에 간 req 수가 8 정도일 가능성 (서버 로그 없어 확정 불가)

---

# PART C. CPU 프로파일 상세 (analysis_log/20260413_075749_cpu_profile)

## C.1 시스템 토폴로지

| 항목 | 값 |
|---|---|
| CPU | Intel Xeon Platinum 8480+ |
| Logical CPUs | 224 (0-223) |
| Physical cores | 112 (2S × 56C) |
| Threads/core | 2 (SMT 활성) |
| Sockets | 2 |
| NUMA nodes | 2 |
| NUMA node0 | CPUs 0-55, 112-167 (SMT sibling +112) |
| NUMA node1 | CPUs 56-111, 168-223 |
| L1d | 48 KB × 112 = 5.3 MiB |
| L1i | 32 KB × 112 = 3.5 MiB |
| L2 | 2 MiB × 112 = 224 MiB |
| L3 | 105 MB × 2 socket = 210 MB |
| DRAM | DDR5-4800, node0 1,031 GB / node1 1,032 GB |

## C.2 GEMM Scaling 실측 (standalone torch benchmark)

**주의**: 이 섹션은 `torch.set_num_threads(N)` 만 호출하는 standalone 테스트. NUMA binding 없음. 76+ thread 는 2 NUMA 경계를 넘어 실행됨.

실제 hybrid 엔진은 각 engine 이 자기 NUMA 에 bind 되어 56 threads 상한. **76/96/112 thread 데이터는 실제 hybrid 동작과 직접 매핑되지 않음** (standalone 참고용).

### decode_qkv [16×3584] × [3584×3584]

| threads | 4 | 8 | 12 | 16 | 24 | 32 | **48** | 56* | 64 | 76 | 96 | 112 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ms | 1.028 | 0.517 | 0.401 | 0.318 | 0.248 | 0.180 | **0.119** | (보간) | 0.137 | 0.140 | 1.138 | 1.723 |
| GFLOPS | 400 | 794 | 1,025 | 1,293 | 1,654 | 2,290 | **3,456** | ~3,000 | 3,009 | 2,946 | 361 | 239 |

### decode_ffn_up [16×3584] × [3584×9728]

| threads | 4 | 8 | 24 | 32 | **48** | 56* | 64 | 76† | 96† | 112† |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ms | 2.893 | 1.857 | 1.066 | 0.517 | **0.479** | ~0.48 | 0.489 | 2.154 | 1.796 | 1.778 |
| GFLOPS | 386 | 601 | 1,047 | 2,158 | **2,330** | ~2,300 | 2,283 | 518† | 621† | 628† |

† NUMA 경계 넘은 standalone 측정 — hybrid 에는 해당 없음

**48 threads = decode GEMM 최적**, NUMA 내부 (≤56) 에서는 안정적 scaling.

### decode_ffn_dn [16×9728] × [9728×3584]

| threads | 4 | 24 | 32 | **48** | 64 | 76† |
|---:|---:|---:|---:|---:|---:|---:|
| GFLOPS | 416 | 1,853 | 2,010 | **4,817** | 842 | 608† |

### prefill_128 [128×3584] × [3584×9728]

| threads | 4 | 24 | 48 | 64 | 76† | **96†** | 112† |
|---:|---:|---:|---:|---:|---:|---:|---:|
| GFLOPS | 2,124 | 6,380 | 9,477 | 11,783 | 13,563 | **14,019** | 5,240 |

Prefill 은 compute-bound (M=128 충분히 큼) → NUMA 경계 넘어도 96 threads 까지 scaling. 그러나 실제 hybrid 엔진은 NUMA-bound 이므로 **각 engine 최대 56 threads 안에서만 사용 가능**.

## C.3 Attention Scaling (순수 torch SDPA, 1.5B 스펙)

1.5B Qwen2.5: num_kv_heads=2, head_dim=64, seq_len=256.

| batch | threads=4 | 8 | 16 | 24 | 48 | 56* |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.021 | 0.021 | 0.023 | 0.023 | 0.026 | ~0.027 |
| 4 | 0.029 | 0.023 | 0.022 | 0.023 | 0.025 | ~0.026 |
| 8 | 0.041 | 0.027 | 0.023 | 0.024 | 0.027 | ~0.028 |
| 16 | 0.065 | 0.040 | 0.030 | 0.030 | 0.030 | ~0.031 |

**순수 torch SDPA 는 빠름 (0.02-0.07 ms), thread 수 거의 무관**.

**⚠ 그러나 vLLM 실제 경로는 IPEX PagedAttention**:
- `single_query_cached_kv_attention` 이 per-seq loop
- batch=16 → 16× 호출
- KV cache paged 접근 (L3/DRAM 에서 cold read)
- 실제 layer 내 attention 시간은 순수 SDPA 의 10-20×

## C.4 Memory BW (STREAM, DRAM 스트리밍)

| threads | 1 | 4 | 16 | 24 | **48** | 56* | 76† | 96† | 112† |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 MB | 9.5 | 37.1 | 103.4 | 124.3 | **122.8** | ~120 | 117.6 | 116.0 | 111.0 |
| 1 GB | 8.3 | 27.3 | 83.3 | 101.3 | **129.8** | ~125 | 120.6 | 125.1 | 127.7 |
|  |  |  |  |  | (GB/s) |  |  |  |  |

**48 threads 에서 NUMA-local BW 포화 (~129 GB/s)**.
이론치 물리 2S DDR5-4800 × 8ch ≈ 300 GB/s 의 **43%**.
- 2 NUMA cross-socket 트래픽 + OMP 동기화
- single NUMA 만으로는 이론 ~150 GB/s 의 87% 활용 → 합리적

실제 hybrid 에선 각 engine 이 NUMA-local DRAM 만 읽으므로 **engine 당 ~130 GB/s 이 BW 상한**.

## C.5 vLLM 실제 CPU inference Thread Sweep (1.5B, 3 runs 평균)

`VLLM_CPU_OMP_THREADS_BIND` 에 명시적 core list 전달. **주의: core list 가 0-111 범위 내이지만, 56 이상은 NUMA 1 를 포함**.

| threads | tps | elapsed (s) | 참고 |
|---:|---:|---:|---|
| 8 | 13.77 | 4.65 | NUMA 0 내부 |
| 16 | 18.92 | 3.38 | NUMA 0 내부 |
| 24 | 22.16 | 2.89 | NUMA 0 내부 |
| **32** | **23.55** | **2.72** | NUMA 0 내부 ⭐ |
| 48 | 23.30 | 2.75 | NUMA 0 내부 |
| 76 | **0.49** | **131.3** | **NUMA 경계 넘음** (56 + 20 on NUMA 1) |
| 96 | 19.59 | 3.27 | 양 NUMA 에 분산 |
| 112 | 18.24 | 3.51 | 양 NUMA balanced |

```
tps ↑
  25│          ███ ██                                                   
    │       ██ ███ ██                                  ██               
  20│       ██ ███ ██                                  ██ ██            
    │    ██ ██ ███ ██                                  ██ ██            
  15│    ██ ██ ███ ██                                  ██ ██            
    │ ██ ██ ██ ███ ██                                  ██ ██            
  10│ ██ ██ ██ ███ ██                                  ██ ██            
    │ ██ ██ ██ ███ ██                                  ██ ██            
   5│ ██ ██ ██ ███ ██                                  ██ ██            
    │ ██ ██ ██ ███ ██      ▎                           ██ ██            
   0└─────────────────────────────────────────────────────────────→ threads
       8  16  24  32  48   76†    96†   112†
                      ⭐   ↑ cross-NUMA artifact
                         0.49 tps (48× 추락)
                    profile standalone 한정, hybrid 에선 발생 안 함
```

### 의미

- **32-48 threads (NUMA 내부) 가 vLLM CPU inference 최적** (~23 tps)
- **76 thread cliff 는 NUMA 경계 넘은 standalone 아티팩트** — 실제 hybrid 엔진 (각 engine NUMA-bound, 최대 56 threads) 에선 발생 불가
- **실제 hybrid 엔진 engine 당 최대 56 threads 내에서 최적 설정 = 48 threads**

## C.6 Layer Breakdown (transformers forward, 1.5B)

batch=1, input_ids [1, 128], 5 iter 평균 (첫 iter 제외):

| threads | Total (ms) | Attention | MLP/FFN | Other |
|---:|---:|---:|---:|---:|
| **24** | **103.0** | 31.5 (30.6%) | 47.3 (45.9%) | 24.3 (23.5%) |
| 76† | 156.1 | 41.1 | 82.1 | 32.8 |
| 96† | 285.1 | 56.4 | 192.6 | 36.1 |
| 112† | 281.0 | 58.3 | 185.4 | 37.3 |

† NUMA 경계 넘은 standalone — hybrid 실제 동작 아님

**24 threads 에서 forward 101.3 ms**. 7B 추정 (weight × 4.7) = ~475 ms/step batch=1.

## C.7 Intel 환경

```
torch: 2.9.0+cu130 (CUDA build, but CPU ops 사용 가능)
IPEX: 2.8.0+gitcb81bf2
mkldnn available: True
AVX-512F, VNNI, BF16, AMX-BF16, AMX-INT8: all Yes

ONEDNN_MAX_CPU_ISA: not set (auto-detect → AMX)
HAS_CPU_OPS: True
HAS_CPU_UTILS: True
```

---

# PART D. 결론 (코드/실측 기반)

## D.1 IPEX 소스 분석 + profile 교차 검증 결과

### IPEX `single_query_cached_kv_attention` 구조 (확정)

- 디스패처 (`PagedAttentionKrnl.cpp:1846-2015`): `num_heads × batch > threads × 2` 이면 VNNI, 아니면 **Flash Decoding (FD)** 선택
- 7B (28 heads, kv_heads=4), batch=16, 56 threads: `28 × 16 > 56 × 2 = 448 > 112` → True, 하지만 `beam_size >= 4` 조건 불만족 → **FD kernel 사용**
- FD kernel 병렬화 (line 1230): `#pragma omp parallel for collapse(3)` 로 **seq × partition × head_group 3-D OMP** (per-seq loop 가 아님)

이전 보고서의 "per-seq loop 폭발" 주장 완전 철회. IPEX 는 batch 를 정상적으로 OMP 병렬 처리.

### Profile 교차 검증 (Section 6 vLLM thread sweep, 1.5B CPU)

- 32 threads @ 1.5B solo decode: 23.55 tps = **per step 42 ms**
- 7B 추정 (weight 4.67×): per step batch=1 ≈ **~200 ms**
- 56 threads 에서도 유사 (memory BW bound)

### 실측과 대조

| 구성 | 실측 wall | per-step (역산) | 이론 추정 (batch=1 기준) | 이론 대비 |
|---|---:|---:|---:|---:|
| max_seqs=1 (2 engines) | 480 s | **3.75 s** (480/128) | 200 ms | **19×** |
| wave=16 (1 engine?) | 2,053 s | **16 s** | 200 ms | **80×** |

max_seqs=1 도 이론 19× 느림 → **솔로 decode 자체가 profile 의 solo 값과 크게 안 맞음**. GPU burst 가 CPU 에 dispatch 를 늦추는 효과 포함됐을 가능성.

### 정확한 원인 (진단 가능 범위)

1. **arg_utils.py `default=1` 버그 (확정)**: wave=16 실행 시 `num_cpu_engines=1` 로 resolve 됐을 가능성 매우 높음 → 단일 engine 에 16 req 몰림 → batch=16 FD kernel 단일 NUMA 에서 실행
2. **IPEX FD kernel batch=16 → 80× 느려진 것의 원인 (확정 불가, 가설)**:
   - KV cache paged access pattern: 16 seq × 서로 다른 block_table → scatter DRAM access
   - context_len 증가에 따른 partition 수 증가
   - schedule(static, 1) 의 불균형 (seq 별 context_len 차이)
3. **max_seqs=1 이 나은 이유**: batch=1 → KV paged access 도 1 seq 분만 → cache friendly

### 확정 불가한 것 (추가 실험 필요)

- 실제 실행 시 num_cpu_engines 값 (서버 로그 `[HYBRID-LAUNCH] num_cpu_engines=...`)
- MLP 와 attention 의 분리 시간 (ONEDNN_VERBOSE + 구간별 timer)
- KV cache L3 miss ratio (perf stat hardware counters)

## D.2 max_seqs=1 이 4× 개선하는 이유 (미확정)

per-step batch=1 이 batch=16 보다 4.3× 빠른 현상은 실측이나, **근본 원인은 D.1 과 동일하게 미확정**. batch 가 작으면 무언가가 개선되는데, 그게 attention 인지 KV access 인지 L3 인지 소스 코드 확인 필요.

실측만 가능한 사실:
- max_seqs=1 wall 480s (사용자 실측)
- wave=16 wall 2,053s (4회 재현)
- 비율 4.3×

## D.3 그러나 gpu_only 대비 여전히 34× 느린 이유

현재 workload (500 req × 128 / 128) 에서 GPU 는 500 req 를 **14 s 에 단독 처리 가능**. CPU 에 req 를 보내는 것은 pure overhead:

- GPU output TP: 16,501 tok/s
- CPU 기여 (16 req × 128 tok / 480 s ≈ 4.3 tok/s) = **GPU 의 0.026%**

**Amdahl's law**: 느린 구성요소가 추가되면 전체 wall 증가. GPU 가 과잉 공급 상태에서 CPU 는 wall 을 늘리기만 함.

## D.4 "hybrid < gpu_only" 달성 조건

현재 구조 (request-level partition) 로는 **GPU 가 실제로 포화** 될 때만 가능:
- **70B TP=8** (weight 140 GB → GPU KV/batch 압박)
- **long context** (16K+ input)
- **대량 burst** (rate-limited 2,000+ req)

또는 **구조 변경** (Spec decode CPU drafter, NEO, ScoutAttention — ideation 로드맵 참조).

## D.5 CPU Profile 의 "76t 금지" 는 hybrid 에 해당하지 않음

이전 초안의 "76 thread 절대 금지" 주장은 정정:
- 76t cliff 는 **standalone torch** (NUMA 경계 넘음) 에서 발생
- 실제 hybrid 엔진은 각 engine 이 자기 NUMA 에 bind → 56 threads 상한
- **hybrid 에서는 76t 상황이 애초에 발생 불가**

실제 hybrid 엔진의 valid 설정 범위: **각 engine 4~56 threads 내**. profile 기반 최적은 48 threads per engine.

---

# PART E. 권장 설정 및 다음 단계

## E.1 현재 workload (500 × 128/128) 용

**workload 자체가 hybrid 로 이득이 없음**. GPU 단독으로 충분.

만약 실험 목적으로 hybrid 를 돌리려면:

```bash
HYBRID_NUM_CPU_ENGINES=0        # auto → 2 (2 NUMA)
HYBRID_CPU_THREADS=48           # engine 당 48 (총 96 cores, 시스템 여유)
HYBRID_CPU_MAX_SEQS=1           # attention 폭발 회피
HYBRID_ROUTING_STRATEGY=throughput-adaptive
HYBRID_ROUTING_PRIORITY=gpu-first  # GPU 포화 시에만 CPU
```

기대 wall: GPU single ~14 s + CPU 1 req tail ~30 s = **~45 s** (추정, 실측 필요).

## E.2 GPU 포화 workload 로 전환

```bash
# 70B 다운로드
huggingface-cli download Qwen/Qwen2.5-72B-Instruct

# env: TP=8, long context, 많은 req
TENSOR_PARALLEL_SIZE=8
EXTRA_SERVE_ARGS="--max-model-len 8192"
NUM_PROMPTS=2000
INPUT_LEN=2048
OUTPUT_LEN=512
REQUEST_RATE=50   # rate-limited
```

이 환경에서 GPU 가 queue 쌓이면 CPU 가 overflow 받아 실질 기여 가능.

## E.3 구조적 변경 (중기)

- Spec decode CPU drafter (A1, ideation Tier 2)
- NEO 비대칭 배치 분할 (B1)
- ScoutAttention layer-ahead (B2)

---

# PART F. 미확정 사항 (현재 데이터로 답 못함)

1. **정확한 CPU dispatch 개수** (16 vs 32) — TPOT 역산은 16 과 일치하지만 서버 `[HYBRID-WAVE]` 로그 없어 확정 불가
2. **각 engine 의 실제 batch 크기** — wave closed 시점, in_flight 변동 기록 없음
3. **per-step time 의 이론 대비 4× 추가 오버헤드 원인** — L3 경합/OMP barrier/IPEX dispatch/NUMA traffic 중 무엇이 지배인지 분리 측정 필요 (ONEDNN_VERBOSE + perf stat 로 구분 가능)
4. **max_seqs=1 실측 8 분의 CPU dispatch 분포** — 16 req 고정인지, fewer 인지

다음 실험 시 서버 stdout (`[HYBRID-ROUTER-STATS]`, `[HYBRID-WAVE]` markers) 을 수집하면 이 모두 확정 가능.

---

# PART G. 데이터 소스

**H100x8 물리 벤치 (이번 세션)**:
- gpu_only: `eval/results/20260413_081534_G_H100_80GB_HBM3_x8_Qwen2.5-7B-Instruct/`
- hybrid wave=16 × 4 runs: `eval/results/20260413_{081840,123757,132948,141742}_H_C_H100_80GB_HBM3_x8_Qwen2.5-7B-Instruct/`

**CPU 프로파일 최종본**:
- `eval/analysis_log/20260413_075749_cpu_profile/`

**코드 참조**:
- `vllm/v1/engine/hybrid_core.py::_find_wave_open_cpu` (line 440-462)
- `vllm/v1/engine/hybrid_core.py::_route_wave_batch` (line 464-506)
- `vllm/v1/engine/hybrid_core.py::_find_available_cpu` (line 303-312)
- `vllm/v1/engine/hybrid_core.py::_create_cpu_vllm_config` (chunked_prefill=False)
- `vllm/v1/attention/backends/cpu_attn.py::_IPEXPagedAttention.forward_decode` (per-seq loop)

**관련 문서**:
- `experiment_result/20260412_142000_h100x4_x8_physical_wave_batch_and_cpu_profile/` (H100x8 1.5B 케이스, 7B 와 대조)
- `ideation/20260413_1400_consolidated_optimization_roadmap.md`
