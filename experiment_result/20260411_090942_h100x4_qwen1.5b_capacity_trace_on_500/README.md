# H100x4 Qwen2.5-1.5B — GPU-only vs Hybrid (capacity + cpu-first, **TRACE=1**)

`20260411_090942_h100x4_qwen1.5b_capacity_trace_on_500`

> **목적**: `h100x4_qwen1.5b_hybrid_smoke.env` 를 그대로 사용해 GPU-only 와 hybrid 를
> 500-prompt 로 비교.
> **결과 한 줄**: hybrid wall = **7.6 × GPU-only**, output throughput = GPU-only 의 **7.4 %**.
> **원인**: env 의 `VLLM_HYBRID_TRACE=1` + `VLLM_HYBRID_TRACE_EVERY=1` 로 인한 stdout I/O
> 폭주가 API server 메인 thread 를 blocking 하여 GPU lane 까지 끌어내림. CPU 코드 자체의
> 성능 저하가 아니라 **로깅 오버헤드**가 본질.

---

## 1. 환경

| 항목 | 값 |
|---|---|
| Host | KVM guest, `hypervisor` flag |
| GPU | NVIDIA H100 80GB HBM3 × 4 |
| CPU | Intel Xeon Platinum 8480+ (1S × 96 vCPU, NUMA=1) |
| ISA | AVX-512 F/VNNI/BF16/FP16, AVX-VNNI, **AMX (bf16/int8/tile)** |
| RAM | 944 GB |
| vLLM | `019c7121a` (h100_cu13) |
| torch | 2.9.0+cu130 |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |

세부 메타: `environment.json`.

## 2. 설정 (env: `h100x4_qwen1.5b_hybrid_smoke.env`)

| 항목 | 값 |
|---|---|
| TP | 4 |
| max-model-len | 4096 |
| 프롬프트 | random in=128, out=128, num_prompts=500, request_rate=inf |
| Routing strategy | **capacity** |
| Routing priority | **cpu-first** |
| `cpu_max_num_seqs` (resolver) | 1 |
| `cpu_threads` (resolver) | **96 (auto)** |
| `cpu_kvcache_gb` (resolver) | **377 (auto)** |
| `numa_aware` | true |
| `num_cpu_engines` | 1 |
| `ONEDNN_ISA` | AVX512_CORE_AMX |
| **`VLLM_HYBRID_TRACE`** | **1** ⚠️ |
| **`VLLM_HYBRID_TRACE_EVERY`** | **1** ⚠️ |

## 3. 결과

### 3.1 GPU-only

| 지표 | 값 |
|---|---|
| Successful | **500 / 500** |
| Duration (s) | 3.68 |
| Wall (s) | **13.96** |
| Total input / output tokens | 63 699 / 61 590 |
| Request throughput (req/s) | **135.92** |
| Output throughput (tok/s) | **16 742.80** |
| Mean TTFT (ms) | 736.69 |
| P99 TTFT (ms) | 888.56 |
| Mean TPOT (ms) | **22.36** |
| P99 TPOT (ms) | 23.05 |

### 3.2 Hybrid (capacity + cpu-first, **TRACE=1**)

| 지표 | 값 |
|---|---|
| Successful | **500 / 500** |
| Duration (s) | 49.54 |
| Wall (s) | **106.66** |
| Total input / output tokens | 63 699 / 61 590 |
| Request throughput (req/s) | **10.09** |
| Output throughput (tok/s) | **1 243.31** |
| Mean TTFT (ms) | 1225.77 |
| P99 TTFT (ms) | 1550.19 |
| Mean TPOT (ms) | **60.06** |
| P99 TPOT (ms) | 61.36 |

### 3.3 Delta (Hybrid − GPU-only)

| 지표 | Δ | 비고 |
|---|---|---|
| Wall | **+92.70 s** | hybrid 가 7.64 × |
| Duration | +45.86 s | hybrid 가 13.5 × |
| Output throughput | **−15 499 tok/s** | hybrid / gpu = **0.074 ×** |
| Request throughput | −125.83 req/s | 동일 패턴 |
| Mean TTFT | +489 ms | |
| Mean TPOT | **+37.7 ms (1.69×)** | GPU 자체가 느려짐 — 호스팅 process bottleneck 신호 |

### 3.4 Router stats (capacity, cpu-first)

```
Router stats [356 reqs]: GPU=4.4 tok/s (354 reqs), CPU=2.7 tok/s (2 reqs), cpu_ratio=0.6%, in_flight=1/1
Router stats [501 reqs]: GPU=3.8 tok/s (499 reqs), CPU=2.7 tok/s (2 reqs), cpu_ratio=0.4%, in_flight=1/1
Router stats [501 reqs]: GPU=25.0 → 27.5 tok/s (499 reqs), CPU=2.7 tok/s (2 reqs), cpu_ratio=0.4%, in_flight=1/1
```

- 501 req 중 **CPU 2 / GPU 499**.
- CPU lane 은 in_flight=1/1 로 끝까지 점유 (2 reqs 만 쥐고 못 끝냄). capacity router 는 CPU slot 이 안 비니 새 req 를 모두 GPU 로 보냄. **abort 누수 없음**.
- GPU tok/s 가 stat 사이에 4.4 → 27.5 로 변화 — burst 직후 trace I/O 폭주로 stat 측정 자체가 흔들림.

### 3.5 CPU decode kernel path

```
totals = { custom_avx: 0, ipex: 7112, sdpa_batched: 0, sdpa_loop: 0 }
```

전부 IPEX (Tech_done v3 와 동일).

---

## 4. 원인 분석 — TRACE=1 의 영향

### 직접 증거

1. **`mean_tpot_ms` 가 22.4 → 60.1 ms** 로 부풀음. GPU executor 가 느려진 게 아니라, **API server 메인 thread 의 응답 송신 경로가 정체**된 것을 의미. CPU 자체의 decode 는 별도 process (`CPU_EngineCore_1 pid=51014`) 에 격리돼 있어 GPU TPOT 에 직접 영향 못 줌.
2. 같은 1.5B 를 같은 하드웨어에서 **TRACE=0** + throughput-adaptive 로 돌린 직전 실험 (`20260411_085801_h100x4_qwen1.5b_thro_adaptive_500`) 은 hybrid wall 25.67 s ≈ GPU-only wall 25.70 s 였음. 즉 hybrid 코드 자체는 GPU lane 을 망치지 않음. **유일한 변수가 TRACE=1**.
3. server log 11267 line 중 대부분이 `[HYBRID-CPU-ATTN] decode call#NNNN`. `VLLM_HYBRID_TRACE_EVERY=1` 이라 매 decode call 마다 stdout 으로 print → API server / CPU engine / dispatcher / router 가 모두 `print` 의 GIL+IO lock 위에 직렬화.

### 메커니즘

- vllm 의 모든 hybrid marker 는 Python `logging` 또는 `print` 로 stdout 에 emit. nohup 으로 파일 redirect 했지만 OS-level write 에는 여전히 lock.
- API server 메인 thread 가 응답 송신 + 라우팅 + dispatch + log emit 를 담당 → log emit 가 io block 되면 응답 송신이 직접 지연.
- GPU executor 자체는 별도 process (TP4 의 worker) 라 영향 작지만, 결과를 메인 thread 가 받아 client 로 흘려보내는 구간에서 직렬화.
- → GPU TPOT 부풀고, request throughput 도 1/8 로 추락.

### 부수 관측

- abort slot leak 패치는 정상 (in_flight=1/1 끝까지 유지, 0 으로 안 떨어져도 stuck 아님 — 1.5B + in/out=128 의 CPU req 가 그만큼 오래 걸린 것).
- AMX env 활성, IPEX path 정상 — CPU 코드 무결성에는 문제 없음.

---

## 5. 결론 / 다음 액션

1. **`VLLM_HYBRID_TRACE=1` 은 production benchmark 에 절대 사용 금지.** smoke 에서 첫 부팅/marker 확인용으로만 쓰고, 실제 throughput 측정에는 `=0` 을 강제해야 함.
2. **재실험 권장**: 동일 env 에서 `VLLM_HYBRID_TRACE=0`(또는 `VLLM_HYBRID_TRACE_EVERY=500` 이상) 으로 다시 돌려서 capacity + cpu-first 의 진짜 throughput 측정. 직전 throughput-adaptive 1.5B 결과와 비교 가능.
3. **env 파일 cleanup**: `h100x4_qwen1.5b_hybrid_smoke.env` 의 헤더 코멘트는 "10-prompt smoke" 로 되어 있는데 실제 `NUM_PROMPTS=500`. 이름과 내용 불일치도 정리 대상. 또한 `VLLM_HYBRID_TRACE` 기본값을 0 으로 두고, smoke 가 필요할 때만 명시적으로 1 로 켜는 패턴이 안전.

---

## 6. 디렉토리

```
20260411_090942_h100x4_qwen1.5b_capacity_trace_on_500/
├── README.md
├── environment.json
├── summary.json
├── env_files_used/
│   └── h100x4_qwen1.5b_hybrid_smoke.env
├── run_gpu_only/
│   ├── server_gpu_only.log
│   └── bench_result/                     (gpu_only.json + monitor csv 등)
└── run_hybrid/
    ├── server_hybrid.log                 (11267 line — TRACE=1 으로 매우 큼)
    ├── hybrid_markers.txt                (RESOLVE/LAUNCH/ENV/PROC/WORKER/CLIENT/CPU-ATTN/Router stats 추출)
    └── bench_result/                     (hybrid.json + monitor csv 등)
```
