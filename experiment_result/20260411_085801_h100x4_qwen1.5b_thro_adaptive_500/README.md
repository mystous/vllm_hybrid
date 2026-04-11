# H100x4 Qwen2.5-1.5B — GPU-only vs Hybrid (throughput-adaptive + cpu-first)

`20260411_085801_h100x4_qwen1.5b_thro_adaptive_500`

> **목적**: 동일 환경(H100x4 + Xeon 8480+ KVM, 944 GB RAM) 에서 **`Qwen/Qwen2.5-1.5B-Instruct`**
> 모델로 GPU-only 와 throughput-adaptive + cpu-first hybrid 를 500-prompt burst 로 비교.
> 결과: **두 모드 throughput 사실상 동일 (Hybrid / GPU-only ≈ 1.008×)**. throughput-adaptive
> 라우터가 CPU 가 처리 못함을 감지하고 모든 트래픽을 GPU 로 우회시켜, hybrid 가 GPU-only 와
> 같은 wall time 에 같은 결과로 종료.

---

## 1. 환경

| 항목 | 값 |
|---|---|
| Host | KVM guest, `hypervisor` flag |
| GPU | NVIDIA H100 80GB HBM3 × 4 |
| CPU | Intel Xeon Platinum 8480+ (1S × 96 vCPU, **Thread/core=1**, NUMA=1) |
| ISA | AVX-512 F/VNNI/BF16/FP16, AVX-VNNI, **AMX (bf16/int8/tile)** |
| RAM | 944 GB |
| vLLM | `019c7121a` (h100_cu13, abort slot leak fix 포함) |
| torch | 2.9.0+cu130, IPEX 2.8.x |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (BF16) |

세부 환경 메타: `environment.json`.

## 2. 설정 (env: `h100x4_Qwen2.5-1.5B_cpu_first_thro.env`)

| 항목 | 값 |
|---|---|
| TP | 4 |
| max-model-len | (auto) |
| 프롬프트 | random in=512, out=512, num_prompts=500, request_rate=inf |
| Routing strategy | **throughput-adaptive** |
| Routing priority | **cpu-first** |
| `HYBRID_CPU_MAX_SEQS` (user) | 2 |
| `cpu_max_num_seqs` (resolver 결과) | **1** (원칙으로 강제) |
| `HYBRID_CPU_THREADS` | **8** (사용자 의도적 — 1.5B 작은 op 의 OMP overhead 회피 목적) |
| `HYBRID_CPU_KVCACHE_GB` | 8 |
| `HYBRID_NUMA_AWARE` | false |
| `HYBRID_NUM_CPU_ENGINES` | 1 |
| `HYBRID_CPU_PREFILL_THRESHOLD` | 1024 (INPUT_LEN=512 보다 크므로 prefill 도 CPU 후보) |
| `ONEDNN_ISA` | AVX512_CORE_AMX (자동 감지) |

## 3. 결과

### 3.1 GPU-only (`run_gpu_only/bench_result/gpu_only.json`)

| 지표 | 값 |
|---|---|
| Successful | **500 / 500** |
| Duration (s) | 12.99 |
| Wall (s) | 25.70 |
| Total input / output tokens | 255 169 / 245 718 |
| Request throughput (req/s) | **38.50** |
| Output throughput (tok/s) | **18 918.66** |
| Mean TTFT (ms) | 1140.11 |
| Median TTFT (ms) | 1109.63 |
| P99 TTFT (ms) | 1746.04 |
| Mean TPOT (ms) | 23.16 |
| Median TPOT (ms) | 22.90 |
| P99 TPOT (ms) | 40.20 |

### 3.2 Hybrid throughput-adaptive + cpu-first (`run_hybrid/bench_result/hybrid.json`)

| 지표 | 값 |
|---|---|
| Successful | **500 / 500** |
| Duration (s) | 12.88 |
| Wall (s) | 25.67 |
| Total input / output tokens | 255 169 / 245 796 |
| Request throughput (req/s) | **38.81** |
| Output throughput (tok/s) | **19 078.11** |
| Mean TTFT (ms) | 1159.75 |
| Median TTFT (ms) | 1149.64 |
| P99 TTFT (ms) | 1766.40 |
| Mean TPOT (ms) | 22.84 |
| Median TPOT (ms) | 22.56 |
| P99 TPOT (ms) | 41.22 |

### 3.3 Delta (Hybrid − GPU-only)

| 지표 | Δ | 비고 |
|---|---|---|
| Duration | **−0.10 s** | 노이즈 수준 |
| Wall | **−0.03 s** | 노이즈 수준 |
| Output throughput | **+159.4 tok/s (+0.84 %)** | 노이즈 수준 |
| Request throughput | +0.31 req/s | 노이즈 수준 |
| Mean TTFT | +19.6 ms | hybrid 가 약간 ↑ (CPU 라우팅 시도 overhead) |
| P99 TTFT | +20.4 ms | 동일 패턴 |
| Mean TPOT | −0.32 ms | 노이즈 수준 |
| P99 TPOT | +1.02 ms | 노이즈 수준 |
| **Ratio (hybrid / gpu_only output_thrput)** | **1.0084 ×** | **사실상 동일** |

### 3.4 Router stats (mid/end-bench, throughput-adaptive)

```
Router stats [503 reqs]: GPU=58.9~59.5 tok/s (501 reqs),
                         CPU=0.0 tok/s (2 reqs),
                         cpu_ratio=0.4%, in_flight=1/1, adaptive_slots=1
```

10 회 sampling 모두 동일 패턴: **CPU 는 503 req 중 2 만 받았고**, 그 2 도 `0.0 tok/s` 로 (즉 prefill 단계에서 막힘 — 1.5B prefill 512 token × `threads=8` 에서 매우 느림). throughput-adaptive 의 `adaptive_slots` 는 1 로 유지 (default).

전체 marker 추출본: `run_hybrid/hybrid_markers.txt`.

---

## 4. 핵심 관측 / 결론

### 4.1 **throughput-adaptive 가 정확히 동작**

- `cpu-first` 인데도 **503 reqs 중 2 만 CPU 로**.
- 첫 1~2 req 가 CPU 로 갔다가 처리량 측정 결과 GPU 가 압도적이라 **이후 트래픽 전체가 GPU 로 우회**.
- 결과: **hybrid wall ≈ GPU-only wall** (25.67 vs 25.70 s).
- abort slot leak 도 재현 안 됨 — capacity in_flight=1/1 로 끝까지 유지되며 stuck 없음.

### 4.2 **hybrid lane 의 production value = 0** (이 batch shape 에서)

- CPU lane 이 처리한 토큰 ≈ 0 (Router stat `0.0 tok/s × 2 reqs`).
- 추가 wall time 비용도 0 (오히려 -30 ms; 노이즈 수준).
- 즉 **hybrid 가 GPU-only 대비 손해도 이득도 없는 무해한 safety net**. throughput-adaptive 의 가치는 명확히 입증.

### 4.3 **`HYBRID_CPU_THREADS=8` 의 영향**

- env 가 의도적으로 8 thread 만 부여 (1.5B 의 작은 op 에서 OMP overhead 회피 목적의 사용자 가설).
- 결과: 96 코어 중 8 OMP worker 만 활용 → 1.5B prefill 512 token 도 단일 시퀀스에서 매우 오래 걸림 → CPU lane 사실상 정지.
- 다음 실험에서 `threads=auto(=96)` 로 되돌리면 CPU lane 이 살아날 수 있는지 검증할 가치가 있음.

### 4.4 **dev 1.5B 와의 정성 비교**

| 환경 | CPU per-req (CPU lane) | adaptive slot 결과 |
|---|---|---|
| dev (i9-12900KF, 16 thread) | ≈ 9.9 tok/s (Tech_done v3 F2) | CPU 로 ~10 % 트래픽 |
| H100/Xeon 8480+ KVM (8 thread) | **0.0 tok/s** (probe 만, prefill 미완) | CPU 로 0.4 % |

threads=8 의 효과가 압도적. dev 의 16 thread 보다도 적기 때문에 in/out=512 의 prefill 무게를 못 견딘 것으로 해석.

---

## 5. 다음 액션 후보

| 옵션 | 내용 |
|---|---|
| A | **`HYBRID_CPU_THREADS=0` (auto=96) 로 동일 실험 재실행** — threads 가 충분할 때 throughput-adaptive 가 CPU 를 어떻게 운영하는지 측정. |
| B | **7B 또는 70B 모델로 동일 셋업** — 모델 크기 늘어나면 GPU per-token 도 비싸져 ratio 가 hybrid 에 유리해질 수 있음 (이전 진단의 가설 검증). |
| C | **int8/w8a8 quant + AMX_INT8** — AMX 가치 검증 (BW 부담 절반). |
| D | **Routing strategy 비교** — 동일 모델/프롬프트에 capacity vs round-robin vs length-aware 비교. |

---

## 6. 디렉토리 구성

```
20260411_085801_h100x4_qwen1.5b_thro_adaptive_500/
├── README.md                            # 본 문서
├── environment.json                     # 호스트/CPU/GPU/SW 메타
├── summary.json                         # gpu_only / hybrid / delta / findings
├── env_files_used/
│   └── h100x4_Qwen2.5-1.5B_cpu_first_thro.env
├── run_gpu_only/
│   ├── server_gpu_only.log              # 부팅 로그
│   └── bench_result/
│       ├── gpu_only.json                # benchmark_serving 결과
│       ├── gpu_only_bench.log
│       ├── gpu_only_monitor_cpu.csv     # 1 Hz CPU per-core util
│       ├── gpu_only_monitor_gpu.csv
│       ├── monitor_gpu_only.log
│       └── system_info.json
└── run_hybrid/
    ├── server_hybrid.log                # 부팅 로그 (HYBRID-* marker 포함)
    ├── hybrid_markers.txt               # marker 만 추출 (RESOLVE/LAUNCH/ENV/PROC/WORKER/CLIENT/CPU-ATTN/Router stats)
    └── bench_result/
        ├── hybrid.json
        ├── hybrid_bench.log
        ├── hybrid_monitor_cpu.csv
        ├── hybrid_monitor_gpu.csv
        ├── monitor_hybrid.log
        └── system_info.json
```
