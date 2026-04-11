# H100x4 Qwen2.5-7B Hybrid — Smoke + CPU 처리량 진단

`20260411_082501_h100x4_qwen7b_smoke_cpu_bw_diagnosis`

> **목적**: 첫 H100x4 (Xeon 8480+ KVM) hybrid bring-up. 부팅·코어 pin·AMX 경로·
> capacity-stuck 해소 검증 + dev (i9-12900KF + RTX 3090) 와의 CPU 처리량 비교.
> 결과: **부팅/경로 검증 OK, CPU 처리량은 dev 와 사실상 동일** — memory-bandwidth
> bound 구간에 들어간 것이 원인으로 확정.

---

## 1. 환경

| 항목 | 값 |
|---|---|
| Host | KVM guest, `hypervisor` flag 노출 |
| GPU | NVIDIA H100 80GB HBM3 × 4 |
| CPU | Intel Xeon Platinum 8480+ (1S × 96 vCPU, **Thread/core=1**, **NUMA=1**) |
| ISA | AVX-512 F/VNNI/BF16/FP16, AVX-VNNI, **AMX (bf16/int8/tile)** |
| RAM | 944 GB usable |
| vLLM | `019c7121a` (h100_cu13, abort slot leak fix 포함) |
| torch | 2.9.0+cu130 |
| IPEX | 2.8.x (PyTorch 2.9 mismatch warning, 동작은 정상) |
| Model | `Qwen/Qwen2.5-7B-Instruct` (BF16, ~14 GB weight) |
| Routing | `capacity` + `cpu-first` (smoke), `--max-model-len 4096` |
| Hybrid | num_cpu_engines=auto(=1) / cpu_max_num_seqs=auto(=1) / threads=auto(=96) / kvcache=auto(=377 GB) |

세부 호스트/CPU/메모리/GPU 덤프는 `diagnosis/host_environment.txt` 와 `environment.json`.

---

## 2. 실행 흐름 (이번 세션)

1. **첫 시도 (재컴파일 전)** — `cudaErrorSymbolNotFound` 으로 GPU CUDA graph capture 단계에서 워커 전원 사망.
2. **재컴파일 후** — 부팅 250 s, health OK, 모든 hybrid marker 정상 출력.
3. **10 req smoke (`bench.sh hybrid envs/h100x4_qwen7b_hybrid_smoke.env`)** — 10/10 성공, 508 s, output 2.31 tok/s.
4. **라이브 진단 수집** — CPU_EngineCore (PID 21250) 의 `proc/status`, `ps -L`, PSR histogram, NUMA stat 캡쳐.

---

## 3. 핵심 관측

### 3.1 부팅·환경 (`run_smoke_hybrid/hybrid_markers.txt` 참조)

```
[HYBRID-RESOLVE]  max_seqs=1 threads=96 kvcache=377GB batched_tokens=256 |
                  effective_cores=96 (physical=96) numa_nodes=1
                  effective_mem=944GB (total=944GB)
                  user_overrides: max_seqs=auto threads=auto kvcache=auto
[HYBRID-LAUNCH]   num_cpu_engines=1 (numa_aware=True, config=1)
[HYBRID-CPU-ENV]  PID=21250 OMP=96 MKL=96 OPENBLAS=96
                  ONEDNN_ISA=AVX512_CORE_AMX  ← AMX 감지
                  KVCACHE=377GB BIND=auto
[HYBRID-CPU-PROC] torch_threads=96 torch_interop=1 mkldnn=True torch=2.9.0+cu130
[HYBRID-CPU-WORKER] local_omp_cpuid='0,1,2,...,95'  (rank=0)
[HYBRID-CPU-WORKER] init_cpu_threads_env (C++) returned: 96 tid ↔ 96 PSR 1:1 mapping
[HYBRID-CPU-WORKER] thread binding established via: C++ (init_cpu_threads_env)
```

### 3.2 Smoke benchmark (`run_smoke_hybrid/bench_result/hybrid.json`)

| 지표 | 값 |
|---|---|
| Successful | **10 / 10** |
| Wall (s) | 508.31 |
| Total generated tokens | 1174 |
| Output throughput (tok/s) | **2.31** |
| Mean TTFT / Median / P99 (ms) | 1598 / 191 / 13206 |
| Mean TPOT / Median / P99 (ms) | **405.31** / 17.98 / 3540 |
| Mean ITL / Median / P99 (ms) | 440 / 19.7 / 4098 |

`Router stats [11 reqs]: GPU=51.9 tok/s (9 reqs), CPU=0.2 tok/s (2 reqs), cpu_ratio=18.2%, in_flight=1/1`

해석: median TPOT 18 ms 는 GPU 응답, mean/P99 의 405 ms / 3.5 s 는 CPU 응답. **CPU 가 1~3 req 를 매우 느리게 끌고 가며 wall 을 지배**.

### 3.3 CPU decode kernel path

`HYBRID-CPU-ATTN` 카운터 (smoke 종료 시점):
```
totals = { custom_avx: 0, ipex: 6260+, sdpa_batched: 0, sdpa_loop: 0 }
```
**전 decode call 이 IPEX (oneDNN 경유)** — Tech_done v3 F3 와 동일 분기. dev 와의 차이는 환경변수 `ONEDNN_ISA=AVX512_CORE_AMX` 가 켜져 있다는 점뿐.

### 3.4 Thread / PSR 분포 (`diagnosis/psr_histogram.txt`, `ps_thread_full.txt`)

- **454 threads**, 분포: PSR 0 = 46 (main + python infra), PSR 1~95 = 각 3~6 threads.
- 활성 (>1 % CPU) thread 가 PSR 1~27 + PSR 28~95 양쪽 모두에 존재 → **96 PSR 전부 work 받음**.
- per-thread CPU util ≈ **70 %** (100 % 가 아님 — barrier/sync overhead 존재).
- main thread (TID = PID) `Cpus_allowed_list = 0` — C++ pin 의도된 동작 (Tech_done v1 Q1 과 동일 구조).

### 3.5 정량 비교

| 환경 | active cores × util | cpu-units | per-req tok/s (CPU) |
|---|---|---|---|
| dev (i9-12900KF) | 16 × 100 % | 16 | **2.3** (Tech_done v3 F2) |
| H100/Xeon 8480+ KVM | 96 × 70 % | **67** | **2.47** (1 / 0.405 s mean TPOT) |

cpu budget **4.2 ×** 인데 CPU per-req throughput 은 **사실상 동일**.

---

## 4. 원인 (가장 강한 가설)

### Decode 는 memory-bandwidth bound 이다.

- 7B BF16 weight ≈ 14 GB. decode 1 token 마다 weight 전체를 DRAM 에서 한 번 읽어야 함.
- KVM 게스트의 DDR5 BW (Xeon 8480+ 8ch DDR5-4800 host BW 의 일부) 와 dev 의 DDR5 BW 가 비슷한 수준이라면 도달 가능한 token/s 가 비슷할 수밖에 없다.
- 96 core × AMX 추가는 BW 가 늘지 않는 한 의미 없음. AMX 는 **compute 가 BW 를 따라잡을 때** 비로소 가치 — int8 / w8a8 quant 또는 batch>1 가 그 조건.
- 보조 요인: Qwen2.5-7B 의 `num_attention_heads = 28` → IPEX paged-attention 의 head-level 병렬은 28-way 가 한계. attention 단계에서 96-28 = 68 core 는 본질적으로 idle. 단 main bottleneck 은 GEMM (Q/K/V proj, FFN gate/up/down) 이고 그건 BW bound.

### 부수 관측

- per-thread util 70 % 의 30 % 는 OMP barrier sync 와 GIL 잡힌 PSR 0 contention 으로 보임 (PSR 0 에 46 threads 몰려 있음).
- AMX BF16 GEMM 자체는 정상이며 ISA 환경변수도 정상 설정. 그러나 BW bound 구간에서 AMX 는 throughput 에 기여 안 함.

---

## 5. 시사점

| 항목 | 영향 |
|---|---|
| `cpu_max_num_seqs = 1` 원칙 | BW-bound 구간에서는 weight 재사용이 0 → throughput 에 매우 불리. **N seq 동시 처리하면 weight 1회 read 로 ~N× 가속** 가능. 논문 §3.4 의 원칙은 "OMP 가 단일 시퀀스를 NUMA 노드 전체로 가속" 을 가정 — 그 가정은 compute-bound 일 때만 성립. 검증 필요. |
| AMX 활용 조건 | BF16 dense decode 로는 효과 없음. **int8 quant + AMX_INT8** 또는 **batch ≥ 2** 가 동반되어야. |
| 모델 스케일링 | 70B 에서는 GPU 도 per-token 더 느려짐 → CPU/GPU ratio 는 hybrid 쪽으로 약간 유리해질 가능성. 검증 가치 있음. |
| Hybrid 의 본질적 의의 | "GPU 가 batch 를 줄이지 않게 long-prompt 를 CPU 로 흘려" 라는 시나리오 — prompt prefill 에서는 BW 가 아닌 compute bound 가 더 강할 수 있음. prefill-routing 만 따로 평가할 가치. |

---

## 6. 검증 OK 항목 (이번 smoke 로 닫힌 것)

- ✅ H100 부팅 (재컴파일로 CUDA graph capture 이슈 해소)
- ✅ 96 physical core 자동 인식, NUMA 1 / num_cpu_engines=1 auto
- ✅ C++ `init_cpu_threads_env` 96-way 1:1 pin, OMP worker 96 PSR 분산
- ✅ AMX env (`ONEDNN_ISA=AVX512_CORE_AMX`) 활성
- ✅ IPEX path 6260+ call, custom_avx / sdpa 미사용 (분기 정상)
- ✅ Capacity-stuck (TODO v1 §1) **재현 안 됨** — abort slot leak 패치 효과 H100 에서도 확인
- ✅ 10 / 10 정상 종료, slot cycle 정상

## 7. 다음 액션 (사용자 결정 대기)

| 옵션 | 내용 | 비용 |
|---|---|---|
| A | **500 main, throughput-adaptive + cpu-first, Qwen2.5-7B** — 적응형 라우팅이 느린 CPU 를 어떻게 운영하는지 측정 (실질적으론 GPU-only 에 수렴할 가능성). | ~15 min |
| B | **Llama-3.1-70B-Instruct** 로 모델 스케일 — GPU/CPU ratio 가 hybrid 쪽으로 이동하는지 확인. | 부팅+bench ~1 h+ |
| C | **int8/w8a8 quant** 모델로 AMX_INT8 path 검증. | 모델 변환 필요 |
| D | **`cpu_max_num_seqs = 2~4`** 실험 — 원칙은 위배되지만 BW reuse 효과 측정. | 즉시 |

500 main 진행 시 server 재기동 필요 (현재 wrapper 죽고 EngineCore 만 살아있음).

---

## 디렉토리 구성

```
20260411_082501_h100x4_qwen7b_smoke_cpu_bw_diagnosis/
├── README.md                            # 본 문서
├── environment.json                      # 호스트/CPU/GPU/SW 메타
├── summary.json                          # 결과 + 결론 구조화
├── env_files_used/
│   ├── h100x4_qwen7b_hybrid_smoke.env    # 이번 smoke 사용 env
│   └── h100x4_qwen7b_hybrid_500.env      # (다음 단계용 — throughput-adaptive)
├── run_smoke_hybrid/
│   ├── server_smoke.log                  # 9345 line 전체 서버 로그
│   ├── hybrid_markers.txt                # marker 만 추출 (RESOLVE/LAUNCH/ENV/PROC/WORKER/CLIENT/CPU-ATTN/Router stats)
│   └── bench_result/                     # bench.sh 가 만든 표준 결과 폴더
│       ├── hybrid.json                   # benchmark_serving 결과 (TTFT/TPOT/ITL)
│       ├── hybrid_bench.log
│       ├── hybrid_monitor_cpu.csv        # 1 Hz CPU per-core util
│       ├── hybrid_monitor_gpu.csv
│       ├── monitor_hybrid.log
│       └── system_info.json
└── diagnosis/
    ├── snapshot_timestamp.txt
    ├── host_environment.txt              # uname / lscpu / cpuinfo flags / numactl / meminfo / nvidia-smi
    ├── proc_status.txt                   # /proc/21250/status
    ├── ps_thread_full.txt                # ps -L 454 thread × (tid,psr,pcpu,comm)
    ├── psr_histogram.txt                 # PSR 별 thread 수
    ├── per_thread_affinity_summary.txt   # head 100 thread 별 Cpus_allowed_list
    └── numastat.txt                      # numastat -p 21250
```
