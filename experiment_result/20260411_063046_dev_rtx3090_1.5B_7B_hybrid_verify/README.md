# Dev RTX 3090 — Qwen2.5 1.5B/7B Hybrid vs GPU-only 검증

**Timestamp (KST)**: 2026-04-11 06:30:46
**Runs 시작 (KST)**: 1.5B GPU 06:04:12 → 1.5B Hybrid 06:07:12 → 7B GPU 06:18:58 → 7B Hybrid 06:21:32
**목적**: hybrid 모드에서 CPU 물리 코어 16개가 모두 사용되는지, 끝까지 정상 동작하는지, 병목이 무엇인지 확인
**결과 요약**: 코어 사용 ✓ / 무결성 ✓ / 병목 = CPU per-request latency (구조적, 로직 문제 아님)

---

## 1. 실행 환경

| 항목 | 값 |
|---|---|
| **host** | mystousUbuntu (Ubuntu 22.04, Linux 6.8.0-90-generic), docker `mystous/vllm_hybrid:cu13_v0.6_h100x4` |
| **CPU** | 12th Gen Intel Core i9-12900KF (1S × 16 physical = 8 P + 8 E, 24 logical) |
| **CPU ISA** | AVX, AVX2, **AVX-VNNI**, FMA, F16C (**no AVX-512, no AMX**) |
| **NUMA** | 1 node (node0 = CPU 0–23) |
| **Memory** | 63 GB |
| **GPU** | NVIDIA GeForce RTX 3090 × 1, 24 GB VRAM, driver 580.126.09 |
| **vLLM** | `0.1.dev8475+g78fa48cb8` |
| **torch** | `2.9.0+cu130` |
| **CUDA** | 13.0 |
| **IPEX** | 설치됨 (decode path primary) |
| **Extensions** | `_C` (CUDA main) ✓, `_C_cpu_ops` ✓ (dev 는 AVX-512 없어 stub fallback), `_C_utils` ✓ (`init_cpu_threads_env` 등록) |

## 2. Code base 상태

| 항목 | 값 |
|---|---|
| branch | `h100_cu13` |
| HEAD | `a0d15b3788d40fd85a19e1635bd2d30b08a5bc71` |
| HEAD subject | `docs: append TODO v3 progress snapshot and Tech_done v2 policy correction` |
| uncommitted changes at run time | none (working tree clean) |

**Recent commits (run 시점 기준)**:
- `a0d15b378` docs: append TODO v3 progress snapshot and Tech_done v2 policy correction
- `c31bc5877` docs: sync project docs to current codebase
- `84e9c8201` chore: add benchmark results and archive old_TODO_TASKS.md
- `ad70b8f4d` fix: hybrid CPU engine end-to-end fixes — _C_utils extension, NUMA auto, diagnostics
- `ee01e68cf` docs: add Tech_done.md and append v2 work logs (hybrid verification session)

**핵심 관련 소스 파일** (=hybrid 동작의 load-bearing 경로):
- `vllm/v1/engine/hybrid_core.py` — `CapacityAwareRouter`, `_resolve_cpu_params`, `_resolve_num_cpu_engines`, `launch_hybrid_engines`
- `vllm/v1/engine/core_client.py` — `HybridAsyncMPClient` / `HybridSyncMPClient`, resolver write-back, dispatch
- `vllm/v1/worker/cpu_worker.py` — `init_device`, `_get_autobind_cpu_ids`, `execute_model` trace
- `vllm/v1/attention/backends/cpu_attn.py` — `_IPEXPagedAttention`, decode path counter
- `vllm/platforms/intel_cpu_utils.py` — feature detection, NUMA, oneDNN ISA env
- `vllm/worker/worker_base.py` — heterogeneous heuristic 우회 (hybrid CPU engine)
- `vllm/config.py` — `HybridConfig` (auto sentinels)
- `vllm/engine/arg_utils.py` — CLI 기본값
- `csrc/cpu/utils.cpp` — `init_cpu_threads_env` C++ 구현 (OpenMP + `sched_setaffinity` + `numa_set_membind`)
- `csrc/cpu/torch_bindings_utils.cpp` — `_C_utils` torch library 등록
- `cmake/cpu_utils_extension.cmake` — `_C_utils` standalone extension 빌드 (OpenMP + libnuma 만 요구, SIMD 의존 없음)

## 3. 사용 파일

| 카테고리 | 경로 |
|---|---|
| Serve script | `eval/serve.sh` |
| Bench script | `eval/bench.sh` |
| Monitor script | `eval/monitor.py` |
| Benchmark tool | `benchmarks/benchmark_serving.py` (dataset=random, in=out=128, 500 prompts, request_rate=inf) |
| Env file (1.5B) | `eval/envs/dev_rtx3090_500.env` |
| Env file (7B) | `eval/envs/dev_rtx3090_qwen7b_500.env` |
| Models | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` (HF cache) |

각 env file 의 스냅샷은 `env_files_used/` 에 복사.

## 4. Hybrid config (적용된 값)

| 파라미터 | 값 | 비고 |
|---|---|---|
| `hybrid-mode` | `parallel-batch` | |
| `hybrid-num-cpu-engines` | `1` | auto → NUMA nodes = 1 |
| `hybrid-cpu-max-seqs` | `1` per engine | 설계 원칙 고정 |
| `hybrid-cpu-threads` | auto = 16 | 물리 코어 전부 |
| `hybrid-cpu-kvcache-gb` | 8 (1.5B) / 16 (7B) | |
| `hybrid-numa-aware` | false (1.5B) / true (7B) | dev 는 1 node 이므로 실제 동작 동일 |
| `hybrid-routing-strategy` | `capacity` | |
| `hybrid-routing-priority` | `cpu-first` | |
| OMP pin target cores | `1,3,5,7,9,11,13,15,16,17,18,19,20,21,22,23` | P-core SMT-odd + E-core 전부 (16 physical cores) |
| OMP pin method | C++ `torch.ops._C_utils.init_cpu_threads_env` | `sched_setaffinity` 1:1 pin |

## 5. 결과 요약

### 5.1 Bench metrics (500 prompts, random in=out=128)

| 지표 | 1.5B GPU | 1.5B Hybrid | 7B GPU | 7B Hybrid |
|---|---:|---:|---:|---:|
| completed | 500/500 | 500/500 | 500/500 | 500/500 |
| wall time (s) | **14.31** | **34.90** | **38.78** | **109.31** |
| duration (s) | 8.27 | 17.07 | 30.69 | 48.94 |
| req TP (req/s) | 60.47 | 29.30 | 16.29 | 10.22 |
| output TP (tok/s) | 7448.67 | 3608.67 | 2030.61 | 1273.73 |
| total TP (tok/s) | 15152 | 7341 | 4106 | 2575 |
| mean TTFT (ms) | 3050 | 11573 | 10990 | 15147 |
| median TTFT (ms) | 2002 | 10370 | 7055 | 10867 |
| P99 TTFT (ms) | 5926 | 14760 | 22032 | 26749 |
| mean TPOT (ms) | 27.8 | 30.0 | 108.5 | 116.1 |
| P99 TPOT (ms) | 30.9 | 34.8 | 382.9 | 374.9 |
| total_input_tokens | 63699 | 63699 | 63699 | 63699 |
| total_output_tokens | 61590 | 61590 | 62327 | 62342 |

### 5.2 CPU 코어 사용률 (monitor.py 1 Hz 샘플링)

| 지표 | 1.5B GPU | 1.5B Hybrid | 7B GPU | 7B Hybrid |
|---|---:|---:|---:|---:|
| pinned 16 코어 avg (%) | 4.4 | **77.4 (run 전체) / 96.5 (busy window)** | 4.9 | **95.0 / 97.7** |
| pinned 16 코어 max (%) | 10.1 | **100.0** | 12.4 | **100.0** |
| 전체 24 logical avg (%) | 6.0 | 55.3 | 5.2 | 67.5 |
| GPU util avg (%) | 50.2 | **21.0** | 84.9 | **26.9** |
| GPU util max (%) | 98 | 97 | 100 | 100 |
| GPU power avg (W) | 254 | **147** | 343 | **163** |
| CPU busy window (s) | — | 3.96 → 29.65 (26 samples) | — | **3.96 → 108.13 (102 samples)** |

**Busy window 내 per-core 평균** (pinned 16 코어):

| 영역 | 1.5B Hybrid | 7B Hybrid |
|---|---:|---:|
| P-core odd (1,3,5,7,9,11,13,15) | 95.7 ~ 97.6% | 95.8 ~ 96.3% |
| E-core (16–23) | 97.0% (전원) | **99.4%** (전원) |
| P-core 짝수 siblings (0,2,4,...,14) — pin 대상 아님 | 5 ~ 9% | 7 ~ 11% (c00=37.9, main thread 공유) |

### 5.3 Hybrid 동작 무결성 지표

| | 1.5B Hybrid | 7B Hybrid |
|---|---|---|
| `[HYBRID-CLIENT] → cpu:0` dispatches | 2 | 2 |
| `[HYBRID-CLIENT] → gpu` dispatches | 499 | 499 |
| `cpu_ratio` | 0.4% | 0.4% |
| Router 최종 상태 | `in_flight=0/1` | slot released after req 2 |
| CPU throughput per req | **9.9 tok/s** | **2.3 tok/s** |
| GPU throughput per req | 8.3~8.9 tok/s | 4.4~5.9 tok/s |
| decode path counters | `ipex=3500+, sdpa_*=0, custom_avx=0` | `ipex=7000+, sdpa_*=0, custom_avx=0` |
| OMP pin via | C++ `init_cpu_threads_env` | C++ `init_cpu_threads_env` |
| OMP tid 1:1 → 16 cores | ✓ (로그 확인) | ✓ (로그 확인) |
| 500/500 successful | ✓ | ✓ |

### 5.4 1.5B vs 7B 스케일링 비교

| 지표 | 1.5B | 7B | 배수 |
|---|---:|---:|---:|
| CPU per-req latency | ~13 s | ~55 s | **×4.2** |
| CPU busy window | 26 s | 102 s | **×3.9** |
| CPU per-req throughput | 9.9 tok/s | 2.3 tok/s | ×0.23 |
| Hybrid wall penalty vs GPU | ×2.44 | ×2.82 | 유사 |
| GPU util drop | 50→21 % | 85→27 % | 유사 |
| pinned 16 코어 포화 | 96.5% | 97.7% | ≈ 동일 |

## 6. 병목 분석

### 6.1 근본 원인

`T_hybrid = max(T_GPU_main_batch, T_CPU_tail)` 구도에서 **CPU tail 이 지배**:

- **1.5B**: GPU 혼자 500 reqs 를 8.3 초에 처리 가능. CPU 는 한 request 를 13 초에 처리.
  → GPU 는 8 초 만에 479 reqs 다 끝내고 나머지 9 초는 CPU 의 마지막 2 reqs 를 기다림.
- **7B**: GPU 혼자 500 reqs 를 30.7 초에 처리 가능. CPU 는 한 request 를 55 초에 처리.
  → GPU 는 30 초 만에 479 reqs 다 끝내고 나머지 19 초는 CPU tail 대기.

### 6.2 병목 차원

| 차원 | 판정 |
|---|---|
| **CPU decode throughput (tok/s)** | **이것이 bottleneck**. dev 는 AVX-512/AMX 없고 DDR4 (DDR5 아님) 수준 대역, IPEX 가 AVX2+VNNI path 로 동작 → 1.5B 9.9 tok/s / 7B 2.3 tok/s 에 갇힘 |
| OMP thread pinning | 아님 — 16 코어 96~99% 사용, 완벽 |
| Attention kernel | 아님 — 100% IPEX oneDNN C++ (fallback 0 건) |
| CPU slot cycle / router dispatch | 아님 — slot 정상 반납, capacity router 설계대로 동작 |
| Correctness | 아님 — 4 runs 전부 500/500 성공 |

### 6.3 Hybrid gain 이 나오는 조건

- GPU 가 혼자 처리하지 못할 정도로 포화된 고부하 (큰 배치 / 긴 decode / 고 QPS)
- 또는 CPU throughput 이 극적으로 올라가야 함:
  - AVX-512 + VNNI → IPEX oneDNN VNNI path 활성
  - AMX-BF16 → oneDNN `AVX512_CORE_AMX` (Sapphire Rapids Xeon 8480+)
  - DDR5 6400 MT/s + multi-channel 대역폭
  - 멀티 NUMA node (`num_cpu_engines = num_numa`) → per-engine 1 seq × N node = N concurrent CPU seqs
- dev 는 위 조건 모두 부재 → **로직 검증 전용 환경**. 성능 실측은 H100x8 + Xeon 2-socket 에서 수행 예정 (`TODO.md §3`)

### 6.4 1.5B ↔ 7B 교차 검증 (재현성)

- 16 코어 pin, IPEX decode path, capacity router cycle, `cpu_dispatches=2` — 양쪽 모델에서 **동일한 패턴**으로 재현
- 7B `CPU=2.3 tok/s` 는 `Tech_done.md v1` (이전 세션) 기록과 **정확히 일치** → 재현성 확보
- 모델 크기가 4.67× 커질 때 CPU per-token 이 4.3× 느려짐 (합리적, 선형 스케일링) → CPU 경로에 체계적 regression 없음

## 7. 파일 구성

```
20260411_063046_dev_rtx3090_1.5B_7B_hybrid_verify/
├── README.md                     ← 본 문서
├── environment.json              ← host/hw/sw/git/설정 메타데이터
├── summary.json                  ← 모든 수치 한데 모은 기계 가독형
├── env_files_used/
│   ├── dev_rtx3090_500.env       ← 1.5B env 스냅샷
│   └── dev_rtx3090_qwen7b_500.env ← 7B env 스냅샷
├── run_1.5B_gpu_only/
│   ├── gpu_only.json             ← benchmark_serving.py 결과
│   ├── gpu_only_bench.log        ← bench 실행 stdout
│   ├── gpu_only_monitor_cpu.csv  ← monitor.py per-core CPU util (1 Hz)
│   ├── gpu_only_monitor_gpu.csv  ← monitor.py GPU util/power (1 Hz)
│   ├── monitor_gpu_only.log      ← monitor.py debug log
│   ├── server.log                ← vLLM 서버 stdout
│   └── system_info.json          ← bench.sh 수집 sysinfo
├── run_1.5B_hybrid/
│   ├── hybrid.json
│   ├── hybrid_bench.log
│   ├── hybrid_monitor_cpu.csv
│   ├── hybrid_monitor_gpu.csv
│   ├── monitor_hybrid.log
│   ├── per_thread_psr.log        ← 1 Hz per-thread PSR/pcpu 샘플 (CPU_EngineCore)
│   ├── server.log                ← 서버 stdout + hybrid diagnostic markers
│   └── system_info.json
├── run_7B_gpu_only/ ← (동일 구조)
└── run_7B_hybrid/ ← (동일 구조, per_thread_psr.log 포함)
```

## 8. 재현 절차

```bash
# 1. GPU-only 1.5B
./eval/serve.sh gpu_only eval/envs/dev_rtx3090_500.env &
# wait for /health 200
./eval/bench.sh gpu_only eval/envs/dev_rtx3090_500.env
pkill -f api_server

# 2. Hybrid 1.5B
./eval/serve.sh hybrid eval/envs/dev_rtx3090_500.env &
# wait
./eval/bench.sh hybrid eval/envs/dev_rtx3090_500.env
pkill -f api_server; pkill -f EngineCore

# 3. GPU-only 7B
./eval/serve.sh gpu_only eval/envs/dev_rtx3090_qwen7b_500.env &
./eval/bench.sh gpu_only eval/envs/dev_rtx3090_qwen7b_500.env
pkill -f api_server

# 4. Hybrid 7B
./eval/serve.sh hybrid eval/envs/dev_rtx3090_qwen7b_500.env &
./eval/bench.sh hybrid eval/envs/dev_rtx3090_qwen7b_500.env
pkill -f api_server; pkill -f EngineCore
```

OMP 1:1 pin 확인 + per-thread PSR 샘플링:
```bash
# hybrid 서버 부팅 후 CPU_EngineCore PID 확인
grep "HYBRID-CPU-WORKER.*init_cpu_threads_env (C\+\+) returned" server.log
# 이후 PID 로 1 Hz per-thread 샘플링
for i in $(seq 1 300); do
  echo "=== sample $i ($(date +%T)) ===" >> psr.log
  ps -L -p ${CPU_PID} -o tid,psr,pcpu,comm --no-headers >> psr.log
  sleep 1
done &
```

## 9. 결론

1. **CPU 물리 코어 16개 모두 사용됨** — 1.5B/7B 양쪽에서 pinned 16 코어가 busy window 동안 96.5~97.7% 평균 사용, 특히 E-core 8개는 99.4% 로 상시 고정. OMP tid ↔ core 1:1 pin 은 C++ `init_cpu_threads_env` 로 등록 확인.
2. **끝까지 정상 동작함** — 4 runs 전부 500/500 successful. Decode path 100% IPEX. Router slot cycle 정상. CPU=2.3 tok/s (7B) 는 이전 세션 Tech_done.md v1 과 정확히 일치하는 재현성.
3. **병목은 CPU per-request decode throughput** — dev AVX2 환경 + RTX 3090 의 조합은 `T_CPU(1 req) > T_GPU(500 reqs)` 구간이라 hybrid gain 구간이 존재하지 않음. 이는 **로직 결함이 아니라 하드웨어 매트릭스의 근본 한계** 이며 논문 §Limitations 의 "작은 GPU / 짧은 tail workload 에서 hybrid gain 없음" 과 일치.
4. **Hybrid gain 실측은 타겟 환경으로 이관해야 함** — H100x8 + Xeon 8480+ 2-socket (AVX-512/VNNI/AMX-BF16 + DDR5 + 2 NUMA node) 에서 수행 예정 (`TODO.md §3`).
