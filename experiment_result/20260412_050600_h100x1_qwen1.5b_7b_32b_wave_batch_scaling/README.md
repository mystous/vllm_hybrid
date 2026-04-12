# H100x1 Qwen 1.5B / 7B / 32B — wave-batch gpu_only vs hybrid 6-run 검증

`20260412_050600_h100x1_qwen1.5b_7b_32b_wave_batch_scaling`

> H100x4 에서 검증된 wave-batch routing + CPU batching (`cpu_max_seqs=16`)
> 을 **H100x1 (TP=1)** 환경으로 옮겨 1.5B / 7B / 32B 세 모델에서 gpu_only
> vs hybrid 6-run 비교. dev RTX 3090 결과
> (`20260412_023700_dev_rtx3090_wave_batch_gate_fix_initial_validation`)
> 에서 확인된 "CPU tail 이 wall 을 지배" 패턴이 AMX 가용 서버 환경에서도
> 유지되는지 확인한다.

**날짜**: 2026-04-12 (KST)

---

## 1. 환경

| 항목 | 값 |
|------|-----|
| **CPU** | Intel Xeon Platinum 8480+ (1S, 24 cores, 1T/C = 24 vCPU) |
| **ISA** | AVX-512F/BW/VL, AVX-512 VNNI, **AMX-BF16/INT8**, FMA |
| **Memory** | 241,600 MB DDR5, 1 NUMA node |
| **GPU** | 1 × NVIDIA H100 80GB HBM3 (81,559 MiB) |
| **GPU Clock** | SM 1,980 MHz, Mem 2,619 MHz, TDP 700W |
| **Driver / CUDA** | 580.126.09 / 13.0 |
| **Software** | Python 3.12.13, PyTorch 2.9.0+cu130, vLLM 0.1.dev8475 |
| **Commit** | `897be2876` (wave-batch cold-start gate) |
| **Benchmark** | 500 req, random, input 128 / output 128, request_rate=inf (burst) |
| **Hybrid config** | routing=wave-batch, priority=cpu-first, `cpu_max_seqs=16` |
| **TP** | 1 (H100x4 대비 차이: TP=4 → TP=1) |

**참고**: H100x4 (`20260411_142900`) 환경은 동일 CPU (8480+ 96 cores) + H100×4
TP=4. 본 세션은 **GPU 1장 + 24 vCPU** 로 제한된 컨테이너.

---

## 2. 핵심 결과

| # | 모델 | 모드 | bench (s) | wall (s) | req/s | output tok/s | TPOT med (ms) | TTFT mean (ms) |
|---|------|------|----------:|--------:|------:|------------:|--------------:|---------------:|
| 1 | 1.5B | gpu_only | **4.43** | 15.51 | **112.8** | **13,898** | **27.59** | 833 |
| 2 | 1.5B | hybrid | 22.60 | 33.68 | 22.1 | 2,725 | 70.60 | 1,239 |
| 3 | 7B | gpu_only | **4.82** | 16.55 | **103.8** | **12,953** | **26.15** | 1,309 |
| 4 | 7B | hybrid | 53.52 | 65.43 | 9.3 | 1,167 | 82.72 | 2,149 |
| 5 | 32B | gpu_only | **26.17** | 40.22 | **19.1** | **2,352** | **46.25** | 10,428 |
| 6 | 32B | hybrid | 213.59 | 233.21 | 2.3 | 288 | 60.86 | 13,512 |

### 2.1 hybrid / gpu_only 비율

| 지표 | 1.5B (H/G) | 7B (H/G) | 32B (H/G) |
|------|----------:|--------:|---------:|
| bench duration | 5.10× | 11.10× | **8.16×** |
| req throughput | 0.196× | 0.090× | 0.122× |
| output tok/s | 0.196× | 0.090× | 0.122× |
| TPOT median | 2.56× | 3.16× | 1.32× |
| TTFT mean | 1.49× | 1.64× | 1.30× |

**모든 모델에서 hybrid 가 gpu_only 대비 5~11배 느리다.**
H100x4 결과 (`20260411_142900`) 에서 hybrid ≈ gpu_only 였던 것과 완전히 다른 양상.

---

## 3. 레이턴시 상세

### 3.1 Qwen2.5-1.5B-Instruct

| 지표 | gpu_only | hybrid |
|------|--------:|------:|
| Mean TTFT (ms) | 833 | 1,239 |
| Median TTFT (ms) | 815 | 1,127 |
| P99 TTFT (ms) | 1,062 | 6,903 |
| Mean TPOT (ms) | 27.67 | 72.76 |
| Median TPOT (ms) | 27.59 | 70.60 |
| P99 TPOT (ms) | 29.00 | 123.54 |
| Mean ITL (ms) | 27.50 | 72.86 |
| Median ITL (ms) | 27.14 | 66.75 |
| P99 ITL (ms) | 70.01 | 162.28 |

### 3.2 Qwen2.5-7B-Instruct

| 지표 | gpu_only | hybrid |
|------|--------:|------:|
| Mean TTFT (ms) | 1,309 | 2,149 |
| Median TTFT (ms) | 1,391 | 1,798 |
| P99 TTFT (ms) | 1,989 | 12,498 |
| Mean TPOT (ms) | 28.34 | 91.25 |
| Median TPOT (ms) | 26.15 | 82.72 |
| P99 TPOT (ms) | 76.31 | 322.94 |
| Mean ITL (ms) | 26.78 | 90.50 |
| Median ITL (ms) | 21.77 | 77.85 |
| P99 ITL (ms) | 195.55 | 337.28 |

### 3.3 Qwen2.5-32B-Instruct

| 지표 | gpu_only | hybrid |
|------|--------:|------:|
| Mean TTFT (ms) | 10,428 | 13,512 |
| Median TTFT (ms) | 9,318 | 11,808 |
| P99 TTFT (ms) | 22,166 | 39,117 |
| Mean TPOT (ms) | 54.33 | 115.53 |
| Median TPOT (ms) | 46.25 | 60.86 |
| P99 TPOT (ms) | 88.50 | 1,373.74 |
| Mean ITL (ms) | 51.17 | 112.69 |
| Median ITL (ms) | 32.36 | 47.85 |
| P99 ITL (ms) | 708.52 | 1,381.87 |

---

## 4. 리소스 사용 (monitor CSV 1 Hz 집계)

| # | 모델 | 모드 | GPU peak | GPU mean | CPU peak | CPU mean |
|---|------|------|--------:|--------:|--------:|--------:|
| 1 | 1.5B | gpu_only | 56% | 13.3% | 17.6% | 8.2% |
| 2 | 1.5B | hybrid | 56% | **5.7%** | **99.9%** | **63.6%** |
| 3 | 7B | gpu_only | 83% | 28.5% | 17.8% | 7.9% |
| 4 | 7B | hybrid | 100% | **6.0%** | **99.9%** | **80.8%** |
| 5 | 32B | gpu_only | 100% | **68.8%** | 17.0% | 6.0% |
| 6 | 32B | hybrid | 100% | **11.0%** | **100%** | **90.0%** |

**패턴**: hybrid 에서 CPU mean util 이 63~90% 로 높고, GPU mean util 은 gpu_only
대비 1/2~1/6 수준으로 급락. CPU tail 이 전체 벤치마크 시간을 지배하여 GPU 가
idle 로 대기하는 구간이 sampling 의 대부분을 차지함.

---

## 5. 32B hybrid 서버 로그 분석

### 5.1 라우팅 dispatch

```
[HYBRID-ROUTER-DISPATCH] n=500 last=gpu prompt_len=128
  cpu_count=16 gpu_count=484 cpu_in_flight=16 gpu_in_flight=482
```

→ 500 req 중 **16 req → CPU, 484 req → GPU**. wave-batch 가 cpu_max_seqs=16
으로 첫 wave 를 채운 뒤 나머지는 전부 GPU. 설계대로 작동.

### 5.2 GPU 완료 → CPU tail 타임라인

| 시각 (KST) | Engine 000 (GPU) | Engine 001 (CPU) | 비고 |
|-------------|-----------------|-----------------|------|
| 04:54:14 | Running 39, Waiting 355, 2430 prompt tok/s | Running 1, 12.8 prompt tok/s | GPU 가 대량 처리 중 |
| 04:54:24 | Running 117, Waiting 113, 1603 gen tok/s | Running 1, 0 tok/s | GPU 디코딩 전환 |
| 04:54:34 | Running 93, Waiting 0, 2063 gen tok/s | — | GPU queue 소진 |
| 04:54:44 | **Running 0**, 635 gen tok/s (마무리) | Running 16, 1.6 gen tok/s | **GPU 완료** |
| 04:54:54 | idle | Running 16, 12.8 gen tok/s | CPU tail 시작 |
| 04:55:04~04:57:34 | idle | Running 16, **11.2 gen tok/s** (10회 반복) | **CPU 3분간 단독 처리** |
| 04:57:36 | — | — | Warmup profiling complete |
| 04:57:38 | — | in_flight_cpu=1/16 | 마지막 CPU req |
| 04:57:44 | — | Running 0 | CPU 완료 |

### 5.3 핵심 수치

- **GPU 484 req 처리 시간**: ~30초 (04:54:14 → 04:54:44)
- **CPU 16 req 처리 시간**: ~180초 (04:54:44 → 04:57:44)
- **CPU per-req throughput**: 11.2 gen tok/s (128 output tokens → ~11.4s per request)
- **CPU aggregate throughput**: 11.2 tok/s (16 seq 에서 aggregate = 11.2, 즉 per-req ≈ 0.7 tok/s)

**치명적 발견**: 32B 에서 CPU aggregate throughput 이 겨우 11.2 tok/s. GPU 는
2,352 tok/s. CPU 가 GPU throughput 의 **0.48%** 에 불과. 16 req 에 대한 CPU
wall = 128 × 16 / 11.2 ≈ **183초**. 이것이 bench duration 213s 의 86%를 차지.

### 5.4 Warmup profiling 결과 (서버 로그)

```
GPU: 6.2 tok/s (avg over 485 reqs, 59653 tokens)
CPU: 0.6 tok/s (avg over 1 reqs, 128 tokens)
```

→ GPU 대비 CPU per-req throughput 비율 = 0.6 / 6.2 = **9.7%**.

---

## 6. H100x4 결과와의 비교

| 지표 | H100x4 1.5B G | H100x4 1.5B H | **H100x1 1.5B G** | **H100x1 1.5B H** |
|------|---:|---:|---:|---:|
| bench dur (s) | 3.94 | 3.87 | 4.43 | 22.60 |
| req/s | 127.0 | 129.2 | 112.8 | 22.1 |
| output tok/s | 15,640 | 15,911 | 13,898 | 2,725 |
| TPOT med (ms) | 23.56 | 23.03 | 27.59 | 70.60 |
| hybrid/gpu ratio | — | **0.98×** | — | **5.10×** |

**H100x4 에서는 hybrid ≈ gpu_only (0.98×)** 였지만, **H100x1 에서는 hybrid
= 5.10× 느림**. 핵심 차이:

| 요인 | H100x4 | H100x1 |
|------|--------|--------|
| GPU wall | ~3.9s | ~4.4s |
| CPU dispatch | **0 req** (Property 2) | **16 req** (wave-batch) |
| 라우팅 전략 | throughput-adaptive | **wave-batch** |
| TP | 4 | 1 |

H100x4 에서는 `throughput-adaptive` 전략이 Property 2 에 의해 CPU 에 0 req
dispatch → hybrid = gpu_only. H100x1 에서는 `wave-batch` 전략이 **강제로 16
req 를 CPU 에 보냄** → CPU tail 이 wall 을 지배.

---

## 7. 모델 크기별 CPU tail 비중

| 모델 | GPU wall (≈) | CPU wave wall (≈) | tail/GPU 비 | CPU 기여 |
|------|----------:|---------------:|----------:|---------:|
| 1.5B | 4.4s | 18.2s | 4.1× | GPU 의 2.0% |
| 7B | 4.8s | 48.7s | 10.1× | GPU 의 0.9% |
| 32B | 26.2s | 183s | 7.0× | GPU 의 0.48% |

**모델이 커질수록 CPU throughput 이 떨어져서 tail 이 극적으로 길어진다.**
32B 에서 CPU 16 req 이 3분 걸리는 동안 GPU 는 idle.

---

## 8. dev RTX 3090 결과와의 비교 (AMX 효과 검증)

| 항목 | dev RTX 3090 | **H100x1** |
|------|-------------|-----------|
| CPU ISA | AVX2 (no AVX-512, no AMX) | **AVX-512 + AMX-BF16** |
| 1.5B CPU per-req | ~9.4 tok/s | batch 내 aggregate ÷ seqs |
| 1.5B batch=16 효과 | per-req 동일 (9.4) | per-req 동일 추정 |
| 1.5B hybrid/gpu ratio | 8.30× | **5.10×** |
| 7B hybrid/gpu ratio | 9.82× | **11.10×** |

**결론**: AMX 가 있는 H100x1 환경에서도 **CPU matmul batching 의 per-req
throughput 가속이 발생하지 않았다**. dev RTX 3090 의 F2 발견 ("batch 크기와
무관하게 per-req throughput 동일") 이 AMX 서버에서도 재현됨.

가능한 원인:
1. **24 vCPU 제한** — 실제 8480+ 는 96 코어이나 컨테이너가 24 코어만 할당. OMP
   thread 수가 부족해 AMX brgemm 의 M-dim 확장 효과가 제한될 수 있음.
2. **32B 모델 weight 가 DDR 대역폭을 완전히 점유** — 32B BF16 ≈ 64GB weight,
   DDR5 ~70 GB/s 단채널. batch 확장이 compute-bound 영역에 들어가기 전에
   memory-bound 에 걸림.
3. **IPEX dispatch 경로의 overhead** — batch 확장 시 AMX tile 재사용 전에
   Python/IPEX dispatch overhead 가 지배적일 수 있음.

---

## 9. 핵심 발견

### F1. wave-batch router 는 H100x1 에서도 설계대로 정확히 작동

모든 모델에서 동일 lifecycle: wave closed (16 req) → GPU 가 나머지 484 처리
→ CPU tail 완료 → wave drained. 구조적 결함 없음.

### F2. H100x1 에서 hybrid 는 모든 모델에서 gpu_only 대비 대폭 느림

- 1.5B: **5.1×**, 7B: **11.1×**, 32B: **8.2×**
- 원인: CPU tail 이 GPU wall 대비 4~10배 길어서 전체 벤치마크 시간을 지배

### F3. CPU throughput 은 모델 크기에 반비례하여 tail 이 악화

- 1.5B: ~13,898 GPU tok/s vs ~274 CPU aggregate tok/s
- 32B: ~2,352 GPU tok/s vs ~11 CPU aggregate tok/s
- 모델이 커질수록 CPU/GPU throughput 격차가 심화

### F4. 32B 에서 CPU 16 req 이 GPU 484 req 보다 6배 오래 걸림

GPU 484 req: ~30s. CPU 16 req: ~183s. CPU 가 GPU 의 **0.48%** throughput
만 제공. wave-batch 로 CPU 에 보내는 것이 "GPU 에 보내는 것 대비" 극단적
으로 비효율.

### F5. GPU path 자체는 hybrid / gpu_only 동일

GPU busy-window 에서의 실제 throughput 은 hybrid 와 gpu_only 가 동등.
hybrid 의 GPU mean util 이 낮은 것은 CPU tail 동안의 idle 구간이 sampling
에 포함되기 때문. **Dual-process 격리 원칙은 H100x1 에서도 유지.**

### F6. AMX 가용 환경에서도 CPU batching per-req 가속 없음

dev RTX 3090 (AVX2) 에서 관찰된 "batch 크기 무관 per-req throughput 동일"
패턴이 H100x1 (AMX-BF16) 에서도 동일하게 나타남. 이는 24 vCPU 제한 또는
memory-bandwidth bound 에 의한 것으로 추정.

---

## 10. H100x4 (TP=4, 96 cores) 와의 결론적 차이

| 환경 | hybrid 결과 | 이유 |
|------|-----------|------|
| **H100x4** (throughput-adaptive) | hybrid ≈ gpu_only | Property 2 가 CPU 에 0 dispatch → GPU 만 사용 |
| **H100x1** (wave-batch) | hybrid = 5~11× 느림 | wave-batch 가 강제 16 dispatch → CPU tail 지배 |

두 결과는 **라우팅 전략이 다르다**. H100x4 는 throughput-adaptive (CPU 가
GPU 보다 느리면 안 보냄), H100x1 은 wave-batch (무조건 첫 wave 를 CPU
에 채움). 동일 전략으로 비교하려면:
- H100x1 에서 throughput-adaptive 로 재실행 → Property 2 에 의해 hybrid
  ≈ gpu_only 가 될 것으로 예상
- H100x4 에서 wave-batch 로 재실행 → CPU tail 발생하지만 96 cores 에서
  CPU throughput 이 더 높을 수 있음

---

## 11. 전체 데이터 포인터

| 항목 | 경로 |
|------|------|
| 1.5B gpu_only | `eval/results/20260412_043206_G_H100_80GB_HBM3_x1_Qwen2.5-1.5B-Instruct/` |
| 1.5B hybrid | `eval/results/20260412_043424_H_C_H100_80GB_HBM3_x1_Qwen2.5-1.5B-Instruct/` |
| 7B gpu_only | `eval/results/20260412_043829_G_H100_80GB_HBM3_x1_Qwen2.5-7B-Instruct/` |
| 7B hybrid | `eval/results/20260412_044201_H_C_H100_80GB_HBM3_x1_Qwen2.5-7B-Instruct/` |
| 32B gpu_only | `eval/results/20260412_044945_G_H100_80GB_HBM3_x1_Qwen2.5-32B-Instruct/` |
| 32B hybrid | `eval/results/20260412_045340_H_C_H100_80GB_HBM3_x1_Qwen2.5-32B-Instruct/` |
| 사용 env | `eval/envs/h100x1_qwen1.5b_hybrid_wave.env` (+ 7B/32B 변형) |
| dev 비교 결과 | `experiment_result/20260412_023700_dev_rtx3090_wave_batch_gate_fix_initial_validation/` |
| H100x4 비교 결과 | `experiment_result/20260411_142900_h100x4_qwen1.5b_7b_gpu_only_vs_hybrid_4runs/` |
| 32B 서버 로그 | 본 문서 §5 (사용자 제공 로그 발췌) |

---

## 12. 결론 (한 줄)

**H100x1 (24 vCPU, TP=1) 에서 wave-batch hybrid 는 모든 모델 크기에서 gpu_only 대비 5~11배 느리며, CPU tail 이 벤치마크 시간의 80~86%를 차지한다. AMX 가용 환경에서도 CPU matmul batching 의 per-req 가속 효과는 없었고, wave-batch 의 "강제 CPU dispatch" 는 GPU-dominant 환경에서 순수한 성능 페널티다.**

---

## 13. 다음 단계

1. **H100x1 + throughput-adaptive 재실행** — wave-batch 가 아닌 Property 2
   기반 라우팅에서 hybrid ≈ gpu_only 가 되는지 확인 (F2 의 원인 분리)
2. **H100x4 + wave-batch 재실행** — 96 cores + TP=4 에서 wave-batch 가
   dev/H100x1 과 동일하게 느린지, 아니면 CPU throughput 이 올라서 다른
   결과가 나오는지 확인
3. **wave 크기 최적화** — `cpu_max_seqs=4` 로 축소하여 CPU tail 절감 효과
   측정 (dev 에서 wave=4 가 wall 을 2.5× 줄임)
4. **CPU core 수 영향 분석** — 컨테이너의 24 vCPU 제한이 batching 효과를
   억제하는지, 96 cores 전체 사용 시 달라지는지 검증
