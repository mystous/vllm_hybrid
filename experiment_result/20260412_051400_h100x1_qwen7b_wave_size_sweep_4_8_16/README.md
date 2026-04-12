# H100x1 Qwen2.5-7B-Instruct — wave 크기 sweep (4 / 8 / 16) vs gpu_only

`20260412_051400_h100x1_qwen7b_wave_size_sweep_4_8_16`

> wave-batch 의 `cpu_max_seqs` (= wave 크기) 를 16 → 8 → 4 로 줄이면
> CPU tail 이 얼마나 줄어드는지 H100x1 에서 7B 로 실측.
> gpu_only baseline 포함 총 4-run 비교.

**날짜**: 2026-04-12 (KST)

---

## 1. 환경

| 항목 | 값 |
|------|-----|
| **CPU** | Intel Xeon Platinum 8480+ (1S, 24 cores, 1T/C = 24 vCPU) |
| **ISA** | AVX-512F/BW/VL, AVX-512 VNNI, AMX-BF16/INT8 |
| **Memory** | 241,600 MB DDR5, 1 NUMA node |
| **GPU** | 1 × NVIDIA H100 80GB HBM3 |
| **Software** | vLLM 0.1.dev8475, PyTorch 2.9.0+cu130, CUDA 13.0 |
| **Benchmark** | 500 req, random, input 128 / output 128, request_rate=inf |
| **Routing** | wave-batch, cpu-first |
| **TP** | 1 |

---

## 2. 핵심 결과

| wave | bench (s) | wall (s) | req/s | output tok/s | TPOT med (ms) | TTFT mean (ms) | P99 TTFT (ms) |
|-----:|----------:|--------:|------:|------------:|--------------:|---------------:|--------------:|
| gpu_only | **4.82** | **16.55** | **103.83** | **12,953** | **26.15** | **1,309** | **1,989** |
| 16 | 53.52 | 65.43 | 9.34 | 1,167 | 82.72 | 2,149 | 12,498 |
| 8 | 49.19 | 61.34 | 10.16 | 1,270 | 80.53 | 1,952 | 9,859 |
| 4 | **47.91** | **59.26** | **10.44** | **1,304** | **78.82** | **1,597** | **2,366** |

### 2.1 hybrid 내 wave 크기 비율 (wave=16 기준)

| 지표 | w=8 / w=16 | w=4 / w=16 |
|------|----------:|---------:|
| bench duration | 0.919× | **0.895×** |
| wall time | 0.937× | **0.906×** |
| req throughput | 1.088× | **1.117×** |
| output tok/s | 1.088× | **1.117×** |
| TPOT median | 0.974× | **0.953×** |
| TTFT mean | 0.908× | **0.743×** |
| P99 TTFT | 0.789× | **0.189×** |

### 2.2 hybrid vs gpu_only 비율

| 지표 | w=16 / G | w=8 / G | w=4 / G |
|------|--------:|-------:|-------:|
| bench duration | 11.10× | 10.21× | **9.94×** |
| req throughput | 0.090× | 0.098× | **0.101×** |
| TPOT median | 3.16× | 3.08× | **3.01×** |
| TTFT mean | 1.64× | 1.49× | **1.22×** |
| P99 TTFT | 6.28× | 4.96× | **1.19×** |

---

## 3. 레이턴시 상세

### 3.1 TTFT (Time to First Token)

| wave | mean (ms) | median (ms) | p99 (ms) |
|-----:|---------:|-----------:|--------:|
| gpu_only | 1,309 | 1,391 | 1,989 |
| 16 | 2,149 | 1,798 | **12,498** |
| 8 | 1,952 | 1,832 | **9,859** |
| 4 | 1,597 | 1,555 | **2,366** |

**P99 TTFT 에서 wave=4 가 극적으로 개선**: 12,498 → 2,366 ms (5.3× 감소).
wave=4 의 P99 TTFT (2,366 ms) 는 gpu_only (1,989 ms) 와 유사한 수준.

### 3.2 TPOT (Time per Output Token)

| wave | mean (ms) | median (ms) | p99 (ms) |
|-----:|---------:|-----------:|--------:|
| gpu_only | 28.34 | 26.15 | 76.31 |
| 16 | 91.25 | 82.72 | 322.94 |
| 8 | 85.26 | 80.53 | 309.62 |
| 4 | 81.75 | 78.82 | **205.73** |

P99 TPOT 도 wave=4 에서 유의미하게 감소 (323 → 206 ms).

### 3.3 ITL (Inter-Token Latency)

| wave | mean (ms) | median (ms) | p99 (ms) |
|-----:|---------:|-----------:|--------:|
| gpu_only | 26.78 | 21.77 | 195.55 |
| 16 | 90.50 | 77.85 | 337.28 |
| 8 | 84.37 | 72.93 | 302.17 |
| 4 | 80.81 | 73.08 | **248.37** |

---

## 4. 리소스 사용

| wave | GPU peak | GPU mean | CPU peak | CPU mean | 모니터 samples |
|-----:|--------:|--------:|--------:|--------:|-------:|
| gpu_only | 83% | 28.5% | 17.8% | 7.9% | 15 |
| 16 | 100% | 6.0% | 99.9% | 80.8% | 61 |
| 8 | 100% | 5.9% | 99.9% | 79.3% | 57 |
| 4 | 100% | 6.4% | 100% | 78.7% | 55 |

GPU mean util 은 세 wave 크기에서 거의 동일 (~6%). CPU mean util 도
유사 (~79-81%). 이는 GPU 가 처리하는 req 수 (484~496) 가 비슷하고,
CPU 처리 시간이 여전히 벤치마크 wall 을 지배하기 때문.

---

## 5. CPU tail 분석

| wave | GPU req | CPU req | 추정 GPU wall (s) | 추정 CPU tail (s) | tail/GPU 비 |
|-----:|--------:|--------:|------------------:|------------------:|----------:|
| 16 | 484 | 16 | ~4.8 | ~48.7 | 10.1× |
| 8 | 492 | 8 | ~4.8 | ~44.4 | 9.3× |
| 4 | 496 | 4 | ~4.8 | ~43.1 | 9.0× |

CPU tail = bench_duration - GPU wall (≈ gpu_only bench duration).

**wave 크기 4배 축소 (16→4) 에 대해 CPU tail 은 11% 밖에 줄지 않음**
(48.7 → 43.1s). dev RTX 3090 에서 관찰된 "wave 크기에 선형 비례하는 tail
감소" 패턴과 다름.

### 5.1 왜 tail 이 선형으로 줄지 않는가?

dev RTX 3090 1.5B 에서는:
- wave=16: CPU tail ≈ 67s
- wave=4: CPU tail ≈ 17s → **4× 감소** (선형)

H100x1 7B 에서는:
- wave=16: CPU tail ≈ 48.7s
- wave=4: CPU tail ≈ 43.1s → **1.13× 감소** (거의 안 줄어듦)

**원인**: 7B 모델에서 CPU per-req 처리 시간이 매우 길기 때문 (추정
~40-50s per request). wave=4 라도 4 req 를 sequential 하게 처리하면
~40s+ 가 걸린다. 즉 **wave 크기를 줄여도 1 req 의 latency 가 이미
~40s 이상이라 바닥이 있다**.

비교:
- 1.5B CPU per-req: ~128 tok / 9.4 tok/s ≈ **13.6s**
- 7B CPU per-req: ~128 tok / 2.3 tok/s ≈ **55.6s**

7B 에서는 단 1 req 만 CPU 에 보내도 ~55s 의 tail 이 생긴다.
wave=4 (43.1s tail) 가 1-req 추정 (55.6s) 보다 짧은 것은 4 req 를
batch 로 parallel 처리하기 때문이지만, 개선폭이 제한적.

---

## 6. 핵심 발견

### F1. wave 크기 축소는 throughput 과 레이턴시를 개선하지만 한계가 있다

| 지표 | w=16 → w=4 개선 |
|------|---------------:|
| req throughput | +11.7% |
| bench duration | -10.5% |
| TTFT mean | -25.7% |
| **P99 TTFT** | **-81.1%** |
| TPOT median | -4.7% |
| P99 TPOT | -36.3% |

P99 TTFT 개선이 가장 극적 (12.5s → 2.4s). 이는 CPU 에 보내지는 req 수가
16→4 로 줄면서 CPU-routed request 의 TTFT outlier 가 감소하기 때문.

### F2. 그러나 bench duration 은 여전히 gpu_only 대비 ~10배

wave=4 (47.9s) 는 여전히 gpu_only (4.8s) 의 **9.94×**. wave 크기를 1로
줄이더라도 7B CPU per-req latency (~55s) 가 GPU wall (~4.8s) 보다 11× 느리므로,
hybrid 가 gpu_only 를 이길 수 없는 구조.

### F3. P99 TTFT 는 wave=4 에서 gpu_only 에 근접

wave=4 P99 TTFT = 2,366 ms vs gpu_only P99 TTFT = 1,989 ms.
차이가 **1.19×** 로 좁혀짐. 이는 CPU 에 dispatch 되는 req 이 적어서
GPU 경로의 대부분 요청이 gpu_only 와 동일한 TTFT 를 경험하기 때문.

### F4. GPU 처리 경로는 wave 크기와 무관하게 동일

세 wave 크기에서 GPU mean util (~6%), GPU peak (100%) 이 거의 동일.
GPU 에 dispatch 되는 req 수 (484~496) 차이가 미미하고, GPU 자체의 처리
속도는 wave 크기에 의해 영향 받지 않음. Dual-process 격리 유지.

---

## 7. dev RTX 3090 1.5B wave sweep 결과와의 비교

| 환경 | w=16 wall | w=4 wall | 개선 | 이유 |
|------|--------:|--------:|-----:|------|
| dev 1.5B | 85.93s | 34.89s | **2.46×** | CPU per-req 13.6s → tail 이 wave 에 선형 |
| **H100x1 7B** | 65.43s | 59.26s | **1.10×** | CPU per-req 55.6s → 바닥이 높아서 wave 축소 효과 미미 |

**결론**: wave 크기 축소의 효과는 **CPU per-req latency 가 짧은 모델** (1.5B)
에서 극적이고, **CPU per-req latency 가 긴 모델** (7B) 에서는 제한적.

---

## 8. 전체 데이터 포인터

| 항목 | 경로 |
|------|------|
| 7B gpu_only | `eval/results/20260412_043829_G_H100_80GB_HBM3_x1_Qwen2.5-7B-Instruct/` |
| 7B hybrid wave=16 | `eval/results/20260412_044201_H_C_H100_80GB_HBM3_x1_Qwen2.5-7B-Instruct/` |
| 7B hybrid wave=8 | `eval/results/20260412_051035_H_C_H100_80GB_HBM3_x1_Qwen2.5-7B-Instruct/` |
| 7B hybrid wave=4 | `eval/results/20260412_051353_H_C_H100_80GB_HBM3_x1_Qwen2.5-7B-Instruct/` |
| Env wave=16 | `eval/envs/h100x1_qwen7b_hybrid_wave.env` |
| Env wave=8 | `eval/envs/h100x1_qwen7b_hybrid_wave_8re.env` |
| Env wave=4 | `eval/envs/h100x1_qwen7b_hybrid_wave_4re.env` |
| 관련 6-run 보고서 | `experiment_result/20260412_050600_h100x1_qwen1.5b_7b_32b_wave_batch_scaling/` |
| dev 1.5B wave sweep | `experiment_result/20260412_023700_dev_rtx3090_wave_batch_gate_fix_initial_validation/` |

---

## 9. 결론

**H100x1 7B 에서 wave 크기를 16→4 로 줄이면 P99 TTFT 가 81% 개선 (12.5s→2.4s) 되고 throughput 이 12% 향상되지만, bench duration 은 여전히 gpu_only 의 ~10배다. 7B 의 CPU per-req latency (~55s) 가 GPU wall (~4.8s) 대비 11배 느려서, wave 크기를 아무리 줄여도 hybrid 가 gpu_only 를 이길 수 없는 구조적 한계가 이 환경에 존재한다.**

---

## 10. 다음 단계

1. **1.5B 에서 동일 wave sweep (4/8/16)** — 7B 대비 CPU per-req 가 4배 빠르므로 wave 축소 효과가 더 클 것으로 예상
2. **wave=1 실험** — CPU 에 1 req 만 보내면 overhead 가 최소. hybrid 가 gpu_only + 1 req 의 tail 로 수렴하는지 확인
3. **H100x4 (96 cores) 에서 wave sweep** — 24 vCPU 제한이 풀리면 CPU per-req throughput 이 올라가서 다른 결과가 나올 수 있음
