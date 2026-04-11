# H100x4 Qwen2.5-32B-Instruct — gpu_only / hybrid baseline + 1.5B/7B/32B scaling

`20260411_145900_h100x4_qwen32b_gpu_only_vs_hybrid_baseline`

> 라우팅 fix (commit `3f528123e`) 가 적용된 상태에서 32B 의 첫 baseline.
> 1.5B / 7B / 32B 동일 shape (500 req × 128/128) 으로 직접 비교 가능.

## 1. 32B 결과

| 모델 | 모드 | bench dur (s) | wall (s) | req/s | output tok/s | TPOT (ms) | TTFT (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| 32B | gpu_only | 7.01 | 17.35 | 71.4 | 8,774 | 41.82 | 1,746 |
| 32B | **hybrid** | 7.09 | 17.21 | 70.5 | 8,674 | 41.93 | 1,802 |

| 지표 | hybrid / gpu_only |
|---|---:|
| bench duration | 1.011× |
| TPOT | 1.003× |
| output tok/s | 0.989× |
| req throughput | 0.987× |

→ **32B 도 hybrid ≈ gpu_only** (전부 ±2% 노이즈 범위). 1.5B/7B 와 동일한
패턴 — Property 2 의 expected-finish gate 가 32B 에서도 모든 요청을 GPU 로
보내고 있음.

라우터 분배 (server stdout 의 `[HYBRID-ROUTER-STATS]` 최종):

```
finished=501 GPU=20.2 tok/s (501 reqs), CPU=0.0 tok/s (0 reqs),
cpu_ratio=0.0%, in_flight_cpu=0/1, adaptive_slots=1
```

→ **501 dispatched, 501 GPU, 0 CPU** — 32B 도 H100 GPU 가 CPU 보다 압도적
으로 빠른 영역이라 expected-finish 비교에서 CPU 가 절대 못 이김.

## 2. 1.5B → 7B → 32B scaling 비교 (전부 동일 shape, 동일 fix)

| 모델 | 모드 | bench dur (s) | TPOT (ms) | output tok/s | GPU mean util |
|---|---|---:|---:|---:|---:|
| **1.5B** | gpu_only | 3.94 | 23.56 | 15,640 | 13.9 % |
| 1.5B | hybrid | 3.87 | 23.03 | 15,911 | 13.1 % |
| **7B** | gpu_only | 4.02 | 24.73 | 15,492 | 19.2 % |
| 7B | hybrid | 3.93 | 23.04 | 15,984 | 13.8 % |
| **32B** | gpu_only | 7.01 | 41.82 | 8,774 | **42.6 %** |
| 32B | hybrid | 7.09 | 41.93 | 8,674 | **43.3 %** |

### 2.1 Scaling 추이

| 지표 | 1.5B → 7B | 7B → 32B | 1.5B → 32B |
|---|---:|---:|---:|
| TPOT 비율 | 1.05× | **1.69×** | 1.78× |
| output tok/s 비율 | 0.99× | **0.57×** | 0.56× |
| bench duration 비율 | 1.02× | **1.74×** | 1.78× |
| GPU mean util 비율 | 1.38× | **2.22×** | 3.06× |

**관찰 1**: 1.5B → 7B 는 TPOT/throughput 거의 동일. **GPU 가 1.5B/7B 둘
다에서 underutilized** (mean 14~19%) 라서 모델 크기 차이가 wall 에 거의
영향 없음. compute 가 아닌 batch dispatch / launch overhead 가 지배.

**관찰 2**: 7B → 32B 는 TPOT 가 **1.69× 증가** (24.7 → 41.8 ms). 모델 크기
~4.5× 인데 step time 1.69× 증가는 compute-bound 의 약한 신호. GPU mean util
이 19% → 43% 로 2.2× 증가 — **32B 가 처음으로 GPU 를 의미 있게 부하**.

**관찰 3**: 32B 에서도 GPU mean util 43% 는 여전히 sub-saturation. **64 GB
weight + 4 GPU TP 의 batching 으로 batch~480 시퀀스 (KV cache 약 200 GB
에 fit) 가 들어가지만 step time 자체가 짧아서 (~40ms) GPU 가 여유 있음**.
즉 H100x4 는 32B 에도 over-provisioned. paper 의 ninja gap 측정을 위해서는
**모델을 더 키우거나 (70B+), batch 를 더 늘리거나, 더 긴 context 를 써야**
함.

### 2.2 1.5B → 32B 의 wall scaling 의미

1.5B 가 7B 와 거의 같은 wall 인 것은 **H100x4 가 1.5B/7B 에 너무 강해서**
라는 점 — `T = launch_overhead + compute(model_size, batch)` 에서 launch
오버헤드가 compute 보다 큰 영역. 32B 에서야 처음으로 compute 가 launch 를
넘어서며 wall 이 ~1.78× 늘어남.

이것이 paper 의 1.5B/7B/32B scaling section 에 들어갈 baseline. **"H100x4
는 32B 까지도 GPU saturated 가 아니다"** 가 핵심 메시지가 됨.

## 3. 32B 의 ninja gap 발현 가능성 (현재 결과 + 분석)

| 후보 | 32B 에서 효과 예측 | 근거 |
|---|---|---|
| **A1 spec decode (Qwen2.5-1.5B drafter)** | **~30~50% TPOT 단축 가능** | 32B GPU TPOT 41.82 ms 가 spec decode 로 1.5~2× 단축 → ~22~28 ms. 32B verification step ~40ms, draft 1.5B ~10~15ms 로 충분히 GPU 를 follow 가능. **본 환경에서 첫 ninja gap 후보** |
| **A2 KV cache CPU offload** | 0% (현재 KV 가 batch 한계 아님) → **batch ↑ 시 1.5~2×** | 현재 batch ~480 이 GPU mean util 43% 에서 만족. KV 가 ~200 GB 사용 중 GPU HBM 240 GB 여유. KV 를 CPU 로 빼면 batch 1500+ 가능 → GPU saturated → 그때 ninja gap |
| **A3 long-context P/D** | 32K context 워크로드에서만 | 본 baseline 은 input 128 이라 무관 |
| **A4 AMX-INT8 활성화** | 단독 0% → A1 의 곱셈 인자 | INT8 path 가 없어서 측정 불가 |

**결론**: 32B baseline 자체에서는 hybrid penalty 0 + ninja gap 0 이 정상.
**ninja gap 을 정량적으로 보이려면 spec decode (A1) 구현이 필수**. 32B 가
1.5B/7B 보다 spec decode 의 효과가 더 클 것으로 예측되는 이유:

1. **GPU TPOT 가 더 큼** (41.82 ms vs 23.56 ms) — verify step 이 더 비싸도
   draft 가 그 안에 충분히 K=4~8 토큰 propose 가능.
2. **Token 분포가 더 다양** — 32B 의 logit 이 더 sharp 할 가능성 있고 (큰
   모델 + instruct tuning), accept rate 가 더 높을 가능성.
3. **GPU 가 underutilized** — verify step 의 compute 여유가 있어 spec decode
   의 batched verify 가 free.

## 4. 부수 결과 — 부팅 시간

| 모델 | gpu_only ready | hybrid ready |
|---|---:|---:|
| 1.5B | ~10~80 s | ~80~100 s |
| 7B | ~90 s | ~100 s |
| **32B** | **~100 s** | **~150 s** |

32B 는 weight 64 GB 로딩 + CUDA graph capture (TP=4 에서 graph 0.72 GB ×
4 = 2.9 GB) 때문에 부팅이 ~100~150 초. env 의 `SERVER_READY_TIMEOUT=1800`
이 충분.

## 5. 데이터 포인터

| 항목 | 경로 |
|---|---|
| 32B gpu_only | `eval/results/20260411_145417_G_H100_80GB_HBM3_x4_Qwen2.5-32B-Instruct/` |
| 32B hybrid | `eval/results/20260411_145805_H_C_H100_80GB_HBM3_x4_Qwen2.5-32B-Instruct/` |
| 사용한 env | `eval/envs/h100x4_qwen32b_hybrid.env` |
| 1.5B/7B 직전 4-run 결과 | `experiment_result/20260411_142900_h100x4_qwen1.5b_7b_gpu_only_vs_hybrid_4runs/` |
| 회귀 추적 + 라우팅 fix | `experiment_result/20260411_141500_h100x4_qwen1.5b_routing_regression_root_cause_fix/` |
| Ninja gap 전략 노트 | `experiment_result/20260411_143500_h100x4_isa_verification_and_ninja_gap_strategy/` |

## 6. 다음 단계

1. **A1 spec decode (CPU drafter) 구현 + 32B 측정**. 본 baseline 의 32B
   TPOT 41.82 ms 를 ~22~28 ms 로 줄일 수 있는지가 ninja gap 의 정량 증명.
2. **70B (Llama-3.3-70B) baseline 측정** — H100x4 에 weight 140 GB 로 KV
   에 더 큰 압력. KV offload demo 의 후보.
3. **Long-context (4K → 16K → 32K) shape 으로 32B 재측정** — input length
   가 늘면 prefill 비용이 늘어 GPU 가 saturated 될 가능성, A3 P/D
   disaggregation 효과 관찰 가능.
4. **64K context, batch 100** 같은 KV 한계 워크로드로 A2 KV offload demo.
