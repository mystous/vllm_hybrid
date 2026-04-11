# H100x4 Qwen2.5-1.5B — GPU-only vs Hybrid 비교 (최신 run)

`20260411_130959_h100x4_qwen1.5b_gpu_vs_hybrid_latest_compare`

> 대상 env: `eval/envs/h100x4_qwen1.5b_hybrid.env`
> 비교한 가장 최근 2개 run (동일 env, 같은 5 분 내 수행):
> - GPU-only: `eval/results/20260411_125604_G_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/`
> - Hybrid (H_C): `eval/results/20260411_125824_H_C_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/`

## 1. 실험 조건 (동일)

| 항목 | 값 |
|---|---|
| 모델 | Qwen/Qwen2.5-1.5B-Instruct |
| 하드웨어 | H100 80GB HBM3 × 4, Xeon 8480+ (96C/1 NUMA), 967 GB DDR5 |
| 소프트웨어 | vLLM 0.1.dev8475+g78fa48cb8, torch 2.9.0+cu130, CUDA 13.0 |
| TP | 4 |
| NUM_PROMPTS / input / output | 500 / 128 / 128 |
| Request rate | inf (burst 1.0) |
| Routing (hybrid) | `throughput-adaptive`, `cpu-first`, warmup=10, prefill-threshold=1024 |
| `num_cpu_engines` / `cpu_max_seqs` / `threads` / `kvcache_gb` | 0/0/0/0 (auto) |
| TRACE | `VLLM_HYBRID_TRACE=0`, `TRACE_EVERY=0` (silent) |

## 2. 핵심 결과

| 지표 | GPU-only | Hybrid | Hybrid / GPU-only |
|---|---:|---:|---:|
| benchmark duration (s) | **3.64** | **44.64** | **12.3×** |
| wall time (s) | 12.89 | 93.08 | 7.22× |
| request throughput (req/s) | 137.40 | 11.20 | 0.082× |
| output throughput (tok/s) | 16,924.8 | 1,379.6 | 0.082× |
| total token throughput (tok/s) | 34,429.1 | 2,806.4 | 0.082× |
| mean TTFT (ms) | 727.2 | 1,291.9 | 1.78× |
| p99 TTFT (ms) | 904.9 | 1,721.1 | 1.90× |
| mean TPOT (ms) | **22.10** | **60.34** | **2.73×** |
| median TPOT (ms) | 22.14 | 59.49 | 2.69× |
| p99 TPOT (ms) | 22.58 | 60.82 | 2.69× |
| mean ITL (ms) | 22.07 | 60.10 | 2.72× |
| p99 ITL (ms) | 46.08 | 145.06 | 3.15× |

→ 이 run 에서는 hybrid 가 **gpu-only 대비 전면적으로 손해** (throughput 0.082×, wall 7.2×).

## 3. 리소스 사용 프로파일 (monitor CSV)

| 지표 | GPU-only (11 rows) | Hybrid (83 rows) |
|---|---|---|
| GPU avg util (4-GPU mean) peak / mean | 60% / 13.7% | 49% / 1.4% |
| GPU avg power (4-GPU mean) peak / mean | 1022 W / 647 W | 847 W / 571 W |
| GPU0 mem used | 74.4 GB | 74.4 GB |
| CPU avg util peak / mean | 12.0% / 6.7% | **96.8% / 84.7%** |
| CPU busy rows (≥20%) | 0 / 11 | **76 / 83** |

관찰:
- GPU-only: CPU 사실상 idle, GPU 가 짧고 강하게 달리고 3.64 s 에 완료.
- Hybrid: **CPU 가 run 대부분 구간 80~97% saturated**. 반대로 GPU mean util 은 1.4% 로 사실상 놀았다. 전력도 GPU 쪽이 더 내려가 있다 (647 → 571 W).
- 즉 이 run 은 routing 이 **거의 전적으로 CPU 경로**로 흘렀고, 그 결과 CPU BW/compute 가 직렬 병목이 되어 전체 wall 을 끌어올렸다.

## 4. 기대치와의 불일치 — "왜 throughput-adaptive 가 GPU 로 회피하지 못했는가"

직전 성공 대조군인 `experiment_result/20260411_085801_h100x4_qwen1.5b_thro_adaptive_500/` 에서는
같은 `throughput-adaptive + cpu-first + silent` 조합으로 hybrid wall ≈ gpu-only wall (ratio ≈ 1.01×)
을 달성한 적이 있다. 이번 125824 run 은 같은 env 를 썼는데도 hybrid 가 CPU 쪽으로 쏠려 ×7.2 slowdown.

가설 후보 (본 리포트에서 **확정 아님**, 다음 실험 필요):

1. **Throughput-adaptive EMA 상태가 cold** — warmup 10 건 이후에도 EMA 가 CPU 우위로 시작해
   초반 선택이 CPU 에 고착되는 케이스. cpu-first 의존도가 실제보다 강하게 작동했을 가능성.
2. **Prefill threshold 1024** + input 128 조합 — prefill 은 CPU 제외이므로 decode 쪽이 전량 router
   판단 대상인데, CPU decode 가 "1 토큰 latency"만 보면 순간적으로 값싸 보여 cpu-first 가 계속 CPU 로
   보낸다.
3. **CapacityAwareRouter 의 cpu_in_flight 한계** — `cpu_max_seqs=1` (NUMA=1 → engine 1개) 임에도
   불구하고 decode step 단위로는 여러 요청이 빠르게 CPU engine 을 통과하며 잔류 시간이 작게 보여
   router 가 계속 CPU 로 보냈다.
4. **085801 과의 diff** — 해당 run 당시 코드/patch 상태와 125824 의 코드 상태 사이에 라우팅 관련
   변경이 있었을 가능성. 최근 commit `9bccbe651 silent per-req/per-call stdout + H100 experiment env cleanup`
   이후 env 는 silent 유지 중이므로 stdout 자체는 이번 원인에서 배제된다 (GPU-only TPOT 가 22.1 ms 로
   정상이고, 125604 가 직접 그 증거).

→ **재현성 확인 필요**: 같은 env 로 3회 재실행 + `HYBRID_STATS_LOG_INTERVAL=50` 을 INFO 레벨로
   승격시켜 라우터가 몇 건을 CPU/GPU 로 보냈는지 직접 카운트해야 결론 가능.

## 5. 정상 여부 판정 (이 run 에 한해서)

| 지표 | GPU-only (125604) | 정상 판단 |
|---|---|---|
| TPOT 22.1 ms | ✓ 과거 H100 085801 GPU-only (~22 ms) 와 일치 | 정상 |
| wall 12.89 s (500 req × bench 3.64 s + 서버 부팅/측정 overhead) | ✓ 기대 범위 | 정상 |

| 지표 | Hybrid (125824) | 정상 판단 |
|---|---|---|
| TPOT 60 ms (× GPU-only 의 2.69배) | ✗ 085801 에서는 hybrid TPOT ≈ gpu-only TPOT 이어야 함 | **비정상** |
| CPU mean util 84.7% for 76/83 rows | ✗ throughput-adaptive + cpu-first 는 슬로우 루프 1~2회 후 GPU 로 쏠려야 함 | **비정상** |
| GPU mean util 1.4% | ✗ hybrid 는 GPU 를 주 경로로 써야 함 | **비정상** |

→ 이 run 의 hybrid 는 **이 env 의 설계 의도 (hybrid wall ≈ gpu-only wall) 를 만족하지 못했다**.
   env 파일은 정상, 코드 쪽의 router 동작에 regression 이 있을 가능성이 높다.

## 6. 다음 액션

1. `h100x4_qwen1.5b_hybrid.env` 로 동일 실험을 **2~3 회 재실행**해 125824 가 1회 fluctuation 인지
   재현 가능한 regression 인지 확인.
2. 재현되면 `HYBRID_STATS_LOG_INTERVAL` 을 활성화해 router 의 CPU/GPU 카운트 로그를 직접 수집.
3. 같은 env 에서 `HYBRID_ROUTING_STRATEGY=capacity` / `gpu-first` 로 바꿨을 때의 거동 차이를 체크해
   throughput-adaptive EMA 경로의 regression 인지, CapacityAwareRouter 전반인지 분리.
4. `git log --oneline b88a5522f..HEAD -- vllm/v1/engine/hybrid_core.py vllm/v1/engine/core_client.py`
   로 085801 이후 router 수정이 있었는지 확인.

## 7. 원본 파일 포인터

| 항목 | 경로 |
|---|---|
| GPU-only result JSON | `eval/results/20260411_125604_G_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/gpu_only.json` |
| GPU-only bench log | `eval/results/20260411_125604_G_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/gpu_only_bench.log` |
| GPU-only CPU/GPU monitor CSV | 같은 디렉토리 `gpu_only_monitor_{cpu,gpu}.csv` |
| Hybrid result JSON | `eval/results/20260411_125824_H_C_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/hybrid.json` |
| Hybrid bench log | `eval/results/20260411_125824_H_C_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/hybrid_bench.log` |
| Hybrid CPU/GPU monitor CSV | 같은 디렉토리 `hybrid_monitor_{cpu,gpu}.csv` |
| 대조군 (정상 작동 reference) | `experiment_result/20260411_085801_h100x4_qwen1.5b_thro_adaptive_500/` |
| stdout regression 분석 (이번 원인 아님) | `experiment_result/20260411_121509_analysis_stdout_scaling_on_fast_hardware/` |
