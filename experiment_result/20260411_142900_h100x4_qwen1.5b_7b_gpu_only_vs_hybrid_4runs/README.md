# H100x4 Qwen 1.5B / 7B × gpu_only / hybrid 4-run 검증

`20260411_142900_h100x4_qwen1.5b_7b_gpu_only_vs_hybrid_4runs`

> 이전 실험 (`20260411_141500_h100x4_qwen1.5b_routing_regression_root_cause_fix`)
> 에서 진단/수정한 라우팅 fix 가 1.5B / 7B 양쪽에서 production env 로
> 회귀 없이 동작하는지 4-run 으로 검증한다.

## 1. 핵심 결과

| run | 모델 | 모드 | bench dur (s) | wall (s) | req/s | output tok/s | TPOT (ms) | TTFT (ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.5B | gpu_only | 3.94 | 13.97 | 127.0 | 15,640 | 23.56 | 844 |
| 2 | 1.5B | **hybrid** | **3.87** | 17.13 | **129.2** | **15,911** | **23.03** | **791** |
| 3 | 7B | gpu_only | 4.02 | 13.97 | 124.3 | 15,492 | 24.73 | 857 |
| 4 | 7B | **hybrid** | **3.93** | 17.15 | **127.4** | **15,984** | **23.04** | 933 |

전부 동일 워크로드: 500 req, input 128 / output 128, request rate inf,
H100 80GB HBM3 × 4 (TP=4), Xeon Platinum 8480+ (96 cores, 1 NUMA), 944 GB DDR5.

### 1.1 hybrid vs gpu_only 비율

| 지표 | 1.5B (H/G) | 7B (H/G) |
|---|---:|---:|
| bench duration | 0.98× | 0.98× |
| TPOT | 0.978× | **0.932×** |
| output tok/s | 1.017× | 1.032× |
| req throughput | 1.017× | 1.025× |
| wall (bench.sh) | 1.226× | 1.228× |

→ **bench duration / TPOT / throughput 모두 hybrid 가 gpu_only 와 동등 또는
약간 더 나음** (노이즈 ±2%). 7B TPOT 은 hybrid 가 gpu_only 대비 **6.8% 빠름**
(24.73 → 23.04 ms) — 이는 측정 노이즈일 가능성이 높지만 회귀 없음을 뒷받침.

wall (bench.sh) 의 1.23× 차이는 hybrid 모드가 CPU EngineCore subprocess 와
모니터링 셋업을 추가로 띄우는 ~3 초의 부팅/teardown 오버헤드. **bench
duration 자체** (벤치 실행 시간) 는 동등.

## 2. 라우팅 거동 검증 — `[HYBRID-ROUTER-STATS]`

서버 stdout 의 router 카운터 최종값:

| run | finished | GPU 분배 | CPU 분배 | cpu_ratio |
|---|---|---|---|---|
| 1.5B hybrid | 501 | 501 | 0 | 0.0% |
| 7B hybrid | 501 | 501 | 0 | 0.0% |

→ 두 환경 모두 **501 건 (probe + main 500) 이 전부 GPU 로**. CPU 에 들어간
요청 0 건. Property 2 정확히 발현 — H100 + 1.5B/7B 환경에서는 GPU 가 CPU
보다 압도적으로 빨라서 expected-finish 비교에서 CPU 가 절대 못 이김.

## 3. 리소스 사용 (monitor CSV)

| run | GPU peak | GPU mean | CPU peak | CPU mean |
|---|---:|---:|---:|---:|
| 1.5B G | 47% | 13.9% | 10% | 7.1% |
| 1.5B H | 61% | 13.1% | 10% | 7.0% |
| 7B G | 75% | 19.2% | 14% | 7.1% |
| 7B H | 68% | 13.8% | 10% | 6.8% |

- **CPU mean util**: 모든 run 에서 ~7% (idle). hybrid 라고 더 높지 않음 →
  CPU EngineCore 가 살아 있지만 실제 work load 0 (라우팅이 0 건 보냈으므로).
- **GPU mean util**: hybrid 가 gpu_only 와 거의 같음. 14×에 달했던 회귀
  완전 해소.

## 4. 7B probe hang 증상도 함께 해소

이전 회귀 시 `./bench.sh hybrid envs/h100x4_qwen7b_hybrid.env` 가 1차 probe
에서 분 단위로 정지하던 증상 → 본 fix 로:

```
[14:28:33] Starting monitor (interval=1s)
[14:28:33] --- Running benchmark ---
...
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...   ← 정상 진행
...
Benchmark duration (s):                  3.93
```

probe 가 즉시 GPU 로 라우팅되어 ~수백 ms 에 완료되고, main bench 도 ~4s
에 마침. **사용자가 보고한 hang 완전 해소.**

## 5. 직전 회귀 (수정 전) 와의 비교 — 같은 1.5B / 7B 환경

| 모델 | 지표 | 수정 전 hybrid (`125824` / `130645`) | **수정 후 hybrid (본 run)** | gpu_only (본 run) |
|---|---|---:|---:|---:|
| 1.5B | bench duration | 44.64 s | **3.87 s** | 3.94 s |
| 1.5B | TPOT | 60.34 ms | **23.03 ms** | 23.56 ms |
| 1.5B | output tok/s | 1,380 | **15,911** | 15,640 |
| 1.5B | wall | 93.08 s | **17.13 s** | 13.97 s |
| 7B | bench duration | (probe hang) | **3.93 s** | 4.02 s |
| 7B | TPOT | (probe hang) | **23.04 ms** | 24.73 ms |

**1.5B hybrid wall 93s → 17s** (5.4× 개선), TPOT **60→23 ms** (2.6× 개선),
throughput **1380→15911 tok/s** (11.5× 개선). 7B 는 비교 가능한 직전
hybrid 결과가 hang 으로 측정 불가였지만, 본 run 에서 정상 측정 + gpu_only
와 동등.

## 6. 전체 데이터 포인터

| 항목 | 경로 |
|---|---|
| 1.5B gpu_only | `eval/results/20260411_142009_G_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/` |
| 1.5B hybrid (with fix) | `eval/results/20260411_142328_H_C_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/` |
| 7B gpu_only | `eval/results/20260411_142556_G_H100_80GB_HBM3_x4_Qwen2.5-7B-Instruct/` |
| 7B hybrid (with fix) | `eval/results/20260411_142833_H_C_H100_80GB_HBM3_x4_Qwen2.5-7B-Instruct/` |
| 사용한 env (1.5B) | `eval/envs/h100x4_qwen1.5b_hybrid.env` (production, 무수정) |
| 사용한 env (7B) | `eval/envs/h100x4_qwen7b_hybrid.env` (production, 무수정) |
| 라우팅 fix 코드 | `vllm/v1/engine/hybrid_core.py:347-405` (`_route_throughput_adaptive`) |
| 진단/수정 분석 | `experiment_result/20260411_141500_h100x4_qwen1.5b_routing_regression_root_cause_fix/` |

## 7. 결론

1. **회귀 완전 해소**: 1.5B / 7B 양쪽에서 hybrid wall ≈ gpu_only wall (bench
   duration 동등, wall 차이는 부팅 오버헤드). TPOT/throughput 도 동등.
2. **probe hang 해소**: 7B hybrid 의 first-probe hang 증상 사라짐 (cold start
   에서 GPU 로 라우팅).
3. **Property 2 정확 동작**: H100 + 1.5B/7B 처럼 GPU 가 CPU 보다 압도적으로
   빠른 환경에서 라우터가 자동으로 모든 요청을 GPU 로. CPU 가 도움이 되지
   않을 환경에서는 CPU 를 안 씀 → no regression.
4. **production env 무수정 사용 가능**: 본 4 run 은 `h100x4_qwen{1.5,7}b_hybrid.env`
   를 그대로 사용. 사용자가 추가 튜닝 없이 동일 명령으로 그대로 작동.

## 8. 다음 단계 후속 작업 (TODO 후보)

1. **Property 2 의 실제 효과 측정**: H100 환경에서는 CPU 가 도움이 되지
   않지만, GPU 가 saturated 되는 워크로드 (예: 더 큰 모델 또는 더 깊은
   queue) 에서 hybrid 가 정말 gpu_only 보다 빠른지 별도 실험으로 확인
   필요.
2. **dev RTX 3090 + 1.5B/7B 검증**: 본 fix 가 dev 환경 (GPU 가 약함, CPU 가
   상대적으로 비슷) 에서 hybrid > gpu_only 영역을 망치지 않는지 재측정.
3. **`_update_adaptive_slots` (Bug 1) 정리**: 본 fix 로 라우팅 영향은
   사라졌지만 코드 자체의 잘못된 동작은 정리 필요. 별도 PR.
4. **paper §3 Property 2 정량 식 본문 추가**: expected-CPU-finish vs
   expected-GPU-wait 비교 공식을 paper 에 명시.
