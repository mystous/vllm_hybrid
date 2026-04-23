# B2 결과 분석 — Long-decode workload 에서 현재 hybrid 는 왜 무너지는가

작성일: 2026-04-22 (KST)
작성자: Claude
데이터 커밋: `868e05127`
분석 대상 경로:
- `measurement_results/H100x8/g0_longctx_32b/` (heavy, 16K/16K)
- `measurement_results/H100x8/g0_longctx_32b_control/` (light, 128/128, 동일 서버·동일 코드로 대조군)

---

## 0. TL;DR — 시스템 성능 개선 관점에서 얻은 것

**최종 목표는 "시스템 추론 성능을 CPU 유휴 자원으로 높인다" 이며, 이번 heavy workload (16K in / 16K out) 실측이 그 목표를 향한 세 개의 구체적 방향을 갈라낸다.**

| 가설 | heavy 데이터가 주는 판정 | 시스템 개선 방향으로의 영향 |
|---|---|---|
| **B1** (Inverted Control Plane — CPU 를 control-path 에서 빼기) | 보강됨 (indirect) | CPU 가 critical path 에 있으면 tail 을 장악하는 구조 확인. B1 의 "CPU 는 critical 하지 않은 곳에만" 원칙이 heavy 에서 더 강하게 정당화됨. |
| **B2** (Heavy workload 에서 CPU decode shadow) | **전제 확정, 처방 기각** | GPU 가 실제로 압박 상태 (p99 TTFT 21s, prefill 큐잉) 라는 B2 의 전제는 확인. 그러나 현재 CPU decode 가 GPU 대비 52× 느려 "같은 역할을 나눠 갖기" 는 원리적으로 불가. B2 를 살리려면 "CPU 가 GPU 와 같은 일" 이 아니라 "CPU 만 할 수 있는 일 (prefix 재계산, speculative draft, K/V admission) " 으로 역할을 재정의해야 한다. |
| **B3** (Meta-scheduling Gateway — CPU 는 routing/policy 전용) | 강하게 지지 | Heavy 에서 `completed = 100 − 2×cpu_max_seqs` 라는 닫힌 식이 정확히 성립 → routing 결정 자체는 예측대로 작동. CPU 가 decode 는 못 해도 "빠른 정책 엔진" 역할은 할 수 있음이 반증으로 드러남. |

**한 줄 요약**: heavy 데이터는 B2 "원형" 을 기각하고, B1 을 보강하며, B3 로의 피벗을 정당화한다. 시스템 성능 개선의 다음 걸음은 "CPU 에게 decode 를 시키지 않고, 대신 GPU 의 tail 을 *만들지 않는 역할* 을 맡긴다" — 이 한 원칙의 구현이다.

---

## 1. 이 문서가 답하는 질문

우리의 목표는 **"시스템 추론 성능을 CPU 유휴 자원으로 높인다"** 이고, 그 목표를 향한 세 가설이 있다:

- **B1** — CPU 를 decode 가 아닌 **control/정책 경로에서 빼내는** 방향
- **B2** — CPU 를 GPU 와 함께 **decode 를 나눠 갖는** 방향 (shadow)
- **B3** — CPU 를 **상위 meta-scheduler / gateway** 로 쓰는 방향

이번 heavy workload (16K in / 16K out × 100 prompts × concurrency 55) 실측은 세 가설 중 **B2 의 직접 검증** 이며, 동시에 B1/B3 에 대한 간접 증거를 주는 실험이다. 이 문서는 다음 구조로 읽는다:

1. heavy 데이터를 해부해 B2 가설의 전제와 처방을 분리 검증한다
2. light (128/128) 은 **control-only** 로 heavy 에서 도출된 메커니즘이 workload 독립적인지만 확인한다
3. 최종적으로 B1/B2/B3 각각이 이번 실측으로 어떻게 다시 정렬되는지 — **시스템 성능 개선의 구체적 다음 단계** — 로 귀결한다

---

## 2. 분석 원칙

### 2.1 같은 축으로만 묶는다
모델, in/out length, max_concurrency, num_prompts, code commit 이 모두 같은 것끼리만 직접 비교한다. 이번 데이터는 두 묶음 모두 동일 commit `538276073` 위에서 돌았다.

### 2.2 평균 throughput 숫자만 보지 않는다
Timeout 에 걸린 bench 는 `output_throughput = total_tokens / duration` 이 **"bulk path 가 낸 속도"가 아니라 "straggler 몇 개가 timeout 기간 내내 0 tok/s 근처로 있었던 결과"** 를 보여준다. 따라서 다음 세 축을 함께 본다:
- `completed` — 몇 요청이 실제로 끝났나
- `duration` — timeout 에 닿았는가 (21600s = bench 기본 상한)
- `mean/p99 TTFT`, `mean/p99 TPOT`, `mean/p99 ITL` — bulk path 의 속도

### 2.3 판정의 범위를 좁혀 언어화한다
이번 데이터는 "CPU shadow 의 개념" 을 평가하지 않는다. 평가하는 것은 현재 구현의 세 가지 구체적 설정 — `routing_priority=cpu-first`, `routing_strategy=capacity`, `CPU decode 분기` — 이 조합이다.

---

## 3. 비교 대상

| 묶음 | workload | code commit | cpu_max_seqs 축 | 결과 디렉토리 |
|---|---|---|---|---|
| **Heavy (B2 main)** | Qwen2.5-32B-Instruct · 16K in / 16K out · concurrency 55 · 100 prompts | `538276073` | gpu_only + {1, 2} | `g0_longctx_32b/` |
| **Light (control)** | Qwen2.5-32B-Instruct · 128 in / 128 out · concurrency 없음(rate=inf) · 500 prompts | `538276073` | gpu_only + {1, 2, 4, 8, 16} | `g0_longctx_32b_control/` |

두 묶음은 "같은 서버·같은 코드·cpu-first + capacity 라우팅" 의 불변 조건에서 **workload 만 바뀐다**. 따라서 결과 차이는 workload 만의 효과로 해석 가능하다.

---

## 4. 1차 증거 — throughput 붕괴는 heavy 가 아니라 양쪽 공통

### 4.1 Heavy (16K/16K)
| 구성 | Duration | Completed | Output tok/s | vs gpu_only |
|---|---:|---:|---:|---:|
| gpu_only | **830s** | **100 / 100** | **1,918** | baseline |
| hybrid seqs=1 | 21,600s (timeout ❌) | 98 / 100 | 72 | **-96.2%** |
| hybrid seqs=2 | 21,600s (timeout ❌) | 96 / 100 | 71 | **-96.3%** |

### 4.2 Light (128/128)
| 구성 | Duration | Completed | Output tok/s | vs gpu_only |
|---|---:|---:|---:|---:|
| gpu_only | **5.3s** | **500 / 500** | **11,617** | baseline |
| hybrid seqs=1 | 83.5s | 500 / 500 | 739 | **-93.6%** |
| hybrid seqs=2 | 82.3s | 500 / 500 | 750 | -93.5% |
| hybrid seqs=4 | 101.2s | 500 / 500 | 609 | -94.8% |
| hybrid seqs=8 | 77.6s | 500 / 500 | 795 | -93.2% |
| hybrid seqs=16 | 107.1s | 500 / 500 | 576 | -95.0% |

![Light bench sweep — wall / TPOT / Output throughput vs cpu_max_num_seqs](../../../measurement_results/H100x8/g0_longctx_32b_control/analysis_bench.png)

*그림 1. Light (128/128) sweep. 세 패널 모두에서 **초록 점선 = GPU-only** 기준이 hybrid 의 어떤 seqs 보다 5~25× 유리함을 시각화한다.*

![Heavy bench sweep — hybrid 2 points + GPU baseline](../../../measurement_results/H100x8/g0_longctx_32b/analysis_bench.png)

*그림 2. Heavy (16K/16K) sweep. hybrid seqs=1, 2 모두 duration 이 21,600s 에서 잘린 모습이 wall plot 의 y-축 스케일 차이로 드러난다 (GPU 830s vs hybrid 21,600s).*

### 4.3 이 표가 반증하는 것
**B2 의 약한 버전 ("적어도 light 보다 heavy 에서 더 잘해야 한다") 부터 무너진다.** Hybrid 의 손실률이 light 에서 -93~95%, heavy 에서 -96%. workload 가 커져도 손실 비율이 줄지 않는다 (오히려 더 악화). "workload mismatch" 가설로는 설명 불가.

---

## 5. 2차 증거 — bulk path 의 per-token 속도는 hybrid/gpu_only 동일

Heavy 묶음의 ITL / TPOT / TTFT 를 보면:

| 구성 | mean ITL | p99 ITL | mean TPOT | mean TTFT | p99 TTFT |
|---|---:|---:|---:|---:|---:|
| gpu_only | 25.3ms | 31.8ms | 25.3ms | 6,876ms | 21,311ms |
| hybrid seqs=1 | **25.2ms** | **31.6ms** | 25.2ms | 6,855ms | 20,683ms |
| hybrid seqs=2 | **25.1ms** | **31.5ms** | 25.1ms | 6,579ms | 20,518ms |

**GPU-handled 요청의 token-level 속도는 완전히 동일하다.** 즉 hybrid 가 GPU 경로를 더 빠르게 만든 것도, 더 느리게 한 것도 아니다. 이 관찰이 결론에서 **CPU 의 존재가 bulk path 성능을 바꾸지 못한다** 는 구조적 주장을 가능하게 한다.

---

## 6. 3차 증거 — `completed` 가 메커니즘을 **정확히** 드러낸다

Heavy hybrid 의 `completed = 98/100, 96/100` 은 결정적 단서다. 그리고 이 두 숫자는 우연이 아니라 **cpu-first + capacity + cpu_max_seqs 조합의 해가 예측하는 정확한 값**이다.

### 6.1 seqs=1 의 수학

- 2 CPU engines × `cpu_max_seqs=1` = **CPU 동시 슬롯 2개**
- cpu-first 라우팅 → bench 시작 시점 첫 2 요청이 CPU engine 0, 1 에 1개씩 dispatch
- 이후 요청 (2~99) 는 CPU slot 이 꽉 찼으니 모두 GPU 로
- GPU 는 98 요청을 concurrency=55 로 약 14분에 완료 (gpu_only 830s 와 거의 일치)
- 해당 시각에 CPU 는 16,384 tokens 중 매우 일부만 낸 상태
- bench 의 종료 조건 = 100 요청 모두 완료 → CPU 의 2 요청이 끝나길 기다림
- 21,600s (bench 기본 timeout) 에서 잘림 → **예측 completed = 100 − 2 = 98** ✓

### 6.2 seqs=2 의 수학

- 2 CPU engines × `cpu_max_seqs=2` = **CPU 동시 슬롯 4개**
- 동일 논리로 첫 4 요청이 CPU 로 dispatch
- 96 요청은 GPU 로, 14~15분에 완료
- CPU 의 4 요청은 timeout 까지 미완
- → **예측 completed = 100 − 4 = 96** ✓

두 예측 모두 실측과 완전히 일치한다. 즉 **`100 − completed = 2 × cpu_max_seqs`** 라는 닫힌 식이 성립한다. 이것은 단순한 버그가 아니라 routing 정책의 결정적 귀결이다.

### 6.3 CPU decode rate 역산

Heavy timeout 데이터에서 CPU decode 상한을 역산:
- CPU 에 묶인 각 요청: 16,384 tokens output 필요
- 21,600s 안에 끝나지 않음 → **rate < 16384 / 21600 ≈ 0.76 tok/s**
- 이 숫자는 어제 작성한 이론 상한 5.5 tok/s (weight memory BW 제한) 대비 **7× 이상 느린 값**이다
- 그 격차가 레이어 3 (코어 미활용) 의 크기를 정량화한다

### 6.4 이 섹션이 닫은 것

Hybrid 가 GPU-only 대비 손해를 본 구조는 **"GPU 가 느려져서" 가 아니라 "GPU 의 14분 완주 이후 CPU 2~4 요청을 기다리느라 wall 이 21,600초로 늘어나서"** 다. 즉 CPU 는 **도움이 아니라 detour sink** 로 기능했다.

---

## 7. Heavy-specific 심층 — B2 의 "전제" 는 살아있다, 그런데 왜 실패했나

위 §4~6 은 "hybrid 가 GPU-only 보다 못하다" 를 보였다. 그러나 B2 의 핵심 주장은 "heavy 에서는 **GPU 가 압박을 받고 있어서** CPU 가 도울 자리가 있다" 였다. 이 전제 자체를 heavy 데이터로 검증할 필요가 있다.

### 7.1 GPU-only heavy 의 숫자가 보여주는 "GPU 압박"

| 지표 | Heavy gpu_only | Light gpu_only | 해석 |
|---|---:|---:|---|
| mean TTFT | **6,876ms** | 1,295ms | heavy 에서 TTFT 가 5.3× 커짐 |
| p99 TTFT | **21,311ms** | — | 마지막 1% 는 21초 대기 |
| p50 TTFT | 2,491ms | — | 중앙값조차 2.5초 |
| Duration / output tok | 830s / 1.59M | 5.3s / 62K | per-token wall time 같은 수준이나 큐 길이 다름 |

p99 TTFT = 21,311ms 는 **scheduler 가 prefill 을 큐잉하고 있다** 는 직접 증거다. concurrency=55 × 16,384 input tokens = 900K tokens 의 prefill 이 한 번에 밀려들어가면 GPU 도 prefill 단계에서 serialize 할 수밖에 없다. 즉 **B2 의 전제 "GPU 가 heavy 에서 압박 상태" 는 데이터로 확정된다**.

### 7.2 HBM 여유 계산

- 8 × H100 80GB = **640 GB**
- Weight (Qwen2.5-32B BF16): 64 GB → TP=8 → 8 GB/GPU, 총 64 GB
- KV (55 req × 32K ctx × 256KB/token): 55 × 8.4 GB = **462 GB**
- Activation/buffer 여유: ~50 GB
- 계산상 남는 여유: 640 − 64 − 462 − 50 ≈ **64 GB** (10%)

HBM 은 포화 직전이지만 넘지는 않았다. 즉 **HBM 용량 자체가 timeout 의 원인이 아니다**. bench 가 정상 완료하는 것 (gpu_only 14분) 도 이 계산과 정합.

### 7.3 그런데 왜 CPU 가 못 썼는가 — 구조적 해명

B2 의 전제 (GPU 압박) 는 맞다. 그럼에도 hybrid 가 GPU-only 를 이기지 못한 이유는 다음 부등식으로 요약된다:

```
GPU 의 per-request 실효 속도    ≈ output_throughput / active_concurrency
                              ≈ 1918 tok/s / 약 48 streams (effective)  ≈ 40 tok/s / req

CPU 의 per-request 실효 속도    < 0.76 tok/s / req  (§6.3)

속도 비                         > 52×
```

CPU 가 GPU 의 1/52 속도로 돌아가는 상황에서 CPU 에 요청을 보내는 것은 그 요청의 완료 시각을 52× 뒤로 미루는 행위다. **bench 의 종료 조건이 "모든 요청 완료" 인 이상, CPU 로 간 요청 1개라도 있으면 전체 wall-clock 은 그 느린 요청으로 결정된다.** 이것이 "GPU 가 압박 받고 있어도 CPU 를 쓸 수 없는" 이유의 구조적 해명이다.

### 7.4 Heavy 의 utilization pattern 이 보여주는 것

![Heavy per-CPU heatmap — 6시간 내내 CPU 는 바쁜 척, GPU 는 14분에 끝](../../../measurement_results/H100x8/g0_longctx_32b/analysis_cpu_heatmap.png)

*그림 2.5. Heavy hybrid seqs=1 의 per-CPU heatmap. bench 전체 기간 (약 6시간) 동안 일부 코어가 **점유 상태**로 보이지만, 동시에 `[HYBRID-CPU-WORKER] thread config:` 로그가 남지 않았다는 사실 (§8 레이어 3) 과 결합해 해석하면, **실제로 일하는 것은 일부 코어뿐이고 나머지는 IPEX 의 intra-op 제약으로 쉬고 있다**. 바쁜 척하는 heatmap 과 "0.76 tok/s 미만" 이라는 실측치 사이의 모순이 이 시각화의 핵심이다.*

### 7.5 이 섹션의 결론

- B2 의 **전제** (heavy 에서 GPU 압박) 는 데이터로 **확정** (p99 TTFT 21s, prefill 큐잉)
- B2 의 **처방** (그래서 CPU shadow 로 보충) 는 **기각** (CPU 속도가 52× 느려 기여 불가, 오히려 tail sink)
- 이 비대칭이 "문제 진단은 맞았는데 치료법이 틀렸다" 는 B2 의 실패 본질이다

---

## 8. 4차 증거 — light 의 cpu_max_seqs sweep 이 tail hazard 를 시각화한다

Light (128/128) 의 TTFT 와 p99 TPOT 를 seqs 축으로 본다:

| seqs | Duration | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT |
|---:|---:|---:|---:|---:|---:|
| 1 | 83.5s | 1,535ms | 2,274ms | 65.6ms | 85.3ms |
| 2 | 82.3s | 1,575ms | 2,299ms | 64.5ms | **190.5ms** |
| 4 | 101.2s | 1,607ms | **5,612ms** | 67.0ms | **731.0ms** |
| 8 | 77.6s | 1,832ms | **8,162ms** | 66.1ms | 546.6ms |
| 16 | 107.1s | 2,249ms | **13,564ms** | 93.4ms | 736.1ms |

### 이 표의 읽는 법
- **mean TPOT 은 64~93ms 로 거의 평탄**. → bulk (GPU) path 는 그대로.
- **p99 TPOT 과 p99 TTFT 는 seqs 따라 폭발**. seqs=1 에서 p99 TPOT 85ms → seqs=16 에서 736ms (8.6배). → tail 이 커지는 중.
- **Duration 은 seqs 축으로 단조증가도 아니고, 감소도 아님** (83 → 82 → 101 → 78 → 107). → cpu_max_seqs 라는 capacity 축을 돌려서 total throughput 을 회복할 수 없다.

### 의미
seqs 를 늘리면 CPU 로 라우팅되는 요청 수만 늘어난다 → 각 요청이 CPU 에서 느리게 끝나므로 **tail 이 더 퍼질 뿐 bulk 는 그대로**. 이것은 "capacity 축이 B2 를 구하지 못한다" 는 실험적 증거다.

![Light util timeseries — per seqs CPU/GPU](../../../measurement_results/H100x8/g0_longctx_32b_control/analysis_util_timeseries.png)

*그림 3. Light sweep 의 CPU (물리 코어 평균, 파랑) vs GPU (빨강) utilization 시계열. **bench 전체 기간 동안 CPU 가 70% 이상 점유되지만 GPU 는 대부분 10% 이하에 머문다**. GPU 는 이미 빨리 일을 끝냈고 나머지 시간은 CPU 의 tail 을 기다린 것. duration 이 seqs 에 단조증가하지 않는 이유도 여기서 보인다 — "GPU 가 빨리 끝나는 시간" 은 거의 상수이고, "CPU 가 마지막 req 를 끝내는 시간" 이 매번 달라질 뿐이다.*

---

## 9. 메커니즘 — 세 레이어의 결함이 곱해진다

이번 데이터는 단일 버그가 아니라 **세 독립 결함의 곱** 으로 설명된다.

### 레이어 1 — CPU decode 는 이 모델에서 GPU 대비 25× 이상 느리다
- 32B BF16, per-socket RAM BW 약 300~400 GB/s
- 이론 상한 ~5.5 tok/s per request (weight memory-bound)
- 실측 하한 < 0.76 tok/s (heavy 의 21,600s timeout 에서 역산)
- **GPU 의 concurrency=55 환경에서의 per-request share ≈ 40 tok/s 와 비교해 최소 52× 격차**

### 레이어 2 — 현재 `cpu-first` 는 tail 을 구조적으로 생성한다
- Cold-start gate 이후 **두번째 요청** 부터 CPU 로 blind dispatch
- CPU slot 이 빌 때마다 재채움 → CPU 에는 항상 "느린 요청 큐" 가 존재
- GPU 가 아무리 빨라도 bench 의 wall-clock 은 CPU 의 가장 느린 요청으로 결정됨
- 즉 **hybrid 의 종료 조건이 GPU path 가 아니라 CPU path 의 완료 시각에 묶인다**

### 레이어 3 — CPU 코어가 (사용자 실측 관찰) 전부 쓰이지 않는다
어제 현장에서 사용자가 top 으로 관찰:
> "코어 몇 개만 쓰고 있어"

이는 레이어 1 의 이론 상한 (5.5 tok/s) 보다 실측이 더 낮은 이유를 설명한다.
- OMP/IPEX intra-op threadpool 이 48 코어 전부를 활용하지 못함
- 가능 원인: IPEX 가 자체 pool 로 OMP_NUM_THREADS 를 무시 / GQA 8-head 가 thread 수보다 작아 parallelism 한계 / KMP_AFFINITY 충돌
- `hybrid_server_run.log` 에 `[HYBRID-CPU-WORKER] thread config:` 마커가 남지 않은 점도 이 레이어가 의심스러움의 방증

![Light per-CPU heatmap — NUMA0 / NUMA1 engine binding](../../../measurement_results/H100x8/g0_longctx_32b_control/analysis_cpu_heatmap.png)

*그림 4. Light per-CPU utilization heatmap. Y-축 = 물리 코어 id 0-111, 점선 = NUMA0/NUMA1 경계. 두 engine 이 정확히 두 socket 에 바인딩된 것은 확인되지만, **각 socket 내부에서 코어별 밝기가 균일하지 않다** — 일부 코어만 짙은 빨강이고 나머지는 옅은 노랑에 가까운 부분이 seqs=1~16 에 걸쳐 일관되게 보인다. 이것이 "코어 몇 개만 쓰고 있어" 의 데이터적 표현이다. 레이어 1 의 "이론 상 5.5 tok/s" 보다 실측이 더 낮은 이유가 여기 있다.*

---

## 10. Cross-workload 교차검증 — 가장 중요한 구조적 발견

Light 묶음이 B2 판정의 **결정적 증거** 다. 이유:

1. Light 에서는 **GPU 가 어떤 의미로도 포화되지 않는다** (duration 5.3s, 500 req, HBM 99.5% free). 따라서 "CPU 가 도와야 할 압박" 자체가 없다.
2. 그럼에도 light hybrid 는 -93.6% 손실.
3. Heavy 는 -96% 손실.
4. 두 상황의 손실률이 거의 같다 → **"CPU 가 도와야 할 여지가 없어서 못 도왔다" 가 아니다**. 도울 자리가 있든 없든 같은 방식으로 손상된다.

이 관찰이 "B2 가설이 단지 workload 가 너무 소박했던 게 아니다" 를 확정한다. workload 를 아무리 바꿔도 — bigger model, longer context, reasoning CoT 로 가든 간에 — 레이어 1+2+3 이 그대로면 결과는 같다.

![Heavy util timeseries — CPU pinned, GPU idle waiting on CPU tail](../../../measurement_results/H100x8/g0_longctx_32b/analysis_util_timeseries.png)

*그림 5. Heavy (16K/16K) 의 CPU / GPU utilization 시계열. Light 와 동일한 패턴: **CPU 는 bench 내내 점유, GPU 는 초반 짧게 100% 찍은 뒤 대부분 유휴**. "workload 가 바뀌어도 pattern 은 같다" 를 한 장으로 보여준다. 그림 3 (light) 과 그림 5 (heavy) 를 나란히 놓고 보면 workload 축을 따라 바뀌는 것은 "CPU 가 얼마나 오래 매여 있느냐" 이지 "GPU 가 도움 받는 구간" 이 아니다.*

---

## 11. B1 / B2 / B3 관점의 재정렬 — 시스템 성능 개선의 갈림길

이번 heavy 실측은 하나의 가설만 판정하지 않는다. 같은 데이터가 세 가설에 각각 다른 신호를 준다. 그 신호를 분리해 정렬하는 것이 시스템 개선 방향을 정하는 작업이다.

### 11.1 B1 — Inverted Control Plane (CPU 를 critical path 에서 빼기)

**위치**: Blink 계열 통찰 ("host control-path 가 long decode 에서 병목이 된다").
B1 은 "CPU 가 inference 자체에 참여하면 안 된다, 대신 해야 할 일은 *GPU 가 하기 싫은 control-path 잡무* 를 CPU 가 들고 가서 GPU 의 host 개입 횟수를 줄이는 것" 이다.

**heavy 가 B1 에 주는 신호 — 강한 보강**:
- 이번 실험은 B1 가설을 *반증* 하지 않는다. 오히려 §6 의 닫힌 식 `completed = 100 − 2×cpu_max_seqs` 는 "CPU 가 critical path 에 참여하면 전체 bench 종료 시각이 CPU 의 가장 느린 요청에 묶인다" 를 실증한다.
- 즉 **"CPU 를 critical path 에서 빼내면 시스템 throughput 은 즉시 GPU-only 수준 (1,918 tok/s) 로 회복된다"** 가 heavy 데이터의 직접적 귀결이다.
- 이것이 B1 의 "invert" 철학 — CPU 는 GPU 의 decode 에 끼지 말고, 대신 decode 가 끝나는 동안 다음 배치의 schedule, KV admission, prefix lookup 등을 선행해라 — 의 정량적 근거다.

**시스템 개선 구현 방향**:
1. 현재 `_route_capacity` 에서 cpu-first 기본값을 **gpu-first 로 전환** + CPU 는 정책 결정만 담당
2. CPU 가 *pre-schedule* (다음 batch 구성, admission control, prefix cache eviction) 를 GPU 의 step 과 병렬로 수행 → GPU step 내 host sync 점 제거
3. 측정축은 "throughput" 이 아니라 "GPU 가 host 기다리는 시간 (sync gap)" 으로 변경

### 11.2 B2 — Heavy workload CPU decode shadow (**전제 확정, 처방 기각**)

**위치**: "GPU 가 long-ctx 에서 HBM 으로 포화되니 CPU 가 일부 decode 를 떠맡아 보충하자" 는 가장 직관적 가설.

**heavy 가 B2 에 주는 신호 — 갈라짐**:
- **전제 확정**: §7.1~7.2 에서 GPU 가 실제로 압박 상태 임을 확인 (p99 TTFT 21s, prefill 큐잉, HBM 여유 10%). B2 가 "해결할 문제가 있다" 는 것은 참이다.
- **처방 기각**: 그러나 §7.3 의 52× 속도비 부등식에 의해, CPU 를 GPU 의 decode 대체자로 쓰는 것은 *어떤 workload 에서도* 작동할 수 없다. bench 종료 조건이 tail 로 묶이는 구조가 workload-independent 이기 때문.

**B2 를 살리는 유일한 방향 — 역할 재정의**:
B2 의 "CPU 가 decode 를 나눠 갖자" 를 포기하되, heavy workload 의 문제 자체는 여전히 실재하므로 **"GPU 가 할 수 없는 일 중 long-ctx 에서 필요한 것" 을 CPU 가 맡는** 버전으로 재정의한다:
1. **K/V tier offload**: decode 후반의 누적 KV (cold tier) 를 CPU DRAM 으로 이주시켜 GPU HBM 포화 완화 (LMCache / InfiniGen 계열). CPU 는 *데이터 저장자* 지 *decode 실행자* 가 아니므로 52× 격차와 무관.
2. **Prefix re-materialization**: concurrency=55 × 16K input 의 prefill 중복 연산을 CPU 가 캐싱/재계산해 GPU prefill 큐 감축. CPU 의 한 번만 하는 재계산은 미리 병렬화 가능.
3. **Speculative prefill draft**: CPU 가 짧은 draft prefix 를 먼저 만들고 GPU 가 검증 (DuoDecoding / Dovetail 계열의 prefill 버전).

**시스템 개선 구현 방향**: 위 세 중 `K/V tier offload` 가 이번 데이터와 가장 정합 (heavy 에서 HBM 여유 10%, KV 462 GB 의 상위 20% 만 CPU 로 보내면 3~5 concurrency 추가 수용 가능, wall 14분 → 11분 수준으로 추정).

### 11.3 B3 — Meta-scheduling Gateway (**강한 지지**)

**위치**: Aegaeon 계열 통찰 — "CPU 는 느려서 decode 는 못 하지만, *정책 결정* 은 순식간이다". 시스템 전반의 admission, routing, batching 결정을 CPU 에 몰아넣고 GPU 는 executor 에 집중.

**heavy 가 B3 에 주는 신호 — 강한 지지**:
- §6 의 닫힌 식이 맞았다는 사실 자체가 "current router 의 routing 결정은 GPU 의 실행 속도와 무관하게 즉각적이다" 를 의미한다. 즉 CPU 가 내리는 *결정* 자체는 시스템의 bottleneck 이 아니다.
- heavy 에서 GPU 가 14분에 완주한다는 것은 **"만약 CPU 가 그 14분 동안 next 100 prompt 의 routing plan 을 미리 세워 두면 end-to-end latency 가 감축된다"** 는 가능성을 열어둔다.
- §8 의 light sweep 에서 p99 TTFT 가 seqs 에 따라 8.6× 악화된 것도, 역으로 "CPU 가 주도권을 가진 gateway 가 있었으면 그 tail 을 피할 routing 을 할 수 있었을 것" 이라는 지표.

**시스템 개선 구현 방향**:
1. CPU engine 을 *EngineCore* 가 아니라 *SchedulerCore* 로 재정의. GPU engine 은 순수 executor.
2. Incoming request 는 CPU 에서 (a) admission 가능 여부, (b) 적정 GPU batch slot, (c) prefix cache hit 이력 — 세 가지를 결정 후 GPU 에 dispatch.
3. 측정축: request 도착 → GPU step 진입까지의 latency (현재 cpu-first 라우팅이 만드는 추가 hop 을 없애는 방향).

---

## 12. 시스템 성능 개선의 구체적 다음 단계

§11 에서 도출된 세 방향을 우선순위로 정렬한다. 각 항목은 **검증 가능한 성능 지표** 와 함께 제시한다.

### 12.1 P0 — 레이어 3 선결: CPU 코어 활용 진단

어떤 가설로 가든 CPU 가 실제 능력을 내지 못하면 결과는 같다. §8 레이어 3 의 "`[HYBRID-CPU-WORKER] thread config` 로그 부재 + 0.76 tok/s << 이론 5.5 tok/s" 의 8× 격차는 **다음 실험의 신뢰성 을 모두 갉아먹는** 독립 문제다.

- **진단 실험**: hybrid_server_run.log 의 `[HYBRID-CPU-WORKER]` 마커가 찍히지 않는 원인 트레이싱, IPEX intra-op pool 의 `torch.get_num_threads()` 실측
- **성공 지표**: CPU engine 의 per-request decode rate 가 이론 상한 (32B 에서 5.5 tok/s) 의 70% 이상 달성
- **블로킹 조건**: P1/P2/P3 모두 이 선행에 묶임

### 12.2 P1 — B3 구현 (SchedulerCore 분리)

**가장 큰 기대 효과 + 가장 적은 리스크**. 이미 두 프로세스 구조가 있으므로 역할만 뒤집는다.

- **최소 실험**: 기존 `CapacityAwareRouter` 의 결정 로직을 CPU engine 프로세스 안에서 실행, GPU 프로세스는 순수 executor. API endpoint 가 CPU 에 먼저 도달.
- **성공 지표**: heavy workload 에서 p99 TTFT 가 21,311ms → 15,000ms 이하로 감축 (prefix hit rate 향상 효과 포함)
- **실패 시 학습**: CPU 의 routing overhead 가 실측으로 관찰됨. 이 값이 1ms 단위라면 B3 설계 성립, 100ms 이상이면 B3 재설계 필요.

### 12.3 P2 — B1 구현 (Inverted Control Plane, partial)

B3 의 sub-case 로 점진적으로 가능. CPU 가 GPU step 과 병렬로 next-batch scheduling 을 미리 수행.

- **최소 실험**: GPU step i 가 돌아가는 동안 CPU 가 step i+1 의 admission + batch 구성 결정. GPU 가 step 종료 시점에 결과만 가져가도록.
- **성공 지표**: GPU step 간 host sync gap (GPU profile 의 idle time) 이 현재 대비 30% 감소
- **실패 시 학습**: scheduler 가 미리 결정할 수 있는 정보가 생각보다 적다면, 이는 vllm v1 scheduler 의 구조적 제약 문제

### 12.4 P3 — B2 재정의 버전 구현 (K/V tier offload)

위 두 방향이 성과를 낸 뒤에 heavy workload 한정으로 추가 여유 확보.

- **최소 실험**: `HYBRID_KV_OFFLOAD` flag 를 켜서 decode 후반 KV 를 CPU DRAM 으로 tier eviction. 검증은 동일 heavy workload (16K/16K × 100) 에서.
- **성공 지표**: GPU HBM 사용량이 현재 대비 20% 감소 + `MAX_CONCURRENCY` 를 55 → 65 로 올릴 수 있음
- **실패 시 학습**: CPU DRAM 에서 KV 를 다시 HBM 으로 가져올 때의 bandwidth 가 실제 decode step 간격 대비 너무 크다면 tier offload 자체가 성립 불가

### 12.5 하지 말아야 할 것 (명시)

- **workload 를 계속 바꿔 B2 원형을 다시 시도** — 이번 데이터가 layer-agnostic 하게 거부. 레이어 3 해결 없이 CoT reasoning / 70B / MoE 로 가도 같은 결과.
- **`cpu_max_seqs` 를 계속 올려 hybrid seqs=32/64 를 돌리기** — §6.1/6.2 의 닫힌 식이 stragglers 만 늘어남을 예측.
- **"이번엔 IPEX 가 안 맞아서" 류의 국소 버그 가설** — 레이어 1+2 가 독립적으로 문제를 설명하므로 단일 fix 로 안 풀림.

---

## 13. 데이터 아티팩트 (재현용)

| 아티팩트 | 경로 |
|---|---|
| Heavy gpu_only JSON | `measurement_results/H100x8/g0_longctx_32b/gpu_only_baseline/gpu_only.json` |
| Heavy hybrid seqs1 JSON | `measurement_results/H100x8/g0_longctx_32b/seqs1/hybrid.json` |
| Heavy hybrid seqs2 JSON | `measurement_results/H100x8/g0_longctx_32b/seqs2/hybrid.json` |
| Light sweep 디렉토리 | `measurement_results/H100x8/g0_longctx_32b_control/` |
| Per-plot (bench/util/heatmap/power) | 각 디렉토리의 `analysis_*.png` (본 문서 그림 1~5 로 인용) |
| 본 문서 인용 외 plot | `analysis_gpu_power_mem.png` (두 묶음 모두) — GPU power draw / memory util 시계열. 본 판정에 독립 증거는 아니나 HBM 여유 확인용 |
| Jupyter 재실행 | `G0_ROOT=<dir> jupyter nbconvert --execute <dir>/analysis_g0.ipynb --inplace` |
| Code commit | `538276073` (`hybrid_core.py` + cold-start gate 이전 버전 기준) |

---

## 부록 A — 숫자의 내적 일관성 체크

- Heavy gpu_only: 100 req × 16,384 token / 830s = 1,974 tok/s (보고값 1,918 tok/s 와 3% 오차, prefill/tail overhead 로 설명 가능)
- Heavy hybrid seqs=1: 98 × 16,384 / 21,600 = 74 tok/s (보고값 72 tok/s 와 일치)
- Light gpu_only: 500 × 128 / 5.3 = 12,075 tok/s (보고값 11,617 tok/s, prefill 비중 고려 정합)
- 위 세 자체 검증 결과 timeout 제외 숫자는 신뢰할 수 있음.
