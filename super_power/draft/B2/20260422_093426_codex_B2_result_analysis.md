# B2 결과 분석 — Workload Redefinition 가설 판정

작성일: 2026-04-22  
작성자: Codex  
목적: `eval/results` 아래에 올라온 B2 실험 결과를, 비교 가능한 범위만 엄격히 묶어서 해석하고, B2 가설에 대해 구조적으로 판정한다.

## 1. 질문

B2의 질문은 단순히 "큰 모델/큰 workload를 돌려보자"가 아니었다.  
정확한 질문은 다음과 같다.

**현재 128/128 중심 workload 에서는 CPU shadow 의 ROI가 보이지 않았는데, long-context / long-decode / high-KV-pressure workload 로 가면 CPU shadow 가 실제로 기여할 구조적 자리가 생기는가?**

따라서 이번 분석의 목표는 다음 둘을 구분하는 것이다.

- `큰 workload에서 current hybrid가 실제로 살아나는가`
- `큰 workload에서 current hybrid는 여전히 죽지만, 그 원인이 workload 부적합이 아니라 구조적 tail hazard 인가`

## 2. 분석 원칙

이번 문서는 단순 결과 나열을 하지 않는다.  
결론을 오염시키지 않기 위해 아래 원칙으로 본다.

### 2.1 직접 비교 가능한 결과만 묶는다

동일 모델, 동일 입력/출력 길이, 동일 concurrency, 동일 prompt 수인 경우만 직접 비교한다.

### 2.2 throughput 숫자를 그대로 믿지 않는다

장시간 timeout 이 개입된 실험에서는 `평균 throughput` 이 bulk path 성능이 아니라 `소수 straggler` 의 영향일 수 있다.  
따라서 completed count, duration, TTFT, TPOT, ITL 을 같이 본다.

### 2.3 이번 판정은 "current hybrid" 에 대한 것이다

이번 결과로 기각되는 것은 `CPU shadow 전체 개념`이 아니라,  
현재 구현된 `CPU decode 분기형 hybrid` 가설이다.

## 3. 비교 대상 분리

`eval/results` 에서 이번 B2 해석에 직접 비교 가능한 묶음은 아래 셋이다.

### 3.1 Large-workload 묶음

- [gpu_only.json](/vllm_hybrid/eval/results/20260421_085735_G_H100_80GB_HBM3_x8_Qwen2.5-32B-Instruct/gpu_only.json)
- [hybrid seqs1](/vllm_hybrid/eval/results/20260421_091604_H_C_H100_80GB_HBM3_x8_Qwen2.5-32B-Instruct_seqs1/hybrid.json)
- [hybrid seqs2](/vllm_hybrid/eval/results/20260421_152038_H_C_H100_80GB_HBM3_x8_Qwen2.5-32B-Instruct_seqs2/hybrid.json)

이 셋의 공통 workload:

- model: `Qwen/Qwen2.5-32B-Instruct`
- num_prompts: `100`
- max_concurrency: `55`
- random_input_len: `16384`
- random_output_len: `16384`

즉, **16K input / 16K output / concurrency 55** 의 long-decode workload 이다.

### 3.2 Short-workload 보조 묶음

- [hybrid seqs4](/vllm_hybrid/eval/results/20260421_071441_H_C_H100_80GB_HBM3_x8_Qwen2.5-32B-Instruct_seqs4/hybrid.json)
- [hybrid seqs8](/vllm_hybrid/eval/results/20260421_072012_H_C_H100_80GB_HBM3_x8_Qwen2.5-32B-Instruct_seqs8/hybrid.json)
- [hybrid seqs16](/vllm_hybrid/eval/results/20260421_072521_H_C_H100_80GB_HBM3_x8_Qwen2.5-32B-Instruct_seqs16/hybrid.json)

이 셋은 다른 workload 이다.

- num_prompts: `500`
- random_input_len: `128`
- random_output_len: `128`

즉, **128/128 short workload** 이다.  
따라서 B2의 long-decode 본판정에는 보조 참고로만 사용한다.

## 4. 핵심 시각 자료

이번 분석에서 실제로 참고할 가치가 높은 그림은 벤치마크 요약 그림 두 장이다.  
이유는 단순하다. 이번 B2의 핵심 질문은 `heavy workload` 와 `light control workload` 에서 hybrid 위치가 어떻게 달라지는가이기 때문이다.

### 4.1 Heavy workload 요약

파일:
[analysis_bench.png](/vllm_hybrid/measurement_results/H100x8/g0_longctx_32b/analysis_bench.png)

![Heavy workload benchmark summary](/vllm_hybrid/measurement_results/H100x8/g0_longctx_32b/analysis_bench.png)

이 그림은 `16K/16K, 100 prompts` 묶음의 GPU-only와 hybrid(`seqs1`, `seqs2`) 차이를 한 장에 요약한다.  
이번 문서의 본판정은 이 그림이 보여주는 분포를 텍스트로 해석한 것이다.

### 4.2 Light control workload 요약

파일:
[analysis_bench.png](/vllm_hybrid/measurement_results/H100x8/g0_longctx_32b_control/analysis_bench.png)

![Control workload benchmark summary](/vllm_hybrid/measurement_results/H100x8/g0_longctx_32b_control/analysis_bench.png)

이 그림은 `128/128, 500 prompts` 의 control 묶음에서 `seqs1/2/4/8/16` 변화가 어떻게 나타나는지 보여준다.  
heavy workload 가 아닌 기존 short workload 에서 current hybrid 의 scaling 한계가 어디서 드러나는지 보조적으로 읽을 수 있다.

### 4.3 보조 시계열 그림

파일:
[analysis_util_timeseries.png](/vllm_hybrid/measurement_results/H100x8/g0_longctx_32b/analysis_util_timeseries.png)

![Heavy workload utilization timeseries](/vllm_hybrid/measurement_results/H100x8/g0_longctx_32b/analysis_util_timeseries.png)

이 그림은 heavy workload 에서의 시간축 변화를 보여주므로, `bulk는 진행되지만 일부 tail request 가 오래 남는다`는 해석을 보조하는 용도로만 참고한다.  
다만 최종 판정은 JSON/bench 로그 숫자를 기준으로 하고, 이 그림은 보조 증거로만 사용한다.

### 4.4 CPU heatmap 보조 그림

파일:
[analysis_cpu_heatmap.png](/vllm_hybrid/measurement_results/H100x8/g0_longctx_32b/analysis_cpu_heatmap.png)

![Heavy workload CPU heatmap](/vllm_hybrid/measurement_results/H100x8/g0_longctx_32b/analysis_cpu_heatmap.png)

이 그림은 CPU 사용 패턴을 직관적으로 보여주지만, 단독으로 해석하면 "CPU를 많이 썼으니 유의미했다"는 잘못된 결론으로 흐르기 쉽다.  
따라서 이 문서에서는 **tail 구간에서도 CPU 자원이 실질적으로 구조를 구제하지 못했다**는 점을 보조하는 시각 자료로만 사용한다.

### 4.5 GPU power / memory 보조 그림

파일:
[analysis_gpu_power_mem.png](/vllm_hybrid/measurement_results/H100x8/g0_longctx_32b/analysis_gpu_power_mem.png)

![Heavy workload GPU power and memory](/vllm_hybrid/measurement_results/H100x8/g0_longctx_32b/analysis_gpu_power_mem.png)

이 그림은 heavy workload 에서 GPU 메모리/전력 흐름을 보여준다.  
특히 long-context workload 에서 GPU 쪽 pressure 자체는 충분히 존재했음을 보조적으로 확인하는 자료로 쓸 수 있다. 다만 여기서도 핵심 판정은 그래프 자체가 아니라, `GPU pressure 가 있어도 current hybrid 가 ROI를 회복하지 못했다`는 JSON/bench 결과와 함께 읽어야 한다.

## 5. Large-workload 관측 사실

### 4.1 GPU-only 기준선

`gpu_only`:

- output throughput: `1918.53 tok/s`
- total token throughput: `3893.50 tok/s`
- mean TTFT: `6875.64 ms`
- mean TPOT: `25.29 ms`
- completed: `100 / 100`
- duration: `829.58 s`

이 값은 long-decode workload 에서의 GPU-only 기준선이다.

### 4.2 Hybrid 결과

`hybrid seqs1`:

- output throughput: `72.16 tok/s`
- total token throughput: `146.49 tok/s`
- mean TTFT: `6854.53 ms`
- mean TPOT: `25.15 ms`
- completed: `98 / 100`
- duration: `21600.60 s`

`hybrid seqs2`:

- output throughput: `71.40 tok/s`
- total token throughput: `144.22 tok/s`
- mean TTFT: `6579.33 ms`
- mean TPOT: `25.08 ms`
- completed: `96 / 100`
- duration: `21600.81 s`

### 4.3 표면적인 ratio

GPU-only 대비 output throughput ratio:

- seqs1: `72.16 / 1918.53 = 3.76%`
- seqs2: `71.40 / 1918.53 = 3.72%`

표면적으로는 hybrid 가 거의 완전히 붕괴한 것처럼 보인다.

## 6. 왜 이 숫자를 그대로 해석하면 안 되는가

이번 B2에서 가장 중요한 포인트는 여기다.

### 5.1 TTFT / TPOT / ITL 은 거의 동일하다

GPU-only 와 hybrid seqs1/2 를 비교하면:

- mean TTFT: 거의 동일한 범위
- mean TPOT: 거의 동일한 범위
- mean ITL: 거의 동일한 범위

즉, **완료된 request 들의 token-step 동작 자체는 GPU-only 와 비슷하게 보인다.**

### 5.2 그런데 throughput 만 26배 가까이 붕괴한다

이 조합은 bulk path 전체가 느려졌다는 신호라기보다,  
**소수의 request 가 끝나지 않아 wall clock 을 비정상적으로 늘린 경우**에 더 가깝다.

### 5.3 실제로 duration 이 6시간 timeout 이다

hybrid seqs1/2 는 둘 다:

- duration ≈ `21600 s`

즉, 정상 종료 시간이 아니라 **6시간 제한 시간에 걸린 결과**다.

### 5.4 completed count 도 100이 아니다

- seqs1: `98 / 100`
- seqs2: `96 / 100`

즉, 전체 100개 중 대다수는 끝났지만,  
마지막 `2~4` 개가 끝나지 못하고 timeout 으로 전체 throughput 을 망쳤다.

## 7. 구조적 해석

이제 관측 사실을 하나의 구조로 엮으면 다음과 같다.

### 6.1 bulk path 는 완전히 무너진 것이 아니다

TTFT / TPOT / ITL 이 GPU-only 와 유사하다는 것은,  
완료된 request 들은 대체로 GPU-only 와 비슷한 token-step dynamics 를 가졌다는 뜻이다.

즉, 현재 결과는:

**"모든 요청이 천천히 처리되었다"**가 아니라  
**"대부분은 비슷하게 처리되었지만, 일부 tail request 가 병적으로 오래 걸렸다"**에 가깝다.

### 6.2 current hybrid 의 위험은 평균이 아니라 tail 에 있다

이 long-decode workload 에서 current hybrid 는:

- bulk throughput 향상 신호를 보여주지 못했고
- 반대로 소수 request 에 대해 극단적인 tail hazard 를 만든다

즉, 문제는 단순 성능 저하가 아니라 **tail amplification** 이다.

### 6.3 이건 B2 가설의 일부를 부정한다

B2의 기대 중 하나는:

**"큰 workload 로 가면 current hybrid 도 상대적으로 나아질 수 있다"**

였는데, 이번 결과는 그 기대를 지지하지 않는다.

정확히는:

**long-decode / high-KV-pressure workload 가 current hybrid 를 살려주지 못했다.**

## 8. 하지만 여기서 "CPU shadow 전체 기각"으로 가면 안 되는 이유

이 지점이 논리적으로 중요하다.

이번 결과는 `CPU shadow 전체 개념` 을 기각하지 않는다.  
기각하는 것은 다음 구조다.

**현재의 request-level CPU decode 분기형 hybrid**

즉, 지금 구조는:

- 일부 요청을 CPU path 로 보냄
- long-decode request 가 CPU tail hazard 로 변함
- 그 straggler 가 전체 wall clock 을 잡아먹음

이 결과는 오히려 반대로 다음을 시사한다.

**CPU는 main decode path 에 들어가면 안 되고, shadow plane 으로만 써야 한다.**

즉 이번 B2 결과는 ideation 문서들의 공통 결론을 강화한다.

## 9. Short-workload 보조 신호

128/128 short workload 쪽 hybrid 결과는 다음과 같다.

- seqs4: output throughput `609.39 tok/s`
- seqs8: output throughput `794.75 tok/s`
- seqs16: output throughput `575.99 tok/s`

이 보조 신호가 말하는 것은:

- batching 을 늘린다고 monotonic 하게 좋아지지 않는다
- sweet spot 이 좁다
- current hybrid 는 안정적인 scaling 구조가 아니다

즉, short workload 에서도 구조적 한계가 있고,  
long workload 에서는 그 한계가 tail hazard 로 더 심하게 드러난다.

## 10. B2 가설 판정

### 9.1 판정 질문

B2의 원 질문:

**"Workload 를 키우면 CPU shadow 의 ROI 가 살아나는가?"**

### 9.2 현재 구현 기준 판정

현재 구현된 hybrid 기준으로는:

**아니오.**

더 정확히는:

- large workload 가 current hybrid 의 구조적 약점을 덮어주지 못했다
- 오히려 long-decode 에서 CPU-routed tail request 가 전체 결과를 붕괴시켰다

### 9.3 개념 수준 판정

하지만 CPU shadow 개념 전체에 대해서는:

**판정 보류**

왜냐하면 지금 검증된 것은 shadow plane 자체가 아니라,  
CPU가 일부 요청을 직접 decode 하는 구조의 실패이기 때문이다.

## 11. 이번 결과가 다음 단계에 주는 의미

이번 B2 결과는 다음 의사결정을 강하게 지지한다.

### 10.1 다음 우선순위는 B1이다

B2가 큰 workload 를 줘도 current hybrid 를 살려주지 못했다면,  
이제 볼 것은 "어디서 tail 이 생기느냐"다.

즉 다음 질문은 자연스럽게 B1이 된다.

**host control-path / routing / request-level CPU 분기가 long-decode tail 을 만드는가?**

### 10.2 B3는 아직 후순위다

meta-scheduling gateway 는 구조적으로 흥미롭지만,  
지금 먼저 봐야 할 것은 외부 orchestration 이 아니라:

- current hybrid 내부에서
- 왜 소수 request 가 6시간 straggler 로 남는지

이다.

즉 B3는 현재 시점에선 장기 구조 후보이지,  
이번 B2 직후의 직접 후속은 아니다.

## 12. 최종 결론

이번 B2 결과는 다음 한 문장으로 요약된다.

**큰 workload 는 current hybrid 를 살려주지 못했다. bulk path 는 GPU-only 와 비슷해 보이지만, CPU 경로로 빠진 소수 long-decode request 가 극단적인 tail hazard 가 되어 전체 결과를 무너뜨린다.**

따라서 이번 분석의 논리적 결론은:

1. `current hybrid` 는 long-decode 에서도 breakthrough 경로가 아니다  
2. 이번 결과는 `CPU shadow 전체 기각` 이 아니라 `CPU decode 분기형 구조 기각` 으로 읽어야 한다  
3. 다음은 B1, 즉 **inverted control plane / control-path 원인 규명** 으로 넘어가는 것이 맞다
