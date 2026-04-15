# Hybrid 개선 방향 재정리 — H100x8 / RTX3090 로그 교차 검증 후

**작성 시각**: 2026-04-14 22:07:44 UTC  
**작성자**: Codex  
**검토 대상**:

- `/vllm_hybrid/ideation/20260413_0916_vLLM CPU-GPU concurrent inference optimize method.md`
- `/vllm_hybrid/ideation/20260413_0933_vllm_hybrid_perf_improvement_B_plans.md`
- `/vllm_hybrid/ideation/20260413_0944_deep-research-report.md`
- `/vllm_hybrid/ideation/20260413_1400_consolidated_optimization_roadmap.md`
- `/vllm_hybrid/eval/h100x8/20260414_213415_codex_h100x8_log_analysis.md`
- `eval/basic/H100x8/*/{hybrid.json,hybrid_server_boot.log,hybrid_server_run.log,hybrid_bench.log}`
- `eval/basic/RTX3090/*/{gpu_only.json,hybrid.json,hybrid_serve.log,hybrid_bench.log}`

## 한 줄 결론

어제 문서들의 메인 아이디어는 맞다. 현재 목표는 **request-level hybrid 를 유지한 채 CPU 동시 request 처리가 실제 계산 이득으로 이어지게 만들어, CPU가 더 많은 request 를 맡을수록 전체 job completion 이 빨라지게 하는 것**이다.

오늘 H100x8/RTX3090 로그가 더 강하게 보여준 것은 다음이다.

- 지금의 `wave-batch + cpu-first` 는 동시성을 늘린 것이 아니라 tail 을 키운 것이다.
- 문제의 본질은 `CPU request 수`가 아니라 `CPU batch 가 진짜 batch 로 동작하지 않는다`는 점이다.
- 따라서 다음 개선의 중심은 single-request polishing 보다 **진짜 동시성 확대**여야 한다.

---

## 1. 오늘 로그로 다시 확인된 핵심 사실

### 1-1. 현재 실패 모드는 “CPU batch scaling 부재”다

H100x8:

- `cpu_max_num_seqs=1` 이어도 CPU 2 req 가 마지막까지 남아 6~7분 tail
- `cpu_max_num_seqs=16` 은 CPU 32 req tail 로 35분대 재앙

RTX3090:

- 1.5B: `8.11s -> 23.06s`
- 7B: `6.52s -> 89.56s`
- 둘 다 CPU 4 req 가 마지막 wall time 을 결정

이건 단순히 “CPU가 느리다”보다 더 구체적이다.

- req 수를 늘렸는데
- per-request cost 가 충분히 내려가지 않고
- barrier, memory traffic, per-seq loop 비용만 누적돼
- 전체 종료 시간이 CPU tail 에 묶인다

즉 지금 문제는 **동시 req 확대가 계산 이득으로 연결되지 않는다**는 것이다.

### 1-2. wave-batch 는 현재 구조에서 가짜 동시성이다

오늘 로그에서 공통적으로 보이는 구조:

- 초반 burst 에서 CPU wave 를 먼저 채움
- GPU bulk 는 먼저 거의 끝남
- CPU wave drain 이 마지막 wall time 결정

이건 HPC식 동시성 확대가 아니다.

- 여러 req를 함께 넣었지만
- 실제로 더 싼 batch kernel 로 합쳐진 게 아니고
- 느린 request owner 를 여러 개 동시에 붙잡은 것에 가깝다

정리하면:

- `cpu_max_num_seqs` 증가 = 현재는 성능 개선이 아니라 tail 증폭
- `wave-batch` = 지금은 failure amplifier
- batch-aware 계산 구조가 생기기 전까지 기본 전략이 되면 안 된다

### 1-3. bring-up 은 끝났고, 남은 건 CPU batch throughput 문제다

이미 로그로 확인된 것:

- CPU engine launch
- NUMA 선택
- `local_omp_cpuid`
- C++ pinning
- IPEX decode 경로 사용

따라서 다음 단계의 질문은 이것이다.

**CPU가 여러 request 를 함께 처리할 때 왜 GPU처럼 scaling 이 안 나오는가, 그리고 그 구조를 어떻게 깨서 CPU inflight 증가를 실제 completion gain 으로 바꿀 수 있는가**

---

## 2. 어제 문서들의 메인 아이디어를 오늘 기준으로 다시 묶으면

## 2-1. 1순위는 진짜 동시성 확대다

본선은 여기다.

- request-level hybrid 의 목표는 CPU가 더 많은 request 를 맡는 것
- 그 목표를 이루려면 여러 request 를 함께 처리할 때 실제 계산 효율이 올라가야 한다
- 지금처럼 `cpu_max_num_seqs` 만 커지고 계산 구조가 그대로면 tail 만 커진다

즉 먼저 필요한 것은:

- inter-request batching 이 attention / FFN / memory access 에서 실제 이득을 내게 만드는 것
- req 수 증가가 throughput scaling 으로 이어지게 만드는 것

이 축에 속하는 아이디어:

- Head Folding
- VNNI pre-pack / load-once-pack-twice
- batch-aware decode attention 커널
- IPEX attention/FFN 경로의 batch scaling 재검토
- NUMA-local multi-engine
- barrier / sync 감소
- AMX / VNNI / AVX-512 dispatch

이건 단순 미세 최적화가 아니다. **request-level hybrid 를 살리는 HPC식 구조 개선**이다.

### 왜 이게 본선인가

오늘 로그가 말하는 건 단순하다.

- `2 req`, `4 req`, `32 req` 로 늘려도
- CPU 쪽이 더 효율적으로 일한 증거는 없고
- tail 만 커졌다

따라서 먼저 물어야 할 것은:

- 여러 req를 함께 넣었을 때 실제로 어떤 연산이 amortize 되는가
- attention 이 per-seq loop 로 남아 있는가
- FFN 이 larger-M GEMM 으로 바뀌는가
- memory traffic 이 줄어드는가

이걸 못 바꾸면 정책을 바꿔도 같은 실패가 반복된다.

## 2-2. 2순위는 single-request 최적화다

이건 필요하다. 다만 본선보다 앞세우면 안 된다.

single-request 최적화 항목:

- thread 수 재조정
- affinity / NUMA pinning 정교화
- shape-thread cliff 회피
- IPEX 경로 점검
- single-request CPU tok/s 개선

이게 필요한 이유:

- base kernel 이 너무 느리면 batch scaling 도 제한된다
- 하지만 이것만으로는 CPU가 더 많은 request 를 맡아도 안 무너지는 구조가 되지 않는다

즉 위치는:

- 필요조건
- 충분조건은 아님

## 2-3. 3순위는 CPU와 GPU가 함께 더 큰 총량을 소화할 작업점 만들기다

이건 어제 문서의 FP8 / INT4 KV, LMCache, CPU DRAM 확장 쪽이다.

이 축의 의미:

- 목표는 GPU가 더 많은 request 를 하는 것이 아니다
- 목표는 CPU와 GPU가 함께 더 많은 총 request 를 끝낼 수 있는 작업점을 넓히는 것이다

그 전제는:

- CPU batch scaling 이 실제로 생겨야 하고
- GPU 쪽도 batch ceiling 이 낮지 않아야 한다

우선순위는 이렇게 보는 게 맞다.

- FP8 KV:
  - 구현 비용이 낮고 즉시 측정 가능
  - GPU batch ceiling 변화 확인에 적합
- INT4 KV + CPU DRAM / LMCache:
  - 70B / long-context / large batch 에서 의미가 큼
  - short-context tail 문제의 직접 해답은 아님

즉 B3 는 살아 있으나, **CPU가 더 많은 request 를 맡는 작업점을 넓히는 주변 조건 강화**다.

---

## 3. 어제 문서에서 보강 또는 수정돼야 할 지점

## 3-1. H100x8 특수 현상처럼 읽히는 부분은 줄여야 한다

RTX3090 로그도 동일한 tail 패턴을 보였기 때문이다.

보강 포인트:

- `wave-batch + cpu-first + batch scaling 없는 CPU path` 는 single-NUMA dev 에서도 실패
- 따라서 문제의 1차 원인은 머신보다는 정책 + CPU batch 비효율
- H100x8 의 2-NUMA 이점은 CPU path 를 덜 망치게 하는 보조 요인이지, 본질적 해결은 아님

## 3-2. `cpu_max_num_seqs` 증가는 전제 조건이 붙어야 한다

지금처럼 쓰면 안 된다.

- 여러 req를 함께 처리할 때 계산 이득이 없는 상태에서
- `cpu_max_num_seqs` 를 올리면
- CPU가 더 많은 request 를 처리하는 게 아니라 CPU tail 을 더 많이 남기게 된다

따라서 문서상 표현은 이렇게 바뀌어야 한다.

- `cpu_max_num_seqs` 증가는 목표가 아니라 검증용 파라미터
- 먼저 CPU batch scaling 형성이 선행
- 그 다음에야 CPU 처리량 확대 실험이 의미 있음

## 3-3. 성공 조건은 “CPU가 더 많은 request 를 처리하면서 전체 completion 을 줄였는가”여야 한다

다음 조건을 동시에 만족해야 진짜 개선이다.

1. CPU가 baseline 보다 더 많은 request 를 처리할 것
2. CPU batch tok/s 가 실제로 증가할 것
3. GPU bulk 완료 후 CPU-only tail 이 baseline 보다 짧아질 것
4. hybrid wall time 이 gpu_only 대비 악화되지 않거나, 최소한 이전 hybrid 보다 좋아질 것

즉 success metric 은 단순 CPU tok/s 가 아니다.

- CPU batch tok/s 증가
- CPU requests handled 증가
- total wall time 감소

이 세 개가 같이 가야 한다.

---

## 4. 지금 시점의 실전 우선순위

## 4-1. 1단계: CPU batch 가 왜 진짜 batch 가 아닌지 깨는 실험

우선순위:

1. Head Folding 가능성 확인
2. VNNI pre-pack / load-once-pack-twice 적용 범위 정리
3. IPEX decode attention / FFN 경로에서 batch-aware 또는 shape-aware 대체 커널 검토
4. thread 수 / NUMA / shape 절벽 회피 재측정
5. barrier / sync / per-seq loop 비용 분해

여기서 보는 지표:

- `num_seqs=1/2/4/8/16` 에서 CPU tok/s scaling
- per-seq latency 감소율
- hybrid 에서 CPU wave size 증가 시 tail 변화

## 4-2. 2단계: 그 위에서 CPU에 더 많은 request 를 맡겨도 되는 작업점 찾기

전제:

- 1단계에서 CPU batch scaling 이 실제로 생겼을 것

그 다음 실험:

1. `cpu_max_num_seqs=1` 에서 strict baseline 재측정
2. `2`, `4`, `8` 순으로 올리되 tail 과 wall time 을 함께 기록
3. `wave-batch` 를 유지할지, `throughput-adaptive` 로 바꿀지 비교

핵심 질문:

- CPU req 수가 늘었을 때 total completion 이 실제로 앞당겨지는가

## 4-3. 3단계: GPU batch ceiling 확장

이 단계는 CPU batch scaling 이 어느 정도 확보된 뒤 의미가 크다.

1. `--kv-cache-dtype fp8` 활성화
2. GPU-only / hybrid 각각 batch ceiling, wall time, tail 변화 측정
3. long-context / 70B 에서만 INT4 KV + CPU DRAM / LMCache 검토

즉 B3 는 여전히 중요하지만, 현재 short burst 문제의 1차 해결책은 아니다.

---

## 5. 최종 판단

오늘 로그를 반영하면, 어제 문서들의 메인 아이디어는 이렇게 다시 요약할 수 있다.

- **request-level hybrid 는 유지한다**
- **여러 request 를 함께 처리할 때 실제로 더 싸지도록 CPU 계산 구조를 바꾼다**
- **그 위에서 CPU가 맡는 request 수를 늘린다**
- **CPU request 증가가 total wall time 감소로 이어지는 작업점을 찾는다**

반대로 지금 틀린 방향은 이렇다.

- CPU batch scaling 이 없는데 `cpu_max_num_seqs` 만 올리는 것
- `wave-batch` 로 CPU wave 를 먼저 채우는 것
- CPU가 더 많은 request 를 맡는 것 자체를 성공으로 보는 것

가장 정확한 한 문장은 이것이다.

> 목표는 CPU가 더 많은 request 를 처리하는 것이다. 단, 그 request 들이 tail 로 남지 않도록 여러 request 를 함께 처리할 때 실제 계산 이득이 생기는 구조를 먼저 만들어야 한다.

