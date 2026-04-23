# CPU 유휴 자원 활용 기반 시스템 추론 성능 향상 방향 재정의

작성일: 2026-04-21  
작성 기준: 기존 NinjaGap 문서는 의도적으로 배제하고, 현재 코드 상태 + 현재 실측 결과 + 외부 공개 자료만 사용

## 1. 목표와 제외 조건

### 목표

- GPU만으로 처리하던 추론 경로에서 **CPU 유휴 자원을 실제로 동원해 시스템 전체 처리량을 높인다**
- 단순 CPU 사용률 상승이 아니라, **end-to-end throughput / latency 관점에서 의미 있는 개선**만 목표로 한다

### 이번 문서에서 제외한 것

- pruning, sparsity 재학습, speculative decoding 같은 **모델 자체를 바꾸는 기법**
- calibration 의존성이 큰 activation quantization 중심 접근
- "코드가 있으니 싸다" 수준의 근거 없는 재시도

이 문서는 **모델러식 가정**이 아니라, 현재 시스템 구조와 실측 실패 패턴에 맞는 **시스템/커널/스케줄링 관점**의 후보만 남긴다.

## 2. 현재 코드에서 확인된 사실

다음은 `/vllm_hybrid` 현재 코드에서 직접 확인한 사실이다.

### 2.1 현재 hybrid 라우팅은 요청 단위 분기다

- `vllm/v1/engine/core_client.py`
- `vllm/v1/engine/hybrid_core.py`

현재 구조는 요청을 GPU 엔진 또는 CPU 엔진으로 **통째로** 라우팅한다.  
즉, 한 decode step 안에서 같은 요청의 일부 연산만 CPU로 보내고 나머지는 GPU에서 병렬 처리하는 구조가 아니다.

이 점이 중요하다. CPU 유휴 자원을 활용한다면서도, 실제 코드 레벨에서는 아직 **request-internal split** 이 없다.

### 2.2 CPU batch 설정도 보수적이다

- `vllm/v1/engine/hybrid_core.py`

`_resolve_cpu_params()` 경로에서 `cpu_max_num_seqs` 는 명시하지 않으면 기본적으로 매우 보수적으로 해석된다. 현재 구현상 CPU가 많은 시퀀스를 적극적으로 흡수해서 amortize 하도록 설계된 상태가 아니다.

### 2.3 현재 커스텀 CPU 가속 경로는 Q8_0 MLP hot path 중심이다

- `vllm/v1/worker/hot_path_wiring.py`
- `csrc/cpu/quant_q8_0.cpp`
- `csrc/cpu/torch_bindings_hybrid.cpp`

커스텀 CPU kernel 실험의 중심은 `q8_0_linear` 계열이며, attention/라우팅/overlap 을 시스템 수준에서 재설계한 구조는 아니다.

### 2.4 sparse 관련 flag 흔적은 있어도 실제 활성 wiring 은 보이지 않는다

코드 검색 기준으로 `HYBRID_SPARSE_BITMASK` 같은 문자열은 남아 있지만, 현재 `HybridConfig` 와 런타임 wiring 에서 실제로 이어진 경로는 확인되지 않았다. 즉, 적어도 현재 기준 주력 경로는 sparse 쪽이 아니다.

## 3. 현재 실측에서 확인된 사실

다음 문서를 기준으로 확인했다.

- `measurement_results/H100x8/g0_00_qwen2.5_32b_base/README.md`
- `measurement_results/H100x8/g0_06_1_qwen2.5_32b_v2(fail)/README.md`
- `measurement_results/H100x8/g0_11_qwen2.5_32b_phase1(fail)/README.md`

### 3.1 이미 실패한 공통 패턴

- `§06` / `§06-1` 은 seqs=1 쪽 개선은 있었지만, batch가 커질수록 구조적으로 무너졌다
- `§06-1 v2` 는 v1 대비 회귀했다
- `§11 Phase 1` attention reroute 역시 회귀했다

### 3.2 실패의 공통축

현재까지의 실패는 거의 전부 **CPU가 decode compute 를 더 많이 떠안게 만드는 방향**에서 발생했다.

정리하면:

- MLP INT8/Q8_0 kernel 강화
- CPU attention path 재라우팅

둘 다 "CPU가 decode hot path compute 를 더 잘 해보자"는 시도였고, 현재 실측으로는 유의미한 승리가 없다.

### 3.3 따라서 다음 시도는 축을 바꿔야 한다

같은 compute 축에서 커널만 또 바꾸는 것은, 현재 데이터만 놓고 보면 성공 확률이 낮다.  
지금 필요한 것은 "CPU 연산을 더 빠르게"가 아니라, **CPU에게 어떤 역할을 주면 GPU와 겹쳐서 시스템 전체가 빨라지느냐**다.

## 4. 외부 자료에서 확인된 큰 흐름

이번 문서의 외부 자료는 "CPU를 GPU 대체물로 쓰는가", "CPU를 겹치는 자원으로 쓰는가", "모델을 바꾸는가"를 기준으로 나눠서 읽었다.

### 4.1 HeteGen: CPU-GPU 이종 병렬 + overlap

출처:

- HeteGen paper PDF: <https://arxiv.org/pdf/2403.01164>
- HeteGen arXiv abs: <https://arxiv.org/abs/2403.01164>
- HeteGen DOI: <https://doi.org/10.48550/arXiv.2403.01164>
- 요약 페이지: <https://huggingface.co/papers/2403.01164>

핵심:

- CPU와 GPU를 단순 offload 대상으로 보지 않고, **이종 병렬 자원**으로 본다
- 핵심 문제를 "CPU가 느리다"가 아니라 **I/O bottleneck 과 overlap 부재**로 본다
- CPU와 GPU 연산을 비동기적으로 겹쳐서 지연을 줄인다

이 관점은 현재 로컬 실측과 잘 맞는다.  
로컬 실패는 "CPU 단독 kernel 승부"가 잘 안 됐다는 것이고, HeteGen 류는 애초에 **단독 승부를 목표로 하지 않는다**.

### 4.2 Splitwise: prefill/decode phase 분리

출처:

- Splitwise paper PDF: <https://arxiv.org/pdf/2311.18677>
- Splitwise arXiv abs: <https://arxiv.org/abs/2311.18677>
- Splitwise DOI: <https://doi.org/10.48550/arXiv.2311.18677>
- Splitwise paper page: <https://huggingface.co/papers/2311.18677>
- Microsoft Research 소개: <https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/?lang=zh-cn>

핵심:

- prefill 은 compute-intensive
- decode 는 memory-intensive
- 따라서 phase 별로 다른 자원에 태우는 것이 합리적

직접적으로 CPU idle 활용 구조와 1:1 대응하진 않지만, 중요한 교훈이 있다.  
**모든 토큰/모든 layer를 같은 장치에 붙잡아 둘 필요가 없다**는 것이다.

### 4.3 xFasterTransformer: CPU 커널 스택 자체는 여전히 중요하지만, 이것만으로는 부족하다

출처:

- 공식 GitHub: <https://github.com/intel/xFasterTransformer>
- 참고: 이번 조사에서는 xFasterTransformer 자체의 독립 논문/DOI는 확인하지 못했다. 따라서 1차 근거는 공식 저장소 README 와 코드다.

핵심:

- x86/Xeon 전용으로 LLM CPU inference 를 강하게 최적화한 스택이다
- AMX/AVX512 기반 CPU 경로를 광범위하게 제공한다

하지만 현재 로컬 데이터와 합치면 해석은 제한적이다.  
`xFT가 빠르다`는 사실은 곧바로 `우리 구조에서 CPU가 decode hot path 를 더 먹어야 한다`는 결론으로 이어지지 않는다.  
현재 우리는 **커널 절대 성능 부족**만의 문제가 아니라, **시스템 역할 분배 부재** 문제도 같이 안고 있다.

### 4.4 KTransformers: heterogeneous placement 가 실전 축이라는 증거

출처:

- 공식 사이트: <https://www.ktransformers.net/>
- GitHub: <https://github.com/kvcache-ai/ktransformers/>
- 논문 DOI: <https://doi.org/10.1145/3731569.3764843>
- 논문 메타데이터 확인용: <https://www.researchgate.net/publication/396443066_KTransformers_Unleashing_the_Full_Potential_of_CPUGPU_Hybrid_Inference_for_MoE_Models>

핵심:

- CPU/GPU heterogeneous computing 을 프레임워크 핵심으로 둔다
- placement / parallelism 자체를 1급 최적화 대상으로 본다

우리에게 중요한 건 "그 구현을 그대로 쓰자"가 아니다.  
외부 실전 프레임워크도 결국 **placement와 분업**을 핵심 축으로 본다는 점이다.

### 4.5 PowerInfer: 빠를 수는 있지만 이번 기준에서는 제외

출처:

- PowerInfer paper PDF: <https://arxiv.org/pdf/2312.12456>
- PowerInfer arXiv abs: <https://arxiv.org/abs/2312.12456>
- PowerInfer DOI: <https://doi.org/10.48550/arXiv.2312.12456>
- GitHub: <https://github.com/SJTU-IPADS/PowerInfer>

핵심:

- activation locality / hot neuron 관찰을 활용해 일부 연산을 CPU/GPU에 다르게 배치한다
- 성능 면에선 흥미롭지만, 모델/활성 분포 의존성이 강하다

이번 목표는 "명확한 이론이 있는 시스템 개선" 이므로, PowerInfer 류는 **후보군에서는 참고만 하고 1순위에서는 제외**한다.

## 5. 이 자료들을 합친 현재 판단

### 결론 1. 다음 코드는 "더 센 CPU decode kernel" 보다 "CPU-GPU 비대칭 분업" 으로 가는 것이 맞다

이건 추측이 아니라 현재 근거가 가리키는 방향이다.

- 로컬 코드: 아직 request-internal split 이 없다
- 로컬 측정: CPU가 decode compute 를 더 많이 맡는 방향은 이미 여러 번 실패했다
- 외부 자료: HeteGen, Splitwise, KTransformers 모두 **이종 분업 / overlap / placement** 를 핵심으로 본다

즉, 다음 개발의 본질은:

**CPU가 GPU 대신 dense decode kernel 을 끝까지 수행하게 만드는 것**이 아니라,
**CPU와 GPU가 서로 다른 일을 동시에 하도록 시스템을 다시 짜는 것**이다.

### 결론 2. 첫 후보는 request-internal asymmetric split / overlap 이다

구체적으로는 다음 류다.

- 같은 요청의 decode step 안에서 일부 work는 GPU, 일부 work는 CPU에 분배
- CPU가 critical path 전체를 먹지 않도록, **겹칠 수 있는 조각**만 맡김
- 라우터가 "이 요청은 CPU/이 요청은 GPU"가 아니라, **한 요청 안의 역할 분담**을 하도록 변경

현재 코드 기준 첫 타깃 파일은 다음이 된다.

- `vllm/v1/engine/hybrid_core.py`
- `vllm/v1/engine/core_client.py`
- 필요시 `vllm/v1/worker/cpu_worker.py`
- 필요시 `vllm/v1/worker/cpu_model_runner.py`

## 6. 새 후보 우선순위

### A. 1순위: Request-internal heterogeneous split prototype

핵심 아이디어:

- 기존 request-level CPU/GPU 라우팅을 유지하되,
- 별도 실험 flag 아래에서 **decode step 내부 비대칭 분업** prototype 추가

이 방향이 맞는 이유:

- 현재 코드의 가장 큰 구조적 빈칸을 메운다
- 실패한 compute-kernel 축을 반복하지 않는다
- CPU 유휴 자원을 "독립 경로"가 아니라 "겹치는 경로"로 쓴다

예상되는 첫 구현 형태:

- 전체 layer split까지 한 번에 가지 않는다
- 먼저 CPU에 넘길 수 있는 **명확히 분리 가능한 비핵심/보조성 작업** 또는 **메모리 성격이 강한 일부 작업**을 선정한다
- GPU 경로와 overlap 가능한지부터 본다

주의:

- 이 단계는 반드시 tracing/profiling 전제여야 한다
- "CPU 사용률이 올랐다"는 지표는 무의미하다
- 기준은 wall time, outTP, overlap 정도다

### B. 2순위: CPU가 맡는 역할을 decode compute 가 아니라 memory/serving 보조 작업으로 재정의

예:

- KV 관련 보조 처리
- CPU-side staging / transfer preparation
- scheduler assist / batching assist

이 방향은 HeteGen/APEX 류와도 닿아 있다.  
중요한 점은 CPU가 GPU의 대체물이 아니라, **GPU의 병목을 가리는 자원**이 되는 것이다.

### C. 3순위: CPU kernel stack 교체는 단독 과제가 아니라 보조 과제로만 취급

예:

- IPEX / oneDNN / xFT 계열 kernel 활용 가능성 재검토

하지만 현재 데이터상 이것을 1순위로 올릴 근거는 약하다.  
시스템 분업이 없는 상태에서 커널만 바꾸는 것은 이미 여러 번 실패했다.

## 7. 이번 라운드에서 제외하는 후보

### 제외 1. pruning / sparse retraining / calibration-heavy 방식

이유:

- 모델 자체를 바꾼다
- 정확도/품질 리스크가 크다
- 이번 목표인 "명확한 이론이 있는 시스템 개선"과 다르다

### 제외 2. activation quantization 중심 새 시도

이유:

- 현재 실패 축과 가깝다
- calibration / outlier 대응 문제가 다시 붙는다
- 시스템 역할 분담 문제를 해결하지 못한다

### 제외 3. CPU attention/MLP 단독 고도화 재시도

이유:

- 지금까지의 실측에서 이미 여러 번 부정적 신호가 나왔다
- 같은 축 재시도는 최소한 routing/overlap 설계 없이 할 이유가 없다

## 8. 바로 이어서 할 코드 작업

### Step 1. 현재 hybrid 경로의 실제 step-level 병목 계측 포인트 추가

목적:

- CPU/GPU 각 경로에서 decode step 당 무엇이 얼마나 걸리는지 분해
- overlap 가능 지점을 찾는다

후보 위치:

- `vllm/v1/engine/core_client.py`
- `vllm/v1/engine/hybrid_core.py`
- `vllm/v1/worker/cpu_model_runner.py`

산출:

- 요청 단위가 아니라 step 단위 trace
- GPU wait / CPU wait / scheduler wait / transfer wait 분해

### Step 2. request-internal split 을 위한 최소 제어면 추가

목적:

- 현재 "요청 전체를 CPU 또는 GPU" 구조를 깨고,
- 실험 flag 아래에서 일부 작업만 CPU side worker 로 보내는 제어면 확보

핵심은 처음부터 완전한 layer split 이 아니라, **작은 제어면을 먼저 만드는 것**이다.

### Step 3. overlap 가능한 첫 조각 선정

후보:

- CPU가 맡아도 정확도 리스크가 없는 작업
- GPU compute 와 동시에 돌릴 수 있는 작업
- 결과 병합 비용이 작아야 함

이 단계는 profiling 결과를 본 뒤 정해야 한다. 지금 당장 특정 op 를 찍어 정하는 것은 다시 추측으로 돌아간다.

## 9. 내가 지금 추천하는 단일 방향

한 줄로 줄이면 이렇다.

**다음 개발은 CPU kernel 추가 최적화가 아니라, hybrid runtime 을 request-level 라우터에서 step-level heterogeneous scheduler 로 끌어올리는 방향이어야 한다.**

이 판단은 다음 3개가 동시에 맞물려서 나온다.

- 로컬 코드가 아직 그 수준의 분업 구조를 갖고 있지 않다
- 로컬 실측은 CPU decode compute 강화 축에서 연속 실패했다
- 외부 연구와 실전 OSS는 모두 heterogeneous placement / overlap 을 성능 축으로 본다

## 10. 참고 자료

- HeteGen PDF: <https://arxiv.org/pdf/2403.01164>
- HeteGen DOI: <https://doi.org/10.48550/arXiv.2403.01164>
- HeteGen summary: <https://huggingface.co/papers/2403.01164>
- Splitwise PDF: <https://arxiv.org/pdf/2311.18677>
- Splitwise DOI: <https://doi.org/10.48550/arXiv.2311.18677>
- Splitwise summary: <https://huggingface.co/papers/2311.18677>
- Splitwise (Microsoft Research): <https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/?lang=zh-cn>
- xFasterTransformer official repo: <https://github.com/intel/xFasterTransformer>
- KTransformers official site: <https://www.ktransformers.net/>
- KTransformers GitHub: <https://github.com/kvcache-ai/ktransformers/>
- KTransformers DOI: <https://doi.org/10.1145/3731569.3764843>
- PowerInfer PDF: <https://arxiv.org/pdf/2312.12456>
- PowerInfer DOI: <https://doi.org/10.48550/arXiv.2312.12456>
- PowerInfer GitHub: <https://github.com/SJTU-IPADS/PowerInfer>
