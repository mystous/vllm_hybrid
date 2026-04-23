# CPU Shadow Breakthrough 가설 정리 v1

작성일: 2026-04-21  
작성 기준: `super_power/ideation` 내 3개 문서 + 추가 통찰(B1/B2/B3) 결합  
목표: "CPU 유휴 자원을 활용해 시스템 추론 성능을 높인다"는 목적 아래, 기존 문서가 약하게 다룬 독립 가설을 구조적으로 정리하고 다음 검증 순서를 고정한다.

## 1. 전제: 지금까지 무엇이 실패했고, 무엇이 아직 안 검증됐는가

현재까지 로컬 코드/실측이 보여준 핵심은 단순하다.

- CPU가 decode hot path compute 를 더 많이 맡는 방향은 이미 반복적으로 실패했다
- 특히 dense MLP/attention 을 CPU kernel 로 직접 더 빠르게 만들겠다는 축은 실측상 breakthrough 신호가 없었다
- 따라서 다음 breakthrough 는 `CPU가 더 계산한다`가 아니라, `CPU가 다른 역할을 맡는다`는 방향에서 찾아야 한다

이 전제는 아래 ideation 문서들의 공통 결론과도 맞는다.

- [cpu_idle_acceleration_ideation_20260421.md](/vllm_hybrid/super_power/ideation/cpu_idle_acceleration_ideation_20260421.md)
- [20260421_openai.md](/vllm_hybrid/super_power/ideation/20260421_openai.md)
- [20260421_104039_cpu_idle_utilization_system_inference_directions.md](/vllm_hybrid/super_power/ideation/20260421_104039_cpu_idle_utilization_system_inference_directions.md)

즉, 지금부터의 문제는 "CPU를 어떻게 더 바쁘게 만들까"가 아니라:

**CPU가 어떤 역할을 맡아야 GPU의 token-step 임계 경로를 줄이거나, 적어도 방해하지 않으면서 시스템 처리량을 올릴 수 있느냐**다.

## 2. 공통 구조: 세 문서가 실제로 수렴하는 하나의 축

세 문서를 통합하면 공통축은 아래 한 줄로 요약된다.

**CPU는 main decoder 가 아니라 shadow plane 이어야 한다.**

여기서 shadow plane 은 두 층으로 나뉜다.

### 2.1 Shadow Control Plane

CPU가 맡는 일:

- batch planning
- request admission
- prefix / KV-aware scheduling
- next-step 준비

핵심은 CPU가 decode compute 본체를 대신하는 것이 아니라, GPU가 다음 step 에 바로 들어갈 수 있도록 준비를 앞당기는 것이다.

### 2.2 Shadow Data Plane

CPU가 맡는 일:

- cold KV tier
- async prefetch
- KV compression / staging
- layer-ahead hint 계산

핵심은 CPU가 "주 계산"을 하지 않고, GPU의 메모리 병목과 준비 지연을 줄이는 쪽에 서는 것이다.

이 구조를 기준으로 보면, 기존 문서에서 상대적으로 덜 강조되었지만 독립적으로 의미 있는 breakthrough 가설은 세 개뿐이다.

## 3. Breakthrough 가설 1 — B1 Inverted Control Plane

### 핵심 질문

CPU를 더 쓰는 것이 아니라, **CPU가 token-step 임계 경로에 덜 개입하게 만들면** 오히려 더 빨라지는가?

### 가설

Fast GPU 앞에서는 host control-path 자체가 병목이 될 수 있다.  
우리 hybrid 가 지금까지 집중한 것은 "CPU에게 어떤 compute 를 더 시킬까"였다.  
하지만 Blink 계열의 메시지는 정반대다.

**CPU는 계산을 더 맡을수록 아니라, step 제어 경로에서 빠질수록 이득일 수 있다.**

### 이론적 배경

- Blink: <https://arxiv.org/abs/2407.20242>

Blink 가 말하는 본질은, fast GPU 환경에서 host-side scheduling / continuous batching / block table 갱신 / token-step dispatch 가 자체 병목이 될 수 있다는 점이다.

### 구체적 구조

- scheduler 는 별도 CPU 프로세스에서 lookahead planning 수행
- GPU worker 에는 ready batch queue 를 미리 공급
- GPU 는 매 step 마다 host 의 세밀한 재결정을 기다리지 않고, 준비된 입력을 가능한 한 연속적으로 실행
- CPU는 "매 step 개입"이 아니라 "미리 준비" 역할로 이동

즉, 이 가설은 CPU shadow plane 의 **control plane 극단화**다.

### 왜 기존 문서 대비 독립적인가

기존 문서들도 scheduler 를 언급하긴 했지만, 대체로 "CPU가 뭘 더 맡을까"의 보조항목으로 다뤘다.  
B1은 반대로:

**breakthrough 자체를 control-plane 제거/재배치에서 찾는다**는 점에서 독립적이다.

### 검증 방법

현재 §06-1 v1 상태를 baseline 으로 두고, 먼저 코드 변경 없이 아래만 본다.

- scheduler 호출 시간
- batch 재구성 시간
- block table / metadata 준비 시간
- worker dispatch 대기 시간
- decode step total 에서 host control-path 가 차지하는 비율

### 승산 판단 기준

- host control-path 가 step total 의 의미 있는 비중을 차지하면 진입 가치 있음
- 비중이 작으면 B1은 즉시 기각 가능

### breakthrough 성격

- 구현은 어렵지만
- 검증 자체는 상대적으로 싸고
- 성공 시 기존 kernel 중심 사고를 완전히 버릴 수 있음

## 4. Breakthrough 가설 2 — B2 Workload Redefinition

### 핵심 질문

우리가 지금까지 보고 있던 workload 자체가 CPU shadow 기법의 ROI를 죽이고 있던 것은 아닌가?

### 가설

현재 `Qwen2.5-32B × 128/128` workload 는 CPU shadow 가 빛날 수 없는 구간일 가능성이 높다.

즉,

- GPU HBM 압박이 낮고
- KV pressure 가 거의 없고
- host-side bottleneck 도 상대적으로 약한 구간이라면

어떤 CPU shadow 기법도 이 workload 에서는 이득이 안 날 수 있다.

### 구조적 의미

이 가설은 "코드를 바꾸는 breakthrough"가 아니라,  
**무엇을 대상으로 breakthrough 를 탐색해야 하는지를 재정의하는 breakthrough**다.

이 점이 중요하다.  
잘못된 workload 위에서 좋은 아이디어를 기각하면, 방향 전체가 틀어질 수 있다.

### 이론적 배경

- [20260421_openai.md](/vllm_hybrid/super_power/ideation/20260421_openai.md) 의 KV pressure 분석
- cold KV / InfiniGen / ScoutAttention 류는 long-context / high-KV-pressure 구간에서 의미가 커짐
- current workload 에서는 CPU shadow 의 자리가 거의 없을 수 있음

### 구체적 구조

workload 를 아래처럼 다시 본다.

- short input / short output
- long input / short output
- long input / long output
- reasoning-heavy long decode
- shared-prefix 멀티턴

즉, "현재 환경에서 뭐가 빠르냐"가 아니라:

**어느 workload 구간에서 CPU shadow 의 자리가 실제로 생기느냐**를 먼저 본다.

### 검증 방법

- `eval/large_envs/` 아래의 준비된 env 들로 baseline 재측정
- gpu_only 대비 hybrid 비율이 어느 구간에서 의미 있게 개선되는지 확인
- 특히 KV pressure 와 decode length 가 커질수록 CPU shadow 후보의 ROI가 올라가는지 확인

### 왜 breakthrough 인가

이건 측정만 하는 일처럼 보이지만 실제로는 그렇지 않다.  
만약 workload 를 바꾸는 순간 이전에 가치 없어 보이던 cold KV / async prefetch / ScoutAttention 계열이 살아난다면,
문제 정의 자체가 바뀐다.

즉 B2는:

**어떤 breakthrough 를 시도할지 결정하는 상위 breakthrough**다.

### 현재 추천 순위

세 가설 중 **가장 먼저 해야 한다.**

이유는 간단하다.

- 코드 변경이 없다
- 리스크가 없다
- 정보 가치가 크다
- B1/B3 의 ROI 판정 근거가 된다

## 5. Breakthrough 가설 3 — B3 Meta-scheduling Gateway

### 핵심 질문

vLLM 내부를 크게 뜯지 않고도, CPU를 coordination 전담 계층으로 분리하면 가치가 있는가?

### 가설

vLLM 을 compute engine 으로 단순화하고,  
그 앞단에 CPU-only gateway 를 둬서 request dispatch / prefix routing / multi-model switching / priority queue / admission control 을 전담시키면,
CPU는 계산보다 coordination 에서 더 큰 가치를 낼 수 있다.

### 이론적 배경

- Aegaeon: <https://arxiv.org/abs/2510.06460>

문서 내 인용된 요지는, 멀티모델 풀링과 goodput 최적화 관점에서 coordination 층이 별도 가치가 있다는 점이다.

### 구체적 구조

- 현재 `CapacityAwareRouter` 류 로직을 vLLM 내부 보조가 아니라 외부 gateway 로 승격
- vLLM 은 토큰 계산에 집중
- gateway 는 다음을 담당
  - request classification
  - prefix-aware routing
  - model/priority tier 선택
  - admission / queue shaping

즉 이 가설은 CPU shadow plane 의 **system-level control plane 외부화**다.

### 왜 기존 문서 대비 독립적인가

기존 문서들은 대부분 vLLM 내부 scheduler / KV / attention 에 집중했다.  
B3는 그보다 바깥 계층을 별도 프로세스로 세워:

**CPU의 가치를 compute가 아닌 orchestration 에서 찾는다**는 점에서 독립적이다.

### 장점

- vLLM 내부 deep patch 를 줄일 수 있다
- 여러 GPU 엔진 / 여러 모델 / priority tier 조합으로 확장성이 높다
- 재사용 가능한 시스템 자산이 될 수 있다

### 단점

- 단일 모델 / 단일 엔진 환경에서는 과설계일 수 있다
- API 계층이 하나 늘어난다
- 현재 목표가 단일 hybrid 성능이면 직접 이득이 작을 수 있다

### breakthrough 성격

- 구현 공수는 중간
- 정보 가치는 B1/B2보다 낮을 수 있으나
- 장기 시스템 구조로는 가장 재사용성이 높다

## 6. 세 가설의 관계

세 가설은 경쟁 관계가 아니라 계층 관계다.

### B2는 문제 정의를 고친다

먼저 CPU shadow 가 실제로 작동할 workload 구간이 있는지 본다.

### B1은 단일 엔진 내부의 병목을 푼다

workload 상 가치가 보이면, 그 다음은 vLLM 내부 control path 를 줄이는 쪽으로 간다.

### B3는 시스템 외부 계층으로 확장한다

내부 개선보다 coordination 가치가 더 크면, gateway 구조로 나간다.

즉 구조는 아래와 같다.

```text
B2: 어디서 이득이 나는가?
  -> CPU shadow 가 의미 있는 workload 구간을 찾는다

B1: 단일 엔진 내부에서 왜 막히는가?
  -> host control-path 병목이면 inverted control plane 으로 간다

B3: coordination 을 바깥으로 뺄 가치가 있는가?
  -> 멀티엔진/멀티모델/priority 문제라면 gateway 로 간다
```

## 7. 비교표

| 기준 | B1 Inverted Control Plane | B2 Workload 전환 | B3 Meta-scheduling Gateway |
|---|---|---|---|
| 검증 비용 | 작음 (profile) | 작음 (측정) | 중 |
| 코드 변경 규모 | 큼 | 0 | 중 |
| 정보 가치 | 큼 | 큼 | 중 |
| 구조적 독립성 | 높음 | 매우 높음 | 높음 |
| 실패 리스크 | scheduler bug | 거의 없음 | 계층 복잡도 증가 |
| 성공 시 파급력 | 큼 | 큼 | 큼 |

## 8. 최종 판단

가장 먼저 할 것은 B2다.

이유:

- 지금까지의 기각 중 일부는 아이디어가 아니라 workload 가 틀렸기 때문일 수 있다
- B2는 무위험 고정보다 가치가 큰 선행 검증이다
- B1과 B3는 모두 B2의 결과를 봐야 의미 있게 우선순위를 정할 수 있다

그 다음 순서는 다음처럼 잡는 것이 맞다.

### 1단계

**B2 Workload Redefinition**

- large workload baseline 재측정
- CPU shadow ROI 구간 확인

### 2단계

**B1 Inverted Control Plane**

- 만약 long-decode / KV-pressure workload 에서도 hybrid 가 안 뜬다면
- 원인이 host control-path 인지 먼저 확인

### 3단계

**B3 Meta-scheduling Gateway**

- coordination 문제가 단일 엔진 내부보다 상위 계층에서 더 크다고 판단될 때
- 또는 여러 엔진/모델/tier 운영 가치가 커질 때

## 9. 한 줄 결론

기존 문서가 공통으로 말한 것은 `CPU는 main decoder 가 아니라 shadow plane 이어야 한다`는 점이다.  
그 위에 추가 통찰까지 합치면, 다음 breakthrough 탐색 순서는:

**B2로 문제 구간을 다시 정의하고, B1로 host control-path 병목을 겨냥하고, B3를 장기 coordination 구조로 검토하는 것**이다.

이 셋 중 당장 가장 값진 다음 행동은:

**B2 workload 전환 실측이다.**

