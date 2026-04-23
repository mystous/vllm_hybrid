# B3 — Meta-scheduling Gateway: 설계 및 실행 계획

## Part I · 맥락

### 1.1 출발점

X (Pipelined Async CPU Executor) 는 실측으로 기각됐다. single CPU engine 내에서 `step_with_batch_queue` pipeline 은 같은 req 의 연속 decode step 을 중복 compute 하여 sync 대비 절반 throughput 에 머문다는 것이 H100 실측 (light workload duration 382s→747s, per-engine generation 2.8→1.3 tok/s) 로 확인됐다.

이 실험은 동시에 **X 를 포기해야 할 이유뿐 아니라 방향 전환의 근거**도 제공했다. sync hybrid 는 지표상 정상 동작 중이고, 단일-엔진 내부 최적화의 ROI 는 작다. 남은 축은 "엔진 내부를 더 짜기" 가 아니라 **엔진 바깥의 coordination 계층을 재정의하는 것**이다.

### 1.2 범용 배포 전제

지금까지의 의사결정은 Qwen2.5-32B 단일 모델 × 단일 hybrid 엔진 구성을 암묵 전제로 했다. 실제 배포는 여러 모델 / 여러 엔진 / priority tier 가 섞이는 환경이다. 이 전제에서 breakthrough 가설들을 다시 보면:

- **B1 Inverted Control Plane** — 엔진 내부 host control-path 최적화. 모델과 무관한 엔진 성능 이득은 있으나 범용 배포의 요구 (모델 혼용, prefix 공유, priority) 에 기여하지 않는다.
- **B3 Meta-scheduling Gateway** — 엔진 바깥에 coordination 계층 (request classification, prefix-aware routing, multi-model switching, admission control). 범용 배포에서 요구되는 운영 축 자체.

단일 모델 관점의 throughput 개선은 B1 이 더 직접적이지만, **범용 배포 상품성**이라는 축에서는 B3 의 가치가 지배적이다. X 기각 이후 우리가 우선할 것은 이쪽이다.

### 1.3 불가침 원칙

설계의 최상위 제약:

**기존 GPU 경로 / vLLM 엔진 내부 / 기존 hybrid 경로는 건드리지 않는다.**

- 현재 GPU-only 로 vLLM 을 쓰는 사용자는 gateway 를 알 필요도, 쓸 필요도 없다. 코드 경로, 바이너리, 엔드포인트, 설정 모두 **unchanged** 여야 한다.
- Gateway 는 **선택적 앞단 레이어**다. 기존 엔진을 **감싸는 래퍼**가 아니라 **여러 엔진을 조율하고 싶을 때만 추가로 얹는 층**.
- 기존 `CapacityAwareRouter` 같은 hybrid 내부 로직을 gateway 로 "승격" 한다고 하더라도, hybrid 엔진 자체는 gateway 없이 단독으로 기존과 동일하게 동작해야 한다.

이 원칙이 무너지는 설계는 B3 아래서 허용하지 않는다. X 의 실패는 "멀쩡한 sync 경로를 async 로 대체하려 한" 결정에서 비롯됐고, 같은 종류의 과몰입을 Gateway 에서 반복하지 않는다.

---

## Part II · 설계

### 2.1 목표와 비목표

**목표**
- vLLM 을 compute engine 으로 유지한 채, 그 **앞단에 옵션으로 올릴 수 있는** coordination 계층 (gateway) 을 제공한다.
- 여러 엔진 (GPU-only, hybrid, 서로 다른 모델) 을 하나의 gateway 뒤에 세워 운영할 수 있다.
- prefix locality 를 gateway 가 의식해 라우팅에 반영한다.
- priority tier 간 SLO 를 gateway 수준에서 관리한다.

**비목표**
- **기존 GPU 경로 / 단일 vLLM 엔진 / 기존 hybrid 엔진의 동작을 변경하지 않는다.** Gateway 미사용 시 바이너리 차이, 엔드포인트 차이, 설정 차이 0.
- 단일 엔진 내부 scheduler 를 수정하지 않는다 (B1 영역).
- 단일 모델 단일 엔진 환경의 throughput 개선을 추구하지 않는다.
- vLLM core.py 를 건드리지 않는다 (CLAUDE.md 원칙).

### 2.2 책임 분리

Gateway 가 존재하는 구성에서만 다음 책임이 gateway 로 이전된다. 미사용 구성에서는 기존 위치가 그대로 유지된다.

| 영역 | 단일 엔진 (gateway off) | multi-engine (gateway on) |
|---|---|---|
| Request 수신 / OpenAI API 호환 | APIServer (변경 없음) | **Gateway → Engine APIServer** |
| 엔진 선택 | N/A (단일) | **Gateway** |
| capacity / priority routing | N/A | **Gateway** |
| prefix hash 계산 / 매칭 | 엔진 내부 scheduler (변경 없음) | 엔진 (변경 없음) + Gateway 가 hint 만 관리 |
| admission control / queue shaping | 없음 | **Gateway** |
| 모델 / priority tier 분리 | 없음 | **Gateway** |
| KV cache 관리 | 엔진 (변경 없음) | 엔진 (변경 없음) |
| scheduler.schedule / model.forward | 엔진 (변경 없음) | 엔진 (변경 없음) |
| hybrid 내부 (GPU + CPU 엔진 조합) | `hybrid_core.py` (변경 없음) | 하나의 engine 단위로 gateway 에 노출, 내부 구조 불가시 |

핵심: gateway 는 엔진에 **아무 것도 요구하지 않는다**. 엔진 입장에서 gateway 는 단지 또 하나의 OpenAI 클라이언트일 뿐이다. 이 대칭성이 "기존 경로 unchanged" 를 구조적으로 보장한다.

### 2.3 구조 개요

Gateway 는 **옵션 레이어**다. 같은 코드베이스에서 두 배치 형태를 모두 지원한다.

```
[A] 기존 구성 (gateway 없음, 단일 엔진)
    ─────────────────────────────────
    Client ──► vLLM 엔진 (APIServer + engine)
                   (GPU-only 또는 기존 hybrid)
    * 현재 사용자의 경로. 변경 없음.

[B] Gateway 구성 (multi-engine / multi-model / priority 용)
    ─────────────────────────────────
    ┌─────────────────────────────────────────────┐
    │  Gateway (CPU-only, 옵션 프로세스)          │
    │  - OpenAI API 호환 엔드포인트              │
    │  - Request classifier / router             │
    │  - Prefix index (global hash → engine)     │
    │  - Admission & priority queue              │
    │  - Engine pool manager (health, capacity)  │
    └────┬────────┬────────┬──────────────────────┘
         │        │        │
      ┌──▼──┐  ┌──▼──┐  ┌──▼──┐
      │ Eng │  │ Eng │  │ Eng │   … vLLM 엔진들 (변경 없음)
      │  A  │  │  B  │  │  C  │     A/B 중 어떤 구성으로 기동하든
      └─────┘  └─────┘  └─────┘     엔진은 [A] 의 엔진과 동일 바이너리
```

Gateway 는 엔진을 **OpenAI 호환 API 클라이언트** 로만 대한다. 엔진 입장에서 gateway 가 존재하는지 알 필요 없다. 그래서 gateway 를 끄면 [A] 경로가 그대로 남는다 — 새 코드 경로가 끼어드는 일 없다.

구현 형태는 두 단계로 간다:
1. **Phase 1**: gateway = 독립 프로세스 (HTTP proxy). 엔진은 기존 `vllm serve` 를 변경 없이 띄우고, gateway 가 앞에 붙는다.
2. **후속 단계** (필요 시): 같은 머신에서 latency 가 문제되면 in-process plugin 형태도 추가 제공. 단 이것도 gateway 를 **기동한 경우에만** 활성.

### 2.4 평가 지표 재설정

단일 모델 throughput 은 더 이상 주 지표가 아니다. Gateway 가 존재 가치를 가지는 지표로 재설정한다.

- **Multi-model goodput** — 동시에 서빙되는 모델들의 SLO 달성 request 수 / 초
- **Prefix hit rate** — gateway 가 prefix-aware 라우팅을 할 때 engine-level prefix cache 히트율
- **Priority SLO 달성률** — tier 별 TTFT / TPOT 상한 준수율
- **Router overhead** — gateway 추가 hop 의 p50 / p99 latency

**Regression 상한 (불가침 원칙의 측정적 보장)**: gateway 미사용 구성 ([A] 경로) 의 throughput / latency 는 B3 도입 전후 차이 0%. 단일 엔진에 gateway 를 얹은 구성 ([B] 의 단일 engine 특수 케이스) 에서는 hop 오버헤드가 p99 +5ms 이하, tput -5% 이하.

---

## Part III · Phase 계획

각 Phase 는 **하나의 답을 얻고 끝나는** 단위로 끊는다. 다음 Phase 의 의미는 이전 Phase 결과로 결정된다.

### Phase 0 · 설계 확정과 평가 workload 정의

**목적.** 구현 전에 평가 가능한 비교 기준을 세운다.

**결정해야 할 것.**
- 대상 모델 조합 (최소 2개). 크기가 크게 다른 조합 (예: 7B + 32B) 이 goodput 차이를 드러내기 좋음
- priority tier 정의 (interactive / batch 2-tier 가 최소)
- prefix locality workload (공통 system prompt 를 공유하는 multi-turn 시나리오)
- router overhead 측정법 (gateway 우회 baseline 대비)

**산출물.** `02_evaluation_plan.md` — 모델 조합 / workload / 지표 / baseline 정의

**코드 변경.** 없음

### Phase 1 · Gateway skeleton

**목적.** 독립 프로세스 gateway 를 세우고 단일 엔진을 pass-through 한다. 기존 경로에는 손대지 않는다.

**범위.**
- OpenAI 호환 프런트엔드 (chat/completions, completions)
- 단일 engine 대상 pass-through
- capacity-based routing (현재 `hybrid_core.py` 의 router 로직을 **복제해서** gateway 쪽에 독립 구현. 기존 `hybrid_core.py` 는 변경 없음)
- health check / engine registry

**검증 기준.**
- **[A] 경로 regression 0** — gateway 를 기동하지 않은 상태에서 기존 vLLM 엔진의 tput/latency/엔드포인트 응답이 B3 도입 이전과 동일 (byte-level 응답 diff)
- [B] 단일 engine 구성에서 tput/latency 차이 5% 이내, p99 hop overhead < 5ms

**산출물.** `03_phase1_skeleton.md` + `gateway/` 코드 (vLLM 와 분리된 모듈)

**Exit 조건 (가치 있는 실패).** [B] overhead 가 5% 를 크게 초과하면 in-process plugin 형태도 옵션으로 추가.

### Phase 2 · Multi-engine capacity + priority routing

**목적.** gateway 뒤에 2개 이상 엔진을 붙이고 capacity / priority 기반 dispatch 를 구현한다.

**범위.**
- 여러 엔진 동시 관리 (engine registry + health polling)
- 엔진별 in-flight 추적
- priority tier 2개 — interactive (짧은 응답 우선) vs batch
- tier 별 SLO 위반률 계측

**검증 기준.**
- 두 엔진 합산 goodput 이 각 엔진 단독 합보다 ≥ 10% 개선 (priority mix workload)
- high-priority tier 의 TTFT p99 가 low 대비 2배 이상 낮음

**산출물.** `04_phase2_multi_engine.md`

**Exit 조건.** 개선이 보이지 않으면 gateway 레벨 router 알고리즘 (JSQ, LWL, SRSF 등) 바꿔가며 재측정. 그래도 10% 아래면 prefix locality 가 있는 workload 로 옮겨 Phase 3 먼저 진행.

### Phase 3 · Prefix-aware routing

**목적.** gateway 가 request 의 prefix hash 를 기반으로 해당 prefix 를 캐시하고 있을 가능성이 높은 엔진으로 라우팅한다. engine-level prefix cache 를 gateway 가 **소유하지 않고 힌트만 준다**.

**범위.**
- request 앞부분 hash (현재 vLLM prefix hashing 과 호환)
- prefix → engine 매핑 index (gateway 내부, LRU)
- 엔진 선택 시 prefix hit 예측치를 capacity 와 결합한 score 로 고름

**검증 기준.**
- 공통 system prompt workload 에서 엔진별 prefix hit rate 가 round-robin 대비 유의미하게 상승 (예: 2 배 이상)
- 같은 workload 의 TTFT p99 개선

**산출물.** `05_phase3_prefix_routing.md`

**리스크.** engine-level cache 상태를 gateway 가 정확히 모르면 hit 예측이 부정확. hit feedback (엔진 → gateway 로 실제 hit 여부) 을 도입해 예측을 교정한다.

### Phase 4 · Multi-model pooling

**목적.** gateway 뒤에 서로 다른 모델 엔진을 세우고, request 의 모델 선택까지 gateway 가 담당한다.

**범위.**
- model 지정 request (OpenAI `model` 필드) 를 gateway 가 해석해 해당 모델 엔진 풀로 라우팅
- 모델별 엔진 수 / tier 설정
- fallback / degradation (요청 모델이 unavailable 하면 정책에 따라 다운그레이드 or reject)

**검증 기준.**
- 두 모델 혼합 workload 에서 goodput 이 "각 모델 단독 운영" 합 대비 유지 또는 개선
- 모델 간 간섭 없음 (한 모델 burst 가 다른 모델 latency 에 영향 ≤ 10%)

**산출물.** `06_phase4_multi_model.md`

---

## Part IV · 결정 포인트와 리스크

### 4.1 설계 결정이 필요한 지점

**D1. Gateway 구현 형태의 기본값.** 독립 프로세스 (HTTP proxy) 와 in-process plugin 중 기본을 정한다. 독립 프로세스가 기존 경로 ([A]) 에 영향이 0 이라 **초기 기본값**. in-process 는 overhead 가 문제될 때 옵션으로 추가 — 기본 경로는 건드리지 않는다.

**D2. Gateway 용 routing 로직의 소스.** 기존 `hybrid_core.CapacityAwareRouter` 를 **그대로 옮기지 않고 복제 후 분리**한다. 기존 hybrid 경로는 `CapacityAwareRouter` 를 계속 사용 (내부 용도). gateway 는 독자 구현. 공용 추상화는 두 경로가 안정화된 후에만 고려.

**D3. prefix index 의 gateway-local 저장 vs 분산.** 단일 gateway 에서는 local 이 단순하고 충분. 나중에 gateway 를 scale out 할 때만 분산 문제가 생긴다 — 그때 다시 본다.

### 4.2 리스크

**R1. Gateway 가 추가 hop 이 되어 [B] 경로 latency regression.** Phase 1 검증 기준에서 5% / 5ms 상한을 명시했다. 초과 시 in-process plugin 옵션 추가. [A] 경로 regression 은 불가침이며 0 을 유지.

**R2. vLLM 의 prefix hashing 이 버전 변경으로 깨질 수 있음.** prefix hit feedback loop (엔진이 실제 hit 여부를 gateway 로 알려줌) 를 두어 예측이 틀려도 자기교정 가능하게 한다.

**R3. multi-model 환경에서 KV 메모리 overcommit.** 같은 GPU 에 여러 모델이 로드되는 상황은 Phase 4 의 범위 밖 (하드웨어 풀링 문제). 초기엔 엔진 ↔ GPU 고정 매핑.

**R4. 기존 CLAUDE.md 의 `core.py` 무수정 원칙 및 GPU 경로 불가침 원칙.** B3 는 vLLM 외부 계층이라 구조적으로 유리. 그래도 엔진 내부 prefix hash 같은 정보가 필요해지면 `hybrid_core.py` 확장으로만 해결하고 core.py 는 손대지 않는다. GPU 경로 (기존 `vllm serve` 경로) 는 어떤 Phase 에서도 코드 변경 대상이 아니다.

### 4.3 Phase 간 재사용 / 기각 조건

- Phase 1 통과 = gateway 구조 성립 + [A] 경로 regression 0 확인 → 진행
- Phase 2 실패 (10% 이하 개선) 시 Phase 3 의 prefix locality 쪽으로 pivot. gateway 구조는 유지
- Phase 3 도 실패하면 B3 재검토 — coordination 가치가 이 시나리오에서는 낮다는 결론
- Phase 4 는 Phase 2/3 중 하나라도 성공한 뒤에만 진입

---

## Part V · 즉시 다음 할 일

Phase 0 착수. 구체적으로:

1. 대상 모델 2개를 고른다. 제안: Qwen2.5-7B-Instruct (현재 dev 환경과 호환) + Qwen2.5-32B-Instruct (현재 H100 환경)
2. workload 3종 정의: (a) single-model baseline, (b) priority-mix (2 tier), (c) prefix-heavy (공통 system prompt)
3. 평가 지표 네 가지 (multi-model goodput, prefix hit, priority SLO, router overhead) 에 측정 스크립트 설계
4. 결과를 `02_evaluation_plan.md` 로 기록

Phase 0 산출물이 나오면 Phase 1 skeleton 구현에 착수한다.
