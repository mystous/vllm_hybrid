# CPU Shadow Assists — New Idea 통합 설계

## Part I · 왜 이 문서가 다시 필요한가

세 번의 기각이 쌓였다. X (CPU async executor) 는 same-req 중복 compute 로 sync 대비 절반 throughput, B2 (heavy workload CPU decode shadow) 는 -96%, B3 (Meta-scheduling Gateway) 는 우리 포크의 차별점과 직교, B1 (Inverted Control Plane) 은 Blink 논문의 재구성에 가까워 독자 기여가 얇았다. 네 시도의 공통 실패 원인은 "가설이 크지만 우리만의 고유 지점이 없었다" 는 것이다.

`super_power/ideation/cpu_idle_acceleration_ideation_20260421.md` 를 다시 읽으면 이 문제의 해결 단서가 §2 (새 아이디어 8가지) 와 §4 (논문 지형의 빈 공간 4가지) 에 이미 놓여 있다. 이 문서는 그 8가지를 한 곳에 묶되, **직접 대응 논문이 없어 우리만의 독자 기여가 되는 것**과 **기존 연구의 적용·조합·역전으로 가치가 나는 것** 두 축으로 분리해 정리한다. 어느 쪽이 옳은지는 지금 판정할 수 없다. 각 축의 성격이 다르고, 추진 순서도 다르다.

독창 쪽은 **돌파 가치가 크나 근거가 얇다**. 선행 연구 적용 쪽은 **이득 범위가 알려져 있으나 우리 포크에서 고유성이 낮다**. 두 축을 동시에 운영하되, 공통 전제인 profile (현재 workload 에서 어느 CPU 유휴 구간이 실재하는지) 을 먼저 확정한 뒤 각 아이디어의 진입/기각이 결정된다.

## Part II · 독창 축 — 직접 대응 논문 없음

### 2.1 · CPU-assisted 동적 배치 planner (`IDE_001`)

vLLM scheduler 는 GPU 주 프로세스의 Python thread 에서 돈다. 매 step `scheduler.schedule` + metadata 조립 (position ids, attention mask, rope cache) 이 GPU forward 의 앞에 걸린다. 유휴 CPU engine 이 **다음 step 의 batch 구성과 metadata** 를 미리 만들어 GPU 에 건네면 scheduler 비용을 hide 한다. 방향은 Blink 와 같지만, **scheduler 오프로드 타깃을 hybrid 포크의 CPU engine 으로 특정**한다는 점이 차별점.

- 이론적 근거 — [vLLM anatomy blog (2025-09-05)](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) 의 "CPU-bound output processing" 지적, [APEX (arXiv 2506.03296)](https://arxiv.org/abs/2506.03296) async overlap 원리
- 리스크 — X 의 실패 패턴 (lookahead 중복 dispatch) 을 반복하지 않으려면 정확성 계약 선행
- 진입 조건 — sync hybrid 의 step total 에서 scheduler+metadata 구간이 ≥ 10% 확인

### 2.2 · CPU prefill-assist for medium context (`IDE_002`)

128 토큰 단문에선 GPU prefill 이 빠르다. 그러나 512~2K 중간 구간에선 GPU 가 prefill-compute-bound 가 되고 CPU 는 유휴로 남는다. 이 구간에 CPU 가 **초기 1~2 layer embedding / position encoding / rope table** 을 미리 준비해 GPU prefill TTFT 를 단축한다. prefill disaggregation 자체는 [DistServe (arXiv 2401.09670)](https://arxiv.org/abs/2401.09670) 에 있지만 GPU-GPU 분리만 다루고, **CPU 가 GPU prefill 을 보조하는 구조는 공개 연구에 직접 대응이 없다**.

- 리스크 — 근거 등급 D. 실측 전에는 이득 크기 불명
- 진입 조건 — medium context workload (512~2K) 에서 GPU prefill 이 실제로 compute-bound 이고 CPU 가 유휴임을 프로파일로 확인

### 2.3 · CPU Background Compiler (`IDE_003`)

§06 의 load-time Q8_0 변환은 이미 있지만, runtime 중에도 CPU 는 대부분 유휴다. 이 유휴 시간에 다음 layer 의 weight layout 을 **AMX-friendly tile packing** 하거나, activation 범위 통계를 계속 수집해 **다음 batch 의 동적 quantization scale** 을 업데이트한다. 공개 연구의 quantization 파이프라인은 "offline pre-pack" 이거나 "일회성 calibration" 이며, **runtime 중 CPU 가 지속적으로 다음 단계를 준비하는 online incremental 구조는 직접 대응이 없다**.

- 이론적 근거 — [APEX](https://arxiv.org/abs/2506.03296), [Async KV Prefetch (arXiv 2504.06319)](https://arxiv.org/abs/2504.06319) 의 "PCIe prefetch" 원리를 compute 쪽으로 확장
- 리스크 — D. quantization 정합성 유지 비용이 이득을 초과할 수 있음

### 2.4 · GPU-idle-phase CPU Burst (`IDE_004`)

Decode step 내부에서 GPU 는 phase 를 번갈아 탄다. attention (memory-bound) 구간에선 GPU 연산기가 놀고, linear matmul (compute-bound) 구간에선 memory 가 논다. 이 **sublayer phase 에 맞춰 CPU 가 서로 다른 보조 작업** 을 수행한다.

| GPU phase | CPU 작업 후보 |
|---|---|
| attention (memory-bound) | 다음 step scheduling, detokenize, grammar update |
| linear (compute-bound) | KV prefetch/evict, speculative draft, logits post-process |

[NEO (arXiv 2411.01142)](https://arxiv.org/abs/2411.01142) 의 asymmetric pipeline 을 **sublayer phase 단위로 세분화** 하는 연구는 공개된 것이 없다.

- 리스크 — D. sublayer 수준 정확한 phase 경계 검출과 CPU 작업 스위칭 오버헤드 필요
- 진입 조건 — GPU phase timing 을 step 내부에서 안정적으로 감지 가능한지 프로파일로 확인

## Part III · 선행 연구 적용 축

### 3.1 · CPU drafter + GPU verifier (`IDE_005`)

[Dovetail (arXiv 2412.18934)](https://arxiv.org/abs/2412.18934) 은 약한 GPU + 강한 CPU 환경에서 "CPU=target / GPU=drafter" 로 갔다. 우리는 정반대 — 강한 GPU + 유휴 CPU. 동일 논리를 역전해 CPU 가 작은 drafter (예: Qwen2.5-0.5B Q8_0) 를 돌리고 GPU 가 32B verifier 를 돌린다. single-request 로 보면 CPU drafter 가 느려 이득 없다는 것이 기존 §18 NinjaGap 에서의 판정이었고, 여기서 새 각도는 **여러 request 를 큐로 다중화**해 CPU drafter 의 느림을 request 수로 상쇄한다.

- 선행 연구 — [Leviathan et al. (arXiv 2211.17192)](https://arxiv.org/abs/2211.17192) spec decode 원전, [EAGLE (arXiv 2401.15077)](https://arxiv.org/abs/2401.15077) / [Medusa (arXiv 2401.10774)](https://arxiv.org/abs/2401.10774) tree draft, [DuoDecoding (arXiv 2503.00784)](https://arxiv.org/abs/2503.00784)
- 리스크 — 근거 B. 우리 hybrid 구조로 옮길 때 정확도 영향과 수락률이 측정 필요

### 3.2 · Cold-KV CPU Partial Attention (`IDE_006`)

기존 "CPU-side Cold KV staging" 정의는 기각한다. 인용했던 [InfiniGen (arXiv 2406.19707)](https://arxiv.org/abs/2406.19707) / [LMCache (arXiv 2510.09665)](https://arxiv.org/abs/2510.09665) / [Mooncake (arXiv 2407.00079)](https://arxiv.org/abs/2407.00079) 의 cold-KV 저장 기능은 이미 vLLM 업스트림에 들어와 있고 본 포크에도 inherit 되어 있다 (`vllm/v1/kv_offload/`, `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`, `lmcache_integration/` ≈ 2.4 kLOC). 기본은 비활성이지만 `--kv-transfer-config` 로 켜지는 운영 결정이지 신규 구현 대상이 아니다.

대신 IDE_006 을 **cold KV 블록 위에서 CPU 엔진이 partial attention 을 직접 수행**하는 구도로 재정의한다. LMCache / OffloadingConnector 가 cold KV 를 CPU DRAM 으로 내려 둔 상태를 그대로 활용하되, GPU 로 다시 끌어올리는 대신 **CPU 엔진이 cold 블록에 대한 Q·K^T softmax + V 가중합을 계산**해 partial output 과 log-sum-exp(LSE) 만 GPU 로 전송한다. GPU 는 hot KV 의 partial output·LSE 와 **online softmax** 로 합산만 한다. 합산 자체는 vLLM 내부에 이미 존재하는 prefix/suffix attention merge 경로(`csrc/attention/merge_attn_states.cu`, `vllm/v1/attention/backends/flash_attn.py:1134`)를 재사용한다. 본 포크의 차별점인 **CPU compute engine** 과 cold-KV 계층을 결합시키는 통합 시도다.

- 선행 연구 — [NEO (arXiv 2411.01142, OpenReview umgy9tWBLA)](https://openreview.net/forum?id=umgy9tWBLA) 가 가장 가깝다. asymmetric GPU/CPU pipeline 에서 attention compute + KV state 일부의 CPU offload 를 이미 다룬다. FastDecode 계열 CPU-side attention, [CachedAttention (USENIX ATC'24)](https://prongs1996.github.io/assets/pdf/CachedAttention.pdf) 다계층 KV reuse 도 인접 영역
- 독자 기여 — NEO / FastDecode 의 CPU attention 은 단일 메모리 계층을 가정하고, LMCache / OffloadingConnector 의 cold-KV 는 GPU 재호출만 한다. 차별점은 좁게 잡는다 — **vLLM native cold-tier offloading (LMCache/OffloadingConnector) 에 이미 내려간 KV 를 GPU reload 없이 CPU 에서 partial attention 으로 소비하고, vLLM 내부 LSE merge 경로로 합산하는 구조** 는 공개 연구·상용 시스템에서 **직접 대응이 확인되지 않았다**.
- 리스크 — 근거 C. (i) partial attention merge 결과가 GPU-only 결과와 사전 합의된 tolerance(rtol/atol) 내에서 일치해야 한다. FlashAttention 경로의 비결정성으로 bitwise 일치는 비현실적이며, 기준값은 PLN 단계에서 결정한다 (Constraint "GPU 결과와 동일" 의 운용 해석). (ii) CPU attention throughput + Q/partial 전송 비용이 GPU full-reload + full-attention 대비 net win 인 sweet spot 이 좁을 수 있다. (iii) OffloadingConnector 가 저장한 CPU tensor 를 attention 커널 입력으로 안전하게 복원할 layout/owner 계약이 미정.
- 현재 workload (128/128) 기여 — 0. cold KV 자체가 발생하지 않는다.
- 가치 성격 — 장기 + 독자 기여. workload 가 long-context (≥ 8K) 로 전환되고 CPU 엔진의 idle 시간이 attention 계산에 쓰일 만큼 충분할 때 작동.
- 진입 조건 — (a) long-context workload 전환, (b) CPU 엔진의 attention throughput 측정, (c) partial attention merge 가 GPU-only 결과와 사전 정의된 tolerance(rtol/atol) 내 일치, (d) GQA(Qwen2.5-7B 의 Q=32 / KV-heads=4 등) 에서 K/V split + head broadcast 동작 확인, (e) IDE_001/002/004 와 충돌하지 않는 CPU 작업 priority/conflict matrix 정의.

### 3.3 · CPU Speculative Logits Rerank (`IDE_007`)

GPU 가 top-k logits 를 출력하고 sampling 직전에, CPU 가 **양자화된 보조 모델의 last few layers** 만으로 logits 를 재평가. 두 결과가 합의하면 빠른 sampling, 불일치면 GPU 재확인. [EAGLE / Medusa](https://arxiv.org/abs/2401.15077) 계열 draft model 의 역방향.

- 리스크 — D. 정확도 영향 평가 필수. 근거 약함
- 우선순위 — 하위. 실측 전 진입 불가

### 3.4 · Constrained Decoding 전담 CPU Worker (`IDE_008`)

JSON mode / 함수 호출 / regex 제약 같은 **constrained decoding** 에서는 매 step grammar state update + token mask 계산이 CPU-bound 다. hybrid engine 의 CPU engine 1개를 "grammar/constraint 전용 워커" 로 재정의하고 일반 요청은 GPU, constrained 요청은 CPU 가 mask 계산 + GPU 가 masked sampling 으로 처리.

- 선행 연구 — [XGrammar](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar), [Guiding LLMs The Right Way (arXiv 2403.06988)](https://arxiv.org/abs/2403.06988), [llama.cpp grammar](https://github.com/ggml-org/llama.cpp)
- 한계 — workload 특이. 일반 텍스트 생성엔 기여 0
- 가치 성격 — 특정 워크로드 전용의 운영 기능

## Part IV · 우선순위 판단

| ID | 아이디어 | 축 | 근거 등급 | 현재 workload 기여 | 장기 가치 | 진입 조건 |
|---|---|---|:---:|:---:|:---:|---|
| `IDE_001` | 2.1 CPU-assisted planner | 독창 | C | 가능성 있음 | 중 | profile 에서 scheduler+metadata 비중 ≥ 10% |
| `IDE_002` | 2.2 CPU prefill-assist (medium) | 독창 | D | medium ctx 한정 | 중 | 512~2K 에서 GPU compute-bound 확인 |
| `IDE_003` | 2.3 CPU Background Compiler | 독창 | D | 불명 | 중 | quantization 정합성 유지 가능성 확인 |
| `IDE_004` | 2.4 Phase-aware CPU Burst | 독창 | D | 불명 | 높음 | sublayer phase 경계 감지 가능성 확인 |
| `IDE_005` | 3.1 CPU drafter + GPU verifier | 적용 | B | 가능성 있음 | 높음 (구조적) | 다중화 request 큐 가정 성립 확인 |
| `IDE_006` | 3.2 Cold-KV CPU Partial Attention | 적용 | C | 0 (128/128) | 높음 (long-ctx + 독자 기여) | long-ctx 전환 + CPU attn throughput + tolerance(rtol/atol) 일치 + GQA 동작 + 자원 경합 matrix |
| `IDE_007` | 3.3 Speculative logits rerank | 적용 | D | 불명 | 낮음 | 정확도 영향 측정 |
| `IDE_008` | 3.4 Constrained decoding worker | 적용 | B | workload 특이 | 중 (특수) | constrained workload 시점 |

두 축은 다른 순서로 진행한다. **독창 축은 profile 을 선행 관문으로 공유**한다 (2.1 의 scheduler 비중, 2.2 의 medium-ctx GPU compute 비중, 2.4 의 sublayer phase 감지 가능성). 한 번의 profile run 이 세 아이디어의 진입 여부를 동시에 판정한다. **선행 연구 적용 축은 workload 와 운영 요건에 따라 트리거**된다 — 3.1 은 spec decode 수락률 측정이, 3.2 는 long-context 전환이, 3.4 는 JSON/tool-calling 수요가 진입 조건이다.

## Part V · 경계선

- **기존 GPU 경로, 기존 vLLM 엔진, 기존 hybrid 엔진은 무변경이 기본값**이다. 여기의 어떤 아이디어도 활성화되지 않은 구성에서는 현재 바이너리·동작 그대로.
- **vLLM core.py 는 건드리지 않는다** (CLAUDE.md 원칙).
- 각 아이디어는 **숫자로 진입·기각**이 결정된다. D 급 근거는 profile 없이 진입하지 않는다.
- 기각된 과거 축 (X, B1, B2, B3) 의 교훈을 반복하지 않는다. 특히 X 의 "lookahead 중복 dispatch" 패턴이 2.1 에서 재발할 위험을 의식한다.

## Part VI · 지금 할 일

Phase 0 으로 **한 번의 profile run** 만 돌린다. 이 run 이 독창 축 4개의 진입 조건을 동시에 관측하도록 구성한다.

1. sync hybrid 에서 `VLLM_HYBRID_PROFILE=1` + sublayer breakdown 을 켜고 light / medium (512~2K) / heavy 세 workload 돌린다.
2. 수집 대상:
   - scheduler.schedule + metadata 조립의 step total 비중 (2.1 진입 조건)
   - medium context 에서 GPU prefill 의 compute vs memory 구간 구분 (2.2)
   - decode step 내 sublayer phase 경계 감지 가능성 (2.4)
   - CPU engine 의 idle 비중 — 어느 phase 에서 얼마나 유휴인지 (2.3 의 quantization preparation 에 쓸 시간 창 존재 확인)
3. 결과를 `01_phase0_profile.md` 로 기록하고 4개 독창 아이디어의 진입 여부를 동시에 판정한다.

Phase 0 이 끝난 뒤에야 이 디렉토리에 `02_*`, `03_*` 가 생긴다. 판정 결과에 따라 어떤 아이디어는 즉시 기각되고 어떤 아이디어는 설계 상세 문서가 따라 붙는다.

## Part VII · ID Trace

CLAUDE.md Ground RULE 에 따라 본 저장소에서 사용되는 모든 ID 를 이 섹션에서 추적한다. Tree 는 Mermaid 로 작성하며 위→아래 방향으로 깊이가 늘어난다. 새로운 ID 가 등장할 때마다 Legend 와 Tree 양쪽을 함께 갱신한다.

### Legend (prefix)

| Prefix | 의미 |
|---|---|
| `IDE` | Idea — 구현 후보 |
| `PLN` | Plan — PoC / microbench 플랜 |
| `TSK` | Task — feature 구현 작업 단위 |
| `TST` | Test — 정확도·throughput·통합성 검증 단위 |
| `FEA` | Feature — 본 코드 베이스 단위 기능 |

> 넘버링 규칙·할당 현황·상태값은 [`shadow_assists/id_registry.md`](id_registry.md) 가 단일 출처. 본 Legend 는 prefix 의미 요약일 뿐이며, 새 prefix 추가 시 양쪽을 함께 갱신한다.

### Trace Tree

```mermaid
flowchart TB
    ROOT["super_power/ideation/<br/>cpu_idle_acceleration_ideation_20260421.md"]

    ROOT --> AXIS_A["독창 축<br/>(직접 대응 논문 없음)"]
    ROOT --> AXIS_B["선행 연구 적용 축"]

    AXIS_A --> IDE_001["IDE_001<br/>CPU-assisted 동적 배치 planner"]
    AXIS_A --> IDE_002["IDE_002<br/>CPU prefill-assist (medium context)"]
    AXIS_A --> IDE_003["IDE_003<br/>CPU Background Compiler"]
    AXIS_A --> IDE_004["IDE_004<br/>GPU-idle-phase CPU Burst"]

    AXIS_B --> IDE_005["IDE_005<br/>CPU drafter + GPU verifier"]
    AXIS_B --> IDE_006["IDE_006<br/>Cold-KV CPU Partial Attention"]
    AXIS_B --> IDE_007["IDE_007<br/>CPU Speculative Logits Rerank"]
    AXIS_B --> IDE_008["IDE_008<br/>Constrained Decoding 전담 CPU Worker"]
```

> 축 노드(`독창 축`, `선행 연구 적용 축`)는 분류용 가상 노드이며 ID 를 가지지 않는다. 진입 판정 후 각 IDE 노드 아래로 후속 prefix(예: `FEA_###` → `TSK_###` → `TST_###`)가 매달리며 Tree 가 아래로 깊어진다.
