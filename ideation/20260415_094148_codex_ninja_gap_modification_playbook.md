# Ninja Gap 수정 방안 총정리 — 합리적 수정 항목, 우선순위, 판정 기준

**작성 시각**: 2026-04-15 09:41:48 KST  
**작성자**: Codex

## 입력 문서

- [20260415_092738_claude_HPC_breakthrough_principles_v4.md](/vllm_hybrid/ideation/20260415_092738_claude_HPC_breakthrough_principles_v4.md)
- [20260415_085858_codex_hybrid_improvement_integrated_rewrite.md](/vllm_hybrid/ideation/20260415_085858_codex_hybrid_improvement_integrated_rewrite.md)
- [TODO.md](/vllm_hybrid/TODO.md)

## 목적

이 문서는 `Ninja Gap`을 실제로 만들 가능성이 있는 수정 방안만 추려서, **무엇을 바꿀지**, **어떤 순서로 바꿀지**, **언제 중단하거나 방향을 바꿀지**를 명확히 적는다.

핵심 전제:

- **request-level hybrid 유지**
- 목표는 **CPU가 더 많은 request 를 처리하면서 total wall time 을 줄이는 것**
- 따라서 본선은 **CPU batch 가 진짜 batch scaling 을 보이도록 만드는 것**

---

## 1. 현재 병목 정의

실측으로 확정된 현재 병목은 다음이다.

- `wave-batch + cpu-first` 는 CPU wave 를 먼저 채우고 GPU bulk 종료 후 CPU tail 을 남긴다
- `cpu_max_num_seqs` 증가는 현재 구조에서 throughput gain 이 아니라 tail 증폭으로 나타난다
- CPU path bring-up 은 이미 끝났고, 남은 질문은 **왜 CPU batch scaling 이 안 나오는가**이다

즉 지금의 핵심 문제는:

> 여러 request 를 CPU에 동시에 줘도 per-request cost 가 충분히 내려가지 않는다

이 정의가 바뀌지 않는 한 `routing`만 바꿔도 결과는 바뀌지 않는다.

---

## 2. 수정 방안의 우선순위 원칙

수정 방안은 다섯 층으로 나눈다.

1. **계측**
2. **저위험 설정 / 호출 경로 변경**
3. **batch scaling 을 만드는 kernel / dataflow 수정**
4. **그 위에서 routing / inflight 정책 재평가**
5. **병행 트랙: spec decode / long-context / large-model**

중요한 원칙:

- 한 단계의 성공은 단일 지표가 아니라 아래 4개를 같이 본다
  - CPU batch tok/s 증가
  - CPU handled requests 증가
  - CPU tail 감소
  - `hybrid wall / gpu_only wall` 개선
- 한 가지만 좋아져도 다음 단계로 넘어가지 않는다

---

## 3. 수정 방안 전체 목록

아래는 합리적인 수정 항목을 **당장 가능 / 중간 규모 / 큰 구조 변경**으로 나눈 것이다.

## 3-1. 계측과 기준선

### M1. CPU batch scaling sweep 추가

무엇:

- `num_seqs=1/2/4/8/16` 에서 CPU-only tok/s, per-request latency, total latency 측정

왜:

- 이걸 먼저 모르면 `wave-batch` 실패가 kernel 문제인지, sync 문제인지, memory 문제인지 구분이 안 된다

위치:

- `eval/cpu_profile_dev.sh`
- `eval/cpu_profile.sh`
- 필요하면 `eval/basic/*` 보조 스크립트

성공 신호:

- `num_seqs` 증가 시 CPU batch tok/s 증가
- per-request latency 가 완만히만 증가

### M2. sublayer 시간 분해

무엇:

- QKV / attention / O proj / gate / up / SiLU / down / norm 별 시간 비중 측정

왜:

- 어떤 sublayer를 fusion / LUT / pre-pack 대상으로 먼저 잡아야 할지 결정 가능

위치:

- `vllm/v1/worker/cpu_worker.py`
- 필요시 `csrc/cpu/utils.cpp` 또는 별도 profiling hook

### M3. sync / barrier / memory-wait 비중 측정

무엇:

- per-step 에서 barrier, sync, memory wait 비중 계측

왜:

- batch scaling 실패가 계산 부족인지, sync 지배인지 구분해야 한다

### M4. cache-fit 검증

무엇:

- staging buffer, LUT, fused intermediate 가 L1/L2/L3 안에 닫히는지 확인

왜:

- cache-fit 실패 시 cascade / fusion / LUT 모두 DDR 왕복으로 무력화될 수 있다

---

## 3-2. 저위험 설정 / 호출 경로 변경

### C1. `wave-batch` 기본 전략 해제

무엇:

- 기본 실험 전략을 `wave-batch` 에서 내리고 `throughput-adaptive` 또는 strict continuous baseline 으로 재측정

왜:

- 현재 `wave-batch` 는 가짜 동시성 증폭기다

위치:

- `eval/envs/*.env`
- [hybrid_core.py](/vllm_hybrid/vllm/v1/engine/hybrid_core.py)

### C2. `cpu_max_num_seqs=1` baseline 고정

무엇:

- batch scaling 확보 전 기본 baseline 은 `1`

왜:

- scaling 이 없는 상태에서 `2/4/8/16` 은 실패 재현용이지 개선용이 아니다

### C3. `chunked_prefill` 재점검

무엇:

- CPU config 에서 강제 비활성화된 `chunked_prefill`을 재실험 가능한 옵션으로 분리

왜:

- 현재 CPU prefill 직렬화가 batch scaling을 더 악화시킬 수 있다

위치:

- [hybrid_core.py](/vllm_hybrid/vllm/v1/engine/hybrid_core.py)
- `_create_cpu_vllm_config()`

### C4. Huge Pages

무엇:

- huge page 기반 weight / cache mapping 실험

왜:

- TLB pressure 감소 가능성

주의:

- 즉시 큰 개선을 가정하지 말고, low-risk system tweak 로만 본다

### C5. IPEX WoQ INT8 path 실험

무엇:

- IPEX weight-only INT8 경로를 실제 CPU hybrid path 에 연결해 실측

왜:

- low-risk 로 CPU request throughput baseline 을 끌어올릴 수 있는지 확인

위치:

- [cpu_worker.py](/vllm_hybrid/vllm/v1/worker/cpu_worker.py)
- model load / optimize hook

---

## 3-3. batch scaling 을 만드는 kernel / dataflow 수정

이 층이 `Ninja Gap` 본선입니다.

### K1. Head Folding

무엇:

- decode batch / head dimension 을 fold 해서 larger-M 연산으로 재배치

왜:

- request-level inflight 증가가 실제 GEMM 이득으로 연결될 가능성이 가장 크다

### K2. VNNI pre-pack / load-once-pack-twice

무엇:

- weight 또는 attention operand 를 미리 layout-aware 하게 재배치
- runtime repack 을 줄임

왜:

- batch 증가 시 runtime 재배치 오버헤드가 scaling 을 죽일 가능성이 크다

### K3. batch-aware decode attention

무엇:

- 현재 decode attention 이 per-seq loop 성격을 강하게 띠는지 확인하고, batch-aware 경로를 명시적으로 추가

왜:

- 현재 batch scaling 실패의 핵심 의심 지점이다

위치:

- [vllm/v1/attention/backends/cpu_attn.py](/vllm_hybrid/vllm/v1/attention/backends/cpu_attn.py)
- `csrc/cpu/*`

### K4. QKV fusion

무엇:

- `W_q`, `W_k`, `W_v` 를 concat / fused execution

왜:

- 동일 입력을 세 번 읽는 비용을 줄일 수 있다

### K5. Gate + Up fusion / interleave

무엇:

- SwiGLU 계열 `gate`, `up` projection 을 fused 또는 interleaved 로 실행

왜:

- 동일 입력 read amortization

### K6. Softmax / SiLU LUT

무엇:

- transcendental / activation 경로를 LUT 또는 cheap approximation 으로 치환

왜:

- batch-aware kernel 안에서 scalar/elementwise overhead를 줄일 수 있다

### K7. LUT GEMV / low-bit native path

무엇:

- T-MAC 계열 lookup-table low-bit path prototype

왜:

- low-bit + LUT는 batch scaling 과 dataflow 이득을 동시에 가져올 수 있는 가장 공격적인 후보 중 하나

주의:

- 강한 후보지만 현재 코드베이스에서는 **강한 가설**

### K8. AVX/AMX binary dispatch

무엇:

- batch/shape 별로 AVX-512 경로와 AMX 경로를 나눠 선택

왜:

- 모든 decode shape 에서 AMX가 유리하다고 가정하면 안 된다

### K9. AVX -> AMX cascade pipeline

무엇:

- AVX dequant / pack
- AMX matmul
- 필요시 prefetch / DSA load

를 타일 기반 파이프라인으로 겹치게 설계

왜:

- AVX와 AMX는 경쟁 관계만이 아니라 연속 단계로도 설계할 수 있다

주의:

- `zmm`와 `tile register`는 직접 연결되지 않음
- **타일 버퍼가 cache 안에 닫히는지** 먼저 봐야 한다

### K10. NUMA-local multi-engine 강화

무엇:

- CPU engine / buffer / memory binding 을 NUMA-local 하게 더 명시화

왜:

- H100x8 같은 2-NUMA 서버에서 batch scaling 을 유지하려면 locality 가 중요하다

### K11. barrier / sync 감소

무엇:

- OpenMP / worker sync / per-step barrier 재검토

왜:

- scaling 실패가 compute 가 아니라 barrier 지배일 수 있다

---

## 3-4. routing / inflight 정책 재평가

이건 반드시 **kernel/dataflow 수정 뒤**에 와야 한다.

### R1. `cpu_max_num_seqs` knee point 탐색

무엇:

- `1 -> 2 -> 4 -> 8` 순으로 확대

왜:

- scaling 이 생긴 뒤에야 CPU가 더 많은 request 를 맡을 최적점이 의미 있다

### R2. `wave-batch` vs `throughput-adaptive`

무엇:

- 동일 shape, 동일 model 에서 routing 비교

왜:

- CPU batch scaling 이 생긴 뒤엔 routing 이 다시 성능에 의미를 가질 수 있다

### R3. Property 2 gate 재평가

무엇:

- CPU throughput 이 올라간 뒤 gate가 실제로 CPU를 선택하는지 확인

왜:

- 지금은 CPU가 느려서 어떤 gate를 줘도 실패한다
- 나중엔 gate가 다시 핵심이 될 수 있다

---

## 3-5. 병행 트랙

이건 중요하지만 immediate mainline 과는 구분해야 한다.

### P1. Spec decode CPU drafter

왜:

- `TODO.md` 기준 A1 1순위
- request-level mainline 이 기대만큼 안 올라오면 가장 유력한 병행 카드

위치:

- [core_client.py](/vllm_hybrid/vllm/v1/engine/core_client.py)
- [hybrid_core.py](/vllm_hybrid/vllm/v1/engine/hybrid_core.py)
- CPU engine / verifier interface

### P2. FP8 KV cache

왜:

- low-risk 로 total operating point 확장

### P3. INT4 KV + CPU DRAM / LMCache

왜:

- 70B / long-context 에서 의미

주의:

- short 128/128 burst 문제의 직접 해법은 아님

### P4. Long-context / 70B 전용 offload

왜:

- current short burst 와 분리된 트랙으로 관리해야 함

---

## 4. 코드 수정 위치별 매핑

### 엔진 / 라우팅

- [vllm/v1/engine/hybrid_core.py](/vllm_hybrid/vllm/v1/engine/hybrid_core.py)
  - routing strategy 기본값
  - `cpu_max_num_seqs`
  - wave-batch / throughput-adaptive 비교
  - CPU config 생성 시 `chunked_prefill` 옵션화
  - Property 2 gate 조정

- [vllm/v1/engine/core_client.py](/vllm_hybrid/vllm/v1/engine/core_client.py)
  - CPU/GPU dispatch
  - finished/inflight accounting
  - spec decode 병행 트랙 진입 시 verifier interface

### CPU worker / 모델 로딩

- [vllm/v1/worker/cpu_worker.py](/vllm_hybrid/vllm/v1/worker/cpu_worker.py)
  - thread count / affinity / pinning
  - IPEX optimize hook
  - batch/shape 기반 ISA dispatch
  - profiling hook

### CPU attention / kernel

- [vllm/v1/attention/backends/cpu_attn.py](/vllm_hybrid/vllm/v1/attention/backends/cpu_attn.py)
  - batch-aware decode attention
  - IPEX path / custom kernel path 분기

- `csrc/cpu/*`
  - fusion kernel
  - LUT path
  - pre-pack
  - AVX/AMX cascade
  - sparse / bitmask path

### 평가 스크립트

- `/vllm_hybrid/eval/*`
  - num_seqs sweep
  - batch scaling / tail / wall ratio 보고
  - CPU-only / hybrid / gpu-only 비교

---

## 5. 단계별 실행 순서

## Phase 0. 기준선 계측

해야 할 일:

1. `num_seqs=1/2/4/8/16` CPU-only sweep
2. sublayer breakdown
3. sync / memory-wait 비중 확인
4. cache-fit 점검

통과 기준:

- 어떤 shape 에서라도 batch scaling 신호를 찾거나
- scaling 실패 원인이 어디인지 분해가 가능해야 한다

## Phase 1. 저위험 수정

해야 할 일:

1. `wave-batch` 기본선 해제
2. `cpu_max_num_seqs=1` strict baseline
3. `chunked_prefill` 옵션화 실험
4. Huge Pages
5. IPEX WoQ INT8 실험

통과 기준:

- baseline 재정립
- low-risk tweak 가 실제 hot path 에 의미 있는지 확인

## Phase 2. 본선 kernel/dataflow

해야 할 일:

1. Head Folding
2. pre-pack
3. batch-aware decode attention
4. QKV / Gate-Up fusion
5. barrier / sync 감소
6. AVX/AMX dispatch
7. 가능하면 cascade pipeline prototype

통과 기준:

- CPU batch tok/s 증가
- per-request cost 완만화
- tail 감소 방향 확인

## Phase 3. routing 재평가

해야 할 일:

1. `cpu_max_num_seqs` knee point 탐색
2. `wave-batch` vs `throughput-adaptive`
3. Property 2 gate 재평가

통과 기준:

- CPU handled requests 증가
- wall ratio 개선

## Phase 4. 병행 트랙 / 장문 트랙

해야 할 일:

1. spec decode drafter
2. FP8 KV
3. INT4 KV + CPU DRAM / LMCache
4. 70B / long-context baseline

---

## 6. Stop / Go 규칙

### 경우 1. CPU가 더 많이 처리했는데 wall ratio 가 안 좋아짐

해석:

- CPU throughput 이 아니라 CPU tail 만 늘어났다

조치:

- 성공으로 보지 않음
- `cpu_max_num_seqs` 확대 중단

### 경우 2. tok/s 는 올랐는데 tail 이 그대로

해석:

- kernel 은 빨라졌지만 routing / prefill / 상태 전이가 tail 을 다시 만들고 있다

조치:

- router, gate, prefill 직렬화 재검토

### 경우 3. kernel 수정 후에도 batch scaling 이 안 나옴

해석:

- hot path 를 못 건드렸거나
- cache-fit / sync / runtime repack 이 지배적

조치:

- 다음 단계로 가지 않음
- 계측 단계로 되돌아감

### 경우 4. request-level mainline 이 일정 수준 이상 안 올라옴

해석:

- 현재 구조에서 `Ninja Gap` 생성이 구조적으로 어렵다

조치:

- spec decode drafter 병행 트랙 비중 확대

---

## 7. 무엇을 당장 하지 말아야 하나

- batch scaling 확인 전 `cpu_max_num_seqs` 확대
- `wave-batch`를 기본 전략으로 유지
- CPU handled requests 증가만 보고 성공으로 판단
- bring-up 재증명 실험 반복
- short burst 문제를 안 푼 상태에서 long-context 전용 해법을 mainline 으로 올리기

---

## 8. 최종 제안

가장 합리적인 수정 방향은 다음 순서입니다.

1. **계측 강화**
2. **batch scaling 을 만드는 kernel/dataflow 수정**
3. **그 뒤 routing 재평가**
4. **mainline 이 막히면 spec decode 병행 트랙 확대**

즉 `Ninja Gap`을 만들기 위한 핵심 수정은:

- `wave-batch` 폐기 또는 후순위화
- `cpu_max_num_seqs` 확대 전 batch scaling 확보
- Head Folding / pre-pack / batch-aware attention / fusion / barrier 감소 / AVX-AMX dispatch
- cache-fit 검증
- 그 다음에만 routing / gate 최적화

한 문장으로 요약하면:

> CPU가 더 많은 request 를 맡게 하려면, 먼저 여러 request 를 함께 처리할 때 실제로 더 싸지는 구조를 만들어야 합니다. `Ninja Gap`은 routing 이 아니라 kernel/dataflow 에서 시작됩니다.

