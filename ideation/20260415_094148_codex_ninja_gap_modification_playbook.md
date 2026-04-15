# Ninja Gap Modification Playbook — 구현 상태 반영 재작성본

**최초 작성**: 2026-04-15 09:41:48 KST  
**재작성**: 2026-04-15 14:29:54 KST  
**작성자**: Codex

## 입력 문서

- [20260415_092738_claude_HPC_breakthrough_principles_v4.md](/vllm_hybrid/ideation/20260415_092738_claude_HPC_breakthrough_principles_v4.md)
- [20260415_085858_codex_hybrid_improvement_integrated_rewrite.md](/vllm_hybrid/ideation/20260415_085858_codex_hybrid_improvement_integrated_rewrite.md)
- [TODO.md](/vllm_hybrid/TODO.md)
- [analysis_h100.ipynb](/vllm_hybrid/eval/basic/H100x8/analysis_h100.ipynb)

## 0. 이 문서의 역할

이 문서는 `Ninja Gap`을 만들기 위한 수정 방안을 단순 나열하지 않는다. 다음 네 가지를 분리한다.

- **이미 구현된 것**: 성능 향상분으로 다시 계산하지 않는다
- **현재 실측으로 확정된 실패 구조**: H100x8/GPU-only 비교에서 보이는 병목
- **Ninja Gap을 만들 수 있는 신규 수정**: 실제로 CPU batch scaling을 만들 가능성이 있는 변경
- **Stop/Go 규칙**: 어떤 결과가 나오면 계속하고, 어떤 결과가 나오면 버릴지

핵심 전제는 유지한다.

- **request-level hybrid 유지**
- 목표는 **CPU가 더 많은 request를 처리하면서 total wall time을 줄이는 것**
- 본선은 routing이 아니라 **CPU batch가 진짜 batch scaling을 보이도록 kernel/dataflow를 바꾸는 것**

## 1. 현재 숫자로 본 Ninja Gap

H100x8 현재 기준선:

| 구분 | CPU req | GPU req | wall | GPU-only 대비 |
|---|---:|---:|---:|---:|
| GPU-only | 0 | 500 | 14.01s | 1.0x |
| Hybrid seq=1 best | 2 | 498 | 364.41s | 26.0x |
| Hybrid seq=1 worst | 2 | 498 | 417.67s | 29.8x |
| Hybrid seq=16 best | 32 | 468 | 1993.89s | 142.3x |
| Hybrid seq=16 worst | 32 | 468 | 2003.19s | 143.0x |

따라서 현재 구조에서 보이는 사실은 명확하다.

- CPU가 `2 req`만 가져가도 wall time이 `~6분`으로 늘어난다
- CPU가 `32 req`를 가져가면 wall time이 `~33분`으로 늘어난다
- `cpu_max_num_seqs` 확대는 현재 상태에서 throughput gain이 아니라 tail amplification이다
- pinning이 `112..167/168..223`에서 `0..55/56..111`로 바뀌어도 이 실패 구조는 유지된다

Ninja Gap의 실질 목표는 다음이다.

> CPU가 가져가는 request 수를 늘리면서도 `max(T_gpu_bulk, T_cpu_work)`에서 `T_cpu_work`가 wall을 지배하지 않게 만드는 것.

즉 단일 CPU request를 조금 빠르게 하는 것만으로는 부족하다. 여러 request를 같이 처리할 때 per-request cost가 실제로 낮아져야 한다.

## 2. 이미 구현된 항목과 제외 규칙

아래 항목은 코드에 이미 들어가 있다. 이 문서에서 **추가 성능 향상분으로 산출하지 않는다**.

| 항목 | 상태 | 근거 | gain 계산 |
|---|---|---|---|
| CPU engine launch | 구현됨 | `hybrid_core.py`, `core_client.py` | 제외 |
| `wave-batch` routing | 구현됨 | `CapacityAwareRouter` | 제외 |
| `throughput-adaptive` routing | 구현됨 | `CapacityAwareRouter` | 제외 |
| `cpu_max_num_seqs=1` auto baseline | 구현됨 | `_resolve_cpu_params()` | 제외 |
| CPU `chunked_prefill=False` | 구현됨 | `_create_cpu_vllm_config()` | 제외 |
| CPU core pinning | 구현됨 | `CPUWorker.init_device()` + `_C_utils.init_cpu_threads_env` | 제외 |
| NUMA node 기반 CPU 선택 | 구현됨 | `hybrid_config.numa_bind_node`, `_get_autobind_cpu_ids()` | 제외 |
| NUMA memory bind | 구현됨 | `numa_set_membind()`, `NUMAAllocator.bind_to_node()` | 제외 |
| affinity reset after fork | 구현됨 | `_setup_cpu_process_env()` | 제외 |
| feature 기반 ONEDNN ISA 설정 | 구현됨 | `intel_cpu_utils.py` | 제외 |
| VNNI INT8 GEMM 토대 | 부분구현 | `csrc/cpu/gemm_vnni.*`, `torch_bindings_hybrid.cpp` | hot path 연결분만 계산 |
| attn/mlp coarse profiling | 부분구현 | `cpu_worker.py` forward hook | sublayer 확장분만 계산 |

중요한 해석:

- “NUMA locality를 해야 한다”는 말은 지금 문서에서 더 이상 신규 gain 항목이 아니다. 기본 골격은 이미 있다.
- “chunked prefill을 꺼야 한다”도 신규 gain이 아니다. 이미 꺼져 있다.
- “VNNI가 있다”도 신규 gain이 아니다. 현재 hybrid decode hot path에 실제로 연결되어 batch scaling을 만든 경우만 gain으로 본다.
- `wave-batch`와 `throughput-adaptive`의 존재 자체는 개선이 아니다. routing은 kernel/dataflow 개선 뒤 재평가할 대상이다.

## 3. 실패 구조의 정확한 정의

현재 실패는 “CPU가 느리다”가 아니라 더 구체적으로 다음이다.

> CPU scheduler가 여러 request를 한 batch로 잡아도, 실제 hot path가 그 request들을 GPU처럼 효율적인 larger-M 연산으로 바꾸지 못한다.

그 결과:

- request 수만 늘어난다
- memory traffic은 req별로 반복된다
- runtime packing/repacking이 반복된다
- OMP barrier/sync 비용이 누적된다
- GPU가 끝난 뒤 CPU wave drain이 wall을 결정한다

따라서 잘못된 개선 방향은 다음이다.

- `cpu_max_num_seqs`만 올리기
- `wave-batch`를 더 크게 만들기
- pinning/NUMA bring-up을 다시 증명하기
- CPU handled request 수만 보고 성공 판단하기

맞는 개선 방향은 다음이다.

- 여러 request를 함께 처리할 때 input read, weight read, KV scan, activation, projection 비용이 amortize되도록 hot path를 바꾼다
- batch scaling이 생긴 뒤에만 CPU inflight/routing을 키운다

## 4. 성공 지표

Ninja Gap은 단일 metric으로 판정하지 않는다.

| 축 | 봐야 할 값 | 성공 방향 |
|---|---|---|
| CPU scaling | `cost(batch=N) / cost(batch=1)` | N보다 훨씬 작아야 함 |
| CPU throughput | CPU-only tok/s, req/s | batch 증가와 함께 증가 |
| Tail | GPU bulk 이후 CPU-only drain | 감소 |
| Wall ratio | `hybrid wall / gpu_only wall` | 감소 |
| CPU contribution | CPU handled req | 증가 |

성공으로 인정하려면 최소한 다음 네 가지가 같이 좋아져야 한다.

- CPU가 더 많은 request를 처리한다
- CPU batch tok/s가 오른다
- CPU tail이 줄어든다
- hybrid wall ratio가 좋아진다

CPU handled request만 늘고 wall이 나빠지면 실패다.

## 5. 경유지

현재 위치는 G0 이전이다. CPU batch scaling의 원인 분해가 아직 부족하다.

| Gate | 목표 | 통과 조건 | 실패 시 |
|---|---|---|---|
| G0 | 기준선 분해 | seq=1/2/4/8/16 CPU-only scaling, sublayer breakdown 확보 | 계측부터 보강 |
| G1 | scaling 징후 | 4req cost ≤ 2x single, tail < 100s, wall ratio < 8x | hot path 못 건드린 것 |
| G2 | routing 재평가 | 4req cost ≤ 1.5x single, tail < 10s, wall ratio < 1.5x | routing/gate 재설계 |
| G3 | Ninja Gap | CPU req 증가 + tail 제거 + wall ≤ GPU-only | 목표 달성 |

클로드 문서의 `G1/G2/G3` 숫자는 방향성으로는 좋지만, 현재 코드에서 확정 수치가 아니다. 여기서는 **실험 통과 기준**으로만 사용한다.

## 6. 실행 우선순위

### Tier -1. 계측 재정의

목표:

- CPU batch가 왜 batch scaling을 못 만드는지 분해한다.

수정:

- `eval/cpu_profile*.sh`에 `num_seqs=1/2/4/8/16` sweep 고정
- CPU-only와 hybrid CPU engine의 동일 shape 비교
- `cpu_worker.py` coarse `attn/mlp` hook을 QKV / O / Gate / Up / SiLU / Down / Norm 수준으로 확장
- per-step barrier/sync time, memory wait, packing/repacking time marker 추가
- H100x8와 dev 머신의 결과를 같은 CSV schema로 저장

성공 산출물:

- `batch_scaling_ratio = step_ms(batch=N) / step_ms(batch=1)`
- `per_req_cost = step_ms / active_reqs`
- sublayer별 top bottleneck
- `num_seqs` 증가 시 어떤 sublayer가 폭증하는지

이 단계에서 새 성능을 기대하지 않는다. 이 단계의 목적은 “어디를 고쳐야 하는지”를 확정하는 것이다.

### Tier 0. 기준선 방어

목표:

- 잘못된 routing/tail 증폭을 막고, 이후 kernel 실험을 안정적으로 비교한다.

수정:

- 기본 실험은 `cpu_max_num_seqs=1`로 고정
- `wave-batch`는 기본 전략에서 내리고 비교 대상으로만 유지
- `throughput-adaptive`와 strict continuous baseline을 같은 workload에서 비교
- CPU `chunked_prefill=False`는 이미 구현됨으로 표시하고, 필요 시 옵션화만 한다
- Huge Pages와 IPEX WoQ INT8은 별도 low-risk 실험으로 분리

주의:

- 이 Tier는 Ninja Gap 본선이 아니다
- `wave-batch` 제거로 tail이 줄어도 CPU 처리량 기여가 없으면 성공으로 보지 않는다

### Tier 1. Mainline Hot Path 연결

목표:

- 이미 있는 VNNI/ISA 토대를 실제 hybrid CPU decode hot path에 연결한다.

근거:

- KTransformers는 CPU/GPU hybrid가 CPU compute와 CPU-GPU synchronization에 막힌다는 문제를 명시하고, AMX-specialized CPU kernels와 tiling-aware layout을 핵심 최적화로 제시한다
- SGLang/KTransformers 통합 글은 AMX-optimized CPU kernels, AVX-512/AMX 동적 전환, coordination overhead 감소가 hybrid throughput에 직접 영향을 준다고 설명한다
- 이 근거는 “ISA 설정만으로 충분하다”가 아니라, **실제 hot path kernel/layout 연결이 필요하다**는 쪽을 뒷받침한다

수정 후보:

- VNNI INT8 GEMM 경로를 실제 Qwen2.5 CPU linear hot path에 연결
- load-time weight pre-pack cache 추가
- runtime repack이 step마다 발생하는지 계측 후 제거
- `batch=1/2/4/8/16` shape별 AVX/VNNI/oneDNN path dispatch 기록
- `ONEDNN_MAX_CPU_ISA` 설정 존재가 아니라 실제 primitive dispatch를 로그/프로파일로 확인

성공 조건:

- hot path가 실제로 바뀐 로그 또는 marker가 있어야 한다
- `num_seqs=4`에서 per-request cost가 감소해야 한다
- 단일 req만 빨라지고 batch scaling이 없으면 다음 Tier로 못 간다

### Tier 2. 진짜 Batch Scaling Kernel/Dataflow

목표:

- CPU inflight 증가가 실제 larger-M 효율 또는 data reuse로 이어지게 한다.

근거:

- T-MAC은 low-bit CPU inference에서 dequantize-then-compute 대신 LUT 기반 mixed-precision GEMM을 사용해 dequant overhead와 dataflow 문제를 정면으로 줄인다
- T-MAN은 NPU 문맥이지만, dequant / load / matmul을 pipeline으로 겹치고 tiling/layout을 통합해 decode/prefill 모두에서 dataflow 이득을 만든다는 점이 중요하다
- SparAMX는 AMX와 sparsity를 linear layer와 attention에 적용해 CPU token generation 경로에도 kernel-level 개선 여지가 있음을 보여준다
- 따라서 이 Tier의 핵심은 “더 많은 request를 넣기”가 아니라, **request 증가가 data reuse와 larger effective matrix work로 바뀌도록 hot path를 재구성하는 것**이다

수정 후보:

1. **Head Folding**
   - decode batch와 head dimension을 재배치해 작은 GEMV 반복을 더 큰 matrix 연산으로 바꾼다
   - 기대 효과는 request 수 증가가 kernel efficiency 증가로 연결되는 것

2. **batch-aware decode attention**
   - per-seq loop 성격의 decode attention을 batch-aware path로 분리
   - KV scan / score / softmax / value accumulation이 req별 반복되는지 제거 후보를 찾는다

3. **QKV fusion**
   - 동일 hidden state를 Q/K/V projection에서 반복 read하지 않게 한다
   - 이미 fused weight가 모델 구조에 있는지 먼저 확인하고, 없으면 CPU path 전용 concat/fused call을 추가한다

4. **Gate + Up fusion**
   - SwiGLU의 gate/up이 같은 input을 두 번 읽는 비용을 줄인다
   - gate/up/down 전체를 무리하게 한 번에 합치기보다, input read amortization부터 본다

5. **Softmax/SiLU LUT**
   - scalar transcendental overhead가 확인된 경우에만 적용한다
   - top bottleneck이 아니면 후순위다

6. **barrier/sync 감소**
   - OMP parallel region을 sublayer마다 새로 열고 닫는지 확인
   - thread team 재사용, chunk scheduling, layer/block 단위 persistent region 가능성을 본다

성공 조건:

- `num_seqs=4`가 `1 req` 대비 cost 2x 이하로 내려가야 한다
- CPU handled req를 늘렸을 때 wall이 악화되지 않아야 한다

### Tier 3. AVX/AMX Cascade Pipeline

목표:

- AVX와 AMX를 단순 binary switch가 아니라 타일 파이프라인으로 설계한다.

근거:

- KTransformers 계열 자료는 decode/prefill shape에 따라 AVX-512와 AMX의 유리한 영역이 다르다는 관점을 제공한다
- T-MAN의 3-stage pipeline은 NPU DMA/vector/matrix 구조이므로 x86 CPU에 직접 복사할 수는 없지만, `load -> vector dequant/pack -> matrix compute`를 pipeline으로 겹친다는 설계 원칙은 AVX/AMX cascade 후보의 근거가 된다
- 이 근거는 아직 우리 코드베이스에서 실측 검증되지 않았으므로, 본 문서에서는 **강한 가설**로 취급한다

구조:

- prefetch 또는 DSA 후보: `tile k+2` load
- AVX-512: `tile k+1` dequant / pack / elementwise
- AMX: `tile k` matmul

주의:

- AVX `zmm`와 AMX `tile register`는 직접 연결되지 않는다
- 중간 tile buffer가 L1/L2/L3에 닫혀야 한다
- cache-fit 실패 시 pipeline이 아니라 DDR 왕복 증가가 된다

수정:

- tile size별 buffer footprint 계산
- L2 fit 가능한 staging layout 설계
- `batch=1`은 AVX path, `batch>=N`은 AMX/cascade path 같은 shape-aware dispatch
- 전환 비용과 tile config 비용을 profile marker로 분리

성공 조건:

- cascade path가 특정 shape에서 standalone AVX 또는 standalone AMX보다 빠르다
- memory wait 비중이 증가하지 않는다

### Tier 4. Routing 재평가

조건:

- Tier 1~3에서 CPU batch scaling이 실제로 생긴 뒤에만 진입한다.

수정:

- `cpu_max_num_seqs` knee point 탐색: `1 -> 2 -> 4 -> 8 -> 16`
- `wave-batch` vs `throughput-adaptive` vs strict continuous 비교
- Property 2 gate 재평가
- CPU engine별 observed throughput을 gate에 반영

성공 조건:

- CPU req 증가와 wall ratio 개선이 동시에 나와야 한다
- CPU wave가 tail로 고착되면 즉시 이전 Tier로 되돌린다

### Tier 5. 병행 트랙

이 트랙은 중요하지만 이 문서의 mainline과 분리한다.

- Spec decode CPU drafter
- FP8 KV cache
- INT4 KV + CPU DRAM / LMCache
- 70B / long-context 전용 offload

이들은 request-level CPU batch scaling 실패를 숨기는 도피처로 쓰면 안 된다. mainline이 일정 기간 실패할 때 병행 비중을 높인다.

근거:

- DuoDecoding은 CPU draft model과 GPU target model을 병렬화하는 heterogeneous speculative decoding을 제시한다
- NEO는 GPU attention compute/KV-cache 일부를 CPU로 offload하고 asymmetric GPU-CPU pipelining과 load-aware scheduling을 사용해 H100에서도 GPU-only 대비 throughput 개선을 보고한다
- 다만 둘 다 현재 문서의 mainline인 “같은 request-level CPU executor의 batch scaling”과는 다른 구조이므로 병행 트랙으로 둔다

## 7. 코드 수정 위치

### 계측

- [vllm/v1/worker/cpu_worker.py](/vllm_hybrid/vllm/v1/worker/cpu_worker.py)
  - sublayer hook 확장
  - step-level barrier/sync marker
  - actual thread count / active req / scheduled token logging

- `eval/cpu_profile*.sh`
  - `num_seqs` sweep
  - CPU-only / hybrid CPU engine 비교

- [eval/basic/H100x8/analysis_h100.ipynb](/vllm_hybrid/eval/basic/H100x8/analysis_h100.ipynb)
  - GPU-only 대비 wall / dispatch / pinning 비교

### 라우팅

- [vllm/v1/engine/hybrid_core.py](/vllm_hybrid/vllm/v1/engine/hybrid_core.py)
  - default strategy
  - `cpu_max_num_seqs`
  - `wave-batch` admission
  - `throughput-adaptive` gate
  - CPU config 옵션화

- [vllm/v1/engine/core_client.py](/vllm_hybrid/vllm/v1/engine/core_client.py)
  - dispatch accounting
  - request finished accounting
  - CPU/GPU observed throughput feedback

### CPU hot path

- [vllm/v1/attention/backends/cpu_attn.py](/vllm_hybrid/vllm/v1/attention/backends/cpu_attn.py)
  - batch-aware decode attention
  - IPEX path vs custom path 분기

- `csrc/cpu/*`
  - VNNI pre-pack hot path 연결
  - fusion kernel
  - LUT path
  - AVX/AMX dispatch
  - cascade prototype

- [vllm/v1/worker/cpu_model_runner.py](/vllm_hybrid/vllm/v1/worker/cpu_model_runner.py)
  - model load-time optimize/pre-pack hook
  - NUMA-local allocator 일관성 확인

## 8. PR 단위 작업 순서

### PR 1. Batch Scaling Profiler

내용:

- CPU-only `num_seqs=1/2/4/8/16` sweep
- hybrid CPU engine 동일 metric logging
- CSV/JSON schema 고정

완료 조건:

- H100x8에서 seq별 step time과 per-request cost가 표로 나온다

### PR 2. Fine-grained Sublayer Profiler

내용:

- attn/mlp coarse hook을 QKV/O/Gate/Up/SiLU/Down/Norm 수준으로 확장
- packing/repacking marker 추가

완료 조건:

- batch scaling 실패의 top-2 원인이 sublayer 단위로 보인다

### PR 3. VNNI/Pre-pack Hot Path Wiring

내용:

- 기존 VNNI 토대를 Qwen CPU linear hot path에 실제 연결
- load-time pre-pack cache
- shape별 dispatch log

완료 조건:

- 로그로 실제 VNNI/pre-pack path 사용 확인
- seq=4 per-request cost 개선

### PR 4. Batch-aware Attention Prototype

내용:

- CPU decode attention에서 req 간 결합이 가능한 부분 분리
- per-seq loop 제거 후보를 prototype으로 구현

완료 조건:

- seq=4/8에서 attention cost가 sublinear하게 증가

### PR 5. Fusion Prototype

내용:

- QKV 또는 Gate+Up 중 profiler상 더 큰 쪽부터 구현
- input read amortization 확인

완료 조건:

- memory-read dominated 구간이 줄어든다

### PR 6. Routing Re-enable

내용:

- batch scaling이 확인된 shape에서만 `cpu_max_num_seqs` 확대
- routing strategy 비교

완료 조건:

- CPU handled request 증가와 wall ratio 개선이 동시에 나온다

## 9. Stop / Go 규칙

### Case 1. CPU handled req는 늘었지만 wall이 악화

판정:

- 실패

조치:

- `cpu_max_num_seqs` 확대 중단
- kernel/dataflow 단계로 되돌아감

### Case 2. CPU tok/s는 올랐지만 tail이 그대로

판정:

- 부분 실패

가능 원인:

- routing이 여전히 wave tail을 만든다
- prefill/decode 경계가 직렬화되어 있다
- finished/inflight accounting이 늦다

조치:

- routing/gate/prefill 상태 전이 재검토

### Case 3. 단일 req만 빨라지고 batch scaling 없음

판정:

- Ninja Gap 관점에서는 실패

조치:

- single-request 최적화로만 분류
- CPU inflight 확대 근거로 사용하지 않음

### Case 4. kernel 수정 후 metric 변화 없음

판정:

- hot path 미타격

조치:

- profile marker로 실제 호출 여부 확인
- 다음 kernel로 넘어가지 말고 계측으로 복귀

## 10. 하지 말아야 할 것

- batch scaling 확인 전 `cpu_max_num_seqs` 확대
- `wave-batch`를 기본 전략으로 유지
- NUMA/pinning bring-up 재증명에 시간 소모
- 이미 구현된 chunked prefill off를 신규 gain으로 계산
- VNNI 토대 존재만 보고 INT8 성능 향상으로 계산
- CPU request 수 증가만 보고 성공 판단
- H100x8 short burst 문제를 못 푼 상태에서 long-context 전용 해법을 mainline으로 올리기

## 11. 최종 정리

Ninja Gap을 만들려면 다음 순서가 가장 합리적이다.

1. **CPU batch scaling을 계측으로 분해**
2. **기존 VNNI/ISA 토대를 실제 hot path에 연결**
3. **Head Folding / batch-aware attention / fusion으로 진짜 batch scaling 생성**
4. **AVX/AMX cascade는 cache-fit이 확인된 shape에서만 적용**
5. **그 뒤에만 routing과 `cpu_max_num_seqs`를 다시 키움**

한 문장으로 요약하면:

> 현재 문제는 CPU에 request를 너무 적게 준 것이 아니라, 많이 줬을 때 계산이 싸지지 않는 것입니다. Ninja Gap은 CPU inflight 확대가 아니라, inflight 확대가 실제 data reuse와 kernel efficiency로 바뀌는 순간에 생깁니다.

## 12. 근거 자료와 적용 수준

아래 자료는 이 문서의 수정 방향을 뒷받침하는 근거다. 단, 외부 수치를 현재 코드베이스의 예상 개선률로 직접 환산하지 않는다. 모든 수치는 “가능성의 근거”이지 “우리 구현의 보장된 gain”이 아니다.

| 자료 | 링크 | 핵심 내용 | 이 문서에서 쓰는 방식 | 적용 수준 |
|---|---|---|---|---|
| H100x8 local analysis notebook | [analysis_h100.ipynb](/vllm_hybrid/eval/basic/H100x8/analysis_h100.ipynb) | GPU-only `14.01s`, hybrid seq=1 `364~418s`, seq=16 `1994~2003s`; CPU dispatch 2/32 req tail 확인 | 현재 실패 구조와 Ninja Gap 크기 정의 | 직접 근거 |
| H100x8 log analysis | [20260414_213415_codex_h100x8_log_analysis.md](/vllm_hybrid/eval/h100x8/20260414_213415_codex_h100x8_log_analysis.md) | CPU engine launch, wave close/drain, CPU tail 구조 정리 | CPU path bring-up은 완료됐고 남은 병목은 batch scaling이라는 판단 | 직접 근거 |
| T-MAC | [Microsoft Research](https://www.microsoft.com/en-us/research/publication/t-mac-cpu-renaissance-via-table-lookup-for-low-bit-llm-deployment-on-edge/), [GitHub](https://github.com/microsoft/T-MAC), [arXiv 2407.00088](https://arxiv.org/abs/2407.00088) | LUT 기반 low-bit CPU mpGEMM, dequantization 회피, CPU low-bit throughput/energy 개선 | LUT, low-bit native path, dataflow 재설계의 근거 | 강한 후보 |
| T-MAN | [Hugging Face paper page](https://huggingface.co/papers/2511.11248), [T-MAC/t-man code path](https://github.com/microsoft/T-MAC/tree/main/t-man) | unified table lookup, fused dequantization, three-stage pipeline, NPU decode/prefill dataflow 개선 | AVX/AMX cascade와 load/vector/matrix pipeline의 설계 힌트 | 가설적 차용 |
| KTransformers | [MADSys publication](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/), [project site](https://ktransformers.net/) | CPU/GPU hybrid에서 CPU compute limit과 synchronization overhead를 문제로 보고, AMX-specialized kernels와 layout 최적화를 사용 | CPU hot path kernel/layout 직접 수정 필요성의 근거 | 구조적 근거 |
| SGLang + KTransformers | [LMSYS blog](https://lmsys.org/blog/2025-10-22-KTransformers/), [SGLang issue #11425](https://github.com/sgl-project/sglang/issues/11425) | AMX optimized CPU kernels, dynamic AMX/AVX-512 switching, CPU/GPU coordination overhead 감소 | ISA dispatch, coordination/routing 재평가의 근거 | 구조적 근거 |
| SparAMX | [Hugging Face paper page](https://huggingface.co/papers/2502.12444), [arXiv 2502.12444](https://arxiv.org/abs/2502.12444) | AMX + unstructured sparsity로 linear layer와 attention CPU 경로 개선 | AMX/sparsity/attention kernel 후보의 근거 | 후보 |
| NEO | [OpenReview](https://openreview.net/forum?id=umgy9tWBLA), [GitHub](https://github.com/NEO-MLSys25/NEO) | CPU offload, asymmetric GPU-CPU pipelining, load-aware scheduling; H100 throughput 개선 보고 | CPU/GPU 협업과 pipeline/load-aware scheduling이 유효할 수 있다는 근거 | 병행/참조 |
| DuoDecoding | [Hugging Face paper page](https://huggingface.co/papers/2503.00784), [arXiv 2503.00784](https://arxiv.org/abs/2503.00784), [GitHub](https://github.com/KaiLv69/DuoDecoding) | CPU draft model + GPU target model의 heterogeneous speculative decoding | Spec decode CPU drafter 병행 트랙의 근거 | 병행 트랙 |

## 13. 근거별 연결되는 수정 항목

| 수정 항목 | 직접 근거 | 보조 근거 | 주의 |
|---|---|---|---|
| Batch scaling profiler | H100x8 local analysis | KTransformers 문제 정의 | 계측은 gain이 아니라 방향 확정 |
| VNNI/pre-pack hot path wiring | KTransformers, SGLang+KTransformers | 현재 `gemm_vnni.*` 부분구현 | 기존 토대 자체는 gain 제외 |
| Head Folding | H100x8 CPU batch failure | T-MAC dataflow 관점 | 직접 논문 근거보다는 프로젝트 병목 기반 후보 |
| batch-aware decode attention | H100x8 tail, SparAMX attention speedup | NEO attention offload | CPU attention hot path 계측 후 착수 |
| QKV / Gate-Up fusion | KTransformers layout/fusion 관점 | T-MAN fused dataflow | 이미 fused된 모델 구조인지 먼저 확인 |
| LUT low-bit path | T-MAC | T-MAN | 구현 난도 높음, 강한 후보지만 즉시 gain 아님 |
| AVX/AMX dispatch | KTransformers, SGLang+KTransformers | Intel feature availability in H100x8 logs | `ONEDNN_MAX_CPU_ISA` 설정과 shape별 dispatch는 다름 |
| AVX/AMX cascade pipeline | T-MAN pipeline idea | Codex AVX/AMX analysis | x86 직접 근거는 약하므로 prototype 검증 필수 |
| routing 재평가 | H100x8 wave failure | NEO load-aware scheduling | kernel scaling 이후에만 의미 있음 |
| spec decode CPU drafter | DuoDecoding | TODO A1 | request-level mainline과 분리 |

## 14. 근거 사용 원칙

- 외부 논문의 speedup 수치를 현재 vLLM hybrid에 직접 곱하지 않는다
- 이미 구현된 기능을 다시 개선 항목으로 세지 않는다
- `구조적 근거`와 `직접 구현 근거`를 구분한다
- H100x8 short burst 문제의 직접 근거는 로컬 로그/JSON이다
- T-MAC/T-MAN/KTransformers/SparAMX는 “어떤 방향이 가능성이 있는지”를 보여주는 근거다
- NEO/DuoDecoding은 mainline이 아니라 병행 트랙의 근거다
