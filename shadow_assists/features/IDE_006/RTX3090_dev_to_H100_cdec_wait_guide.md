# RTX3090 개발 가이드 — TSK_019 `cdec_wait` 분석과 H100 이관 기준

> 목적: RTX 3090 개발 머신에서 `TSK_019`의 현재 병목을 빠르게 읽고 수정한 뒤,
> H100 서버로 옮겨 최종 판정할 수 있도록 현재 판단을 한 문서에 정리합니다.
>
> 범위: `IDE_006` / `TSK_019`의 `v1.5` 계열에서 관찰된 `cdec_wait` 중심 구조,
> 정적 분석 결과, 3090에서 유효한 수정 범위, H100에서만 확정 가능한 항목.

---

## 1. 한 줄 결론

`cdec_wait`의 **존재 이유는 하드웨어보다 아키텍처/알고리즘 문제**입니다.

- 왜 기다리는가: 구조 문제입니다.
- 얼마나 오래 기다리는가: 구조 + 하드웨어 둘 다 영향을 줍니다.
- 따라서 **3090에서 구조를 고치고**, **H100에서 최종 성능을 판정**하는 전략이 맞습니다.

---

## 2. 지금 질문에 대한 직접 답

질문:

> "cdec_wait은 하드웨어 특성이 아니라 알고리즘 문제 아닌가?"

답:

**대체로 맞습니다. 다만 절대 시간과 일부 세부 병목은 하드웨어 영향도 큽니다.**

정확히 나누면:

### 2.1 `cdec_wait`가 생기는 이유

이건 거의 **구조 문제**입니다.

현재 코드에서는 CPU cdec 경로가 layer 내부 critical path에 들어와 있고,
GPU path가 끝난 뒤 layer-end에서 CPU 결과를 기다립니다.

핵심 위치:

- `vllm/model_executor/layers/attention/attention.py`
  - cdec submit
  - GPU attention 실행
  - `cdec_future.result()` 대기

즉,

1. CPU 쪽 attention을 백그라운드로 던집니다.
2. GPU forward를 먼저 실행합니다.
3. layer를 끝내기 전에 CPU 결과를 반드시 기다립니다.

이 설계에서는 CPU가 느리면 `cdec_wait`가 생길 수밖에 없습니다.
이건 3090이든 H100이든 동일합니다.

### 2.2 `cdec_wait`가 얼마나 큰가

이건 **하드웨어 영향을 받습니다**.

다음 요소들이 바뀝니다.

- CPU 코어 수
- CPU ISA (AVX-512 / AMX)
- 메모리 대역폭
- NUMA 구조
- GPU 개수
- TP=1 vs TP=8
- NCCL collective 비용
- pinned memory / PCIe / NVLink / host topology

즉 **원인은 구조**, **크기는 구조 + 하드웨어**입니다.

---

## 3. 왜 RTX 3090에서 개발해도 되는가

3090 머신은 `TSK_019`의 현재 문제 중 **가장 중요한 절반 이상**을 충분히 드러냅니다.

3090에서 바로 검증 가능한 것은 다음입니다.

### 3.1 구조 문제

- layer마다 host fence가 생기는가
- cdec metadata를 hot path에서 다시 만드는가
- block table을 매 layer 재조립하는가
- Q/K/V transfer를 과도하게 반복하는가
- `decide_mode`가 불필요한 배치 조합을 두 번 하는가
- `NeoCpuKvBuffer`를 Python 자료구조 hot path로 쓰고 있는가

이건 하드웨어와 거의 무관한 성질입니다.

### 3.2 Python / staging / metadata 오버헤드

3090에서도 바로 드러납니다.

예:

- `.tolist()`
- `torch.tensor(block_table_rows)`
- `get_block_ids()` 반복 호출
- `ensure_capacity()` 반복 호출
- `re.search(layer_name)` 같은 반복 처리
- `seq_lens.cpu()` fallback
- pinned q/k/v alloc / copy / cast / reshape 반복

이건 3090에서 고쳐도 H100에서 그대로 이득이 납니다.

### 3.3 잘못된 발화 구조

예:

- cdec가 모든 layer에서 무조건 발화하는가
- 실제로는 sparse fire인데 NEO 인프라 비용만 큰가
- overlap이 있다고 생각했지만 실제로는 stream 재합류 때문에 거의 직렬인가

이것도 3090에서 충분히 판단 가능합니다.

---

## 4. 왜 H100에서 다시 봐야 하는가

3090에서 구조를 고친 뒤에도, H100에서만 확정 가능한 영역이 분명히 있습니다.

### 4.1 TP / NCCL 영역

H100 서버에서는 TP=8 같은 multi-GPU 설정이 붙습니다.
이 경우 CPU cdec만 느린 것이 아니라 다음도 같이 비용이 됩니다.

- NCCL all-reduce
- cross-device reduce
- CUDA event sync
- multi-worker coordination

3090 단일 GPU에서는 이 비용이 작거나 아예 없습니다.

따라서 3090에서 좋아진 구조가 H100에서는 NCCL에 묻힐 수도 있습니다.

### 4.2 CPU ISA 차이

3090 개발 머신의 CPU는 보통 다음 제약이 있습니다.

- AMX 없음
- AVX-512도 불완전하거나 fuse-off 가능
- NUMA 구조 단순

반면 H100 서버의 Xeon SPR 계열은:

- AVX-512 native
- AMX native
- NUMA 강함
- host memory bandwidth 훨씬 큼

즉 pacpu kernel의 **절대 성능 수치**는 3090 머신에서 그대로 예측할 수 없습니다.

### 4.3 NUMA / pinned memory / swap path

CPU resident KV, pinned host buffer, swap_in/out 경로는
단일 소켓/단순 토폴로지보다 다중 소켓 H100 서버에서 훨씬 민감합니다.

3090에서 깨끗해 보여도 H100에서 병목이 터질 수 있습니다.

---

## 5. 현재 코드 기준으로 본 `cdec_wait`의 구조적 root

아래는 현재 `v1.5` 계열 코드와 문서 기준으로, 정적 분석만으로도 거의 확실한 오버헤드 축입니다.

### 5.1 Root A — layer-end host fence

핵심 위치:

- `vllm/model_executor/layers/attention/attention.py`

현재 구조:

1. cdec용 CPU 작업 submit
2. GPU attention forward
3. layer 끝에서 `cdec_future.result()`

의미:

- step time 하한이 CPU cdec time에 잠깁니다.
- GPU가 아무리 빨라도 CPU가 늦으면 결국 기다립니다.
- 이것은 구조 문제입니다.

중요:

이 구조가 유지되는 한, pacpu를 조금 빠르게 만드는 것만으로는 한계가 큽니다.

### 5.2 Root B — cdec setup을 layer마다 다시 수행

핵심 위치:

- `vllm/model_executor/layers/attention/attention.py`

매 layer 반복되는 것들:

- cdec slice 확인
- `seq_lens` fallback
- `ensure_capacity`
- `get_block_ids`
- `max_blocks_per_seq`
- Python `block_table_rows` 조립
- `torch.tensor(...)` materialize
- `seq_lengths.tolist()`

이건 “CPU attention 계산”이 아니라 “CPU attention을 호출하기 위한 Python 준비”입니다.

따라서 이 부분은 **step/sub-batch 단위 precompute**로 빼는 것이 정답입니다.

### 5.3 Root C — Q/K/V transfer 경로가 너무 비쌈

핵심 위치:

- `vllm/model_executor/layers/attention/attention.py`

현재 비용:

- `q.to(torch.float16)`
- `k.to(torch.float16)`
- `v.to(torch.float16)`
- pinned CPU copy
- CPU kernel output 생성
- `out_buf.to(output.device).to(output.dtype)`

즉 layer당:

- cast 3회
- D2H copy 3회
- H2D copy 1회
- output dtype 재변환

이건 구조적으로 복사 횟수가 많습니다.

### 5.4 Root D — `decide_mode`의 이중 materialization

핵심 위치:

- `vllm/v1/core/sched/mode_selector.py`

현재 구조는:

- `gpu_only_batch`
- `batches[0]`
- `batches[1]`

를 만들면서 prefill/gdec를 중복 적재합니다.

문제:

- `SubBatch.add_*`는 단순 append가 아니라 perfdata 갱신까지 포함합니다.
- sparse cdec 상황에서는 이 비용이 순수 overhead가 됩니다.

정리:

두-sub-batch가 실제로 필요할 때만 두 번째 구조를 만드는 lazy 전략이 더 맞습니다.

### 5.5 Root E — queue / block accounting의 선형 재계산

핵심 위치:

- `vllm/v1/core/sched/neo_scheduler.py`

현재:

- `sum(self._get_block_needed(...))`
- `sum(...)` again

를 step마다 반복합니다.

req 수가 많아지면 스케줄러 비용도 선형으로 커집니다.

이건 swap_in/out 시점에 증감되는 running counter로 줄일 수 있습니다.

### 5.6 Root F — `NeoCpuKvBuffer`의 Python hot path

핵심 위치:

- `vllm/v1/core/sched/neo_cpu_kv_buffer.py`

문제:

- Python list 기반 `block_ids`
- 요청별 반복 조회
- 락 잡고 `ensure_capacity`
- 락 잡고 `get_block_ids`
- hot path에서 block table 재구성

이건 alloc/free 시점으로 옮겨야 합니다.
attention hot path에 두면 계속 손해입니다.

### 5.7 Root G — `forward_double` overlap window가 짧음

핵심 위치:

- `vllm/v1/worker/sub_batch_executor.py`

구조:

- stage 0에서 두 stream으로 launch
- 곧바로 `wait_stream`으로 재합류

의미:

- 겉으로는 비동기처럼 보이지만 실제 overlap window는 짧습니다.
- 다시 layer 단위 fence 구조로 돌아옵니다.

즉 “NEO가 발화한다”와 “실제 throughput이 오른다”가 다를 수 있습니다.

---

## 6. 3090에서 먼저 해결해야 할 것

3090에서 먼저 손대야 하는 것은 “절대 성능 숫자”가 아니라 “구조적으로 낭비인 부분”입니다.

우선순위를 나누면 다음과 같습니다.

### 6.1 최우선 — 구조를 바꾸는 작업

#### A. cdec metadata를 layer hot path에서 빼기

목표:

- step 또는 sub-batch 시작 시 1회 계산
- layer에서는 lookup만 하도록 변경

옮겨야 할 것:

- `seq_lengths`
- `req_ids`
- `block_table_cpu`
- `max_blocks_per_seq`
- cdec row mapping

효과:

- Python list / tensor materialization 감소
- layer 수가 많을수록 이득 큼

#### B. transfer를 packed 경로로 줄이기

목표:

- q/k/v를 3개 tensor로 따로 다루지 않기
- 가능하면 하나의 packed staging buffer 사용

효과:

- cast / copy 횟수 축소
- launch 수도 감소

#### C. `cdec_future.result()` 빈도를 줄이는 방향 검토

중요:

이게 가장 큰 본질입니다.

가능한 방향:

- 모든 layer에서 cdec를 기다리지 않기
- 일부 layer만 CPU로 보내기
- 여러 layer 또는 더 큰 단위에서 fence하기
- cdec fire 대상을 더 공격적으로 줄이기

이 부분은 구현 난도가 높지만, 구조 효과는 제일 큽니다.

### 6.2 중간 우선순위 — scheduler / queue 비용 줄이기

#### D. `decide_mode` lazy materialization

현재처럼:

- gpu_only
- batch0
- batch1

를 다 만들어 비교하지 말고,
필요할 때만 분기 materialize하는 쪽이 유리합니다.

#### E. queue/block usage counter화

현재 `sum(...)` 반복 대신:

- `gpu_block_needed`
- `cpu_block_needed`

를 상태로 유지하고 swap/prefill 시점에 증감시킵니다.

### 6.3 낮지만 확실한 우선순위 — hot path 잡음 제거

제거 대상:

- `re.search(layer_name)`
- 반복 env lookup
- debug counter 분기
- `.tolist()` 남용
- 반복 `torch.tensor(...)`

개별 이득은 작아도 layer × step × worker로 누적되면 무시하기 어렵습니다.

---

## 7. 3090에서 해도 되는 일 vs H100에서만 확정할 일

### 7.1 3090에서 해도 되는 일

다음은 3090에서 수정하고 검증해도 충분히 의미 있습니다.

#### 구조 / 코드 정리

- hot path metadata precompute
- transfer packing
- Python materialization 제거
- block table caching
- lazy `SubBatch` 생성
- scheduler counter화
- unnecessary sync 제거

#### correctness / 발화 검증

- cdec path가 정말 발화하는지
- silent skip 없는지
- shape mismatch 없는지
- CPU/GPU 경로가 deadlock 없이 도는지
- layered overlap 구조가 의도대로 동작하는지

#### microbench 성격 검증

- 어떤 코드가 layer당 반복되는지
- 어떤 분기가 예상보다 자주 발화하는지
- cdec path 준비 비용이 실제 compute보다 큰지

### 7.2 H100에서만 최종 확정할 일

다음은 H100에서 다시 반드시 봐야 합니다.

#### 절대 throughput

- `vanilla` 대비 실제 순이득인지
- TP=8 환경에서 여전히 이득이 남는지

#### NCCL / TP 상호작용

- CPU cdec를 줄였더니 NCCL이 새 병목이 되는지
- `forward_double` overlap이 TP collective에 묻히는지

#### CPU ISA 이득

- AVX-512 / AMX path가 실제로 얼마나 이득인지
- 3090 개발 머신에서 안 보이던 kernel 계층 병목이 있는지

#### NUMA / swap path

- CPU resident KV
- pinned host buffer
- swap_in/out
- multi-socket locality

이 부분은 H100 서버에서 최종 확인해야 합니다.

---

## 8. 실전 판단 기준

3090에서 개발할 때는 다음 질문으로 판단하시면 됩니다.

### 8.1 "이 문제는 3090에서도 유효한가?"

다음이면 **예**입니다.

- Python hot path 문제
- metadata 재조립 문제
- layer-end wait 구조 문제
- transfer 횟수 과다 문제
- scheduler 중복 계산 문제

### 8.2 "이 문제는 H100에서만 판정 가능한가?"

다음이면 **예**입니다.

- 절대 throughput 승패
- TP=8 collective 영향
- AVX-512 vs AMX 실제 차이
- NUMA/pinned host 최적화의 최종 효과

### 8.3 "지금 바꾸면 리스크 없이 좋은가?"

다음이면 거의 항상 바꿔도 됩니다.

- 같은 정보를 매 layer마다 다시 만드는 코드
- 같은 batch를 여러 구조로 중복 materialize하는 코드
- 반복 string parsing / list build / tensor build
- debug용 hot path 분기

---

## 9. 현재 기준 권장 작업 순서

3090에서 실제로 작업한다면 순서를 이렇게 가져가는 것이 좋습니다.

### Phase 1 — hot path 정리

1. cdec metadata를 step/sub-batch precompute로 이동
2. `seq_lens` / `block_table` / `req_ids` 반복 생성 제거
3. `layer_name` parsing 제거
4. debug/env lookup 정리

### Phase 2 — transfer 경로 축소

1. q/k/v staging packing
2. output buffer 재사용
3. dtype cast 위치 최소화

### Phase 3 — scheduler 비용 축소

1. `decide_mode` lazy화
2. queue/block counter화
3. `NeoCpuKvBuffer` hot path 캐시화

### Phase 4 — 구조 재검토

1. 모든 layer cdec wait 구조가 맞는지 재판단
2. selective cdec 또는 coarser fence 가능성 검토
3. 실제로 CPU가 critical path에 들어갈 가치가 있는 req/layer만 태우는 방향 검토

### Phase 5 — H100 이관

1. TP=8에서 측정
2. NCCL / cdec / swap 비율 재분석
3. AVX-512 / AMX 실측
4. NUMA / pinned / swap path 최종 조정

---

## 10. 코드상 바로 봐야 하는 파일

### 가장 먼저

- `vllm/model_executor/layers/attention/attention.py`
  - cdec setup / transfer / future wait / result apply

- `vllm/v1/core/sched/mode_selector.py`
  - two-sub-batch 조합 비용

- `vllm/v1/core/sched/neo_scheduler.py`
  - queue / block accounting

- `vllm/v1/core/sched/neo_cpu_kv_buffer.py`
  - block allocation / lookup / hot path 자료구조

- `vllm/v1/worker/sub_batch_executor.py`
  - 실제 overlap이 얼마나 되는지

### 같이 참고

- `vllm/model_executor/models/llama.py`
  - `forward_neo_pipelined`, `neo_preproj`, `neo_postproj`

- `csrc/cpu/pacpu/pacpu.ispc`
  - CPU kernel 자체는 현재 “구조 문제 정리 후” 다시 보는 것이 좋음

---

## 11. 최종 요약

이 문서의 핵심은 다음입니다.

1. `cdec_wait`의 **원인은 구조 문제**입니다.
2. `cdec_wait`의 **절대 크기와 최종 승패는 하드웨어 영향**을 받습니다.
3. 따라서 **3090은 구조 개선용 머신**, **H100은 최종 성능 판정 머신**으로 쓰는 게 맞습니다.
4. 지금 당장 3090에서 가장 가치 있는 작업은:
   - hot path metadata 제거
   - transfer 경로 축소
   - scheduler 중복 계산 제거
   - layer-end wait 구조 재검토
5. 반대로 AVX-512/AMX의 최종 숫자, TP=8/NCCL과의 상호작용, NUMA 최적화의 승패는 H100에서만 확정할 수 있습니다.

즉 지금 개발 전략은 다음 한 줄로 정리됩니다.

> **3090에서 구조를 정리하고, H100에서 숫자를 판정합니다.**

