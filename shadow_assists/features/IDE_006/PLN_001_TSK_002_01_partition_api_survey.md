**↑ 부모 TSK**: [`TSK_002`](TSK_002.md) · **↑ 부모 PLN**: [`PLN_001`](PLN_001.md) · **↟ 조부 IDE**: [`IDE_006`](README.md) · **선행 TSK**: [`TSK_001`](TSK_001.md) (kernel 진입점)

---

# PLN_001_TSK_002_01 — OffloadingConnector partition API survey

| 항목 | 값 |
|---|---|
| 산출 단계 | [`TSK_002`](TSK_002.md) §4.1 (선행 게이트) |
| 종류 | read-only 코드 조사 + 설계 결정 (코드 변경 없음) |
| 작성일 | 2026-04-26 |
| 결정 사항 | 6 개 (§3 Decisions) — Phase 1 (§4.2 metadata schema) 부터 본 doc 결정에 따름 |
| Scope lock | BF16/FP16, non-FP8, non-MLA, full attention, 단일 KV group, decode-first (PLN_001 §3) |

> **목적**: TSK_002 §3.2 의 "단정 금지" 박스가 요구한 선행 조사. 현재 vLLM v1 abstraction 으로 임의의 hot/cold 분할이 노출 가능한지, 아니면 prefix-suffix 모델로 제한되는지 결정. 본 doc 의 결정은 §4.2 metadata schema → §4.7 e2e smoke 까지 모든 후속 단계의 design 입력.

---

## 1. TL;DR

1. **partition shape = (a) prefix-suffix** — `OffloadingManager.lookup()` (`vllm/v1/kv_offload/abstract.py:94`) 가 "first N contiguous offloaded blocks" 만 반환. abstract.py 를 재작성하지 않고 사용 가능한 유일한 형태. 이미 scheduler 가 동일 패턴으로 lookup 하고 있음 (`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:187-190`).
2. **flash_attn return_lse 는 이미 노출** — `flash_attn_varlen_func(..., return_softmax_lse=True)` 가 `vllm/v1/attention/backends/flash_attn.py:1163, 1191, 1204` 에서 cascade attention 에 사용 중. `merge_attn_states(output, prefix_O, prefix_LSE, suffix_O, suffix_LSE)` 가 표준 merge.
3. **dispatcher hook = cascade attention 경로 확장** — `flash_attn.py:1123-1214` 의 `cascade_attention` 가 정확히 우리가 필요한 prefix(GPU) + suffix(GPU) + LSE merge 의 구조. cold(CPU) + hot(GPU) + LSE merge 로 일반화 / 별도 함수 추가.
4. **단일 KV group 가정 유지** — `vllm/v1/kv_offload/worker/cpu_gpu.py:138-139` assert 와 정합. multi-group 은 후속.
5. **TSK_002 §3.1 의 metadata schema 단순화 가능** — 임의 set 이 아닌 prefix-suffix 면 `cold_block_ids[num_seqs, max_cold_blocks]` 텐서가 불필요. 시퀀스별 정수 `num_cold_blocks[num_seqs]` 만 있으면 충분.

---

## 2. Findings

### 2.1 `OffloadingManager.lookup()` — prefix-only contiguous count

`vllm/v1/kv_offload/abstract.py:94-113` (실제 docstring 직접 인용):

```
Finds the length of the maximal series of blocks, starting from the
first one, that are all offloaded.
...
Returns:
    An integer representing the maximal number of blocks that
    are currently offloaded, or None if the lookup should be retried
    later. Returning None will delay the request handling by the vLLM
    scheduler.
```

→ **임의의 hot/cold 분할 set 을 반환하는 method 가 abstract 에 없다**. `LoadStoreSpec` / `prepare_load` / `prepare_store` 도 "어느 block 을 transfer 할지" 의 spec 으로, scheduler 측에서 partition 정보를 얻는 용도가 아님.

### 2.2 Scheduler 가 이미 동일 패턴으로 lookup

`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:187-190`:

```python
hits = self.manager.lookup(
    offload_keys[start_block_idx:],
    req_status.req_context,
)
```

→ scheduler 측에서 각 request 의 prefix 부분이 몇 block 까지 offloaded 인지 이미 계산 중. **이 hit count 가 그대로 `num_cold_blocks` 의 source**. `OffloadingConnectorScheduler.get_num_new_matched_tokens()` 에서 호출.

### 2.3 cascade attention 의 LSE merge 패턴

`vllm/v1/attention/backends/flash_attn.py:1163-1214` 의 `cascade_attention()` (직접 read 한 코드 일부):

```python
prefix_output, prefix_lse = flash_attn_varlen_func(
    q=query, k=key_cache, v=value_cache,
    ...
    block_table=block_table[:1],            # shared prefix (1 행)
    return_softmax_lse=True,
    ...
)

suffix_output, suffix_lse = flash_attn_varlen_func(
    q=query, k=key_cache, v=value_cache,
    ...
    block_table=block_table[:, num_common_kv_blocks:],   # suffix slice
    return_softmax_lse=True,
    ...
)

merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
```

**TSK_002 가 다룰 변형**: prefix path 를 GPU → **CPU partial-attention kernel** (`forward_partial_with_lse`, TSK_001) 로 교체. block_table slicing 은 `block_table[:, :num_cold_blocks]` (cold) / `block_table[:, num_cold_blocks:]` (hot). 모든 시퀀스가 동일 cold count 인 cascade 의 shared-prefix 와 달리, 본 케이스는 **시퀀스별 cold count 가 다름**.

### 2.4 CPU kernel 시그니처 — 이미 LSE 반환

`vllm/v1/attention/ops/cpu_partial_attention.py:333-407` `forward_partial_with_lse(...)` 가:

- 입력: Q (BF16/FP16), cold K/V (canonical int8 → KVViewAdapter), `cu_seqlens_q`, `query_positions`, `seq_lens_total`, `causal`
- 출력: `(O_cold, LSE_cold)` — `LSE_cold.shape = [num_q_heads, num_tokens]`, `dtype = float32`

→ `merge_attn_states` 의 입력 시그니처와 정확 일치. **wrapper 호출만 하면 됨**.

### 2.5 KVViewAdapter — zero-copy 보장

`vllm/v1/attention/ops/kv_view_adapter.py:60-180` 가 canonical int8 page → typed BF16/FP16 view 를 `torch.as_strided` 로 zero-copy 노출. CPU partial-attention kernel 은 cold KV 를 추가 copy 없이 read.

### 2.6 model_runner 의 stream

`vllm/v1/worker/model_runner.py:135` 에 `self.output_copy_stream = torch.cuda.Stream(self.device)` 가 이미 정의 — async copy 패턴이 vLLM v1 에 정착돼 있음. CPU partial 의 H2D `(O_cold_gpu, LSE_cold_gpu)` 도 이 패턴 재사용 가능.

### 2.7 사용 예시 — `tests/v1/kv_offload/test_cpu_offloading.py:197`

connector 설정 패턴:
```python
KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "cpu_bytes_to_use": <int>,
        "block_size": 64,
    },
)
```

`eval/envs/ide006_cold_kv*.env` 에 사용 중인 형태와 정합 → **본 TSK 는 별도 connector config 변경 없이** `enable_hot_cold_split=True` flag 만 추가하면 활성.

---

## 3. Decisions

본 doc 의 핵심 산출. 후속 phase 의 design 잠금.

### Decision 1 — partition shape: **(a) prefix-suffix**

| 후보 | 채택? | 사유 |
|---|---|---|
| (a) prefix-suffix (cold = first N blocks per req) | ✓ | abstract.py 무수정 가능. `manager.lookup()` 의 출력이 정확히 이 형태. scheduler 가 이미 hit count 산출 중 (`scheduler.py:187`). cascade attention 의 block_table slicing 패턴 (`flash_attn.py:1174, 1202`) 과 동형. |
| (b) 신규 connector hook (임의 set) | × | abstract.py 에 새 abstractmethod 추가 필요 → 모든 connector 구현체 (LMCache 등) 깨짐. scope lock 위반. |
| (c) scheduler-side 별도 추적 자료구조 | × | OffloadingManager 와 별도로 partition 을 추적하면 evict 발생 시 동기화 비용. lookup 결과를 신뢰하면 충분. |

**의미**: TSK_002 §3.1 의 metadata 필드 4 개 중:
- ✓ `enable_hot_cold_split: bool` — 그대로
- ✓ `num_cold_blocks: Tensor[num_seqs]` — `cold_block_lens` 와 통합 (시퀀스별 정수 1 개)
- ✗ `cold_block_ids: Tensor[num_seqs, max_cold_blocks]` — **불필요** (block_table[:, :num_cold_blocks] slice 면 됨)
- ✗ `hot_block_table: Tensor[num_seqs, max_hot_blocks]` — **불필요** (block_table[:, num_cold_blocks:] slice)

→ schema 단순화. Phase 1 (§4.2) 작업량 감소.

### Decision 2 — flash_attn return_lse: **기존 옵션 재사용**

`flash_attn_varlen_func(..., return_softmax_lse=True)` 가 이미 cascade 경로에서 활성. 별도 wrapper / 옵션 추가 불필요. hot subset 호출 시 동일 kw 전달.

### Decision 3 — dispatcher hook 위치: **cascade_attention 경로 확장**

| 후보 | 채택? | 사유 |
|---|---|---|
| `flash_attn.py:1123-1214` 의 `cascade_attention()` 변형 / 새 함수 추가 | ✓ | 이미 prefix(GPU) + suffix(GPU) + LSE merge 의 정확한 구조. cold(CPU) + hot(GPU) + LSE merge 로 일반화 자연스러움. |
| `model_runner.py` 의 attention forward 분기 | △ | 가능하지만 attention backend 의 LSE 처리 로직이 backend 측에 모이는 게 vLLM 스타일 |
| `flash_attn.py:751-841` 의 `use_cascade` if-else 직전 | × | 그 위치는 cascade 와 무관한 standard path 기준 — 주변 코드 가독성 떨어짐 |

**구체 위치**: `cascade_attention()` 옆에 새 함수 `hot_cold_attention()` 추가 (또는 `cascade_attention` 의 prefix path 를 callable 로 일반화). `attn_metadata.enable_hot_cold_split` 분기는 그 위 (model_runner forward) 에서.

### Decision 4 — async stream owner: **`output_copy_stream` 패턴 재사용**

기존 `self.output_copy_stream` (model_runner.py:135) 의 패턴을 따라 추가 stream 1 개 (`self.cpu_partial_stream`) 신설. CPU partial kernel 은 main stream 에서 호출 (CPU 작업이라 stream 분리 의미 없음), Q 의 D2H + (O_cold, LSE_cold) 의 H2D copy 만 별도 stream 으로 main 의 GPU hot 과 overlap.

상세 동기화 로직은 PLN_001 §4.3 의 overlap profile (TST_002 단계 B) 결과로 결정 — 본 doc 은 stream 갯수 / owner 만 lock.

### Decision 5 — scheduler populate point: **scheduler.py:945-959 사이**

`vllm/v1/core/sched/scheduler.py` 의 흐름 (agent 조사 결과):
- `:945` `self.block_tables.apply_staged_writes()` — block 할당 finalize
- `:959` `connector.build_connector_meta()` — connector metadata build

이 사이에 `OffloadingConnectorScheduler.get_num_new_matched_tokens()` 의 hit count 를 attention metadata 의 `num_cold_blocks` 필드로 populate. metadata 자체는 worker 측 `FlashAttentionMetadataBuilder.build()` 가 최종 packing.

### Decision 6 — single KV group 유지

`cpu_gpu.py:138-139` 의 assert 그대로 유지. multi-group 은 후속 TSK 로 미룸 (PLN_001 §3 scope lock 과 정합).

---

## 4. Risks / Surprises

### R1. cascade attention heuristic gate 와의 분리

`flash_attn.py:1045-1120` 의 `use_cascade_attention()` 가 SM count + tile size + block layout 으로 cascade 활성을 동적 결정. 본 TSK 의 hot/cold split 은 **cascade heuristic 과 독립적으로 활성** 되어야 함 (`enable_hot_cold_split` flag 가 별도 게이트). 두 경로가 동시 활성될 때의 우선순위: **cold/hot 우선** (cascade 는 모든 시퀀스가 공유 prefix 가질 때만 의미. cold/hot 은 각 시퀀스 독립).

### R2. block_table padding (CUDA graph mode)

`model_runner.py:746-748` 가 `query` / `seq_len` 버퍼를 `self.max_num_reqs + 1` 까지 padding (FULL cudagraph). cold/hot split 이 이 padding 을 보존하는지 확인 필요. **runtime check 필요** (Open Q1).

### R3. cascade_attention 의 batch 가정과 상이

cascade 는 *all sequences share the same prefix* 가정으로 `block_table[:1]` (1 행) 만 prefix 에 사용. 본 TSK 는 **시퀀스별 cold count 가 다름** → `block_table[:, :num_cold_blocks[i]]` 처럼 시퀀스별 slicing. 이 불일치를 어떻게 cu_seqlens 로 표현할지 — Phase 3 (§4.4) 에서 구현 시 주의.

### R4. cold KV 의 물리 위치

CPU partial-attention kernel 은 cold KV 를 CPU DRAM 에서 read 한다고 가정. 그러나 OffloadingConnector 의 lifecycle 상 "lookup hit = CPU 에 있다" 는 사실인지, 아니면 GPU↔CPU 전송이 이미 시작/완료된 상태인지 — runtime 측 확인 필요 (Open Q2).

### R5. block eviction 동시성

scheduler step → attention forward 사이에 OffloadingManager 가 block 을 evict 할 가능성. `num_cold_blocks` 가 forward 시점에도 valid 한지 — runtime check (Open Q3).

---

## 5. Open Questions (runtime check 필요)

1. **block_table padding 호환**: cold/hot split 이 CUDA graph 의 max_num_reqs+1 padding 을 깨뜨리지 않는지. → Phase 1 (§4.2) 의 metadata schema 결정 후 small repl 로 검증.
2. **cold KV 의 forward 시점 위치**: OffloadingConnector lookup hit ↔ 실제 CPU 메모리 보유. → `cpu_gpu.py:198-260` worker stream 동작 검토.
3. **block eviction**: scheduler step ↔ forward 간 race. → `OffloadingManager` 의 `prepare_load` 가 block 잠금하는지 코드 재확인.
4. **mixed prefill-decode batch 의 query position**: `cpu_partial_attention.py:342` 의 `query_positions` 가 두 종류 batch 에서 정확한지. → TSK_002 §4.7 e2e smoke 의 첫 검증 항목.
5. **prefill chunked attention**: TSK_002 §1 의 "decode 우선" 정책 유지. prefill 의 hot/cold 분할은 TSK_004+ 로 미룸.

---

## 6. References

### 6.1 vLLM 코드 (read-only 확인)

| File | Lines | 요점 |
|---|---|---|
| `vllm/v1/kv_offload/abstract.py` | 94-113 | `lookup()` docstring — prefix-only contiguous count |
| `vllm/v1/kv_offload/worker/cpu_gpu.py` | 138-139 | 단일 KV group assert |
| `vllm/v1/kv_offload/worker/cpu_gpu.py` | 198-260 | offloading worker stream / transfer |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | 115-268 | `OffloadingConnectorScheduler` |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | 187-190 | scheduler 측 lookup 호출 |
| `vllm/v1/attention/backends/flash_attn.py` | 751-841 | `use_cascade` 분기 (heuristic gate 직후) |
| `vllm/v1/attention/backends/flash_attn.py` | 1045-1120 | `use_cascade_attention()` heuristic |
| `vllm/v1/attention/backends/flash_attn.py` | 1123-1214 | `cascade_attention()` — 본 TSK 의 model 함수 |
| `vllm/v1/attention/backends/flash_attn.py` | 1163, 1191, 1204 | `flash_attn_varlen_func(..., return_softmax_lse=True)` |
| `vllm/v1/attention/backends/flash_attn.py` | 1214 | `merge_attn_states(output, prefix_O, prefix_LSE, suffix_O, suffix_LSE)` |
| `vllm/v1/attention/ops/merge_attn_states.py` | 9-47 | merge 시그니처 |
| `vllm/v1/attention/ops/cpu_partial_attention.py` | 333-407 | TSK_001 산출물 — `forward_partial_with_lse` |
| `vllm/v1/attention/ops/kv_view_adapter.py` | 60-180 | TSK_001 산출물 — zero-copy view |
| `vllm/v1/worker/model_runner.py` | 135 | `output_copy_stream` 정의 |
| `vllm/v1/worker/model_runner.py` | 746-748 | block_table padding (CUDA graph) |
| `vllm/v1/core/sched/scheduler.py` | 945-959 | metadata populate 영역 |
| `tests/v1/kv_offload/test_cpu_offloading.py` | 191-204 | KVTransferConfig 사용 예 |

### 6.2 IDE_006 doc

- 부모 TSK: [`TSK_002`](TSK_002.md) §3.2 (단정 금지 박스), §4.1 (선행 게이트), §8 (Open Q1·Q2·Q3·Q4)
- 부모 PLN: [`PLN_001`](PLN_001.md) §3 (scope lock), §4.3 (overlap profile)
- 선행 TSK: [`TSK_001`](TSK_001.md) §4.2c (portable C++), §4.3 (wrapper dispatch)
- 조부 IDE: [`IDE_006`](README.md) §6.2 (4 축)

---

## 7. Phase 1 (§4.2 metadata schema) 입력

본 doc 의 결정에 따라 Phase 1 의 schema 가 다음과 같이 lock:

```python
# vllm/v1/attention/backends/utils.py 또는 metadata 정의 모듈
class AttentionMetadata:
    # 기존 필드 ...
    enable_hot_cold_split: bool = False    # default: 기존 동작
    num_cold_blocks: torch.Tensor | None = None   # [num_seqs] int32, prefix 기준
```

`hot_block_table` / `cold_block_ids` 는 신설하지 않음 — `block_table` slicing 으로 충분 (Decision 1). 시퀀스별 가변 cold count 는 cu_seqlens 형태로 cascade variant 에 전달.

→ Phase 1 작업: schema 추가 + `enable_hot_cold_split=False` 의 기존 동작 무변경 검증.

---

## 8. Change Log

| 날짜 | 변경 | 사유 |
|---|---|---|
| 2026-04-26 | 본 doc 초안 작성 (TSK_002 §4.1 산출) | TSK_002 §3.2 "단정 금지" 박스가 요구한 선행 게이트. read-only 코드 조사 (8 개 파일) + 6 결정 (partition shape / return_lse / dispatcher / async stream / scheduler populate / KV group lock). 채택안: prefix-suffix 모델 (Decision 1). schema 단순화 효과: TSK_002 §3.1 의 4 필드 → 2 필드 (`enable_hot_cold_split` + `num_cold_blocks`). cascade_attention path (`flash_attn.py:1123-1214`) 가 사실상 동일 패턴이라 재사용 가능 (Decision 3). dispatcher / async stream / populate point 모두 lock. R1 ~ R5 risk 와 5 개 runtime open question 명시. Phase 1 (§4.2) 진입 가능. |

---

**↑ 부모 TSK**: [`TSK_002`](TSK_002.md) · **↑ 부모 PLN**: [`PLN_001`](PLN_001.md) · **↟ 조부 IDE**: [`IDE_006`](README.md) · **선행 TSK**: [`TSK_001`](TSK_001.md)
