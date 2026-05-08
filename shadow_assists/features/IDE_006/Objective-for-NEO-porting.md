# Objective for NEO Porting — vLLM 적재 검증 항목

| 항목 | 값 |
|---|---|
| 작성 | 2026-05-08 |
| 영역 | IDE_006 / TSK_019 (NEO performance max) |
| 기준 분기 | `feat/ide006-tsk019-neo-performance-max` (HEAD `93ea1fcd5f`, v1.2) |
| 비교 분기 | `feat/ide006-neo-asymmetric` (`27b1bf15fa`, v41) / commit `eeed0d46fc` (v38) |
| 결과 근거 | try44 (Phase 2-A v1.2) / try45 (Phase 2-B skip 제거) / try46 (Phase 3 v38, 진행 중) |
| 본 문서 | NEO 논문의 mechanism 별 *구현* 상태 + *동작* 상태 + *근거 위치* 단일 출처 |

---

## A. 개략 표 — at-a-glance

> Legend: ✅ 정상 / 🔶 부분 / ❌ 미구현·미발화 / ⚠️ 발화는 했으나 결과 부적절

| # | 항목 | 구현 | 동작 |
|---|---|---|---|
| 1 | request 단위 KV exclusive ownership | ✅ | ⚠️ |
| 2 | CPU 가 attention 직접 수행 | ✅ | ⚠️ |
| 3 | Asymmetric Pipelining (`forward_double`) | ✅ | ⚠️ |
| 4 | Stage 분할 (first / double / last) | ✅ | ⚠️ |
| 5 | 6 단계 Scheduler | 🔶 | ❌ |
| 6 | Mode Select (pipelined vs sequential) | 🔶 | ❌ |
| 7 | 3-way attention dispatch (prefill / GPU dec / CPU dec) | ✅ | ❌ |
| 8 | swap_out / swap_in 동시 발생 안 함 invariant | ✅ | ✅ |
| 9 | `paged_attention_cpu` (pacpu) kernel | ✅ | ⚠️ |
| 10 | Q/K/V D2H transfer + qkvtr_e sync | ✅ | ⚠️ |
| 11 | `NeoSchedulerOutput → SchedulerOutput` 변환 (sub_batches attach) | ✅ | ⚠️ |
| 12 | Worker side b0_eff / b1_eff 정렬 | ✅ | ⚠️ |
| 13 | `forward_pipeline` overlap (진짜 CPU↔GPU 병렬) | ✅ | ⚠️ |
| 14 | KV migration 정책 (LRU-like GPU↔CPU 양방향) | ❌ | ❌ |
| 15 | NEO 가 vanilla > 능가 (throughput win) | ⚠️ | ❌ |
| 16 | CPU utilization HIGH (CLAUDE.md objective) | ⚠️ | ❌ |
| 17 | token correctness (분포 동등) | 🔶 (v41) | ⚠️ |
| 18 | deadlock 회피 | ✅ | ✅ |
| 19 | silent worker crash 0 (CUDA device-side assert 없음) | ✅ | ✅ |

---

## B. 상세 표 — 각 항목의 상태 + 근거 + gap

| # | 항목 | 구현 위치 (file:line / commit) | 동작 근거 (try / probe / log) | gap (수정 plan input) |
|---|---|---|---|---|
| 1 | KV exclusive ownership | `vllm/v1/core/sched/neo_cpu_kv_buffer.py` (`NeoCpuKvBuffer`) | try44 P8: `new_swap_out=4,210` / `new_swap_in=0` 매 step | 한쪽 (CPU) 으로 단방향 swap 만 — bi-directional 부재 → #14 미구현 영향 |
| 2 | CPU attention 직접 수행 | `csrc/cpu/pacpu/pacpu.ispc` (ISPC kernel) + `vllm/v1/attention/ops/neo_pacpu.py` (Python wrapper) + `torch.ops.pacpu.paged_attention_cpu` | try44 P7=1600 (warmup cap, `_interesting=False` 100%) / try46 NEO FORK STAT `active=4217/4300` | v1.2 의 cdec dispatch path 가 try22 skip 으로 차단. v38 에선 정상 fire |
| 3 | Asymmetric Pipelining `forward_double` | `vllm/v1/worker/sub_batch_executor.py:231` (`forward_double`) + `vllm/model_executor/models/llama.py:533~607` (`forward_neo_pipelined`) | try45 P6=8 (b0_eff=35, b1_eff=195, boundary=35) → forward_double 진입 → CUDA OOM @ `preproj/input_layernorm` | forward_double 진입 시 GPU mem 잔여 ~92 MiB 만 — 추가 126 MiB alloc 실패. memory budget 관리 필요 |
| 4 | Stage 분할 | `llama.py:533~607` (`forward_neo_pipelined`) → `forward_pipeline:297` → `forward_double:231` | try45 traceback 상 `llama.py:607` `out0, out1 = executor.forward_pipeline(...)` 도달 | 정상 chain 도달, OOM 으로 측정 미완 |
| 5 | 6 단계 Scheduler | `vllm/v1/core/sched/neo_scheduler.py` + `vllm/v1/core/sched/neo_scheduler_adapter.py` (super().schedule() 후 NeoScheduler 동기) | try44 P8: 매 step `new_swap_in=0` — Step 3 (swap-in) 미발화. Step 6 (mode select) 도 항상 sequential | swap-in path 의 enable / Step 3 의 candidate evaluation 활성화 필요 |
| 6 | Mode Select pipelined vs sequential | `neo_scheduler_adapter.py:128` (`ZeroPerfPredictor` fallback + `TablePerfPredictor` lazy init) | TablePerfPredictor 의 *table 미작성* → 항상 ZeroPerfPredictor → mode 선택 logic 발화 안 함 (NEO_redesign §1.4 참조) | ModelProfiler 로 table 채우거나 heuristic 기반 mode 선택 추가 |
| 7 | 3-way attention dispatch | `vllm/model_executor/layers/attention/attention.py:773~893` (`cdec_future` submit + `_get_neo_cdec_executor`) | try44 P7=1600 / `_interesting=False` 100% → `forward_context.neo_cdec_token_slice / seq_slice / req_ids` 모두 None | forward_context 의 neo_cdec_* 가 set 안 됨 → 호출 chain 의 worker side stash 누락 (chain #11 → #12 영역) |
| 8 | swap_out/swap_in 비동시 invariant | `vllm/v1/engine/core.py:_handle_neo_swaps:496` | try44 P8 모든 step 에서 둘 중 하나만 attached (관찰: swap_out 만, swap_in 0) | 정상 |
| 9 | pacpu kernel | `csrc/cpu/pacpu/CMakeLists.txt` + `pacpu.ispc` + `core.h` + `_C.so` 빌드 | try46 build OK + import OK / try46 NEO FORK STAT 4,217 dispatch | v1.2 build 도 OK. firing 빈도 차이는 chain 통로 (try22 skip) 때문 |
| 10 | Q/K/V D2H transfer | `sub_batch_executor.py` 의 transfer stream + qkvtr_e event sync | try45 OOM 으로 D2H 실제 전송 도달 못 함 / try46 v38 에선 정상 fire (FORK STAT) | forward_double OOM 회피 후 측정 |
| 11 | sub_batches attach | `neo_scheduler_adapter.py:600~666` (`output.neo_sub_batches = [b0_ids, b1_ids]`) | try44 P5=535 (cdec_ids 항상 0 매 step) / try45 P5=4 step (cdec_ids=195 정상 attach) | v1.2: try22 skip 가 SWAPPED_OUT 을 outer loop 에서 제거 → vllm_ids 에 안 포함 → cdec_ids=[]. v38: 정상 attach |
| 12 | Worker b0_eff/b1_eff | `gpu_model_runner.py:1064~1135` (`_neo_b0_eff_for_step` + `_neo_b1_eff_for_step` + swap_states) | try45 P6=8 (8 worker × 1 step) — boundary=35, b0_eff=35 b1_eff=195 정상 split | 정상 분기 후 forward_double OOM. b0/b1 분기 자체는 OK |
| 13 | forward_pipeline overlap | `sub_batch_executor.py:297` (`forward_pipeline`) — CUDA stream 분리 + executor.submit | try44: 미발화 / try45: OOM 진입 / try46 v38: 4,217 회 active dispatch (진짜 병렬 추정) | 측정 보류 — try46 완료 후 wall vs CPU pacpu vs GPU forward 시계열 비교 |
| 14 | KV migration 정책 (양방향) | NEO 표준 = LRU 기반 / **vLLM port 미구현** | swap_out 만 단방향 fire (try44 4,210), swap_in 0 → 결국 vanilla preempt path 로 흘러감 | LRU policy 도입 + swap_in candidate 평가 logic + KV CPU→GPU 복원 path |
| 15 | NEO > vanilla throughput | E2E gate (PLN_001 §5.6 의 win 기준 +3.91% v38 / +3.13% v41) | try44 v1.2 output_tps **1154** vs vanilla 추정 ~1100 (이득 ~0%) / try46 v38 진행 중 (예상 +3.91%) | chain 통로 닫힘 → throughput 이득 발현 안 됨. #5 (swap-in path) + #14 (migration) 활성화 필요 |
| 16 | CPU utilization HIGH | pacpu thread pool active 측정 | try44 cdec 0회 → CPU pacpu thread idle / try46 v38 4,217 fire → CPU 사용 high 추정. py-spy dump 분석 보류 | py-spy 5 sample 분석 (try44 의 5 dump + try46 의 5 dump). thread state distribution 측정 |
| 17 | token correctness | v41 적재 (no-fastmath BF16 kernel + ISPC `--opt=fast-math` 제거 + C++ `-Ofast` 제거) | v41 500p token loss **2.84%**, 1000p **10.38%** (PLN_001 §5.6) / try44 chain 0 → 정확도 vanilla 와 동일 / try45 OOM 으로 측정 미완 | v41 baseline 충분. chain firing 시에도 v41 정합 유지 검증 |
| 18 | deadlock 회피 | TSK_019 try22 deadlock fix (`engine/core.py:_handle_neo_swaps` 의 deferred_free 영역을 *gate 앞* 으로 이동) | try44 정상 완료 (3548s wall, 16.4K step, no stuck) | 정상 |
| 19 | silent worker crash 0 | TSK_015 4.5.2.c B-NEW fix (`scheduler.py` 의 auto-swap-in 제거 + adapter 의 predictive 영역에 일원화) | try44 / try46 정상 / try45 OOM (CUDA assert 아님 — 정상 OOM 처리) | 정상 |

---

## C. 항목 별 상세

### 1. request 단위 KV exclusive ownership

**의도 (NEO 논문)**
> "For any request that has already been prefilled in the system, its KV cache will either reside entirely in the GPU-cache—designated as a 'GPU-request'—or entirely in the CPU-cache."

mirror / partial split 아닌 **exclusive ownership**. swap 시 한쪽 → 다른 쪽으로 *완전* 이동.

**구현**
- `vllm/v1/core/sched/neo_cpu_kv_buffer.py` — `NeoCpuKvBuffer` 가 CPU pinned tensor pool 관리. layer × max_block × kv × dim 4D layout.
- `vllm/v1/engine/core.py:_handle_neo_swaps:496` — swap_out_req_ids 받으면 GPU KV → CPU buffer copy + GPU block free.
- `swap_in` 도 동일 구조 — CPU buffer → GPU block + buffer free (구현은 있으나 미발화).

**동작 (현재까지)**
| try | swap_out | swap_in |
|---|---|---|
| try44 (v1.2 + skip 유지) | 4,210 | **0** |
| try45 (skip 제거) | 측정 미완 (OOM) | — |
| try46 (v38) | 진행 중 | — (FORK STAT 측정 보류) |

→ swap_out 은 정상 fire, **swap_in path 가 항상 미발화**. 결국 SWAPPED_OUT reqs 는 `_handle_neo_swaps` 의 deferred_preempt path 로 흘러서 **vanilla preempt** 처리 (status=PREEMPTED + waiting.prepend) → 다음 ready 시 *처음부터 prefill* — KV migration 의 본래 의도 (CPU 에 잔류, GPU 만 evict) 와 다름.

**Gap → 수정 plan input**
- swap_in candidate evaluation logic 활성화 (`neo_scheduler.py` Step 3)
- swap_in 발화 조건: `kv_usage < swap_in_threshold (= 0.95 × swap_out_threshold)` 시 CPU 큐에서 head 이동
- KV CPU→GPU 복원 path 의 worker 측 wiring (`_handle_neo_swaps` 의 swap_in 영역 활성화)

### 2. CPU 가 attention 직접 수행

**의도 (NEO 논문)**
> "CPU 는 b1 (offloaded) 의 decoding attention 을 *직접* 처리. Q 만 GPU→CPU, KV 는 CPU 상주, attention output 만 CPU→GPU."

GPU 가 cdec request 의 attention 을 *처리할 의지조차 없음* (다른 일 중) → CPU 결과가 *유일한* 결과 → Q dependency dilemma 회피.

**구현**
- `csrc/cpu/pacpu/pacpu.ispc` — ISPC 로 작성된 CPU paged attention kernel. `paged_attention_cpu` 함수.
- `csrc/cpu/pacpu/core.h` — C++ wrapper.
- `csrc/cpu/pacpu/CMakeLists.txt` — ISPC + C++ 빌드 정의.
- `vllm/v1/attention/ops/neo_pacpu.py` — Python wrapper (`torch.ops.pacpu.paged_attention_cpu`).

**v37/v41 적재 history**
- v37 (`neo_pacpu_v37`) — 첫 적재 (fast-math + `-Ofast`).
- v41 (HEAD) — fast-math 제거 + `-Ofast` 제거. token loss 39% 감소. data_t = `__bf16` (BF16 native).

**동작**
- try46 (v38) NEO FORK STAT: `active=4,217 / 4,300 (98.1%)` — kernel call 정상 fire.
- try44 (v1.2): P7 `_interesting=False` 100% → `forward_context.neo_cdec_*` None → kernel call 0회.

**Gap**
- v1.2 chain 통로 (try22 skip) 가 닫혀서 kernel 까지 도달 안 함. kernel 자체는 OK.

### 3. Asymmetric Pipelining `forward_double`

**의도 (NEO 논문 + NEO_code_deepdive §4.4)**
> "두 sub-batch 가 *layer 단위 어긋남* 으로 attention dependency chain 을 분리. pipeline depth 1 with layer offset."

```
batch 0 : o0   |=>  postproj[i] → preproj[i+1]  | attention[i+1]   |=> [o0']
batch 1 : qkv1 |=>       attention[i]           | postproj[i] → preproj[i+1] |=> qkv1'
```

`cur_layer_id = (self.layer_id + cur_stage) % num_layers` — stage 0 = batch[1] layer i, stage 1 = batch[0] layer i+1.

**구현**
- `vllm/model_executor/models/llama.py:533~607` — `forward_neo_pipelined`, `neo_preproj`, `neo_postproj`.
- `vllm/v1/worker/sub_batch_executor.py:231` — `forward_double` (cb.preproj + attention + postproj).
- `sub_batch_executor.py:297` — `forward_pipeline` (전 layer 의 ping-pong loop).

**동작**
- try45 (skip 제거): chain 정상 도달 — `llama.py:607: out0, out1 = executor.forward_pipeline(sub_batches, embeddings_list)` → forward_double:231 → preproj:563 → input_layernorm:361 → **OOM**.
- try46 v38: NEO FORK STAT active=4217 / 정상 fire 추정.

**Crash 정확 위치** (try45)
```
File "/workspace/vllm_hybrid/vllm/model_executor/models/llama.py", line 607, in forward_neo_pipelined
File "/workspace/vllm_hybrid/vllm/v1/worker/sub_batch_executor.py", line 297, in forward_pipeline
File "/workspace/vllm_hybrid/vllm/v1/worker/sub_batch_executor.py", line 231, in forward_double
File "/workspace/vllm_hybrid/vllm/model_executor/models/llama.py", line 563, in preproj
File "/workspace/vllm_hybrid/vllm/model_executor/models/llama.py", line 361, in neo_preproj
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 126.00 MiB. GPU 5 has a total capacity of 79.18 GiB of which 92.00 MiB is free.
```

**Gap**
- forward_double 진입 시 GPU mem 잔여가 너무 적어 추가 alloc 실패. workspace tensor (preproj 의 hidden + residual buffer) 의 reuse / workspace pool / `gpu_memory_utilization` 의 forward_double 마진 예약 필요.

### 4. Stage 분할 (`forward_first_stage` / `forward_double` / `forward_last_stage`)

**의도**
- `forward_first_stage`: input projection (Q/K/V) + RoPE — 한 batch 의 layer 0 을 미리 처리.
- `forward_double`: layer 1 ~ N-1 의 ping-pong.
- `forward_last_stage`: 마지막 layer attention + output projection.

**구현**
- `llama.py:533~607` — `forward_neo_pipelined` 가 위 3 stage 를 호출.
- `sub_batch_executor.py:231~297` — 각 stage 별 함수.

**동작**
- try45 traceback: `forward_neo_pipelined` 까지 정상 진입 (`llama.py:607` 도달) → `forward_pipeline` → `forward_double` 도달 → preproj 에서 OOM.

**Gap**
- 정상 chain 이지만 OOM 으로 stage 의 *완전 실행* 도달 못 함.

### 5. 6 단계 Scheduler

**의도 (NEO_code_deepdive §3)**
1. Reserve budget for existing GPU decoding
2. swap_out — GPU over-budget 면 last running 을 CPU 큐로 preempt
3. swap_in — CPU 큐의 head 를 GPU 로 옮길 여유 있으면 진행
4. 새 prefilling 후보 분류 — GPU 빈 자리 우선
5. pipelined vs sequential 결정 (mode select)
6. 실제 prefilling launch

**구현**
- `vllm/v1/core/sched/neo_scheduler.py` — NeoScheduler 의 `schedule()` 메소드 (6 단계).
- `vllm/v1/core/sched/neo_scheduler_adapter.py:585` — `output = super().schedule()` 후 NeoScheduler 동기화 (line 600~666 의 cdec_ids 추출 + sub_batches attach).
- threshold env: `VLLM_NEO_SWAP_OUT_RATIO` (`neo_scheduler.py:214`), `VLLM_NEO_PREDICTIVE_THRESHOLD` (`neo_scheduler_adapter.py:471`).

**동작**
- Step 1, 2 정상 fire (try44 swap_out=4,210)
- **Step 3 (swap_in) 매 step 0 회** — CPU 큐 head 의 candidate evaluation 미발화 또는 조건 불충족
- **Step 5 (mode select) 항상 sequential** — predictor 기반 logic 미작동

**Gap**
- Step 3 활성화: swap_in candidate 의 KV slot 추정 + threshold 기반 fire
- Step 5 활성화: TablePerfPredictor 의 table 채우기 또는 heuristic mode select

### 6. Mode Select (pipelined vs sequential)

**의도 (NEO_code_deepdive §3.1)**
```python
pipelined_time  = (batches[0].gpu_time + batches[1].gpu_time) * num_layers
seqential_time  = max(...)
return batches if pipelined_rate > seqential_rate else [gpu_only_batch]
```

predictor 가 두 batch 의 GPU/CPU 시간 추정 → 더 빠른 mode 선택.

**구현**
- `neo_scheduler_adapter.py:128` — `self.predictor = ZeroPerfPredictor()` (fallback) + `self.table_predictor = TablePerfPredictor(self.vllm_config)` (lazy).
- `vllm/v1/core/sched/perfpredictor.py` — `ZeroPerfPredictor`, `TablePerfPredictor` 의 인터페이스.

**동작**
- ZeroPerfPredictor 가 모든 시간 0 반환 → mode select 항상 sequential
- TablePerfPredictor 의 table 미작성 (ModelProfiler 미구현)

**Gap**
- ModelProfiler 로 table 채우기 또는 heuristic (e.g., b1_count > threshold 면 pipelined) 도입.

### 7. 3-way attention dispatch (prefill / GPU decode / CPU decode)

**의도 (NEO_code_deepdive §4.5)**
한 SubBatch 안에 prefill / gdec / cdec 가 섞일 수 있음 → `_attention` 가 3-way:
1. prefill (flash_attn varlen_fwd)
2. GPU decode (paged_attention)
3. CPU decode (`torch.ops.pacpu.paged_attention_cpu`)

**구현**
- `vllm/model_executor/layers/attention/attention.py:773~893` — `cdec_future` 의 `_get_neo_cdec_executor().submit()` (background thread).
- `forward_context.neo_cdec_token_slice / seq_slice / req_ids` 가 cdec 영역 marker.

**동작**
- try44 P7=1600 (warmup cap) — `_interesting=False` 100% → `_tok=None _seq=None _req_ids=None` 매 호출.
- try46 (v38) FORK STAT active=4,217 → 정상 fire.

**Gap**
- v1.2 의 cdec dispatch path: scheduler.py:407 try22 skip 으로 SWAPPED_OUT 이 outer loop 에서 제거 → adapter cdec_ids=[] → forward_context 의 neo_cdec_* 미 set → attention.py 의 cdec_future submit 0회.
- 수정 좌표: try22 skip 의 *조건부 활성화* (chain firing 상태에서만 skip 우회) 또는 SWAPPED_OUT 의 *별도 schedule branch* 도입.

### 8. swap_out / swap_in 동시 발생 안 함 invariant

**의도** — 한 step 에 둘 중 하나만. 동시 발화 시 KV pool oscillation.

**구현**
- `vllm/v1/engine/core.py:_handle_neo_swaps:496` — deferred_free 영역에서 swap_out 우선 처리, swap_in 은 다음 step 으로.
- `neo_scheduler.py` 의 6 단계 algorithm 구조 자체가 둘을 분리.

**동작**
- try44 P8: 모든 step 에서 둘 중 하나만 attached (관찰: swap_out 만 fire, swap_in 항상 0).

**Gap**
- 정상.

### 9. `paged_attention_cpu` (pacpu) ISPC kernel

**의도 (NEO_code_deepdive §4.5)**
NEO 의 *pacpu* C++/ISPC kernel — `torch.ops.pacpu.paged_attention_cpu` 로 dispatch.

**구현**
- `csrc/cpu/pacpu/pacpu.ispc` — ISPC kernel (parallel for loop, vector ISA 활용).
- `csrc/cpu/pacpu/core.h` — C++ wrapper + `data_t = __bf16` (v41).
- `csrc/cpu/pacpu/CMakeLists.txt` — ISPC + C++ 빌드 정의 (v41 에서 `--opt=fast-math` 제거).
- `vllm/v1/attention/ops/neo_pacpu.py` — Python wrapper (`@torch.ops.pacpu` 등록).

**동작**
- try46 build OK / import OK / FORK STAT active=4,217 → kernel 정상 fire.
- try44 의 v1.2 도 build OK / import OK / 단지 호출 안 됨 (chain 통로 차단).

**Gap**
- 정상.

### 10. Q/K/V D2H transfer + qkvtr_e sync

**의도 (NEO_code_deepdive §4.6)**
`num_cdecs > 0` 시 `_attention` 진입 *전* Q/K/V 를 CPU 로 D2H. CPU 통신 stream 사용 + qkvtr_e event 로 sync.

**구현**
- `vllm/v1/worker/sub_batch_executor.py` 의 transfer stream + event mechanism.
- `attention.py:867` — `cdec_future = _get_neo_cdec_executor().submit(...)` 직전 D2H 처리.

**동작**
- try45: forward_double 진입 후 OOM → 실제 D2H 도달 못 함.
- try46 (v38): FORK STAT active=4,217 → 정상 fire 추정.

**Gap**
- forward_double OOM 회피 후 정확 측정.

### 11. `NeoSchedulerOutput → SchedulerOutput` 변환 (sub_batches attach)

**의도**
NEO 의 SubBatch 결정이 vLLM 의 SchedulerOutput 에 매핑되도록.

**구현**
- `neo_scheduler_adapter.py:600~666` (Phase B 적재):
  ```python
  vllm_ids = list(output.num_scheduled_tokens.keys())
  cdec_ids = [rid for rid in vllm_ids
              if requests[rid].status == SWAPPED_OUT]
  if cdec_ids and len(cdec_ids) < len(vllm_ids):
      output.neo_sub_batches = [b0_ids, b1_ids]
      output.neo_sub_batch_cdec_slices = [...]
      output.neo_sub_batch_cdec_seq_slices = [...]
  ```

**동작**
- try44 P5=535 (cdec_ids=0 매 step) — vllm_ids 에 SWAPPED_OUT 안 포함 (try22 skip 효과)
- try45 P5=4 step (cdec_ids=195 정상 attach) — try22 skip 제거 후 정상

**Gap**
- v1.2: try22 skip path 가 SWAPPED_OUT 을 outer loop 에서 미리 제거 → adapter 가 cdec_ids 못 만듦.

### 12. Worker side b0_eff / b1_eff 정렬

**의도**
worker 의 input_batch req 순서 ↔ scheduler 의 b0/b1 list 매핑 + swap_states 로 b0/b1 boundary 확보.

**구현**
- `vllm/v1/worker/gpu_model_runner.py:1064~1135` — `_neo_b0_eff_for_step`, `_neo_b1_eff_for_step` 계산 + swap_states reorder.
- depth 7 fix (commit `42fdcd5f4a` etc.) — boundary 계산 + greedy walk swap.

**동작**
- try45 P6=8 (8 worker × 1 step) — boundary=35, b0_eff=35, b1_eff=195 정상 split.

**Gap**
- 정상 분기. forward_double OOM 으로 다음 step 측정 미완.

### 13. `forward_pipeline` overlap (CPU↔GPU 진짜 병렬)

**의도 (NEO_code_deepdive §4.1)**
```
GPU:  [batches[0] linear] → [attention] → [post-attn]
CPU:                [batches[1] CPU attention (별도 thread)]
```

**구현**
- `sub_batch_executor.py:297` (`forward_pipeline`) — CUDA stream 분리 + executor.submit (background thread for cdec).
- v37 architectural fix (CUDA stream 분리) 로 진짜 병렬 보장.

**동작**
- try44 미발화
- try45 OOM 진입
- try46 v38 4,217 회 active dispatch — 진짜 병렬 추정 (측정 보류)

**Gap**
- try46 완료 후 wall vs CPU pacpu vs GPU forward 시계열 비교 측정.

### 14. KV migration 정책 (양방향)

**의도 (NEO 논문)**
- LRU-like policy 로 GPU↔CPU KV 를 양방향 migrate
- swap_out: GPU pressure 시 last-recently-used 를 CPU 로
- swap_in: CPU 의 head (가장 ready 한) 를 GPU 로

**구현**
- swap_out: 구현됨 (`_handle_neo_swaps` 의 swap_out_req_ids 처리)
- **swap_in: vLLM port 미구현** — `neo_scheduler.py:Step 3` 이 placeholder. CPU 큐의 head candidate evaluation 미작동.

**동작**
- try44: swap_out 4,210 / swap_in 0 → 결국 vanilla preempt path 로 흘러감 (`_handle_neo_swaps` 의 deferred_free 영역에서 status=PREEMPTED + waiting.prepend).
- 즉 NEO 의 *KV CPU 잔류 + 다음 ready 시 swap_in* 이 아닌 *GPU evict + 처음부터 prefill* (vanilla preempt) 로 동작.

**Gap (수정 plan 의 핵심)**
- LRU policy 적재
- swap_in candidate evaluation logic (`neo_scheduler.py:Step 3`)
- swap_in 발화 조건: `kv_usage < swap_in_threshold` 시 CPU 큐 head 이동
- KV CPU→GPU 복원 path (`_handle_neo_swaps` 의 swap_in 영역)

### 15. NEO 가 vanilla > 능가 (throughput win)

**의도 (CLAUDE.md objective + NEO 논문)**
- CLAUDE.md objective: GPU 활용률 + 결과 동등 + CPU 활용률 극대화 → cluster 전체 성능 향상
- NEO 논문 정량 우위 영역 (PLN_001 §5.6): 500p 50:50 H100×8 70B 에서 v38 +3.91% / v41 +3.13%

**현재까지**
- try44 v1.2: output_tps **1,154** vs vanilla 추정 ~1100 (이득 ~0% — chain 0 → vanilla preempt 등가)
- try46 v38 진행 중: 예상 +3.91% (PLN_001 §5.6 보고치 재확인)
- try45 OOM 으로 측정 미완

**Gap**
- chain 통로 닫힘 (try22 skip) + swap_in 미발화 → throughput 이득 발현 안 됨
- #5 (swap-in path) + #14 (migration) 활성화 필요

### 16. CPU utilization HIGH

**의도 (CLAUDE.md objective)**
> "CPU 의 활용률이 Idle 또는 낮은 Utilization 을 허락하지 않는다."

pacpu thread pool 이 실제 active.

**현재까지**
- try44 cdec 0 회 → CPU pacpu thread idle (NEO objective 위반)
- try46 v38 4,217 fire → CPU 사용 high 추정

**Gap**
- py-spy 5 sample 분석 (try44 의 5 dump + try46 의 5 dump 비교) 에서 thread state distribution 측정
- CPU util 직접 측정 (top, mpstat, perf 등) 추가

### 17. token correctness (분포 동등)

**의도 (CLAUDE.md Constraint)**
> "GPU만 사용했을 때와 결과 값이 달라져서는 안됨."
> 운영 해석: token-level bit-exact 가 아니라 **분포·의도 수준의 유사성**. binding gate = per-token logprob max abs diff + sequence PPL relative diff.

**구현**
- v41 적재 (no-fastmath BF16 kernel + ISPC `--opt=fast-math` 제거 + C++ `-Ofast` 제거)
- TST_003 verdict gate (IDE_006 의 `verdict_overall = verdict_d_ii`)

**현재까지** (PLN_001 §5.6)
| variant | 500p tok loss | 500p output_tps | 1000p tok loss | 1000p output_tps |
|---|---|---|---|---|
| vanilla | 0% | 2190.83 | 0% | 1869.61 |
| v37 | 12% | 1928.63 | — | — |
| v38 | 4.69% | 2276.42 (+3.91%) | 12.89% | 1815.81 |
| v41 | **2.84%** | 2259.26 (+3.13%) | **10.38%** | 1626.59 |

**Gap**
- v41 baseline 충분
- chain firing 시에도 v41 정합 유지 검증 필요 (try46 종료 후 token 비교)

### 18. deadlock 회피

**의도** — KV pool 영구 stuck (예: 99.33% 고정) 발생 안 함.

**구현**
- TSK_019 try22 deadlock fix (commit history): `engine/core.py:_handle_neo_swaps` 의 deferred_free 영역을 *gate 앞* 으로 이동 → 매 step 자연 발화.
- TSK_019 try26: vanilla preempt path 채택 (`status=PREEMPTED + num_computed_tokens=0 + waiting.prepend`).

**현재까지**
- try44: 정상 완료 (3,548s wall, 16.4K step, no stuck)

**Gap**
- 정상.

### 19. silent worker crash 0 (CUDA device-side assert)

**의도** — `input_batch.block_table.np[req_idx]` stale → cross-req KV contamination → CUDA device-side assert → silent worker crash 시나리오 (try10~try15 root) 의 회피.

**구현**
- TSK_015 4.5.2.c B-NEW fix: scheduler 의 auto-swap-in 제거 → adapter 의 predictive 영역에 일원화.
- TSK_015 4.5.2.c v33~v38: parallel CUDA stream + skip_gpu_attn 등 chain 계 fix.

**현재까지**
- try44 / try46 정상 (try46 진행 중 crash 0)
- try45 OOM (CUDA assert 와는 별도, 정상 OOM 처리 — EngineCore EngineDeadError)

**Gap**
- 정상.

---

## D. 참조 (Reference)

### NEO 논문
- arXiv: <https://arxiv.org/abs/2411.01142>
- MLSys 2025: <https://proceedings.mlsys.org/paper_files/paper/2025/hash/66a026c0d17040889b50f0dfa650e5e0-Abstract-Conference.html>
- GitHub (원 SwiftLLM): <https://github.com/NEO-MLSys25/NEO>

### IDE_006 내부 history
- `shadow_assists/features/IDE_006/NEO_redesign.md` — 4 차 재정의 결정 history
- `shadow_assists/features/IDE_006/NEO_code_deepdive.md` — NEO algorithm 분석 (논문용 reference)
- `shadow_assists/features/IDE_006/PLN_001.md` — IDE_006 plan
- `shadow_assists/features/IDE_006/PLN_001_neo_baseline_results.md` §5.6 — Phase D NEO chain (v37~v41) 측정 결과

### 본 분석 plan v2 산출물
- `shadow_assists/features/IDE_006/analysis_runtime_trace_v2.md` — try44 의 8 probe quantitative trace
- `shadow_assists/features/IDE_006/analysis_invariants_v2.md` — 4 invariant 도입 commit + NEO 충돌 매핑
- (예정) `analysis_pyspy_dumps_v2.md` — try44/45/46 의 py-spy stack 분석
- (예정) `analysis_v38_runtime_v2.md` — Phase 3 try46 v38 runtime trace (대조군)
- (예정) `analysis_invariant_repro_v2.md` — Phase 4 try47 의 minimal repro lifecycle
- (예정) `analysis_final_v2.md` — Phase 6 종합 + 다음 *수정 plan* input

### 측정 결과 디렉토리
- `eval/results/20260508_211625_try44_anal_v2_phase2A/` — try44 (v1.2 baseline)
- `eval/results/20260508_222510_try45_anal_v2_phase2B/` — try45 (skip 제거 → OOM)
- `eval/results/20260508_223224_try46_anal_v2_phase3_v38/` — try46 (v38 대조군, 진행 중)

### 핵심 commit
- `93ea1fcd5f` — HEAD (v1.2: launcher CLI args + try43 vanilla +74% 검증)
- `27b1bf15fa` — feat/ide006-neo-asymmetric (v41: no-fastmath FP16 kernel + skip_gpu_attn)
- `eeed0d46fc` — v38 (NEO throughput > vanilla 검증 PASS — 500p +3.91%)
- `f231bada8c` — v1.1 (NEO surgery stack 통합 + try28 정상 동작 검증)

---

## E. Change log

| 일자 | 변경 |
|---|---|
| 2026-05-08 | 본 문서 신규 작성 — try44/45/46 (진행 중) 결과 + Phase 5 git history 분석 + NEO_code_deepdive 의 mechanism 19 항목 추출 |
