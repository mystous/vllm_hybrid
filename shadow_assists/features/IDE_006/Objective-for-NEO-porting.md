# Objective for NEO Porting — vLLM 적재 검증 항목

| 항목 | 값 |
|---|---|
| 작성 | 2026-05-08 (최종 update 2026-05-11 — 22 항목 확장) |
| 영역 | IDE_006 / TSK_019 (NEO performance max) |
| 기준 분기 | `feat/ide006-tsk019-neo-performance-max` (HEAD = Option I+K+C+L 적재, uncommitted) |
| 비교 분기 | `feat/ide006-neo-asymmetric` (`27b1bf15fa`, v41) / commit `eeed0d46fc` (v38) / v1.3 = `f2678c2f4` (try51 stack) |
| 결과 근거 | try77~try81 (2026-05-10/11 회차) — Option I/K/C/L 누적 fix + chain firing 98.9% 달성 |
| 본 문서 | NEO 논문의 mechanism 별 *구현* 상태 + *동작* 상태 + *근거 위치* 단일 출처. **2026-05-11 22 항목 확장** — 우리 fix 영역 (Option I/L/M 의 CPU resident queue 영구화 / 매 step 증분 CPU block alloc / swap-in GPU/CPU block 동기화) 신설. |

---

## A. 개략 표 — at-a-glance

> Legend: ✅ 정상 / 🔶 부분 / ❌ 미구현·미발화 / ⚠️ 발화는 했으나 결과 부적절 / ⏳ 측정 미수행

**측정 기준**: try81 (2026-05-11 Option I+K+C+L 적재 mid-stop ~30분 진행) — chain firing 98.9% 영역.

| # | 항목 | 구현 | 동작 | try81 측정 |
|---|---|---|---|---|
| 1 | request 단위 KV exclusive ownership | ✅ | ✅ | SWAP_OUT_CALL=160 |
| 2 | CPU 가 attention 직접 수행 | ✅ | ✅ | active=2276/2300 = **98.96%** |
| 3 | Asymmetric Pipelining (`forward_double`) | ✅ | ✅ | OOM=0 (forward_double 정상 fire) |
| 4 | Stage 분할 (first / double / last) | ✅ | ✅ | OOM=0 |
| 5 | 6 단계 Scheduler | ✅ | ✅ | Step 1~6 모두 fire (D15+D16 + decide_mode) |
| 6 | Mode Select (pipelined vs sequential) | ✅ | ✅ | decide_mode 매 step 호출 (Option C) |
| 7 | 3-way attention dispatch (prefill / GPU dec / CPU dec) | ✅ | ✅ | NEO CDEC CALL=185,300/worker (≈1.48M 합) |
| 8 | swap_out / swap_in 동시 발생 안 함 invariant | ✅ | ✅ | 둘 다 attached 0 회 |
| 9 | `paged_attention_cpu` (pacpu) kernel | ✅ | ✅ | D11 OOB=**0** (try80 의 18560 대비) |
| 10 | Q/K/V D2H transfer + qkvtr_e sync | ✅ | ✅ | pacpu 호출 시 자동 fire |
| 11 | `NeoSchedulerOutput → SchedulerOutput` 변환 (sub_batches attach) | ✅ | ✅ | eligible=2276 (FORK active>0 184 회) |
| 12 | Worker side b0_eff / b1_eff 정렬 | ✅ | ✅ | reject_b1_empty=0 reject_prefix_fail=0 |
| 13 | `forward_pipeline` overlap (진짜 CPU↔GPU 병렬) | ✅ | ✅ | active 98.96% (try44 의 0% 대비) |
| 14 | KV migration 정책 (LRU-like GPU↔CPU 양방향) | 🔶 | 🔶 | swap_out=160 / swap_in shape mismatch=5912 (#22 Option M 영역) |
| 15 | NEO 가 vanilla > 능가 (throughput win) | ❌ | ❌ | **400 tps vs vanilla 4690** (11.7× **저하**) |
| 16 | CPU utilization HIGH (CLAUDE.md objective) | ✅ | ⏳ | (worker py-spy 미수행 — CDEC CALL 1.48M 으로 추정) |
| 17 | token correctness (분포 동등) | 🔶 (v41) | ⏳ | (run 미완주 mid-stop — 측정 보류) |
| 18 | deadlock 회피 | ✅ | ✅ | EngineDead=0 |
| 19 | silent worker crash 0 (CUDA device-side assert 없음) | ✅ | ✅ | AssertionError=0 CUDA-assert=0 SEGV=0 |
| **20** | **CPU resident queue 영구화** (NEO `cpu_decoding_q` 정합) | ✅ | ✅ | mirror_set_size 안정 영역 10 (1040 회 분포) — Option I MIN_BUFFER guard |
| **21** | **매 step 증분 CPU block alloc** (NEO `omit_last=False` 정합) | ✅ | ✅ | NEO BUF EXTEND=64 (FAIL=0) — Option L `ensure_capacity` |
| **22** | **swap-in GPU/CPU block 수 동기화** (mirror reqs GPU 복귀) | ❌ | ❌ | shape mismatch=5912 → mirror reqs GPU 복귀 불가 → CPU 단독 decode → #15 저하 root |

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
| **20** | **CPU resident queue 영구화** (NEO `cpu_decoding_q` 등가) | `neo_scheduler_adapter.py:1051-1080` (Option I MIN_BUFFER guard) — D4 swap_in path 의 `_max_swap_in = min(env, max(0, len(mirror) - MIN_BUFFER))` | try81: mirror_set_size 안정 영역 10 (1040 회 분포). NEO swiftllm/server/scheduler.py 의 `cpu_decoding_q` deque (swap_out~swap_in 잔류) 의 동등 path | 정상. env `VLLM_NEO_MIRROR_MIN_BUFFER` default 8 |
| **21** | **매 step 증분 CPU block alloc** (NEO `omit_last=False` 등가) | `neo_cpu_kv_buffer.py:204-265` (Option L `ensure_capacity` API) + `attention.py:797-840` (caller) | try81: NEO BUF EXTEND=64 (FAIL=0), **D11 OOB=0** (try80 의 18560 대비). NEO swiftllm/server/block_manager.py:153-162 의 `cpu_block_manager.alloc(cdec_reqs, omit_last=False)` 등가 logic 모방 | 정상. env `VLLM_NEO_OPTION_L` default 1. NEO `DeviceBlockManager.alloc` 의 increment logic (line 78-112) 모방 |
| **22** | **swap-in GPU/CPU block 수 동기화** (mirror reqs GPU 복귀) | (미적재 — Option M 영역). NEO 정통: swap-in 시 `gpu_block_manager.alloc(swap_in_reqs)` 의 size 가 CPU buffer 의 *현재 alloc block 수* 와 일치해야. 우리 `kv_cache_manager.neo_swap_in_alloc(req)` (`kv_cache_manager.py:463-498`) 는 `n_tokens = req.num_computed_tokens` 만 보고 GPU alloc — CPU buffer 의 Option L extend 한 block 수와 mismatch | try81: shape mismatch=5912 회 (swap-in 시도 7360 중 80%). 진짜 swap-in done ≈ 1448 또는 미만 | Option M 적재 필요 — `neo_swap_in_alloc` 가 CPU buffer 의 `get_block_ids(req)` size 와 동기화하거나, swap_out 시 *upfront max_total_tokens block alloc* 후 swap-in 시 일치 |

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
- try77~try81 모두 crash 0.

**Gap**
- 정상.

---

### 20. CPU resident queue 영구화 (NEO `cpu_decoding_q` 정합)

**의도 (NEO 정통)**

NEO `swiftllm/server/scheduler.py:107-187` 의 `cpu_decoding_q: deque[Request]` — swap_out 시 `appendleft(victim)`, swap_in 시 `popleft()`. **swap_out 과 swap_in 사이의 모든 reqs 의 영구 큐** — 그 안의 reqs 가 *매 step* `_decide_mode_and_gen_batch.Step 3` 에서 `batches[1].add_cdec(req)` 으로 CDEC dispatch.

> "Step 3: split CPU decoding requests. `for req in self.cpu_decoding_q: batches[next_batch_idx].add_cdec(req); remains = self._get_remains(batches); ...`"

핵심 — `cpu_decoding_q` 가 *충분히 잔류* 해야 CDEC dispatch fire 영역 확보.

**구현**
- `vllm/v1/core/sched/neo_scheduler_adapter.py:854-893` (D1 mirror add) — swap_out 시 `_neo_cpu_resident_mirror.add(rid)`.
- `vllm/v1/core/sched/neo_scheduler_adapter.py:1051-1080` (**Option I MIN_BUFFER guard** — 2026-05-11 적재) — D4 swap_in path 의 `_max_swap_in = min(env, max(0, len(mirror) - MIN_BUFFER))`. mirror size 가 MIN_BUFFER 미만이면 swap_in 0 — *영구 큐 시간 확보*.
- env: `VLLM_NEO_MIRROR_MIN_BUFFER` (default 8).

**우리 결손 영역 (Option I 전)**
- D4 swap_in cap=2 + D15+D16 swap_out cap=2 + cooldown 5 의 *균형 결과 mirror size 거의 항상 0~2*
- D17C/D19 의 cdec_cands 추출 시점 (line 793) 이 D1 mirror add (line 854) 보다 *앞* → mirror = D4 가 비운 후 = 거의 0
- 따라서 *어떤 step 도 sub_batches attach 영역 진입 못 함* → chain firing 0 (try73~try78)

**Option I 적재 후 (try81)**
- mirror_set_size 안정 영역 10 (1040 회 분포), 최대 56 (cap 도달 22 회)
- D17C `cdec_cands_oc` 가 *매 step non-empty* → sub_batches attach 영역 진입 가능 → chain firing 98.9%

**Gap**
- 정상. NEO 정통의 *영구 큐 시간 확보* 등가 메커니즘 입증.

### 21. 매 step 증분 CPU block alloc (NEO `omit_last=False` 정합)

**의도 (NEO 정통)**

NEO `swiftllm/server/block_manager.py:153-162` 의 `_alloc_blocks_for_batch`:
```python
def _alloc_blocks_for_batch(self, batch: SubBatch):
    return (
        self.gpu_block_manager.alloc(batch.all_reqs[:batch.num_prgds], ...),
        self.cpu_block_manager.alloc(batch.all_reqs[batch.num_prgds:], omit_last=False)
        # ↑ batch.num_prgds 이후 = cdec_reqs. 매 step 호출 — 증분 alloc.
    )
```

`DeviceBlockManager.alloc` (line 78-112) 의 logic:
```python
tgt_num_blks = (seq_lens - 1) // self.block_size + 1
new_num_blks = tgt_num_blks - seq_num_blks  # ← 증분
```

핵심 — `pacpu/core.h:21-22` 의 `block_pos = (seq_len-1)/BLOCK_SIZE; block_id = block_table[block_pos]` — 매 step seq_len 진행 시 block_table[block_pos] 가 *valid block* 이어야. 매 step 증분 alloc 의 *진짜 이유*.

**우리 결손 영역 (Option L 전)**
- `vllm/v1/core/sched/neo_cpu_kv_buffer.py:155-202` 의 `alloc` 가 *swap-out 시점 1회만* 호출 가능 (재 alloc 시 `ValueError`).
- worker `_neo_handle_kv_swap.swap_out` (`gpu_model_runner.py:6424`) 의 `buf.alloc(req_id, num_blocks=len(gpu_blocks))` — *swap-out 시점 GPU blocks 와 동일 수* 만.
- Option K 의 `num_new_tokens=1` 진행 시 seq_len 증가 → block_pos 가 *swap-out 시점 nblock 너머* 도달 → invalid block_id → pacpu `brute::store_kv` SIGSEGV (try60-γ root)
- D11 dynamic precheck (`attention.py:798-880`) 가 OOB 시 cdec dispatch skip — try80 에서 D11 OOB **18560** 회 fire.

**구현 (Option L — 2026-05-11 적재)**
- `vllm/v1/core/sched/neo_cpu_kv_buffer.py:204-265` — `NeoCpuKvBuffer.ensure_capacity(req_id, target_num_blocks)`:
  - 현재 alloc 의 block 수 < target → 차이만큼 free pool 에서 추가 alloc
  - 충분하면 noop
  - thread-safe (`@_neo_synchronized`)
  - NEO `DeviceBlockManager.alloc` 의 increment logic 정합
- `vllm/model_executor/layers/attention/attention.py:797-840` — worker attention.py 의 cdec dispatch 직전 호출:
  ```python
  for _rid_l, _seq_len_l in zip(_req_ids, _seq_lens_optL):
      _target_nblk_l = (_seq_len_l + 15) // 16
      buf.ensure_capacity(_rid_l, _target_nblk_l)
  ```
- env: `VLLM_NEO_OPTION_L` (default 1).

**Option L 적재 후 (try81)**
- NEO BUF EXTEND count=64 (FAIL=0) — CPU pool 충분
- **D11 OOB count=0** (try80 의 18560 대비 100% 감소)
- NEO CDEC CALL=185,300/worker (try80 의 32 대비 +469× 폭증) — pacpu kernel 의 *실효 dispatch* 입증

**Gap**
- 정상. NEO 정통의 매 step 증분 alloc 등가.

### 22. swap-in GPU/CPU block 수 동기화 (mirror reqs GPU 복귀)

**의도 (NEO 정통)**

NEO `swiftllm/server/block_manager.py:170-192` 의 `_initiate_swap`:
```python
def _initiate_swap(self, reqs, is_swap_out, use_itm=False, omit_last=True):
    src_blk_pids = src_block_manager.free(reqs, int(use_itm))
    dst_blk_vids, dst_blk_pids = dst_block_manager.alloc(reqs, omit_last=omit_last)
    return src_blk_pids, dst_blk_vids, dst_blk_pids
```

swap-in 시 (is_swap_out=False) dst = GPU block manager. `dst_block_manager.alloc(reqs, omit_last=False)` — GPU 에 *현재 seq_len 의 모든 block alloc*. CPU buffer 의 block 수 와 *일치*.

핵심 — `_alloc_blocks_for_batch` 가 매 step `gpu_block_manager.alloc(prgd_reqs, ...)` + `cpu_block_manager.alloc(cdec_reqs, omit_last=False)`. **GPU 와 CPU 둘 다 매 step alloc** — *swap-in 시점에 자동 일치*.

**우리 결손 영역 (Option M 미적재)**

- `vllm/v1/core/kv_cache_manager.py:463-498` 의 `neo_swap_in_alloc(request)`:
  ```python
  n_tokens = request.num_computed_tokens  # ← request side num_computed
  num_blocks = self.coordinator.get_num_blocks_to_allocate(
      request_id=request.request_id,
      num_tokens=n_tokens, ...
  )
  ```
  GPU alloc = `ceil(num_computed / 16)`. *request.num_computed_tokens 기준*.

- 그러나 *CPU buffer 의 block 수* 는 Option L 의 `ensure_capacity` 가 매 step extend → CPU 의 `len(get_block_ids(req))` 가 *request.num_computed_tokens / 16 보다 클 수 있음* (CPU 가 더 빨리 extend).

- worker `_neo_swap_in_one_req` (`gpu_model_runner.py:6507-6541`):
  ```python
  k_cpu, v_cpu = buf.copy_layer_out(req_id, layer)  # CPU 의 모든 blocks
  kv[0][gpu_idx] = k_gpu  # gpu_idx size = swap-in 시점 alloc 한 GPU blocks 수
  ```
  *k_cpu.shape[0] > gpu_idx.size* 시 broadcast mismatch.

**현재까지 (try81)**
- swap-in shape mismatch warning **5912 회** — 5912/7360 = 80% swap-in 시도 fail
- 실제 swap-in 성공 ≈ 1448 회 또는 미만
- mirror reqs 가 *GPU 복귀 못 함* → 그 reqs 의 token 진행은 *CPU 단독 decode 만* → throughput 11× 저하 root (#15)

**Gap (Option M 적재 plan)**

**M1 (단순)**: swap_out 시 `buf.alloc` 의 num_blocks 를 *upfront max_total_tokens 기준* 으로 alloc. 그러면 swap-in 시점 GPU alloc 도 *max_total* — 자동 일치. CPU pool 부족 시 swap_out fail → vanilla preempt fallback.

**M2 (정통)**: `neo_swap_in_alloc` 가 `n_tokens` 대신 *CPU buffer 의 `get_block_ids(req)` size * 16* 기준 GPU alloc. NEO 정통의 *full pipeline* 정합.

본 항목 미적재 = mirror reqs 가 *영원히 CPU 단독 decode*. NEO paper 의 *GPU+CPU 동시 동작* (14% gain on H100) 영역 진입 못 함.

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
- `eval/results/20260510_235401_try77_v4_K_OptI_only/` — Option I single (mirror MIN_BUFFER) infra prerequisite PASS
- `eval/results/20260511_000346_try78_v4_K_OptI_plus_OptC/` — I+C (D10 가드 충돌 진단)
- `eval/results/20260511_001239_try80_v4_K_OptIKC/` — I+K+C (chain firing 96% + D11 OOB 18560)
- `eval/results/20260511_003320_try81_v4_K_OptIKCL/` — **I+K+C+L (chain firing 98.9% + D11 OOB 0 + crash 0)**

### NEO source 정합 검증 (2026-05-11)
- `/tmp/NEO_ref/` (clone of `NEO-MLSys25/NEO`)
- 핵심 paths:
  - `swiftllm/server/scheduler.py:142-235` — `_decide_mode_and_gen_batch` Step 3 (load-aware cdec)
  - `swiftllm/server/block_manager.py:153-162` — `_alloc_blocks_for_batch` (gpu+cpu 매 step 증분)
  - `swiftllm/server/block_manager.py:170-192` — `_initiate_swap` (swap-in 시 dst alloc)
  - `pacpu/core.h:11-31` — `brute::store_kv` (block_pos=(seq_len-1)/BS, block_table[block_pos])
- 우리 4 단계 fix 의 NEO 정합 검증 완료 (#20 Option I / #21 Option L / #22 Option M 영역)

### 핵심 commit
- `93ea1fcd5f` — HEAD (v1.2: launcher CLI args + try43 vanilla +74% 검증)
- `27b1bf15fa` — feat/ide006-neo-asymmetric (v41: no-fastmath FP16 kernel + skip_gpu_attn)
- `eeed0d46fc` — v38 (NEO throughput > vanilla 검증 PASS — 500p +3.91%)
- `f231bada8c` — v1.1 (NEO surgery stack 통합 + try28 정상 동작 검증)
- `f2678c2f4` — **v1.3 (try51 stack — chain firing 0.6% 첫 측정)**
- **HEAD (uncommitted, 2026-05-11)** — Option I (`neo_scheduler_adapter.py:1051-1080`) + Option K (`scheduler.py:483-512`) + Option C (`neo_scheduler_adapter.py:793-870`) + Option L (`neo_cpu_kv_buffer.py:204-265` + `attention.py:797-840`). **chain firing 98.9% 달성** — commit 대기 (사용자 명시 후만, NEO ground rule).

---

## E. Change log

| 일자 | 변경 |
|---|---|
| 2026-05-08 | 본 문서 신규 작성 — try44/45/46 (진행 중) 결과 + Phase 5 git history 분석 + NEO_code_deepdive 의 mechanism 19 항목 추출 |
| 2026-05-11 | **22 항목 확장 + try81 (Option I+K+C+L) 기준 상태 update**. (1) 헤더 metadata update — HEAD = Option I+K+C+L 적재 (uncommitted), 측정 회차 try77~try81. (2) A. 개략 표 — 22 항목 + try81 측정값 column 추가. 기존 19 항목 의 동작 상태 ⚠️/❌ 영역들이 try81 에서 *대부분 ✅* 로 update (chain firing 98.9% 달성). (3) B. 상세 표 — #20/#21/#22 추가. (4) C. 항목 별 상세 — #20 (CPU resident queue 영구화, Option I MIN_BUFFER guard), #21 (매 step 증분 CPU block alloc, Option L `ensure_capacity`), #22 (swap-in GPU/CPU block 동기화, Option M 미적재) 신설. (5) D. 참조 — NEO github / paper / PLN_001 §5.13 추가. (6) **잔존 결손**: #15 (throughput 11× 저하), #22 (swap-in shape mismatch 5912 회 → mirror reqs GPU 복귀 불가). Option M 적재 가 본 phase 의 *다음 step*. (7) 본 update 는 NEO github (`NEO-MLSys25/NEO`) 소스 + paper (MLSys 2025) 직접 분석 후 *소스 fact 기반* 작성. |
