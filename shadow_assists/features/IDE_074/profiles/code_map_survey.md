# 코드 지도 조사 (읽기 전용 에이전트 산출, 2026-09-17)

조사 완료했습니다. 아래는 요청하신 경계점 매핑입니다. 모든 경로는 **컨테이너 `sgl-kt` 내부 절대경로**이며, 파일은 하나도 수정하지 않았습니다.

---

# 0. 소스 트리 위치 (중요: 두 벌이 있고 "실행되는 쪽"이 다름)

| 구성요소 | 실제 실행되는 경로 | 비고 |
|---|---|---|
| SGLang (patched) | `/sgl-workspace/sglang/python/sglang/` | git `71de97b` + 로컬 수정 10개 파일 (`git status` 확인) |
| kt_kernel **Python** | `/usr/local/lib/python3.12/dist-packages/kt_kernel/` | **이게 런타임에 import 됨.** `experts_base.py` 1067줄 = 소스트리(940줄)에 `/tmp/cf_py_patch.py`, `/tmp/cf_skip_patch.py` 가 덧씌워진 버전 |
| kt_kernel **Python 소스트리** | `/sgl-workspace/ktransformers/kt-kernel/python/experts_base.py` | 940줄. **CALLBACK_FREE 패치 없음 → 실행 안 됨.** 혼동 주의 |
| kt-kernel **C++** | `/sgl-workspace/ktransformers/kt-kernel/` | `cpu_backend/`, `operators/amx/`, `ext_bindings.cpp`. `.so` 문자열과 일치 확인함 (`KT_CF_SKIP_EMPTY_DEF`, `[kt-cf]`, `[kt-tq]`, `[kt-wrap]`, `Profiling Results` 모두 `.so` 내 존재) |
| 빌드된 확장 | `/usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so` | |

모델 파라미터 (config.json 확인): `hidden_size=6144`, `num_experts=160`, `num_experts_per_tok=8`, `moe_intermediate_size=2560`, `num_hidden_layers=62`, bf16.
런 파라미터(IDE_071/072 문서 기준): `--kt-num-gpu-experts 96`, `--kt-cpuinfer 96`, `--kt-threadpool-count 2`, `--kt-max-deferred-experts-per-token 8`.

> ⚠️ **`KT_GPU_EXPERTS_PER_LAYER` 는 현재 컨테이너 코드에 존재하지 않습니다 (미확인/미적용).** 호스트 문서 `~/projects/vllm_hybrid/shadow_assists/features/IDE_071/OPTION_REGISTRY.md:72` 에 "sglang kt_ep_wrapper.py (IDE_070 패치, 컨테이너 로컬) … 패치 적용 시에만" 이라 기록돼 있으나, 현재 `/sgl-workspace/sglang` 전체 grep 결과 해당 문자열 0건입니다. 즉 지금 트리에서는 `KTConfig.num_gpu_experts = server_args.kt_num_gpu_experts` (전 층 동일 96) 로만 동작합니다.

---

# 1. Router/topk → hot/cold 분리

| # | file::function::line | 인용 |
|---|---|---|
| 1a | `/sgl-workspace/sglang/python/sglang/srt/models/qwen3_moe.py::Qwen3MoeSparseMoeBlock.forward_normal::328-335` | `router_logits, _ = self.gate(hidden_states)` / `topk_output = self.topk(hidden_states, router_logits, expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id,),)` ← **로컬 패치** (upstream 은 인자 없음) |
| 1b | `/sgl-workspace/sglang/python/sglang/srt/layers/moe/topk.py::TopK.forward_cuda::568-575` | `topk_output = select_experts(hidden_states=..., layer_id=self.layer_id, router_logits=..., expert_location_dispatch_info=expert_location_dispatch_info,)` |
| 1c | `.../topk.py::_post_process_topk_ids::1976-1978` | `topk_ids = _biased_grouped_topk_postprocess(topk_ids, expert_location_dispatch_info, num_token_non_padded)` ← **여기서 logical→physical 재매핑**. hotmap(`--init-expert-location`)에 의해 hot expert 가 물리 id 0..95 로 모임 = "hot/cold 분리의 실질적 근거" |
| 1d | `.../topk.py::select_experts::2314` | `topk_ids, topk_weights, recorder_topk_ids = _post_process_topk_ids(` |
| 1e (GPU 쪽 마스킹) | `/sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py::mask_cpu_expert_ids::132-148` | `@torch.compile(dynamic=True, backend=get_compiler_backend())` … `topk_ids[topk_ids >= num_gpu_experts] = -1` |
| 1f | `.../kt_ep_wrapper.py::KTEPWrapperMethod.apply::408` | `masked_topk_ids = mask_cpu_expert_ids(topk_ids, self.num_gpu_experts)` — **in-place 이므로 CPU submit(394) 이 먼저 일어나야 함** |
| 1g (CPU 쪽 마스크) | `.../kt_ep_wrapper.py::KTEPWrapperMethod.create_weights::252-256` | `_gpu_mask = torch.zeros(num_experts, dtype=torch.bool)` / `_gpu_mask[: self.num_gpu_experts] = True` → `gpu_experts_mask=_gpu_mask` (263) ← **로컬 패치** |
| 1h | `/usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py::BaseMoEWrapper.__init__::477-490` | `self.gpu_experts_mask = torch.empty(num_experts, dtype=torch.bool, device="cpu", pin_memory=True)` … `self._gpu_experts_mask_gpu = self.gpu_experts_mask.to(...)` |
| 1i (C++ skip) | `/sgl-workspace/ktransformers/kt-kernel/operators/common.hpp::GeneralMOEConfig::should_skip_expert::256-258` | `return expert_id < 0 \|\| expert_id >= expert_num \|\| (gpu_experts_mask && gpu_experts_mask[expert_id]);` ← **CPU 가 hot expert 를 건너뛰는 유일한 지점** |
| 1j (deferred 선택) | `.../experts_base.py::BaseMoEWrapper.select_deferred_experts::543-586` | `protected_k = max(0, min(int(protected_k), topk))` (567) / `if protected_k == 0: deferred_ids = expert_ids.clone(); immediate_ids = torch.full_like(expert_ids, -1)` (568-571) |
| 1k (protected_k 계산) | `.../experts_base.py::BaseMoEWrapper._prepare_forward_cpu_buffers::643-645` | `protected_k = self.num_experts_per_tok - self.max_deferred_experts_per_token` / `immediate_ids, deferred_ids = self.select_deferred_experts(topk_ids_long, topk_weights, protected_k)` |

**핵심 귀결 (deferred=8, top_k=8 설정):** `protected_k = 8-8 = 0` → **line 568-571 분기**를 탄다. 즉 `immediate_ids` 전부 `-1`, `deferred_ids = topk_ids` 전량. topk 기반 선택(573) 은 **실행되지 않는다**. `KT_COLD_DEFER` (553-566) 도 미설정이라 미사용.

---

# 2. CPU 입력 전송 (DtoH → pinned ring buffer)

모두 `/usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py`.

| # | function::line | 인용 |
|---|---|---|
| 2a | `KExpertsCPUBuffer::216`, `buffer_depth::227` | `buffer_depth: int = 2` |
| 2b | `KExpertsCPUBuffer.get_buffer::237-291` | `batch_size = (cls.channel, hidden_states.shape[0])` (239) / `if batch_size in cls.capture_buffers: return cls.capture_buffers[batch_size]` (243-244) |
| 2c | 버퍼 텐서 정의 `::249-276` | `input_tensor_cpu … dtype=torch.bfloat16` (250) / `immediate_experts_ids_cpu … dtype=torch.long` (254) / `deferred_experts_ids_cpu = torch.full(..., -1, dtype=torch.long)` (258) / `weights_cpu … dtype=torch.float32` (262) / `output_cpu … dtype=torch.bfloat16` (266) / `bsz_tensor_cpu … dtype=torch.int32` (270) / `output_gpu …` (274) — 전부 `pin_memory=True` |
| 2d | `BaseMoEWrapper._prepare_forward_cpu_buffers::628-629` | `current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth` / `next_slot = (current_slot + 1) % KExpertsCPUBuffer.buffer_depth` |
| 2e | **DtoH 복사 4건** `::650-654` | `input_tensor_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)` / `weights_cpu[current_slot].copy_(topk_weights, non_blocking=True)` / `immediate_experts_ids_cpu[current_slot].copy_(immediate_ids, non_blocking=True)` / `if deferred_ids is not None: deferred_experts_ids_cpu[current_slot].copy_(deferred_ids, non_blocking=True)` |
| 2f | 길이 가드 `_check_qlen_fits_cpp_buffers::588-606` | `if qlen > self.chunked_prefill_size: raise ValueError(...)` |

**스트림**: `torch.cuda.current_stream(x.device)` — `kt_ep_wrapper.py::KTEPWrapperMethod.submit::346-348` 에서 `torch.cuda.current_stream(x.device).cuda_stream` 를 넘기지만, `.copy_()` 자체는 **호출 시점의 current stream** (디코드 그래프 캡처 중에는 캡처 스트림) 에서 실행됨. 별도 copy stream 없음.

**페이로드 (bs 토큰, hidden=6144, k=8):**

| 텐서 | dtype | 행당 바이트 | bs=1 | bs=64 |
|---|---|---|---|---|
| hidden (DtoH) | bf16 | 12,288 | 12 KiB | 768 KiB |
| immediate ids (DtoH) | int64 | 64 | 64 B | 4 KiB |
| deferred ids (DtoH) | int64 | 64 | 64 B | 4 KiB |
| weights (DtoH) | fp32 | 32 | 32 B | 2 KiB |
| **output (HtoD)** | bf16 | 12,288 | 12 KiB | 768 KiB |

`bsz_tensor_cpu` 는 GPU 로부터 복사되지 않음 — 할당 시 `rows` 로 고정 채워짐 (270). 즉 **런타임 토큰 수는 항상 캡처 shape 와 동일**하다고 가정된 구조.

---

# 3. CPU job 제출

| # | file::function::line | 인용 |
|---|---|---|
| 3a | `.../kt_ep_wrapper.py::KTEPWrapperMethod.submit::346-348` | `self.wrapper.submit_forward(x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream)` |
| 3b | `.../experts_base.py::BaseMoEWrapper.submit_forward::806` | 진입점 |
| 3c | `…::822-824` | `_immediate_ids, deferred_ids, _buffers, current_slot, next_slot = self._prepare_forward_cpu_buffers(...)` |
| 3d | **skip-empty-imm** `…::852-855` | `_cf_skip_imm = (bool(_os_sk.environ.get("KT_CALLBACK_FREE")) and bool(_os_sk.environ.get("KT_CF_SKIP_EMPTY_IMM"))` / `and incremental and not sync_submit` / `and self.max_deferred_experts_per_token >= self.num_experts_per_tok)` |
| 3e | task 생성 `…::856-864` | `immediate_task = (0, 0) if _cf_skip_imm else self.moe.forward_task(bsz_slot_tensor.data_ptr(), ...size(-1), ids.data_ptr(), weights.data_ptr(), input.data_ptr(), output_cpu[current_slot].data_ptr(), incremental,)` |
| 3f | **CF 경로 pending** `…::879-886` | `if _os_cf.environ.get("KT_CALLBACK_FREE"):` / `self._cf_pending_imm = immediate_task; self._cf_pending_ci = _ci_s; self._cf_pending_stream = cuda_stream` / `else: _ci_s.submit_with_cuda_stream(cuda_stream, immediate_task)` |
| 3g | deferred task `…::903-917` | `deferred_task = self.moe.forward_task(..., output_cpu[next_slot].data_ptr(), False,)` / `elif getattr(self, "_cf_pending_imm", None) is not None: self._cf_pending_def = deferred_task` |
| 3h | **패킷 확정 + GPU 트리거** `…::919-935` | `_cf_mask_args = (int(self.gpu_experts_mask.data_ptr()), int(self.gpu_experts_mask.numel()), int(output_cpu[next_slot].shape[-1]) * 2)` (924) / `if torch.cuda.is_current_stream_capturing(): _slot = _ci.arm_packet(_imm, _def if _has else _dummy, _has, *_cf_mask_args)` (925-926) / `else: … self._cf_eager_slot = _ci.arm_packet(...) … _ci.rearm_packet(self._cf_eager_slot, ...)` (928-932) / `_ci.go_on_stream(self._cf_pending_stream, _slot)` (933) / `self._cf_last_slot = _slot` (934) |

## C++ 측

| # | file::function::line | 인용 |
|---|---|---|
| 3i | `cpu_backend/cpuinfer.h::CPUInfer::arm_packet::172-183` | `int slot = n_packets_.fetch_add(1);` (174) / `p.imm_fn = (void (*)(void*))imm.first; p.imm_args = (void*)imm.second;` (177) / `__atomic_store_n(&p.armed, true, __ATOMIC_RELEASE);` (181) |
| 3j | `cpuinfer.h::CPUInfer::go_on_stream::193-197` | `int r = cu_write_val32_((void*)user_cuda_stream, (unsigned long long)(uintptr_t)(go_dev_ + slot), 1u, 0u);` ← **`cuStreamWriteValue32` (graph-capturable)** |
| 3k | `cpuinfer.h::CPUInfer::ensure_cf_::155-169` | `cudaHostAlloc((void**)&go_host_, kMaxPackets * sizeof(unsigned int), cudaHostAllocMapped \| cudaHostAllocPortable)` (158) / `poller_ = std::thread([this]() { this->poll_loop_(); });` (167) |
| 3l | `cpuinfer.h::CPUInfer::poll_loop_::205-223` | `if (__atomic_load_n(&go_host_[slot], __ATOMIC_ACQUIRE) != 1u) continue;` (210) / `if (p.imm_fn) p.imm_fn(p.imm_args);   // enqueues into task_queue_` (214) / `if (!any) { for (int k = 0; k < 32; ++k) __builtin_ia32_pause(); }` (221) ← **전용 busy-poll 호스트 스레드** |
| 3m | `ext_bindings.cpp::MOEBindings<T>::ForwardBindings::inner::240-244` | `args_->cpuinfer->enqueue(&TP_MOE<T>::forward_binding, args_->moe, args_->qlen, args_->k, args_->expert_ids, args_->weights, args_->input, args_->output, args_->incremental);` |
| 3n | `ext_bindings.cpp::…::cpuinfer_interface::245-250` | `Args* args = new Args{nullptr, moe.get(), qlen, k, expert_ids, weights, input, output, incremental};` ← **호출마다 `new` (누수/할당 비용, CF 경로에선 캡처 시 1회)** |
| 3o | `cpuinfer.h::CPUInfer::enqueue::79-82` | `task_queue_->enqueue([=]() { std::invoke(f, *obj, args...); });` |
| 3p | (비-CF 경로) `cpuinfer.h::submit_with_cuda_stream::91-100` | `cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);` |
| 3q | (동기 경로) `cpuinfer.h::submit::84-89` | `*((CPUInfer**)args) = this; func(args);` |

## NUMA 서브풀 (2 pools)

| # | file::function::line | 인용 |
|---|---|---|
| 3r | `.../experts_base.py::_MoEBase._get_cpu_infer::334-352` | `worker_config.subpool_count = threadpool_count` (349) / `subpool_thread_count = [cpuinfer_threads // threadpool_count + (1 if i < cpuinfer_threads % threadpool_count else 0) ...]` (344-347) → 96 threads / 2 = **48+48** |
| 3s | `operators/moe-tp.hpp::TP_MOE_Common::forward::217-219` | `pool->dispense_backend()->do_numa_job([this, pool, qlen, k, expert_ids, input, weights](int numa_id) { tps[numa_id]->forward(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id]); });` |
| 3t | `cpu_backend/worker_pool.cpp::NumaJobDistributor::do_numa_job::348 / 369` , `WorkerPool::get_subpool::478`, `dispense_backend::480` | `InNumaPool* WorkerPool::get_subpool(int numa_id) { return numa_worker_pools[numa_id].get(); }` |
| 3u | `moe-tp.hpp::TP_MOE_Common 생성자::130-135` | `tp_config.intermediate_size /= tp_count;` ← NUMA 별로 intermediate 를 반씩 분할 (2560 → 1280) |

## 사용 가능한 job 식별자

| 식별자 | 어디서 | 상태 |
|---|---|---|
| `slot` (packet index) | `arm_packet` 반환값, Python `self._cf_last_slot` / `self._cf_eager_slot` | ✅ 유일하게 안정적. 캡처된 graph(bs) × layer 마다 고유. 62 layer × 12 bs ≈ 744 슬롯, `kMaxPackets=16384` |
| `layer_idx` | Python `self.layer_idx`; C++ 은 `MOEConfig.layer_idx` (`utils/amx.py::load_weights::525` `moe_config.layer_idx = self.layer_idx`) | ⚠️ `FwdArgsView` (cpuinfer.h:138) 에는 없음. `Packet.imm_args→moe` 포인터로만 간접 도달 |
| `slot`(ring buffer) | `layer_idx % 2` | ✅ |
| epoch / sequence / job id | — | ❌ **존재하지 않음 (미확인 아님, 부재 확인)**. `grep -rn 'epoch\|seq_id\|job_id' cpu_backend/ experts_base.py` → 0건 |
| 통계 카운터 | `cpuinfer.h::140` `std::atomic<long> n_def_skipped_{0}, n_def_run_{0};` | ✅ 있으나 출력 경로 없음 |

---

# 4. CPU 연산 단계 타이머 (KT_PHASE_PROF)

**매크로**: `operators/amx/moe_base.hpp::1` — `#define FORWARD_TIME_PROFILE 1` (**항상 켜짐**, 같은 파일 24행의 주석처리된 정의와 혼동 주의), 파일 끝 `::871` `#undef FORWARD_TIME_PROFILE`. `.so` 문자열로 컴파일 포함 확인함.

## 4-a. Prefill 경로 (`qlen > 1`) — `operators/amx/moe_base.hpp::AMX_MOE_BASE::forward_prefill::208`

| 타이머 | 측정 라인 | 실제로 덮는 구간 |
|---|---|---|
| `prepare` | `::304-310` `prepare_time = duration_cast<microseconds>(now_time - last).count();` | 219-302: expert별 토큰 히스토그램(`m_local_pos_`, `m_local_num_`) + **BufferA/BufferC 스크래치 풀 포인터 재배치** (`set_data`, `align64`). GEMM 전 메타 준비 |
| `cpy_input` | `::342-348` | 312-340: `memcpy(m_local_input_ptr_[expert]+pos*hidden, (ggml_bf16_t*)input + i*hidden, sizeof(ggml_bf16_t)*hidden)` ← **DtoH 가 아니라 "CPU RAM 내부 gather"**. pinned buffer → expert별 연속 배열로 토큰 복사 (bf16, hidden=6144 → 12 KiB/토큰/expert). `KT_FUSE_QIN` 이면 여기서 양자화까지 융합(323-330) |
| `q_input` | `::366-386` | 350-364: `gate_up_ba_[e]->from_mat(...)` = **입력 활성 int8/int4 양자화** (`KT_QA_PAR` 시 블록 병렬) |
| `up_gate` | `::406-412` | 388-404: `do_gate_up_gemm` ×2 (gate/up) + `to_mat` |
| `act` | `::416-422` | 414: `apply_activation(activated_expert, nth, qlen)` (SiLU·mul) |
| `q_down` | `::440-446` | 424-438: `down_ba_[e]->from_mat/from_mat_block` = down GEMM 입력 양자화 |
| `down` | `::459-465` | 448-457: `do_down_gemm` + `to_mat` |
| **`weight`** | `::492-497` | 467-490: **top-k 가중합 리덕션**. `__m512 weight = _mm512_set1_ps(weights[i*k+j]);` … `x0 = _mm512_fmadd_ps(down_output0, weight, x0);` → `output` (fp32) 에 누적. **가중치 로딩이 아니라 "expert 출력 weighted-sum"** |
| `total` | `::498-499` | `forward_total_time = ... (end_time - start_time)` |

**출력 라인** `moe_base.hpp::forward_prefill::500-506`:
```cpp
static bool _pp_on = std::getenv("KT_PHASE_PROF") != nullptr; static thread_local unsigned _pp_cnt = 0;
if (_pp_on && ((++_pp_cnt) & 63) == 0) printf(
    "Profiling Results (numa[%d]): activated_expert: %d, prepare: %ld us, cpy_input: %ld us, q_input: %ld us, "
    "up_gate: %ld us, act: %ld us, q_down: %ld us, down: %ld us, weight: %ld us, total: %ld us, max_local_num: %d, qlen: %d\n",
```
- 단위: **마이크로초 (`std::chrono::microseconds`, `high_resolution_clock`)**
- 샘플링: **`& 63` → 64회에 1회**, 카운터는 `thread_local` (NUMA worker 스레드별로 독립)
- `numa[%d]` = `tp_part_idx` (0 또는 1)

## 4-b. Decode 경로 (`qlen == 1`) — `moe_base.hpp::AMX_MOE_BASE::forward_decode::518`

디스패치: `moe_base.hpp::AMX_MOE_BASE::forward::190-196`
```cpp
if (qlen > 1) { forward_prefill(qlen, k, expert_ids, weights, input, output); }
else { forward_decode(k, expert_ids, weights, input, output); }
```

| 타이머 | 라인 |
|---|---|
| `q_input` | `::604-610` (prepare/cpy_input **없음** — 521-524 에서 아예 변수 선언 자체가 빠짐) |
| `up_gate` | `::630-636` |
| `act` | `::640-646` |
| `q_down` | `::656-662` |
| `down` | `::675-681` |
| `weight` | `::703-708` (683-701 의 AVX-512 fmadd 가중합) |
| **출력** | `::711-715` `printf("Profiling Results (numa[%d]): activated_expert: %d, q_input: %ld us, up_gate: ... total: %ld us\n", ...)` |

> 🔴 **주의: decode 쪽 printf(711) 는 `KT_PHASE_PROF` 게이트도, `& 63` 샘플링도 없습니다.** `FORWARD_TIME_PROFILE` 가 moe_base.hpp:1 에서 무조건 정의돼 있으므로, **bs=1 디코드에서는 층마다 NUMA 2개씩 매 호출 stdout 에 찍힙니다** (62층 × 2 = 124 줄/토큰). 타이밍 프로브 추가 전에 이 부분의 오버헤드/로그 오염을 먼저 확인하는 것을 권합니다. `.so` 문자열에 해당 포맷(“…activated_expert: %d, q_input: …”, prepare 없는 버전) 존재 확인함.

## 4-c. 래퍼 레벨 (`[kt-wrap]`) — `operators/moe-tp.hpp::TP_MOE_Common::forward::204-227`

```cpp
static bool _wp_on = std::getenv("KT_PHASE_PROF") != nullptr; static thread_local unsigned _wp_cnt = 0;   // 215
auto _wp_t0 = std::chrono::steady_clock::now();                                                          // 216
pool->dispense_backend()->do_numa_job([...](int numa_id) { tps[numa_id]->forward(...); });                // 217-219
auto _wp_t1 = std::chrono::steady_clock::now();                                                          // 220
merge_results(qlen, output, incremental);                                                                // 221
if (_wp_on && ((++_wp_cnt) & 63) == 0) {                                                                 // 222
  printf("[kt-wrap] qlen: %d, numa_job: %ld us, merge: %ld us, incremental: %d\n", ...);                 // 224
```
- `numa_job` = 2개 NUMA 서브풀 fork-join 전체, `merge` = NUMA 결과 합산 + (incremental 시) 이전 층 deferred 가산.
- 이 블록은 `#ifdef` 밖 → **항상 컴파일됨**, `KT_PHASE_PROF` 로만 게이트, 64회 1샘플. `steady_clock` / µs.

---

# 5. 완료 신호 & 결과 HtoD

| # | file::function::line | 인용 |
|---|---|---|
| 5a (CPU→flag) | `cpu_backend/cpuinfer.h::CPUInfer::poll_loop_::215-218` | `volatile unsigned int* d = done_host_ + slot;` / `task_queue_->enqueue([d]() { __atomic_store_n(d, 1u, __ATOMIC_RELEASE); });` ← **immediate task 뒤·deferred task 앞에 "done setter" 를 큐에 삽입** (= `sync(allow_pending=1)` 과 동일 의미) |
| 5b | `cpuinfer.h::CPUInfer::poll_loop_::216-219` | `bool run_def = (p.def_fn != nullptr);` / `if (run_def && skip_empty_def_on_() && def_is_empty_(p)) { run_def = false; n_def_skipped_.fetch_add(1, ...); }` / `if (run_def) { p.def_fn(p.def_args); n_def_run_.fetch_add(1, ...); }` |
| 5c (flag 메모리) | `cpuinfer.h::ensure_cf_::160-164` | `cudaHostAlloc((void**)&done_host_, kMaxPackets*sizeof(unsigned int), cudaHostAllocMapped\|cudaHostAllocPortable)` / `cudaHostGetDevicePointer((void**)&done_dev_, (void*)done_host_, 0);` ← **mapped pinned (managed 아님)** |
| 5d (GPU 대기) | `cpuinfer.h::CPUInfer::wait_done_on_stream::199-204` | `int r1 = cu_wait_val32_((void*)user_cuda_stream, (unsigned long long)(uintptr_t)(done_dev_ + slot), 1u, 1u);` / `int r2 = cu_write_val32_(..., done_dev_ + slot, 0u, 0u);` ← **`cuStreamWaitValue32(GEQ 1)` + 리셋 write. busy-poll 커널 아님, SM 점유 0** |
| 5e (드라이버 심볼) | `cpuinfer.h::ensure_driver_fns_::288-296` | `cu_wait_val32_ = (cuWaitVal32_t)dlsym(h, "cuStreamWaitValue32_v2");` … `cu_write_val32_ = (cuWriteVal32_t)dlsym(h, "cuStreamWriteValue32_v2");` |
| 5f (Python 진입) | `.../experts_base.py::BaseMoEWrapper.sync_forward::953` | |
| 5g | `…::996-997` | `if _os.environ.get("KT_CALLBACK_FREE") and getattr(self, "_cf_last_slot", None) is not None:` / `self._active_cpu_infer().wait_done_on_stream(cuda_stream, self._cf_last_slot)` |
| 5h (대안경로) | `…::998-1004` | `elif _os.environ.get("SGL_INTERLEAVE"): … _ci.wait_signal_on_stream(cuda_stream, _slot)` / `else: self._active_cpu_infer().sync_with_cuda_stream(cuda_stream, allow_pending)` ← 후자는 `cudaLaunchHostFunc` (`cpuinfer.h::118-125`) |
| 5i (**HtoD**) | `…::copy_forward_output_to_device::937-951` | `current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth` (949) / `output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=True)` (950) / `return output_gpu[current_slot]` (951) |
| 5j | `…::sync_forward::1006` | `return self.copy_forward_output_to_device(hidden_states)` |

**스트림**: 5d 의 memop 과 5i 의 HtoD 는 **같은 current stream** (디코드에선 캡처된 그래프 스트림). 별도 stream/event 없음. 디바이스 동기화 0회.

⚠️ `sync_forward::994-996` 의 들여쓰기가 특이함 — `import os as _os` (995) 는 `else:` 블록 안, `if _os.environ.get(...)` (996) 는 그 블록 **밖**. NPU 경로(987-993)를 타면 `_os` 가 미정의라 `NameError` 가 납니다(CUDA 에선 무해). 프로브 삽입 시 이 들여쓰기를 건드리지 않도록 주의.

---

# 6. Hot expert GPU 연산 + 결합 + all-reduce

| # | file::function::line | 인용 |
|---|---|---|
| 6a | `.../kt_ep_wrapper.py::KTEPWrapperMethod.apply::394-395` | `if self.tp_rank == 0:` / `self.submit(layer, dispatch_output)` ← **CPU submit 이 GPU 큐잉보다 먼저** |
| 6b | `…::apply::418` | `gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)` |
| 6c | `…::apply::421-429` | `output = gpu_combine_input.hidden_states` / `if self.tp_rank == 0:` / `cpu_output = self.sync(x)` / `output = output + cpu_output` ← **hot + cold 합산 지점 (rank0 에서만; rank1-3 은 GPU 출력만)** |
| 6d | `…::apply::397-403` | `if self.num_gpu_experts == 0:` fast path (현 설정 96 이므로 미사용) |
| 6e | `/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py::FusedMoE.run_moe_core::1530-1535` | `return self.quant_method.apply(layer=self, dispatch_output=dispatch_output,)` |
| 6f | `/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8.py::Fp8MoEMethod.apply::2607-2614` | `elif self.runner.runner_backend.is_triton(): quant_info = self.get_triton_quant_info(layer)` / `return self.runner.run(dispatch_output, quant_info)` |
| 6g | **fused_moe 커널 호출부** `/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton.py::fused_experts_none_to_triton::221-245` | `from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (fused_experts,)` / `output = fused_experts(hidden_states=dispatch_output.hidden_states, w1=quant_info.w13_weight, w2=quant_info.w2_weight, topk_output=dispatch_output.topk_output, moe_runner_config=runner_config, ... use_fp8_w8a8=quant_info.use_fp8_w8a8, ...)` |
| 6h | `/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/runner.py::MoeRunner.run::143-147` | `if self.fused_func is not None and not self.lora_enabled: return self.fused_func(dispatch_output, quant_info, self.config)` |
| 6i | `…/fused_moe_triton/layer.py::FusedMoE.forward_impl::1501-1509` | `final_hidden_states = self.dispatcher.combine(combine_input=combine_input)` / `if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1): final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)` — `reduce_results` 기본값 `False` (`layer.py::245`), Qwen3 경로에선 미사용 |
| 6j | **TP all-reduce** `/sgl-workspace/sglang/python/sglang/srt/models/qwen3_moe.py::Qwen3MoeSparseMoeBlock.forward_normal::343-348` | `if self.tp_size > 1 and not should_skip_post_experts_all_reduce(is_tp_path=True):` / `final_hidden_states = moe_tensor_model_parallel_all_reduce(final_hidden_states)` ← **rank0 의 cold 결과가 여기서 4 rank 에 퍼짐** |
| 6k | residual add | `.../qwen3_moe.py::Qwen3MoeDecoderLayer.forward::853-860` | `hidden_states = self.mlp(hidden_states, forward_batch)` / `hidden_states, residual = self.layer_communicator.postprocess_layer(hidden_states, residual, forward_batch)` |

**shared expert**: Qwen3-Coder-480B 는 shared expert 없음 (`Qwen3MoeSparseMoeBlock.__init__::252-286` 에 shared_experts 생성 없음). 따라서 결합은 **hot(GPU) + cold(CPU) 두 항 뿐**.

---

# 7. CUDA Graph 캡처/재생

| # | file::function::line | 인용 |
|---|---|---|
| 7a | `/sgl-workspace/sglang/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py::DecodeCudaGraphRunner.capture_prepare::330-337` | `KTMoEWrapper.set_capture_batch_sizes(sorted({b * self.captured_req_width for b in self.capture_bs}))` ← **로컬 패치**: flat token 수 기준으로 등록 |
| 7b | `…::capture_one_shape::1155-1201` | `def run_once(): … out = forward(forward_batch.input_ids, forward_batch.positions, forward_batch, **kwargs)` |
| 7c | `…::capture_one_shape::1231-1236` | `self.backend.capture_one(shape_key, run_once, capture_inputs=None, post_warmup_hook=post_warmup_hook,)` |
| 7d | `/sgl-workspace/sglang/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py::FullCudaGraphBackend.capture_one::112-119` | `for _ in range(2): self._device_module.synchronize(); self._tp_group.barrier(); forward_fn()` ← **워밍업 2회 (eager)** |
| 7e | `…::capture_one::135-136` | `with graph_ctx(cuda_graph=graph, pool=self._pool, stream=self._capture_stream):` / `out = forward_fn()` |
| 7f | **재생** `…::FullCudaGraphBackend.replay::151-158` | `self._graphs[shape_key].replay()` / `return self._outputs[shape_key]` |
| 7g | `…/runner/decode_cuda_graph_runner.py::DecodeCudaGraphRunner.execute::1402-1421` | `with timer_ctx, self.backend.replay_session():` / `self.load_batch(forward_batch, pp_proxy_tensors)` / `output = self.backend.replay(self._replay_graph_key, forward_batch)` |
| 7h | `/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py::ModelRunner.forward_decode(주변)::1874-1890` | `ret = self.decode_cuda_graph_runner.execute(forward_batch, pp_proxy_tensors=pp_proxy_tensors,)` ← 로컬 패치로 `_interleaved_execute` 분기가 앞에 붙음(1877-1886), 단 `SGL_INTERLEAVE` 미설정이면 우회 |

## 그래프 안에 들어간 것 vs 호스트에서 매 재생마다 도는 것

**그래프에 캡처되는 GPU 노드 (재생마다 실행, Python 개입 0):**
1. DtoH 4건 (`experts_base.py::650-654`)
2. `go_dev_[slot] = 1` — `cuStreamWriteValue32` (`cpuinfer.h::195`)
3. hot expert triton fused_moe 커널 (`triton.py::234`)
4. `wait(done_dev_[slot] >= 1)` + `done_dev_[slot] = 0` — `cuStreamWaitValue32`/`WriteValue32` (`cpuinfer.h::201-202`)
5. HtoD `output_gpu ← output_cpu` (`experts_base.py::950`)
6. `output = output + cpu_output` (`kt_ep_wrapper.py::429`), all-reduce (`qwen3_moe.py::346`)

**캡처 시점에만 1회 실행되고 재생 시엔 돌지 않는 Python:**
- `select_deferred_experts`(543)의 텐서 연산은 **캡처됨**(GPU op), 하지만 `.item()` 없는 브랜치만 사용됨
- `self.moe.forward_task(...)` → `new Args` (ext_bindings.cpp:248) — **캡처 시 1회**
- `arm_packet` (cpuinfer.h:172) — **캡처 시 1회, 슬롯 영구 보존**
- `KExpertsCPUBuffer.get_buffer` 딕셔너리 조회

**재생마다 도는 호스트 Python (타임스탬프 찍기 좋은 지점):**

| 위치 | 재생당 호출 횟수 |
|---|---|
| `decode_cuda_graph_runner.py::execute::1391` | 1 |
| `decode_cuda_graph_runner.py::load_batch::1245` (1403 에서 호출) | 1 |
| `full_cuda_graph_backend.py::replay::157` (`graph.replay()`) | 1 |
| **층별 Python** | **0 — 존재하지 않음** |

> 📌 따라서 **층별 GPU-side 타임라인은 호스트 Python 에서 얻을 수 없습니다.** 층 단위 타임스탬프는 (a) `poll_loop_` 에서 `go_host_[slot]` 관측 시각, (b) `task_queue.cpp::worker` 의 `tq_before_/tq_after_`, (c) done-setter 람다(cpuinfer.h:218) 실행 시각 — 이 세 CPU-side 지점에서만 가능합니다. GPU-side 는 그래프에 추가 `cuStreamWriteValue32` 노드를 심어 mapped-pinned 카운터를 쓰고, `poll_loop_` 스레드가 그 변화를 관측한 호스트 시각으로 대용해야 합니다 (디바이스 sync 불필요).

**ensure_cf_ 의 캡처 제약**: `cudaHostAlloc` 은 캡처 중 호출 불가(err 900). `arm_packet::173` 의 `ensure_cf_()` 는 `go_host_ != nullptr` 이면 즉시 return(156) 하므로, **7d 의 워밍업 2회 (eager)** 에서 할당이 먼저 끝나야 합니다. 같은 이유로 `model_runner.py::_il_register_ladder_hooks` 도 `assert not torch.cuda.is_current_stream_capturing()` 후 더미 memop 으로 선할당합니다.

---

# 8. Deferred experts — producer/consumer

**결론: 층 L → 층 L+1 (같은 decode step 내부, 다음 층). step t → t+1 이 아님.**

| # | file::function::line | 인용 |
|---|---|---|
| 8a (상태 dict) | `.../experts_base.py::BaseMoEWrapper::424` | `_layer_has_pending_deferred: Dict[int, bool] = {}` (클래스 변수, 전 층 공유) |
| 8b (init) | `…::__init__::500` | `BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False` |
| 8c (**consumer**) | `…::submit_forward::851` | `incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)` |
| 8d | `…::submit_forward::856-864` | `immediate_task = … self.moe.forward_task(..., output_cpu[current_slot].data_ptr(), incremental,)` |
| 8e (**producer**) | `…::submit_forward::903-911` | `deferred_task = self.moe.forward_task(..., deferred_experts_ids_cpu[current_slot].data_ptr(), ..., output_cpu[next_slot].data_ptr(), False,)` ← **다음 층의 current_slot 에 기록** |
| 8f | `…::submit_forward::899, 918` | `BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False` … `= True` |
| 8g (**실제 가산**) | `operators/amx/moe_base.hpp::TP_MOE<AMX_MOE_BASE<...>>::merge_results::829-838` | `if (incremental) { for (int e = 0; e < config.hidden_size; e += 32) { avx512_32xbf16_to_32xfp32((__m512i*)((ggml_bf16_t*)output + token_nth*config.hidden_size + e), &x0, &x1); *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), x0); ... } }` ← **출력 버퍼에 이미 들어있던 이전 층 deferred 결과를 읽어 누적** |
| 8h | `moe-tp.hpp::TP_MOE_Common::forward::221` | `merge_results(qlen, output, incremental);` |
| 8i (링 슬롯) | `…::_prepare_forward_cpu_buffers::628-629` | `current_slot = self.layer_idx % 2`, `next_slot = (current_slot+1) % 2` → 층 L 의 next_slot == 층 L+1 의 current_slot |
| 8j (마지막 층 차단) | `.../kt_ep_wrapper.py::KTEPWrapperMethod.create_weights::231-237` | `if (... and self.kt_config.layer_idx == self.kt_config.num_layers - 1): layer_max_deferred = 0` |
| 8k (skip-empty-def, C++) | `cpu_backend/cpuinfer.h::CPUInfer::def_is_empty_::142-148` | `for (int i = 0; i < qlen * a->k; ++i) { int64_t e = ids[i]; if (e >= 0 && e < p.n_experts && !p.gpu_mask[e]) return false; }` / `memset((void*)a->output, 0, (size_t)qlen * p.out_row_bytes);` ← cold 가 하나도 없으면 **출력 0 채우고 skip** (`KT_CF_SKIP_EMPTY_DEF`) |

**deferred=8 설정에서의 실제 동작 체인** (층 L, current_slot = L%2):
1. `protected_k=0` → `immediate_ids` 전부 -1, `deferred_ids` = topk 전량
2. `incremental = True` (L≥1) + `KT_CF_SKIP_EMPTY_IMM` → `immediate_task = (0,0)` → `Packet.imm_fn = nullptr` → poller 가 건너뜀. **`output_cpu[L%2]` 는 덮어써지지 않고, 층 L-1 이 써 둔 deferred 결과를 그대로 보존**
3. `deferred_task` 는 `output_cpu[(L+1)%2]` 에 `incremental=False` 로 기록
4. `sync_forward` (949-950) 가 `output_cpu[L%2]` 를 GPU 로 복사 → **층 L 이 받는 cold 기여는 층 L-1 의 hidden state 로 계산된 것** (1-layer stale)
5. 층 0: `incremental=False` → immediate task 실행, expert 전부 -1 → `output_cpu[0]` 에 0 기록

**epoch / sequence 카운터: 존재하지 않음** (§3 표 참조). 정합성은 순전히 "Python 이 층 순서대로 submit → 단일 `task_queue_` FIFO" 에 의존합니다 (`cpu_backend/task_queue.h::22-49` 의 MPSC 큐 + 단일 `workerThread`).

---

# 9. 기존 계측 env var 전수

| env var | file::function::line | 출력 내용 | 샘플링 |
|---|---|---|---|
| **`KT_TQ_TIMING`** | `cpu_backend/task_queue.cpp::23` `static bool tq_on = std::getenv("KT_TQ_TIMING") != nullptr;` | `[kt-tq] n=%zu exec_us p50/p90/p99/mean \| hist(<5,<30,<100,<300,<600,<1k,<2k,<4k,>=4k) \| wait_us p50/p90/p99/mean \| busy=%.2f` | `::36` `if (tq_exec.size() >= 2048)` → **2048 태스크마다 1회 flush** (stderr) |
| ↳ 측정 지점 | `task_queue.cpp::TaskQueue::worker::106-108` | `if (tq_on) tq_before_(); next->task(); if (tq_on) tq_after_();` | |
| ↳ before/after | `task_queue.cpp::tq_before_::27-30`, `tq_after_::32-48` | `tq_wait.push_back(... (tq_t0 - tq_prev_end) ...)` / `unsigned us = ... (t1 - tq_t0) ...` | `steady_clock`, **µs**, `thread_local` |
| **`KT_PHASE_PROF`** (prefill) | `operators/amx/moe_base.hpp::forward_prefill::500-506` | `Profiling Results (numa[%d]): activated_expert, prepare, cpy_input, q_input, up_gate, act, q_down, down, weight, total (us), max_local_num, qlen` | `((++_pp_cnt) & 63) == 0` → **64회 1회**, `thread_local`, stdout |
| **(게이트 없음)** (decode) | `moe_base.hpp::forward_decode::711-715` | `Profiling Results (numa[%d]): activated_expert, q_input, up_gate, act, q_down, down, weight, total (us)` | 🔴 **없음 — 매 호출 출력** |
| **`KT_PHASE_PROF`** (wrapper) | `operators/moe-tp.hpp::TP_MOE_Common::forward::215, 222-227` | `[kt-wrap] qlen: %d, numa_job: %ld us, merge: %ld us, incremental: %d` | `((++_wp_cnt) & 63) == 0` → **64회 1회**, stdout |
| **`KT_AVX_RB`** | `operators/amx/la/amx_kernels.hpp::GemmKernel224Int4::avx_rb_on::1768-1771` | — (성능 스위치) | `std::getenv(...) != nullptr` → **`KT_AVX_RB=0` 도 ON** (사용자 지적대로) |
| ↳ 사용처 | `amx_kernels.hpp::integer_mat_mul::2638-2652` | `if constexpr (!amx_or_avx && std::is_same_v<K, GemmKernel224Int4>) { if (K::avx_rb_on()) { ... K::avx_kernel_rb(...) ... return; } }` — **AVX(vec_mul) 경로에서만** |
| ↳ AMX/AVX 선택 | `operators/amx/moe.hpp::do_gate_up_gemm::253` / `do_down_gemm::266` | `if (qlen > amx_min_qlen_() \|\| m >= amx_min_rows_()) { amx::mat_mul(...) } else { amx::vec_mul(...) }` |
| `KT_AMX_MIN_QLEN` | `moe.hpp::amx_min_qlen_::243-246` | 기본값 `4 * expert_num / num_experts_per_tok` = `4*160/8 = 80` | |
| `KT_AMX_MIN_ROWS` | `moe.hpp::amx_min_rows_::239-242` | 기본 `1<<30` (사실상 비활성) | |
| `KT_AVX_PF` | `amx_kernels.hpp::avx_pf_dist::1772-1775` | prefetch 거리, 기본 0 | |
| `KT_AMX_BCACHE` | `amx_kernels.hpp::1893` | B 행렬 캐시 | |
| `KT_AMX_A_LOADD` | `amx_kernels.hpp::1897` | | |
| `KT_HUGEPAGE` | `moe_base.hpp::kt_hp_alloc::15-22` | `madvise(p, rsz, MADV_HUGEPAGE)` | |
| `KT_FUSE_QIN` | `moe_base.hpp::kt_fuse_qin_::510-513`, 사용 `::322-330` | gather+quant 융합 | |
| `KT_QA_PAR` | `moe_base.hpp::kt_qa_par_::514-517`, 사용 `::353-359`, `::424-430` | 양자화 블록 병렬 | |
| `KT_DUMP_A` | `moe_base.hpp::368-379` | `/tmp/dumpA_q%d_%d_%s.bin` 4회 덤프 | `dump_ct < 4 && qlen>=10 && tp_part_idx==0` |
| `KT_CF_SKIP_EMPTY_DEF` | `cpuinfer.h::skip_empty_def_on_::139`, 사용 `::217` | deferred 전부 hot 이면 skip | 카운터 `n_def_skipped_/n_def_run_` (140) — **출력 경로 없음** |
| **`KT_CALLBACK_FREE`** | `experts_base.py::submit_forward::853, 880`, `sync_forward::996` | §3, §5 참조 | Python only |
| **`KT_CF_SKIP_EMPTY_IMM`** | `experts_base.py::submit_forward::853` | §3d | Python only |
| `KT_COLD_DEFER` / `KT_COLD_TAU` | `experts_base.py::select_deferred_experts::553, 555` | 점수 τ 기반 defer | 현 설정 미사용 |
| `KT_DUMP_TOPK` | `experts_base.py::_prepare_forward_cpu_buffers::634-642` | `/tmp/topk_dump.pt` | `_n` 회 |
| `KT_FORCE_SYNC_SUBMIT` | `experts_base.py::_should_bypass_stream_callback::75` | 동기 fallback | |
| `KT_NAN_PROBE` / `KT_NAN_PROBE_EVERY` | `kt_ep_wrapper.py::38-39`, `_kt_probe_accum::42-65`, 호출 `::424-428` | `KT_NAN_PROBE layer=%s calls=%d cpu_expert_nan=%d gpu_expert_nan=%d` | 기본 **4000회 1회** (`.item()` 로 D2H sync 발생 → 프로파일링 시 끄세요) |
| `KT_DEBUG_NEGIDS` | `kt_ep_wrapper.py::submit::341-345` | `[KT-NEGIDS] layer=... neg_count=...` | 1회 |
| `SGL_DEBUG_REMAP` | `topk.py::select_experts::2102-2109`, `_post_process_topk_ids::1971-1981` | `[SEL-DEBUG]`, `[REMAP-DEBUG]` | 8회 / 1회 |
| `SGL_INTERLEAVE`, `SGL_IL_LADDER`, `SGL_IL_SEQ`, `SGL_IL_HALF`, `SGL_IL_PRIVATE_POOL`, `SGL_INTERLEAVE_MIN_BS` | `model_runner.py::init_cuda_graphs::994-1027`, `_il_register_ladder_hooks::1400-1470`, `_interleaved_execute::1472-1560`; `full_cuda_graph_backend.py::capture_session::73-80` | 듀얼 마이크로배치 | 현 목표 설정에선 미사용 |
| `KT_MOE_PHASE_TIMING` / `_INTERVAL` | `operators/llamafile/moe.hpp::43, 51` | llamafile 백엔드 전용 | **AMXINT4 경로 아님 → 무효** |
| `KT_SFT_PROFILE` | `operators/sft_profile.hpp::168` | SFT 전용 | 무효 |

---

# 10. C++ task queue 의 job 식별자 & 무-동기 타임스탬프 부착 지점

## 현존 식별자

| 구조체/필드 | file::line | 비고 |
|---|---|---|
| `CPUInfer::Packet` | `cpuinfer.h::132-137` | `imm_fn/imm_args/def_fn/def_args/gpu_mask/n_experts/out_row_bytes/armed` — **slot index 가 배열 첨자** |
| `CPUInfer::FwdArgsView` | `cpuinfer.h::138` | `{ void* ci; void* moe; intptr_t qlen_ptr; int k; intptr_t expert_ids; intptr_t weights; intptr_t input; intptr_t output; bool incremental; }` — `def_is_empty_` 가 쓰는 뷰. **layer_idx 없음**, `moe` 포인터로만 층 구분 가능 |
| `MOEBindings<T>::ForwardBindings::Args` | `ext_bindings.cpp::229-239` | 위와 동일 레이아웃 (첫 필드 `CPUInfer* cpuinfer` = `void* ci`) |
| `TaskQueue::Node` | `task_queue.h::32-37` | `{ std::function<void()> task; std::atomic<Node*> next; }` — **id 필드 없음** |
| `CPUInfer::SignalArgs` | `cpuinfer.h::255-258` | `{ CPUInfer* ci; int slot; }` — SGL_INTERLEAVE 전용 |
| `MOEConfig::layer_idx` | `utils/amx.py::load_weights::410 / 525 / 864` | `moe_config.layer_idx = self.layer_idx` — C++ `config.layer_idx` 로 존재 |

**job id / epoch / sequence number: 없음.** slot 이 사실상 유일한 안정 키이며, `(layer, capture_bs)` 조합당 1개로 고정 발급됩니다 (`arm_packet::174` `n_packets_.fetch_add(1)`).

## 디바이스 동기화 없이 타임스탬프를 붙일 수 있는 지점 (읽기 전용 제안)

| 지점 | file::line | 왜 안전한가 |
|---|---|---|
| **A. go 관측 시각** | `cpuinfer.h::poll_loop_::210-211` — `if (__atomic_load_n(&go_host_[slot], ...) != 1u) continue;` 직후 | 이미 전용 busy-poll 스레드. `steady_clock::now()` 를 `t_go_[slot]` 배열에 저장. GPU 개입 0 |
| **B. imm/def 큐잉 시각** | `cpuinfer.h::poll_loop_::214, 219` | `p.imm_fn()` / `p.def_fn()` 는 enqueue 만 함 (실행 아님) → 큐 대기 시간 분리 측정 가능 |
| **C. done 기록 시각** | `cpuinfer.h::poll_loop_::218` — `task_queue_->enqueue([d]() { __atomic_store_n(d, 1u, __ATOMIC_RELEASE); });` | 람다에 `slot` 을 캡처시켜 `t_done_[slot] = now()` 추가. worker 스레드에서 실행되므로 정확히 "CPU MoE 완료" 시각 |
| **D. 태스크 exec/wait** | `task_queue.cpp::worker::106-108` (`tq_before_`/`tq_after_`) | **이미 구현됨** (`KT_TQ_TIMING`). slot 을 알려면 `Node` 에 필드 추가 또는 `std::function` 대신 태그 구조체 필요 |
| **E. 단계별 µs** | `moe_base.hpp::500-506` / `moe-tp.hpp::222-227` | **이미 구현됨** (`KT_PHASE_PROF`). 다만 64-샘플 stride 이고 slot/layer 표기가 없음 → `tp_part_idx` 만 있음. `config_.layer_idx` 를 포맷에 추가하면 층 귀속 가능 |
| **F. GPU 타임라인 (동기 없이)** | `cpuinfer.h::go_on_stream::195` 패턴 복제 | 별도 mapped-pinned `u32 tick_[N]` 을 만들고, 그래프 캡처 시 관심 지점마다 `cu_write_val32_(stream, tick_dev_+i, seq, 0)` 노드를 추가. 호스트 poller 가 `tick_host_[i]` 변화를 관측한 시각을 기록 → **`cudaEventQuery`/`Synchronize` 불필요**. `wait_done_on_stream::201-202` 가 이미 wait+reset 두 memop 을 넣고 있으므로 그래프 노드 추가 자체는 검증된 패턴 |
| **G. Python 재생 경계** | `decode_cuda_graph_runner.py::execute::1391`, `full_cuda_graph_backend.py::replay::157` | 재생당 1회. 층 해상도는 없음 (§7 참조) |

**피해야 할 것**: `experts_base.py::730-731 / 788-789` 의 `(deferred_experts_ids_cpu[...] >= 0).any().item()` 과 `kt_ep_wrapper.py::58-59` 의 `.item()` 은 D2H 동기를 유발합니다. `submit_forward` (806) 경로에는 `.item()` 이 없으니 이 성질을 깨지 않도록 프로브를 넣어야 합니다.

---

# 11. Prefill 과의 차이 요약

| 항목 | Decode (CUDA graph, bs∈capture_bs) | Prefill (`--cuda-graph-backend-prefill disabled` → eager) |
|---|---|---|
| C++ 커널 | `moe_base.hpp::forward_decode::518` (qlen==1, bs=1일 때) / `forward_prefill::208` (bs>1) | 항상 `forward_prefill::208` |
| `prepare`/`cpy_input` 타이머 | decode 커널엔 **없음** (521-524) | 있음 (304-310, 342-348) |
| 로그 게이트 | decode printf 는 **게이트 없음** (711) | `KT_PHASE_PROF` + 64-stride (500-501) |
| 패킷 슬롯 | `arm_packet` — graph shape 마다 새 슬롯, 영구 (`experts_base.py::925-926`) | `rearm_packet` — 층당 `self._cf_eager_slot` 1개 재사용 (`::928-932`) |
| 버퍼 | `capture_buffers[(channel, rows)]` 캐시 히트 (`::243-244`) | `temp_buffer` — rows 가 바뀌면 **매번 pinned 재할당** (`::289-290`). chunk 8192 → 8192×6144×2 = 96 MiB pinned |
| Python 층별 실행 | 재생 시 **0회** | 층·chunk 마다 전부 실행 |
| AMX/AVX | qlen=1 → `vec_mul` (AVX, `KT_AVX_RB` 적용) | qlen>80 → `amx::mat_mul` (`moe.hpp::253`) |
| `_wait_device` | `torch.cuda.is_current_stream_capturing()` → 즉시 return (`::153-156`) | CUDA 경로에선 `sync_submit=False` 라 호출 안 됨 |
| qlen 가드 | bs ≤ 64 | `_check_qlen_fits_cpp_buffers::601` — `qlen > chunked_prefill_size(8192)` 면 예외 |

---

# 12. 미확인 / 부재 항목

1. **`KT_GPU_EXPERTS_PER_LAYER`** — 컨테이너 코드에 **없음**. 호스트 문서에만 존재 (IDE_070 패치 미적용 상태). `~/projects/vllm_hybrid/shadow_assists/features/IDE_071/OPTION_REGISTRY.md:72`.
2. **epoch / sequence / job id** — C++·Python 어디에도 **없음** (grep 0건). slot 이 유일한 키.
3. **`Packet` → layer_idx 매핑 테이블** — 없음. Python `self._cf_last_slot` 은 층 객체별로 최신 1개만 보존.
4. **`FwdArgsView` 의 layer 정보** — 없음 (`cpuinfer.h::138`).
5. **실행 중 서버 프로세스** — 현재 컨테이너에 살아있는 `launch_server` 없음 (defunct `[sglang::schedul]` 좀비만 존재) → 런타임 env 실측 불가, 설정값은 IDE_071/072 문서의 `launch_cmd` 기준입니다.
6. **shared expert 경로** — Qwen3-Coder-480B 는 shared expert 미보유 → 해당 결합 지점 없음.
