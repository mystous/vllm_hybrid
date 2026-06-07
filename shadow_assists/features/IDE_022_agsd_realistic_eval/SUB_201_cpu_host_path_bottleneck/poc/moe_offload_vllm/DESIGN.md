# DESIGN — vLLM MoE Expert CPU Offload (kt-kernel 통합)

> **상위**: SUB_201 / poc/moe_offload (SGLang 1단계 +18.4 % per-model, +136.8 % tps-per-GPU 결과)
> **목표**: 동일 lever 를 vLLM 위에서 재구현. 본 PoC = `feat/spec-decode-tuning` 브랜치 working tree 직접 수정.
> **작성 일자**: 2026-06-06 (작업 머신: B200 8 GPU + Xeon Platinum 8570 AMX)

## 1. vLLM FusedMoE 호출 chain (현황)

```
Qwen3MoeSparseMoeBlock.forward (qwen3_moe.py:224)
 → SharedFusedMoE.forward (shared_fused_moe.py: thin wrapper)
   → FusedMoE.forward (layer.py:1538)
     → DefaultMoERunner.forward (moe_runner_base.py:462 → forward_dispatch → _forward_impl)
       → _apply_quant_method (moe_runner_base.py:384)
         → quant_method.apply (UnquantizedFusedMoEMethod.apply, unquantized_fused_moe_method.py:253)
           → forward_cuda → forward_native → self.moe_kernel.apply (triton fused_experts)
```

핵심 지점:
- `topk_weights, topk_ids = router.select_experts(...)` (moe_runner_base.py:405) → 이 시점에 expert routing 결정 완료.
- 직후 `quant_method.apply(layer, x, topk_weights, topk_ids, ...)` 호출 — **본 hook 위치**.
- `expert_map`: -1=글로벌 expert 가 로컬에 없음, 0..local_num_experts-1=로컬 GPU index. EP 분할의 native 메커니즘.

## 2. SGLang 통합 패턴 (참조)

`sglang/srt/layers/moe/kt_ep_wrapper.py` 의 `KTEPWrapperMethod.apply` (라인 2489):

```
1. (cpu_stream) staging_buffer.copy_(x); wrapper.submit_forward(staging_buffer, topk_ids_full, topk_weights, cpu_stream)
2. (main_stream)   masked_topk_ids = mask_and_remap_expert_ids(topk_ids, gpu_mask, logical_to_gpu_idx)  # CPU expert -> -1, GPU expert -> 0..N-1
                   gpu_output = gpu_method.apply(layer, masked_dispatch_output)
3. (cpu_stream) cpu_output_on_gpu = wrapper.sync_forward(staging_buffer, cpu_stream)
4. (main_stream) wait_event(cpu_stream) → output = gpu_output + cpu_output
```

KT-kernel API (kt_kernel/experts_base.py:377/457):
- `submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream) -> None`
- `sync_forward(hidden_states, cuda_stream) -> output_gpu` (이미 GPU에 올라온 결과)
- `KTMoEWrapper(layer_idx, num_experts, num_experts_per_tok, hidden_size, moe_intermediate_size, gpu_experts_mask, cpuinfer_threads, threadpool_count, weight_path, chunked_prefill_size, method="AMXBF16", numa_nodes=...)`

## 3. vLLM 측 통합 결정 (단순화)

본 PoC 는 시간 예산 5-7일 내 7단계 e2e 도달이 목표. 따라서 **non-invasive monkey-patch** 방식으로 통합 (loader 수정 없음 → 다음 단계에서 합법적 hook으로 격상 가능).

### 3.1 진입점

`UnquantizedFusedMoEMethod` 의 인스턴스에 `_kt_wrapper` 속성 부여:
- env flag `VLLM_MOE_CPU_OFFLOAD=1` → 모델 load 후 each Qwen3MoeSparseMoeBlock 의 self.experts.quant_method 에 KTMoEWrapper 한 개 부착 (layer_idx 별).
- `forward_native`/`forward_cuda` 에서 `_kt_wrapper is not None` 이면 분기.

### 3.2 분기 로직 (apply 함수 내)

```python
def forward_cuda(self, layer, x, topk_weights, topk_ids, shared_experts_input):
    if getattr(self, "_kt_wrapper", None) is not None:
        # KT path
        return self._kt_forward(layer, x, topk_weights, topk_ids, shared_experts_input)
    return self.forward_native(layer, x, topk_weights, topk_ids, shared_experts_input)
```

`_kt_forward` 흐름:
1. staging_buffer = `x.clone()` (단순화. shared buffer 는 다음 iteration).
2. `cpu_stream.wait_stream(main); with cpu_stream: wrapper.submit_forward(staging_buffer, topk_ids, topk_weights, cpu_stream.cuda_stream)`.
3. main_stream 에서 `masked_topk_ids = where(gpu_mask[topk_ids], topk_ids_remapped, -1)` (CPU expert id → -1; GPU expert id → 0..num_gpu_experts-1).
4. main: `gpu_out = fused_experts(x, layer.w13_weight, layer.w2_weight, masked_topk_ids, topk_weights, ...)` — vLLM 의 기존 triton kernel. `expert_map` 효과를 이미 갖는 `-1` semantic 그대로.
5. cpu_stream: `cpu_out_on_gpu = wrapper.sync_forward(staging_buffer, cpu_stream.cuda_stream)`.
6. main: wait_event(cpu_done) → `return gpu_out + cpu_out_on_gpu`.

> **결정 1**: PoC 1차에서는 **`num_gpu_experts=N`** (e.g. 32) — 동일하게 처음 N 개의 expert id 가 GPU 로컬, 나머지는 CPU. 단순한 split. SGLang B 결과는 32/128 split.
> **결정 2**: 모든 weight 는 `weight_path = HF 모델 디렉토리` → kt-kernel 의 self-loader 가 disk → shared-mem mmap. vLLM 의 weight loader 도 모든 128 expert weight 를 GPU 에 로드. **이중 메모리**가 되지만, CPU expert weight 는 vLLM 가 사용 안 함 (mask 처리). 다음 단계에서 conditional load.
> **결정 3**: TP=1 만 지원 (1단계 결과와 동일). EP=1, DP=1.

### 3.3 모델 위치 — Qwen3MoE

Qwen3MoeSparseMoeBlock.experts = SharedFusedMoE(num_experts=128, top_k=8, hidden_size=2048, intermediate_size=768, ...). FusedMoE.quant_method = UnquantizedFusedMoEMethod (BF16).

## 4. 단계별 wire-up

| 단계 | 위치 | 작업 |
|---|---|---|
| 2 | `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` | env flag 분기 + `_kt_forward` 메서드 추가 |
| 3 | `vllm/model_executor/layers/fused_moe/kt_kernel_binding.py` (신규) | `KTMoEWrapper` import + lazy init wrapper + `make_qwen3_wrapper(layer_idx, ...)` 헬퍼 |
| 4 | `vllm/model_executor/models/qwen3_moe.py` (Qwen3MoeForCausalLM.__init__ 끝부분) | 모든 MoE block 순회 → `wrapper.load_weights(physical_to_logical_map)` 호출 |
| 5 | (3 안에) | cuda_stream + Event 정의, double-buffer 추가 |
| 6 | smoke test | `VLLM_MOE_CPU_OFFLOAD=1 vllm serve ...` boot + 10p correctness compare |
| 7 | e2e | sharegpt 100p × conc=8, A vs B |

## 5. 위험 + 회피

- **kt-kernel 모듈 import** = `/workspace/sglang_kt_prj/lib/python3.12/site-packages/kt_kernel` — vllm venv 에는 없음. **sys.path 주입** 또는 **`uv pip install kt-kernel` 을 vllm_dev_prj 에**. 1단계 agent 가 sglang_kt_prj 에 설치했으므로, vllm 측에서는 두 번째 옵션 (vllm_dev_prj 에 추가 install) 이 sane.
- **weight_path** = HF cache. Qwen3-30B-A3B-Instruct-2507 의 safetensors 가 위치. kt-kernel BF16 path 는 safetensors 의 expert key 를 직접 파싱.
- **physical_to_logical_map** = EPLB 미사용 시 `torch.arange(num_experts)`.
- **correctness**: BF16 산술 비결합성으로 token-level bit-exact 불가. CLAUDE.md §Constraint 운영해석에 따라 **logprob max-abs-diff < 0.1, PPL relative diff < 0.05** 게이트 사용 (10 prompt × max_tokens=128).
- **cuda_stream 인자** = `int` (raw stream pointer). PyTorch 의 `Stream.cuda_stream` property 가 정확한 값.

## 6. 다음 단계 (단계 2 시작 위치)

- env flag: `VLLM_MOE_CPU_OFFLOAD` (env_vars.py 등록), `VLLM_MOE_NUM_GPU_EXPERTS` (기본 32), `VLLM_MOE_CPUINFER_THREADS` (기본 112), `VLLM_MOE_THREADPOOL_COUNT` (기본 2), `VLLM_MOE_KT_METHOD` (기본 `AMXBF16`).
- `_kt_forward` 추가 (forward_cuda 우선).
- import 차단: kt_kernel import 실패 시 OFF 로 fallback 후 warning.
