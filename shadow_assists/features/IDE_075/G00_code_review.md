# G00 — GLM-4.7-FP8 + kt FP8_PERCHANNEL 정상 출력 차단 원인 코드 검토 (읽기 전용 에이전트, 2026-09-17)

조사 완료. 컨테이너에는 현재 **Qwen3-Coder-480B / AMXINT4 서버가 가동 중**이어서(PID 1015118) CPU를 점유하는 커널 테스트는 실행하지 않고 정적 분석만 했습니다. 파일 수정은 하지 않았습니다.

---

# GLM-4.7-FP8 + kt `FP8_PERCHANNEL` 깨진 출력 원인 분석

## 0. 확인된 사실 (체크포인트 / 실행 설정)

**`/models/hub/models--zai-org--GLM-4.7-FP8/snapshots/*/config.json`**
- `quantization_config`: compressed-tensors `"format": "float-quantized"`, weights `"strategy": "channel"`, `"num_bits": 8`, **`weight_block_size` 없음** (per-channel), input_activations `"dynamic": true, "strategy": "token"`.
- `n_routed_experts=160, n_shared_experts=1, num_experts_per_tok=8, routed_scaling_factor=2.5, scoring_func="sigmoid", topk_method="noaux_tc", n_group=1, topk_group=1, norm_topk_prob=true, first_k_dense_replace=3, num_hidden_layers=92, hidden_size=5120, moe_intermediate_size=1536`.

체크포인트 실측 shape:
```
model.layers.3.mlp.experts.0.gate_proj.weight        [1536, 5120]  F8_E4M3
model.layers.3.mlp.experts.0.gate_proj.weight_scale  [1536, 1]     F32
model.layers.3.mlp.experts.0.down_proj.weight        [5120, 1536]  F8_E4M3
model.layers.3.mlp.experts.0.down_proj.weight_scale  [5120, 1]     F32
```
→ kt 가 기대하는 포맷과 **정확히 일치**.

실제 실행 커맨드(`/home/mystous/projects/vllm_hybrid/eval/results/IDE_074_20260917/glm/BASIC4/G_135723/launch_cmd.sh`)에는 `--tp 4`, `--disable-shared-experts-fusion`, `--kt-method FP8_PERCHANNEL --kt-num-gpu-experts 80 --kt-cpuinfer 100 --kt-threadpool-count 2 --kt-max-deferred-experts-per-token 2` 가 들어 있습니다.

---

## 1. Python 디스패치 / 가중치 포맷 (질문 1)

- `/usr/local/lib/python3.12/dist-packages/kt_kernel/experts.py::_create_inference_wrapper:345-357` — `FP8_PERCHANNEL` → `backend_cls = NativeMoEWrapper`.
- `/usr/local/lib/python3.12/dist-packages/kt_kernel/utils/amx.py::NativeMoEWrapper._create_loader:719`
  ```python
  elif method == "FP8_PERCHANNEL":
      return FP8SafeTensorLoader(weight_path, scale_suffix="weight_scale")
  ```
  → suffix 를 **명시적으로** 넘기므로 `loader.py::FP8SafeTensorLoader.__init__:366` 에서 `self._is_per_channel = True` 로 즉시 확정되고, `_detect_format` 의 자동 감지(shape 기반 per-channel/blockwise 판별) 분기는 타지 않습니다. **`weight_block_size` 나 `quantization_config` 를 읽는 코드는 kt 쪽에 전혀 없습니다** — `--kt-method` 가 유일한 스위치이므로 "`weight_block_size` 누락" 자체는 문제가 아닙니다. (반대로 `--kt-method FP8`(블록) 을 줬다면 suffix 자동감지로 `weight_scale` 을 찾아 `_is_per_channel` 은 True 가 되지만 C++ 은 group_size=128 블록으로 해석 → 확실히 깨짐.)
- `/usr/local/lib/python3.12/dist-packages/kt_kernel/utils/amx.py::NativeMoEWrapper.load_weights:978-982`
  ```python
  elif self.method == "FP8_PERCHANNEL":
      moe_config.quant_config.bits = 8
      moe_config.quant_config.per_channel = True
      self.moe = AMXFP8PerChannel_MOE(moe_config)
  ```
  → C++ 클래스 `AMX_FP8_PERCHANNEL_MOE_TP<amx::GemmKernel224FP8PerChannel>`. `group_size` 는 설정하지 않음(기본 0, per-channel 경로에서 미사용 — `common.hpp:222-228`).
- 스케일 dtype 검증: `amx.py:823-827` `assert self.gate_scales[0].dtype == torch.float32`.
- 스케일 로드: `/usr/local/lib/python3.12/dist-packages/kt_kernel/utils/loader.py::FP8SafeTensorLoader.load_experts:512-534`
  ```python
  gate_s_key = f"{experts_prefix}.{exp_id}.{gate_name}.{self._scale_suffix}"
  ...
  if self._is_per_channel:
      if gate_scale.dim() == 2 and gate_scale.shape[1] == 1:
          gate_scale = gate_scale.squeeze(1)      # [N,1] -> [N]
  ```
  포맷 감지는 `MOE_FORMATS["deepseek"] = ("{base}.mlp.experts", "gate_proj","up_proj","down_proj")` (loader.py:348) 로 GLM 과 일치. expert 개수는 `has_tensor(f"{prefix}.{i}.gate_proj.weight")` 로 세어 **160** 을 얻습니다.

**결론(1): Python 로딩/디스패치 경로는 올바릅니다.**

---

## 2. C++ FP8 per-channel 커널 (질문 2)

- `/sgl-workspace/ktransformers/kt-kernel/operators/amx/fp8-perchannel-moe.hpp::AMX_FP8_PERCHANNEL_MOE_TP::derived_init:52-56` — `per_channel` 미설정 시 throw.
- 가중치 재배열: `operators/amx/la/amx_raw_buffers.hpp::BufferBFP8PerChannelImpl::from_mat:754-789` — 스케일은 `memcpy(d + n_start, d_src + n_start, ...)` 로 **행 단위 선형 복사**, 가중치 재배열 루프는 검증된 블록 FP8 `BufferBFP8Impl::from_mat:325-365` 와 **바이트 단위로 동일**(`mat_offset[8] = {0,2,4,6,1,3,5,7}`).
- FP8→BF16 변환: `amx_raw_kernels.hpp:289-320` 의 LUT 재사용(`GemmKernel224FP8::fp8x64_to_bf16x64`). E4M3(subnormal 포함) 지수 매핑을 손검산해봤고 정확합니다(예: E=0,M=1 → 2⁻⁹ → bf16 hi=0x3B ✓, E=1 → hi=0x3C ✓).
- 스케일 적용: `amx_raw_kernels.hpp::GemmKernel224FP8PerChannel::apply_scale_perchannel:682-702`
  ```cpp
  __m512 bs_lo = _mm512_loadu_ps(bb->get_scale(n_begin));
  __m512 bs_hi = _mm512_loadu_ps(bb->get_scale(n_begin + TILE_N));
  ... _mm512_mul_ps(c_lo, bs_lo) ...
  ```
  → **출력 행(= per-output-channel) 단위**, K 전체 누적 후 1회 적용(`float_mat_vec_perchannel:816-836`). 수학적으로 `y = (Σ q·a) * s` 로 per-channel 규약과 일치.
- **활성값 양자화는 전혀 하지 않습니다.** `BufferA = BufferABF16Impl` (amx_raw_kernels.hpp:660) → 입력은 BF16 그대로, `_mm512_dpbf16_ps` 로 FP32 누적. 즉 per-token dynamic FP8 양자화(GPU 쪽)는 CPU 에 없고, CPU 가 오히려 더 정확합니다. → **activation scheme 불일치는 garbage 원인이 아님.**
- 하드코딩 블록: `N_BLOCK = 128`, `K_BLOCK = 7168` (amx_raw_kernels.hpp:641-642). 이는 **양자화 블록이 아니라 타일링 블록**이며 per-channel 규약과 충돌하지 않습니다. GLM 수치로 모두 정합: gate/up `n=768(TP2), k=5120`, down `n=5120, k=768` 전부 `N_STEP=32 / K_STEP=32` 배수.
- TP 분할 로딩: `fp8-perchannel-moe.hpp::TP_MOE<...>::load_weights:602-700` — gate/up 은 N 분할(`gate_up_scale_src_offset = i * tpc.intermediate_size`), down 은 K 분할 + 스케일 전량 복사(`tp_down_scale_elems = tpc.hidden_size`). 1536/2=768 로 균등 분할이므로 정합. (다만 `i * tp_weight_elems` 식은 **균등 분할 가정** — 블록 FP8 은 `intermediate_offsets[i]` 를 쓰는데(fp8-moe.hpp:671) 여기는 아님. GLM 에선 무해하나 잠재 버그.)
- expert 인덱싱: `load_weights:521-553` 과 `TP_MOE::load_weights:648-693` 이 둘 다 `expert_map()` 을 적용하지만 dst/src 가 서로 상쇄되어 **self-consistent** 합니다(블록 FP8 과 동일 패턴). `init_expert_location=trivial` 이므로 identity.
- 잠재 이슈(현재 미발현): `BufferBFP8PerChannelImpl::required_size:730` 은 `n*k + 4*n` 을 **64B 라운드업하지 않습니다**(블록 FP8 은 `:300-305` 에서 라운드업). GLM 수치는 우연히 64 배수라 안전하지만, 다른 shape 에서는 `__m512i*` 정렬 load 가 깨질 수 있습니다.

**결론(2): FP8 per-channel 커널 자체에서 "출력이 완전히 깨질" 수준의 결함은 발견되지 않았습니다.** 유일한 정량 검증은 `examples/test_fp8_perchannel_moe.py` 의 **상대 L1 오차 15% 임계값**(같은 파일 하단 `threshold = 15.0`)으로 매우 느슨합니다.

---

## 3. 체크포인트 키 네이밍 / layer_idx 매핑 (질문 3)

- kt 가 찾는 키: `amx.py::NativeMoEWrapper.load_weights:775-779`
  ```python
  _candidates = [f"model.layers.{self.layer_idx}", f"language_model.model.layers.{...}", ...]
  ```
  → `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.{weight,weight_scale}`. 체크포인트와 **완전 일치**.
- `layer_idx` 는 `kt_ep_wrapper.py::create_kt_config_from_server_args:135-137` 에서 `layer_id` 를 그대로 받고, 그 `layer_id` 는 `fused_moe_triton/layer.py:370` 에서 FusedMoE 의 전역 디코더 레이어 인덱스입니다. GLM 의 MoE 레이어는 3~91 이고 체크포인트 키도 전역 인덱스라 **오프셋 버그 없음**(`first_k_dense_replace` 보정 불필요). MoE 레이어 여부는 `glm4_moe.py::Glm4MoeDecoderLayer._is_layer_sparse:902-906` 의 `layer_id >= self.config.first_k_dense_replace` 로 결정됩니다.
- `num_layers=92` → `kt_ep_wrapper.py:247-253` 이 layer 91 의 deferred 를 0 으로 강제(마지막 레이어의 이월분 누락 방지). 정상.

**결론(3): 키 네이밍·레이어 인덱스 모두 정상. `weight_scale_inv` 를 기대하지 않고 `weight_scale`([out,1]) 를 올바르게 읽습니다.**

---

## 4. SGLang GLM 라우팅 / routed_scaling_factor (질문 4) — **여기가 문제**

### 4-1. topk 가중치에는 2.5 가 들어있지 않습니다
- `glm4_moe.py::Glm4MoeSparseMoeBlock.__init__:437-450`
  ```python
  apply_routed_scaling_factor_on_output=getattr(
      self.experts, "should_fuse_routed_scaling_factor_in_topk", False)
  ```
- `fused_moe_triton/layer.py:443-461` — `should_fuse_routed_scaling_factor_in_topk` 는 `ModelOptNvFp4FusedMoEMethod` / (`Fp8MoEMethod` + cutlass|trtllm_routed) 일 때만 True. 여기서 `quant_method` 는 **`KTEPWrapperMethod`** 이므로 **False**.
- GLM 라우팅은 `topk.py::biased_grouped_topk_gpu` 에서 `num_experts=160`(비2의거듭제곱), `num_expert_group=1` 이라 flashinfer/grouped 분기를 모두 건너뛰고 `topk.py:1688-1706` 의 ungrouped Triton `jit_gate` 로 갑니다. 여기에 `apply_routed_scaling_factor_on_output=False` 가 전달 → **topk_weights 에 2.5 미포함** (sigmoid + `e_score_correction_bias` + renormalize 는 정상).

### 4-2. GPU 경로는 러너 내부에서 2.5 를 곱합니다
- `moe_runner/triton.py::TritonRunnerCore.run:164` → `routed_scaling_factor=self.config.routed_scaling_factor`
- `moe_runner/triton_utils/fused_moe.py:851-923`
  ```python
  if routed_scaling_factor is None: routed_scaling_factor = 1.0
  ...
  out_slice.mul_(routed_scaling_factor)   /  moe_sum_reduce(..., routed_scaling_factor)
  ```
  → `gpu_method.apply()` 결과 `G = 2.5·g`.

### 4-3. CPU(kt) 경로는 2.5 를 곱하지 않습니다
- `operators/amx/moe_base.hpp:479` / `:703` — `weight = _mm512_set1_ps(weights[i*k+j])` 로 **topk_weights 만** 적용. kt_kernel 전체에 `routed_scaling` 문자열이 존재하지 않습니다(grep 결과 0건).
- `kt_ep_wrapper.py::KTEPWrapperMethod.apply:439-445`
  ```python
  cpu_output = self.sync(x)
  output = output + cpu_output      # 2.5·g  +  c
  ```

### 4-4. 그 결과 prefill 과 decode 가 서로 다르게 틀립니다
- `glm4_moe.py::Glm4MoeSparseMoeBlock.forward_normal:608-610` (prefill / non-capture)
  ```python
  final_hidden_states = self.experts(hidden_states, topk_output)
  if not _is_cuda and not _use_aiter:
      final_hidden_states *= self.routed_scaling_factor   # CUDA 에서는 실행 안 됨
  ```
  → 결과 `2.5g + c` (**CPU 기여가 2.5배 과소**)
- `glm4_moe.py::Glm4MoeSparseMoeBlock.forward_normal_dual_stream:583-585` (CUDA graph capture/decode)
  ```python
  final_hidden_states = self.experts(hidden_states, topk_output)
  if not _is_cuda or isinstance(self.experts.quant_method, KTEPWrapperMethod):
      final_hidden_states *= self.routed_scaling_factor
  ```
  → 결과 `2.5·(2.5g + c) = 6.25g + 2.5c` (**GPU 기여가 2.5배 과대**)

즉 `forward_normal` 에는 `deepseek_v2.py:1144-1151` 에 있는 KT 분기
```python
if (not _is_cuda and not _is_musa and not _is_xpu and not _use_aiter
    or isinstance(self.experts.quant_method, KTEPWrapperMethod)):
    final_hidden_states *= self.routed_scaling_factor
```
가 **누락**되어 있고, 설령 추가해도 GPU 항이 이중 스케일되므로 근본적으로 **"GPU는 러너 내부 스케일 / CPU는 미스케일" 조합이 잘못**입니다. (이 파일은 `/sgl-workspace/sglang-upstream` 과 diff 결과 완전 동일 → 로컬 개조가 아닌 업스트림 상태입니다.)

### 4-5. 왜 Qwen3-Coder-480B(AMXINT4) 는 정상인가
`/sgl-workspace/sglang/python/sglang/srt/models/qwen3_moe.py` 에는 **`routed_scaling_factor` 가 단 한 번도 등장하지 않습니다**(grep 0건). Qwen3-MoE 는 스케일 팩터가 없으므로 위 불일치가 무해합니다. → **"GLM만 깨지고 Qwen은 정상"이라는 관측과 정확히 일치합니다.**

### 4-6. shared expert / expert-id 리맵 (현재는 문제 아님, 단 지뢰)
- 실행에 `--disable-shared-experts-fusion` 이 있어 `num_fused_shared_experts=0` → shared expert 는 `Glm4MoeMLP` 로 GPU 에서 별도 계산(`glm4_moe.py:456-470`, `_forward_shared_experts:676-680`). 정상.
- **만약 이 플래그를 빼면**: `glm4_moe.py::shared_experts_fusion_disable_reason:1174-1192` 가 CUDA cap≥8.0 & ep_size=1 에서 `None` 을 반환 → fusion ON → `num_experts=161`, topk 마지막 슬롯 id=160 (`topk.py:1335-1346`). 그러면 `kt_ep_wrapper.py::mask_cpu_expert_ids:163` 이 160(≥80)을 -1 로 만들어 GPU 에서 제외하고, CPU 는 `is_gpu_expert(160)=false` 로 계산하려 하는데 로더는 160개만 주므로 `fp8-perchannel-moe.hpp:672` 의 `config.gate_projs[0][expert_id]` 가 **OOB** → crash/garbage. 반드시 플래그 유지 필요.
- physical↔logical: `init_expert_location='trivial'`, `ep_num_redundant_experts=0` → identity 160. `process_weights_after_loading:308-313` 이 넘기는 맵과 C++ `expert_map` 이 정합.
- GPU 마스크: 로컬 패치(`kt_ep_wrapper.py:269-272`, 업스트림에는 없음)가 `_gpu_mask[:80]=True` 를 만들어 `mask_cpu_expert_ids(topk_ids, 80)` 와 일치시킵니다. 정상.

---

## 5. 업스트림 문서 / 이슈 노트 (질문 5)

- `/sgl-workspace/ktransformers/doc/en/kt-kernel/Native-Precision-Tutorial.md:37` — `| **GLM-4.7** | FP8_PERCHANNEL, BF16 |` → 공식 지원 모델로 등재.
- 같은 문서 `:140-165` 의 **공식 GLM-4.7 레시피**는 `--tensor-parallel-size 8`, `--disable-shared-experts-fusion`, 그리고 **`--kt-enable-dynamic-expert-update`** 를 포함합니다. 그런데 이 빌드의 `server_args.py` 에는 kt 인자가 6개(`kt_weight_path/kt_method/kt_cpuinfer/kt_threadpool_count/kt_num_gpu_experts/kt_max_deferred_experts_per_token`)뿐입니다(`server_args.py:3045-3070`). → **문서가 가정하는 kt-sglang 포크(layerwise prefill / dynamic expert update 포함)와 현재 `/sgl-workspace/sglang` 통합이 다릅니다.** 실제로 업스트림 `kt_ep_wrapper.py` 는 `KTMoEWrapper(..., num_gpu_experts=...)` 를 넘기는데 설치된 `kt_kernel` 의 `experts.py::__new__:130` 은 `gpu_experts_mask` 를 필수 인자로 요구 → 로컬 패치 없이는 TypeError. 즉 **버전 스큐가 존재**합니다.
- `kt-kernel/README.md:193` 은 GLM-4.7 을 **BF16** 행에만 예시로 적고 있습니다.
- `kt-kernel/test/test_swiglu_limit_method_guards.py` 는 FP8_PERCHANNEL 을 "clamp 미검증" 방법으로 거부하는 가드만 테스트(정확도 테스트 아님). `FP8_PERCHANNEL` 수치 검증은 `examples/test_fp8_perchannel_moe.py`(임계 15%) 와 `bench/bench_fp8_perchannel_moe.py` 뿐입니다.

---

## 6. 가설 순위 + 최소 진단

### H1 (최우선, 확신도 높음) — `routed_scaling_factor=2.5` 가 GPU 기여에만 적용됨
- 코드: `moe_runner/triton_utils/fused_moe.py:851-923`(GPU에 ×2.5) + `operators/amx/moe_base.hpp:479`(CPU는 topk_weights만) + `kt_ep_wrapper.py::apply:445` `output = output + cpu_output` + `glm4_moe.py:609-610`(prefill 미적용) vs `glm4_moe.py:584-585`(decode ×2.5).
- 예상 증상: 89개 MoE 레이어 전부에서 라우팅 합이 어긋나고 **prefill 과 decode 가 서로 다른 배율** → 완전 비문, EOS 미도달(max_tokens 소진). Qwen3 는 스케일 팩터가 없어 무해.
- **최소 진단**
  1. `--kt-num-gpu-experts 160` (전 expert GPU) 으로 1요청. CPU 기여가 0 이 되어 prefill 은 정상, decode 는 `×2.5` 한 번만 남아 여전히 깨짐 → H1 확정. (반대로 `--kt-num-gpu-experts 0` 이면 `kt_ep_wrapper.py:411-419` fast path 로 CPU only 가 되어 prefill 은 2.5 누락, decode 는 정상 — 두 조합의 비대칭 자체가 결정적 증거)
  2. `--disable-cuda-graph` 로 decode 도 `forward_normal` 을 타게 하여 prefill/decode 배율을 일치시킴 → 여전히 틀리지만 **출력의 "붕괴 양상"이 바뀌어야** 함(일관되게 CPU 기여만 1/2.5). 바뀌면 H1.
  3. 코드 무수정 수치 확인: `KT_DUMP_TOPK=1`(`experts_base.py:635-645`) 로 topk_weights 를 덤프 → 각 토큰 행 합이 `1.0`(2.5 미포함) 인지 확인. 합이 1.0 이면 GPU 러너가 ×2.5 를 담당하고 CPU 는 못 받는다는 것이 확정.

### H2 (중간) — prefill/decode 경로 배율 불일치 (H1 의 하위 케이스지만 독립 검증 가능)
- 코드: `glm4_moe.py:584-585` vs `:609-610`. decode 만 KT 분기가 있음(`deepseek_v2.py:1144-1151` 에는 양쪽 다 있음).
- 최소 진단: 위 (2)번. 또는 `max_tokens=1` 짧은 프롬프트(prefill만)와 긴 생성(decode 위주) 출력의 붕괴 성격을 비교.

### H3 (중간-낮음) — `--kt-max-deferred-experts-per-token 2` 의 레이어-이월 근사
- 코드: `experts_base.py::select_deferred_experts:567-587` (protected_k = 8-2 = 6) + `experts_base.py:628-629` `next_slot = (layer_idx+1) % buffer_depth(=2)` → **레이어 L 의 하위 2개 expert 출력이 레이어 L+1 의 CPU 출력 버퍼에 더해집니다**(`incremental` 병합, `moe_base.hpp::merge_results:841`).
- GLM 은 hotmap 없이 `gpu_experts=0..79` 라 CPU 로 가는 트래픽이 ~50% → 이월 오차가 Qwen(hotmap + cold expert만 CPU) 보다 훨씬 큼.
- 최소 진단: `--kt-max-deferred-experts-per-token 0` 으로 재기동 후 동일 프롬프트. 단독으로는 "완전 비문"을 설명하기 어려우므로 H1 수정 후 남는 열화 확인용.

### H4 (낮음, 그러나 지뢰) — shared-expert fusion 이 켜지면 CPU 측 OOB
- 코드: `glm4_moe.py:1174-1192` (CUDA 에서 기본 fusion ON) + `topk.py:1335-1342` (id=160) + `kt_ep_wrapper.py:163` + `fp8-perchannel-moe.hpp:672` `config.gate_projs[0][expert_id]` (벡터 길이 160).
- 현재 실행에는 `--disable-shared-experts-fusion` 이 있어 발현 안 됨. 그러나 이 플래그를 빼면 즉시 garbage/segfault.
- 최소 진단: 로그에서 `TP MOE layer N, ..., expert num: 160` (moe-tp.hpp:63-64) 이 **160** 인지 확인. 161 이면 H4.

### H5 (낮음) — FP8 per-channel 커널 자체의 수치 결함
- per-channel 전용 경로에는 AMX 타일 커널이 없고 `avx_kernel_4` 만 존재(`amx_raw_kernels.hpp:705-810`)하며, 블록 FP8 은 AMX 가 있으면 `K::amx_kernel` 을 씁니다(`float_mat_vec_kgroup:553-556`). 즉 **per-channel 은 블록 FP8 의 "덜 검증된" AVX512 경로를 재사용**합니다. 다만 `from_mat` 재배열·LUT·누적 레인 배치가 블록 FP8 과 동일하고, `to_mat`(`amx_raw_buffers.hpp:607-623`) 이 자연 열 순서를 가정하는 것과 `apply_scale_perchannel` 의 `loadu_ps(d + n_begin)` 이 정합하므로 결함 가능성은 낮습니다.
- 최소 진단(서버 부하 때문에 저는 실행하지 않았음): 서버를 쓰지 않는 시간대에
  `python3 /sgl-workspace/ktransformers/kt-kernel/examples/test_fp8_perchannel_moe.py`
  → 상대 L1 오차가 **수 % 이내**면 커널 무죄. 다만 해당 스크립트의 통과 임계는 15% 로 느슨하므로 **출력된 수치 자체를 보세요**(1~3% 면 FP8 양자화 오차 수준, 30%+ 면 per-channel 스케일 정렬 버그).
- 추가 검증(비침습): `KT_NAN_PROBE=1`(`kt_ep_wrapper.py:38-65`) 로 CPU/GPU expert 출력 NaN 카운터 확인. NaN 이 0 이면 커널 폭주형 버그는 배제되고 H1(스케일) 쪽으로 수렴.

### H6 (배제됨)
- 활성값 양자화 불일치: CPU 는 `BufferABF16Impl` 로 **양자화하지 않음** → `activation_scheme: dynamic/token` 은 CPU 무관.
- `weight_block_size` 누락 처리: kt 는 `quantization_config` 를 읽지 않고 `--kt-method` 로만 분기 → 무관.
- `weight_scale` vs `weight_scale_inv`: `amx.py:719` 가 `scale_suffix="weight_scale"` 를 명시 → 올바름. shape `[N,1]` → `[N]` squeeze 도 정상(`loader.py:526-534`).
- layer_idx 오프셋(`first_k_dense_replace`): 전역 인덱스 그대로 사용, 정상.
- physical/logical 리맵, `--kt-num-gpu-experts 80` 마스크: identity + `[:80]=True` 로 GPU 마스킹과 일치, 정상.

---

## 7. 권장 확인 순서 (가장 빠른 것부터)

1. `KT_DUMP_TOPK=1` 로 topk_weights 행 합 = 1.0 확인 → H1 즉시 확정 (재기동 1회, 추론 1회).
2. `--kt-num-gpu-experts 160` 과 `--kt-num-gpu-experts 0` 두 극단을 각각 1요청씩 비교 → "한쪽만 정상"이면 H1/H2 확정.
3. 로그에서 `Created FP8_PERCHANNEL_MOE_TP <tp> at numa <n> (backend=...)` (`fp8-perchannel-moe.hpp:58-61`) 와 `TP MOE layer N ... expert num: 160` 확인 → 백엔드/expert 수 점검(H4 배제).
4. `--kt-max-deferred-experts-per-token 0` → 잔여 열화 측정(H3).
5. 서버 유휴 시 `examples/test_fp8_perchannel_moe.py` 수치 확인(H5).

---

## 8. 확장 단계 확정 결과 (2026-09-17 17:00~18:00, 진단 D1~D5 + 커널 bisect)

### 8-1. 진단 실행 결과 (4문항 게이트, GLM-4.7-FP8 BASIC4)

| 진단 | 구성 | 결과 |
|---|---|---|
| D1 | 무수정 + `--kt-num-gpu-experts 160` (CPU 기여 0) | NOT_RUN (TP4 GPU OOM, 160 expert 상주 불가) |
| D2 | rsf 수정(`glm_rsf_patch.py`) + 기본 80 | BLOCKED_NORMAL_OUTPUT 0/4 (문장 붕괴) |
| D3 | rsf 수정 + deferred 0 | BLOCKED_NORMAL_OUTPUT 0/4 |
| D4 | rsf 수정 + `--kt-num-gpu-experts 0` (전 CPU) | BLOCKED_NORMAL_OUTPUT 0/4 |
| D5 | 무수정 + 전 CPU + deferred 0 | BLOCKED_NORMAL_OUTPUT 0/4 |

→ 스케일·deferred·GPU/CPU 분할 어느 축으로도 회복되지 않음. H1(rsf) 은 실재하는 결함이지만 **단독 원인이 아님**. H5(커널) 로 이동.

### 8-2. 커널 단독 테스트 + bisect

- `examples/test_fp8_perchannel_moe.py` / `test_fp8_moe.py`: 설치 .so(v1~v3), 원본 .so(e29357f7, 09-09 빌드) 모두 **출력 전부 0 (Mean relative L1 100%)**. upstream 6d460cc 클린 빌드(`/tmp/kt_up`) 는 PASS 0.5829%.
- 로컬 수정 파일 10개를 upstream 트리에 하나씩 이식: A(moe-tp.hpp) PASS, B(cpuinfer/task_queue/ext_bindings) PASS, D(CMakeLists) PASS, C 그룹 FAIL → 파일별: moe.hpp PASS, **moe_base.hpp FAIL**, amx_kernels.hpp PASS, amx_buffers.hpp PASS.

### 8-3. 근본 원인 (확정): `moe_base.hpp` 의 if-constexpr dangling-else

IDE_046-b(`KT_QA_PAR`)·IDE_051(`KT_FUSE_QIN`) 이 넣은 hunk 3곳 (입력 gather, gate_up A 양자화, down A 양자화) 의 형태:

```cpp
if constexpr (requires(typename T::BufferA& x_, ...) { x_.from_mat_row(...); }) if (env_flag && qlen >= 10) {
  /* 새 경로 */
} else
/* 원본 경로 (direct_or_pool / pool->do_work_stealing_job ...) */
```

C++ 에서 `else` 는 가장 가까운 `if (env_flag ...)` 에 결합하므로 원본 경로도 `if constexpr` 의 본문에 속한다. `from_mat_row/from_mat_block` 은 INT4 계열 `BufferAImpl` 등에만 있고, FP8·FP8PerChannel·BF16 커널의 `BufferA = BufferABF16Impl` (`operators/amx/la/amx_raw_buffers.hpp:51`, `amx_raw_kernels.hpp:320/660`) 에는 없다. 따라서 이들 커널에서는 `requires` 가 거짓 → 새 경로와 **원본 경로가 함께 컴파일에서 제거** → 입력 gather(memcpy) 도, A 버퍼 양자화도, down A 양자화도 수행되지 않은 채 GEMM → 출력 0. 컴파일 시점 결정이므로 환경변수·서버 인자 어느 것으로도 회복 불가 (D2~D5 와 일치). INT4 커널은 `requires` 참 → 원본 경로 유지 → Qwen3-Coder-480B(AMXINT4) 무영향 (§4-5 의 결론 유지, 이유는 수정).

원본 .so(e29357f7) 도 실패한 이유: 그 .so 자체가 IDE_051 이후 로컬 트리로 빌드된 것 (09-09) 이라 동일 결함 포함.

### 8-4. 수정

`eval/ide075/glm_fp8_dispatch_fix.py` — 3곳을 `bool flag=false; if constexpr(...) { if (env) { 새 경로; flag=true; } } if (!flag) 원본;` 으로 재구성. 환경변수 미설정 시 INT4 경로 동작은 수정 전과 동일. 재빌드 .so 로 (1) kt FP8 단독 테스트 2종, (2) GLM 게이트 D6(rsf 수정 포함) / D7(rsf 미수정) 를 실행해 rsf 결함(§4)의 효과를 분리한다. 결과는 FULL_REPORT §G00 에 기록.

### 8-5. 기준 답안 대조 (IDE_073 G-GPU8: GLM-4.7-FP8 전 GPU TP8, kt 없음, 같은 quality20 greedy·thinking off)

| 구성 | Q0 | Q1 | Q2 | Q3 | 4문항 정답 |
|---|---|---|---|---|---|
| D6 = dispatch 수정 + rsf 패치 | 129자 공통 접두 후 분기 (289 vs 325자, 답 18 동일) | **완전 일치** (421자) | **완전 일치** (885자) | **완전 일치** (296자) | 4/4 |
| D7 = dispatch 수정, rsf 원본 | 접두 0자 | 1자 | 0자 | 21자 | 4/4 (문장·풀이 경로 다름) |
| D2 = rsf 패치, dispatch 미수정 | 0자 (3,639자 붕괴) | 0 | 1 | 6 | 0/4 |

해석: (1) dispatch 수정만으로도 답은 맞지만 greedy 시퀀스가 첫 토큰부터 갈림 → CPU 기여의 2.5 배 누락은 분포를 바꾸는 실재 결함 (H1 확정). (2) 두 수정을 함께 적용하면 전 GPU 기준과 3/4 문항이 bit-동일, 나머지 1 문항도 129자까지 동일 후 분기 (BF16 비결합·구성 차이에 따른 cascading divergence 범위; CLAUDE.md 정확도 해석과 일치). (3) GLM CPU 경로는 두 수정을 모두 적용해야 GPU-only 와 동등.
