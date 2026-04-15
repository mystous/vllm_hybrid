# 04. IPEX WoQ INT8 (Weight-Only Quantization)

**Tier**: 0
**상태**: ⭕ 미구현
**예상 이득**: 2× decode (memory-bound). PPL 열화 <0.5

---

## 왜 필요한가

decode 는 **memory-bound**. 7B BF16 weight = 14 GB, decode 1 step 당 weight 전체 DDR read → DDR BW (SPR 약 300 GB/s / socket) 가 ceiling. INT8 weight 로 절반 → DDR read 절반 → decode throughput 2× 잠재.

**중요 성공 조건 차이**:
- "IPEX API 가 호출됐다" → 성공 아님
- **CPU linear hot path 가 실제로 WoQ INT8 kernel 로 치환되고, `num_seqs > 1` 에서 per-req cost 가 내려가야** 성공

---

## 기술적 배경

### Weight-Only Quantization (WoQ)

| 방식 | Weight 저장 | Weight dequant | Activation | 연산 |
|---|---|---|---|---|
| BF16 native | BF16 | - | BF16 | BF16 × BF16 |
| **WoQ INT8** | INT8 + scale | BF16 on-the-fly | BF16 | BF16 × BF16 (dequant 후) |
| W8A8 | INT8 + scale | - | INT8 | INT8 × INT8 (VNNI) |
| WoQ INT4 (GPTQ/AWQ) | INT4 | BF16 on-the-fly | BF16 | BF16 × BF16 |

**WoQ INT8** 은 weight 만 INT8 저장. dequant 를 GEMM kernel 내부에서 수행 (fused). **정확도 열화 <0.5 PPL** (논문 다수).

### IPEX `ipex.llm.optimize()` WoQ 경로

```python
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import WoqWeightDtype, WoqLowpMode

qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
    weight_dtype=WoqWeightDtype.INT8,
    lowp_mode=WoqLowpMode.BF16,  # activation dtype
)
model = ipex.llm.optimize(
    model, dtype=torch.bfloat16,
    quantization_config=qconfig,
    inplace=True,
)
```

현재 `vllm/v1/worker/cpu_worker.py` 의 `ipex.llm.optimize` 호출에 `quantization_config` **미전달** — grep `WoqWeightDtype|weight_only_quant` 결과 없음.

### 호환성 사전 검증

- **IPEX 버전**: vLLM hybrid 는 IPEX 2.4.x 사용. WoQ API 는 2.1.0+ 이지만 kernel 안정성은 버전별 차이. `ipex.quantization.get_weight_only_quant_qconfig_mapping` 존재 여부 확인
- **vLLM model loading 경로**: HuggingFace weight → `torch.nn.Linear` → IPEX `replace_module` 방식. IPEX 의 `_IPEXLinearWithWoq` 또는 유사 class 로 치환되는지 hook 로 확인
- **성공 마커**: `[IPEX] Replaced Linear with IPEXLinearWithWoq` 류 로그 / `type(model.layers[0].self_attn.qkv_proj)` 결과

### Intel-Extension-for-PyTorch 유지보수 리스크

- GitHub 상에 **2026-03-30 archived** 표시 사례 보고 (deep-research §"남은 불확실성 3")
- 장기적으로 Intel 의 공식 추천 path 가 바뀔 가능성 — 단기 (3-6개월) 내 교체 가능성 낮음

---

## 관련 참고 문헌

- **Intel "LLM Weight-Only Quantization on Intel CPU with Intel Extension for PyTorch"** (ICML'24 workshop): https://arxiv.org/abs/2407.07304
- **IPEX WoQ API 문서**: https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/sq_recipe_tuning_api.html
- **IPEX qconfig mapping API**: `intel_extension_for_pytorch.quantization.get_weight_only_quant_qconfig_mapping` 소스
- **deep-research report §"남은 불확실성 3"**: `/vllm_hybrid/ideation/20260415_1629_deep-research-report.md`
- **Codex playbook §6 Tier 1 참고**: `/vllm_hybrid/ideation/20260415_094148_codex_ninja_gap_modification_playbook.md`
- **GPTQ/AWQ 원전**: Frantar et al. (2023) "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (ICLR 2023), Lin et al. (2023) "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"

---

## 구체 작업

- [ ] **IPEX 버전 + WoQ API 존재 확인**: `python -c "from intel_extension_for_pytorch.quantization import WoqWeightDtype"` 성공 여부
- [ ] **vLLM model replacement path 조사**: `ipex.llm.optimize` 가 어느 module 을 어느 class 로 치환하는지 (`_IPEXLinearWithWoq` 등)
- [ ] **`cpu_worker.py` 수정**: `ipex.llm.optimize(model, dtype=..., quantization_config=qconfig, inplace=True)` 로 qconfig 전달
- [ ] **`HybridConfig` 에 flag 추가**: `cpu_woq_int8: bool = False` (auto 아님 — 명시 opt-in)
- [ ] **CLI 옵션**: `--hybrid-cpu-woq-int8`
- [ ] **성공 마커 로그**: `[HYBRID-CPU-WORKER] WoQ enabled — Linear → IPEXLinearWithWoq (N layers replaced)`
- [ ] **hot path 치환 확인**: worker 의 execute_model 내부에서 `self.model.layers[0].self_attn.qkv_proj.__class__` 가 WoQ class 인지
- [ ] **정확도 비교**: gpu_only vs hybrid+WoQ 의 출력 token 동일성 (greedy decoding) 또는 MMLU/HellaSwag 스코어 <0.5pp 차이
- [ ] **per-req cost 측정**: `num_seqs=1/4` 각각 step_ms 비교 — **`num_seqs>1` 에서 cost 하락 확인이 핵심**

---

## 성공 조건

1. ✅ "IPEX optimize 호출됨" 이 아니라, model 의 Linear layer 가 실제 WoQ class 로 치환됨
2. ✅ Memory footprint 2× 감소 (weight 14GB → 7GB for 7B)
3. ✅ `num_seqs>1` 에서 per-req cost 감소 (memory BW 여유가 scaling 에 기여)
4. ✅ PPL 열화 <0.5
5. ✅ G0 baseline 대비 decode step time 1.3–2× 감소

**case 3 (§01 Stop/Go)**: 단일 req 만 빨라지고 batch scaling 없음 → single-req 최적화로 분류, Ninja Gap 기여 아님

---

## 의존성

- **선행**: §01 G0 계측, §03 Huge Pages (권장)
- **후속**: §13 T-MAC LUT GEMV INT4 로 넘어가면 **대체됨** (INT4 가 INT8 보다 2× 더 작음)
- **스택 호환성**: §07 ISA dispatch, §08 fusion, §10 head folding 과 독립 병행

---

## 리스크

- **IPEX WoQ 가 vLLM model structure 와 충돌**: HuggingFace 의 Qwen2.5 구조가 `QKVParallelLinear` (vLLM custom) 를 쓰는데 IPEX 의 replace_module 이 이를 인식 못할 수 있음. 해결: IPEX patch 또는 pre-IPEX 에서 weight unpack
- **WoQ kernel 이 AVX-512 VNNI 경로를 탐**: AMX 기반 INT8 경로와 분리되어 AMX pre-pack 이득과 충돌 가능
- **IPEX 버전 업그레이드 시 API 변경**: `quantization_config` 형식 바뀔 수 있음
- **정확도 열화가 특정 모델에서 큼**: Qwen2.5 outlier activation 이 INT8 허용 범위 초과 가능 — GPTQ/AWQ calibrated weight 로 fallback

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_WOQ_INT8` | `0` (기본) / `1` | IPEX WoQ INT8 경로 활성 |

전체 flag 테이블: [README.md](./README.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `vllm/v1/worker/cpu_worker.py` — `ipex.llm.optimize` 호출부
- `vllm/config.py` — `HybridConfig`
- `vllm/engine/arg_utils.py` — CLI 옵션
- `vllm/model_executor/layers/linear.py` — vLLM custom Linear (IPEX patch 필요 지점)
