# 16. SparAMX AVX-512 Bitmask Sparse

**Tier**: 근거 등급 **B** (유사 HW 실측). `TODO.md` §3-A Tier 1 후보 **우선순위 1**.
**상태 (2026-04-20)**: ⭕ 미구현 — 공식 구현 공개 확인됨, **preflight-first** 진행.
**예상 이득**: linear **1.42×**, attention **1.14×** (SparAMX 실측, **Xeon SPR** — 우리 H100 호스트 CPU 와 동일).
**공식 구현**: MIT 라이선스, PyTorch C++ extension 형태 — 이식 가능.
**우선순위 근거**: Tier 1 후보 4개 중 측정 환경 일치도 최상 (CPU SPR 동일). 보고 수치 직접 비교 가능.

---

## 현재 판단 (2026-04-20) — preflight-first

### 결론

§16 은 우선순위 1 이지만 **첫 코드 작업은 sparse kernel 구현이 아니다**. `§16 preflight harness` 부터 진행한다.

이유:
- Sparse kernel 이득은 **pruned model** 전제. Pruning 성립 안 하면 kernel 이식 의미 없음
- Pruning feasibility (정확도 열화 <2%) 를 offline 에서 먼저 확인 가능 — kernel 공수보다 훨씬 싸다
- Preflight 실패 시 우선순위 2 (§22) 또는 3 (§28) 으로 즉시 이동 — sunk cost 최소

### 분기

- Preflight 통과 + 공식 impl 재사용 가능 → **SparAMX 이식 경로**
- Preflight 통과 + 재사용 곤란 → sparse kernel 자체 구현 검토 (→ §28 xFT 와 비교 재평가)
- Preflight 실패 (PPL 열화 >2%) → §22 NEO 또는 §28 xFT 이식으로 이동

### 현재 코드 상태

- Repo 에 sparse path **없음**. `HYBRID_SPARSE_BITMASK` env flag 는 manifest 에만 있고 `HybridConfig` / CLI / worker path 에 **미연결**
- 현재 CPU custom path: dense `vnni_int8_gemm`, `q8_0_linear` (§06-1 v1) 까지만
- 즉 §16 은 현재 기준으로 "flag 명만 존재 / 구현 0" 상태

---

## Step 0 / 0-1 결과 — SparAMX 공식 구현 상세

### 메타

- **Repo**: https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/tree/main/SparAMX
- **Paper**: https://arxiv.org/abs/2502.12444 (AbouElhamayed et al.)
- **License**: **MIT** (vLLM Apache 2.0 과 호환 — 이식 OK)
- **빌드**: `pip install -r requirements.txt` + `python setup.py install`

### 구조

**5 개 C++ extension** (`csrc/`):
| extension | 용도 |
|---|---|
| `sparse_linear` | 핵심 sparse kernel (본 이식 대상) |
| `avx_sparse_linear` | AVX-512 fallback |
| `quantized_sparse_linear` | sparse + quant (§16 + §24 병합 시 후보) |
| `quantized_dense_linear` | dense + quant (비교용) |
| `dense_linear` | dense baseline (비교용) |

**6 개 Python Linear class** (`layer/`): `SparseLinear`, `AVXSparseLinear`, `OneDNNLinear`, `DenseLinear`, `QuantizedSparseLinear`, `QuantizedDenseLinear`.

**Compile flags** (setup.py):
```
-march=sapphirerapids -O3 -DNDEBUG
-mavx512f -mavx512dq -mamx-tile -mamx-int8 -mamx-bf16
-fopenmp -lgomp
```
→ 우리 `cmake/cpu_hybrid_extension.cmake` flag 와 호환 (`-mamx-*` 추가 필요 가능).

### `SparseLinear` 핵심 API

- **상속**: `class SparseLinear(nn.Linear)`
- **변환**: `SparseLinear.from_linear(standard_linear)` classmethod
- **Weight storage (3 tensor)**:
  - `weight_metadata` — **bitmap, int32 packed**. `bitmap_tensor = (weights != 0)` → `bits_to_int32()`
  - `weight_values` — **BF16 compact** (nonzero only)
  - `weight_values_is` — per-thread index offsets
- **Forward**:
  ```python
  output = torch.zeros(B, S, out_cols, dtype=fp32, device=input.device)
  sparse_linear.forward(input, weight_metadata, weight_values,
                         weight_values_is, out_cols, output)
  return output.to(bfloat16)
  ```
- **입력 shape**: `(batch, seq, in_features)` — 3D
- **FP32 accumulate → BF16 cast** 로 numeric 안정성 확보

### Attention sparse

논문 섹션이 "attention 에 unstructured sparsity" 를 언급하고 **1.14×** 보고. 그러나 README 는 구체 API 명시 안 함. `custom_llama_attention.py` 류의 파일이 repo 에 있으니 확인 가능하지만, **1차 이식 scope 는 MLP linear 만** 으로 한정하고 attention 은 후속.

### Pruning 은 외부 의존

SparAMX repo 는 **pruning 수행 안 함**. "이미 pruned 된 weight 를 sparse kernel 로 돌리는 역할" 만 담당. 사전 단계로 Wanda / SparseGPT 로 pruned checkpoint 생성해야 함.

---

## 리스크

### (A) `§04 WoQ INT8 기각 패턴 재현` — 최고 리스크

`SparseLinear.from_linear(nn.Linear)` 는 standard `nn.Linear` 가정. 그러나 vLLM 의 `QKVParallelLinear`, `MergedColumnParallelLinear` 는 `LinearBase` 계열 (multi-column fused, TP 지원). §04 IPEX WoQ 가 정확히 여기서 막혀서 기각됐음.

**완화**: §06 `_Q8_0LinearMethod` 가 `quant_method.apply` swap 으로 우회한 전략 동일하게 적용. `_SparAMXLinearMethod` wrapper 작성.

### (B) 3D ↔ 2D shape 불일치

SparAMX forward 는 `(batch, seq, in_features)` 3D. vLLM 은 `(num_tokens, hidden)` 2D. 어댑터 필요 (input unsqueeze, output squeeze 또는 내부 view 변경).

### (C) Pruning 후 PPL 열화

Qwen2.5 의 activation outlier 로 Wanda/SparseGPT 적용 시 2% 이상 열화 가능. Calibration set 조정 필요. → **preflight 로 선검증**.

### (D) MLP sparsity batch↑ 시 소멸 (Polar Sparsity)

서로 다른 input 이 서로 다른 neuron 활성화 → batch 전체에선 dense. **batch>1 에서 MLP sparse 이득 축소**. Attention head sparsity 는 batch-invariant 하지만 MLP 만 이식 시 batch scaling 이득 제한적.

### (E) Attention head pruning 의 생성 품질 영향

PPL 유지돼도 long-text coherence 손상 가능. MMLU/BBH 종합 평가 필요 (attention 이식 단계에서).

---

## 실행 계획 (preflight-first)

### Step 1 — Preflight (pruning + 정확도 검증)

- [ ] **Pruning library 선택**: Wanda (H100 ~수 시간) 또는 SparseGPT (~하루). Wanda 권장 (가볍고 빠름)
- [ ] **Qwen2.5-32B 50% unstructured pruned checkpoint 생성**
- [ ] **PPL 검증**: WikiText-2 에서 dense 대비 열화 <2%
- [ ] **layer 별 실제 zero ratio 측정** — 50% pruning 요청했을 때 각 layer 의 sparsity 실분포
- [ ] **(옵션) MMLU 일부 평가**: 장기 coherence 신호 확인

**산출물**: `tools/pruning/preflight/` 아래 script + 결과 JSON (layer zero ratio / PPL delta). `measurement_results/H100x8/g0_16_preflight_qwen2.5_32b/` 에 실측치.

**판정**:
- PPL 열화 <2% + layer zero ratio ≥ 40% → Step 2 진입
- PPL 열화 ≥2% → §22 NEO 또는 §28 xFT 로 이동. §16 보류

### Step 2 — SparAMX 이식 (MLP Linear 우선, attention 후속)

Preflight 통과 시:

- [ ] **SparAMX repo clone + build test** (standalone 환경에서 SparAMX 자체가 SPR 에서 동작 확인)
- [ ] **C++ kernel 이식**: `csrc/sparse_linear.cpp` + `csrc/avx_sparse_linear.cpp` → 우리 `csrc/cpu/sparse_amx.cpp` (또는 동일 이름 2 개 파일)
- [ ] **우리 `_C_cpu_ops` 에 torch op 등록**: `torch.ops._C_cpu_ops.sparse_linear_bf16` (또는 유사)
- [ ] **Weight format 변환 utility**: dense → (metadata / values / values_is) 3 tensor. Pruned checkpoint 로딩 시 1회 변환
- [ ] **`_SparAMXLinearMethod` wrapper**: `vllm/v1/worker/hot_path_wiring.py` 에 §06 `_Q8_0LinearMethod` 패턴 답습
- [ ] **3D↔2D shape 어댑터**: forward 진입 시 `input.unsqueeze(0)`, 출력 시 `.squeeze(0)` (또는 kernel 내부 2D 처리 수정)
- [ ] **patch 함수**: `patch_mlp_to_sparamx(model, hybrid_config, ...)` — Qwen2 MLP `gate_up_proj` / `down_proj` 만 대상 (§06 와 동일 scope)
- [ ] **CLI / HybridConfig flag**: `HybridConfig.sparse_bitmask: bool` + `--hybrid-sparse-bitmask` + `_create_cpu_vllm_config` passthrough (§06/§11 패턴)
- [ ] **torch_bindings + guard**: `HAS_CPU_OPS` + op registered + arch allowlist + LoRA off (§06 5겹 guard 답습)

### Step 3 — 측정 + 판정

- [ ] **gpu_only + hybrid sweep**: seqs 1/2/4/8/16 측정
- [ ] **정확도 smoke**: 100 prompt greedy top-1 token identical 검증
- [ ] **G1 / G2 재판정**:
  - G1: hybrid outTP ≥ base outTP at seqs 1/2/4/8/16
  - G2: hybrid outTP ≥ gpu_only × 0.30 at any seqs
- [ ] **Attention head pruning 확장** (Step 2 성공 시): `custom_llama_attention` 포팅 검토

### Step 4 (조건부) — 실패 시 이동

- vLLM Linear 호환성 막힘 (§04 패턴 재현) → §28 xFT 로 이동
- Kernel 이식 빌드 실패 지속 → §22 NEO 로 이동
- 측정에서 G1 미통과 → Tier 1 후보 우선순위 2 (§22) 재평가

---

## 성공 조건

**Preflight 단계**:
1. ✅ PPL 열화 <2% (WikiText-2)
2. ✅ layer zero ratio ≥ 40% (50% pruning 요청 시)

**이식 단계**:
3. ✅ Greedy top-1 token identical (또는 exact match ≥95% fallback)
4. ✅ Linear layer 단독 1.3× 이상 가속 (batch=1 decode, kernel 독립 측정)
5. ✅ seqs=1 에서 hybrid outTP ≥ base (G1 부분 통과)
6. ✅ Memory BW 감소 20~40% (`perf stat -e uncore_imc/mem_bw`)

**확장 (attention 포함)**:
7. ✅ attention head pruning 후 1.14× 이상 (batch=16)
8. ✅ batch 확장에도 attention sparse 이득 유지 (MLP 는 축소 인정)

---

## 의존성

- **선행**: §06 hot path wiring (infra 재사용, 완료), §06-1 v1 (kernel baseline, 완료)
- **병행 후보**: §15 AMX pre-pack (sparse pre-packed layout, Tier 2 보류 중이나 이식 단계에서 필요 여부 재평가)
- **후속**: §24 W8A8 (activation INT8) — `quantized_sparse_linear` extension 재사용 가능

---

## 기술적 배경 (참고)

### Unstructured vs Structured Sparsity

- **Unstructured**: random 위치 0. 정확도 유지 쉬움. SparAMX 는 bitmap + `_mm512_maskz_fmadd_ps` 로 해결
- **Structured (2:4, 4:8)**: NVIDIA Ampere+ 만 HW 지원. Intel CPU 는 software 구현 → unstructured 와 동일

### AMX 와 sparse 의 충돌

AMX 는 dense tile matmul 전제. Sparse 는 두 단계로 회피:
- **Coarse (tile level)**: mask all-zero 인 tile 은 `tdpbf16ps` 생략
- **Fine (element level)**: 부분 sparse tile 은 AVX-512 fallback 에서 `maskz_fmadd_ps`

### Pruning 방법 (preflight 에서 선택)

- **Wanda** (Sun et al. 2023, arXiv 2306.11695): activation-aware, calibration 빠름
- **SparseGPT** (Frantar et al. 2023, arXiv 2301.00774): second-order info, 정확도 우수 but 느림
- **Magnitude pruning**: 단순 baseline

---

## 관련 참고 문헌

- **SparAMX**: https://arxiv.org/abs/2502.12444 — AbouElhamayed et al., Xeon SPR 실측 linear 1.42× / attn 1.14×
- **SparAMX 공식 구현**: https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/tree/main/SparAMX (MIT)
- **Wanda pruning**: https://arxiv.org/abs/2306.11695
- **SparseGPT**: https://arxiv.org/abs/2301.00774
- **Michel et al. "Are Sixteen Heads Really Better than One?"**: https://arxiv.org/abs/1905.10650
- **Intel Intrinsics `_mm512_maskz_fmadd_ps`**: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- **Polar Sparsity**: batch-invariant sparsity 분석 (논문 참조)

---

## 실행 flag (현재 미연결)

| flag | 값 | 의미 | 현재 상태 |
|---|---|---|---|
| `HYBRID_SPARSE_BITMASK` | `0` (기본) / `1` | SparAMX bitmask sparse 경로 | **env / manifest 에만 존재. HybridConfig / CLI / worker 미연결** — Step 2 에서 구현 |

향후 추가 예정: `HybridConfig.sparse_bitmask: bool`, `--hybrid-sparse-bitmask` CLI, `_create_cpu_vllm_config` passthrough.

---

## 관련 코드 위치

**Preflight 단계**:
- 신규: `tools/pruning/preflight/wanda_prune.py` (Wanda 기반, Qwen2.5-32B)
- 신규: `tools/pruning/preflight/ppl_eval.py` (WikiText-2)
- 산출물: `measurement_results/H100x8/g0_16_preflight_qwen2.5_32b/`

**이식 단계** (Step 2 진입 후):
- 신규: `csrc/cpu/sparse_amx.cpp` (또는 `sparse_linear.cpp` + `avx_sparse_linear.cpp` 2 파일) — SparAMX impl 이식
- 수정: `csrc/cpu/torch_bindings_hybrid.cpp` — op 등록
- 수정: `vllm/v1/worker/hot_path_wiring.py` — `_SparAMXLinearMethod` + `patch_mlp_to_sparamx`
- 수정: `vllm/v1/worker/cpu_model_runner.py` — load_model 훅에 patch 호출
- 수정: `vllm/config.py`, `vllm/engine/arg_utils.py` — flag 추가
- 수정: `eval/serve.sh` — env → CLI 변환
- 신규 env: `eval/envs/g0_h100x8_qwen32b_16.env`

---

## 관련 문서

- [TODO.md §3-A](../TODO.md) — Tier 1 후보 우선순위
- [NinjaGap_Todo/README.md Tier 1 후보](README.md)
- [SSOT 원인 트리 / Gate 재정의](../Tech_done.md) v8 §SSOT-*
- [§22 NEO asymmetric](22_neo_asymmetric.md) — preflight 실패 시 이동 후보
- [§28 xFasterTransformer 이식](28_xft_kernel_porting.md) — vLLM Linear 호환성 실패 시 이동 후보
- [Tier 2 / 기각 / 보류 backlog](../old_doc/NinjaGap_backlog_tier2_20260420.md)
