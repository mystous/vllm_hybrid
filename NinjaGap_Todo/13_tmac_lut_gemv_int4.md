# 13. T-MAC LUT-Based GEMV INT4

**Tier**: 2
**상태**: ⭕ 미구현
**예상 이득**: INT4 **4×** (T-MAC 실측, CPU 22 tok/s > NPU 10.4 tok/s)
**근거 등급**: D → 강한 가설 (SPR+AMX 재검증 필수)

---

## 왜 필요한가

INT4 quantization 으로 weight 8× 작아짐 (BF16 대비). Memory-bound decode 에서 DDR read 8× 감소 → 이론상 8× throughput. 그러나 전통적 dequant-then-compute 경로는:
```
INT4 weight 읽기 → BF16 dequant (vectorized) → BF16 matmul (AMX)
```
dequant 가 compute 와 분리되어 **dequant overhead** 발생. T-MAC 은 이 overhead 를 **LUT precompute** 로 제거.

**Ninja Gap 관점 가장 큰 단일 이득**. 경로 1 누적 이론 상한 35× 중 3.0× 단독 기여.

---

## 기술적 배경

### Dequant-then-compute 한계

```cpp
// 전통적 W4A16
int4_weight = load_compressed(W, idx);     // 4-bit
bf16_weight = dequant(int4_weight, scale); // expand 4 → 16 bit
result += bf16_weight * bf16_input;        // AMX BF16 GEMM
```

- 매 layer 마다 dequant 가 weight read 뒤에 끼어듦
- AMX tile layout 과 dequant 결과 layout 이 다르면 repack 추가
- dequant 와 matmul 이 하나의 fused primitive 가 아니면 L2 ping-pong

### T-MAC 의 LUT 방식

**핵심 아이디어**: INT4 값의 가능한 조합 (`2^4 = 16`) 마다 **input 과 미리 곱한 결과** 를 precompute → runtime 은 `vpshufb` 1-cycle lookup.

```
LUT[16 entries] = {
    input * 0,
    input * 1,
    input * 2,
    ...
    input * 15,  // actually signed: -8..7
}

Runtime:
weight_idx = load_int4(W, idx)         // 4-bit index
result += LUT[weight_idx]              // vpshufb, 1 cycle
```

- Dequant 없음 — **곱셈 자체가 lookup**
- `vpshufb` 는 port 5, 1 cycle throughput
- 512-bit zmm 에 4×128-bit lane → 64 parallel lookups

### Group-wise LUT

Weight 는 64/128 단위 group 으로 scale 공유. 각 group 마다:
1. Input block (input tile) 을 읽음 (예: 64 elements)
2. LUT[16] 을 **input × {-8..7} * scale** 로 precompute (16 × 64-element 곱)
3. Weight INT4 index 로 lookup → accumulate

**Amortization**: LUT 계산은 group 당 1회, lookup 은 group 내 weight 개수만큼.

### AMX 와의 조합

T-MAC 은 원래 edge CPU (Snapdragon ARM) 에서 검증. x86 AMX 와 조합:
- AMX tile 은 matmul 전용 — LUT 연산과 직접 결합 불가
- 하지만 **AVX-512 `vpshufb` LUT + AMX matmul 의 hybrid kernel** 가능
  - 초입: INT4 → LUT 결과 (BF16 partial) 를 zmm register 에 생성
  - zmm → tile register 로 `tileloadd` (매 16 cache line)
  - AMX tile matmul 로 accumulate
- 이 hybrid 는 §14 AVX/AMX Cascade 와 자연스럽게 이어짐

### SPR+AMX 재검증 필요성

T-MAC 실측 수치 (4×) 는:
- Snapdragon X Elite (ARM, NPU 10.4 tok/s 대비 CPU 22 tok/s)
- Intel Meteor Lake (edge x86, AVX-512 없음)

Sapphire Rapids (SPR, server CPU) + AMX 조합 성능은 **미검증**. 가능성:
- AMX 가 대형 GEMM 에서 더 빠르므로 T-MAC LUT 상대 이득 축소 가능
- 하지만 memory BW 절감 (INT4 vs BF16) 은 절대적
- 실험으로 확정해야 할 부분

### INT4 quantization 파이프라인

Weight 를 INT4 로 만드는 방법:
- **GPTQ** (Frantar et al.): calibration-based, second-order info 사용, 정확도 우수
- **AWQ** (Lin et al.): activation-aware, outlier protect
- **Round-to-nearest (RTN)**: 단순, 정확도 낮음
- **HQQ (Half-Quadratic Quant)**: zero-shot, 빠른 quant

기존 HuggingFace 에 GPTQ/AWQ quantized Qwen2.5 model 존재 (`Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`). 이걸 사용.

---

## 관련 참고 문헌

- **T-MAC (EuroSys'25)**: Wei et al. "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge" https://arxiv.org/pdf/2407.00088
- **T-MAC GitHub**: https://github.com/microsoft/T-MAC
- **Microsoft Research T-MAC page**: https://www.microsoft.com/en-us/research/publication/t-mac-cpu-renaissance-via-table-lookup-for-low-bit-llm-deployment-on-edge/
- **T-MAN (arXiv 2511.11248)**: https://arxiv.org/html/2511.11248v1 — NPU 확장, 3-stage pipeline
- **LUT Tensor Core (ISCA'25)**: https://dl.acm.org/doi/10.1145/3695053.3731057
- **GPTQ paper (Frantar et al. 2023)**: https://arxiv.org/abs/2210.17323
- **AWQ paper (Lin et al. 2023)**: https://arxiv.org/abs/2306.00978
- **HQQ**: Mobius Labs — https://github.com/mobiusml/hqq
- **Intel Intrinsics Guide `vpshufb`**: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- **Codex playbook Tier 2 LUT low-bit path**: `/vllm_hybrid/ideation/20260415_094148_codex_ninja_gap_modification_playbook.md`
- **Claude Part 2-8 T-MAC LUT GEMV**: `/vllm_hybrid/ideation/20260415_094130_claude_ninja_gap_comprehensive_plan.md`

---

## 구체 작업

### Quantization 파이프라인
- [ ] **GPTQ/AWQ Qwen2.5-7B-Instruct-Int4 weight 준비**: HuggingFace 에 기존 존재
- [ ] **Group size 결정**: GPTQ 기본 128, T-MAC 은 64 권장 — 측정으로 결정
- [ ] **Weight format 변환**: HuggingFace safetensors → T-MAC 의 LUT-friendly layout
  - group-wise packing: (n_groups, group_size/2) INT4 pairs
  - scale/zero: (n_groups,) BF16

### LUT kernel 구현
- [ ] **`csrc/cpu/lut_gemv.cpp`** (신규)
  - `lut_gemv_int4_bf16(input, weight_int4, scales, output, M, N, K, group_size)`
  - Group loop: 각 group 마다 LUT precompute + inner GEMV
  - LUT 계산: `lut[16] = input * {-8..7} * scale` (BF16, zmm resident)
  - Inner GEMV: `vpshufb(lut, weight_int4_indices)` 를 K 방향 accumulate
- [ ] **Tiling strategy**: L1 fit 을 위한 group/M/N 축 tile
- [ ] **Prefetch**: 다음 group weight 를 `prefetcht2`
- [ ] **IPEX bypass**: 이 kernel 은 IPEX optimized Linear 를 완전히 대체

### 통합
- [ ] **torch op 등록**: `torch.ops._C_cpu_ops.lut_gemv_int4_bf16`
- [ ] **Linear module patch**: W4A16 quantized model 로드 시 우리 kernel 로 치환
- [ ] **Load-time pre-pack cache**: INT4 weight 를 T-MAC group layout 으로 1회 변환 + `~/.cache/vllm_hybrid/prepack/<model>_tmac.pt`

### 검증
- [ ] **정확도**: GPTQ reference (IPEX INT4) vs LUT kernel 결과 <1e-2
- [ ] **PPL**: WikiText-2 / MMLU 에서 BF16 baseline 대비 열화 <2%
- [ ] **Throughput**: batch=1 decode tok/s, batch=4/16 scaling
- [ ] **AMX 대비**: 동일 weight (INT4) 에 대해 T-MAC LUT vs AMX BF16 (dequant 후) 비교
- [ ] **SPR+AMX 재검증 핵심**: edge CPU 실측 4× 가 SPR 에서 어느 수준인지 확정

---

## 성공 조건

1. ✅ PPL 열화 <2%
2. ✅ decode batch=1 throughput 3× 이상 (BF16 baseline 대비)
3. ✅ Memory footprint 1/4 (INT4 + scale)
4. ✅ `batch_scaling_ratio` 가 BF16 path 보다 개선 (weight memory 여유 → scaling 친화)
5. ✅ SPR+AMX 에서 T-MAC 효과 확인 (edge 대비 degrade 2× 이상 발생 시 D → 드롭)

---

## 의존성

- **선행**: §01 G0 계측, §06 hot path wiring, §09 LUT infra (`lut_ops.cpp`)
- **대체 관계**: §04 IPEX WoQ INT8 은 본 기법 성공 시 **대체됨** (INT4 가 INT8 보다 2× 더 memory 절감)
- **병행**: §14 AVX/AMX Cascade (LUT + tile matmul hybrid), §15 AMX Pre-pack (T-MAC 전용 layout)
- **후속**: 장기적으로 INT3, INT2 로 확장 가능 (bit↓ 선형 가속)

---

## 리스크

- **⚠ SPR+AMX 재검증 실패** (D → 드롭): edge CPU 와 SPR 은 memory hierarchy 차이 (L3 210MB vs 24MB). SPR 은 AMX 가 더 강해서 T-MAC 이득 축소 가능. **실험 없이는 알 수 없음**
- **⚠ Staging overhead**: LUT precompute 비용이 group 크기 대비 크면 이득 상쇄. group=128 이상일 때만 positive
- **GPTQ/AWQ PPL 열화**: Qwen2.5 의 outlier activation 이 INT4 에서 정확도 유지 어려울 수 있음. AWQ 권장
- **T-MAC 소스 코드 x86 포팅**: T-MAC 원본은 ARM SVE / NEON 중심. `vpshufb` 기반 x86 이식이 어느 정도 되어 있는지 GitHub 확인 필수
- **Linear module patch 가 IPEX 와 충돌**: §06 hot path wiring 과 동일 문제, 해결 패턴 공유

---

## 스택 호환성

- §04 WoQ INT8 **대체**
- §06 Hot Path Wiring 의 torch custom op infra 재사용
- §09 LUT Softmax 와 동일 `lut_ops.cpp` infra
- §14 AVX/AMX Cascade 의 자연스러운 확장 (LUT → tile matmul pipeline)
- §15 AMX Pre-pack 의 T-MAC 전용 group layout

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_TMAC_LUT_INT4` | `0` (기본) / `1` | T-MAC LUT GEMV INT4 경로 |

전체 flag 테이블: [README.md](./README.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `csrc/cpu/lut_gemv.cpp` — (신규)
- `csrc/cpu/lut_ops.cpp` — (공통 LUT infra, §09)
- `csrc/cpu/torch_bindings_hybrid.cpp` — 등록
- `vllm/model_executor/layers/linear.py` — Linear patch
- `vllm/v1/worker/cpu_model_runner.py` — load-time pre-pack hook
- 참조: T-MAC GitHub `t-man` 디렉토리 (https://github.com/microsoft/T-MAC/tree/main/t-man)
