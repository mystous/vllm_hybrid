# 24. Dynamic Activation Quantization (W8A8 full pipeline)

**Tier**: 1/2 (§23 확장)
**상태**: ⭕ 미구현
**관계**: §23 (Weight-only Q8_0) 위에 **activation 도 INT8 동적 양자화** 를 추가해 full W8A8 를 달성. VNNI `VPDPBUSD` / AMX-INT8 peak throughput 을 실제로 끌어내는 필수 단계.

---

## 왜 필요한가

§23 (Weight Q8_0 + Activation BF16) 단계에서:
- Weight: INT8 로 읽고 load 시 BF16 로 dequantize-on-fly
- Matmul: `bf16 × bf16 → fp32 accumulate`
- VNNI `VPDPBUSD` 같은 INT8×INT8 경로는 **사용 안 됨** — WoQ 만으로는 peak throughput 도달 불가

W8A8 full 은:
- Activation 을 matmul 직전 FP32/BF16 → INT8 로 동적 양자화 (per-token scale)
- Matmul: `int8 × int8 → int32` (VNNI / AMX-INT8)
- 출력 dequantize: `int32 × (weight_scale × act_scale) → BF16`

실측 기준 MLP 336ms 중:
- §23 WoQ 만 적용 시 예상: **~170ms (2×)** — dequant 후 BF16 matmul 가 병목
- §24 W8A8 full 추가 시 예상: **~120ms (2.8×)** — INT8 matmul 로 peak throughput

즉 §23 대비 **추가 30~40% 절감** 여지.

---

## 기술적 배경

### Per-token activation quantization

```
activation_fp32  shape=(B, T, H)
├─ per-token scale  s_a = max(|x|) / 127    shape=(B, T, 1)
├─ q_act = round(x / s_a).to(int8)
└─ outlier channel → FP16 fallback (SmoothQuant)
```

- B=batch, T=token count, H=hidden. per-token 은 T 축 별 독립 scale
- outlier channel 보정 (SmoothQuant 류) 이 없으면 Qwen2.5 같은 모델에서 PPL 열화 가능

### VNNI INT8 matmul path

```
int8 act (B, T, H)
× int8 weight (H, N_out)     ← §23 Q8_0 block
= int32 accumulator (B, T, N_out)
→ int32 × (s_w * s_a) = fp32
→ BF16 output
```

- 현 `csrc/cpu/gemm_vnni.cpp` 는 weight-only; activation 경로 없음
- activation quant kernel 신규 필요 (`csrc/cpu/quant_act_int8.cpp` 예상)

### AMX-INT8 path (§14 cascade 와 결합)

SPR 의 AMX-INT8 tile (`_tile_dpbssd`):
```
TMUL int8[16, 64] × int8[64, 16] → int32[16, 16]
```

- 16×16 accumulator 가 1 cycle 에 512 ops
- VNNI (16-wide VPDPBUSD) 대비 8~16× peak

본 §24 는 VNNI W8A8 먼저, AMX-INT8 는 §14 cascade 에 위임.

### activation outlier 처리

Qwen2.5-32B 는 일부 layer 에 outlier magnitude 가 큼. 대처:
- **(A) per-token scale 만** — 정확도 열화 수용 (PPL +0.2~0.5)
- **(B) SmoothQuant** — weight 로 outlier 흡수 (offline calibration 필요)
- **(C) mixed precision** — outlier channel 을 FP16 로 별도 처리

본 § 첫 단계는 (A), 정확도 지표 확인 후 (B) 또는 (C) 추가.

---

## 관련 참고 문헌

- **SmoothQuant (Xiao et al., 2023)**: https://arxiv.org/abs/2211.10438 — activation outlier 를 weight 로 이전하는 post-training calibration
- **LLM.int8 (Dettmers et al., 2022)**: mixed precision activation quant
- **xFasterTransformer W8A8 kernel**: https://github.com/intel/xFasterTransformer/blob/main/src/kernels/ — Intel 공식 SPR W8A8 구현, 참조
- **Intel AMX Programming**: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html (TMUL intrinsics)
- **§23 CPU Native Quantization**: Weight-only 전제

---

## 구체 작업

- [ ] **정확도 pre-check**: SmoothQuant 없이 per-token scale 만으로 Qwen2.5-32B 의 MMLU 열화 측정 (Offline, pytorch fake-quant)
- [ ] **activation quant kernel**: `csrc/cpu/quant_act_int8.cpp` — FP32/BF16 → INT8 + scale (AVX-512 `_mm512_cvtps_epi8` 기반)
- [ ] **VNNI W8A8 GEMM**: `gemm_vnni_w8a8` custom op — int8 act × int8 weight + dual scale. `gemm_vnni.cpp` 확장
- [ ] **Linear method 확장**: `CPUNativeLinearMethod` 의 `apply` 에서 activation quant → W8A8 GEMM → dequant 순으로 chain
- [ ] **dispatch flag**: `HYBRID_CPU_NATIVE_QUANT=q8_0_w8a8` 로 activation quant 활성
- [ ] **outlier 측정**: 각 layer activation 의 max/mean 비율 기록 (debug mode)
- [ ] **SmoothQuant 옵션 (후속)**: offline migration factor 계산 → weight 에 적용 후 저장. `--quantization cpu_native_q8_0_sq`
- [ ] **성능 측정**: §23 대비 MLP sublayer ms 변화

---

## 성공 조건

1. ✅ `[HYBRID-KERNEL-W8A8]` 로그가 찍히고 VNNI VPDPBUSD 실행 확인 (`perf stat -e` 로 uops)
2. ✅ MLP sublayer ms 가 §23 (170ms) 대비 **≤ 130ms**
3. ✅ PPL 열화 <0.5pp (SmoothQuant 없이), < 0.2pp (SmoothQuant 적용)
4. ✅ 모델별 outlier 측정 로그로 문제 layer 식별 가능
5. ✅ `num_seqs≥4` 에서 per-req cost 추가 감소 (INT8 matmul 의 batch 이득 반영)

---

## 의존성

- **선행**: §23 CPU Native Quantization (weight Q8_0 가 먼저 동작해야 함)
- **병행**: §07 ISA dispatch (VNNI/AMX 선택)
- **후속**: §14 AVX/AMX Cascade (AMX-INT8 로 peak throughput)

---

## 리스크

- **정확도 열화가 특정 모델에서 큼**: Qwen2.5 outlier 가 per-token scale 만으로 커버 안 되면 SmoothQuant 필수. 추가 offline step
- **activation quant 자체 overhead**: 매 matmul 전에 quant kernel 추가 실행 → net gain 이 작아질 수 있음. `fused_quant_gemm` 형태로 합쳐야
- **Mixed precision 관리**: outlier channel 을 FP16 로 처리하면 matmul 이 두 경로로 쪼개짐 (INT8 + FP16) — layout 복잡도 증가
- **PPL 측정 시간**: 정확도 검증용 MMLU 는 CPU engine 에서 돌리면 몇 시간. 단축 방법 (samples=subset) 필요

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `HYBRID_CPU_NATIVE_QUANT` | `q8_0_w8a8` | §23 Q8_0 위에 activation W8 추가 |
| `HYBRID_W8A8_SMOOTHQUANT` | `0` (기본) / `1` | SmoothQuant offline migration 활성 |
| `HYBRID_W8A8_OUTLIER_LOG` | `0` / `1` | activation outlier 통계 기록 (debug) |

---

## 관련 코드 위치

- `csrc/cpu/gemm_vnni.cpp` — W8A8 경로 추가
- `csrc/cpu/quant_act_int8.cpp` — **신규** (activation quant kernel)
- `vllm/model_executor/layers/quantization/cpu_native.py` — §23 확장
- `vllm/v1/worker/cpu_model_runner.py` — SmoothQuant migration hook (offline path)
