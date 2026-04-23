# 23. CPU Native Quantization (llama.cpp Q8_0 / Q4_K dispatch)

**Tier**: 1 (G1 핵심)
**상태**: ⭕ 미구현 (build artifact 일부 존재 — `csrc/cpu/gemm_vnni.cpp`, `csrc/cpu/quant_q8_0.cpp`)
**관계**: **§06 의 구체 구현체**. §06 는 "hot path 연결 infra" 라는 방향, 본 §23 은 "실제로 어떤 quantization scheme 을 쓰고 어떻게 등록하는가" 의 실행 계획. §04 IPEX WoQ INT8 기각 이후 대체 경로.

---

## 왜 필요한가

실측 (H100x8 + Qwen2.5-32B):
- CPU engine step = 468 ms, 중 **MLP (gate_up + down) 71% = 336 ms**
- AMX 미활용 BF16 기반으로 추정되며, AVX-512 peak 224 TFLOPS vs H100 1456 TFLOPS (6.5× 격차) 의 전량을 쓰지도 못함
- 나머지 4.4× 성능 격차의 상당 부분이 "kernel 비효율 + WoQ 부재" 에서 유래

§04 (IPEX `ipex.llm.optimize` + WoqWeightDtype) 루트는 vLLM 의 `QKVParallelLinear` 구조와 비호환으로 2026-04-19 기각. 대안으로 **이미 자산으로 존재하는 VNNI INT8 / Q8_0 / Q4_K 커널을 vLLM Linear 에 직접 dispatch** 하는 경로가 유일한 실효 방법.

---

## 기술적 배경

### Quantization scheme 옵션

| scheme | weight 저장 | 디캡슐 | activation | 커널 | block | ROI (decode memory-bound) |
|---|---|---|---|---|---|---|
| Q8_0 (llama.cpp) | INT8 + fp16 scale | block 별 dequant | BF16/FP32 | VNNI dot | 32 elem | 1.8~2.5× |
| Q4_K (llama.cpp) | INT4 + scales | block 별 | BF16/FP32 | VNNI + LUT | 256 elem | 2.5~3.5× |
| W8A8 (full INT8) | INT8 + per-tensor scale | 불필요 | **INT8 (동적)** | VNNI / AMX-INT8 | — | 2.0~3.0× (§24 와 결합) |
| W4A16 (GPTQ 류) | INT4 + group scale | block 별 | BF16 | AMX-BF16 dequant | 128 | 2.0~2.8× |

본 §23 은 **Q8_0 + Q4_K 먼저 포팅** (llama.cpp 가 이미 SPR 에서 검증), W8A8 은 §24 로 분리.

### vLLM 통합 지점

`vllm/model_executor/layers/quantization/` 에 신규 config class 추가:
- `vllm/model_executor/layers/quantization/cpu_native.py` (신규)
  - `CPUNativeQuantConfig(QuantizationConfig)` — `q8_0`, `q4_k` scheme 선택
  - `get_supported_act_dtypes() → [torch.bfloat16]`
  - `get_min_capability()` — CUDA 전용이 아니므로 CPU-only 체크 로직 별도
  - `get_quant_method()` → `CPUNativeLinearMethod`
- `CPUNativeLinearMethod`:
  - `create_weights()` — quantized int8/int4 storage + scale tensor
  - `process_weights_after_loading()` — HF BF16 weight → Q8_0/Q4_K 로 사전 양자화 (1회)
  - `apply(x, layer)` — `torch.ops._C_cpu_ops.gemm_q8_0(..)` 호출

### Torch custom op 등록

`csrc/cpu/torch_bindings_hybrid.cpp` 에:
```cpp
m.def("gemm_q8_0(Tensor x, Tensor qweight, Tensor scales, ...) -> Tensor");
m.impl("gemm_q8_0", torch::kCPU, &gemm_q8_0_dispatch);
```

Python 쪽:
```python
torch.ops._C_cpu_ops.gemm_q8_0(x, qweight, scales, group_size)
```

### activation dtype 처리

llama.cpp 의 Q8_0 dot product 는 `int8 × int8 → int32 accumulate → BF16/FP32 scale` 이며 activation 도 INT8 로 양자화되어야 throughput 최대. 본 §23 에서는 **activation 은 BF16 유지** (dequantize-on-fly), **full W8A8 은 §24 에서**.

### pre-pack / repack

Q8_0 block format: `{int8[32], fp16 scale}` packed 구조. HF checkpoint 은 BF16 tensor — load 시점에 1회 변환 후 별도 buffer 유지. IPEX 의 `weights_prepack=True` 와 **직접 충돌** 하므로 Q8_0 적용 layer 는 IPEX optimize 경로에서 제외해야 함.

---

## 관련 참고 문헌

- **llama.cpp GGUF quantization schemes**: https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c — Q8_0, Q4_K_M 정확한 layout
- **GGML Q4_K 디자인**: https://github.com/ggerganov/llama.cpp/pull/1684
- **IPEX LLM WoQ INT8 post-mortem**: `NinjaGap_Todo/04_ipex_woq_int8.md` (왜 IPEX 루트가 막혔는지)
- **vLLM Quantization backend 구조**: `vllm/model_executor/layers/quantization/awq.py` — reference. AWQ 는 CUDA-only 지만 class 구조는 동일
- **본 프로젝트 기존 커널**: `csrc/cpu/gemm_vnni.cpp` (VPDPBUSD 6×16 micro-kernel), `csrc/cpu/quant_q8_0.cpp` (Q8_0 변환)
- **§06 Hot Path wiring**: 본 § 는 §06 의 Option C (torch custom op + post-load hook) 실체화

---

## 구체 작업

- [ ] **scheme 선택 확정**: Q8_0 먼저 (구현 간단, 이미 kernel 존재). Q4_K 는 후속.
- [ ] **`torch_bindings_hybrid.cpp` 검토** — `gemm_vnni` 가 torch op 로 등록돼 있는지, schema 가 Q8_0 에 맞는지
- [ ] **`quant_q8_0.cpp` 의 변환 함수 확인** — HF BF16 weight → Q8_0 block 변환 + reverse 검증
- [ ] **신규 파일 `vllm/model_executor/layers/quantization/cpu_native.py`**
  - `CPUNativeQuantConfig`, `CPUNativeLinearMethod` 구현
  - `--quantization cpu_native_q8_0` CLI 옵션 추가
- [ ] **CPU-only 체크** — `CpuPlatform` 에서만 활성화 (GPU engine 에서는 AWQ/Marlin 계속 사용)
- [ ] **Hybrid 연동** — GPU engine 은 BF16 원본, CPU engine 만 Q8_0 변환본 로드. 즉 2 copy of weight. 메모리 영향 평가
- [ ] **pre-pack cache** — 모델 최초 load 시 BF16→Q8_0 변환 시간 측정, 캐시 저장 (`~/.cache/vllm_hybrid/prepack/<model_hash>_q8_0.pt`)
- [ ] **dispatch marker**: `[HYBRID-KERNEL-Q8_0] layer=<name> M=.. N=.. K=.. time=..ms`
- [ ] **정확도 검증**: gpu_only (BF16) vs hybrid(CPU=Q8_0) 의 greedy 출력 일치도, MMLU/HellaSwag 변화 < 0.5pp
- [ ] **성능 측정**: `num_seqs=1/4/16` 에서 MLP sublayer ms 변화
- [ ] **IPEX bypass 경로**: Q8_0 Linear 는 IPEX optimize 대상에서 제외 (충돌 방지)

---

## 성공 조건

1. ✅ `[HYBRID-KERNEL-Q8_0]` 로그가 모든 MLP Linear 에서 찍힘 (IPEX fallback 0%)
2. ✅ MLP sublayer (gate_up + down) ms 가 현재 336ms → **≤ 170ms (2× 이상 감소)**
3. ✅ PPL 또는 greedy 출력 변화 < 0.5pp
4. ✅ 모델 로드 시간 +30s 이하 (Q8_0 변환 overhead 허용 범위)
5. ✅ GPU engine 동작 영향 없음 (BF16 경로 그대로)
6. ✅ `num_seqs>1` 에서도 per-req cost 유지 또는 감소 (scaling 확인)

---

## 의존성

- **선행**: §01 G0 계측, §05 OMP env, §06 hot path wiring (infra)
- **교체**: ~~§04 IPEX WoQ INT8~~ (기각, 본 § 가 대체 경로)
- **병행**: §07 ISA binary dispatch (AVX-512 vs AMX 분기 시 VNNI Q8_0 vs AMX Q8_0 선택)
- **후속**: §13 T-MAC LUT INT4 (Q4_K 의 LUT-기반 더 공격적 버전), §14 AVX/AMX Cascade
- **연관**: §24 (activation quant 추가하면 W8A8 full), §28 xFasterTransformer 참조

---

## 리스크

- **CPU 메모리 2배 (BF16 + Q8_0 사본)**: H100x8 서버 DDR5 용량 확인 필요. 32B BF16 61GB × 2 = 122GB per engine × 2 engine → 244GB. 서버 총 DDR 가 수 TB 급이면 문제 없음
- **정확도 열화**: Qwen2.5 의 outlier activation 이 Q8_0 block-level scale 로 표현 안 될 수 있음. group_size 작게 또는 Q4_K_M 검증
- **IPEX 충돌**: `ipex.optimize` 가 Q8_0 tensor 를 BF16 로 재변환하려 할 수 있음. Linear 단위 opt-out 필요
- **torch.compile graph break**: custom op 가 compile path 에 graph break 일으킬 가능성 — 측정
- **embed_tokens / lm_head 예외**: 2GB 근처 단일 tensor 는 block layout 이 맞지 않을 수 있음. BF16 유지

---

## 스택 호환성

- §06 Hot Path wiring: 본 § 가 §06 의 Option C 실체
- §07 ISA Binary Dispatch: 본 § kernel 을 AVX-512 전용으로 두고 AMX 경로는 §14 cascade 로
- §08 Kernel Fusion: Q8_0 gate_up fused kernel (gate 와 up 을 한 번에) 확장 가능
- §13 T-MAC LUT: Q4_K 로 전환 시 LUT GEMV 경로 병합

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_CPU_NATIVE_QUANT` | `0` (기본) / `q8_0` / `q4_k` | CPU 전용 양자화 scheme 선택 |

또는 CLI: `--quantization cpu_native_q8_0`.

전체 flag 테이블: [README.md](./README.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `csrc/cpu/gemm_vnni.cpp` — VNNI INT8 GEMM (재사용)
- `csrc/cpu/quant_q8_0.cpp` — Q8_0 변환 (재사용)
- `csrc/cpu/torch_bindings_hybrid.cpp` — torch custom op 등록 (보강)
- `vllm/model_executor/layers/quantization/cpu_native.py` — **신규**
- `vllm/model_executor/layers/quantization/__init__.py` — config 등록
- `vllm/v1/worker/cpu_model_runner.py::load_model` — IPEX optimize 전 Q8_0 변환 hook
- `vllm/_custom_ops.py` — `HAS_CPU_OPS` + `gemm_q8_0` callable 노출
