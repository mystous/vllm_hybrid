# 07. ISA Binary Dispatch (AVX-512 ↔ AMX)

**Tier**: 1
**상태**: 🔶 부분 구현 (fallback chain 존재, 명시적 batch-based 없음)
**예상 이득**: decode 1.5–2.22× (KTransformers 실측)

---

## 왜 필요한가

AMX 는 "큰 matrix × 큰 matrix" 에 최적. AVX-512 는 "작은 matrix 또는 GEMV" 에 최적. 경계는 대략 **M (batch) > 4** 근처:

| Shape | AVX-512 VNNI | AMX BF16 | 승자 |
|---|---|---|---|
| M=1 GEMV | 빠름 | 느림 (tile 초기화 + clock down) | AVX-512 |
| M=4 | 대등 | 대등 | 상황 |
| M=16 GEMM | 느림 | 빠름 (tile 가득 활용) | AMX |

**실패 2** (Claude 3겹 진단): AMX 고정 시 `batch=1` 에서 AVX-512 대비 **2.22× 손해** (KTransformers 실측).

현재 `cpu_attn.py` decode 경로에 `custom_avx → ipex → sdpa_batched → sdpa_loop` fallback chain 존재하지만 **batch size 기반 명시적 dispatch 가 아님** — IPEX 내부 dispatcher 의존.

---

## 기술적 배경

### Intel ISA 발전 경로

| ISA | Year | 주요 추가 | 레지스터 |
|---|---|---|---|
| AVX-512 F | 2017 (Skylake-X) | 512-bit SIMD | zmm0-31 |
| AVX-512 VNNI | 2019 (Cascade Lake) | `VPDPBUSD`, `VPDPWSSD` INT8/INT16 dot product | zmm |
| AVX-512 BF16 | 2020 (Cooper Lake) | `VCVTNE2PS2BF16`, `VDPBF16PS` | zmm |
| **AMX** | 2023 (Sapphire Rapids) | Tile register (16×64 bytes × 8), `TDPBSSD` (INT8), `TDPBF16PS` (BF16) | tmm0-7 |

### AMX tile 초기화 비용

- `ldtilecfg` → `tileloadd` → `tdpbsxd` → `tilestored` → `tilerelease`
- **Tile register 전환에 ~수백 cycle** (processor state save/restore)
- 소규모 matmul 에서는 고정 비용 > 이득

### KTransformers dispatch 방식

```
batch_size <= 4 → AVX-512 VNNI (decode)
batch_size > 4  → AMX BF16 (prefill + batched decode)
```

실측:
- batch=1: AVX-512 VNNI 가 AMX BF16 대비 2.22× 빠름
- batch=16: AMX BF16 이 AVX-512 대비 5× 빠름 (tile 가득 활용)

### oneDNN 의 ISA 설정

`ONEDNN_MAX_CPU_ISA` 환경변수:
- `AVX512_CORE_AMX` — AMX 우선 (SPR 에서 기본)
- `AVX512_CORE_BF16` — AMX 제외, AVX-512 BF16
- `AVX512_CORE_VNNI` — INT8 VNNI
- `AVX512_CORE` — VNNI/BF16 제외

**문제**: `ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX` 설정이 있다고 해서 oneDNN 이 모든 shape 에 AMX 를 쓰는 것은 아님. 내부 dispatcher 가 shape 별 cost model 을 가지는데, 그 model 이 작은 M 에서 AMX 를 선택하는 경향 (clock down 무시).

### Explicit batch-based dispatch

우리가 직접 제어:
```cpp
// csrc/cpu/dispatch.cpp (신규)
if (batch_size > threshold) {
    return amx_bf16_gemm(A, B, C, ...);
} else {
    return avx512_vnni_gemm(A, B, C, ...);
}
```

또는 Python 쪽:
```python
def linear_forward(x, weight, ...):
    M = x.shape[0]
    if M > 4:
        return torch.ops._C_cpu_ops.amx_bf16_gemm(x, weight)
    else:
        return torch.ops._C_cpu_ops.avx512_vnni_gemm(x, weight)
```

### IPEX 내부 dispatcher 충돌

IPEX 의 `_IPEXLinearFusionCPU` 가 자체적으로 kernel 선택. 우리 dispatch 를 먹이려면:
- Option A: IPEX 치환 **전에** 우리 dispatch 설치 (Linear 를 직접 대체)
- Option B: IPEX 의 custom primitive 로 등록 (어려움, 버전 의존)

---

## 관련 참고 문헌

- **KTransformers AMX doc**: https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md
- **KTransformers SOSP'25 paper**: https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf
- **Intel AMX Programming Guide**: https://cdrdv2-public.intel.com/671368/architecture-instruction-set-extensions-programming-reference.pdf §Chapter 3 "Intel AMX"
- **Intel Optimization Reference Manual §20 AMX**: https://www.intel.com/content/www/us/en/content-details/814198/intel-64-and-ia-32-architectures-optimization-reference-manual.html
- **oneDNN ISA dispatcher**: https://oneapi-src.github.io/oneDNN/dev_guide_dispatcher_control.html
- **SGLang + KTransformers blog**: https://lmsys.org/blog/2025-10-22-KTransformers/ — dynamic AMX/AVX-512 switching
- **AMX clock down analysis**: Intel 은 공식적으로 AMX clock penalty 를 "minimal" 이라 주장하지만, KTransformers 실측은 차이 존재를 시사
- **현재 코드**: `vllm/v1/attention/backends/cpu_attn.py` 의 `_decode_path_counts` (fallback chain counter)

---

## 구체 작업

- [ ] **§06 hot path wiring 선행 확인**
- [ ] **AMX BF16 GEMM kernel 구현** (`csrc/cpu/amx_bf16_gemm.cpp` — tile layout 이용)
- [ ] **AVX-512 VNNI GEMM kernel 기존 확인/보강** (`csrc/cpu/gemm_vnni.cpp` — INT8 이미 있음, BF16 도 필요하면 추가)
- [ ] **Dispatch 함수 등록**: `torch.ops._C_cpu_ops.cpu_gemm_dispatch(x, w, ..., batch_size)` 가 threshold 에 따라 분기
- [ ] **Threshold 탐색 실험**: `batch=1,2,3,4,5,6,8,16` 각각에서 AVX-512 vs AMX 비교. G0 계측 결과로 threshold 확정 (예상 4 근처)
- [ ] **Linear module patch**: 선별된 layer 를 dispatch 함수로 치환
- [ ] **`[HYBRID-KERNEL-DISPATCH]` 로그**: `shape=... chose=avx512_vnni|amx_bf16 time=...`
- [ ] **IPEX dispatcher 와의 충돌 검증**: 우리 dispatch 가 Linear 치환 후에도 IPEX 의 다른 fusion (Residual+Norm 등) 과 공존하는지

---

## 성공 조건

1. ✅ `num_seqs=1` 에서 AVX-512 VNNI 가 기본 — AMX 대비 최소 1.5× 빠름
2. ✅ `num_seqs>=8` 에서 AMX BF16 사용 — AVX-512 대비 최소 3× 빠름
3. ✅ Threshold 경계 (batch=4 근처) 에서 dispatch 정확히 작동
4. ✅ decode 전반 1.5–2.22× 개선
5. ✅ `num_seqs` 증가 시 per-req cost 가 AMX path 에서 선형 이하로 증가 (batch scaling 이득의 시작)

---

## 의존성

- **선행**: §06 hot path wiring (우리 kernel 을 호출할 infra)
- **병행**: §15 AMX pre-pack (AMX 가 연산하기 전 weight 를 tile layout 으로 준비)
- **후속**: §14 AVX/AMX cascade 의 전제 (binary dispatch → pipeline dispatch)

---

## 리스크

- **IPEX 내부 dispatcher 충돌**: 우리가 Linear 를 직접 치환하면 IPEX 의 fused attention / MLP 최적화 잃음. 해결: fusion 은 §08 에서 별도 구현
- **Threshold 가 shape / model 별로 다름**: batch=4 가 만능 아님. tuning 이 model/hardware specific
- **AMX clock down**: BF16 AMX 이 주변 core 의 AVX-512 주파수를 약간 낮춘다는 미확정 보고. H100x8 2-NUMA engine 이 각자 AMX 사용 시 서로 간섭 가능 (NUMA 교차 영향 낮음)
- **IPEX 를 끄고 직접 구현이 더 느릴 위험**: IPEX 는 layer fusion + weight prepack + scratchpad caching 등 복합 최적화. 우리 dispatch 만 켜고 fusion/prepack 없으면 IPEX 보다 느림

---

## 스택 호환성

- §08 Kernel Fusion: 각 dispatch 안에서 fused kernel 호출
- §13 T-MAC LUT GEMV INT4: 또 하나의 dispatch 분기로 추가
- §14 AVX/AMX Cascade: binary dispatch → 3-stage pipeline dispatch 로 확장
- §15 AMX Pre-pack: AMX path 선택 시 pre-packed weight 사용

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_ISA_DISPATCH` | `auto` (기본) / `avx512` / `amx` / `cascade` | Dispatch 정책 |

전체 flag 테이블: [README.md](./README.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `csrc/cpu/gemm_vnni.cpp` — AVX-512 VNNI
- `csrc/cpu/amx_bf16_gemm.cpp` — (신규) AMX BF16
- `csrc/cpu/dispatch.cpp` — (신규) shape-based dispatch
- `csrc/cpu/torch_bindings_hybrid.cpp` — 등록
- `cmake/cpu_hybrid_extension.cmake` — 빌드 대상 추가
- `vllm/v1/attention/backends/cpu_attn.py` — `_decode_path_counts` 확장
- `vllm/v1/worker/cpu_worker.py` — execute_model pre-dispatch hook
