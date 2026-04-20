# 28. xFasterTransformer Kernel 이식 (참조 구현 경로)

**Tier**: 1 또는 2 / **근거 Tier 1 후보** (Intel 공식 SPR CPU LLM stack)
**상태**: ⭕ 미구현 (외부 repo 참조만)
**근거 등급**: **B** (Intel 공식 검증된 구현). Intel 블로그에 SPR 실측 수치 공개.
**관계**: **§23 + §24 + §14 + §08 을 한 번에 해결하는 alternative route**. 자체 구현 대신 Intel 이 공식 유지하는 xFasterTransformer (xFT) 의 검증된 커널을 이식.
**우선순위 근거**: 2026-04-20 Tier 1 후보 정리 시 선정. 자체 kernel 시도 (§06-1 v2, §11 Phase 1) 연속 실패 후 안정적 alternative. 구현량 큼 (submodule 통합 + build 연결) 대신 성공 확률 높음.

---

## 왜 필요한가

§23 (CPU native quantization), §24 (W8A8), §14 (AMX cascade), §08 (kernel fusion) 을 자체 구현하면:
- 각 기법당 수 주 ~ 수 개월
- 상호 interaction 미검증
- SPR specific bug 를 스스로 찾아야 함

**Intel xFasterTransformer (xFT)** 는 이미 SPR 에서 검증된 CPU LLM inference stack:
- AMX-BF16, AMX-INT8, AVX-512 VNNI 커널 모두 포함
- LLaMA, Qwen, ChatGLM 구조 지원
- paged attention + fused layernorm + W8A8/W4A16 quantization
- Intel 공식 유지 (2024년부터 active development)

**방향**: 자체 구현을 **xFT 커널 포팅 + vLLM Linear 에 dispatch** 로 대체. vLLM 의 model 구조 유지 + xFT 의 커널 품질 활용.

---

## 기술적 배경

### xFT 아키텍처 요약

```
python/xfastertransformer/automodel.py
  └─ model 구조 (LLaMA/Qwen 등)
      └─ src/layers/*.cpp (attention, mlp)
          └─ src/kernels/*.cpp
              ├─ amx_gemm.cpp (BF16/INT8 tile)
              ├─ bert_util.cpp (fused layernorm)
              ├─ paged_attention.cpp
              └─ ...
```

vLLM 과의 차이:
- xFT 는 **모델 구조 자체** 도 재정의 (HF checkpoint 을 자체 tensor format 으로 변환). vLLM 은 `QKVParallelLinear` 등 자체 구조.
- 따라서 xFT 의 **kernel only** 를 추출해 vLLM Linear/Attention 의 backend 로 호출하는 bridge 가 필요.

### 이식 범위 (커널만)

| xFT source | 이식 대상 vLLM 경로 |
|---|---|
| `src/kernels/amx_gemm.cpp` | `csrc/cpu/gemm_vnni.cpp` 와 통합 or 대체 (§23/§24) |
| `src/kernels/paged_attention.cpp` | `csrc/cpu/batch_attention.cpp` 와 통합 (§25) |
| `src/kernels/bert_util.cpp` (fused layernorm) | `csrc/cpu/` 신규 파일 (§08 의 실체) |
| `src/layers/mlp.cpp` 의 gate_up fused | `csrc/cpu/gemm_vnni.cpp` 확장 (§08 fusion) |

### Weight layout 변환

xFT 는 자체 layout (tile-friendly) 로 weight 저장. vLLM 의 BF16 HF layout → xFT layout 은 one-time 변환 (§23 의 pre-pack 과 동일 성격).

### Torch custom op bridge

```cpp
// csrc/cpu/xft_bridge.cpp
#include "xft/kernels/amx_gemm.h"

Tensor xft_amx_gemm_wrapper(Tensor x, Tensor xft_weight, Tensor scales) {
    // xft kernel call
    xft::amx_gemm_int8(x.data_ptr(), ...);
    return output;
}

TORCH_LIBRARY_FRAGMENT(_C_cpu_ops, m) {
    m.def("xft_amx_gemm(Tensor x, Tensor w, Tensor s) -> Tensor", &xft_amx_gemm_wrapper);
}
```

### 라이선스

xFT: **Apache-2.0**. vLLM 도 Apache-2.0. **호환**. 소스 재사용 가능 (저작권 고지 포함).

---

## 관련 참고 문헌

- **xFasterTransformer repo**: https://github.com/intel/xFasterTransformer
- **xFT 성능 블로그 (Intel)**: https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Run-Large-Language-Models-on-CPU-with-xFasterTransformer/post/1573869
- **xFT vs IPEX 벤치마크** (외부): 7B 기준 SPR 에서 IPEX 대비 1.2~1.5× (decode)
- **xFT W8A8 구현**: `src/kernels/amx_gemm_int8.cpp`
- **§23, §24, §25 와의 관계**: 본 § 는 이들 self-implementation 의 대안 경로. 병행 추진 가능.

---

## 구체 작업

- [ ] **xFT repo 빌드 시도**: SPR 에서 `cmake + make` 성공 확인, 테스트 포함
- [ ] **커널 API 매핑 표 작성**: xFT 의 kernel 함수 → vLLM Linear method 호출점
- [ ] **bridge 코드 `csrc/cpu/xft_bridge.cpp` 신규**: xFT header include + torch custom op wrapper
- [ ] **cmake 통합**: `cmake/cpu_hybrid_extension.cmake` 에 xFT source 링크 옵션
- [ ] **Weight layout 변환기**: HF BF16 → xFT layout (one-time, `~/.cache/vllm_hybrid/xft_prepack/`)
- [ ] **vLLM Linear dispatch**: §23 의 `CPUNativeLinearMethod.apply` 에서 xFT kernel 호출 옵션 추가 (`scheme=xft`)
- [ ] **fused layernorm**: §08 과 결합. `csrc/cpu/fused_norm.cpp` 신규 또는 xFT `bert_util` 직접 link
- [ ] **paged attention**: §25 와 결합. xFT `paged_attention.cpp` 를 reference 로 또는 직접 wrapper
- [ ] **정확도 검증**: xFT kernel 출력 vs GPU BF16 allclose
- [ ] **성능 비교**: §23 self-implemented vs xFT (어느 쪽이 SPR 에서 더 빠른지)

---

## 성공 조건

1. ✅ xFT AMX-INT8 kernel 이 vLLM CPU Linear 에서 호출됨
2. ✅ MLP sublayer ms 가 §23 self-impl 대비 **동등 또는 우수**
3. ✅ PPL 변화 < 0.5pp
4. ✅ 빌드 파이프라인에 xFT submodule 통합 (CMake 의존성 깔끔)
5. ✅ license 호환 (Apache-2.0 × Apache-2.0) 확인

---

## 의존성

- **선행**: §06 Hot Path wiring infra (Linear dispatch 구조)
- **병행/대체**: §23 self-implementation — xFT 가 동등하거나 우수하면 §23 포기 가능
- **통합 대상**: §08 fused layernorm, §25 GQA attention, §14 AMX cascade

---

## 리스크

- **xFT 의 모델 구조 가정**: xFT 가 HF checkpoint 을 자체 layout 으로 변환하는데, QKVParallelLinear 같은 vLLM custom 구조는 xFT 에서 읽어들일 수 없음. **kernel 만 추출** 가능한지가 관건
- **Header/ABI 안정성**: xFT 는 공식 C++ API 로 제공되지만 vLLM 빌드 시스템과 의존성 충돌 가능. cmake fetch_content vs submodule vs prebuilt .so 중 선택
- **tile intrinsics 의존**: xFT 가 glibc 특정 버전 / gcc 13+ / intel toolchain 에 의존할 수 있음. H100 서버 환경 호환성 검증
- **유지보수 부담**: xFT 업데이트 시 bridge 코드도 따라가야
- **성능 불확실**: self-impl 이 오히려 vLLM 구조에 맞아 빠를 가능성

---

## 스택 호환성

- §23 CPU Native Quantization: 동일 Linear method 내에서 "scheme=self / scheme=xft" 선택지
- §08 Kernel Fusion: xFT 의 fused kernel 재사용으로 fusion 작업 생략
- §14 AMX Cascade: xFT 의 AMX tile 경로로 대체
- §25 GQA attention: xFT 의 paged attention 참조 또는 직접 사용

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `HYBRID_CPU_NATIVE_QUANT` | `xft_int8` / `xft_bf16` | §23 의 scheme 옵션으로 통합 |
| `HYBRID_XFT_KERNEL` | `0` (기본) / `1` | xFT bridge 활성 |

---

## 관련 코드 위치

- `csrc/cpu/xft_bridge.cpp` — **신규**
- `cmake/cpu_hybrid_extension.cmake` — xFT dependency 추가
- `third_party/xFasterTransformer/` — submodule 위치 후보
- `vllm/model_executor/layers/quantization/cpu_native.py` — scheme 분기 추가
- `csrc/cpu/gemm_vnni.cpp` — self-impl 과의 공존/선택
