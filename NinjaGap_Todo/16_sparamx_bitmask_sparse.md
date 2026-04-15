# 16. SparAMX AVX-512 Bitmask Sparse

**Tier**: 2
**상태**: 🔶 부분 구현 (dense `int8_gemm_vnni` 존재, sparse 경로 없음)
**예상 이득**: linear 1.42×, attention 1.14× (SparAMX 실측, Xeon SPR)

---

## 왜 필요한가

LLM weight 에는 자연적 sparsity 가 존재:
- Dense pruning 50% 까지 정확도 열화 <1% (Sun et al. "A Simple and Effective Pruning Approach for LLMs")
- Attention head 절반 이상이 실제 기여 미미 (Michel et al. "Are Sixteen Heads Really Better than One?")

Sparse GEMM 을 통해:
- Weight memory 절반 (zero-pad 대신 mask)
- 연산 절반 (유효 원소만)
- **Memory bandwidth 절감** (decode 는 memory-bound → 직접 효과)

SparAMX 가 Xeon SPR 에서 linear **1.42×**, attention **1.14×** 실측 (H100 에 준하는 production CPU). Sparse + AMX 결합으로 batch scaling 에도 기여.

---

## 기술적 배경

### Unstructured vs Structured Sparsity

**Unstructured** (random 위치 0):
- 정확도 유지 쉬움
- Memory layout 복잡, 실제 속도 이득 어려움
- **SparAMX 방식**: bitmask + `_mm512_mask_fmadd_ps` 로 해결

**Structured** (2:4, 4:8 등 규칙):
- NVIDIA Ampere+ 의 hardware 지원
- Intel CPU 는 hardware 지원 없음 → software 로 unstructured 와 동일
- 정확도 약간 열화

### SparAMX Bitmask 방식

**Weight format**:
```
W_dense[M, N]  →  (W_compact[nnz], mask[M, N/64] of uint64_t)
```

mask 비트 0/1 이 weight 위치의 zero/nonzero 를 표현. `nnz = popcount(mask)`.

**Runtime**:
```cpp
for (k = 0; k < K; k += 64) {
    uint64_t m = mask[row][k/64];
    __m512i masked_fma = _mm512_maskz_fmadd_ps(m, input[k:k+16], weight[k:k+16], accum);
    // k 위치에 weight[k] 가 0 이면 mask bit 0 → fmadd 생략
}
```

- `_mm512_maskz_fmadd_ps` 는 **K register (64-bit mask)** 를 predicate 로 사용
- 0 원소는 연산도 memory fetch 도 skip

### AMX 확장

AMX 기본은 dense tile matmul. Sparse 확장은:
- Tile 단위 mask 검사: tile 의 모든 원소가 0 이면 `tdpbf16ps` 생략
- Sub-tile sparsity: tile 내부 부분 0 — AMX 자체는 full tile 연산이므로 0 을 skip 할 수 없음 → AVX-512 fallback 필요

**SparAMX 실제 구현**:
- Coarse mask (tile level) 로 AMX tile 전체 skip
- Fine mask (element level) 로 AVX-512 fallback 경로에서 `maskz`

### Batch-invariance 차이

- **MLP sparsity**: batch↑ 시 소멸 (Polar Sparsity paper). 서로 다른 input 이 서로 다른 neuron 활성화 → batch 전체에선 dense. **batch>1 에서 MLP sparse 이득 사라짐**
- **Attention head sparsity**: batch-invariant (각 head 의 중요도는 모델 structural). batch 어느 크기에도 유효

결론: **Sparsity 는 attention head pruning 에 초점**, MLP 는 unstructured sparsity 로 한정적 이득.

### Weight Pruning 파이프라인

- **Wanda** (Sun et al. 2023): activation-aware, calibration 필요
- **SparseGPT** (Frantar et al. 2023): second-order info, 정확도 우수
- **Magnitude pruning**: 단순, 기준선

50% 2:4 또는 unstructured 50% 을 목표.

---

## 관련 참고 문헌

- **SparAMX (HF 2502.12444)**: AbouElhamayed et al. "SparAMX: Accelerating Compressed LLMs Token Generation on AMX-powered CPUs" https://huggingface.co/papers/2502.12444
- **Wanda pruning (Sun et al. 2023)**: https://arxiv.org/abs/2306.11695
- **SparseGPT (Frantar et al. 2023)**: https://arxiv.org/abs/2301.00774
- **Polar Sparsity paper**: batch-invariant sparsity 분석
- **Michel et al. "Are Sixteen Heads Really Better than One?"**: https://arxiv.org/abs/1905.10650
- **Intel Intrinsics `_mm512_maskz_fmadd_ps`**: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- **NVIDIA 2:4 Sparsity**: https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/ — 비교 참고
- **현재 코드**: `csrc/cpu/gemm_vnni.cpp` 의 dense `int8_gemm_vnni`

---

## 구체 작업

### Pruning 파이프라인
- [ ] **Pruning library 선택**: Wanda or SparseGPT
- [ ] **Qwen2.5-7B 50% unstructured pruned checkpoint 준비**
- [ ] **PPL 검증**: WikiText-2 에서 열화 <2% 확인

### Kernel 구현
- [ ] **`csrc/cpu/sparse_amx.cpp`** (신규)
  - `sparse_gemm_avx512_bitmask(input, weight_compact, mask, output, M, N, K)`
  - `_mm512_maskz_fmadd_ps` 기반 AVX-512 path
- [ ] **Tile-level AMX skip**: mask 가 all-zero 인 tile 은 AMX tdpbf16ps 생략
- [ ] **Sub-tile path**: 부분 sparse tile 은 AVX-512 fallback
- [ ] **Weight format 변환**: dense → (compact + mask) 저장

### 통합
- [ ] **torch op 등록**: `torch.ops._C_cpu_ops.sparse_gemm_bitmask_bf16`
- [ ] **Linear patch**: pruned model 로드 시 sparse kernel 로 치환
- [ ] **§15 pre-pack 연계**: sparse weight 도 tile layout 로 pre-pack

### 검증
- [ ] **정확도**: pruned model 의 output 이 reference (dense matmul) 과 일치 (mask 가 정확히 zero 를 표현)
- [ ] **Attention head pruning 실험**: 32 heads 중 8-16 개 제거 후 PPL
- [ ] **Batch scaling**: `batch=1/4/16` 에서 MLP sparse 의 이득 축소 현상 확인 (Polar Sparsity)
- [ ] **Memory BW 측정**: `perf stat -e uncore_imc/mem_bw` — sparse 에서 절감 확인

---

## 성공 조건

1. ✅ PPL 열화 <2% (WikiText-2)
2. ✅ Linear layer 단독 1.3× 이상 가속 (batch=1 decode)
3. ✅ Attention head pruning 적용 시 1.14× 이상
4. ✅ Memory BW 감소 20-40%
5. ✅ `batch=16` 에서도 attention sparse 는 이득 유지 (MLP 는 축소 인정)

---

## 의존성

- **선행**: §06 hot path wiring, §07 ISA dispatch (AVX-512 VNNI/BF16 경로)
- **병행**: §15 AMX pre-pack (sparse pre-packed layout)
- **후속**: 구조적 sparsity (2:4) 는 별도 검토, 현재는 unstructured 만

---

## 리스크

- **Pruning 후 PPL 열화**: Qwen2.5 outlier 때문에 Wanda/SparseGPT 적용 시 2% 이상 열화 가능. calibration set 조정 필요
- **MLP sparsity batch↑ 소멸**: Polar Sparsity 현상 — MLP 에 한정 최적화 의미 제한
- **Attention head pruning 의 생성 품질 영향**: PPL 은 유지되어도 long-text coherence 손상 가능 — MMLU/BBH 종합 평가
- **Bitmask 저장 overhead**: dense 대비 mask (N/8 bytes per row) 추가 저장. 50% sparsity 에선 compact (N/2) + mask (N/8) = N*5/8 vs dense N → 37.5% 절감 (50% 기대 대비 축소)

---

## 스택 호환성

- §13 T-MAC LUT GEMV 와 **대체 아닌 추가** (다른 경로)
- §15 AMX pre-pack: sparse weight 도 pre-pack 대상
- §11 Batch-aware decode attention: attention head pruning 이 batch-aware kernel 과 자연스럽게 조합

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_SPARSE_BITMASK` | `0` (기본) / `1` | SparAMX bitmask sparse 경로 |

전체 flag 테이블: [README.md](./README.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `csrc/cpu/sparse_amx.cpp` — (신규)
- `csrc/cpu/gemm_vnni.cpp` — 기존 dense VNNI GEMM 참조
- `csrc/cpu/torch_bindings_hybrid.cpp` — 등록
- `vllm/model_executor/layers/linear.py` — Linear patch
- Pruning scripts: `tools/pruning/wanda_prune.py` (신규, 외부 라이브러리 연동)
