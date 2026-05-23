# SUB_038 — B5 libxsmm JIT shape-specific kernel plan

> **parent**: TSK_019 / N 문서 영역 B B5 / O 분석 §7.2 ★★ B-tier 후보
> **출처**: 사용자 명시 (turn 22) — "3개 전부 다 진행"
> **선행**: SUB_036 (Path A 500p baseline) 결과 + SUB_037 (B4 SPARAMX) 결과 — B-tier 의 일반적 가치 검증 후 진입.

---

## 1. 배경

### 1.1 libxsmm

| 항목 | 값 |
|---|---|
| 출처 | [libxsmm GitHub](https://github.com/libxsmm/libxsmm) + arXiv 2304.12576 (PyTorch-TPP) |
| 핵심 | shape-specific GEMM / matmul kernel 의 **JIT 생성** — runtime 에 size 별 최적 kernel 영역 |
| 비교 | ISPC = compile-time SIMD intrinsics generation. libxsmm = runtime JIT으로 shape 별 정확한 kernel |
| NEO 적용 | qk_product / av_product 의 shape (HEAD_DIM=128, NUM_Q_HEADS=64, BLOCK_SIZE=16) 별 JIT kernel |

### 1.2 PyTorch-TPP 사용 사례

- Intel PyTorch-TPP 가 libxsmm 위에 attention/MLP kernel build
- BF16 / FP16 / INT8 모두 지원
- KTransformers, IPEX 가 비슷한 방식

## 2. 현 코드 surface vs libxsmm 적용

### 2.1 현 코드 = ISPC compile-time

- `pacpu.ispc::qk_product` = ISPC `--target=avx512spr-x16 -O3`. 컴파일 시점 SIMD lane 결정.
- shape: `[NUM_Q_HEADS=64, HEAD_DIM=128] × [BLOCK_SIZE=16, HEAD_DIM=128] = [NUM_Q_HEADS, BLOCK_SIZE]`
- 한 shape 만 — compile-time 결정으로 충분.

### 2.2 libxsmm 적용 가능 영역

| 영역 | 가치 |
|---|---|
| qk_product BF16 GEMM | AMX 가 이미 적용 (qk_amx) — libxsmm 가 AMX 보다 더 빠르려면 specific tile schedule 필요. **불확실** |
| av_product FP16 mul-accumulate | scalar broadcast × vec (matmul 아님) — libxsmm 의 BRGEMM/EMM 적용 어려움. **부적합** |
| **softmax + log-sum-exp** | libxsmm 는 GEMM 전용 → softmax 별도 kernel 필요. **부적합** |

→ **libxsmm 적용 surface 가 좁음** — av_product 의 matmul 변환 (SUB_039 와 중복) 후에야 의미.

### 2.3 추가 위험

| risk | 설명 |
|---|---|
| 외부 dependency 추가 | libxsmm install + cmake link + ABI compat (Torch C++ ABI vs libxsmm) |
| JIT overhead | 첫 호출 시 ms 단위 JIT 컴파일 — request 첫 cdec 의 latency spike |
| SPR AMX vs libxsmm tile schedule | libxsmm 의 AMX schedule 이 우리 hand-tuned `qk_amx` 보다 빠르지 않을 수 있음 |

## 3. 권고 — SUB_038 보류 / SUB_037, SUB_039 우선

- libxsmm 적용 가치가 av_product 의 matmul 변환 (SUB_039) 후에야 발생
- 외부 dependency 추가 비용 vs 효과 불확실
- → **SUB_037 (B4 SPARAMX) → SUB_039 (av_amx 확장) → SUB_038 (libxsmm)** 순서가 합리적
- SUB_036 결과 + SUB_037 결과 본 후 SUB_038 진입 여부 재평가

## 4. 진입 조건

- SUB_037 의 ARI dispatch 가 ≥3% win
- SUB_039 의 av_amx 확장이 ≥3% win
- 둘 중 1+ 가 win 일 때만 libxsmm 의 추가 ROI 검토

→ **현재는 plan 단계 보류**. SUB_036/037/039 결과 후 재평가.
