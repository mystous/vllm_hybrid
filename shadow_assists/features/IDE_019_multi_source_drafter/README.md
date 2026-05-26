# IDE_019 — CPU Multi-Source Spec Drafter

> **scope**: ngram (GPU) + suffix (GPU) + **Jacobi (CPU AVX-512)** + **AMX draft head (CPU)** 의 multi-source AGSD 통합.
> **paper angle**: SwiftSpec (ASPLOS'26) 의 GPU-GPU disaggregation 과 다른 — GPU + **CPU heterogeneous** 의 첫 case.
> **status**: ✅ design + skeleton 작성 완료 / ⚠ Jacobi + AMX draft head 구현 별도 turn.

---

## 1. 이론적 배경

### 1.1 본 IDE 의 novelty

| 영역 | 현재 spec decoding | 본 IDE_019 |
|---|---|---|
| draft source | ngram (CPU) OR suffix (GPU) 단일 | 4-source AGSD multi-router |
| CPU draft head | 직접 대응 논문 없음 | AMX BF16 matmul (SUB_106 입증 22 TFLOPS) |
| Jacobi parallel decode | EAGLE / Medusa 의 GPU only | CPU AVX-512 vectorize Jacobi |
| acceptance rate vs source | per-workload best (SUB_011 measured chat 81.2%) | per-workload best-source 자동 선택 |

### 1.2 3 sub-task

| TSK | 영역 | scope | priority |
|---|---|---|---|
| TSK_035 | Jacobi lookahead + AVX-512 kernel | lossless guarantee proof + CPU AVX-512 vectorize | ★ |
| TSK_036 | AMX draft head on small model | Qwen 0.5B AMX → ≤ 5 ms / draft step | ★★ |
| TSK_037 | AGSD multi-source integration on canonical | router 4-method 분기 + per-workload best-source rule | ★★ |

---

## 2. 구현 방향

### 2.1 Jacobi AVX-512 kernel (TSK_035)

```cpp
// src/jacobi_avx512/jacobi_kernel.cpp
// Jacobi parallel decoding: 동일 sequence 의 K 개 candidate token 을
// AVX-512 lane 으로 parallel iterate.
void jacobi_iterate_avx512(
    const __bf16* prev_hidden,    // [B, K, hidden]
    const __bf16* W_lm_head,       // [hidden, vocab]
    int32_t* candidates_out,       // [B, K]
    int B, int K, int hidden, int vocab,
    int max_iters
);
```

### 2.2 AMX draft head (TSK_036)

```cpp
// src/amx_draft_head/qwen_draft.cpp
// IDE_016 TSK_026 의 amx_matmul_bf16 재사용
// Qwen 0.5B forward on CPU AMX:
//   embed → 24 layers × (attn + MLP) → lm_head
// target: ≤ 5 ms / draft step (per batch)
struct QwenDraftHead {
    int hidden;        // 896
    int num_layers;    // 24
    int num_heads;     // 14
    int kv_heads;      // 2
    int intermediate;  // 4864
};
```

### 2.3 4-source AGSD router (TSK_037)

```python
# src/router/multi_source_router.py
# /tmp/sub094_router.py 의 확장
# 4 method: vanilla, ngram, suffix, cpu_amx_draft
# 분기 rule: workload classifier (SUB_076 PoC) + model size + acceptance rate history
```

---

## 3. 측정 결과 input

- [SUB_106](../IDE_015_cpu_extreme_util/SUB_106_amx_microbench/RESULTS.md) — AMX 22 TFLOPS peak (TSK_036 의 throughput lower bound)
- [SUB_011](../IDE_006/TSK_020/idea/IDE_011_acceptance_rate_direct_measure.md) — chat K=6.69 α=81.2% (TSK_037 의 per-workload best-source 입력)
- [SUB_168](../IDE_015_cpu_extreme_util/SUB_168_cpu_task_candidate_matrix/RESULTS.md) — task F (AMX draft head) 의 phase mapping

---

## 4. 검증

- TSK_035: Jacobi 의 lossless guarantee proof (rejection sampler 와 통합)
- TSK_036: AMX draft head 의 acceptance rate vs ngram (target: chat α ≥ 60%)
- TSK_037: canonical multi-source 통합 → per-workload best-source 자동 분기 vs single-source baseline +5-15% throughput

---

## 5. dependencies

- IDE_016 TSK_026 (AMX matmul kernel) — TSK_036 의 base
- IDE_017 TSK_028 (DMA pinned pool) — CPU → GPU 의 draft logits transfer
- IDE_018 TSK_033 task F — IDE_019 의 task F (AMX draft head) 의 dispatch
