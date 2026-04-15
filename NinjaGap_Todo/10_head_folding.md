# 10. Head Folding (GEMV → GEMM)

**Tier**: 1
**상태**: ⭕ 미구현
**예상 이득**: decode attention 1.5–2× (SGLang blog)

---

## 왜 필요한가

decode 단계 attention 의 Q 는 **M=1** (new token 1개). 즉 `Q (1 × d_head) @ K^T (d_head × seq_len) = score (1 × seq_len)` — 이건 **GEMV** (matrix × vector). GEMV 는 AMX tile 의 16×16 구조를 **1/16 만 활용** — 심각한 underutilization.

**Head Folding**: 여러 request 의 Q 를 concat + head 축을 batch 로 접어서 **M=n_heads × batch** 형태의 GEMM 으로 변환. AMX tile 이 가득 참.

실패 1 의 본질 중 하나: batch=1 attention 이 GEMV 라 AMX 가 GEMM 기회 없음. Head Folding 은 **"batch 가 작아도 AMX 가 busy 해지는" 우회**.

---

## 기술적 배경

### Head Folding 이란

**Standard decode attention** (per-seq loop):
```
for seq in range(batch):
    Q = Q_batched[seq]          # shape (1, n_heads, head_dim)
    K = K_cache[seq]              # shape (seq_len, n_heads, head_dim)
    score = Q @ K.transpose()     # shape (1, n_heads, seq_len) — per-head GEMV
    ...
```

**Head Folded attention**:
```
Q_folded = concat([Q[s] for s in range(batch)], dim=0)  # shape (batch * n_heads, 1, head_dim)
# or for MLA/shared: fold over heads
Q_folded = Q.reshape(batch * n_heads, 1, head_dim)
# GEMM: (batch * n_heads, 1) @ (1, head_dim) @ (head_dim, seq_len) — effective M = batch * n_heads
```

batch=1, n_heads=32 → **M=32** GEMM (AMX tile 가득 참).

### GQA (Qwen) 에서의 변형

Qwen2.5 는 **Grouped-Query Attention**:
- `n_kv_heads < n_query_heads` (예: 32 query heads, 8 kv heads, group=4)
- 같은 group 의 query 들이 같은 K, V 공유

Head Fold 시:
- Query heads 는 batch 차원으로 폴드 가능
- K, V 는 head 축 복제 (memory 낭비) or group-aware access (복잡)
- **Group-aware**: 같은 group 의 query 들을 M 차원에 모으고, K/V 는 1회 로드 + 여러 query 와 계산

### MLA (DeepSeek) 에서의 직접 적용

DeepSeek V2/V3 의 Multi-head Latent Attention: Q projection 이 저랭크 latent 로 압축 후 head 확장. Head Folding 이 구조적으로 자연스럽게 맞음. 하지만 Qwen2.5 에는 MLA 없음.

### SGLang 실측

SGLang CPU backend blog 에서 Head Folding 으로 decode attention **1.5–2× 가속** 보고. Qwen 계열 GQA 에서도 변형 적용됨.

### IPEX `single_query_cached_kv_attention` 대체

현재 CPU attention 경로:
```
cpu_attn.py → IPEX single_query_cached_kv_attention
         → per-seq loop, paged KV access
```

이 함수가 **per-seq GEMV** 구조. Head Folding 구현은 이 함수를 교체하거나 우회.

---

## 관련 참고 문헌

- **SGLang CPU backend blog**: https://lmsys.org/blog/2025-10-22-KTransformers/ — Head Folding 실측
- **DeepSeek V2 paper "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"**: https://arxiv.org/abs/2405.04434 — MLA 원전
- **GQA paper (Ainslie et al. 2023) "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"**: https://arxiv.org/abs/2305.13245
- **Flash Attention 2 paper**: Dao (2023) — batch/head 축 병렬화 원리 (CPU 이식 가능)
- **PagedAttention paper (Kwon et al. 2023) — vLLM 원전**: https://arxiv.org/abs/2309.06180
- **Codex playbook Tier 2 Head Folding**: `/vllm_hybrid/ideation/20260415_094148_codex_ninja_gap_modification_playbook.md`
- **IPEX single_query_cached_kv_attention 구현**: https://github.com/intel/intel-extension-for-pytorch/

---

## 구체 작업

### 설계
- [ ] **Qwen2.5 GQA 구조 분석**: `n_query_heads=32, n_kv_heads=8, group=4, head_dim=128`
- [ ] **Fold scheme 결정**:
  - Option A: query heads 만 M 축으로 폴드 (K/V 복제) — 메모리 ↑ 32×
  - Option B: group-aware fold (같은 group 의 4 query → M=4, K/V 1회 로드)
  - Option C: batch × heads → M (multi-sequence 에서만 이득)
- [ ] **Tile 경계 설계**: M=32 (AMX tile 가득), K=head_dim=128, N=seq_len (변동)

### 구현
- [ ] **`csrc/cpu/fold_attention.cpp`** (신규)
  - input: Q `(batch, n_q_heads, head_dim)`, K paged, V paged, block_table
  - fold: Q 를 `(M, head_dim)` 으로 재배치
  - GEMM 1: Q_folded @ K^T → score `(M, seq_len)`
  - Softmax (online, §09 LUT 사용)
  - GEMM 2: score @ V → output `(M, head_dim)`
  - unfold: output 을 원래 shape 으로 복원
- [ ] **PagedAttention block_table 접근**: vLLM 의 paged KV 구조 그대로 사용
- [ ] **AMX tile scheduling**: `ldtilecfg` 1회 + loop 내 `tileloadd/tdpbf16ps/tilestored`
- [ ] **torch op 등록**: `torch.ops._C_cpu_ops.fold_attention_bf16`
- [ ] **cpu_attn.py 수정**: `_IPEXPagedAttention.forward_decode` 에서 `fold_attention_bf16` 호출 분기

### 검증
- [ ] **정확도**: 기존 IPEX path vs Head Folded 결과 BF16 tolerance
- [ ] **성능**: batch=1, 2, 4, 8 에서 per-step attention 시간 비교
- [ ] **MLA 모델** (DeepSeek) 에서도 작동하는지 optional 검증

---

## 성공 조건

1. ✅ 정확도 tolerance 통과 (BF16 rel error <1e-2)
2. ✅ batch=1 에서 decode attention 1.5× 이상 가속 (AMX tile 활용도 증가 확인)
3. ✅ batch=4/8 에서 2× 이상 가속 (multi-seq 이득 추가)
4. ✅ `batch_scaling_ratio` 개선 기여도 측정 (attention 만 분리 시)
5. ✅ G0 대비 decode step time 감소 — attention 이 top bottleneck 일 때 효과 큼

---

## 의존성

- **선행**: §01 G0 계측 (attention 이 top 인지), §06 hot path wiring, §07 ISA dispatch (AMX path 존재)
- **병행**: §09 LUT Softmax (folded attention 내부 softmax 를 LUT 으로)
- **후속**: §11 Batch-aware decode attention 은 본 Head Folding 의 일반화 버전 — 통합 가능

---

## 리스크

- **GQA fold 가 K/V 복제 필요 시 메모리 폭증**: 32 × head_dim × seq_len × layers. 1M context 에선 부담. Option B (group-aware) 가 필수
- **vLLM PagedAttention block_table 접근이 복잡**: paged 구조 때문에 K fold 시 non-contiguous access — cache miss 폭증 위험. block_table 을 먼저 gather 해야 함
- **IPEX 대체 시 Flash attention 등 기타 최적화 잃음**: IPEX 는 Flash decoding 원리 이미 일부 적용. 우리 Head Fold 가 그보다 확실히 빨라야 의미
- **§11 Batch-aware attention 과 중복 영역**: 하나로 통합하는 게 유지보수에 유리. 선택 필요

---

## 스택 호환성

- §09 LUT Softmax: fold_attention 내부 softmax 를 LUT 으로
- §11 Batch-aware decode attention: Head Folding 을 일반화 — **선택 or 통합**
- §14 AVX/AMX Cascade: fold 된 GEMM 이 cascade 의 matmul stage 가 됨

---

## 관련 코드 위치

- `csrc/cpu/fold_attention.cpp` — (신규)
- `csrc/cpu/torch_bindings_hybrid.cpp` — 등록
- `vllm/v1/attention/backends/cpu_attn.py` — `_IPEXPagedAttention.forward_decode` 분기
- `vllm/attention/backends/pagedattention.py` — paged KV 구조 참조
- `csrc/cpu/batch_attention.cpp` — `batch16_paged_attention_v1` 기존 구현 참조 (§11 연결)
