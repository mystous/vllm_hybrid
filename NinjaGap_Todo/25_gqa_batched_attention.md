# 25. GQA-aware Batched Paged Attention (CPU decode)

**Tier**: 2 (G2 핵심)
**상태**: ⭕ 미구현 (기존 `csrc/cpu/batch_attention.cpp` 가 GQA 미인식 상태로 존재)
**관계**: **§11 Batch-aware Decode Attention 의 GQA 확장**. §11 기본 방향은 유지하고 본 § 에서 "GQA ratio 5:1 를 어떻게 활용해 KV bandwidth 를 줄이는가" 를 구체화.

---

## 왜 필요한가

실측 (H100x8 + Qwen2.5-32B) sublayer breakdown:
- Attention Core (QK/softmax/V) = 105.6 ms/step (22.5%)
- QKV Projection = 38.5 ms (8.2%)
- Attention Output = 29.2 ms (6.2%)
- 합 = 173.3 ms = **37% of step**

Qwen2.5-32B 는 **GQA (grouped query attention) 40:8**:
- `num_attention_heads = 40` (Q head)
- `num_key_value_heads = 8` (KV head)
- ratio = 5:1 — 매 K/V head 를 **5개 Q head 가 공유**

현재 CPU paged attention 경로 (`cpu_attn.py` / `batch_attention.cpp`) 는:
- Q 40 head × KV 8 head 를 naive 하게 처리 — K 를 5번 재로드 (중복)
- decode 단계는 token 1개만 생성하므로 batch=1 에서는 GQA 이득이 **구조적으로 반영 안 됨**

**기대 이득**: K/V 를 1회만 읽고 Q 5개와 동시 연산 → attn_core 의 memory bandwidth 병목 감소. 예상 1.4~1.6×.

---

## 기술적 배경

### 현재 CPU attention path

`vllm/v1/attention/backends/cpu_attn.py` 의 decode 경로:
```
for each layer:
  for each batch_idx:
    for each query_head in 40:
      kv_head_idx = query_head // 5   # GQA mapping
      K = load_from_paged_cache(kv_head_idx, ...)
      V = load_from_paged_cache(kv_head_idx, ...)
      scores = Q[query_head] @ K.T
      out[query_head] = softmax(scores) @ V
```

문제: 같은 `kv_head_idx` 에 대해 `K`, `V` 를 5번 로드 → DDR BW 낭비.

### GQA-aware 경로

```
for each layer:
  for each batch_idx:
    for each kv_head in 8:
      K = load_from_paged_cache(kv_head, ...)     # 1 load
      V = load_from_paged_cache(kv_head, ...)     # 1 load
      Q_group = Q[kv_head*5 : kv_head*5+5]        # 5 Q heads
      scores = Q_group @ K.T                       # (5, seq_len)
      out[kv_head*5 : kv_head*5+5] = softmax(scores) @ V
```

- K/V load 5× 감소
- Q_group @ K.T 는 `(5, head_dim) × (head_dim, seq_len) = (5, seq_len)` 으로 GEMM 형태 → VNNI/AMX 로 가속
- Q 5개 + K 1개 조합은 register 에 상주 가능한 크기 (head_dim=128, 5×128×2bytes = 1.28 KB)

### AMX tile 활용

SPR AMX-BF16 tile = 16×32 BF16. head_dim=128 이면 Q_group (5, 128) 를 tile 1개에 담을 수 있음 (padded to 16). K 는 (128, seq_len) → seq_len chunk 단위로 tile.

- decode batch=1 + GQA 5:1 + AMX-BF16 tile = "single-token decode 가 GEMM 처럼 보임"
- §10 Head Folding 과 철학 동일, GQA 를 1차 folding factor 로 활용

### paged attention block layout 재고

현재 paged cache: `(num_blocks, block_size, num_kv_heads, head_dim)`. 이 layout 은 GQA-aware 에서도 그대로 사용 가능하지만, `kv_head` 축이 stride=1 인지 확인 필요 (contiguous load 위해).

### batch>1 조합

batch=4 이면 effective Q = 4 × 5 = 20 rows. AMX tile (16) 하나로 커버 — batch 더 키울수록 tile utilization 향상. **§11 batch-aware 의 자연스러운 확장**.

---

## 관련 참고 문헌

- **GQA 원논문 (Ainslie et al., 2023)**: https://arxiv.org/abs/2305.13245
- **FlashAttention-2 (Dao, 2023)**: GQA-aware batched load 패턴 — CPU 포팅 기반 참고
- **xFasterTransformer attention kernel**: SPR AMX 활용 decode attention 예제
- **llama.cpp ggml-cuda attention**: GQA 처리 방식 (`llama_kv_cache_view`)
- **본 프로젝트 기존 커널**: `csrc/cpu/batch_attention.cpp` (BF16 batch-16 paged attention — GQA 비활성)
- **§11 Batch-aware Decode Attention** (상위 방향)
- **§10 Head Folding** (GEMV → GEMM, GQA group 을 folding factor 로 사용)

---

## 구체 작업

- [ ] **`cpu_attn.py` 검토**: 현재 decode loop 가 Q-head 단위인지 KV-head 단위인지. `_decode_path_counts` 의 경로별 분포 확인
- [ ] **GQA group 인지 유무 확인**: `self.num_queries_per_kv` 같은 attribute 가 이미 있는지
- [ ] **`batch_attention.cpp` 확장**: Q_group (5 heads) × K (1 head) → softmax → V 의 AMX-BF16 tile 구현
  - Q_group × K.T: tile 1개
  - softmax: AVX-512 exp / reduce_max
  - scores × V: tile 1개
- [ ] **torch custom op 등록**: `torch.ops._C_cpu_ops.gqa_paged_attention_bf16(q, paged_k, paged_v, block_table, ...)`
- [ ] **dispatch**: `cpu_attn.py` 에서 GQA 감지 (`num_queries_per_kv > 1`) 시 신규 op 호출, 아니면 기존 경로
- [ ] **paged block stride 검증**: contiguous load 조건 만족 확인
- [ ] **batch>1 경로**: §11 과 병합. `(batch × group, head_dim)` tile 구성
- [ ] **정확도 검증**: GPU BF16 vs CPU GQA-aware 의 출력 tensor allclose (rtol=1e-3)
- [ ] **성능 측정**: attn_core sublayer ms 변화 (106ms → 목표 ≤ 70ms)

---

## 성공 조건

1. ✅ `_decode_path_counts` 에 `gqa_batched` 경로 신설, 호출 비율 100%
2. ✅ attn_core sublayer ms **≤ 70 ms** (현재 106ms → 34% 감소)
3. ✅ batch=4 에서 attn_core 가 batch=1 대비 **sub-linear 증가** (4× 아닌 2~2.5×)
4. ✅ GPU 와 출력 allclose (rtol=1e-3)

---

## 의존성

- **선행**: §01 G0 계측, §11 Batch-aware decode attn (개념적 선행)
- **병행**: §10 Head Folding (GQA group 을 folding factor 로 결합)
- **후속**: §14 AVX/AMX Cascade (AMX tile 활용), §16 SparAMX (attention sparsity)

---

## 리스크

- **GQA 인식 로직 누락**: vLLM V1 의 attention backend 가 GQA 를 이미 flatten 해서 전달하면 원 구조가 사라짐. config 에서 `num_kv_heads` 직접 읽어야
- **AMX tile fragmentation**: head_dim=128 이 tile 크기(32 BF16)와 정수 배 관계여야. 128/32 = 4 번 tile 누적 필요
- **block_size 영향**: paged cache 의 block_size (보통 16 or 128) 가 tile 과 불일치 시 경계 처리 복잡
- **batch-1 에만 이득**: batch=16 이면 기존 flatten 경로가 이미 GEMM 화 — GQA-aware 이득 축소

---

## 스택 호환성

- §10 Head Folding: GQA group (5) × batch (N) 를 folding factor 로 곱
- §11 Batch-aware: 본 § 가 GQA 측면, §11 가 batch 측면 — 결합 시 양방향 확장
- §16 SparAMX: GQA group 내에서 일부 Q head attention sparse 가능

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `HYBRID_GQA_BATCHED_ATTN` | `0` (기본) / `1` | GQA-aware kernel dispatch |
| `HYBRID_BATCH_AWARE_ATTN` | (§11 flag) | 본 § 와 결합 |

---

## 관련 코드 위치

- `csrc/cpu/batch_attention.cpp` — 확장 대상
- `vllm/v1/attention/backends/cpu_attn.py` — dispatch 경로 수정
- `csrc/cpu/torch_bindings_hybrid.cpp` — 신규 op 등록
- `vllm/attention/ops/paged_attn.py` — reference (GPU 경로 비교용)
