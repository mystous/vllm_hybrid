# 21. ScoutAttention Layer-Ahead

**Tier**: 장거리 (8K+ context)
**상태**: ⭕ 미구현
**예상 이득**: decoding **5.1×** (장문). GPU idle 57% → <5%
**Ninja Gap 기여도**: 현 workload 제한적. 8K+ context 에서 큼

---

## 왜 필요한가

Long-context 에서 attention KV scan 이 dominant. GPU 가 decode 의 attention 단계에 긴 시간 (57% idle due to memory wait) 소모.

**ScoutAttention**: CPU 가 **1 layer 앞서** Q 예측 → top-k KV block 선별 → partial attention 을 CPU 가 계산 → GPU 는 hot block 만 처리 → 결과 합산.

**원리**:
- Layer i+1 의 Q ≈ Layer i 의 Q (cosine sim 0.93+)
- → Layer i 처리 중 CPU 가 미리 Q_{i+1} ≈ Q_i 로 attention score 예측 가능
- → top-k block 선별
- → CPU 가 **small-k attention** 을 layer i+1 완료 직전까지 준비

**현 workload (128 ctx)** 에선 full attention 이 빠르므로 의미 없음. **8K+** 에서 선별 이득.

---

## 기술적 배경

### Layer-ahead Q 예측

Transformer layer 간 Q 변화:
- `Q_{i+1} = proj_q(layer_i_output)` 이지만, layer_i_output 은 residual stream 에서 점진적 변화
- Cosine similarity (`Q_i`, `Q_{i+1}`) ≈ 0.93+ (실증)
- → `Q_{i+1}` 을 `Q_i` 로 근사해 **미리** attention score 계산

### Top-k KV Block Selection

전체 seq_len 대신 top-k block 만 attention:
- k = 64~256 block (seq_len 8K → 512 block, 128 block 선별 = 25%)
- Attention 의 sparsity 활용 (Streaming/Sliding/Sink + Top-k 조합)

### CPU-side Partial Attention

```
CPU (layer i 중):
    Q_predicted = Q_i  # 근사
    scores = Q_predicted @ K_all[sampled_blocks]^T
    top_k_blocks = topk(scores, k)
    partial_attn = softmax(top_k_scores) @ V[top_k_blocks]

GPU (layer i+1):
    Q_actual = proj_q(layer_i_output)
    hot_attn = Q_actual @ K[hot_blocks]^T @ V[hot_blocks]
    final_attn = merge(partial_attn, hot_attn)
```

**정확도 vs 속도 trade**: top-k 가 작을수록 빠르지만 정확도 손실. 실험으로 k 확정.

### 근사 Attention 의 한계

- PPL 열화 목표 <2.1% (ScoutAttention 실측)
- **특정 workload** (long-context QA, summarization) 에선 성능 유지, generation 의 다양성 (narrative) 에선 손실 가능
- Benchmark: WikiText perplexity, LongBench, RULER

### vLLM 포팅 복잡도

현재 vLLM V1:
- Layer loop 가 Python level 에서 sequential
- Layer i 완료 전 layer i+1 의 predictive 계산 동시 실행 = **cross-layer pipeline** 필요
- C++ 에서 pipeline runner 구축 필요 (§17 Core Group Pipeline 과 유사)

---

## 관련 참고 문헌

- **ScoutAttention paper**: (구체 citation 필요 — ideation 문서에 "ScoutAttention 5.1×" 언급, 원논문 검색 필요)
- **StreamingLLM (Xiao et al. 2024)**: https://arxiv.org/abs/2309.17453 — attention sink + streaming
- **H2O (Zhang et al. 2023)**: https://arxiv.org/abs/2306.14048 — Heavy-hitter KV eviction
- **Quest (Tang et al. 2024)**: https://arxiv.org/abs/2406.10774 — query-aware KV selection
- **DejaVu (Liu et al. ICML'23)**: contextual sparsity prediction
- **SparQ (Ribar et al. 2024)**: https://arxiv.org/abs/2312.04985 — sparse query attention
- **Layer-wise cosine similarity in transformers (Kobayashi et al. 2020)**: https://arxiv.org/abs/2004.10102 — hidden state layer 간 유사도
- **Codex 1630 superset Part 4-4**: `/vllm_hybrid/ideation/20260415_1630_ninja_gap_superset.md`

---

## 구체 작업

### 사전 평가
- [ ] **8K+ context workload 구축** (§TODO H100 §3.2)
- [ ] **Layer-ahead Q 유사도 실측**: Qwen2.5 에서 cosine sim (Q_i, Q_{i+1}) 분포
- [ ] **Top-k 선별 오차 측정**: k ∈ {32, 64, 128, 256} 별 attention full vs top-k 비교

### 설계
- [ ] **Prediction 방식**: naive `Q_{i+1} ≈ Q_i`, 또는 light projection
- [ ] **Top-k 선정 알고리즘**: cumulative score threshold or fixed k
- [ ] **CPU-GPU handoff**: CPU partial_attn 결과를 GPU 에 전달 후 merge
- [ ] **Accuracy vs speed curve**: PPL vs k

### 구현
- [ ] **`csrc/cpu/scout_attention.cpp`** (신규) — CPU side layer-ahead prediction + partial attention
- [ ] **`vllm/v1/attention/backends/scout_attn.py`** — attention backend 확장
- [ ] **Cross-layer pipeline runner** (§17 과 유사)
- [ ] **Merge 로직**: CPU partial + GPU hot → final attention output

### 검증
- [ ] **정확도**: WikiText-2 / LongBench PPL 열화 <2.1%
- [ ] **Decoding speed**: 8K/16K context 에서 TPOT
- [ ] **GPU idle 감소 확인**: nvidia-smi dmon 로 GPU util 변화
- [ ] **Ablation**: k 별 정확도/속도 trade

---

## 성공 조건

1. ✅ PPL 열화 <2.1%
2. ✅ 8K context decode 3× 이상 가속
3. ✅ 16K 에서 5× 목표 (ScoutAttention 원 논문 수치)
4. ✅ GPU idle 시간 57% → <10%
5. ✅ 짧은 context (<2K) 에서는 bypass 로 기존 성능 유지

---

## 의존성

- **선행**: §01 G0 계측, §11 Batch-aware decode attention (CPU attention kernel), §17 pipeline infra 일부 재사용
- **선행**: 8K+ workload 가 프로젝트 target 이어야 의미

---

## 리스크

- **현 workload (128 ctx) 무효**: 이득 0
- **Cross-layer pipeline 구현 복잡도 매우 높음**: C++ level 에서 cross-layer dependency 관리 non-trivial
- **정확도 열화가 생성 품질에 비가시적 손상**: PPL 은 유지 되어도 narrative coherence 손상. 종합 평가 필수
- **Layer-ahead Q 유사도가 model/prompt 에 따라 다름**: 특정 케이스 (long context 전환) 에서 sim 낮음 → 예측 실패
- **vLLM 포팅 큰 작업**: 설계 부터 시작

---

## 스택 호환성

- §11 Batch-aware decode attention: scout 내부 partial attention 이 본 kernel 사용
- §20 KV offload: scout top-k 선별로 CPU 에 필요한 KV 만 유지 → memory 절감
- §22 NEO asymmetric 과 유사 아이디어, 독립 path

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_SCOUT_ATTN` | `0` (기본) / `1` | Layer-ahead ScoutAttention 활성 |

전체 flag 테이블: [README.md](./README.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `csrc/cpu/scout_attention.cpp` — (신규)
- `vllm/v1/attention/backends/scout_attn.py` — (신규)
- `vllm/v1/attention/backends/cpu_attn.py` — 기존 reference
- 별도 설계 문서: `docs/SCOUT_ATTENTION_DESIGN.md` (사전 필수)
