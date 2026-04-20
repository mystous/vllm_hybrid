# 11. Batch-aware Decode Attention

**Tier**: 2 (원리만, 실측 수치 없음)
**근거 등급**: D (강한 가설, 환경 미검증). SparAMX 의 CPU attention 1.14× 외엔 CPU 대상 보고 수치 없음.
**상태**: ✗ **Phase 1 기각 (2026-04-20)** — IPEX 우회 + 기존 `batch16_paged_attention_v1` dispatch 활성 만으로는 이득 없음. 측정 결과 `measurement_results/H100x8/g0_11_qwen2.5_32b_phase1(fail)/` 참고.
**예상 이득**: batch=16 scaling 5.3× → 10–12× 개선 (목표, 미검증)

## Phase 1 실패 요약 (2026-04-20)

Option A (기존 kernel 재라우팅) 로 시도. 측정 결과:

| seqs | §06-1 v1 | §11 Phase 1 | Δ |
|---:|---:|---:|---:|
| 1 | 1,196.3 | 1,056.5 | −11.7% |
| 2 | 794.0 | 735.3 | −7.4% |
| 4 | 496.2 | 501.1 | +1.0% |
| 8 | 272.3 | 258.4 | −5.1% |

실패 원인:
1. seqs 2/4/8 구간이 `batch16_paged_attention_v1` 의 **remainder path** (per-seq loop). IPEX 와 memory access 구조 동일 → 이득 구조적으로 없음
2. Layout 일관성 유지 위한 prefill IPEX → SDPA fallback 의 오버헤드가 순손실
3. seqs=16 까지 돌리지 않음 (하위 구간 전체 regression 확인돼 중단)

Phase 2 (v2 신규 kernel, {1,4,8,16,32} template + block coalescing) 로 재시도 여부는 Tier 1 후보 (§13/§16/§22/§28) 검토 이후 재평가.

---

## 왜 필요한가

IPEX 의 `single_query_cached_kv_attention` 은 **per-seq loop 의 Python/C++ 혼합 구조** — 각 시퀀스마다 별도 GEMV 수행, KV paged access 반복. 한 step 의 batch=16 실행 시 batch=1 대비 **5.3× 느림** (Claude 실패 1 수치) — 선형 기대 16× 대비 3× 실패.

**원인 분해**:
- per-seq KV paged access → block_table 간접 참조, cache miss 반복
- per-seq softmax → 각각 독립, SIMD 기회 분산
- per-seq GEMV → AMX tile underutilization (M=1)
- barrier overhead: 각 seq 끝날 때 동기화

**해결**: batch 단위 재구성 — head-parallel + page-coalesced access + 배치 GEMM.

---

## 기술적 배경

### 현재 `batch16_paged_attention_v1` 의 한계

`csrc/cpu/batch_attention.cpp` 에 BF16 batch=16 paged attention kernel **이미 구현됨**. `cpu_attn.py:964` 의 `custom_avx` fallback path 로 호출.

**한계**:
- batch=16 hardcoded (`_v1` 접미사가 이를 암시)
- 동적 batch size 미지원 → batch=1, 4, 8, 32 에서는 fallback
- IPEX 의 기본 decode path 가 여전히 사용됨 (우리 경로는 `custom_avx` fallback 일 때만 호출)

### Batch-aware 의 핵심 설계

**1. Head-parallel**
```
반복 축: heads (32) × batch (B) — 총 32B 개 independent tasks
각 task: (1, head_dim) Q × (seq_len, head_dim) K^T → (1, seq_len) scores
```
OMP parallel for over `32B` tasks (충분한 병렬도).

**2. Page-coalesced KV access**
vLLM PagedAttention 은 KV 를 16-token block 단위로 저장. block_table 이 seq → [block_ids] 매핑.
- 같은 block id 를 가진 여러 시퀀스는 **block 1회 로드 + 여러 Q 와 계산**
- L2 resident block 활용 극대화

**3. Fused softmax + online max/sum**
Flash Attention 2 의 online softmax 원리 — tile 단위 running max/sum 갱신.

**4. Value accumulation**
score @ V: (B*heads, seq_len) × (seq_len, head_dim) → (B*heads, head_dim)
- seq_len 이 context 길이 (변동) — tiled reduction 필요
- V block 도 paged 구조 그대로

### Dynamic batch size 지원

**Option A — Template specialization**:
- `batch_paged_attention_v2<BATCH_SIZE=1|4|8|16|32>` compile-time 분기
- 각 size 에 맞게 tile schedule 최적화
- runtime 에 `if (B == 1) ... else if (B == 4) ...` dispatch

**Option B — Dynamic tiling**:
- B 를 runtime param, tile schedule 은 B / tile_B 기반
- flexibility 높음, per-size 최적화는 불가능

권장: **Option A (compile-time 분기)** for {1, 4, 8, 16, 32} + Option B fallback.

### GQA 고려사항

Qwen2.5 GQA: group=4. 같은 group 의 query heads 는 K, V 공유. Batch-aware kernel 설계 시:
- Task 축을 `(B * n_kv_heads * group)` 로 정의
- 같은 group 의 query 4개는 같은 K, V 에 대해 consecutive GEMV
- K, V 1회 로드 + 4 queries 계산 → memory traffic 4× 감소

---

## 관련 참고 문헌

- **Flash Attention 2 (Dao 2023)**: https://arxiv.org/abs/2307.08691 — batch/head 축 병렬화
- **Flash Decoding**: https://crfm.stanford.edu/2023/10/12/flashdecoding.html — decode 전용 batch-aware
- **PagedAttention (Kwon et al. 2023)**: https://arxiv.org/abs/2309.06180
- **Codex playbook Tier 2 batch-aware decode attention**: `/vllm_hybrid/ideation/20260415_094148_codex_ninja_gap_modification_playbook.md`
- **SparAMX attention (Hugging Face 2502.12444)**: https://huggingface.co/papers/2502.12444 — CPU attention 1.14× 개선
- **NEO attention offload**: https://openreview.net/forum?id=umgy9tWBLA — asymmetric attention
- **기존 구현**: `csrc/cpu/batch_attention.cpp` 의 `batch16_paged_attention_v1`
- **H100x8 wave=16 재앙 실측**: `/vllm_hybrid/eval/basic/H100x8/20260415_031045_worst_case_timeframe.md`

---

## 구체 작업

### 설계 단계
- [ ] **§01 G0 계측 결과로 attention 이 top bottleneck 인지 확인**
- [ ] **GQA fold scheme 결정**: group-aware (K/V 공유) vs naive (K/V 복제)
- [ ] **Batch size spec 결정**: {1, 2, 4, 8, 16, 32} specialization 목록
- [ ] **Task schedule 설계**: `(B × n_kv_heads × group)` 을 OMP thread 에 분배

### 구현
- [ ] **`csrc/cpu/batch_attention_v2.cpp`** (신규) — 기존 v1 확장
  - `batch_paged_attention_v2<BF16, B>(Q, K_cache, V_cache, block_table, output)`
  - template for B ∈ {1, 4, 8, 16, 32}
  - head-parallel OMP schedule
  - online softmax (Flash Attention 2 원리)
  - paged K/V access with block coalescing
- [ ] **Dynamic dispatch wrapper**: runtime B → template instance 선택
- [ ] **torch op 등록**: `torch.ops._C_cpu_ops.batch_paged_attention_v2_bf16`
- [ ] **`cpu_attn.py` 수정**: IPEX `single_query_cached_kv_attention` 대체
  - 우리 kernel 이 기본 경로가 되고, IPEX 는 fallback (PRE fix 와 반대)

### 검증
- [ ] **정확도**: IPEX vs v2 결과 BF16 tolerance (<1e-2 rel error)
- [ ] **Batch scaling 측정**: `batch=1, 2, 4, 8, 16` per-step attention time
- [ ] **목표**: `batch_scaling_ratio_attention = step(B) / step(1)` — 현재 5.3× → 2× 이하
- [ ] **GQA group sharing 검증**: K, V memory traffic `perf` counter

---

## 성공 조건

1. ✅ 정확도 tolerance 통과
2. ✅ Batch scaling: batch=4 에서 per-req cost 1 req 대비 1.5× 이하
3. ✅ batch=16 per-step attention 시간이 IPEX 대비 2× 빠름
4. ✅ K, V DRAM read bandwidth `batch=16` 에서 batch=1 대비 <2× (coalescing 작동)
5. ✅ G0 대비 decode step time 30–50% 감소 (attention top bottleneck 시)

---

## 의존성

- **선행**: §01 G0 계측, §06 hot path wiring, §07 ISA dispatch (AMX path)
- **병행**: §09 LUT Softmax (내부 softmax), §10 Head Folding (중복 영역 — 통합 권장)
- **후속**: §14 AVX/AMX Cascade 가 본 kernel 을 matmul stage 로 사용

---

## 리스크

- **높음: IPEX 내부 FD kernel 재구현**: Flash decoding + GQA + paged KV + online softmax 조합은 구현 복잡도 최상
- **GQA group sharing 구현 오류 시 정확도 폭락**: Q-K 매칭이 틀리면 출력 garbage. 유닛 테스트 세밀히 필요
- **paged KV access 의 cache miss**: block_table 이 random access → prefetch hint 없으면 L2 miss 폭증. `PREFETCHT2` intrinsic 추가
- **OMP schedule 오버헤드**: 32 heads × B 가 너무 많은 task 면 schedule overhead 가 gain 상쇄. tile_B × tile_head group 으로 coarsen

---

## 스택 호환성

- §10 Head Folding 과 **중복 영역** — 실제로는 하나의 kernel 로 통합 설계 필요
- §09 LUT Softmax 내부 softmax 를 1-pass LUT 로
- §14 Cascade pipeline 의 matmul stage 후보
- §15 AMX pre-pack 의 혜택 (K, V layout 도 pre-pack 가능)

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_BATCH_AWARE_ATTN` | `off` (기본) / `v1` / `v2` | v1=batch16 hardcoded (기존), v2=동적 |

전체 flag 테이블: [README.md](./README.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `csrc/cpu/batch_attention.cpp` — 기존 v1 (batch=16 hardcoded)
- `csrc/cpu/batch_attention_v2.cpp` — (신규)
- `csrc/cpu/torch_bindings_hybrid.cpp` — 등록
- `vllm/v1/attention/backends/cpu_attn.py` — dispatch 분기
- `vllm/attention/backends/pagedattention.py` — paged KV helper
