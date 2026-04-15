# 20. KV Cache CPU Tier Offload (InfiniGen)

**Tier**: 장거리 (70B / 큰 batch 전용)
**상태**: 🔶 부분 구현 (`--cpu-offload-gb` 용량 기반 offload 만)
**예상 이득**: 동시 시퀀스 3×, throughput 2–3× (70B / batch 1500+)
**Ninja Gap 기여도**: 7B 현 workload 0. 70B 에서 큼

---

## 왜 필요한가

70B BF16 weight = 140GB. H100 80GB 4 GPU (TP=4) → GPU 당 35GB. KV cache 여유 심각히 축소.
- 16K context × 1500 req 동시 시: KV ~수백 GB 필요
- HBM 만으로는 불가능 → CPU DRAM (TB 단위) 활용 필수

현재 vLLM `--cpu-offload-gb` 는 **단순 용량 기반**. InfiniGen 은 **predictive prefetching + tier-aware block_table** 로 latency 숨김.

현 workload (7B, 128/128) 에서는 KV cache 가 HBM 에 여유 → 본 기법 **의미 없음**. 70B 시나리오에서만.

---

## 기술적 배경

### Tier-aware PagedAttention

현재 PagedAttention:
```
block_table[seq] = [block_id_0, block_id_1, ...]
block[block_id] → HBM physical address
```

Tier-aware:
```
block_table[seq] = [(tier, block_id), ...]
tier ∈ {HBM_HOT, HBM_WARM, CPU_COLD}
```

- HBM_HOT: 최근 access
- HBM_WARM: 중간
- CPU_COLD: DRAM 에만

### Eviction 정책

**LRU**: least recently used block 을 HBM → CPU 로 demote
**Attention-aware**: attention weight 가 낮은 block 이 eviction 후보 (InfiniGen)
**Recency + Access frequency**: LFU-LRU hybrid

### Predictive Prefetch (InfiniGen 핵심)

**원리**: 다음 attention 이 필요할 KV block 을 **실제 attention 전에 미리 transfer**.
- Layer i 의 Q 를 부분 계산 → attention pattern 예측 → 필요 block 을 layer i attention 시작 전 HBM 으로 transfer
- 예측 정확도 > 90% (InfiniGen 실측)

**구체 알고리즘**:
1. Q 의 subset (first few dims) 을 먼저 계산
2. Subset Q 를 sampled K 와 inner product → rough attention score
3. Top-k block 을 prefetch 대상으로 선정
4. Full Q 계산 + 실제 attention 은 prefetch 완료 후

### DMA + pinned memory

```cpp
cudaHostAlloc(&pinned_buf, size, cudaHostAllocDefault);
cudaMemcpyAsync(gpu_buf, pinned_buf, size, cudaMemcpyHostToDevice, prefetch_stream);
cudaStreamSynchronize(prefetch_stream);  // 또는 event 기반 async wait
```

- Pinned memory: page-locked, DMA 직접 가능
- Separate CUDA stream: compute 와 overlap
- PCIe 4.0 x16: ~32GB/s → 1MB block ~30us transfer (single block)

### LMCache 스타일 Prefix Reuse

Prompt 앞부분이 동일한 요청 (예: system prompt) 의 KV 를 재사용:
- Hash 기반 prefix matching
- CPU DRAM 에 shared prefix cache 저장
- Request arrival 시 prefix hit → KV copy (prefill 생략)

---

## 관련 참고 문헌

- **InfiniGen (Lee et al. OSDI'24)**: https://arxiv.org/abs/2406.19707 — predictive KV prefetching
- **LMCache**: https://github.com/LMCache/LMCache — prefix reuse
- **FlexGen (Sheng et al. ICML'23)**: https://arxiv.org/abs/2303.06865 — heterogeneous offload
- **Petals (Borzunov et al. NeurIPS'23)**: distributed LLM with offload
- **vLLM `--cpu-offload-gb` 문서**: https://docs.vllm.ai/en/latest/serving/offloading.html
- **PagedAttention (Kwon et al. SOSP'23)**: https://arxiv.org/abs/2309.06180
- **PCIe 4.0 specs**: 32 GB/s x16
- **TODO v5 §1.3 A2 KV cache CPU tier offload**: `/vllm_hybrid/old_doc/TODO_v5_20260415.md`

---

## 구체 작업

### 사전 평가
- [ ] **70B workload 환경 구축** (§TODO H100 §3.1)
- [ ] **KV cache 압박 측정**: batch 1500+ 에서 HBM KV 용량 부족 확인
- [ ] **PCIe BW 측정**: `nvidia-smi dmon` 으로 host-device transfer

### Tier-aware PagedAttention
- [ ] **`vllm/v1/core/kv_cache_manager.py`** 에 tier 필드 추가
  - `block_table[seq] = [(tier, block_id), ...]`
- [ ] **Allocator 확장**: HBM + CPU DRAM 두 tier pool
- [ ] **Tier transition API**: `promote(block)`, `demote(block)`

### Eviction
- [ ] **LRU 기본**: HBM 에서 가장 오래 된 block → CPU
- [ ] **Attention-aware eviction** (InfiniGen 스타일): 추가 가능
- [ ] **Watermark**: HBM 사용률 X% 초과 시 eviction trigger

### Prefetch
- [ ] **Predictive prefetch** (InfiniGen): subset Q 로 예측 + top-k prefetch
  - 구현 복잡도 높음
- [ ] **Naive prefetch**: 다음 layer access 를 미리 시작 (간단, 효과 축소)

### DMA
- [ ] **Pinned memory pool**: `cudaHostAlloc` 으로 CPU side buffer
- [ ] **Separate CUDA stream**: `prefetch_stream`
- [ ] **Attention kernel 에 swap-in trigger**: block_table 의 tier 가 COLD 면 prefetch wait

### 검증
- [ ] **70B + batch 1500 demo**: GPU-only OOM, offload enabled 성공
- [ ] **Throughput**: offload 전후 비교 (목표 2-3×)
- [ ] **Latency**: TPOT p99 악화 없음 (prefetch 로 숨김)

---

## 성공 조건

1. ✅ 70B + batch 1500 에서 OOM 없이 동작
2. ✅ Throughput 2× 이상 (GPU-only batch ~500 대비)
3. ✅ TPOT p99 악화 20% 이하
4. ✅ Prefetch hit rate 90%+ (InfiniGen 목표)

---

## 의존성

- **선행**: §TODO 70B baseline 구축
- **무관**: 7B / 짧은 context 에서는 건드리지 않음

---

## 리스크

- **현 workload 무효**: 7B / 128/128 에서 이득 없음
- **Predictive prefetch 정확도 의존**: hit rate 낮으면 penalty (unused transfer)
- **PCIe BW 한계**: 동시 req 다수 시 bandwidth 포화
- **vLLM V1 scheduler 와 통합 복잡**: block_table 수정이 core scheduler 에 광범위 영향
- **Correctness**: tier transition 중 race condition 시 garbage KV → output 오류

---

## 스택 호환성

- §19 P/D disaggregation 과 조합: CPU prefill → GPU decode 에서 KV 가 CPU 에 남아 있음 (offload 자연스러움)
- §21 ScoutAttention 과 orthogonal
- 경로 1 기법과 무관 (CPU decode compute 가 아니라 KV storage)

---

## 관련 코드 위치

- `vllm/v1/core/kv_cache_manager.py` — block_table, allocator
- `vllm/v1/attention/backends/*` — attention kernel 의 swap-in trigger
- `vllm/engine/arg_utils.py` — `--cpu-offload-gb` 기존
- `vllm/config.py` — offload config 확장
- 참조: InfiniGen GitHub (공개 시)
