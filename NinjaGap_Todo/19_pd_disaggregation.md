# 19. P/D Disaggregation (Prefill/Decode 분리)

**Tier**: 장거리 (long-context 전용)
**상태**: 🔶 부분 구현 (`vllm/engine/disaggregated/` stub 존재, hybrid 와 미통합)
**예상 이득**: 16K input 에서 GPU TPOT p99 개선
**Ninja Gap 기여도**: **현 workload (128/128) 에서 0**. Long-ctx 전용

---

## 왜 필요한가

Long-context (16K+ input) 에서 GPU 는 prefill 에 긴 시간 소모 → decode stall. Prefill 은 compute-bound (GEMM 위주, parallelizable), decode 는 memory-bound (GEMV 위주).

**CPU 의 상대적 우위**: AMX BF16 은 prefill 대형 GEMM 에서 경쟁력 있음 (batch × seq_len 이 큼, tile 가득). CPU 가 prefill 을 담당하면 GPU 는 decode 전념 → TPOT p99 개선.

현 workload (128 input / 128 output) 에서는 prefill 이 짧아 의미 없음. 장거리 workload 전제.

---

## 기술적 배경

### Disaggregated Serving 개념

**Splitwise (MS Research 2024)**, **DistServe (UCSD 2024)** 의 원리:
- Prefill-optimized cluster (GPU-heavy) → decode-optimized cluster (memory-heavy)
- KV cache 를 prefill 노드에서 decode 노드로 transfer
- 각 단계의 HW 특성에 맞춰 최적화

우리 변형: CPU = prefill cluster, GPU = decode cluster. **Intra-machine** disaggregation.

### Hand-off 메커니즘

Request lifecycle:
```
1. Request arrival → CPU engine (prefill dispatch)
2. CPU: prefill 수행 (AMX BF16 GEMM), K/V 생성
3. CPU → GPU: K/V cache transfer via DMA
4. GPU: decode 수행 (IPEX single_query_cached_kv_attention)
5. Response stream
```

### KV cache transfer

7B 기준 prefill 출력 K/V:
- shape: `(seq_len, n_kv_heads × head_dim × 2 × n_layers)`
- 16K seq × 8 heads × 128 dim × 2 (K+V) × 80 layers × 2 bytes (BF16) = **5.2GB per request**
- PCIe 4.0 x16 ~32GB/s → transfer ~160ms

**비용 분석**:
- GPU decode 1 req 16K context: TPOT ~50ms × 16K → 820s → KV transfer 160ms 는 무시 가능
- 단 **batch 16K req 동시**: transfer bandwidth 병목 가능

### vLLM disaggregated stub

`vllm/engine/disaggregated/` 에 `coordinator.py`, `kv_transfer.py` stub. **hybrid engine 과 미통합** — 실제 prefill/decode split 경로 없음.

### CPU prefill 성능 실증 부족

- IPEX AMX BF16 이 대형 GEMM (M=1024+) 에서 H100 대비 얼마나 빠른지 unknown
- SPR AMX theoretical peak 8 TFLOPS (BF16) per socket × 2 = 16 TFLOPS
- H100 BF16 theoretical 1 PFLOP per GPU × 4 TP=4 = 4 PFLOP (250× 차이)
- 단 prefill 은 memory 제약 (weight + KV) 가 CPU 에 유리할 수 있음

---

## 관련 참고 문헌

- **Splitwise (Patel et al. MICRO'24)**: https://arxiv.org/abs/2311.18677
- **DistServe (Zhong et al. OSDI'24)**: https://arxiv.org/abs/2401.09670
- **Mooncake (Moonshot AI)**: https://arxiv.org/abs/2407.00079 — disaggregated serving
- **vLLM disaggregated serving proposal**: https://github.com/vllm-project/vllm/issues/8498
- **PagedAttention (Kwon et al. 2023)**: https://arxiv.org/abs/2309.06180 — KV cache 구조
- **IPEX AMX BF16 prefill**: https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance.html
- **TODO v5 §1.4 A3 Long-context P/D disagg**: `/vllm_hybrid/old_doc/TODO_v5_20260415.md`

---

## 구체 작업

### 사전 평가
- [ ] **Long-ctx workload 환경 구축**: 16K/32K input 벤치 (§TODO H100 확장 실험 §3.2)
- [ ] **GPU prefill 과 decode 의 비율 측정**: 현재 GPU 가 prefill 에 얼마나 시간 소비하는지
- [ ] **CPU AMX BF16 prefill throughput 실측**: 동일 shape 에서 GPU 대비

### 설계
- [ ] **Hand-off 프로토콜**: Request 가 prefill 단계 완료 시점에 CPU → GPU 이동 신호
- [ ] **KV transfer 경로**: `cudaMemcpyAsync` + pinned host memory + `torch.cuda.Stream`
- [ ] **CPU prefill-only engine**: decode 경로 비활성, AMX BF16 최적화 집중
- [ ] **GPU decode-only engine**: prefill 수신 거부 (long-ctx 만)

### 구현
- [ ] **`vllm/engine/disaggregated/coordinator.py`** 확장 — hybrid 구조와 병합
- [ ] **`vllm/engine/disaggregated/kv_transfer.py`** — 실제 DMA path 구현
- [ ] **`HybridConfig.disaggregated_mode: bool = False`** 필드
- [ ] **Router 수정**: long-ctx request → CPU prefill, short-ctx → 기존 hybrid 경로
- [ ] **GPU side KV receive**: paged block layout 으로 placement

### 검증
- [ ] **정확도**: P/D split vs non-split 결과 동일
- [ ] **TPOT p99 16K input**: 개선 여부
- [ ] **KV transfer 시간**: PCIe BW utilization

---

## 성공 조건

1. ✅ 16K input 에서 GPU TPOT p99 20%+ 감소
2. ✅ KV transfer 가 decode 시간 대비 <10%
3. ✅ Short-ctx workload 에서는 P/D off 로 기존 성능 유지
4. ✅ 정확도 열화 0
5. ✅ 32K input 에서도 작동 (scalability)

---

## 의존성

- **선행**: §TODO H100 long-context workload 구축 (현재 없음)
- **선행**: §06 hot path wiring, §07 AMX dispatch (CPU prefill 최적화)
- **무관**: 현 workload (128/128) 에서는 본 기법 skip

---

## 리스크

- **현 workload 무효**: 128/128 bench 에서 prefill 비중 작음 → 이득 0
- **KV transfer bandwidth 병목**: 동시 16K req 다수 시 PCIe 포화
- **CPU AMX prefill 이 GPU 보다 여전히 느림** (가능성 큼): 이 경우 P/D 가 아니라 GPU prefill + CPU decode 역방향이 더 적합할 수도
- **vLLM disaggregated stub 성숙도 불명**: 실제 production-ready 에 필요한 작업량 불확실

---

## 스택 호환성

- §20 KV Offload: prefill 결과를 GPU 대신 CPU DRAM 저장 → 다른 P/D variant
- 경로 1 (§02-§17) 과 독립 trajectory
- 경로 2 의 장거리 track (§21 ScoutAttention, §22 NEO asymmetric) 과 함께 고려

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_PD_DISAGG` | `0` (기본) / `1` | P/D split 활성 (long-ctx 전용) |

전체 flag 테이블: [README.md](./README.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `vllm/engine/disaggregated/coordinator.py` — 기존 stub
- `vllm/engine/disaggregated/kv_transfer.py` — 기존 stub
- `vllm/v1/engine/hybrid_core.py` — P/D router 분기
- `vllm/config.py` — `HybridConfig.disaggregated_mode`
- `vllm/v1/attention/backends/cpu_attn.py` — CPU prefill path
