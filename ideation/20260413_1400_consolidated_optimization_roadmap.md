# vLLM Hybrid 최적화 로드맵 — 실측 기반 통합 업데이트

**Timestamp**: 2026-04-13 14:00
**이전 문서 통합**: 0411 논문 서베이 / 0412 wave-batch 분석 / 0412 many-core 서베이 / 0413 B-plans / 0413 deep-research / 0413 CPU-GPU 병행 최적화 방법
**실측 데이터**: dev RTX3090 5 runs + H100x1 KVM 8 runs + H100x4 KVM + H100x4 물리 + H100x8 물리

---

## 1. 실측으로 확정된 사실 (가설 → 팩트 전환)

### 확정 1: CPU matmul batching 효과 = 0

- dev (AVX2 16C), H100x1 KVM (AMX 24C), H100x4 KVM (AMX 76C) 모두 동일
- `cpu_max_num_seqs=1/4/8/16` 전부 per-req throughput 동일 (7B: 2.3 tok/s, 1.5B: 9.4 tok/s)
- **원인**: IPEX `single_query_cached_kv_attention` 이 batch 내 seq 별 for-loop. matmul M-dim 확장이 attention 의 per-seq KV scan 에 가려짐
- **Layer breakdown 실측 (H100x4 KVM)**: FFN 72%, Attention 18%, Other 10% → **FFN 이 지배적이나 batching 효과가 없는 이유는 attention 의 per-seq 구조와 oneDNN tiling 문제가 복합**
- **→ 이전 가설 "batch 로 BW amortization" 폐기. wave-batch 전략의 전제 소멸**

### 확정 2: KVM 은 L3 16 MB + BW 26.5 GB/s 이중 제한

- 물리 Xeon 8480+: L3 105 MB, BW ~300 GB/s
- KVM guest: **L3 16 MB (15%), BW 26.5 GB/s (8.8%)**
- 동일 thread 수에서 KVM CPU 가 물리 대비 **10× 느림**
- **→ KVM 환경에서의 CPU 최적화는 물리 머신과 완전히 다른 전략 필요**

### 확정 3: oneDNN FFN GEMM 에 76 thread 절벽

- N=9728 (FFN up/down) 에서 76 thread 가 64 thread 대비 **5-8× 성능 추락**
- N=3584 (QKV) 는 76 thread 에서 최고 성능 → shape × thread 조합 의존
- **→ 단일 OMP_NUM_THREADS 를 전 연산에 적용하는 것이 근본 문제**

### 확정 4: 2-socket NUMA 분할이 many-core 의 핵심

- H100x8 물리 (2S × 56C): CPU tail **5.2초**, hybrid TPOT 21.50 ms (gpu_only 20.56 대비 1.05×)
- H100x4 물리 (1S × 96C): CPU tail **53초**, hybrid TPOT 65.39 ms (3.04×)
- H100x4 KVM (1S × 96 vCPU): CPU tail **18초** (BW 제한이 오히려 thread 압력을 낮춤)
- **→ `num_cpu_engines = num_numa` 설계가 실측으로 증명됨. SNC 또는 2-socket 이 필수**

### 확정 5: 물리 H100x8 에서 hybrid 가 "거의 작동"

- TPOT: 20.56 → 21.50 ms (**+4.6% 만**)
- Wall: 12.89 → 18.77 s (1.46×)
- **→ "hybrid < gpu_only" 는 아직 달성 안 됐지만 TPOT 기준으로 거의 등가. wall time 이 문제**

---

## 2. 이전 방안들의 상태 재평가

### 이전 문서 (0411 서베이) 의 A1~A4 + D1 + E1

| ID | 방안 | 이전 우선순위 | 실측 후 재평가 | 비고 |
|---|---|---|---|---|
| A1 | Spec decode CPU drafter | ⭐⭐⭐⭐⭐ | **⭐⭐⭐⭐⭐ 유지** | CPU 가 "다른 일" 을 하는 유일한 경로. 물리 머신에서 실현 가능 |
| A2 | KV cache CPU offload | ⭐⭐⭐⭐ | **⭐⭐⭐⭐ 유지 (B3 으로 구체화)** | KIVI INT4 + LMCache 가 핵심. 0413 B-plans 에서 구체 설계 완료 |
| A3 | P/D disaggregation | ⭐⭐⭐⭐ | ⭐⭐⭐ 하락 | 현재 workload (128/128 short) 에서 효과 미미. 16K+ 에서만 의미 |
| A4 | AMX-INT8 dispatch | ⭐⭐⭐ | **⭐⭐⭐⭐ 상승** | 0413 문서에서 Head Folding + VNNI pre-pack 으로 구체화. GEMM 절벽 우회 가능성 |
| D1 | IPEX WoQ INT8 | Phase 0 | **폐기** | Layer breakdown 에서 FFN 72% 이지만 batching 효과 0 → INT8 로 FFN 2× 해도 attention per-seq 구조가 남음 |
| E1 | MoE expert offload | ⭐⭐⭐ | ⭐⭐ 하락 | Dense 모델 실험에 집중. MoE 는 별도 트랙 |

### 0413 새 문서의 B1~B4

| ID | 방안 | 핵심 | 실측 적합성 |
|---|---|---|---|
| **B1** | **NEO 비대칭 어텐션 오프로딩** | GPU linear 실행 중 CPU 가 일부 req 의 attention 을 병렬 처리 | ⭐⭐⭐⭐ — H100x8 (2-NUMA) 에서 가장 자연스럽게 적용. 현재 "request-level partition" 을 "layer-level partition" 으로 전환 |
| **B2** | **ScoutAttention CPU 선행 계산** | CPU 가 1 layer 앞서 top-k KV block 예측 + partial attention | ⭐⭐⭐⭐⭐ — CPU→GPU PCIe 전송보다 CPU 직접 attention 이 6.7× 효율. Layer pipeline 으로 GPU idle 최소화 |
| **B3** | **KV INT4 + CPU DRAM 확장** | GPU KV 4× 축소 → batch 3-4× → GPU saturation | ⭐⭐⭐⭐ — 70B / 16K+ 에서 핵심. FP8 KV 는 vLLM 기본 지원 (즉시) |
| **B4** | **활성화 희소성 CPU 보정** | Dense 모델 40-50% 희소성 → GPU 가 중요 뉴런만 GEMM, CPU 가 보정 | ⭐⭐⭐ — batch 커지면 희소성 소멸 (Polar Sparsity 경고) |

### 0413 deep-research 핵심 인사이트

- **"CPU 는 부가 처리가 아니라 vLLM 성능 상한을 결정하는 병목"** — 작은 모델 + 빠른 GPU 에서 CPU 오버헤드가 지배적
- **vLLM V1 의 엔진 코어 busy loop** 이 CPU starvation 에 민감 → hybrid 로 CPU 코어를 뺏으면 오히려 GPU 가 놀 수 있음
- **최소 코어: 2 + N(GPU 수)** — H100x4 는 최소 6 코어를 vLLM 시스템에 남겨야 함

### Head Folding + VNNI Pre-Pack (사용자 제안)

| 기법 | 효과 | 적용 조건 |
|---|---|---|
| Head Folding (GEMV → GEMM) | AMX tile 을 꽉 채워 decode 가속 | MLA 구조 (DeepSeek) 에서 직접 적용. GQA (Qwen) 에서는 batch fold 변형 필요 |
| Load Once Pack Twice | Attention 의 NT/NN GEMM 에서 KV 를 1 회만 로드, register 에서 2 형태 pack | 모델 무관, 즉시 적용 가능. `csrc/cpu/` 커널 수정 |

---

## 3. 통합 로드맵 — 우선순위 재정렬

### Tier 0: 즉시 (설정 변경만, 1일)

**0-A. H100x4 물리에서 `HYBRID_CPU_THREADS=48` 테스트**
- GEMM 절벽 (76t) 회피. 48t 에서 모든 FFN shape 안정
- 7B hybrid 가 완료되지 않았던 문제 해결 가능
- 나머지 48 core 는 vLLM system (API server, EngineCore, GPU workers) 에 사용

**0-B. vLLM FP8 KV cache 활성화**
- `--kv-cache-dtype fp8` 한 줄 → 즉시 2× KV 메모리 절감
- H100 하드웨어 FP8 텐서코어 활용
- batch size 2× → GPU util 상승 → hybrid 에서 CPU overflow 발생 조건 생성

### Tier 1: 단기 (1-2주)

**1-A. ScoutAttention 개념 검증 (B2)**
- CPU 가 1 layer 앞서 top-k KV block 예측 → partial attention 계산
- 기존 dual-process 인프라 (ZMQ, CPU EngineCore) 를 그대로 활용
- **vLLM 스케줄러 변경 불필요** — CPU worker 가 내부적으로 layer pipeline
- Phase 1: cos similarity (Q_i ≈ Q_{i+1}) 실측 (Qwen2.5-7B/32B)
- 검증 지표: cos sim > 0.90 이면 진행, < 0.85 면 폐기

**1-B. NEO 비대칭 배치 분할 (B1) — H100x8 물리 전용**
- H100x8 이 2-NUMA + TPOT 1.05× (거의 등가) 인 환경에서 즉시 테스트 가능
- 매 decode step 에서 batch 를 GPU-batch (attention@HBM) + CPU-batch (attention@DRAM) 로 분할
- GPU linear 실행 시간 동안 CPU attention 겹침
- **현재 wave-batch 의 "request 단위" 를 "step 단위" 로 전환하는 것이 핵심 변경**

### Tier 2: 중기 (2-4주)

**2-A. CPU Attention 커널 교체**
- IPEX `single_query_cached_kv_attention` → batch-aware `flash_attn_varlen_func` (decode 에도 사용)
- 또는 `_C_cpu_ops.batch16_paged_attention_v1` (기존 AVX-512 커널) 을 IPEX 보다 우선 사용
- Head Folding: batch 16 × KV head 4 = M=64 로 fold → AMX tile 4 개 full 활용
- VNNI Pre-Pack: 모델 로드 시 weight 를 VNNI layout 으로 변환, attention 에서 "Load Once Pack Twice"

**2-B. Spec Decode CPU Drafter (A1)**
- 0.5-1B drafter (Qwen2.5-0.5B) 를 CPU EngineCore 에서 기동
- ZMQ 로 GPU verifier 에 draft token 전달
- 32B target + 0.5B draft: TPOT 46→22 ms 예상 (DuoDecoding 기준 2.05×)
- **물리 머신에서만 실행** — KVM 의 BW 26.5 GB/s 에서는 drafter 도 느림

**2-C. KIVI INT4 KV + CPU DRAM 확장 (B3)**
- GPU KV 4× 축소 → batch 288 (70B 기준, BF16 72 → INT4 288)
- CPU DRAM 에 overflow KV 저장 + LMCache 프리픽스 재사용
- 70B / 16K+ context 에서 핵심

### Tier 3: 장기 (1-2개월)

**3-A. 연산별 동적 Thread 수 (Sandwich 기법)**
- Prefill: 전체 코어 (compute-bound, GEMM scaling OK)
- Decode: LLC 클러스터당 1-2 core (BW-bound, thread 줄여야 빠름)
- `omp_set_num_threads()` 를 prefill/decode 전환 시 동적 호출
- oneDNN FFN 절벽 근본 해결

**3-B. 활성화 희소성 CPU 보정 (B4 + TEAL)**
- Dense 모델의 40% MLP 희소성 활용
- GPU: 중요 뉴런 60% GEMM, CPU: 나머지 40% INT8 보정
- 배치 1-8 에서만 유효 (batch 64+ 에서 희소성 소멸)

---

## 4. 폐기 / 후순위로 강등된 방안

| 방안 | 이유 |
|---|---|
| **wave-batch routing** | matmul batching 효과 0 이 실측으로 확정. wave 단위로 묶어도 per-req throughput 동일. continuous batching 으로 복귀 권장 |
| **D1 IPEX WoQ INT8 (CPU weight 단독)** | FFN 이 72% 지배적이나 attention per-seq 구조가 남아 total 개선 제한. Head Folding (2-A) 가 더 근본적 |
| **core_ratio 튜닝 (0.25~0.8)** | KVM 환경 한정 workaround. 물리 머신에서는 2-NUMA 분할이 답 |
| **cpu_max_num_seqs 증가** | batching 효과 0 으로 의미 없음. 1 이 최적 (continuous batching 에서) |

---

## 5. 환경별 권장 설정

### H100x8 물리 (2S × 56C, 2 NUMA) — 가장 유망

```bash
HYBRID_NUM_CPU_ENGINES=2       # 2 NUMA 자동
HYBRID_CPU_THREADS=0           # auto → 56 per NUMA
HYBRID_CPU_MAX_SEQS=1          # continuous batching (wave-batch 폐기)
HYBRID_ROUTING_STRATEGY=throughput-adaptive  # capacity gate
HYBRID_ROUTING_PRIORITY=cpu-first
```

### H100x4 물리 (1S × 96C, 1 NUMA) — GEMM 절벽 회피

```bash
HYBRID_NUM_CPU_ENGINES=1
HYBRID_CPU_THREADS=48          # 76t 절벽 회피, 나머지는 system 용
HYBRID_CPU_MAX_SEQS=1
HYBRID_ROUTING_STRATEGY=throughput-adaptive
```

### H100x4 KVM (1S × 96 vCPU, L3=16MB, BW=26.5 GB/s)

```bash
HYBRID_CPU_THREADS=24          # BW 포화 16-24t, 그 이상 무의미
HYBRID_CPU_MAX_SEQS=1
# ⚠ CPU 경로의 실용적 가치 제한. spec decode drafter 가 유일한 의미 있는 사용
```

---

## 6. 다음 즉시 행동

1. **H100x4 물리에서 `HYBRID_CPU_THREADS=48` + 7B hybrid 재시도** — GEMM 절벽 회피 확인
2. **H100x8 물리에서 `cpu_profile.sh` 실행** — 2-socket 의 GEMM/attention/BW scaling 프로파일
3. **Qwen2.5-7B 에서 Q_i vs Q_{i+1} cos similarity 측정** — ScoutAttention (B2) 적용 가능성 판단
4. **`--kv-cache-dtype fp8` 로 gpu_only baseline 재측정** — batch size 변화 확인

---

## 7. 참고 문헌 인덱스

| 논문/프로젝트 | arXiv / 출처 | 관련 방안 |
|---|---|---|
| DuoDecoding | 2503.00784 | A1 (spec decode) |
| ScoutAttention | 2603.27138 (DAC'26) | B2 (layer-ahead CPU attention) |
| NEO | 2411.01142 (MLSys'25) | B1 (비대칭 배치 분할) |
| KIVI | 2402.02750 (ICML'24) | B3 (KV INT4) |
| TEAL | 2408.14690 (ICLR'25) | B4 (활성화 희소성) |
| Sandwich | 2507.18454 | Tier 3 (decode thread 축소) |
| InfiniGen | 2406.19707 (OSDI'24) | A2/B3 (speculative KV prefetch) |
| SparAMX | 2502.12444 | A4 (AMX sparse decode) |
| Polar Sparsity | 2505.14884 | B4 경고 (batch 시 희소성 소멸) |
| LMCache | 2510.09665 | B3 (prefix KV 재사용) |
| SGLang Head Folding | SGLang blog | Head Folding GEMV→GEMM |
| AMD vLLM Multi-Instance | AMD blog 2025 | SNC/NUMA 분할 |
| llama.cpp #9588 | GitHub | barrier false sharing |
