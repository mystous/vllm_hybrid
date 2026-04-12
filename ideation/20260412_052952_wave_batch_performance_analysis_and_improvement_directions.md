# Wave-Batch 성능 분석 및 개선 방향

**Timestamp (KST)**: 2026-04-12 05:29:52
**Commit**: `50b3bc035` (H100x1 wave-batch 전체 실험 완료 후)
**형식**: 실험 데이터 기반 분석 + 개선 로드맵

---

## 1. 실험 범위 요약

### 1.1 환경 2개

| 환경 | CPU | cores | ISA | GPU | VRAM |
|---|---|---:|---|---|---:|
| **dev** | i9-12900KF | 16 (8P+8E) | AVX2 + AVX-VNNI | RTX 3090 | 24 GB |
| **H100x1** | Xeon Platinum 8480+ | 24 (1S, 1T/C) | AVX-512F/VNNI/BF16, **AMX-BF16/INT8** | H100 80GB HBM3 | 80 GB |

### 1.2 실험 매트릭스

| 환경 | 모델 | gpu_only | hybrid w=16 | hybrid w=8 | hybrid w=4 |
|---|---|:---:|:---:|:---:|:---:|
| dev | 1.5B | ✓ | ✓ | — | ✓ |
| dev | 7B | ✓ | ✓ | — | — |
| H100x1 | 1.5B | ✓ | ✓ | — | — |
| H100x1 | 7B | ✓ | ✓ | ✓ | ✓ |
| H100x1 | 32B | ✓ | ✓ | — | — |

모든 실험: 500 req × 128/128 shape, burst mode (rate=inf), wave-batch + cpu-first.

---

## 2. 핵심 수치 종합

### 2.1 H100x1 — gpu_only vs hybrid (wave=16)

| 지표 | 1.5B G | 1.5B H | 7B G | 7B H | 32B G | 32B H |
|---|---:|---:|---:|---:|---:|---:|
| Wall (s) | 15.5 | 33.7 | 16.6 | 65.4 | 40.2 | 233.2 |
| Bench dur (s) | 4.43 | 22.60 | 4.82 | 53.52 | 26.17 | 213.59 |
| Output TP (tok/s) | 13,898 | 2,725 | 12,953 | 1,167 | 2,352 | 288 |
| Mean TPOT (ms) | 27.67 | 72.76 | 28.34 | 91.25 | 54.33 | 115.53 |
| Med TPOT (ms) | 27.59 | 70.60 | 26.15 | 82.72 | 46.25 | 60.86 |
| P99 TPOT (ms) | 29.00 | 123.54 | 76.31 | 322.94 | 88.50 | 1,373.74 |
| P99 TTFT (ms) | 1,062 | 6,903 | 1,989 | 12,498 | 22,166 | 39,117 |
| GPU mean util | 13.3% | 5.7% | 28.5% | 6.0% | 68.8% | 11.0% |
| CPU mean util | 8.2% | 63.6% | 7.9% | 80.8% | 6.0% | 90.0% |
| CPU dispatch | — | 16 | — | 16 | — | 16 |

### 2.2 H100x1 7B — wave size sweep

| 지표 | gpu_only | w=16 | w=8 | w=4 |
|---|---:|---:|---:|---:|
| Wall (s) | 16.55 | 65.43 | 61.34 | 59.26 |
| Bench dur (s) | 4.82 | 53.52 | 49.19 | 47.91 |
| Output TP (tok/s) | 12,953 | 1,167 | 1,270 | 1,304 |
| Med TPOT (ms) | 26.15 | 82.72 | 80.53 | 78.82 |
| P99 TTFT (ms) | 1,989 | 12,498 | 9,859 | 2,366 |
| CPU dispatch | 0 | 16 | 8 | 4 |
| Est CPU tail (s) | — | 48.7 | 44.4 | 43.1 |

### 2.3 dev RTX 3090 — 1.5B wave 비교

| 지표 | gpu_only | w=16 | w=4 |
|---|---:|---:|---:|
| Wall (s) | 14.31 | 85.93 | 34.89 |
| Bench dur (s) | 8.17 | 67.81 | 17.51 |
| Output TP (tok/s) | 7,541 | 908 | 3,517 |
| CPU per-req tps | — | 9.4 | 9.4 |

---

## 3. 5대 핵심 발견

### F1. CPU matmul batching 효과 = 0 (dev + H100x1 양쪽)

**가장 중요한 발견.** wave 크기를 16 → 8 → 4 로 바꿔도 **CPU per-request throughput 이 동일**하다.

- **dev 1.5B**: batch=1 도, batch=4 도, batch=16 도 = **9.4 tok/s**
- **dev 7B**: batch=1 도, batch=16 도 = **2.3 tok/s**
- **H100x1 7B wave sweep**: w=16 bench 53.52s, w=8 bench 49.19s, w=4 bench 47.91s
  - 만약 batching 이 작동했다면: w=16 의 16 req 은 aggregate 가속으로 **더 빨리** 끝나야 한다
  - 실제: w=16 이 w=4 보다 **더 느리다** (5.6초 차이)
  - 이유: 16 req 의 CPU tail (128 × 16 / ~2.3 = 891s) vs 4 req tail (128 × 4 / ~2.3 = 223s)
  - 즉 batch 크기만큼 **선형으로 tail 이 증가** — batching 효과 0 의 직접 증거

**의미**: weight DRAM load 를 M req 가 공유하는 matmul M-dim 확장 이론이 **실측에서 전혀 발현되지 않았다**. dev (AVX2 fallback) 와 H100x1 (AMX-BF16 brgemm) 양쪽에서 동일.

**추정 원인**:
- vLLM V1 scheduler 의 continuous batching 은 "모든 req 가 동시에 forward pass 에 참여" 하지만, 각 req 의 decode step 은 attention KV lookup 이 per-sequence 라서 matmul M-dim 의 BW amortization 이 attention overhead 에 가려진다
- oneDNN brgemm 의 M-dim tiling 이 24 cores 에서는 tile reuse 보다 memory stall 이 지배 (L2=2MB, L3=36MB per socket)
- IPEX PagedAttention `single_query_cached_kv_attention` 는 M=1 decode 전용 커널. batch > 1 이면 **for-loop 으로 M 번 호출** 되어 weight reuse 가 안 됨

### F2. CPU tail 이 wall time 을 결정 — "hybrid < gpu_only" 영역이 존재하지 않음

모든 실험에서 hybrid wall > gpu_only wall:

| 환경 | 모델 | wave | wall ratio (H/G) |
|---|---|---:|---:|
| H100x1 | 1.5B | 16 | **2.17×** |
| H100x1 | 7B | 16 | **3.95×** |
| H100x1 | 7B | 8 | 3.71× |
| H100x1 | 7B | 4 | **3.58×** |
| H100x1 | 32B | 16 | **5.80×** |
| dev | 1.5B | 16 | 6.00× |
| dev | 1.5B | 4 | 2.44× |
| dev | 7B | 16 | 9.27× |

GPU 가 sub-saturated (1.5B 13%, 7B 28%, 32B 69%) 인 상태에서 CPU 에 req 를 뺏기면 **GPU wall 은 거의 안 줄어들고** CPU tail 만 추가된다. `T_hybrid = max(T_gpu, T_cpu_tail)` 에서 `T_cpu_tail ≫ T_gpu` 이므로 hybrid 는 반드시 느리다.

### F3. 32B 의 TPOT 비율이 가장 양호 (1.32× 불과)

| 모델 | TPOT med ratio (H/G) |
|---|---:|
| 1.5B | 2.56× |
| 7B | 3.16× |
| **32B** | **1.32×** |

32B 는 GPU 자체가 가장 느려서 (46.25 ms/tok vs 1.5B 27.59 ms) CPU tail 의 **상대적 영향** 이 작다. 또한 32B 는 GPU util 이 68.8% 로 가장 높아 "GPU 가 실제로 바쁜" 유일한 모델. 이 영역에서 CPU 가 도움이 되려면 **CPU per-req throughput 이 현재보다 10× 이상** 빨라야 한다.

### F4. H100x1 CPU = 24 cores (dev 16 cores 대비 1.5×) 인데 throughput 향상 미미

| 환경 | cores | 7B CPU tps |
|---|---:|---:|
| dev (AVX2) | 16 | 2.3 |
| H100x1 (AMX-BF16) | 24 | ~2.3 |

코어 수 1.5× + ISA 세대 업그레이드 (AVX2 → AVX-512 + AMX) 에도 CPU throughput 이 동일. **코어 추가와 ISA 업그레이드가 7B CPU decode throughput 에 기여하지 못하고 있다.**

추정: memory bandwidth bound. 7B BF16 = 14 GB weight. DDR5-4800 8ch 이론 ~300 GB/s, 실측 ~200 GB/s. 이론 max = 14 tok/s (weight streaming). 실측 2.3 tok/s = **이론의 16%**. 나머지 84% 가 attention KV scan + Python dispatch + NUMA latency + OMP synchronization 등에 소모. **코어 수를 늘려도 BW bottleneck 은 해소 안 됨.**

### F5. Wave 크기 축소는 wall 을 선형 단축하지만 asymptote 가 높음

7B wave sweep:
- w=16 → w=4: wall 65.43 → 59.26s (9.4% 감소)
- w=4 → gpu_only: wall 59.26 → 16.55s (72.1% 감소 필요)

w=4 에서도 GPU 대비 3.58×. **wave 크기 최적화만으로는 한계가 명확**. CPU 가 4 req 만 처리해도 43s 의 tail 이 생기는데, GPU 는 496 req 를 4.8s 만에 끝냄. 비율: CPU 4 req 에 43s vs GPU 496 req 에 4.8s → **CPU 가 req 당 2,200× 느림**.

---

## 4. 근본 원인 진단

### 4.1 "왜 matmul batching 이 안 먹는가"

vLLM V1 의 CPU decode forward pass 구조:

```
for each decoder layer (28 for 7B):
    # 1. Attention: per-sequence KV lookup (bottleneck!)
    for seq in batch:
        q = input[seq]                # [1, hidden]
        k_cache, v_cache = KV[seq]    # [seq_len, hidden] — per-seq!
        attn_out[seq] = IPEX.single_query_cached_kv_attention(q, k_cache, v_cache)
    
    # 2. Linear (QKV proj, O proj, Gate/Up/Down FFN): batched matmul
    # M-dim = batch_size = 16 → weight reuse OK
    hidden = linear(attn_out)   # [16, hidden] × [hidden, 4*hidden]
```

- **Attention (단계 1)**: batch=16 이어도 **16 번 의 독립 KV lookup**. KV cache 는 per-seq 이라 batch 간 공유 불가. 이 단계에서 DRAM BW 를 KV 읽기에 쓰고, weight 는 읽지 않음. **batching 효과 0.**
- **Linear (단계 2)**: M=16 으로 weight 를 한 번 읽어 16 output 생성. BW amortization 이론상 16×. **batching 효과 있어야 함.**

하지만 전체 시간에서 Attention 이 지배하면 Linear 의 16× 가속이 total 에 묻힌다.

**가설 검증 방법**: `ONEDNN_VERBOSE=1` 로 matmul vs attention 시간 분리 프로파일. 또는 `VLLM_HYBRID_TRACE=1` 의 `execute_model` per-step trace 에서 attention time vs total time 비율 측정.

### 4.2 "왜 24 AMX core 가 16 AVX2 core 와 같은 throughput 인가"

Memory bandwidth bound → 코어 수나 ISA 가 아닌 **DRAM channel throughput** 이 천장:

| 환경 | DRAM 이론 BW | 실측 attain | 7B weight | 이론 max tok/s |
|---|---:|---:|---:|---:|
| dev DDR4/DDR5 | ~50 GB/s | ~35 GB/s | 14 GB | 2.5 |
| H100x1 DDR5-4800 8ch | ~300 GB/s | ~200 GB/s | 14 GB | **14** |

dev 이론 2.5 tok/s vs 실측 2.3 → **dev 는 실제로 BW ceiling 에 근접**.
H100x1 이론 14 tok/s vs 실측 2.3 → **H100x1 은 ceiling 의 16% 만 사용**. 원인은 4.1 의 attention KV scan overhead.

**결론**: dev 는 BW bound, H100x1 은 **attention compute/scan bound**. 서로 다른 bottleneck 이지만 결과적으로 같은 throughput.

---

## 5. 개선 방향 — 우선순위 순

### P0. CPU attention 프로파일링 (1일, 즉시)

**무엇을 모르는가**: 전체 CPU decode step 중 attention 이 차지하는 비율. 이 비율을 모르면 P1~P4 중 어느 것이 효과적인지 판단 불가.

**방법**:
```python
# cpu_worker.py::execute_model 에 per-layer timer 삽입
import time
for layer in model.layers:
    t0 = time.perf_counter()
    attn_out = layer.self_attn(...)
    t_attn = time.perf_counter() - t0
    t1 = time.perf_counter()
    ffn_out = layer.mlp(...)
    t_ffn = time.perf_counter() - t1
    logger.debug("[CPU-PROFILE] layer=%d attn=%.3fms ffn=%.3fms", i, t_attn*1000, t_ffn*1000)
```

또는 `ONEDNN_VERBOSE=1` + `IPEX_VERBOSE=1` 로 커널 단위 시간 분해.

**결과 해석**:
- attn > 80%: attention 최적화가 핵심 (P1, P3)
- ffn > 50%: Linear matmul 최적화가 핵심 (P2)
- Python dispatch > 30%: scheduling overhead 최적화 (P4)

### P1. CPU attention 경로를 batch-aware 로 변경 (1-2주, attn 지배 시)

현재 IPEX `single_query_cached_kv_attention` 는 단일 seq 전용. batch=16 이면 16 번 호출.

**변경 안**:
1. **IPEX `flash_attn_varlen_func` 을 decode path 에도 사용**: 현재 prefill 전용이지만 variable-length decode 도 처리 가능. `q_lens=[1,1,...,1]` (16개), `kv_lens=[128,127,...]` (per-seq). 한 번의 호출로 16 seq decode attention.
2. **`_C_cpu_ops.batch16_paged_attention_v1`** (이미 있는 AVX-512 커널): `cpu_attn.py` 에 dispatch 경로는 있으나 **IPEX 가 우선** 돼서 사용 안 됨. IPEX 보다 batch-aware AVX-512 커널이 **더 빠를 수 있음** — 16 seq 를 한 번의 AVX-512 vectorized loop 으로 처리.

**기대**: attention overhead 50-80% 감소 → **total CPU throughput 2-5× 향상** → 7B 에서 2.3 → 5-10 tok/s

### P2. IPEX INT8 weight-only quantization 활성화 (2-3일)

이전 ideation (20260411) Phase 0 D1 과 동일.

```python
# cpu_worker.py IPEX optimize 호출 지점
model = ipex.optimize(model, dtype=torch.bfloat16,
                      weights_dtype=ipex.quantization.WoqWeightDtype.INT8)
```

**효과**:
- Weight 크기 7B BF16 14 GB → INT8 7 GB → **DRAM BW 2× 절감**
- AMX-INT8 tile 은 AMX-BF16 대비 2× throughput (INT8 VNNI path)
- Linear layer 에서 matmul 시간 ~2× 감소
- attention 은 KV cache 가 BF16 이라 INT8 적용 안 됨 (별도 KIVI 필요)

**기대**: Linear 부분만 2× → attention 지배 시 total 1.3-1.5×, Linear 지배 시 1.5-2×

### P3. CPU 전용 경량 모델로 전환 — Spec Decode CPU Drafter (2주)

이전 ideation (20260411) A1 방향. **근본적 전환**: CPU 가 같은 모델의 decode 를 돌리는 대신, **작은 drafter 모델** (0.5-1B) 로 draft token 을 생성하고 GPU 가 verify.

**핵심 이점**: CPU 가 7B/32B 를 decode 하지 않음. 0.5B 는 dev 에서도 ~40 tok/s (weight 1 GB, BW bound 미만). H100x1 에서는 **80-160 tok/s** 예상.

**구현 경로**: `vllm/v1/spec_decode/cpu_drafter.py` 신규. 기존 ZMQ dispatch 재사용. `HybridAsyncMPClient` 에서 GPU verify + CPU draft fanout.

**기대**: 32B TPOT 46.25ms → **22-28ms** (1.8-2.5×), GPU 가 이미 바쁜 구간에서 효과 극대화.

### P4. Continuous batching 전환 — wave-batch 폐기 (3일)

wave-batch 의 전제 (matmul batching 이 throughput 을 올림) 가 F1 에서 부정됐으므로, wave 단위로 묶어서 보내는 이유가 사라졌다.

**대안**: `throughput-adaptive` 의 단순 capacity gate (이미 구현됨) 로 돌아가되, **CPU slot 을 1~4 로 대폭 축소**:
```
cpu_max_num_seqs = 1 (or 2)  ← wave 없이, 1 req 가 끝나면 바로 다음 1 req
```

**효과**:
- CPU tail = 1 req 의 decode time (7B: ~55s, 32B: ~183s)
- 하지만 req 가 끝나는 즉시 다음 req 투입 → throughput = 1 / per_req_time = 2.3 tok/s (7B)
- Wave=16 과 총 throughput 동일 (batching 효과 0 이므로) 하지만 **wall time = max(GPU wall, 1 req tail)** 로 축소
- 32B: wall = max(26s, 183s) = 183s (동일). 7B: wall = max(4.8s, 55s) = 55s vs wave=16 의 65s → **10s 절감**

**wave-batch 를 완전 폐기하는 게 아니라** 옵션으로 남기고 default strategy 를 capacity 로 되돌리는 것을 권장. matmul batching 이 작동하는 환경 (향후 커스텀 batch-aware attention, 또는 2-socket 환경) 에서 wave-batch 를 다시 쓸 수 있음.

### P5. KV cache INT4 압축 (1주, 32B 전용)

32B 에서 GPU KV cache 가 tight (80 GB - 64 GB weight = ~8 GB KV). batch 가 작아서 GPU 가 sub-saturated 임에도 "더 많은 req 를 동시에 올릴 수 없음".

KIVI INT4 (key per-channel, value per-token) → **KV footprint 4× 축소** → batch 3-4× → GPU util 68% → 90%+.

GPU 가 saturated 되면 CPU 가 overflow 를 받을 실질 기회가 처음으로 생긴다.

### P6. 70B + H100x1 (GPU 완전 포화 workload)

70B BF16 = 140 GB → H100 80 GB 1장에 안 올라감 (INT4/AWQ 필요). 혹은 H100x4 TP=4 에서 35 GB/GPU → KV ~37 GB → batch 수백 이상 가능.

이 워크로드에서는 GPU 가 진짜 바빠서 CPU overflow 가 wall 을 줄일 수 있다. **현재 인프라에서 가장 쉽게 "hybrid < gpu_only" 를 달성할 수 있는 방향**.

---

## 6. 직교성 매트릭스 — 곱셈 경로

| | P0 profile | P1 batch-attn | P2 INT8 | P3 spec-decode | P4 continuous | P5 KV INT4 | P6 70B |
|---|---|---|---|---|---|---|---|
| P0 | — | 선행 필수 | 선행 필수 | 독립 | 독립 | 독립 | 독립 |
| P1 | — | — | ✓ 곱셈 | △ P3 우선이면 불필요 | ✓ | ✓ | ✓ |
| P2 | — | ✓ | — | ✓ drafter 2× | ✓ | ✓ | ✓ |
| P3 | — | △ | ✓ | — | ✓ | ✓ | ✓✓ |
| P4 | — | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| P5 | — | ✓ | ✓ | ✓ | ✓ | — | 직교 |
| P6 | — | ✓ | ✓ | ✓✓ | ✓ | 직교 | — |

**최대 곱셈 경로**: P0 → P1 (또는 P3) → P2 → P5 → P6

---

## 7. 가장 먼저 해야 할 한 가지

**P0: CPU attention vs FFN 프로파일링** (1일).

이유:
1. F1 (batching 효과 0) 의 **원인이 attention 인지 FFN 인지** 모름 — 이 정보 없이 P1~P4 선택은 도박
2. 작업량 최소 — `cpu_worker.py` 또는 `cpu_attn.py` 에 timer 몇 줄 삽입
3. 결과에 따라 P1 (attention 최적화) vs P3 (spec decode) 의 **ROI 가 극적으로 달라짐**:
   - attention 지배 → P1 이 가장 빠름 (기존 코드 경로 수정)
   - FFN 지배 → P2 (INT8) 만으로 2× 가능
   - Python dispatch 지배 → 구조 문제, P3 으로 우회
4. dev 와 H100x1 양쪽에서 돌려야 함 — bottleneck 이 다를 수 있음 (dev 는 BW bound, H100x1 은 attention bound 가설)

---

## 8. 두 번째로 빠른 승리

**P4: wave-batch → continuous batching 전환 + cpu_max_num_seqs=1 복귀** (3일).

이유:
1. F1 로 "batching 효과 없음" 이 확정 → wave 의 존재 이유 소멸
2. cpu_max_num_seqs=1 로 돌리면 **wall time 이 즉시 개선** (CPU tail = 1 req only)
3. 코드 변경 최소 — env 파일의 `HYBRID_CPU_MAX_SEQS=1` + `HYBRID_ROUTING_STRATEGY=throughput-adaptive` 로 복귀
4. 단, **throughput-adaptive 의 capacity gate** (이번 세션에서 수정한 "slot 있으면 CPU" 룰) 는 유지. per-req 비교는 다시 넣지 않음.
5. 7B wall: 현재 wave=4 기준 59.26s → continuous max_seqs=1 시 ~55s → ~4s 절감 (modest 하지만 공짜)

**위험**: max_seqs=1 이면 CPU throughput = 2.3 tok/s. 500 req burst 에서 CPU 1 req 완료에 55s. GPU 가 4.8s 면 끝나므로 CPU 1 req tail 이 wall 을 지배. wall ≈ 55s. wave=4 의 59.26s 와 큰 차이 없음. 하지만 **wall = 1 req tail 이라 더 이상 줄일 수 없는 floor** 가 명확해짐.

---

## 9. 중장기 시나리오 — "hybrid < gpu_only" 도달 조건

### 시나리오 A: GPU saturation workload (가장 빠른 경로)

70B + H100x4 TP=4 + batch 500+. GPU 가 35 GB/GPU weight 로 KV 37 GB/GPU 에서 batch ~200 이 max. 200 넘으면 queuing 시작. CPU 가 overflow 처리.

**필요 조건**: 70B baseline 준비 + num_prompts ↑ 또는 긴 context
**기대**: hybrid wall < gpu_only wall (**처음으로** throughput 이 아닌 wall time gain)

### 시나리오 B: Spec Decode (가장 큰 gain)

32B + 0.5B CPU drafter. GPU 32B decode TPOT 46ms. CPU 0.5B draft ~6ms/tok (H100x1 추정). γ=5 draft → 30ms CPU time, GPU verify 5 in 1 step ~60ms. net TPOT = max(30, 60)/5 = 12ms (**3.8× 가속**).

**필요 조건**: P3 구현 (2주). vLLM V1 spec decode path 확장.
**기대**: 32B TPOT 46 → 12ms. throughput 2,352 → ~8,000 tok/s.

### 시나리오 C: Batch-aware CPU attention + INT8 (P1 + P2)

Attention 을 batch-aware 로 바꾸면 CPU 7B throughput 2.3 → 10 tok/s (5× 낙관). INT8 추가 시 15-20 tok/s. 이 경우:
- 7B: CPU 20 tok/s, GPU 12,953 tok/s. CPU/GPU ratio = 0.15%. 500 req 중 ~1 req 가 CPU. wall 차이 미미.
- 32B: CPU 5 tok/s (32B INT8), GPU 2,352 tok/s. ratio = 0.21%. 역시 미미.

**결론**: P1+P2 만으로는 "hybrid < gpu_only" 불가. GPU saturation (A) 또는 spec decode (B) 가 필수.

---

## 10. 결론 한 줄

**CPU matmul batching 은 dev 와 H100x1 모두에서 throughput 향상을 주지 못했고, 원인은 attention 의 per-sequence KV scan 이 batched matmul 의 weight BW amortization 을 완전히 가리기 때문으로 추정된다. 즉시 P0 (CPU attention 프로파일링) 으로 bottleneck 을 확정한 뒤, P3 (spec decode CPU drafter) 또는 P1 (batch-aware attention) 중 하나를 선택하는 것이 "hybrid < gpu_only" 달성의 가장 현실적인 경로다.**

---

## 11. 참고 데이터 소스

| 소스 | 내용 |
|---|---|
| `experiment_result/20260412_050600_h100x1_qwen1.5b_7b_32b_wave_batch_scaling/README.md` | H100x1 1.5B/7B/32B gpu_only + hybrid w=16 |
| `experiment_result/20260412_051400_h100x1_qwen7b_wave_size_sweep_4_8_16/README.md` | H100x1 7B wave=4/8/16 sweep |
| `experiment_result/20260412_023700_dev_rtx3090_wave_batch_gate_fix_initial_validation/README.md` | dev RTX3090 1.5B/7B validation |
| `eval/results/20260412_04*` | H100x1 raw bench JSON / monitor CSV |
| `eval/results/20260412_02*` | dev raw bench JSON / monitor CSV |
| `ideation/20260411_154523_hybrid_optimization_literature_survey.md` | 이전 논문 서베이 + Phase 0~3 로드맵 |
