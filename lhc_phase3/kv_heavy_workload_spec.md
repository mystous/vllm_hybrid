# LHC Phase 3 — Task D: KV-heavy workload spec (NEO swap 발화)

**날짜**: 2026-06-08
**상위**: `lhc_phase3/PHASE3_VERDICT.md`

---

## 0. 배경 / 문제 (Phase 2 lesson)

Phase 2의 DSA 통합 측정(sonnet 200p × 3 sweep, conc=64, max-tok=512) 에서:
- DSA hook (NEO swap-out scatter) 실행 횟수 **0건** — boot log 에 `[LHC DSA] lane ENABLED` 는 떴지만 swap event 자체가 발화 안함.
- → Δ = -0.21% noise. lane 가치 측정 불가.

원인: sonnet 200p, KV 압박 < 임계치 → swap 미발생.

Phase 3 Task D 임무: **NEO swap ≥ 10건/min** AND **DSA hook coverage ≥ 50%** 가 보장되는 workload 3개 정의.

---

## 1. workload 정의

### 1.1 W-D1 — long-context 32K

| 파라미터 | 값 |
|---|---|
| 모델 | Llama-3.1-8B-Instruct (TP=8) |
| `max-model-len` | 32768 |
| target input | 28000 token (= 0.85 × 32K) |
| `max-tokens` | 4096 |
| 프롬프트 수 | 80 |
| concurrency | 8 (각 8 stream = 8 × 28K = 224K context, KV cache 압박) |
| workload type | sonnet (반복 문구로 입력 padding) |
| 측정 시간 | ≈ 8-10 분/run |

**근거**: 단일 request KV ≈ 28K tok × 32 layers × 8 KV-heads × 128 dim × 2 (K,V) × 2 (bf16) = 458 MB. 8 stream 동시 = 3.66 GB KV. TP=8 분산 → rank당 ≈ 460 MB. B200 184 GB 의 model + activation + KV → KV 전용 영역에 압박 충분히 가해짐. 200 동시면 swap-out 임계 진입.

→ NEO swap fire 기대. DSA host scatter (block staging) 가 hot path 로 활성.

### 1.2 W-D2 — multi-tenant + LoRA churn

| 파라미터 | 값 |
|---|---|
| 모델 | Llama-3.1-8B-Instruct (TP=8) + 4 LoRA adapter |
| `max-model-len` | 16384 |
| target input | 8000 token |
| `max-tokens` | 1024 |
| 프롬프트 수 | 240 (4 tenant × 60) |
| LoRA rotation | request 별 균등 rotation 4 adapter |
| concurrency | 32 |
| workload type | mix (chat 50% + code 50%) |
| 측정 시간 | ≈ 6-8 분 |

**근거**: LoRA adapter swap 이 host ↔ device 전송 발화. tenant 분리 + 다양한 LoRA → cache eviction churn 강제. DSA lane 의 H2D adapter copy 가 hot.

### 1.3 W-D3 — prefix-cache heavy (sharegpt prefix 강제 중복)

| 파라미터 | 값 |
|---|---|
| 모델 | Llama-3.1-8B-Instruct (TP=8) |
| `max-model-len` | 16384 |
| target input | 12000 token (8000 token shared prefix + 4000 unique) |
| `max-tokens` | 512 |
| 프롬프트 수 | 300 |
| concurrency | 64 |
| workload type | sonnet w/ shared 8K prefix → 강제 4-way prefix-cache hit |
| 측정 시간 | ≈ 5-7 분 |

**근거**: shared prefix 8K → prefix-cache hit 가 ≥ 50% expected. host-side radix-tree match (Phase 3 Task C **C3 winner**) 가 hot path. AMX byte scan 의 ortho-lane 가치 측정.

---

## 2. 측정 절차 (Task G 와 공유)

각 workload 별:
1. boot vllm serve (구성별: vanilla / lhc_dsa / lhc_amx / lhc_dsa_amx / lhc_full+suffix / lhc_full+fp8kv).
2. warmup: 동일 workload × 20 프롬프트.
3. sweep × 5 (seed 42-46).
4. boot log 에서 hook counter 추출:
   - `[LHC DSA] dsa_lane_stats: ops=, bytes=, fails=`
   - `[NEO] swap_out_count, swap_in_count`
   - `[C3 prefix] amx_lane_hits=`
5. gate 판정:
   - NEO swap ≥ 10 / min
   - DSA op_count ≥ 50% of NEO swap count
   - AMX C3 hits ≥ 100 / min (prefix-heavy 만)

---

## 3. 산출물 위치

```
lhc_phase3/runs_D/
├── W-D1/{cfg}_boot.log + {cfg}_s{1..5}.json + {cfg}_hook_stats.json
├── W-D2/...
└── W-D3/...
lhc_phase3/kv_heavy_workload_result.md   ← post-run summary (Task G 결과 통합)
```

---

## 4. 의존성

- W-D1: **DSA multi-engine (Task E)** 가 functional PASS 한 뒤. 그러나 single-engine 만 가능해도 hook coverage 측정은 가능.
- W-D2: vLLM LoRA serving stack 확인 필요 (`--enable-lora` + 4 adapter path).
- W-D3: vLLM `--enable-prefix-caching` (이미 default).

→ W-D3 부터 Task A (host dedicated WQ 완료) 와 무관하게 vanilla / lhc_amx (C3) 만 비교 가능. Task A 기다리는 동안 W-D3 pilot 실행 권고.
