# SUB_051 — Medusa multiple draft heads (CPU)

> **parent**: TSK_020 / 카테고리 A (Advanced spec decode, CPU draft)
> **status**: 대기 (plan only)
> **effort**: 3-5 일
> **CPU% target**: 30-50% / **throughput 가설**: SUB_047 대비 +10-20%
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §1

---

## 1. Mechanism

Medusa 는 main model 의 last hidden state 위에 **N (~4-5) 개의 병렬 small heads** 를 학습 — 각 head 가 +1, +2, ..., +N 위치의 token 분포를 직접 예측. autoregressive 가 아니라 **단일 step 으로 N 개 draft** 생성.

각 head 의 top-K candidate 들이 **tree 구조** 로 verify (Medusa-style tree attention).

본 lever 변형: heads 자체를 **CPU 에서 forward** (heads 가 작고 parallel → CPU 영역 유리).

```
[main GPU forward, last hidden]
  ↓
[CPU: N heads × top-K candidate 동시 forward (parallel)]
  ↓ (N×K candidate tree 영역 H2D)
[main GPU verify tree (1 forward)]
  ↓
[accept tree path]
```

## 2. 출처

| 자료 | 위치 |
|---|---|
| Medusa paper | [arXiv 2401.10774](https://arxiv.org/abs/2401.10774) — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" |
| Reference impl | GitHub `FasterDecoding/Medusa` |
| vLLM 내장 | `vllm/v1/spec_decode/medusa.py` (이미 존재, GPU heads) |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `vllm/v1/spec_decode/medusa.py` | heads device 옵션 추가 |
| `vllm/config/speculative.py` | `draft_device_type` field (SUB_050 와 공유) |
| `vllm/v1/worker/gpu_model_runner.py` | hidden state H2D + CPU heads forward + tree H2D |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | Medusa heads pretrained ckpt 검토 (FasterDecoding/medusa-llama-3-70b 등) | 0.5 일 |
| Phase 1 | `draft_device_type` field share (SUB_050 dependency) | 0.5 일 |
| Phase 2 | CPU heads parallel forward (N heads × top-K) | 1 일 |
| Phase 3 | tree H2D + GPU tree attention verify | 1 일 |
| Phase 4 | 정확도 + throughput 측정 | 1 일 |
| 총 | | **4 일** |

## 5. CPU% target / throughput 가설

- N heads (~4-5) × top-K candidate (~5-10) = 20-50 candidate vector 동시 forward
- heads small (per-head ~50-100M, total ~250-500M) + parallel
- CPU SIMD 8 thread/rank 영역 head batch forward ~3-5ms
- 가설: spec depth N=5 + top-K=5 영역 tree → accept rate ~65-70% (depth 5 → token/step ~4-5)
- throughput: SUB_047 +10-20% (단 tree expansion overhead 변수)

## 6. Risk

| 위험 | 완화 |
|---|---|
| tree attention 영역 GPU 영역 overhead 영역 small batch 영역 잘 안 fit | candidate top-K 조정 (K=3 → K=5 sweep) |
| Medusa heads pretrained ckpt 영역 Llama 3.3 70B 영역 본 head 영역 없을 수 있음 | medusa-1 영역 generic Llama-3 head 활용 또는 self-train (effort ↑↑) |
| heads forward 영역 K candidate 영역 CPU 영역 memory bandwidth 영역 bound | heads weight 영역 cache locality 영역 (small total weight) |

## 7. Dependencies

- SUB_050 와 `draft_device_type` field 공유 (먼저 land 권장)
- Medusa heads pretrained ckpt (FasterDecoding/medusa-llama-3 series)

## 8. Acceptance criteria

- [ ] CPU heads forward 성공 + tree verify 정상
- [ ] sonnet 500p × 8192 throughput ≥ 10,956 tps
- [ ] CPU busy ≥ 30%
- [ ] 정확도: top-1 token-id ≥ 99% vs vanilla
