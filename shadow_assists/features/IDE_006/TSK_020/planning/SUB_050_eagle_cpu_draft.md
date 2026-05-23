# SUB_050 — Eagle/Eagle2 CPU draft head

> **parent**: TSK_020 / 카테고리 A (Advanced spec decode, CPU draft)
> **status**: 대기 (plan only)
> **effort**: medium-large (3-5 일)
> **CPU% target**: 40-60% / **throughput 가설**: SUB_047 대비 +20~30% 가능
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §1

---

## 1. Mechanism

Eagle 은 small autoregressive draft head (1-layer transformer + LM head) 가 main model 의 **last hidden state** 를 입력으로 받아 다음 K token 의 후보를 생성한다. 그 후 main GPU model 이 1 forward 로 K 개를 동시 verify 후 accept/reject.

Eagle2 는 Eagle1 의 정적 chain 을 **dynamic tree** 로 확장 — 각 step 의 token 확률 분포에 따라 tree depth/breadth 조정.

본 lever 의 변형: draft head 자체를 **CPU 에서 inference** (head 가 작아서 ~50-200M params, CPU SIMD 로 충분). main GPU 는 verify only.

```
[main GPU forward, last hidden]
  ↓ (hidden state H2D→D2H copy)
[CPU draft head, autoregressive K step]
  ↓ (K draft tokens H2D→D2H copy)
[main GPU verify K+1 token batch (1 forward)]
  ↓
[accept up to first reject point]
```

## 2. 출처

| 자료 | 위치 |
|---|---|
| Eagle1 paper | [arXiv 2401.15077](https://arxiv.org/abs/2401.15077) — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" |
| Eagle2 paper | [arXiv 2406.16858](https://arxiv.org/abs/2406.16858) — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" |
| Reference impl | GitHub `SafeAILab/EAGLE` |
| vLLM 내장 | `vllm/v1/spec_decode/eagle.py` (이미 존재, GPU draft 지원) |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `vllm/v1/spec_decode/eagle.py` | draft model device 옵션 추가 (`draft_device="cpu"` arg) |
| `vllm/config/speculative.py` | `SpeculativeConfig.draft_device_type` field 추가 (default "auto", 옵션 "cpu"/"cuda") |
| `vllm/v1/worker/gpu_model_runner.py` | CPU draft model load + hidden state H2D/D2H copy path |
| `vllm/v1/spec_decode/utils.py` | CPU draft head loader (transformers + BF16) |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | Eagle head 영역 vLLM 내 적재 검토 (existing impl 검토) | 0.5 일 |
| Phase 1 | `SpeculativeConfig.draft_device_type` field + arg parse | 0.5 일 |
| Phase 2 | CPU draft model loader + transformers integration | 1 일 |
| Phase 3 | hidden state H2D/D2H async pattern | 1 일 |
| Phase 4 | 정확도 verify (Eagle accept rate ≥ 70% sonnet) + throughput 측정 | 1 일 |
| 총 | | **4 일** |

## 5. CPU% target / throughput 가설

- draft head 가 small (50-200M) + autoregressive K step (K=7 default) → CPU 영역 sustained compute
- CPU SIMD (AVX-512 BF16) + 8 thread/rank 사용 시 head step ~5-10ms/step
- K=7 step × 8 ms = ~56 ms/decode-step 의 CPU 영역 작업
- GPU 영역 verify 는 1 forward = ~7ms (full batch) → CPU 와 GPU pipeline overlap 시 CPU active 영역 50%+ 가능
- 가설: Eagle accept rate ~70% (vs ngram 60%) → throughput +15-25% over SUB_047

## 6. Risk

| 위험 | 완화 |
|---|---|
| head autoregressive 의 CPU latency 가 GPU verify wall 압도 (K step 직렬) | head 더 작게 (50M 이하) + AMX 활용 |
| hidden state H2D/D2H 영역 transfer overhead (4096 dim × bf16 = 8KB/step) | pinned mem + non_blocking |
| Eagle head pretrained ckpt 영역 model 별 (Llama 3.3 70B 영역 ckpt 영역 확인) | yuhuili/EAGLE-LLaMA3.3-Instruct-70B (Hugging Face 영역 있음) |
| vLLM 영역 Eagle adapter 영역 draft device 옵션 미지원 영역 깊은 변경 필요 | Phase 1 영역 patch 영역 land + upstream PR 영역 함께 |

## 7. Dependencies

- vLLM `eagle.py` 영역 GPU draft 영역 작동 확인 (smoke test)
- Eagle head pretrained ckpt 영역 다운로드 (yuhuili/EAGLE 영역 series)
- SUB_047 기 적재 (cap=8 영역 ngram base — Eagle 영역 대체)

## 8. Acceptance criteria

- [ ] CPU draft model load 성공 + verify forward 정상
- [ ] sonnet 500p × 8192 throughput ≥ 10,956 tps (SUB_047 Best 와 동등 이상)
- [ ] CPU busy ≥ 40% (Objective 정합)
- [ ] 정확도 검증: top-1 token-id 일치율 ≥ 99% (vs vanilla)
- [ ] 3-run avg / min / max 영역 variance < 1%
