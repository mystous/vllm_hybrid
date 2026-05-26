# CLAUDE.md — IDE_019 구현 시 알아야 할 것

## 0. 핵심 규칙
- AMX intrinsic 은 Sapphire Rapids prod 머신에서만 직접 verify (dev 머신 미지원)
- Jacobi 의 lossless guarantee proof 필수 — 통합 sampling 과 정합성 보장
- physical core 0-111, HT 시블링 금지, 최대 100 core
- 측정 KST 표시
- 측정 1-run default
- commit/push 명시 지시 시만

## 1. Phase A input

| Phase A finding | IDE_019 의 의미 |
|---|---|
| SUB_106 AMX 22 TFLOPS peak | TSK_036 의 lower bound (Qwen 0.5B forward 의 throughput budget) |
| SUB_011 acceptance rate (chat α 81.2%) | TSK_037 의 per-workload best-source 입력 |
| SUB_117 N=32 10.24 TFLOPS available | task F (AMX draft) 의 CPU compute capacity |
| SUB_166 DMA 35 μs / 54 GB/s | draft logits CPU → GPU 의 transfer cost |

## 2. 통합 위치 (vLLM)

- AGSD router (`/tmp/sub094_router.py`) 의 4-method 분기 확장
- 새 method `cpu_amx_draft` 추가: HTTP POST classifier → CPU draft inference → DMA push → GPU verify
- ENV `VLLM_USE_CPU_DRAFT=1` 으로 activate

## 3. AMX draft head 의 model 선택

| model | hidden | layers | shape mapping | 측정 |
|---|---:|---:|---|---|
| Qwen 2.5-0.5B | 896 | 24 | TSK_036 main | SUB_106 의 qwen7b shape 보다 ~6× 작음 — fast |
| Qwen 2.5-1.5B | 1536 | 28 | TSK_036 secondary | SUB_106 의 qwen32b shape 보다 ~16× 작음 |
| distilled Llama-3 8B → 1.5B | 1536 | 28 | alternative | 추가 학습 필요 |

→ Qwen 0.5B 가 본 IDE 의 default draft model (target ≤ 5 ms / step).

## 4. risk + fallback

| risk | severity | fallback |
|---|---|---|
| AMX draft head 의 acceptance rate 가 ngram 보다 낮음 (per-workload) | medium | per-workload best-source 자동 분기 (TSK_037) |
| Jacobi 의 lossless proof breakdown (rejection sampler 통합) | high | per-token verify rejection 강화 |
| CPU → GPU draft logits transfer overhead (DMA 35 μs × per-step) | low | batch multiple draft steps (amortize) |
| 4-source router 의 overhead 가 net gain 잠식 | medium | per-request fast-path (classification 우회) |
