# SUB_253 — Contextual activation sparsity (R6, GPU-direct 신규) — no-go (2026-06-16)

> 10라운드 루프 R6. 신규(vLLM 미적용): dense 모델 contextual activation sparsity 없음. GPU-direct
> (MLP matmul FLOP↓). Llama MLP down_proj(SiLU(gate(x))*up(x))의 중간활성(4d) 하위 |k%|를 0으로.

## probe (8B 전 레이어 MLP 패치, teacher-forcing)
| prune% | top1 일치 | max_logit_diff |
|---|---:|---:|
| 30% | 0.990 | 0.78 |
| 50% | 0.907 | 1.74 |
| 70% | 0.917 | 4.09 |
| 90% | 0.732 | 13.20 |

## 판정 = no-go (3가지)
1. **무손실 sparsity 낮음**: 30%도 logit_diff 0.78>게이트0.5, 50%+ top1 0.91로 분포 붕괴.
2. **FLOP 절감 미미**: 중간활성 prune은 down_proj(MLP의 ~1/3) 입력만 → 30%×1/3 ≈ 전체 decode ~7%.
3. **tensorcore 비호환(결정적)**: 비정형 sparsity는 sparse-gather 오버헤드 > dense TC 이득 → 실 GPU
   속도 이득 ~0. gate+up(MLP 2/3)까지 줄이려면 **trained predictor(Deja Vu)=training territory**.

## R7 (동거) — 양자화 self-speculative decoding probe (`exp/probe_quantselfdraft.py`)
draft=같은 모델 저비트, target=4bit. decode memory-bound → c≈b/4.
| draft b | accept(vs 4bit) | c | 1/(c+(1−a)) |
|---|---:|---:|---:|
| 2bit | 0.000 (RTN 파괴) | 0.50 | 0.67 |
| 3bit | 0.812 | 0.75 | 1.07 (낙관 상한) |
→ no-go: 3bit 1.07은 verify비용·2모델 메모리·커널 비효율 무시한 상한; 실측 break-even 이하. 저비트 draft는
싸질수록(2bit) accept 붕괴(0.0). vLLM 양자화 self-draft 미적용(신규)이나 net 무이득.

## 함의 (R4/R5/R6/R7 연속)
GPU-direct + 신규 + training-free + gate-safe 후보 3연속 no-go — 각기 다른 구조적 이유
(R4 CPU오버랩 천장 / R5 draft-accept 커플링 net-neg / R6 비정형 sparsity TC비호환+predictor training).
**메타 재확인**: 효과 통로는 비트폭(upstream) 또는 학습(training territory)으로만 귀결.
산출물: `exp/probe_actsparsity.py`.
