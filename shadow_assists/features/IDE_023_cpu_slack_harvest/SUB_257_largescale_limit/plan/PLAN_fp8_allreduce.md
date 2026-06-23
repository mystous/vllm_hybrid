# 신규 알고리즘 계획: B200 multimem-PTX 저정밀 all-reduce + 신규 오차보상

## 동기 (측정)
대형모델(R1-671B/K-EXAONE-236B/405B) decode는 통신-bound (all-reduce ~31% cudagraph / ~80% eager).
TP 통신 불가피(부분합 GPU간 합산). vLLM/PyTorch는 **bf16 AR만**(FP8 multimem 미구현) → FP8 AR이 novel 코드 영역.

## 신규성 (config 아님, 코드)
- FP8로 통신량 2×↓ + B200 NVLink5 multimem PTX(in-fabric reduce) + **신규 오차보상**으로 정밀도 회복.
- 단일-패스 추론이라 training식 error-feedback 부적용 → 새 보상 기법 발명 필요(=논문 핵심).

## Phase
1. FP8 2-shot AR 구현(reduce-scatter FP8 + fp32 누적 + all-gather). bf16 AR 대비 지연·정밀도.
2. 오차보상: per-channel 동적스케일 + in-kernel stochastic rounding + outlier 채널 bf16 잔차(hybrid).
   목표: rel_err 2.6%→게이트(<~0.5%) 수준.
3. vLLM AR 경로 env-gated 통합 → 대형모델 end-to-end throughput + 분포동등 게이트.
4. 판정: 게이트 통과+통신↓=novel win(논문) / 실패=negative 특성화.

## 게이트
분포동등(max_logprob_diff≤0.5, ppl_rel≤0.1) 유지하며 AR 지연↓ → 대형모델 throughput↑.
