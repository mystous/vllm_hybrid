# 논문 골격 (워크샵 8쪽) — 초안 2026-09-08 14:35, K1 판정 반영

**가제**: Predicting the Bottleneck Transition in CPU–GPU Hybrid MoE Serving: A Pre-registered Mechanistic Model for Hot-Expert Residency

## 1. Introduction (1쪽)
- 문제: 480B급 MoE 를 GPU 4장 + 2소켓 Xeon 으로 서빙할 때, GPU 에 상주시킬 hot expert 수 H 는 운영자가 손으로 정한다 (KTransformers `--kt-num-gpu-experts`, CoX-MoE 정적 배치). H 를 늘리면 CPU 의 cold expert 스트리밍 (DDR 대역폭 한계) 이 줄지만 GPU expert 연산이 늘고 KV 자리가 준다.
- 관찰 (실측): H=80 에서 스텝 66ms = GPU 26 + CPU 대기 40 (DDR-bound); H=96 에서 39ms = GPU 27.7 + CPU 대기 9.9 (GPU-expert-bound). 전이점이 H=80~96 사이에 있다. C=32 처리량 319.9 → 490.9 tok/s (+53.5%), GSM8K 97.0%.
- 기여: (1) 마이크로벤치·라우팅 트레이스 파라미터만의 기계 모델 (8개 항, 순환 적합 없음) (2) 사전 등록 예측 — 12셀 격자, 셀별 오차 분포·순위 일치 (3) 전이점 위치의 사전 예측 (4) 부정 결과: micro-batch 교차 실행이 대역폭 한계 decode 에서 구조적으로 손해인 이유.
- 선행과의 위치 (K1): MoE-Lightning (HRM turning point, PCIe 예산 하 비율 결정, 오프라인 배치), MoE-Lens (하이브리드 성능 모델 94% 평균, CPU attention 토폴로지), CoX-MoE (VRAM 용량 하 AMX cold + GPU hot 배치, 오차 미보고), KTransformers (배치 1, 고정 배치). **H 축 전이점의 사전 등록 예측과 셀별 오차 분포는 선행에 없음.**

## 2. Background & Measurement Setup (1쪽)
- 스택: SGLang + KTransformers kt-kernel (AMXINT4), Qwen3-Coder-480B-A35B FP8, TP=4 H100, Xeon 8480+×2 (turbo OFF 2.0GHz), DDR5 실측 읽기 ~390 GB/s.
- 라우팅 트레이스 (sonnet, 18,776 토큰): hot-H 커버리지 (64: 94.2%, 80: 97.1%, 96: 98.6%, 112: 99.5%), 층당 기대 distinct cold expert (B=32: 12.4 / 6.7 / 3.2 / 1.3).

## 3. Mechanistic Step Model (2쪽)
- 구조 (프로파일 분해 근거): 스텝 = Σ층 [GPU 비-expert + GPU hot-expert + CPU 노출]; 직렬.
- 항과 출처 표: CPU 층 법칙 (expert 당 55µs + 행당 19µs + 고정 ~60µs; 480B 마이크로벤치, T=1 평평 구간 = 430 GB/s), GPU hot-expert 커널 (graph 재생 격자: 40 + 4.26·D_h µs/층), 비-expert 커널 범주 (프로파일), prefill 요청당 비용 × admission 배치 (out=1 벤치 + 스케줄러 로그), deferred 숨김 창 (MoE 이전 GPU 구간; topk 덤프로 cold 적중 = N/8 확인; KTransformers 의 deferral 창 서술과 일치), KV 용량 → 실효 배치, DDR 2-스트림 간섭 (같은 소켓 0.73 / 다른 소켓 0.89).
- 재예측 (파라미터 미사용 기측정 9셀): 중앙값 18.3% — 잔차 2종 (저동시성 prefill 겹침, KV thrash) 명시.

## 4. Pre-registered Prediction (1.5쪽)
- 12셀 (H×C×ctx×KV, 무작위 + 극단 강제), predictions.json 커밋 5e6459b8b (측정 전). 셀별 오차·순위 일치·binding 자원 예측 적중표. 전이점 그림: H 축에서 CPU 노출 vs GPU expert 시간의 교차.
- 실패 셀의 기전 분석 (있는 그대로).

## 5. Policy & DDR Contention (1쪽, 보조)
- 정책 = 모델 위 격자 탐색: 워크로드별 (H, N, KV) 선택 사전 등록 (C=64·긴 컨텍스트에서 H=80 선택 — hot 을 줄이고 KV 를 택함) → 실측 대조.
- DDR 경합: HiCache 공유 prefix 워크로드에서 host 읽기 × cold 스트리밍 간섭이 binding 인지 실측 (MoE-Lens 의 비-binding 결론과 대조).

## 6. Negative Result: Interleaved Micro-batches (0.5쪽)
- 두 graph 동시 재생: 격리 5중 + 비블로킹 신호식 대기 + 층 사다리 후에도 두 통신그룹 collective 정지. 그러나 모델이 먼저 말해 주는 것: 분할은 층당 CPU 호출 2회 → CPU 고정비·스트리밍 2배 → 순차 −20% (예측 112 vs 실측 109ms). 대역폭 한계 decode 에서 micro-batch overlap 은 구조적으로 손해.

## 7. Related Work (0.5쪽) — K1_DELTA 표 요약
## 8. Conclusion (0.25쪽)

## 그림 목록
1. 전이점: H 축 CPU 노출 vs GPU expert (예측 곡선 + 실측점)  2. 모델 항·출처 도식  3. 12셀 예측-실측 산점도 (오차 ±20% 밴드)  4. 정상 상태 스텝 분해 (H=80 vs 96 프로파일)  5. DDR 2-스트림 간섭  6. 교차 실행 부정 결과 (순차 모드 예측 vs 실측)
