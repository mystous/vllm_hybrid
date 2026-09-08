# 논문 골격 (워크샵 8쪽) — 초안 2026-09-08 14:35, K1 판정 반영

**가제**: Predicting the Bottleneck Transition in CPU–GPU Hybrid MoE Serving: A Pre-registered Mechanistic Model for Hot-Expert Residency

> **M1 최종 (09-09 02:40)**: 트레이스 불일치를 고친 v3 셀 8개에서 사전 등록 모델 (v1 원본·v2 모두) 중앙값 |오차| 8%, 순위 일치 100% → 주장 2 (마이크로벤치·트레이스 파라미터만으로 절대 TPOT 사전 예측 ≤20%) **복원**. 남은 결함 = hot-96 소형 KV 초과 구간의 retract 항 (1셀 +47%).
> **M1/v2 검증 반영 (09-08 19:55, 역사)**: 절대-TPOT ≤20% 게이트는 실패(중앙값 M1 52%·v2 새셀 60%, 전부 긴 컨텍스트 과소). 그러나 구성 **순위 일치는 M1 93%·새 셀 100%**, 짧은 컨텍스트 절대값 ±7~30%. → 주장을 "절대 예측"에서 "**순위·전이점 사전 예측 + 짧은 컨텍스트 절대 예측**"으로 조정. 긴 컨텍스트 절대 TPOT 는 명시적 future work.

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

> **메커니즘 기여 추가 (09-09 00:55, IDE_033 계열)**: 논문의 주 기여를 "예측 모델" 에서 **"CPU 큐 중심 하이브리드 실행 재설계 + 모델" 로 격상**. (1) callback-free CPU↔GPU 핸드오프 (mapped go/done 플래그 + 스트림 memop + 폴러) (2) 전량-deferral 체제에서 빈 immediate 작업 제거 → 큐 = [신호, deferred] (3) 스핀 스레드의 AMX 워커 HT 형제 회피 배치. 합계 hot-96 C32 491→600 (+22.5%), C64 408→760 (+86%), 정확도 무손실. 계측 근거: TaskQueue 작업별 시간 (큐 99% 바쁨 → CPU-bound 판정), 프로파일 갭 구조. §3 모델은 CPU 항을 in-situ 계측으로 재교정하고 "메커니즘이 전이점 (H) 을 옮긴다" 를 §4 에 추가.

> **추가 기여 (09-09 02:20, IDE_034)**: (4) **phase-가중 hot expert 선택** — prefill 은 cold (token,expert) 쌍 수, decode 는 배치당 distinct cold expert 수가 비용이라 최적 집합이 다름 (현행 prompt 기반 hotmap: prefill 99.2% / decode 96.0%). α 스윕으로 처리량 최대점 (α=0.25) 과 TTFT↔TPOT 단조 교환 실측. 누적: C32 491→661 (+35%), C64 408→884 (+116%), 정확도 무손실. §3 모델에 in-situ CPU 법칙 (40+90+71·D_c µs, DDR 천장 80%) 과 phase 별 D_c 항, §5 정책에 α 결정 편입. 그림 추가: α–(TTFT, TPOT, tput) 곡선.

