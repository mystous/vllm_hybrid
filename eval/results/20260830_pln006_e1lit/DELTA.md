# PLN_006 E-1 — 문헌 정밀 조사 결과 (K1 게이트 판정)

조사일: 2026-08-30. 방법: 위험 지대 6종 웹 조사 (attention I/O 이론 / DNN 가속기 하한 / LLM 해석 모델 / offload 최적성 / MoE hybrid 비용모델 / 이종 pebbling 이론).

## 1. 인접 선행 delta 표

| 계열 | 대표 선행 | 그들이 한 것 | 우리와의 거리 |
|---|---|---|---|
| Attention I/O 하한 | Saha-Ye "I/O Complexity of Attention" (2402.07443), backward (2410.09397), 근사 attention (2605.23751) | red-blue pebbling으로 attention의 타이트 하한, FlashAttention 최적성 증명 | 단일 디바이스 2계층 (SRAM↔HBM), 연산기 1곳. expert 배치 없음 |
| DNN 가속기 하한 | HPCA'20 conv 가속기 (1911.05662), CNN comm bounds (2204.08279) | conv 층의 off-chip 이동량 하한 + 도달 dataflow | 역시 단일 가속기 2계층. 추론 서빙·MoE 아님 |
| **offload 최적성** | **FlexGen (2303.06865)** | offload 전략 search space 정의 + **자기 search space 내 계산 순서가 I/O 최적의 2× 이내임을 증명** + LP 탐색 | **가장 가까운 선행.** 단 dense 모델·전송-기반 offload (가중치를 GPU로 나름)·단일 GPU. "CPU가 그 자리에서 계산" 옵션이 배치 변수에 없고, 하한은 자기 search space 내부 한정 |
| LLM 해석 모델 | GenZ, LLMCompass, GUIDE, LLM-CoOpt (2602.09323) | 하드웨어 스펙 → 처리량 예측 (offload 포함) | 예측 모델이지 **하한 아님** — "이보다 빠를 수 없다"를 말하지 않음. 사전등록 검증 규율도 없음 |
| MoE hybrid 비용모델 | KTransformers SOSP'25 (ARI 분석), TriMoE (2603.01058), CoX-MoE (2605.17889), Beyond Uniform Experts (2606.29982) | 자기 정책의 비용 계산·배치 휴리스틱 | "우리 정책이 빠르다"의 근거용 비용모델. 정책 무관 하한·최적성 gap 없음 |
| **이종 pebbling 이론** | **다중 프로세서 red-blue pebbling (SPAA'24, 2409.03898)**, Savage k-계층, 분리 메모리 pebbling (MEMSYS'25) | 프로세서 여러 개 + 공유 slow memory의 계산·통신·메모리 트레이드오프 하한. 근사 불가능성 결과 포함 | **이론 기계는 이미 존재.** 우리가 발명할 것이 아니라 인용·적용할 것 |

## 2. K1 판정: **조건부 PASS**

- "tiered MoE 배치의 이동량 하한 + 실측 검증" 동일 주장 선행: **0건** (게이트의 죽는 조건 ≥2 미달)
- 단, 포위망이 조밀함: 이론 기계 (다중 프로세서 pebbling) 존재 + 가장 가까운 시스템 선행 (FlexGen 2×) 존재 + 우리 설정의 비용모델 다수. **"새 이론" 주장은 불가능하고 해서도 안 됨.**

## 3. 주장 위치 수정 (플랜 검증의 결론)

논문의 기여를 다음으로 재정의한다:

1. **첫 적용 + 예측 검증**: 다중 프로세서 pebbling 기계를 "CPU에 연산기가 있는 MoE expert 배치" 문제에 처음 실체화하고, 사전등록 예측으로 검증 (GenZ류와의 차별 = 하한 + 예측 규율)
2. **★ 헤드라인 지표 = 최적성 gap**: "실측 시스템 (kt-hybrid) 은 하한의 몇 %인가" — 어느 선행도 계산한 적 없는 수치. gap이 작으면 "스케줄링 논문들이 쫓는 여지가 이만큼뿐"이라는 지형 판정, gap이 크면 "헤드룸이 이만큼"이라는 기회 지도. **어느 쪽이 나와도 가치 있는 비자명 결과** → K3 게이트의 안전판
3. FlexGen delta 명문화 의무: 그들의 2× 는 전송-기반·dense·자기 search space 내부. 우리는 연산기-양쪽·MoE·정책 무관 하한

## 4. PLN_006 수정 사항

- §2 E0b: 하한 유도를 다중 프로세서 pebbling (SPAA'24) 위에 명시적으로 구축 (인용 기반)
- §2 E0c / §3 E1: 각 재예측·예측 셀에 **하한 대비 gap(%)** 컬럼 추가
- §6 판정: 논문 골격의 헤드라인을 "MoE hybrid 서빙의 최적성 gap 지도"로
