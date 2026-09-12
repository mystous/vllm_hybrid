# RBC-Attn v0.2 — 최종 결과

이 문서의 모든 수치는 `results/v02/**` 의 원본 JSON 에서 생성됐다. 측정하지 않은 값은 `NOT_RUN` 으로 남기고 계산해 채우지 않았다.

## 측정 환경

- GPU: NVIDIA H100 80GB HBM3 (UUID 43a78ae4-f89f-ca72-2ff2-27e8a8a27f74), SM 132
- 사용 장치 수 4, 논리 ordinal 0
- 버전: {'flashinfer': '0.6.17', 'available': True, 'error': None, 'torch': '2.13.0+cu130', 'cuda': '13.0', 'device': 'NVIDIA H100 80GB HBM3', 'triton': '3.7.1'}
- 생성 시각 기준 원본: 2026-09-12T00:44:11

## 1. 최종 판정 표 (§14.1)

감소율은 §1.3 의 `gain = 1 - T_new/T_best_baseline` 이다. 음수는 기준선보다 **느려졌다**는 뜻이다.

| 입력 | mask 출발점 | 비교 모드 | 최강 기준선/시간 | RBC 선택/시간 | 감소율 | non-Q RBC 비율 | 정확성 | 판정 |
|---|---|---|---:|---:|---:|---:|---|---|
| clustered 256:16:8 | GPU | pool-fresh | q_outer / 767.0us | signature / 811.7us | -5.8% | 0/189 (0%) | pass | 개선 없음 |
| common_private 256:16:8 | GPU | pool-fresh | q_outer / 717.3us | bounded_merge / 780.1us | -8.8% | 0/189 (0%) | pass | 개선 없음 |
| random 256:16:8 | GPU | pool-fresh | q_outer / 765.1us | signature / 826.2us | -8.0% | 0/189 (0%) | pass | 개선 없음 |
| clustered 512:16:4 | GPU | pool-fresh | q_outer / 959.2us | signature / 1026.7us | -7.0% | 0/189 (0%) | pass | 개선 없음 |
| common_private 512:16:4 | GPU | pool-fresh | q_outer / 861.9us | bounded_merge / 989.1us | -14.7% | 0/189 (0%) | pass | 개선 없음 |
| random 512:16:4 | GPU | pool-fresh | q_outer / 955.6us | bounded_merge / 1141.2us | -19.4% | 0/189 (0%) | pass | 개선 없음 |
| clustered 512:16:8 | GPU | pool-fresh | q_outer / 933.3us | signature / 1040.9us | -11.5% | 0/189 (0%) | pass | 개선 없음 |
| common_private 512:16:8 | GPU | pool-fresh | q_outer / 891.6us | bounded_merge / 940.7us | -5.5% | 0/189 (0%) | pass | 개선 없음 |
| random 512:16:8 | GPU | pool-fresh | q_outer / 953.1us | bounded_merge / 1085.0us | -13.8% | 0/189 (0%) | pass | 개선 없음 |
| 실제 capture | 실제 위치 | 동일 경계 | NOT_RUN | — | — | — | — | NOT_RUN (capture 없음) |

## 2. §13.1 GPU 단계 게이트

- Nq512~8192 / G8: 0/27 영역 통과
- Nq128~512 / G4·8·16: 46/81 영역 통과, 감소율 중앙값 -35.7%, 최대 -69.5%

## 3. P4 선택기 품질 (§7.7 진단 기준)

- 교정 seed [0, 1, 2, 3, 4] / 평가 seed [101, 102, 103, 104, 105, 106, 107, 108, 109]
- regret 중앙값 3.22% (기준 3%), 최악 215.11% (기준 5%)
- `RBC_active_fraction` = 0/189 = 0.000
- 선택기 CPU 중앙값 799us (probe + 선택된 계획 생성)

## 4. 변경별 효과 분해 (§14.1 두 번째 표)

GPU 실행시간의 상대 변화다. **음수가 시간 감소(개선)**, 양수가 악화다. 각 행은 같은 탐색 예산 안에서 정책별 최적 max_m 을 고른 뒤 패턴별 값의 중앙값이다. 마지막 열은 같은 실행기에서 RBC 계획이 Q-outer 계획보다 얼마나 느린지이며, 양수면 RBC 분해가 손해라는 뜻이다.

| 변경 | 단독 효과 | 누적 효과 | 공통 Q-outer 도 얻은 효과 | RBC−Q (양수=RBC 손해) |
|---|---|---|---|---|
| P1 (강제 분할 제거) | 1.8% | 1.8% | 1.8% (Q-outer 기준) | 30.5% (RBC−Q) |
| P2 (direct output) | -7.8% | -6.1% | -7.8% (Q-outer 기준) | 37.6% (RBC−Q) |
| P3 (BN=64) | 35.3% | 27.0% | 35.3% (Q-outer 기준) | 9.6% (RBC−Q) |
| P3 (late V) | 3.1% | -3.2% | 3.1% (Q-outer 기준) | 28.3% (RBC−Q) |
| P3 (FULL fast path) | 0.1% | -6.0% | 0.1% (Q-outer 기준) | 32.7% (RBC−Q) |
| P4 (교정 선택기) | 아래 §3 | — | — | — |
| P5 (arena·단일 전송) | 아래 §5 | — | — | — |
| P6 (whole-query 기본) | 기본값 변경 (pipeline 은 비교 옵션) | — | — | — |

NCU 바이트와 일반 실행시간의 측정 조건은 다르다 (NCU 는 profiling 활성, 실행시간은 profiling 없음). 두 값을 같은 조건의 인과로 연결하지 않았다.

## 5. P5 runtime 비용 분해 (§8.7)

| 입력 | 후보 | cpu plan+pack | DMA | GPU | wall (fresh_gpu) | descriptor live/capacity | H2D | hot pin/alloc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| common_private 512:16:4 | q_outer | 405us | 42us | 214us | 862us | 25704/545740B | 66568B x2 | 0/0 |
| common_private 512:16:4 | signature | 528us | 34us | 205us | 992us | 42088/545740B | 42088B x1 | 0/0 |
| common_private 512:16:4 | bounded_merge | 522us | 47us | 212us | 989us | 25704/545740B | 66568B x2 | 0/0 |
| random 512:16:4 | q_outer | 474us | 42us | 252us | 956us | 36280/545740B | 98296B x2 | 0/0 |
| random 512:16:4 | signature | 675us | 33us | 241us | 1161us | 54856/545740B | 54856B x1 | 0/0 |
| random 512:16:4 | bounded_merge | 594us | 49us | 291us | 1141us | 36696/545740B | 98712B x2 | 0/0 |
| clustered 512:16:4 | q_outer | 445us | 46us | 265us | 959us | 36592/545740B | 99224B x2 | 0/0 |
| clustered 512:16:4 | signature | 563us | 31us | 229us | 1027us | 42032/545740B | 42032B x1 | 0/0 |
| clustered 512:16:4 | bounded_merge | 555us | 44us | 293us | 1074us | 39216/545740B | 101848B x2 | 0/0 |
| common_private 256:16:8 | q_outer | 316us | 46us | 170us | 717us | 15504/280524B | 40064B x2 | 0/0 |
| common_private 256:16:8 | signature | 391us | 34us | 167us | 781us | 23696/280524B | 23696B x1 | 0/0 |
| common_private 256:16:8 | bounded_merge | 361us | 48us | 175us | 780us | 15504/280524B | 40064B x2 | 0/0 |
| random 256:16:8 | q_outer | 343us | 44us | 186us | 765us | 19232/280524B | 51248B x2 | 0/0 |
| random 256:16:8 | signature | 413us | 34us | 168us | 826us | 23728/280524B | 23728B x1 | 0/0 |
| random 256:16:8 | bounded_merge | 386us | 45us | 196us | 832us | 19760/280524B | 51776B x2 | 0/0 |
| clustered 256:16:8 | q_outer | 339us | 45us | 191us | 767us | 19320/280524B | 51512B x2 | 0/0 |
| clustered 256:16:8 | signature | 410us | 34us | 169us | 812us | 20792/280524B | 20792B x1 | 0/0 |
| clustered 256:16:8 | bounded_merge | 379us | 46us | 195us | 820us | 20280/280524B | 52472B x2 | 0/0 |
| common_private 512:16:8 | q_outer | 463us | 42us | 189us | 892us | 30824/545740B | 79880B x2 | 0/0 |
| common_private 512:16:8 | signature | 580us | 32us | 200us | 1024us | 47208/545740B | 47208B x1 | 0/0 |
| common_private 512:16:8 | bounded_merge | 511us | 43us | 187us | 941us | 30824/545740B | 79880B x2 | 0/0 |
| random 512:16:8 | q_outer | 507us | 43us | 212us | 953us | 38312/545740B | 102344B x2 | 0/0 |
| random 512:16:8 | signature | 612us | 32us | 230us | 1093us | 46904/545740B | 46904B x1 | 0/0 |
| random 512:16:8 | bounded_merge | 579us | 45us | 258us | 1085us | 39416/545740B | 103448B x2 | 0/0 |
| clustered 512:16:8 | q_outer | 489us | 43us | 211us | 933us | 38448/545740B | 102752B x2 | 0/0 |
| clustered 512:16:8 | signature | 573us | 32us | 217us | 1041us | 41440/545740B | 41440B x1 | 0/0 |
| clustered 512:16:8 | bounded_merge | 556us | 50us | 247us | 1053us | 40352/545740B | 104656B x2 | 0/0 |

## 6. 해석 제한

- 이 결과는 합성 mask 연산자 수준이다. 모델 tok/s·서비스 SLO·신규성 주장으로 확장하지 않는다.
- Q-outer fallback·pinned pool·일반 kernel 개선은 RBC 분해의 성과가 아니다. §4 표에서 '공통 Q-outer 도 얻은 효과' 와 'RBC 추가 효과' 를 분리했다.
- KV 합집합 바이트로 HBM 비용을 계산하지 않았다. 방문량(`kv_block_visits`) 은 발행량 특징이며 실측 HBM 바이트가 아니다.
- pipeline 결과가 있어도 chunk serial 대비 개선을 whole-query 개선으로 쓰지 않는다.

