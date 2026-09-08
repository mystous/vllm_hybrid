# InfiniGen (OSDI'24) — K1 정독 노트

- **서지**: Wonbeom Lee, Jungi Lee, Junghwan Seo, Jaewoong Sim (SNU). "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management." OSDI 2024. arXiv 2406.19707. 코드 github.com/snu-comparch/InfiniGen.
- **전문 접근 여부**: **불가**. 이 머신에서 arxiv.org / usenix.org 로의 HTTPS 연결이 타임아웃 (WebFetch·curl 모두). WebSearch 스니펫 (arXiv abs/pdf, USENIX 페이지, GitHub README, 2차 해설) 만 확보. 아래 인용은 스니펫에 나타난 문장이며 원문 페이지 대조는 못 했음. 원문에 없는 추정은 "추측" 표시.

## 문제·결정 변수
- 대상: dense LLM (OPT-13B 등) 의 **KV cache 를 CPU 메모리에 전량 두고** 매 decode 스텝 필요한 일부 토큰의 K/V 만 GPU 로 가져오는 offloading 서빙 (FlexGen 계열 위에 구현).
- 결정 변수: (a) 어떤 토큰의 KV 를 이번 층에서 GPU 로 prefetch 할지 — 이전 층 attention 입력과 다음 층 query weight 일부·key cache 일부로 "minimal rehearsal" 하여 중요 토큰을 추측 (b) CPU 측 KV pool 용량 한도 도달 시 어떤 항목을 evict 할지 (counter / FIFO / LRU). 오프라인으로 weight skewing (SVD 기반) 수행.
- 스니펫 인용: "InfiniGen leverages the key insight that a few important tokens that are essential for computing the subsequent attention layer in the Transformer can be speculated by performing a minimal rehearsal with the inputs of the current layer and part of the query weight and key cache of the subsequent layer."
- 스니펫 인용: "when the size of the CPU memory reaches a user-defined limit, the KV cache pool manager selects a victim KV entry for eviction ... In the counter-based policy, the pool manager increments a counter for each prefetched KV entry and selects a victim with the smallest count".

## 비용 모델 파라미터 출처와 검증
- 사전 예측용 성능 모델 **없음**. 설계 근거는 "PCIe 전송량 감소" 라는 정성적 논리 + 실측 (OPT-13B, 입력 1920 / 출력 128 / 배치 20, H2O·FlexGen·UVM 대비 1.63–32.93× latency, 처리량 27.36→41.99 tok/s vs H2O 21.31→25.70).
- 스니펫 인용: "The performance benefit mainly comes from the significantly reduced amount of KV cache to load from the CPU memory due to the dynamic approach."

## expert·KV 경합 취급
- MoE / expert 언급 없음 (스니펫 범위). KV 만 호스트에 둠. 가중치는 "GPU 에 최대한, 나머지 CPU" (FlexGen 설정) — 가중치·KV 가 호스트 대역폭을 공유한다는 모델링은 확인 안 됨.
- **호스트 DRAM 대역폭 모델링**: 아니오 (추측 포함). 병목은 일관되게 **PCIe** 로 서술 ("reduces the waste of PCIe bandwidth by only transferring the keys and values critical for attention computation"). DRAM 읽기 대역폭을 별도 항으로 두는 문장은 스니펫에서 발견 못 함.

## D1/D2/D3 판정
| 조건 | 판정 | 근거 |
|---|---|---|
| D1 expert 상주 수 + KV 배치 공동 결정 | **아니오** | MoE 아님. 결정 변수는 KV 토큰 선택·evict 뿐 |
| D2 마이크로벤치-only 사전 예측 + 오차 보고 | **아니오** | 예측 모델 없음, 전부 실측 |
| D3 binding 자원 전이점 명시 분석 | **아니오** | PCIe 단일 병목 가정. 전이점 논의 인용 불가 |
| 호스트 DRAM BW 모델링 (KV 계열 추가) | 아니오 (추측) | PCIe 만 언급 |

## 우리와의 delta
- 같은 점: KV 를 호스트에 두고 매 스텝 일부만 GPU 로 읽는 구조 (우리 HiCache 경로와 유사한 데이터 흐름).
- 다른 점: InfiniGen 은 "무엇을 읽을지" (선택·정확도 trade-off) 이고, 우리는 "호스트 읽기가 cold expert 스트리밍과 같은 DDR 예산을 쓴다" 는 경합 항과 그 결과의 사전 예측. 정확도 손실 없는 exact 경로라는 점도 다름. 위협도 **하**.
