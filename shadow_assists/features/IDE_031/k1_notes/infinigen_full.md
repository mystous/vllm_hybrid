# InfiniGen — K1 전문 정독 노트

- **서지**: Wonbeom Lee, Jungi Lee, Junghwan Seo, Jaewoong Sim (SNU). "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management". arXiv 2406.19707v1 (2024-06-28), OSDI'24. 18쪽.
- **전문 확보 경로**: HF `Chelsea707/arxiv-cs-2020-2025-pdfs` → `data/2024/2406_19xxx/2406.19707.pdf` (3.0 MB) → pypdf. **전문 확보**.

## 무대·토폴로지
- 무대 = 오프로딩 기반 추론 (FlexGen·UVM 위) 의 **지연 (wall-clock)** 측정, dense 모델 (OPT 6.7/13/30B, Llama-2 7/13B). 원문: "we measure the wall clock time during inference with varying batch sizes and sequence lengths." MoE 항 **없음** (grep 결과 "mixture" 는 참고문헌 제목뿐).
- 토폴로지: KV 전량 CPU 메모리 풀, 가중치는 GPU 우선. "we explicitly locate all the KV cache in the CPU memory. The model parameters are stored in the GPU memory as much as possible, with the remainder in the CPU memory." 머신: RTX A6000 48 GB + Xeon Gold 6136 DDR4-2666 96 GB, PCIe 3.0.
- 핵심 기법: 층 i−1 에서 층 i 의 attention 패턴을 부분 가중치로 리허설 → 중요 토큰 KV 만 prefetch. "InfiniGen speculates on the attention pattern of the next layer (Layer i) using the attention input of Layer i−1, a partial query weight, and a partial key cache of Layer i."

## 결정 변수
- 층별로 GPU 로 가져올 KV 엔트리 수 (α 임계값, 평균 <10%, 상한 20%), CPU 풀 용량 초과 시 counter 기반 evict. "we allow sending up to 20% of the total KV cache to the GPU". expert 상주 결정 **없음**, GPU 메모리 분할 결정 없음.

## 비용 모델 파라미터 출처·검증·오차
- **성능 모델 없음**. "performance model / cost model / analytic" 문자열 없음. 지연은 실측 (Fig. 14–18). Fig. 18 분해: "the key performance bottleneck of FlexGen and H2O is the data transfer overhead, which occupies 96.9% and 91.8% of the execution time" — 사후 실측 분해이지 사전 예측 아님.

## expert·KV 경합 취급
- expert 없음. 대역폭 항은 PCIe 뿐: "due to the low PCIe bandwidth between the CPU and GPU". 호스트 DRAM 대역폭이 다른 스트림과 공유된다는 항 **없음**.

## D1·D2·D3 판정
| 조건 | 판정 | 근거 |
|---|---|---|
| D1 | **아니오** | MoE 아님, expert 상주 없음. 결정 대상은 KV 엔트리 선택뿐. 인용 불가 |
| D2 | **아니오** | 예측 모델 없음. 인용 불가 |
| D3 | **아니오** | binding 은 PCIe 전송으로 고정 ("the data transfer overhead ... 96.9%"), 전이점 분석 없음 |
| DDR 공유 항 | **아니오** | 인용 불가 |

## 우리와의 delta
- 우리 격자의 "KV 위치 (GPU vs host)" 축에서 host KV 읽기를 줄이는 알고리즘 기법 (근사, 정확도 손실 수반). 우리 기여는 정확도 무손실 조건 (CLAUDE.md 제약) 아래 분할 결정이므로 직교. 관련 연구에 "KV 오프로드 시 PCIe 가 지배" 근거로 인용 가능. 위협도 **하**.
