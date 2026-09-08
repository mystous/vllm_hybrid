# Fiddler (ICLR'25) — K1 정독 노트

- **서지**: Keisuke Kamahori, Tian Tang, Yile Gu, Kan Zhu, Baris Kasikci (UW/Tsinghua). "Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models." ICLR 2025. arXiv 2402.07033 (v3, 2025-05-01).
- **전문 접근 여부**: **전문 확보 (17쪽, 부록 A~F 포함)**.
- **전문 확보 경로**: HF 데이터셋 `Chelsea707/arxiv-cs-2020-2025-pdfs` `data/2024/2402_07xxx/2402.07033.pdf` (curl) → pypdf 추출. huggingface.co/papers 초록·GitHub README 로 교차 확인.

## 문제·결정 변수
- 대상: MoE (Mixtral-8x7B 16-bit, Phi-3.5-MoE) 를 단일 GPU 24/48GB 에서 **단일 요청·저지연** (batch 1, 긴 prefill, beam search).
- 정적 배치: 비-expert 층 전부 GPU + 인기 expert 를 용량만큼 GPU (오프라인 프로파일). 인용 (§3.4): "we place frequently used experts on the GPU based on offline profiling. For this, we select as many experts as the memory capacity permits in order of popularity to maximize the hit rate".
- 동적 결정 (Algorithm 1, expert·층 단위): GPU 상주면 GPU; 아니면 `cpu_lat(s) > gpu_lat(s) + trans_lat()` 이면 가중치 전송 후 GPU, 아니면 CPU 계산.
- KV 는 GPU (언급 없음 = 기본 GPU). 결정 변수에 KV 없음.

## 비용 모델 파라미터 출처와 검증
- 모델: GPU 시간 상수, CPU 시간 입력 수 s 에 선형. 인용 (§3.3): "gpu_lat(s) returns a constant value, while cpu_lat(s) returns a value proportional to s, multiplied by another constant. These constants are determined in the initialization phase."
- **파라미터 출처 = 초기화 시 마이크로벤치** (엔드투엔드 아님): "We also measure the latency to copy weights and execute experts on either the CPU or the GPU with different input sizes to inform the decision at runtime." 부록 A: W_copy, A_copy, GPU_N, CPU_N 을 32회 측정.
- 예측 검증: **없음** (실측 처리량 비교만). 오차 수치 인용 불가.

## expert·KV 경합 취급
- **없음**. KV 배치는 변수가 아니고, expert 상주 수는 "as many experts as the memory capacity permits" 로 KV 와 무관하게 채움. 대역폭은 PCIe (가중치 전송 vs activation 전송) 만: "the latency for transferring weights from CPU memory to GPU memory is about 2-5 times longer than the actual computation time."
- 호스트 DRAM 대역폭 항 없음 — CPU 는 "latency bounded by the computation part, not the memory movement part" 로 본다 (§3.1).

## D1/D2/D3 판정
| 조건 | 판정 | 근거 (원문) |
|---|---|---|
| D1 expert 상주 수 + KV 배치 공동 결정 | **아니오** | "we select as many experts as the memory capacity permits in order of popularity" — KV 항 없음. |
| D2 마이크로벤치-only 사전 예측 + 오차 보고 | **부분 (마이크로벤치 파라미터) / 아니오 (사전 예측·오차)** | 파라미터는 마이크로벤치 ("These constants are determined in the initialization phase") 이나 용도는 expert 단위 런타임 결정이고, 처리량 사전 예측·오차 보고 없음. |
| D3 binding 자원 전이점 | **부분 (입력 크기 축, 정성)** | "(c) is advantageous when the input size to an expert is small, while (b) is better if the input size is above some threshold, even with large communication overhead." 전이점 위치를 모델로 예측·보고하지는 않음 (Algorithm 1 의 비교식이 암묵적 전이점). |

## 우리와의 delta
- 같은 점: expert 당 CPU 시간 선형 모델 (우리의 expert당 t_s + 행당 t_r 과 같은 형태), 마이크로벤치 파라미터, 인기 expert 를 GPU 에 정적 배치.
- 다른 점: Fiddler 는 batch 1 지연이고 DDR 대역폭·KV·동시성 없음. 우리는 배치 서빙에서 H 와 KV 를 DDR 예산으로 함께 결정, 처리량 사전 예측. 위협도 **하~중** (마이크로벤치 파라미터 선행이나 D2 의 "처리량 사전 예측 + 오차" 는 없음).
