# Klotski — arXiv 2502.06888 (v1, 2025-02-09)

## 서지 / 전문 접근
- Zhiyuan Fang, Yuegui Huang, Zicong Hong, Yufeng Lyu, Wuhui Chen, Yue Yu, Fan Yu, Zibin Zheng (SYSU / HKUST / Huawei / PCL). "Klotski: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline". 학회 미표기. FlexGen 위 3k LoC.
- 전문 접근: **예** — HF 미러 PDF(15쪽) 텍스트 추출.

## 결정 변수
- n(배치 그룹 내 배치 수) 와 배치 크기; 텐서 배치(VRAM/DRAM/disk, expert·gate·attention·KV cache·activation 유형별, 층 단위 가능); prefetch 할 hot expert 수 K(=top-k 기본); expert 연산 순서(hot 먼저, 이후 전송 완료 순); 양자화(HQQ 4bit)·sparse attention 옵션.
- offline 대량 배치 throughput 지향(zig-zag 스케줄). RTX 3090 + Xeon 5318Y / H800 + Xeon 8470, Mixtral-8×7B/8×22B.

## 비용 모델 파라미터 출처 / 예측 검증 / 보고 오차
- **마이크로벤치 파라미터로 계획을 세움**: "Klotski measures the computation times and transmission durations of the model's various layers based on their shapes, data types, and other relevant information in the current environment. These results are cached locally." → 부등식 (4)~(7): n·t_c_A ≥ t_IO_G; n(t_c_A+t_c_G) ≥ t_IO_G + K·t_IO_E; … + Σ_{i∈Q} t_c_Ei ≥ t_IO_G + (K+len(Q))·t_IO_E + t_IO_A. Q 길이는 "based on statistical data".
- 출력은 최소 n(버블 0 조건) 이지 처리량 예측이 아님: "n should be set to the smallest integer that satisfies the inequality group". 예측 오차 보고 없음. 평가에서는 계산된 n 대신 최대치를 사용: "We use the maximum n (= 15) from Figure 14 to show a better result than the default computed n".

## expert 와 KV 의 메모리·대역폭 경합 취급
- KV 는 FlexGen 식으로 offload·prefetch(스트림 4개 중 2개가 KV 적재/저장). n 상한이 KV 용량으로 결정: "If n is too large, it will introduce a significant KV cache." 메모리 제약 M_peak_GPU < M_GPU 는 모든 텐서 공통.
- 그러나 부등식의 I/O 항은 가중치(t_IO_G, t_IO_E, t_IO_A) 만이며 KV 읽기 대역폭은 항에 없음. expert 상주 수는 K=top-k 고정이라 "함께 결정" 이 아님.

## 캐시 크기 결정 방식
- **휴리스틱**: GPU 상주 hot expert 수 K = top-k("K is by default equal to k in top-k"), hot 선정은 expert 상관 테이블(wikitext-2 사전 실행, 온라인 갱신). 남는 VRAM 은 adaptive tensor placement 로 층 단위 배치("placing the experts of the first three layers in VRAM").

## 판정
- **D1: 아니오(부분 근접).** 메모리 용량 제약은 공통이나 대역폭 예산은 가중치 I/O 만 모델링하고 expert 상주 수는 고정. 인용: "If n is too large, it will introduce a significant KV cache" / 부등식 (7) 에 KV 항 없음.
- **D2: 부분.** 마이크로벤치만으로 스케줄(n) 을 계획하지만 처리량·스텝시간 사전 예측과 오차 보고가 없고 평가는 계산된 n 을 쓰지 않음. 인용: "Klotski measures the computation times and transmission durations ... These results are cached locally" / "n should be set to the smallest integer that satisfies the inequality group".
- **D3: 부분(경험적, n 축).** "when n is small, the throughput is low because the I/O time is much longer than the computation time ... When n reaches a sufficiently large value, the slope ... gradually approaches zero, indicating that most of the inter- and intra-layer bubbles have been eliminated." 축이 n(배치 그룹) 이며 H 축 전이는 없음.

## 우리와의 delta
- **K1 에서 가장 가까운 선행(D2 부분).** 우리 논문에서 반드시 인용·대조: Klotski 는 마이크로벤치 → 계획(n) 이고 우리는 마이크로벤치 → **스텝 시간·처리량 사전 예측(격자, 오차 보고)** → 분할 정책(H, KV 위치). 우리 모델에는 Klotski 에 없는 항이 셋: CPU expert 연산(compute offloading; Klotski 는 weight streaming 만), KV DRAM 읽기와 cold expert 스트리밍의 DDR 공유, H(GPU 상주 expert 수) 를 결정 변수로.
- Klotski 의 hot expert prefetch 정확도(58.89%) 와 "prefetched experts 100% participate" 구분은 우리 트레이스 항(distinct hot/cold 기대값) 검증 방식 참고.
