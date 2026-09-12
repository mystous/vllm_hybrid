# RBC-Attn v0.2 — 전체 실험 기록 (요약 + 원시 데이터)

지시서 `RBC_Attn_v0.2` 에 따른 수정과 측정의 전량 기록이다. 모든 수치는 `results/v02/**` 의 원본 JSON 에서 생성했고, 측정하지 않은 값은 계산해 채우지 않고 `NOT_RUN` / `—` 로 남겼다.


## 한눈에

- **전체 시간 판정: 개선 없음.** E4 의 18개 조합 전부 q_outer 가 가장 빠르다 (Q대비 중앙값 +11.7%).
- **GPU 시간만 보면 이기는 영역이 있다.** §13.1 게이트 46/81 영역 통과, 감소율 중앙값 −35.7%, 최대 −69.5%.
- **이득의 원인은 설계의 주장과 다르다.** 통과 영역 대부분에서 KV 방문 비가 1.00 이다. KV 재읽기 감소가 아니라 Q-outer union 직사각형의 padded dot 낭비가 사라진 것이다.
- **막는 것은 CPU 계획비다.** Nq512 에서 405~528us 로 §8.7 목표 50us 의 8~10배, GPU 이득 20~75us 보다 크다.
- **정확성은 전부 통과.** numeric 54/54, exact-edge 27/27, 테스트 154건.


## 측정 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (UUID `43a78ae4-f89f-ca72-2ff2-27e8a8a27f74`) |
| SM 수 | 132 |
| 사용 장치 수 / 논리 ordinal | 4 / 0 |
| torch / CUDA | 2.13.0+cu130 / 13.0 |
| triton | 3.7.1 |
| flashinfer | 0.6.17 (available=True) |
| 원본 생성 시각 | 2026-09-12T00:44:11 |

시간 측정은 전부 profiling 없는 CUDA event / graph replay 다 (§7.5). NCU 는 이번 작업에서 실행하지 않았다.


## 구현 변경 요약

| 항목 | 파일 | 내용 |
|---|---|---|
| P1 강제 분할 제거 | `src/planner.cpp` | `min_tasks` 기본 0. 과거 2×SM 분할은 legacy 후보로만 유지. 선택 후 재분할 없음 |
| P2 direct output | `src/planner.cpp`, `rbc/kernels.py` | degree 기반 출력 소유권. degree==1 행은 main 이 정규화 결과를 직접 기록하고 partial/merge 를 건너뛴다. reduction CSR 은 multi-owner 행만 |
| P3 K microtile | `rbc/kernels.py` | `BN` 서브루프, `LATE_V` variant, `FULL` membership fast path |
| P4 교정 선택기 | `rbc/select.py`, `src/planner.cpp` | 계획 없이 후보 특징만 뽑는 `rbc_probe` C ABI + (BM,G) 구간별 NNLS 회귀 + 보수적 선택 규칙 |
| P5 3단계 runtime | `rbc/slots.py` | `prepare_capacity`/`plan_into`/`submit`. 2 slot, event 의존성, C++ 이 pinned slab 에 직접 쓰기, FULL 이면 H2D 1회, capacity 초과는 명시적 실패 |
| P6 whole-query | `stream_bench.py` | 기본 경로는 whole-query. chunk pipeline 은 비교 옵션 (§13.1 미충족으로 확대 안 함) |
| §8.6 플래너 최적화 | `src/planner.cpp` | worker별 스크래치 재사용, Task 풀(capacity 유지), 그룹별 CSR 직접 접근, block→위치 이진탐색 제거, reduction CSR 을 count→prefix sum→fill 로 |


## E0 — 기준선 복구 (§11 E0)

입력: Nq512 / Nk65536 / G8 / D128 / BK128 / sel16, seed 42. 반복 9 × graph replay 100.
clock 상태: `{'clocks.sm': '345 MHz', 'clocks.max.sm': '1980 MHz', 'clocks.mem': '2619 MHz', 'power.draw': '72.01 W', 'power.limit': '700.00 W', 'temperature.gpu': '38', 'persistence_mode': 'Disabled', 'compute_mode': 'Default'}`

### 패턴별 최강

| 패턴 | 정책 | max_m | split | GPU 시간 | tasks | partial |
|---|---|---|---|---:|---:|---:|
| common_private | rbc | M16 | none | 75.4us | 256 | 0 |
| random | q_outer | M16 | none | 102.0us | 256 | 0 |
| clustered | q_outer | M16 | none | 104.7us | 256 | 0 |

### 전량 (정확성 통과 셀)

| 패턴 | 정책 | max_m | split | GPU us | tasks | partial | BM | rel_err | plan ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clustered | kv_outer | 16 | legacy_2SM | 188.2 | 8052 | 8192 | 2 | 2.70e-03 | 1.85 |
| clustered | kv_outer | 16 | none | 188.7 | 8052 | 8192 | 2 | 2.70e-03 | 1.85 |
| clustered | kv_outer | 32 | legacy_2SM | 229.9 | 7796 | 8192 | 4 | 2.70e-03 | 1.72 |
| clustered | kv_outer | 32 | none | 229.8 | 7796 | 8192 | 4 | 2.70e-03 | 1.74 |
| clustered | kv_outer | 64 | legacy_2SM | 218.4 | 7322 | 8192 | 4 | 2.70e-03 | 1.64 |
| clustered | kv_outer | 64 | none | 218.4 | 7322 | 8192 | 4 | 2.70e-03 | 1.66 |
| clustered | kv_outer | 128 | legacy_2SM | 394.3 | 6587 | 8192 | 8 | 2.70e-03 | 1.59 |
| clustered | kv_outer | 128 | none | 394.1 | 6587 | 8192 | 8 | 2.70e-03 | 1.63 |
| clustered | q_outer | 16 | legacy_2SM | 107.8 | 264 | 32 | 2 | 2.57e-03 | 0.54 |
| clustered | q_outer | 16 | none | 104.7 | 256 | 0 | 2 | 2.57e-03 | 0.68 |
| clustered | q_outer | 32 | legacy_2SM | 144.9 | 264 | 1056 | 4 | 2.55e-03 | 0.73 |
| clustered | q_outer | 32 | none | 199.0 | 128 | 0 | 4 | 2.57e-03 | 0.52 |
| clustered | q_outer | 64 | legacy_2SM | 301.3 | 264 | 2112 | 8 | 2.77e-03 | 0.80 |
| clustered | q_outer | 64 | none | 564.7 | 64 | 0 | 8 | 2.57e-03 | 0.50 |
| clustered | q_outer | 128 | legacy_2SM | 793.4 | 264 | 4224 | 16 | 2.38e-03 | 0.92 |
| clustered | q_outer | 128 | none | 3075.4 | 32 | 0 | 16 | 2.57e-03 | 0.47 |
| clustered | rbc | 16 | legacy_2SM | 136.0 | 495 | 0 | 2 | 2.57e-03 | 0.76 |
| clustered | rbc | 16 | none | 133.1 | 495 | 0 | 2 | 2.57e-03 | 0.59 |
| clustered | rbc | 32 | legacy_2SM | 192.1 | 469 | 22 | 4 | 2.57e-03 | 0.56 |
| clustered | rbc | 32 | none | 192.1 | 469 | 22 | 4 | 2.57e-03 | 0.61 |
| clustered | rbc | 64 | legacy_2SM | 151.2 | 560 | 314 | 4 | 2.55e-03 | 6.74 |
| clustered | rbc | 64 | none | 151.2 | 560 | 314 | 4 | 2.55e-03 | 0.66 |
| clustered | rbc | 128 | legacy_2SM | 286.4 | 654 | 674 | 8 | 2.55e-03 | 0.70 |
| clustered | rbc | 128 | none | 287.9 | 654 | 674 | 8 | 2.55e-03 | 0.73 |
| clustered | signature_only | 16 | legacy_2SM | 109.4 | 529 | 68 | 2 | 2.57e-03 | 0.58 |
| clustered | signature_only | 16 | none | 109.5 | 529 | 68 | 2 | 2.57e-03 | 0.55 |
| clustered | signature_only | 32 | legacy_2SM | 152.0 | 563 | 204 | 4 | 2.55e-03 | 0.58 |
| clustered | signature_only | 32 | none | 151.9 | 563 | 204 | 4 | 2.55e-03 | 0.59 |
| clustered | signature_only | 64 | legacy_2SM | 151.7 | 609 | 404 | 4 | 2.55e-03 | 0.64 |
| clustered | signature_only | 64 | none | 151.4 | 609 | 404 | 4 | 2.55e-03 | 0.69 |
| clustered | signature_only | 128 | legacy_2SM | 265.3 | 681 | 719 | 8 | 2.55e-03 | 0.68 |
| clustered | signature_only | 128 | none | 269.2 | 681 | 719 | 8 | 2.55e-03 | 0.66 |
| common_private | kv_outer | 16 | legacy_2SM | 154.3 | 6132 | 8180 | 2 | 2.65e-03 | 1.52 |
| common_private | kv_outer | 16 | none | 154.6 | 6132 | 8180 | 2 | 2.65e-03 | 1.57 |
| common_private | kv_outer | 32 | legacy_2SM | 166.8 | 5108 | 8180 | 4 | 2.65e-03 | 1.33 |
| common_private | kv_outer | 32 | none | 166.8 | 5108 | 8180 | 4 | 2.65e-03 | 1.35 |
| common_private | kv_outer | 64 | legacy_2SM | 290.0 | 4596 | 8180 | 8 | 2.65e-03 | 1.34 |
| common_private | kv_outer | 64 | none | 290.4 | 4596 | 8180 | 8 | 2.65e-03 | 1.47 |
| common_private | kv_outer | 128 | legacy_2SM | 686.9 | 4340 | 8180 | 16 | 2.65e-03 | 1.34 |
| common_private | kv_outer | 128 | none | 686.8 | 4340 | 8180 | 16 | 2.65e-03 | 1.34 |
| common_private | q_outer | 16 | legacy_2SM | 79.4 | 264 | 32 | 2 | 2.65e-03 | 0.52 |
| common_private | q_outer | 16 | none | 76.6 | 256 | 0 | 2 | 2.65e-03 | 12.28 |
| common_private | q_outer | 32 | legacy_2SM | 94.2 | 264 | 1056 | 4 | 2.73e-03 | 0.56 |
| common_private | q_outer | 32 | none | 128.8 | 128 | 0 | 4 | 2.65e-03 | 0.41 |
| common_private | q_outer | 64 | legacy_2SM | 181.1 | 264 | 2112 | 8 | 2.62e-03 | 0.67 |
| common_private | q_outer | 64 | none | 324.6 | 64 | 0 | 8 | 2.65e-03 | 0.41 |
| common_private | q_outer | 128 | legacy_2SM | 499.3 | 264 | 4224 | 16 | 2.71e-03 | 0.83 |
| common_private | q_outer | 128 | none | 1779.1 | 32 | 0 | 16 | 2.65e-03 | 0.40 |
| common_private | rbc | 16 | legacy_2SM | 78.6 | 264 | 32 | 2 | 2.65e-03 | 0.50 |
| common_private | rbc | 16 | none | 75.4 | 256 | 0 | 2 | 2.65e-03 | 0.60 |
| common_private | rbc | 32 | legacy_2SM | 115.8 | 640 | 1024 | 4 | 2.67e-03 | 0.58 |
| common_private | rbc | 32 | none | 114.0 | 640 | 1024 | 4 | 2.67e-03 | 0.61 |
| common_private | rbc | 64 | legacy_2SM | 191.6 | 576 | 1024 | 8 | 2.67e-03 | 0.59 |
| common_private | rbc | 64 | none | 193.8 | 576 | 1024 | 8 | 2.67e-03 | 0.61 |
| common_private | rbc | 128 | legacy_2SM | 524.6 | 544 | 1024 | 16 | 2.67e-03 | 0.73 |
| common_private | rbc | 128 | none | 524.6 | 544 | 1024 | 16 | 2.67e-03 | 0.56 |
| common_private | signature_only | 16 | legacy_2SM | 90.5 | 768 | 1024 | 2 | 2.67e-03 | 0.62 |
| common_private | signature_only | 16 | none | 90.5 | 768 | 1024 | 2 | 2.67e-03 | 0.63 |
| common_private | signature_only | 32 | legacy_2SM | 116.5 | 640 | 1024 | 4 | 2.67e-03 | 0.55 |
| common_private | signature_only | 32 | none | 116.1 | 640 | 1024 | 4 | 2.67e-03 | 0.64 |
| common_private | signature_only | 64 | legacy_2SM | 191.8 | 576 | 1024 | 8 | 2.67e-03 | 0.55 |
| common_private | signature_only | 64 | none | 198.3 | 576 | 1024 | 8 | 2.67e-03 | 0.57 |
| common_private | signature_only | 128 | legacy_2SM | 524.1 | 544 | 1024 | 16 | 2.67e-03 | 0.69 |
| common_private | signature_only | 128 | none | 524.1 | 544 | 1024 | 16 | 2.67e-03 | 0.59 |
| random | kv_outer | 16 | legacy_2SM | 185.9 | 8036 | 8168 | 2 | 2.27e-03 | 1.98 |
| random | kv_outer | 16 | none | 186.7 | 8036 | 8168 | 2 | 2.27e-03 | 1.96 |
| random | kv_outer | 32 | legacy_2SM | 228.5 | 7798 | 8168 | 4 | 2.27e-03 | 1.84 |
| random | kv_outer | 32 | none | 229.0 | 7798 | 8168 | 4 | 2.27e-03 | 1.86 |
| random | kv_outer | 64 | legacy_2SM | 218.1 | 7354 | 8168 | 4 | 2.27e-03 | 1.74 |
| random | kv_outer | 64 | none | 218.5 | 7354 | 8168 | 4 | 2.27e-03 | 1.74 |
| random | kv_outer | 128 | legacy_2SM | 387.5 | 6541 | 8168 | 8 | 2.27e-03 | 1.72 |
| random | kv_outer | 128 | none | 387.4 | 6541 | 8168 | 8 | 2.27e-03 | 1.89 |
| random | q_outer | 16 | legacy_2SM | 102.2 | 264 | 32 | 2 | 2.48e-03 | 0.53 |
| random | q_outer | 16 | none | 102.0 | 256 | 0 | 2 | 2.48e-03 | 0.69 |
| random | q_outer | 32 | legacy_2SM | 137.5 | 264 | 1056 | 4 | 2.31e-03 | 0.73 |
| random | q_outer | 32 | none | 196.3 | 128 | 0 | 4 | 2.48e-03 | 0.51 |
| random | q_outer | 64 | legacy_2SM | 278.3 | 264 | 2112 | 8 | 2.38e-03 | 0.85 |
| random | q_outer | 64 | none | 527.0 | 64 | 0 | 8 | 2.48e-03 | 0.51 |
| random | q_outer | 128 | legacy_2SM | 731.6 | 264 | 4224 | 16 | 2.47e-03 | 1.21 |
| random | q_outer | 128 | none | 2729.0 | 32 | 0 | 16 | 2.48e-03 | 0.51 |
| random | rbc | 16 | legacy_2SM | 142.9 | 408 | 0 | 2 | 2.48e-03 | 0.66 |
| random | rbc | 16 | none | 142.5 | 408 | 0 | 2 | 2.48e-03 | 0.68 |
| random | rbc | 32 | legacy_2SM | 199.9 | 526 | 534 | 4 | 2.48e-03 | 1.00 |
| random | rbc | 32 | none | 200.1 | 526 | 534 | 4 | 2.48e-03 | 0.79 |
| random | rbc | 64 | legacy_2SM | 173.0 | 1099 | 1778 | 4 | 2.35e-03 | 0.97 |
| random | rbc | 64 | none | 172.9 | 1099 | 1778 | 4 | 2.35e-03 | 1.01 |
| random | rbc | 128 | legacy_2SM | 289.7 | 1780 | 3265 | 8 | 2.44e-03 | 1.10 |
| random | rbc | 128 | none | 294.9 | 1780 | 3265 | 8 | 2.44e-03 | 1.14 |
| random | signature_only | 16 | legacy_2SM | 110.4 | 616 | 416 | 2 | 2.48e-03 | 0.68 |
| random | signature_only | 16 | none | 110.8 | 616 | 416 | 2 | 2.48e-03 | 0.69 |
| random | signature_only | 32 | legacy_2SM | 157.7 | 807 | 997 | 4 | 2.48e-03 | 0.77 |
| random | signature_only | 32 | none | 153.6 | 807 | 997 | 4 | 2.48e-03 | 0.81 |
| random | signature_only | 64 | legacy_2SM | 165.6 | 1163 | 1853 | 4 | 2.35e-03 | 0.92 |
| random | signature_only | 64 | none | 165.5 | 1163 | 1853 | 4 | 2.35e-03 | 0.94 |
| random | signature_only | 128 | legacy_2SM | 289.7 | 1780 | 3265 | 8 | 2.44e-03 | 1.09 |
| random | signature_only | 128 | none | 294.3 | 1780 | 3265 | 8 | 2.44e-03 | 1.12 |


## E1 — 최종 계획 선택과 분할 (§11 E1)

실행기: D (P1+P2+P3: direct output, FULL fast path, BN=BK). 입력 Nq512/Nk65536/G8/D128/sel16, seed 42.

**핵심 관찰**: KV 방문량(`KV_visits`)은 max_m 을 키우면 줄지만 GPU 시간은 급격히 늘어난다. 이득원(KV 재읽기 감소)과 비용(병렬성·padding)이 반대 방향이다.

| 패턴 | 정책 | split | max_m | BM | ntasks | KV_visits | padded | partial | main us | merge us | GPU us |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clustered | kv_outer | legacy_2SM | 16 | 2 | 8052 | 8052 | 128832 | 8192 | 169.2 | 12.9 | 188.3 |
| clustered | kv_outer | legacy_2SM | 32 | 4 | 7796 | 7796 | 124944 | 8192 | 210.9 | 12.6 | 230.1 |
| clustered | kv_outer | legacy_2SM | 64 | 4 | 7322 | 7322 | 118096 | 8192 | 200.3 | 12.6 | 220.1 |
| clustered | kv_outer | legacy_2SM | 128 | 8 | 6587 | 6587 | 109440 | 8192 | 372.9 | 12.7 | 395.8 |
| clustered | kv_outer | none | 16 | 2 | 8052 | 8052 | 128832 | 8192 | 168.7 | 12.8 | 189.1 |
| clustered | kv_outer | none | 32 | 4 | 7796 | 7796 | 124944 | 8192 | 209.8 | 13.0 | 232.2 |
| clustered | kv_outer | none | 64 | 4 | 7322 | 7322 | 118096 | 8192 | 197.9 | 12.4 | 219.5 |
| clustered | kv_outer | none | 128 | 8 | 6587 | 6587 | 109440 | 8192 | 373.9 | 12.7 | 394.7 |
| clustered | q_outer | legacy_2SM | 16 | 2 | 264 | 8052 | 128832 | 32 | 102.1 | 4.4 | 105.3 |
| clustered | q_outer | legacy_2SM | 32 | 4 | 264 | 7796 | 249472 | 1056 | 137.5 | 5.7 | 142.7 |
| clustered | q_outer | legacy_2SM | 64 | 8 | 264 | 7322 | 468608 | 2112 | 292.7 | 6.5 | 300.9 |
| clustered | q_outer | legacy_2SM | 128 | 16 | 264 | 6587 | 843136 | 4224 | 780.8 | 9.8 | 797.4 |
| clustered | q_outer | none | 16 | 2 | 256 | 8052 | 128832 | 0 | 105.0 | 0.0 | 105.2 |
| clustered | q_outer | none | 32 | 4 | 128 | 7796 | 249472 | 0 | 202.5 | 0.0 | 202.9 |
| clustered | q_outer | none | 64 | 8 | 64 | 7322 | 468608 | 0 | 562.5 | 0.0 | 562.4 |
| clustered | q_outer | none | 128 | 16 | 32 | 6587 | 843136 | 0 | 3070.9 | 0.0 | 3070.1 |
| clustered | rbc | legacy_2SM | 16 | 2 | 495 | 8052 | 128832 | 0 | 131.9 | 0.0 | 132.2 |
| clustered | rbc | legacy_2SM | 32 | 4 | 469 | 7796 | 124944 | 22 | 188.3 | 4.5 | 192.0 |
| clustered | rbc | legacy_2SM | 64 | 4 | 560 | 7322 | 118096 | 314 | 147.0 | 4.3 | 152.1 |
| clustered | rbc | legacy_2SM | 128 | 8 | 654 | 6587 | 109440 | 674 | 281.6 | 4.7 | 292.5 |
| clustered | rbc | none | 16 | 2 | 495 | 8052 | 128832 | 0 | 136.8 | 0.0 | 131.7 |
| clustered | rbc | none | 32 | 4 | 469 | 7796 | 124944 | 22 | 188.5 | 4.4 | 192.3 |
| clustered | rbc | none | 64 | 4 | 560 | 7322 | 118096 | 314 | 147.8 | 4.3 | 152.2 |
| clustered | rbc | none | 128 | 8 | 654 | 6587 | 109440 | 674 | 284.4 | 4.6 | 297.2 |
| clustered | signature_only | legacy_2SM | 16 | 2 | 529 | 8052 | 128832 | 68 | 107.5 | 4.4 | 111.2 |
| clustered | signature_only | legacy_2SM | 32 | 4 | 563 | 7796 | 124944 | 204 | 144.9 | 4.4 | 149.1 |
| clustered | signature_only | legacy_2SM | 64 | 4 | 609 | 7322 | 118096 | 404 | 147.4 | 4.3 | 152.6 |
| clustered | signature_only | legacy_2SM | 128 | 8 | 681 | 6587 | 109440 | 719 | 264.3 | 4.7 | 274.6 |
| clustered | signature_only | none | 16 | 2 | 529 | 8052 | 128832 | 68 | 106.0 | 4.3 | 109.6 |
| clustered | signature_only | none | 32 | 4 | 563 | 7796 | 124944 | 204 | 148.4 | 4.4 | 147.4 |
| clustered | signature_only | none | 64 | 4 | 609 | 7322 | 118096 | 404 | 148.0 | 4.5 | 152.8 |
| clustered | signature_only | none | 128 | 8 | 681 | 6587 | 109440 | 719 | 268.9 | 4.9 | 275.1 |
| common_private | kv_outer | legacy_2SM | 16 | 2 | 6132 | 6132 | 98112 | 8180 | 134.1 | 13.4 | 154.2 |
| common_private | kv_outer | legacy_2SM | 32 | 4 | 5108 | 5108 | 98112 | 8180 | 146.6 | 13.3 | 167.9 |
| common_private | kv_outer | legacy_2SM | 64 | 8 | 4596 | 4596 | 98112 | 8180 | 269.6 | 13.4 | 291.4 |
| common_private | kv_outer | legacy_2SM | 128 | 16 | 4340 | 4340 | 98112 | 8180 | 663.7 | 12.9 | 686.1 |
| common_private | kv_outer | none | 16 | 2 | 6132 | 6132 | 98112 | 8180 | 133.6 | 13.3 | 154.0 |
| common_private | kv_outer | none | 32 | 4 | 5108 | 5108 | 98112 | 8180 | 146.9 | 13.3 | 168.2 |
| common_private | kv_outer | none | 64 | 8 | 4596 | 4596 | 98112 | 8180 | 269.6 | 13.0 | 290.2 |
| common_private | kv_outer | none | 128 | 16 | 4340 | 4340 | 98112 | 8180 | 664.1 | 13.1 | 694.2 |
| common_private | q_outer | legacy_2SM | 16 | 2 | 264 | 6132 | 98112 | 32 | 75.5 | 4.4 | 78.7 |
| common_private | q_outer | legacy_2SM | 32 | 4 | 264 | 5108 | 163456 | 1056 | 88.9 | 5.7 | 93.8 |
| common_private | q_outer | legacy_2SM | 64 | 8 | 264 | 4596 | 294144 | 2112 | 173.1 | 6.3 | 181.2 |
| common_private | q_outer | legacy_2SM | 128 | 16 | 264 | 4340 | 555520 | 4224 | 483.3 | 9.4 | 498.7 |
| common_private | q_outer | none | 16 | 2 | 256 | 6132 | 98112 | 0 | 77.0 | 0.0 | 76.9 |
| common_private | q_outer | none | 32 | 4 | 128 | 5108 | 163456 | 0 | 131.9 | 0.0 | 132.0 |
| common_private | q_outer | none | 64 | 8 | 64 | 4596 | 294144 | 0 | 324.3 | 0.0 | 327.0 |
| common_private | q_outer | none | 128 | 16 | 32 | 4340 | 555520 | 0 | 1751.9 | 0.0 | 1751.7 |
| common_private | rbc | legacy_2SM | 16 | 2 | 264 | 6132 | 98112 | 32 | 75.5 | 4.4 | 78.4 |
| common_private | rbc | legacy_2SM | 32 | 4 | 640 | 5108 | 98112 | 1024 | 107.2 | 5.8 | 113.2 |
| common_private | rbc | legacy_2SM | 64 | 8 | 576 | 4596 | 98112 | 1024 | 192.5 | 5.9 | 197.1 |
| common_private | rbc | legacy_2SM | 128 | 16 | 544 | 4340 | 98112 | 1024 | 517.5 | 5.8 | 526.2 |
| common_private | rbc | none | 16 | 2 | 256 | 6132 | 98112 | 0 | 75.1 | 0.0 | 75.2 |
| common_private | rbc | none | 32 | 4 | 640 | 5108 | 98112 | 1024 | 106.8 | 5.7 | 112.8 |
| common_private | rbc | none | 64 | 8 | 576 | 4596 | 98112 | 1024 | 191.2 | 5.7 | 191.3 |
| common_private | rbc | none | 128 | 16 | 544 | 4340 | 98112 | 1024 | 517.4 | 5.6 | 530.3 |
| common_private | signature_only | legacy_2SM | 16 | 2 | 768 | 6132 | 98112 | 1024 | 84.5 | 5.7 | 90.4 |
| common_private | signature_only | legacy_2SM | 32 | 4 | 640 | 5108 | 98112 | 1024 | 107.0 | 5.7 | 113.0 |
| common_private | signature_only | legacy_2SM | 64 | 8 | 576 | 4596 | 98112 | 1024 | 185.0 | 5.7 | 191.4 |
| common_private | signature_only | legacy_2SM | 128 | 16 | 544 | 4340 | 98112 | 1024 | 516.9 | 5.8 | 539.6 |
| common_private | signature_only | none | 16 | 2 | 768 | 6132 | 98112 | 1024 | 84.7 | 5.6 | 90.6 |
| common_private | signature_only | none | 32 | 4 | 640 | 5108 | 98112 | 1024 | 106.8 | 5.7 | 112.9 |
| common_private | signature_only | none | 64 | 8 | 576 | 4596 | 98112 | 1024 | 191.1 | 5.7 | 191.4 |
| common_private | signature_only | none | 128 | 16 | 544 | 4340 | 98112 | 1024 | 516.9 | 5.7 | 533.1 |
| random | kv_outer | legacy_2SM | 16 | 2 | 8036 | 8036 | 128576 | 8168 | 166.8 | 12.7 | 186.3 |
| random | kv_outer | legacy_2SM | 32 | 4 | 7798 | 7798 | 124896 | 8168 | 209.0 | 12.7 | 229.3 |
| random | kv_outer | legacy_2SM | 64 | 4 | 7354 | 7354 | 118448 | 8168 | 198.6 | 12.8 | 218.2 |
| random | kv_outer | legacy_2SM | 128 | 8 | 6541 | 6541 | 107904 | 8168 | 365.8 | 12.6 | 386.2 |
| random | kv_outer | none | 16 | 2 | 8036 | 8036 | 128576 | 8168 | 165.9 | 12.7 | 186.4 |
| random | kv_outer | none | 32 | 4 | 7798 | 7798 | 124896 | 8168 | 209.0 | 12.8 | 229.7 |
| random | kv_outer | none | 64 | 4 | 7354 | 7354 | 118448 | 8168 | 197.8 | 12.7 | 218.3 |
| random | kv_outer | none | 128 | 8 | 6541 | 6541 | 107904 | 8168 | 365.8 | 12.3 | 386.3 |
| random | q_outer | legacy_2SM | 16 | 2 | 264 | 8036 | 128576 | 32 | 98.6 | 4.4 | 101.9 |
| random | q_outer | legacy_2SM | 32 | 4 | 264 | 7798 | 249536 | 1056 | 132.1 | 5.8 | 137.2 |
| random | q_outer | legacy_2SM | 64 | 8 | 264 | 7354 | 470656 | 2112 | 269.9 | 6.4 | 277.9 |
| random | q_outer | legacy_2SM | 128 | 16 | 264 | 6541 | 837248 | 4224 | 721.3 | 9.1 | 734.1 |
| random | q_outer | none | 16 | 2 | 256 | 8036 | 128576 | 0 | 102.5 | 0.0 | 102.5 |
| random | q_outer | none | 32 | 4 | 128 | 7798 | 249536 | 0 | 199.1 | 0.0 | 202.1 |
| random | q_outer | none | 64 | 8 | 64 | 7354 | 470656 | 0 | 524.7 | 0.0 | 524.9 |
| random | q_outer | none | 128 | 16 | 32 | 6541 | 837248 | 0 | 2663.5 | 0.0 | 2663.8 |
| random | rbc | legacy_2SM | 16 | 2 | 408 | 8036 | 128576 | 0 | 138.7 | 0.0 | 138.5 |
| random | rbc | legacy_2SM | 32 | 4 | 526 | 7798 | 124896 | 534 | 196.5 | 4.6 | 200.5 |
| random | rbc | legacy_2SM | 64 | 4 | 1099 | 7354 | 118448 | 1778 | 166.9 | 6.5 | 174.5 |
| random | rbc | legacy_2SM | 128 | 8 | 1780 | 6541 | 107904 | 3265 | 281.9 | 8.2 | 299.6 |
| random | rbc | none | 16 | 2 | 408 | 8036 | 128576 | 0 | 138.9 | 0.0 | 138.8 |
| random | rbc | none | 32 | 4 | 526 | 7798 | 124896 | 534 | 196.3 | 4.5 | 201.5 |
| random | rbc | none | 64 | 4 | 1099 | 7354 | 118448 | 1778 | 167.0 | 6.3 | 173.8 |
| random | rbc | none | 128 | 8 | 1780 | 6541 | 107904 | 3265 | 281.8 | 8.1 | 299.9 |
| random | signature_only | legacy_2SM | 16 | 2 | 616 | 8036 | 128576 | 416 | 107.1 | 4.4 | 111.3 |
| random | signature_only | legacy_2SM | 32 | 4 | 807 | 7798 | 124896 | 997 | 149.2 | 5.2 | 153.9 |
| random | signature_only | legacy_2SM | 64 | 4 | 1163 | 7354 | 118448 | 1853 | 158.9 | 6.7 | 166.3 |
| random | signature_only | legacy_2SM | 128 | 8 | 1780 | 6541 | 107904 | 3265 | 281.7 | 8.2 | 300.0 |
| random | signature_only | none | 16 | 2 | 616 | 8036 | 128576 | 416 | 106.1 | 4.3 | 110.5 |
| random | signature_only | none | 32 | 4 | 807 | 7798 | 124896 | 997 | 152.3 | 5.4 | 153.3 |
| random | signature_only | none | 64 | 4 | 1163 | 7354 | 118448 | 1853 | 159.2 | 6.6 | 166.6 |
| random | signature_only | none | 128 | 8 | 1780 | 6541 | 107904 | 3265 | 288.7 | 8.3 | 300.2 |


## E2 — 변경별 효과 분리 (§11 E2)

입력 Nq512/Nk65536/G8/D128/sel16, seed 42. 반복 3 × replay 50. 셀 648개, 실패 0건.

행 A=v0.1 보존 / B=P1 / C=+direct output / D_P3a=+BN64 / D_P3b=+late V / D_P3c=+FULL fast path. 각 행에서 세 정책을 모두 실행했다.

### common_private

| 행 | q_outer us | M | signature us | M | rbc us | M |
|---|---:|---:|---:|---:|---:|---:|
| A_v01 | 80.5 | M16 | 90.4 | M16 | 80.6 | M16 |
| B_P1 | 80.3 | M16 | 90.4 | M16 | 82.0 | M16 |
| C_P2 | 75.9 | M16 | 90.6 | M16 | 75.5 | M16 |
| D_P3a | 98.7 | M16 | 100.4 | M16 | 98.5 | M16 |
| D_P3b | 75.3 | M16 | 90.5 | M16 | 75.6 | M16 |
| D_P3c | 75.5 | M16 | 90.5 | M16 | 75.6 | M16 |

### random

| 행 | q_outer us | M | signature us | M | rbc us | M |
|---|---:|---:|---:|---:|---:|---:|
| A_v01 | 106.4 | M16 | 115.5 | M16 | 144.7 | M16 |
| B_P1 | 108.3 | M16 | 112.5 | M16 | 149.5 | M16 |
| C_P2 | 99.9 | M16 | 112.1 | M16 | 143.8 | M16 |
| D_P3a | 135.1 | M16 | 111.4 | M16 | 157.8 | M16 |
| D_P3b | 103.0 | M16 | 110.7 | M16 | 140.1 | M16 |
| D_P3c | 100.0 | M16 | 112.8 | M16 | 139.8 | M16 |

### clustered

| 행 | q_outer us | M | signature us | M | rbc us | M |
|---|---:|---:|---:|---:|---:|---:|
| A_v01 | 107.6 | M16 | 114.7 | M16 | 137.1 | M16 |
| B_P1 | 110.7 | M16 | 111.4 | M16 | 141.3 | M16 |
| C_P2 | 102.3 | M16 | 113.0 | M16 | 137.4 | M16 |
| D_P3a | 138.9 | M16 | 107.9 | M16 | 148.1 | M16 |
| D_P3b | 104.8 | M16 | 109.7 | M16 | 132.1 | M16 |
| D_P3c | 102.8 | M16 | 109.7 | M16 | 132.7 | M16 |

### A 대비 상대 변화 (각 정책의 최적 max_m 기준, 음수=개선)

| 정책 | 패턴 | A 기준 | B_P1 | C_P2 | D_P3a | D_P3b | D_P3c |
|---|---|---:|---:|---:|---:|---:|---:|
| q_outer | common_private | 80.5us | -0.2% | -5.7% | +22.7% | -6.3% | -6.1% |
| q_outer | random | 106.4us | +1.8% | -6.1% | +27.0% | -3.2% | -6.0% |
| q_outer | clustered | 107.6us | +2.9% | -5.0% | +29.0% | -2.6% | -4.5% |
| signature_only | common_private | 90.4us | -0.0% | +0.2% | +11.1% | +0.1% | +0.1% |
| signature_only | random | 115.5us | -2.7% | -3.0% | -3.6% | -4.2% | -2.4% |
| signature_only | clustered | 114.7us | -2.9% | -1.5% | -6.0% | -4.4% | -4.4% |
| rbc | common_private | 80.6us | +1.7% | -6.3% | +22.3% | -6.2% | -6.2% |
| rbc | random | 144.7us | +3.4% | -0.6% | +9.1% | -3.2% | -3.3% |
| rbc | clustered | 137.1us | +3.0% | +0.2% | +8.0% | -3.7% | -3.2% |

### 전량 원시 셀

(반복별 값. `rep` 는 무작위 순서로 실행한 반복 번호다.)

| 패턴 | 행 | 정책 | max_m | rep | GPU us | tasks | partial | multi | FULL | BN | lateV | rel_err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clustered | A_v01 | q_outer | 16 | 0 | 107.64 | 264 | 528 | 512 | N | 128 | N | 2.57e-03 |
| clustered | A_v01 | q_outer | 16 | 1 | 107.53 | 264 | 528 | 512 | N | 128 | N | 2.57e-03 |
| clustered | A_v01 | q_outer | 16 | 2 | 111.42 | 264 | 528 | 512 | N | 128 | N | 2.57e-03 |
| clustered | A_v01 | q_outer | 32 | 0 | 148.17 | 264 | 1056 | 512 | N | 128 | N | 2.55e-03 |
| clustered | A_v01 | q_outer | 32 | 1 | 148.38 | 264 | 1056 | 512 | N | 128 | N | 2.55e-03 |
| clustered | A_v01 | q_outer | 32 | 2 | 142.79 | 264 | 1056 | 512 | N | 128 | N | 2.55e-03 |
| clustered | A_v01 | q_outer | 64 | 0 | 312.25 | 264 | 2112 | 512 | N | 128 | N | 2.77e-03 |
| clustered | A_v01 | q_outer | 64 | 1 | 311.36 | 264 | 2112 | 512 | N | 128 | N | 2.77e-03 |
| clustered | A_v01 | q_outer | 64 | 2 | 312.35 | 264 | 2112 | 512 | N | 128 | N | 2.77e-03 |
| clustered | A_v01 | q_outer | 128 | 0 | 811.05 | 264 | 4224 | 512 | N | 128 | N | 2.38e-03 |
| clustered | A_v01 | q_outer | 128 | 1 | 811.44 | 264 | 4224 | 512 | N | 128 | N | 2.38e-03 |
| clustered | A_v01 | q_outer | 128 | 2 | 799.79 | 264 | 4224 | 512 | N | 128 | N | 2.38e-03 |
| clustered | A_v01 | rbc | 16 | 0 | 137.12 | 495 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | A_v01 | rbc | 16 | 1 | 137.05 | 495 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | A_v01 | rbc | 16 | 2 | 141.99 | 495 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | A_v01 | rbc | 32 | 0 | 187.23 | 469 | 524 | 512 | N | 128 | N | 2.57e-03 |
| clustered | A_v01 | rbc | 32 | 1 | 192.35 | 469 | 524 | 512 | N | 128 | N | 2.57e-03 |
| clustered | A_v01 | rbc | 32 | 2 | 193.32 | 469 | 524 | 512 | N | 128 | N | 2.57e-03 |
| clustered | A_v01 | rbc | 64 | 0 | 147.52 | 560 | 682 | 512 | N | 128 | N | 2.55e-03 |
| clustered | A_v01 | rbc | 64 | 1 | 147.68 | 560 | 682 | 512 | N | 128 | N | 2.55e-03 |
| clustered | A_v01 | rbc | 64 | 2 | 152.67 | 560 | 682 | 512 | N | 128 | N | 2.55e-03 |
| clustered | A_v01 | rbc | 128 | 0 | 296.70 | 654 | 910 | 512 | N | 128 | N | 2.55e-03 |
| clustered | A_v01 | rbc | 128 | 1 | 297.54 | 654 | 910 | 512 | N | 128 | N | 2.55e-03 |
| clustered | A_v01 | rbc | 128 | 2 | 296.67 | 654 | 910 | 512 | N | 128 | N | 2.55e-03 |
| clustered | A_v01 | signature_only | 16 | 0 | 111.87 | 529 | 546 | 512 | Y | 128 | N | 2.57e-03 |
| clustered | A_v01 | signature_only | 16 | 1 | 115.11 | 529 | 546 | 512 | Y | 128 | N | 2.57e-03 |
| clustered | A_v01 | signature_only | 16 | 2 | 114.75 | 529 | 546 | 512 | Y | 128 | N | 2.57e-03 |
| clustered | A_v01 | signature_only | 32 | 0 | 159.75 | 563 | 618 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | A_v01 | signature_only | 32 | 1 | 158.70 | 563 | 618 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | A_v01 | signature_only | 32 | 2 | 158.72 | 563 | 618 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | A_v01 | signature_only | 64 | 0 | 152.79 | 609 | 731 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | A_v01 | signature_only | 64 | 1 | 158.61 | 609 | 731 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | A_v01 | signature_only | 64 | 2 | 153.23 | 609 | 731 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | A_v01 | signature_only | 128 | 0 | 294.50 | 681 | 936 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | A_v01 | signature_only | 128 | 1 | 299.31 | 681 | 936 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | A_v01 | signature_only | 128 | 2 | 287.93 | 681 | 936 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | B_P1 | q_outer | 16 | 0 | 110.68 | 256 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | q_outer | 16 | 1 | 111.07 | 256 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | q_outer | 16 | 2 | 110.71 | 256 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | q_outer | 32 | 0 | 213.22 | 128 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | q_outer | 32 | 1 | 212.35 | 128 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | q_outer | 32 | 2 | 212.48 | 128 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | q_outer | 64 | 0 | 577.59 | 64 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | q_outer | 64 | 1 | 570.69 | 64 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | q_outer | 64 | 2 | 593.20 | 64 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | q_outer | 128 | 0 | 3166.60 | 32 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | q_outer | 128 | 1 | 3072.10 | 32 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | q_outer | 128 | 2 | 3075.63 | 32 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | rbc | 16 | 0 | 139.35 | 495 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | rbc | 16 | 1 | 141.28 | 495 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | rbc | 16 | 2 | 142.21 | 495 | 512 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | rbc | 32 | 0 | 193.31 | 469 | 524 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | rbc | 32 | 1 | 192.64 | 469 | 524 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | rbc | 32 | 2 | 186.50 | 469 | 524 | 512 | N | 128 | N | 2.57e-03 |
| clustered | B_P1 | rbc | 64 | 0 | 148.02 | 560 | 682 | 512 | N | 128 | N | 2.55e-03 |
| clustered | B_P1 | rbc | 64 | 1 | 152.81 | 560 | 682 | 512 | N | 128 | N | 2.55e-03 |
| clustered | B_P1 | rbc | 64 | 2 | 153.08 | 560 | 682 | 512 | N | 128 | N | 2.55e-03 |
| clustered | B_P1 | rbc | 128 | 0 | 285.73 | 654 | 910 | 512 | N | 128 | N | 2.55e-03 |
| clustered | B_P1 | rbc | 128 | 1 | 289.71 | 654 | 910 | 512 | N | 128 | N | 2.55e-03 |
| clustered | B_P1 | rbc | 128 | 2 | 296.86 | 654 | 910 | 512 | N | 128 | N | 2.55e-03 |
| clustered | B_P1 | signature_only | 16 | 0 | 112.14 | 529 | 546 | 512 | Y | 128 | N | 2.57e-03 |
| clustered | B_P1 | signature_only | 16 | 1 | 111.33 | 529 | 546 | 512 | Y | 128 | N | 2.57e-03 |
| clustered | B_P1 | signature_only | 16 | 2 | 111.40 | 529 | 546 | 512 | Y | 128 | N | 2.57e-03 |
| clustered | B_P1 | signature_only | 32 | 0 | 158.95 | 563 | 618 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | B_P1 | signature_only | 32 | 1 | 153.67 | 563 | 618 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | B_P1 | signature_only | 32 | 2 | 153.43 | 563 | 618 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | B_P1 | signature_only | 64 | 0 | 157.17 | 609 | 731 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | B_P1 | signature_only | 64 | 1 | 158.94 | 609 | 731 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | B_P1 | signature_only | 64 | 2 | 158.74 | 609 | 731 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | B_P1 | signature_only | 128 | 0 | 288.18 | 681 | 936 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | B_P1 | signature_only | 128 | 1 | 287.88 | 681 | 936 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | B_P1 | signature_only | 128 | 2 | 298.94 | 681 | 936 | 512 | Y | 128 | N | 2.55e-03 |
| clustered | C_P2 | q_outer | 16 | 0 | 102.27 | 256 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | q_outer | 16 | 1 | 101.78 | 256 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | q_outer | 16 | 2 | 105.93 | 256 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | q_outer | 32 | 0 | 199.51 | 128 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | q_outer | 32 | 1 | 207.59 | 128 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | q_outer | 32 | 2 | 205.24 | 128 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | q_outer | 64 | 0 | 579.31 | 64 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | q_outer | 64 | 1 | 587.00 | 64 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | q_outer | 64 | 2 | 588.86 | 64 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | q_outer | 128 | 0 | 3106.80 | 32 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | q_outer | 128 | 1 | 3175.75 | 32 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | q_outer | 128 | 2 | 3089.74 | 32 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | rbc | 16 | 0 | 137.61 | 495 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | rbc | 16 | 1 | 136.64 | 495 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | rbc | 16 | 2 | 137.43 | 495 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | rbc | 32 | 0 | 184.80 | 469 | 22 | 10 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | rbc | 32 | 1 | 191.91 | 469 | 22 | 10 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | rbc | 32 | 2 | 191.51 | 469 | 22 | 10 | N | 128 | N | 2.57e-03 |
| clustered | C_P2 | rbc | 64 | 0 | 152.54 | 560 | 314 | 144 | N | 128 | N | 2.55e-03 |
| clustered | C_P2 | rbc | 64 | 1 | 152.19 | 560 | 314 | 144 | N | 128 | N | 2.55e-03 |
| clustered | C_P2 | rbc | 64 | 2 | 146.82 | 560 | 314 | 144 | N | 128 | N | 2.55e-03 |
| clustered | C_P2 | rbc | 128 | 0 | 289.76 | 654 | 674 | 276 | N | 128 | N | 2.55e-03 |
| clustered | C_P2 | rbc | 128 | 1 | 285.80 | 654 | 674 | 276 | N | 128 | N | 2.55e-03 |
| clustered | C_P2 | rbc | 128 | 2 | 296.89 | 654 | 674 | 276 | N | 128 | N | 2.55e-03 |
| clustered | C_P2 | signature_only | 16 | 0 | 113.00 | 529 | 68 | 34 | Y | 128 | N | 2.57e-03 |
| clustered | C_P2 | signature_only | 16 | 1 | 113.34 | 529 | 68 | 34 | Y | 128 | N | 2.57e-03 |
| clustered | C_P2 | signature_only | 16 | 2 | 112.89 | 529 | 68 | 34 | Y | 128 | N | 2.57e-03 |
| clustered | C_P2 | signature_only | 32 | 0 | 157.90 | 563 | 204 | 98 | Y | 128 | N | 2.55e-03 |
| clustered | C_P2 | signature_only | 32 | 1 | 152.77 | 563 | 204 | 98 | Y | 128 | N | 2.55e-03 |
| clustered | C_P2 | signature_only | 32 | 2 | 153.19 | 563 | 204 | 98 | Y | 128 | N | 2.55e-03 |
| clustered | C_P2 | signature_only | 64 | 0 | 156.39 | 609 | 404 | 185 | Y | 128 | N | 2.55e-03 |
| clustered | C_P2 | signature_only | 64 | 1 | 157.67 | 609 | 404 | 185 | Y | 128 | N | 2.55e-03 |
| clustered | C_P2 | signature_only | 64 | 2 | 158.49 | 609 | 404 | 185 | Y | 128 | N | 2.55e-03 |
| clustered | C_P2 | signature_only | 128 | 0 | 298.06 | 681 | 719 | 295 | Y | 128 | N | 2.55e-03 |
| clustered | C_P2 | signature_only | 128 | 1 | 299.04 | 681 | 719 | 295 | Y | 128 | N | 2.55e-03 |
| clustered | C_P2 | signature_only | 128 | 2 | 287.63 | 681 | 719 | 295 | Y | 128 | N | 2.55e-03 |
| clustered | D_P3a | q_outer | 16 | 0 | 134.72 | 256 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | q_outer | 16 | 1 | 139.34 | 256 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | q_outer | 16 | 2 | 138.86 | 256 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | q_outer | 32 | 0 | 270.03 | 128 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | q_outer | 32 | 1 | 270.08 | 128 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | q_outer | 32 | 2 | 269.86 | 128 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | q_outer | 64 | 0 | 673.90 | 64 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | q_outer | 64 | 1 | 669.03 | 64 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | q_outer | 64 | 2 | 668.55 | 64 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | q_outer | 128 | 0 | 2882.43 | 32 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | q_outer | 128 | 1 | 2897.77 | 32 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | q_outer | 128 | 2 | 2848.66 | 32 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | rbc | 16 | 0 | 140.59 | 495 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | rbc | 16 | 1 | 148.11 | 495 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | rbc | 16 | 2 | 150.69 | 495 | 0 | 0 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | rbc | 32 | 0 | 211.34 | 469 | 22 | 10 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | rbc | 32 | 1 | 209.63 | 469 | 22 | 10 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | rbc | 32 | 2 | 209.29 | 469 | 22 | 10 | N | 64 | N | 2.74e-03 |
| clustered | D_P3a | rbc | 64 | 0 | 161.48 | 560 | 314 | 144 | N | 64 | N | 2.55e-03 |
| clustered | D_P3a | rbc | 64 | 1 | 157.70 | 560 | 314 | 144 | N | 64 | N | 2.55e-03 |
| clustered | D_P3a | rbc | 64 | 2 | 158.05 | 560 | 314 | 144 | N | 64 | N | 2.55e-03 |
| clustered | D_P3a | rbc | 128 | 0 | 224.30 | 654 | 674 | 276 | N | 64 | N | 2.55e-03 |
| clustered | D_P3a | rbc | 128 | 1 | 232.72 | 654 | 674 | 276 | N | 64 | N | 2.55e-03 |
| clustered | D_P3a | rbc | 128 | 2 | 232.84 | 654 | 674 | 276 | N | 64 | N | 2.55e-03 |
| clustered | D_P3a | signature_only | 16 | 0 | 107.85 | 529 | 68 | 34 | Y | 64 | N | 2.74e-03 |
| clustered | D_P3a | signature_only | 16 | 1 | 107.56 | 529 | 68 | 34 | Y | 64 | N | 2.74e-03 |
| clustered | D_P3a | signature_only | 16 | 2 | 111.81 | 529 | 68 | 34 | Y | 64 | N | 2.74e-03 |
| clustered | D_P3a | signature_only | 32 | 0 | 163.47 | 563 | 204 | 98 | Y | 64 | N | 2.74e-03 |
| clustered | D_P3a | signature_only | 32 | 1 | 159.11 | 563 | 204 | 98 | Y | 64 | N | 2.74e-03 |
| clustered | D_P3a | signature_only | 32 | 2 | 163.69 | 563 | 204 | 98 | Y | 64 | N | 2.74e-03 |
| clustered | D_P3a | signature_only | 64 | 0 | 164.20 | 609 | 404 | 185 | Y | 64 | N | 2.55e-03 |
| clustered | D_P3a | signature_only | 64 | 1 | 165.98 | 609 | 404 | 185 | Y | 64 | N | 2.55e-03 |
| clustered | D_P3a | signature_only | 64 | 2 | 163.83 | 609 | 404 | 185 | Y | 64 | N | 2.55e-03 |
| clustered | D_P3a | signature_only | 128 | 0 | 222.01 | 681 | 719 | 295 | Y | 64 | N | 2.55e-03 |
| clustered | D_P3a | signature_only | 128 | 1 | 222.27 | 681 | 719 | 295 | Y | 64 | N | 2.55e-03 |
| clustered | D_P3a | signature_only | 128 | 2 | 221.74 | 681 | 719 | 295 | Y | 64 | N | 2.55e-03 |
| clustered | D_P3b | q_outer | 16 | 0 | 101.96 | 256 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | q_outer | 16 | 1 | 105.40 | 256 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | q_outer | 16 | 2 | 104.84 | 256 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | q_outer | 32 | 0 | 205.42 | 128 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | q_outer | 32 | 1 | 202.83 | 128 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | q_outer | 32 | 2 | 200.91 | 128 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | q_outer | 64 | 0 | 580.31 | 64 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | q_outer | 64 | 1 | 584.87 | 64 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | q_outer | 64 | 2 | 566.42 | 64 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | q_outer | 128 | 0 | 3073.20 | 32 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | q_outer | 128 | 1 | 3162.53 | 32 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | q_outer | 128 | 2 | 3157.70 | 32 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | rbc | 16 | 0 | 136.36 | 495 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | rbc | 16 | 1 | 131.80 | 495 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | rbc | 16 | 2 | 132.11 | 495 | 0 | 0 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | rbc | 32 | 0 | 192.03 | 469 | 22 | 10 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | rbc | 32 | 1 | 191.67 | 469 | 22 | 10 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | rbc | 32 | 2 | 187.92 | 469 | 22 | 10 | N | 128 | Y | 2.57e-03 |
| clustered | D_P3b | rbc | 64 | 0 | 146.89 | 560 | 314 | 144 | N | 128 | Y | 2.55e-03 |
| clustered | D_P3b | rbc | 64 | 1 | 152.55 | 560 | 314 | 144 | N | 128 | Y | 2.55e-03 |
| clustered | D_P3b | rbc | 64 | 2 | 152.36 | 560 | 314 | 144 | N | 128 | Y | 2.55e-03 |
| clustered | D_P3b | rbc | 128 | 0 | 285.97 | 654 | 674 | 276 | N | 128 | Y | 2.55e-03 |
| clustered | D_P3b | rbc | 128 | 1 | 285.73 | 654 | 674 | 276 | N | 128 | Y | 2.55e-03 |
| clustered | D_P3b | rbc | 128 | 2 | 286.06 | 654 | 674 | 276 | N | 128 | Y | 2.55e-03 |
| clustered | D_P3b | signature_only | 16 | 0 | 109.66 | 529 | 68 | 34 | Y | 128 | Y | 2.57e-03 |
| clustered | D_P3b | signature_only | 16 | 1 | 110.41 | 529 | 68 | 34 | Y | 128 | Y | 2.57e-03 |
| clustered | D_P3b | signature_only | 16 | 2 | 109.46 | 529 | 68 | 34 | Y | 128 | Y | 2.57e-03 |
| clustered | D_P3b | signature_only | 32 | 0 | 153.26 | 563 | 204 | 98 | Y | 128 | Y | 2.55e-03 |
| clustered | D_P3b | signature_only | 32 | 1 | 152.95 | 563 | 204 | 98 | Y | 128 | Y | 2.55e-03 |
| clustered | D_P3b | signature_only | 32 | 2 | 152.90 | 563 | 204 | 98 | Y | 128 | Y | 2.55e-03 |
| clustered | D_P3b | signature_only | 64 | 0 | 152.56 | 609 | 404 | 185 | Y | 128 | Y | 2.55e-03 |
| clustered | D_P3b | signature_only | 64 | 1 | 152.32 | 609 | 404 | 185 | Y | 128 | Y | 2.55e-03 |
| clustered | D_P3b | signature_only | 64 | 2 | 148.50 | 609 | 404 | 185 | Y | 128 | Y | 2.55e-03 |
| clustered | D_P3b | signature_only | 128 | 0 | 264.32 | 681 | 719 | 295 | Y | 128 | Y | 2.55e-03 |
| clustered | D_P3b | signature_only | 128 | 1 | 268.26 | 681 | 719 | 295 | Y | 128 | Y | 2.55e-03 |
| clustered | D_P3b | signature_only | 128 | 2 | 275.23 | 681 | 719 | 295 | Y | 128 | Y | 2.55e-03 |
| clustered | D_P3c | q_outer | 16 | 0 | 101.66 | 256 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | q_outer | 16 | 1 | 106.25 | 256 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | q_outer | 16 | 2 | 102.77 | 256 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | q_outer | 32 | 0 | 206.97 | 128 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | q_outer | 32 | 1 | 199.73 | 128 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | q_outer | 32 | 2 | 207.08 | 128 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | q_outer | 64 | 0 | 586.00 | 64 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | q_outer | 64 | 1 | 564.74 | 64 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | q_outer | 64 | 2 | 583.02 | 64 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | q_outer | 128 | 0 | 3074.74 | 32 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | q_outer | 128 | 1 | 3007.57 | 32 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | q_outer | 128 | 2 | 3078.77 | 32 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | rbc | 16 | 0 | 136.70 | 495 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | rbc | 16 | 1 | 132.73 | 495 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | rbc | 16 | 2 | 132.31 | 495 | 0 | 0 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | rbc | 32 | 0 | 191.53 | 469 | 22 | 10 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | rbc | 32 | 1 | 185.84 | 469 | 22 | 10 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | rbc | 32 | 2 | 184.26 | 469 | 22 | 10 | N | 128 | N | 2.57e-03 |
| clustered | D_P3c | rbc | 64 | 0 | 152.54 | 560 | 314 | 144 | N | 128 | N | 2.55e-03 |
| clustered | D_P3c | rbc | 64 | 1 | 152.52 | 560 | 314 | 144 | N | 128 | N | 2.55e-03 |
| clustered | D_P3c | rbc | 64 | 2 | 147.25 | 560 | 314 | 144 | N | 128 | N | 2.55e-03 |
| clustered | D_P3c | rbc | 128 | 0 | 297.38 | 654 | 674 | 276 | N | 128 | N | 2.55e-03 |
| clustered | D_P3c | rbc | 128 | 1 | 294.32 | 654 | 674 | 276 | N | 128 | N | 2.55e-03 |
| clustered | D_P3c | rbc | 128 | 2 | 297.02 | 654 | 674 | 276 | N | 128 | N | 2.55e-03 |
| clustered | D_P3c | signature_only | 16 | 0 | 109.67 | 529 | 68 | 34 | Y | 128 | N | 2.57e-03 |
| clustered | D_P3c | signature_only | 16 | 1 | 109.51 | 529 | 68 | 34 | Y | 128 | N | 2.57e-03 |
| clustered | D_P3c | signature_only | 16 | 2 | 112.91 | 529 | 68 | 34 | Y | 128 | N | 2.57e-03 |
| clustered | D_P3c | signature_only | 32 | 0 | 152.72 | 563 | 204 | 98 | Y | 128 | N | 2.55e-03 |
| clustered | D_P3c | signature_only | 32 | 1 | 147.15 | 563 | 204 | 98 | Y | 128 | N | 2.55e-03 |
| clustered | D_P3c | signature_only | 32 | 2 | 152.72 | 563 | 204 | 98 | Y | 128 | N | 2.55e-03 |
| clustered | D_P3c | signature_only | 64 | 0 | 147.34 | 609 | 404 | 185 | Y | 128 | N | 2.55e-03 |
| clustered | D_P3c | signature_only | 64 | 1 | 147.46 | 609 | 404 | 185 | Y | 128 | N | 2.55e-03 |
| clustered | D_P3c | signature_only | 64 | 2 | 153.10 | 609 | 404 | 185 | Y | 128 | N | 2.55e-03 |
| clustered | D_P3c | signature_only | 128 | 0 | 264.63 | 681 | 719 | 295 | Y | 128 | N | 2.55e-03 |
| clustered | D_P3c | signature_only | 128 | 1 | 264.65 | 681 | 719 | 295 | Y | 128 | N | 2.55e-03 |
| clustered | D_P3c | signature_only | 128 | 2 | 264.73 | 681 | 719 | 295 | Y | 128 | N | 2.55e-03 |
| common_private | A_v01 | q_outer | 16 | 0 | 80.71 | 264 | 528 | 512 | N | 128 | N | 2.65e-03 |
| common_private | A_v01 | q_outer | 16 | 1 | 80.37 | 264 | 528 | 512 | N | 128 | N | 2.65e-03 |
| common_private | A_v01 | q_outer | 16 | 2 | 80.45 | 264 | 528 | 512 | N | 128 | N | 2.65e-03 |
| common_private | A_v01 | q_outer | 32 | 0 | 96.44 | 264 | 1056 | 512 | N | 128 | N | 2.73e-03 |
| common_private | A_v01 | q_outer | 32 | 1 | 93.66 | 264 | 1056 | 512 | N | 128 | N | 2.73e-03 |
| common_private | A_v01 | q_outer | 32 | 2 | 93.99 | 264 | 1056 | 512 | N | 128 | N | 2.73e-03 |
| common_private | A_v01 | q_outer | 64 | 0 | 186.45 | 264 | 2112 | 512 | N | 128 | N | 2.62e-03 |
| common_private | A_v01 | q_outer | 64 | 1 | 180.98 | 264 | 2112 | 512 | N | 128 | N | 2.62e-03 |
| common_private | A_v01 | q_outer | 64 | 2 | 187.61 | 264 | 2112 | 512 | N | 128 | N | 2.62e-03 |
| common_private | A_v01 | q_outer | 128 | 0 | 500.26 | 264 | 4224 | 512 | N | 128 | N | 2.71e-03 |
| common_private | A_v01 | q_outer | 128 | 1 | 499.22 | 264 | 4224 | 512 | N | 128 | N | 2.71e-03 |
| common_private | A_v01 | q_outer | 128 | 2 | 499.05 | 264 | 4224 | 512 | N | 128 | N | 2.71e-03 |
| common_private | A_v01 | rbc | 16 | 0 | 81.05 | 264 | 528 | 512 | N | 128 | N | 2.65e-03 |
| common_private | A_v01 | rbc | 16 | 1 | 80.58 | 264 | 528 | 512 | N | 128 | N | 2.65e-03 |
| common_private | A_v01 | rbc | 16 | 2 | 80.52 | 264 | 528 | 512 | N | 128 | N | 2.65e-03 |
| common_private | A_v01 | rbc | 32 | 0 | 116.72 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | rbc | 32 | 1 | 117.08 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | rbc | 32 | 2 | 116.89 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | rbc | 64 | 0 | 207.08 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | rbc | 64 | 1 | 207.29 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | rbc | 64 | 2 | 207.16 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | rbc | 128 | 0 | 588.25 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | rbc | 128 | 1 | 571.91 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | rbc | 128 | 2 | 571.58 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | signature_only | 16 | 0 | 90.42 | 768 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | signature_only | 16 | 1 | 90.41 | 768 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | signature_only | 16 | 2 | 90.32 | 768 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | signature_only | 32 | 0 | 120.06 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | signature_only | 32 | 1 | 116.99 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | signature_only | 32 | 2 | 116.81 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | signature_only | 64 | 0 | 207.10 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | signature_only | 64 | 1 | 207.03 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | signature_only | 64 | 2 | 207.59 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | signature_only | 128 | 0 | 583.17 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | signature_only | 128 | 1 | 571.72 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | A_v01 | signature_only | 128 | 2 | 571.75 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | q_outer | 16 | 0 | 80.13 | 256 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | q_outer | 16 | 1 | 80.48 | 256 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | q_outer | 16 | 2 | 80.27 | 256 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | q_outer | 32 | 0 | 137.86 | 128 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | q_outer | 32 | 1 | 134.58 | 128 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | q_outer | 32 | 2 | 134.28 | 128 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | q_outer | 64 | 0 | 330.35 | 64 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | q_outer | 64 | 1 | 331.20 | 64 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | q_outer | 64 | 2 | 330.68 | 64 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | q_outer | 128 | 0 | 1801.66 | 32 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | q_outer | 128 | 1 | 1782.12 | 32 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | q_outer | 128 | 2 | 1773.57 | 32 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | rbc | 16 | 0 | 81.96 | 256 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | rbc | 16 | 1 | 83.15 | 256 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | rbc | 16 | 2 | 80.57 | 256 | 512 | 512 | N | 128 | N | 2.65e-03 |
| common_private | B_P1 | rbc | 32 | 0 | 116.46 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | rbc | 32 | 1 | 116.78 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | rbc | 32 | 2 | 116.68 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | rbc | 64 | 0 | 207.21 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | rbc | 64 | 1 | 207.31 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | rbc | 64 | 2 | 207.44 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | rbc | 128 | 0 | 571.48 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | rbc | 128 | 1 | 571.50 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | rbc | 128 | 2 | 571.61 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | signature_only | 16 | 0 | 90.14 | 768 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | signature_only | 16 | 1 | 90.37 | 768 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | signature_only | 16 | 2 | 90.51 | 768 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | signature_only | 32 | 0 | 118.31 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | signature_only | 32 | 1 | 116.94 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | signature_only | 32 | 2 | 117.01 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | signature_only | 64 | 0 | 212.97 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | signature_only | 64 | 1 | 207.16 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | signature_only | 64 | 2 | 215.12 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | signature_only | 128 | 0 | 571.41 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | signature_only | 128 | 1 | 571.29 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | B_P1 | signature_only | 128 | 2 | 571.88 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | q_outer | 16 | 0 | 75.49 | 256 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | q_outer | 16 | 1 | 75.85 | 256 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | q_outer | 16 | 2 | 78.38 | 256 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | q_outer | 32 | 0 | 129.01 | 128 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | q_outer | 32 | 1 | 128.98 | 128 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | q_outer | 32 | 2 | 128.89 | 128 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | q_outer | 64 | 0 | 325.51 | 64 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | q_outer | 64 | 1 | 324.62 | 64 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | q_outer | 64 | 2 | 338.71 | 64 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | q_outer | 128 | 0 | 1769.11 | 32 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | q_outer | 128 | 1 | 1777.27 | 32 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | q_outer | 128 | 2 | 1777.56 | 32 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | rbc | 16 | 0 | 75.23 | 256 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | rbc | 16 | 1 | 75.48 | 256 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | rbc | 16 | 2 | 75.48 | 256 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | C_P2 | rbc | 32 | 0 | 117.09 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | rbc | 32 | 1 | 117.12 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | rbc | 32 | 2 | 117.11 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | rbc | 64 | 0 | 207.30 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | rbc | 64 | 1 | 207.18 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | rbc | 64 | 2 | 207.35 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | rbc | 128 | 0 | 579.01 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | rbc | 128 | 1 | 572.19 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | rbc | 128 | 2 | 571.56 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | signature_only | 16 | 0 | 91.56 | 768 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | signature_only | 16 | 1 | 90.63 | 768 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | signature_only | 16 | 2 | 90.39 | 768 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | signature_only | 32 | 0 | 116.90 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | signature_only | 32 | 1 | 117.34 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | signature_only | 32 | 2 | 117.38 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | signature_only | 64 | 0 | 212.82 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | signature_only | 64 | 1 | 207.11 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | signature_only | 64 | 2 | 215.20 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | signature_only | 128 | 0 | 579.00 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | signature_only | 128 | 1 | 572.29 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | C_P2 | signature_only | 128 | 2 | 572.20 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3a | q_outer | 16 | 0 | 98.42 | 256 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | q_outer | 16 | 1 | 98.69 | 256 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | q_outer | 16 | 2 | 99.25 | 256 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | q_outer | 32 | 0 | 171.78 | 128 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | q_outer | 32 | 1 | 172.97 | 128 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | q_outer | 32 | 2 | 173.03 | 128 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | q_outer | 64 | 0 | 375.68 | 64 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | q_outer | 64 | 1 | 375.67 | 64 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | q_outer | 64 | 2 | 375.80 | 64 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | q_outer | 128 | 0 | 1678.32 | 32 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | q_outer | 128 | 1 | 1625.50 | 32 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | q_outer | 128 | 2 | 1694.16 | 32 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | rbc | 16 | 0 | 98.54 | 256 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | rbc | 16 | 1 | 98.62 | 256 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | rbc | 16 | 2 | 98.51 | 256 | 0 | 0 | N | 64 | N | 2.66e-03 |
| common_private | D_P3a | rbc | 32 | 0 | 121.42 | 640 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | rbc | 32 | 1 | 121.67 | 640 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | rbc | 32 | 2 | 125.03 | 640 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | rbc | 64 | 0 | 170.83 | 576 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | rbc | 64 | 1 | 166.41 | 576 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | rbc | 64 | 2 | 167.12 | 576 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | rbc | 128 | 0 | 541.79 | 544 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | rbc | 128 | 1 | 551.38 | 544 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | rbc | 128 | 2 | 560.55 | 544 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | signature_only | 16 | 0 | 100.44 | 768 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | signature_only | 16 | 1 | 100.04 | 768 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | signature_only | 16 | 2 | 100.78 | 768 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | signature_only | 32 | 0 | 120.12 | 640 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | signature_only | 32 | 1 | 121.32 | 640 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | signature_only | 32 | 2 | 125.46 | 640 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | signature_only | 64 | 0 | 166.88 | 576 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | signature_only | 64 | 1 | 166.58 | 576 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | signature_only | 64 | 2 | 166.83 | 576 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | signature_only | 128 | 0 | 540.33 | 544 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | signature_only | 128 | 1 | 540.82 | 544 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3a | signature_only | 128 | 2 | 540.96 | 544 | 1024 | 512 | Y | 64 | N | 2.71e-03 |
| common_private | D_P3b | q_outer | 16 | 0 | 75.21 | 256 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | q_outer | 16 | 1 | 75.38 | 256 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | q_outer | 16 | 2 | 75.35 | 256 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | q_outer | 32 | 0 | 132.48 | 128 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | q_outer | 32 | 1 | 128.67 | 128 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | q_outer | 32 | 2 | 129.23 | 128 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | q_outer | 64 | 0 | 325.11 | 64 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | q_outer | 64 | 1 | 325.03 | 64 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | q_outer | 64 | 2 | 325.69 | 64 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | q_outer | 128 | 0 | 1778.48 | 32 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | q_outer | 128 | 1 | 1768.46 | 32 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | q_outer | 128 | 2 | 1804.67 | 32 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | rbc | 16 | 0 | 75.56 | 256 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | rbc | 16 | 1 | 75.57 | 256 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | rbc | 16 | 2 | 75.63 | 256 | 0 | 0 | N | 128 | Y | 2.65e-03 |
| common_private | D_P3b | rbc | 32 | 0 | 112.27 | 640 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | rbc | 32 | 1 | 112.65 | 640 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | rbc | 32 | 2 | 112.59 | 640 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | rbc | 64 | 0 | 191.25 | 576 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | rbc | 64 | 1 | 191.40 | 576 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | rbc | 64 | 2 | 191.38 | 576 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | rbc | 128 | 0 | 522.79 | 544 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | rbc | 128 | 1 | 523.46 | 544 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | rbc | 128 | 2 | 523.38 | 544 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | signature_only | 16 | 0 | 90.50 | 768 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | signature_only | 16 | 1 | 90.45 | 768 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | signature_only | 16 | 2 | 90.61 | 768 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | signature_only | 32 | 0 | 112.46 | 640 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | signature_only | 32 | 1 | 112.36 | 640 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | signature_only | 32 | 2 | 112.56 | 640 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | signature_only | 64 | 0 | 197.44 | 576 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | signature_only | 64 | 1 | 191.73 | 576 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | signature_only | 64 | 2 | 191.53 | 576 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | signature_only | 128 | 0 | 523.49 | 544 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | signature_only | 128 | 1 | 523.42 | 544 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3b | signature_only | 128 | 2 | 523.48 | 544 | 1024 | 512 | Y | 128 | Y | 2.67e-03 |
| common_private | D_P3c | q_outer | 16 | 0 | 75.30 | 256 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | q_outer | 16 | 1 | 75.86 | 256 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | q_outer | 16 | 2 | 75.53 | 256 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | q_outer | 32 | 0 | 128.67 | 128 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | q_outer | 32 | 1 | 129.05 | 128 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | q_outer | 32 | 2 | 129.02 | 128 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | q_outer | 64 | 0 | 324.99 | 64 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | q_outer | 64 | 1 | 325.28 | 64 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | q_outer | 64 | 2 | 325.46 | 64 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | q_outer | 128 | 0 | 1775.11 | 32 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | q_outer | 128 | 1 | 1775.89 | 32 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | q_outer | 128 | 2 | 1778.55 | 32 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | rbc | 16 | 0 | 75.58 | 256 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | rbc | 16 | 1 | 75.47 | 256 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | rbc | 16 | 2 | 75.65 | 256 | 0 | 0 | N | 128 | N | 2.65e-03 |
| common_private | D_P3c | rbc | 32 | 0 | 112.29 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | rbc | 32 | 1 | 112.66 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | rbc | 32 | 2 | 112.62 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | rbc | 64 | 0 | 190.96 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | rbc | 64 | 1 | 191.47 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | rbc | 64 | 2 | 191.39 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | rbc | 128 | 0 | 525.63 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | rbc | 128 | 1 | 523.50 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | rbc | 128 | 2 | 524.01 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | signature_only | 16 | 0 | 90.46 | 768 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | signature_only | 16 | 1 | 90.65 | 768 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | signature_only | 16 | 2 | 90.14 | 768 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | signature_only | 32 | 0 | 115.23 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | signature_only | 32 | 1 | 112.42 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | signature_only | 32 | 2 | 115.97 | 640 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | signature_only | 64 | 0 | 191.27 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | signature_only | 64 | 1 | 191.51 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | signature_only | 64 | 2 | 191.51 | 576 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | signature_only | 128 | 0 | 523.28 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | signature_only | 128 | 1 | 522.99 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| common_private | D_P3c | signature_only | 128 | 2 | 523.83 | 544 | 1024 | 512 | Y | 128 | N | 2.67e-03 |
| random | A_v01 | q_outer | 16 | 0 | 106.67 | 264 | 528 | 512 | N | 128 | N | 2.48e-03 |
| random | A_v01 | q_outer | 16 | 1 | 104.90 | 264 | 528 | 512 | N | 128 | N | 2.48e-03 |
| random | A_v01 | q_outer | 16 | 2 | 106.38 | 264 | 528 | 512 | N | 128 | N | 2.48e-03 |
| random | A_v01 | q_outer | 32 | 0 | 141.56 | 264 | 1056 | 512 | N | 128 | N | 2.31e-03 |
| random | A_v01 | q_outer | 32 | 1 | 142.74 | 264 | 1056 | 512 | N | 128 | N | 2.31e-03 |
| random | A_v01 | q_outer | 32 | 2 | 138.13 | 264 | 1056 | 512 | N | 128 | N | 2.31e-03 |
| random | A_v01 | q_outer | 64 | 0 | 286.39 | 264 | 2112 | 512 | N | 128 | N | 2.38e-03 |
| random | A_v01 | q_outer | 64 | 1 | 288.71 | 264 | 2112 | 512 | N | 128 | N | 2.38e-03 |
| random | A_v01 | q_outer | 64 | 2 | 288.81 | 264 | 2112 | 512 | N | 128 | N | 2.38e-03 |
| random | A_v01 | q_outer | 128 | 0 | 743.99 | 264 | 4224 | 512 | N | 128 | N | 2.47e-03 |
| random | A_v01 | q_outer | 128 | 1 | 743.04 | 264 | 4224 | 512 | N | 128 | N | 2.47e-03 |
| random | A_v01 | q_outer | 128 | 2 | 735.72 | 264 | 4224 | 512 | N | 128 | N | 2.47e-03 |
| random | A_v01 | rbc | 16 | 0 | 144.67 | 408 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | A_v01 | rbc | 16 | 1 | 144.53 | 408 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | A_v01 | rbc | 16 | 2 | 149.47 | 408 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | A_v01 | rbc | 32 | 0 | 199.93 | 526 | 829 | 512 | N | 128 | N | 2.48e-03 |
| random | A_v01 | rbc | 32 | 1 | 201.17 | 526 | 829 | 512 | N | 128 | N | 2.48e-03 |
| random | A_v01 | rbc | 32 | 2 | 201.59 | 526 | 829 | 512 | N | 128 | N | 2.48e-03 |
| random | A_v01 | rbc | 64 | 0 | 172.58 | 1099 | 1802 | 512 | N | 128 | N | 2.35e-03 |
| random | A_v01 | rbc | 64 | 1 | 173.71 | 1099 | 1802 | 512 | N | 128 | N | 2.35e-03 |
| random | A_v01 | rbc | 64 | 2 | 174.34 | 1099 | 1802 | 512 | N | 128 | N | 2.35e-03 |
| random | A_v01 | rbc | 128 | 0 | 320.84 | 1780 | 3266 | 512 | Y | 128 | N | 2.44e-03 |
| random | A_v01 | rbc | 128 | 1 | 322.82 | 1780 | 3266 | 512 | Y | 128 | N | 2.44e-03 |
| random | A_v01 | rbc | 128 | 2 | 322.75 | 1780 | 3266 | 512 | Y | 128 | N | 2.44e-03 |
| random | A_v01 | signature_only | 16 | 0 | 112.98 | 616 | 720 | 512 | Y | 128 | N | 2.48e-03 |
| random | A_v01 | signature_only | 16 | 1 | 115.74 | 616 | 720 | 512 | Y | 128 | N | 2.48e-03 |
| random | A_v01 | signature_only | 16 | 2 | 115.54 | 616 | 720 | 512 | Y | 128 | N | 2.48e-03 |
| random | A_v01 | signature_only | 32 | 0 | 162.78 | 807 | 1110 | 512 | Y | 128 | N | 2.48e-03 |
| random | A_v01 | signature_only | 32 | 1 | 158.59 | 807 | 1110 | 512 | Y | 128 | N | 2.48e-03 |
| random | A_v01 | signature_only | 32 | 2 | 163.29 | 807 | 1110 | 512 | Y | 128 | N | 2.48e-03 |
| random | A_v01 | signature_only | 64 | 0 | 166.01 | 1163 | 1866 | 512 | Y | 128 | N | 2.35e-03 |
| random | A_v01 | signature_only | 64 | 1 | 165.32 | 1163 | 1866 | 512 | Y | 128 | N | 2.35e-03 |
| random | A_v01 | signature_only | 64 | 2 | 165.97 | 1163 | 1866 | 512 | Y | 128 | N | 2.35e-03 |
| random | A_v01 | signature_only | 128 | 0 | 319.43 | 1780 | 3266 | 512 | Y | 128 | N | 2.44e-03 |
| random | A_v01 | signature_only | 128 | 1 | 322.73 | 1780 | 3266 | 512 | Y | 128 | N | 2.44e-03 |
| random | A_v01 | signature_only | 128 | 2 | 310.96 | 1780 | 3266 | 512 | Y | 128 | N | 2.44e-03 |
| random | B_P1 | q_outer | 16 | 0 | 108.27 | 256 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | q_outer | 16 | 1 | 108.04 | 256 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | q_outer | 16 | 2 | 108.37 | 256 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | q_outer | 32 | 0 | 208.74 | 128 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | q_outer | 32 | 1 | 209.07 | 128 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | q_outer | 32 | 2 | 205.89 | 128 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | q_outer | 64 | 0 | 537.76 | 64 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | q_outer | 64 | 1 | 531.08 | 64 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | q_outer | 64 | 2 | 536.95 | 64 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | q_outer | 128 | 0 | 2755.66 | 32 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | q_outer | 128 | 1 | 2777.21 | 32 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | q_outer | 128 | 2 | 2773.83 | 32 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | rbc | 16 | 0 | 146.97 | 408 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | rbc | 16 | 1 | 149.53 | 408 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | rbc | 16 | 2 | 149.52 | 408 | 512 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | rbc | 32 | 0 | 200.37 | 526 | 829 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | rbc | 32 | 1 | 201.30 | 526 | 829 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | rbc | 32 | 2 | 200.20 | 526 | 829 | 512 | N | 128 | N | 2.48e-03 |
| random | B_P1 | rbc | 64 | 0 | 168.06 | 1099 | 1802 | 512 | N | 128 | N | 2.35e-03 |
| random | B_P1 | rbc | 64 | 1 | 169.00 | 1099 | 1802 | 512 | N | 128 | N | 2.35e-03 |
| random | B_P1 | rbc | 64 | 2 | 171.45 | 1099 | 1802 | 512 | N | 128 | N | 2.35e-03 |
| random | B_P1 | rbc | 128 | 0 | 310.65 | 1780 | 3266 | 512 | Y | 128 | N | 2.44e-03 |
| random | B_P1 | rbc | 128 | 1 | 315.01 | 1780 | 3266 | 512 | Y | 128 | N | 2.44e-03 |
| random | B_P1 | rbc | 128 | 2 | 322.38 | 1780 | 3266 | 512 | Y | 128 | N | 2.44e-03 |
| random | B_P1 | signature_only | 16 | 0 | 114.80 | 616 | 720 | 512 | Y | 128 | N | 2.48e-03 |
| random | B_P1 | signature_only | 16 | 1 | 112.47 | 616 | 720 | 512 | Y | 128 | N | 2.48e-03 |
| random | B_P1 | signature_only | 16 | 2 | 112.18 | 616 | 720 | 512 | Y | 128 | N | 2.48e-03 |
| random | B_P1 | signature_only | 32 | 0 | 159.90 | 807 | 1110 | 512 | Y | 128 | N | 2.48e-03 |
| random | B_P1 | signature_only | 32 | 1 | 158.68 | 807 | 1110 | 512 | Y | 128 | N | 2.48e-03 |
| random | B_P1 | signature_only | 32 | 2 | 158.45 | 807 | 1110 | 512 | Y | 128 | N | 2.48e-03 |
| random | B_P1 | signature_only | 64 | 0 | 170.34 | 1163 | 1866 | 512 | Y | 128 | N | 2.35e-03 |
| random | B_P1 | signature_only | 64 | 1 | 171.81 | 1163 | 1866 | 512 | Y | 128 | N | 2.35e-03 |
| random | B_P1 | signature_only | 64 | 2 | 172.34 | 1163 | 1866 | 512 | Y | 128 | N | 2.35e-03 |
| random | B_P1 | signature_only | 128 | 0 | 313.52 | 1780 | 3266 | 512 | Y | 128 | N | 2.44e-03 |
| random | B_P1 | signature_only | 128 | 1 | 311.18 | 1780 | 3266 | 512 | Y | 128 | N | 2.44e-03 |
| random | B_P1 | signature_only | 128 | 2 | 322.75 | 1780 | 3266 | 512 | Y | 128 | N | 2.44e-03 |
| random | C_P2 | q_outer | 16 | 0 | 99.88 | 256 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | q_outer | 16 | 1 | 99.58 | 256 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | q_outer | 16 | 2 | 102.77 | 256 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | q_outer | 32 | 0 | 196.33 | 128 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | q_outer | 32 | 1 | 203.91 | 128 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | q_outer | 32 | 2 | 196.25 | 128 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | q_outer | 64 | 0 | 542.64 | 64 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | q_outer | 64 | 1 | 545.96 | 64 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | q_outer | 64 | 2 | 546.25 | 64 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | q_outer | 128 | 0 | 2839.77 | 32 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | q_outer | 128 | 1 | 2757.83 | 32 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | q_outer | 128 | 2 | 2773.62 | 32 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | rbc | 16 | 0 | 143.22 | 408 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | rbc | 16 | 1 | 144.12 | 408 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | rbc | 16 | 2 | 143.79 | 408 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | C_P2 | rbc | 32 | 0 | 193.88 | 526 | 534 | 217 | N | 128 | N | 2.48e-03 |
| random | C_P2 | rbc | 32 | 1 | 201.59 | 526 | 534 | 217 | N | 128 | N | 2.48e-03 |
| random | C_P2 | rbc | 32 | 2 | 196.39 | 526 | 534 | 217 | N | 128 | N | 2.48e-03 |
| random | C_P2 | rbc | 64 | 0 | 172.49 | 1099 | 1778 | 488 | N | 128 | N | 2.35e-03 |
| random | C_P2 | rbc | 64 | 1 | 173.67 | 1099 | 1778 | 488 | N | 128 | N | 2.35e-03 |
| random | C_P2 | rbc | 64 | 2 | 168.69 | 1099 | 1778 | 488 | N | 128 | N | 2.35e-03 |
| random | C_P2 | rbc | 128 | 0 | 315.77 | 1780 | 3265 | 511 | Y | 128 | N | 2.44e-03 |
| random | C_P2 | rbc | 128 | 1 | 311.20 | 1780 | 3265 | 511 | Y | 128 | N | 2.44e-03 |
| random | C_P2 | rbc | 128 | 2 | 322.54 | 1780 | 3265 | 511 | Y | 128 | N | 2.44e-03 |
| random | C_P2 | signature_only | 16 | 0 | 112.06 | 616 | 416 | 208 | Y | 128 | N | 2.48e-03 |
| random | C_P2 | signature_only | 16 | 1 | 114.42 | 616 | 416 | 208 | Y | 128 | N | 2.48e-03 |
| random | C_P2 | signature_only | 16 | 2 | 111.10 | 616 | 416 | 208 | Y | 128 | N | 2.48e-03 |
| random | C_P2 | signature_only | 32 | 0 | 161.77 | 807 | 997 | 399 | Y | 128 | N | 2.48e-03 |
| random | C_P2 | signature_only | 32 | 1 | 157.80 | 807 | 997 | 399 | Y | 128 | N | 2.48e-03 |
| random | C_P2 | signature_only | 32 | 2 | 163.64 | 807 | 997 | 399 | Y | 128 | N | 2.48e-03 |
| random | C_P2 | signature_only | 64 | 0 | 170.78 | 1163 | 1853 | 499 | Y | 128 | N | 2.35e-03 |
| random | C_P2 | signature_only | 64 | 1 | 166.12 | 1163 | 1853 | 499 | Y | 128 | N | 2.35e-03 |
| random | C_P2 | signature_only | 64 | 2 | 172.12 | 1163 | 1853 | 499 | Y | 128 | N | 2.35e-03 |
| random | C_P2 | signature_only | 128 | 0 | 315.65 | 1780 | 3265 | 511 | Y | 128 | N | 2.44e-03 |
| random | C_P2 | signature_only | 128 | 1 | 314.02 | 1780 | 3265 | 511 | Y | 128 | N | 2.44e-03 |
| random | C_P2 | signature_only | 128 | 2 | 311.24 | 1780 | 3265 | 511 | Y | 128 | N | 2.44e-03 |
| random | D_P3a | q_outer | 16 | 0 | 134.60 | 256 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | q_outer | 16 | 1 | 135.11 | 256 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | q_outer | 16 | 2 | 136.50 | 256 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | q_outer | 32 | 0 | 271.81 | 128 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | q_outer | 32 | 1 | 273.87 | 128 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | q_outer | 32 | 2 | 263.68 | 128 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | q_outer | 64 | 0 | 624.06 | 64 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | q_outer | 64 | 1 | 614.66 | 64 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | q_outer | 64 | 2 | 629.48 | 64 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | q_outer | 128 | 0 | 2558.11 | 32 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | q_outer | 128 | 1 | 2484.22 | 32 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | q_outer | 128 | 2 | 2505.23 | 32 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | rbc | 16 | 0 | 157.81 | 408 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | rbc | 16 | 1 | 153.82 | 408 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | rbc | 16 | 2 | 176.57 | 408 | 0 | 0 | N | 64 | N | 2.44e-03 |
| random | D_P3a | rbc | 32 | 0 | 236.54 | 526 | 534 | 217 | N | 64 | N | 2.44e-03 |
| random | D_P3a | rbc | 32 | 1 | 226.68 | 526 | 534 | 217 | N | 64 | N | 2.44e-03 |
| random | D_P3a | rbc | 32 | 2 | 225.97 | 526 | 534 | 217 | N | 64 | N | 2.44e-03 |
| random | D_P3a | rbc | 64 | 0 | 180.71 | 1099 | 1778 | 488 | N | 64 | N | 2.28e-03 |
| random | D_P3a | rbc | 64 | 1 | 175.19 | 1099 | 1778 | 488 | N | 64 | N | 2.28e-03 |
| random | D_P3a | rbc | 64 | 2 | 175.62 | 1099 | 1778 | 488 | N | 64 | N | 2.28e-03 |
| random | D_P3a | rbc | 128 | 0 | 233.81 | 1780 | 3265 | 511 | Y | 64 | N | 2.37e-03 |
| random | D_P3a | rbc | 128 | 1 | 243.73 | 1780 | 3265 | 511 | Y | 64 | N | 2.37e-03 |
| random | D_P3a | rbc | 128 | 2 | 242.88 | 1780 | 3265 | 511 | Y | 64 | N | 2.37e-03 |
| random | D_P3a | signature_only | 16 | 0 | 111.12 | 616 | 416 | 208 | Y | 64 | N | 2.44e-03 |
| random | D_P3a | signature_only | 16 | 1 | 111.36 | 616 | 416 | 208 | Y | 64 | N | 2.44e-03 |
| random | D_P3a | signature_only | 16 | 2 | 113.33 | 616 | 416 | 208 | Y | 64 | N | 2.44e-03 |
| random | D_P3a | signature_only | 32 | 0 | 164.23 | 807 | 997 | 399 | Y | 64 | N | 2.44e-03 |
| random | D_P3a | signature_only | 32 | 1 | 161.53 | 807 | 997 | 399 | Y | 64 | N | 2.44e-03 |
| random | D_P3a | signature_only | 32 | 2 | 165.42 | 807 | 997 | 399 | Y | 64 | N | 2.44e-03 |
| random | D_P3a | signature_only | 64 | 0 | 176.51 | 1163 | 1853 | 499 | Y | 64 | N | 2.28e-03 |
| random | D_P3a | signature_only | 64 | 1 | 171.92 | 1163 | 1853 | 499 | Y | 64 | N | 2.28e-03 |
| random | D_P3a | signature_only | 64 | 2 | 172.54 | 1163 | 1853 | 499 | Y | 64 | N | 2.28e-03 |
| random | D_P3a | signature_only | 128 | 0 | 242.66 | 1780 | 3265 | 511 | Y | 64 | N | 2.37e-03 |
| random | D_P3a | signature_only | 128 | 1 | 234.98 | 1780 | 3265 | 511 | Y | 64 | N | 2.37e-03 |
| random | D_P3a | signature_only | 128 | 2 | 234.33 | 1780 | 3265 | 511 | Y | 64 | N | 2.37e-03 |
| random | D_P3b | q_outer | 16 | 0 | 102.45 | 256 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | q_outer | 16 | 1 | 103.01 | 256 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | q_outer | 16 | 2 | 103.19 | 256 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | q_outer | 32 | 0 | 202.48 | 128 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | q_outer | 32 | 1 | 199.35 | 128 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | q_outer | 32 | 2 | 196.62 | 128 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | q_outer | 64 | 0 | 541.58 | 64 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | q_outer | 64 | 1 | 546.67 | 64 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | q_outer | 64 | 2 | 525.85 | 64 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | q_outer | 128 | 0 | 2787.14 | 32 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | q_outer | 128 | 1 | 2845.71 | 32 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | q_outer | 128 | 2 | 2748.96 | 32 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | rbc | 16 | 0 | 142.61 | 408 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | rbc | 16 | 1 | 140.10 | 408 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | rbc | 16 | 2 | 139.56 | 408 | 0 | 0 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | rbc | 32 | 0 | 200.65 | 526 | 534 | 217 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | rbc | 32 | 1 | 200.18 | 526 | 534 | 217 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | rbc | 32 | 2 | 200.92 | 526 | 534 | 217 | N | 128 | Y | 2.48e-03 |
| random | D_P3b | rbc | 64 | 0 | 172.31 | 1099 | 1778 | 488 | N | 128 | Y | 2.35e-03 |
| random | D_P3b | rbc | 64 | 1 | 167.80 | 1099 | 1778 | 488 | N | 128 | Y | 2.35e-03 |
| random | D_P3b | rbc | 64 | 2 | 174.14 | 1099 | 1778 | 488 | N | 128 | Y | 2.35e-03 |
| random | D_P3b | rbc | 128 | 0 | 289.69 | 1780 | 3265 | 511 | Y | 128 | Y | 2.44e-03 |
| random | D_P3b | rbc | 128 | 1 | 289.23 | 1780 | 3265 | 511 | Y | 128 | Y | 2.44e-03 |
| random | D_P3b | rbc | 128 | 2 | 289.14 | 1780 | 3265 | 511 | Y | 128 | Y | 2.44e-03 |
| random | D_P3b | signature_only | 16 | 0 | 110.56 | 616 | 416 | 208 | Y | 128 | Y | 2.48e-03 |
| random | D_P3b | signature_only | 16 | 1 | 111.76 | 616 | 416 | 208 | Y | 128 | Y | 2.48e-03 |
| random | D_P3b | signature_only | 16 | 2 | 110.73 | 616 | 416 | 208 | Y | 128 | Y | 2.48e-03 |
| random | D_P3b | signature_only | 32 | 0 | 157.55 | 807 | 997 | 399 | Y | 128 | Y | 2.48e-03 |
| random | D_P3b | signature_only | 32 | 1 | 159.36 | 807 | 997 | 399 | Y | 128 | Y | 2.48e-03 |
| random | D_P3b | signature_only | 32 | 2 | 158.75 | 807 | 997 | 399 | Y | 128 | Y | 2.48e-03 |
| random | D_P3b | signature_only | 64 | 0 | 165.16 | 1163 | 1853 | 499 | Y | 128 | Y | 2.35e-03 |
| random | D_P3b | signature_only | 64 | 1 | 161.25 | 1163 | 1853 | 499 | Y | 128 | Y | 2.35e-03 |
| random | D_P3b | signature_only | 64 | 2 | 166.19 | 1163 | 1853 | 499 | Y | 128 | Y | 2.35e-03 |
| random | D_P3b | signature_only | 128 | 0 | 288.42 | 1780 | 3265 | 511 | Y | 128 | Y | 2.44e-03 |
| random | D_P3b | signature_only | 128 | 1 | 298.32 | 1780 | 3265 | 511 | Y | 128 | Y | 2.44e-03 |
| random | D_P3b | signature_only | 128 | 2 | 300.07 | 1780 | 3265 | 511 | Y | 128 | Y | 2.44e-03 |
| random | D_P3c | q_outer | 16 | 0 | 99.71 | 256 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | q_outer | 16 | 1 | 103.19 | 256 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | q_outer | 16 | 2 | 99.99 | 256 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | q_outer | 32 | 0 | 202.43 | 128 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | q_outer | 32 | 1 | 203.68 | 128 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | q_outer | 32 | 2 | 203.46 | 128 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | q_outer | 64 | 0 | 541.23 | 64 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | q_outer | 64 | 1 | 525.70 | 64 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | q_outer | 64 | 2 | 546.57 | 64 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | q_outer | 128 | 0 | 2753.23 | 32 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | q_outer | 128 | 1 | 2761.25 | 32 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | q_outer | 128 | 2 | 2727.30 | 32 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | rbc | 16 | 0 | 143.32 | 408 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | rbc | 16 | 1 | 139.71 | 408 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | rbc | 16 | 2 | 139.83 | 408 | 0 | 0 | N | 128 | N | 2.48e-03 |
| random | D_P3c | rbc | 32 | 0 | 199.58 | 526 | 534 | 217 | N | 128 | N | 2.48e-03 |
| random | D_P3c | rbc | 32 | 1 | 195.94 | 526 | 534 | 217 | N | 128 | N | 2.48e-03 |
| random | D_P3c | rbc | 32 | 2 | 193.49 | 526 | 534 | 217 | N | 128 | N | 2.48e-03 |
| random | D_P3c | rbc | 64 | 0 | 172.03 | 1099 | 1778 | 488 | N | 128 | N | 2.35e-03 |
| random | D_P3c | rbc | 64 | 1 | 168.61 | 1099 | 1778 | 488 | N | 128 | N | 2.35e-03 |
| random | D_P3c | rbc | 64 | 2 | 167.89 | 1099 | 1778 | 488 | N | 128 | N | 2.35e-03 |
| random | D_P3c | rbc | 128 | 0 | 297.89 | 1780 | 3265 | 511 | Y | 128 | N | 2.44e-03 |
| random | D_P3c | rbc | 128 | 1 | 288.91 | 1780 | 3265 | 511 | Y | 128 | N | 2.44e-03 |
| random | D_P3c | rbc | 128 | 2 | 295.11 | 1780 | 3265 | 511 | Y | 128 | N | 2.44e-03 |
| random | D_P3c | signature_only | 16 | 0 | 113.31 | 616 | 416 | 208 | Y | 128 | N | 2.48e-03 |
| random | D_P3c | signature_only | 16 | 1 | 111.37 | 616 | 416 | 208 | Y | 128 | N | 2.48e-03 |
| random | D_P3c | signature_only | 16 | 2 | 112.79 | 616 | 416 | 208 | Y | 128 | N | 2.48e-03 |
| random | D_P3c | signature_only | 32 | 0 | 157.62 | 807 | 997 | 399 | Y | 128 | N | 2.48e-03 |
| random | D_P3c | signature_only | 32 | 1 | 155.79 | 807 | 997 | 399 | Y | 128 | N | 2.48e-03 |
| random | D_P3c | signature_only | 32 | 2 | 159.37 | 807 | 997 | 399 | Y | 128 | N | 2.48e-03 |
| random | D_P3c | signature_only | 64 | 0 | 165.04 | 1163 | 1853 | 499 | Y | 128 | N | 2.35e-03 |
| random | D_P3c | signature_only | 64 | 1 | 160.74 | 1163 | 1853 | 499 | Y | 128 | N | 2.35e-03 |
| random | D_P3c | signature_only | 64 | 2 | 163.70 | 1163 | 1853 | 499 | Y | 128 | N | 2.35e-03 |
| random | D_P3c | signature_only | 128 | 0 | 297.34 | 1780 | 3265 | 511 | Y | 128 | N | 2.44e-03 |
| random | D_P3c | signature_only | 128 | 1 | 288.89 | 1780 | 3265 | 511 | Y | 128 | N | 2.44e-03 |
| random | D_P3c | signature_only | 128 | 2 | 289.19 | 1780 | 3265 | 511 | Y | 128 | N | 2.44e-03 |


## §13.1 GPU 단계 게이트 — 영역 탐색

게이트: non-Q RBC 계획이 최강 공통 Q-outer 대비 GPU 시간을 5% 이상 줄이는 held-out 영역이 있는가. `is_q_fallback` 인 계획(= Q-outer 와 구조가 같은 계획)은 non-Q 로 세지 않는다.

### sweep A — Nq512/2048/8192, G8, seed101

| Nq | G | sel | 패턴 | Q-outer us | M | signature us | merge us | Δ non-Q | 게이트 | 승자 | KV 방문비 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|
| 512 | 8 | 8 | clustered | 57.8 | M16 | 62.6 | 63.9 | +8.3% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 8 | common_private | 43.6 | M16 | 55.2 | 66.7 | +26.7% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 8 | random | 56.4 | M16 | 63.4 | 73.2 | +12.5% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 16 | clustered | 105.7 | M16 | 109.5 | 137.9 | +3.6% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 16 | common_private | 77.9 | M16 | 90.2 | 112.5 | +15.8% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 16 | random | 102.3 | M16 | 109.2 | 144.3 | +6.7% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 32 | clustered | 199.2 | M16 | 204.5 | 267.6 | +2.7% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 32 | common_private | 147.1 | M16 | 156.5 | 200.3 | +6.4% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 32 | random | 188.9 | M16 | 206.3 | 262.8 | +9.2% | 미통과 | signature_only | 1.00x |
| 2048 | 8 | 8 | clustered | 205.7 | M16 | 212.2 | 209.1 | +1.6% | 미통과 | rbc | 1.00x |
| 2048 | 8 | 8 | common_private | 154.6 | M16 | 186.8 | 204.6 | +20.8% | 미통과 | signature_only | 1.00x |
| 2048 | 8 | 8 | random | 199.4 | M16 | 207.4 | 216.2 | +4.0% | 미통과 | signature_only | 1.00x |
| 2048 | 8 | 16 | clustered | 390.2 | M16 | 391.6 | 426.4 | +0.4% | 미통과 | signature_only | 1.00x |
| 2048 | 8 | 16 | common_private | 287.1 | M16 | 320.8 | 352.6 | +11.7% | 미통과 | signature_only | 1.00x |
| 2048 | 8 | 16 | random | 374.4 | M16 | 384.5 | 404.7 | +2.7% | 미통과 | signature_only | 1.00x |
| 2048 | 8 | 32 | clustered | 755.6 | M16 | 750.6 | 811.4 | -0.7% | 미통과 | signature_only | 1.00x |
| 2048 | 8 | 32 | common_private | 557.0 | M16 | 581.9 | 638.1 | +4.5% | 미통과 | signature_only | 1.00x |
| 2048 | 8 | 32 | random | 715.1 | M16 | 731.0 | 779.1 | +2.2% | 미통과 | signature_only | 1.00x |
| 8192 | 8 | 8 | clustered | 738.5 | M16 | 746.8 | 749.6 | +1.1% | 미통과 | signature_only | 1.00x |
| 8192 | 8 | 8 | common_private | 547.6 | M16 | 651.9 | 726.6 | +19.1% | 미통과 | signature_only | 1.00x |
| 8192 | 8 | 8 | random | 729.7 | M16 | 742.7 | 743.5 | +1.8% | 미통과 | signature_only | 1.00x |
| 8192 | 8 | 16 | clustered | 1431.5 | M16 | 1409.9 | 1429.5 | -1.5% | 미통과 | signature_only | 1.00x |
| 8192 | 8 | 16 | common_private | 1056.4 | M16 | 1146.5 | 1282.2 | +8.5% | 미통과 | signature_only | 1.00x |
| 8192 | 8 | 16 | random | 1371.5 | M16 | 1406.8 | 1402.3 | +2.3% | 미통과 | rbc | 1.00x |
| 8192 | 8 | 32 | clustered | 2756.6 | M16 | 2722.1 | 2733.6 | -1.3% | 미통과 | signature_only | 1.00x |
| 8192 | 8 | 32 | common_private | 2054.4 | M16 | 2134.5 | 2475.1 | +3.9% | 미통과 | signature_only | 1.00x |
| 8192 | 8 | 32 | random | 2603.3 | M16 | 2690.5 | 2645.3 | +1.6% | 미통과 | rbc | 1.00x |

통과 0/27 영역

### sweep B — Nq128/256/512, G4/8/16, seed101~103

| Nq | G | sel | 패턴 | Q-outer us | M | signature us | merge us | Δ non-Q | 게이트 | 승자 | KV 방문비 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|
| 128 | 4 | 8 | clustered | 76.4 | M16 | 27.1 | 28.1 | -64.5% | 통과 | signature_only | 1.00x |
| 128 | 4 | 8 | common_private | 50.7 | M16 | 20.9 | 25.9 | -58.8% | 통과 | signature_only | 1.00x |
| 128 | 4 | 8 | random | 75.0 | M16 | 28.7 | 30.2 | -61.7% | 통과 | signature_only | 1.00x |
| 128 | 4 | 16 | clustered | 149.6 | M16 | 50.1 | 52.7 | -66.5% | 통과 | signature_only | 1.00x |
| 128 | 4 | 16 | common_private | 96.0 | M16 | 34.9 | 45.0 | -63.6% | 통과 | signature_only | 1.00x |
| 128 | 4 | 16 | random | 144.2 | M16 | 56.4 | 54.6 | -62.1% | 통과 | rbc | 0.93x |
| 128 | 4 | 32 | clustered | 287.4 | M16 | 104.6 | 98.6 | -65.7% | 통과 | rbc | 0.89x |
| 128 | 4 | 32 | common_private | 185.4 | M16 | 56.6 | 73.0 | -69.5% | 통과 | signature_only | 1.00x |
| 128 | 4 | 32 | random | 263.3 | M16 | 87.4 | 88.1 | -66.8% | 통과 | signature_only | 0.48x |
| 128 | 8 | 8 | clustered | 44.6 | M16 | 29.4 | 30.0 | -34.1% | 통과 | signature_only | 1.00x |
| 128 | 8 | 8 | common_private | 34.4 | M16 | 21.7 | 25.5 | -36.8% | 통과 | signature_only | 1.00x |
| 128 | 8 | 8 | random | 44.2 | M16 | 31.9 | 41.7 | -27.8% | 통과 | signature_only | 0.99x |
| 128 | 8 | 16 | clustered | 85.1 | M16 | 50.9 | 73.5 | -40.2% | 통과 | signature_only | 1.00x |
| 128 | 8 | 16 | common_private | 64.8 | M16 | 33.7 | 42.2 | -47.9% | 통과 | signature_only | 1.00x |
| 128 | 8 | 16 | random | 84.0 | M16 | 54.3 | 75.3 | -35.3% | 통과 | signature_only | 1.00x |
| 128 | 8 | 32 | clustered | 161.8 | M16 | 103.3 | 139.4 | -36.1% | 통과 | signature_only | 1.00x |
| 128 | 8 | 32 | common_private | 122.1 | M16 | 57.5 | 71.3 | -52.9% | 통과 | signature_only | 1.00x |
| 128 | 8 | 32 | random | 155.4 | M16 | 95.3 | 127.2 | -38.7% | 통과 | signature_only | 1.00x |
| 128 | 16 | 8 | clustered | 27.4 | M16 | 34.4 | 34.8 | +25.7% | 미통과 | signature_only | 0.97x |
| 128 | 16 | 8 | common_private | 25.7 | M16 | 26.6 | 26.7 | +3.5% | 미통과 | signature_only | 0.75x |
| 128 | 16 | 8 | random | 28.1 | M16 | 38.5 | 38.7 | +36.8% | 미통과 | signature_only | 0.99x |
| 128 | 16 | 16 | clustered | 48.0 | M16 | 59.9 | 60.0 | +24.8% | 미통과 | signature_only | 0.98x |
| 128 | 16 | 16 | common_private | 46.0 | M16 | 43.3 | 42.5 | -7.7% | 통과 | rbc | 0.75x |
| 128 | 16 | 16 | random | 47.0 | M16 | 68.7 | 68.5 | +45.6% | 미통과 | rbc | 0.98x |
| 128 | 16 | 32 | clustered | 86.9 | M16 | 125.3 | 126.0 | +44.1% | 미통과 | signature_only | 0.98x |
| 128 | 16 | 32 | common_private | 83.8 | M16 | 70.1 | 69.7 | -16.9% | 통과 | rbc | 0.75x |
| 128 | 16 | 32 | random | 83.2 | M16 | 124.3 | 123.9 | +49.0% | 미통과 | rbc | 0.97x |
| 256 | 4 | 8 | clustered | 77.9 | M16 | 34.1 | 34.9 | -56.3% | 통과 | signature_only | 1.00x |
| 256 | 4 | 8 | common_private | 52.0 | M16 | 35.9 | 44.6 | -30.9% | 통과 | signature_only | 1.00x |
| 256 | 4 | 8 | random | 79.0 | M16 | 36.6 | 40.5 | -53.7% | 통과 | signature_only | 1.00x |
| 256 | 4 | 16 | clustered | 148.2 | M16 | 65.1 | 63.1 | -57.4% | 통과 | rbc | 0.94x |
| 256 | 4 | 16 | common_private | 96.2 | M16 | 59.9 | 76.4 | -37.7% | 통과 | signature_only | 1.00x |
| 256 | 4 | 16 | random | 144.1 | M16 | 71.1 | 78.0 | -50.7% | 통과 | signature_only | 1.00x |
| 256 | 4 | 32 | clustered | 282.0 | M16 | 128.6 | 118.5 | -58.0% | 통과 | rbc | 0.89x |
| 256 | 4 | 32 | common_private | 184.0 | M16 | 107.4 | 139.7 | -41.6% | 통과 | signature_only | 1.00x |
| 256 | 4 | 32 | random | 264.0 | M16 | 134.5 | 170.5 | -49.0% | 통과 | signature_only | 1.00x |
| 256 | 8 | 8 | clustered | 49.2 | M16 | 36.4 | 36.9 | -26.1% | 통과 | signature_only | 1.00x |
| 256 | 8 | 8 | common_private | 38.6 | M16 | 37.3 | 45.3 | -3.4% | 미통과 | signature_only | 1.00x |
| 256 | 8 | 8 | random | 49.0 | M16 | 38.8 | 49.1 | -20.7% | 통과 | signature_only | 1.00x |
| 256 | 8 | 16 | clustered | 87.7 | M16 | 60.8 | 83.9 | -30.7% | 통과 | signature_only | 1.00x |
| 256 | 8 | 16 | common_private | 69.6 | M16 | 61.3 | 75.3 | -11.9% | 통과 | signature_only | 1.00x |
| 256 | 8 | 16 | random | 87.5 | M16 | 63.2 | 95.1 | -27.8% | 통과 | signature_only | 1.00x |
| 256 | 8 | 32 | clustered | 166.4 | M16 | 116.2 | 169.0 | -30.2% | 통과 | signature_only | 1.00x |
| 256 | 8 | 32 | common_private | 125.4 | M16 | 106.3 | 137.9 | -15.2% | 통과 | signature_only | 1.00x |
| 256 | 8 | 32 | random | 159.2 | M16 | 116.3 | 181.2 | -26.9% | 통과 | signature_only | 1.00x |
| 256 | 16 | 8 | clustered | 34.1 | M16 | 44.2 | 43.2 | +26.7% | 미통과 | rbc | 1.00x |
| 256 | 16 | 8 | common_private | 30.3 | M16 | 47.8 | 47.7 | +57.1% | 미통과 | rbc | 0.75x |
| 256 | 16 | 8 | random | 33.6 | M16 | 47.6 | 47.4 | +40.9% | 미통과 | rbc | 0.99x |
| 256 | 16 | 16 | clustered | 58.0 | M16 | 74.8 | 74.9 | +29.0% | 미통과 | signature_only | 0.99x |
| 256 | 16 | 16 | common_private | 53.4 | M16 | 78.2 | 77.8 | +45.6% | 미통과 | rbc | 0.75x |
| 256 | 16 | 16 | random | 55.3 | M16 | 79.2 | 79.5 | +43.4% | 미통과 | signature_only | 0.99x |
| 256 | 16 | 32 | clustered | 105.8 | M16 | 147.4 | 152.2 | +39.3% | 미통과 | signature_only | 0.98x |
| 256 | 16 | 32 | common_private | 96.4 | M16 | 135.5 | 135.5 | +40.6% | 미통과 | signature_only | 0.75x |
| 256 | 16 | 32 | random | 98.6 | M16 | 148.8 | 150.4 | +51.0% | 미통과 | signature_only | 0.97x |
| 512 | 4 | 8 | clustered | 80.2 | M16 | 62.0 | 61.3 | -23.5% | 통과 | rbc | 0.97x |
| 512 | 4 | 8 | common_private | 56.2 | M16 | 52.9 | 65.8 | -5.9% | 통과 | signature_only | 1.00x |
| 512 | 4 | 8 | random | 81.3 | M16 | 66.6 | 65.8 | -19.1% | 통과 | rbc | 0.97x |
| 512 | 4 | 16 | clustered | 149.7 | M16 | 116.0 | 110.1 | -26.4% | 통과 | rbc | 0.94x |
| 512 | 4 | 16 | common_private | 99.0 | M16 | 90.2 | 113.4 | -8.9% | 통과 | signature_only | 1.00x |
| 512 | 4 | 16 | random | 147.7 | M16 | 119.7 | 118.3 | -19.9% | 통과 | rbc | 0.94x |
| 512 | 4 | 32 | clustered | 284.6 | M16 | 213.1 | 217.0 | -25.1% | 통과 | signature_only | 0.89x |
| 512 | 4 | 32 | common_private | 186.9 | M16 | 157.9 | 211.8 | -15.5% | 통과 | signature_only | 1.00x |
| 512 | 4 | 32 | random | 270.6 | M16 | 226.9 | 273.8 | -16.2% | 통과 | signature_only | 1.00x |
| 512 | 8 | 8 | clustered | 57.6 | M16 | 62.4 | 67.3 | +8.3% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 8 | common_private | 44.3 | M16 | 55.5 | 66.3 | +25.3% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 8 | random | 56.9 | M16 | 63.1 | 73.2 | +10.9% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 16 | clustered | 106.2 | M16 | 109.7 | 134.0 | +3.3% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 16 | common_private | 78.6 | M16 | 90.7 | 114.2 | +15.4% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 16 | random | 101.8 | M16 | 109.5 | 140.0 | +7.6% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 32 | clustered | 200.3 | M16 | 208.5 | 266.7 | +4.1% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 32 | common_private | 147.1 | M16 | 157.2 | 200.5 | +6.9% | 미통과 | signature_only | 1.00x |
| 512 | 8 | 32 | random | 189.7 | M16 | 207.0 | 267.8 | +9.1% | 미통과 | signature_only | 1.00x |
| 512 | 16 | 8 | clustered | 59.4 | M16 | 76.6 | 77.4 | +28.9% | 미통과 | signature_only | 1.00x |
| 512 | 16 | 8 | common_private | 56.7 | M16 | 72.6 | 72.1 | +27.1% | 미통과 | rbc | 0.75x |
| 512 | 16 | 8 | random | 59.1 | M16 | 79.7 | 79.5 | +34.6% | 미통과 | rbc | 0.99x |
| 512 | 16 | 16 | clustered | 105.9 | M16 | 137.1 | 137.3 | +29.4% | 미통과 | signature_only | 0.99x |
| 512 | 16 | 16 | common_private | 99.9 | M16 | 114.7 | 114.6 | +14.7% | 미통과 | rbc | 0.75x |
| 512 | 16 | 16 | random | 102.4 | M16 | 140.1 | 139.9 | +36.6% | 미통과 | rbc | 0.99x |
| 512 | 16 | 32 | clustered | 198.6 | M16 | 266.5 | 268.9 | +34.2% | 미통과 | signature_only | 0.98x |
| 512 | 16 | 32 | common_private | 185.5 | M16 | 199.6 | 199.6 | +7.6% | 미통과 | signature_only | 0.75x |
| 512 | 16 | 32 | random | 186.8 | M16 | 268.4 | 270.2 | +43.7% | 미통과 | signature_only | 0.97x |

통과 46/81 영역, 감소율 중앙값 -35.7%, 최대 -69.5%

| 축 | 통과/전체 |
|---|---|
| Nq | 128=20/27, 256=17/27, 512=9/27 |
| G | 16=2/27, 4=27/27, 8=17/27 |
| sel | 16=16/27, 32=16/27, 8=14/27 |
| 패턴 | clustered=15/27, common_private=16/27, random=15/27 |


## P4 — 시간 모델 교정 (§7)

교정 seed [0, 1, 2, 3, 4] / Nq [128, 256, 512, 2048] / G [4, 8] / sel [8, 16, 32] / 패턴 ['common_private', 'random', 'clustered'] / max_m [16, 32, 64, 128].
표본 4320개. 시간 측정 방식: cuda graph replay, profiling 없음.

회귀는 (BM, G) 구간별 **비음수 최소제곱(NNLS)** + 상대오차 가중이다. 음수 계수를 허용하면 외삽에서 음수 시간을 만들고 그것이 '가장 빠른 후보'로 보이게 된다 (실제로 한 번 발생해 regret 415% 를 만들었다).

특징 벡터:

```
const, l_work, l_tail, padded_per_sm, n_programs_per_sm, unique_blocks, multi_rows, empty_rows, k_p50_per_sm, k_p90, imbalance, full_ratio
```

| BM:G | n | 잔차 p50 | p90 | max | 관측 y 범위 us | diff margin |
|---|---:|---:|---:|---:|---:|---:|
| 1:4 | 2 | 0.000 | 0.000 | 0.000 | 23~23 | 전역 |
| 1:8 | 4 | 0.004 | 0.006 | 0.007 | 25~25 | 전역 |
| 2:4 | 47 | 0.041 | 0.090 | 0.123 | 27~169 | 전역 |
| 2:8 | 612 | 0.078 | 0.166 | 0.317 | 21~833 | 0.232 |
| 4:4 | 855 | 0.054 | 0.148 | 0.285 | 21~841 | 0.215 |
| 4:8 | 778 | 0.045 | 0.149 | 0.245 | 26~1090 | 전역 |
| 8:4 | 622 | 0.076 | 0.202 | 0.581 | 26~1041 | 전역 |
| 8:8 | 466 | 0.080 | 0.137 | 0.273 | 46~2207 | 전역 |
| 16:4 | 334 | 0.104 | 0.282 | 0.335 | 49~1897 | 전역 |
| 16:8 | 300 | 0.097 | 0.250 | 0.381 | 120~5026 | 전역 |
| 32:4 | 300 | 0.101 | 0.276 | 0.330 | 72~6050 | 전역 |

후보 **차이** 예측의 상대 잔차 (error_margin 의 근거): p50 0.097 / p90 0.217 (n=720)

계획 생성 CPU 비용 중앙값 (선택기의 `extra_prepare` 근거):

| 후보 | us |
|---|---:|
| q_outer | 279 |
| signature | 343 |
| bounded_merge | 326 |

→ `extra_prepare` = 64us


형상 구간별 최적 max_m lookup: `{'4:128': 16, '8:128': 16, '4:256': 16, '8:256': 16, '4:512': 16, '8:512': 16, '4:2048': 16, '8:2048': 16}`


### 교정 원시 표본 (4320개)

(형상·후보·max_m 별 실측 GPU 시간과 특징값)

| seed | Nq | G | sel | 패턴 | 후보 | max_m | BM | 실측 us | tasks | sum_k | max_k | padded | multi | uniq | k_p50 | k_p90 | full비 |
|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 4 | 8 | clustered | bounded_merge | 16 | 1 | 23.36 | 128 | 1024 | 8 | 16384 | 0 | 450 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 56.20 | 123 | 1010 | 23 | 16160 | 0 | 437 | 8.0 | 8.0 | 0.958 |
| 2 | 128 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 54.03 | 119 | 1000 | 23 | 16000 | 0 | 436 | 8.0 | 8.0 | 0.918 |
| 3 | 128 | 4 | 8 | clustered | bounded_merge | 16 | 2 | 37.84 | 125 | 1013 | 14 | 16208 | 0 | 445 | 8.0 | 8.0 | 0.974 |
| 4 | 128 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 48.42 | 121 | 1001 | 19 | 16016 | 0 | 438 | 8.0 | 8.0 | 0.943 |
| 0 | 128 | 4 | 8 | clustered | bounded_merge | 32 | 2 | 27.80 | 129 | 1000 | 8 | 16000 | 7 | 450 | 8.0 | 8.0 | 0.977 |
| 1 | 128 | 4 | 8 | clustered | bounded_merge | 32 | 2 | 28.52 | 128 | 962 | 8 | 15392 | 13 | 437 | 8.0 | 8.0 | 0.965 |
| 2 | 128 | 4 | 8 | clustered | bounded_merge | 32 | 2 | 28.45 | 131 | 973 | 8 | 15568 | 17 | 436 | 8.0 | 8.0 | 0.967 |
| 3 | 128 | 4 | 8 | clustered | bounded_merge | 32 | 2 | 27.72 | 131 | 941 | 8 | 15056 | 19 | 445 | 8.0 | 8.0 | 0.966 |
| 4 | 128 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 29.99 | 131 | 976 | 8 | 15616 | 12 | 438 | 8.0 | 8.0 | 0.973 |
| 0 | 128 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 31.42 | 145 | 931 | 8 | 14896 | 36 | 450 | 8.0 | 8.0 | 0.969 |
| 1 | 128 | 4 | 8 | clustered | bounded_merge | 64 | 2 | 29.06 | 140 | 892 | 8 | 14272 | 37 | 437 | 8.0 | 8.0 | 0.974 |
| 2 | 128 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 31.10 | 147 | 908 | 8 | 14528 | 46 | 436 | 8.0 | 8.0 | 0.977 |
| 3 | 128 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 29.90 | 148 | 882 | 8 | 14112 | 46 | 445 | 8.0 | 8.0 | 0.976 |
| 4 | 128 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 31.72 | 141 | 931 | 8 | 14896 | 30 | 438 | 8.0 | 8.0 | 0.984 |
| 0 | 128 | 4 | 8 | clustered | bounded_merge | 128 | 4 | 33.51 | 162 | 818 | 8 | 13088 | 66 | 450 | 5.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 8 | clustered | bounded_merge | 128 | 4 | 32.88 | 158 | 806 | 8 | 12896 | 66 | 437 | 6.0 | 8.0 | 1.000 |
| 2 | 128 | 4 | 8 | clustered | bounded_merge | 128 | 4 | 32.26 | 166 | 815 | 8 | 13040 | 84 | 436 | 5.0 | 8.0 | 1.000 |
| 3 | 128 | 4 | 8 | clustered | bounded_merge | 128 | 4 | 32.66 | 168 | 825 | 8 | 13200 | 70 | 445 | 5.0 | 8.0 | 1.000 |
| 4 | 128 | 4 | 8 | clustered | bounded_merge | 128 | 4 | 32.78 | 167 | 840 | 8 | 13440 | 74 | 438 | 5.0 | 8.0 | 1.000 |
| 0 | 128 | 4 | 8 | clustered | q_outer | 16 | 4 | 75.22 | 32 | 1024 | 32 | 16384 | 0 | 450 | 32.0 | 32.0 | 0.000 |
| 1 | 128 | 4 | 8 | clustered | q_outer | 16 | 4 | 77.46 | 32 | 1010 | 32 | 16160 | 0 | 437 | 32.0 | 32.0 | 0.000 |
| 2 | 128 | 4 | 8 | clustered | q_outer | 16 | 4 | 76.78 | 32 | 1000 | 32 | 16000 | 0 | 436 | 32.0 | 32.0 | 0.000 |
| 3 | 128 | 4 | 8 | clustered | q_outer | 16 | 4 | 76.78 | 32 | 1013 | 32 | 16208 | 0 | 445 | 32.0 | 32.0 | 0.000 |
| 4 | 128 | 4 | 8 | clustered | q_outer | 16 | 4 | 77.11 | 32 | 1001 | 32 | 16016 | 0 | 438 | 32.0 | 32.0 | 0.000 |
| 0 | 128 | 4 | 8 | clustered | q_outer | 32 | 8 | 221.03 | 16 | 1000 | 64 | 32000 | 0 | 450 | 64.0 | 64.0 | 0.000 |
| 1 | 128 | 4 | 8 | clustered | q_outer | 32 | 8 | 213.18 | 16 | 962 | 64 | 30784 | 0 | 437 | 62.0 | 64.0 | 0.000 |
| 2 | 128 | 4 | 8 | clustered | q_outer | 32 | 8 | 213.83 | 16 | 973 | 64 | 31136 | 0 | 436 | 62.0 | 64.0 | 0.000 |
| 3 | 128 | 4 | 8 | clustered | q_outer | 32 | 8 | 216.00 | 16 | 941 | 64 | 30112 | 0 | 445 | 59.0 | 64.0 | 0.000 |
| 4 | 128 | 4 | 8 | clustered | q_outer | 32 | 8 | 218.66 | 16 | 976 | 64 | 31232 | 0 | 438 | 64.0 | 64.0 | 0.000 |
| 0 | 128 | 4 | 8 | clustered | q_outer | 64 | 16 | 597.60 | 8 | 931 | 125 | 59584 | 0 | 450 | 115.0 | 122.9 | 0.000 |
| 1 | 128 | 4 | 8 | clustered | q_outer | 64 | 16 | 559.37 | 8 | 892 | 116 | 57088 | 0 | 437 | 113.0 | 115.3 | 0.000 |
| 2 | 128 | 4 | 8 | clustered | q_outer | 64 | 16 | 578.39 | 8 | 908 | 121 | 58112 | 0 | 436 | 115.0 | 120.3 | 0.000 |
| 3 | 128 | 4 | 8 | clustered | q_outer | 64 | 16 | 571.84 | 8 | 882 | 119 | 56448 | 0 | 445 | 111.5 | 117.6 | 0.000 |
| 4 | 128 | 4 | 8 | clustered | q_outer | 64 | 16 | 604.33 | 8 | 931 | 127 | 59584 | 0 | 438 | 117.5 | 123.5 | 0.000 |
| 0 | 128 | 4 | 8 | clustered | q_outer | 128 | 32 | 2448.00 | 4 | 818 | 215 | 104704 | 0 | 450 | 203.0 | 211.7 | 0.000 |
| 1 | 128 | 4 | 8 | clustered | q_outer | 128 | 32 | 2467.40 | 4 | 806 | 218 | 103168 | 0 | 437 | 199.0 | 213.8 | 0.000 |
| 2 | 128 | 4 | 8 | clustered | q_outer | 128 | 32 | 2482.54 | 4 | 815 | 218 | 104320 | 0 | 436 | 204.0 | 214.4 | 0.000 |
| 3 | 128 | 4 | 8 | clustered | q_outer | 128 | 32 | 2383.82 | 4 | 825 | 209 | 105600 | 0 | 445 | 206.0 | 208.4 | 0.000 |
| 4 | 128 | 4 | 8 | clustered | q_outer | 128 | 32 | 2493.39 | 4 | 840 | 218 | 107520 | 0 | 438 | 210.5 | 216.2 | 0.000 |
| 0 | 128 | 4 | 8 | clustered | signature | 16 | 1 | 23.35 | 128 | 1024 | 8 | 16384 | 0 | 450 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 8 | clustered | signature | 16 | 2 | 27.59 | 129 | 1010 | 8 | 16160 | 4 | 437 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 4 | 8 | clustered | signature | 16 | 2 | 28.38 | 133 | 1000 | 8 | 16000 | 12 | 436 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 4 | 8 | clustered | signature | 16 | 2 | 27.65 | 131 | 1013 | 8 | 16208 | 6 | 445 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 4 | 8 | clustered | signature | 16 | 4 | 31.48 | 132 | 1001 | 8 | 16016 | 7 | 438 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 4 | 8 | clustered | signature | 32 | 2 | 26.81 | 134 | 1000 | 8 | 16000 | 12 | 450 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 8 | clustered | signature | 32 | 2 | 28.56 | 136 | 962 | 8 | 15392 | 21 | 437 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 4 | 8 | clustered | signature | 32 | 2 | 27.85 | 138 | 973 | 8 | 15568 | 23 | 436 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 4 | 8 | clustered | signature | 32 | 2 | 29.02 | 142 | 941 | 8 | 15056 | 29 | 445 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 4 | 8 | clustered | signature | 32 | 4 | 33.07 | 138 | 976 | 8 | 15616 | 18 | 438 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 4 | 8 | clustered | signature | 64 | 4 | 33.36 | 150 | 931 | 8 | 14896 | 41 | 450 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 8 | clustered | signature | 64 | 2 | 28.12 | 146 | 892 | 8 | 14272 | 43 | 437 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 4 | 8 | clustered | signature | 64 | 4 | 33.09 | 152 | 908 | 8 | 14528 | 51 | 436 | 7.0 | 8.0 | 1.000 |
| 3 | 128 | 4 | 8 | clustered | signature | 64 | 4 | 33.35 | 154 | 882 | 8 | 14112 | 51 | 445 | 7.0 | 8.0 | 1.000 |
| 4 | 128 | 4 | 8 | clustered | signature | 64 | 4 | 33.16 | 146 | 931 | 8 | 14896 | 35 | 438 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 4 | 8 | clustered | signature | 128 | 4 | 33.42 | 162 | 818 | 8 | 13088 | 66 | 450 | 5.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 8 | clustered | signature | 128 | 4 | 32.82 | 158 | 806 | 8 | 12896 | 66 | 437 | 6.0 | 8.0 | 1.000 |
| 2 | 128 | 4 | 8 | clustered | signature | 128 | 4 | 32.45 | 166 | 815 | 8 | 13040 | 84 | 436 | 5.0 | 8.0 | 1.000 |
| 3 | 128 | 4 | 8 | clustered | signature | 128 | 4 | 33.17 | 168 | 825 | 8 | 13200 | 70 | 445 | 5.0 | 8.0 | 1.000 |
| 4 | 128 | 4 | 8 | clustered | signature | 128 | 4 | 33.47 | 167 | 840 | 8 | 13440 | 74 | 438 | 5.0 | 8.0 | 1.000 |
| 0 | 128 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 49.25 | 32 | 640 | 20 | 10240 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 1 | 128 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 48.95 | 32 | 640 | 20 | 10240 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 2 | 128 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 49.80 | 32 | 640 | 20 | 10240 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 3 | 128 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 48.36 | 32 | 640 | 20 | 10240 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 4 | 128 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 49.32 | 32 | 640 | 20 | 10240 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 0 | 128 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 26.32 | 144 | 576 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 128 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 26.13 | 144 | 576 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 128 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 25.85 | 144 | 576 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 128 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 26.73 | 144 | 576 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 128 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 27.14 | 144 | 576 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 128 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 49.17 | 136 | 544 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 128 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 49.06 | 136 | 544 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 128 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 48.88 | 136 | 544 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 128 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 48.68 | 136 | 544 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 128 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 49.32 | 136 | 544 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 128 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 72.09 | 132 | 528 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 128 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 72.51 | 132 | 528 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 128 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 72.19 | 132 | 528 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 128 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 72.18 | 132 | 528 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 128 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 72.64 | 132 | 528 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 128 | 4 | 8 | common_private | q_outer | 16 | 4 | 51.61 | 32 | 640 | 20 | 10240 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 1 | 128 | 4 | 8 | common_private | q_outer | 16 | 4 | 50.01 | 32 | 640 | 20 | 10240 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 2 | 128 | 4 | 8 | common_private | q_outer | 16 | 4 | 50.84 | 32 | 640 | 20 | 10240 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 3 | 128 | 4 | 8 | common_private | q_outer | 16 | 4 | 50.79 | 32 | 640 | 20 | 10240 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 4 | 128 | 4 | 8 | common_private | q_outer | 16 | 4 | 51.93 | 32 | 640 | 20 | 10240 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 0 | 128 | 4 | 8 | common_private | q_outer | 32 | 8 | 129.69 | 16 | 576 | 36 | 18432 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 1 | 128 | 4 | 8 | common_private | q_outer | 32 | 8 | 128.53 | 16 | 576 | 36 | 18432 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 2 | 128 | 4 | 8 | common_private | q_outer | 32 | 8 | 127.93 | 16 | 576 | 36 | 18432 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 3 | 128 | 4 | 8 | common_private | q_outer | 32 | 8 | 130.44 | 16 | 576 | 36 | 18432 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 4 | 128 | 4 | 8 | common_private | q_outer | 32 | 8 | 131.51 | 16 | 576 | 36 | 18432 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 0 | 128 | 4 | 8 | common_private | q_outer | 64 | 16 | 344.63 | 8 | 544 | 68 | 34816 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 1 | 128 | 4 | 8 | common_private | q_outer | 64 | 16 | 342.61 | 8 | 544 | 68 | 34816 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 2 | 128 | 4 | 8 | common_private | q_outer | 64 | 16 | 343.64 | 8 | 544 | 68 | 34816 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 3 | 128 | 4 | 8 | common_private | q_outer | 64 | 16 | 342.46 | 8 | 544 | 68 | 34816 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 4 | 128 | 4 | 8 | common_private | q_outer | 64 | 16 | 344.63 | 8 | 544 | 68 | 34816 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 0 | 128 | 4 | 8 | common_private | q_outer | 128 | 32 | 1579.28 | 4 | 528 | 132 | 67584 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 1 | 128 | 4 | 8 | common_private | q_outer | 128 | 32 | 1570.60 | 4 | 528 | 132 | 67584 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 2 | 128 | 4 | 8 | common_private | q_outer | 128 | 32 | 1576.62 | 4 | 528 | 132 | 67584 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 3 | 128 | 4 | 8 | common_private | q_outer | 128 | 32 | 1574.83 | 4 | 528 | 132 | 67584 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 4 | 128 | 4 | 8 | common_private | q_outer | 128 | 32 | 1588.30 | 4 | 528 | 132 | 67584 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 0 | 128 | 4 | 8 | common_private | signature | 16 | 4 | 21.32 | 160 | 640 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 128 | 4 | 8 | common_private | signature | 16 | 4 | 21.11 | 160 | 640 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 128 | 4 | 8 | common_private | signature | 16 | 4 | 21.48 | 160 | 640 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 128 | 4 | 8 | common_private | signature | 16 | 4 | 21.39 | 160 | 640 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 128 | 4 | 8 | common_private | signature | 16 | 4 | 21.36 | 160 | 640 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 128 | 4 | 8 | common_private | signature | 32 | 8 | 71.25 | 144 | 576 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 128 | 4 | 8 | common_private | signature | 32 | 8 | 26.02 | 144 | 576 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 128 | 4 | 8 | common_private | signature | 32 | 8 | 26.62 | 144 | 576 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 128 | 4 | 8 | common_private | signature | 32 | 8 | 26.89 | 144 | 576 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 128 | 4 | 8 | common_private | signature | 32 | 8 | 26.71 | 144 | 576 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 128 | 4 | 8 | common_private | signature | 64 | 16 | 49.76 | 136 | 544 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 128 | 4 | 8 | common_private | signature | 64 | 16 | 48.86 | 136 | 544 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 128 | 4 | 8 | common_private | signature | 64 | 16 | 50.17 | 136 | 544 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 128 | 4 | 8 | common_private | signature | 64 | 16 | 50.11 | 136 | 544 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 128 | 4 | 8 | common_private | signature | 64 | 16 | 49.53 | 136 | 544 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 128 | 4 | 8 | common_private | signature | 128 | 32 | 72.40 | 132 | 528 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 128 | 4 | 8 | common_private | signature | 128 | 32 | 72.48 | 132 | 528 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 128 | 4 | 8 | common_private | signature | 128 | 32 | 73.10 | 132 | 528 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 128 | 4 | 8 | common_private | signature | 128 | 32 | 72.78 | 132 | 528 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 128 | 4 | 8 | common_private | signature | 128 | 32 | 72.74 | 132 | 528 | 4 | 10240 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 128 | 4 | 8 | random | bounded_merge | 16 | 4 | 70.19 | 90 | 999 | 30 | 15984 | 0 | 440 | 8.0 | 23.0 | 0.574 |
| 1 | 128 | 4 | 8 | random | bounded_merge | 16 | 4 | 68.26 | 84 | 994 | 30 | 15904 | 0 | 448 | 8.0 | 23.0 | 0.518 |
| 2 | 128 | 4 | 8 | random | bounded_merge | 16 | 4 | 71.28 | 104 | 1010 | 30 | 16160 | 0 | 435 | 8.0 | 15.0 | 0.731 |
| 3 | 128 | 4 | 8 | random | bounded_merge | 16 | 4 | 70.24 | 91 | 999 | 30 | 15984 | 0 | 430 | 8.0 | 23.0 | 0.602 |
| 4 | 128 | 4 | 8 | random | bounded_merge | 16 | 4 | 67.34 | 93 | 1002 | 30 | 16032 | 0 | 446 | 8.0 | 23.0 | 0.609 |
| 0 | 128 | 4 | 8 | random | bounded_merge | 32 | 4 | 30.47 | 166 | 967 | 8 | 15472 | 74 | 440 | 7.0 | 8.0 | 0.901 |
| 1 | 128 | 4 | 8 | random | bounded_merge | 32 | 4 | 30.84 | 168 | 963 | 8 | 15408 | 73 | 448 | 7.0 | 8.0 | 0.894 |
| 2 | 128 | 4 | 8 | random | bounded_merge | 32 | 2 | 29.78 | 155 | 981 | 8 | 15696 | 55 | 435 | 7.0 | 8.0 | 0.899 |
| 3 | 128 | 4 | 8 | random | bounded_merge | 32 | 4 | 30.70 | 174 | 960 | 8 | 15360 | 73 | 430 | 7.0 | 8.0 | 0.904 |
| 4 | 128 | 4 | 8 | random | bounded_merge | 32 | 4 | 31.25 | 161 | 972 | 8 | 15552 | 66 | 446 | 7.0 | 8.0 | 0.890 |
| 0 | 128 | 4 | 8 | random | bounded_merge | 64 | 4 | 30.25 | 228 | 900 | 8 | 14400 | 106 | 440 | 5.0 | 7.0 | 0.963 |
| 1 | 128 | 4 | 8 | random | bounded_merge | 64 | 4 | 31.29 | 220 | 915 | 8 | 14640 | 106 | 448 | 5.0 | 7.1 | 0.960 |
| 2 | 128 | 4 | 8 | random | bounded_merge | 64 | 4 | 30.12 | 216 | 918 | 8 | 14688 | 106 | 435 | 5.0 | 7.5 | 0.954 |
| 3 | 128 | 4 | 8 | random | bounded_merge | 64 | 4 | 31.05 | 228 | 905 | 8 | 14480 | 107 | 430 | 5.0 | 7.0 | 0.959 |
| 4 | 128 | 4 | 8 | random | bounded_merge | 64 | 4 | 32.01 | 211 | 923 | 8 | 14768 | 110 | 446 | 6.0 | 7.0 | 0.948 |
| 0 | 128 | 4 | 8 | random | bounded_merge | 128 | 4 | 34.16 | 318 | 799 | 8 | 12784 | 124 | 440 | 1.0 | 6.0 | 1.000 |
| 1 | 128 | 4 | 8 | random | bounded_merge | 128 | 4 | 33.77 | 297 | 822 | 8 | 13152 | 121 | 448 | 1.0 | 6.0 | 1.000 |
| 2 | 128 | 4 | 8 | random | bounded_merge | 128 | 4 | 31.48 | 301 | 815 | 7 | 13040 | 128 | 435 | 1.0 | 6.0 | 1.000 |
| 3 | 128 | 4 | 8 | random | bounded_merge | 128 | 4 | 33.03 | 302 | 806 | 8 | 12896 | 126 | 430 | 1.0 | 6.0 | 1.000 |
| 4 | 128 | 4 | 8 | random | bounded_merge | 128 | 4 | 34.10 | 283 | 838 | 8 | 13408 | 125 | 446 | 1.0 | 6.0 | 1.000 |
| 0 | 128 | 4 | 8 | random | q_outer | 16 | 4 | 74.88 | 32 | 999 | 32 | 15984 | 0 | 440 | 31.0 | 32.0 | 0.000 |
| 1 | 128 | 4 | 8 | random | q_outer | 16 | 4 | 76.51 | 32 | 994 | 32 | 15904 | 0 | 448 | 31.0 | 32.0 | 0.000 |
| 2 | 128 | 4 | 8 | random | q_outer | 16 | 4 | 76.18 | 32 | 1010 | 32 | 16160 | 0 | 435 | 32.0 | 32.0 | 0.000 |
| 3 | 128 | 4 | 8 | random | q_outer | 16 | 4 | 74.81 | 32 | 999 | 32 | 15984 | 0 | 430 | 31.5 | 32.0 | 0.000 |
| 4 | 128 | 4 | 8 | random | q_outer | 16 | 4 | 77.61 | 32 | 1002 | 32 | 16032 | 0 | 446 | 31.0 | 32.0 | 0.000 |
| 0 | 128 | 4 | 8 | random | q_outer | 32 | 8 | 213.95 | 16 | 967 | 64 | 30944 | 0 | 440 | 60.0 | 62.0 | 0.000 |
| 1 | 128 | 4 | 8 | random | q_outer | 32 | 8 | 211.83 | 16 | 963 | 62 | 30816 | 0 | 448 | 60.0 | 62.0 | 0.000 |
| 2 | 128 | 4 | 8 | random | q_outer | 32 | 8 | 214.62 | 16 | 981 | 64 | 31392 | 0 | 435 | 61.0 | 63.0 | 0.000 |
| 3 | 128 | 4 | 8 | random | q_outer | 32 | 8 | 208.79 | 16 | 960 | 64 | 30720 | 0 | 430 | 60.5 | 61.5 | 0.000 |
| 4 | 128 | 4 | 8 | random | q_outer | 32 | 8 | 213.82 | 16 | 972 | 63 | 31104 | 0 | 446 | 61.0 | 62.5 | 0.000 |
| 0 | 128 | 4 | 8 | random | q_outer | 64 | 16 | 554.76 | 8 | 900 | 117 | 57600 | 0 | 440 | 112.0 | 115.6 | 0.000 |
| 1 | 128 | 4 | 8 | random | q_outer | 64 | 16 | 566.28 | 8 | 915 | 117 | 58560 | 0 | 448 | 115.5 | 116.3 | 0.000 |
| 2 | 128 | 4 | 8 | random | q_outer | 64 | 16 | 576.18 | 8 | 918 | 120 | 58752 | 0 | 435 | 115.0 | 117.9 | 0.000 |
| 3 | 128 | 4 | 8 | random | q_outer | 64 | 16 | 559.43 | 8 | 905 | 117 | 57920 | 0 | 430 | 113.5 | 117.0 | 0.000 |
| 4 | 128 | 4 | 8 | random | q_outer | 64 | 16 | 576.92 | 8 | 923 | 119 | 59072 | 0 | 446 | 115.0 | 118.3 | 0.000 |
| 0 | 128 | 4 | 8 | random | q_outer | 128 | 32 | 2342.16 | 4 | 799 | 207 | 102272 | 0 | 440 | 199.0 | 205.5 | 0.000 |
| 1 | 128 | 4 | 8 | random | q_outer | 128 | 32 | 2403.35 | 4 | 822 | 210 | 105216 | 0 | 448 | 206.0 | 209.7 | 0.000 |
| 2 | 128 | 4 | 8 | random | q_outer | 128 | 32 | 2352.17 | 4 | 815 | 205 | 104320 | 0 | 435 | 204.0 | 205.0 | 0.000 |
| 3 | 128 | 4 | 8 | random | q_outer | 128 | 32 | 2355.05 | 4 | 806 | 205 | 103168 | 0 | 430 | 201.0 | 204.4 | 0.000 |
| 4 | 128 | 4 | 8 | random | q_outer | 128 | 32 | 2477.67 | 4 | 838 | 216 | 107264 | 0 | 446 | 209.0 | 214.5 | 0.000 |
| 0 | 128 | 4 | 8 | random | signature | 16 | 4 | 32.58 | 150 | 999 | 8 | 15984 | 42 | 440 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 8 | random | signature | 16 | 4 | 32.82 | 154 | 994 | 8 | 15904 | 47 | 448 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 4 | 8 | random | signature | 16 | 2 | 27.90 | 142 | 1010 | 8 | 16160 | 26 | 435 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 4 | 8 | random | signature | 16 | 4 | 32.33 | 152 | 999 | 8 | 15984 | 40 | 430 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 4 | 8 | random | signature | 16 | 4 | 33.11 | 147 | 1002 | 8 | 16032 | 38 | 446 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 4 | 8 | random | signature | 32 | 4 | 32.91 | 181 | 967 | 8 | 15472 | 82 | 440 | 7.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 8 | random | signature | 32 | 4 | 32.77 | 184 | 963 | 8 | 15408 | 82 | 448 | 7.0 | 8.0 | 1.000 |
| 2 | 128 | 4 | 8 | random | signature | 32 | 2 | 28.80 | 170 | 981 | 8 | 15696 | 65 | 435 | 7.0 | 8.0 | 1.000 |
| 3 | 128 | 4 | 8 | random | signature | 32 | 4 | 31.65 | 189 | 960 | 8 | 15360 | 80 | 430 | 6.0 | 8.0 | 1.000 |
| 4 | 128 | 4 | 8 | random | signature | 32 | 4 | 32.30 | 177 | 972 | 8 | 15552 | 78 | 446 | 7.0 | 8.0 | 1.000 |
| 0 | 128 | 4 | 8 | random | signature | 64 | 4 | 33.36 | 234 | 900 | 8 | 14400 | 107 | 440 | 5.0 | 7.0 | 1.000 |
| 1 | 128 | 4 | 8 | random | signature | 64 | 4 | 32.88 | 226 | 915 | 8 | 14640 | 109 | 448 | 5.0 | 7.0 | 1.000 |
| 2 | 128 | 4 | 8 | random | signature | 64 | 4 | 32.44 | 223 | 918 | 8 | 14688 | 108 | 435 | 5.0 | 7.0 | 1.000 |
| 3 | 128 | 4 | 8 | random | signature | 64 | 4 | 32.75 | 234 | 905 | 8 | 14480 | 109 | 430 | 5.0 | 7.0 | 1.000 |
| 4 | 128 | 4 | 8 | random | signature | 64 | 4 | 33.26 | 219 | 923 | 8 | 14768 | 113 | 446 | 5.0 | 7.0 | 1.000 |
| 0 | 128 | 4 | 8 | random | signature | 128 | 4 | 34.23 | 318 | 799 | 8 | 12784 | 124 | 440 | 1.0 | 6.0 | 1.000 |
| 1 | 128 | 4 | 8 | random | signature | 128 | 4 | 33.33 | 297 | 822 | 8 | 13152 | 121 | 448 | 1.0 | 6.0 | 1.000 |
| 2 | 128 | 4 | 8 | random | signature | 128 | 4 | 30.98 | 301 | 815 | 7 | 13040 | 128 | 435 | 1.0 | 6.0 | 1.000 |
| 3 | 128 | 4 | 8 | random | signature | 128 | 4 | 33.04 | 302 | 806 | 8 | 12896 | 126 | 430 | 1.0 | 6.0 | 1.000 |
| 4 | 128 | 4 | 8 | random | signature | 128 | 4 | 34.25 | 283 | 838 | 8 | 13408 | 125 | 446 | 1.0 | 6.0 | 1.000 |
| 0 | 128 | 4 | 16 | clustered | bounded_merge | 16 | 2 | 67.03 | 121 | 2005 | 28 | 32080 | 0 | 509 | 16.0 | 16.0 | 0.931 |
| 1 | 128 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 109.49 | 113 | 1976 | 46 | 31616 | 0 | 497 | 16.0 | 16.0 | 0.856 |
| 2 | 128 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 93.20 | 116 | 1948 | 38 | 31168 | 0 | 483 | 16.0 | 16.0 | 0.912 |
| 3 | 128 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 89.65 | 121 | 1992 | 38 | 31872 | 0 | 489 | 16.0 | 16.0 | 0.947 |
| 4 | 128 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 116.21 | 115 | 1968 | 50 | 31488 | 0 | 491 | 16.0 | 16.0 | 0.891 |
| 0 | 128 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 53.09 | 135 | 1909 | 16 | 30544 | 24 | 509 | 16.0 | 16.0 | 0.948 |
| 1 | 128 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 52.12 | 136 | 1816 | 16 | 29056 | 35 | 497 | 16.0 | 16.0 | 0.927 |
| 2 | 128 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 51.37 | 134 | 1845 | 16 | 29520 | 27 | 483 | 16.0 | 16.0 | 0.953 |
| 3 | 128 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 52.26 | 136 | 1788 | 16 | 28608 | 32 | 489 | 16.0 | 16.0 | 0.961 |
| 4 | 128 | 4 | 16 | clustered | bounded_merge | 32 | 8 | 79.95 | 137 | 1849 | 16 | 29616 | 29 | 491 | 16.0 | 16.0 | 0.937 |
| 0 | 128 | 4 | 16 | clustered | bounded_merge | 64 | 4 | 70.26 | 163 | 1687 | 27 | 26992 | 68 | 509 | 11.0 | 16.0 | 0.963 |
| 1 | 128 | 4 | 16 | clustered | bounded_merge | 64 | 4 | 77.64 | 165 | 1575 | 27 | 25200 | 80 | 497 | 9.0 | 16.0 | 0.965 |
| 2 | 128 | 4 | 16 | clustered | bounded_merge | 64 | 4 | 68.76 | 161 | 1576 | 27 | 25216 | 73 | 483 | 9.0 | 16.0 | 0.956 |
| 3 | 128 | 4 | 16 | clustered | bounded_merge | 64 | 4 | 78.66 | 160 | 1599 | 27 | 25584 | 72 | 489 | 11.0 | 16.0 | 0.962 |
| 4 | 128 | 4 | 16 | clustered | bounded_merge | 64 | 8 | 81.02 | 162 | 1687 | 16 | 27056 | 68 | 491 | 12.5 | 16.0 | 0.969 |
| 0 | 128 | 4 | 16 | clustered | bounded_merge | 128 | 4 | 58.12 | 194 | 1329 | 16 | 21264 | 103 | 509 | 6.0 | 16.0 | 1.000 |
| 1 | 128 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 76.54 | 198 | 1283 | 16 | 20544 | 112 | 497 | 6.0 | 14.0 | 1.000 |
| 2 | 128 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 74.43 | 189 | 1250 | 16 | 20064 | 113 | 483 | 6.0 | 14.0 | 1.000 |
| 3 | 128 | 4 | 16 | clustered | bounded_merge | 128 | 4 | 58.28 | 192 | 1359 | 16 | 21744 | 108 | 489 | 6.0 | 15.9 | 1.000 |
| 4 | 128 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 72.36 | 195 | 1343 | 16 | 21552 | 104 | 491 | 6.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 16 | clustered | q_outer | 16 | 4 | 150.23 | 32 | 2005 | 64 | 32080 | 0 | 509 | 64.0 | 64.0 | 0.000 |
| 1 | 128 | 4 | 16 | clustered | q_outer | 16 | 4 | 149.74 | 32 | 1976 | 64 | 31616 | 0 | 497 | 64.0 | 64.0 | 0.000 |
| 2 | 128 | 4 | 16 | clustered | q_outer | 16 | 4 | 149.86 | 32 | 1948 | 64 | 31168 | 0 | 483 | 64.0 | 64.0 | 0.000 |
| 3 | 128 | 4 | 16 | clustered | q_outer | 16 | 4 | 150.97 | 32 | 1992 | 64 | 31872 | 0 | 489 | 64.0 | 64.0 | 0.000 |
| 4 | 128 | 4 | 16 | clustered | q_outer | 16 | 4 | 147.39 | 32 | 1968 | 64 | 31488 | 0 | 491 | 64.0 | 64.0 | 0.001 |
| 0 | 128 | 4 | 16 | clustered | q_outer | 32 | 8 | 439.12 | 16 | 1909 | 128 | 61088 | 0 | 509 | 121.0 | 128.0 | 0.000 |
| 1 | 128 | 4 | 16 | clustered | q_outer | 32 | 8 | 421.18 | 16 | 1816 | 128 | 58112 | 0 | 497 | 116.0 | 124.5 | 0.000 |
| 2 | 128 | 4 | 16 | clustered | q_outer | 32 | 8 | 429.64 | 16 | 1845 | 128 | 59040 | 0 | 483 | 114.5 | 128.0 | 0.000 |
| 3 | 128 | 4 | 16 | clustered | q_outer | 32 | 8 | 443.21 | 16 | 1788 | 128 | 57216 | 0 | 489 | 112.0 | 128.0 | 0.000 |
| 4 | 128 | 4 | 16 | clustered | q_outer | 32 | 8 | 433.73 | 16 | 1849 | 128 | 59168 | 0 | 491 | 117.0 | 127.5 | 0.000 |
| 0 | 128 | 4 | 16 | clustered | q_outer | 64 | 16 | 1139.84 | 8 | 1687 | 243 | 107968 | 0 | 509 | 205.5 | 233.9 | 0.000 |
| 1 | 128 | 4 | 16 | clustered | q_outer | 64 | 16 | 1017.80 | 8 | 1575 | 213 | 100800 | 0 | 497 | 197.5 | 204.6 | 0.000 |
| 2 | 128 | 4 | 16 | clustered | q_outer | 64 | 16 | 1069.88 | 8 | 1576 | 229 | 100864 | 0 | 483 | 196.5 | 220.6 | 0.000 |
| 3 | 128 | 4 | 16 | clustered | q_outer | 64 | 16 | 1030.84 | 8 | 1599 | 217 | 102336 | 0 | 489 | 207.0 | 214.2 | 0.000 |
| 4 | 128 | 4 | 16 | clustered | q_outer | 64 | 16 | 1089.57 | 8 | 1687 | 231 | 107968 | 0 | 491 | 219.0 | 227.5 | 0.000 |
| 0 | 128 | 4 | 16 | clustered | q_outer | 128 | 32 | 3836.85 | 4 | 1329 | 343 | 170112 | 0 | 509 | 331.0 | 341.5 | 0.000 |
| 1 | 128 | 4 | 16 | clustered | q_outer | 128 | 32 | 3867.06 | 4 | 1283 | 345 | 164224 | 0 | 497 | 318.5 | 340.5 | 0.000 |
| 2 | 128 | 4 | 16 | clustered | q_outer | 128 | 32 | 3891.69 | 4 | 1250 | 349 | 160000 | 0 | 483 | 312.0 | 342.1 | 0.000 |
| 3 | 128 | 4 | 16 | clustered | q_outer | 128 | 32 | 3866.67 | 4 | 1359 | 348 | 173952 | 0 | 489 | 339.0 | 345.9 | 0.000 |
| 4 | 128 | 4 | 16 | clustered | q_outer | 128 | 32 | 4029.79 | 4 | 1343 | 360 | 171904 | 0 | 491 | 330.0 | 351.6 | 0.000 |
| 0 | 128 | 4 | 16 | clustered | signature | 16 | 2 | 49.31 | 135 | 2005 | 16 | 32080 | 14 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 4 | 16 | clustered | signature | 16 | 2 | 49.21 | 137 | 1976 | 16 | 31616 | 20 | 497 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 4 | 16 | clustered | signature | 16 | 2 | 50.13 | 136 | 1948 | 16 | 31168 | 18 | 483 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 4 | 16 | clustered | signature | 16 | 2 | 48.06 | 134 | 1992 | 16 | 31872 | 12 | 489 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 4 | 16 | clustered | signature | 16 | 4 | 56.38 | 138 | 1968 | 16 | 31488 | 17 | 491 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 16 | clustered | signature | 32 | 4 | 59.61 | 146 | 1909 | 16 | 30544 | 34 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 4 | 16 | clustered | signature | 32 | 4 | 56.70 | 152 | 1816 | 16 | 29056 | 49 | 497 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 4 | 16 | clustered | signature | 32 | 4 | 57.90 | 146 | 1845 | 16 | 29520 | 39 | 483 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 4 | 16 | clustered | signature | 32 | 4 | 57.51 | 148 | 1788 | 16 | 28608 | 43 | 489 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 4 | 16 | clustered | signature | 32 | 8 | 79.40 | 151 | 1849 | 16 | 29616 | 39 | 491 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 16 | clustered | signature | 64 | 4 | 60.03 | 170 | 1687 | 16 | 26992 | 70 | 509 | 10.0 | 16.0 | 1.000 |
| 1 | 128 | 4 | 16 | clustered | signature | 64 | 4 | 59.48 | 171 | 1575 | 16 | 25200 | 82 | 497 | 8.0 | 16.0 | 1.000 |
| 2 | 128 | 4 | 16 | clustered | signature | 64 | 4 | 56.87 | 169 | 1576 | 16 | 25216 | 79 | 483 | 9.0 | 16.0 | 1.000 |
| 3 | 128 | 4 | 16 | clustered | signature | 64 | 4 | 60.01 | 167 | 1599 | 16 | 25584 | 76 | 489 | 10.0 | 16.0 | 1.000 |
| 4 | 128 | 4 | 16 | clustered | signature | 64 | 8 | 79.38 | 169 | 1687 | 16 | 27056 | 73 | 491 | 11.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 16 | clustered | signature | 128 | 4 | 57.80 | 194 | 1329 | 16 | 21264 | 103 | 509 | 6.0 | 16.0 | 1.000 |
| 1 | 128 | 4 | 16 | clustered | signature | 128 | 8 | 74.92 | 198 | 1283 | 16 | 20544 | 112 | 497 | 6.0 | 14.0 | 1.000 |
| 2 | 128 | 4 | 16 | clustered | signature | 128 | 8 | 74.05 | 189 | 1250 | 16 | 20064 | 113 | 483 | 6.0 | 14.0 | 1.000 |
| 3 | 128 | 4 | 16 | clustered | signature | 128 | 4 | 58.22 | 192 | 1359 | 16 | 21744 | 108 | 489 | 6.0 | 15.9 | 1.000 |
| 4 | 128 | 4 | 16 | clustered | signature | 128 | 8 | 72.36 | 195 | 1343 | 16 | 21552 | 104 | 491 | 6.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 93.65 | 32 | 1280 | 40 | 20480 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 1 | 128 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 95.43 | 32 | 1280 | 40 | 20480 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 2 | 128 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 94.80 | 32 | 1280 | 40 | 20480 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 3 | 128 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 95.91 | 32 | 1280 | 40 | 20480 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 4 | 128 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 95.20 | 32 | 1280 | 40 | 20480 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 0 | 128 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 45.12 | 144 | 1152 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 45.99 | 144 | 1152 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 46.04 | 144 | 1152 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 45.89 | 144 | 1152 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 46.33 | 144 | 1152 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 87.08 | 136 | 1088 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 87.36 | 136 | 1088 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 88.20 | 136 | 1088 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 87.57 | 136 | 1088 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 88.58 | 136 | 1088 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 118.97 | 132 | 1056 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 120.07 | 132 | 1056 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 120.00 | 132 | 1056 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 119.84 | 132 | 1056 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 120.05 | 132 | 1056 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 4 | 16 | common_private | q_outer | 16 | 4 | 97.10 | 32 | 1280 | 40 | 20480 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 1 | 128 | 4 | 16 | common_private | q_outer | 16 | 4 | 96.01 | 32 | 1280 | 40 | 20480 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 2 | 128 | 4 | 16 | common_private | q_outer | 16 | 4 | 96.54 | 32 | 1280 | 40 | 20480 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 3 | 128 | 4 | 16 | common_private | q_outer | 16 | 4 | 96.79 | 32 | 1280 | 40 | 20480 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 4 | 128 | 4 | 16 | common_private | q_outer | 16 | 4 | 97.56 | 32 | 1280 | 40 | 20480 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 0 | 128 | 4 | 16 | common_private | q_outer | 32 | 8 | 247.70 | 16 | 1152 | 72 | 36864 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 1 | 128 | 4 | 16 | common_private | q_outer | 32 | 8 | 251.28 | 16 | 1152 | 72 | 36864 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 2 | 128 | 4 | 16 | common_private | q_outer | 32 | 8 | 249.05 | 16 | 1152 | 72 | 36864 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 3 | 128 | 4 | 16 | common_private | q_outer | 32 | 8 | 252.01 | 16 | 1152 | 72 | 36864 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 4 | 128 | 4 | 16 | common_private | q_outer | 32 | 8 | 249.84 | 16 | 1152 | 72 | 36864 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 0 | 128 | 4 | 16 | common_private | q_outer | 64 | 16 | 661.98 | 8 | 1088 | 136 | 69632 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 1 | 128 | 4 | 16 | common_private | q_outer | 64 | 16 | 660.44 | 8 | 1088 | 136 | 69632 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 2 | 128 | 4 | 16 | common_private | q_outer | 64 | 16 | 662.82 | 8 | 1088 | 136 | 69632 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 3 | 128 | 4 | 16 | common_private | q_outer | 64 | 16 | 662.76 | 8 | 1088 | 136 | 69632 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 4 | 128 | 4 | 16 | common_private | q_outer | 64 | 16 | 657.83 | 8 | 1088 | 136 | 69632 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 0 | 128 | 4 | 16 | common_private | q_outer | 128 | 32 | 3040.53 | 4 | 1056 | 264 | 135168 | 0 | 512 | 264.0 | 264.0 | 0.030 |
| 1 | 128 | 4 | 16 | common_private | q_outer | 128 | 32 | 3035.37 | 4 | 1056 | 264 | 135168 | 0 | 512 | 264.0 | 264.0 | 0.030 |
| 2 | 128 | 4 | 16 | common_private | q_outer | 128 | 32 | 3072.93 | 4 | 1056 | 264 | 135168 | 0 | 512 | 264.0 | 264.0 | 0.030 |
| 3 | 128 | 4 | 16 | common_private | q_outer | 128 | 32 | 3039.09 | 4 | 1056 | 264 | 135168 | 0 | 512 | 264.0 | 264.0 | 0.030 |
| 4 | 128 | 4 | 16 | common_private | q_outer | 128 | 32 | 3054.30 | 4 | 1056 | 264 | 135168 | 0 | 512 | 264.0 | 264.0 | 0.030 |
| 0 | 128 | 4 | 16 | common_private | signature | 16 | 4 | 34.85 | 160 | 1280 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 16 | common_private | signature | 16 | 4 | 33.63 | 160 | 1280 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 4 | 16 | common_private | signature | 16 | 4 | 35.26 | 160 | 1280 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 4 | 16 | common_private | signature | 16 | 4 | 35.26 | 160 | 1280 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 4 | 16 | common_private | signature | 16 | 4 | 35.77 | 160 | 1280 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 4 | 16 | common_private | signature | 32 | 8 | 43.42 | 144 | 1152 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 16 | common_private | signature | 32 | 8 | 41.88 | 144 | 1152 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 4 | 16 | common_private | signature | 32 | 8 | 45.61 | 144 | 1152 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 4 | 16 | common_private | signature | 32 | 8 | 43.04 | 144 | 1152 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 4 | 16 | common_private | signature | 32 | 8 | 46.04 | 144 | 1152 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 4 | 16 | common_private | signature | 64 | 16 | 87.09 | 136 | 1088 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 16 | common_private | signature | 64 | 16 | 86.94 | 136 | 1088 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 4 | 16 | common_private | signature | 64 | 16 | 88.15 | 136 | 1088 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 4 | 16 | common_private | signature | 64 | 16 | 87.31 | 136 | 1088 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 4 | 16 | common_private | signature | 64 | 16 | 88.62 | 136 | 1088 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 4 | 16 | common_private | signature | 128 | 32 | 118.78 | 132 | 1056 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 4 | 16 | common_private | signature | 128 | 32 | 120.01 | 132 | 1056 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 4 | 16 | common_private | signature | 128 | 32 | 120.17 | 132 | 1056 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 4 | 16 | common_private | signature | 128 | 32 | 119.62 | 132 | 1056 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 4 | 16 | common_private | signature | 128 | 32 | 120.28 | 132 | 1056 | 8 | 20480 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 4 | 16 | random | bounded_merge | 16 | 4 | 137.85 | 43 | 1946 | 62 | 31136 | 0 | 501 | 58.0 | 62.0 | 0.093 |
| 1 | 128 | 4 | 16 | random | bounded_merge | 16 | 4 | 136.11 | 50 | 1958 | 62 | 31328 | 0 | 503 | 44.5 | 62.0 | 0.181 |
| 2 | 128 | 4 | 16 | random | bounded_merge | 16 | 4 | 137.48 | 39 | 1948 | 62 | 31168 | 0 | 505 | 60.0 | 62.0 | 0.030 |
| 3 | 128 | 4 | 16 | random | bounded_merge | 16 | 4 | 138.48 | 55 | 1973 | 62 | 31568 | 0 | 502 | 31.0 | 62.0 | 0.200 |
| 4 | 128 | 4 | 16 | random | bounded_merge | 16 | 4 | 137.93 | 54 | 1960 | 62 | 31360 | 0 | 505 | 31.0 | 62.0 | 0.202 |
| 0 | 128 | 4 | 16 | random | bounded_merge | 32 | 4 | 53.88 | 300 | 1792 | 16 | 28672 | 124 | 501 | 2.0 | 14.0 | 0.893 |
| 1 | 128 | 4 | 16 | random | bounded_merge | 32 | 4 | 56.25 | 276 | 1837 | 16 | 29392 | 117 | 503 | 2.0 | 14.0 | 0.884 |
| 2 | 128 | 4 | 16 | random | bounded_merge | 32 | 4 | 53.92 | 290 | 1819 | 16 | 29104 | 125 | 505 | 2.0 | 14.0 | 0.893 |
| 3 | 128 | 4 | 16 | random | bounded_merge | 32 | 4 | 53.07 | 271 | 1839 | 16 | 29424 | 121 | 502 | 2.0 | 14.0 | 0.890 |
| 4 | 128 | 4 | 16 | random | bounded_merge | 32 | 4 | 54.46 | 277 | 1849 | 16 | 29584 | 122 | 505 | 2.0 | 14.0 | 0.890 |
| 0 | 128 | 4 | 16 | random | bounded_merge | 64 | 8 | 69.86 | 464 | 1579 | 13 | 25280 | 128 | 501 | 1.0 | 10.0 | 1.000 |
| 1 | 128 | 4 | 16 | random | bounded_merge | 64 | 4 | 55.86 | 451 | 1629 | 14 | 26064 | 128 | 503 | 1.0 | 10.0 | 1.000 |
| 2 | 128 | 4 | 16 | random | bounded_merge | 64 | 8 | 72.39 | 446 | 1630 | 15 | 26096 | 128 | 505 | 1.0 | 11.0 | 1.000 |
| 3 | 128 | 4 | 16 | random | bounded_merge | 64 | 4 | 56.47 | 462 | 1612 | 14 | 25792 | 128 | 502 | 1.0 | 11.0 | 1.000 |
| 4 | 128 | 4 | 16 | random | bounded_merge | 64 | 4 | 55.52 | 443 | 1653 | 15 | 26448 | 128 | 505 | 1.0 | 11.0 | 1.000 |
| 0 | 128 | 4 | 16 | random | bounded_merge | 128 | 8 | 63.92 | 642 | 1277 | 12 | 20576 | 128 | 501 | 1.0 | 6.0 | 1.000 |
| 1 | 128 | 4 | 16 | random | bounded_merge | 128 | 8 | 64.26 | 642 | 1297 | 10 | 20832 | 128 | 503 | 1.0 | 6.0 | 1.000 |
| 2 | 128 | 4 | 16 | random | bounded_merge | 128 | 8 | 63.38 | 639 | 1299 | 11 | 20864 | 128 | 505 | 1.0 | 6.0 | 1.000 |
| 3 | 128 | 4 | 16 | random | bounded_merge | 128 | 8 | 61.42 | 654 | 1279 | 11 | 20608 | 128 | 502 | 1.0 | 6.0 | 1.000 |
| 4 | 128 | 4 | 16 | random | bounded_merge | 128 | 8 | 65.91 | 636 | 1326 | 11 | 21296 | 128 | 505 | 1.0 | 6.0 | 1.000 |
| 0 | 128 | 4 | 16 | random | q_outer | 16 | 4 | 144.69 | 32 | 1946 | 64 | 31136 | 0 | 501 | 61.0 | 62.0 | 0.000 |
| 1 | 128 | 4 | 16 | random | q_outer | 16 | 4 | 146.38 | 32 | 1958 | 64 | 31328 | 0 | 503 | 61.0 | 63.9 | 0.000 |
| 2 | 128 | 4 | 16 | random | q_outer | 16 | 4 | 145.07 | 32 | 1948 | 63 | 31168 | 0 | 505 | 61.0 | 62.0 | 0.000 |
| 3 | 128 | 4 | 16 | random | q_outer | 16 | 4 | 144.71 | 32 | 1973 | 64 | 31568 | 0 | 502 | 62.0 | 63.0 | 0.000 |
| 4 | 128 | 4 | 16 | random | q_outer | 16 | 4 | 149.18 | 32 | 1960 | 64 | 31360 | 0 | 505 | 61.5 | 64.0 | 0.000 |
| 0 | 128 | 4 | 16 | random | q_outer | 32 | 8 | 396.91 | 16 | 1792 | 120 | 57344 | 0 | 501 | 113.0 | 116.0 | 0.000 |
| 1 | 128 | 4 | 16 | random | q_outer | 32 | 8 | 401.79 | 16 | 1837 | 119 | 58784 | 0 | 503 | 114.5 | 118.5 | 0.000 |
| 2 | 128 | 4 | 16 | random | q_outer | 32 | 8 | 403.59 | 16 | 1819 | 120 | 58208 | 0 | 505 | 114.5 | 117.5 | 0.000 |
| 3 | 128 | 4 | 16 | random | q_outer | 32 | 8 | 404.04 | 16 | 1839 | 120 | 58848 | 0 | 502 | 115.5 | 118.5 | 0.000 |
| 4 | 128 | 4 | 16 | random | q_outer | 32 | 8 | 412.18 | 16 | 1849 | 121 | 59168 | 0 | 505 | 115.5 | 119.5 | 0.000 |
| 0 | 128 | 4 | 16 | random | q_outer | 64 | 16 | 949.94 | 8 | 1579 | 203 | 101056 | 0 | 501 | 197.5 | 201.6 | 0.000 |
| 1 | 128 | 4 | 16 | random | q_outer | 64 | 16 | 996.90 | 8 | 1629 | 211 | 104256 | 0 | 503 | 205.0 | 209.6 | 0.000 |
| 2 | 128 | 4 | 16 | random | q_outer | 64 | 16 | 990.70 | 8 | 1630 | 212 | 104320 | 0 | 505 | 204.5 | 210.6 | 0.000 |
| 3 | 128 | 4 | 16 | random | q_outer | 64 | 16 | 978.54 | 8 | 1612 | 209 | 103168 | 0 | 502 | 203.5 | 206.9 | 0.000 |
| 4 | 128 | 4 | 16 | random | q_outer | 64 | 16 | 1008.57 | 8 | 1653 | 214 | 105792 | 0 | 505 | 206.0 | 212.6 | 0.000 |
| 0 | 128 | 4 | 16 | random | q_outer | 128 | 32 | 3627.05 | 4 | 1277 | 323 | 163456 | 0 | 501 | 318.5 | 321.8 | 0.000 |
| 1 | 128 | 4 | 16 | random | q_outer | 128 | 32 | 3757.23 | 4 | 1297 | 337 | 166016 | 0 | 503 | 322.0 | 333.4 | 0.000 |
| 2 | 128 | 4 | 16 | random | q_outer | 128 | 32 | 3764.92 | 4 | 1299 | 337 | 166272 | 0 | 505 | 325.0 | 334.0 | 0.000 |
| 3 | 128 | 4 | 16 | random | q_outer | 128 | 32 | 3763.70 | 4 | 1279 | 337 | 163712 | 0 | 502 | 317.0 | 331.3 | 0.000 |
| 4 | 128 | 4 | 16 | random | q_outer | 128 | 32 | 3772.04 | 4 | 1326 | 335 | 169728 | 0 | 505 | 330.5 | 333.8 | 0.000 |
| 0 | 128 | 4 | 16 | random | signature | 16 | 4 | 55.86 | 205 | 1946 | 16 | 31136 | 101 | 501 | 13.0 | 16.0 | 1.000 |
| 1 | 128 | 4 | 16 | random | signature | 16 | 4 | 56.58 | 198 | 1958 | 16 | 31328 | 98 | 503 | 14.0 | 16.0 | 1.000 |
| 2 | 128 | 4 | 16 | random | signature | 16 | 2 | 48.36 | 211 | 1948 | 16 | 31168 | 113 | 505 | 14.0 | 15.0 | 1.000 |
| 3 | 128 | 4 | 16 | random | signature | 16 | 4 | 57.47 | 188 | 1973 | 16 | 31568 | 91 | 502 | 14.0 | 16.0 | 1.000 |
| 4 | 128 | 4 | 16 | random | signature | 16 | 4 | 57.03 | 200 | 1960 | 16 | 31360 | 94 | 505 | 14.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 16 | random | signature | 32 | 4 | 57.69 | 316 | 1792 | 16 | 28672 | 126 | 501 | 2.0 | 14.0 | 1.000 |
| 1 | 128 | 4 | 16 | random | signature | 32 | 4 | 59.59 | 292 | 1837 | 16 | 29392 | 125 | 503 | 2.0 | 14.0 | 1.000 |
| 2 | 128 | 4 | 16 | random | signature | 32 | 4 | 57.04 | 306 | 1819 | 16 | 29104 | 127 | 505 | 2.0 | 14.0 | 1.000 |
| 3 | 128 | 4 | 16 | random | signature | 32 | 4 | 58.07 | 287 | 1839 | 16 | 29424 | 124 | 502 | 2.0 | 14.0 | 1.000 |
| 4 | 128 | 4 | 16 | random | signature | 32 | 4 | 58.86 | 293 | 1849 | 16 | 29584 | 124 | 505 | 1.0 | 14.0 | 1.000 |
| 0 | 128 | 4 | 16 | random | signature | 64 | 8 | 70.36 | 464 | 1579 | 13 | 25280 | 128 | 501 | 1.0 | 10.0 | 1.000 |
| 1 | 128 | 4 | 16 | random | signature | 64 | 4 | 55.79 | 451 | 1629 | 14 | 26064 | 128 | 503 | 1.0 | 10.0 | 1.000 |
| 2 | 128 | 4 | 16 | random | signature | 64 | 8 | 71.86 | 446 | 1630 | 15 | 26096 | 128 | 505 | 1.0 | 11.0 | 1.000 |
| 3 | 128 | 4 | 16 | random | signature | 64 | 4 | 56.62 | 462 | 1612 | 14 | 25792 | 128 | 502 | 1.0 | 11.0 | 1.000 |
| 4 | 128 | 4 | 16 | random | signature | 64 | 4 | 55.82 | 443 | 1653 | 15 | 26448 | 128 | 505 | 1.0 | 11.0 | 1.000 |
| 0 | 128 | 4 | 16 | random | signature | 128 | 8 | 64.14 | 642 | 1277 | 12 | 20576 | 128 | 501 | 1.0 | 6.0 | 1.000 |
| 1 | 128 | 4 | 16 | random | signature | 128 | 8 | 63.87 | 642 | 1297 | 10 | 20832 | 128 | 503 | 1.0 | 6.0 | 1.000 |
| 2 | 128 | 4 | 16 | random | signature | 128 | 8 | 63.18 | 639 | 1299 | 11 | 20864 | 128 | 505 | 1.0 | 6.0 | 1.000 |
| 3 | 128 | 4 | 16 | random | signature | 128 | 8 | 62.06 | 654 | 1279 | 11 | 20608 | 128 | 502 | 1.0 | 6.0 | 1.000 |
| 4 | 128 | 4 | 16 | random | signature | 128 | 8 | 65.04 | 636 | 1326 | 11 | 21296 | 128 | 505 | 1.0 | 6.0 | 1.000 |
| 0 | 128 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 248.05 | 106 | 3826 | 111 | 61216 | 0 | 509 | 32.0 | 46.0 | 0.800 |
| 1 | 128 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 209.11 | 105 | 3785 | 92 | 60560 | 0 | 499 | 32.0 | 45.0 | 0.802 |
| 2 | 128 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 273.11 | 92 | 3679 | 123 | 58864 | 0 | 490 | 32.0 | 60.7 | 0.658 |
| 3 | 128 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 273.54 | 100 | 3819 | 125 | 61104 | 0 | 499 | 32.0 | 56.0 | 0.722 |
| 4 | 128 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 264.64 | 99 | 3702 | 121 | 59232 | 0 | 498 | 32.0 | 52.2 | 0.748 |
| 0 | 128 | 4 | 32 | clustered | bounded_merge | 32 | 4 | 100.18 | 157 | 3423 | 32 | 54768 | 67 | 509 | 24.0 | 32.0 | 0.937 |
| 1 | 128 | 4 | 32 | clustered | bounded_merge | 32 | 4 | 96.81 | 151 | 3238 | 32 | 51808 | 62 | 499 | 24.0 | 32.0 | 0.946 |
| 2 | 128 | 4 | 32 | clustered | bounded_merge | 32 | 4 | 97.86 | 156 | 3307 | 32 | 52912 | 63 | 490 | 24.0 | 32.0 | 0.926 |
| 3 | 128 | 4 | 32 | clustered | bounded_merge | 32 | 4 | 97.65 | 161 | 3257 | 32 | 52112 | 76 | 499 | 24.0 | 32.0 | 0.927 |
| 4 | 128 | 4 | 32 | clustered | bounded_merge | 32 | 8 | 142.18 | 158 | 3277 | 32 | 52720 | 67 | 498 | 23.0 | 32.0 | 0.926 |
| 0 | 128 | 4 | 32 | clustered | bounded_merge | 64 | 4 | 145.79 | 201 | 2647 | 54 | 42352 | 110 | 509 | 11.0 | 30.0 | 0.944 |
| 1 | 128 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 141.00 | 191 | 2509 | 32 | 40368 | 109 | 499 | 10.0 | 31.0 | 0.967 |
| 2 | 128 | 4 | 32 | clustered | bounded_merge | 64 | 4 | 99.07 | 187 | 2553 | 32 | 40848 | 105 | 490 | 10.0 | 32.0 | 0.970 |
| 3 | 128 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 178.43 | 196 | 2632 | 54 | 42448 | 107 | 499 | 10.5 | 31.5 | 0.959 |
| 4 | 128 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 142.06 | 194 | 2692 | 32 | 43392 | 107 | 498 | 11.5 | 32.0 | 0.967 |
| 0 | 128 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 118.26 | 226 | 1773 | 32 | 29872 | 125 | 509 | 6.0 | 17.0 | 1.000 |
| 1 | 128 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 121.58 | 230 | 1686 | 32 | 28960 | 127 | 499 | 5.0 | 16.1 | 1.000 |
| 2 | 128 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 120.50 | 220 | 1728 | 32 | 29376 | 126 | 490 | 5.5 | 19.0 | 1.000 |
| 3 | 128 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 124.93 | 232 | 1840 | 32 | 30544 | 124 | 499 | 6.0 | 17.9 | 1.000 |
| 4 | 128 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 117.88 | 226 | 1768 | 32 | 30000 | 126 | 498 | 6.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 32 | clustered | q_outer | 16 | 4 | 280.66 | 32 | 3826 | 128 | 61216 | 0 | 509 | 126.0 | 128.0 | 0.000 |
| 1 | 128 | 4 | 32 | clustered | q_outer | 16 | 4 | 286.95 | 32 | 3785 | 128 | 60560 | 0 | 499 | 124.0 | 128.0 | 0.000 |
| 2 | 128 | 4 | 32 | clustered | q_outer | 16 | 4 | 284.45 | 32 | 3679 | 128 | 58864 | 0 | 490 | 117.0 | 128.0 | 0.000 |
| 3 | 128 | 4 | 32 | clustered | q_outer | 16 | 4 | 283.89 | 32 | 3819 | 128 | 61104 | 0 | 499 | 123.0 | 128.0 | 0.000 |
| 4 | 128 | 4 | 32 | clustered | q_outer | 16 | 4 | 286.35 | 32 | 3702 | 128 | 59232 | 0 | 498 | 122.0 | 128.0 | 0.005 |
| 0 | 128 | 4 | 32 | clustered | q_outer | 32 | 8 | 775.60 | 16 | 3423 | 239 | 109536 | 0 | 509 | 218.0 | 234.0 | 0.000 |
| 1 | 128 | 4 | 32 | clustered | q_outer | 32 | 8 | 762.32 | 16 | 3238 | 230 | 103616 | 0 | 499 | 211.5 | 226.5 | 0.000 |
| 2 | 128 | 4 | 32 | clustered | q_outer | 32 | 8 | 803.43 | 16 | 3307 | 256 | 105824 | 0 | 490 | 201.5 | 241.5 | 0.000 |
| 3 | 128 | 4 | 32 | clustered | q_outer | 32 | 8 | 793.05 | 16 | 3257 | 250 | 104224 | 0 | 499 | 204.5 | 241.0 | 0.000 |
| 4 | 128 | 4 | 32 | clustered | q_outer | 32 | 8 | 765.24 | 16 | 3277 | 239 | 104864 | 0 | 498 | 204.0 | 231.0 | 0.000 |
| 0 | 128 | 4 | 32 | clustered | q_outer | 64 | 16 | 1763.42 | 8 | 2647 | 382 | 169408 | 0 | 509 | 327.5 | 373.6 | 0.000 |
| 1 | 128 | 4 | 32 | clustered | q_outer | 64 | 16 | 1564.11 | 8 | 2509 | 337 | 160576 | 0 | 499 | 313.5 | 332.8 | 0.000 |
| 2 | 128 | 4 | 32 | clustered | q_outer | 64 | 16 | 1746.37 | 8 | 2553 | 378 | 163392 | 0 | 490 | 316.5 | 370.3 | 0.000 |
| 3 | 128 | 4 | 32 | clustered | q_outer | 64 | 16 | 1668.06 | 8 | 2632 | 359 | 168448 | 0 | 499 | 339.0 | 357.6 | 0.000 |
| 4 | 128 | 4 | 32 | clustered | q_outer | 64 | 16 | 1783.07 | 8 | 2692 | 383 | 172288 | 0 | 498 | 342.0 | 378.8 | 0.000 |
| 0 | 128 | 4 | 32 | clustered | q_outer | 128 | 32 | 5072.16 | 4 | 1773 | 457 | 226944 | 0 | 509 | 446.0 | 456.4 | 0.000 |
| 1 | 128 | 4 | 32 | clustered | q_outer | 128 | 32 | 4949.64 | 4 | 1686 | 447 | 215808 | 0 | 499 | 421.5 | 442.5 | 0.000 |
| 2 | 128 | 4 | 32 | clustered | q_outer | 128 | 32 | 5180.15 | 4 | 1728 | 468 | 221184 | 0 | 490 | 443.0 | 464.7 | 0.000 |
| 3 | 128 | 4 | 32 | clustered | q_outer | 128 | 32 | 5261.40 | 4 | 1840 | 476 | 235520 | 0 | 499 | 460.5 | 475.7 | 0.000 |
| 4 | 128 | 4 | 32 | clustered | q_outer | 128 | 32 | 5226.36 | 4 | 1768 | 473 | 226304 | 0 | 498 | 437.0 | 466.4 | 0.000 |
| 0 | 128 | 4 | 32 | clustered | signature | 16 | 4 | 108.69 | 147 | 3826 | 32 | 61216 | 36 | 509 | 32.0 | 32.0 | 1.000 |
| 1 | 128 | 4 | 32 | clustered | signature | 16 | 2 | 87.98 | 144 | 3785 | 32 | 60560 | 34 | 499 | 32.0 | 32.0 | 1.000 |
| 2 | 128 | 4 | 32 | clustered | signature | 16 | 2 | 90.86 | 155 | 3679 | 32 | 58864 | 54 | 490 | 30.0 | 32.0 | 1.000 |
| 3 | 128 | 4 | 32 | clustered | signature | 16 | 2 | 88.04 | 148 | 3819 | 32 | 61104 | 41 | 499 | 32.0 | 32.0 | 1.000 |
| 4 | 128 | 4 | 32 | clustered | signature | 16 | 4 | 105.92 | 153 | 3702 | 32 | 59232 | 44 | 498 | 32.0 | 32.0 | 1.000 |
| 0 | 128 | 4 | 32 | clustered | signature | 32 | 4 | 108.51 | 173 | 3423 | 32 | 54768 | 78 | 509 | 21.0 | 32.0 | 1.000 |
| 1 | 128 | 4 | 32 | clustered | signature | 32 | 4 | 106.98 | 167 | 3238 | 32 | 51808 | 75 | 499 | 18.0 | 32.0 | 1.000 |
| 2 | 128 | 4 | 32 | clustered | signature | 32 | 4 | 104.15 | 171 | 3307 | 32 | 52912 | 76 | 490 | 20.0 | 32.0 | 1.000 |
| 3 | 128 | 4 | 32 | clustered | signature | 32 | 4 | 105.86 | 177 | 3257 | 32 | 52112 | 82 | 499 | 21.0 | 32.0 | 1.000 |
| 4 | 128 | 4 | 32 | clustered | signature | 32 | 8 | 139.95 | 174 | 3277 | 32 | 52720 | 76 | 498 | 18.5 | 32.0 | 1.000 |
| 0 | 128 | 4 | 32 | clustered | signature | 64 | 4 | 107.74 | 208 | 2647 | 32 | 42352 | 112 | 509 | 11.0 | 28.0 | 1.000 |
| 1 | 128 | 4 | 32 | clustered | signature | 64 | 8 | 138.29 | 199 | 2509 | 32 | 40368 | 112 | 499 | 10.0 | 27.2 | 1.000 |
| 2 | 128 | 4 | 32 | clustered | signature | 64 | 4 | 98.83 | 195 | 2553 | 32 | 40848 | 109 | 490 | 9.0 | 31.0 | 1.000 |
| 3 | 128 | 4 | 32 | clustered | signature | 64 | 8 | 140.58 | 204 | 2632 | 32 | 42448 | 108 | 499 | 10.0 | 31.0 | 1.000 |
| 4 | 128 | 4 | 32 | clustered | signature | 64 | 8 | 138.15 | 202 | 2692 | 32 | 43392 | 108 | 498 | 11.0 | 31.0 | 1.000 |
| 0 | 128 | 4 | 32 | clustered | signature | 128 | 8 | 117.64 | 226 | 1773 | 32 | 29872 | 125 | 509 | 6.0 | 17.0 | 1.000 |
| 1 | 128 | 4 | 32 | clustered | signature | 128 | 8 | 121.55 | 230 | 1686 | 32 | 28960 | 127 | 499 | 5.0 | 16.1 | 1.000 |
| 2 | 128 | 4 | 32 | clustered | signature | 128 | 8 | 120.73 | 220 | 1728 | 32 | 29376 | 126 | 490 | 5.5 | 19.0 | 1.000 |
| 3 | 128 | 4 | 32 | clustered | signature | 128 | 8 | 125.59 | 232 | 1840 | 32 | 30544 | 124 | 499 | 6.0 | 17.9 | 1.000 |
| 4 | 128 | 4 | 32 | clustered | signature | 128 | 8 | 118.15 | 226 | 1768 | 32 | 30000 | 126 | 498 | 6.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 178.51 | 32 | 2560 | 80 | 40960 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 1 | 128 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 178.46 | 32 | 2560 | 80 | 40960 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 2 | 128 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 178.00 | 32 | 2560 | 80 | 40960 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 3 | 128 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 179.75 | 32 | 2560 | 80 | 40960 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 4 | 128 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 179.23 | 32 | 2560 | 80 | 40960 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 0 | 128 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 73.16 | 144 | 2304 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 73.32 | 144 | 2304 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 73.60 | 144 | 2304 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 73.26 | 144 | 2304 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 74.77 | 144 | 2304 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 161.63 | 136 | 2176 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 162.47 | 136 | 2176 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 162.05 | 136 | 2176 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 162.70 | 136 | 2176 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 162.56 | 136 | 2176 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 214.15 | 128 | 2048 | 16 | 39936 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 214.45 | 128 | 2048 | 16 | 39936 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 214.10 | 128 | 2048 | 16 | 39936 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 214.53 | 128 | 2048 | 16 | 39936 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 214.82 | 128 | 2048 | 16 | 39936 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 32 | common_private | q_outer | 16 | 4 | 184.71 | 32 | 2560 | 80 | 40960 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 1 | 128 | 4 | 32 | common_private | q_outer | 16 | 4 | 184.46 | 32 | 2560 | 80 | 40960 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 2 | 128 | 4 | 32 | common_private | q_outer | 16 | 4 | 184.84 | 32 | 2560 | 80 | 40960 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 3 | 128 | 4 | 32 | common_private | q_outer | 16 | 4 | 186.42 | 32 | 2560 | 80 | 40960 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 4 | 128 | 4 | 32 | common_private | q_outer | 16 | 4 | 186.10 | 32 | 2560 | 80 | 40960 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 0 | 128 | 4 | 32 | common_private | q_outer | 32 | 8 | 484.92 | 16 | 2304 | 144 | 73728 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 1 | 128 | 4 | 32 | common_private | q_outer | 32 | 8 | 489.62 | 16 | 2304 | 144 | 73728 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 2 | 128 | 4 | 32 | common_private | q_outer | 32 | 8 | 488.02 | 16 | 2304 | 144 | 73728 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 3 | 128 | 4 | 32 | common_private | q_outer | 32 | 8 | 486.93 | 16 | 2304 | 144 | 73728 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 4 | 128 | 4 | 32 | common_private | q_outer | 32 | 8 | 489.22 | 16 | 2304 | 144 | 73728 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 0 | 128 | 4 | 32 | common_private | q_outer | 64 | 16 | 1284.60 | 8 | 2176 | 272 | 139264 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 1 | 128 | 4 | 32 | common_private | q_outer | 64 | 16 | 1291.52 | 8 | 2176 | 272 | 139264 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 2 | 128 | 4 | 32 | common_private | q_outer | 64 | 16 | 1286.88 | 8 | 2176 | 272 | 139264 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 3 | 128 | 4 | 32 | common_private | q_outer | 64 | 16 | 1286.32 | 8 | 2176 | 272 | 139264 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 4 | 128 | 4 | 32 | common_private | q_outer | 64 | 16 | 1301.76 | 8 | 2176 | 272 | 139264 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 0 | 128 | 4 | 32 | common_private | q_outer | 128 | 32 | 5770.12 | 4 | 2048 | 512 | 262144 | 0 | 512 | 512.0 | 512.0 | 0.031 |
| 1 | 128 | 4 | 32 | common_private | q_outer | 128 | 32 | 5795.97 | 4 | 2048 | 512 | 262144 | 0 | 512 | 512.0 | 512.0 | 0.031 |
| 2 | 128 | 4 | 32 | common_private | q_outer | 128 | 32 | 5789.26 | 4 | 2048 | 512 | 262144 | 0 | 512 | 512.0 | 512.0 | 0.031 |
| 3 | 128 | 4 | 32 | common_private | q_outer | 128 | 32 | 5790.38 | 4 | 2048 | 512 | 262144 | 0 | 512 | 512.0 | 512.0 | 0.031 |
| 4 | 128 | 4 | 32 | common_private | q_outer | 128 | 32 | 5803.17 | 4 | 2048 | 512 | 262144 | 0 | 512 | 512.0 | 512.0 | 0.031 |
| 0 | 128 | 4 | 32 | common_private | signature | 16 | 4 | 56.52 | 160 | 2560 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 4 | 32 | common_private | signature | 16 | 4 | 57.11 | 160 | 2560 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 4 | 32 | common_private | signature | 16 | 4 | 56.95 | 160 | 2560 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 4 | 32 | common_private | signature | 16 | 4 | 57.47 | 160 | 2560 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 4 | 32 | common_private | signature | 16 | 4 | 56.90 | 160 | 2560 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 32 | common_private | signature | 32 | 8 | 72.67 | 144 | 2304 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 4 | 32 | common_private | signature | 32 | 8 | 73.20 | 144 | 2304 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 4 | 32 | common_private | signature | 32 | 8 | 73.05 | 144 | 2304 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 4 | 32 | common_private | signature | 32 | 8 | 73.23 | 144 | 2304 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 4 | 32 | common_private | signature | 32 | 8 | 73.52 | 144 | 2304 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 32 | common_private | signature | 64 | 16 | 161.77 | 136 | 2176 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 4 | 32 | common_private | signature | 64 | 16 | 162.26 | 136 | 2176 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 4 | 32 | common_private | signature | 64 | 16 | 162.02 | 136 | 2176 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 4 | 32 | common_private | signature | 64 | 16 | 162.73 | 136 | 2176 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 4 | 32 | common_private | signature | 64 | 16 | 162.58 | 136 | 2176 | 16 | 40960 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 32 | common_private | signature | 128 | 32 | 213.66 | 128 | 2048 | 16 | 39936 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 4 | 32 | common_private | signature | 128 | 32 | 214.90 | 128 | 2048 | 16 | 39936 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 4 | 32 | common_private | signature | 128 | 32 | 214.62 | 128 | 2048 | 16 | 39936 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 4 | 32 | common_private | signature | 128 | 32 | 214.39 | 128 | 2048 | 16 | 39936 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 4 | 32 | common_private | signature | 128 | 32 | 215.15 | 128 | 2048 | 16 | 39936 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 4 | 32 | random | bounded_merge | 16 | 4 | 253.71 | 32 | 3691 | 122 | 59056 | 0 | 512 | 116.0 | 119.9 | 0.000 |
| 1 | 128 | 4 | 32 | random | bounded_merge | 16 | 4 | 262.97 | 32 | 3718 | 121 | 59488 | 0 | 512 | 116.0 | 119.0 | 0.000 |
| 2 | 128 | 4 | 32 | random | bounded_merge | 16 | 4 | 259.62 | 32 | 3733 | 122 | 59728 | 0 | 512 | 117.0 | 120.8 | 0.000 |
| 3 | 128 | 4 | 32 | random | bounded_merge | 16 | 4 | 258.34 | 32 | 3725 | 123 | 59600 | 0 | 512 | 116.5 | 120.9 | 0.000 |
| 4 | 128 | 4 | 32 | random | bounded_merge | 16 | 4 | 259.22 | 32 | 3753 | 122 | 60048 | 0 | 512 | 117.0 | 120.9 | 0.000 |
| 0 | 128 | 4 | 32 | random | bounded_merge | 32 | 8 | 127.50 | 557 | 3288 | 26 | 52624 | 128 | 512 | 2.0 | 21.0 | 0.973 |
| 1 | 128 | 4 | 32 | random | bounded_merge | 32 | 4 | 89.40 | 551 | 3315 | 27 | 53040 | 128 | 512 | 2.0 | 21.0 | 0.973 |
| 2 | 128 | 4 | 32 | random | bounded_merge | 32 | 4 | 90.27 | 534 | 3323 | 28 | 53168 | 128 | 512 | 2.0 | 21.0 | 0.969 |
| 3 | 128 | 4 | 32 | random | bounded_merge | 32 | 4 | 91.12 | 553 | 3312 | 27 | 52992 | 128 | 512 | 2.0 | 21.0 | 0.970 |
| 4 | 128 | 4 | 32 | random | bounded_merge | 32 | 4 | 93.14 | 539 | 3325 | 28 | 53200 | 128 | 512 | 2.0 | 21.0 | 0.973 |
| 0 | 128 | 4 | 32 | random | bounded_merge | 64 | 8 | 107.36 | 973 | 2611 | 19 | 41936 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 128 | 4 | 32 | random | bounded_merge | 64 | 8 | 112.97 | 984 | 2644 | 21 | 42384 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 128 | 4 | 32 | random | bounded_merge | 64 | 8 | 113.14 | 945 | 2668 | 19 | 42912 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 128 | 4 | 32 | random | bounded_merge | 64 | 8 | 116.18 | 962 | 2639 | 21 | 42384 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 128 | 4 | 32 | random | bounded_merge | 64 | 8 | 106.92 | 978 | 2634 | 19 | 42288 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 128 | 4 | 32 | random | bounded_merge | 128 | 8 | 92.12 | 1289 | 1782 | 10 | 30016 | 128 | 512 | 1.0 | 2.0 | 1.000 |
| 1 | 128 | 4 | 32 | random | bounded_merge | 128 | 8 | 93.90 | 1286 | 1788 | 11 | 30048 | 128 | 512 | 1.0 | 2.0 | 1.000 |
| 2 | 128 | 4 | 32 | random | bounded_merge | 128 | 8 | 92.48 | 1281 | 1805 | 11 | 30320 | 128 | 512 | 1.0 | 2.0 | 1.000 |
| 3 | 128 | 4 | 32 | random | bounded_merge | 128 | 16 | 143.38 | 1281 | 1769 | 9 | 29936 | 128 | 512 | 1.0 | 2.0 | 1.000 |
| 4 | 128 | 4 | 32 | random | bounded_merge | 128 | 8 | 90.04 | 1309 | 1786 | 10 | 30000 | 128 | 512 | 1.0 | 2.0 | 1.000 |
| 0 | 128 | 4 | 32 | random | q_outer | 16 | 4 | 262.96 | 32 | 3691 | 122 | 59056 | 0 | 512 | 116.0 | 119.9 | 0.000 |
| 1 | 128 | 4 | 32 | random | q_outer | 16 | 4 | 270.81 | 32 | 3718 | 121 | 59488 | 0 | 512 | 116.0 | 119.0 | 0.000 |
| 2 | 128 | 4 | 32 | random | q_outer | 16 | 4 | 266.56 | 32 | 3733 | 122 | 59728 | 0 | 512 | 117.0 | 120.8 | 0.000 |
| 3 | 128 | 4 | 32 | random | q_outer | 16 | 4 | 266.52 | 32 | 3725 | 123 | 59600 | 0 | 512 | 116.5 | 120.9 | 0.000 |
| 4 | 128 | 4 | 32 | random | q_outer | 16 | 4 | 267.43 | 32 | 3753 | 122 | 60048 | 0 | 512 | 117.0 | 120.9 | 0.000 |
| 0 | 128 | 4 | 32 | random | q_outer | 32 | 8 | 689.22 | 16 | 3288 | 213 | 105216 | 0 | 512 | 205.0 | 211.5 | 0.000 |
| 1 | 128 | 4 | 32 | random | q_outer | 32 | 8 | 702.78 | 16 | 3315 | 217 | 106080 | 0 | 512 | 208.0 | 211.5 | 0.000 |
| 2 | 128 | 4 | 32 | random | q_outer | 32 | 8 | 701.46 | 16 | 3323 | 214 | 106336 | 0 | 512 | 207.0 | 212.0 | 0.000 |
| 3 | 128 | 4 | 32 | random | q_outer | 32 | 8 | 708.92 | 16 | 3312 | 216 | 105984 | 0 | 512 | 207.5 | 213.0 | 0.000 |
| 4 | 128 | 4 | 32 | random | q_outer | 32 | 8 | 718.05 | 16 | 3325 | 220 | 106400 | 0 | 512 | 207.5 | 212.0 | 0.000 |
| 0 | 128 | 4 | 32 | random | q_outer | 64 | 16 | 1549.50 | 8 | 2611 | 336 | 167104 | 0 | 512 | 325.5 | 334.6 | 0.000 |
| 1 | 128 | 4 | 32 | random | q_outer | 64 | 16 | 1599.51 | 8 | 2644 | 346 | 169216 | 0 | 512 | 330.0 | 338.3 | 0.000 |
| 2 | 128 | 4 | 32 | random | q_outer | 64 | 16 | 1605.26 | 8 | 2668 | 347 | 170752 | 0 | 512 | 332.5 | 342.1 | 0.000 |
| 3 | 128 | 4 | 32 | random | q_outer | 64 | 16 | 1566.50 | 8 | 2639 | 338 | 168896 | 0 | 512 | 331.5 | 334.5 | 0.000 |
| 4 | 128 | 4 | 32 | random | q_outer | 64 | 16 | 1578.80 | 8 | 2634 | 338 | 168576 | 0 | 512 | 328.0 | 337.3 | 0.000 |
| 0 | 128 | 4 | 32 | random | q_outer | 128 | 32 | 5010.77 | 4 | 1782 | 454 | 228096 | 0 | 512 | 446.0 | 452.2 | 0.000 |
| 1 | 128 | 4 | 32 | random | q_outer | 128 | 32 | 5118.42 | 4 | 1788 | 463 | 228864 | 0 | 512 | 443.5 | 457.3 | 0.000 |
| 2 | 128 | 4 | 32 | random | q_outer | 128 | 32 | 5149.08 | 4 | 1805 | 458 | 231040 | 0 | 512 | 452.0 | 456.2 | 0.000 |
| 3 | 128 | 4 | 32 | random | q_outer | 128 | 32 | 5029.31 | 4 | 1769 | 454 | 226432 | 0 | 512 | 440.0 | 451.0 | 0.000 |
| 4 | 128 | 4 | 32 | random | q_outer | 128 | 32 | 5022.10 | 4 | 1786 | 451 | 228608 | 0 | 512 | 447.0 | 450.1 | 0.000 |
| 0 | 128 | 4 | 32 | random | signature | 16 | 4 | 96.69 | 321 | 3691 | 30 | 59056 | 128 | 512 | 3.0 | 27.0 | 1.000 |
| 1 | 128 | 4 | 32 | random | signature | 16 | 4 | 99.24 | 308 | 3718 | 30 | 59488 | 128 | 512 | 3.0 | 28.0 | 1.000 |
| 2 | 128 | 4 | 32 | random | signature | 16 | 4 | 99.28 | 302 | 3733 | 31 | 59728 | 128 | 512 | 3.0 | 28.0 | 1.000 |
| 3 | 128 | 4 | 32 | random | signature | 16 | 4 | 97.96 | 305 | 3725 | 32 | 59600 | 127 | 512 | 3.0 | 28.0 | 1.000 |
| 4 | 128 | 4 | 32 | random | signature | 16 | 4 | 98.73 | 299 | 3753 | 31 | 60048 | 128 | 512 | 3.0 | 28.0 | 1.000 |
| 0 | 128 | 4 | 32 | random | signature | 32 | 8 | 123.10 | 561 | 3288 | 26 | 52624 | 128 | 512 | 2.0 | 21.0 | 1.000 |
| 1 | 128 | 4 | 32 | random | signature | 32 | 4 | 94.74 | 555 | 3315 | 26 | 53040 | 128 | 512 | 2.0 | 21.0 | 1.000 |
| 2 | 128 | 4 | 32 | random | signature | 32 | 4 | 93.41 | 539 | 3323 | 26 | 53168 | 128 | 512 | 2.0 | 21.0 | 1.000 |
| 3 | 128 | 4 | 32 | random | signature | 32 | 4 | 96.23 | 557 | 3312 | 26 | 52992 | 128 | 512 | 2.0 | 21.0 | 1.000 |
| 4 | 128 | 4 | 32 | random | signature | 32 | 4 | 95.84 | 543 | 3325 | 28 | 53200 | 128 | 512 | 2.0 | 21.0 | 1.000 |
| 0 | 128 | 4 | 32 | random | signature | 64 | 8 | 108.09 | 973 | 2611 | 19 | 41936 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 128 | 4 | 32 | random | signature | 64 | 8 | 113.99 | 984 | 2644 | 21 | 42384 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 128 | 4 | 32 | random | signature | 64 | 8 | 112.13 | 945 | 2668 | 19 | 42912 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 128 | 4 | 32 | random | signature | 64 | 8 | 118.29 | 962 | 2639 | 21 | 42384 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 128 | 4 | 32 | random | signature | 64 | 8 | 107.10 | 978 | 2634 | 19 | 42288 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 128 | 4 | 32 | random | signature | 128 | 8 | 91.64 | 1289 | 1782 | 10 | 30016 | 128 | 512 | 1.0 | 2.0 | 1.000 |
| 1 | 128 | 4 | 32 | random | signature | 128 | 8 | 93.74 | 1286 | 1788 | 11 | 30048 | 128 | 512 | 1.0 | 2.0 | 1.000 |
| 2 | 128 | 4 | 32 | random | signature | 128 | 8 | 91.67 | 1281 | 1805 | 11 | 30320 | 128 | 512 | 1.0 | 2.0 | 1.000 |
| 3 | 128 | 4 | 32 | random | signature | 128 | 16 | 142.19 | 1281 | 1769 | 9 | 29936 | 128 | 512 | 1.0 | 2.0 | 1.000 |
| 4 | 128 | 4 | 32 | random | signature | 128 | 8 | 89.98 | 1309 | 1786 | 10 | 30000 | 128 | 512 | 1.0 | 2.0 | 1.000 |
| 0 | 128 | 8 | 8 | clustered | bounded_merge | 16 | 1 | 24.60 | 128 | 1024 | 8 | 16384 | 0 | 450 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 43.80 | 127 | 1023 | 15 | 16368 | 0 | 437 | 8.0 | 8.0 | 0.986 |
| 2 | 128 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 43.29 | 125 | 1018 | 15 | 16288 | 0 | 436 | 8.0 | 8.0 | 0.965 |
| 3 | 128 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 40.06 | 126 | 1015 | 14 | 16240 | 0 | 445 | 8.0 | 8.0 | 0.986 |
| 4 | 128 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 33.57 | 127 | 1019 | 11 | 16304 | 0 | 438 | 8.0 | 8.0 | 0.994 |
| 0 | 128 | 8 | 8 | clustered | bounded_merge | 32 | 1 | 24.90 | 128 | 1024 | 8 | 16384 | 0 | 450 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 8 | clustered | bounded_merge | 32 | 2 | 42.69 | 125 | 1010 | 15 | 16160 | 0 | 437 | 8.0 | 8.0 | 0.980 |
| 2 | 128 | 8 | 8 | clustered | bounded_merge | 32 | 2 | 42.60 | 121 | 1000 | 15 | 16000 | 0 | 436 | 8.0 | 8.0 | 0.936 |
| 3 | 128 | 8 | 8 | clustered | bounded_merge | 32 | 2 | 41.63 | 125 | 1013 | 14 | 16208 | 0 | 445 | 8.0 | 8.0 | 0.974 |
| 4 | 128 | 8 | 8 | clustered | bounded_merge | 32 | 4 | 44.11 | 126 | 1001 | 11 | 16096 | 3 | 438 | 8.0 | 8.0 | 0.985 |
| 0 | 128 | 8 | 8 | clustered | bounded_merge | 64 | 2 | 30.23 | 129 | 1000 | 8 | 16000 | 7 | 450 | 8.0 | 8.0 | 0.977 |
| 1 | 128 | 8 | 8 | clustered | bounded_merge | 64 | 2 | 30.66 | 128 | 962 | 8 | 15392 | 13 | 437 | 8.0 | 8.0 | 0.965 |
| 2 | 128 | 8 | 8 | clustered | bounded_merge | 64 | 2 | 29.46 | 131 | 973 | 8 | 15568 | 17 | 436 | 8.0 | 8.0 | 0.967 |
| 3 | 128 | 8 | 8 | clustered | bounded_merge | 64 | 2 | 30.13 | 131 | 941 | 8 | 15056 | 19 | 445 | 8.0 | 8.0 | 0.966 |
| 4 | 128 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 37.03 | 131 | 976 | 8 | 15728 | 12 | 438 | 8.0 | 8.0 | 0.973 |
| 0 | 128 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 42.21 | 145 | 931 | 8 | 14944 | 36 | 450 | 8.0 | 8.0 | 0.969 |
| 1 | 128 | 8 | 8 | clustered | bounded_merge | 128 | 2 | 33.95 | 140 | 892 | 8 | 14272 | 37 | 437 | 8.0 | 8.0 | 0.974 |
| 2 | 128 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 40.97 | 147 | 908 | 8 | 14768 | 46 | 436 | 8.0 | 8.0 | 0.977 |
| 3 | 128 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 41.79 | 148 | 882 | 8 | 14176 | 46 | 445 | 8.0 | 8.0 | 0.976 |
| 4 | 128 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 43.11 | 141 | 931 | 8 | 15008 | 30 | 438 | 8.0 | 8.0 | 0.984 |
| 0 | 128 | 8 | 8 | clustered | q_outer | 16 | 2 | 45.13 | 64 | 1024 | 16 | 16384 | 0 | 450 | 16.0 | 16.0 | 0.000 |
| 1 | 128 | 8 | 8 | clustered | q_outer | 16 | 2 | 44.51 | 64 | 1023 | 16 | 16368 | 0 | 437 | 16.0 | 16.0 | 0.001 |
| 2 | 128 | 8 | 8 | clustered | q_outer | 16 | 2 | 44.68 | 64 | 1018 | 16 | 16288 | 0 | 436 | 16.0 | 16.0 | 0.006 |
| 3 | 128 | 8 | 8 | clustered | q_outer | 16 | 2 | 45.64 | 64 | 1015 | 16 | 16240 | 0 | 445 | 16.0 | 16.0 | 0.009 |
| 4 | 128 | 8 | 8 | clustered | q_outer | 16 | 2 | 44.62 | 64 | 1019 | 16 | 16304 | 0 | 438 | 16.0 | 16.0 | 0.005 |
| 0 | 128 | 8 | 8 | clustered | q_outer | 32 | 4 | 106.81 | 32 | 1024 | 32 | 32768 | 0 | 450 | 32.0 | 32.0 | 0.000 |
| 1 | 128 | 8 | 8 | clustered | q_outer | 32 | 4 | 108.58 | 32 | 1010 | 32 | 32320 | 0 | 437 | 32.0 | 32.0 | 0.000 |
| 2 | 128 | 8 | 8 | clustered | q_outer | 32 | 4 | 106.64 | 32 | 1000 | 32 | 32000 | 0 | 436 | 32.0 | 32.0 | 0.000 |
| 3 | 128 | 8 | 8 | clustered | q_outer | 32 | 4 | 107.28 | 32 | 1013 | 32 | 32416 | 0 | 445 | 32.0 | 32.0 | 0.000 |
| 4 | 128 | 8 | 8 | clustered | q_outer | 32 | 4 | 107.27 | 32 | 1001 | 32 | 32032 | 0 | 438 | 32.0 | 32.0 | 0.000 |
| 0 | 128 | 8 | 8 | clustered | q_outer | 64 | 8 | 302.17 | 16 | 1000 | 64 | 64000 | 0 | 450 | 64.0 | 64.0 | 0.000 |
| 1 | 128 | 8 | 8 | clustered | q_outer | 64 | 8 | 296.96 | 16 | 962 | 64 | 61568 | 0 | 437 | 62.0 | 64.0 | 0.000 |
| 2 | 128 | 8 | 8 | clustered | q_outer | 64 | 8 | 298.35 | 16 | 973 | 64 | 62272 | 0 | 436 | 62.0 | 64.0 | 0.000 |
| 3 | 128 | 8 | 8 | clustered | q_outer | 64 | 8 | 298.44 | 16 | 941 | 64 | 60224 | 0 | 445 | 59.0 | 64.0 | 0.000 |
| 4 | 128 | 8 | 8 | clustered | q_outer | 64 | 8 | 297.43 | 16 | 976 | 64 | 62464 | 0 | 438 | 64.0 | 64.0 | 0.000 |
| 0 | 128 | 8 | 8 | clustered | q_outer | 128 | 16 | 1587.80 | 8 | 931 | 125 | 119168 | 0 | 450 | 115.0 | 122.9 | 0.000 |
| 1 | 128 | 8 | 8 | clustered | q_outer | 128 | 16 | 1470.60 | 8 | 892 | 116 | 114176 | 0 | 437 | 113.0 | 115.3 | 0.000 |
| 2 | 128 | 8 | 8 | clustered | q_outer | 128 | 16 | 1532.04 | 8 | 908 | 121 | 116224 | 0 | 436 | 115.0 | 120.3 | 0.000 |
| 3 | 128 | 8 | 8 | clustered | q_outer | 128 | 16 | 1516.47 | 8 | 882 | 119 | 112896 | 0 | 445 | 111.5 | 117.6 | 0.000 |
| 4 | 128 | 8 | 8 | clustered | q_outer | 128 | 16 | 1605.46 | 8 | 931 | 127 | 119168 | 0 | 438 | 117.5 | 123.5 | 0.000 |
| 0 | 128 | 8 | 8 | clustered | signature | 16 | 1 | 24.75 | 128 | 1024 | 8 | 16384 | 0 | 450 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 8 | clustered | signature | 16 | 2 | 30.43 | 129 | 1023 | 8 | 16368 | 2 | 437 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 8 | 8 | clustered | signature | 16 | 2 | 29.58 | 131 | 1018 | 8 | 16288 | 6 | 436 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 8 | 8 | clustered | signature | 16 | 2 | 32.43 | 130 | 1015 | 8 | 16240 | 4 | 445 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 8 | 8 | clustered | signature | 16 | 2 | 30.33 | 129 | 1019 | 8 | 16304 | 2 | 438 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 8 | 8 | clustered | signature | 32 | 1 | 24.84 | 128 | 1024 | 8 | 16384 | 0 | 450 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 8 | clustered | signature | 32 | 2 | 30.99 | 129 | 1010 | 8 | 16160 | 4 | 437 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 8 | 8 | clustered | signature | 32 | 2 | 32.09 | 133 | 1000 | 8 | 16000 | 12 | 436 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 8 | 8 | clustered | signature | 32 | 2 | 30.98 | 131 | 1013 | 8 | 16208 | 6 | 445 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 8 | 8 | clustered | signature | 32 | 4 | 35.03 | 132 | 1001 | 8 | 16096 | 7 | 438 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 8 | 8 | clustered | signature | 64 | 2 | 30.80 | 134 | 1000 | 8 | 16000 | 12 | 450 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 8 | clustered | signature | 64 | 2 | 31.75 | 136 | 962 | 8 | 15392 | 21 | 437 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 8 | 8 | clustered | signature | 64 | 2 | 31.74 | 138 | 973 | 8 | 15568 | 23 | 436 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 8 | 8 | clustered | signature | 64 | 2 | 31.49 | 142 | 941 | 8 | 15056 | 29 | 445 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 8 | 8 | clustered | signature | 64 | 4 | 40.52 | 138 | 976 | 8 | 15728 | 18 | 438 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 8 | 8 | clustered | signature | 128 | 4 | 42.70 | 150 | 931 | 8 | 14944 | 41 | 450 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 8 | clustered | signature | 128 | 2 | 32.96 | 146 | 892 | 8 | 14272 | 43 | 437 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 8 | 8 | clustered | signature | 128 | 4 | 41.13 | 152 | 908 | 8 | 14768 | 51 | 436 | 7.0 | 8.0 | 1.000 |
| 3 | 128 | 8 | 8 | clustered | signature | 128 | 4 | 40.78 | 154 | 882 | 8 | 14176 | 51 | 445 | 7.0 | 8.0 | 1.000 |
| 4 | 128 | 8 | 8 | clustered | signature | 128 | 4 | 41.59 | 146 | 931 | 8 | 15008 | 35 | 438 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 34.43 | 64 | 768 | 12 | 12288 | 0 | 512 | 12.0 | 12.0 | 0.333 |
| 1 | 128 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 34.95 | 64 | 768 | 12 | 12288 | 0 | 512 | 12.0 | 12.0 | 0.333 |
| 2 | 128 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 34.15 | 64 | 768 | 12 | 12288 | 0 | 512 | 12.0 | 12.0 | 0.333 |
| 3 | 128 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 34.24 | 64 | 768 | 12 | 12288 | 0 | 512 | 12.0 | 12.0 | 0.333 |
| 4 | 128 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 34.64 | 64 | 768 | 12 | 12288 | 0 | 512 | 12.0 | 12.0 | 0.333 |
| 0 | 128 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 25.56 | 160 | 640 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 128 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 26.10 | 160 | 640 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 128 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 25.84 | 160 | 640 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 128 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 26.41 | 160 | 640 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 128 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 26.10 | 160 | 640 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 128 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 45.73 | 144 | 576 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 128 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 45.72 | 144 | 576 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 128 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 46.16 | 144 | 576 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 128 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 45.85 | 144 | 576 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 128 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 46.87 | 144 | 576 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 128 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 119.91 | 136 | 544 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 128 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 120.04 | 136 | 544 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 128 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 120.05 | 136 | 544 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 128 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 119.63 | 136 | 544 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 128 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 120.54 | 136 | 544 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 128 | 8 | 8 | common_private | q_outer | 16 | 2 | 34.30 | 64 | 768 | 12 | 12288 | 0 | 512 | 12.0 | 12.0 | 0.333 |
| 1 | 128 | 8 | 8 | common_private | q_outer | 16 | 2 | 35.87 | 64 | 768 | 12 | 12288 | 0 | 512 | 12.0 | 12.0 | 0.333 |
| 2 | 128 | 8 | 8 | common_private | q_outer | 16 | 2 | 34.67 | 64 | 768 | 12 | 12288 | 0 | 512 | 12.0 | 12.0 | 0.333 |
| 3 | 128 | 8 | 8 | common_private | q_outer | 16 | 2 | 34.70 | 64 | 768 | 12 | 12288 | 0 | 512 | 12.0 | 12.0 | 0.333 |
| 4 | 128 | 8 | 8 | common_private | q_outer | 16 | 2 | 34.86 | 64 | 768 | 12 | 12288 | 0 | 512 | 12.0 | 12.0 | 0.333 |
| 0 | 128 | 8 | 8 | common_private | q_outer | 32 | 4 | 68.66 | 32 | 640 | 20 | 20480 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 1 | 128 | 8 | 8 | common_private | q_outer | 32 | 4 | 70.07 | 32 | 640 | 20 | 20480 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 2 | 128 | 8 | 8 | common_private | q_outer | 32 | 4 | 69.61 | 32 | 640 | 20 | 20480 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 3 | 128 | 8 | 8 | common_private | q_outer | 32 | 4 | 69.64 | 32 | 640 | 20 | 20480 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 4 | 128 | 8 | 8 | common_private | q_outer | 32 | 4 | 70.25 | 32 | 640 | 20 | 20480 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 0 | 128 | 8 | 8 | common_private | q_outer | 64 | 8 | 173.48 | 16 | 576 | 36 | 36864 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 1 | 128 | 8 | 8 | common_private | q_outer | 64 | 8 | 174.91 | 16 | 576 | 36 | 36864 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 2 | 128 | 8 | 8 | common_private | q_outer | 64 | 8 | 174.84 | 16 | 576 | 36 | 36864 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 3 | 128 | 8 | 8 | common_private | q_outer | 64 | 8 | 175.80 | 16 | 576 | 36 | 36864 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 4 | 128 | 8 | 8 | common_private | q_outer | 64 | 8 | 175.48 | 16 | 576 | 36 | 36864 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 0 | 128 | 8 | 8 | common_private | q_outer | 128 | 16 | 892.60 | 8 | 544 | 68 | 69632 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 1 | 128 | 8 | 8 | common_private | q_outer | 128 | 16 | 897.56 | 8 | 544 | 68 | 69632 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 2 | 128 | 8 | 8 | common_private | q_outer | 128 | 16 | 899.36 | 8 | 544 | 68 | 69632 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 3 | 128 | 8 | 8 | common_private | q_outer | 128 | 16 | 899.45 | 8 | 544 | 68 | 69632 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 4 | 128 | 8 | 8 | common_private | q_outer | 128 | 16 | 893.71 | 8 | 544 | 68 | 69632 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 0 | 128 | 8 | 8 | common_private | signature | 16 | 2 | 21.35 | 192 | 768 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 128 | 8 | 8 | common_private | signature | 16 | 2 | 21.46 | 192 | 768 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 128 | 8 | 8 | common_private | signature | 16 | 2 | 21.19 | 192 | 768 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 128 | 8 | 8 | common_private | signature | 16 | 2 | 21.53 | 192 | 768 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 128 | 8 | 8 | common_private | signature | 16 | 2 | 22.01 | 192 | 768 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 128 | 8 | 8 | common_private | signature | 32 | 4 | 26.30 | 160 | 640 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 128 | 8 | 8 | common_private | signature | 32 | 4 | 26.11 | 160 | 640 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 128 | 8 | 8 | common_private | signature | 32 | 4 | 26.20 | 160 | 640 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 128 | 8 | 8 | common_private | signature | 32 | 4 | 26.55 | 160 | 640 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 128 | 8 | 8 | common_private | signature | 32 | 4 | 26.86 | 160 | 640 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 128 | 8 | 8 | common_private | signature | 64 | 8 | 45.90 | 144 | 576 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 128 | 8 | 8 | common_private | signature | 64 | 8 | 45.87 | 144 | 576 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 128 | 8 | 8 | common_private | signature | 64 | 8 | 46.29 | 144 | 576 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 128 | 8 | 8 | common_private | signature | 64 | 8 | 46.69 | 144 | 576 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 128 | 8 | 8 | common_private | signature | 64 | 8 | 46.78 | 144 | 576 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 128 | 8 | 8 | common_private | signature | 128 | 16 | 119.86 | 136 | 544 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 128 | 8 | 8 | common_private | signature | 128 | 16 | 120.18 | 136 | 544 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 128 | 8 | 8 | common_private | signature | 128 | 16 | 119.75 | 136 | 544 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 128 | 8 | 8 | common_private | signature | 128 | 16 | 121.22 | 136 | 544 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 128 | 8 | 8 | common_private | signature | 128 | 16 | 120.68 | 136 | 544 | 4 | 12288 | 128 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 128 | 8 | 8 | random | bounded_merge | 16 | 2 | 43.58 | 117 | 1011 | 15 | 16176 | 0 | 440 | 8.0 | 8.0 | 0.852 |
| 1 | 128 | 8 | 8 | random | bounded_merge | 16 | 2 | 44.37 | 115 | 1009 | 15 | 16144 | 0 | 448 | 8.0 | 14.0 | 0.824 |
| 2 | 128 | 8 | 8 | random | bounded_merge | 16 | 2 | 43.27 | 123 | 1019 | 15 | 16304 | 0 | 435 | 8.0 | 8.0 | 0.931 |
| 3 | 128 | 8 | 8 | random | bounded_merge | 16 | 2 | 43.84 | 119 | 1015 | 15 | 16240 | 0 | 430 | 8.0 | 8.0 | 0.876 |
| 4 | 128 | 8 | 8 | random | bounded_merge | 16 | 2 | 43.01 | 121 | 1016 | 15 | 16256 | 0 | 446 | 8.0 | 8.0 | 0.906 |
| 0 | 128 | 8 | 8 | random | bounded_merge | 32 | 4 | 56.18 | 111 | 999 | 15 | 16000 | 6 | 440 | 8.0 | 15.0 | 0.734 |
| 1 | 128 | 8 | 8 | random | bounded_merge | 32 | 4 | 56.28 | 112 | 994 | 15 | 15920 | 11 | 448 | 8.0 | 15.0 | 0.716 |
| 2 | 128 | 8 | 8 | random | bounded_merge | 32 | 2 | 46.73 | 116 | 1010 | 15 | 16160 | 2 | 435 | 8.0 | 11.0 | 0.822 |
| 3 | 128 | 8 | 8 | random | bounded_merge | 32 | 4 | 57.08 | 120 | 999 | 15 | 16000 | 14 | 430 | 8.0 | 8.6 | 0.782 |
| 4 | 128 | 8 | 8 | random | bounded_merge | 32 | 4 | 56.81 | 113 | 1002 | 15 | 16064 | 5 | 446 | 8.0 | 15.0 | 0.769 |
| 0 | 128 | 8 | 8 | random | bounded_merge | 64 | 4 | 42.28 | 166 | 967 | 8 | 15504 | 74 | 440 | 7.0 | 8.0 | 0.901 |
| 1 | 128 | 8 | 8 | random | bounded_merge | 64 | 4 | 42.57 | 168 | 963 | 8 | 15440 | 73 | 448 | 7.0 | 8.0 | 0.894 |
| 2 | 128 | 8 | 8 | random | bounded_merge | 64 | 2 | 32.55 | 155 | 981 | 8 | 15696 | 55 | 435 | 7.0 | 8.0 | 0.899 |
| 3 | 128 | 8 | 8 | random | bounded_merge | 64 | 4 | 42.54 | 174 | 960 | 8 | 15408 | 74 | 430 | 7.0 | 8.0 | 0.905 |
| 4 | 128 | 8 | 8 | random | bounded_merge | 64 | 4 | 42.61 | 161 | 972 | 8 | 15584 | 66 | 446 | 7.0 | 8.0 | 0.890 |
| 0 | 128 | 8 | 8 | random | bounded_merge | 128 | 4 | 42.27 | 229 | 900 | 8 | 14592 | 107 | 440 | 5.0 | 7.0 | 0.971 |
| 1 | 128 | 8 | 8 | random | bounded_merge | 128 | 4 | 40.74 | 220 | 915 | 8 | 14752 | 106 | 448 | 5.0 | 7.1 | 0.960 |
| 2 | 128 | 8 | 8 | random | bounded_merge | 128 | 4 | 42.30 | 217 | 918 | 8 | 14768 | 106 | 435 | 5.0 | 7.4 | 0.961 |
| 3 | 128 | 8 | 8 | random | bounded_merge | 128 | 4 | 42.22 | 228 | 905 | 8 | 14640 | 107 | 430 | 5.0 | 7.0 | 0.959 |
| 4 | 128 | 8 | 8 | random | bounded_merge | 128 | 4 | 42.30 | 211 | 923 | 8 | 14864 | 110 | 446 | 6.0 | 7.0 | 0.948 |
| 0 | 128 | 8 | 8 | random | q_outer | 16 | 2 | 44.37 | 64 | 1011 | 16 | 16176 | 0 | 440 | 16.0 | 16.0 | 0.013 |
| 1 | 128 | 8 | 8 | random | q_outer | 16 | 2 | 44.05 | 64 | 1009 | 16 | 16144 | 0 | 448 | 16.0 | 16.0 | 0.015 |
| 2 | 128 | 8 | 8 | random | q_outer | 16 | 2 | 45.32 | 64 | 1019 | 16 | 16304 | 0 | 435 | 16.0 | 16.0 | 0.005 |
| 3 | 128 | 8 | 8 | random | q_outer | 16 | 2 | 44.02 | 64 | 1015 | 16 | 16240 | 0 | 430 | 16.0 | 16.0 | 0.009 |
| 4 | 128 | 8 | 8 | random | q_outer | 16 | 2 | 45.98 | 64 | 1016 | 16 | 16256 | 0 | 446 | 16.0 | 16.0 | 0.008 |
| 0 | 128 | 8 | 8 | random | q_outer | 32 | 4 | 106.30 | 32 | 999 | 32 | 31968 | 0 | 440 | 31.0 | 32.0 | 0.000 |
| 1 | 128 | 8 | 8 | random | q_outer | 32 | 4 | 106.72 | 32 | 994 | 32 | 31808 | 0 | 448 | 31.0 | 32.0 | 0.000 |
| 2 | 128 | 8 | 8 | random | q_outer | 32 | 4 | 107.48 | 32 | 1010 | 32 | 32320 | 0 | 435 | 32.0 | 32.0 | 0.000 |
| 3 | 128 | 8 | 8 | random | q_outer | 32 | 4 | 106.12 | 32 | 999 | 32 | 31968 | 0 | 430 | 31.5 | 32.0 | 0.000 |
| 4 | 128 | 8 | 8 | random | q_outer | 32 | 4 | 108.64 | 32 | 1002 | 32 | 32064 | 0 | 446 | 31.0 | 32.0 | 0.000 |
| 0 | 128 | 8 | 8 | random | q_outer | 64 | 8 | 293.46 | 16 | 967 | 64 | 61888 | 0 | 440 | 60.0 | 62.0 | 0.000 |
| 1 | 128 | 8 | 8 | random | q_outer | 64 | 8 | 287.09 | 16 | 963 | 62 | 61632 | 0 | 448 | 60.0 | 62.0 | 0.000 |
| 2 | 128 | 8 | 8 | random | q_outer | 64 | 8 | 299.01 | 16 | 981 | 64 | 62784 | 0 | 435 | 61.0 | 63.0 | 0.000 |
| 3 | 128 | 8 | 8 | random | q_outer | 64 | 8 | 292.01 | 16 | 960 | 64 | 61440 | 0 | 430 | 60.5 | 61.5 | 0.000 |
| 4 | 128 | 8 | 8 | random | q_outer | 64 | 8 | 297.09 | 16 | 972 | 63 | 62208 | 0 | 446 | 61.0 | 62.5 | 0.000 |
| 0 | 128 | 8 | 8 | random | q_outer | 128 | 16 | 1484.99 | 8 | 900 | 117 | 115200 | 0 | 440 | 112.0 | 115.6 | 0.000 |
| 1 | 128 | 8 | 8 | random | q_outer | 128 | 16 | 1488.09 | 8 | 915 | 117 | 117120 | 0 | 448 | 115.5 | 116.3 | 0.000 |
| 2 | 128 | 8 | 8 | random | q_outer | 128 | 16 | 1526.56 | 8 | 918 | 120 | 117504 | 0 | 435 | 115.0 | 117.9 | 0.000 |
| 3 | 128 | 8 | 8 | random | q_outer | 128 | 16 | 1480.87 | 8 | 905 | 117 | 115840 | 0 | 430 | 113.5 | 117.0 | 0.000 |
| 4 | 128 | 8 | 8 | random | q_outer | 128 | 16 | 1510.20 | 8 | 923 | 119 | 118144 | 0 | 446 | 115.0 | 118.3 | 0.000 |
| 0 | 128 | 8 | 8 | random | signature | 16 | 2 | 31.66 | 139 | 1011 | 8 | 16176 | 22 | 440 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 8 | random | signature | 16 | 2 | 31.67 | 141 | 1009 | 8 | 16144 | 26 | 448 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 8 | 8 | random | signature | 16 | 2 | 31.68 | 133 | 1019 | 8 | 16304 | 10 | 435 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 8 | 8 | random | signature | 16 | 2 | 31.80 | 137 | 1015 | 8 | 16240 | 18 | 430 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 8 | 8 | random | signature | 16 | 2 | 31.90 | 135 | 1016 | 8 | 16256 | 14 | 446 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 8 | 8 | random | signature | 32 | 4 | 40.72 | 150 | 999 | 8 | 16000 | 42 | 440 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 8 | random | signature | 32 | 4 | 41.34 | 154 | 994 | 8 | 15920 | 47 | 448 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 8 | 8 | random | signature | 32 | 2 | 32.89 | 142 | 1010 | 8 | 16160 | 26 | 435 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 8 | 8 | random | signature | 32 | 4 | 41.01 | 152 | 999 | 8 | 16000 | 40 | 430 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 8 | 8 | random | signature | 32 | 4 | 42.21 | 147 | 1002 | 8 | 16064 | 38 | 446 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 8 | 8 | random | signature | 64 | 4 | 40.95 | 181 | 967 | 8 | 15504 | 82 | 440 | 7.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 8 | random | signature | 64 | 4 | 41.37 | 184 | 963 | 8 | 15440 | 82 | 448 | 7.0 | 8.0 | 1.000 |
| 2 | 128 | 8 | 8 | random | signature | 64 | 2 | 32.31 | 170 | 981 | 8 | 15696 | 65 | 435 | 7.0 | 8.0 | 1.000 |
| 3 | 128 | 8 | 8 | random | signature | 64 | 4 | 40.77 | 189 | 960 | 8 | 15408 | 80 | 430 | 6.0 | 8.0 | 1.000 |
| 4 | 128 | 8 | 8 | random | signature | 64 | 4 | 41.72 | 177 | 972 | 8 | 15584 | 78 | 446 | 7.0 | 8.0 | 1.000 |
| 0 | 128 | 8 | 8 | random | signature | 128 | 4 | 41.85 | 234 | 900 | 8 | 14592 | 107 | 440 | 5.0 | 7.0 | 1.000 |
| 1 | 128 | 8 | 8 | random | signature | 128 | 4 | 40.90 | 226 | 915 | 8 | 14752 | 109 | 448 | 5.0 | 7.0 | 1.000 |
| 2 | 128 | 8 | 8 | random | signature | 128 | 4 | 40.68 | 223 | 918 | 8 | 14768 | 108 | 435 | 5.0 | 7.0 | 1.000 |
| 3 | 128 | 8 | 8 | random | signature | 128 | 4 | 41.13 | 234 | 905 | 8 | 14640 | 109 | 430 | 5.0 | 7.0 | 1.000 |
| 4 | 128 | 8 | 8 | random | signature | 128 | 4 | 42.49 | 219 | 923 | 8 | 14864 | 113 | 446 | 5.0 | 7.0 | 1.000 |
| 0 | 128 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 70.56 | 127 | 2042 | 26 | 32672 | 0 | 509 | 16.0 | 16.0 | 0.990 |
| 1 | 128 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 80.48 | 125 | 2030 | 30 | 32480 | 0 | 497 | 16.0 | 16.0 | 0.970 |
| 2 | 128 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 66.06 | 124 | 2009 | 24 | 32144 | 0 | 483 | 16.0 | 16.0 | 0.975 |
| 3 | 128 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 69.82 | 125 | 2017 | 27 | 32272 | 0 | 489 | 16.0 | 16.0 | 0.983 |
| 4 | 128 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 71.46 | 124 | 2020 | 28 | 32320 | 0 | 491 | 16.0 | 16.0 | 0.964 |
| 0 | 128 | 8 | 16 | clustered | bounded_merge | 32 | 2 | 73.67 | 121 | 2005 | 28 | 32080 | 0 | 509 | 16.0 | 16.0 | 0.931 |
| 1 | 128 | 8 | 16 | clustered | bounded_merge | 32 | 2 | 80.05 | 117 | 1976 | 31 | 31616 | 0 | 497 | 16.0 | 16.0 | 0.895 |
| 2 | 128 | 8 | 16 | clustered | bounded_merge | 32 | 2 | 70.95 | 118 | 1948 | 28 | 31168 | 0 | 483 | 16.0 | 16.0 | 0.938 |
| 3 | 128 | 8 | 16 | clustered | bounded_merge | 32 | 2 | 70.42 | 122 | 1992 | 27 | 31872 | 0 | 489 | 16.0 | 16.0 | 0.960 |
| 4 | 128 | 8 | 16 | clustered | bounded_merge | 32 | 4 | 101.18 | 123 | 1968 | 30 | 31728 | 5 | 491 | 16.0 | 16.0 | 0.930 |
| 0 | 128 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 73.96 | 135 | 1909 | 16 | 30720 | 24 | 509 | 16.0 | 16.0 | 0.948 |
| 1 | 128 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 73.83 | 136 | 1816 | 16 | 29296 | 35 | 497 | 16.0 | 16.0 | 0.924 |
| 2 | 128 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 72.02 | 135 | 1845 | 16 | 29696 | 29 | 483 | 16.0 | 16.0 | 0.960 |
| 3 | 128 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 74.90 | 136 | 1788 | 16 | 28768 | 32 | 489 | 16.0 | 16.0 | 0.961 |
| 4 | 128 | 8 | 16 | clustered | bounded_merge | 64 | 8 | 92.05 | 137 | 1849 | 16 | 30000 | 29 | 491 | 16.0 | 16.0 | 0.934 |
| 0 | 128 | 8 | 16 | clustered | bounded_merge | 128 | 4 | 74.85 | 164 | 1687 | 16 | 27792 | 68 | 509 | 11.0 | 16.0 | 0.979 |
| 1 | 128 | 8 | 16 | clustered | bounded_merge | 128 | 4 | 100.66 | 165 | 1575 | 27 | 25872 | 79 | 497 | 9.0 | 16.0 | 0.963 |
| 2 | 128 | 8 | 16 | clustered | bounded_merge | 128 | 4 | 72.78 | 163 | 1576 | 16 | 26192 | 74 | 483 | 9.0 | 16.0 | 0.981 |
| 3 | 128 | 8 | 16 | clustered | bounded_merge | 128 | 4 | 106.68 | 160 | 1599 | 27 | 26496 | 72 | 489 | 11.0 | 16.0 | 0.962 |
| 4 | 128 | 8 | 16 | clustered | bounded_merge | 128 | 8 | 105.15 | 162 | 1687 | 16 | 27600 | 68 | 491 | 12.5 | 16.0 | 0.968 |
| 0 | 128 | 8 | 16 | clustered | q_outer | 16 | 2 | 85.81 | 64 | 2042 | 32 | 32672 | 0 | 509 | 32.0 | 32.0 | 0.003 |
| 1 | 128 | 8 | 16 | clustered | q_outer | 16 | 2 | 83.83 | 64 | 2030 | 32 | 32480 | 0 | 497 | 32.0 | 32.0 | 0.009 |
| 2 | 128 | 8 | 16 | clustered | q_outer | 16 | 2 | 85.83 | 64 | 2009 | 32 | 32144 | 0 | 483 | 32.0 | 32.0 | 0.019 |
| 3 | 128 | 8 | 16 | clustered | q_outer | 16 | 2 | 86.23 | 64 | 2017 | 32 | 32272 | 0 | 489 | 32.0 | 32.0 | 0.015 |
| 4 | 128 | 8 | 16 | clustered | q_outer | 16 | 2 | 83.52 | 64 | 2020 | 32 | 32320 | 0 | 491 | 32.0 | 32.0 | 0.014 |
| 0 | 128 | 8 | 16 | clustered | q_outer | 32 | 4 | 206.48 | 32 | 2005 | 64 | 64160 | 0 | 509 | 64.0 | 64.0 | 0.000 |
| 1 | 128 | 8 | 16 | clustered | q_outer | 32 | 4 | 209.63 | 32 | 1976 | 64 | 63232 | 0 | 497 | 64.0 | 64.0 | 0.000 |
| 2 | 128 | 8 | 16 | clustered | q_outer | 32 | 4 | 212.16 | 32 | 1948 | 64 | 62336 | 0 | 483 | 64.0 | 64.0 | 0.000 |
| 3 | 128 | 8 | 16 | clustered | q_outer | 32 | 4 | 211.70 | 32 | 1992 | 64 | 63744 | 0 | 489 | 64.0 | 64.0 | 0.000 |
| 4 | 128 | 8 | 16 | clustered | q_outer | 32 | 4 | 210.00 | 32 | 1968 | 64 | 62976 | 0 | 491 | 64.0 | 64.0 | 0.001 |
| 0 | 128 | 8 | 16 | clustered | q_outer | 64 | 8 | 584.65 | 16 | 1909 | 128 | 122176 | 0 | 509 | 121.0 | 128.0 | 0.000 |
| 1 | 128 | 8 | 16 | clustered | q_outer | 64 | 8 | 582.24 | 16 | 1816 | 128 | 116224 | 0 | 497 | 116.0 | 124.5 | 0.000 |
| 2 | 128 | 8 | 16 | clustered | q_outer | 64 | 8 | 597.05 | 16 | 1845 | 128 | 118080 | 0 | 483 | 114.5 | 128.0 | 0.000 |
| 3 | 128 | 8 | 16 | clustered | q_outer | 64 | 8 | 590.16 | 16 | 1788 | 128 | 114432 | 0 | 489 | 112.0 | 128.0 | 0.000 |
| 4 | 128 | 8 | 16 | clustered | q_outer | 64 | 8 | 587.08 | 16 | 1849 | 128 | 118336 | 0 | 491 | 117.0 | 127.5 | 0.000 |
| 0 | 128 | 8 | 16 | clustered | q_outer | 128 | 16 | 3021.86 | 8 | 1687 | 243 | 215936 | 0 | 509 | 205.5 | 233.9 | 0.000 |
| 1 | 128 | 8 | 16 | clustered | q_outer | 128 | 16 | 2657.15 | 8 | 1575 | 213 | 201600 | 0 | 497 | 197.5 | 204.6 | 0.000 |
| 2 | 128 | 8 | 16 | clustered | q_outer | 128 | 16 | 2853.02 | 8 | 1576 | 229 | 201728 | 0 | 483 | 196.5 | 220.6 | 0.000 |
| 3 | 128 | 8 | 16 | clustered | q_outer | 128 | 16 | 2729.98 | 8 | 1599 | 217 | 204672 | 0 | 489 | 207.0 | 214.2 | 0.000 |
| 4 | 128 | 8 | 16 | clustered | q_outer | 128 | 16 | 2857.59 | 8 | 1687 | 231 | 215936 | 0 | 491 | 219.0 | 227.5 | 0.000 |
| 0 | 128 | 8 | 16 | clustered | signature | 16 | 2 | 51.70 | 129 | 2042 | 16 | 32672 | 2 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 8 | 16 | clustered | signature | 16 | 2 | 51.88 | 131 | 2030 | 16 | 32480 | 6 | 497 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 8 | 16 | clustered | signature | 16 | 2 | 52.45 | 132 | 2009 | 16 | 32144 | 8 | 483 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 8 | 16 | clustered | signature | 16 | 2 | 51.56 | 131 | 2017 | 16 | 32272 | 6 | 489 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 8 | 16 | clustered | signature | 16 | 2 | 51.89 | 132 | 2020 | 16 | 32320 | 8 | 491 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 8 | 16 | clustered | signature | 32 | 2 | 54.03 | 135 | 2005 | 16 | 32080 | 14 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 8 | 16 | clustered | signature | 32 | 2 | 55.69 | 137 | 1976 | 16 | 31616 | 20 | 497 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 8 | 16 | clustered | signature | 32 | 2 | 57.19 | 136 | 1948 | 16 | 31168 | 18 | 483 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 8 | 16 | clustered | signature | 32 | 2 | 55.07 | 134 | 1992 | 16 | 31872 | 12 | 489 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 8 | 16 | clustered | signature | 32 | 4 | 70.84 | 138 | 1968 | 16 | 31728 | 17 | 491 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 8 | 16 | clustered | signature | 64 | 4 | 74.09 | 146 | 1909 | 16 | 30720 | 34 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 8 | 16 | clustered | signature | 64 | 4 | 71.18 | 152 | 1816 | 16 | 29296 | 49 | 497 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 8 | 16 | clustered | signature | 64 | 4 | 71.49 | 146 | 1845 | 16 | 29696 | 39 | 483 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 8 | 16 | clustered | signature | 64 | 4 | 72.58 | 148 | 1788 | 16 | 28768 | 43 | 489 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 8 | 16 | clustered | signature | 64 | 8 | 96.52 | 151 | 1849 | 16 | 30000 | 39 | 491 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 8 | 16 | clustered | signature | 128 | 4 | 74.40 | 170 | 1687 | 16 | 27792 | 70 | 509 | 10.0 | 16.0 | 1.000 |
| 1 | 128 | 8 | 16 | clustered | signature | 128 | 4 | 71.97 | 171 | 1575 | 16 | 25872 | 82 | 497 | 8.0 | 16.0 | 1.000 |
| 2 | 128 | 8 | 16 | clustered | signature | 128 | 4 | 70.61 | 169 | 1576 | 16 | 26192 | 79 | 483 | 9.0 | 16.0 | 1.000 |
| 3 | 128 | 8 | 16 | clustered | signature | 128 | 4 | 73.37 | 167 | 1599 | 16 | 26496 | 76 | 489 | 10.0 | 16.0 | 1.000 |
| 4 | 128 | 8 | 16 | clustered | signature | 128 | 8 | 94.00 | 169 | 1687 | 16 | 27600 | 73 | 491 | 11.0 | 16.0 | 1.000 |
| 0 | 128 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 62.51 | 64 | 1536 | 24 | 24576 | 0 | 512 | 24.0 | 24.0 | 0.333 |
| 1 | 128 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 63.69 | 64 | 1536 | 24 | 24576 | 0 | 512 | 24.0 | 24.0 | 0.333 |
| 2 | 128 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 63.49 | 64 | 1536 | 24 | 24576 | 0 | 512 | 24.0 | 24.0 | 0.333 |
| 3 | 128 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 64.50 | 64 | 1536 | 24 | 24576 | 0 | 512 | 24.0 | 24.0 | 0.333 |
| 4 | 128 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 64.12 | 64 | 1536 | 24 | 24576 | 0 | 512 | 24.0 | 24.0 | 0.333 |
| 0 | 128 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 41.63 | 160 | 1280 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 44.83 | 160 | 1280 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 43.60 | 160 | 1280 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 42.01 | 160 | 1280 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 44.33 | 160 | 1280 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 79.23 | 144 | 1152 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 80.15 | 144 | 1152 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 80.09 | 144 | 1152 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 80.78 | 144 | 1152 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 80.79 | 144 | 1152 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 207.91 | 136 | 1088 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 209.77 | 136 | 1088 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 211.22 | 136 | 1088 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 211.81 | 136 | 1088 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 210.96 | 136 | 1088 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 8 | 16 | common_private | q_outer | 16 | 2 | 65.29 | 64 | 1536 | 24 | 24576 | 0 | 512 | 24.0 | 24.0 | 0.333 |
| 1 | 128 | 8 | 16 | common_private | q_outer | 16 | 2 | 65.11 | 64 | 1536 | 24 | 24576 | 0 | 512 | 24.0 | 24.0 | 0.333 |
| 2 | 128 | 8 | 16 | common_private | q_outer | 16 | 2 | 65.23 | 64 | 1536 | 24 | 24576 | 0 | 512 | 24.0 | 24.0 | 0.333 |
| 3 | 128 | 8 | 16 | common_private | q_outer | 16 | 2 | 65.52 | 64 | 1536 | 24 | 24576 | 0 | 512 | 24.0 | 24.0 | 0.333 |
| 4 | 128 | 8 | 16 | common_private | q_outer | 16 | 2 | 65.27 | 64 | 1536 | 24 | 24576 | 0 | 512 | 24.0 | 24.0 | 0.333 |
| 0 | 128 | 8 | 16 | common_private | q_outer | 32 | 4 | 135.85 | 32 | 1280 | 40 | 40960 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 1 | 128 | 8 | 16 | common_private | q_outer | 32 | 4 | 136.07 | 32 | 1280 | 40 | 40960 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 2 | 128 | 8 | 16 | common_private | q_outer | 32 | 4 | 136.19 | 32 | 1280 | 40 | 40960 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 3 | 128 | 8 | 16 | common_private | q_outer | 32 | 4 | 136.73 | 32 | 1280 | 40 | 40960 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 4 | 128 | 8 | 16 | common_private | q_outer | 32 | 4 | 135.98 | 32 | 1280 | 40 | 40960 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 0 | 128 | 8 | 16 | common_private | q_outer | 64 | 8 | 340.44 | 16 | 1152 | 72 | 73728 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 1 | 128 | 8 | 16 | common_private | q_outer | 64 | 8 | 339.55 | 16 | 1152 | 72 | 73728 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 2 | 128 | 8 | 16 | common_private | q_outer | 64 | 8 | 342.62 | 16 | 1152 | 72 | 73728 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 3 | 128 | 8 | 16 | common_private | q_outer | 64 | 8 | 340.98 | 16 | 1152 | 72 | 73728 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 4 | 128 | 8 | 16 | common_private | q_outer | 64 | 8 | 340.35 | 16 | 1152 | 72 | 73728 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 0 | 128 | 8 | 16 | common_private | q_outer | 128 | 16 | 1740.64 | 8 | 1088 | 136 | 139264 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 1 | 128 | 8 | 16 | common_private | q_outer | 128 | 16 | 1743.86 | 8 | 1088 | 136 | 139264 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 2 | 128 | 8 | 16 | common_private | q_outer | 128 | 16 | 1743.79 | 8 | 1088 | 136 | 139264 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 3 | 128 | 8 | 16 | common_private | q_outer | 128 | 16 | 1745.44 | 8 | 1088 | 136 | 139264 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 4 | 128 | 8 | 16 | common_private | q_outer | 128 | 16 | 1753.76 | 8 | 1088 | 136 | 139264 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 0 | 128 | 8 | 16 | common_private | signature | 16 | 2 | 36.14 | 192 | 1536 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 16 | common_private | signature | 16 | 2 | 33.83 | 192 | 1536 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 8 | 16 | common_private | signature | 16 | 2 | 34.55 | 192 | 1536 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 8 | 16 | common_private | signature | 16 | 2 | 34.23 | 192 | 1536 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 8 | 16 | common_private | signature | 16 | 2 | 35.56 | 192 | 1536 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 8 | 16 | common_private | signature | 32 | 4 | 44.95 | 160 | 1280 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 16 | common_private | signature | 32 | 4 | 43.37 | 160 | 1280 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 8 | 16 | common_private | signature | 32 | 4 | 44.29 | 160 | 1280 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 8 | 16 | common_private | signature | 32 | 4 | 44.09 | 160 | 1280 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 8 | 16 | common_private | signature | 32 | 4 | 45.08 | 160 | 1280 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 8 | 16 | common_private | signature | 64 | 8 | 80.76 | 144 | 1152 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 16 | common_private | signature | 64 | 8 | 80.34 | 144 | 1152 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 8 | 16 | common_private | signature | 64 | 8 | 81.41 | 144 | 1152 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 8 | 16 | common_private | signature | 64 | 8 | 80.51 | 144 | 1152 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 8 | 16 | common_private | signature | 64 | 8 | 80.55 | 144 | 1152 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 8 | 16 | common_private | signature | 128 | 16 | 210.32 | 136 | 1088 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 128 | 8 | 16 | common_private | signature | 128 | 16 | 209.78 | 136 | 1088 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 128 | 8 | 16 | common_private | signature | 128 | 16 | 210.06 | 136 | 1088 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 128 | 8 | 16 | common_private | signature | 128 | 16 | 211.63 | 136 | 1088 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 128 | 8 | 16 | common_private | signature | 128 | 16 | 211.38 | 136 | 1088 | 8 | 24576 | 128 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 128 | 8 | 16 | random | bounded_merge | 16 | 2 | 77.81 | 105 | 2018 | 31 | 32288 | 0 | 501 | 16.0 | 31.0 | 0.665 |
| 1 | 128 | 8 | 16 | random | bounded_merge | 16 | 2 | 78.39 | 103 | 2020 | 31 | 32320 | 0 | 503 | 16.0 | 31.0 | 0.632 |
| 2 | 128 | 8 | 16 | random | bounded_merge | 16 | 2 | 79.86 | 106 | 2023 | 31 | 32368 | 0 | 505 | 16.0 | 31.0 | 0.677 |
| 3 | 128 | 8 | 16 | random | bounded_merge | 16 | 2 | 79.76 | 104 | 2020 | 31 | 32320 | 0 | 502 | 16.0 | 31.0 | 0.648 |
| 4 | 128 | 8 | 16 | random | bounded_merge | 16 | 2 | 80.37 | 100 | 2016 | 31 | 32256 | 0 | 505 | 16.0 | 31.0 | 0.587 |
| 0 | 128 | 8 | 16 | random | bounded_merge | 32 | 4 | 103.14 | 130 | 1946 | 31 | 31152 | 53 | 501 | 16.0 | 29.1 | 0.458 |
| 1 | 128 | 8 | 16 | random | bounded_merge | 32 | 4 | 113.65 | 133 | 1958 | 31 | 31376 | 60 | 503 | 15.0 | 29.0 | 0.530 |
| 2 | 128 | 8 | 16 | random | bounded_merge | 32 | 2 | 82.12 | 130 | 1948 | 31 | 31168 | 60 | 505 | 15.0 | 30.0 | 0.406 |
| 3 | 128 | 8 | 16 | random | bounded_merge | 32 | 4 | 104.40 | 119 | 1973 | 31 | 31632 | 41 | 502 | 16.0 | 30.0 | 0.496 |
| 4 | 128 | 8 | 16 | random | bounded_merge | 32 | 4 | 104.13 | 133 | 1960 | 31 | 31392 | 54 | 505 | 16.0 | 30.0 | 0.516 |
| 0 | 128 | 8 | 16 | random | bounded_merge | 64 | 4 | 74.41 | 300 | 1792 | 16 | 28992 | 124 | 501 | 2.0 | 14.0 | 0.893 |
| 1 | 128 | 8 | 16 | random | bounded_merge | 64 | 4 | 78.57 | 276 | 1837 | 16 | 29632 | 117 | 503 | 2.0 | 14.0 | 0.884 |
| 2 | 128 | 8 | 16 | random | bounded_merge | 64 | 4 | 75.08 | 290 | 1819 | 16 | 29296 | 125 | 505 | 2.0 | 14.0 | 0.893 |
| 3 | 128 | 8 | 16 | random | bounded_merge | 64 | 4 | 73.25 | 271 | 1839 | 16 | 29584 | 121 | 502 | 2.0 | 14.0 | 0.890 |
| 4 | 128 | 8 | 16 | random | bounded_merge | 64 | 4 | 75.57 | 277 | 1849 | 16 | 29856 | 122 | 505 | 2.0 | 14.0 | 0.890 |
| 0 | 128 | 8 | 16 | random | bounded_merge | 128 | 8 | 113.87 | 464 | 1579 | 13 | 26368 | 128 | 501 | 1.0 | 10.0 | 1.000 |
| 1 | 128 | 8 | 16 | random | bounded_merge | 128 | 4 | 70.86 | 451 | 1629 | 14 | 26896 | 128 | 503 | 1.0 | 10.0 | 1.000 |
| 2 | 128 | 8 | 16 | random | bounded_merge | 128 | 8 | 111.34 | 446 | 1630 | 15 | 26896 | 128 | 505 | 1.0 | 11.0 | 1.000 |
| 3 | 128 | 8 | 16 | random | bounded_merge | 128 | 4 | 70.29 | 462 | 1612 | 14 | 26528 | 128 | 502 | 1.0 | 11.0 | 1.000 |
| 4 | 128 | 8 | 16 | random | bounded_merge | 128 | 4 | 70.72 | 443 | 1653 | 15 | 27264 | 128 | 505 | 1.0 | 11.0 | 1.000 |
| 0 | 128 | 8 | 16 | random | q_outer | 16 | 2 | 83.11 | 64 | 2018 | 32 | 32288 | 0 | 501 | 32.0 | 32.0 | 0.015 |
| 1 | 128 | 8 | 16 | random | q_outer | 16 | 2 | 83.97 | 64 | 2020 | 32 | 32320 | 0 | 503 | 32.0 | 32.0 | 0.014 |
| 2 | 128 | 8 | 16 | random | q_outer | 16 | 2 | 84.09 | 64 | 2023 | 32 | 32368 | 0 | 505 | 32.0 | 32.0 | 0.012 |
| 3 | 128 | 8 | 16 | random | q_outer | 16 | 2 | 83.49 | 64 | 2020 | 32 | 32320 | 0 | 502 | 32.0 | 32.0 | 0.014 |
| 4 | 128 | 8 | 16 | random | q_outer | 16 | 2 | 84.59 | 64 | 2016 | 32 | 32256 | 0 | 505 | 32.0 | 32.0 | 0.016 |
| 0 | 128 | 8 | 16 | random | q_outer | 32 | 4 | 203.70 | 32 | 1946 | 64 | 62272 | 0 | 501 | 61.0 | 62.0 | 0.000 |
| 1 | 128 | 8 | 16 | random | q_outer | 32 | 4 | 203.91 | 32 | 1958 | 64 | 62656 | 0 | 503 | 61.0 | 63.9 | 0.000 |
| 2 | 128 | 8 | 16 | random | q_outer | 32 | 4 | 203.38 | 32 | 1948 | 63 | 62336 | 0 | 505 | 61.0 | 62.0 | 0.000 |
| 3 | 128 | 8 | 16 | random | q_outer | 32 | 4 | 203.66 | 32 | 1973 | 64 | 63136 | 0 | 502 | 62.0 | 63.0 | 0.000 |
| 4 | 128 | 8 | 16 | random | q_outer | 32 | 4 | 210.57 | 32 | 1960 | 64 | 62720 | 0 | 505 | 61.5 | 64.0 | 0.000 |
| 0 | 128 | 8 | 16 | random | q_outer | 64 | 8 | 548.87 | 16 | 1792 | 120 | 114688 | 0 | 501 | 113.0 | 116.0 | 0.000 |
| 1 | 128 | 8 | 16 | random | q_outer | 64 | 8 | 550.01 | 16 | 1837 | 119 | 117568 | 0 | 503 | 114.5 | 118.5 | 0.000 |
| 2 | 128 | 8 | 16 | random | q_outer | 64 | 8 | 549.78 | 16 | 1819 | 120 | 116416 | 0 | 505 | 114.5 | 117.5 | 0.000 |
| 3 | 128 | 8 | 16 | random | q_outer | 64 | 8 | 549.39 | 16 | 1839 | 120 | 117696 | 0 | 502 | 115.5 | 118.5 | 0.000 |
| 4 | 128 | 8 | 16 | random | q_outer | 64 | 8 | 565.00 | 16 | 1849 | 121 | 118336 | 0 | 505 | 115.5 | 119.5 | 0.000 |
| 0 | 128 | 8 | 16 | random | q_outer | 128 | 16 | 2525.63 | 8 | 1579 | 203 | 202112 | 0 | 501 | 197.5 | 201.6 | 0.000 |
| 1 | 128 | 8 | 16 | random | q_outer | 128 | 16 | 2618.89 | 8 | 1629 | 211 | 208512 | 0 | 503 | 205.0 | 209.6 | 0.000 |
| 2 | 128 | 8 | 16 | random | q_outer | 128 | 16 | 2643.67 | 8 | 1630 | 212 | 208640 | 0 | 505 | 204.5 | 210.6 | 0.000 |
| 3 | 128 | 8 | 16 | random | q_outer | 128 | 16 | 2602.06 | 8 | 1612 | 209 | 206336 | 0 | 502 | 203.5 | 206.9 | 0.000 |
| 4 | 128 | 8 | 16 | random | q_outer | 128 | 16 | 2660.51 | 8 | 1653 | 214 | 211584 | 0 | 505 | 206.0 | 212.6 | 0.000 |
| 0 | 128 | 8 | 16 | random | signature | 16 | 2 | 54.65 | 151 | 2018 | 16 | 32288 | 46 | 501 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 8 | 16 | random | signature | 16 | 2 | 54.54 | 153 | 2020 | 16 | 32320 | 50 | 503 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 8 | 16 | random | signature | 16 | 2 | 54.89 | 150 | 2023 | 16 | 32368 | 44 | 505 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 8 | 16 | random | signature | 16 | 2 | 54.95 | 152 | 2020 | 16 | 32320 | 48 | 502 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 8 | 16 | random | signature | 16 | 2 | 54.94 | 156 | 2016 | 16 | 32256 | 56 | 505 | 15.0 | 16.0 | 1.000 |
| 0 | 128 | 8 | 16 | random | signature | 32 | 4 | 72.97 | 205 | 1946 | 16 | 31152 | 101 | 501 | 13.0 | 16.0 | 1.000 |
| 1 | 128 | 8 | 16 | random | signature | 32 | 4 | 71.45 | 198 | 1958 | 16 | 31376 | 98 | 503 | 14.0 | 16.0 | 1.000 |
| 2 | 128 | 8 | 16 | random | signature | 32 | 2 | 54.56 | 211 | 1948 | 16 | 31168 | 113 | 505 | 14.0 | 15.0 | 1.000 |
| 3 | 128 | 8 | 16 | random | signature | 32 | 4 | 71.60 | 188 | 1973 | 16 | 31632 | 91 | 502 | 14.0 | 16.0 | 1.000 |
| 4 | 128 | 8 | 16 | random | signature | 32 | 4 | 71.78 | 200 | 1960 | 16 | 31392 | 94 | 505 | 14.0 | 16.0 | 1.000 |
| 0 | 128 | 8 | 16 | random | signature | 64 | 4 | 74.15 | 316 | 1792 | 16 | 28992 | 126 | 501 | 2.0 | 14.0 | 1.000 |
| 1 | 128 | 8 | 16 | random | signature | 64 | 4 | 75.61 | 292 | 1837 | 16 | 29632 | 125 | 503 | 2.0 | 14.0 | 1.000 |
| 2 | 128 | 8 | 16 | random | signature | 64 | 4 | 71.48 | 306 | 1819 | 16 | 29296 | 127 | 505 | 2.0 | 14.0 | 1.000 |
| 3 | 128 | 8 | 16 | random | signature | 64 | 4 | 72.64 | 287 | 1839 | 16 | 29584 | 124 | 502 | 2.0 | 14.0 | 1.000 |
| 4 | 128 | 8 | 16 | random | signature | 64 | 4 | 74.38 | 293 | 1849 | 16 | 29856 | 124 | 505 | 1.0 | 14.0 | 1.000 |
| 0 | 128 | 8 | 16 | random | signature | 128 | 8 | 113.66 | 464 | 1579 | 13 | 26368 | 128 | 501 | 1.0 | 10.0 | 1.000 |
| 1 | 128 | 8 | 16 | random | signature | 128 | 4 | 69.96 | 451 | 1629 | 14 | 26896 | 128 | 503 | 1.0 | 10.0 | 1.000 |
| 2 | 128 | 8 | 16 | random | signature | 128 | 8 | 111.55 | 446 | 1630 | 15 | 26896 | 128 | 505 | 1.0 | 11.0 | 1.000 |
| 3 | 128 | 8 | 16 | random | signature | 128 | 4 | 70.93 | 462 | 1612 | 14 | 26528 | 128 | 502 | 1.0 | 11.0 | 1.000 |
| 4 | 128 | 8 | 16 | random | signature | 128 | 4 | 69.62 | 443 | 1653 | 15 | 27264 | 128 | 505 | 1.0 | 11.0 | 1.000 |
| 0 | 128 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 159.19 | 120 | 4009 | 62 | 64144 | 0 | 509 | 32.0 | 32.0 | 0.916 |
| 1 | 128 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 147.60 | 121 | 3995 | 62 | 63920 | 0 | 499 | 32.0 | 32.0 | 0.938 |
| 2 | 128 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 152.60 | 118 | 3938 | 61 | 63008 | 0 | 490 | 32.0 | 32.0 | 0.918 |
| 3 | 128 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 149.62 | 117 | 3954 | 62 | 63264 | 0 | 499 | 32.0 | 32.0 | 0.894 |
| 4 | 128 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 157.97 | 117 | 3934 | 63 | 62944 | 0 | 498 | 32.0 | 32.0 | 0.903 |
| 0 | 128 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 204.47 | 112 | 3826 | 62 | 61264 | 4 | 509 | 32.0 | 42.9 | 0.839 |
| 1 | 128 | 8 | 32 | clustered | bounded_merge | 32 | 2 | 147.59 | 110 | 3785 | 62 | 60560 | 0 | 499 | 32.0 | 44.1 | 0.860 |
| 2 | 128 | 8 | 32 | clustered | bounded_merge | 32 | 2 | 154.21 | 102 | 3679 | 62 | 58864 | 3 | 490 | 32.0 | 53.0 | 0.746 |
| 3 | 128 | 8 | 32 | clustered | bounded_merge | 32 | 2 | 158.86 | 107 | 3819 | 63 | 61104 | 1 | 499 | 32.0 | 53.0 | 0.785 |
| 4 | 128 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 194.11 | 113 | 3702 | 61 | 60064 | 12 | 498 | 32.0 | 47.0 | 0.814 |
| 0 | 128 | 8 | 32 | clustered | bounded_merge | 64 | 4 | 137.66 | 157 | 3423 | 32 | 55744 | 65 | 509 | 24.0 | 32.0 | 0.930 |
| 1 | 128 | 8 | 32 | clustered | bounded_merge | 64 | 4 | 139.42 | 151 | 3238 | 32 | 52704 | 62 | 499 | 24.0 | 32.0 | 0.944 |
| 2 | 128 | 8 | 32 | clustered | bounded_merge | 64 | 4 | 136.12 | 156 | 3307 | 32 | 54192 | 63 | 490 | 24.0 | 32.0 | 0.925 |
| 3 | 128 | 8 | 32 | clustered | bounded_merge | 64 | 4 | 139.49 | 161 | 3257 | 32 | 53760 | 76 | 499 | 24.0 | 32.0 | 0.927 |
| 4 | 128 | 8 | 32 | clustered | bounded_merge | 64 | 8 | 188.50 | 158 | 3277 | 32 | 55344 | 66 | 498 | 23.0 | 32.0 | 0.916 |
| 0 | 128 | 8 | 32 | clustered | bounded_merge | 128 | 4 | 136.29 | 201 | 2647 | 41 | 47248 | 110 | 509 | 12.0 | 30.0 | 0.949 |
| 1 | 128 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 190.08 | 191 | 2509 | 32 | 45600 | 109 | 499 | 10.0 | 31.0 | 0.967 |
| 2 | 128 | 8 | 32 | clustered | bounded_merge | 128 | 4 | 142.05 | 187 | 2553 | 32 | 46672 | 105 | 490 | 10.0 | 32.0 | 0.970 |
| 3 | 128 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 181.97 | 197 | 2632 | 32 | 48192 | 107 | 499 | 11.0 | 31.4 | 0.979 |
| 4 | 128 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 191.82 | 194 | 2692 | 32 | 48384 | 107 | 498 | 12.0 | 32.0 | 0.966 |
| 0 | 128 | 8 | 32 | clustered | q_outer | 16 | 2 | 164.67 | 64 | 4009 | 64 | 64144 | 0 | 509 | 64.0 | 64.0 | 0.022 |
| 1 | 128 | 8 | 32 | clustered | q_outer | 16 | 2 | 160.22 | 64 | 3995 | 64 | 63920 | 0 | 499 | 64.0 | 64.0 | 0.025 |
| 2 | 128 | 8 | 32 | clustered | q_outer | 16 | 2 | 162.62 | 64 | 3938 | 64 | 63008 | 0 | 490 | 64.0 | 64.0 | 0.040 |
| 3 | 128 | 8 | 32 | clustered | q_outer | 16 | 2 | 162.77 | 64 | 3954 | 64 | 63264 | 0 | 499 | 64.0 | 64.0 | 0.036 |
| 4 | 128 | 8 | 32 | clustered | q_outer | 16 | 2 | 160.85 | 64 | 3934 | 64 | 62944 | 0 | 498 | 64.0 | 64.0 | 0.041 |
| 0 | 128 | 8 | 32 | clustered | q_outer | 32 | 4 | 400.22 | 32 | 3826 | 128 | 122432 | 0 | 509 | 126.0 | 128.0 | 0.000 |
| 1 | 128 | 8 | 32 | clustered | q_outer | 32 | 4 | 405.18 | 32 | 3785 | 128 | 121120 | 0 | 499 | 124.0 | 128.0 | 0.000 |
| 2 | 128 | 8 | 32 | clustered | q_outer | 32 | 4 | 399.14 | 32 | 3679 | 128 | 117728 | 0 | 490 | 117.0 | 128.0 | 0.000 |
| 3 | 128 | 8 | 32 | clustered | q_outer | 32 | 4 | 403.31 | 32 | 3819 | 128 | 122208 | 0 | 499 | 123.0 | 128.0 | 0.000 |
| 4 | 128 | 8 | 32 | clustered | q_outer | 32 | 4 | 404.31 | 32 | 3702 | 128 | 118464 | 0 | 498 | 122.0 | 128.0 | 0.005 |
| 0 | 128 | 8 | 32 | clustered | q_outer | 64 | 8 | 1076.57 | 16 | 3423 | 239 | 219072 | 0 | 509 | 218.0 | 234.0 | 0.000 |
| 1 | 128 | 8 | 32 | clustered | q_outer | 64 | 8 | 1042.78 | 16 | 3238 | 230 | 207232 | 0 | 499 | 211.5 | 226.5 | 0.000 |
| 2 | 128 | 8 | 32 | clustered | q_outer | 64 | 8 | 1139.25 | 16 | 3307 | 256 | 211648 | 0 | 490 | 201.5 | 241.5 | 0.000 |
| 3 | 128 | 8 | 32 | clustered | q_outer | 64 | 8 | 1112.34 | 16 | 3257 | 250 | 208448 | 0 | 499 | 204.5 | 241.0 | 0.000 |
| 4 | 128 | 8 | 32 | clustered | q_outer | 64 | 8 | 1061.13 | 16 | 3277 | 239 | 209728 | 0 | 498 | 204.0 | 231.0 | 0.000 |
| 0 | 128 | 8 | 32 | clustered | q_outer | 128 | 16 | 4694.98 | 8 | 2647 | 382 | 338816 | 0 | 509 | 327.5 | 373.6 | 0.000 |
| 1 | 128 | 8 | 32 | clustered | q_outer | 128 | 16 | 4137.17 | 8 | 2509 | 337 | 321152 | 0 | 499 | 313.5 | 332.8 | 0.000 |
| 2 | 128 | 8 | 32 | clustered | q_outer | 128 | 16 | 4661.48 | 8 | 2553 | 378 | 326784 | 0 | 490 | 316.5 | 370.3 | 0.000 |
| 3 | 128 | 8 | 32 | clustered | q_outer | 128 | 16 | 4421.94 | 8 | 2632 | 359 | 336896 | 0 | 499 | 339.0 | 357.6 | 0.000 |
| 4 | 128 | 8 | 32 | clustered | q_outer | 128 | 16 | 4689.94 | 8 | 2692 | 383 | 344576 | 0 | 498 | 342.0 | 378.8 | 0.000 |
| 0 | 128 | 8 | 32 | clustered | signature | 16 | 2 | 98.10 | 136 | 4009 | 32 | 64144 | 16 | 509 | 32.0 | 32.0 | 1.000 |
| 1 | 128 | 8 | 32 | clustered | signature | 16 | 2 | 96.42 | 135 | 3995 | 32 | 63920 | 14 | 499 | 32.0 | 32.0 | 1.000 |
| 2 | 128 | 8 | 32 | clustered | signature | 16 | 2 | 98.42 | 138 | 3938 | 32 | 63008 | 20 | 490 | 32.0 | 32.0 | 1.000 |
| 3 | 128 | 8 | 32 | clustered | signature | 16 | 2 | 101.28 | 137 | 3954 | 32 | 63264 | 20 | 499 | 32.0 | 32.0 | 1.000 |
| 4 | 128 | 8 | 32 | clustered | signature | 16 | 2 | 99.05 | 139 | 3934 | 32 | 62944 | 22 | 498 | 32.0 | 32.0 | 1.000 |
| 0 | 128 | 8 | 32 | clustered | signature | 32 | 4 | 131.96 | 147 | 3826 | 32 | 61264 | 36 | 509 | 32.0 | 32.0 | 1.000 |
| 1 | 128 | 8 | 32 | clustered | signature | 32 | 2 | 101.19 | 144 | 3785 | 32 | 60560 | 34 | 499 | 32.0 | 32.0 | 1.000 |
| 2 | 128 | 8 | 32 | clustered | signature | 32 | 2 | 102.33 | 155 | 3679 | 32 | 58864 | 54 | 490 | 30.0 | 32.0 | 1.000 |
| 3 | 128 | 8 | 32 | clustered | signature | 32 | 2 | 101.85 | 148 | 3819 | 32 | 61104 | 41 | 499 | 32.0 | 32.0 | 1.000 |
| 4 | 128 | 8 | 32 | clustered | signature | 32 | 4 | 134.28 | 153 | 3702 | 32 | 60064 | 44 | 498 | 32.0 | 32.0 | 1.000 |
| 0 | 128 | 8 | 32 | clustered | signature | 64 | 4 | 136.68 | 173 | 3423 | 32 | 55744 | 78 | 509 | 21.0 | 32.0 | 1.000 |
| 1 | 128 | 8 | 32 | clustered | signature | 64 | 4 | 133.59 | 167 | 3238 | 32 | 52704 | 75 | 499 | 18.0 | 32.0 | 1.000 |
| 2 | 128 | 8 | 32 | clustered | signature | 64 | 4 | 131.76 | 171 | 3307 | 32 | 54192 | 76 | 490 | 20.0 | 32.0 | 1.000 |
| 3 | 128 | 8 | 32 | clustered | signature | 64 | 4 | 131.71 | 177 | 3257 | 32 | 53760 | 82 | 499 | 21.0 | 32.0 | 1.000 |
| 4 | 128 | 8 | 32 | clustered | signature | 64 | 8 | 182.19 | 174 | 3277 | 32 | 55344 | 76 | 498 | 18.5 | 32.0 | 1.000 |
| 0 | 128 | 8 | 32 | clustered | signature | 128 | 4 | 132.71 | 208 | 2647 | 32 | 47248 | 112 | 509 | 11.0 | 28.0 | 1.000 |
| 1 | 128 | 8 | 32 | clustered | signature | 128 | 8 | 177.58 | 199 | 2509 | 32 | 45600 | 112 | 499 | 10.0 | 27.2 | 1.000 |
| 2 | 128 | 8 | 32 | clustered | signature | 128 | 4 | 124.26 | 195 | 2553 | 32 | 46672 | 109 | 490 | 9.0 | 31.0 | 1.000 |
| 3 | 128 | 8 | 32 | clustered | signature | 128 | 8 | 169.67 | 204 | 2632 | 32 | 48192 | 108 | 499 | 10.0 | 31.0 | 1.000 |
| 4 | 128 | 8 | 32 | clustered | signature | 128 | 8 | 178.72 | 202 | 2692 | 32 | 48384 | 108 | 498 | 11.0 | 31.0 | 1.000 |
| 0 | 128 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 118.36 | 64 | 3072 | 48 | 49152 | 0 | 512 | 48.0 | 48.0 | 0.333 |
| 1 | 128 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 119.23 | 64 | 3072 | 48 | 49152 | 0 | 512 | 48.0 | 48.0 | 0.333 |
| 2 | 128 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 119.88 | 64 | 3072 | 48 | 49152 | 0 | 512 | 48.0 | 48.0 | 0.333 |
| 3 | 128 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 119.27 | 64 | 3072 | 48 | 49152 | 0 | 512 | 48.0 | 48.0 | 0.333 |
| 4 | 128 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 120.13 | 64 | 3072 | 48 | 49152 | 0 | 512 | 48.0 | 48.0 | 0.333 |
| 0 | 128 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 71.69 | 160 | 2560 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 71.36 | 160 | 2560 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 71.29 | 160 | 2560 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 71.66 | 160 | 2560 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 71.39 | 160 | 2560 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 145.50 | 144 | 2304 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 145.42 | 144 | 2304 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 145.82 | 144 | 2304 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 145.76 | 144 | 2304 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 145.66 | 144 | 2304 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 390.49 | 136 | 2176 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 390.31 | 136 | 2176 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 391.69 | 136 | 2176 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 390.64 | 136 | 2176 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 389.88 | 136 | 2176 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 8 | 32 | common_private | q_outer | 16 | 2 | 121.40 | 64 | 3072 | 48 | 49152 | 0 | 512 | 48.0 | 48.0 | 0.333 |
| 1 | 128 | 8 | 32 | common_private | q_outer | 16 | 2 | 123.04 | 64 | 3072 | 48 | 49152 | 0 | 512 | 48.0 | 48.0 | 0.333 |
| 2 | 128 | 8 | 32 | common_private | q_outer | 16 | 2 | 123.61 | 64 | 3072 | 48 | 49152 | 0 | 512 | 48.0 | 48.0 | 0.333 |
| 3 | 128 | 8 | 32 | common_private | q_outer | 16 | 2 | 123.72 | 64 | 3072 | 48 | 49152 | 0 | 512 | 48.0 | 48.0 | 0.333 |
| 4 | 128 | 8 | 32 | common_private | q_outer | 16 | 2 | 123.24 | 64 | 3072 | 48 | 49152 | 0 | 512 | 48.0 | 48.0 | 0.333 |
| 0 | 128 | 8 | 32 | common_private | q_outer | 32 | 4 | 257.63 | 32 | 2560 | 80 | 81920 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 1 | 128 | 8 | 32 | common_private | q_outer | 32 | 4 | 259.21 | 32 | 2560 | 80 | 81920 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 2 | 128 | 8 | 32 | common_private | q_outer | 32 | 4 | 259.15 | 32 | 2560 | 80 | 81920 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 3 | 128 | 8 | 32 | common_private | q_outer | 32 | 4 | 259.62 | 32 | 2560 | 80 | 81920 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 4 | 128 | 8 | 32 | common_private | q_outer | 32 | 4 | 257.83 | 32 | 2560 | 80 | 81920 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 0 | 128 | 8 | 32 | common_private | q_outer | 64 | 8 | 668.01 | 16 | 2304 | 144 | 147456 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 1 | 128 | 8 | 32 | common_private | q_outer | 64 | 8 | 662.58 | 16 | 2304 | 144 | 147456 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 2 | 128 | 8 | 32 | common_private | q_outer | 64 | 8 | 671.14 | 16 | 2304 | 144 | 147456 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 3 | 128 | 8 | 32 | common_private | q_outer | 64 | 8 | 660.67 | 16 | 2304 | 144 | 147456 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 4 | 128 | 8 | 32 | common_private | q_outer | 64 | 8 | 662.35 | 16 | 2304 | 144 | 147456 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 0 | 128 | 8 | 32 | common_private | q_outer | 128 | 16 | 3400.81 | 8 | 2176 | 272 | 278528 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 1 | 128 | 8 | 32 | common_private | q_outer | 128 | 16 | 3398.86 | 8 | 2176 | 272 | 278528 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 2 | 128 | 8 | 32 | common_private | q_outer | 128 | 16 | 3396.70 | 8 | 2176 | 272 | 278528 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 3 | 128 | 8 | 32 | common_private | q_outer | 128 | 16 | 3404.01 | 8 | 2176 | 272 | 278528 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 4 | 128 | 8 | 32 | common_private | q_outer | 128 | 16 | 3409.57 | 8 | 2176 | 272 | 278528 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 0 | 128 | 8 | 32 | common_private | signature | 16 | 2 | 57.33 | 192 | 3072 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 8 | 32 | common_private | signature | 16 | 2 | 58.47 | 192 | 3072 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 8 | 32 | common_private | signature | 16 | 2 | 57.71 | 192 | 3072 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 8 | 32 | common_private | signature | 16 | 2 | 57.09 | 192 | 3072 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 8 | 32 | common_private | signature | 16 | 2 | 57.92 | 192 | 3072 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 8 | 32 | common_private | signature | 32 | 4 | 71.63 | 160 | 2560 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 8 | 32 | common_private | signature | 32 | 4 | 71.93 | 160 | 2560 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 8 | 32 | common_private | signature | 32 | 4 | 71.07 | 160 | 2560 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 8 | 32 | common_private | signature | 32 | 4 | 72.57 | 160 | 2560 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 8 | 32 | common_private | signature | 32 | 4 | 71.36 | 160 | 2560 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 8 | 32 | common_private | signature | 64 | 8 | 145.43 | 144 | 2304 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 8 | 32 | common_private | signature | 64 | 8 | 145.50 | 144 | 2304 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 8 | 32 | common_private | signature | 64 | 8 | 145.60 | 144 | 2304 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 8 | 32 | common_private | signature | 64 | 8 | 145.78 | 144 | 2304 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 8 | 32 | common_private | signature | 64 | 8 | 145.99 | 144 | 2304 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 8 | 32 | common_private | signature | 128 | 16 | 390.76 | 136 | 2176 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 128 | 8 | 32 | common_private | signature | 128 | 16 | 390.50 | 136 | 2176 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 128 | 8 | 32 | common_private | signature | 128 | 16 | 390.65 | 136 | 2176 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 128 | 8 | 32 | common_private | signature | 128 | 16 | 390.47 | 136 | 2176 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 128 | 8 | 32 | common_private | signature | 128 | 16 | 390.37 | 136 | 2176 | 16 | 49152 | 128 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 128 | 8 | 32 | random | bounded_merge | 16 | 2 | 150.97 | 67 | 3946 | 63 | 63136 | 0 | 512 | 62.0 | 63.0 | 0.087 |
| 1 | 128 | 8 | 32 | random | bounded_merge | 16 | 2 | 149.69 | 68 | 3963 | 63 | 63408 | 0 | 512 | 62.0 | 63.0 | 0.098 |
| 2 | 128 | 8 | 32 | random | bounded_merge | 16 | 2 | 149.78 | 71 | 3972 | 63 | 63552 | 0 | 512 | 62.0 | 63.0 | 0.144 |
| 3 | 128 | 8 | 32 | random | bounded_merge | 16 | 2 | 148.98 | 71 | 3964 | 63 | 63424 | 0 | 512 | 62.0 | 63.0 | 0.146 |
| 4 | 128 | 8 | 32 | random | bounded_merge | 16 | 2 | 149.78 | 73 | 3971 | 63 | 63536 | 0 | 512 | 61.0 | 63.0 | 0.177 |
| 0 | 128 | 8 | 32 | random | bounded_merge | 32 | 4 | 133.13 | 289 | 3691 | 31 | 59344 | 128 | 512 | 4.0 | 28.0 | 0.777 |
| 1 | 128 | 8 | 32 | random | bounded_merge | 32 | 4 | 182.68 | 272 | 3718 | 57 | 59680 | 125 | 512 | 4.0 | 28.0 | 0.748 |
| 2 | 128 | 8 | 32 | random | bounded_merge | 32 | 4 | 203.36 | 265 | 3733 | 57 | 60000 | 126 | 512 | 4.0 | 28.0 | 0.735 |
| 3 | 128 | 8 | 32 | random | bounded_merge | 32 | 4 | 132.74 | 267 | 3725 | 32 | 59792 | 124 | 512 | 4.0 | 28.0 | 0.732 |
| 4 | 128 | 8 | 32 | random | bounded_merge | 32 | 4 | 209.94 | 262 | 3753 | 57 | 60208 | 126 | 512 | 4.0 | 28.9 | 0.740 |
| 0 | 128 | 8 | 32 | random | bounded_merge | 64 | 8 | 211.05 | 557 | 3288 | 26 | 54080 | 128 | 512 | 2.0 | 21.0 | 0.973 |
| 1 | 128 | 8 | 32 | random | bounded_merge | 64 | 4 | 125.52 | 551 | 3315 | 27 | 54416 | 128 | 512 | 2.0 | 21.0 | 0.973 |
| 2 | 128 | 8 | 32 | random | bounded_merge | 64 | 4 | 127.14 | 534 | 3323 | 28 | 54560 | 128 | 512 | 2.0 | 21.0 | 0.969 |
| 3 | 128 | 8 | 32 | random | bounded_merge | 64 | 4 | 127.87 | 553 | 3312 | 27 | 54528 | 128 | 512 | 2.0 | 21.0 | 0.970 |
| 4 | 128 | 8 | 32 | random | bounded_merge | 64 | 4 | 128.84 | 539 | 3325 | 28 | 54448 | 128 | 512 | 2.0 | 21.0 | 0.973 |
| 0 | 128 | 8 | 32 | random | bounded_merge | 128 | 8 | 170.36 | 973 | 2611 | 19 | 46992 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 128 | 8 | 32 | random | bounded_merge | 128 | 8 | 172.43 | 984 | 2644 | 21 | 47344 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 128 | 8 | 32 | random | bounded_merge | 128 | 8 | 174.65 | 945 | 2668 | 19 | 47600 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 128 | 8 | 32 | random | bounded_merge | 128 | 8 | 178.87 | 962 | 2639 | 21 | 47328 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 128 | 8 | 32 | random | bounded_merge | 128 | 8 | 160.39 | 978 | 2634 | 19 | 47168 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 128 | 8 | 32 | random | q_outer | 16 | 2 | 155.49 | 64 | 3946 | 64 | 63136 | 0 | 512 | 62.0 | 63.0 | 0.038 |
| 1 | 128 | 8 | 32 | random | q_outer | 16 | 2 | 162.31 | 64 | 3963 | 64 | 63408 | 0 | 512 | 62.0 | 63.0 | 0.034 |
| 2 | 128 | 8 | 32 | random | q_outer | 16 | 2 | 160.72 | 64 | 3972 | 64 | 63552 | 0 | 512 | 62.0 | 63.7 | 0.031 |
| 3 | 128 | 8 | 32 | random | q_outer | 16 | 2 | 156.55 | 64 | 3964 | 64 | 63424 | 0 | 512 | 62.0 | 63.7 | 0.033 |
| 4 | 128 | 8 | 32 | random | q_outer | 16 | 2 | 156.84 | 64 | 3971 | 64 | 63536 | 0 | 512 | 62.0 | 64.0 | 0.031 |
| 0 | 128 | 8 | 32 | random | q_outer | 32 | 4 | 374.06 | 32 | 3691 | 122 | 118112 | 0 | 512 | 116.0 | 119.9 | 0.000 |
| 1 | 128 | 8 | 32 | random | q_outer | 32 | 4 | 383.58 | 32 | 3718 | 121 | 118976 | 0 | 512 | 116.0 | 119.0 | 0.000 |
| 2 | 128 | 8 | 32 | random | q_outer | 32 | 4 | 381.35 | 32 | 3733 | 122 | 119456 | 0 | 512 | 117.0 | 120.8 | 0.000 |
| 3 | 128 | 8 | 32 | random | q_outer | 32 | 4 | 379.92 | 32 | 3725 | 123 | 119200 | 0 | 512 | 116.5 | 120.9 | 0.000 |
| 4 | 128 | 8 | 32 | random | q_outer | 32 | 4 | 379.01 | 32 | 3753 | 122 | 120096 | 0 | 512 | 117.0 | 120.9 | 0.000 |
| 0 | 128 | 8 | 32 | random | q_outer | 64 | 8 | 971.13 | 16 | 3288 | 213 | 210432 | 0 | 512 | 205.0 | 211.5 | 0.000 |
| 1 | 128 | 8 | 32 | random | q_outer | 64 | 8 | 973.86 | 16 | 3315 | 217 | 212160 | 0 | 512 | 208.0 | 211.5 | 0.000 |
| 2 | 128 | 8 | 32 | random | q_outer | 64 | 8 | 970.47 | 16 | 3323 | 214 | 212672 | 0 | 512 | 207.0 | 212.0 | 0.000 |
| 3 | 128 | 8 | 32 | random | q_outer | 64 | 8 | 970.30 | 16 | 3312 | 216 | 211968 | 0 | 512 | 207.5 | 213.0 | 0.000 |
| 4 | 128 | 8 | 32 | random | q_outer | 64 | 8 | 997.11 | 16 | 3325 | 220 | 212800 | 0 | 512 | 207.5 | 212.0 | 0.000 |
| 0 | 128 | 8 | 32 | random | q_outer | 128 | 16 | 4105.96 | 8 | 2611 | 336 | 334208 | 0 | 512 | 325.5 | 334.6 | 0.000 |
| 1 | 128 | 8 | 32 | random | q_outer | 128 | 16 | 4254.30 | 8 | 2644 | 346 | 338432 | 0 | 512 | 330.0 | 338.3 | 0.000 |
| 2 | 128 | 8 | 32 | random | q_outer | 128 | 16 | 4276.16 | 8 | 2668 | 347 | 341504 | 0 | 512 | 332.5 | 342.1 | 0.000 |
| 3 | 128 | 8 | 32 | random | q_outer | 128 | 16 | 4148.60 | 8 | 2639 | 338 | 337792 | 0 | 512 | 331.5 | 334.5 | 0.000 |
| 4 | 128 | 8 | 32 | random | q_outer | 128 | 16 | 4162.24 | 8 | 2634 | 338 | 337152 | 0 | 512 | 328.0 | 337.3 | 0.000 |
| 0 | 128 | 8 | 32 | random | signature | 16 | 2 | 96.21 | 189 | 3946 | 32 | 63136 | 122 | 512 | 29.0 | 31.0 | 1.000 |
| 1 | 128 | 8 | 32 | random | signature | 16 | 2 | 96.69 | 188 | 3963 | 32 | 63408 | 120 | 512 | 29.0 | 31.0 | 1.000 |
| 2 | 128 | 8 | 32 | random | signature | 16 | 2 | 96.01 | 185 | 3972 | 32 | 63552 | 114 | 512 | 29.0 | 31.0 | 1.000 |
| 3 | 128 | 8 | 32 | random | signature | 16 | 2 | 95.83 | 185 | 3964 | 32 | 63424 | 114 | 512 | 29.0 | 31.0 | 1.000 |
| 4 | 128 | 8 | 32 | random | signature | 16 | 2 | 95.42 | 183 | 3971 | 32 | 63536 | 110 | 512 | 29.0 | 31.0 | 1.000 |
| 0 | 128 | 8 | 32 | random | signature | 32 | 4 | 124.73 | 321 | 3691 | 30 | 59344 | 128 | 512 | 3.0 | 27.0 | 1.000 |
| 1 | 128 | 8 | 32 | random | signature | 32 | 4 | 126.57 | 308 | 3718 | 30 | 59680 | 128 | 512 | 3.0 | 28.0 | 1.000 |
| 2 | 128 | 8 | 32 | random | signature | 32 | 4 | 128.09 | 302 | 3733 | 31 | 60000 | 128 | 512 | 3.0 | 28.0 | 1.000 |
| 3 | 128 | 8 | 32 | random | signature | 32 | 4 | 125.13 | 305 | 3725 | 32 | 59792 | 127 | 512 | 3.0 | 28.0 | 1.000 |
| 4 | 128 | 8 | 32 | random | signature | 32 | 4 | 126.97 | 299 | 3753 | 31 | 60208 | 128 | 512 | 3.0 | 28.0 | 1.000 |
| 0 | 128 | 8 | 32 | random | signature | 64 | 8 | 195.80 | 561 | 3288 | 26 | 54080 | 128 | 512 | 2.0 | 21.0 | 1.000 |
| 1 | 128 | 8 | 32 | random | signature | 64 | 4 | 120.72 | 555 | 3315 | 26 | 54416 | 128 | 512 | 2.0 | 21.0 | 1.000 |
| 2 | 128 | 8 | 32 | random | signature | 64 | 4 | 118.98 | 539 | 3323 | 26 | 54560 | 128 | 512 | 2.0 | 21.0 | 1.000 |
| 3 | 128 | 8 | 32 | random | signature | 64 | 4 | 120.56 | 557 | 3312 | 26 | 54528 | 128 | 512 | 2.0 | 21.0 | 1.000 |
| 4 | 128 | 8 | 32 | random | signature | 64 | 4 | 122.19 | 543 | 3325 | 28 | 54448 | 128 | 512 | 2.0 | 21.0 | 1.000 |
| 0 | 128 | 8 | 32 | random | signature | 128 | 8 | 170.34 | 973 | 2611 | 19 | 46992 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 128 | 8 | 32 | random | signature | 128 | 8 | 172.03 | 984 | 2644 | 21 | 47344 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 128 | 8 | 32 | random | signature | 128 | 8 | 174.37 | 945 | 2668 | 19 | 47600 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 128 | 8 | 32 | random | signature | 128 | 8 | 178.62 | 962 | 2639 | 21 | 47328 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 128 | 8 | 32 | random | signature | 128 | 8 | 160.35 | 978 | 2634 | 19 | 47168 | 128 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 256 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 63.20 | 247 | 2012 | 23 | 32192 | 0 | 502 | 8.0 | 8.0 | 0.968 |
| 1 | 256 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 64.45 | 247 | 2032 | 23 | 32512 | 0 | 498 | 8.0 | 8.0 | 0.957 |
| 2 | 256 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 63.42 | 239 | 1995 | 23 | 31920 | 0 | 494 | 8.0 | 8.0 | 0.929 |
| 3 | 256 | 4 | 8 | clustered | bounded_merge | 16 | 2 | 42.13 | 246 | 2003 | 14 | 32048 | 0 | 494 | 8.0 | 8.0 | 0.965 |
| 4 | 256 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 63.20 | 242 | 2014 | 23 | 32224 | 0 | 500 | 8.0 | 8.0 | 0.935 |
| 0 | 256 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 35.54 | 259 | 1956 | 8 | 31296 | 26 | 502 | 8.0 | 8.0 | 0.960 |
| 1 | 256 | 4 | 8 | clustered | bounded_merge | 32 | 2 | 35.51 | 258 | 1951 | 8 | 31216 | 26 | 498 | 8.0 | 8.0 | 0.963 |
| 2 | 256 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 35.59 | 259 | 1930 | 8 | 30880 | 30 | 494 | 8.0 | 8.0 | 0.965 |
| 3 | 256 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 35.94 | 262 | 1898 | 8 | 30368 | 34 | 494 | 8.0 | 8.0 | 0.974 |
| 4 | 256 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 35.36 | 261 | 1943 | 8 | 31088 | 28 | 500 | 8.0 | 8.0 | 0.963 |
| 0 | 256 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 40.72 | 296 | 1824 | 8 | 29184 | 84 | 502 | 8.0 | 8.0 | 0.975 |
| 1 | 256 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 38.96 | 284 | 1824 | 8 | 29184 | 76 | 498 | 8.0 | 8.0 | 0.976 |
| 2 | 256 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 40.54 | 290 | 1797 | 8 | 28752 | 89 | 494 | 8.0 | 8.0 | 0.977 |
| 3 | 256 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 40.72 | 289 | 1802 | 8 | 28832 | 78 | 494 | 8.0 | 8.0 | 0.982 |
| 4 | 256 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 40.03 | 286 | 1854 | 8 | 29664 | 70 | 500 | 8.0 | 8.0 | 0.979 |
| 0 | 256 | 4 | 8 | clustered | bounded_merge | 128 | 4 | 43.39 | 333 | 1617 | 8 | 25872 | 145 | 502 | 4.0 | 8.0 | 1.000 |
| 1 | 256 | 4 | 8 | clustered | bounded_merge | 128 | 4 | 42.33 | 320 | 1647 | 8 | 26352 | 136 | 498 | 5.5 | 8.0 | 1.000 |
| 2 | 256 | 4 | 8 | clustered | bounded_merge | 128 | 4 | 43.23 | 324 | 1590 | 8 | 25440 | 160 | 494 | 5.0 | 8.0 | 1.000 |
| 3 | 256 | 4 | 8 | clustered | bounded_merge | 128 | 8 | 53.24 | 335 | 1608 | 8 | 25776 | 153 | 494 | 5.0 | 8.0 | 1.000 |
| 4 | 256 | 4 | 8 | clustered | bounded_merge | 128 | 4 | 43.64 | 339 | 1683 | 8 | 26928 | 149 | 500 | 5.0 | 8.0 | 1.000 |
| 0 | 256 | 4 | 8 | clustered | q_outer | 16 | 4 | 77.95 | 64 | 2012 | 32 | 32192 | 0 | 502 | 32.0 | 32.0 | 0.000 |
| 1 | 256 | 4 | 8 | clustered | q_outer | 16 | 4 | 77.20 | 64 | 2032 | 32 | 32512 | 0 | 498 | 32.0 | 32.0 | 0.000 |
| 2 | 256 | 4 | 8 | clustered | q_outer | 16 | 4 | 78.91 | 64 | 1995 | 32 | 31920 | 0 | 494 | 32.0 | 32.0 | 0.000 |
| 3 | 256 | 4 | 8 | clustered | q_outer | 16 | 4 | 76.98 | 64 | 2003 | 32 | 32048 | 0 | 494 | 32.0 | 32.0 | 0.000 |
| 4 | 256 | 4 | 8 | clustered | q_outer | 16 | 4 | 78.47 | 64 | 2014 | 32 | 32224 | 0 | 500 | 32.0 | 32.0 | 0.000 |
| 0 | 256 | 4 | 8 | clustered | q_outer | 32 | 8 | 214.08 | 32 | 1956 | 64 | 62592 | 0 | 502 | 62.0 | 64.0 | 0.000 |
| 1 | 256 | 4 | 8 | clustered | q_outer | 32 | 8 | 218.73 | 32 | 1951 | 64 | 62432 | 0 | 498 | 63.0 | 64.0 | 0.000 |
| 2 | 256 | 4 | 8 | clustered | q_outer | 32 | 8 | 217.90 | 32 | 1930 | 64 | 61760 | 0 | 494 | 61.0 | 64.0 | 0.000 |
| 3 | 256 | 4 | 8 | clustered | q_outer | 32 | 8 | 216.80 | 32 | 1898 | 64 | 60736 | 0 | 494 | 60.5 | 64.0 | 0.000 |
| 4 | 256 | 4 | 8 | clustered | q_outer | 32 | 8 | 216.11 | 32 | 1943 | 64 | 62176 | 0 | 500 | 61.0 | 64.0 | 0.000 |
| 0 | 256 | 4 | 8 | clustered | q_outer | 64 | 16 | 607.42 | 16 | 1824 | 126 | 116736 | 0 | 502 | 113.0 | 123.5 | 0.000 |
| 1 | 256 | 4 | 8 | clustered | q_outer | 64 | 16 | 588.96 | 16 | 1824 | 122 | 116736 | 0 | 498 | 115.0 | 121.0 | 0.000 |
| 2 | 256 | 4 | 8 | clustered | q_outer | 64 | 16 | 612.21 | 16 | 1797 | 128 | 115008 | 0 | 494 | 113.0 | 120.5 | 0.000 |
| 3 | 256 | 4 | 8 | clustered | q_outer | 64 | 16 | 574.21 | 16 | 1802 | 120 | 115328 | 0 | 494 | 115.0 | 119.0 | 0.000 |
| 4 | 256 | 4 | 8 | clustered | q_outer | 64 | 16 | 601.43 | 16 | 1854 | 127 | 118656 | 0 | 500 | 117.0 | 121.5 | 0.000 |
| 0 | 256 | 4 | 8 | clustered | q_outer | 128 | 32 | 2387.98 | 8 | 1617 | 215 | 206976 | 0 | 502 | 201.5 | 210.1 | 0.000 |
| 1 | 256 | 4 | 8 | clustered | q_outer | 128 | 32 | 2434.31 | 8 | 1647 | 218 | 210816 | 0 | 498 | 204.5 | 218.0 | 0.000 |
| 2 | 256 | 4 | 8 | clustered | q_outer | 128 | 32 | 2442.57 | 8 | 1590 | 218 | 203520 | 0 | 494 | 199.5 | 209.6 | 0.000 |
| 3 | 256 | 4 | 8 | clustered | q_outer | 128 | 32 | 2350.97 | 8 | 1608 | 209 | 205824 | 0 | 494 | 204.5 | 208.3 | 0.000 |
| 4 | 256 | 4 | 8 | clustered | q_outer | 128 | 32 | 2451.54 | 8 | 1683 | 218 | 215424 | 0 | 500 | 210.5 | 215.9 | 0.000 |
| 0 | 256 | 4 | 8 | clustered | signature | 16 | 2 | 34.14 | 263 | 2012 | 8 | 32192 | 15 | 502 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 4 | 8 | clustered | signature | 16 | 2 | 34.01 | 259 | 2032 | 8 | 32512 | 8 | 498 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 4 | 8 | clustered | signature | 16 | 2 | 35.78 | 265 | 1995 | 8 | 31920 | 22 | 494 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 4 | 8 | clustered | signature | 16 | 2 | 34.34 | 264 | 2003 | 8 | 32048 | 18 | 494 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 4 | 8 | clustered | signature | 16 | 4 | 37.95 | 264 | 2014 | 8 | 32224 | 15 | 500 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 4 | 8 | clustered | signature | 32 | 4 | 41.96 | 276 | 1956 | 8 | 31296 | 42 | 502 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 4 | 8 | clustered | signature | 32 | 2 | 35.74 | 275 | 1951 | 8 | 31216 | 43 | 498 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 4 | 8 | clustered | signature | 32 | 4 | 42.82 | 277 | 1930 | 8 | 30880 | 47 | 494 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 4 | 8 | clustered | signature | 32 | 4 | 42.80 | 279 | 1898 | 8 | 30368 | 48 | 494 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 4 | 8 | clustered | signature | 32 | 4 | 42.44 | 280 | 1943 | 8 | 31088 | 45 | 500 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 4 | 8 | clustered | signature | 64 | 4 | 44.08 | 306 | 1824 | 8 | 29184 | 93 | 502 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 4 | 8 | clustered | signature | 64 | 4 | 42.16 | 295 | 1824 | 8 | 29184 | 85 | 498 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 4 | 8 | clustered | signature | 64 | 4 | 43.70 | 300 | 1797 | 8 | 28752 | 98 | 494 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 4 | 8 | clustered | signature | 64 | 4 | 42.84 | 299 | 1802 | 8 | 28832 | 87 | 494 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 4 | 8 | clustered | signature | 64 | 4 | 44.08 | 297 | 1854 | 8 | 29664 | 80 | 500 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 4 | 8 | clustered | signature | 128 | 4 | 43.80 | 333 | 1617 | 8 | 25872 | 145 | 502 | 4.0 | 8.0 | 1.000 |
| 1 | 256 | 4 | 8 | clustered | signature | 128 | 4 | 42.56 | 320 | 1647 | 8 | 26352 | 136 | 498 | 5.5 | 8.0 | 1.000 |
| 2 | 256 | 4 | 8 | clustered | signature | 128 | 4 | 43.12 | 324 | 1590 | 8 | 25440 | 160 | 494 | 5.0 | 8.0 | 1.000 |
| 3 | 256 | 4 | 8 | clustered | signature | 128 | 8 | 53.20 | 335 | 1608 | 8 | 25776 | 153 | 494 | 5.0 | 8.0 | 1.000 |
| 4 | 256 | 4 | 8 | clustered | signature | 128 | 4 | 43.51 | 339 | 1683 | 8 | 26928 | 149 | 500 | 5.0 | 8.0 | 1.000 |
| 0 | 256 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 51.42 | 64 | 1279 | 20 | 20464 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 1 | 256 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 51.03 | 64 | 1279 | 20 | 20464 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 2 | 256 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 51.72 | 64 | 1279 | 20 | 20464 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 3 | 256 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 51.71 | 64 | 1279 | 20 | 20464 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 4 | 256 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 52.37 | 64 | 1279 | 20 | 20464 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 0 | 256 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 45.38 | 288 | 1151 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 256 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 44.57 | 288 | 1151 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 256 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 45.40 | 288 | 1151 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 256 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 44.94 | 288 | 1151 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 256 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 45.55 | 288 | 1151 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 256 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 71.90 | 272 | 1087 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 256 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 71.38 | 272 | 1087 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 256 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 72.82 | 272 | 1087 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 256 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 71.98 | 272 | 1087 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 256 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 72.78 | 272 | 1087 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 256 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 138.61 | 264 | 1055 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 256 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 135.44 | 264 | 1055 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 256 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 135.93 | 264 | 1055 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 256 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 135.00 | 264 | 1055 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 256 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 136.00 | 264 | 1055 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 256 | 4 | 8 | common_private | q_outer | 16 | 4 | 51.73 | 64 | 1279 | 20 | 20464 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 1 | 256 | 4 | 8 | common_private | q_outer | 16 | 4 | 53.70 | 64 | 1279 | 20 | 20464 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 2 | 256 | 4 | 8 | common_private | q_outer | 16 | 4 | 52.80 | 64 | 1279 | 20 | 20464 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 3 | 256 | 4 | 8 | common_private | q_outer | 16 | 4 | 51.76 | 64 | 1279 | 20 | 20464 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 4 | 256 | 4 | 8 | common_private | q_outer | 16 | 4 | 52.51 | 64 | 1279 | 20 | 20464 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 0 | 256 | 4 | 8 | common_private | q_outer | 32 | 8 | 126.88 | 32 | 1151 | 36 | 36832 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 1 | 256 | 4 | 8 | common_private | q_outer | 32 | 8 | 131.12 | 32 | 1151 | 36 | 36832 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 2 | 256 | 4 | 8 | common_private | q_outer | 32 | 8 | 129.55 | 32 | 1151 | 36 | 36832 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 3 | 256 | 4 | 8 | common_private | q_outer | 32 | 8 | 127.57 | 32 | 1151 | 36 | 36832 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 4 | 256 | 4 | 8 | common_private | q_outer | 32 | 8 | 128.51 | 32 | 1151 | 36 | 36832 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 0 | 256 | 4 | 8 | common_private | q_outer | 64 | 16 | 337.01 | 16 | 1087 | 68 | 69568 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 1 | 256 | 4 | 8 | common_private | q_outer | 64 | 16 | 341.80 | 16 | 1087 | 68 | 69568 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 2 | 256 | 4 | 8 | common_private | q_outer | 64 | 16 | 338.82 | 16 | 1087 | 68 | 69568 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 3 | 256 | 4 | 8 | common_private | q_outer | 64 | 16 | 339.26 | 16 | 1087 | 68 | 69568 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 4 | 256 | 4 | 8 | common_private | q_outer | 64 | 16 | 341.97 | 16 | 1087 | 68 | 69568 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 0 | 256 | 4 | 8 | common_private | q_outer | 128 | 32 | 1528.72 | 8 | 1055 | 132 | 135040 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 1 | 256 | 4 | 8 | common_private | q_outer | 128 | 32 | 1538.67 | 8 | 1055 | 132 | 135040 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 2 | 256 | 4 | 8 | common_private | q_outer | 128 | 32 | 1534.22 | 8 | 1055 | 132 | 135040 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 3 | 256 | 4 | 8 | common_private | q_outer | 128 | 32 | 1540.27 | 8 | 1055 | 132 | 135040 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 4 | 256 | 4 | 8 | common_private | q_outer | 128 | 32 | 1541.73 | 8 | 1055 | 132 | 135040 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 0 | 256 | 4 | 8 | common_private | signature | 16 | 4 | 36.28 | 320 | 1279 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 256 | 4 | 8 | common_private | signature | 16 | 4 | 36.42 | 320 | 1279 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 256 | 4 | 8 | common_private | signature | 16 | 4 | 37.00 | 320 | 1279 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 256 | 4 | 8 | common_private | signature | 16 | 4 | 36.21 | 320 | 1279 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 256 | 4 | 8 | common_private | signature | 16 | 4 | 36.60 | 320 | 1279 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 256 | 4 | 8 | common_private | signature | 32 | 8 | 45.85 | 288 | 1151 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 256 | 4 | 8 | common_private | signature | 32 | 8 | 45.73 | 288 | 1151 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 256 | 4 | 8 | common_private | signature | 32 | 8 | 46.19 | 288 | 1151 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 256 | 4 | 8 | common_private | signature | 32 | 8 | 45.05 | 288 | 1151 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 256 | 4 | 8 | common_private | signature | 32 | 8 | 45.64 | 288 | 1151 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 256 | 4 | 8 | common_private | signature | 64 | 16 | 72.48 | 272 | 1087 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 256 | 4 | 8 | common_private | signature | 64 | 16 | 72.44 | 272 | 1087 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 256 | 4 | 8 | common_private | signature | 64 | 16 | 73.31 | 272 | 1087 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 256 | 4 | 8 | common_private | signature | 64 | 16 | 71.67 | 272 | 1087 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 256 | 4 | 8 | common_private | signature | 64 | 16 | 73.27 | 272 | 1087 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 256 | 4 | 8 | common_private | signature | 128 | 32 | 136.84 | 264 | 1055 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 256 | 4 | 8 | common_private | signature | 128 | 32 | 136.89 | 264 | 1055 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 256 | 4 | 8 | common_private | signature | 128 | 32 | 136.06 | 264 | 1055 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 256 | 4 | 8 | common_private | signature | 128 | 32 | 134.88 | 264 | 1055 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 256 | 4 | 8 | common_private | signature | 128 | 32 | 135.96 | 264 | 1055 | 4 | 20464 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 256 | 4 | 8 | random | bounded_merge | 16 | 4 | 82.22 | 172 | 1990 | 31 | 31840 | 0 | 500 | 8.0 | 23.0 | 0.527 |
| 1 | 256 | 4 | 8 | random | bounded_merge | 16 | 4 | 81.21 | 183 | 2000 | 30 | 32000 | 0 | 504 | 8.0 | 23.0 | 0.601 |
| 2 | 256 | 4 | 8 | random | bounded_merge | 16 | 4 | 81.53 | 182 | 2004 | 30 | 32064 | 0 | 499 | 8.0 | 23.0 | 0.596 |
| 3 | 256 | 4 | 8 | random | bounded_merge | 16 | 4 | 75.29 | 177 | 1996 | 30 | 31936 | 0 | 506 | 8.0 | 23.0 | 0.568 |
| 4 | 256 | 4 | 8 | random | bounded_merge | 16 | 4 | 73.59 | 178 | 2000 | 30 | 32000 | 0 | 504 | 8.0 | 23.0 | 0.569 |
| 0 | 256 | 4 | 8 | random | bounded_merge | 32 | 4 | 41.42 | 336 | 1928 | 8 | 30848 | 146 | 500 | 7.0 | 8.0 | 0.901 |
| 1 | 256 | 4 | 8 | random | bounded_merge | 32 | 4 | 41.23 | 321 | 1946 | 8 | 31136 | 124 | 504 | 7.0 | 8.0 | 0.899 |
| 2 | 256 | 4 | 8 | random | bounded_merge | 32 | 4 | 41.40 | 324 | 1946 | 8 | 31136 | 126 | 499 | 7.0 | 8.0 | 0.896 |
| 3 | 256 | 4 | 8 | random | bounded_merge | 32 | 4 | 42.83 | 337 | 1931 | 8 | 30896 | 140 | 506 | 7.0 | 8.0 | 0.904 |
| 4 | 256 | 4 | 8 | random | bounded_merge | 32 | 4 | 41.39 | 323 | 1943 | 8 | 31088 | 133 | 504 | 7.0 | 8.0 | 0.890 |
| 0 | 256 | 4 | 8 | random | bounded_merge | 64 | 4 | 45.19 | 461 | 1794 | 8 | 28704 | 219 | 500 | 5.0 | 7.0 | 0.958 |
| 1 | 256 | 4 | 8 | random | bounded_merge | 64 | 4 | 45.04 | 436 | 1838 | 8 | 29408 | 209 | 504 | 5.0 | 8.0 | 0.961 |
| 2 | 256 | 4 | 8 | random | bounded_merge | 64 | 4 | 45.62 | 439 | 1831 | 8 | 29296 | 216 | 499 | 5.0 | 7.0 | 0.954 |
| 3 | 256 | 4 | 8 | random | bounded_merge | 64 | 4 | 46.47 | 455 | 1815 | 8 | 29040 | 218 | 506 | 5.0 | 7.0 | 0.952 |
| 4 | 256 | 4 | 8 | random | bounded_merge | 64 | 4 | 45.82 | 428 | 1840 | 8 | 29440 | 217 | 504 | 6.0 | 7.0 | 0.952 |
| 0 | 256 | 4 | 8 | random | bounded_merge | 128 | 4 | 48.17 | 633 | 1588 | 8 | 25408 | 250 | 500 | 1.0 | 6.0 | 1.000 |
| 1 | 256 | 4 | 8 | random | bounded_merge | 128 | 4 | 49.36 | 596 | 1646 | 8 | 26336 | 246 | 504 | 1.0 | 6.0 | 1.000 |
| 2 | 256 | 4 | 8 | random | bounded_merge | 128 | 8 | 62.33 | 606 | 1627 | 8 | 26048 | 255 | 499 | 1.0 | 6.0 | 1.000 |
| 3 | 256 | 4 | 8 | random | bounded_merge | 128 | 4 | 49.01 | 621 | 1609 | 8 | 25744 | 249 | 506 | 1.0 | 6.0 | 1.000 |
| 4 | 256 | 4 | 8 | random | bounded_merge | 128 | 8 | 59.01 | 594 | 1647 | 8 | 26368 | 252 | 504 | 1.0 | 6.0 | 1.000 |
| 0 | 256 | 4 | 8 | random | q_outer | 16 | 4 | 76.07 | 64 | 1990 | 32 | 31840 | 0 | 500 | 31.0 | 32.0 | 0.000 |
| 1 | 256 | 4 | 8 | random | q_outer | 16 | 4 | 77.90 | 64 | 2000 | 32 | 32000 | 0 | 504 | 31.0 | 32.0 | 0.000 |
| 2 | 256 | 4 | 8 | random | q_outer | 16 | 4 | 77.89 | 64 | 2004 | 32 | 32064 | 0 | 499 | 31.5 | 32.0 | 0.000 |
| 3 | 256 | 4 | 8 | random | q_outer | 16 | 4 | 79.17 | 64 | 1996 | 32 | 31936 | 0 | 506 | 31.0 | 32.0 | 0.000 |
| 4 | 256 | 4 | 8 | random | q_outer | 16 | 4 | 78.24 | 64 | 2000 | 32 | 32000 | 0 | 504 | 31.0 | 32.0 | 0.000 |
| 0 | 256 | 4 | 8 | random | q_outer | 32 | 8 | 214.54 | 32 | 1928 | 64 | 61696 | 0 | 500 | 60.0 | 62.0 | 0.000 |
| 1 | 256 | 4 | 8 | random | q_outer | 32 | 8 | 216.48 | 32 | 1946 | 64 | 62272 | 0 | 504 | 61.0 | 62.9 | 0.000 |
| 2 | 256 | 4 | 8 | random | q_outer | 32 | 8 | 214.45 | 32 | 1946 | 64 | 62272 | 0 | 499 | 61.0 | 63.0 | 0.000 |
| 3 | 256 | 4 | 8 | random | q_outer | 32 | 8 | 216.43 | 32 | 1931 | 64 | 61792 | 0 | 506 | 60.5 | 62.0 | 0.000 |
| 4 | 256 | 4 | 8 | random | q_outer | 32 | 8 | 212.03 | 32 | 1943 | 63 | 62176 | 0 | 504 | 61.0 | 62.0 | 0.000 |
| 0 | 256 | 4 | 8 | random | q_outer | 64 | 16 | 555.96 | 16 | 1794 | 117 | 114816 | 0 | 500 | 112.0 | 115.5 | 0.000 |
| 1 | 256 | 4 | 8 | random | q_outer | 64 | 16 | 569.54 | 16 | 1838 | 121 | 117632 | 0 | 504 | 115.5 | 117.5 | 0.000 |
| 2 | 256 | 4 | 8 | random | q_outer | 64 | 16 | 578.37 | 16 | 1831 | 120 | 117184 | 0 | 499 | 115.0 | 118.0 | 0.000 |
| 3 | 256 | 4 | 8 | random | q_outer | 64 | 16 | 570.90 | 16 | 1815 | 117 | 116160 | 0 | 506 | 114.0 | 116.5 | 0.000 |
| 4 | 256 | 4 | 8 | random | q_outer | 64 | 16 | 572.14 | 16 | 1840 | 119 | 117760 | 0 | 504 | 115.0 | 118.0 | 0.000 |
| 0 | 256 | 4 | 8 | random | q_outer | 128 | 32 | 2320.58 | 8 | 1588 | 207 | 203264 | 0 | 500 | 198.5 | 202.8 | 0.000 |
| 1 | 256 | 4 | 8 | random | q_outer | 128 | 32 | 2369.79 | 8 | 1646 | 213 | 210688 | 0 | 504 | 208.0 | 210.9 | 0.000 |
| 2 | 256 | 4 | 8 | random | q_outer | 128 | 32 | 2345.50 | 8 | 1627 | 209 | 208256 | 0 | 499 | 204.5 | 206.2 | 0.000 |
| 3 | 256 | 4 | 8 | random | q_outer | 128 | 32 | 2329.39 | 8 | 1609 | 206 | 205952 | 0 | 506 | 203.0 | 205.3 | 0.000 |
| 4 | 256 | 4 | 8 | random | q_outer | 128 | 32 | 2414.83 | 8 | 1647 | 216 | 210816 | 0 | 504 | 204.5 | 212.5 | 0.000 |
| 0 | 256 | 4 | 8 | random | signature | 16 | 4 | 42.87 | 306 | 1990 | 8 | 31840 | 91 | 500 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 4 | 8 | random | signature | 16 | 4 | 40.52 | 298 | 2000 | 8 | 32000 | 76 | 504 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 4 | 8 | random | signature | 16 | 2 | 36.15 | 300 | 2004 | 8 | 32064 | 78 | 499 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 4 | 8 | random | signature | 16 | 4 | 41.47 | 304 | 1996 | 8 | 31936 | 85 | 506 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 4 | 8 | random | signature | 16 | 4 | 41.21 | 300 | 2000 | 8 | 32000 | 82 | 504 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 4 | 8 | random | signature | 32 | 4 | 44.89 | 366 | 1928 | 8 | 30848 | 163 | 500 | 7.0 | 8.0 | 1.000 |
| 1 | 256 | 4 | 8 | random | signature | 32 | 4 | 43.11 | 351 | 1946 | 8 | 31136 | 144 | 504 | 7.0 | 8.0 | 1.000 |
| 2 | 256 | 4 | 8 | random | signature | 32 | 4 | 45.01 | 355 | 1946 | 8 | 31136 | 145 | 499 | 7.0 | 8.0 | 1.000 |
| 3 | 256 | 4 | 8 | random | signature | 32 | 4 | 47.70 | 367 | 1931 | 8 | 30896 | 158 | 506 | 7.0 | 8.0 | 1.000 |
| 4 | 256 | 4 | 8 | random | signature | 32 | 4 | 45.75 | 355 | 1943 | 8 | 31088 | 156 | 504 | 7.0 | 8.0 | 1.000 |
| 0 | 256 | 4 | 8 | random | signature | 64 | 4 | 48.25 | 474 | 1794 | 8 | 28704 | 221 | 500 | 4.0 | 7.0 | 1.000 |
| 1 | 256 | 4 | 8 | random | signature | 64 | 4 | 47.85 | 448 | 1838 | 8 | 29408 | 213 | 504 | 5.0 | 7.0 | 1.000 |
| 2 | 256 | 4 | 8 | random | signature | 64 | 4 | 47.85 | 453 | 1831 | 8 | 29296 | 220 | 499 | 5.0 | 7.0 | 1.000 |
| 3 | 256 | 4 | 8 | random | signature | 64 | 4 | 48.08 | 469 | 1815 | 8 | 29040 | 225 | 506 | 5.0 | 7.0 | 1.000 |
| 4 | 256 | 4 | 8 | random | signature | 64 | 4 | 49.60 | 443 | 1840 | 8 | 29440 | 223 | 504 | 5.0 | 7.0 | 1.000 |
| 0 | 256 | 4 | 8 | random | signature | 128 | 4 | 48.84 | 633 | 1588 | 8 | 25408 | 250 | 500 | 1.0 | 6.0 | 1.000 |
| 1 | 256 | 4 | 8 | random | signature | 128 | 4 | 49.25 | 596 | 1646 | 8 | 26336 | 246 | 504 | 1.0 | 6.0 | 1.000 |
| 2 | 256 | 4 | 8 | random | signature | 128 | 8 | 61.80 | 606 | 1627 | 8 | 26048 | 255 | 499 | 1.0 | 6.0 | 1.000 |
| 3 | 256 | 4 | 8 | random | signature | 128 | 4 | 48.88 | 621 | 1609 | 8 | 25744 | 249 | 506 | 1.0 | 6.0 | 1.000 |
| 4 | 256 | 4 | 8 | random | signature | 128 | 8 | 58.80 | 594 | 1647 | 8 | 26368 | 252 | 504 | 1.0 | 6.0 | 1.000 |
| 0 | 256 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 107.78 | 234 | 3925 | 41 | 62800 | 0 | 509 | 16.0 | 16.0 | 0.910 |
| 1 | 256 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 118.37 | 232 | 3977 | 46 | 63632 | 0 | 509 | 16.0 | 16.0 | 0.883 |
| 2 | 256 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 115.99 | 229 | 3889 | 46 | 62224 | 0 | 508 | 16.0 | 16.0 | 0.891 |
| 3 | 256 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 106.19 | 236 | 3936 | 41 | 62976 | 0 | 499 | 16.0 | 16.0 | 0.926 |
| 4 | 256 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 125.89 | 230 | 3940 | 50 | 63040 | 0 | 508 | 16.0 | 16.0 | 0.887 |
| 0 | 256 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 61.53 | 275 | 3683 | 16 | 58928 | 64 | 509 | 16.0 | 16.0 | 0.949 |
| 1 | 256 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 63.14 | 273 | 3687 | 16 | 58992 | 65 | 509 | 16.0 | 16.0 | 0.938 |
| 2 | 256 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 63.62 | 274 | 3643 | 16 | 58288 | 64 | 508 | 16.0 | 16.0 | 0.959 |
| 3 | 256 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 61.87 | 275 | 3599 | 16 | 57584 | 67 | 499 | 16.0 | 16.0 | 0.956 |
| 4 | 256 | 4 | 16 | clustered | bounded_merge | 32 | 8 | 88.85 | 278 | 3642 | 16 | 58304 | 68 | 508 | 16.0 | 16.0 | 0.941 |
| 0 | 256 | 4 | 16 | clustered | bounded_merge | 64 | 4 | 79.88 | 329 | 3259 | 27 | 52144 | 140 | 509 | 10.0 | 16.0 | 0.976 |
| 1 | 256 | 4 | 16 | clustered | bounded_merge | 64 | 4 | 83.46 | 326 | 3247 | 27 | 51952 | 152 | 509 | 10.0 | 16.0 | 0.959 |
| 2 | 256 | 4 | 16 | clustered | bounded_merge | 64 | 4 | 74.85 | 325 | 3169 | 27 | 50704 | 148 | 508 | 9.0 | 16.0 | 0.967 |
| 3 | 256 | 4 | 16 | clustered | bounded_merge | 64 | 4 | 82.38 | 321 | 3261 | 27 | 52176 | 134 | 499 | 11.0 | 16.0 | 0.964 |
| 4 | 256 | 4 | 16 | clustered | bounded_merge | 64 | 8 | 108.76 | 326 | 3323 | 27 | 53232 | 139 | 508 | 11.0 | 16.0 | 0.967 |
| 0 | 256 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 90.92 | 382 | 2629 | 16 | 42096 | 202 | 509 | 5.0 | 16.0 | 1.000 |
| 1 | 256 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 88.53 | 390 | 2635 | 16 | 42176 | 221 | 509 | 6.0 | 14.0 | 1.000 |
| 2 | 256 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 91.30 | 378 | 2512 | 16 | 40256 | 221 | 508 | 6.0 | 14.0 | 1.000 |
| 3 | 256 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 90.94 | 396 | 2621 | 16 | 42112 | 220 | 499 | 6.0 | 15.0 | 1.000 |
| 4 | 256 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 93.84 | 395 | 2696 | 16 | 43216 | 209 | 508 | 6.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 16 | clustered | q_outer | 16 | 4 | 149.35 | 64 | 3925 | 64 | 62800 | 0 | 509 | 64.0 | 64.0 | 0.000 |
| 1 | 256 | 4 | 16 | clustered | q_outer | 16 | 4 | 146.62 | 64 | 3977 | 64 | 63632 | 0 | 509 | 64.0 | 64.0 | 0.000 |
| 2 | 256 | 4 | 16 | clustered | q_outer | 16 | 4 | 146.37 | 64 | 3889 | 64 | 62224 | 0 | 508 | 64.0 | 64.0 | 0.000 |
| 3 | 256 | 4 | 16 | clustered | q_outer | 16 | 4 | 146.68 | 64 | 3936 | 64 | 62976 | 0 | 499 | 64.0 | 64.0 | 0.000 |
| 4 | 256 | 4 | 16 | clustered | q_outer | 16 | 4 | 148.58 | 64 | 3940 | 64 | 63040 | 0 | 508 | 64.0 | 64.0 | 0.001 |
| 0 | 256 | 4 | 16 | clustered | q_outer | 32 | 8 | 422.34 | 32 | 3683 | 128 | 117856 | 0 | 509 | 117.0 | 128.0 | 0.000 |
| 1 | 256 | 4 | 16 | clustered | q_outer | 32 | 8 | 415.47 | 32 | 3687 | 128 | 117984 | 0 | 509 | 116.0 | 127.8 | 0.000 |
| 2 | 256 | 4 | 16 | clustered | q_outer | 32 | 8 | 425.42 | 32 | 3643 | 128 | 116576 | 0 | 508 | 114.5 | 128.0 | 0.000 |
| 3 | 256 | 4 | 16 | clustered | q_outer | 32 | 8 | 419.61 | 32 | 3599 | 128 | 115168 | 0 | 499 | 112.5 | 128.0 | 0.000 |
| 4 | 256 | 4 | 16 | clustered | q_outer | 32 | 8 | 420.61 | 32 | 3642 | 128 | 116544 | 0 | 508 | 115.5 | 126.9 | 0.000 |
| 0 | 256 | 4 | 16 | clustered | q_outer | 64 | 16 | 1139.28 | 16 | 3259 | 243 | 208576 | 0 | 509 | 197.0 | 234.0 | 0.000 |
| 1 | 256 | 4 | 16 | clustered | q_outer | 64 | 16 | 1038.74 | 16 | 3247 | 220 | 207808 | 0 | 509 | 198.5 | 219.5 | 0.000 |
| 2 | 256 | 4 | 16 | clustered | q_outer | 64 | 16 | 1098.57 | 16 | 3169 | 230 | 202816 | 0 | 508 | 198.0 | 223.0 | 0.000 |
| 3 | 256 | 4 | 16 | clustered | q_outer | 64 | 16 | 1080.58 | 16 | 3261 | 226 | 208704 | 0 | 499 | 210.0 | 220.0 | 0.000 |
| 4 | 256 | 4 | 16 | clustered | q_outer | 64 | 16 | 1085.69 | 16 | 3323 | 231 | 212672 | 0 | 508 | 211.0 | 224.5 | 0.000 |
| 0 | 256 | 4 | 16 | clustered | q_outer | 128 | 32 | 3829.88 | 8 | 2629 | 348 | 336512 | 0 | 509 | 331.0 | 344.5 | 0.000 |
| 1 | 256 | 4 | 16 | clustered | q_outer | 128 | 32 | 3955.05 | 8 | 2635 | 360 | 337280 | 0 | 509 | 328.5 | 351.6 | 0.000 |
| 2 | 256 | 4 | 16 | clustered | q_outer | 128 | 32 | 3824.02 | 8 | 2512 | 349 | 321536 | 0 | 508 | 313.0 | 338.5 | 0.000 |
| 3 | 256 | 4 | 16 | clustered | q_outer | 128 | 32 | 3838.16 | 8 | 2621 | 350 | 335488 | 0 | 499 | 335.0 | 348.6 | 0.000 |
| 4 | 256 | 4 | 16 | clustered | q_outer | 128 | 32 | 3955.21 | 8 | 2696 | 360 | 345088 | 0 | 508 | 336.5 | 357.2 | 0.000 |
| 0 | 256 | 4 | 16 | clustered | signature | 16 | 4 | 66.01 | 276 | 3925 | 16 | 62800 | 40 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 4 | 16 | clustered | signature | 16 | 2 | 62.41 | 273 | 3977 | 16 | 63632 | 36 | 509 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 4 | 16 | clustered | signature | 16 | 2 | 61.20 | 276 | 3889 | 16 | 62224 | 44 | 508 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 4 | 16 | clustered | signature | 16 | 2 | 60.80 | 272 | 3936 | 16 | 62976 | 32 | 499 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 4 | 16 | clustered | signature | 16 | 4 | 67.20 | 276 | 3940 | 16 | 63040 | 37 | 508 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 16 | clustered | signature | 32 | 4 | 73.07 | 300 | 3683 | 16 | 58928 | 84 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 4 | 16 | clustered | signature | 32 | 4 | 73.01 | 302 | 3687 | 16 | 58992 | 92 | 509 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 4 | 16 | clustered | signature | 32 | 4 | 72.13 | 300 | 3643 | 16 | 58288 | 88 | 508 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 4 | 16 | clustered | signature | 32 | 4 | 69.56 | 301 | 3599 | 16 | 57584 | 89 | 499 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 4 | 16 | clustered | signature | 32 | 8 | 94.61 | 307 | 3642 | 16 | 58304 | 90 | 508 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 16 | clustered | signature | 64 | 4 | 71.07 | 341 | 3259 | 16 | 52144 | 142 | 509 | 9.0 | 16.0 | 1.000 |
| 1 | 256 | 4 | 16 | clustered | signature | 64 | 4 | 73.92 | 340 | 3247 | 16 | 51952 | 158 | 509 | 9.0 | 16.0 | 1.000 |
| 2 | 256 | 4 | 16 | clustered | signature | 64 | 4 | 71.84 | 340 | 3169 | 16 | 50704 | 158 | 508 | 9.0 | 16.0 | 1.000 |
| 3 | 256 | 4 | 16 | clustered | signature | 64 | 4 | 71.85 | 336 | 3261 | 16 | 52176 | 145 | 499 | 10.0 | 16.0 | 1.000 |
| 4 | 256 | 4 | 16 | clustered | signature | 64 | 8 | 94.34 | 340 | 3323 | 16 | 53232 | 147 | 508 | 10.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 16 | clustered | signature | 128 | 8 | 91.12 | 382 | 2629 | 16 | 42096 | 202 | 509 | 5.0 | 16.0 | 1.000 |
| 1 | 256 | 4 | 16 | clustered | signature | 128 | 8 | 89.35 | 390 | 2635 | 16 | 42176 | 221 | 509 | 6.0 | 14.0 | 1.000 |
| 2 | 256 | 4 | 16 | clustered | signature | 128 | 8 | 91.08 | 378 | 2512 | 16 | 40256 | 221 | 508 | 6.0 | 14.0 | 1.000 |
| 3 | 256 | 4 | 16 | clustered | signature | 128 | 8 | 90.78 | 396 | 2621 | 16 | 42112 | 220 | 499 | 6.0 | 15.0 | 1.000 |
| 4 | 256 | 4 | 16 | clustered | signature | 128 | 8 | 93.13 | 395 | 2696 | 16 | 43216 | 209 | 508 | 6.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 93.95 | 64 | 2558 | 40 | 40928 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 1 | 256 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 94.13 | 64 | 2558 | 40 | 40928 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 2 | 256 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 93.27 | 64 | 2558 | 40 | 40928 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 3 | 256 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 93.80 | 64 | 2558 | 40 | 40928 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 4 | 256 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 93.30 | 64 | 2558 | 40 | 40928 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 0 | 256 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 77.12 | 288 | 2302 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 77.60 | 288 | 2302 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 77.43 | 288 | 2302 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 77.25 | 288 | 2302 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 77.18 | 288 | 2302 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 129.76 | 272 | 2174 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 129.24 | 272 | 2174 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 129.96 | 272 | 2174 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 130.40 | 272 | 2174 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 130.95 | 272 | 2174 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 228.60 | 264 | 2110 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 228.91 | 264 | 2110 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 229.37 | 264 | 2110 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 228.93 | 264 | 2110 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 229.22 | 264 | 2110 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 4 | 16 | common_private | q_outer | 16 | 4 | 96.19 | 64 | 2558 | 40 | 40928 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 1 | 256 | 4 | 16 | common_private | q_outer | 16 | 4 | 97.41 | 64 | 2558 | 40 | 40928 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 2 | 256 | 4 | 16 | common_private | q_outer | 16 | 4 | 95.79 | 64 | 2558 | 40 | 40928 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 3 | 256 | 4 | 16 | common_private | q_outer | 16 | 4 | 96.64 | 64 | 2558 | 40 | 40928 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 4 | 256 | 4 | 16 | common_private | q_outer | 16 | 4 | 96.15 | 64 | 2558 | 40 | 40928 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 0 | 256 | 4 | 16 | common_private | q_outer | 32 | 8 | 246.60 | 32 | 2302 | 72 | 73664 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 1 | 256 | 4 | 16 | common_private | q_outer | 32 | 8 | 246.52 | 32 | 2302 | 72 | 73664 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 2 | 256 | 4 | 16 | common_private | q_outer | 32 | 8 | 244.07 | 32 | 2302 | 72 | 73664 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 3 | 256 | 4 | 16 | common_private | q_outer | 32 | 8 | 247.21 | 32 | 2302 | 72 | 73664 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 4 | 256 | 4 | 16 | common_private | q_outer | 32 | 8 | 246.10 | 32 | 2302 | 72 | 73664 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 0 | 256 | 4 | 16 | common_private | q_outer | 64 | 16 | 660.27 | 16 | 2174 | 136 | 139136 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 1 | 256 | 4 | 16 | common_private | q_outer | 64 | 16 | 669.38 | 16 | 2174 | 136 | 139136 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 2 | 256 | 4 | 16 | common_private | q_outer | 64 | 16 | 663.41 | 16 | 2174 | 136 | 139136 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 3 | 256 | 4 | 16 | common_private | q_outer | 64 | 16 | 668.32 | 16 | 2174 | 136 | 139136 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 4 | 256 | 4 | 16 | common_private | q_outer | 64 | 16 | 665.86 | 16 | 2174 | 136 | 139136 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 0 | 256 | 4 | 16 | common_private | q_outer | 128 | 32 | 2964.49 | 8 | 2110 | 264 | 270080 | 0 | 512 | 264.0 | 264.0 | 0.030 |
| 1 | 256 | 4 | 16 | common_private | q_outer | 128 | 32 | 2971.29 | 8 | 2110 | 264 | 270080 | 0 | 512 | 264.0 | 264.0 | 0.030 |
| 2 | 256 | 4 | 16 | common_private | q_outer | 128 | 32 | 2973.69 | 8 | 2110 | 264 | 270080 | 0 | 512 | 264.0 | 264.0 | 0.030 |
| 3 | 256 | 4 | 16 | common_private | q_outer | 128 | 32 | 2970.03 | 8 | 2110 | 264 | 270080 | 0 | 512 | 264.0 | 264.0 | 0.030 |
| 4 | 256 | 4 | 16 | common_private | q_outer | 128 | 32 | 2974.53 | 8 | 2110 | 264 | 270080 | 0 | 512 | 264.0 | 264.0 | 0.030 |
| 0 | 256 | 4 | 16 | common_private | signature | 16 | 4 | 60.45 | 320 | 2558 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 4 | 16 | common_private | signature | 16 | 4 | 59.76 | 320 | 2558 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 4 | 16 | common_private | signature | 16 | 4 | 60.82 | 320 | 2558 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 4 | 16 | common_private | signature | 16 | 4 | 60.03 | 320 | 2558 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 4 | 16 | common_private | signature | 16 | 4 | 60.16 | 320 | 2558 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 4 | 16 | common_private | signature | 32 | 8 | 77.49 | 288 | 2302 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 4 | 16 | common_private | signature | 32 | 8 | 76.94 | 288 | 2302 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 4 | 16 | common_private | signature | 32 | 8 | 77.46 | 288 | 2302 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 4 | 16 | common_private | signature | 32 | 8 | 77.41 | 288 | 2302 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 4 | 16 | common_private | signature | 32 | 8 | 77.02 | 288 | 2302 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 4 | 16 | common_private | signature | 64 | 16 | 129.96 | 272 | 2174 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 4 | 16 | common_private | signature | 64 | 16 | 129.80 | 272 | 2174 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 4 | 16 | common_private | signature | 64 | 16 | 130.54 | 272 | 2174 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 4 | 16 | common_private | signature | 64 | 16 | 130.12 | 272 | 2174 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 4 | 16 | common_private | signature | 64 | 16 | 130.14 | 272 | 2174 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 4 | 16 | common_private | signature | 128 | 32 | 228.24 | 264 | 2110 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 4 | 16 | common_private | signature | 128 | 32 | 229.20 | 264 | 2110 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 4 | 16 | common_private | signature | 128 | 32 | 229.11 | 264 | 2110 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 4 | 16 | common_private | signature | 128 | 32 | 228.98 | 264 | 2110 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 4 | 16 | common_private | signature | 128 | 32 | 229.10 | 264 | 2110 | 8 | 40928 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 4 | 16 | random | bounded_merge | 16 | 4 | 134.87 | 90 | 3902 | 62 | 62432 | 0 | 510 | 58.0 | 62.0 | 0.117 |
| 1 | 256 | 4 | 16 | random | bounded_merge | 16 | 4 | 138.24 | 102 | 3915 | 62 | 62640 | 0 | 511 | 31.0 | 62.0 | 0.164 |
| 2 | 256 | 4 | 16 | random | bounded_merge | 16 | 4 | 135.12 | 85 | 3898 | 62 | 62368 | 0 | 512 | 59.0 | 62.0 | 0.074 |
| 3 | 256 | 4 | 16 | random | bounded_merge | 16 | 4 | 135.83 | 110 | 3927 | 62 | 62832 | 0 | 512 | 31.0 | 61.0 | 0.201 |
| 4 | 256 | 4 | 16 | random | bounded_merge | 16 | 4 | 136.01 | 109 | 3916 | 62 | 62656 | 0 | 512 | 31.0 | 61.2 | 0.205 |
| 0 | 256 | 4 | 16 | random | bounded_merge | 32 | 4 | 76.29 | 573 | 3636 | 16 | 58176 | 249 | 510 | 2.0 | 14.0 | 0.894 |
| 1 | 256 | 4 | 16 | random | bounded_merge | 32 | 4 | 76.33 | 556 | 3676 | 16 | 58816 | 242 | 511 | 2.0 | 14.0 | 0.887 |
| 2 | 256 | 4 | 16 | random | bounded_merge | 32 | 4 | 74.41 | 561 | 3653 | 16 | 58448 | 247 | 512 | 2.0 | 14.0 | 0.898 |
| 3 | 256 | 4 | 16 | random | bounded_merge | 32 | 4 | 75.85 | 544 | 3686 | 16 | 58976 | 240 | 512 | 2.0 | 14.0 | 0.891 |
| 4 | 256 | 4 | 16 | random | bounded_merge | 32 | 4 | 79.91 | 544 | 3702 | 16 | 59232 | 241 | 512 | 2.0 | 14.0 | 0.892 |
| 0 | 256 | 4 | 16 | random | bounded_merge | 64 | 8 | 104.18 | 920 | 3207 | 14 | 51328 | 256 | 510 | 1.0 | 10.0 | 1.000 |
| 1 | 256 | 4 | 16 | random | bounded_merge | 64 | 4 | 81.13 | 896 | 3265 | 14 | 52240 | 256 | 511 | 1.0 | 11.0 | 1.000 |
| 2 | 256 | 4 | 16 | random | bounded_merge | 64 | 8 | 105.86 | 878 | 3272 | 15 | 52368 | 256 | 512 | 1.0 | 11.0 | 1.000 |
| 3 | 256 | 4 | 16 | random | bounded_merge | 64 | 4 | 81.96 | 908 | 3256 | 15 | 52096 | 256 | 512 | 1.0 | 11.0 | 1.000 |
| 4 | 256 | 4 | 16 | random | bounded_merge | 64 | 4 | 82.46 | 869 | 3305 | 15 | 52880 | 256 | 512 | 1.0 | 11.0 | 1.000 |
| 0 | 256 | 4 | 16 | random | bounded_merge | 128 | 8 | 97.98 | 1287 | 2591 | 12 | 41648 | 256 | 510 | 1.0 | 6.0 | 1.000 |
| 1 | 256 | 4 | 16 | random | bounded_merge | 128 | 8 | 99.09 | 1281 | 2613 | 11 | 41936 | 256 | 511 | 1.0 | 6.0 | 1.000 |
| 2 | 256 | 4 | 16 | random | bounded_merge | 128 | 8 | 95.88 | 1252 | 2635 | 11 | 42320 | 256 | 512 | 1.0 | 6.0 | 1.000 |
| 3 | 256 | 4 | 16 | random | bounded_merge | 128 | 8 | 95.60 | 1283 | 2588 | 11 | 41648 | 256 | 512 | 1.0 | 6.0 | 1.000 |
| 4 | 256 | 4 | 16 | random | bounded_merge | 128 | 8 | 96.36 | 1259 | 2640 | 11 | 42400 | 256 | 512 | 1.0 | 6.0 | 1.000 |
| 0 | 256 | 4 | 16 | random | q_outer | 16 | 4 | 143.58 | 64 | 3902 | 64 | 62432 | 0 | 510 | 61.0 | 63.0 | 0.000 |
| 1 | 256 | 4 | 16 | random | q_outer | 16 | 4 | 142.87 | 64 | 3915 | 64 | 62640 | 0 | 511 | 61.0 | 63.0 | 0.000 |
| 2 | 256 | 4 | 16 | random | q_outer | 16 | 4 | 143.63 | 64 | 3898 | 64 | 62368 | 0 | 512 | 61.0 | 62.0 | 0.000 |
| 3 | 256 | 4 | 16 | random | q_outer | 16 | 4 | 144.88 | 64 | 3927 | 64 | 62832 | 0 | 512 | 61.5 | 63.0 | 0.000 |
| 4 | 256 | 4 | 16 | random | q_outer | 16 | 4 | 148.81 | 64 | 3916 | 64 | 62656 | 0 | 512 | 61.0 | 64.0 | 0.000 |
| 0 | 256 | 4 | 16 | random | q_outer | 32 | 8 | 392.00 | 32 | 3636 | 120 | 116352 | 0 | 510 | 114.0 | 117.9 | 0.000 |
| 1 | 256 | 4 | 16 | random | q_outer | 32 | 8 | 396.14 | 32 | 3676 | 120 | 117632 | 0 | 511 | 115.0 | 118.0 | 0.000 |
| 2 | 256 | 4 | 16 | random | q_outer | 32 | 8 | 407.74 | 32 | 3653 | 123 | 116896 | 0 | 512 | 114.5 | 117.9 | 0.000 |
| 3 | 256 | 4 | 16 | random | q_outer | 32 | 8 | 399.68 | 32 | 3686 | 121 | 117952 | 0 | 512 | 116.0 | 118.9 | 0.000 |
| 4 | 256 | 4 | 16 | random | q_outer | 32 | 8 | 401.89 | 32 | 3702 | 121 | 118464 | 0 | 512 | 115.5 | 120.0 | 0.000 |
| 0 | 256 | 4 | 16 | random | q_outer | 64 | 16 | 994.88 | 16 | 3207 | 209 | 205248 | 0 | 510 | 200.0 | 207.0 | 0.000 |
| 1 | 256 | 4 | 16 | random | q_outer | 64 | 16 | 1013.72 | 16 | 3265 | 212 | 208960 | 0 | 511 | 204.0 | 210.0 | 0.000 |
| 2 | 256 | 4 | 16 | random | q_outer | 64 | 16 | 1023.89 | 16 | 3272 | 216 | 209408 | 0 | 512 | 205.0 | 211.0 | 0.000 |
| 3 | 256 | 4 | 16 | random | q_outer | 64 | 16 | 1008.77 | 16 | 3256 | 212 | 208384 | 0 | 512 | 205.0 | 209.0 | 0.000 |
| 4 | 256 | 4 | 16 | random | q_outer | 64 | 16 | 1019.43 | 16 | 3305 | 214 | 211520 | 0 | 512 | 206.5 | 211.5 | 0.000 |
| 0 | 256 | 4 | 16 | random | q_outer | 128 | 32 | 3671.52 | 8 | 2591 | 335 | 331648 | 0 | 510 | 323.0 | 331.5 | 0.000 |
| 1 | 256 | 4 | 16 | random | q_outer | 128 | 32 | 3726.94 | 8 | 2613 | 341 | 334464 | 0 | 511 | 326.0 | 337.5 | 0.000 |
| 2 | 256 | 4 | 16 | random | q_outer | 128 | 32 | 3727.20 | 8 | 2635 | 339 | 337280 | 0 | 512 | 331.5 | 337.6 | 0.000 |
| 3 | 256 | 4 | 16 | random | q_outer | 128 | 32 | 3680.93 | 8 | 2588 | 336 | 331264 | 0 | 512 | 325.5 | 333.2 | 0.000 |
| 4 | 256 | 4 | 16 | random | q_outer | 128 | 32 | 3679.86 | 8 | 2640 | 334 | 337920 | 0 | 512 | 330.5 | 334.0 | 0.000 |
| 0 | 256 | 4 | 16 | random | signature | 16 | 4 | 70.70 | 406 | 3902 | 16 | 62432 | 201 | 510 | 14.0 | 16.0 | 1.000 |
| 1 | 256 | 4 | 16 | random | signature | 16 | 4 | 72.51 | 395 | 3915 | 16 | 62640 | 197 | 511 | 14.0 | 16.0 | 1.000 |
| 2 | 256 | 4 | 16 | random | signature | 16 | 4 | 72.70 | 417 | 3898 | 16 | 62368 | 215 | 512 | 14.0 | 15.0 | 1.000 |
| 3 | 256 | 4 | 16 | random | signature | 16 | 4 | 70.69 | 384 | 3927 | 16 | 62832 | 187 | 512 | 14.0 | 16.0 | 1.000 |
| 4 | 256 | 4 | 16 | random | signature | 16 | 4 | 70.15 | 399 | 3916 | 16 | 62656 | 186 | 512 | 14.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 16 | random | signature | 32 | 4 | 78.16 | 605 | 3636 | 16 | 58176 | 251 | 510 | 2.0 | 14.0 | 1.000 |
| 1 | 256 | 4 | 16 | random | signature | 32 | 4 | 82.59 | 588 | 3676 | 16 | 58816 | 252 | 511 | 2.0 | 14.0 | 1.000 |
| 2 | 256 | 4 | 16 | random | signature | 32 | 4 | 80.25 | 593 | 3653 | 16 | 58448 | 250 | 512 | 2.0 | 14.0 | 1.000 |
| 3 | 256 | 4 | 16 | random | signature | 32 | 4 | 77.24 | 576 | 3686 | 16 | 58976 | 246 | 512 | 2.0 | 14.0 | 1.000 |
| 4 | 256 | 4 | 16 | random | signature | 32 | 4 | 80.22 | 576 | 3702 | 16 | 59232 | 245 | 512 | 2.0 | 14.0 | 1.000 |
| 0 | 256 | 4 | 16 | random | signature | 64 | 8 | 103.61 | 920 | 3207 | 14 | 51328 | 256 | 510 | 1.0 | 10.0 | 1.000 |
| 1 | 256 | 4 | 16 | random | signature | 64 | 4 | 81.90 | 896 | 3265 | 14 | 52240 | 256 | 511 | 1.0 | 11.0 | 1.000 |
| 2 | 256 | 4 | 16 | random | signature | 64 | 8 | 106.44 | 878 | 3272 | 15 | 52368 | 256 | 512 | 1.0 | 11.0 | 1.000 |
| 3 | 256 | 4 | 16 | random | signature | 64 | 4 | 80.95 | 908 | 3256 | 15 | 52096 | 256 | 512 | 1.0 | 11.0 | 1.000 |
| 4 | 256 | 4 | 16 | random | signature | 64 | 4 | 82.53 | 869 | 3305 | 15 | 52880 | 256 | 512 | 1.0 | 11.0 | 1.000 |
| 0 | 256 | 4 | 16 | random | signature | 128 | 8 | 93.90 | 1287 | 2591 | 12 | 41648 | 256 | 510 | 1.0 | 6.0 | 1.000 |
| 1 | 256 | 4 | 16 | random | signature | 128 | 8 | 98.59 | 1281 | 2613 | 11 | 41936 | 256 | 511 | 1.0 | 6.0 | 1.000 |
| 2 | 256 | 4 | 16 | random | signature | 128 | 8 | 96.52 | 1252 | 2635 | 11 | 42320 | 256 | 512 | 1.0 | 6.0 | 1.000 |
| 3 | 256 | 4 | 16 | random | signature | 128 | 8 | 95.21 | 1283 | 2588 | 11 | 41648 | 256 | 512 | 1.0 | 6.0 | 1.000 |
| 4 | 256 | 4 | 16 | random | signature | 128 | 8 | 95.46 | 1259 | 2640 | 11 | 42400 | 256 | 512 | 1.0 | 6.0 | 1.000 |
| 0 | 256 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 250.30 | 202 | 7459 | 111 | 119344 | 0 | 509 | 32.0 | 49.0 | 0.770 |
| 1 | 256 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 266.02 | 202 | 7574 | 124 | 121184 | 0 | 509 | 32.0 | 52.9 | 0.743 |
| 2 | 256 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 300.51 | 184 | 7360 | 123 | 117760 | 0 | 510 | 32.0 | 61.7 | 0.658 |
| 3 | 256 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 274.41 | 205 | 7557 | 125 | 120912 | 0 | 500 | 32.0 | 53.0 | 0.771 |
| 4 | 256 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 287.99 | 200 | 7434 | 121 | 118944 | 0 | 509 | 32.0 | 52.1 | 0.756 |
| 0 | 256 | 4 | 32 | clustered | bounded_merge | 32 | 8 | 179.88 | 316 | 6591 | 32 | 105648 | 135 | 509 | 23.5 | 32.0 | 0.941 |
| 1 | 256 | 4 | 32 | clustered | bounded_merge | 32 | 4 | 116.08 | 309 | 6589 | 32 | 105424 | 130 | 509 | 25.0 | 32.0 | 0.945 |
| 2 | 256 | 4 | 32 | clustered | bounded_merge | 32 | 4 | 112.62 | 315 | 6550 | 32 | 104800 | 128 | 510 | 24.0 | 32.0 | 0.940 |
| 3 | 256 | 4 | 32 | clustered | bounded_merge | 32 | 4 | 113.90 | 312 | 6528 | 32 | 104448 | 137 | 500 | 25.0 | 32.0 | 0.928 |
| 4 | 256 | 4 | 32 | clustered | bounded_merge | 32 | 8 | 177.32 | 317 | 6459 | 32 | 103680 | 133 | 509 | 21.0 | 32.0 | 0.936 |
| 0 | 256 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 250.70 | 394 | 5179 | 57 | 83488 | 217 | 509 | 11.0 | 31.0 | 0.939 |
| 1 | 256 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 169.92 | 386 | 5215 | 32 | 83680 | 221 | 509 | 11.0 | 31.0 | 0.964 |
| 2 | 256 | 4 | 32 | clustered | bounded_merge | 64 | 4 | 111.52 | 377 | 5203 | 32 | 83248 | 212 | 510 | 10.0 | 32.0 | 0.967 |
| 3 | 256 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 193.98 | 379 | 5389 | 54 | 86928 | 206 | 500 | 11.0 | 32.0 | 0.961 |
| 4 | 256 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 175.50 | 383 | 5356 | 32 | 86384 | 204 | 509 | 11.0 | 32.0 | 0.969 |
| 0 | 256 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 152.11 | 445 | 3560 | 32 | 60032 | 253 | 509 | 6.0 | 17.0 | 1.000 |
| 1 | 256 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 136.46 | 452 | 3501 | 32 | 59024 | 255 | 509 | 6.0 | 17.0 | 1.000 |
| 2 | 256 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 140.59 | 439 | 3473 | 32 | 59040 | 253 | 510 | 6.0 | 19.0 | 1.000 |
| 3 | 256 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 141.89 | 453 | 3559 | 32 | 59536 | 251 | 500 | 6.0 | 17.0 | 1.000 |
| 4 | 256 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 157.03 | 452 | 3549 | 32 | 59936 | 248 | 509 | 6.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 32 | clustered | q_outer | 16 | 4 | 281.81 | 64 | 7459 | 128 | 119344 | 0 | 509 | 125.0 | 128.0 | 0.000 |
| 1 | 256 | 4 | 32 | clustered | q_outer | 16 | 4 | 279.89 | 64 | 7574 | 128 | 121184 | 0 | 509 | 122.0 | 128.0 | 0.000 |
| 2 | 256 | 4 | 32 | clustered | q_outer | 16 | 4 | 279.91 | 64 | 7360 | 128 | 117760 | 0 | 510 | 117.5 | 128.0 | 0.000 |
| 3 | 256 | 4 | 32 | clustered | q_outer | 16 | 4 | 280.30 | 64 | 7557 | 128 | 120912 | 0 | 500 | 123.0 | 128.0 | 0.000 |
| 4 | 256 | 4 | 32 | clustered | q_outer | 16 | 4 | 281.48 | 64 | 7434 | 128 | 118944 | 0 | 509 | 124.0 | 128.0 | 0.002 |
| 0 | 256 | 4 | 32 | clustered | q_outer | 32 | 8 | 818.99 | 32 | 6591 | 256 | 210912 | 0 | 509 | 209.0 | 234.0 | 0.000 |
| 1 | 256 | 4 | 32 | clustered | q_outer | 32 | 8 | 807.33 | 32 | 6589 | 255 | 210848 | 0 | 509 | 208.0 | 229.8 | 0.000 |
| 2 | 256 | 4 | 32 | clustered | q_outer | 32 | 8 | 814.88 | 32 | 6550 | 256 | 209600 | 0 | 510 | 200.0 | 241.8 | 0.000 |
| 3 | 256 | 4 | 32 | clustered | q_outer | 32 | 8 | 802.65 | 32 | 6528 | 250 | 208896 | 0 | 500 | 206.0 | 241.8 | 0.000 |
| 4 | 256 | 4 | 32 | clustered | q_outer | 32 | 8 | 771.05 | 32 | 6459 | 239 | 206688 | 0 | 509 | 202.5 | 229.9 | 0.000 |
| 0 | 256 | 4 | 32 | clustered | q_outer | 64 | 16 | 1774.25 | 16 | 5179 | 382 | 331456 | 0 | 509 | 325.0 | 373.5 | 0.000 |
| 1 | 256 | 4 | 32 | clustered | q_outer | 64 | 16 | 1742.80 | 16 | 5215 | 374 | 333760 | 0 | 509 | 327.0 | 355.0 | 0.000 |
| 2 | 256 | 4 | 32 | clustered | q_outer | 64 | 16 | 1744.98 | 16 | 5203 | 378 | 332992 | 0 | 510 | 331.5 | 368.0 | 0.000 |
| 3 | 256 | 4 | 32 | clustered | q_outer | 64 | 16 | 1788.42 | 16 | 5389 | 387 | 344896 | 0 | 500 | 345.5 | 373.0 | 0.000 |
| 4 | 256 | 4 | 32 | clustered | q_outer | 64 | 16 | 1776.63 | 16 | 5356 | 383 | 342784 | 0 | 509 | 340.5 | 375.5 | 0.000 |
| 0 | 256 | 4 | 32 | clustered | q_outer | 128 | 32 | 5148.96 | 8 | 3560 | 472 | 455680 | 0 | 509 | 446.0 | 465.7 | 0.000 |
| 1 | 256 | 4 | 32 | clustered | q_outer | 128 | 32 | 5286.56 | 8 | 3501 | 485 | 448128 | 0 | 509 | 439.5 | 461.9 | 0.000 |
| 2 | 256 | 4 | 32 | clustered | q_outer | 128 | 32 | 5123.60 | 8 | 3473 | 469 | 444544 | 0 | 510 | 440.0 | 468.3 | 0.000 |
| 3 | 256 | 4 | 32 | clustered | q_outer | 128 | 32 | 5184.87 | 8 | 3559 | 476 | 455552 | 0 | 500 | 444.5 | 475.3 | 0.000 |
| 4 | 256 | 4 | 32 | clustered | q_outer | 128 | 32 | 5152.55 | 8 | 3549 | 473 | 454272 | 0 | 509 | 444.0 | 469.5 | 0.000 |
| 0 | 256 | 4 | 32 | clustered | signature | 16 | 4 | 127.18 | 299 | 7459 | 32 | 119344 | 79 | 509 | 32.0 | 32.0 | 1.000 |
| 1 | 256 | 4 | 32 | clustered | signature | 16 | 2 | 107.48 | 298 | 7574 | 32 | 121184 | 83 | 509 | 32.0 | 32.0 | 1.000 |
| 2 | 256 | 4 | 32 | clustered | signature | 16 | 4 | 130.22 | 309 | 7360 | 32 | 117760 | 104 | 510 | 31.0 | 32.0 | 1.000 |
| 3 | 256 | 4 | 32 | clustered | signature | 16 | 2 | 108.64 | 294 | 7557 | 32 | 120912 | 79 | 500 | 32.0 | 32.0 | 1.000 |
| 4 | 256 | 4 | 32 | clustered | signature | 16 | 4 | 132.87 | 304 | 7434 | 32 | 118944 | 85 | 509 | 32.0 | 32.0 | 1.000 |
| 0 | 256 | 4 | 32 | clustered | signature | 32 | 8 | 174.27 | 347 | 6591 | 32 | 105648 | 153 | 509 | 19.0 | 32.0 | 1.000 |
| 1 | 256 | 4 | 32 | clustered | signature | 32 | 4 | 131.71 | 340 | 6589 | 32 | 105424 | 153 | 509 | 19.0 | 32.0 | 1.000 |
| 2 | 256 | 4 | 32 | clustered | signature | 32 | 4 | 125.17 | 345 | 6550 | 32 | 104800 | 152 | 510 | 20.0 | 32.0 | 1.000 |
| 3 | 256 | 4 | 32 | clustered | signature | 32 | 4 | 127.28 | 344 | 6528 | 32 | 104448 | 157 | 500 | 21.0 | 32.0 | 1.000 |
| 4 | 256 | 4 | 32 | clustered | signature | 32 | 8 | 175.21 | 349 | 6459 | 32 | 103680 | 151 | 509 | 17.0 | 32.0 | 1.000 |
| 0 | 256 | 4 | 32 | clustered | signature | 64 | 8 | 168.47 | 409 | 5179 | 32 | 83488 | 223 | 509 | 10.0 | 29.0 | 1.000 |
| 1 | 256 | 4 | 32 | clustered | signature | 64 | 8 | 162.90 | 402 | 5215 | 32 | 83680 | 228 | 509 | 11.0 | 27.0 | 1.000 |
| 2 | 256 | 4 | 32 | clustered | signature | 64 | 4 | 122.16 | 393 | 5203 | 32 | 83248 | 218 | 510 | 10.0 | 31.0 | 1.000 |
| 3 | 256 | 4 | 32 | clustered | signature | 64 | 8 | 168.05 | 395 | 5389 | 32 | 86928 | 210 | 500 | 11.0 | 32.0 | 1.000 |
| 4 | 256 | 4 | 32 | clustered | signature | 64 | 8 | 169.78 | 399 | 5356 | 32 | 86384 | 209 | 509 | 11.0 | 32.0 | 1.000 |
| 0 | 256 | 4 | 32 | clustered | signature | 128 | 8 | 151.51 | 445 | 3560 | 32 | 60032 | 253 | 509 | 6.0 | 17.0 | 1.000 |
| 1 | 256 | 4 | 32 | clustered | signature | 128 | 8 | 134.98 | 452 | 3501 | 32 | 59024 | 255 | 509 | 6.0 | 17.0 | 1.000 |
| 2 | 256 | 4 | 32 | clustered | signature | 128 | 8 | 142.01 | 439 | 3473 | 32 | 59040 | 253 | 510 | 6.0 | 19.0 | 1.000 |
| 3 | 256 | 4 | 32 | clustered | signature | 128 | 8 | 140.85 | 453 | 3559 | 32 | 59536 | 251 | 500 | 6.0 | 17.0 | 1.000 |
| 4 | 256 | 4 | 32 | clustered | signature | 128 | 8 | 158.52 | 452 | 3549 | 32 | 59936 | 248 | 509 | 6.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 175.69 | 64 | 5116 | 80 | 81856 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 1 | 256 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 175.26 | 64 | 5116 | 80 | 81856 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 2 | 256 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 177.18 | 64 | 5115 | 80 | 81840 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 3 | 256 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 178.27 | 64 | 5116 | 80 | 81856 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 4 | 256 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 176.64 | 64 | 5116 | 80 | 81856 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 0 | 256 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 138.80 | 288 | 4604 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 138.65 | 288 | 4604 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 138.99 | 288 | 4603 | 16 | 81840 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 139.52 | 288 | 4604 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 139.75 | 288 | 4604 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 241.70 | 272 | 4348 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 242.77 | 272 | 4348 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 241.30 | 272 | 4347 | 16 | 81840 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 242.48 | 272 | 4348 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 242.87 | 272 | 4348 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 418.25 | 256 | 4092 | 16 | 79808 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 418.45 | 256 | 4092 | 16 | 79808 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 418.42 | 256 | 4092 | 16 | 79808 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 418.53 | 256 | 4092 | 16 | 79808 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 418.34 | 256 | 4092 | 16 | 79808 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 32 | common_private | q_outer | 16 | 4 | 179.53 | 64 | 5116 | 80 | 81856 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 1 | 256 | 4 | 32 | common_private | q_outer | 16 | 4 | 182.08 | 64 | 5116 | 80 | 81856 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 2 | 256 | 4 | 32 | common_private | q_outer | 16 | 4 | 183.75 | 64 | 5115 | 80 | 81840 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 3 | 256 | 4 | 32 | common_private | q_outer | 16 | 4 | 184.32 | 64 | 5116 | 80 | 81856 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 4 | 256 | 4 | 32 | common_private | q_outer | 16 | 4 | 182.12 | 64 | 5116 | 80 | 81856 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 0 | 256 | 4 | 32 | common_private | q_outer | 32 | 8 | 473.54 | 32 | 4604 | 144 | 147328 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 1 | 256 | 4 | 32 | common_private | q_outer | 32 | 8 | 478.55 | 32 | 4604 | 144 | 147328 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 2 | 256 | 4 | 32 | common_private | q_outer | 32 | 8 | 479.73 | 32 | 4603 | 144 | 147296 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 3 | 256 | 4 | 32 | common_private | q_outer | 32 | 8 | 479.24 | 32 | 4604 | 144 | 147328 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 4 | 256 | 4 | 32 | common_private | q_outer | 32 | 8 | 471.93 | 32 | 4604 | 144 | 147328 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 0 | 256 | 4 | 32 | common_private | q_outer | 64 | 16 | 1295.63 | 16 | 4348 | 272 | 278272 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 1 | 256 | 4 | 32 | common_private | q_outer | 64 | 16 | 1289.19 | 16 | 4348 | 272 | 278272 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 2 | 256 | 4 | 32 | common_private | q_outer | 64 | 16 | 1299.97 | 16 | 4347 | 272 | 278208 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 3 | 256 | 4 | 32 | common_private | q_outer | 64 | 16 | 1296.07 | 16 | 4348 | 272 | 278272 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 4 | 256 | 4 | 32 | common_private | q_outer | 64 | 16 | 1298.33 | 16 | 4348 | 272 | 278272 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 0 | 256 | 4 | 32 | common_private | q_outer | 128 | 32 | 5865.09 | 8 | 4092 | 512 | 523776 | 0 | 512 | 511.5 | 512.0 | 0.031 |
| 1 | 256 | 4 | 32 | common_private | q_outer | 128 | 32 | 5870.44 | 8 | 4092 | 512 | 523776 | 0 | 512 | 511.5 | 512.0 | 0.031 |
| 2 | 256 | 4 | 32 | common_private | q_outer | 128 | 32 | 5873.00 | 8 | 4092 | 512 | 523776 | 0 | 512 | 511.5 | 512.0 | 0.031 |
| 3 | 256 | 4 | 32 | common_private | q_outer | 128 | 32 | 5884.01 | 8 | 4092 | 512 | 523776 | 0 | 512 | 511.5 | 512.0 | 0.031 |
| 4 | 256 | 4 | 32 | common_private | q_outer | 128 | 32 | 5878.90 | 8 | 4092 | 512 | 523776 | 0 | 512 | 511.5 | 512.0 | 0.031 |
| 0 | 256 | 4 | 32 | common_private | signature | 16 | 4 | 107.83 | 320 | 5116 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 4 | 32 | common_private | signature | 16 | 4 | 107.99 | 320 | 5116 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 4 | 32 | common_private | signature | 16 | 4 | 108.48 | 320 | 5115 | 16 | 81840 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 4 | 32 | common_private | signature | 16 | 4 | 108.77 | 320 | 5116 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 4 | 32 | common_private | signature | 16 | 4 | 108.88 | 320 | 5116 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 32 | common_private | signature | 32 | 8 | 139.10 | 288 | 4604 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 4 | 32 | common_private | signature | 32 | 8 | 138.13 | 288 | 4604 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 4 | 32 | common_private | signature | 32 | 8 | 139.90 | 288 | 4603 | 16 | 81840 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 4 | 32 | common_private | signature | 32 | 8 | 139.42 | 288 | 4604 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 4 | 32 | common_private | signature | 32 | 8 | 139.08 | 288 | 4604 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 32 | common_private | signature | 64 | 16 | 242.15 | 272 | 4348 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 4 | 32 | common_private | signature | 64 | 16 | 243.54 | 272 | 4348 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 4 | 32 | common_private | signature | 64 | 16 | 241.93 | 272 | 4347 | 16 | 81840 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 4 | 32 | common_private | signature | 64 | 16 | 242.39 | 272 | 4348 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 4 | 32 | common_private | signature | 64 | 16 | 243.13 | 272 | 4348 | 16 | 81856 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 32 | common_private | signature | 128 | 32 | 418.22 | 256 | 4092 | 16 | 79808 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 4 | 32 | common_private | signature | 128 | 32 | 418.75 | 256 | 4092 | 16 | 79808 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 4 | 32 | common_private | signature | 128 | 32 | 418.56 | 256 | 4092 | 16 | 79808 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 4 | 32 | common_private | signature | 128 | 32 | 418.59 | 256 | 4092 | 16 | 79808 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 4 | 32 | common_private | signature | 128 | 32 | 418.53 | 256 | 4092 | 16 | 79808 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 4 | 32 | random | bounded_merge | 16 | 4 | 257.21 | 64 | 7428 | 123 | 118848 | 0 | 512 | 116.0 | 120.0 | 0.000 |
| 1 | 256 | 4 | 32 | random | bounded_merge | 16 | 4 | 260.64 | 64 | 7410 | 124 | 118560 | 0 | 512 | 116.0 | 119.0 | 0.000 |
| 2 | 256 | 4 | 32 | random | bounded_merge | 16 | 4 | 256.64 | 64 | 7467 | 123 | 119472 | 0 | 512 | 117.0 | 119.7 | 0.000 |
| 3 | 256 | 4 | 32 | random | bounded_merge | 16 | 4 | 255.82 | 64 | 7430 | 123 | 118880 | 0 | 512 | 116.0 | 120.0 | 0.000 |
| 4 | 256 | 4 | 32 | random | bounded_merge | 16 | 4 | 264.42 | 64 | 7496 | 122 | 119936 | 0 | 512 | 117.0 | 120.0 | 0.000 |
| 0 | 256 | 4 | 32 | random | bounded_merge | 32 | 8 | 194.54 | 1088 | 6594 | 27 | 105536 | 256 | 512 | 2.0 | 21.0 | 0.967 |
| 1 | 256 | 4 | 32 | random | bounded_merge | 32 | 8 | 188.51 | 1097 | 6597 | 29 | 105584 | 256 | 512 | 2.0 | 21.0 | 0.972 |
| 2 | 256 | 4 | 32 | random | bounded_merge | 32 | 4 | 131.30 | 1086 | 6619 | 28 | 105904 | 256 | 512 | 2.0 | 21.0 | 0.978 |
| 3 | 256 | 4 | 32 | random | bounded_merge | 32 | 4 | 135.31 | 1124 | 6585 | 27 | 105360 | 256 | 512 | 2.0 | 21.0 | 0.979 |
| 4 | 256 | 4 | 32 | random | bounded_merge | 32 | 4 | 134.02 | 1072 | 6626 | 28 | 106016 | 256 | 512 | 2.0 | 21.0 | 0.973 |
| 0 | 256 | 4 | 32 | random | bounded_merge | 64 | 8 | 162.90 | 1943 | 5230 | 19 | 83968 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 256 | 4 | 32 | random | bounded_merge | 64 | 8 | 160.16 | 1942 | 5282 | 21 | 84800 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 256 | 4 | 32 | random | bounded_merge | 64 | 8 | 167.78 | 1908 | 5307 | 20 | 85360 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 256 | 4 | 32 | random | bounded_merge | 64 | 8 | 170.90 | 1951 | 5252 | 21 | 84288 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 256 | 4 | 32 | random | bounded_merge | 64 | 8 | 176.36 | 1946 | 5279 | 20 | 84736 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 256 | 4 | 32 | random | bounded_merge | 128 | 8 | 135.70 | 2569 | 3543 | 10 | 59792 | 256 | 512 | 1.0 | 2.0 | 1.000 |
| 1 | 256 | 4 | 32 | random | bounded_merge | 128 | 16 | 258.67 | 2559 | 3574 | 11 | 60256 | 256 | 512 | 1.0 | 2.0 | 1.000 |
| 2 | 256 | 4 | 32 | random | bounded_merge | 128 | 8 | 138.16 | 2577 | 3573 | 11 | 60240 | 256 | 512 | 1.0 | 2.0 | 1.000 |
| 3 | 256 | 4 | 32 | random | bounded_merge | 128 | 16 | 242.34 | 2560 | 3549 | 12 | 60240 | 256 | 512 | 1.0 | 2.0 | 1.000 |
| 4 | 256 | 4 | 32 | random | bounded_merge | 128 | 16 | 238.30 | 2588 | 3580 | 10 | 60224 | 256 | 512 | 1.0 | 2.0 | 1.000 |
| 0 | 256 | 4 | 32 | random | q_outer | 16 | 4 | 265.81 | 64 | 7428 | 123 | 118848 | 0 | 512 | 116.0 | 120.0 | 0.000 |
| 1 | 256 | 4 | 32 | random | q_outer | 16 | 4 | 268.04 | 64 | 7410 | 124 | 118560 | 0 | 512 | 116.0 | 119.0 | 0.000 |
| 2 | 256 | 4 | 32 | random | q_outer | 16 | 4 | 263.85 | 64 | 7467 | 123 | 119472 | 0 | 512 | 117.0 | 119.7 | 0.000 |
| 3 | 256 | 4 | 32 | random | q_outer | 16 | 4 | 267.36 | 64 | 7430 | 123 | 118880 | 0 | 512 | 116.0 | 120.0 | 0.000 |
| 4 | 256 | 4 | 32 | random | q_outer | 16 | 4 | 265.36 | 64 | 7496 | 122 | 119936 | 0 | 512 | 117.0 | 120.0 | 0.000 |
| 0 | 256 | 4 | 32 | random | q_outer | 32 | 8 | 690.33 | 32 | 6594 | 214 | 211008 | 0 | 512 | 206.0 | 212.9 | 0.000 |
| 1 | 256 | 4 | 32 | random | q_outer | 32 | 8 | 692.38 | 32 | 6597 | 216 | 211104 | 0 | 512 | 207.0 | 211.9 | 0.000 |
| 2 | 256 | 4 | 32 | random | q_outer | 32 | 8 | 687.18 | 32 | 6619 | 214 | 211808 | 0 | 512 | 207.0 | 211.0 | 0.000 |
| 3 | 256 | 4 | 32 | random | q_outer | 32 | 8 | 689.09 | 32 | 6585 | 216 | 210720 | 0 | 512 | 206.0 | 212.0 | 0.000 |
| 4 | 256 | 4 | 32 | random | q_outer | 32 | 8 | 709.08 | 32 | 6626 | 222 | 212032 | 0 | 512 | 205.0 | 212.9 | 0.000 |
| 0 | 256 | 4 | 32 | random | q_outer | 64 | 16 | 1581.97 | 16 | 5230 | 339 | 334720 | 0 | 512 | 326.5 | 335.5 | 0.000 |
| 1 | 256 | 4 | 32 | random | q_outer | 64 | 16 | 1597.99 | 16 | 5282 | 345 | 338048 | 0 | 512 | 329.5 | 339.5 | 0.000 |
| 2 | 256 | 4 | 32 | random | q_outer | 64 | 16 | 1601.61 | 16 | 5307 | 346 | 339648 | 0 | 512 | 330.0 | 340.0 | 0.000 |
| 3 | 256 | 4 | 32 | random | q_outer | 64 | 16 | 1593.94 | 16 | 5252 | 342 | 336128 | 0 | 512 | 331.0 | 336.5 | 0.000 |
| 4 | 256 | 4 | 32 | random | q_outer | 64 | 16 | 1631.68 | 16 | 5279 | 352 | 337856 | 0 | 512 | 329.0 | 339.5 | 0.000 |
| 0 | 256 | 4 | 32 | random | q_outer | 128 | 32 | 4940.96 | 8 | 3543 | 453 | 453504 | 0 | 512 | 441.5 | 448.8 | 0.000 |
| 1 | 256 | 4 | 32 | random | q_outer | 128 | 32 | 5035.31 | 8 | 3574 | 462 | 457472 | 0 | 512 | 444.5 | 457.1 | 0.000 |
| 2 | 256 | 4 | 32 | random | q_outer | 128 | 32 | 5034.22 | 8 | 3573 | 457 | 457344 | 0 | 512 | 447.0 | 452.8 | 0.000 |
| 3 | 256 | 4 | 32 | random | q_outer | 128 | 32 | 4956.28 | 8 | 3549 | 454 | 454272 | 0 | 512 | 444.5 | 451.9 | 0.000 |
| 4 | 256 | 4 | 32 | random | q_outer | 128 | 32 | 4952.50 | 8 | 3580 | 454 | 458240 | 0 | 512 | 446.5 | 454.0 | 0.000 |
| 0 | 256 | 4 | 32 | random | signature | 16 | 4 | 131.36 | 612 | 7428 | 31 | 118848 | 256 | 512 | 3.0 | 28.0 | 1.000 |
| 1 | 256 | 4 | 32 | random | signature | 16 | 4 | 135.21 | 621 | 7410 | 32 | 118560 | 255 | 512 | 3.0 | 28.0 | 1.000 |
| 2 | 256 | 4 | 32 | random | signature | 16 | 4 | 135.70 | 602 | 7467 | 31 | 119472 | 256 | 512 | 3.0 | 28.0 | 1.000 |
| 3 | 256 | 4 | 32 | random | signature | 16 | 4 | 127.85 | 620 | 7430 | 32 | 118880 | 255 | 512 | 3.0 | 28.0 | 1.000 |
| 4 | 256 | 4 | 32 | random | signature | 16 | 4 | 131.69 | 599 | 7496 | 32 | 119936 | 255 | 512 | 3.0 | 28.0 | 1.000 |
| 0 | 256 | 4 | 32 | random | signature | 32 | 8 | 185.96 | 1098 | 6594 | 27 | 105536 | 256 | 512 | 2.0 | 21.0 | 1.000 |
| 1 | 256 | 4 | 32 | random | signature | 32 | 8 | 182.78 | 1105 | 6597 | 28 | 105584 | 256 | 512 | 2.0 | 21.0 | 1.000 |
| 2 | 256 | 4 | 32 | random | signature | 32 | 4 | 139.84 | 1093 | 6619 | 26 | 105904 | 256 | 512 | 2.0 | 21.0 | 1.000 |
| 3 | 256 | 4 | 32 | random | signature | 32 | 4 | 139.90 | 1130 | 6585 | 26 | 105360 | 256 | 512 | 2.0 | 21.0 | 1.000 |
| 4 | 256 | 4 | 32 | random | signature | 32 | 4 | 139.09 | 1080 | 6626 | 28 | 106016 | 256 | 512 | 2.0 | 21.0 | 1.000 |
| 0 | 256 | 4 | 32 | random | signature | 64 | 8 | 163.92 | 1943 | 5230 | 19 | 83968 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 256 | 4 | 32 | random | signature | 64 | 8 | 159.48 | 1942 | 5282 | 21 | 84800 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 256 | 4 | 32 | random | signature | 64 | 8 | 162.13 | 1908 | 5307 | 20 | 85360 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 256 | 4 | 32 | random | signature | 64 | 8 | 165.02 | 1951 | 5252 | 21 | 84288 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 256 | 4 | 32 | random | signature | 64 | 8 | 170.30 | 1946 | 5279 | 20 | 84736 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 256 | 4 | 32 | random | signature | 128 | 8 | 130.81 | 2569 | 3543 | 10 | 59792 | 256 | 512 | 1.0 | 2.0 | 1.000 |
| 1 | 256 | 4 | 32 | random | signature | 128 | 16 | 249.47 | 2559 | 3574 | 11 | 60256 | 256 | 512 | 1.0 | 2.0 | 1.000 |
| 2 | 256 | 4 | 32 | random | signature | 128 | 8 | 134.23 | 2577 | 3573 | 11 | 60240 | 256 | 512 | 1.0 | 2.0 | 1.000 |
| 3 | 256 | 4 | 32 | random | signature | 128 | 16 | 234.09 | 2560 | 3549 | 12 | 60240 | 256 | 512 | 1.0 | 2.0 | 1.000 |
| 4 | 256 | 4 | 32 | random | signature | 128 | 16 | 238.46 | 2588 | 3580 | 10 | 60224 | 256 | 512 | 1.0 | 2.0 | 1.000 |
| 0 | 256 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 40.28 | 254 | 2039 | 12 | 32624 | 0 | 502 | 8.0 | 8.0 | 0.993 |
| 1 | 256 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 48.08 | 253 | 2045 | 15 | 32720 | 0 | 498 | 8.0 | 8.0 | 0.979 |
| 2 | 256 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 48.69 | 252 | 2035 | 15 | 32560 | 0 | 494 | 8.0 | 8.0 | 0.981 |
| 3 | 256 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 46.14 | 252 | 2035 | 14 | 32560 | 0 | 494 | 8.0 | 8.0 | 0.981 |
| 4 | 256 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 48.06 | 253 | 2041 | 15 | 32656 | 0 | 500 | 8.0 | 8.0 | 0.983 |
| 0 | 256 | 8 | 8 | clustered | bounded_merge | 32 | 2 | 53.08 | 249 | 2012 | 15 | 32192 | 1 | 502 | 8.0 | 8.0 | 0.976 |
| 1 | 256 | 8 | 8 | clustered | bounded_merge | 32 | 2 | 48.33 | 251 | 2032 | 15 | 32512 | 0 | 498 | 8.0 | 8.0 | 0.976 |
| 2 | 256 | 8 | 8 | clustered | bounded_merge | 32 | 2 | 48.14 | 243 | 1995 | 15 | 31920 | 0 | 494 | 8.0 | 8.0 | 0.949 |
| 3 | 256 | 8 | 8 | clustered | bounded_merge | 32 | 2 | 46.08 | 246 | 2003 | 14 | 32048 | 0 | 494 | 8.0 | 8.0 | 0.965 |
| 4 | 256 | 8 | 8 | clustered | bounded_merge | 32 | 4 | 64.76 | 250 | 2014 | 15 | 32304 | 3 | 500 | 8.0 | 8.0 | 0.972 |
| 0 | 256 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 46.37 | 259 | 1956 | 8 | 31328 | 26 | 502 | 8.0 | 8.0 | 0.962 |
| 1 | 256 | 8 | 8 | clustered | bounded_merge | 64 | 2 | 37.21 | 258 | 1951 | 8 | 31216 | 26 | 498 | 8.0 | 8.0 | 0.963 |
| 2 | 256 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 47.25 | 260 | 1930 | 8 | 31024 | 32 | 494 | 8.0 | 8.0 | 0.967 |
| 3 | 256 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 47.06 | 262 | 1898 | 8 | 30496 | 34 | 494 | 8.0 | 8.0 | 0.973 |
| 4 | 256 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 46.92 | 261 | 1943 | 8 | 31216 | 28 | 500 | 8.0 | 8.0 | 0.964 |
| 0 | 256 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 54.33 | 296 | 1824 | 8 | 29440 | 84 | 502 | 8.0 | 8.0 | 0.975 |
| 1 | 256 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 51.55 | 285 | 1824 | 8 | 29296 | 77 | 498 | 8.0 | 8.0 | 0.980 |
| 2 | 256 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 53.86 | 290 | 1797 | 8 | 29184 | 89 | 494 | 8.0 | 8.0 | 0.977 |
| 3 | 256 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 53.31 | 289 | 1802 | 8 | 29072 | 78 | 494 | 8.0 | 8.0 | 0.982 |
| 4 | 256 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 53.41 | 286 | 1854 | 8 | 30000 | 70 | 500 | 8.0 | 8.0 | 0.979 |
| 0 | 256 | 8 | 8 | clustered | q_outer | 16 | 2 | 48.37 | 128 | 2039 | 16 | 32624 | 0 | 502 | 16.0 | 16.0 | 0.004 |
| 1 | 256 | 8 | 8 | clustered | q_outer | 16 | 2 | 49.84 | 128 | 2045 | 16 | 32720 | 0 | 498 | 16.0 | 16.0 | 0.001 |
| 2 | 256 | 8 | 8 | clustered | q_outer | 16 | 2 | 49.03 | 128 | 2035 | 16 | 32560 | 0 | 494 | 16.0 | 16.0 | 0.006 |
| 3 | 256 | 8 | 8 | clustered | q_outer | 16 | 2 | 49.30 | 128 | 2035 | 16 | 32560 | 0 | 494 | 16.0 | 16.0 | 0.006 |
| 4 | 256 | 8 | 8 | clustered | q_outer | 16 | 2 | 49.01 | 128 | 2041 | 16 | 32656 | 0 | 500 | 16.0 | 16.0 | 0.003 |
| 0 | 256 | 8 | 8 | clustered | q_outer | 32 | 4 | 108.63 | 64 | 2012 | 32 | 64384 | 0 | 502 | 32.0 | 32.0 | 0.000 |
| 1 | 256 | 8 | 8 | clustered | q_outer | 32 | 4 | 107.30 | 64 | 2032 | 32 | 65024 | 0 | 498 | 32.0 | 32.0 | 0.000 |
| 2 | 256 | 8 | 8 | clustered | q_outer | 32 | 4 | 107.92 | 64 | 1995 | 32 | 63840 | 0 | 494 | 32.0 | 32.0 | 0.000 |
| 3 | 256 | 8 | 8 | clustered | q_outer | 32 | 4 | 106.93 | 64 | 2003 | 32 | 64096 | 0 | 494 | 32.0 | 32.0 | 0.000 |
| 4 | 256 | 8 | 8 | clustered | q_outer | 32 | 4 | 107.88 | 64 | 2014 | 32 | 64448 | 0 | 500 | 32.0 | 32.0 | 0.000 |
| 0 | 256 | 8 | 8 | clustered | q_outer | 64 | 8 | 306.50 | 32 | 1956 | 64 | 125184 | 0 | 502 | 62.0 | 64.0 | 0.000 |
| 1 | 256 | 8 | 8 | clustered | q_outer | 64 | 8 | 311.11 | 32 | 1951 | 64 | 124864 | 0 | 498 | 63.0 | 64.0 | 0.000 |
| 2 | 256 | 8 | 8 | clustered | q_outer | 64 | 8 | 309.56 | 32 | 1930 | 64 | 123520 | 0 | 494 | 61.0 | 64.0 | 0.000 |
| 3 | 256 | 8 | 8 | clustered | q_outer | 64 | 8 | 308.78 | 32 | 1898 | 64 | 121472 | 0 | 494 | 60.5 | 64.0 | 0.000 |
| 4 | 256 | 8 | 8 | clustered | q_outer | 64 | 8 | 309.20 | 32 | 1943 | 64 | 124352 | 0 | 500 | 61.0 | 64.0 | 0.000 |
| 0 | 256 | 8 | 8 | clustered | q_outer | 128 | 16 | 1626.67 | 16 | 1824 | 126 | 233472 | 0 | 502 | 113.0 | 123.5 | 0.000 |
| 1 | 256 | 8 | 8 | clustered | q_outer | 128 | 16 | 1573.53 | 16 | 1824 | 122 | 233472 | 0 | 498 | 115.0 | 121.0 | 0.000 |
| 2 | 256 | 8 | 8 | clustered | q_outer | 128 | 16 | 1635.61 | 16 | 1797 | 128 | 230016 | 0 | 494 | 113.0 | 120.5 | 0.000 |
| 3 | 256 | 8 | 8 | clustered | q_outer | 128 | 16 | 1547.19 | 16 | 1802 | 120 | 230656 | 0 | 494 | 115.0 | 119.0 | 0.000 |
| 4 | 256 | 8 | 8 | clustered | q_outer | 128 | 16 | 1614.65 | 16 | 1854 | 127 | 237312 | 0 | 500 | 117.0 | 121.5 | 0.000 |
| 0 | 256 | 8 | 8 | clustered | signature | 16 | 2 | 36.51 | 258 | 2039 | 8 | 32624 | 4 | 502 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 8 | 8 | clustered | signature | 16 | 2 | 36.49 | 259 | 2045 | 8 | 32720 | 6 | 498 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 8 | 8 | clustered | signature | 16 | 2 | 36.77 | 260 | 2035 | 8 | 32560 | 8 | 494 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 8 | 8 | clustered | signature | 16 | 2 | 37.18 | 260 | 2035 | 8 | 32560 | 8 | 494 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 8 | 8 | clustered | signature | 16 | 2 | 36.91 | 259 | 2041 | 8 | 32656 | 6 | 500 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 8 | 8 | clustered | signature | 32 | 2 | 36.82 | 263 | 2012 | 8 | 32192 | 15 | 502 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 8 | 8 | clustered | signature | 32 | 2 | 36.87 | 259 | 2032 | 8 | 32512 | 8 | 498 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 8 | 8 | clustered | signature | 32 | 2 | 39.99 | 265 | 1995 | 8 | 31920 | 22 | 494 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 8 | 8 | clustered | signature | 32 | 2 | 37.20 | 264 | 2003 | 8 | 32048 | 18 | 494 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 8 | 8 | clustered | signature | 32 | 4 | 46.27 | 264 | 2014 | 8 | 32304 | 15 | 500 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 8 | 8 | clustered | signature | 64 | 4 | 50.49 | 276 | 1956 | 8 | 31328 | 42 | 502 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 8 | 8 | clustered | signature | 64 | 2 | 40.24 | 275 | 1951 | 8 | 31216 | 43 | 498 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 8 | 8 | clustered | signature | 64 | 4 | 52.85 | 277 | 1930 | 8 | 31024 | 47 | 494 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 8 | 8 | clustered | signature | 64 | 4 | 50.78 | 279 | 1898 | 8 | 30496 | 48 | 494 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 8 | 8 | clustered | signature | 64 | 4 | 52.55 | 280 | 1943 | 8 | 31216 | 45 | 500 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 8 | 8 | clustered | signature | 128 | 4 | 54.09 | 306 | 1824 | 8 | 29440 | 93 | 502 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 8 | 8 | clustered | signature | 128 | 4 | 52.13 | 295 | 1824 | 8 | 29296 | 85 | 498 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 8 | 8 | clustered | signature | 128 | 4 | 54.75 | 300 | 1797 | 8 | 29184 | 98 | 494 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 8 | 8 | clustered | signature | 128 | 4 | 52.93 | 299 | 1802 | 8 | 29072 | 87 | 494 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 8 | 8 | clustered | signature | 128 | 4 | 54.10 | 297 | 1854 | 8 | 30000 | 80 | 500 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 38.54 | 128 | 1535 | 12 | 24560 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 1 | 256 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 37.56 | 128 | 1535 | 12 | 24560 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 2 | 256 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 39.34 | 128 | 1535 | 12 | 24560 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 3 | 256 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 38.65 | 128 | 1535 | 12 | 24560 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 4 | 256 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 38.03 | 128 | 1535 | 12 | 24560 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 0 | 256 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 46.05 | 320 | 1279 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 256 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 45.84 | 320 | 1279 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 256 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 46.53 | 320 | 1279 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 256 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 45.86 | 320 | 1279 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 256 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 46.14 | 320 | 1279 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 256 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 68.10 | 288 | 1151 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 256 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 67.85 | 288 | 1151 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 256 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 68.52 | 288 | 1151 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 256 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 68.28 | 288 | 1151 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 256 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 68.26 | 288 | 1151 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 256 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 181.70 | 272 | 1087 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 256 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 181.29 | 272 | 1087 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 256 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 180.62 | 272 | 1087 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 256 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 179.56 | 272 | 1087 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 256 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 180.17 | 272 | 1087 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 256 | 8 | 8 | common_private | q_outer | 16 | 2 | 39.43 | 128 | 1535 | 12 | 24560 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 1 | 256 | 8 | 8 | common_private | q_outer | 16 | 2 | 37.74 | 128 | 1535 | 12 | 24560 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 2 | 256 | 8 | 8 | common_private | q_outer | 16 | 2 | 38.08 | 128 | 1535 | 12 | 24560 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 3 | 256 | 8 | 8 | common_private | q_outer | 16 | 2 | 39.93 | 128 | 1535 | 12 | 24560 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 4 | 256 | 8 | 8 | common_private | q_outer | 16 | 2 | 38.72 | 128 | 1535 | 12 | 24560 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 0 | 256 | 8 | 8 | common_private | q_outer | 32 | 4 | 71.82 | 64 | 1279 | 20 | 40928 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 1 | 256 | 8 | 8 | common_private | q_outer | 32 | 4 | 71.33 | 64 | 1279 | 20 | 40928 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 2 | 256 | 8 | 8 | common_private | q_outer | 32 | 4 | 72.21 | 64 | 1279 | 20 | 40928 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 3 | 256 | 8 | 8 | common_private | q_outer | 32 | 4 | 71.99 | 64 | 1279 | 20 | 40928 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 4 | 256 | 8 | 8 | common_private | q_outer | 32 | 4 | 72.14 | 64 | 1279 | 20 | 40928 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 0 | 256 | 8 | 8 | common_private | q_outer | 64 | 8 | 183.03 | 32 | 1151 | 36 | 73664 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 1 | 256 | 8 | 8 | common_private | q_outer | 64 | 8 | 179.69 | 32 | 1151 | 36 | 73664 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 2 | 256 | 8 | 8 | common_private | q_outer | 64 | 8 | 182.27 | 32 | 1151 | 36 | 73664 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 3 | 256 | 8 | 8 | common_private | q_outer | 64 | 8 | 181.83 | 32 | 1151 | 36 | 73664 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 4 | 256 | 8 | 8 | common_private | q_outer | 64 | 8 | 182.47 | 32 | 1151 | 36 | 73664 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 0 | 256 | 8 | 8 | common_private | q_outer | 128 | 16 | 909.68 | 16 | 1087 | 68 | 139136 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 1 | 256 | 8 | 8 | common_private | q_outer | 128 | 16 | 905.69 | 16 | 1087 | 68 | 139136 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 2 | 256 | 8 | 8 | common_private | q_outer | 128 | 16 | 900.51 | 16 | 1087 | 68 | 139136 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 3 | 256 | 8 | 8 | common_private | q_outer | 128 | 16 | 911.23 | 16 | 1087 | 68 | 139136 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 4 | 256 | 8 | 8 | common_private | q_outer | 128 | 16 | 907.77 | 16 | 1087 | 68 | 139136 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 0 | 256 | 8 | 8 | common_private | signature | 16 | 2 | 37.83 | 384 | 1535 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 256 | 8 | 8 | common_private | signature | 16 | 2 | 37.71 | 384 | 1535 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 256 | 8 | 8 | common_private | signature | 16 | 2 | 38.05 | 384 | 1535 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 256 | 8 | 8 | common_private | signature | 16 | 2 | 38.45 | 384 | 1535 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 256 | 8 | 8 | common_private | signature | 16 | 2 | 37.94 | 384 | 1535 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 256 | 8 | 8 | common_private | signature | 32 | 4 | 46.22 | 320 | 1279 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 256 | 8 | 8 | common_private | signature | 32 | 4 | 45.66 | 320 | 1279 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 256 | 8 | 8 | common_private | signature | 32 | 4 | 46.10 | 320 | 1279 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 256 | 8 | 8 | common_private | signature | 32 | 4 | 47.06 | 320 | 1279 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 256 | 8 | 8 | common_private | signature | 32 | 4 | 46.39 | 320 | 1279 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 256 | 8 | 8 | common_private | signature | 64 | 8 | 68.03 | 288 | 1151 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 256 | 8 | 8 | common_private | signature | 64 | 8 | 67.69 | 288 | 1151 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 256 | 8 | 8 | common_private | signature | 64 | 8 | 68.32 | 288 | 1151 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 256 | 8 | 8 | common_private | signature | 64 | 8 | 68.65 | 288 | 1151 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 256 | 8 | 8 | common_private | signature | 64 | 8 | 68.55 | 288 | 1151 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 256 | 8 | 8 | common_private | signature | 128 | 16 | 181.26 | 272 | 1087 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 256 | 8 | 8 | common_private | signature | 128 | 16 | 180.91 | 272 | 1087 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 256 | 8 | 8 | common_private | signature | 128 | 16 | 180.72 | 272 | 1087 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 256 | 8 | 8 | common_private | signature | 128 | 16 | 179.88 | 272 | 1087 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 256 | 8 | 8 | common_private | signature | 128 | 16 | 180.03 | 272 | 1087 | 4 | 24560 | 256 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 256 | 8 | 8 | random | bounded_merge | 16 | 2 | 48.00 | 236 | 2023 | 15 | 32368 | 0 | 500 | 8.0 | 8.0 | 0.864 |
| 1 | 256 | 8 | 8 | random | bounded_merge | 16 | 2 | 47.72 | 237 | 2025 | 15 | 32400 | 0 | 504 | 8.0 | 8.0 | 0.871 |
| 2 | 256 | 8 | 8 | random | bounded_merge | 16 | 2 | 47.83 | 243 | 2035 | 15 | 32560 | 0 | 499 | 8.0 | 8.0 | 0.911 |
| 3 | 256 | 8 | 8 | random | bounded_merge | 16 | 2 | 51.27 | 234 | 2023 | 15 | 32368 | 0 | 506 | 8.0 | 8.0 | 0.851 |
| 4 | 256 | 8 | 8 | random | bounded_merge | 16 | 2 | 50.08 | 239 | 2029 | 15 | 32464 | 0 | 504 | 8.0 | 8.0 | 0.884 |
| 0 | 256 | 8 | 8 | random | bounded_merge | 32 | 4 | 70.40 | 223 | 1990 | 15 | 31856 | 15 | 500 | 8.0 | 15.0 | 0.716 |
| 1 | 256 | 8 | 8 | random | bounded_merge | 32 | 4 | 67.21 | 227 | 2000 | 15 | 32016 | 14 | 504 | 8.0 | 14.4 | 0.759 |
| 2 | 256 | 8 | 8 | random | bounded_merge | 32 | 2 | 54.36 | 225 | 2004 | 15 | 32064 | 12 | 499 | 8.0 | 15.0 | 0.743 |
| 3 | 256 | 8 | 8 | random | bounded_merge | 32 | 4 | 65.74 | 228 | 1996 | 15 | 31952 | 18 | 506 | 8.0 | 15.0 | 0.741 |
| 4 | 256 | 8 | 8 | random | bounded_merge | 32 | 4 | 71.16 | 224 | 2000 | 15 | 32032 | 12 | 504 | 8.0 | 15.0 | 0.741 |
| 0 | 256 | 8 | 8 | random | bounded_merge | 64 | 4 | 54.93 | 336 | 1928 | 8 | 30880 | 146 | 500 | 7.0 | 8.0 | 0.901 |
| 1 | 256 | 8 | 8 | random | bounded_merge | 64 | 4 | 54.20 | 321 | 1946 | 8 | 31168 | 124 | 504 | 7.0 | 8.0 | 0.899 |
| 2 | 256 | 8 | 8 | random | bounded_merge | 64 | 4 | 55.34 | 325 | 1946 | 8 | 31152 | 127 | 499 | 7.0 | 8.0 | 0.899 |
| 3 | 256 | 8 | 8 | random | bounded_merge | 64 | 4 | 56.30 | 337 | 1931 | 8 | 30960 | 141 | 506 | 7.0 | 8.0 | 0.903 |
| 4 | 256 | 8 | 8 | random | bounded_merge | 64 | 4 | 55.22 | 323 | 1943 | 8 | 31152 | 133 | 504 | 7.0 | 8.0 | 0.890 |
| 0 | 256 | 8 | 8 | random | bounded_merge | 128 | 4 | 61.31 | 462 | 1794 | 8 | 29040 | 220 | 500 | 5.0 | 7.0 | 0.962 |
| 1 | 256 | 8 | 8 | random | bounded_merge | 128 | 4 | 60.60 | 436 | 1838 | 8 | 29568 | 209 | 504 | 5.0 | 8.0 | 0.961 |
| 2 | 256 | 8 | 8 | random | bounded_merge | 128 | 4 | 61.26 | 440 | 1831 | 8 | 29488 | 216 | 499 | 5.0 | 7.0 | 0.957 |
| 3 | 256 | 8 | 8 | random | bounded_merge | 128 | 4 | 62.46 | 455 | 1815 | 8 | 29248 | 218 | 506 | 5.0 | 7.0 | 0.952 |
| 4 | 256 | 8 | 8 | random | bounded_merge | 128 | 4 | 61.39 | 428 | 1840 | 8 | 29648 | 217 | 504 | 6.0 | 7.0 | 0.952 |
| 0 | 256 | 8 | 8 | random | q_outer | 16 | 2 | 49.34 | 128 | 2023 | 16 | 32368 | 0 | 500 | 16.0 | 16.0 | 0.011 |
| 1 | 256 | 8 | 8 | random | q_outer | 16 | 2 | 49.00 | 128 | 2025 | 16 | 32400 | 0 | 504 | 16.0 | 16.0 | 0.010 |
| 2 | 256 | 8 | 8 | random | q_outer | 16 | 2 | 49.82 | 128 | 2035 | 16 | 32560 | 0 | 499 | 16.0 | 16.0 | 0.006 |
| 3 | 256 | 8 | 8 | random | q_outer | 16 | 2 | 49.25 | 128 | 2023 | 16 | 32368 | 0 | 506 | 16.0 | 16.0 | 0.012 |
| 4 | 256 | 8 | 8 | random | q_outer | 16 | 2 | 49.46 | 128 | 2029 | 16 | 32464 | 0 | 504 | 16.0 | 16.0 | 0.009 |
| 0 | 256 | 8 | 8 | random | q_outer | 32 | 4 | 108.55 | 64 | 1990 | 32 | 63680 | 0 | 500 | 31.0 | 32.0 | 0.000 |
| 1 | 256 | 8 | 8 | random | q_outer | 32 | 4 | 108.34 | 64 | 2000 | 32 | 64000 | 0 | 504 | 31.0 | 32.0 | 0.000 |
| 2 | 256 | 8 | 8 | random | q_outer | 32 | 4 | 107.27 | 64 | 2004 | 32 | 64128 | 0 | 499 | 31.5 | 32.0 | 0.000 |
| 3 | 256 | 8 | 8 | random | q_outer | 32 | 4 | 107.66 | 64 | 1996 | 32 | 63872 | 0 | 506 | 31.0 | 32.0 | 0.000 |
| 4 | 256 | 8 | 8 | random | q_outer | 32 | 4 | 106.56 | 64 | 2000 | 32 | 64000 | 0 | 504 | 31.0 | 32.0 | 0.000 |
| 0 | 256 | 8 | 8 | random | q_outer | 64 | 8 | 308.34 | 32 | 1928 | 64 | 123392 | 0 | 500 | 60.0 | 62.0 | 0.000 |
| 1 | 256 | 8 | 8 | random | q_outer | 64 | 8 | 304.12 | 32 | 1946 | 64 | 124544 | 0 | 504 | 61.0 | 62.9 | 0.000 |
| 2 | 256 | 8 | 8 | random | q_outer | 64 | 8 | 302.40 | 32 | 1946 | 64 | 124544 | 0 | 499 | 61.0 | 63.0 | 0.000 |
| 3 | 256 | 8 | 8 | random | q_outer | 64 | 8 | 305.48 | 32 | 1931 | 64 | 123584 | 0 | 506 | 60.5 | 62.0 | 0.000 |
| 4 | 256 | 8 | 8 | random | q_outer | 64 | 8 | 301.54 | 32 | 1943 | 63 | 124352 | 0 | 504 | 61.0 | 62.0 | 0.000 |
| 0 | 256 | 8 | 8 | random | q_outer | 128 | 16 | 1491.08 | 16 | 1794 | 117 | 229632 | 0 | 500 | 112.0 | 115.5 | 0.000 |
| 1 | 256 | 8 | 8 | random | q_outer | 128 | 16 | 1542.00 | 16 | 1838 | 121 | 235264 | 0 | 504 | 115.5 | 117.5 | 0.000 |
| 2 | 256 | 8 | 8 | random | q_outer | 128 | 16 | 1545.80 | 16 | 1831 | 120 | 234368 | 0 | 499 | 115.0 | 118.0 | 0.000 |
| 3 | 256 | 8 | 8 | random | q_outer | 128 | 16 | 1509.46 | 16 | 1815 | 117 | 232320 | 0 | 506 | 114.0 | 116.5 | 0.000 |
| 4 | 256 | 8 | 8 | random | q_outer | 128 | 16 | 1521.93 | 16 | 1840 | 119 | 235520 | 0 | 504 | 115.0 | 118.0 | 0.000 |
| 0 | 256 | 8 | 8 | random | signature | 16 | 2 | 39.69 | 276 | 2023 | 8 | 32368 | 40 | 500 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 8 | 8 | random | signature | 16 | 2 | 39.00 | 275 | 2025 | 8 | 32400 | 38 | 504 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 8 | 8 | random | signature | 16 | 2 | 37.84 | 269 | 2035 | 8 | 32560 | 26 | 499 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 8 | 8 | random | signature | 16 | 2 | 40.06 | 278 | 2023 | 8 | 32368 | 44 | 506 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 8 | 8 | random | signature | 16 | 2 | 39.63 | 273 | 2029 | 8 | 32464 | 34 | 504 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 8 | 8 | random | signature | 32 | 4 | 52.89 | 306 | 1990 | 8 | 31856 | 91 | 500 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 8 | 8 | random | signature | 32 | 4 | 50.34 | 298 | 2000 | 8 | 32016 | 76 | 504 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 8 | 8 | random | signature | 32 | 2 | 40.53 | 300 | 2004 | 8 | 32064 | 78 | 499 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 8 | 8 | random | signature | 32 | 4 | 51.02 | 304 | 1996 | 8 | 31952 | 85 | 506 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 8 | 8 | random | signature | 32 | 4 | 51.77 | 300 | 2000 | 8 | 32032 | 82 | 504 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 8 | 8 | random | signature | 64 | 4 | 55.86 | 366 | 1928 | 8 | 30880 | 163 | 500 | 7.0 | 8.0 | 1.000 |
| 1 | 256 | 8 | 8 | random | signature | 64 | 4 | 53.50 | 351 | 1946 | 8 | 31168 | 144 | 504 | 7.0 | 8.0 | 1.000 |
| 2 | 256 | 8 | 8 | random | signature | 64 | 4 | 55.54 | 355 | 1946 | 8 | 31152 | 145 | 499 | 7.0 | 8.0 | 1.000 |
| 3 | 256 | 8 | 8 | random | signature | 64 | 4 | 59.98 | 367 | 1931 | 8 | 30960 | 158 | 506 | 7.0 | 8.0 | 1.000 |
| 4 | 256 | 8 | 8 | random | signature | 64 | 4 | 56.28 | 355 | 1943 | 8 | 31152 | 156 | 504 | 7.0 | 8.0 | 1.000 |
| 0 | 256 | 8 | 8 | random | signature | 128 | 4 | 60.41 | 474 | 1794 | 8 | 29040 | 221 | 500 | 4.0 | 7.0 | 1.000 |
| 1 | 256 | 8 | 8 | random | signature | 128 | 4 | 60.64 | 448 | 1838 | 8 | 29568 | 213 | 504 | 5.0 | 7.0 | 1.000 |
| 2 | 256 | 8 | 8 | random | signature | 128 | 4 | 60.77 | 453 | 1831 | 8 | 29488 | 220 | 499 | 5.0 | 7.0 | 1.000 |
| 3 | 256 | 8 | 8 | random | signature | 128 | 4 | 60.18 | 469 | 1815 | 8 | 29248 | 225 | 506 | 5.0 | 7.0 | 1.000 |
| 4 | 256 | 8 | 8 | random | signature | 128 | 4 | 61.42 | 443 | 1840 | 8 | 29648 | 223 | 504 | 5.0 | 7.0 | 1.000 |
| 0 | 256 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 89.60 | 251 | 4064 | 31 | 65024 | 0 | 509 | 16.0 | 16.0 | 0.976 |
| 1 | 256 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 90.02 | 248 | 4045 | 30 | 64720 | 0 | 509 | 16.0 | 16.0 | 0.962 |
| 2 | 256 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 79.55 | 250 | 4037 | 27 | 64592 | 0 | 508 | 16.0 | 16.0 | 0.982 |
| 3 | 256 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 77.82 | 250 | 4039 | 27 | 64624 | 0 | 499 | 16.0 | 16.0 | 0.981 |
| 4 | 256 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 80.45 | 248 | 4039 | 28 | 64624 | 0 | 508 | 16.0 | 16.0 | 0.965 |
| 0 | 256 | 8 | 16 | clustered | bounded_merge | 32 | 4 | 130.40 | 238 | 3925 | 31 | 62912 | 4 | 509 | 16.0 | 16.0 | 0.922 |
| 1 | 256 | 8 | 16 | clustered | bounded_merge | 32 | 2 | 89.54 | 237 | 3977 | 31 | 63632 | 0 | 509 | 16.0 | 16.0 | 0.907 |
| 2 | 256 | 8 | 16 | clustered | bounded_merge | 32 | 2 | 89.94 | 232 | 3889 | 31 | 62224 | 0 | 508 | 16.0 | 16.0 | 0.909 |
| 3 | 256 | 8 | 16 | clustered | bounded_merge | 32 | 2 | 89.01 | 240 | 3936 | 31 | 62976 | 0 | 499 | 16.0 | 16.0 | 0.951 |
| 4 | 256 | 8 | 16 | clustered | bounded_merge | 32 | 4 | 118.35 | 241 | 3940 | 30 | 63280 | 5 | 508 | 16.0 | 16.0 | 0.922 |
| 0 | 256 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 84.22 | 275 | 3683 | 16 | 59536 | 64 | 509 | 16.0 | 16.0 | 0.950 |
| 1 | 256 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 86.08 | 273 | 3687 | 16 | 59232 | 65 | 509 | 16.0 | 16.0 | 0.936 |
| 2 | 256 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 84.95 | 275 | 3643 | 16 | 58928 | 66 | 508 | 16.0 | 16.0 | 0.958 |
| 3 | 256 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 83.42 | 275 | 3599 | 16 | 58176 | 67 | 499 | 16.0 | 16.0 | 0.954 |
| 4 | 256 | 8 | 16 | clustered | bounded_merge | 64 | 8 | 165.26 | 278 | 3642 | 16 | 58960 | 68 | 508 | 16.0 | 16.0 | 0.935 |
| 0 | 256 | 8 | 16 | clustered | bounded_merge | 128 | 4 | 93.80 | 331 | 3259 | 16 | 54320 | 140 | 509 | 10.0 | 16.0 | 0.985 |
| 1 | 256 | 8 | 16 | clustered | bounded_merge | 128 | 4 | 112.26 | 326 | 3247 | 27 | 53376 | 151 | 509 | 10.0 | 16.0 | 0.957 |
| 2 | 256 | 8 | 16 | clustered | bounded_merge | 128 | 4 | 92.22 | 327 | 3169 | 16 | 52576 | 149 | 508 | 9.0 | 16.0 | 0.980 |
| 3 | 256 | 8 | 16 | clustered | bounded_merge | 128 | 4 | 109.98 | 321 | 3261 | 27 | 54080 | 134 | 499 | 11.0 | 16.0 | 0.964 |
| 4 | 256 | 8 | 16 | clustered | bounded_merge | 128 | 8 | 184.02 | 326 | 3323 | 27 | 54816 | 139 | 508 | 11.0 | 16.0 | 0.964 |
| 0 | 256 | 8 | 16 | clustered | q_outer | 16 | 2 | 87.59 | 128 | 4064 | 32 | 65024 | 0 | 509 | 32.0 | 32.0 | 0.008 |
| 1 | 256 | 8 | 16 | clustered | q_outer | 16 | 2 | 87.27 | 128 | 4045 | 32 | 64720 | 0 | 509 | 32.0 | 32.0 | 0.013 |
| 2 | 256 | 8 | 16 | clustered | q_outer | 16 | 2 | 88.23 | 128 | 4037 | 32 | 64592 | 0 | 508 | 32.0 | 32.0 | 0.015 |
| 3 | 256 | 8 | 16 | clustered | q_outer | 16 | 2 | 87.94 | 128 | 4039 | 32 | 64624 | 0 | 499 | 32.0 | 32.0 | 0.014 |
| 4 | 256 | 8 | 16 | clustered | q_outer | 16 | 2 | 87.38 | 128 | 4039 | 32 | 64624 | 0 | 508 | 32.0 | 32.0 | 0.014 |
| 0 | 256 | 8 | 16 | clustered | q_outer | 32 | 4 | 203.20 | 64 | 3925 | 64 | 125600 | 0 | 509 | 64.0 | 64.0 | 0.000 |
| 1 | 256 | 8 | 16 | clustered | q_outer | 32 | 4 | 205.95 | 64 | 3977 | 64 | 127264 | 0 | 509 | 64.0 | 64.0 | 0.000 |
| 2 | 256 | 8 | 16 | clustered | q_outer | 32 | 4 | 208.17 | 64 | 3889 | 64 | 124448 | 0 | 508 | 64.0 | 64.0 | 0.000 |
| 3 | 256 | 8 | 16 | clustered | q_outer | 32 | 4 | 205.02 | 64 | 3936 | 64 | 125952 | 0 | 499 | 64.0 | 64.0 | 0.000 |
| 4 | 256 | 8 | 16 | clustered | q_outer | 32 | 4 | 208.39 | 64 | 3940 | 64 | 126080 | 0 | 508 | 64.0 | 64.0 | 0.001 |
| 0 | 256 | 8 | 16 | clustered | q_outer | 64 | 8 | 596.88 | 32 | 3683 | 128 | 235712 | 0 | 509 | 117.0 | 128.0 | 0.000 |
| 1 | 256 | 8 | 16 | clustered | q_outer | 64 | 8 | 605.17 | 32 | 3687 | 128 | 235968 | 0 | 509 | 116.0 | 127.8 | 0.000 |
| 2 | 256 | 8 | 16 | clustered | q_outer | 64 | 8 | 612.12 | 32 | 3643 | 128 | 233152 | 0 | 508 | 114.5 | 128.0 | 0.000 |
| 3 | 256 | 8 | 16 | clustered | q_outer | 64 | 8 | 608.16 | 32 | 3599 | 128 | 230336 | 0 | 499 | 112.5 | 128.0 | 0.000 |
| 4 | 256 | 8 | 16 | clustered | q_outer | 64 | 8 | 606.96 | 32 | 3642 | 128 | 233088 | 0 | 508 | 115.5 | 126.9 | 0.000 |
| 0 | 256 | 8 | 16 | clustered | q_outer | 128 | 16 | 3021.97 | 16 | 3259 | 243 | 417152 | 0 | 509 | 197.0 | 234.0 | 0.000 |
| 1 | 256 | 8 | 16 | clustered | q_outer | 128 | 16 | 2761.44 | 16 | 3247 | 220 | 415616 | 0 | 509 | 198.5 | 219.5 | 0.000 |
| 2 | 256 | 8 | 16 | clustered | q_outer | 128 | 16 | 2869.06 | 16 | 3169 | 230 | 405632 | 0 | 508 | 198.0 | 223.0 | 0.000 |
| 3 | 256 | 8 | 16 | clustered | q_outer | 128 | 16 | 2824.37 | 16 | 3261 | 226 | 417408 | 0 | 499 | 210.0 | 220.0 | 0.000 |
| 4 | 256 | 8 | 16 | clustered | q_outer | 128 | 16 | 2884.29 | 16 | 3323 | 231 | 425344 | 0 | 508 | 211.0 | 224.5 | 0.000 |
| 0 | 256 | 8 | 16 | clustered | signature | 16 | 2 | 60.95 | 261 | 4064 | 16 | 65024 | 10 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 8 | 16 | clustered | signature | 16 | 2 | 61.23 | 264 | 4045 | 16 | 64720 | 16 | 509 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 8 | 16 | clustered | signature | 16 | 2 | 60.90 | 262 | 4037 | 16 | 64592 | 12 | 508 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 8 | 16 | clustered | signature | 16 | 2 | 61.35 | 262 | 4039 | 16 | 64624 | 12 | 499 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 8 | 16 | clustered | signature | 16 | 2 | 61.24 | 264 | 4039 | 16 | 64624 | 16 | 508 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 8 | 16 | clustered | signature | 32 | 4 | 81.39 | 276 | 3925 | 16 | 62912 | 40 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 8 | 16 | clustered | signature | 32 | 2 | 68.69 | 273 | 3977 | 16 | 63632 | 36 | 509 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 8 | 16 | clustered | signature | 32 | 2 | 69.01 | 276 | 3889 | 16 | 62224 | 44 | 508 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 8 | 16 | clustered | signature | 32 | 2 | 63.49 | 272 | 3936 | 16 | 62976 | 32 | 499 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 8 | 16 | clustered | signature | 32 | 4 | 83.66 | 276 | 3940 | 16 | 63280 | 37 | 508 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 8 | 16 | clustered | signature | 64 | 4 | 91.69 | 300 | 3683 | 16 | 59536 | 84 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 8 | 16 | clustered | signature | 64 | 4 | 90.95 | 302 | 3687 | 16 | 59232 | 92 | 509 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 8 | 16 | clustered | signature | 64 | 4 | 90.51 | 300 | 3643 | 16 | 58928 | 88 | 508 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 8 | 16 | clustered | signature | 64 | 4 | 86.12 | 301 | 3599 | 16 | 58176 | 89 | 499 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 8 | 16 | clustered | signature | 64 | 8 | 162.17 | 307 | 3642 | 16 | 58960 | 90 | 508 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 8 | 16 | clustered | signature | 128 | 4 | 89.39 | 341 | 3259 | 16 | 54320 | 142 | 509 | 9.0 | 16.0 | 1.000 |
| 1 | 256 | 8 | 16 | clustered | signature | 128 | 4 | 92.56 | 340 | 3247 | 16 | 53376 | 158 | 509 | 9.0 | 16.0 | 1.000 |
| 2 | 256 | 8 | 16 | clustered | signature | 128 | 4 | 90.67 | 340 | 3169 | 16 | 52576 | 158 | 508 | 9.0 | 16.0 | 1.000 |
| 3 | 256 | 8 | 16 | clustered | signature | 128 | 4 | 90.14 | 336 | 3261 | 16 | 54080 | 145 | 499 | 10.0 | 16.0 | 1.000 |
| 4 | 256 | 8 | 16 | clustered | signature | 128 | 8 | 153.78 | 340 | 3323 | 16 | 54816 | 147 | 508 | 10.0 | 16.0 | 1.000 |
| 0 | 256 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 68.81 | 128 | 3070 | 24 | 49120 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 1 | 256 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 68.68 | 128 | 3070 | 24 | 49120 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 2 | 256 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 68.19 | 128 | 3070 | 24 | 49120 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 3 | 256 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 67.88 | 128 | 3070 | 24 | 49120 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 4 | 256 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 69.09 | 128 | 3070 | 24 | 49120 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 0 | 256 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 75.58 | 320 | 2558 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 75.61 | 320 | 2558 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 75.96 | 320 | 2558 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 76.37 | 320 | 2558 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 75.49 | 320 | 2558 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 117.47 | 288 | 2302 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 117.27 | 288 | 2302 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 118.03 | 288 | 2302 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 118.84 | 288 | 2302 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 117.40 | 288 | 2302 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 316.65 | 272 | 2174 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 316.48 | 272 | 2174 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 319.69 | 272 | 2174 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 318.77 | 272 | 2174 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 316.71 | 272 | 2174 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 8 | 16 | common_private | q_outer | 16 | 2 | 71.55 | 128 | 3070 | 24 | 49120 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 1 | 256 | 8 | 16 | common_private | q_outer | 16 | 2 | 71.62 | 128 | 3070 | 24 | 49120 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 2 | 256 | 8 | 16 | common_private | q_outer | 16 | 2 | 71.81 | 128 | 3070 | 24 | 49120 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 3 | 256 | 8 | 16 | common_private | q_outer | 16 | 2 | 70.61 | 128 | 3070 | 24 | 49120 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 4 | 256 | 8 | 16 | common_private | q_outer | 16 | 2 | 70.31 | 128 | 3070 | 24 | 49120 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 0 | 256 | 8 | 16 | common_private | q_outer | 32 | 4 | 132.98 | 64 | 2558 | 40 | 81856 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 1 | 256 | 8 | 16 | common_private | q_outer | 32 | 4 | 133.71 | 64 | 2558 | 40 | 81856 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 2 | 256 | 8 | 16 | common_private | q_outer | 32 | 4 | 133.76 | 64 | 2558 | 40 | 81856 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 3 | 256 | 8 | 16 | common_private | q_outer | 32 | 4 | 133.78 | 64 | 2558 | 40 | 81856 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 4 | 256 | 8 | 16 | common_private | q_outer | 32 | 4 | 132.91 | 64 | 2558 | 40 | 81856 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 0 | 256 | 8 | 16 | common_private | q_outer | 64 | 8 | 347.46 | 32 | 2302 | 72 | 147328 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 1 | 256 | 8 | 16 | common_private | q_outer | 64 | 8 | 348.97 | 32 | 2302 | 72 | 147328 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 2 | 256 | 8 | 16 | common_private | q_outer | 64 | 8 | 346.44 | 32 | 2302 | 72 | 147328 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 3 | 256 | 8 | 16 | common_private | q_outer | 64 | 8 | 349.24 | 32 | 2302 | 72 | 147328 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 4 | 256 | 8 | 16 | common_private | q_outer | 64 | 8 | 348.69 | 32 | 2302 | 72 | 147328 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 0 | 256 | 8 | 16 | common_private | q_outer | 128 | 16 | 1754.28 | 16 | 2174 | 136 | 278272 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 1 | 256 | 8 | 16 | common_private | q_outer | 128 | 16 | 1768.09 | 16 | 2174 | 136 | 278272 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 2 | 256 | 8 | 16 | common_private | q_outer | 128 | 16 | 1759.07 | 16 | 2174 | 136 | 278272 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 3 | 256 | 8 | 16 | common_private | q_outer | 128 | 16 | 1760.59 | 16 | 2174 | 136 | 278272 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 4 | 256 | 8 | 16 | common_private | q_outer | 128 | 16 | 1751.17 | 16 | 2174 | 136 | 278272 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 0 | 256 | 8 | 16 | common_private | signature | 16 | 2 | 62.49 | 384 | 3070 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 8 | 16 | common_private | signature | 16 | 2 | 62.51 | 384 | 3070 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 8 | 16 | common_private | signature | 16 | 2 | 61.77 | 384 | 3070 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 8 | 16 | common_private | signature | 16 | 2 | 62.46 | 384 | 3070 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 8 | 16 | common_private | signature | 16 | 2 | 62.58 | 384 | 3070 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 8 | 16 | common_private | signature | 32 | 4 | 75.54 | 320 | 2558 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 8 | 16 | common_private | signature | 32 | 4 | 75.38 | 320 | 2558 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 8 | 16 | common_private | signature | 32 | 4 | 75.87 | 320 | 2558 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 8 | 16 | common_private | signature | 32 | 4 | 76.58 | 320 | 2558 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 8 | 16 | common_private | signature | 32 | 4 | 75.57 | 320 | 2558 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 8 | 16 | common_private | signature | 64 | 8 | 117.72 | 288 | 2302 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 8 | 16 | common_private | signature | 64 | 8 | 117.90 | 288 | 2302 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 8 | 16 | common_private | signature | 64 | 8 | 117.67 | 288 | 2302 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 8 | 16 | common_private | signature | 64 | 8 | 118.98 | 288 | 2302 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 8 | 16 | common_private | signature | 64 | 8 | 117.89 | 288 | 2302 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 8 | 16 | common_private | signature | 128 | 16 | 316.15 | 272 | 2174 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 256 | 8 | 16 | common_private | signature | 128 | 16 | 316.06 | 272 | 2174 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 256 | 8 | 16 | common_private | signature | 128 | 16 | 316.32 | 272 | 2174 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 256 | 8 | 16 | common_private | signature | 128 | 16 | 319.30 | 272 | 2174 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 256 | 8 | 16 | common_private | signature | 128 | 16 | 315.82 | 272 | 2174 | 8 | 49120 | 256 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 256 | 8 | 16 | random | bounded_merge | 16 | 2 | 93.03 | 199 | 4024 | 31 | 64384 | 0 | 510 | 16.0 | 31.0 | 0.581 |
| 1 | 256 | 8 | 16 | random | bounded_merge | 16 | 2 | 92.99 | 211 | 4039 | 31 | 64624 | 0 | 511 | 16.0 | 31.0 | 0.670 |
| 2 | 256 | 8 | 16 | random | bounded_merge | 16 | 2 | 93.58 | 211 | 4040 | 31 | 64640 | 0 | 512 | 16.0 | 31.0 | 0.669 |
| 3 | 256 | 8 | 16 | random | bounded_merge | 16 | 2 | 92.94 | 205 | 4029 | 31 | 64464 | 0 | 512 | 16.0 | 31.0 | 0.625 |
| 4 | 256 | 8 | 16 | random | bounded_merge | 16 | 2 | 92.07 | 201 | 4027 | 31 | 64432 | 0 | 512 | 16.0 | 31.0 | 0.596 |
| 0 | 256 | 8 | 16 | random | bounded_merge | 32 | 4 | 130.87 | 264 | 3902 | 31 | 62480 | 109 | 510 | 16.0 | 29.7 | 0.485 |
| 1 | 256 | 8 | 16 | random | bounded_merge | 32 | 4 | 130.50 | 251 | 3915 | 31 | 62704 | 103 | 511 | 16.0 | 30.0 | 0.473 |
| 2 | 256 | 8 | 16 | random | bounded_merge | 32 | 4 | 131.79 | 264 | 3898 | 31 | 62400 | 116 | 512 | 15.0 | 30.0 | 0.439 |
| 3 | 256 | 8 | 16 | random | bounded_merge | 32 | 4 | 132.17 | 247 | 3927 | 31 | 62928 | 90 | 512 | 16.0 | 30.0 | 0.501 |
| 4 | 256 | 8 | 16 | random | bounded_merge | 32 | 4 | 129.15 | 266 | 3916 | 31 | 62720 | 106 | 512 | 15.0 | 30.0 | 0.516 |
| 0 | 256 | 8 | 16 | random | bounded_merge | 64 | 4 | 102.65 | 573 | 3636 | 16 | 58672 | 249 | 510 | 2.0 | 14.0 | 0.894 |
| 1 | 256 | 8 | 16 | random | bounded_merge | 64 | 4 | 104.95 | 556 | 3676 | 16 | 59232 | 242 | 511 | 2.0 | 14.0 | 0.887 |
| 2 | 256 | 8 | 16 | random | bounded_merge | 64 | 4 | 99.94 | 561 | 3653 | 16 | 58912 | 247 | 512 | 2.0 | 14.0 | 0.895 |
| 3 | 256 | 8 | 16 | random | bounded_merge | 64 | 4 | 102.21 | 544 | 3686 | 16 | 59280 | 240 | 512 | 2.0 | 14.0 | 0.891 |
| 4 | 256 | 8 | 16 | random | bounded_merge | 64 | 4 | 108.92 | 544 | 3702 | 16 | 59648 | 242 | 512 | 2.0 | 14.0 | 0.893 |
| 0 | 256 | 8 | 16 | random | bounded_merge | 128 | 8 | 171.12 | 920 | 3207 | 14 | 53216 | 256 | 510 | 1.0 | 10.0 | 1.000 |
| 1 | 256 | 8 | 16 | random | bounded_merge | 128 | 4 | 102.21 | 896 | 3265 | 14 | 53824 | 256 | 511 | 1.0 | 11.0 | 1.000 |
| 2 | 256 | 8 | 16 | random | bounded_merge | 128 | 8 | 166.91 | 878 | 3272 | 15 | 54016 | 256 | 512 | 1.0 | 11.0 | 1.000 |
| 3 | 256 | 8 | 16 | random | bounded_merge | 128 | 4 | 104.25 | 908 | 3256 | 15 | 53424 | 256 | 512 | 1.0 | 11.0 | 1.000 |
| 4 | 256 | 8 | 16 | random | bounded_merge | 128 | 4 | 103.50 | 869 | 3305 | 15 | 54528 | 256 | 512 | 1.0 | 11.0 | 1.000 |
| 0 | 256 | 8 | 16 | random | q_outer | 16 | 2 | 85.46 | 128 | 4024 | 32 | 64384 | 0 | 510 | 32.0 | 32.0 | 0.017 |
| 1 | 256 | 8 | 16 | random | q_outer | 16 | 2 | 86.67 | 128 | 4039 | 32 | 64624 | 0 | 511 | 32.0 | 32.0 | 0.013 |
| 2 | 256 | 8 | 16 | random | q_outer | 16 | 2 | 87.49 | 128 | 4040 | 32 | 64640 | 0 | 512 | 32.0 | 32.0 | 0.013 |
| 3 | 256 | 8 | 16 | random | q_outer | 16 | 2 | 87.35 | 128 | 4029 | 32 | 64464 | 0 | 512 | 32.0 | 32.0 | 0.015 |
| 4 | 256 | 8 | 16 | random | q_outer | 16 | 2 | 87.96 | 128 | 4027 | 32 | 64432 | 0 | 512 | 32.0 | 32.0 | 0.016 |
| 0 | 256 | 8 | 16 | random | q_outer | 32 | 4 | 199.18 | 64 | 3902 | 64 | 124864 | 0 | 510 | 61.0 | 63.0 | 0.000 |
| 1 | 256 | 8 | 16 | random | q_outer | 32 | 4 | 199.26 | 64 | 3915 | 64 | 125280 | 0 | 511 | 61.0 | 63.0 | 0.000 |
| 2 | 256 | 8 | 16 | random | q_outer | 32 | 4 | 200.86 | 64 | 3898 | 64 | 124736 | 0 | 512 | 61.0 | 62.0 | 0.000 |
| 3 | 256 | 8 | 16 | random | q_outer | 32 | 4 | 203.17 | 64 | 3927 | 64 | 125664 | 0 | 512 | 61.5 | 63.0 | 0.000 |
| 4 | 256 | 8 | 16 | random | q_outer | 32 | 4 | 208.24 | 64 | 3916 | 64 | 125312 | 0 | 512 | 61.0 | 64.0 | 0.000 |
| 0 | 256 | 8 | 16 | random | q_outer | 64 | 8 | 557.06 | 32 | 3636 | 120 | 232704 | 0 | 510 | 114.0 | 117.9 | 0.000 |
| 1 | 256 | 8 | 16 | random | q_outer | 64 | 8 | 562.92 | 32 | 3676 | 120 | 235264 | 0 | 511 | 115.0 | 118.0 | 0.000 |
| 2 | 256 | 8 | 16 | random | q_outer | 64 | 8 | 574.70 | 32 | 3653 | 123 | 233792 | 0 | 512 | 114.5 | 117.9 | 0.000 |
| 3 | 256 | 8 | 16 | random | q_outer | 64 | 8 | 573.49 | 32 | 3686 | 121 | 235904 | 0 | 512 | 116.0 | 118.9 | 0.000 |
| 4 | 256 | 8 | 16 | random | q_outer | 64 | 8 | 570.52 | 32 | 3702 | 121 | 236928 | 0 | 512 | 115.5 | 120.0 | 0.000 |
| 0 | 256 | 8 | 16 | random | q_outer | 128 | 16 | 2616.35 | 16 | 3207 | 209 | 410496 | 0 | 510 | 200.0 | 207.0 | 0.000 |
| 1 | 256 | 8 | 16 | random | q_outer | 128 | 16 | 2671.18 | 16 | 3265 | 212 | 417920 | 0 | 511 | 204.0 | 210.0 | 0.000 |
| 2 | 256 | 8 | 16 | random | q_outer | 128 | 16 | 2730.64 | 16 | 3272 | 216 | 418816 | 0 | 512 | 205.0 | 211.0 | 0.000 |
| 3 | 256 | 8 | 16 | random | q_outer | 128 | 16 | 2665.70 | 16 | 3256 | 212 | 416768 | 0 | 512 | 205.0 | 209.0 | 0.000 |
| 4 | 256 | 8 | 16 | random | q_outer | 128 | 16 | 2667.06 | 16 | 3305 | 214 | 423040 | 0 | 512 | 206.5 | 211.5 | 0.000 |
| 0 | 256 | 8 | 16 | random | signature | 16 | 2 | 62.45 | 313 | 4024 | 16 | 64384 | 114 | 510 | 15.0 | 16.0 | 1.000 |
| 1 | 256 | 8 | 16 | random | signature | 16 | 2 | 63.45 | 301 | 4039 | 16 | 64624 | 90 | 511 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 8 | 16 | random | signature | 16 | 2 | 63.28 | 301 | 4040 | 16 | 64640 | 90 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 8 | 16 | random | signature | 16 | 2 | 63.28 | 307 | 4029 | 16 | 64464 | 102 | 512 | 15.0 | 16.0 | 1.000 |
| 4 | 256 | 8 | 16 | random | signature | 16 | 2 | 62.46 | 311 | 4027 | 16 | 64432 | 110 | 512 | 15.0 | 16.0 | 1.000 |
| 0 | 256 | 8 | 16 | random | signature | 32 | 4 | 88.64 | 406 | 3902 | 16 | 62480 | 201 | 510 | 14.0 | 16.0 | 1.000 |
| 1 | 256 | 8 | 16 | random | signature | 32 | 4 | 90.68 | 395 | 3915 | 16 | 62704 | 197 | 511 | 14.0 | 16.0 | 1.000 |
| 2 | 256 | 8 | 16 | random | signature | 32 | 4 | 90.68 | 417 | 3898 | 16 | 62400 | 215 | 512 | 14.0 | 15.0 | 1.000 |
| 3 | 256 | 8 | 16 | random | signature | 32 | 4 | 89.01 | 384 | 3927 | 16 | 62928 | 187 | 512 | 14.0 | 16.0 | 1.000 |
| 4 | 256 | 8 | 16 | random | signature | 32 | 4 | 88.53 | 399 | 3916 | 16 | 62720 | 186 | 512 | 14.0 | 16.0 | 1.000 |
| 0 | 256 | 8 | 16 | random | signature | 64 | 4 | 99.59 | 605 | 3636 | 16 | 58672 | 251 | 510 | 2.0 | 14.0 | 1.000 |
| 1 | 256 | 8 | 16 | random | signature | 64 | 4 | 103.22 | 588 | 3676 | 16 | 59232 | 252 | 511 | 2.0 | 14.0 | 1.000 |
| 2 | 256 | 8 | 16 | random | signature | 64 | 4 | 100.61 | 593 | 3653 | 16 | 58912 | 250 | 512 | 2.0 | 14.0 | 1.000 |
| 3 | 256 | 8 | 16 | random | signature | 64 | 4 | 97.67 | 576 | 3686 | 16 | 59280 | 246 | 512 | 2.0 | 14.0 | 1.000 |
| 4 | 256 | 8 | 16 | random | signature | 64 | 4 | 100.70 | 576 | 3702 | 16 | 59648 | 245 | 512 | 2.0 | 14.0 | 1.000 |
| 0 | 256 | 8 | 16 | random | signature | 128 | 8 | 170.91 | 920 | 3207 | 14 | 53216 | 256 | 510 | 1.0 | 10.0 | 1.000 |
| 1 | 256 | 8 | 16 | random | signature | 128 | 4 | 102.82 | 896 | 3265 | 14 | 53824 | 256 | 511 | 1.0 | 11.0 | 1.000 |
| 2 | 256 | 8 | 16 | random | signature | 128 | 8 | 166.99 | 878 | 3272 | 15 | 54016 | 256 | 512 | 1.0 | 11.0 | 1.000 |
| 3 | 256 | 8 | 16 | random | signature | 128 | 4 | 101.79 | 908 | 3256 | 15 | 53424 | 256 | 512 | 1.0 | 11.0 | 1.000 |
| 4 | 256 | 8 | 16 | random | signature | 128 | 4 | 102.77 | 869 | 3305 | 15 | 54528 | 256 | 512 | 1.0 | 11.0 | 1.000 |
| 0 | 256 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 164.67 | 241 | 7970 | 62 | 127520 | 0 | 509 | 32.0 | 32.0 | 0.935 |
| 1 | 256 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 169.73 | 239 | 7923 | 62 | 126768 | 0 | 509 | 32.0 | 32.0 | 0.931 |
| 2 | 256 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 179.22 | 238 | 7962 | 63 | 127392 | 0 | 510 | 32.0 | 32.0 | 0.913 |
| 3 | 256 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 168.12 | 238 | 7928 | 62 | 126848 | 0 | 500 | 32.0 | 32.0 | 0.921 |
| 4 | 256 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 173.40 | 234 | 7851 | 63 | 125616 | 0 | 509 | 32.0 | 32.0 | 0.908 |
| 0 | 256 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 228.79 | 226 | 7459 | 62 | 120272 | 18 | 509 | 32.0 | 40.0 | 0.850 |
| 1 | 256 | 8 | 32 | clustered | bounded_merge | 32 | 2 | 183.56 | 217 | 7574 | 62 | 121184 | 4 | 509 | 32.0 | 46.4 | 0.810 |
| 2 | 256 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 236.81 | 209 | 7360 | 63 | 117808 | 9 | 510 | 32.0 | 52.2 | 0.762 |
| 3 | 256 | 8 | 32 | clustered | bounded_merge | 32 | 2 | 170.31 | 217 | 7557 | 63 | 120912 | 3 | 500 | 32.0 | 48.4 | 0.822 |
| 4 | 256 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 234.89 | 227 | 7434 | 61 | 120144 | 23 | 509 | 32.0 | 42.4 | 0.828 |
| 0 | 256 | 8 | 32 | clustered | bounded_merge | 64 | 8 | 307.52 | 316 | 6591 | 32 | 109456 | 133 | 509 | 23.0 | 32.0 | 0.937 |
| 1 | 256 | 8 | 32 | clustered | bounded_merge | 64 | 4 | 168.48 | 309 | 6589 | 32 | 106960 | 130 | 509 | 25.0 | 32.0 | 0.942 |
| 2 | 256 | 8 | 32 | clustered | bounded_merge | 64 | 4 | 160.14 | 315 | 6550 | 32 | 108304 | 128 | 510 | 24.0 | 32.0 | 0.933 |
| 3 | 256 | 8 | 32 | clustered | bounded_merge | 64 | 4 | 158.52 | 312 | 6528 | 32 | 107392 | 136 | 500 | 25.0 | 32.0 | 0.928 |
| 4 | 256 | 8 | 32 | clustered | bounded_merge | 64 | 8 | 319.20 | 317 | 6459 | 32 | 108768 | 132 | 509 | 22.0 | 32.0 | 0.926 |
| 0 | 256 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 284.38 | 395 | 5179 | 41 | 95008 | 217 | 509 | 11.0 | 31.0 | 0.952 |
| 1 | 256 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 298.28 | 386 | 5215 | 32 | 93072 | 220 | 509 | 11.0 | 31.0 | 0.961 |
| 2 | 256 | 8 | 32 | clustered | bounded_merge | 128 | 4 | 162.12 | 377 | 5203 | 32 | 94016 | 212 | 510 | 10.0 | 32.0 | 0.967 |
| 3 | 256 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 296.70 | 381 | 5389 | 32 | 97440 | 206 | 500 | 11.0 | 32.0 | 0.972 |
| 4 | 256 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 304.85 | 383 | 5356 | 32 | 96304 | 204 | 509 | 11.0 | 32.0 | 0.968 |
| 0 | 256 | 8 | 32 | clustered | q_outer | 16 | 2 | 164.59 | 128 | 7970 | 64 | 127520 | 0 | 509 | 64.0 | 64.0 | 0.028 |
| 1 | 256 | 8 | 32 | clustered | q_outer | 16 | 2 | 164.97 | 128 | 7923 | 64 | 126768 | 0 | 509 | 64.0 | 64.0 | 0.034 |
| 2 | 256 | 8 | 32 | clustered | q_outer | 16 | 2 | 166.07 | 128 | 7962 | 64 | 127392 | 0 | 510 | 64.0 | 64.0 | 0.029 |
| 3 | 256 | 8 | 32 | clustered | q_outer | 16 | 2 | 161.58 | 128 | 7928 | 64 | 126848 | 0 | 500 | 64.0 | 64.0 | 0.033 |
| 4 | 256 | 8 | 32 | clustered | q_outer | 16 | 2 | 164.93 | 128 | 7851 | 64 | 125616 | 0 | 509 | 64.0 | 64.0 | 0.043 |
| 0 | 256 | 8 | 32 | clustered | q_outer | 32 | 4 | 396.81 | 64 | 7459 | 128 | 238688 | 0 | 509 | 125.0 | 128.0 | 0.000 |
| 1 | 256 | 8 | 32 | clustered | q_outer | 32 | 4 | 393.77 | 64 | 7574 | 128 | 242368 | 0 | 509 | 122.0 | 128.0 | 0.000 |
| 2 | 256 | 8 | 32 | clustered | q_outer | 32 | 4 | 401.04 | 64 | 7360 | 128 | 235520 | 0 | 510 | 117.5 | 128.0 | 0.000 |
| 3 | 256 | 8 | 32 | clustered | q_outer | 32 | 4 | 392.36 | 64 | 7557 | 128 | 241824 | 0 | 500 | 123.0 | 128.0 | 0.000 |
| 4 | 256 | 8 | 32 | clustered | q_outer | 32 | 4 | 396.27 | 64 | 7434 | 128 | 237888 | 0 | 509 | 124.0 | 128.0 | 0.002 |
| 0 | 256 | 8 | 32 | clustered | q_outer | 64 | 8 | 1174.96 | 32 | 6591 | 256 | 421824 | 0 | 509 | 209.0 | 234.0 | 0.000 |
| 1 | 256 | 8 | 32 | clustered | q_outer | 64 | 8 | 1161.27 | 32 | 6589 | 255 | 421696 | 0 | 509 | 208.0 | 229.8 | 0.000 |
| 2 | 256 | 8 | 32 | clustered | q_outer | 64 | 8 | 1155.95 | 32 | 6550 | 256 | 419200 | 0 | 510 | 200.0 | 241.8 | 0.000 |
| 3 | 256 | 8 | 32 | clustered | q_outer | 64 | 8 | 1140.05 | 32 | 6528 | 250 | 417792 | 0 | 500 | 206.0 | 241.8 | 0.000 |
| 4 | 256 | 8 | 32 | clustered | q_outer | 64 | 8 | 1095.88 | 32 | 6459 | 239 | 413376 | 0 | 509 | 202.5 | 229.9 | 0.000 |
| 0 | 256 | 8 | 32 | clustered | q_outer | 128 | 16 | 4687.77 | 16 | 5179 | 382 | 662912 | 0 | 509 | 325.0 | 373.5 | 0.000 |
| 1 | 256 | 8 | 32 | clustered | q_outer | 128 | 16 | 4598.67 | 16 | 5215 | 374 | 667520 | 0 | 509 | 327.0 | 355.0 | 0.000 |
| 2 | 256 | 8 | 32 | clustered | q_outer | 128 | 16 | 4701.19 | 16 | 5203 | 378 | 665984 | 0 | 510 | 331.5 | 368.0 | 0.000 |
| 3 | 256 | 8 | 32 | clustered | q_outer | 128 | 16 | 4763.90 | 16 | 5389 | 387 | 689792 | 0 | 500 | 345.5 | 373.0 | 0.000 |
| 4 | 256 | 8 | 32 | clustered | q_outer | 128 | 16 | 4707.27 | 16 | 5356 | 383 | 685568 | 0 | 509 | 340.5 | 375.5 | 0.000 |
| 0 | 256 | 8 | 32 | clustered | signature | 16 | 2 | 123.22 | 271 | 7970 | 32 | 127520 | 30 | 509 | 32.0 | 32.0 | 1.000 |
| 1 | 256 | 8 | 32 | clustered | signature | 16 | 2 | 119.61 | 273 | 7923 | 32 | 126768 | 34 | 509 | 32.0 | 32.0 | 1.000 |
| 2 | 256 | 8 | 32 | clustered | signature | 16 | 2 | 117.96 | 274 | 7962 | 32 | 127392 | 36 | 510 | 32.0 | 32.0 | 1.000 |
| 3 | 256 | 8 | 32 | clustered | signature | 16 | 2 | 119.58 | 272 | 7928 | 32 | 126848 | 34 | 500 | 32.0 | 32.0 | 1.000 |
| 4 | 256 | 8 | 32 | clustered | signature | 16 | 2 | 126.01 | 278 | 7851 | 32 | 125616 | 44 | 509 | 32.0 | 32.0 | 1.000 |
| 0 | 256 | 8 | 32 | clustered | signature | 32 | 4 | 157.71 | 299 | 7459 | 32 | 120272 | 79 | 509 | 32.0 | 32.0 | 1.000 |
| 1 | 256 | 8 | 32 | clustered | signature | 32 | 2 | 127.16 | 298 | 7574 | 32 | 121184 | 83 | 509 | 32.0 | 32.0 | 1.000 |
| 2 | 256 | 8 | 32 | clustered | signature | 32 | 4 | 161.98 | 309 | 7360 | 32 | 117808 | 104 | 510 | 31.0 | 32.0 | 1.000 |
| 3 | 256 | 8 | 32 | clustered | signature | 32 | 2 | 126.66 | 294 | 7557 | 32 | 120912 | 79 | 500 | 32.0 | 32.0 | 1.000 |
| 4 | 256 | 8 | 32 | clustered | signature | 32 | 4 | 168.34 | 304 | 7434 | 32 | 120144 | 85 | 509 | 32.0 | 32.0 | 1.000 |
| 0 | 256 | 8 | 32 | clustered | signature | 64 | 8 | 286.38 | 347 | 6591 | 32 | 109456 | 153 | 509 | 19.0 | 32.0 | 1.000 |
| 1 | 256 | 8 | 32 | clustered | signature | 64 | 4 | 168.42 | 340 | 6589 | 32 | 106960 | 153 | 509 | 19.0 | 32.0 | 1.000 |
| 2 | 256 | 8 | 32 | clustered | signature | 64 | 4 | 161.78 | 345 | 6550 | 32 | 108304 | 152 | 510 | 20.0 | 32.0 | 1.000 |
| 3 | 256 | 8 | 32 | clustered | signature | 64 | 4 | 160.27 | 344 | 6528 | 32 | 107392 | 157 | 500 | 21.0 | 32.0 | 1.000 |
| 4 | 256 | 8 | 32 | clustered | signature | 64 | 8 | 294.64 | 349 | 6459 | 32 | 108768 | 151 | 509 | 17.0 | 32.0 | 1.000 |
| 0 | 256 | 8 | 32 | clustered | signature | 128 | 8 | 262.79 | 409 | 5179 | 32 | 95008 | 223 | 509 | 10.0 | 29.0 | 1.000 |
| 1 | 256 | 8 | 32 | clustered | signature | 128 | 8 | 275.64 | 402 | 5215 | 32 | 93072 | 228 | 509 | 11.0 | 27.0 | 1.000 |
| 2 | 256 | 8 | 32 | clustered | signature | 128 | 4 | 159.03 | 393 | 5203 | 32 | 94016 | 218 | 510 | 10.0 | 31.0 | 1.000 |
| 3 | 256 | 8 | 32 | clustered | signature | 128 | 8 | 274.01 | 395 | 5389 | 32 | 97440 | 210 | 500 | 11.0 | 32.0 | 1.000 |
| 4 | 256 | 8 | 32 | clustered | signature | 128 | 8 | 281.90 | 399 | 5356 | 32 | 96304 | 209 | 509 | 11.0 | 32.0 | 1.000 |
| 0 | 256 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 121.96 | 128 | 6140 | 48 | 98240 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 1 | 256 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 122.59 | 128 | 6140 | 48 | 98240 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 2 | 256 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 122.02 | 128 | 6139 | 48 | 98224 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 3 | 256 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 121.96 | 128 | 6140 | 48 | 98240 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 4 | 256 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 121.77 | 128 | 6140 | 48 | 98240 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 0 | 256 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 136.54 | 320 | 5116 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 136.12 | 320 | 5116 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 136.60 | 320 | 5115 | 16 | 98224 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 136.49 | 320 | 5116 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 136.62 | 320 | 5116 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 217.34 | 288 | 4604 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 217.67 | 288 | 4604 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 216.91 | 288 | 4603 | 16 | 98224 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 217.66 | 288 | 4604 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 217.47 | 288 | 4604 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 590.15 | 272 | 4348 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 591.14 | 272 | 4348 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 590.50 | 272 | 4347 | 16 | 98224 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 590.72 | 272 | 4348 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 590.93 | 272 | 4348 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 8 | 32 | common_private | q_outer | 16 | 2 | 126.52 | 128 | 6140 | 48 | 98240 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 1 | 256 | 8 | 32 | common_private | q_outer | 16 | 2 | 126.91 | 128 | 6140 | 48 | 98240 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 2 | 256 | 8 | 32 | common_private | q_outer | 16 | 2 | 124.98 | 128 | 6139 | 48 | 98224 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 3 | 256 | 8 | 32 | common_private | q_outer | 16 | 2 | 126.27 | 128 | 6140 | 48 | 98240 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 4 | 256 | 8 | 32 | common_private | q_outer | 16 | 2 | 125.73 | 128 | 6140 | 48 | 98240 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 0 | 256 | 8 | 32 | common_private | q_outer | 32 | 4 | 255.90 | 64 | 5116 | 80 | 163712 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 1 | 256 | 8 | 32 | common_private | q_outer | 32 | 4 | 255.07 | 64 | 5116 | 80 | 163712 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 2 | 256 | 8 | 32 | common_private | q_outer | 32 | 4 | 254.99 | 64 | 5115 | 80 | 163680 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 3 | 256 | 8 | 32 | common_private | q_outer | 32 | 4 | 258.07 | 64 | 5116 | 80 | 163712 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 4 | 256 | 8 | 32 | common_private | q_outer | 32 | 4 | 255.56 | 64 | 5116 | 80 | 163712 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 0 | 256 | 8 | 32 | common_private | q_outer | 64 | 8 | 675.76 | 32 | 4604 | 144 | 294656 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 1 | 256 | 8 | 32 | common_private | q_outer | 64 | 8 | 682.31 | 32 | 4604 | 144 | 294656 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 2 | 256 | 8 | 32 | common_private | q_outer | 64 | 8 | 672.80 | 32 | 4603 | 144 | 294592 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 3 | 256 | 8 | 32 | common_private | q_outer | 64 | 8 | 683.71 | 32 | 4604 | 144 | 294656 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 4 | 256 | 8 | 32 | common_private | q_outer | 64 | 8 | 676.29 | 32 | 4604 | 144 | 294656 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 0 | 256 | 8 | 32 | common_private | q_outer | 128 | 16 | 3419.40 | 16 | 4348 | 272 | 556544 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 1 | 256 | 8 | 32 | common_private | q_outer | 128 | 16 | 3407.90 | 16 | 4348 | 272 | 556544 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 2 | 256 | 8 | 32 | common_private | q_outer | 128 | 16 | 3414.26 | 16 | 4347 | 272 | 556416 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 3 | 256 | 8 | 32 | common_private | q_outer | 128 | 16 | 3411.86 | 16 | 4348 | 272 | 556544 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 4 | 256 | 8 | 32 | common_private | q_outer | 128 | 16 | 3431.67 | 16 | 4348 | 272 | 556544 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 0 | 256 | 8 | 32 | common_private | signature | 16 | 2 | 106.66 | 384 | 6140 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 8 | 32 | common_private | signature | 16 | 2 | 106.74 | 384 | 6140 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 8 | 32 | common_private | signature | 16 | 2 | 106.61 | 384 | 6139 | 16 | 98224 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 8 | 32 | common_private | signature | 16 | 2 | 106.84 | 384 | 6140 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 8 | 32 | common_private | signature | 16 | 2 | 106.72 | 384 | 6140 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 8 | 32 | common_private | signature | 32 | 4 | 136.58 | 320 | 5116 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 8 | 32 | common_private | signature | 32 | 4 | 136.66 | 320 | 5116 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 8 | 32 | common_private | signature | 32 | 4 | 136.74 | 320 | 5115 | 16 | 98224 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 8 | 32 | common_private | signature | 32 | 4 | 136.72 | 320 | 5116 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 8 | 32 | common_private | signature | 32 | 4 | 136.84 | 320 | 5116 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 8 | 32 | common_private | signature | 64 | 8 | 217.30 | 288 | 4604 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 8 | 32 | common_private | signature | 64 | 8 | 217.80 | 288 | 4604 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 8 | 32 | common_private | signature | 64 | 8 | 217.64 | 288 | 4603 | 16 | 98224 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 8 | 32 | common_private | signature | 64 | 8 | 217.35 | 288 | 4604 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 8 | 32 | common_private | signature | 64 | 8 | 217.30 | 288 | 4604 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 8 | 32 | common_private | signature | 128 | 16 | 591.62 | 272 | 4348 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 256 | 8 | 32 | common_private | signature | 128 | 16 | 591.22 | 272 | 4348 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 256 | 8 | 32 | common_private | signature | 128 | 16 | 589.56 | 272 | 4347 | 16 | 98224 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 256 | 8 | 32 | common_private | signature | 128 | 16 | 589.72 | 272 | 4348 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 256 | 8 | 32 | common_private | signature | 128 | 16 | 589.94 | 272 | 4348 | 16 | 98240 | 256 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 256 | 8 | 32 | random | bounded_merge | 16 | 2 | 175.50 | 143 | 7918 | 63 | 126688 | 0 | 512 | 61.0 | 63.0 | 0.154 |
| 1 | 256 | 8 | 32 | random | bounded_merge | 16 | 2 | 174.63 | 137 | 7899 | 63 | 126384 | 0 | 512 | 62.0 | 63.0 | 0.108 |
| 2 | 256 | 8 | 32 | random | bounded_merge | 16 | 2 | 175.97 | 144 | 7939 | 63 | 127024 | 0 | 512 | 61.0 | 63.0 | 0.160 |
| 3 | 256 | 8 | 32 | random | bounded_merge | 16 | 2 | 181.84 | 140 | 7929 | 63 | 126864 | 0 | 512 | 62.0 | 63.0 | 0.129 |
| 4 | 256 | 8 | 32 | random | bounded_merge | 16 | 2 | 175.57 | 147 | 7945 | 63 | 127120 | 0 | 512 | 61.0 | 63.0 | 0.183 |
| 0 | 256 | 8 | 32 | random | bounded_merge | 32 | 4 | 220.95 | 543 | 7428 | 59 | 119376 | 256 | 512 | 4.0 | 29.0 | 0.754 |
| 1 | 256 | 8 | 32 | random | bounded_merge | 32 | 4 | 215.57 | 551 | 7410 | 57 | 119088 | 252 | 512 | 4.0 | 28.0 | 0.754 |
| 2 | 256 | 8 | 32 | random | bounded_merge | 32 | 4 | 205.28 | 530 | 7467 | 57 | 119936 | 253 | 512 | 4.0 | 28.0 | 0.744 |
| 3 | 256 | 8 | 32 | random | bounded_merge | 32 | 4 | 241.85 | 545 | 7430 | 59 | 119440 | 250 | 512 | 4.0 | 28.0 | 0.734 |
| 4 | 256 | 8 | 32 | random | bounded_merge | 32 | 4 | 164.48 | 523 | 7496 | 32 | 120400 | 252 | 512 | 5.0 | 29.0 | 0.730 |
| 0 | 256 | 8 | 32 | random | bounded_merge | 64 | 8 | 342.31 | 1088 | 6594 | 27 | 108304 | 256 | 512 | 2.0 | 21.0 | 0.967 |
| 1 | 256 | 8 | 32 | random | bounded_merge | 64 | 8 | 341.55 | 1097 | 6597 | 29 | 108464 | 256 | 512 | 2.0 | 21.0 | 0.972 |
| 2 | 256 | 8 | 32 | random | bounded_merge | 64 | 4 | 184.26 | 1086 | 6619 | 28 | 108736 | 256 | 512 | 2.0 | 21.0 | 0.978 |
| 3 | 256 | 8 | 32 | random | bounded_merge | 64 | 4 | 191.90 | 1124 | 6585 | 27 | 108384 | 256 | 512 | 2.0 | 21.0 | 0.979 |
| 4 | 256 | 8 | 32 | random | bounded_merge | 64 | 4 | 185.06 | 1072 | 6626 | 28 | 108528 | 256 | 512 | 2.0 | 21.0 | 0.973 |
| 0 | 256 | 8 | 32 | random | bounded_merge | 128 | 8 | 284.77 | 1943 | 5230 | 19 | 94096 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 256 | 8 | 32 | random | bounded_merge | 128 | 8 | 287.59 | 1942 | 5282 | 21 | 94752 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 256 | 8 | 32 | random | bounded_merge | 128 | 8 | 276.25 | 1908 | 5307 | 20 | 94992 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 256 | 8 | 32 | random | bounded_merge | 128 | 8 | 286.27 | 1951 | 5252 | 21 | 94352 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 256 | 8 | 32 | random | bounded_merge | 128 | 8 | 288.02 | 1946 | 5279 | 20 | 94928 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 256 | 8 | 32 | random | q_outer | 16 | 2 | 159.71 | 128 | 7918 | 64 | 126688 | 0 | 512 | 62.0 | 64.0 | 0.033 |
| 1 | 256 | 8 | 32 | random | q_outer | 16 | 2 | 163.67 | 128 | 7899 | 64 | 126384 | 0 | 512 | 62.0 | 63.0 | 0.036 |
| 2 | 256 | 8 | 32 | random | q_outer | 16 | 2 | 160.98 | 128 | 7939 | 64 | 127024 | 0 | 512 | 62.0 | 64.0 | 0.031 |
| 3 | 256 | 8 | 32 | random | q_outer | 16 | 2 | 160.37 | 128 | 7929 | 64 | 126864 | 0 | 512 | 62.0 | 63.0 | 0.032 |
| 4 | 256 | 8 | 32 | random | q_outer | 16 | 2 | 159.75 | 128 | 7945 | 64 | 127120 | 0 | 512 | 62.0 | 64.0 | 0.030 |
| 0 | 256 | 8 | 32 | random | q_outer | 32 | 4 | 374.58 | 64 | 7428 | 123 | 237696 | 0 | 512 | 116.0 | 120.0 | 0.000 |
| 1 | 256 | 8 | 32 | random | q_outer | 32 | 4 | 376.02 | 64 | 7410 | 124 | 237120 | 0 | 512 | 116.0 | 119.0 | 0.000 |
| 2 | 256 | 8 | 32 | random | q_outer | 32 | 4 | 376.21 | 64 | 7467 | 123 | 238944 | 0 | 512 | 117.0 | 119.7 | 0.000 |
| 3 | 256 | 8 | 32 | random | q_outer | 32 | 4 | 375.01 | 64 | 7430 | 123 | 237760 | 0 | 512 | 116.0 | 120.0 | 0.000 |
| 4 | 256 | 8 | 32 | random | q_outer | 32 | 4 | 371.48 | 64 | 7496 | 122 | 239872 | 0 | 512 | 117.0 | 120.0 | 0.000 |
| 0 | 256 | 8 | 32 | random | q_outer | 64 | 8 | 985.69 | 32 | 6594 | 214 | 422016 | 0 | 512 | 206.0 | 212.9 | 0.000 |
| 1 | 256 | 8 | 32 | random | q_outer | 64 | 8 | 1003.56 | 32 | 6597 | 216 | 422208 | 0 | 512 | 207.0 | 211.9 | 0.000 |
| 2 | 256 | 8 | 32 | random | q_outer | 64 | 8 | 993.88 | 32 | 6619 | 214 | 423616 | 0 | 512 | 207.0 | 211.0 | 0.000 |
| 3 | 256 | 8 | 32 | random | q_outer | 64 | 8 | 990.60 | 32 | 6585 | 216 | 421440 | 0 | 512 | 206.0 | 212.0 | 0.000 |
| 4 | 256 | 8 | 32 | random | q_outer | 64 | 8 | 1015.59 | 32 | 6626 | 222 | 424064 | 0 | 512 | 205.0 | 212.9 | 0.000 |
| 0 | 256 | 8 | 32 | random | q_outer | 128 | 16 | 4201.28 | 16 | 5230 | 339 | 669440 | 0 | 512 | 326.5 | 335.5 | 0.000 |
| 1 | 256 | 8 | 32 | random | q_outer | 128 | 16 | 4224.10 | 16 | 5282 | 345 | 676096 | 0 | 512 | 329.5 | 339.5 | 0.000 |
| 2 | 256 | 8 | 32 | random | q_outer | 128 | 16 | 4270.45 | 16 | 5307 | 346 | 679296 | 0 | 512 | 330.0 | 340.0 | 0.000 |
| 3 | 256 | 8 | 32 | random | q_outer | 128 | 16 | 4198.29 | 16 | 5252 | 342 | 672256 | 0 | 512 | 331.0 | 336.5 | 0.000 |
| 4 | 256 | 8 | 32 | random | q_outer | 128 | 16 | 4340.76 | 16 | 5279 | 352 | 675712 | 0 | 512 | 329.0 | 339.5 | 0.000 |
| 0 | 256 | 8 | 32 | random | signature | 16 | 2 | 116.20 | 369 | 7918 | 32 | 126688 | 226 | 512 | 29.0 | 31.0 | 1.000 |
| 1 | 256 | 8 | 32 | random | signature | 16 | 2 | 115.74 | 375 | 7899 | 32 | 126384 | 238 | 512 | 29.0 | 31.0 | 1.000 |
| 2 | 256 | 8 | 32 | random | signature | 16 | 2 | 115.48 | 368 | 7939 | 32 | 127024 | 224 | 512 | 29.0 | 31.0 | 1.000 |
| 3 | 256 | 8 | 32 | random | signature | 16 | 2 | 111.86 | 372 | 7929 | 32 | 126864 | 232 | 512 | 29.0 | 31.0 | 1.000 |
| 4 | 256 | 8 | 32 | random | signature | 16 | 2 | 115.93 | 365 | 7945 | 32 | 127120 | 218 | 512 | 29.0 | 32.0 | 1.000 |
| 0 | 256 | 8 | 32 | random | signature | 32 | 4 | 167.79 | 612 | 7428 | 31 | 119376 | 256 | 512 | 3.0 | 28.0 | 1.000 |
| 1 | 256 | 8 | 32 | random | signature | 32 | 4 | 171.46 | 621 | 7410 | 32 | 119088 | 255 | 512 | 3.0 | 28.0 | 1.000 |
| 2 | 256 | 8 | 32 | random | signature | 32 | 4 | 170.42 | 602 | 7467 | 31 | 119936 | 256 | 512 | 3.0 | 28.0 | 1.000 |
| 3 | 256 | 8 | 32 | random | signature | 32 | 4 | 163.30 | 620 | 7430 | 32 | 119440 | 255 | 512 | 3.0 | 28.0 | 1.000 |
| 4 | 256 | 8 | 32 | random | signature | 32 | 4 | 167.30 | 599 | 7496 | 32 | 120400 | 255 | 512 | 3.0 | 28.0 | 1.000 |
| 0 | 256 | 8 | 32 | random | signature | 64 | 8 | 306.56 | 1098 | 6594 | 27 | 108304 | 256 | 512 | 2.0 | 21.0 | 1.000 |
| 1 | 256 | 8 | 32 | random | signature | 64 | 8 | 303.70 | 1105 | 6597 | 28 | 108464 | 256 | 512 | 2.0 | 21.0 | 1.000 |
| 2 | 256 | 8 | 32 | random | signature | 64 | 4 | 176.26 | 1093 | 6619 | 26 | 108736 | 256 | 512 | 2.0 | 21.0 | 1.000 |
| 3 | 256 | 8 | 32 | random | signature | 64 | 4 | 182.26 | 1130 | 6585 | 26 | 108384 | 256 | 512 | 2.0 | 21.0 | 1.000 |
| 4 | 256 | 8 | 32 | random | signature | 64 | 4 | 177.83 | 1080 | 6626 | 28 | 108528 | 256 | 512 | 2.0 | 21.0 | 1.000 |
| 0 | 256 | 8 | 32 | random | signature | 128 | 8 | 276.45 | 1943 | 5230 | 19 | 94096 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 256 | 8 | 32 | random | signature | 128 | 8 | 277.49 | 1942 | 5282 | 21 | 94752 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 256 | 8 | 32 | random | signature | 128 | 8 | 276.06 | 1908 | 5307 | 20 | 94992 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 256 | 8 | 32 | random | signature | 128 | 8 | 286.05 | 1951 | 5252 | 21 | 94352 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 256 | 8 | 32 | random | signature | 128 | 8 | 288.98 | 1946 | 5279 | 20 | 94928 | 256 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 512 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 84.54 | 489 | 3994 | 31 | 63904 | 0 | 509 | 8.0 | 8.0 | 0.964 |
| 1 | 512 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 68.23 | 494 | 4031 | 23 | 64496 | 0 | 509 | 8.0 | 8.0 | 0.966 |
| 2 | 512 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 69.95 | 484 | 3991 | 23 | 63856 | 0 | 509 | 8.0 | 8.0 | 0.946 |
| 3 | 512 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 88.16 | 493 | 4019 | 23 | 64304 | 0 | 511 | 8.0 | 8.0 | 0.964 |
| 4 | 512 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 80.06 | 486 | 4018 | 23 | 64288 | 0 | 511 | 8.0 | 8.0 | 0.944 |
| 0 | 512 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 61.00 | 518 | 3887 | 8 | 62192 | 50 | 509 | 8.0 | 8.0 | 0.969 |
| 1 | 512 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 61.25 | 515 | 3867 | 8 | 61872 | 51 | 509 | 8.0 | 8.0 | 0.968 |
| 2 | 512 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 61.40 | 516 | 3889 | 8 | 62224 | 53 | 509 | 8.0 | 8.0 | 0.965 |
| 3 | 512 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 61.71 | 520 | 3860 | 8 | 61760 | 52 | 511 | 8.0 | 8.0 | 0.977 |
| 4 | 512 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 62.14 | 524 | 3875 | 8 | 62000 | 61 | 511 | 8.0 | 8.0 | 0.967 |
| 0 | 512 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 62.90 | 585 | 3609 | 8 | 57744 | 164 | 509 | 8.0 | 8.0 | 0.979 |
| 1 | 512 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 62.62 | 569 | 3639 | 8 | 58224 | 148 | 509 | 8.0 | 8.0 | 0.977 |
| 2 | 512 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 63.77 | 574 | 3642 | 8 | 58272 | 163 | 509 | 8.0 | 8.0 | 0.976 |
| 3 | 512 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 62.89 | 576 | 3615 | 8 | 57840 | 153 | 511 | 8.0 | 8.0 | 0.980 |
| 4 | 512 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 64.00 | 577 | 3631 | 8 | 58096 | 158 | 511 | 8.0 | 8.0 | 0.980 |
| 0 | 512 | 4 | 8 | clustered | bounded_merge | 128 | 4 | 65.58 | 656 | 3215 | 8 | 51440 | 284 | 509 | 5.0 | 8.0 | 1.000 |
| 1 | 512 | 4 | 8 | clustered | bounded_merge | 128 | 4 | 66.25 | 646 | 3280 | 8 | 52480 | 277 | 509 | 5.0 | 8.0 | 1.000 |
| 2 | 512 | 4 | 8 | clustered | bounded_merge | 128 | 4 | 65.55 | 646 | 3194 | 8 | 51104 | 305 | 509 | 5.0 | 8.0 | 1.000 |
| 3 | 512 | 4 | 8 | clustered | bounded_merge | 128 | 8 | 88.06 | 663 | 3223 | 8 | 51616 | 294 | 511 | 5.0 | 8.0 | 1.000 |
| 4 | 512 | 4 | 8 | clustered | bounded_merge | 128 | 4 | 66.85 | 670 | 3271 | 8 | 52336 | 302 | 511 | 5.0 | 8.0 | 1.000 |
| 0 | 512 | 4 | 8 | clustered | q_outer | 16 | 4 | 79.41 | 128 | 3994 | 32 | 63904 | 0 | 509 | 32.0 | 32.0 | 0.000 |
| 1 | 512 | 4 | 8 | clustered | q_outer | 16 | 4 | 80.50 | 128 | 4031 | 32 | 64496 | 0 | 509 | 32.0 | 32.0 | 0.000 |
| 2 | 512 | 4 | 8 | clustered | q_outer | 16 | 4 | 79.62 | 128 | 3991 | 32 | 63856 | 0 | 509 | 32.0 | 32.0 | 0.000 |
| 3 | 512 | 4 | 8 | clustered | q_outer | 16 | 4 | 79.70 | 128 | 4019 | 32 | 64304 | 0 | 511 | 32.0 | 32.0 | 0.000 |
| 4 | 512 | 4 | 8 | clustered | q_outer | 16 | 4 | 78.98 | 128 | 4018 | 32 | 64288 | 0 | 511 | 32.0 | 32.0 | 0.000 |
| 0 | 512 | 4 | 8 | clustered | q_outer | 32 | 8 | 209.24 | 64 | 3887 | 64 | 124384 | 0 | 509 | 63.0 | 64.0 | 0.000 |
| 1 | 512 | 4 | 8 | clustered | q_outer | 32 | 8 | 214.95 | 64 | 3867 | 64 | 123744 | 0 | 509 | 62.0 | 64.0 | 0.000 |
| 2 | 512 | 4 | 8 | clustered | q_outer | 32 | 8 | 213.29 | 64 | 3889 | 64 | 124448 | 0 | 509 | 62.0 | 64.0 | 0.000 |
| 3 | 512 | 4 | 8 | clustered | q_outer | 32 | 8 | 214.68 | 64 | 3860 | 64 | 123520 | 0 | 511 | 64.0 | 64.0 | 0.000 |
| 4 | 512 | 4 | 8 | clustered | q_outer | 32 | 8 | 212.67 | 64 | 3875 | 64 | 124000 | 0 | 511 | 62.0 | 64.0 | 0.000 |
| 0 | 512 | 4 | 8 | clustered | q_outer | 64 | 16 | 628.77 | 32 | 3609 | 128 | 230976 | 0 | 509 | 112.5 | 124.0 | 0.000 |
| 1 | 512 | 4 | 8 | clustered | q_outer | 64 | 16 | 611.59 | 32 | 3639 | 123 | 232896 | 0 | 509 | 115.0 | 121.0 | 0.000 |
| 2 | 512 | 4 | 8 | clustered | q_outer | 64 | 16 | 635.06 | 32 | 3642 | 128 | 233088 | 0 | 509 | 114.5 | 123.9 | 0.000 |
| 3 | 512 | 4 | 8 | clustered | q_outer | 64 | 16 | 601.41 | 32 | 3615 | 121 | 231360 | 0 | 511 | 114.5 | 119.0 | 0.000 |
| 4 | 512 | 4 | 8 | clustered | q_outer | 64 | 16 | 601.93 | 32 | 3631 | 127 | 232384 | 0 | 511 | 114.0 | 120.9 | 0.000 |
| 0 | 512 | 4 | 8 | clustered | q_outer | 128 | 32 | 2510.20 | 16 | 3215 | 224 | 411520 | 0 | 509 | 199.0 | 218.5 | 0.000 |
| 1 | 512 | 4 | 8 | clustered | q_outer | 128 | 32 | 2435.71 | 16 | 3280 | 218 | 419840 | 0 | 509 | 204.0 | 218.0 | 0.000 |
| 2 | 512 | 4 | 8 | clustered | q_outer | 128 | 32 | 2454.94 | 16 | 3194 | 218 | 408832 | 0 | 509 | 200.5 | 218.0 | 0.000 |
| 3 | 512 | 4 | 8 | clustered | q_outer | 128 | 32 | 2421.12 | 16 | 3223 | 217 | 412544 | 0 | 511 | 204.5 | 212.0 | 0.000 |
| 4 | 512 | 4 | 8 | clustered | q_outer | 128 | 32 | 2453.74 | 16 | 3271 | 218 | 418688 | 0 | 511 | 206.5 | 215.0 | 0.000 |
| 0 | 512 | 4 | 8 | clustered | signature | 16 | 2 | 62.59 | 528 | 3994 | 8 | 63904 | 35 | 509 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 4 | 8 | clustered | signature | 16 | 2 | 62.25 | 522 | 4031 | 8 | 64496 | 24 | 509 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 4 | 8 | clustered | signature | 16 | 2 | 61.74 | 528 | 3991 | 8 | 63856 | 40 | 509 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 4 | 8 | clustered | signature | 16 | 2 | 62.33 | 526 | 4019 | 8 | 64304 | 32 | 511 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 4 | 8 | clustered | signature | 16 | 4 | 65.70 | 526 | 4018 | 8 | 64288 | 29 | 511 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 4 | 8 | clustered | signature | 32 | 4 | 65.50 | 549 | 3887 | 8 | 62192 | 79 | 509 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 4 | 8 | clustered | signature | 32 | 4 | 66.19 | 551 | 3867 | 8 | 61872 | 86 | 509 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 4 | 8 | clustered | signature | 32 | 4 | 67.15 | 549 | 3889 | 8 | 62224 | 85 | 509 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 4 | 8 | clustered | signature | 32 | 4 | 67.18 | 546 | 3860 | 8 | 61760 | 75 | 511 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 4 | 8 | clustered | signature | 32 | 4 | 68.23 | 559 | 3875 | 8 | 62000 | 93 | 511 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 4 | 8 | clustered | signature | 64 | 4 | 66.70 | 605 | 3609 | 8 | 57744 | 181 | 509 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 4 | 8 | clustered | signature | 64 | 4 | 66.87 | 592 | 3639 | 8 | 58224 | 167 | 509 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 4 | 8 | clustered | signature | 64 | 4 | 68.55 | 594 | 3642 | 8 | 58272 | 180 | 509 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 4 | 8 | clustered | signature | 64 | 4 | 67.03 | 596 | 3615 | 8 | 57840 | 172 | 511 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 4 | 8 | clustered | signature | 64 | 4 | 68.83 | 598 | 3631 | 8 | 58096 | 176 | 511 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 4 | 8 | clustered | signature | 128 | 4 | 65.62 | 656 | 3215 | 8 | 51440 | 284 | 509 | 5.0 | 8.0 | 1.000 |
| 1 | 512 | 4 | 8 | clustered | signature | 128 | 4 | 65.58 | 646 | 3280 | 8 | 52480 | 277 | 509 | 5.0 | 8.0 | 1.000 |
| 2 | 512 | 4 | 8 | clustered | signature | 128 | 4 | 64.91 | 646 | 3194 | 8 | 51104 | 305 | 509 | 5.0 | 8.0 | 1.000 |
| 3 | 512 | 4 | 8 | clustered | signature | 128 | 8 | 85.09 | 663 | 3223 | 8 | 51616 | 294 | 511 | 5.0 | 8.0 | 1.000 |
| 4 | 512 | 4 | 8 | clustered | signature | 128 | 4 | 66.49 | 670 | 3271 | 8 | 52336 | 302 | 511 | 5.0 | 8.0 | 1.000 |
| 0 | 512 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 53.20 | 128 | 2554 | 20 | 40864 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 1 | 512 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 53.75 | 128 | 2554 | 20 | 40864 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 2 | 512 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 55.15 | 128 | 2554 | 20 | 40864 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 3 | 512 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 54.53 | 128 | 2554 | 20 | 40864 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 4 | 512 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 54.48 | 128 | 2554 | 20 | 40864 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 0 | 512 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 66.38 | 576 | 2298 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 512 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 66.29 | 576 | 2298 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 512 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 66.14 | 576 | 2298 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 512 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 66.27 | 576 | 2298 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 512 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 65.98 | 576 | 2298 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 512 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 117.50 | 544 | 2170 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 512 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 117.52 | 544 | 2170 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 512 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 117.74 | 544 | 2170 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 512 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 117.91 | 544 | 2170 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 512 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 116.93 | 544 | 2170 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 512 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 259.85 | 528 | 2106 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 512 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 259.25 | 528 | 2106 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 512 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 259.35 | 528 | 2106 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 512 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 260.07 | 528 | 2106 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 512 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 259.82 | 528 | 2106 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 512 | 4 | 8 | common_private | q_outer | 16 | 4 | 55.67 | 128 | 2554 | 20 | 40864 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 1 | 512 | 4 | 8 | common_private | q_outer | 16 | 4 | 55.93 | 128 | 2554 | 20 | 40864 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 2 | 512 | 4 | 8 | common_private | q_outer | 16 | 4 | 56.29 | 128 | 2554 | 20 | 40864 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 3 | 512 | 4 | 8 | common_private | q_outer | 16 | 4 | 56.44 | 128 | 2554 | 20 | 40864 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 4 | 512 | 4 | 8 | common_private | q_outer | 16 | 4 | 55.05 | 128 | 2554 | 20 | 40864 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 0 | 512 | 4 | 8 | common_private | q_outer | 32 | 8 | 126.97 | 64 | 2298 | 36 | 73536 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 1 | 512 | 4 | 8 | common_private | q_outer | 32 | 8 | 127.71 | 64 | 2298 | 36 | 73536 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 2 | 512 | 4 | 8 | common_private | q_outer | 32 | 8 | 127.02 | 64 | 2298 | 36 | 73536 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 3 | 512 | 4 | 8 | common_private | q_outer | 32 | 8 | 126.44 | 64 | 2298 | 36 | 73536 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 4 | 512 | 4 | 8 | common_private | q_outer | 32 | 8 | 126.44 | 64 | 2298 | 36 | 73536 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 0 | 512 | 4 | 8 | common_private | q_outer | 64 | 16 | 351.44 | 32 | 2170 | 68 | 138880 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 1 | 512 | 4 | 8 | common_private | q_outer | 64 | 16 | 350.08 | 32 | 2170 | 68 | 138880 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 2 | 512 | 4 | 8 | common_private | q_outer | 64 | 16 | 349.83 | 32 | 2170 | 68 | 138880 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 3 | 512 | 4 | 8 | common_private | q_outer | 64 | 16 | 352.80 | 32 | 2170 | 68 | 138880 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 4 | 512 | 4 | 8 | common_private | q_outer | 64 | 16 | 350.66 | 32 | 2170 | 68 | 138880 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 0 | 512 | 4 | 8 | common_private | q_outer | 128 | 32 | 1557.87 | 16 | 2106 | 132 | 269568 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 1 | 512 | 4 | 8 | common_private | q_outer | 128 | 32 | 1541.83 | 16 | 2106 | 132 | 269568 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 2 | 512 | 4 | 8 | common_private | q_outer | 128 | 32 | 1545.71 | 16 | 2106 | 132 | 269568 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 3 | 512 | 4 | 8 | common_private | q_outer | 128 | 32 | 1569.97 | 16 | 2106 | 132 | 269568 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 4 | 512 | 4 | 8 | common_private | q_outer | 128 | 32 | 1553.90 | 16 | 2106 | 132 | 269568 | 0 | 512 | 132.0 | 132.0 | 0.030 |
| 0 | 512 | 4 | 8 | common_private | signature | 16 | 4 | 53.20 | 640 | 2554 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 512 | 4 | 8 | common_private | signature | 16 | 4 | 53.46 | 640 | 2554 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 512 | 4 | 8 | common_private | signature | 16 | 4 | 53.58 | 640 | 2554 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 512 | 4 | 8 | common_private | signature | 16 | 4 | 53.20 | 640 | 2554 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 512 | 4 | 8 | common_private | signature | 16 | 4 | 53.82 | 640 | 2554 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 512 | 4 | 8 | common_private | signature | 32 | 8 | 66.28 | 576 | 2298 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 512 | 4 | 8 | common_private | signature | 32 | 8 | 66.05 | 576 | 2298 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 512 | 4 | 8 | common_private | signature | 32 | 8 | 66.93 | 576 | 2298 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 512 | 4 | 8 | common_private | signature | 32 | 8 | 66.30 | 576 | 2298 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 512 | 4 | 8 | common_private | signature | 32 | 8 | 65.99 | 576 | 2298 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 512 | 4 | 8 | common_private | signature | 64 | 16 | 117.40 | 544 | 2170 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 512 | 4 | 8 | common_private | signature | 64 | 16 | 117.28 | 544 | 2170 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 512 | 4 | 8 | common_private | signature | 64 | 16 | 118.42 | 544 | 2170 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 512 | 4 | 8 | common_private | signature | 64 | 16 | 117.73 | 544 | 2170 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 512 | 4 | 8 | common_private | signature | 64 | 16 | 117.01 | 544 | 2170 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 512 | 4 | 8 | common_private | signature | 128 | 32 | 259.82 | 528 | 2106 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 512 | 4 | 8 | common_private | signature | 128 | 32 | 258.98 | 528 | 2106 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 512 | 4 | 8 | common_private | signature | 128 | 32 | 259.27 | 528 | 2106 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 512 | 4 | 8 | common_private | signature | 128 | 32 | 259.11 | 528 | 2106 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 512 | 4 | 8 | common_private | signature | 128 | 32 | 258.77 | 528 | 2106 | 4 | 40864 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 512 | 4 | 8 | random | bounded_merge | 16 | 4 | 102.98 | 361 | 3988 | 31 | 63808 | 0 | 512 | 8.0 | 23.0 | 0.577 |
| 1 | 512 | 4 | 8 | random | bounded_merge | 16 | 4 | 101.05 | 358 | 3989 | 31 | 63824 | 0 | 512 | 8.0 | 23.0 | 0.580 |
| 2 | 512 | 4 | 8 | random | bounded_merge | 16 | 4 | 103.71 | 355 | 3991 | 31 | 63856 | 0 | 512 | 8.0 | 23.0 | 0.579 |
| 3 | 512 | 4 | 8 | random | bounded_merge | 16 | 4 | 102.90 | 348 | 3983 | 31 | 63728 | 0 | 512 | 8.0 | 23.0 | 0.551 |
| 4 | 512 | 4 | 8 | random | bounded_merge | 16 | 4 | 100.07 | 336 | 3985 | 31 | 63760 | 0 | 512 | 8.0 | 23.0 | 0.516 |
| 0 | 512 | 4 | 8 | random | bounded_merge | 32 | 4 | 66.03 | 650 | 3876 | 8 | 62016 | 253 | 512 | 7.0 | 8.0 | 0.903 |
| 1 | 512 | 4 | 8 | random | bounded_merge | 32 | 4 | 65.96 | 653 | 3871 | 8 | 61936 | 259 | 512 | 7.0 | 8.0 | 0.900 |
| 2 | 512 | 4 | 8 | random | bounded_merge | 32 | 4 | 66.14 | 657 | 3869 | 8 | 61904 | 259 | 512 | 7.0 | 8.0 | 0.899 |
| 3 | 512 | 4 | 8 | random | bounded_merge | 32 | 4 | 66.74 | 656 | 3868 | 8 | 61888 | 267 | 512 | 7.0 | 8.0 | 0.902 |
| 4 | 512 | 4 | 8 | random | bounded_merge | 32 | 4 | 65.04 | 657 | 3863 | 8 | 61808 | 271 | 512 | 7.0 | 8.0 | 0.896 |
| 0 | 512 | 4 | 8 | random | bounded_merge | 64 | 4 | 68.61 | 903 | 3611 | 8 | 57776 | 425 | 512 | 5.0 | 7.0 | 0.958 |
| 1 | 512 | 4 | 8 | random | bounded_merge | 64 | 4 | 68.66 | 889 | 3641 | 8 | 58256 | 423 | 512 | 5.0 | 7.0 | 0.959 |
| 2 | 512 | 4 | 8 | random | bounded_merge | 64 | 4 | 69.92 | 885 | 3639 | 8 | 58224 | 430 | 512 | 5.0 | 7.0 | 0.955 |
| 3 | 512 | 4 | 8 | random | bounded_merge | 64 | 4 | 69.06 | 885 | 3649 | 8 | 58384 | 428 | 512 | 5.0 | 7.0 | 0.958 |
| 4 | 512 | 4 | 8 | random | bounded_merge | 64 | 4 | 68.38 | 870 | 3649 | 8 | 58384 | 436 | 512 | 5.0 | 7.0 | 0.952 |
| 0 | 512 | 4 | 8 | random | bounded_merge | 128 | 8 | 92.79 | 1254 | 3178 | 8 | 50880 | 498 | 512 | 1.0 | 6.0 | 1.000 |
| 1 | 512 | 4 | 8 | random | bounded_merge | 128 | 8 | 94.27 | 1187 | 3272 | 8 | 52368 | 499 | 512 | 1.0 | 6.0 | 1.000 |
| 2 | 512 | 4 | 8 | random | bounded_merge | 128 | 8 | 93.63 | 1203 | 3245 | 8 | 51968 | 509 | 512 | 1.0 | 6.0 | 1.000 |
| 3 | 512 | 4 | 8 | random | bounded_merge | 128 | 4 | 72.70 | 1225 | 3238 | 8 | 51808 | 500 | 512 | 1.0 | 6.0 | 1.000 |
| 4 | 512 | 4 | 8 | random | bounded_merge | 128 | 8 | 92.52 | 1216 | 3247 | 8 | 51984 | 505 | 512 | 1.0 | 6.0 | 1.000 |
| 0 | 512 | 4 | 8 | random | q_outer | 16 | 4 | 81.58 | 128 | 3988 | 32 | 63808 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 1 | 512 | 4 | 8 | random | q_outer | 16 | 4 | 80.27 | 128 | 3989 | 32 | 63824 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 2 | 512 | 4 | 8 | random | q_outer | 16 | 4 | 80.75 | 128 | 3991 | 32 | 63856 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 3 | 512 | 4 | 8 | random | q_outer | 16 | 4 | 81.06 | 128 | 3983 | 32 | 63728 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 4 | 512 | 4 | 8 | random | q_outer | 16 | 4 | 80.39 | 128 | 3985 | 32 | 63760 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 0 | 512 | 4 | 8 | random | q_outer | 32 | 8 | 207.53 | 64 | 3876 | 64 | 124032 | 0 | 512 | 61.0 | 63.0 | 0.000 |
| 1 | 512 | 4 | 8 | random | q_outer | 32 | 8 | 212.86 | 64 | 3871 | 64 | 123872 | 0 | 512 | 60.0 | 62.7 | 0.000 |
| 2 | 512 | 4 | 8 | random | q_outer | 32 | 8 | 209.44 | 64 | 3869 | 63 | 123808 | 0 | 512 | 60.5 | 63.0 | 0.000 |
| 3 | 512 | 4 | 8 | random | q_outer | 32 | 8 | 213.00 | 64 | 3868 | 64 | 123776 | 0 | 512 | 61.0 | 62.0 | 0.000 |
| 4 | 512 | 4 | 8 | random | q_outer | 32 | 8 | 207.35 | 64 | 3863 | 64 | 123616 | 0 | 512 | 60.0 | 62.7 | 0.000 |
| 0 | 512 | 4 | 8 | random | q_outer | 64 | 16 | 597.95 | 32 | 3611 | 122 | 231104 | 0 | 512 | 112.5 | 117.9 | 0.000 |
| 1 | 512 | 4 | 8 | random | q_outer | 64 | 16 | 581.73 | 32 | 3641 | 120 | 233024 | 0 | 512 | 114.0 | 116.9 | 0.000 |
| 2 | 512 | 4 | 8 | random | q_outer | 64 | 16 | 603.30 | 32 | 3639 | 122 | 232896 | 0 | 512 | 113.5 | 117.9 | 0.000 |
| 3 | 512 | 4 | 8 | random | q_outer | 64 | 16 | 593.77 | 32 | 3649 | 120 | 233536 | 0 | 512 | 114.0 | 117.0 | 0.000 |
| 4 | 512 | 4 | 8 | random | q_outer | 64 | 16 | 584.24 | 32 | 3649 | 119 | 233536 | 0 | 512 | 114.0 | 117.0 | 0.000 |
| 0 | 512 | 4 | 8 | random | q_outer | 128 | 32 | 2342.50 | 16 | 3178 | 207 | 406784 | 0 | 512 | 197.5 | 206.0 | 0.000 |
| 1 | 512 | 4 | 8 | random | q_outer | 128 | 32 | 2378.56 | 16 | 3272 | 212 | 418816 | 0 | 512 | 207.0 | 210.5 | 0.000 |
| 2 | 512 | 4 | 8 | random | q_outer | 128 | 32 | 2405.19 | 16 | 3245 | 214 | 415360 | 0 | 512 | 203.0 | 207.5 | 0.000 |
| 3 | 512 | 4 | 8 | random | q_outer | 128 | 32 | 2385.34 | 16 | 3238 | 212 | 414464 | 0 | 512 | 203.0 | 210.0 | 0.000 |
| 4 | 512 | 4 | 8 | random | q_outer | 128 | 32 | 2417.93 | 16 | 3247 | 216 | 415616 | 0 | 512 | 203.0 | 210.5 | 0.000 |
| 0 | 512 | 4 | 8 | random | signature | 16 | 4 | 68.01 | 602 | 3988 | 8 | 63808 | 163 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 4 | 8 | random | signature | 16 | 4 | 66.88 | 597 | 3989 | 8 | 63824 | 152 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 4 | 8 | random | signature | 16 | 2 | 61.70 | 605 | 3991 | 8 | 63856 | 158 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 4 | 8 | random | signature | 16 | 4 | 67.77 | 606 | 3983 | 8 | 63728 | 169 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 4 | 8 | random | signature | 16 | 4 | 67.03 | 607 | 3985 | 8 | 63760 | 171 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 4 | 8 | random | signature | 32 | 4 | 71.60 | 708 | 3876 | 8 | 62016 | 291 | 512 | 7.0 | 8.0 | 1.000 |
| 1 | 512 | 4 | 8 | random | signature | 32 | 4 | 70.42 | 712 | 3871 | 8 | 61936 | 300 | 512 | 7.0 | 8.0 | 1.000 |
| 2 | 512 | 4 | 8 | random | signature | 32 | 4 | 71.44 | 718 | 3869 | 8 | 61904 | 294 | 512 | 7.0 | 8.0 | 1.000 |
| 3 | 512 | 4 | 8 | random | signature | 32 | 4 | 71.46 | 716 | 3868 | 8 | 61888 | 302 | 512 | 7.0 | 8.0 | 1.000 |
| 4 | 512 | 4 | 8 | random | signature | 32 | 4 | 70.95 | 719 | 3863 | 8 | 61808 | 310 | 512 | 7.0 | 8.0 | 1.000 |
| 0 | 512 | 4 | 8 | random | signature | 64 | 4 | 72.65 | 929 | 3611 | 8 | 57776 | 433 | 512 | 5.0 | 7.0 | 1.000 |
| 1 | 512 | 4 | 8 | random | signature | 64 | 4 | 72.91 | 914 | 3641 | 8 | 58256 | 433 | 512 | 5.0 | 7.0 | 1.000 |
| 2 | 512 | 4 | 8 | random | signature | 64 | 4 | 73.36 | 913 | 3639 | 8 | 58224 | 437 | 512 | 5.0 | 7.0 | 1.000 |
| 3 | 512 | 4 | 8 | random | signature | 64 | 4 | 73.49 | 910 | 3649 | 8 | 58384 | 438 | 512 | 5.0 | 7.0 | 1.000 |
| 4 | 512 | 4 | 8 | random | signature | 64 | 4 | 72.46 | 901 | 3649 | 8 | 58384 | 447 | 512 | 5.0 | 7.0 | 1.000 |
| 0 | 512 | 4 | 8 | random | signature | 128 | 8 | 92.88 | 1254 | 3178 | 8 | 50880 | 498 | 512 | 1.0 | 6.0 | 1.000 |
| 1 | 512 | 4 | 8 | random | signature | 128 | 8 | 94.45 | 1187 | 3272 | 8 | 52368 | 499 | 512 | 1.0 | 6.0 | 1.000 |
| 2 | 512 | 4 | 8 | random | signature | 128 | 8 | 93.12 | 1203 | 3245 | 8 | 51968 | 509 | 512 | 1.0 | 6.0 | 1.000 |
| 3 | 512 | 4 | 8 | random | signature | 128 | 4 | 72.60 | 1225 | 3238 | 8 | 51808 | 500 | 512 | 1.0 | 6.0 | 1.000 |
| 4 | 512 | 4 | 8 | random | signature | 128 | 8 | 93.51 | 1216 | 3247 | 8 | 51984 | 505 | 512 | 1.0 | 6.0 | 1.000 |
| 0 | 512 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 151.39 | 468 | 7816 | 41 | 125056 | 0 | 509 | 16.0 | 16.0 | 0.919 |
| 1 | 512 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 170.63 | 462 | 7895 | 50 | 126320 | 0 | 509 | 16.0 | 16.0 | 0.888 |
| 2 | 512 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 195.89 | 459 | 7805 | 56 | 124880 | 0 | 509 | 16.0 | 16.0 | 0.894 |
| 3 | 512 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 151.57 | 470 | 7888 | 41 | 126208 | 0 | 511 | 16.0 | 16.0 | 0.912 |
| 4 | 512 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 180.02 | 465 | 7871 | 50 | 125936 | 0 | 511 | 16.0 | 16.0 | 0.905 |
| 0 | 512 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 105.89 | 540 | 7412 | 16 | 118592 | 109 | 509 | 16.0 | 16.0 | 0.957 |
| 1 | 512 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 105.51 | 542 | 7299 | 16 | 116784 | 129 | 509 | 16.0 | 16.0 | 0.939 |
| 2 | 512 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 109.58 | 547 | 7380 | 16 | 118080 | 125 | 509 | 16.0 | 16.0 | 0.952 |
| 3 | 512 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 107.22 | 546 | 7362 | 16 | 117792 | 123 | 511 | 16.0 | 16.0 | 0.949 |
| 4 | 512 | 4 | 16 | clustered | bounded_merge | 32 | 8 | 158.44 | 553 | 7298 | 16 | 116800 | 131 | 511 | 16.0 | 16.0 | 0.941 |
| 0 | 512 | 4 | 16 | clustered | bounded_merge | 64 | 4 | 106.64 | 651 | 6471 | 27 | 103536 | 283 | 509 | 10.0 | 16.0 | 0.978 |
| 1 | 512 | 4 | 16 | clustered | bounded_merge | 64 | 4 | 118.40 | 650 | 6454 | 27 | 103264 | 297 | 509 | 10.0 | 16.0 | 0.960 |
| 2 | 512 | 4 | 16 | clustered | bounded_merge | 64 | 4 | 117.06 | 660 | 6444 | 27 | 103104 | 303 | 509 | 10.0 | 16.0 | 0.967 |
| 3 | 512 | 4 | 16 | clustered | bounded_merge | 64 | 4 | 127.46 | 643 | 6517 | 27 | 104272 | 273 | 511 | 11.0 | 16.0 | 0.966 |
| 4 | 512 | 4 | 16 | clustered | bounded_merge | 64 | 8 | 183.81 | 649 | 6522 | 27 | 104512 | 274 | 511 | 11.0 | 16.0 | 0.967 |
| 0 | 512 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 144.89 | 761 | 5272 | 16 | 84560 | 401 | 509 | 6.0 | 16.0 | 1.000 |
| 1 | 512 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 137.11 | 779 | 5240 | 16 | 83952 | 441 | 509 | 6.0 | 14.0 | 1.000 |
| 2 | 512 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 141.86 | 778 | 5078 | 16 | 81360 | 443 | 509 | 6.0 | 14.0 | 1.000 |
| 3 | 512 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 141.58 | 780 | 5219 | 16 | 83744 | 437 | 511 | 6.0 | 15.0 | 1.000 |
| 4 | 512 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 145.04 | 786 | 5241 | 16 | 84064 | 430 | 511 | 5.5 | 16.0 | 1.000 |
| 0 | 512 | 4 | 16 | clustered | q_outer | 16 | 4 | 149.36 | 128 | 7816 | 64 | 125056 | 0 | 509 | 64.0 | 64.0 | 0.000 |
| 1 | 512 | 4 | 16 | clustered | q_outer | 16 | 4 | 151.19 | 128 | 7895 | 64 | 126320 | 0 | 509 | 64.0 | 64.0 | 0.000 |
| 2 | 512 | 4 | 16 | clustered | q_outer | 16 | 4 | 148.86 | 128 | 7805 | 64 | 124880 | 0 | 509 | 64.0 | 64.0 | 0.000 |
| 3 | 512 | 4 | 16 | clustered | q_outer | 16 | 4 | 147.95 | 128 | 7888 | 64 | 126208 | 0 | 511 | 64.0 | 64.0 | 0.000 |
| 4 | 512 | 4 | 16 | clustered | q_outer | 16 | 4 | 149.32 | 128 | 7871 | 64 | 125936 | 0 | 511 | 64.0 | 64.0 | 0.000 |
| 0 | 512 | 4 | 16 | clustered | q_outer | 32 | 8 | 411.89 | 64 | 7412 | 128 | 237184 | 0 | 509 | 117.0 | 128.0 | 0.000 |
| 1 | 512 | 4 | 16 | clustered | q_outer | 32 | 8 | 409.44 | 64 | 7299 | 128 | 233568 | 0 | 509 | 115.0 | 125.7 | 0.000 |
| 2 | 512 | 4 | 16 | clustered | q_outer | 32 | 8 | 412.08 | 64 | 7380 | 128 | 236160 | 0 | 509 | 116.0 | 127.7 | 0.000 |
| 3 | 512 | 4 | 16 | clustered | q_outer | 32 | 8 | 409.98 | 64 | 7362 | 128 | 235584 | 0 | 511 | 117.5 | 128.0 | 0.000 |
| 4 | 512 | 4 | 16 | clustered | q_outer | 32 | 8 | 411.98 | 64 | 7298 | 128 | 233536 | 0 | 511 | 115.0 | 127.7 | 0.000 |
| 0 | 512 | 4 | 16 | clustered | q_outer | 64 | 16 | 1171.06 | 32 | 6471 | 242 | 414144 | 0 | 509 | 199.0 | 231.8 | 0.000 |
| 1 | 512 | 4 | 16 | clustered | q_outer | 64 | 16 | 1089.19 | 32 | 6454 | 225 | 413056 | 0 | 509 | 201.0 | 219.8 | 0.000 |
| 2 | 512 | 4 | 16 | clustered | q_outer | 64 | 16 | 1138.29 | 32 | 6444 | 235 | 412416 | 0 | 509 | 202.5 | 228.5 | 0.000 |
| 3 | 512 | 4 | 16 | clustered | q_outer | 64 | 16 | 1103.52 | 32 | 6517 | 226 | 417088 | 0 | 511 | 207.0 | 219.8 | 0.000 |
| 4 | 512 | 4 | 16 | clustered | q_outer | 64 | 16 | 1104.01 | 32 | 6522 | 231 | 417408 | 0 | 511 | 206.0 | 225.6 | 0.000 |
| 0 | 512 | 4 | 16 | clustered | q_outer | 128 | 32 | 4135.75 | 16 | 5272 | 379 | 674816 | 0 | 509 | 324.0 | 352.0 | 0.000 |
| 1 | 512 | 4 | 16 | clustered | q_outer | 128 | 32 | 3953.40 | 16 | 5240 | 360 | 670720 | 0 | 509 | 324.0 | 349.5 | 0.000 |
| 2 | 512 | 4 | 16 | clustered | q_outer | 128 | 32 | 3975.70 | 16 | 5078 | 360 | 649984 | 0 | 509 | 315.5 | 345.5 | 0.000 |
| 3 | 512 | 4 | 16 | clustered | q_outer | 128 | 32 | 3969.93 | 16 | 5219 | 362 | 668032 | 0 | 511 | 331.0 | 349.0 | 0.000 |
| 4 | 512 | 4 | 16 | clustered | q_outer | 128 | 32 | 3957.90 | 16 | 5241 | 360 | 670848 | 0 | 511 | 328.0 | 349.5 | 0.000 |
| 0 | 512 | 4 | 16 | clustered | signature | 16 | 4 | 112.90 | 546 | 7816 | 16 | 125056 | 75 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 4 | 16 | clustered | signature | 16 | 2 | 109.39 | 545 | 7895 | 16 | 126320 | 71 | 509 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 4 | 16 | clustered | signature | 16 | 4 | 118.18 | 550 | 7805 | 16 | 124880 | 81 | 509 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 4 | 16 | clustered | signature | 16 | 2 | 110.88 | 545 | 7888 | 16 | 126208 | 69 | 511 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 4 | 16 | clustered | signature | 16 | 4 | 115.86 | 548 | 7871 | 16 | 125936 | 69 | 511 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 4 | 16 | clustered | signature | 32 | 4 | 115.47 | 586 | 7412 | 16 | 118592 | 150 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 4 | 16 | clustered | signature | 32 | 4 | 116.08 | 601 | 7299 | 16 | 116784 | 183 | 509 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 4 | 16 | clustered | signature | 32 | 4 | 117.23 | 600 | 7380 | 16 | 118080 | 176 | 509 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 4 | 16 | clustered | signature | 32 | 4 | 116.35 | 595 | 7362 | 16 | 117792 | 166 | 511 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 4 | 16 | clustered | signature | 32 | 8 | 156.49 | 610 | 7298 | 16 | 116800 | 179 | 511 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 4 | 16 | clustered | signature | 64 | 4 | 113.73 | 677 | 6471 | 16 | 103536 | 292 | 509 | 9.0 | 16.0 | 1.000 |
| 1 | 512 | 4 | 16 | clustered | signature | 64 | 4 | 113.67 | 680 | 6454 | 16 | 103264 | 314 | 509 | 9.0 | 16.0 | 1.000 |
| 2 | 512 | 4 | 16 | clustered | signature | 64 | 4 | 113.80 | 689 | 6444 | 16 | 103104 | 322 | 509 | 9.0 | 16.0 | 1.000 |
| 3 | 512 | 4 | 16 | clustered | signature | 64 | 4 | 114.20 | 672 | 6517 | 16 | 104272 | 293 | 511 | 10.0 | 16.0 | 1.000 |
| 4 | 512 | 4 | 16 | clustered | signature | 64 | 8 | 152.23 | 676 | 6522 | 16 | 104512 | 291 | 511 | 10.0 | 16.0 | 1.000 |
| 0 | 512 | 4 | 16 | clustered | signature | 128 | 8 | 140.53 | 761 | 5272 | 16 | 84560 | 401 | 509 | 6.0 | 16.0 | 1.000 |
| 1 | 512 | 4 | 16 | clustered | signature | 128 | 8 | 137.89 | 779 | 5240 | 16 | 83952 | 441 | 509 | 6.0 | 14.0 | 1.000 |
| 2 | 512 | 4 | 16 | clustered | signature | 128 | 8 | 141.06 | 778 | 5078 | 16 | 81360 | 443 | 509 | 6.0 | 14.0 | 1.000 |
| 3 | 512 | 4 | 16 | clustered | signature | 128 | 8 | 136.77 | 780 | 5219 | 16 | 83744 | 437 | 511 | 6.0 | 15.0 | 1.000 |
| 4 | 512 | 4 | 16 | clustered | signature | 128 | 8 | 143.40 | 786 | 5241 | 16 | 84064 | 430 | 511 | 5.5 | 16.0 | 1.000 |
| 0 | 512 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 96.97 | 128 | 5108 | 40 | 81728 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 1 | 512 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 96.54 | 128 | 5108 | 40 | 81728 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 2 | 512 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 96.17 | 128 | 5108 | 40 | 81728 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 3 | 512 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 96.00 | 128 | 5108 | 40 | 81728 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 4 | 512 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 95.53 | 128 | 5108 | 40 | 81728 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 0 | 512 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 113.32 | 576 | 4596 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 114.17 | 576 | 4596 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 113.77 | 576 | 4596 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 113.98 | 576 | 4596 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 113.45 | 576 | 4596 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 212.29 | 544 | 4340 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 212.28 | 544 | 4340 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 212.84 | 544 | 4340 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 212.11 | 544 | 4340 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 212.40 | 544 | 4340 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 446.28 | 528 | 4212 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 446.72 | 528 | 4212 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 446.11 | 528 | 4212 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 445.63 | 528 | 4212 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 445.16 | 528 | 4212 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 4 | 16 | common_private | q_outer | 16 | 4 | 98.76 | 128 | 5108 | 40 | 81728 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 1 | 512 | 4 | 16 | common_private | q_outer | 16 | 4 | 99.33 | 128 | 5108 | 40 | 81728 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 2 | 512 | 4 | 16 | common_private | q_outer | 16 | 4 | 99.89 | 128 | 5108 | 40 | 81728 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 3 | 512 | 4 | 16 | common_private | q_outer | 16 | 4 | 99.64 | 128 | 5108 | 40 | 81728 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 4 | 512 | 4 | 16 | common_private | q_outer | 16 | 4 | 98.24 | 128 | 5108 | 40 | 81728 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 0 | 512 | 4 | 16 | common_private | q_outer | 32 | 8 | 235.19 | 64 | 4596 | 72 | 147072 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 1 | 512 | 4 | 16 | common_private | q_outer | 32 | 8 | 238.91 | 64 | 4596 | 72 | 147072 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 2 | 512 | 4 | 16 | common_private | q_outer | 32 | 8 | 239.34 | 64 | 4596 | 72 | 147072 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 3 | 512 | 4 | 16 | common_private | q_outer | 32 | 8 | 242.13 | 64 | 4596 | 72 | 147072 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 4 | 512 | 4 | 16 | common_private | q_outer | 32 | 8 | 239.21 | 64 | 4596 | 72 | 147072 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 0 | 512 | 4 | 16 | common_private | q_outer | 64 | 16 | 674.89 | 32 | 4340 | 136 | 277760 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 1 | 512 | 4 | 16 | common_private | q_outer | 64 | 16 | 678.40 | 32 | 4340 | 136 | 277760 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 2 | 512 | 4 | 16 | common_private | q_outer | 64 | 16 | 684.16 | 32 | 4340 | 136 | 277760 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 3 | 512 | 4 | 16 | common_private | q_outer | 64 | 16 | 681.10 | 32 | 4340 | 136 | 277760 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 4 | 512 | 4 | 16 | common_private | q_outer | 64 | 16 | 682.95 | 32 | 4340 | 136 | 277760 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 0 | 512 | 4 | 16 | common_private | q_outer | 128 | 32 | 2980.18 | 16 | 4212 | 264 | 539136 | 0 | 512 | 263.5 | 264.0 | 0.030 |
| 1 | 512 | 4 | 16 | common_private | q_outer | 128 | 32 | 2986.96 | 16 | 4212 | 264 | 539136 | 0 | 512 | 263.0 | 264.0 | 0.030 |
| 2 | 512 | 4 | 16 | common_private | q_outer | 128 | 32 | 2975.61 | 16 | 4212 | 264 | 539136 | 0 | 512 | 263.5 | 264.0 | 0.030 |
| 3 | 512 | 4 | 16 | common_private | q_outer | 128 | 32 | 3007.01 | 16 | 4212 | 264 | 539136 | 0 | 512 | 264.0 | 264.0 | 0.030 |
| 4 | 512 | 4 | 16 | common_private | q_outer | 128 | 32 | 2984.15 | 16 | 4212 | 264 | 539136 | 0 | 512 | 264.0 | 264.0 | 0.030 |
| 0 | 512 | 4 | 16 | common_private | signature | 16 | 4 | 89.95 | 640 | 5108 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 4 | 16 | common_private | signature | 16 | 4 | 90.04 | 640 | 5108 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 4 | 16 | common_private | signature | 16 | 4 | 90.30 | 640 | 5108 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 4 | 16 | common_private | signature | 16 | 4 | 90.21 | 640 | 5108 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 4 | 16 | common_private | signature | 16 | 4 | 90.38 | 640 | 5108 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 4 | 16 | common_private | signature | 32 | 8 | 113.85 | 576 | 4596 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 4 | 16 | common_private | signature | 32 | 8 | 113.17 | 576 | 4596 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 4 | 16 | common_private | signature | 32 | 8 | 113.70 | 576 | 4596 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 4 | 16 | common_private | signature | 32 | 8 | 113.83 | 576 | 4596 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 4 | 16 | common_private | signature | 32 | 8 | 113.65 | 576 | 4596 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 4 | 16 | common_private | signature | 64 | 16 | 212.08 | 544 | 4340 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 4 | 16 | common_private | signature | 64 | 16 | 212.12 | 544 | 4340 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 4 | 16 | common_private | signature | 64 | 16 | 212.06 | 544 | 4340 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 4 | 16 | common_private | signature | 64 | 16 | 212.35 | 544 | 4340 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 4 | 16 | common_private | signature | 64 | 16 | 212.53 | 544 | 4340 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 4 | 16 | common_private | signature | 128 | 32 | 445.30 | 528 | 4212 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 4 | 16 | common_private | signature | 128 | 32 | 446.35 | 528 | 4212 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 4 | 16 | common_private | signature | 128 | 32 | 446.59 | 528 | 4212 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 4 | 16 | common_private | signature | 128 | 32 | 446.52 | 528 | 4212 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 4 | 16 | common_private | signature | 128 | 32 | 445.62 | 528 | 4212 | 8 | 81728 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 4 | 16 | random | bounded_merge | 16 | 4 | 163.26 | 188 | 7789 | 62 | 124624 | 0 | 512 | 56.5 | 61.0 | 0.125 |
| 1 | 512 | 4 | 16 | random | bounded_merge | 16 | 4 | 164.60 | 195 | 7805 | 62 | 124880 | 0 | 512 | 31.0 | 61.0 | 0.145 |
| 2 | 512 | 4 | 16 | random | bounded_merge | 16 | 4 | 163.38 | 182 | 7788 | 62 | 124608 | 0 | 512 | 58.5 | 61.9 | 0.114 |
| 3 | 512 | 4 | 16 | random | bounded_merge | 16 | 4 | 165.10 | 216 | 7833 | 62 | 125328 | 0 | 512 | 31.0 | 61.0 | 0.185 |
| 4 | 512 | 4 | 16 | random | bounded_merge | 16 | 4 | 164.51 | 201 | 7789 | 62 | 124624 | 0 | 512 | 31.0 | 61.0 | 0.160 |
| 0 | 512 | 4 | 16 | random | bounded_merge | 32 | 4 | 118.52 | 1119 | 7303 | 16 | 116848 | 492 | 512 | 2.0 | 14.0 | 0.894 |
| 1 | 512 | 4 | 16 | random | bounded_merge | 32 | 4 | 116.60 | 1105 | 7343 | 16 | 117488 | 488 | 512 | 2.0 | 14.0 | 0.890 |
| 2 | 512 | 4 | 16 | random | bounded_merge | 32 | 4 | 119.44 | 1117 | 7321 | 16 | 117136 | 489 | 512 | 2.0 | 14.0 | 0.894 |
| 3 | 512 | 4 | 16 | random | bounded_merge | 32 | 4 | 118.22 | 1089 | 7370 | 16 | 117920 | 480 | 512 | 2.0 | 14.0 | 0.892 |
| 4 | 512 | 4 | 16 | random | bounded_merge | 32 | 4 | 116.08 | 1103 | 7334 | 16 | 117344 | 486 | 512 | 2.0 | 14.0 | 0.893 |
| 0 | 512 | 4 | 16 | random | bounded_merge | 64 | 8 | 165.77 | 1790 | 6475 | 16 | 103648 | 511 | 512 | 1.0 | 11.0 | 1.000 |
| 1 | 512 | 4 | 16 | random | bounded_merge | 64 | 8 | 166.41 | 1785 | 6512 | 15 | 104208 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 2 | 512 | 4 | 16 | random | bounded_merge | 64 | 8 | 167.28 | 1748 | 6541 | 15 | 104704 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 3 | 512 | 4 | 16 | random | bounded_merge | 64 | 8 | 171.66 | 1796 | 6520 | 15 | 104336 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 4 | 512 | 4 | 16 | random | bounded_merge | 64 | 4 | 121.71 | 1753 | 6527 | 15 | 104432 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 0 | 512 | 4 | 16 | random | bounded_merge | 128 | 8 | 157.88 | 2557 | 5225 | 12 | 84016 | 512 | 512 | 1.0 | 6.0 | 1.000 |
| 1 | 512 | 4 | 16 | random | bounded_merge | 128 | 8 | 159.19 | 2543 | 5228 | 12 | 83952 | 512 | 512 | 1.0 | 6.0 | 1.000 |
| 2 | 512 | 4 | 16 | random | bounded_merge | 128 | 8 | 160.07 | 2503 | 5270 | 12 | 84624 | 512 | 512 | 1.0 | 6.0 | 1.000 |
| 3 | 512 | 4 | 16 | random | bounded_merge | 128 | 8 | 154.36 | 2557 | 5190 | 11 | 83520 | 512 | 512 | 1.0 | 6.0 | 1.000 |
| 4 | 512 | 4 | 16 | random | bounded_merge | 128 | 8 | 144.34 | 2505 | 5223 | 11 | 83952 | 512 | 512 | 1.0 | 6.0 | 1.000 |
| 0 | 512 | 4 | 16 | random | q_outer | 16 | 4 | 144.67 | 128 | 7789 | 64 | 124624 | 0 | 512 | 61.0 | 63.0 | 0.000 |
| 1 | 512 | 4 | 16 | random | q_outer | 16 | 4 | 151.26 | 128 | 7805 | 64 | 124880 | 0 | 512 | 61.0 | 63.0 | 0.000 |
| 2 | 512 | 4 | 16 | random | q_outer | 16 | 4 | 147.75 | 128 | 7788 | 64 | 124608 | 0 | 512 | 61.0 | 63.0 | 0.000 |
| 3 | 512 | 4 | 16 | random | q_outer | 16 | 4 | 149.15 | 128 | 7833 | 64 | 125328 | 0 | 512 | 61.0 | 63.0 | 0.000 |
| 4 | 512 | 4 | 16 | random | q_outer | 16 | 4 | 148.04 | 128 | 7789 | 64 | 124624 | 0 | 512 | 61.0 | 63.0 | 0.000 |
| 0 | 512 | 4 | 16 | random | q_outer | 32 | 8 | 385.12 | 64 | 7303 | 122 | 233696 | 0 | 512 | 114.0 | 118.0 | 0.000 |
| 1 | 512 | 4 | 16 | random | q_outer | 32 | 8 | 378.36 | 64 | 7343 | 119 | 234976 | 0 | 512 | 115.0 | 118.0 | 0.000 |
| 2 | 512 | 4 | 16 | random | q_outer | 32 | 8 | 394.55 | 64 | 7321 | 122 | 234272 | 0 | 512 | 114.5 | 118.7 | 0.000 |
| 3 | 512 | 4 | 16 | random | q_outer | 32 | 8 | 387.73 | 64 | 7370 | 122 | 235840 | 0 | 512 | 116.0 | 119.7 | 0.000 |
| 4 | 512 | 4 | 16 | random | q_outer | 32 | 8 | 388.02 | 64 | 7334 | 121 | 234688 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 0 | 512 | 4 | 16 | random | q_outer | 64 | 16 | 1027.76 | 32 | 6475 | 219 | 414400 | 0 | 512 | 201.5 | 208.9 | 0.000 |
| 1 | 512 | 4 | 16 | random | q_outer | 64 | 16 | 1035.85 | 32 | 6512 | 212 | 416768 | 0 | 512 | 203.5 | 209.9 | 0.000 |
| 2 | 512 | 4 | 16 | random | q_outer | 64 | 16 | 1056.40 | 32 | 6541 | 217 | 418624 | 0 | 512 | 204.0 | 210.0 | 0.000 |
| 3 | 512 | 4 | 16 | random | q_outer | 64 | 16 | 1048.19 | 32 | 6520 | 213 | 417280 | 0 | 512 | 204.0 | 211.8 | 0.000 |
| 4 | 512 | 4 | 16 | random | q_outer | 64 | 16 | 1040.31 | 32 | 6527 | 213 | 417728 | 0 | 512 | 204.0 | 209.0 | 0.000 |
| 0 | 512 | 4 | 16 | random | q_outer | 128 | 32 | 3723.40 | 16 | 5225 | 339 | 668800 | 0 | 512 | 325.0 | 336.0 | 0.000 |
| 1 | 512 | 4 | 16 | random | q_outer | 128 | 32 | 3846.93 | 16 | 5228 | 345 | 669184 | 0 | 512 | 325.5 | 338.5 | 0.000 |
| 2 | 512 | 4 | 16 | random | q_outer | 128 | 32 | 3805.44 | 16 | 5270 | 343 | 674560 | 0 | 512 | 331.0 | 336.5 | 0.000 |
| 3 | 512 | 4 | 16 | random | q_outer | 128 | 32 | 3666.66 | 16 | 5190 | 335 | 664320 | 0 | 512 | 325.5 | 334.0 | 0.000 |
| 4 | 512 | 4 | 16 | random | q_outer | 128 | 32 | 3679.87 | 16 | 5223 | 333 | 668544 | 0 | 512 | 328.0 | 332.0 | 0.000 |
| 0 | 512 | 4 | 16 | random | signature | 16 | 4 | 119.59 | 809 | 7789 | 16 | 124624 | 406 | 512 | 14.0 | 16.0 | 1.000 |
| 1 | 512 | 4 | 16 | random | signature | 16 | 4 | 117.91 | 795 | 7805 | 16 | 124880 | 396 | 512 | 14.0 | 16.0 | 1.000 |
| 2 | 512 | 4 | 16 | random | signature | 16 | 4 | 121.40 | 827 | 7788 | 16 | 124608 | 417 | 512 | 14.0 | 16.0 | 1.000 |
| 3 | 512 | 4 | 16 | random | signature | 16 | 4 | 117.93 | 777 | 7833 | 16 | 125328 | 380 | 512 | 14.0 | 16.0 | 1.000 |
| 4 | 512 | 4 | 16 | random | signature | 16 | 4 | 118.93 | 814 | 7789 | 16 | 124624 | 395 | 512 | 14.0 | 16.0 | 1.000 |
| 0 | 512 | 4 | 16 | random | signature | 32 | 4 | 126.85 | 1183 | 7303 | 16 | 116848 | 498 | 512 | 2.0 | 14.0 | 1.000 |
| 1 | 512 | 4 | 16 | random | signature | 32 | 4 | 125.56 | 1169 | 7343 | 16 | 117488 | 502 | 512 | 2.0 | 14.0 | 1.000 |
| 2 | 512 | 4 | 16 | random | signature | 32 | 4 | 126.84 | 1181 | 7321 | 16 | 117136 | 499 | 512 | 2.0 | 14.0 | 1.000 |
| 3 | 512 | 4 | 16 | random | signature | 32 | 4 | 126.01 | 1153 | 7370 | 16 | 117920 | 490 | 512 | 2.0 | 14.0 | 1.000 |
| 4 | 512 | 4 | 16 | random | signature | 32 | 4 | 124.88 | 1167 | 7334 | 16 | 117344 | 495 | 512 | 2.0 | 14.0 | 1.000 |
| 0 | 512 | 4 | 16 | random | signature | 64 | 8 | 164.12 | 1790 | 6475 | 16 | 103648 | 511 | 512 | 1.0 | 11.0 | 1.000 |
| 1 | 512 | 4 | 16 | random | signature | 64 | 8 | 166.81 | 1785 | 6512 | 15 | 104208 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 2 | 512 | 4 | 16 | random | signature | 64 | 8 | 167.44 | 1748 | 6541 | 15 | 104704 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 3 | 512 | 4 | 16 | random | signature | 64 | 8 | 165.30 | 1796 | 6520 | 15 | 104336 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 4 | 512 | 4 | 16 | random | signature | 64 | 4 | 122.01 | 1753 | 6527 | 15 | 104432 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 0 | 512 | 4 | 16 | random | signature | 128 | 8 | 152.85 | 2557 | 5225 | 12 | 84016 | 512 | 512 | 1.0 | 6.0 | 1.000 |
| 1 | 512 | 4 | 16 | random | signature | 128 | 8 | 154.31 | 2543 | 5228 | 12 | 83952 | 512 | 512 | 1.0 | 6.0 | 1.000 |
| 2 | 512 | 4 | 16 | random | signature | 128 | 8 | 160.17 | 2503 | 5270 | 12 | 84624 | 512 | 512 | 1.0 | 6.0 | 1.000 |
| 3 | 512 | 4 | 16 | random | signature | 128 | 8 | 149.03 | 2557 | 5190 | 11 | 83520 | 512 | 512 | 1.0 | 6.0 | 1.000 |
| 4 | 512 | 4 | 16 | random | signature | 128 | 8 | 144.54 | 2505 | 5223 | 11 | 83952 | 512 | 512 | 1.0 | 6.0 | 1.000 |
| 0 | 512 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 358.35 | 394 | 14845 | 114 | 237520 | 0 | 509 | 32.0 | 51.7 | 0.747 |
| 1 | 512 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 314.78 | 412 | 15141 | 124 | 242256 | 0 | 509 | 32.0 | 51.0 | 0.774 |
| 2 | 512 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 345.34 | 377 | 14863 | 123 | 237808 | 0 | 509 | 32.0 | 61.0 | 0.680 |
| 3 | 512 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 338.15 | 408 | 15112 | 125 | 241792 | 0 | 511 | 32.0 | 54.0 | 0.765 |
| 4 | 512 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 379.69 | 412 | 15005 | 121 | 240080 | 0 | 511 | 32.0 | 51.0 | 0.788 |
| 0 | 512 | 4 | 32 | clustered | bounded_merge | 32 | 8 | 285.80 | 625 | 13283 | 32 | 212768 | 268 | 509 | 24.0 | 32.0 | 0.936 |
| 1 | 512 | 4 | 32 | clustered | bounded_merge | 32 | 4 | 192.94 | 614 | 13131 | 32 | 210096 | 252 | 509 | 25.0 | 32.0 | 0.945 |
| 2 | 512 | 4 | 32 | clustered | bounded_merge | 32 | 4 | 199.88 | 630 | 13250 | 32 | 212000 | 268 | 509 | 24.5 | 32.0 | 0.935 |
| 3 | 512 | 4 | 32 | clustered | bounded_merge | 32 | 4 | 193.11 | 627 | 13253 | 32 | 212048 | 268 | 511 | 24.0 | 32.0 | 0.925 |
| 4 | 512 | 4 | 32 | clustered | bounded_merge | 32 | 8 | 297.98 | 627 | 13043 | 32 | 209024 | 259 | 511 | 23.0 | 32.0 | 0.937 |
| 0 | 512 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 261.98 | 777 | 10350 | 57 | 166448 | 434 | 509 | 11.0 | 31.0 | 0.951 |
| 1 | 512 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 279.68 | 768 | 10383 | 41 | 166448 | 428 | 509 | 11.0 | 32.0 | 0.964 |
| 2 | 512 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 271.35 | 775 | 10349 | 32 | 165712 | 433 | 509 | 10.0 | 31.6 | 0.964 |
| 3 | 512 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 263.05 | 769 | 10621 | 54 | 170672 | 423 | 511 | 11.0 | 32.0 | 0.965 |
| 4 | 512 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 322.33 | 758 | 10570 | 54 | 170192 | 405 | 511 | 11.0 | 32.0 | 0.961 |
| 0 | 512 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 206.54 | 886 | 7164 | 32 | 121632 | 505 | 509 | 6.0 | 17.5 | 1.000 |
| 1 | 512 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 205.86 | 898 | 7008 | 32 | 118000 | 508 | 509 | 6.0 | 17.0 | 1.000 |
| 2 | 512 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 218.98 | 881 | 6948 | 32 | 117776 | 503 | 509 | 6.0 | 18.0 | 1.000 |
| 3 | 512 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 201.80 | 903 | 7009 | 32 | 117536 | 504 | 511 | 6.0 | 17.0 | 1.000 |
| 4 | 512 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 212.50 | 893 | 7067 | 32 | 118976 | 498 | 511 | 6.0 | 17.0 | 1.000 |
| 0 | 512 | 4 | 32 | clustered | q_outer | 16 | 4 | 280.63 | 128 | 14845 | 128 | 237520 | 0 | 509 | 123.5 | 128.0 | 0.000 |
| 1 | 512 | 4 | 32 | clustered | q_outer | 16 | 4 | 286.73 | 128 | 15141 | 128 | 242256 | 0 | 509 | 124.0 | 128.0 | 0.000 |
| 2 | 512 | 4 | 32 | clustered | q_outer | 16 | 4 | 285.94 | 128 | 14863 | 128 | 237808 | 0 | 509 | 122.0 | 128.0 | 0.000 |
| 3 | 512 | 4 | 32 | clustered | q_outer | 16 | 4 | 284.18 | 128 | 15112 | 128 | 241792 | 0 | 511 | 124.5 | 128.0 | 0.000 |
| 4 | 512 | 4 | 32 | clustered | q_outer | 16 | 4 | 284.81 | 128 | 15005 | 128 | 240080 | 0 | 511 | 125.0 | 128.0 | 0.001 |
| 0 | 512 | 4 | 32 | clustered | q_outer | 32 | 8 | 796.38 | 64 | 13283 | 256 | 425056 | 0 | 509 | 209.0 | 236.4 | 0.000 |
| 1 | 512 | 4 | 32 | clustered | q_outer | 32 | 8 | 800.02 | 64 | 13131 | 255 | 420192 | 0 | 509 | 207.5 | 230.0 | 0.000 |
| 2 | 512 | 4 | 32 | clustered | q_outer | 32 | 8 | 807.51 | 64 | 13250 | 256 | 424000 | 0 | 509 | 206.5 | 235.8 | 0.000 |
| 3 | 512 | 4 | 32 | clustered | q_outer | 32 | 8 | 787.19 | 64 | 13253 | 250 | 424096 | 0 | 511 | 208.0 | 241.4 | 0.000 |
| 4 | 512 | 4 | 32 | clustered | q_outer | 32 | 8 | 801.95 | 64 | 13043 | 256 | 417376 | 0 | 511 | 206.0 | 231.4 | 0.000 |
| 0 | 512 | 4 | 32 | clustered | q_outer | 64 | 16 | 1792.02 | 32 | 10350 | 381 | 662400 | 0 | 509 | 324.5 | 371.8 | 0.000 |
| 1 | 512 | 4 | 32 | clustered | q_outer | 64 | 16 | 1781.07 | 32 | 10383 | 373 | 664512 | 0 | 509 | 328.0 | 353.4 | 0.000 |
| 2 | 512 | 4 | 32 | clustered | q_outer | 64 | 16 | 1896.77 | 32 | 10349 | 400 | 662336 | 0 | 509 | 335.5 | 366.3 | 0.000 |
| 3 | 512 | 4 | 32 | clustered | q_outer | 64 | 16 | 1845.18 | 32 | 10621 | 387 | 679744 | 0 | 511 | 336.5 | 366.9 | 0.000 |
| 4 | 512 | 4 | 32 | clustered | q_outer | 64 | 16 | 1791.92 | 32 | 10570 | 383 | 676480 | 0 | 511 | 335.5 | 368.8 | 0.000 |
| 0 | 512 | 4 | 32 | clustered | q_outer | 128 | 32 | 5460.15 | 16 | 7164 | 501 | 916992 | 0 | 509 | 444.5 | 470.5 | 0.000 |
| 1 | 512 | 4 | 32 | clustered | q_outer | 128 | 32 | 5304.68 | 16 | 7008 | 485 | 897024 | 0 | 509 | 434.0 | 469.5 | 0.000 |
| 2 | 512 | 4 | 32 | clustered | q_outer | 128 | 32 | 5130.23 | 16 | 6948 | 469 | 889344 | 0 | 509 | 437.5 | 468.5 | 0.000 |
| 3 | 512 | 4 | 32 | clustered | q_outer | 128 | 32 | 5294.23 | 16 | 7009 | 484 | 897152 | 0 | 511 | 438.0 | 475.5 | 0.000 |
| 4 | 512 | 4 | 32 | clustered | q_outer | 128 | 32 | 5163.11 | 16 | 7067 | 473 | 904576 | 0 | 511 | 443.5 | 471.0 | 0.000 |
| 0 | 512 | 4 | 32 | clustered | signature | 16 | 4 | 217.02 | 598 | 14845 | 32 | 237520 | 165 | 509 | 32.0 | 32.0 | 1.000 |
| 1 | 512 | 4 | 32 | clustered | signature | 16 | 4 | 218.89 | 590 | 15141 | 32 | 242256 | 153 | 509 | 32.0 | 32.0 | 1.000 |
| 2 | 512 | 4 | 32 | clustered | signature | 16 | 4 | 221.97 | 608 | 14863 | 32 | 237808 | 189 | 509 | 32.0 | 32.0 | 1.000 |
| 3 | 512 | 4 | 32 | clustered | signature | 16 | 4 | 219.93 | 588 | 15112 | 32 | 241792 | 152 | 511 | 32.0 | 32.0 | 1.000 |
| 4 | 512 | 4 | 32 | clustered | signature | 16 | 4 | 226.59 | 594 | 15005 | 32 | 240080 | 150 | 511 | 32.0 | 32.0 | 1.000 |
| 0 | 512 | 4 | 32 | clustered | signature | 32 | 8 | 285.23 | 687 | 13283 | 32 | 212768 | 308 | 509 | 20.0 | 32.0 | 1.000 |
| 1 | 512 | 4 | 32 | clustered | signature | 32 | 4 | 209.22 | 677 | 13131 | 32 | 210096 | 300 | 509 | 20.0 | 32.0 | 1.000 |
| 2 | 512 | 4 | 32 | clustered | signature | 32 | 4 | 211.46 | 690 | 13250 | 32 | 212000 | 311 | 509 | 20.0 | 32.0 | 1.000 |
| 3 | 512 | 4 | 32 | clustered | signature | 32 | 4 | 211.38 | 689 | 13253 | 32 | 212048 | 311 | 511 | 21.0 | 32.0 | 1.000 |
| 4 | 512 | 4 | 32 | clustered | signature | 32 | 8 | 295.16 | 690 | 13043 | 32 | 209024 | 301 | 511 | 19.0 | 32.0 | 1.000 |
| 0 | 512 | 4 | 32 | clustered | signature | 64 | 8 | 249.43 | 807 | 10350 | 32 | 166448 | 444 | 509 | 11.0 | 29.0 | 1.000 |
| 1 | 512 | 4 | 32 | clustered | signature | 64 | 8 | 255.17 | 800 | 10383 | 32 | 166448 | 444 | 509 | 11.0 | 30.0 | 1.000 |
| 2 | 512 | 4 | 32 | clustered | signature | 64 | 8 | 252.14 | 807 | 10349 | 32 | 165712 | 445 | 509 | 10.0 | 30.0 | 1.000 |
| 3 | 512 | 4 | 32 | clustered | signature | 64 | 8 | 253.97 | 801 | 10621 | 32 | 170672 | 436 | 511 | 11.0 | 31.0 | 1.000 |
| 4 | 512 | 4 | 32 | clustered | signature | 64 | 8 | 267.90 | 789 | 10570 | 32 | 170192 | 420 | 511 | 11.0 | 32.0 | 1.000 |
| 0 | 512 | 4 | 32 | clustered | signature | 128 | 8 | 206.75 | 886 | 7164 | 32 | 121632 | 505 | 509 | 6.0 | 17.5 | 1.000 |
| 1 | 512 | 4 | 32 | clustered | signature | 128 | 8 | 206.34 | 898 | 7008 | 32 | 118000 | 508 | 509 | 6.0 | 17.0 | 1.000 |
| 2 | 512 | 4 | 32 | clustered | signature | 128 | 8 | 217.45 | 881 | 6948 | 32 | 117776 | 503 | 509 | 6.0 | 18.0 | 1.000 |
| 3 | 512 | 4 | 32 | clustered | signature | 128 | 8 | 203.32 | 903 | 7009 | 32 | 117536 | 504 | 511 | 6.0 | 17.0 | 1.000 |
| 4 | 512 | 4 | 32 | clustered | signature | 128 | 8 | 213.00 | 893 | 7067 | 32 | 118976 | 498 | 511 | 6.0 | 17.0 | 1.000 |
| 0 | 512 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 179.68 | 128 | 10216 | 80 | 163456 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 1 | 512 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 186.51 | 128 | 10215 | 80 | 163440 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 2 | 512 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 187.28 | 128 | 10215 | 80 | 163440 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 3 | 512 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 182.01 | 128 | 10215 | 80 | 163440 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 4 | 512 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 187.61 | 128 | 10216 | 80 | 163456 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 0 | 512 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 210.32 | 576 | 9192 | 16 | 163456 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 218.15 | 576 | 9191 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 219.06 | 576 | 9191 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 212.69 | 576 | 9191 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 218.39 | 576 | 9192 | 16 | 163456 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 396.51 | 544 | 8680 | 16 | 163456 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 406.89 | 544 | 8679 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 410.17 | 544 | 8679 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 398.16 | 544 | 8679 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 406.81 | 544 | 8680 | 16 | 163456 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 812.75 | 512 | 8168 | 16 | 159360 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 829.28 | 512 | 8168 | 16 | 159360 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 829.35 | 512 | 8168 | 16 | 159360 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 813.26 | 512 | 8168 | 16 | 159360 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 828.92 | 512 | 8168 | 16 | 159360 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 4 | 32 | common_private | q_outer | 16 | 4 | 186.28 | 128 | 10216 | 80 | 163456 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 1 | 512 | 4 | 32 | common_private | q_outer | 16 | 4 | 187.09 | 128 | 10215 | 80 | 163440 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 2 | 512 | 4 | 32 | common_private | q_outer | 16 | 4 | 187.08 | 128 | 10215 | 80 | 163440 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 3 | 512 | 4 | 32 | common_private | q_outer | 16 | 4 | 186.87 | 128 | 10215 | 80 | 163440 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 4 | 512 | 4 | 32 | common_private | q_outer | 16 | 4 | 187.85 | 128 | 10216 | 80 | 163456 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 0 | 512 | 4 | 32 | common_private | q_outer | 32 | 8 | 470.31 | 64 | 9192 | 144 | 294144 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 1 | 512 | 4 | 32 | common_private | q_outer | 32 | 8 | 469.28 | 64 | 9191 | 144 | 294112 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 2 | 512 | 4 | 32 | common_private | q_outer | 32 | 8 | 471.12 | 64 | 9191 | 144 | 294112 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 3 | 512 | 4 | 32 | common_private | q_outer | 32 | 8 | 462.46 | 64 | 9191 | 144 | 294112 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 4 | 512 | 4 | 32 | common_private | q_outer | 32 | 8 | 472.06 | 64 | 9192 | 144 | 294144 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 0 | 512 | 4 | 32 | common_private | q_outer | 64 | 16 | 1338.34 | 32 | 8680 | 272 | 555520 | 0 | 512 | 271.0 | 272.0 | 0.059 |
| 1 | 512 | 4 | 32 | common_private | q_outer | 64 | 16 | 1330.43 | 32 | 8679 | 272 | 555456 | 0 | 512 | 271.0 | 272.0 | 0.059 |
| 2 | 512 | 4 | 32 | common_private | q_outer | 64 | 16 | 1331.34 | 32 | 8679 | 272 | 555456 | 0 | 512 | 271.0 | 272.0 | 0.059 |
| 3 | 512 | 4 | 32 | common_private | q_outer | 64 | 16 | 1335.24 | 32 | 8679 | 272 | 555456 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 4 | 512 | 4 | 32 | common_private | q_outer | 64 | 16 | 1335.62 | 32 | 8680 | 272 | 555520 | 0 | 512 | 271.0 | 272.0 | 0.059 |
| 0 | 512 | 4 | 32 | common_private | q_outer | 128 | 32 | 5882.29 | 16 | 8168 | 512 | 1045504 | 0 | 512 | 510.5 | 512.0 | 0.031 |
| 1 | 512 | 4 | 32 | common_private | q_outer | 128 | 32 | 5894.63 | 16 | 8168 | 512 | 1045504 | 0 | 512 | 510.5 | 512.0 | 0.031 |
| 2 | 512 | 4 | 32 | common_private | q_outer | 128 | 32 | 5892.65 | 16 | 8168 | 512 | 1045504 | 0 | 512 | 510.5 | 512.0 | 0.031 |
| 3 | 512 | 4 | 32 | common_private | q_outer | 128 | 32 | 5885.11 | 16 | 8168 | 512 | 1045504 | 0 | 512 | 510.5 | 512.0 | 0.031 |
| 4 | 512 | 4 | 32 | common_private | q_outer | 128 | 32 | 5895.65 | 16 | 8168 | 512 | 1045504 | 0 | 512 | 510.5 | 512.0 | 0.031 |
| 0 | 512 | 4 | 32 | common_private | signature | 16 | 4 | 157.69 | 640 | 10216 | 16 | 163456 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 4 | 32 | common_private | signature | 16 | 4 | 158.01 | 640 | 10215 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 4 | 32 | common_private | signature | 16 | 4 | 157.97 | 640 | 10215 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 4 | 32 | common_private | signature | 16 | 4 | 158.16 | 640 | 10215 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 4 | 32 | common_private | signature | 16 | 4 | 158.04 | 640 | 10216 | 16 | 163456 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 4 | 32 | common_private | signature | 32 | 8 | 211.23 | 576 | 9192 | 16 | 163456 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 4 | 32 | common_private | signature | 32 | 8 | 211.57 | 576 | 9191 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 4 | 32 | common_private | signature | 32 | 8 | 211.74 | 576 | 9191 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 4 | 32 | common_private | signature | 32 | 8 | 212.43 | 576 | 9191 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 4 | 32 | common_private | signature | 32 | 8 | 211.50 | 576 | 9192 | 16 | 163456 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 4 | 32 | common_private | signature | 64 | 16 | 397.30 | 544 | 8680 | 16 | 163456 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 4 | 32 | common_private | signature | 64 | 16 | 405.22 | 544 | 8679 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 4 | 32 | common_private | signature | 64 | 16 | 396.64 | 544 | 8679 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 4 | 32 | common_private | signature | 64 | 16 | 398.25 | 544 | 8679 | 16 | 163440 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 4 | 32 | common_private | signature | 64 | 16 | 398.94 | 544 | 8680 | 16 | 163456 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 4 | 32 | common_private | signature | 128 | 32 | 812.93 | 512 | 8168 | 16 | 159360 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 4 | 32 | common_private | signature | 128 | 32 | 842.70 | 512 | 8168 | 16 | 159360 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 4 | 32 | common_private | signature | 128 | 32 | 839.55 | 512 | 8168 | 16 | 159360 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 4 | 32 | common_private | signature | 128 | 32 | 812.82 | 512 | 8168 | 16 | 159360 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 4 | 32 | common_private | signature | 128 | 32 | 842.25 | 512 | 8168 | 16 | 159360 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 4 | 32 | random | bounded_merge | 16 | 4 | 267.37 | 128 | 14866 | 122 | 237856 | 0 | 512 | 116.0 | 120.3 | 0.000 |
| 1 | 512 | 4 | 32 | random | bounded_merge | 16 | 4 | 275.17 | 128 | 14804 | 124 | 236864 | 0 | 512 | 116.0 | 119.3 | 0.000 |
| 2 | 512 | 4 | 32 | random | bounded_merge | 16 | 4 | 269.31 | 128 | 14829 | 122 | 237264 | 0 | 512 | 116.0 | 119.0 | 0.000 |
| 3 | 512 | 4 | 32 | random | bounded_merge | 16 | 4 | 269.61 | 128 | 14852 | 123 | 237632 | 0 | 512 | 116.0 | 120.0 | 0.000 |
| 4 | 512 | 4 | 32 | random | bounded_merge | 16 | 4 | 263.70 | 128 | 14917 | 122 | 238672 | 0 | 512 | 117.0 | 120.0 | 0.000 |
| 0 | 512 | 4 | 32 | random | bounded_merge | 32 | 8 | 306.92 | 2151 | 13189 | 28 | 211072 | 512 | 512 | 2.0 | 21.0 | 0.968 |
| 1 | 512 | 4 | 32 | random | bounded_merge | 32 | 8 | 304.70 | 2222 | 13109 | 29 | 209776 | 512 | 512 | 2.0 | 21.0 | 0.980 |
| 2 | 512 | 4 | 32 | random | bounded_merge | 32 | 4 | 210.84 | 2210 | 13154 | 29 | 210464 | 512 | 512 | 2.0 | 21.0 | 0.980 |
| 3 | 512 | 4 | 32 | random | bounded_merge | 32 | 8 | 308.70 | 2221 | 13128 | 27 | 210064 | 512 | 512 | 2.0 | 21.0 | 0.980 |
| 4 | 512 | 4 | 32 | random | bounded_merge | 32 | 4 | 210.12 | 2185 | 13179 | 28 | 210864 | 512 | 512 | 2.0 | 21.0 | 0.977 |
| 0 | 512 | 4 | 32 | random | bounded_merge | 64 | 8 | 269.77 | 3853 | 10486 | 22 | 168240 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 512 | 4 | 32 | random | bounded_merge | 64 | 8 | 271.49 | 3875 | 10479 | 21 | 168112 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 512 | 4 | 32 | random | bounded_merge | 64 | 8 | 272.47 | 3855 | 10521 | 20 | 169104 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 512 | 4 | 32 | random | bounded_merge | 64 | 8 | 279.02 | 3885 | 10483 | 21 | 168464 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 512 | 4 | 32 | random | bounded_merge | 64 | 8 | 266.03 | 3878 | 10505 | 21 | 168848 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 512 | 4 | 32 | random | bounded_merge | 128 | 16 | 448.26 | 5123 | 7088 | 14 | 119680 | 512 | 512 | 1.0 | 2.0 | 1.000 |
| 1 | 512 | 4 | 32 | random | bounded_merge | 128 | 16 | 453.32 | 5064 | 7111 | 11 | 120144 | 512 | 512 | 1.0 | 2.0 | 1.000 |
| 2 | 512 | 4 | 32 | random | bounded_merge | 128 | 8 | 220.17 | 5157 | 7108 | 12 | 119792 | 512 | 512 | 1.0 | 2.0 | 1.000 |
| 3 | 512 | 4 | 32 | random | bounded_merge | 128 | 16 | 448.04 | 5113 | 7110 | 12 | 120400 | 512 | 512 | 1.0 | 2.0 | 1.000 |
| 4 | 512 | 4 | 32 | random | bounded_merge | 128 | 16 | 437.54 | 5139 | 7145 | 11 | 120400 | 512 | 512 | 1.0 | 2.0 | 1.000 |
| 0 | 512 | 4 | 32 | random | q_outer | 16 | 4 | 269.74 | 128 | 14866 | 122 | 237856 | 0 | 512 | 116.0 | 120.3 | 0.000 |
| 1 | 512 | 4 | 32 | random | q_outer | 16 | 4 | 274.93 | 128 | 14804 | 124 | 236864 | 0 | 512 | 116.0 | 119.3 | 0.000 |
| 2 | 512 | 4 | 32 | random | q_outer | 16 | 4 | 269.75 | 128 | 14829 | 122 | 237264 | 0 | 512 | 116.0 | 119.0 | 0.000 |
| 3 | 512 | 4 | 32 | random | q_outer | 16 | 4 | 274.02 | 128 | 14852 | 123 | 237632 | 0 | 512 | 116.0 | 120.0 | 0.000 |
| 4 | 512 | 4 | 32 | random | q_outer | 16 | 4 | 272.89 | 128 | 14917 | 122 | 238672 | 0 | 512 | 117.0 | 120.0 | 0.000 |
| 0 | 512 | 4 | 32 | random | q_outer | 32 | 8 | 685.93 | 64 | 13189 | 219 | 422048 | 0 | 512 | 206.0 | 213.0 | 0.000 |
| 1 | 512 | 4 | 32 | random | q_outer | 32 | 8 | 683.05 | 64 | 13109 | 216 | 419488 | 0 | 512 | 205.0 | 210.0 | 0.000 |
| 2 | 512 | 4 | 32 | random | q_outer | 32 | 8 | 688.23 | 64 | 13154 | 219 | 420928 | 0 | 512 | 206.0 | 210.7 | 0.000 |
| 3 | 512 | 4 | 32 | random | q_outer | 32 | 8 | 673.72 | 64 | 13128 | 215 | 420096 | 0 | 512 | 205.0 | 211.0 | 0.000 |
| 4 | 512 | 4 | 32 | random | q_outer | 32 | 8 | 704.91 | 64 | 13179 | 222 | 421728 | 0 | 512 | 206.0 | 212.0 | 0.000 |
| 0 | 512 | 4 | 32 | random | q_outer | 64 | 16 | 1646.70 | 32 | 10486 | 343 | 671104 | 0 | 512 | 328.5 | 335.9 | 0.000 |
| 1 | 512 | 4 | 32 | random | q_outer | 64 | 16 | 1627.56 | 32 | 10479 | 344 | 670656 | 0 | 512 | 327.0 | 336.7 | 0.000 |
| 2 | 512 | 4 | 32 | random | q_outer | 64 | 16 | 1667.53 | 32 | 10521 | 345 | 673344 | 0 | 512 | 327.5 | 338.9 | 0.000 |
| 3 | 512 | 4 | 32 | random | q_outer | 64 | 16 | 1653.73 | 32 | 10483 | 342 | 670912 | 0 | 512 | 329.0 | 337.8 | 0.000 |
| 4 | 512 | 4 | 32 | random | q_outer | 64 | 16 | 1689.57 | 32 | 10505 | 351 | 672320 | 0 | 512 | 328.0 | 337.9 | 0.000 |
| 0 | 512 | 4 | 32 | random | q_outer | 128 | 32 | 4988.70 | 16 | 7088 | 452 | 907264 | 0 | 512 | 443.0 | 451.0 | 0.000 |
| 1 | 512 | 4 | 32 | random | q_outer | 128 | 32 | 5024.37 | 16 | 7111 | 460 | 910208 | 0 | 512 | 444.5 | 453.0 | 0.000 |
| 2 | 512 | 4 | 32 | random | q_outer | 128 | 32 | 5024.86 | 16 | 7108 | 455 | 909824 | 0 | 512 | 444.5 | 451.5 | 0.000 |
| 3 | 512 | 4 | 32 | random | q_outer | 128 | 32 | 5013.76 | 16 | 7110 | 459 | 910080 | 0 | 512 | 444.5 | 455.0 | 0.000 |
| 4 | 512 | 4 | 32 | random | q_outer | 128 | 32 | 4993.71 | 16 | 7145 | 456 | 914560 | 0 | 512 | 444.5 | 454.0 | 0.000 |
| 0 | 512 | 4 | 32 | random | signature | 16 | 4 | 221.34 | 1216 | 14866 | 31 | 237856 | 512 | 512 | 3.0 | 28.0 | 1.000 |
| 1 | 512 | 4 | 32 | random | signature | 16 | 4 | 226.42 | 1236 | 14804 | 32 | 236864 | 511 | 512 | 3.0 | 28.0 | 1.000 |
| 2 | 512 | 4 | 32 | random | signature | 16 | 4 | 221.51 | 1226 | 14829 | 31 | 237264 | 512 | 512 | 3.0 | 28.0 | 1.000 |
| 3 | 512 | 4 | 32 | random | signature | 16 | 4 | 232.46 | 1218 | 14852 | 32 | 237632 | 510 | 512 | 3.0 | 28.0 | 1.000 |
| 4 | 512 | 4 | 32 | random | signature | 16 | 4 | 235.27 | 1202 | 14917 | 32 | 238672 | 510 | 512 | 3.0 | 28.0 | 1.000 |
| 0 | 512 | 4 | 32 | random | signature | 32 | 8 | 292.67 | 2171 | 13189 | 28 | 211072 | 512 | 512 | 2.0 | 21.0 | 1.000 |
| 1 | 512 | 4 | 32 | random | signature | 32 | 8 | 291.69 | 2234 | 13109 | 28 | 209776 | 512 | 512 | 2.0 | 21.0 | 1.000 |
| 2 | 512 | 4 | 32 | random | signature | 32 | 4 | 218.08 | 2222 | 13154 | 28 | 210464 | 512 | 512 | 2.0 | 21.0 | 1.000 |
| 3 | 512 | 4 | 32 | random | signature | 32 | 8 | 299.50 | 2233 | 13128 | 27 | 210064 | 512 | 512 | 2.0 | 21.0 | 1.000 |
| 4 | 512 | 4 | 32 | random | signature | 32 | 4 | 224.62 | 2199 | 13179 | 28 | 210864 | 512 | 512 | 2.0 | 21.0 | 1.000 |
| 0 | 512 | 4 | 32 | random | signature | 64 | 8 | 275.42 | 3853 | 10486 | 22 | 168240 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 512 | 4 | 32 | random | signature | 64 | 8 | 272.28 | 3875 | 10479 | 21 | 168112 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 512 | 4 | 32 | random | signature | 64 | 8 | 266.88 | 3855 | 10521 | 20 | 169104 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 512 | 4 | 32 | random | signature | 64 | 8 | 281.42 | 3885 | 10483 | 21 | 168464 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 512 | 4 | 32 | random | signature | 64 | 8 | 266.25 | 3878 | 10505 | 21 | 168848 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 512 | 4 | 32 | random | signature | 128 | 16 | 447.16 | 5123 | 7088 | 14 | 119680 | 512 | 512 | 1.0 | 2.0 | 1.000 |
| 1 | 512 | 4 | 32 | random | signature | 128 | 16 | 455.37 | 5064 | 7111 | 11 | 120144 | 512 | 512 | 1.0 | 2.0 | 1.000 |
| 2 | 512 | 4 | 32 | random | signature | 128 | 8 | 223.09 | 5157 | 7108 | 12 | 119792 | 512 | 512 | 1.0 | 2.0 | 1.000 |
| 3 | 512 | 4 | 32 | random | signature | 128 | 16 | 446.18 | 5113 | 7110 | 12 | 120400 | 512 | 512 | 1.0 | 2.0 | 1.000 |
| 4 | 512 | 4 | 32 | random | signature | 128 | 16 | 429.97 | 5139 | 7145 | 11 | 120400 | 512 | 512 | 1.0 | 2.0 | 1.000 |
| 0 | 512 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 67.98 | 506 | 4062 | 13 | 64992 | 0 | 509 | 8.0 | 8.0 | 0.992 |
| 1 | 512 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 60.09 | 507 | 4078 | 15 | 65248 | 0 | 509 | 8.0 | 8.0 | 0.989 |
| 2 | 512 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 72.05 | 505 | 4072 | 15 | 65152 | 0 | 509 | 8.0 | 8.0 | 0.984 |
| 3 | 512 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 67.61 | 505 | 4069 | 14 | 65104 | 0 | 511 | 8.0 | 8.0 | 0.986 |
| 4 | 512 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 63.49 | 508 | 4083 | 15 | 65328 | 0 | 511 | 8.0 | 8.0 | 0.990 |
| 0 | 512 | 8 | 8 | clustered | bounded_merge | 32 | 2 | 72.06 | 494 | 3994 | 15 | 63904 | 1 | 509 | 8.0 | 8.0 | 0.976 |
| 1 | 512 | 8 | 8 | clustered | bounded_merge | 32 | 2 | 70.66 | 498 | 4031 | 15 | 64496 | 0 | 509 | 8.0 | 8.0 | 0.976 |
| 2 | 512 | 8 | 8 | clustered | bounded_merge | 32 | 2 | 72.24 | 488 | 3991 | 15 | 63856 | 0 | 509 | 8.0 | 8.0 | 0.956 |
| 3 | 512 | 8 | 8 | clustered | bounded_merge | 32 | 2 | 73.66 | 494 | 4019 | 15 | 64304 | 0 | 511 | 8.0 | 8.0 | 0.967 |
| 4 | 512 | 8 | 8 | clustered | bounded_merge | 32 | 4 | 92.81 | 498 | 4018 | 15 | 64368 | 3 | 511 | 8.0 | 8.0 | 0.975 |
| 0 | 512 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 81.75 | 518 | 3887 | 8 | 62240 | 50 | 509 | 8.0 | 8.0 | 0.970 |
| 1 | 512 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 81.71 | 515 | 3867 | 8 | 61904 | 51 | 509 | 8.0 | 8.0 | 0.968 |
| 2 | 512 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 81.92 | 517 | 3889 | 8 | 62368 | 55 | 509 | 8.0 | 8.0 | 0.966 |
| 3 | 512 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 82.34 | 520 | 3860 | 8 | 61952 | 52 | 511 | 8.0 | 8.0 | 0.977 |
| 4 | 512 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 82.04 | 524 | 3875 | 8 | 62208 | 61 | 511 | 8.0 | 8.0 | 0.967 |
| 0 | 512 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 86.17 | 585 | 3609 | 8 | 58208 | 164 | 509 | 8.0 | 8.0 | 0.979 |
| 1 | 512 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 85.71 | 570 | 3639 | 8 | 58560 | 149 | 509 | 8.0 | 8.0 | 0.979 |
| 2 | 512 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 86.77 | 574 | 3642 | 8 | 58848 | 163 | 509 | 8.0 | 8.0 | 0.976 |
| 3 | 512 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 85.66 | 576 | 3615 | 8 | 58208 | 153 | 511 | 8.0 | 8.0 | 0.980 |
| 4 | 512 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 86.10 | 577 | 3631 | 8 | 58864 | 158 | 511 | 8.0 | 8.0 | 0.980 |
| 0 | 512 | 8 | 8 | clustered | q_outer | 16 | 2 | 57.45 | 256 | 4062 | 16 | 64992 | 0 | 509 | 16.0 | 16.0 | 0.008 |
| 1 | 512 | 8 | 8 | clustered | q_outer | 16 | 2 | 58.41 | 256 | 4078 | 16 | 65248 | 0 | 509 | 16.0 | 16.0 | 0.004 |
| 2 | 512 | 8 | 8 | clustered | q_outer | 16 | 2 | 57.90 | 256 | 4072 | 16 | 65152 | 0 | 509 | 16.0 | 16.0 | 0.006 |
| 3 | 512 | 8 | 8 | clustered | q_outer | 16 | 2 | 58.39 | 256 | 4069 | 16 | 65104 | 0 | 511 | 16.0 | 16.0 | 0.007 |
| 4 | 512 | 8 | 8 | clustered | q_outer | 16 | 2 | 58.03 | 256 | 4083 | 16 | 65328 | 0 | 511 | 16.0 | 16.0 | 0.003 |
| 0 | 512 | 8 | 8 | clustered | q_outer | 32 | 4 | 107.15 | 128 | 3994 | 32 | 127808 | 0 | 509 | 32.0 | 32.0 | 0.000 |
| 1 | 512 | 8 | 8 | clustered | q_outer | 32 | 4 | 109.45 | 128 | 4031 | 32 | 128992 | 0 | 509 | 32.0 | 32.0 | 0.000 |
| 2 | 512 | 8 | 8 | clustered | q_outer | 32 | 4 | 108.51 | 128 | 3991 | 32 | 127712 | 0 | 509 | 32.0 | 32.0 | 0.000 |
| 3 | 512 | 8 | 8 | clustered | q_outer | 32 | 4 | 108.70 | 128 | 4019 | 32 | 128608 | 0 | 511 | 32.0 | 32.0 | 0.000 |
| 4 | 512 | 8 | 8 | clustered | q_outer | 32 | 4 | 109.06 | 128 | 4018 | 32 | 128576 | 0 | 511 | 32.0 | 32.0 | 0.000 |
| 0 | 512 | 8 | 8 | clustered | q_outer | 64 | 8 | 296.48 | 64 | 3887 | 64 | 248768 | 0 | 509 | 63.0 | 64.0 | 0.000 |
| 1 | 512 | 8 | 8 | clustered | q_outer | 64 | 8 | 302.30 | 64 | 3867 | 64 | 247488 | 0 | 509 | 62.0 | 64.0 | 0.000 |
| 2 | 512 | 8 | 8 | clustered | q_outer | 64 | 8 | 299.25 | 64 | 3889 | 64 | 248896 | 0 | 509 | 62.0 | 64.0 | 0.000 |
| 3 | 512 | 8 | 8 | clustered | q_outer | 64 | 8 | 300.79 | 64 | 3860 | 64 | 247040 | 0 | 511 | 64.0 | 64.0 | 0.000 |
| 4 | 512 | 8 | 8 | clustered | q_outer | 64 | 8 | 301.75 | 64 | 3875 | 64 | 248000 | 0 | 511 | 62.0 | 64.0 | 0.000 |
| 0 | 512 | 8 | 8 | clustered | q_outer | 128 | 16 | 1666.90 | 32 | 3609 | 128 | 461952 | 0 | 509 | 112.5 | 124.0 | 0.000 |
| 1 | 512 | 8 | 8 | clustered | q_outer | 128 | 16 | 1617.66 | 32 | 3639 | 123 | 465792 | 0 | 509 | 115.0 | 121.0 | 0.000 |
| 2 | 512 | 8 | 8 | clustered | q_outer | 128 | 16 | 1700.04 | 32 | 3642 | 128 | 466176 | 0 | 509 | 114.5 | 123.9 | 0.000 |
| 3 | 512 | 8 | 8 | clustered | q_outer | 128 | 16 | 1575.40 | 32 | 3615 | 121 | 462720 | 0 | 511 | 114.5 | 119.0 | 0.000 |
| 4 | 512 | 8 | 8 | clustered | q_outer | 128 | 16 | 1605.51 | 32 | 3631 | 127 | 464768 | 0 | 511 | 114.0 | 120.9 | 0.000 |
| 0 | 512 | 8 | 8 | clustered | signature | 16 | 2 | 62.65 | 518 | 4062 | 8 | 64992 | 12 | 509 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 8 | 8 | clustered | signature | 16 | 2 | 63.86 | 515 | 4078 | 8 | 65248 | 8 | 509 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 8 | 8 | clustered | signature | 16 | 2 | 63.68 | 519 | 4072 | 8 | 65152 | 14 | 509 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 8 | 8 | clustered | signature | 16 | 2 | 63.33 | 519 | 4069 | 8 | 65104 | 14 | 511 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 8 | 8 | clustered | signature | 16 | 2 | 63.59 | 516 | 4083 | 8 | 65328 | 8 | 511 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 8 | 8 | clustered | signature | 32 | 2 | 63.08 | 528 | 3994 | 8 | 63904 | 35 | 509 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 8 | 8 | clustered | signature | 32 | 2 | 63.71 | 522 | 4031 | 8 | 64496 | 24 | 509 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 8 | 8 | clustered | signature | 32 | 2 | 63.79 | 528 | 3991 | 8 | 63856 | 40 | 509 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 8 | 8 | clustered | signature | 32 | 2 | 64.03 | 526 | 4019 | 8 | 64304 | 32 | 511 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 8 | 8 | clustered | signature | 32 | 4 | 81.25 | 526 | 4018 | 8 | 64368 | 29 | 511 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 8 | 8 | clustered | signature | 64 | 4 | 81.00 | 549 | 3887 | 8 | 62240 | 79 | 509 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 8 | 8 | clustered | signature | 64 | 4 | 82.97 | 551 | 3867 | 8 | 61904 | 86 | 509 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 8 | 8 | clustered | signature | 64 | 4 | 85.29 | 549 | 3889 | 8 | 62368 | 85 | 509 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 8 | 8 | clustered | signature | 64 | 4 | 84.08 | 546 | 3860 | 8 | 61952 | 75 | 511 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 8 | 8 | clustered | signature | 64 | 4 | 85.81 | 559 | 3875 | 8 | 62208 | 93 | 511 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 8 | 8 | clustered | signature | 128 | 4 | 83.46 | 605 | 3609 | 8 | 58208 | 181 | 509 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 8 | 8 | clustered | signature | 128 | 4 | 83.96 | 592 | 3639 | 8 | 58560 | 167 | 509 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 8 | 8 | clustered | signature | 128 | 4 | 86.44 | 594 | 3642 | 8 | 58848 | 180 | 509 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 8 | 8 | clustered | signature | 128 | 4 | 84.55 | 596 | 3615 | 8 | 58208 | 172 | 511 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 8 | 8 | clustered | signature | 128 | 4 | 85.56 | 598 | 3631 | 8 | 58864 | 176 | 511 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 44.02 | 256 | 3066 | 12 | 49056 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 1 | 512 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 45.29 | 256 | 3066 | 12 | 49056 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 2 | 512 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 44.24 | 256 | 3066 | 12 | 49056 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 3 | 512 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 43.04 | 256 | 3066 | 12 | 49056 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 4 | 512 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 44.66 | 256 | 3066 | 12 | 49056 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 0 | 512 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 67.82 | 640 | 2554 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 512 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 67.52 | 640 | 2554 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 512 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 67.78 | 640 | 2554 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 512 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 67.68 | 640 | 2554 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 512 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 67.78 | 640 | 2554 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 512 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 110.18 | 576 | 2298 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 512 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 109.50 | 576 | 2298 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 512 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 109.88 | 576 | 2298 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 512 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 109.97 | 576 | 2298 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 512 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 110.03 | 576 | 2298 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 512 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 298.25 | 544 | 2170 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 512 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 296.56 | 544 | 2170 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 512 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 296.64 | 544 | 2170 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 512 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 296.21 | 544 | 2170 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 512 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 296.51 | 544 | 2170 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 512 | 8 | 8 | common_private | q_outer | 16 | 2 | 44.87 | 256 | 3066 | 12 | 49056 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 1 | 512 | 8 | 8 | common_private | q_outer | 16 | 2 | 45.28 | 256 | 3066 | 12 | 49056 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 2 | 512 | 8 | 8 | common_private | q_outer | 16 | 2 | 44.48 | 256 | 3066 | 12 | 49056 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 3 | 512 | 8 | 8 | common_private | q_outer | 16 | 2 | 44.08 | 256 | 3066 | 12 | 49056 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 4 | 512 | 8 | 8 | common_private | q_outer | 16 | 2 | 45.99 | 256 | 3066 | 12 | 49056 | 0 | 512 | 12.0 | 12.0 | 0.334 |
| 0 | 512 | 8 | 8 | common_private | q_outer | 32 | 4 | 72.76 | 128 | 2554 | 20 | 81728 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 1 | 512 | 8 | 8 | common_private | q_outer | 32 | 4 | 73.45 | 128 | 2554 | 20 | 81728 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 2 | 512 | 8 | 8 | common_private | q_outer | 32 | 4 | 73.62 | 128 | 2554 | 20 | 81728 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 3 | 512 | 8 | 8 | common_private | q_outer | 32 | 4 | 73.65 | 128 | 2554 | 20 | 81728 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 4 | 512 | 8 | 8 | common_private | q_outer | 32 | 4 | 74.02 | 128 | 2554 | 20 | 81728 | 0 | 512 | 20.0 | 20.0 | 0.200 |
| 0 | 512 | 8 | 8 | common_private | q_outer | 64 | 8 | 175.97 | 64 | 2298 | 36 | 147072 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 1 | 512 | 8 | 8 | common_private | q_outer | 64 | 8 | 176.89 | 64 | 2298 | 36 | 147072 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 2 | 512 | 8 | 8 | common_private | q_outer | 64 | 8 | 175.62 | 64 | 2298 | 36 | 147072 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 3 | 512 | 8 | 8 | common_private | q_outer | 64 | 8 | 176.01 | 64 | 2298 | 36 | 147072 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 4 | 512 | 8 | 8 | common_private | q_outer | 64 | 8 | 176.08 | 64 | 2298 | 36 | 147072 | 0 | 512 | 36.0 | 36.0 | 0.111 |
| 0 | 512 | 8 | 8 | common_private | q_outer | 128 | 16 | 920.82 | 32 | 2170 | 68 | 277760 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 1 | 512 | 8 | 8 | common_private | q_outer | 128 | 16 | 925.69 | 32 | 2170 | 68 | 277760 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 2 | 512 | 8 | 8 | common_private | q_outer | 128 | 16 | 916.11 | 32 | 2170 | 68 | 277760 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 3 | 512 | 8 | 8 | common_private | q_outer | 128 | 16 | 921.31 | 32 | 2170 | 68 | 277760 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 4 | 512 | 8 | 8 | common_private | q_outer | 128 | 16 | 931.12 | 32 | 2170 | 68 | 277760 | 0 | 512 | 68.0 | 68.0 | 0.059 |
| 0 | 512 | 8 | 8 | common_private | signature | 16 | 2 | 56.52 | 768 | 3066 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 512 | 8 | 8 | common_private | signature | 16 | 2 | 56.40 | 768 | 3066 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 512 | 8 | 8 | common_private | signature | 16 | 2 | 56.53 | 768 | 3066 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 512 | 8 | 8 | common_private | signature | 16 | 2 | 56.14 | 768 | 3066 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 512 | 8 | 8 | common_private | signature | 16 | 2 | 57.48 | 768 | 3066 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 512 | 8 | 8 | common_private | signature | 32 | 4 | 68.02 | 640 | 2554 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 512 | 8 | 8 | common_private | signature | 32 | 4 | 67.57 | 640 | 2554 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 512 | 8 | 8 | common_private | signature | 32 | 4 | 68.26 | 640 | 2554 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 512 | 8 | 8 | common_private | signature | 32 | 4 | 67.70 | 640 | 2554 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 512 | 8 | 8 | common_private | signature | 32 | 4 | 67.73 | 640 | 2554 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 512 | 8 | 8 | common_private | signature | 64 | 8 | 110.83 | 576 | 2298 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 512 | 8 | 8 | common_private | signature | 64 | 8 | 109.60 | 576 | 2298 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 512 | 8 | 8 | common_private | signature | 64 | 8 | 110.88 | 576 | 2298 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 512 | 8 | 8 | common_private | signature | 64 | 8 | 110.31 | 576 | 2298 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 512 | 8 | 8 | common_private | signature | 64 | 8 | 109.97 | 576 | 2298 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 512 | 8 | 8 | common_private | signature | 128 | 16 | 298.96 | 544 | 2170 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 512 | 8 | 8 | common_private | signature | 128 | 16 | 295.91 | 544 | 2170 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 512 | 8 | 8 | common_private | signature | 128 | 16 | 298.40 | 544 | 2170 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 512 | 8 | 8 | common_private | signature | 128 | 16 | 298.84 | 544 | 2170 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 512 | 8 | 8 | common_private | signature | 128 | 16 | 296.57 | 544 | 2170 | 4 | 49056 | 512 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 512 | 8 | 8 | random | bounded_merge | 16 | 2 | 74.73 | 480 | 4049 | 15 | 64784 | 0 | 512 | 8.0 | 8.0 | 0.890 |
| 1 | 512 | 8 | 8 | random | bounded_merge | 16 | 2 | 73.78 | 481 | 4049 | 15 | 64784 | 0 | 512 | 8.0 | 8.0 | 0.894 |
| 2 | 512 | 8 | 8 | random | bounded_merge | 16 | 2 | 74.28 | 479 | 4051 | 15 | 64816 | 0 | 512 | 8.0 | 8.0 | 0.887 |
| 3 | 512 | 8 | 8 | random | bounded_merge | 16 | 2 | 75.13 | 468 | 4035 | 15 | 64560 | 0 | 512 | 8.0 | 8.0 | 0.851 |
| 4 | 512 | 8 | 8 | random | bounded_merge | 16 | 2 | 74.10 | 478 | 4049 | 15 | 64784 | 0 | 512 | 8.0 | 8.0 | 0.883 |
| 0 | 512 | 8 | 8 | random | bounded_merge | 32 | 4 | 103.00 | 452 | 3988 | 15 | 63824 | 25 | 512 | 8.0 | 15.0 | 0.742 |
| 1 | 512 | 8 | 8 | random | bounded_merge | 32 | 4 | 103.80 | 458 | 3989 | 15 | 63856 | 30 | 512 | 8.0 | 14.0 | 0.763 |
| 2 | 512 | 8 | 8 | random | bounded_merge | 32 | 2 | 78.53 | 454 | 3991 | 15 | 63856 | 32 | 512 | 8.0 | 14.0 | 0.744 |
| 3 | 512 | 8 | 8 | random | bounded_merge | 32 | 4 | 104.96 | 447 | 3983 | 15 | 63744 | 28 | 512 | 8.0 | 15.0 | 0.729 |
| 4 | 512 | 8 | 8 | random | bounded_merge | 32 | 4 | 105.09 | 449 | 3985 | 15 | 63808 | 30 | 512 | 8.0 | 15.0 | 0.729 |
| 0 | 512 | 8 | 8 | random | bounded_merge | 64 | 4 | 90.27 | 650 | 3876 | 8 | 62096 | 255 | 512 | 7.0 | 8.0 | 0.904 |
| 1 | 512 | 8 | 8 | random | bounded_merge | 64 | 4 | 89.29 | 653 | 3871 | 8 | 62032 | 260 | 512 | 7.0 | 8.0 | 0.901 |
| 2 | 512 | 8 | 8 | random | bounded_merge | 64 | 4 | 89.42 | 658 | 3869 | 8 | 61984 | 260 | 512 | 7.0 | 8.0 | 0.901 |
| 3 | 512 | 8 | 8 | random | bounded_merge | 64 | 4 | 91.65 | 656 | 3868 | 8 | 61984 | 268 | 512 | 7.0 | 8.0 | 0.902 |
| 4 | 512 | 8 | 8 | random | bounded_merge | 64 | 4 | 88.99 | 658 | 3863 | 8 | 61952 | 272 | 512 | 7.0 | 8.0 | 0.898 |
| 0 | 512 | 8 | 8 | random | bounded_merge | 128 | 4 | 94.26 | 905 | 3611 | 8 | 58352 | 427 | 512 | 5.0 | 7.0 | 0.962 |
| 1 | 512 | 8 | 8 | random | bounded_merge | 128 | 4 | 94.31 | 890 | 3641 | 8 | 58704 | 424 | 512 | 5.0 | 7.0 | 0.961 |
| 2 | 512 | 8 | 8 | random | bounded_merge | 128 | 4 | 95.18 | 886 | 3639 | 8 | 58704 | 430 | 512 | 5.0 | 7.0 | 0.957 |
| 3 | 512 | 8 | 8 | random | bounded_merge | 128 | 4 | 95.75 | 885 | 3649 | 8 | 58752 | 428 | 512 | 5.0 | 7.0 | 0.958 |
| 4 | 512 | 8 | 8 | random | bounded_merge | 128 | 4 | 94.20 | 871 | 3649 | 8 | 58880 | 437 | 512 | 5.0 | 7.0 | 0.954 |
| 0 | 512 | 8 | 8 | random | q_outer | 16 | 2 | 57.36 | 256 | 4049 | 16 | 64784 | 0 | 512 | 16.0 | 16.0 | 0.008 |
| 1 | 512 | 8 | 8 | random | q_outer | 16 | 2 | 57.94 | 256 | 4049 | 16 | 64784 | 0 | 512 | 16.0 | 16.0 | 0.008 |
| 2 | 512 | 8 | 8 | random | q_outer | 16 | 2 | 57.38 | 256 | 4051 | 16 | 64816 | 0 | 512 | 16.0 | 16.0 | 0.009 |
| 3 | 512 | 8 | 8 | random | q_outer | 16 | 2 | 57.65 | 256 | 4035 | 16 | 64560 | 0 | 512 | 16.0 | 16.0 | 0.012 |
| 4 | 512 | 8 | 8 | random | q_outer | 16 | 2 | 57.61 | 256 | 4049 | 16 | 64784 | 0 | 512 | 16.0 | 16.0 | 0.009 |
| 0 | 512 | 8 | 8 | random | q_outer | 32 | 4 | 109.02 | 128 | 3988 | 32 | 127616 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 1 | 512 | 8 | 8 | random | q_outer | 32 | 4 | 109.31 | 128 | 3989 | 32 | 127648 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 2 | 512 | 8 | 8 | random | q_outer | 32 | 4 | 109.48 | 128 | 3991 | 32 | 127712 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 3 | 512 | 8 | 8 | random | q_outer | 32 | 4 | 108.95 | 128 | 3983 | 32 | 127456 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 4 | 512 | 8 | 8 | random | q_outer | 32 | 4 | 109.90 | 128 | 3985 | 32 | 127520 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 0 | 512 | 8 | 8 | random | q_outer | 64 | 8 | 293.34 | 64 | 3876 | 64 | 248064 | 0 | 512 | 61.0 | 63.0 | 0.000 |
| 1 | 512 | 8 | 8 | random | q_outer | 64 | 8 | 297.25 | 64 | 3871 | 64 | 247744 | 0 | 512 | 60.0 | 62.7 | 0.000 |
| 2 | 512 | 8 | 8 | random | q_outer | 64 | 8 | 294.98 | 64 | 3869 | 63 | 247616 | 0 | 512 | 60.5 | 63.0 | 0.000 |
| 3 | 512 | 8 | 8 | random | q_outer | 64 | 8 | 297.53 | 64 | 3868 | 64 | 247552 | 0 | 512 | 61.0 | 62.0 | 0.000 |
| 4 | 512 | 8 | 8 | random | q_outer | 64 | 8 | 294.47 | 64 | 3863 | 64 | 247232 | 0 | 512 | 60.0 | 62.7 | 0.000 |
| 0 | 512 | 8 | 8 | random | q_outer | 128 | 16 | 1586.98 | 32 | 3611 | 122 | 462208 | 0 | 512 | 112.5 | 117.9 | 0.000 |
| 1 | 512 | 8 | 8 | random | q_outer | 128 | 16 | 1564.02 | 32 | 3641 | 120 | 466048 | 0 | 512 | 114.0 | 116.9 | 0.000 |
| 2 | 512 | 8 | 8 | random | q_outer | 128 | 16 | 1592.82 | 32 | 3639 | 122 | 465792 | 0 | 512 | 113.5 | 117.9 | 0.000 |
| 3 | 512 | 8 | 8 | random | q_outer | 128 | 16 | 1571.40 | 32 | 3649 | 120 | 467072 | 0 | 512 | 114.0 | 117.0 | 0.000 |
| 4 | 512 | 8 | 8 | random | q_outer | 128 | 16 | 1555.71 | 32 | 3649 | 119 | 467072 | 0 | 512 | 114.0 | 117.0 | 0.000 |
| 0 | 512 | 8 | 8 | random | signature | 16 | 2 | 63.63 | 544 | 4049 | 8 | 64784 | 64 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 8 | 8 | random | signature | 16 | 2 | 63.76 | 543 | 4049 | 8 | 64784 | 62 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 8 | 8 | random | signature | 16 | 2 | 65.20 | 545 | 4051 | 8 | 64816 | 66 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 8 | 8 | random | signature | 16 | 2 | 64.85 | 556 | 4035 | 8 | 64560 | 88 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 8 | 8 | random | signature | 16 | 2 | 65.29 | 546 | 4049 | 8 | 64784 | 68 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 8 | 8 | random | signature | 32 | 4 | 86.27 | 602 | 3988 | 8 | 63824 | 163 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 8 | 8 | random | signature | 32 | 4 | 84.19 | 597 | 3989 | 8 | 63856 | 152 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 8 | 8 | random | signature | 32 | 2 | 67.28 | 605 | 3991 | 8 | 63856 | 158 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 8 | 8 | random | signature | 32 | 4 | 86.00 | 606 | 3983 | 8 | 63744 | 169 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 8 | 8 | random | signature | 32 | 4 | 84.49 | 607 | 3985 | 8 | 63808 | 171 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 8 | 8 | random | signature | 64 | 4 | 90.38 | 708 | 3876 | 8 | 62096 | 291 | 512 | 7.0 | 8.0 | 1.000 |
| 1 | 512 | 8 | 8 | random | signature | 64 | 4 | 88.83 | 712 | 3871 | 8 | 62032 | 300 | 512 | 7.0 | 8.0 | 1.000 |
| 2 | 512 | 8 | 8 | random | signature | 64 | 4 | 91.13 | 718 | 3869 | 8 | 61984 | 294 | 512 | 7.0 | 8.0 | 1.000 |
| 3 | 512 | 8 | 8 | random | signature | 64 | 4 | 91.34 | 716 | 3868 | 8 | 61984 | 302 | 512 | 7.0 | 8.0 | 1.000 |
| 4 | 512 | 8 | 8 | random | signature | 64 | 4 | 89.15 | 719 | 3863 | 8 | 61952 | 310 | 512 | 7.0 | 8.0 | 1.000 |
| 0 | 512 | 8 | 8 | random | signature | 128 | 4 | 92.31 | 929 | 3611 | 8 | 58352 | 433 | 512 | 5.0 | 7.0 | 1.000 |
| 1 | 512 | 8 | 8 | random | signature | 128 | 4 | 92.25 | 914 | 3641 | 8 | 58704 | 433 | 512 | 5.0 | 7.0 | 1.000 |
| 2 | 512 | 8 | 8 | random | signature | 128 | 4 | 94.06 | 913 | 3639 | 8 | 58704 | 437 | 512 | 5.0 | 7.0 | 1.000 |
| 3 | 512 | 8 | 8 | random | signature | 128 | 4 | 93.49 | 910 | 3649 | 8 | 58752 | 438 | 512 | 5.0 | 7.0 | 1.000 |
| 4 | 512 | 8 | 8 | random | signature | 128 | 4 | 92.15 | 901 | 3649 | 8 | 58880 | 447 | 512 | 5.0 | 7.0 | 1.000 |
| 0 | 512 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 140.11 | 496 | 8062 | 31 | 128992 | 0 | 509 | 16.0 | 16.0 | 0.968 |
| 1 | 512 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 138.90 | 498 | 8087 | 31 | 129392 | 0 | 509 | 16.0 | 16.0 | 0.970 |
| 2 | 512 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 142.23 | 498 | 8078 | 31 | 129248 | 0 | 509 | 16.0 | 16.0 | 0.973 |
| 3 | 512 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 135.13 | 501 | 8091 | 29 | 129456 | 0 | 511 | 16.0 | 16.0 | 0.981 |
| 4 | 512 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 130.70 | 500 | 8102 | 28 | 129632 | 0 | 511 | 16.0 | 16.0 | 0.975 |
| 0 | 512 | 8 | 16 | clustered | bounded_merge | 32 | 4 | 192.42 | 476 | 7816 | 31 | 125296 | 7 | 509 | 16.0 | 16.0 | 0.933 |
| 1 | 512 | 8 | 16 | clustered | bounded_merge | 32 | 2 | 144.93 | 474 | 7895 | 31 | 126320 | 1 | 509 | 16.0 | 16.0 | 0.919 |
| 2 | 512 | 8 | 16 | clustered | bounded_merge | 32 | 4 | 188.97 | 471 | 7805 | 31 | 124896 | 4 | 509 | 16.0 | 16.0 | 0.922 |
| 3 | 512 | 8 | 16 | clustered | bounded_merge | 32 | 2 | 146.02 | 477 | 7888 | 31 | 126208 | 1 | 511 | 16.0 | 16.0 | 0.933 |
| 4 | 512 | 8 | 16 | clustered | bounded_merge | 32 | 4 | 183.80 | 482 | 7871 | 30 | 126208 | 9 | 511 | 16.0 | 16.0 | 0.933 |
| 0 | 512 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 150.41 | 541 | 7412 | 16 | 119472 | 111 | 509 | 16.0 | 16.0 | 0.958 |
| 1 | 512 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 150.83 | 543 | 7299 | 16 | 117392 | 131 | 509 | 16.0 | 16.0 | 0.941 |
| 2 | 512 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 151.70 | 548 | 7380 | 16 | 118768 | 128 | 509 | 16.0 | 16.0 | 0.951 |
| 3 | 512 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 152.09 | 546 | 7362 | 16 | 118672 | 123 | 511 | 16.0 | 16.0 | 0.949 |
| 4 | 512 | 8 | 16 | clustered | bounded_merge | 64 | 8 | 305.50 | 553 | 7298 | 16 | 117984 | 131 | 511 | 16.0 | 16.0 | 0.939 |
| 0 | 512 | 8 | 16 | clustered | bounded_merge | 128 | 4 | 153.91 | 653 | 6471 | 16 | 107024 | 283 | 509 | 10.0 | 16.0 | 0.983 |
| 1 | 512 | 8 | 16 | clustered | bounded_merge | 128 | 4 | 168.71 | 650 | 6454 | 27 | 106192 | 297 | 509 | 10.0 | 16.0 | 0.958 |
| 2 | 512 | 8 | 16 | clustered | bounded_merge | 128 | 4 | 153.58 | 663 | 6444 | 16 | 106176 | 304 | 509 | 10.0 | 16.0 | 0.978 |
| 3 | 512 | 8 | 16 | clustered | bounded_merge | 128 | 4 | 153.73 | 644 | 6517 | 27 | 107200 | 273 | 511 | 11.0 | 16.0 | 0.970 |
| 4 | 512 | 8 | 16 | clustered | bounded_merge | 128 | 8 | 284.78 | 650 | 6522 | 27 | 108320 | 274 | 511 | 11.0 | 16.0 | 0.970 |
| 0 | 512 | 8 | 16 | clustered | q_outer | 16 | 2 | 106.01 | 256 | 8062 | 32 | 128992 | 0 | 509 | 32.0 | 32.0 | 0.016 |
| 1 | 512 | 8 | 16 | clustered | q_outer | 16 | 2 | 105.24 | 256 | 8087 | 32 | 129392 | 0 | 509 | 32.0 | 32.0 | 0.013 |
| 2 | 512 | 8 | 16 | clustered | q_outer | 16 | 2 | 106.55 | 256 | 8078 | 32 | 129248 | 0 | 509 | 32.0 | 32.0 | 0.014 |
| 3 | 512 | 8 | 16 | clustered | q_outer | 16 | 2 | 105.96 | 256 | 8091 | 32 | 129456 | 0 | 511 | 32.0 | 32.0 | 0.012 |
| 4 | 512 | 8 | 16 | clustered | q_outer | 16 | 2 | 106.23 | 256 | 8102 | 32 | 129632 | 0 | 511 | 32.0 | 32.0 | 0.011 |
| 0 | 512 | 8 | 16 | clustered | q_outer | 32 | 4 | 207.23 | 128 | 7816 | 64 | 250112 | 0 | 509 | 64.0 | 64.0 | 0.000 |
| 1 | 512 | 8 | 16 | clustered | q_outer | 32 | 4 | 207.92 | 128 | 7895 | 64 | 252640 | 0 | 509 | 64.0 | 64.0 | 0.000 |
| 2 | 512 | 8 | 16 | clustered | q_outer | 32 | 4 | 206.68 | 128 | 7805 | 64 | 249760 | 0 | 509 | 64.0 | 64.0 | 0.000 |
| 3 | 512 | 8 | 16 | clustered | q_outer | 32 | 4 | 207.04 | 128 | 7888 | 64 | 252416 | 0 | 511 | 64.0 | 64.0 | 0.000 |
| 4 | 512 | 8 | 16 | clustered | q_outer | 32 | 4 | 208.56 | 128 | 7871 | 64 | 251872 | 0 | 511 | 64.0 | 64.0 | 0.000 |
| 0 | 512 | 8 | 16 | clustered | q_outer | 64 | 8 | 584.25 | 64 | 7412 | 128 | 474368 | 0 | 509 | 117.0 | 128.0 | 0.000 |
| 1 | 512 | 8 | 16 | clustered | q_outer | 64 | 8 | 584.51 | 64 | 7299 | 128 | 467136 | 0 | 509 | 115.0 | 125.7 | 0.000 |
| 2 | 512 | 8 | 16 | clustered | q_outer | 64 | 8 | 586.51 | 64 | 7380 | 128 | 472320 | 0 | 509 | 116.0 | 127.7 | 0.000 |
| 3 | 512 | 8 | 16 | clustered | q_outer | 64 | 8 | 585.26 | 64 | 7362 | 128 | 471168 | 0 | 511 | 117.5 | 128.0 | 0.000 |
| 4 | 512 | 8 | 16 | clustered | q_outer | 64 | 8 | 584.76 | 64 | 7298 | 128 | 467072 | 0 | 511 | 115.0 | 127.7 | 0.000 |
| 0 | 512 | 8 | 16 | clustered | q_outer | 128 | 16 | 3093.79 | 32 | 6471 | 242 | 828288 | 0 | 509 | 199.0 | 231.8 | 0.000 |
| 1 | 512 | 8 | 16 | clustered | q_outer | 128 | 16 | 2860.77 | 32 | 6454 | 225 | 826112 | 0 | 509 | 201.0 | 219.8 | 0.000 |
| 2 | 512 | 8 | 16 | clustered | q_outer | 128 | 16 | 3015.76 | 32 | 6444 | 235 | 824832 | 0 | 509 | 202.5 | 228.5 | 0.000 |
| 3 | 512 | 8 | 16 | clustered | q_outer | 128 | 16 | 2890.11 | 32 | 6517 | 226 | 834176 | 0 | 511 | 207.0 | 219.8 | 0.000 |
| 4 | 512 | 8 | 16 | clustered | q_outer | 128 | 16 | 2960.39 | 32 | 6522 | 231 | 834816 | 0 | 511 | 206.0 | 225.6 | 0.000 |
| 0 | 512 | 8 | 16 | clustered | signature | 16 | 2 | 110.14 | 526 | 8062 | 16 | 128992 | 30 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 8 | 16 | clustered | signature | 16 | 2 | 109.78 | 524 | 8087 | 16 | 129392 | 26 | 509 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 8 | 16 | clustered | signature | 16 | 2 | 109.51 | 526 | 8078 | 16 | 129248 | 28 | 509 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 8 | 16 | clustered | signature | 16 | 2 | 110.18 | 523 | 8091 | 16 | 129456 | 22 | 511 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 8 | 16 | clustered | signature | 16 | 2 | 110.20 | 524 | 8102 | 16 | 129632 | 24 | 511 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 8 | 16 | clustered | signature | 32 | 4 | 141.33 | 546 | 7816 | 16 | 125296 | 75 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 8 | 16 | clustered | signature | 32 | 2 | 114.08 | 545 | 7895 | 16 | 126320 | 71 | 509 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 8 | 16 | clustered | signature | 32 | 4 | 148.62 | 550 | 7805 | 16 | 124896 | 81 | 509 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 8 | 16 | clustered | signature | 32 | 2 | 112.79 | 545 | 7888 | 16 | 126208 | 69 | 511 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 8 | 16 | clustered | signature | 32 | 4 | 145.17 | 548 | 7871 | 16 | 126208 | 69 | 511 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 8 | 16 | clustered | signature | 64 | 4 | 146.18 | 586 | 7412 | 16 | 119472 | 150 | 509 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 8 | 16 | clustered | signature | 64 | 4 | 146.88 | 601 | 7299 | 16 | 117392 | 183 | 509 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 8 | 16 | clustered | signature | 64 | 4 | 150.26 | 600 | 7380 | 16 | 118768 | 176 | 509 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 8 | 16 | clustered | signature | 64 | 4 | 147.63 | 595 | 7362 | 16 | 118672 | 166 | 511 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 8 | 16 | clustered | signature | 64 | 8 | 280.54 | 610 | 7298 | 16 | 117984 | 179 | 511 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 8 | 16 | clustered | signature | 128 | 4 | 148.64 | 677 | 6471 | 16 | 107024 | 292 | 509 | 9.0 | 16.0 | 1.000 |
| 1 | 512 | 8 | 16 | clustered | signature | 128 | 4 | 144.89 | 680 | 6454 | 16 | 106192 | 314 | 509 | 9.0 | 16.0 | 1.000 |
| 2 | 512 | 8 | 16 | clustered | signature | 128 | 4 | 150.48 | 689 | 6444 | 16 | 106176 | 322 | 509 | 9.0 | 16.0 | 1.000 |
| 3 | 512 | 8 | 16 | clustered | signature | 128 | 4 | 144.62 | 672 | 6517 | 16 | 107200 | 293 | 511 | 10.0 | 16.0 | 1.000 |
| 4 | 512 | 8 | 16 | clustered | signature | 128 | 8 | 262.44 | 676 | 6522 | 16 | 108320 | 291 | 511 | 10.0 | 16.0 | 1.000 |
| 0 | 512 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 75.74 | 256 | 6132 | 24 | 98112 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 1 | 512 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 76.80 | 256 | 6132 | 24 | 98112 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 2 | 512 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 76.42 | 256 | 6132 | 24 | 98112 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 3 | 512 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 76.60 | 256 | 6132 | 24 | 98112 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 4 | 512 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 76.59 | 256 | 6132 | 24 | 98112 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 0 | 512 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 113.46 | 640 | 5108 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 113.51 | 640 | 5108 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 113.72 | 640 | 5108 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 113.91 | 640 | 5108 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 113.70 | 640 | 5108 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 191.90 | 576 | 4596 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 192.21 | 576 | 4596 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 192.01 | 576 | 4596 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 192.39 | 576 | 4596 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 191.85 | 576 | 4596 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 523.64 | 544 | 4340 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 523.59 | 544 | 4340 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 523.98 | 544 | 4340 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 524.18 | 544 | 4340 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 523.59 | 544 | 4340 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 8 | 16 | common_private | q_outer | 16 | 2 | 77.81 | 256 | 6132 | 24 | 98112 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 1 | 512 | 8 | 16 | common_private | q_outer | 16 | 2 | 78.64 | 256 | 6132 | 24 | 98112 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 2 | 512 | 8 | 16 | common_private | q_outer | 16 | 2 | 79.05 | 256 | 6132 | 24 | 98112 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 3 | 512 | 8 | 16 | common_private | q_outer | 16 | 2 | 78.67 | 256 | 6132 | 24 | 98112 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 4 | 512 | 8 | 16 | common_private | q_outer | 16 | 2 | 79.12 | 256 | 6132 | 24 | 98112 | 0 | 512 | 24.0 | 24.0 | 0.334 |
| 0 | 512 | 8 | 16 | common_private | q_outer | 32 | 4 | 133.83 | 128 | 5108 | 40 | 163456 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 1 | 512 | 8 | 16 | common_private | q_outer | 32 | 4 | 134.34 | 128 | 5108 | 40 | 163456 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 2 | 512 | 8 | 16 | common_private | q_outer | 32 | 4 | 134.88 | 128 | 5108 | 40 | 163456 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 3 | 512 | 8 | 16 | common_private | q_outer | 32 | 4 | 134.81 | 128 | 5108 | 40 | 163456 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 4 | 512 | 8 | 16 | common_private | q_outer | 32 | 4 | 132.87 | 128 | 5108 | 40 | 163456 | 0 | 512 | 40.0 | 40.0 | 0.200 |
| 0 | 512 | 8 | 16 | common_private | q_outer | 64 | 8 | 334.39 | 64 | 4596 | 72 | 294144 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 1 | 512 | 8 | 16 | common_private | q_outer | 64 | 8 | 335.56 | 64 | 4596 | 72 | 294144 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 2 | 512 | 8 | 16 | common_private | q_outer | 64 | 8 | 332.98 | 64 | 4596 | 72 | 294144 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 3 | 512 | 8 | 16 | common_private | q_outer | 64 | 8 | 338.72 | 64 | 4596 | 72 | 294144 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 4 | 512 | 8 | 16 | common_private | q_outer | 64 | 8 | 336.08 | 64 | 4596 | 72 | 294144 | 0 | 512 | 72.0 | 72.0 | 0.111 |
| 0 | 512 | 8 | 16 | common_private | q_outer | 128 | 16 | 1800.21 | 32 | 4340 | 136 | 555520 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 1 | 512 | 8 | 16 | common_private | q_outer | 128 | 16 | 1801.34 | 32 | 4340 | 136 | 555520 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 2 | 512 | 8 | 16 | common_private | q_outer | 128 | 16 | 1797.19 | 32 | 4340 | 136 | 555520 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 3 | 512 | 8 | 16 | common_private | q_outer | 128 | 16 | 1806.10 | 32 | 4340 | 136 | 555520 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 4 | 512 | 8 | 16 | common_private | q_outer | 128 | 16 | 1790.08 | 32 | 4340 | 136 | 555520 | 0 | 512 | 136.0 | 136.0 | 0.059 |
| 0 | 512 | 8 | 16 | common_private | signature | 16 | 2 | 90.93 | 768 | 6132 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 8 | 16 | common_private | signature | 16 | 2 | 92.39 | 768 | 6132 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 8 | 16 | common_private | signature | 16 | 2 | 91.26 | 768 | 6132 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 8 | 16 | common_private | signature | 16 | 2 | 91.21 | 768 | 6132 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 8 | 16 | common_private | signature | 16 | 2 | 92.08 | 768 | 6132 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 8 | 16 | common_private | signature | 32 | 4 | 113.32 | 640 | 5108 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 8 | 16 | common_private | signature | 32 | 4 | 115.54 | 640 | 5108 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 8 | 16 | common_private | signature | 32 | 4 | 113.79 | 640 | 5108 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 8 | 16 | common_private | signature | 32 | 4 | 113.61 | 640 | 5108 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 8 | 16 | common_private | signature | 32 | 4 | 114.87 | 640 | 5108 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 8 | 16 | common_private | signature | 64 | 8 | 191.92 | 576 | 4596 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 8 | 16 | common_private | signature | 64 | 8 | 194.03 | 576 | 4596 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 8 | 16 | common_private | signature | 64 | 8 | 192.07 | 576 | 4596 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 8 | 16 | common_private | signature | 64 | 8 | 192.13 | 576 | 4596 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 8 | 16 | common_private | signature | 64 | 8 | 194.43 | 576 | 4596 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 8 | 16 | common_private | signature | 128 | 16 | 524.02 | 544 | 4340 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 512 | 8 | 16 | common_private | signature | 128 | 16 | 523.52 | 544 | 4340 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 512 | 8 | 16 | common_private | signature | 128 | 16 | 523.69 | 544 | 4340 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 512 | 8 | 16 | common_private | signature | 128 | 16 | 524.15 | 544 | 4340 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 512 | 8 | 16 | common_private | signature | 128 | 16 | 523.53 | 544 | 4340 | 8 | 98112 | 512 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 512 | 8 | 16 | random | bounded_merge | 16 | 2 | 144.12 | 410 | 8044 | 31 | 128704 | 0 | 512 | 16.0 | 31.0 | 0.626 |
| 1 | 512 | 8 | 16 | random | bounded_merge | 16 | 2 | 139.48 | 421 | 8050 | 31 | 128800 | 0 | 512 | 16.0 | 31.0 | 0.667 |
| 2 | 512 | 8 | 16 | random | bounded_merge | 16 | 2 | 140.28 | 415 | 8057 | 31 | 128912 | 0 | 512 | 16.0 | 31.0 | 0.644 |
| 3 | 512 | 8 | 16 | random | bounded_merge | 16 | 2 | 144.61 | 416 | 8048 | 31 | 128768 | 0 | 512 | 16.0 | 31.0 | 0.649 |
| 4 | 512 | 8 | 16 | random | bounded_merge | 16 | 2 | 140.68 | 405 | 8035 | 31 | 128560 | 0 | 512 | 16.0 | 31.0 | 0.607 |
| 0 | 512 | 8 | 16 | random | bounded_merge | 32 | 4 | 202.47 | 518 | 7789 | 31 | 124720 | 205 | 512 | 16.0 | 30.0 | 0.467 |
| 1 | 512 | 8 | 16 | random | bounded_merge | 32 | 4 | 194.41 | 502 | 7805 | 31 | 124960 | 204 | 512 | 16.0 | 30.0 | 0.465 |
| 2 | 512 | 8 | 16 | random | bounded_merge | 32 | 4 | 195.86 | 541 | 7788 | 31 | 124672 | 236 | 512 | 15.0 | 30.0 | 0.478 |
| 3 | 512 | 8 | 16 | random | bounded_merge | 32 | 4 | 203.42 | 490 | 7833 | 31 | 125488 | 178 | 512 | 16.0 | 30.0 | 0.475 |
| 4 | 512 | 8 | 16 | random | bounded_merge | 32 | 4 | 196.61 | 544 | 7789 | 31 | 124768 | 228 | 512 | 15.0 | 30.0 | 0.508 |
| 0 | 512 | 8 | 16 | random | bounded_merge | 64 | 4 | 171.42 | 1119 | 7303 | 16 | 117728 | 492 | 512 | 2.0 | 14.0 | 0.892 |
| 1 | 512 | 8 | 16 | random | bounded_merge | 64 | 4 | 164.42 | 1105 | 7343 | 16 | 118240 | 489 | 512 | 2.0 | 14.0 | 0.890 |
| 2 | 512 | 8 | 16 | random | bounded_merge | 64 | 4 | 166.68 | 1117 | 7321 | 16 | 117952 | 489 | 512 | 2.0 | 14.0 | 0.893 |
| 3 | 512 | 8 | 16 | random | bounded_merge | 64 | 4 | 171.11 | 1089 | 7370 | 16 | 118464 | 481 | 512 | 2.0 | 14.0 | 0.891 |
| 4 | 512 | 8 | 16 | random | bounded_merge | 64 | 4 | 163.14 | 1103 | 7334 | 16 | 118144 | 488 | 512 | 2.0 | 14.0 | 0.894 |
| 0 | 512 | 8 | 16 | random | bounded_merge | 128 | 8 | 303.92 | 1790 | 6475 | 16 | 107024 | 511 | 512 | 1.0 | 11.0 | 1.000 |
| 1 | 512 | 8 | 16 | random | bounded_merge | 128 | 8 | 303.99 | 1785 | 6512 | 15 | 107376 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 2 | 512 | 8 | 16 | random | bounded_merge | 128 | 8 | 298.34 | 1748 | 6541 | 15 | 107936 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 3 | 512 | 8 | 16 | random | bounded_merge | 128 | 8 | 303.28 | 1796 | 6520 | 15 | 107152 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 4 | 512 | 8 | 16 | random | bounded_merge | 128 | 4 | 162.76 | 1753 | 6527 | 15 | 108128 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 0 | 512 | 8 | 16 | random | q_outer | 16 | 2 | 101.09 | 256 | 8044 | 32 | 128704 | 0 | 512 | 32.0 | 32.0 | 0.015 |
| 1 | 512 | 8 | 16 | random | q_outer | 16 | 2 | 103.74 | 256 | 8050 | 32 | 128800 | 0 | 512 | 32.0 | 32.0 | 0.013 |
| 2 | 512 | 8 | 16 | random | q_outer | 16 | 2 | 102.74 | 256 | 8057 | 32 | 128912 | 0 | 512 | 32.0 | 32.0 | 0.014 |
| 3 | 512 | 8 | 16 | random | q_outer | 16 | 2 | 103.10 | 256 | 8048 | 32 | 128768 | 0 | 512 | 32.0 | 32.0 | 0.015 |
| 4 | 512 | 8 | 16 | random | q_outer | 16 | 2 | 102.03 | 256 | 8035 | 32 | 128560 | 0 | 512 | 32.0 | 32.0 | 0.016 |
| 0 | 512 | 8 | 16 | random | q_outer | 32 | 4 | 199.17 | 128 | 7789 | 64 | 249248 | 0 | 512 | 61.0 | 63.0 | 0.000 |
| 1 | 512 | 8 | 16 | random | q_outer | 32 | 4 | 206.62 | 128 | 7805 | 64 | 249760 | 0 | 512 | 61.0 | 63.0 | 0.000 |
| 2 | 512 | 8 | 16 | random | q_outer | 32 | 4 | 202.24 | 128 | 7788 | 64 | 249216 | 0 | 512 | 61.0 | 63.0 | 0.000 |
| 3 | 512 | 8 | 16 | random | q_outer | 32 | 4 | 205.51 | 128 | 7833 | 64 | 250656 | 0 | 512 | 61.0 | 63.0 | 0.000 |
| 4 | 512 | 8 | 16 | random | q_outer | 32 | 4 | 203.48 | 128 | 7789 | 64 | 249248 | 0 | 512 | 61.0 | 63.0 | 0.000 |
| 0 | 512 | 8 | 16 | random | q_outer | 64 | 8 | 545.53 | 64 | 7303 | 122 | 467392 | 0 | 512 | 114.0 | 118.0 | 0.000 |
| 1 | 512 | 8 | 16 | random | q_outer | 64 | 8 | 533.88 | 64 | 7343 | 119 | 469952 | 0 | 512 | 115.0 | 118.0 | 0.000 |
| 2 | 512 | 8 | 16 | random | q_outer | 64 | 8 | 551.91 | 64 | 7321 | 122 | 468544 | 0 | 512 | 114.5 | 118.7 | 0.000 |
| 3 | 512 | 8 | 16 | random | q_outer | 64 | 8 | 553.05 | 64 | 7370 | 122 | 471680 | 0 | 512 | 116.0 | 119.7 | 0.000 |
| 4 | 512 | 8 | 16 | random | q_outer | 64 | 8 | 544.26 | 64 | 7334 | 121 | 469376 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 0 | 512 | 8 | 16 | random | q_outer | 128 | 16 | 2794.49 | 32 | 6475 | 219 | 828800 | 0 | 512 | 201.5 | 208.9 | 0.000 |
| 1 | 512 | 8 | 16 | random | q_outer | 128 | 16 | 2748.58 | 32 | 6512 | 212 | 833536 | 0 | 512 | 203.5 | 209.9 | 0.000 |
| 2 | 512 | 8 | 16 | random | q_outer | 128 | 16 | 2777.24 | 32 | 6541 | 217 | 837248 | 0 | 512 | 204.0 | 210.0 | 0.000 |
| 3 | 512 | 8 | 16 | random | q_outer | 128 | 16 | 2733.13 | 32 | 6520 | 213 | 834560 | 0 | 512 | 204.0 | 211.8 | 0.000 |
| 4 | 512 | 8 | 16 | random | q_outer | 128 | 16 | 2709.50 | 32 | 6527 | 213 | 835456 | 0 | 512 | 204.0 | 209.0 | 0.000 |
| 0 | 512 | 8 | 16 | random | signature | 16 | 2 | 110.05 | 614 | 8044 | 16 | 128704 | 204 | 512 | 15.0 | 16.0 | 1.000 |
| 1 | 512 | 8 | 16 | random | signature | 16 | 2 | 109.60 | 603 | 8050 | 16 | 128800 | 182 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 8 | 16 | random | signature | 16 | 2 | 110.69 | 609 | 8057 | 16 | 128912 | 194 | 512 | 15.0 | 16.0 | 1.000 |
| 3 | 512 | 8 | 16 | random | signature | 16 | 2 | 110.48 | 608 | 8048 | 16 | 128768 | 192 | 512 | 15.0 | 16.0 | 1.000 |
| 4 | 512 | 8 | 16 | random | signature | 16 | 2 | 111.22 | 619 | 8035 | 16 | 128560 | 214 | 512 | 15.0 | 16.0 | 1.000 |
| 0 | 512 | 8 | 16 | random | signature | 32 | 4 | 153.19 | 809 | 7789 | 16 | 124720 | 406 | 512 | 14.0 | 16.0 | 1.000 |
| 1 | 512 | 8 | 16 | random | signature | 32 | 4 | 151.81 | 795 | 7805 | 16 | 124960 | 396 | 512 | 14.0 | 16.0 | 1.000 |
| 2 | 512 | 8 | 16 | random | signature | 32 | 4 | 154.63 | 827 | 7788 | 16 | 124672 | 417 | 512 | 14.0 | 16.0 | 1.000 |
| 3 | 512 | 8 | 16 | random | signature | 32 | 4 | 151.13 | 777 | 7833 | 16 | 125488 | 380 | 512 | 14.0 | 16.0 | 1.000 |
| 4 | 512 | 8 | 16 | random | signature | 32 | 4 | 153.27 | 814 | 7789 | 16 | 124768 | 395 | 512 | 14.0 | 16.0 | 1.000 |
| 0 | 512 | 8 | 16 | random | signature | 64 | 4 | 162.41 | 1183 | 7303 | 16 | 117728 | 498 | 512 | 2.0 | 14.0 | 1.000 |
| 1 | 512 | 8 | 16 | random | signature | 64 | 4 | 160.80 | 1169 | 7343 | 16 | 118240 | 502 | 512 | 2.0 | 14.0 | 1.000 |
| 2 | 512 | 8 | 16 | random | signature | 64 | 4 | 162.44 | 1181 | 7321 | 16 | 117952 | 499 | 512 | 2.0 | 14.0 | 1.000 |
| 3 | 512 | 8 | 16 | random | signature | 64 | 4 | 162.08 | 1153 | 7370 | 16 | 118464 | 490 | 512 | 2.0 | 14.0 | 1.000 |
| 4 | 512 | 8 | 16 | random | signature | 64 | 4 | 160.18 | 1167 | 7334 | 16 | 118144 | 495 | 512 | 2.0 | 14.0 | 1.000 |
| 0 | 512 | 8 | 16 | random | signature | 128 | 8 | 292.77 | 1790 | 6475 | 16 | 107024 | 511 | 512 | 1.0 | 11.0 | 1.000 |
| 1 | 512 | 8 | 16 | random | signature | 128 | 8 | 293.36 | 1785 | 6512 | 15 | 107376 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 2 | 512 | 8 | 16 | random | signature | 128 | 8 | 298.56 | 1748 | 6541 | 15 | 107936 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 3 | 512 | 8 | 16 | random | signature | 128 | 8 | 292.79 | 1796 | 6520 | 15 | 107152 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 4 | 512 | 8 | 16 | random | signature | 128 | 4 | 157.49 | 1753 | 6527 | 15 | 108128 | 512 | 512 | 1.0 | 11.0 | 1.000 |
| 0 | 512 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 270.00 | 480 | 15832 | 62 | 253312 | 0 | 509 | 32.0 | 32.0 | 0.940 |
| 1 | 512 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 263.00 | 481 | 15893 | 62 | 254288 | 0 | 509 | 32.0 | 32.0 | 0.937 |
| 2 | 512 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 274.86 | 478 | 15920 | 63 | 254720 | 0 | 509 | 32.0 | 32.0 | 0.921 |
| 3 | 512 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 260.75 | 482 | 15929 | 62 | 254864 | 0 | 511 | 32.0 | 32.0 | 0.937 |
| 4 | 512 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 264.42 | 480 | 15884 | 63 | 254144 | 0 | 511 | 32.0 | 32.0 | 0.934 |
| 0 | 512 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 373.42 | 448 | 14845 | 63 | 239440 | 38 | 509 | 32.0 | 40.6 | 0.842 |
| 1 | 512 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 360.91 | 440 | 15141 | 62 | 242480 | 8 | 509 | 32.0 | 44.0 | 0.838 |
| 2 | 512 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 376.98 | 430 | 14863 | 63 | 238480 | 23 | 509 | 32.0 | 50.0 | 0.784 |
| 3 | 512 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 360.22 | 443 | 15112 | 63 | 242080 | 15 | 511 | 32.0 | 41.8 | 0.839 |
| 4 | 512 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 362.85 | 455 | 15005 | 61 | 241808 | 31 | 511 | 32.0 | 40.0 | 0.852 |
| 0 | 512 | 8 | 32 | clustered | bounded_merge | 64 | 8 | 575.81 | 626 | 13283 | 32 | 218624 | 268 | 509 | 24.0 | 32.0 | 0.937 |
| 1 | 512 | 8 | 32 | clustered | bounded_merge | 64 | 4 | 271.82 | 615 | 13131 | 32 | 214896 | 252 | 509 | 25.0 | 32.0 | 0.943 |
| 2 | 512 | 8 | 32 | clustered | bounded_merge | 64 | 4 | 283.85 | 630 | 13250 | 32 | 217408 | 266 | 509 | 25.0 | 32.0 | 0.934 |
| 3 | 512 | 8 | 32 | clustered | bounded_merge | 64 | 4 | 282.82 | 627 | 13253 | 32 | 216848 | 266 | 511 | 25.0 | 32.0 | 0.925 |
| 4 | 512 | 8 | 32 | clustered | bounded_merge | 64 | 8 | 562.55 | 627 | 13043 | 32 | 216656 | 258 | 511 | 23.0 | 32.0 | 0.932 |
| 0 | 512 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 489.20 | 779 | 10350 | 41 | 188928 | 434 | 509 | 11.0 | 31.0 | 0.961 |
| 1 | 512 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 510.94 | 768 | 10383 | 57 | 186752 | 428 | 509 | 11.0 | 32.0 | 0.955 |
| 2 | 512 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 490.86 | 776 | 10349 | 32 | 186960 | 433 | 509 | 10.0 | 31.5 | 0.965 |
| 3 | 512 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 522.52 | 771 | 10621 | 54 | 190336 | 424 | 511 | 11.0 | 32.0 | 0.967 |
| 4 | 512 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 493.63 | 759 | 10570 | 32 | 191296 | 405 | 511 | 11.0 | 32.0 | 0.965 |
| 0 | 512 | 8 | 32 | clustered | q_outer | 16 | 2 | 200.63 | 256 | 15832 | 64 | 253312 | 0 | 509 | 64.0 | 64.0 | 0.035 |
| 1 | 512 | 8 | 32 | clustered | q_outer | 16 | 2 | 197.76 | 256 | 15893 | 64 | 254288 | 0 | 509 | 64.0 | 64.0 | 0.031 |
| 2 | 512 | 8 | 32 | clustered | q_outer | 16 | 2 | 199.98 | 256 | 15920 | 64 | 254720 | 0 | 509 | 64.0 | 64.0 | 0.029 |
| 3 | 512 | 8 | 32 | clustered | q_outer | 16 | 2 | 199.99 | 256 | 15929 | 64 | 254864 | 0 | 511 | 64.0 | 64.0 | 0.029 |
| 4 | 512 | 8 | 32 | clustered | q_outer | 16 | 2 | 200.26 | 256 | 15884 | 64 | 254144 | 0 | 511 | 64.0 | 64.0 | 0.031 |
| 0 | 512 | 8 | 32 | clustered | q_outer | 32 | 4 | 395.05 | 128 | 14845 | 128 | 475040 | 0 | 509 | 123.5 | 128.0 | 0.000 |
| 1 | 512 | 8 | 32 | clustered | q_outer | 32 | 4 | 398.54 | 128 | 15141 | 128 | 484512 | 0 | 509 | 124.0 | 128.0 | 0.000 |
| 2 | 512 | 8 | 32 | clustered | q_outer | 32 | 4 | 400.15 | 128 | 14863 | 128 | 475616 | 0 | 509 | 122.0 | 128.0 | 0.000 |
| 3 | 512 | 8 | 32 | clustered | q_outer | 32 | 4 | 399.90 | 128 | 15112 | 128 | 483584 | 0 | 511 | 124.5 | 128.0 | 0.000 |
| 4 | 512 | 8 | 32 | clustered | q_outer | 32 | 4 | 396.51 | 128 | 15005 | 128 | 480160 | 0 | 511 | 125.0 | 128.0 | 0.001 |
| 0 | 512 | 8 | 32 | clustered | q_outer | 64 | 8 | 1137.98 | 64 | 13283 | 256 | 850112 | 0 | 509 | 209.0 | 236.4 | 0.000 |
| 1 | 512 | 8 | 32 | clustered | q_outer | 64 | 8 | 1134.19 | 64 | 13131 | 255 | 840384 | 0 | 509 | 207.5 | 230.0 | 0.000 |
| 2 | 512 | 8 | 32 | clustered | q_outer | 64 | 8 | 1135.44 | 64 | 13250 | 256 | 848000 | 0 | 509 | 206.5 | 235.8 | 0.000 |
| 3 | 512 | 8 | 32 | clustered | q_outer | 64 | 8 | 1124.96 | 64 | 13253 | 250 | 848192 | 0 | 511 | 208.0 | 241.4 | 0.000 |
| 4 | 512 | 8 | 32 | clustered | q_outer | 64 | 8 | 1139.11 | 64 | 13043 | 256 | 834752 | 0 | 511 | 206.0 | 231.4 | 0.000 |
| 0 | 512 | 8 | 32 | clustered | q_outer | 128 | 16 | 4738.32 | 32 | 10350 | 381 | 1324800 | 0 | 509 | 324.5 | 371.8 | 0.000 |
| 1 | 512 | 8 | 32 | clustered | q_outer | 128 | 16 | 4632.39 | 32 | 10383 | 373 | 1329024 | 0 | 509 | 328.0 | 353.4 | 0.000 |
| 2 | 512 | 8 | 32 | clustered | q_outer | 128 | 16 | 5014.58 | 32 | 10349 | 400 | 1324672 | 0 | 509 | 335.5 | 366.3 | 0.000 |
| 3 | 512 | 8 | 32 | clustered | q_outer | 128 | 16 | 4860.00 | 32 | 10621 | 387 | 1359488 | 0 | 511 | 336.5 | 366.9 | 0.000 |
| 4 | 512 | 8 | 32 | clustered | q_outer | 128 | 16 | 4681.15 | 32 | 10570 | 383 | 1352960 | 0 | 511 | 335.5 | 368.8 | 0.000 |
| 0 | 512 | 8 | 32 | clustered | signature | 16 | 2 | 207.35 | 542 | 15832 | 32 | 253312 | 62 | 509 | 32.0 | 32.0 | 1.000 |
| 1 | 512 | 8 | 32 | clustered | signature | 16 | 2 | 210.82 | 543 | 15893 | 32 | 254288 | 62 | 509 | 32.0 | 32.0 | 1.000 |
| 2 | 512 | 8 | 32 | clustered | signature | 16 | 2 | 208.84 | 546 | 15920 | 32 | 254720 | 68 | 509 | 32.0 | 32.0 | 1.000 |
| 3 | 512 | 8 | 32 | clustered | signature | 16 | 2 | 209.83 | 540 | 15929 | 32 | 254864 | 58 | 511 | 32.0 | 32.0 | 1.000 |
| 4 | 512 | 8 | 32 | clustered | signature | 16 | 2 | 216.41 | 544 | 15884 | 32 | 254144 | 64 | 511 | 32.0 | 32.0 | 1.000 |
| 0 | 512 | 8 | 32 | clustered | signature | 32 | 4 | 275.03 | 598 | 14845 | 32 | 239440 | 165 | 509 | 32.0 | 32.0 | 1.000 |
| 1 | 512 | 8 | 32 | clustered | signature | 32 | 4 | 278.44 | 590 | 15141 | 32 | 242480 | 153 | 509 | 32.0 | 32.0 | 1.000 |
| 2 | 512 | 8 | 32 | clustered | signature | 32 | 4 | 282.11 | 608 | 14863 | 32 | 238480 | 189 | 509 | 32.0 | 32.0 | 1.000 |
| 3 | 512 | 8 | 32 | clustered | signature | 32 | 4 | 277.85 | 588 | 15112 | 32 | 242080 | 152 | 511 | 32.0 | 32.0 | 1.000 |
| 4 | 512 | 8 | 32 | clustered | signature | 32 | 4 | 287.28 | 594 | 15005 | 32 | 241808 | 150 | 511 | 32.0 | 32.0 | 1.000 |
| 0 | 512 | 8 | 32 | clustered | signature | 64 | 8 | 508.51 | 687 | 13283 | 32 | 218624 | 308 | 509 | 20.0 | 32.0 | 1.000 |
| 1 | 512 | 8 | 32 | clustered | signature | 64 | 4 | 267.65 | 677 | 13131 | 32 | 214896 | 300 | 509 | 20.0 | 32.0 | 1.000 |
| 2 | 512 | 8 | 32 | clustered | signature | 64 | 4 | 272.60 | 690 | 13250 | 32 | 217408 | 311 | 509 | 20.0 | 32.0 | 1.000 |
| 3 | 512 | 8 | 32 | clustered | signature | 64 | 4 | 268.03 | 689 | 13253 | 32 | 216848 | 311 | 511 | 21.0 | 32.0 | 1.000 |
| 4 | 512 | 8 | 32 | clustered | signature | 64 | 8 | 500.26 | 690 | 13043 | 32 | 216656 | 301 | 511 | 19.0 | 32.0 | 1.000 |
| 0 | 512 | 8 | 32 | clustered | signature | 128 | 8 | 444.53 | 807 | 10350 | 32 | 188928 | 444 | 509 | 11.0 | 29.0 | 1.000 |
| 1 | 512 | 8 | 32 | clustered | signature | 128 | 8 | 432.70 | 800 | 10383 | 32 | 186752 | 444 | 509 | 11.0 | 30.0 | 1.000 |
| 2 | 512 | 8 | 32 | clustered | signature | 128 | 8 | 449.70 | 807 | 10349 | 32 | 186960 | 445 | 509 | 10.0 | 30.0 | 1.000 |
| 3 | 512 | 8 | 32 | clustered | signature | 128 | 8 | 452.18 | 801 | 10621 | 32 | 190336 | 436 | 511 | 11.0 | 31.0 | 1.000 |
| 4 | 512 | 8 | 32 | clustered | signature | 128 | 8 | 439.29 | 789 | 10570 | 32 | 191296 | 420 | 511 | 11.0 | 32.0 | 1.000 |
| 0 | 512 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 142.21 | 256 | 12264 | 48 | 196224 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 1 | 512 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 142.88 | 256 | 12263 | 48 | 196208 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 2 | 512 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 141.82 | 256 | 12263 | 48 | 196208 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 3 | 512 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 142.40 | 256 | 12263 | 48 | 196208 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 4 | 512 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 142.99 | 256 | 12264 | 48 | 196224 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 0 | 512 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 199.68 | 640 | 10216 | 16 | 196224 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 199.97 | 640 | 10215 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 199.67 | 640 | 10215 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 199.84 | 640 | 10215 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 199.94 | 640 | 10216 | 16 | 196224 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 358.40 | 576 | 9192 | 16 | 196224 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 359.31 | 576 | 9191 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 358.63 | 576 | 9191 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 358.09 | 576 | 9191 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 358.97 | 576 | 9192 | 16 | 196224 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 981.79 | 544 | 8680 | 16 | 196224 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 980.84 | 544 | 8679 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 981.37 | 544 | 8679 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 981.28 | 544 | 8679 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 983.14 | 544 | 8680 | 16 | 196224 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 8 | 32 | common_private | q_outer | 16 | 2 | 146.93 | 256 | 12264 | 48 | 196224 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 1 | 512 | 8 | 32 | common_private | q_outer | 16 | 2 | 147.53 | 256 | 12263 | 48 | 196208 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 2 | 512 | 8 | 32 | common_private | q_outer | 16 | 2 | 147.42 | 256 | 12263 | 48 | 196208 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 3 | 512 | 8 | 32 | common_private | q_outer | 16 | 2 | 147.72 | 256 | 12263 | 48 | 196208 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 4 | 512 | 8 | 32 | common_private | q_outer | 16 | 2 | 147.51 | 256 | 12264 | 48 | 196224 | 0 | 512 | 48.0 | 48.0 | 0.334 |
| 0 | 512 | 8 | 32 | common_private | q_outer | 32 | 4 | 256.78 | 128 | 10216 | 80 | 326912 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 1 | 512 | 8 | 32 | common_private | q_outer | 32 | 4 | 254.74 | 128 | 10215 | 80 | 326880 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 2 | 512 | 8 | 32 | common_private | q_outer | 32 | 4 | 257.21 | 128 | 10215 | 80 | 326880 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 3 | 512 | 8 | 32 | common_private | q_outer | 32 | 4 | 258.55 | 128 | 10215 | 80 | 326880 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 4 | 512 | 8 | 32 | common_private | q_outer | 32 | 4 | 257.32 | 128 | 10216 | 80 | 326912 | 0 | 512 | 80.0 | 80.0 | 0.200 |
| 0 | 512 | 8 | 32 | common_private | q_outer | 64 | 8 | 660.94 | 64 | 9192 | 144 | 588288 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 1 | 512 | 8 | 32 | common_private | q_outer | 64 | 8 | 651.82 | 64 | 9191 | 144 | 588224 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 2 | 512 | 8 | 32 | common_private | q_outer | 64 | 8 | 662.66 | 64 | 9191 | 144 | 588224 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 3 | 512 | 8 | 32 | common_private | q_outer | 64 | 8 | 660.92 | 64 | 9191 | 144 | 588224 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 4 | 512 | 8 | 32 | common_private | q_outer | 64 | 8 | 663.45 | 64 | 9192 | 144 | 588288 | 0 | 512 | 144.0 | 144.0 | 0.111 |
| 0 | 512 | 8 | 32 | common_private | q_outer | 128 | 16 | 3509.36 | 32 | 8680 | 272 | 1111040 | 0 | 512 | 271.0 | 272.0 | 0.059 |
| 1 | 512 | 8 | 32 | common_private | q_outer | 128 | 16 | 3521.47 | 32 | 8679 | 272 | 1110912 | 0 | 512 | 271.0 | 272.0 | 0.059 |
| 2 | 512 | 8 | 32 | common_private | q_outer | 128 | 16 | 3514.65 | 32 | 8679 | 272 | 1110912 | 0 | 512 | 271.0 | 272.0 | 0.059 |
| 3 | 512 | 8 | 32 | common_private | q_outer | 128 | 16 | 3532.24 | 32 | 8679 | 272 | 1110912 | 0 | 512 | 272.0 | 272.0 | 0.059 |
| 4 | 512 | 8 | 32 | common_private | q_outer | 128 | 16 | 3519.99 | 32 | 8680 | 272 | 1111040 | 0 | 512 | 271.0 | 272.0 | 0.059 |
| 0 | 512 | 8 | 32 | common_private | signature | 16 | 2 | 157.32 | 768 | 12264 | 16 | 196224 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 8 | 32 | common_private | signature | 16 | 2 | 157.70 | 768 | 12263 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 8 | 32 | common_private | signature | 16 | 2 | 157.39 | 768 | 12263 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 8 | 32 | common_private | signature | 16 | 2 | 157.59 | 768 | 12263 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 8 | 32 | common_private | signature | 16 | 2 | 157.78 | 768 | 12264 | 16 | 196224 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 8 | 32 | common_private | signature | 32 | 4 | 199.12 | 640 | 10216 | 16 | 196224 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 8 | 32 | common_private | signature | 32 | 4 | 199.58 | 640 | 10215 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 8 | 32 | common_private | signature | 32 | 4 | 200.36 | 640 | 10215 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 8 | 32 | common_private | signature | 32 | 4 | 199.95 | 640 | 10215 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 8 | 32 | common_private | signature | 32 | 4 | 200.38 | 640 | 10216 | 16 | 196224 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 8 | 32 | common_private | signature | 64 | 8 | 358.45 | 576 | 9192 | 16 | 196224 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 8 | 32 | common_private | signature | 64 | 8 | 359.46 | 576 | 9191 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 8 | 32 | common_private | signature | 64 | 8 | 359.03 | 576 | 9191 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 8 | 32 | common_private | signature | 64 | 8 | 357.79 | 576 | 9191 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 8 | 32 | common_private | signature | 64 | 8 | 358.64 | 576 | 9192 | 16 | 196224 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 8 | 32 | common_private | signature | 128 | 16 | 980.97 | 544 | 8680 | 16 | 196224 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 512 | 8 | 32 | common_private | signature | 128 | 16 | 981.55 | 544 | 8679 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 512 | 8 | 32 | common_private | signature | 128 | 16 | 981.83 | 544 | 8679 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 512 | 8 | 32 | common_private | signature | 128 | 16 | 981.71 | 544 | 8679 | 16 | 196208 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 512 | 8 | 32 | common_private | signature | 128 | 16 | 982.52 | 544 | 8680 | 16 | 196224 | 512 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 512 | 8 | 32 | random | bounded_merge | 16 | 2 | 268.86 | 286 | 15814 | 63 | 253024 | 0 | 512 | 61.0 | 63.0 | 0.154 |
| 1 | 512 | 8 | 32 | random | bounded_merge | 16 | 2 | 271.19 | 272 | 15783 | 63 | 252528 | 0 | 512 | 61.0 | 63.0 | 0.099 |
| 2 | 512 | 8 | 32 | random | bounded_merge | 16 | 2 | 272.87 | 283 | 15829 | 63 | 253264 | 0 | 512 | 61.0 | 63.0 | 0.142 |
| 3 | 512 | 8 | 32 | random | bounded_merge | 16 | 2 | 273.18 | 281 | 15803 | 63 | 252848 | 0 | 512 | 61.0 | 63.0 | 0.134 |
| 4 | 512 | 8 | 32 | random | bounded_merge | 16 | 2 | 273.82 | 296 | 15846 | 63 | 253536 | 0 | 512 | 61.0 | 63.0 | 0.192 |
| 0 | 512 | 8 | 32 | random | bounded_merge | 32 | 4 | 365.96 | 1073 | 14866 | 57 | 238736 | 507 | 512 | 4.0 | 29.0 | 0.747 |
| 1 | 512 | 8 | 32 | random | bounded_merge | 32 | 4 | 301.26 | 1097 | 14804 | 57 | 237824 | 508 | 512 | 4.0 | 28.0 | 0.756 |
| 2 | 512 | 8 | 32 | random | bounded_merge | 32 | 4 | 301.29 | 1083 | 14829 | 54 | 238272 | 506 | 512 | 4.0 | 28.0 | 0.748 |
| 3 | 512 | 8 | 32 | random | bounded_merge | 32 | 4 | 375.83 | 1067 | 14852 | 59 | 238704 | 502 | 512 | 4.0 | 28.0 | 0.733 |
| 4 | 512 | 8 | 32 | random | bounded_merge | 32 | 4 | 385.96 | 1051 | 14917 | 59 | 239632 | 502 | 512 | 4.0 | 29.0 | 0.731 |
| 0 | 512 | 8 | 32 | random | bounded_merge | 64 | 8 | 580.54 | 2151 | 13189 | 28 | 216448 | 512 | 512 | 2.0 | 21.0 | 0.968 |
| 1 | 512 | 8 | 32 | random | bounded_merge | 64 | 8 | 592.12 | 2222 | 13109 | 29 | 215520 | 512 | 512 | 2.0 | 21.0 | 0.980 |
| 2 | 512 | 8 | 32 | random | bounded_merge | 64 | 4 | 302.87 | 2210 | 13154 | 29 | 216480 | 512 | 512 | 2.0 | 21.0 | 0.980 |
| 3 | 512 | 8 | 32 | random | bounded_merge | 64 | 8 | 580.45 | 2221 | 13128 | 27 | 216016 | 512 | 512 | 2.0 | 21.0 | 0.980 |
| 4 | 512 | 8 | 32 | random | bounded_merge | 64 | 4 | 309.75 | 2185 | 13179 | 28 | 216368 | 512 | 512 | 2.0 | 21.0 | 0.977 |
| 0 | 512 | 8 | 32 | random | bounded_merge | 128 | 8 | 501.49 | 3853 | 10486 | 22 | 188272 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 512 | 8 | 32 | random | bounded_merge | 128 | 8 | 506.75 | 3875 | 10479 | 21 | 188416 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 512 | 8 | 32 | random | bounded_merge | 128 | 8 | 502.89 | 3855 | 10521 | 20 | 188944 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 512 | 8 | 32 | random | bounded_merge | 128 | 8 | 519.03 | 3885 | 10483 | 21 | 189040 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 512 | 8 | 32 | random | bounded_merge | 128 | 8 | 508.30 | 3878 | 10505 | 21 | 189264 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 512 | 8 | 32 | random | q_outer | 16 | 2 | 189.62 | 256 | 15814 | 64 | 253024 | 0 | 512 | 62.0 | 63.0 | 0.033 |
| 1 | 512 | 8 | 32 | random | q_outer | 16 | 2 | 191.01 | 256 | 15783 | 64 | 252528 | 0 | 512 | 62.0 | 63.0 | 0.035 |
| 2 | 512 | 8 | 32 | random | q_outer | 16 | 2 | 190.00 | 256 | 15829 | 64 | 253264 | 0 | 512 | 62.0 | 63.0 | 0.033 |
| 3 | 512 | 8 | 32 | random | q_outer | 16 | 2 | 194.21 | 256 | 15803 | 64 | 252848 | 0 | 512 | 62.0 | 63.0 | 0.033 |
| 4 | 512 | 8 | 32 | random | q_outer | 16 | 2 | 192.23 | 256 | 15846 | 64 | 253536 | 0 | 512 | 62.0 | 64.0 | 0.031 |
| 0 | 512 | 8 | 32 | random | q_outer | 32 | 4 | 373.66 | 128 | 14866 | 122 | 475712 | 0 | 512 | 116.0 | 120.3 | 0.000 |
| 1 | 512 | 8 | 32 | random | q_outer | 32 | 4 | 381.22 | 128 | 14804 | 124 | 473728 | 0 | 512 | 116.0 | 119.3 | 0.000 |
| 2 | 512 | 8 | 32 | random | q_outer | 32 | 4 | 371.97 | 128 | 14829 | 122 | 474528 | 0 | 512 | 116.0 | 119.0 | 0.000 |
| 3 | 512 | 8 | 32 | random | q_outer | 32 | 4 | 381.34 | 128 | 14852 | 123 | 475264 | 0 | 512 | 116.0 | 120.0 | 0.000 |
| 4 | 512 | 8 | 32 | random | q_outer | 32 | 4 | 376.95 | 128 | 14917 | 122 | 477344 | 0 | 512 | 117.0 | 120.0 | 0.000 |
| 0 | 512 | 8 | 32 | random | q_outer | 64 | 8 | 967.65 | 64 | 13189 | 219 | 844096 | 0 | 512 | 206.0 | 213.0 | 0.000 |
| 1 | 512 | 8 | 32 | random | q_outer | 64 | 8 | 966.40 | 64 | 13109 | 216 | 838976 | 0 | 512 | 205.0 | 210.0 | 0.000 |
| 2 | 512 | 8 | 32 | random | q_outer | 64 | 8 | 967.28 | 64 | 13154 | 219 | 841856 | 0 | 512 | 206.0 | 210.7 | 0.000 |
| 3 | 512 | 8 | 32 | random | q_outer | 64 | 8 | 952.24 | 64 | 13128 | 215 | 840192 | 0 | 512 | 205.0 | 211.0 | 0.000 |
| 4 | 512 | 8 | 32 | random | q_outer | 64 | 8 | 984.79 | 64 | 13179 | 222 | 843456 | 0 | 512 | 206.0 | 212.0 | 0.000 |
| 0 | 512 | 8 | 32 | random | q_outer | 128 | 16 | 4366.02 | 32 | 10486 | 343 | 1342208 | 0 | 512 | 328.5 | 335.9 | 0.000 |
| 1 | 512 | 8 | 32 | random | q_outer | 128 | 16 | 4245.28 | 32 | 10479 | 344 | 1341312 | 0 | 512 | 327.0 | 336.7 | 0.000 |
| 2 | 512 | 8 | 32 | random | q_outer | 128 | 16 | 4382.84 | 32 | 10521 | 345 | 1346688 | 0 | 512 | 327.5 | 338.9 | 0.000 |
| 3 | 512 | 8 | 32 | random | q_outer | 128 | 16 | 4302.94 | 32 | 10483 | 342 | 1341824 | 0 | 512 | 329.0 | 337.8 | 0.000 |
| 4 | 512 | 8 | 32 | random | q_outer | 128 | 16 | 4421.51 | 32 | 10505 | 351 | 1344640 | 0 | 512 | 328.0 | 337.9 | 0.000 |
| 0 | 512 | 8 | 32 | random | signature | 16 | 2 | 203.97 | 738 | 15814 | 32 | 253024 | 452 | 512 | 29.0 | 31.0 | 1.000 |
| 1 | 512 | 8 | 32 | random | signature | 16 | 2 | 205.46 | 752 | 15783 | 32 | 252528 | 480 | 512 | 29.0 | 31.0 | 1.000 |
| 2 | 512 | 8 | 32 | random | signature | 16 | 2 | 204.51 | 741 | 15829 | 32 | 253264 | 458 | 512 | 29.0 | 31.0 | 1.000 |
| 3 | 512 | 8 | 32 | random | signature | 16 | 2 | 207.18 | 743 | 15803 | 32 | 252848 | 462 | 512 | 29.0 | 31.0 | 1.000 |
| 4 | 512 | 8 | 32 | random | signature | 16 | 2 | 209.79 | 728 | 15846 | 32 | 253536 | 432 | 512 | 29.0 | 32.0 | 1.000 |
| 0 | 512 | 8 | 32 | random | signature | 32 | 4 | 286.41 | 1216 | 14866 | 31 | 238736 | 512 | 512 | 3.0 | 28.0 | 1.000 |
| 1 | 512 | 8 | 32 | random | signature | 32 | 4 | 290.39 | 1236 | 14804 | 32 | 237824 | 511 | 512 | 3.0 | 28.0 | 1.000 |
| 2 | 512 | 8 | 32 | random | signature | 32 | 4 | 286.85 | 1226 | 14829 | 31 | 238272 | 512 | 512 | 3.0 | 28.0 | 1.000 |
| 3 | 512 | 8 | 32 | random | signature | 32 | 4 | 296.55 | 1218 | 14852 | 32 | 238704 | 510 | 512 | 3.0 | 28.0 | 1.000 |
| 4 | 512 | 8 | 32 | random | signature | 32 | 4 | 301.75 | 1202 | 14917 | 32 | 239632 | 510 | 512 | 3.0 | 28.0 | 1.000 |
| 0 | 512 | 8 | 32 | random | signature | 64 | 8 | 544.83 | 2171 | 13189 | 28 | 216448 | 512 | 512 | 2.0 | 21.0 | 1.000 |
| 1 | 512 | 8 | 32 | random | signature | 64 | 8 | 540.19 | 2234 | 13109 | 28 | 215520 | 512 | 512 | 2.0 | 21.0 | 1.000 |
| 2 | 512 | 8 | 32 | random | signature | 64 | 4 | 283.33 | 2222 | 13154 | 28 | 216480 | 512 | 512 | 2.0 | 21.0 | 1.000 |
| 3 | 512 | 8 | 32 | random | signature | 64 | 8 | 544.73 | 2233 | 13128 | 27 | 216016 | 512 | 512 | 2.0 | 21.0 | 1.000 |
| 4 | 512 | 8 | 32 | random | signature | 64 | 4 | 298.98 | 2199 | 13179 | 28 | 216368 | 512 | 512 | 2.0 | 21.0 | 1.000 |
| 0 | 512 | 8 | 32 | random | signature | 128 | 8 | 502.05 | 3853 | 10486 | 22 | 188272 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 512 | 8 | 32 | random | signature | 128 | 8 | 509.98 | 3875 | 10479 | 21 | 188416 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 512 | 8 | 32 | random | signature | 128 | 8 | 493.72 | 3855 | 10521 | 20 | 188944 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 512 | 8 | 32 | random | signature | 128 | 8 | 518.58 | 3885 | 10483 | 21 | 189040 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 512 | 8 | 32 | random | signature | 128 | 8 | 508.47 | 3878 | 10505 | 21 | 189264 | 512 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 2048 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 228.80 | 1917 | 15830 | 31 | 253280 | 0 | 510 | 8.0 | 8.0 | 0.940 |
| 1 | 2048 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 220.64 | 1925 | 15915 | 31 | 254640 | 0 | 511 | 8.0 | 8.0 | 0.939 |
| 2 | 2048 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 217.58 | 1920 | 15892 | 27 | 254272 | 0 | 511 | 8.0 | 8.0 | 0.934 |
| 3 | 2048 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 227.36 | 1923 | 15912 | 31 | 254592 | 0 | 510 | 8.0 | 8.0 | 0.936 |
| 4 | 2048 | 4 | 8 | clustered | bounded_merge | 16 | 4 | 230.56 | 1907 | 15855 | 31 | 253680 | 0 | 511 | 8.0 | 8.0 | 0.930 |
| 0 | 2048 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 201.08 | 2073 | 15341 | 8 | 245456 | 233 | 510 | 8.0 | 8.0 | 0.965 |
| 1 | 2048 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 200.84 | 2085 | 15335 | 8 | 245360 | 253 | 511 | 8.0 | 8.0 | 0.963 |
| 2 | 2048 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 206.94 | 2074 | 15472 | 8 | 247552 | 222 | 511 | 8.0 | 8.0 | 0.963 |
| 3 | 2048 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 202.62 | 2087 | 15358 | 8 | 245728 | 239 | 510 | 8.0 | 8.0 | 0.970 |
| 4 | 2048 | 4 | 8 | clustered | bounded_merge | 32 | 4 | 203.37 | 2076 | 15325 | 8 | 245200 | 233 | 511 | 8.0 | 8.0 | 0.969 |
| 0 | 2048 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 193.71 | 2325 | 14341 | 8 | 229456 | 659 | 510 | 8.0 | 8.0 | 0.979 |
| 1 | 2048 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 193.76 | 2304 | 14478 | 8 | 231648 | 620 | 511 | 8.0 | 8.0 | 0.982 |
| 2 | 2048 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 199.46 | 2286 | 14561 | 8 | 232976 | 606 | 511 | 8.0 | 8.0 | 0.979 |
| 3 | 2048 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 198.04 | 2318 | 14431 | 8 | 230896 | 648 | 510 | 8.0 | 8.0 | 0.984 |
| 4 | 2048 | 4 | 8 | clustered | bounded_merge | 64 | 4 | 199.82 | 2312 | 14463 | 8 | 231408 | 641 | 511 | 8.0 | 8.0 | 0.983 |
| 0 | 2048 | 4 | 8 | clustered | bounded_merge | 128 | 8 | 257.15 | 2627 | 12772 | 8 | 204384 | 1177 | 510 | 5.0 | 8.0 | 1.000 |
| 1 | 2048 | 4 | 8 | clustered | bounded_merge | 128 | 8 | 257.81 | 2630 | 12803 | 8 | 204896 | 1163 | 511 | 5.0 | 8.0 | 1.000 |
| 2 | 2048 | 4 | 8 | clustered | bounded_merge | 128 | 4 | 199.15 | 2601 | 12910 | 8 | 206560 | 1158 | 511 | 5.0 | 8.0 | 1.000 |
| 3 | 2048 | 4 | 8 | clustered | bounded_merge | 128 | 8 | 265.41 | 2630 | 12815 | 8 | 205152 | 1187 | 510 | 5.0 | 8.0 | 1.000 |
| 4 | 2048 | 4 | 8 | clustered | bounded_merge | 128 | 8 | 266.02 | 2634 | 12817 | 8 | 205104 | 1189 | 511 | 5.0 | 8.0 | 1.000 |
| 0 | 2048 | 4 | 8 | clustered | q_outer | 16 | 4 | 189.64 | 512 | 15830 | 32 | 253280 | 0 | 510 | 32.0 | 32.0 | 0.000 |
| 1 | 2048 | 4 | 8 | clustered | q_outer | 16 | 4 | 191.60 | 512 | 15915 | 32 | 254640 | 0 | 511 | 32.0 | 32.0 | 0.000 |
| 2 | 2048 | 4 | 8 | clustered | q_outer | 16 | 4 | 191.00 | 512 | 15892 | 32 | 254272 | 0 | 511 | 32.0 | 32.0 | 0.000 |
| 3 | 2048 | 4 | 8 | clustered | q_outer | 16 | 4 | 189.48 | 512 | 15912 | 32 | 254592 | 0 | 510 | 32.0 | 32.0 | 0.000 |
| 4 | 2048 | 4 | 8 | clustered | q_outer | 16 | 4 | 190.11 | 512 | 15855 | 32 | 253680 | 0 | 511 | 32.0 | 32.0 | 0.000 |
| 0 | 2048 | 4 | 8 | clustered | q_outer | 32 | 8 | 277.18 | 256 | 15341 | 64 | 490912 | 0 | 510 | 61.0 | 64.0 | 0.000 |
| 1 | 2048 | 4 | 8 | clustered | q_outer | 32 | 8 | 277.24 | 256 | 15335 | 64 | 490720 | 0 | 511 | 60.0 | 64.0 | 0.000 |
| 2 | 2048 | 4 | 8 | clustered | q_outer | 32 | 8 | 278.77 | 256 | 15472 | 64 | 495104 | 0 | 511 | 62.0 | 64.0 | 0.000 |
| 3 | 2048 | 4 | 8 | clustered | q_outer | 32 | 8 | 276.20 | 256 | 15358 | 64 | 491456 | 0 | 510 | 61.0 | 64.0 | 0.000 |
| 4 | 2048 | 4 | 8 | clustered | q_outer | 32 | 8 | 276.46 | 256 | 15325 | 64 | 490400 | 0 | 511 | 61.0 | 64.0 | 0.000 |
| 0 | 2048 | 4 | 8 | clustered | q_outer | 64 | 16 | 599.92 | 128 | 14341 | 128 | 917824 | 0 | 510 | 112.0 | 121.3 | 0.000 |
| 1 | 2048 | 4 | 8 | clustered | q_outer | 64 | 16 | 600.95 | 128 | 14478 | 128 | 926592 | 0 | 511 | 113.0 | 122.0 | 0.000 |
| 2 | 2048 | 4 | 8 | clustered | q_outer | 64 | 16 | 603.54 | 128 | 14561 | 128 | 931904 | 0 | 511 | 114.0 | 124.0 | 0.000 |
| 3 | 2048 | 4 | 8 | clustered | q_outer | 64 | 16 | 603.75 | 128 | 14431 | 128 | 923584 | 0 | 510 | 114.0 | 122.0 | 0.000 |
| 4 | 2048 | 4 | 8 | clustered | q_outer | 64 | 16 | 602.99 | 128 | 14463 | 128 | 925632 | 0 | 511 | 113.0 | 122.3 | 0.000 |
| 0 | 2048 | 4 | 8 | clustered | q_outer | 128 | 32 | 2571.08 | 64 | 12772 | 229 | 1634816 | 0 | 510 | 199.5 | 216.0 | 0.000 |
| 1 | 2048 | 4 | 8 | clustered | q_outer | 128 | 32 | 2572.93 | 64 | 12803 | 229 | 1638784 | 0 | 511 | 199.5 | 215.0 | 0.000 |
| 2 | 2048 | 4 | 8 | clustered | q_outer | 128 | 32 | 2679.06 | 64 | 12910 | 237 | 1652480 | 0 | 511 | 202.5 | 216.4 | 0.000 |
| 3 | 2048 | 4 | 8 | clustered | q_outer | 128 | 32 | 2520.88 | 64 | 12815 | 226 | 1640320 | 0 | 510 | 200.5 | 211.7 | 0.000 |
| 4 | 2048 | 4 | 8 | clustered | q_outer | 128 | 32 | 2574.41 | 64 | 12817 | 231 | 1640576 | 0 | 511 | 201.0 | 213.0 | 0.000 |
| 0 | 2048 | 4 | 8 | clustered | signature | 16 | 4 | 217.72 | 2113 | 15830 | 8 | 253280 | 157 | 510 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 4 | 8 | clustered | signature | 16 | 4 | 217.91 | 2112 | 15915 | 8 | 254640 | 152 | 511 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 4 | 8 | clustered | signature | 16 | 2 | 169.46 | 2116 | 15892 | 8 | 254272 | 164 | 511 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 4 | 8 | clustered | signature | 16 | 4 | 217.48 | 2125 | 15912 | 8 | 254592 | 167 | 510 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 4 | 8 | clustered | signature | 16 | 4 | 216.40 | 2115 | 15855 | 8 | 253680 | 164 | 511 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 4 | 8 | clustered | signature | 32 | 4 | 213.71 | 2216 | 15341 | 8 | 245456 | 369 | 510 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 4 | 8 | clustered | signature | 32 | 4 | 212.68 | 2231 | 15335 | 8 | 245360 | 388 | 511 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 4 | 8 | clustered | signature | 32 | 4 | 213.78 | 2201 | 15472 | 8 | 247552 | 343 | 511 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 4 | 8 | clustered | signature | 32 | 4 | 213.32 | 2215 | 15358 | 8 | 245728 | 359 | 510 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 4 | 8 | clustered | signature | 32 | 4 | 213.67 | 2208 | 15325 | 8 | 245200 | 363 | 511 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 4 | 8 | clustered | signature | 64 | 4 | 205.42 | 2401 | 14341 | 8 | 229456 | 727 | 510 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 4 | 8 | clustered | signature | 64 | 4 | 205.42 | 2374 | 14478 | 8 | 231648 | 677 | 511 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 4 | 8 | clustered | signature | 64 | 4 | 206.28 | 2360 | 14561 | 8 | 232976 | 672 | 511 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 4 | 8 | clustered | signature | 64 | 4 | 204.87 | 2378 | 14431 | 8 | 230896 | 704 | 510 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 4 | 8 | clustered | signature | 64 | 4 | 206.44 | 2379 | 14463 | 8 | 231408 | 699 | 511 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 4 | 8 | clustered | signature | 128 | 8 | 256.17 | 2627 | 12772 | 8 | 204384 | 1177 | 510 | 5.0 | 8.0 | 1.000 |
| 1 | 2048 | 4 | 8 | clustered | signature | 128 | 8 | 256.07 | 2630 | 12803 | 8 | 204896 | 1163 | 511 | 5.0 | 8.0 | 1.000 |
| 2 | 2048 | 4 | 8 | clustered | signature | 128 | 4 | 193.15 | 2601 | 12910 | 8 | 206560 | 1158 | 511 | 5.0 | 8.0 | 1.000 |
| 3 | 2048 | 4 | 8 | clustered | signature | 128 | 8 | 256.54 | 2630 | 12815 | 8 | 205152 | 1187 | 510 | 5.0 | 8.0 | 1.000 |
| 4 | 2048 | 4 | 8 | clustered | signature | 128 | 8 | 256.82 | 2634 | 12817 | 8 | 205104 | 1189 | 511 | 5.0 | 8.0 | 1.000 |
| 0 | 2048 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 118.72 | 512 | 10117 | 20 | 161872 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 1 | 2048 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 119.76 | 512 | 10120 | 20 | 161920 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 2 | 2048 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 118.83 | 512 | 10118 | 20 | 161888 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 3 | 2048 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 118.39 | 512 | 10120 | 20 | 161920 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 4 | 2048 | 4 | 8 | common_private | bounded_merge | 16 | 4 | 118.96 | 512 | 10119 | 20 | 161904 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 0 | 2048 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 188.45 | 2304 | 9093 | 4 | 161872 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 2048 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 189.03 | 2304 | 9096 | 4 | 161920 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 2048 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 188.28 | 2304 | 9094 | 4 | 161888 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 2048 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 187.82 | 2304 | 9096 | 4 | 161920 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 2048 | 4 | 8 | common_private | bounded_merge | 32 | 8 | 188.52 | 2304 | 9095 | 4 | 161904 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 2048 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 378.55 | 2176 | 8581 | 4 | 161872 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 2048 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 380.42 | 2176 | 8584 | 4 | 161920 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 2048 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 379.80 | 2176 | 8582 | 4 | 161888 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 2048 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 379.82 | 2176 | 8584 | 4 | 161920 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 2048 | 4 | 8 | common_private | bounded_merge | 64 | 16 | 379.10 | 2176 | 8583 | 4 | 161904 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 2048 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 978.77 | 2112 | 8325 | 4 | 161872 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 2048 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 989.29 | 2112 | 8328 | 4 | 161920 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 2048 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 982.83 | 2112 | 8326 | 4 | 161888 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 2048 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 990.35 | 2112 | 8328 | 4 | 161920 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 2048 | 4 | 8 | common_private | bounded_merge | 128 | 32 | 984.81 | 2112 | 8327 | 4 | 161904 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 2048 | 4 | 8 | common_private | q_outer | 16 | 4 | 122.90 | 512 | 10117 | 20 | 161872 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 1 | 2048 | 4 | 8 | common_private | q_outer | 16 | 4 | 122.99 | 512 | 10120 | 20 | 161920 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 2 | 2048 | 4 | 8 | common_private | q_outer | 16 | 4 | 121.94 | 512 | 10118 | 20 | 161888 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 3 | 2048 | 4 | 8 | common_private | q_outer | 16 | 4 | 122.24 | 512 | 10120 | 20 | 161920 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 4 | 2048 | 4 | 8 | common_private | q_outer | 16 | 4 | 122.63 | 512 | 10119 | 20 | 161904 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 0 | 2048 | 4 | 8 | common_private | q_outer | 32 | 8 | 160.58 | 256 | 9093 | 36 | 290976 | 0 | 512 | 36.0 | 36.0 | 0.113 |
| 1 | 2048 | 4 | 8 | common_private | q_outer | 32 | 8 | 161.57 | 256 | 9096 | 36 | 291072 | 0 | 512 | 36.0 | 36.0 | 0.113 |
| 2 | 2048 | 4 | 8 | common_private | q_outer | 32 | 8 | 159.82 | 256 | 9094 | 36 | 291008 | 0 | 512 | 36.0 | 36.0 | 0.113 |
| 3 | 2048 | 4 | 8 | common_private | q_outer | 32 | 8 | 161.38 | 256 | 9096 | 36 | 291072 | 0 | 512 | 36.0 | 36.0 | 0.113 |
| 4 | 2048 | 4 | 8 | common_private | q_outer | 32 | 8 | 161.73 | 256 | 9095 | 36 | 291040 | 0 | 512 | 36.0 | 36.0 | 0.113 |
| 0 | 2048 | 4 | 8 | common_private | q_outer | 64 | 16 | 333.73 | 128 | 8581 | 68 | 549184 | 0 | 512 | 67.0 | 68.0 | 0.060 |
| 1 | 2048 | 4 | 8 | common_private | q_outer | 64 | 16 | 330.17 | 128 | 8584 | 68 | 549376 | 0 | 512 | 67.0 | 68.0 | 0.060 |
| 2 | 2048 | 4 | 8 | common_private | q_outer | 64 | 16 | 333.19 | 128 | 8582 | 68 | 549248 | 0 | 512 | 67.0 | 68.0 | 0.060 |
| 3 | 2048 | 4 | 8 | common_private | q_outer | 64 | 16 | 335.36 | 128 | 8584 | 68 | 549376 | 0 | 512 | 67.0 | 68.0 | 0.060 |
| 4 | 2048 | 4 | 8 | common_private | q_outer | 64 | 16 | 333.12 | 128 | 8583 | 68 | 549312 | 0 | 512 | 67.0 | 68.0 | 0.060 |
| 0 | 2048 | 4 | 8 | common_private | q_outer | 128 | 32 | 1526.20 | 64 | 8325 | 132 | 1065600 | 0 | 512 | 130.0 | 132.0 | 0.031 |
| 1 | 2048 | 4 | 8 | common_private | q_outer | 128 | 32 | 1529.35 | 64 | 8328 | 132 | 1065984 | 0 | 512 | 130.0 | 132.0 | 0.031 |
| 2 | 2048 | 4 | 8 | common_private | q_outer | 128 | 32 | 1525.12 | 64 | 8326 | 132 | 1065728 | 0 | 512 | 131.0 | 132.0 | 0.031 |
| 3 | 2048 | 4 | 8 | common_private | q_outer | 128 | 32 | 1532.60 | 64 | 8328 | 132 | 1065984 | 0 | 512 | 131.0 | 132.0 | 0.031 |
| 4 | 2048 | 4 | 8 | common_private | q_outer | 128 | 32 | 1540.25 | 64 | 8327 | 132 | 1065856 | 0 | 512 | 130.0 | 132.0 | 0.031 |
| 0 | 2048 | 4 | 8 | common_private | signature | 16 | 4 | 159.88 | 2560 | 10117 | 4 | 161872 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 2048 | 4 | 8 | common_private | signature | 16 | 4 | 158.63 | 2560 | 10120 | 4 | 161920 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 2048 | 4 | 8 | common_private | signature | 16 | 4 | 160.00 | 2560 | 10118 | 4 | 161888 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 2048 | 4 | 8 | common_private | signature | 16 | 4 | 160.38 | 2560 | 10120 | 4 | 161920 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 2048 | 4 | 8 | common_private | signature | 16 | 4 | 159.89 | 2560 | 10119 | 4 | 161904 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 2048 | 4 | 8 | common_private | signature | 32 | 8 | 190.32 | 2304 | 9093 | 4 | 161872 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 2048 | 4 | 8 | common_private | signature | 32 | 8 | 190.04 | 2304 | 9096 | 4 | 161920 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 2048 | 4 | 8 | common_private | signature | 32 | 8 | 191.16 | 2304 | 9094 | 4 | 161888 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 2048 | 4 | 8 | common_private | signature | 32 | 8 | 191.81 | 2304 | 9096 | 4 | 161920 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 2048 | 4 | 8 | common_private | signature | 32 | 8 | 191.54 | 2304 | 9095 | 4 | 161904 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 2048 | 4 | 8 | common_private | signature | 64 | 16 | 380.69 | 2176 | 8581 | 4 | 161872 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 2048 | 4 | 8 | common_private | signature | 64 | 16 | 386.10 | 2176 | 8584 | 4 | 161920 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 2048 | 4 | 8 | common_private | signature | 64 | 16 | 382.49 | 2176 | 8582 | 4 | 161888 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 2048 | 4 | 8 | common_private | signature | 64 | 16 | 381.97 | 2176 | 8584 | 4 | 161920 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 2048 | 4 | 8 | common_private | signature | 64 | 16 | 386.68 | 2176 | 8583 | 4 | 161904 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 2048 | 4 | 8 | common_private | signature | 128 | 32 | 983.34 | 2112 | 8325 | 4 | 161872 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 2048 | 4 | 8 | common_private | signature | 128 | 32 | 998.70 | 2112 | 8328 | 4 | 161920 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 2048 | 4 | 8 | common_private | signature | 128 | 32 | 983.39 | 2112 | 8326 | 4 | 161888 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 2048 | 4 | 8 | common_private | signature | 128 | 32 | 990.61 | 2112 | 8328 | 4 | 161920 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 2048 | 4 | 8 | common_private | signature | 128 | 32 | 998.06 | 2112 | 8327 | 4 | 161904 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 2048 | 4 | 8 | random | bounded_merge | 16 | 4 | 232.26 | 1389 | 15769 | 31 | 252304 | 0 | 512 | 8.0 | 23.0 | 0.544 |
| 1 | 2048 | 4 | 8 | random | bounded_merge | 16 | 4 | 230.58 | 1416 | 15765 | 31 | 252240 | 0 | 511 | 8.0 | 23.0 | 0.564 |
| 2 | 2048 | 4 | 8 | random | bounded_merge | 16 | 4 | 230.32 | 1431 | 15796 | 31 | 252736 | 0 | 511 | 8.0 | 23.0 | 0.572 |
| 3 | 2048 | 4 | 8 | random | bounded_merge | 16 | 4 | 227.81 | 1376 | 15771 | 31 | 252336 | 0 | 512 | 8.0 | 23.0 | 0.536 |
| 4 | 2048 | 4 | 8 | random | bounded_merge | 16 | 4 | 231.30 | 1388 | 15790 | 31 | 252640 | 0 | 512 | 8.0 | 23.0 | 0.541 |
| 0 | 2048 | 4 | 8 | random | bounded_merge | 32 | 4 | 201.66 | 2588 | 15284 | 8 | 244544 | 1023 | 512 | 7.0 | 8.0 | 0.899 |
| 1 | 2048 | 4 | 8 | random | bounded_merge | 32 | 4 | 203.75 | 2584 | 15305 | 8 | 244880 | 990 | 511 | 7.0 | 8.0 | 0.899 |
| 2 | 2048 | 4 | 8 | random | bounded_merge | 32 | 4 | 203.01 | 2643 | 15268 | 8 | 244288 | 1082 | 511 | 7.0 | 8.0 | 0.897 |
| 3 | 2048 | 4 | 8 | random | bounded_merge | 32 | 4 | 202.82 | 2622 | 15286 | 8 | 244576 | 1032 | 512 | 7.0 | 8.0 | 0.899 |
| 4 | 2048 | 4 | 8 | random | bounded_merge | 32 | 4 | 202.24 | 2613 | 15298 | 8 | 244768 | 1045 | 512 | 7.0 | 8.0 | 0.897 |
| 0 | 2048 | 4 | 8 | random | bounded_merge | 64 | 4 | 204.85 | 3528 | 14352 | 8 | 229632 | 1692 | 512 | 5.0 | 7.0 | 0.958 |
| 1 | 2048 | 4 | 8 | random | bounded_merge | 64 | 8 | 302.76 | 3505 | 14391 | 8 | 230272 | 1672 | 511 | 5.0 | 7.0 | 0.957 |
| 2 | 2048 | 4 | 8 | random | bounded_merge | 64 | 4 | 204.13 | 3531 | 14387 | 8 | 230192 | 1682 | 511 | 5.0 | 7.0 | 0.955 |
| 3 | 2048 | 4 | 8 | random | bounded_merge | 64 | 4 | 204.89 | 3533 | 14401 | 8 | 230416 | 1698 | 512 | 5.0 | 7.0 | 0.954 |
| 4 | 2048 | 4 | 8 | random | bounded_merge | 64 | 4 | 205.79 | 3481 | 14434 | 8 | 230944 | 1690 | 512 | 5.0 | 7.0 | 0.954 |
| 0 | 2048 | 4 | 8 | random | bounded_merge | 128 | 8 | 283.52 | 4846 | 12749 | 8 | 204032 | 1999 | 512 | 1.0 | 6.0 | 1.000 |
| 1 | 2048 | 4 | 8 | random | bounded_merge | 128 | 8 | 285.97 | 4815 | 12842 | 8 | 205584 | 2007 | 511 | 1.0 | 6.0 | 1.000 |
| 2 | 2048 | 4 | 8 | random | bounded_merge | 128 | 8 | 284.89 | 4817 | 12806 | 8 | 204992 | 2009 | 511 | 1.0 | 6.0 | 1.000 |
| 3 | 2048 | 4 | 8 | random | bounded_merge | 128 | 8 | 286.48 | 4845 | 12796 | 8 | 204752 | 2000 | 512 | 1.0 | 6.0 | 1.000 |
| 4 | 2048 | 4 | 8 | random | bounded_merge | 128 | 8 | 286.89 | 4850 | 12806 | 8 | 204976 | 1999 | 512 | 1.0 | 6.0 | 1.000 |
| 0 | 2048 | 4 | 8 | random | q_outer | 16 | 4 | 185.22 | 512 | 15769 | 32 | 252304 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 1 | 2048 | 4 | 8 | random | q_outer | 16 | 4 | 185.46 | 512 | 15765 | 32 | 252240 | 0 | 511 | 31.0 | 32.0 | 0.000 |
| 2 | 2048 | 4 | 8 | random | q_outer | 16 | 4 | 185.56 | 512 | 15796 | 32 | 252736 | 0 | 511 | 31.0 | 32.0 | 0.000 |
| 3 | 2048 | 4 | 8 | random | q_outer | 16 | 4 | 185.05 | 512 | 15771 | 32 | 252336 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 4 | 2048 | 4 | 8 | random | q_outer | 16 | 4 | 184.36 | 512 | 15790 | 32 | 252640 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 0 | 2048 | 4 | 8 | random | q_outer | 32 | 8 | 268.96 | 256 | 15284 | 63 | 489088 | 0 | 512 | 60.0 | 62.0 | 0.000 |
| 1 | 2048 | 4 | 8 | random | q_outer | 32 | 8 | 276.19 | 256 | 15305 | 64 | 489760 | 0 | 511 | 60.0 | 62.0 | 0.000 |
| 2 | 2048 | 4 | 8 | random | q_outer | 32 | 8 | 268.44 | 256 | 15268 | 64 | 488576 | 0 | 511 | 60.0 | 62.0 | 0.000 |
| 3 | 2048 | 4 | 8 | random | q_outer | 32 | 8 | 271.48 | 256 | 15286 | 64 | 489152 | 0 | 512 | 60.0 | 62.0 | 0.000 |
| 4 | 2048 | 4 | 8 | random | q_outer | 32 | 8 | 274.76 | 256 | 15298 | 64 | 489536 | 0 | 512 | 60.0 | 62.0 | 0.000 |
| 0 | 2048 | 4 | 8 | random | q_outer | 64 | 16 | 567.78 | 128 | 14352 | 120 | 918528 | 0 | 512 | 113.0 | 116.0 | 0.000 |
| 1 | 2048 | 4 | 8 | random | q_outer | 64 | 16 | 571.67 | 128 | 14391 | 121 | 921024 | 0 | 511 | 112.0 | 117.0 | 0.000 |
| 2 | 2048 | 4 | 8 | random | q_outer | 64 | 16 | 574.41 | 128 | 14387 | 121 | 920768 | 0 | 511 | 113.0 | 116.0 | 0.000 |
| 3 | 2048 | 4 | 8 | random | q_outer | 64 | 16 | 580.13 | 128 | 14401 | 122 | 921664 | 0 | 512 | 113.0 | 117.0 | 0.000 |
| 4 | 2048 | 4 | 8 | random | q_outer | 64 | 16 | 571.94 | 128 | 14434 | 121 | 923776 | 0 | 512 | 113.0 | 117.0 | 0.000 |
| 0 | 2048 | 4 | 8 | random | q_outer | 128 | 32 | 2384.09 | 64 | 12749 | 212 | 1631872 | 0 | 512 | 200.0 | 207.7 | 0.000 |
| 1 | 2048 | 4 | 8 | random | q_outer | 128 | 32 | 2445.82 | 64 | 12842 | 215 | 1643776 | 0 | 511 | 200.0 | 207.0 | 0.000 |
| 2 | 2048 | 4 | 8 | random | q_outer | 128 | 32 | 2404.00 | 64 | 12806 | 214 | 1639168 | 0 | 511 | 200.0 | 206.7 | 0.000 |
| 3 | 2048 | 4 | 8 | random | q_outer | 128 | 32 | 2346.68 | 64 | 12796 | 210 | 1637888 | 0 | 512 | 200.0 | 206.7 | 0.000 |
| 4 | 2048 | 4 | 8 | random | q_outer | 128 | 32 | 2375.99 | 64 | 12806 | 213 | 1639168 | 0 | 512 | 200.0 | 207.7 | 0.000 |
| 0 | 2048 | 4 | 8 | random | signature | 16 | 4 | 215.61 | 2389 | 15769 | 8 | 252304 | 614 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 4 | 8 | random | signature | 16 | 4 | 215.36 | 2393 | 15765 | 8 | 252240 | 611 | 511 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 4 | 8 | random | signature | 16 | 4 | 215.09 | 2404 | 15796 | 8 | 252736 | 636 | 511 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 4 | 8 | random | signature | 16 | 4 | 214.58 | 2406 | 15771 | 8 | 252336 | 638 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 4 | 8 | random | signature | 16 | 4 | 214.60 | 2403 | 15790 | 8 | 252640 | 647 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 4 | 8 | random | signature | 32 | 4 | 217.25 | 2832 | 15284 | 8 | 244544 | 1184 | 512 | 7.0 | 8.0 | 1.000 |
| 1 | 2048 | 4 | 8 | random | signature | 32 | 4 | 217.72 | 2824 | 15305 | 8 | 244880 | 1156 | 511 | 7.0 | 8.0 | 1.000 |
| 2 | 2048 | 4 | 8 | random | signature | 32 | 4 | 218.38 | 2892 | 15268 | 8 | 244288 | 1239 | 511 | 7.0 | 8.0 | 1.000 |
| 3 | 2048 | 4 | 8 | random | signature | 32 | 4 | 217.59 | 2863 | 15286 | 8 | 244576 | 1188 | 512 | 7.0 | 8.0 | 1.000 |
| 4 | 2048 | 4 | 8 | random | signature | 32 | 4 | 216.89 | 2858 | 15298 | 8 | 244768 | 1219 | 512 | 7.0 | 8.0 | 1.000 |
| 0 | 2048 | 4 | 8 | random | signature | 64 | 4 | 217.42 | 3631 | 14352 | 8 | 229632 | 1728 | 512 | 5.0 | 7.0 | 1.000 |
| 1 | 2048 | 4 | 8 | random | signature | 64 | 8 | 292.34 | 3611 | 14391 | 8 | 230272 | 1707 | 511 | 5.0 | 7.0 | 1.000 |
| 2 | 2048 | 4 | 8 | random | signature | 64 | 4 | 216.84 | 3643 | 14387 | 8 | 230192 | 1719 | 511 | 5.0 | 7.0 | 1.000 |
| 3 | 2048 | 4 | 8 | random | signature | 64 | 4 | 218.14 | 3642 | 14401 | 8 | 230416 | 1743 | 512 | 5.0 | 7.0 | 1.000 |
| 4 | 2048 | 4 | 8 | random | signature | 64 | 4 | 219.17 | 3595 | 14434 | 8 | 230944 | 1729 | 512 | 5.0 | 7.0 | 1.000 |
| 0 | 2048 | 4 | 8 | random | signature | 128 | 8 | 282.36 | 4846 | 12749 | 8 | 204032 | 1999 | 512 | 1.0 | 6.0 | 1.000 |
| 1 | 2048 | 4 | 8 | random | signature | 128 | 8 | 284.64 | 4815 | 12842 | 8 | 205584 | 2007 | 511 | 1.0 | 6.0 | 1.000 |
| 2 | 2048 | 4 | 8 | random | signature | 128 | 8 | 283.39 | 4817 | 12806 | 8 | 204992 | 2009 | 511 | 1.0 | 6.0 | 1.000 |
| 3 | 2048 | 4 | 8 | random | signature | 128 | 8 | 286.29 | 4845 | 12796 | 8 | 204752 | 2000 | 512 | 1.0 | 6.0 | 1.000 |
| 4 | 2048 | 4 | 8 | random | signature | 128 | 8 | 286.84 | 4850 | 12806 | 8 | 204976 | 1999 | 512 | 1.0 | 6.0 | 1.000 |
| 0 | 2048 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 407.75 | 1836 | 31025 | 62 | 496400 | 0 | 510 | 16.0 | 16.0 | 0.895 |
| 1 | 2048 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 461.98 | 1812 | 31116 | 58 | 497856 | 0 | 511 | 16.0 | 16.0 | 0.876 |
| 2 | 2048 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 412.63 | 1798 | 31073 | 62 | 497168 | 0 | 511 | 16.0 | 16.3 | 0.863 |
| 3 | 2048 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 428.63 | 1844 | 31223 | 62 | 499568 | 0 | 510 | 16.0 | 16.0 | 0.896 |
| 4 | 2048 | 4 | 16 | clustered | bounded_merge | 16 | 4 | 402.95 | 1840 | 31139 | 62 | 498224 | 0 | 511 | 16.0 | 16.0 | 0.896 |
| 0 | 2048 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 375.53 | 2208 | 29123 | 27 | 465968 | 533 | 510 | 16.0 | 16.0 | 0.947 |
| 1 | 2048 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 360.55 | 2220 | 28937 | 27 | 462992 | 563 | 511 | 16.0 | 16.0 | 0.942 |
| 2 | 2048 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 351.62 | 2212 | 29256 | 27 | 468096 | 549 | 511 | 16.0 | 16.0 | 0.944 |
| 3 | 2048 | 4 | 16 | clustered | bounded_merge | 32 | 4 | 375.37 | 2195 | 29278 | 27 | 468448 | 510 | 510 | 16.0 | 16.0 | 0.947 |
| 4 | 2048 | 4 | 16 | clustered | bounded_merge | 32 | 8 | 533.66 | 2183 | 29167 | 27 | 466704 | 491 | 511 | 16.0 | 16.0 | 0.942 |
| 0 | 2048 | 4 | 16 | clustered | bounded_merge | 64 | 8 | 476.05 | 2638 | 25642 | 27 | 410304 | 1197 | 510 | 10.0 | 16.0 | 0.959 |
| 1 | 2048 | 4 | 16 | clustered | bounded_merge | 64 | 8 | 506.09 | 2619 | 25851 | 27 | 413760 | 1152 | 511 | 10.0 | 16.0 | 0.958 |
| 2 | 2048 | 4 | 16 | clustered | bounded_merge | 64 | 8 | 494.71 | 2619 | 25962 | 27 | 415600 | 1169 | 511 | 10.0 | 16.0 | 0.960 |
| 3 | 2048 | 4 | 16 | clustered | bounded_merge | 64 | 8 | 499.85 | 2583 | 26030 | 27 | 416512 | 1115 | 510 | 11.0 | 16.0 | 0.958 |
| 4 | 2048 | 4 | 16 | clustered | bounded_merge | 64 | 8 | 525.15 | 2568 | 26035 | 27 | 416736 | 1091 | 511 | 11.0 | 16.0 | 0.958 |
| 0 | 2048 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 395.92 | 3092 | 20641 | 16 | 331520 | 1708 | 510 | 5.0 | 16.0 | 1.000 |
| 1 | 2048 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 414.92 | 3131 | 20498 | 16 | 329936 | 1738 | 511 | 5.0 | 15.0 | 1.000 |
| 2 | 2048 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 404.04 | 3121 | 20756 | 16 | 333440 | 1736 | 511 | 6.0 | 15.0 | 1.000 |
| 3 | 2048 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 415.51 | 3091 | 20654 | 16 | 332480 | 1716 | 510 | 6.0 | 15.0 | 1.000 |
| 4 | 2048 | 4 | 16 | clustered | bounded_merge | 128 | 8 | 416.21 | 3073 | 20728 | 16 | 332960 | 1699 | 511 | 5.0 | 16.0 | 1.000 |
| 0 | 2048 | 4 | 16 | clustered | q_outer | 16 | 4 | 359.07 | 512 | 31025 | 64 | 496400 | 0 | 510 | 64.0 | 64.0 | 0.000 |
| 1 | 2048 | 4 | 16 | clustered | q_outer | 16 | 4 | 360.23 | 512 | 31116 | 64 | 497856 | 0 | 511 | 64.0 | 64.0 | 0.000 |
| 2 | 2048 | 4 | 16 | clustered | q_outer | 16 | 4 | 360.25 | 512 | 31073 | 64 | 497168 | 0 | 511 | 64.0 | 64.0 | 0.000 |
| 3 | 2048 | 4 | 16 | clustered | q_outer | 16 | 4 | 358.93 | 512 | 31223 | 64 | 499568 | 0 | 510 | 64.0 | 64.0 | 0.000 |
| 4 | 2048 | 4 | 16 | clustered | q_outer | 16 | 4 | 360.77 | 512 | 31139 | 64 | 498224 | 0 | 511 | 64.0 | 64.0 | 0.000 |
| 0 | 2048 | 4 | 16 | clustered | q_outer | 32 | 8 | 531.10 | 256 | 29123 | 128 | 931936 | 0 | 510 | 115.0 | 127.0 | 0.000 |
| 1 | 2048 | 4 | 16 | clustered | q_outer | 32 | 8 | 533.74 | 256 | 28937 | 128 | 925984 | 0 | 511 | 114.0 | 128.0 | 0.000 |
| 2 | 2048 | 4 | 16 | clustered | q_outer | 32 | 8 | 531.51 | 256 | 29256 | 128 | 936192 | 0 | 511 | 116.0 | 128.0 | 0.000 |
| 3 | 2048 | 4 | 16 | clustered | q_outer | 32 | 8 | 534.10 | 256 | 29278 | 128 | 936896 | 0 | 510 | 115.0 | 128.0 | 0.000 |
| 4 | 2048 | 4 | 16 | clustered | q_outer | 32 | 8 | 536.99 | 256 | 29167 | 128 | 933344 | 0 | 511 | 115.0 | 128.0 | 0.000 |
| 0 | 2048 | 4 | 16 | clustered | q_outer | 64 | 16 | 1110.20 | 128 | 25642 | 238 | 1641088 | 0 | 510 | 200.0 | 225.3 | 0.000 |
| 1 | 2048 | 4 | 16 | clustered | q_outer | 64 | 16 | 1121.46 | 128 | 25851 | 240 | 1654464 | 0 | 511 | 201.0 | 221.0 | 0.000 |
| 2 | 2048 | 4 | 16 | clustered | q_outer | 64 | 16 | 1147.61 | 128 | 25962 | 245 | 1661568 | 0 | 511 | 205.0 | 225.6 | 0.000 |
| 3 | 2048 | 4 | 16 | clustered | q_outer | 64 | 16 | 1134.64 | 128 | 26030 | 245 | 1665920 | 0 | 510 | 204.5 | 225.3 | 0.000 |
| 4 | 2048 | 4 | 16 | clustered | q_outer | 64 | 16 | 1153.93 | 128 | 26035 | 249 | 1666240 | 0 | 511 | 203.5 | 224.6 | 0.000 |
| 0 | 2048 | 4 | 16 | clustered | q_outer | 128 | 32 | 4152.47 | 64 | 20641 | 377 | 2642048 | 0 | 510 | 324.5 | 352.1 | 0.000 |
| 1 | 2048 | 4 | 16 | clustered | q_outer | 128 | 32 | 4008.68 | 64 | 20498 | 365 | 2623744 | 0 | 511 | 321.0 | 346.4 | 0.000 |
| 2 | 2048 | 4 | 16 | clustered | q_outer | 128 | 32 | 4194.44 | 64 | 20756 | 374 | 2656768 | 0 | 511 | 325.0 | 350.1 | 0.000 |
| 3 | 2048 | 4 | 16 | clustered | q_outer | 128 | 32 | 3979.80 | 64 | 20654 | 357 | 2643712 | 0 | 510 | 322.0 | 348.7 | 0.000 |
| 4 | 2048 | 4 | 16 | clustered | q_outer | 128 | 32 | 4007.40 | 64 | 20728 | 367 | 2653184 | 0 | 511 | 327.0 | 352.7 | 0.000 |
| 0 | 2048 | 4 | 16 | clustered | signature | 16 | 4 | 394.62 | 2212 | 31025 | 16 | 496400 | 337 | 510 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 4 | 16 | clustered | signature | 16 | 4 | 396.20 | 2218 | 31116 | 16 | 497856 | 339 | 511 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 4 | 16 | clustered | signature | 16 | 4 | 395.01 | 2223 | 31073 | 16 | 497168 | 359 | 511 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 4 | 16 | clustered | signature | 16 | 4 | 398.56 | 2205 | 31223 | 16 | 499568 | 315 | 510 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 4 | 16 | clustered | signature | 16 | 4 | 397.29 | 2196 | 31139 | 16 | 498224 | 302 | 511 | 16.0 | 16.0 | 1.000 |
| 0 | 2048 | 4 | 16 | clustered | signature | 32 | 4 | 376.49 | 2422 | 29123 | 16 | 465968 | 722 | 510 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 4 | 16 | clustered | signature | 32 | 4 | 376.28 | 2439 | 28937 | 16 | 462992 | 749 | 511 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 4 | 16 | clustered | signature | 32 | 4 | 377.79 | 2424 | 29256 | 16 | 468096 | 727 | 511 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 4 | 16 | clustered | signature | 32 | 4 | 380.42 | 2397 | 29278 | 16 | 468448 | 685 | 510 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 4 | 16 | clustered | signature | 32 | 8 | 513.20 | 2396 | 29167 | 16 | 466704 | 678 | 511 | 16.0 | 16.0 | 1.000 |
| 0 | 2048 | 4 | 16 | clustered | signature | 64 | 8 | 464.05 | 2749 | 25642 | 16 | 410304 | 1259 | 510 | 9.0 | 16.0 | 1.000 |
| 1 | 2048 | 4 | 16 | clustered | signature | 64 | 8 | 483.06 | 2727 | 25851 | 16 | 413760 | 1211 | 511 | 9.0 | 16.0 | 1.000 |
| 2 | 2048 | 4 | 16 | clustered | signature | 64 | 8 | 468.91 | 2732 | 25962 | 16 | 415600 | 1231 | 511 | 9.0 | 16.0 | 1.000 |
| 3 | 2048 | 4 | 16 | clustered | signature | 64 | 8 | 471.27 | 2698 | 26030 | 16 | 416512 | 1187 | 510 | 10.0 | 16.0 | 1.000 |
| 4 | 2048 | 4 | 16 | clustered | signature | 64 | 8 | 474.71 | 2679 | 26035 | 16 | 416736 | 1163 | 511 | 10.0 | 16.0 | 1.000 |
| 0 | 2048 | 4 | 16 | clustered | signature | 128 | 8 | 396.07 | 3092 | 20641 | 16 | 331520 | 1708 | 510 | 5.0 | 16.0 | 1.000 |
| 1 | 2048 | 4 | 16 | clustered | signature | 128 | 8 | 415.14 | 3131 | 20498 | 16 | 329936 | 1738 | 511 | 5.0 | 15.0 | 1.000 |
| 2 | 2048 | 4 | 16 | clustered | signature | 128 | 8 | 403.74 | 3121 | 20756 | 16 | 333440 | 1736 | 511 | 6.0 | 15.0 | 1.000 |
| 3 | 2048 | 4 | 16 | clustered | signature | 128 | 8 | 414.42 | 3091 | 20654 | 16 | 332480 | 1716 | 510 | 6.0 | 15.0 | 1.000 |
| 4 | 2048 | 4 | 16 | clustered | signature | 128 | 8 | 402.43 | 3073 | 20728 | 16 | 332960 | 1699 | 511 | 5.0 | 16.0 | 1.000 |
| 0 | 2048 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 221.56 | 512 | 20235 | 40 | 323760 | 0 | 512 | 40.0 | 40.0 | 0.202 |
| 1 | 2048 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 222.35 | 512 | 20236 | 40 | 323776 | 0 | 512 | 40.0 | 40.0 | 0.202 |
| 2 | 2048 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 222.19 | 512 | 20235 | 40 | 323760 | 0 | 512 | 40.0 | 40.0 | 0.202 |
| 3 | 2048 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 222.54 | 512 | 20239 | 40 | 323824 | 0 | 512 | 40.0 | 40.0 | 0.202 |
| 4 | 2048 | 4 | 16 | common_private | bounded_merge | 16 | 4 | 222.14 | 512 | 20208 | 40 | 323328 | 0 | 512 | 40.0 | 40.0 | 0.201 |
| 0 | 2048 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 329.71 | 2304 | 18187 | 8 | 323760 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 330.62 | 2304 | 18188 | 8 | 323776 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 329.69 | 2304 | 18187 | 8 | 323760 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 329.62 | 2304 | 18191 | 8 | 323824 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 4 | 16 | common_private | bounded_merge | 32 | 8 | 330.83 | 2304 | 18176 | 8 | 323328 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 691.86 | 2176 | 17163 | 8 | 323760 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 692.41 | 2176 | 17164 | 8 | 323776 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 692.17 | 2176 | 17163 | 8 | 323760 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 693.17 | 2176 | 17167 | 8 | 323824 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 4 | 16 | common_private | bounded_merge | 64 | 16 | 692.64 | 2176 | 17160 | 8 | 323328 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 1764.87 | 2112 | 16651 | 8 | 323760 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 1752.27 | 2112 | 16652 | 8 | 323776 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 1755.12 | 2112 | 16651 | 8 | 323760 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 1765.21 | 2112 | 16655 | 8 | 323824 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 4 | 16 | common_private | bounded_merge | 128 | 32 | 1761.27 | 2112 | 16652 | 8 | 323328 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 4 | 16 | common_private | q_outer | 16 | 4 | 228.58 | 512 | 20235 | 40 | 323760 | 0 | 512 | 40.0 | 40.0 | 0.202 |
| 1 | 2048 | 4 | 16 | common_private | q_outer | 16 | 4 | 229.44 | 512 | 20236 | 40 | 323776 | 0 | 512 | 40.0 | 40.0 | 0.202 |
| 2 | 2048 | 4 | 16 | common_private | q_outer | 16 | 4 | 228.22 | 512 | 20235 | 40 | 323760 | 0 | 512 | 40.0 | 40.0 | 0.202 |
| 3 | 2048 | 4 | 16 | common_private | q_outer | 16 | 4 | 227.29 | 512 | 20239 | 40 | 323824 | 0 | 512 | 40.0 | 40.0 | 0.202 |
| 4 | 2048 | 4 | 16 | common_private | q_outer | 16 | 4 | 228.59 | 512 | 20208 | 40 | 323328 | 0 | 512 | 40.0 | 40.0 | 0.201 |
| 0 | 2048 | 4 | 16 | common_private | q_outer | 32 | 8 | 309.03 | 256 | 18187 | 72 | 581984 | 0 | 512 | 71.0 | 72.0 | 0.113 |
| 1 | 2048 | 4 | 16 | common_private | q_outer | 32 | 8 | 306.88 | 256 | 18188 | 72 | 582016 | 0 | 512 | 71.0 | 72.0 | 0.113 |
| 2 | 2048 | 4 | 16 | common_private | q_outer | 32 | 8 | 306.67 | 256 | 18187 | 72 | 581984 | 0 | 512 | 71.0 | 72.0 | 0.113 |
| 3 | 2048 | 4 | 16 | common_private | q_outer | 32 | 8 | 306.12 | 256 | 18191 | 72 | 582112 | 0 | 512 | 71.0 | 72.0 | 0.113 |
| 4 | 2048 | 4 | 16 | common_private | q_outer | 32 | 8 | 309.61 | 256 | 18176 | 72 | 581632 | 0 | 512 | 71.0 | 72.0 | 0.112 |
| 0 | 2048 | 4 | 16 | common_private | q_outer | 64 | 16 | 640.61 | 128 | 17163 | 136 | 1098432 | 0 | 512 | 134.0 | 136.0 | 0.060 |
| 1 | 2048 | 4 | 16 | common_private | q_outer | 64 | 16 | 644.21 | 128 | 17164 | 136 | 1098496 | 0 | 512 | 134.0 | 136.0 | 0.060 |
| 2 | 2048 | 4 | 16 | common_private | q_outer | 64 | 16 | 643.00 | 128 | 17163 | 136 | 1098432 | 0 | 512 | 134.0 | 136.0 | 0.060 |
| 3 | 2048 | 4 | 16 | common_private | q_outer | 64 | 16 | 645.69 | 128 | 17167 | 136 | 1098688 | 0 | 512 | 134.0 | 136.0 | 0.060 |
| 4 | 2048 | 4 | 16 | common_private | q_outer | 64 | 16 | 647.58 | 128 | 17160 | 136 | 1098240 | 0 | 512 | 134.0 | 136.0 | 0.059 |
| 0 | 2048 | 4 | 16 | common_private | q_outer | 128 | 32 | 2978.94 | 64 | 16651 | 264 | 2131328 | 0 | 512 | 260.0 | 264.0 | 0.031 |
| 1 | 2048 | 4 | 16 | common_private | q_outer | 128 | 32 | 2972.14 | 64 | 16652 | 264 | 2131456 | 0 | 512 | 260.0 | 263.0 | 0.031 |
| 2 | 2048 | 4 | 16 | common_private | q_outer | 128 | 32 | 2976.34 | 64 | 16651 | 264 | 2131328 | 0 | 512 | 260.5 | 264.0 | 0.031 |
| 3 | 2048 | 4 | 16 | common_private | q_outer | 128 | 32 | 2980.55 | 64 | 16655 | 264 | 2131840 | 0 | 512 | 260.0 | 263.0 | 0.031 |
| 4 | 2048 | 4 | 16 | common_private | q_outer | 128 | 32 | 2979.46 | 64 | 16652 | 264 | 2131456 | 0 | 512 | 260.0 | 263.0 | 0.031 |
| 0 | 2048 | 4 | 16 | common_private | signature | 16 | 4 | 274.56 | 2560 | 20235 | 8 | 323760 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 4 | 16 | common_private | signature | 16 | 4 | 274.58 | 2560 | 20236 | 8 | 323776 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 4 | 16 | common_private | signature | 16 | 4 | 274.67 | 2560 | 20235 | 8 | 323760 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 4 | 16 | common_private | signature | 16 | 4 | 275.19 | 2560 | 20239 | 8 | 323824 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 4 | 16 | common_private | signature | 16 | 4 | 274.37 | 2560 | 20208 | 8 | 323328 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 4 | 16 | common_private | signature | 32 | 8 | 330.08 | 2304 | 18187 | 8 | 323760 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 4 | 16 | common_private | signature | 32 | 8 | 330.47 | 2304 | 18188 | 8 | 323776 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 4 | 16 | common_private | signature | 32 | 8 | 330.06 | 2304 | 18187 | 8 | 323760 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 4 | 16 | common_private | signature | 32 | 8 | 330.30 | 2304 | 18191 | 8 | 323824 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 4 | 16 | common_private | signature | 32 | 8 | 330.39 | 2304 | 18176 | 8 | 323328 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 4 | 16 | common_private | signature | 64 | 16 | 691.77 | 2176 | 17163 | 8 | 323760 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 4 | 16 | common_private | signature | 64 | 16 | 695.10 | 2176 | 17164 | 8 | 323776 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 4 | 16 | common_private | signature | 64 | 16 | 692.03 | 2176 | 17163 | 8 | 323760 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 4 | 16 | common_private | signature | 64 | 16 | 693.65 | 2176 | 17167 | 8 | 323824 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 4 | 16 | common_private | signature | 64 | 16 | 693.17 | 2176 | 17160 | 8 | 323328 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 4 | 16 | common_private | signature | 128 | 32 | 1763.68 | 2112 | 16651 | 8 | 323760 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 4 | 16 | common_private | signature | 128 | 32 | 1756.31 | 2112 | 16652 | 8 | 323776 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 4 | 16 | common_private | signature | 128 | 32 | 1753.12 | 2112 | 16651 | 8 | 323760 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 4 | 16 | common_private | signature | 128 | 32 | 1766.14 | 2112 | 16655 | 8 | 323824 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 4 | 16 | common_private | signature | 128 | 32 | 1762.21 | 2112 | 16652 | 8 | 323328 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 4 | 16 | random | bounded_merge | 16 | 4 | 407.34 | 763 | 30783 | 62 | 492528 | 0 | 512 | 46.0 | 61.0 | 0.135 |
| 1 | 2048 | 4 | 16 | random | bounded_merge | 16 | 4 | 409.86 | 779 | 30855 | 62 | 493680 | 0 | 512 | 45.0 | 61.0 | 0.142 |
| 2 | 2048 | 4 | 16 | random | bounded_merge | 16 | 4 | 412.82 | 744 | 30827 | 62 | 493232 | 0 | 512 | 56.0 | 61.0 | 0.124 |
| 3 | 2048 | 4 | 16 | random | bounded_merge | 16 | 4 | 412.26 | 770 | 30840 | 62 | 493440 | 0 | 512 | 46.0 | 61.0 | 0.138 |
| 4 | 2048 | 4 | 16 | random | bounded_merge | 16 | 4 | 414.84 | 792 | 30772 | 62 | 492352 | 0 | 512 | 31.0 | 61.0 | 0.153 |
| 0 | 2048 | 4 | 16 | random | bounded_merge | 32 | 4 | 380.53 | 4417 | 28927 | 16 | 462832 | 1952 | 512 | 2.0 | 14.0 | 0.894 |
| 1 | 2048 | 4 | 16 | random | bounded_merge | 32 | 4 | 373.24 | 4438 | 28953 | 16 | 463248 | 1972 | 512 | 2.0 | 14.0 | 0.893 |
| 2 | 2048 | 4 | 16 | random | bounded_merge | 32 | 4 | 374.89 | 4448 | 28981 | 16 | 463696 | 1943 | 512 | 2.0 | 14.0 | 0.892 |
| 3 | 2048 | 4 | 16 | random | bounded_merge | 32 | 4 | 376.68 | 4398 | 29020 | 16 | 464320 | 1943 | 512 | 2.0 | 14.0 | 0.892 |
| 4 | 2048 | 4 | 16 | random | bounded_merge | 32 | 4 | 375.90 | 4423 | 28910 | 16 | 462560 | 1957 | 512 | 2.0 | 14.0 | 0.894 |
| 0 | 2048 | 4 | 16 | random | bounded_merge | 64 | 8 | 539.42 | 7081 | 25699 | 16 | 411344 | 2046 | 512 | 1.0 | 11.0 | 1.000 |
| 1 | 2048 | 4 | 16 | random | bounded_merge | 64 | 8 | 528.12 | 7079 | 25686 | 15 | 411152 | 2047 | 512 | 1.0 | 11.0 | 1.000 |
| 2 | 2048 | 4 | 16 | random | bounded_merge | 64 | 8 | 525.54 | 7112 | 25685 | 15 | 411088 | 2048 | 512 | 1.0 | 11.0 | 1.000 |
| 3 | 2048 | 4 | 16 | random | bounded_merge | 64 | 8 | 541.76 | 7121 | 25679 | 16 | 410896 | 2046 | 512 | 1.0 | 11.0 | 1.000 |
| 4 | 2048 | 4 | 16 | random | bounded_merge | 64 | 8 | 536.70 | 7022 | 25730 | 15 | 411776 | 2048 | 512 | 1.0 | 11.0 | 1.000 |
| 0 | 2048 | 4 | 16 | random | bounded_merge | 128 | 8 | 498.95 | 10103 | 20604 | 12 | 331296 | 2048 | 512 | 1.0 | 6.0 | 1.000 |
| 1 | 2048 | 4 | 16 | random | bounded_merge | 128 | 8 | 491.50 | 10141 | 20536 | 12 | 330240 | 2048 | 512 | 1.0 | 6.0 | 1.000 |
| 2 | 2048 | 4 | 16 | random | bounded_merge | 128 | 8 | 491.00 | 10058 | 20634 | 12 | 331760 | 2048 | 512 | 1.0 | 6.0 | 1.000 |
| 3 | 2048 | 4 | 16 | random | bounded_merge | 128 | 8 | 495.99 | 10077 | 20611 | 14 | 331488 | 2048 | 512 | 1.0 | 6.0 | 1.000 |
| 4 | 2048 | 4 | 16 | random | bounded_merge | 128 | 8 | 491.50 | 10064 | 20543 | 12 | 330096 | 2048 | 512 | 1.0 | 6.0 | 1.000 |
| 0 | 2048 | 4 | 16 | random | q_outer | 16 | 4 | 344.44 | 512 | 30783 | 64 | 492528 | 0 | 512 | 60.0 | 62.9 | 0.000 |
| 1 | 2048 | 4 | 16 | random | q_outer | 16 | 4 | 344.33 | 512 | 30855 | 64 | 493680 | 0 | 512 | 60.0 | 63.0 | 0.000 |
| 2 | 2048 | 4 | 16 | random | q_outer | 16 | 4 | 345.48 | 512 | 30827 | 64 | 493232 | 0 | 512 | 60.0 | 62.0 | 0.000 |
| 3 | 2048 | 4 | 16 | random | q_outer | 16 | 4 | 345.91 | 512 | 30840 | 64 | 493440 | 0 | 512 | 60.0 | 62.0 | 0.000 |
| 4 | 2048 | 4 | 16 | random | q_outer | 16 | 4 | 343.92 | 512 | 30772 | 64 | 492352 | 0 | 512 | 60.0 | 63.0 | 0.000 |
| 0 | 2048 | 4 | 16 | random | q_outer | 32 | 8 | 498.43 | 256 | 28927 | 121 | 925664 | 0 | 512 | 114.0 | 118.0 | 0.000 |
| 1 | 2048 | 4 | 16 | random | q_outer | 32 | 8 | 502.66 | 256 | 28953 | 121 | 926496 | 0 | 512 | 113.0 | 117.0 | 0.000 |
| 2 | 2048 | 4 | 16 | random | q_outer | 32 | 8 | 509.87 | 256 | 28981 | 122 | 927392 | 0 | 512 | 113.0 | 118.0 | 0.000 |
| 3 | 2048 | 4 | 16 | random | q_outer | 32 | 8 | 494.29 | 256 | 29020 | 120 | 928640 | 0 | 512 | 114.0 | 117.0 | 0.000 |
| 4 | 2048 | 4 | 16 | random | q_outer | 32 | 8 | 500.79 | 256 | 28910 | 122 | 925120 | 0 | 512 | 113.0 | 117.0 | 0.000 |
| 0 | 2048 | 4 | 16 | random | q_outer | 64 | 16 | 996.23 | 128 | 25699 | 214 | 1644736 | 0 | 512 | 201.0 | 208.0 | 0.000 |
| 1 | 2048 | 4 | 16 | random | q_outer | 64 | 16 | 991.88 | 128 | 25686 | 214 | 1643904 | 0 | 512 | 201.0 | 208.3 | 0.000 |
| 2 | 2048 | 4 | 16 | random | q_outer | 64 | 16 | 1014.94 | 128 | 25685 | 217 | 1643840 | 0 | 512 | 201.0 | 207.3 | 0.000 |
| 3 | 2048 | 4 | 16 | random | q_outer | 64 | 16 | 994.71 | 128 | 25679 | 215 | 1643456 | 0 | 512 | 200.0 | 208.0 | 0.000 |
| 4 | 2048 | 4 | 16 | random | q_outer | 64 | 16 | 1012.45 | 128 | 25730 | 216 | 1646720 | 0 | 512 | 201.0 | 207.3 | 0.000 |
| 0 | 2048 | 4 | 16 | random | q_outer | 128 | 32 | 3778.70 | 64 | 20604 | 347 | 2637312 | 0 | 512 | 322.5 | 334.1 | 0.000 |
| 1 | 2048 | 4 | 16 | random | q_outer | 128 | 32 | 3776.36 | 64 | 20536 | 339 | 2628608 | 0 | 512 | 321.5 | 329.7 | 0.000 |
| 2 | 2048 | 4 | 16 | random | q_outer | 128 | 32 | 3812.70 | 64 | 20634 | 343 | 2641152 | 0 | 512 | 323.0 | 331.0 | 0.000 |
| 3 | 2048 | 4 | 16 | random | q_outer | 128 | 32 | 3798.56 | 64 | 20611 | 344 | 2638208 | 0 | 512 | 322.0 | 332.4 | 0.000 |
| 4 | 2048 | 4 | 16 | random | q_outer | 128 | 32 | 3702.95 | 64 | 20543 | 336 | 2629504 | 0 | 512 | 321.0 | 328.0 | 0.000 |
| 0 | 2048 | 4 | 16 | random | signature | 16 | 4 | 399.27 | 3224 | 30783 | 16 | 492528 | 1579 | 512 | 13.0 | 16.0 | 1.000 |
| 1 | 2048 | 4 | 16 | random | signature | 16 | 4 | 400.18 | 3201 | 30855 | 16 | 493680 | 1581 | 512 | 14.0 | 16.0 | 1.000 |
| 2 | 2048 | 4 | 16 | random | signature | 16 | 4 | 398.52 | 3254 | 30827 | 16 | 493232 | 1613 | 512 | 13.0 | 16.0 | 1.000 |
| 3 | 2048 | 4 | 16 | random | signature | 16 | 4 | 401.41 | 3214 | 30840 | 16 | 493440 | 1574 | 512 | 13.0 | 16.0 | 1.000 |
| 4 | 2048 | 4 | 16 | random | signature | 16 | 4 | 398.19 | 3238 | 30772 | 16 | 492352 | 1582 | 512 | 13.0 | 16.0 | 1.000 |
| 0 | 2048 | 4 | 16 | random | signature | 32 | 4 | 396.99 | 4673 | 28927 | 16 | 462832 | 1982 | 512 | 2.0 | 14.0 | 1.000 |
| 1 | 2048 | 4 | 16 | random | signature | 32 | 4 | 398.52 | 4694 | 28953 | 16 | 463248 | 2003 | 512 | 2.0 | 14.0 | 1.000 |
| 2 | 2048 | 4 | 16 | random | signature | 32 | 4 | 396.44 | 4704 | 28981 | 16 | 463696 | 1986 | 512 | 2.0 | 14.0 | 1.000 |
| 3 | 2048 | 4 | 16 | random | signature | 32 | 4 | 397.53 | 4654 | 29020 | 16 | 464320 | 1980 | 512 | 2.0 | 14.0 | 1.000 |
| 4 | 2048 | 4 | 16 | random | signature | 32 | 4 | 396.35 | 4679 | 28910 | 16 | 462560 | 1994 | 512 | 2.0 | 14.0 | 1.000 |
| 0 | 2048 | 4 | 16 | random | signature | 64 | 8 | 524.10 | 7081 | 25699 | 16 | 411344 | 2046 | 512 | 1.0 | 11.0 | 1.000 |
| 1 | 2048 | 4 | 16 | random | signature | 64 | 8 | 523.73 | 7079 | 25686 | 15 | 411152 | 2047 | 512 | 1.0 | 11.0 | 1.000 |
| 2 | 2048 | 4 | 16 | random | signature | 64 | 8 | 518.17 | 7112 | 25685 | 15 | 411088 | 2048 | 512 | 1.0 | 11.0 | 1.000 |
| 3 | 2048 | 4 | 16 | random | signature | 64 | 8 | 526.61 | 7121 | 25679 | 16 | 410896 | 2046 | 512 | 1.0 | 11.0 | 1.000 |
| 4 | 2048 | 4 | 16 | random | signature | 64 | 8 | 521.26 | 7022 | 25730 | 15 | 411776 | 2048 | 512 | 1.0 | 11.0 | 1.000 |
| 0 | 2048 | 4 | 16 | random | signature | 128 | 8 | 495.45 | 10103 | 20604 | 12 | 331296 | 2048 | 512 | 1.0 | 6.0 | 1.000 |
| 1 | 2048 | 4 | 16 | random | signature | 128 | 8 | 490.90 | 10141 | 20536 | 12 | 330240 | 2048 | 512 | 1.0 | 6.0 | 1.000 |
| 2 | 2048 | 4 | 16 | random | signature | 128 | 8 | 490.30 | 10058 | 20634 | 12 | 331760 | 2048 | 512 | 1.0 | 6.0 | 1.000 |
| 3 | 2048 | 4 | 16 | random | signature | 128 | 8 | 493.19 | 10077 | 20611 | 14 | 331488 | 2048 | 512 | 1.0 | 6.0 | 1.000 |
| 4 | 2048 | 4 | 16 | random | signature | 128 | 8 | 491.61 | 10064 | 20543 | 12 | 330096 | 2048 | 512 | 1.0 | 6.0 | 1.000 |
| 0 | 2048 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 807.98 | 1575 | 59221 | 127 | 947536 | 0 | 510 | 32.0 | 54.0 | 0.743 |
| 1 | 2048 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 841.15 | 1542 | 59377 | 127 | 950032 | 0 | 511 | 32.0 | 57.0 | 0.716 |
| 2 | 2048 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 840.50 | 1544 | 59168 | 127 | 946688 | 0 | 511 | 32.0 | 57.0 | 0.716 |
| 3 | 2048 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 827.17 | 1574 | 59630 | 127 | 954080 | 0 | 510 | 32.0 | 56.0 | 0.735 |
| 4 | 2048 | 4 | 32 | clustered | bounded_merge | 16 | 4 | 799.75 | 1607 | 59588 | 127 | 953408 | 0 | 511 | 32.0 | 53.0 | 0.761 |
| 0 | 2048 | 4 | 32 | clustered | bounded_merge | 32 | 8 | 948.38 | 2513 | 52087 | 54 | 833648 | 1088 | 510 | 23.0 | 32.0 | 0.931 |
| 1 | 2048 | 4 | 32 | clustered | bounded_merge | 32 | 8 | 951.63 | 2522 | 51913 | 32 | 830944 | 1073 | 511 | 24.0 | 32.0 | 0.940 |
| 2 | 2048 | 4 | 32 | clustered | bounded_merge | 32 | 8 | 945.73 | 2522 | 52201 | 57 | 835232 | 1103 | 511 | 23.0 | 32.0 | 0.933 |
| 3 | 2048 | 4 | 32 | clustered | bounded_merge | 32 | 4 | 641.18 | 2484 | 52683 | 41 | 842928 | 1038 | 510 | 25.0 | 32.0 | 0.932 |
| 4 | 2048 | 4 | 32 | clustered | bounded_merge | 32 | 8 | 992.80 | 2476 | 52419 | 54 | 839168 | 1027 | 511 | 24.0 | 32.0 | 0.931 |
| 0 | 2048 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 863.87 | 3104 | 41003 | 57 | 658944 | 1732 | 510 | 11.0 | 31.0 | 0.953 |
| 1 | 2048 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 800.69 | 3104 | 41371 | 57 | 664704 | 1694 | 511 | 11.0 | 32.0 | 0.960 |
| 2 | 2048 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 794.51 | 3104 | 41239 | 57 | 661872 | 1723 | 511 | 11.0 | 31.7 | 0.959 |
| 3 | 2048 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 821.97 | 3089 | 41867 | 57 | 672128 | 1706 | 510 | 11.0 | 32.0 | 0.951 |
| 4 | 2048 | 4 | 32 | clustered | bounded_merge | 64 | 8 | 863.03 | 3044 | 42088 | 57 | 676672 | 1664 | 511 | 11.0 | 32.0 | 0.956 |
| 0 | 2048 | 4 | 32 | clustered | bounded_merge | 128 | 16 | 1207.31 | 3563 | 27935 | 32 | 475312 | 2008 | 510 | 6.0 | 18.0 | 1.000 |
| 1 | 2048 | 4 | 32 | clustered | bounded_merge | 128 | 8 | 566.51 | 3594 | 27580 | 32 | 470912 | 2013 | 511 | 6.0 | 17.0 | 1.000 |
| 2 | 2048 | 4 | 32 | clustered | bounded_merge | 128 | 16 | 1216.24 | 3567 | 27855 | 32 | 472480 | 2001 | 511 | 6.0 | 17.0 | 1.000 |
| 3 | 2048 | 4 | 32 | clustered | bounded_merge | 128 | 16 | 1209.77 | 3574 | 27962 | 32 | 475968 | 2010 | 510 | 6.0 | 17.0 | 1.000 |
| 4 | 2048 | 4 | 32 | clustered | bounded_merge | 128 | 16 | 1233.78 | 3557 | 28042 | 32 | 475216 | 2013 | 511 | 6.0 | 17.0 | 1.000 |
| 0 | 2048 | 4 | 32 | clustered | q_outer | 16 | 4 | 696.95 | 512 | 59221 | 128 | 947536 | 0 | 510 | 122.0 | 128.0 | 0.000 |
| 1 | 2048 | 4 | 32 | clustered | q_outer | 16 | 4 | 703.96 | 512 | 59377 | 128 | 950032 | 0 | 511 | 122.0 | 128.0 | 0.000 |
| 2 | 2048 | 4 | 32 | clustered | q_outer | 16 | 4 | 693.55 | 512 | 59168 | 128 | 946688 | 0 | 511 | 120.0 | 128.0 | 0.000 |
| 3 | 2048 | 4 | 32 | clustered | q_outer | 16 | 4 | 694.97 | 512 | 59630 | 128 | 954080 | 0 | 510 | 122.0 | 128.0 | 0.000 |
| 4 | 2048 | 4 | 32 | clustered | q_outer | 16 | 4 | 700.62 | 512 | 59588 | 128 | 953408 | 0 | 511 | 122.0 | 128.0 | 0.000 |
| 0 | 2048 | 4 | 32 | clustered | q_outer | 32 | 8 | 1020.92 | 256 | 52087 | 256 | 1666784 | 0 | 510 | 206.0 | 232.0 | 0.000 |
| 1 | 2048 | 4 | 32 | clustered | q_outer | 32 | 8 | 983.77 | 256 | 51913 | 256 | 1661216 | 0 | 511 | 205.0 | 231.0 | 0.000 |
| 2 | 2048 | 4 | 32 | clustered | q_outer | 32 | 8 | 1024.45 | 256 | 52201 | 256 | 1670432 | 0 | 511 | 204.0 | 236.5 | 0.000 |
| 3 | 2048 | 4 | 32 | clustered | q_outer | 32 | 8 | 1040.79 | 256 | 52683 | 253 | 1685856 | 0 | 510 | 208.0 | 234.5 | 0.000 |
| 4 | 2048 | 4 | 32 | clustered | q_outer | 32 | 8 | 1017.09 | 256 | 52419 | 256 | 1677408 | 0 | 511 | 207.5 | 232.0 | 0.000 |
| 0 | 2048 | 4 | 32 | clustered | q_outer | 64 | 16 | 1747.68 | 128 | 41003 | 381 | 2624192 | 0 | 510 | 320.0 | 363.9 | 0.000 |
| 1 | 2048 | 4 | 32 | clustered | q_outer | 64 | 16 | 1760.82 | 128 | 41371 | 384 | 2647744 | 0 | 511 | 322.0 | 358.0 | 0.000 |
| 2 | 2048 | 4 | 32 | clustered | q_outer | 64 | 16 | 1886.36 | 128 | 41239 | 412 | 2639296 | 0 | 511 | 324.0 | 360.3 | 0.000 |
| 3 | 2048 | 4 | 32 | clustered | q_outer | 64 | 16 | 1821.06 | 128 | 41867 | 396 | 2679488 | 0 | 510 | 329.5 | 369.2 | 0.000 |
| 4 | 2048 | 4 | 32 | clustered | q_outer | 64 | 16 | 1863.43 | 128 | 42088 | 396 | 2693632 | 0 | 511 | 335.5 | 367.3 | 0.000 |
| 0 | 2048 | 4 | 32 | clustered | q_outer | 128 | 32 | 5406.49 | 64 | 27935 | 492 | 3575680 | 0 | 510 | 435.5 | 470.0 | 0.000 |
| 1 | 2048 | 4 | 32 | clustered | q_outer | 128 | 32 | 5169.14 | 64 | 27580 | 473 | 3530240 | 0 | 511 | 432.0 | 458.7 | 0.000 |
| 2 | 2048 | 4 | 32 | clustered | q_outer | 128 | 32 | 5250.07 | 64 | 27855 | 478 | 3565440 | 0 | 511 | 435.5 | 463.0 | 0.000 |
| 3 | 2048 | 4 | 32 | clustered | q_outer | 128 | 32 | 5345.72 | 64 | 27962 | 486 | 3579136 | 0 | 510 | 442.0 | 471.7 | 0.000 |
| 4 | 2048 | 4 | 32 | clustered | q_outer | 128 | 32 | 5378.85 | 64 | 28042 | 485 | 3589376 | 0 | 511 | 447.0 | 468.8 | 0.000 |
| 0 | 2048 | 4 | 32 | clustered | signature | 16 | 4 | 729.66 | 2388 | 59221 | 32 | 947536 | 658 | 510 | 32.0 | 32.0 | 1.000 |
| 1 | 2048 | 4 | 32 | clustered | signature | 16 | 4 | 734.62 | 2418 | 59377 | 32 | 950032 | 697 | 511 | 32.0 | 32.0 | 1.000 |
| 2 | 2048 | 4 | 32 | clustered | signature | 16 | 4 | 729.54 | 2416 | 59168 | 32 | 946688 | 714 | 511 | 32.0 | 32.0 | 1.000 |
| 3 | 2048 | 4 | 32 | clustered | signature | 16 | 4 | 739.32 | 2387 | 59630 | 32 | 954080 | 656 | 510 | 32.0 | 32.0 | 1.000 |
| 4 | 2048 | 4 | 32 | clustered | signature | 16 | 4 | 738.39 | 2379 | 59588 | 32 | 953408 | 633 | 511 | 32.0 | 32.0 | 1.000 |
| 0 | 2048 | 4 | 32 | clustered | signature | 32 | 8 | 892.61 | 2765 | 52087 | 32 | 833648 | 1250 | 510 | 19.0 | 32.0 | 1.000 |
| 1 | 2048 | 4 | 32 | clustered | signature | 32 | 8 | 893.18 | 2772 | 51913 | 32 | 830944 | 1237 | 511 | 19.0 | 32.0 | 1.000 |
| 2 | 2048 | 4 | 32 | clustered | signature | 32 | 8 | 897.62 | 2771 | 52201 | 32 | 835232 | 1254 | 511 | 19.0 | 32.0 | 1.000 |
| 3 | 2048 | 4 | 32 | clustered | signature | 32 | 4 | 670.36 | 2735 | 52683 | 32 | 842928 | 1200 | 510 | 20.0 | 32.0 | 1.000 |
| 4 | 2048 | 4 | 32 | clustered | signature | 32 | 8 | 902.58 | 2730 | 52419 | 32 | 839168 | 1215 | 511 | 19.0 | 32.0 | 1.000 |
| 0 | 2048 | 4 | 32 | clustered | signature | 64 | 8 | 771.20 | 3231 | 41003 | 32 | 658944 | 1775 | 510 | 10.0 | 29.0 | 1.000 |
| 1 | 2048 | 4 | 32 | clustered | signature | 64 | 8 | 751.48 | 3226 | 41371 | 32 | 664704 | 1745 | 511 | 10.0 | 30.0 | 1.000 |
| 2 | 2048 | 4 | 32 | clustered | signature | 64 | 8 | 771.74 | 3229 | 41239 | 32 | 661872 | 1764 | 511 | 10.0 | 30.0 | 1.000 |
| 3 | 2048 | 4 | 32 | clustered | signature | 64 | 8 | 760.73 | 3215 | 41867 | 32 | 672128 | 1757 | 510 | 11.0 | 30.0 | 1.000 |
| 4 | 2048 | 4 | 32 | clustered | signature | 64 | 8 | 763.29 | 3171 | 42088 | 32 | 676672 | 1720 | 511 | 11.0 | 31.0 | 1.000 |
| 0 | 2048 | 4 | 32 | clustered | signature | 128 | 16 | 1194.62 | 3563 | 27935 | 32 | 475312 | 2008 | 510 | 6.0 | 18.0 | 1.000 |
| 1 | 2048 | 4 | 32 | clustered | signature | 128 | 8 | 565.87 | 3594 | 27580 | 32 | 470912 | 2013 | 511 | 6.0 | 17.0 | 1.000 |
| 2 | 2048 | 4 | 32 | clustered | signature | 128 | 16 | 1197.54 | 3567 | 27855 | 32 | 472480 | 2001 | 511 | 6.0 | 17.0 | 1.000 |
| 3 | 2048 | 4 | 32 | clustered | signature | 128 | 16 | 1171.60 | 3574 | 27962 | 32 | 475968 | 2010 | 510 | 6.0 | 17.0 | 1.000 |
| 4 | 2048 | 4 | 32 | clustered | signature | 128 | 16 | 1207.74 | 3557 | 28042 | 32 | 475216 | 2013 | 511 | 6.0 | 17.0 | 1.000 |
| 0 | 2048 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 428.00 | 512 | 40467 | 80 | 647472 | 0 | 512 | 79.0 | 80.0 | 0.202 |
| 1 | 2048 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 428.21 | 512 | 40465 | 80 | 647440 | 0 | 512 | 79.0 | 80.0 | 0.202 |
| 2 | 2048 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 425.93 | 512 | 40214 | 80 | 643424 | 0 | 512 | 79.0 | 80.0 | 0.197 |
| 3 | 2048 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 427.81 | 512 | 40466 | 80 | 647456 | 0 | 512 | 79.0 | 80.0 | 0.202 |
| 4 | 2048 | 4 | 32 | common_private | bounded_merge | 16 | 4 | 429.20 | 512 | 40463 | 80 | 647408 | 0 | 512 | 79.0 | 80.0 | 0.202 |
| 0 | 2048 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 604.32 | 2304 | 36371 | 16 | 647472 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 604.67 | 2304 | 36369 | 16 | 647440 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 602.86 | 2304 | 36262 | 16 | 643424 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 604.13 | 2304 | 36370 | 16 | 647456 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 4 | 32 | common_private | bounded_merge | 32 | 8 | 624.04 | 2297 | 36367 | 27 | 647408 | 2048 | 512 | 16.0 | 16.0 | 0.995 |
| 0 | 2048 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 1301.10 | 2176 | 34323 | 16 | 647472 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 1299.05 | 2176 | 34321 | 16 | 647440 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 1302.55 | 2176 | 34286 | 16 | 643424 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 1300.30 | 2176 | 34322 | 16 | 647456 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 4 | 32 | common_private | bounded_merge | 64 | 16 | 1318.07 | 2174 | 34319 | 27 | 647408 | 2048 | 512 | 16.0 | 16.0 | 0.998 |
| 0 | 2048 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 3143.41 | 2048 | 32288 | 16 | 631296 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 3140.47 | 2048 | 32288 | 16 | 631296 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 3140.96 | 2048 | 32288 | 16 | 627264 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 3142.55 | 2048 | 32288 | 16 | 631296 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 4 | 32 | common_private | bounded_merge | 128 | 32 | 3075.39 | 2047 | 32288 | 27 | 631296 | 2048 | 512 | 16.0 | 16.0 | 0.999 |
| 0 | 2048 | 4 | 32 | common_private | q_outer | 16 | 4 | 441.15 | 512 | 40467 | 80 | 647472 | 0 | 512 | 79.0 | 80.0 | 0.202 |
| 1 | 2048 | 4 | 32 | common_private | q_outer | 16 | 4 | 441.94 | 512 | 40465 | 80 | 647440 | 0 | 512 | 79.0 | 80.0 | 0.202 |
| 2 | 2048 | 4 | 32 | common_private | q_outer | 16 | 4 | 440.08 | 512 | 40214 | 80 | 643424 | 0 | 512 | 79.0 | 80.0 | 0.197 |
| 3 | 2048 | 4 | 32 | common_private | q_outer | 16 | 4 | 442.49 | 512 | 40466 | 80 | 647456 | 0 | 512 | 79.0 | 80.0 | 0.202 |
| 4 | 2048 | 4 | 32 | common_private | q_outer | 16 | 4 | 442.91 | 512 | 40463 | 80 | 647408 | 0 | 512 | 79.0 | 80.0 | 0.202 |
| 0 | 2048 | 4 | 32 | common_private | q_outer | 32 | 8 | 596.70 | 256 | 36371 | 144 | 1163872 | 0 | 512 | 143.0 | 144.0 | 0.113 |
| 1 | 2048 | 4 | 32 | common_private | q_outer | 32 | 8 | 605.23 | 256 | 36369 | 144 | 1163808 | 0 | 512 | 142.0 | 144.0 | 0.113 |
| 2 | 2048 | 4 | 32 | common_private | q_outer | 32 | 8 | 596.14 | 256 | 36262 | 144 | 1160384 | 0 | 512 | 142.0 | 144.0 | 0.109 |
| 3 | 2048 | 4 | 32 | common_private | q_outer | 32 | 8 | 600.39 | 256 | 36370 | 144 | 1163840 | 0 | 512 | 142.0 | 144.0 | 0.113 |
| 4 | 2048 | 4 | 32 | common_private | q_outer | 32 | 8 | 597.98 | 256 | 36367 | 144 | 1163744 | 0 | 512 | 143.0 | 144.0 | 0.113 |
| 0 | 2048 | 4 | 32 | common_private | q_outer | 64 | 16 | 1265.05 | 128 | 34323 | 272 | 2196672 | 0 | 512 | 268.0 | 271.0 | 0.060 |
| 1 | 2048 | 4 | 32 | common_private | q_outer | 64 | 16 | 1275.26 | 128 | 34321 | 272 | 2196544 | 0 | 512 | 268.0 | 272.0 | 0.060 |
| 2 | 2048 | 4 | 32 | common_private | q_outer | 64 | 16 | 1271.31 | 128 | 34286 | 272 | 2194304 | 0 | 512 | 268.0 | 271.0 | 0.058 |
| 3 | 2048 | 4 | 32 | common_private | q_outer | 64 | 16 | 1267.16 | 128 | 34322 | 272 | 2196608 | 0 | 512 | 268.0 | 272.0 | 0.060 |
| 4 | 2048 | 4 | 32 | common_private | q_outer | 64 | 16 | 1269.93 | 128 | 34319 | 272 | 2196416 | 0 | 512 | 268.0 | 271.0 | 0.060 |
| 0 | 2048 | 4 | 32 | common_private | q_outer | 128 | 32 | 5996.45 | 64 | 32288 | 512 | 4132864 | 0 | 512 | 504.5 | 511.0 | 0.032 |
| 1 | 2048 | 4 | 32 | common_private | q_outer | 128 | 32 | 6023.15 | 64 | 32288 | 512 | 4132864 | 0 | 512 | 504.5 | 511.0 | 0.032 |
| 2 | 2048 | 4 | 32 | common_private | q_outer | 128 | 32 | 6003.00 | 64 | 32288 | 512 | 4132864 | 0 | 512 | 504.5 | 511.0 | 0.031 |
| 3 | 2048 | 4 | 32 | common_private | q_outer | 128 | 32 | 5998.74 | 64 | 32288 | 512 | 4132864 | 0 | 512 | 504.5 | 511.0 | 0.032 |
| 4 | 2048 | 4 | 32 | common_private | q_outer | 128 | 32 | 6050.33 | 64 | 32288 | 512 | 4132864 | 0 | 512 | 504.5 | 511.0 | 0.032 |
| 0 | 2048 | 4 | 32 | common_private | signature | 16 | 4 | 498.77 | 2560 | 40467 | 16 | 647472 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 4 | 32 | common_private | signature | 16 | 4 | 499.67 | 2560 | 40465 | 16 | 647440 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 4 | 32 | common_private | signature | 16 | 4 | 495.70 | 2560 | 40214 | 16 | 643424 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 4 | 32 | common_private | signature | 16 | 4 | 498.13 | 2560 | 40466 | 16 | 647456 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 4 | 32 | common_private | signature | 16 | 4 | 498.97 | 2560 | 40463 | 16 | 647408 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 2048 | 4 | 32 | common_private | signature | 32 | 8 | 603.90 | 2304 | 36371 | 16 | 647472 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 4 | 32 | common_private | signature | 32 | 8 | 604.19 | 2304 | 36369 | 16 | 647440 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 4 | 32 | common_private | signature | 32 | 8 | 602.16 | 2304 | 36262 | 16 | 643424 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 4 | 32 | common_private | signature | 32 | 8 | 604.69 | 2304 | 36370 | 16 | 647456 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 4 | 32 | common_private | signature | 32 | 8 | 604.11 | 2304 | 36367 | 16 | 647408 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 2048 | 4 | 32 | common_private | signature | 64 | 16 | 1300.69 | 2176 | 34323 | 16 | 647472 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 4 | 32 | common_private | signature | 64 | 16 | 1299.63 | 2176 | 34321 | 16 | 647440 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 4 | 32 | common_private | signature | 64 | 16 | 1302.87 | 2176 | 34286 | 16 | 643424 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 4 | 32 | common_private | signature | 64 | 16 | 1301.16 | 2176 | 34322 | 16 | 647456 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 4 | 32 | common_private | signature | 64 | 16 | 1300.95 | 2176 | 34319 | 16 | 647408 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 2048 | 4 | 32 | common_private | signature | 128 | 32 | 3144.01 | 2048 | 32288 | 16 | 631296 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 4 | 32 | common_private | signature | 128 | 32 | 3141.06 | 2048 | 32288 | 16 | 631296 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 4 | 32 | common_private | signature | 128 | 32 | 3142.12 | 2048 | 32288 | 16 | 627264 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 4 | 32 | common_private | signature | 128 | 32 | 3142.96 | 2048 | 32288 | 16 | 631296 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 4 | 32 | common_private | signature | 128 | 32 | 3143.79 | 2048 | 32288 | 16 | 631296 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 2048 | 4 | 32 | random | bounded_merge | 16 | 4 | 628.18 | 512 | 58732 | 124 | 939712 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 1 | 2048 | 4 | 32 | random | bounded_merge | 16 | 4 | 627.00 | 512 | 58702 | 122 | 939232 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 2 | 2048 | 4 | 32 | random | bounded_merge | 16 | 4 | 629.12 | 512 | 58818 | 122 | 941088 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 3 | 2048 | 4 | 32 | random | bounded_merge | 16 | 4 | 630.49 | 512 | 58668 | 123 | 938688 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 4 | 2048 | 4 | 32 | random | bounded_merge | 16 | 4 | 632.16 | 513 | 58789 | 123 | 940624 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 0 | 2048 | 4 | 32 | random | bounded_merge | 32 | 8 | 1000.57 | 8706 | 52077 | 30 | 833408 | 2048 | 512 | 2.0 | 21.0 | 0.973 |
| 1 | 2048 | 4 | 32 | random | bounded_merge | 32 | 8 | 1021.81 | 8768 | 52062 | 28 | 833136 | 2048 | 512 | 2.0 | 21.0 | 0.974 |
| 2 | 2048 | 4 | 32 | random | bounded_merge | 32 | 8 | 1003.60 | 8781 | 52037 | 30 | 832624 | 2048 | 512 | 2.0 | 21.0 | 0.976 |
| 3 | 2048 | 4 | 32 | random | bounded_merge | 32 | 8 | 994.94 | 8800 | 51936 | 28 | 831088 | 2048 | 512 | 2.0 | 21.0 | 0.977 |
| 4 | 2048 | 4 | 32 | random | bounded_merge | 32 | 8 | 987.73 | 8708 | 52079 | 28 | 833296 | 2048 | 512 | 2.0 | 21.0 | 0.974 |
| 0 | 2048 | 4 | 32 | random | bounded_merge | 64 | 8 | 919.46 | 15378 | 41567 | 22 | 667376 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 2048 | 4 | 32 | random | bounded_merge | 64 | 8 | 910.01 | 15390 | 41589 | 21 | 667504 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 2048 | 4 | 32 | random | bounded_merge | 64 | 8 | 916.41 | 15380 | 41550 | 20 | 667456 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 2048 | 4 | 32 | random | bounded_merge | 64 | 8 | 917.17 | 15388 | 41486 | 22 | 666496 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 2048 | 4 | 32 | random | bounded_merge | 64 | 8 | 892.33 | 15333 | 41588 | 21 | 667904 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 2048 | 4 | 32 | random | bounded_merge | 128 | 16 | 1598.46 | 20289 | 28193 | 14 | 475648 | 2048 | 512 | 1.0 | 2.0 | 1.000 |
| 1 | 2048 | 4 | 32 | random | bounded_merge | 128 | 16 | 1592.04 | 20308 | 28148 | 12 | 474896 | 2048 | 512 | 1.0 | 2.0 | 1.000 |
| 2 | 2048 | 4 | 32 | random | bounded_merge | 128 | 16 | 1590.16 | 20302 | 28147 | 12 | 474512 | 2048 | 512 | 1.0 | 2.0 | 1.000 |
| 3 | 2048 | 4 | 32 | random | bounded_merge | 128 | 16 | 1589.66 | 20290 | 28135 | 13 | 475008 | 2048 | 512 | 1.0 | 2.0 | 1.000 |
| 4 | 2048 | 4 | 32 | random | bounded_merge | 128 | 16 | 1612.99 | 20335 | 28247 | 11 | 475696 | 2048 | 512 | 1.0 | 2.0 | 1.000 |
| 0 | 2048 | 4 | 32 | random | q_outer | 16 | 4 | 644.72 | 512 | 58732 | 124 | 939712 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 1 | 2048 | 4 | 32 | random | q_outer | 16 | 4 | 638.68 | 512 | 58702 | 122 | 939232 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 2 | 2048 | 4 | 32 | random | q_outer | 16 | 4 | 646.71 | 512 | 58818 | 122 | 941088 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 3 | 2048 | 4 | 32 | random | q_outer | 16 | 4 | 645.99 | 512 | 58668 | 123 | 938688 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 4 | 2048 | 4 | 32 | random | q_outer | 16 | 4 | 643.86 | 512 | 58789 | 124 | 940624 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 0 | 2048 | 4 | 32 | random | q_outer | 32 | 8 | 876.23 | 256 | 52077 | 217 | 1666464 | 0 | 512 | 203.5 | 210.0 | 0.000 |
| 1 | 2048 | 4 | 32 | random | q_outer | 32 | 8 | 887.79 | 256 | 52062 | 217 | 1665984 | 0 | 512 | 203.0 | 211.0 | 0.000 |
| 2 | 2048 | 4 | 32 | random | q_outer | 32 | 8 | 878.99 | 256 | 52037 | 217 | 1665184 | 0 | 512 | 203.0 | 211.0 | 0.000 |
| 3 | 2048 | 4 | 32 | random | q_outer | 32 | 8 | 876.62 | 256 | 51936 | 218 | 1661952 | 0 | 512 | 203.0 | 210.0 | 0.000 |
| 4 | 2048 | 4 | 32 | random | q_outer | 32 | 8 | 891.69 | 256 | 52079 | 219 | 1666528 | 0 | 512 | 203.0 | 210.0 | 0.000 |
| 0 | 2048 | 4 | 32 | random | q_outer | 64 | 16 | 1617.23 | 128 | 41567 | 354 | 2660288 | 0 | 512 | 325.0 | 334.0 | 0.000 |
| 1 | 2048 | 4 | 32 | random | q_outer | 64 | 16 | 1590.62 | 128 | 41589 | 347 | 2661696 | 0 | 512 | 324.0 | 335.0 | 0.000 |
| 2 | 2048 | 4 | 32 | random | q_outer | 64 | 16 | 1592.63 | 128 | 41550 | 346 | 2659200 | 0 | 512 | 325.0 | 334.0 | 0.000 |
| 3 | 2048 | 4 | 32 | random | q_outer | 64 | 16 | 1599.45 | 128 | 41486 | 351 | 2655104 | 0 | 512 | 323.0 | 336.0 | 0.000 |
| 4 | 2048 | 4 | 32 | random | q_outer | 64 | 16 | 1609.98 | 128 | 41588 | 343 | 2661632 | 0 | 512 | 324.0 | 336.0 | 0.000 |
| 0 | 2048 | 4 | 32 | random | q_outer | 128 | 32 | 5052.48 | 64 | 28193 | 463 | 3608704 | 0 | 512 | 440.5 | 451.0 | 0.000 |
| 1 | 2048 | 4 | 32 | random | q_outer | 128 | 32 | 4998.87 | 64 | 28148 | 456 | 3602944 | 0 | 512 | 440.0 | 448.7 | 0.000 |
| 2 | 2048 | 4 | 32 | random | q_outer | 128 | 32 | 5095.85 | 64 | 28147 | 462 | 3602816 | 0 | 512 | 440.0 | 448.0 | 0.000 |
| 3 | 2048 | 4 | 32 | random | q_outer | 128 | 32 | 5002.34 | 64 | 28135 | 459 | 3601280 | 0 | 512 | 440.5 | 449.7 | 0.000 |
| 4 | 2048 | 4 | 32 | random | q_outer | 128 | 32 | 5085.62 | 64 | 28247 | 458 | 3615616 | 0 | 512 | 442.5 | 449.7 | 0.000 |
| 0 | 2048 | 4 | 32 | random | signature | 16 | 4 | 741.76 | 4865 | 58732 | 32 | 939712 | 2044 | 512 | 3.0 | 28.0 | 1.000 |
| 1 | 2048 | 4 | 32 | random | signature | 16 | 4 | 740.27 | 4899 | 58702 | 32 | 939232 | 2047 | 512 | 3.0 | 27.0 | 1.000 |
| 2 | 2048 | 4 | 32 | random | signature | 16 | 4 | 740.63 | 4863 | 58818 | 31 | 941088 | 2047 | 512 | 3.0 | 28.0 | 1.000 |
| 3 | 2048 | 4 | 32 | random | signature | 16 | 4 | 740.77 | 4905 | 58668 | 32 | 938688 | 2043 | 512 | 3.0 | 27.0 | 1.000 |
| 4 | 2048 | 4 | 32 | random | signature | 16 | 4 | 739.76 | 4825 | 58789 | 32 | 940624 | 2041 | 512 | 3.0 | 28.0 | 1.000 |
| 0 | 2048 | 4 | 32 | random | signature | 32 | 8 | 967.01 | 8773 | 52077 | 30 | 833408 | 2048 | 512 | 2.0 | 21.0 | 1.000 |
| 1 | 2048 | 4 | 32 | random | signature | 32 | 8 | 962.73 | 8831 | 52062 | 28 | 833136 | 2048 | 512 | 2.0 | 21.0 | 1.000 |
| 2 | 2048 | 4 | 32 | random | signature | 32 | 8 | 964.27 | 8839 | 52037 | 29 | 832624 | 2048 | 512 | 2.0 | 21.0 | 1.000 |
| 3 | 2048 | 4 | 32 | random | signature | 32 | 8 | 966.94 | 8858 | 51936 | 28 | 831088 | 2048 | 512 | 2.0 | 21.0 | 1.000 |
| 4 | 2048 | 4 | 32 | random | signature | 32 | 8 | 958.32 | 8772 | 52079 | 28 | 833296 | 2048 | 512 | 2.0 | 21.0 | 1.000 |
| 0 | 2048 | 4 | 32 | random | signature | 64 | 8 | 892.91 | 15378 | 41567 | 22 | 667376 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 2048 | 4 | 32 | random | signature | 64 | 8 | 900.47 | 15390 | 41589 | 21 | 667504 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 2048 | 4 | 32 | random | signature | 64 | 8 | 897.58 | 15380 | 41550 | 20 | 667456 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 2048 | 4 | 32 | random | signature | 64 | 8 | 896.11 | 15388 | 41486 | 22 | 666496 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 2048 | 4 | 32 | random | signature | 64 | 8 | 903.57 | 15333 | 41588 | 21 | 667904 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 2048 | 4 | 32 | random | signature | 128 | 16 | 1598.77 | 20289 | 28193 | 14 | 475648 | 2048 | 512 | 1.0 | 2.0 | 1.000 |
| 1 | 2048 | 4 | 32 | random | signature | 128 | 16 | 1587.11 | 20308 | 28148 | 12 | 474896 | 2048 | 512 | 1.0 | 2.0 | 1.000 |
| 2 | 2048 | 4 | 32 | random | signature | 128 | 16 | 1595.22 | 20302 | 28147 | 12 | 474512 | 2048 | 512 | 1.0 | 2.0 | 1.000 |
| 3 | 2048 | 4 | 32 | random | signature | 128 | 16 | 1587.34 | 20290 | 28135 | 13 | 475008 | 2048 | 512 | 1.0 | 2.0 | 1.000 |
| 4 | 2048 | 4 | 32 | random | signature | 128 | 16 | 1609.16 | 20335 | 28247 | 11 | 475696 | 2048 | 512 | 1.0 | 2.0 | 1.000 |
| 0 | 2048 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 215.44 | 2020 | 16129 | 15 | 258064 | 0 | 510 | 8.0 | 8.0 | 0.993 |
| 1 | 2048 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 211.16 | 2019 | 16165 | 15 | 258640 | 0 | 511 | 8.0 | 8.0 | 0.991 |
| 2 | 2048 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 211.27 | 2015 | 16160 | 15 | 258560 | 0 | 511 | 8.0 | 8.0 | 0.985 |
| 3 | 2048 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 218.02 | 2012 | 16150 | 15 | 258400 | 0 | 510 | 8.0 | 8.0 | 0.985 |
| 4 | 2048 | 8 | 8 | clustered | bounded_merge | 16 | 2 | 212.28 | 2013 | 16138 | 15 | 258208 | 0 | 511 | 8.0 | 8.0 | 0.987 |
| 0 | 2048 | 8 | 8 | clustered | bounded_merge | 32 | 4 | 313.14 | 1959 | 15830 | 15 | 253312 | 7 | 510 | 8.0 | 8.0 | 0.966 |
| 1 | 2048 | 8 | 8 | clustered | bounded_merge | 32 | 4 | 299.73 | 1965 | 15915 | 15 | 254720 | 7 | 511 | 8.0 | 8.0 | 0.964 |
| 2 | 2048 | 8 | 8 | clustered | bounded_merge | 32 | 2 | 216.87 | 1952 | 15892 | 15 | 254272 | 0 | 511 | 8.0 | 8.0 | 0.955 |
| 3 | 2048 | 8 | 8 | clustered | bounded_merge | 32 | 4 | 299.63 | 1959 | 15912 | 15 | 254640 | 3 | 510 | 8.0 | 8.0 | 0.960 |
| 4 | 2048 | 8 | 8 | clustered | bounded_merge | 32 | 4 | 313.50 | 1952 | 15855 | 15 | 253760 | 3 | 511 | 8.0 | 8.0 | 0.960 |
| 0 | 2048 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 292.51 | 2074 | 15341 | 8 | 245872 | 235 | 510 | 8.0 | 8.0 | 0.966 |
| 1 | 2048 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 283.29 | 2087 | 15335 | 8 | 245952 | 256 | 511 | 8.0 | 8.0 | 0.964 |
| 2 | 2048 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 285.07 | 2076 | 15472 | 8 | 247888 | 226 | 511 | 8.0 | 8.0 | 0.964 |
| 3 | 2048 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 285.03 | 2088 | 15358 | 8 | 246256 | 241 | 510 | 8.0 | 8.0 | 0.970 |
| 4 | 2048 | 8 | 8 | clustered | bounded_merge | 64 | 4 | 292.74 | 2077 | 15325 | 8 | 245792 | 237 | 511 | 8.0 | 8.0 | 0.969 |
| 0 | 2048 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 281.90 | 2326 | 14341 | 8 | 231360 | 661 | 510 | 8.0 | 8.0 | 0.979 |
| 1 | 2048 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 273.63 | 2305 | 14478 | 8 | 233392 | 621 | 511 | 8.0 | 8.0 | 0.982 |
| 2 | 2048 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 275.05 | 2286 | 14561 | 8 | 234624 | 606 | 511 | 8.0 | 8.0 | 0.979 |
| 3 | 2048 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 272.96 | 2320 | 14431 | 8 | 233120 | 650 | 510 | 8.0 | 8.0 | 0.985 |
| 4 | 2048 | 8 | 8 | clustered | bounded_merge | 128 | 4 | 283.48 | 2313 | 14463 | 8 | 233728 | 642 | 511 | 8.0 | 8.0 | 0.983 |
| 0 | 2048 | 8 | 8 | clustered | q_outer | 16 | 2 | 206.86 | 1024 | 16129 | 16 | 258064 | 0 | 510 | 16.0 | 16.0 | 0.011 |
| 1 | 2048 | 8 | 8 | clustered | q_outer | 16 | 2 | 206.47 | 1024 | 16165 | 16 | 258640 | 0 | 511 | 16.0 | 16.0 | 0.010 |
| 2 | 2048 | 8 | 8 | clustered | q_outer | 16 | 2 | 207.47 | 1024 | 16160 | 16 | 258560 | 0 | 511 | 16.0 | 16.0 | 0.009 |
| 3 | 2048 | 8 | 8 | clustered | q_outer | 16 | 2 | 207.34 | 1024 | 16150 | 16 | 258400 | 0 | 510 | 16.0 | 16.0 | 0.010 |
| 4 | 2048 | 8 | 8 | clustered | q_outer | 16 | 2 | 207.80 | 1024 | 16138 | 16 | 258208 | 0 | 511 | 16.0 | 16.0 | 0.011 |
| 0 | 2048 | 8 | 8 | clustered | q_outer | 32 | 4 | 273.40 | 512 | 15830 | 32 | 506560 | 0 | 510 | 32.0 | 32.0 | 0.000 |
| 1 | 2048 | 8 | 8 | clustered | q_outer | 32 | 4 | 272.87 | 512 | 15915 | 32 | 509280 | 0 | 511 | 32.0 | 32.0 | 0.000 |
| 2 | 2048 | 8 | 8 | clustered | q_outer | 32 | 4 | 273.85 | 512 | 15892 | 32 | 508544 | 0 | 511 | 32.0 | 32.0 | 0.000 |
| 3 | 2048 | 8 | 8 | clustered | q_outer | 32 | 4 | 272.99 | 512 | 15912 | 32 | 509184 | 0 | 510 | 32.0 | 32.0 | 0.000 |
| 4 | 2048 | 8 | 8 | clustered | q_outer | 32 | 4 | 274.44 | 512 | 15855 | 32 | 507360 | 0 | 511 | 32.0 | 32.0 | 0.000 |
| 0 | 2048 | 8 | 8 | clustered | q_outer | 64 | 8 | 583.55 | 256 | 15341 | 64 | 981824 | 0 | 510 | 61.0 | 64.0 | 0.000 |
| 1 | 2048 | 8 | 8 | clustered | q_outer | 64 | 8 | 579.26 | 256 | 15335 | 64 | 981440 | 0 | 511 | 60.0 | 64.0 | 0.000 |
| 2 | 2048 | 8 | 8 | clustered | q_outer | 64 | 8 | 584.97 | 256 | 15472 | 64 | 990208 | 0 | 511 | 62.0 | 64.0 | 0.000 |
| 3 | 2048 | 8 | 8 | clustered | q_outer | 64 | 8 | 574.07 | 256 | 15358 | 64 | 982912 | 0 | 510 | 61.0 | 64.0 | 0.000 |
| 4 | 2048 | 8 | 8 | clustered | q_outer | 64 | 8 | 583.78 | 256 | 15325 | 64 | 980800 | 0 | 511 | 61.0 | 64.0 | 0.000 |
| 0 | 2048 | 8 | 8 | clustered | q_outer | 128 | 16 | 1653.09 | 128 | 14341 | 128 | 1835648 | 0 | 510 | 112.0 | 121.3 | 0.000 |
| 1 | 2048 | 8 | 8 | clustered | q_outer | 128 | 16 | 1706.13 | 128 | 14478 | 128 | 1853184 | 0 | 511 | 113.0 | 122.0 | 0.000 |
| 2 | 2048 | 8 | 8 | clustered | q_outer | 128 | 16 | 1685.78 | 128 | 14561 | 128 | 1863808 | 0 | 511 | 114.0 | 124.0 | 0.000 |
| 3 | 2048 | 8 | 8 | clustered | q_outer | 128 | 16 | 1659.32 | 128 | 14431 | 128 | 1847168 | 0 | 510 | 114.0 | 122.0 | 0.000 |
| 4 | 2048 | 8 | 8 | clustered | q_outer | 128 | 16 | 1667.23 | 128 | 14463 | 128 | 1851264 | 0 | 511 | 113.0 | 122.3 | 0.000 |
| 0 | 2048 | 8 | 8 | clustered | signature | 16 | 2 | 214.27 | 2058 | 16129 | 8 | 258064 | 38 | 510 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 8 | 8 | clustered | signature | 16 | 2 | 215.51 | 2059 | 16165 | 8 | 258640 | 40 | 511 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 8 | 8 | clustered | signature | 16 | 2 | 214.50 | 2067 | 16160 | 8 | 258560 | 52 | 511 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 8 | 8 | clustered | signature | 16 | 2 | 213.68 | 2078 | 16150 | 8 | 258400 | 66 | 510 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 8 | 8 | clustered | signature | 16 | 2 | 214.60 | 2061 | 16138 | 8 | 258208 | 48 | 511 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 8 | 8 | clustered | signature | 32 | 4 | 276.17 | 2113 | 15830 | 8 | 253312 | 157 | 510 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 8 | 8 | clustered | signature | 32 | 4 | 276.05 | 2112 | 15915 | 8 | 254720 | 152 | 511 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 8 | 8 | clustered | signature | 32 | 2 | 211.86 | 2116 | 15892 | 8 | 254272 | 164 | 511 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 8 | 8 | clustered | signature | 32 | 4 | 279.32 | 2125 | 15912 | 8 | 254640 | 167 | 510 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 8 | 8 | clustered | signature | 32 | 4 | 276.88 | 2115 | 15855 | 8 | 253760 | 164 | 511 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 8 | 8 | clustered | signature | 64 | 4 | 274.31 | 2216 | 15341 | 8 | 245872 | 369 | 510 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 8 | 8 | clustered | signature | 64 | 4 | 273.22 | 2231 | 15335 | 8 | 245952 | 388 | 511 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 8 | 8 | clustered | signature | 64 | 4 | 274.72 | 2201 | 15472 | 8 | 247888 | 343 | 511 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 8 | 8 | clustered | signature | 64 | 4 | 276.36 | 2215 | 15358 | 8 | 246256 | 359 | 510 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 8 | 8 | clustered | signature | 64 | 4 | 274.28 | 2208 | 15325 | 8 | 245792 | 363 | 511 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 8 | 8 | clustered | signature | 128 | 4 | 265.79 | 2401 | 14341 | 8 | 231360 | 727 | 510 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 8 | 8 | clustered | signature | 128 | 4 | 265.73 | 2374 | 14478 | 8 | 233392 | 677 | 511 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 8 | 8 | clustered | signature | 128 | 4 | 267.05 | 2360 | 14561 | 8 | 234624 | 672 | 511 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 8 | 8 | clustered | signature | 128 | 4 | 266.05 | 2378 | 14431 | 8 | 233120 | 704 | 510 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 8 | 8 | clustered | signature | 128 | 4 | 268.18 | 2379 | 14463 | 8 | 233728 | 699 | 511 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 153.62 | 1024 | 12165 | 12 | 194640 | 0 | 512 | 12.0 | 12.0 | 0.337 |
| 1 | 2048 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 153.49 | 1024 | 12168 | 12 | 194688 | 0 | 512 | 12.0 | 12.0 | 0.337 |
| 2 | 2048 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 152.72 | 1024 | 12166 | 12 | 194656 | 0 | 512 | 12.0 | 12.0 | 0.337 |
| 3 | 2048 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 152.45 | 1024 | 12168 | 12 | 194688 | 0 | 512 | 12.0 | 12.0 | 0.337 |
| 4 | 2048 | 8 | 8 | common_private | bounded_merge | 16 | 2 | 154.21 | 1024 | 12167 | 12 | 194672 | 0 | 512 | 12.0 | 12.0 | 0.337 |
| 0 | 2048 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 207.30 | 2560 | 10117 | 4 | 194640 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 2048 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 207.59 | 2560 | 10120 | 4 | 194688 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 2048 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 205.80 | 2560 | 10118 | 4 | 194656 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 2048 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 207.30 | 2560 | 10120 | 4 | 194688 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 2048 | 8 | 8 | common_private | bounded_merge | 32 | 4 | 208.48 | 2560 | 10119 | 4 | 194672 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 2048 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 370.10 | 2304 | 9093 | 4 | 194640 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 2048 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 372.51 | 2304 | 9096 | 4 | 194688 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 2048 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 369.52 | 2304 | 9094 | 4 | 194656 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 2048 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 371.84 | 2304 | 9096 | 4 | 194688 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 2048 | 8 | 8 | common_private | bounded_merge | 64 | 8 | 370.00 | 2304 | 9095 | 4 | 194672 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 2048 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 986.38 | 2176 | 8581 | 4 | 194640 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 2048 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 993.17 | 2176 | 8584 | 4 | 194688 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 2048 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 988.68 | 2176 | 8582 | 4 | 194656 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 2048 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 989.79 | 2176 | 8584 | 4 | 194688 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 2048 | 8 | 8 | common_private | bounded_merge | 128 | 16 | 985.69 | 2176 | 8583 | 4 | 194672 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 2048 | 8 | 8 | common_private | q_outer | 16 | 2 | 155.65 | 1024 | 12165 | 12 | 194640 | 0 | 512 | 12.0 | 12.0 | 0.337 |
| 1 | 2048 | 8 | 8 | common_private | q_outer | 16 | 2 | 155.40 | 1024 | 12168 | 12 | 194688 | 0 | 512 | 12.0 | 12.0 | 0.337 |
| 2 | 2048 | 8 | 8 | common_private | q_outer | 16 | 2 | 155.92 | 1024 | 12166 | 12 | 194656 | 0 | 512 | 12.0 | 12.0 | 0.337 |
| 3 | 2048 | 8 | 8 | common_private | q_outer | 16 | 2 | 154.61 | 1024 | 12168 | 12 | 194688 | 0 | 512 | 12.0 | 12.0 | 0.337 |
| 4 | 2048 | 8 | 8 | common_private | q_outer | 16 | 2 | 156.64 | 1024 | 12167 | 12 | 194672 | 0 | 512 | 12.0 | 12.0 | 0.337 |
| 0 | 2048 | 8 | 8 | common_private | q_outer | 32 | 4 | 175.99 | 512 | 10117 | 20 | 323744 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 1 | 2048 | 8 | 8 | common_private | q_outer | 32 | 4 | 175.10 | 512 | 10120 | 20 | 323840 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 2 | 2048 | 8 | 8 | common_private | q_outer | 32 | 4 | 176.04 | 512 | 10118 | 20 | 323776 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 3 | 2048 | 8 | 8 | common_private | q_outer | 32 | 4 | 174.76 | 512 | 10120 | 20 | 323840 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 4 | 2048 | 8 | 8 | common_private | q_outer | 32 | 4 | 175.32 | 512 | 10119 | 20 | 323808 | 0 | 512 | 20.0 | 20.0 | 0.202 |
| 0 | 2048 | 8 | 8 | common_private | q_outer | 64 | 8 | 335.10 | 256 | 9093 | 36 | 581952 | 0 | 512 | 36.0 | 36.0 | 0.113 |
| 1 | 2048 | 8 | 8 | common_private | q_outer | 64 | 8 | 330.33 | 256 | 9096 | 36 | 582144 | 0 | 512 | 36.0 | 36.0 | 0.113 |
| 2 | 2048 | 8 | 8 | common_private | q_outer | 64 | 8 | 336.14 | 256 | 9094 | 36 | 582016 | 0 | 512 | 36.0 | 36.0 | 0.113 |
| 3 | 2048 | 8 | 8 | common_private | q_outer | 64 | 8 | 333.27 | 256 | 9096 | 36 | 582144 | 0 | 512 | 36.0 | 36.0 | 0.113 |
| 4 | 2048 | 8 | 8 | common_private | q_outer | 64 | 8 | 333.29 | 256 | 9095 | 36 | 582080 | 0 | 512 | 36.0 | 36.0 | 0.113 |
| 0 | 2048 | 8 | 8 | common_private | q_outer | 128 | 16 | 908.60 | 128 | 8581 | 68 | 1098368 | 0 | 512 | 67.0 | 68.0 | 0.060 |
| 1 | 2048 | 8 | 8 | common_private | q_outer | 128 | 16 | 893.73 | 128 | 8584 | 68 | 1098752 | 0 | 512 | 67.0 | 68.0 | 0.060 |
| 2 | 2048 | 8 | 8 | common_private | q_outer | 128 | 16 | 916.42 | 128 | 8582 | 68 | 1098496 | 0 | 512 | 67.0 | 68.0 | 0.060 |
| 3 | 2048 | 8 | 8 | common_private | q_outer | 128 | 16 | 889.84 | 128 | 8584 | 68 | 1098752 | 0 | 512 | 67.0 | 68.0 | 0.060 |
| 4 | 2048 | 8 | 8 | common_private | q_outer | 128 | 16 | 893.71 | 128 | 8583 | 68 | 1098624 | 0 | 512 | 67.0 | 68.0 | 0.060 |
| 0 | 2048 | 8 | 8 | common_private | signature | 16 | 2 | 189.25 | 3072 | 12165 | 4 | 194640 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 2048 | 8 | 8 | common_private | signature | 16 | 2 | 189.65 | 3072 | 12168 | 4 | 194688 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 2048 | 8 | 8 | common_private | signature | 16 | 2 | 188.99 | 3072 | 12166 | 4 | 194656 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 2048 | 8 | 8 | common_private | signature | 16 | 2 | 189.86 | 3072 | 12168 | 4 | 194688 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 2048 | 8 | 8 | common_private | signature | 16 | 2 | 189.66 | 3072 | 12167 | 4 | 194672 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 2048 | 8 | 8 | common_private | signature | 32 | 4 | 208.63 | 2560 | 10117 | 4 | 194640 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 2048 | 8 | 8 | common_private | signature | 32 | 4 | 207.17 | 2560 | 10120 | 4 | 194688 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 2048 | 8 | 8 | common_private | signature | 32 | 4 | 207.03 | 2560 | 10118 | 4 | 194656 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 2048 | 8 | 8 | common_private | signature | 32 | 4 | 208.64 | 2560 | 10120 | 4 | 194688 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 2048 | 8 | 8 | common_private | signature | 32 | 4 | 208.10 | 2560 | 10119 | 4 | 194672 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 2048 | 8 | 8 | common_private | signature | 64 | 8 | 375.03 | 2304 | 9093 | 4 | 194640 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 2048 | 8 | 8 | common_private | signature | 64 | 8 | 372.24 | 2304 | 9096 | 4 | 194688 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 2048 | 8 | 8 | common_private | signature | 64 | 8 | 372.39 | 2304 | 9094 | 4 | 194656 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 2048 | 8 | 8 | common_private | signature | 64 | 8 | 375.65 | 2304 | 9096 | 4 | 194688 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 2048 | 8 | 8 | common_private | signature | 64 | 8 | 374.61 | 2304 | 9095 | 4 | 194672 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 2048 | 8 | 8 | common_private | signature | 128 | 16 | 1001.01 | 2176 | 8581 | 4 | 194640 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 1 | 2048 | 8 | 8 | common_private | signature | 128 | 16 | 994.01 | 2176 | 8584 | 4 | 194688 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 2 | 2048 | 8 | 8 | common_private | signature | 128 | 16 | 995.09 | 2176 | 8582 | 4 | 194656 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 3 | 2048 | 8 | 8 | common_private | signature | 128 | 16 | 998.66 | 2176 | 8584 | 4 | 194688 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 4 | 2048 | 8 | 8 | common_private | signature | 128 | 16 | 998.77 | 2176 | 8583 | 4 | 194672 | 2048 | 512 | 4.0 | 4.0 | 1.000 |
| 0 | 2048 | 8 | 8 | random | bounded_merge | 16 | 2 | 218.56 | 1928 | 16006 | 15 | 256096 | 0 | 512 | 8.0 | 8.0 | 0.897 |
| 1 | 2048 | 8 | 8 | random | bounded_merge | 16 | 2 | 217.83 | 1926 | 16006 | 15 | 256096 | 0 | 511 | 8.0 | 8.0 | 0.895 |
| 2 | 2048 | 8 | 8 | random | bounded_merge | 16 | 2 | 219.79 | 1923 | 16036 | 15 | 256576 | 0 | 511 | 8.0 | 8.0 | 0.893 |
| 3 | 2048 | 8 | 8 | random | bounded_merge | 16 | 2 | 218.47 | 1903 | 15993 | 15 | 255888 | 0 | 512 | 8.0 | 8.0 | 0.877 |
| 4 | 2048 | 8 | 8 | random | bounded_merge | 16 | 2 | 219.24 | 1926 | 16042 | 15 | 256672 | 0 | 512 | 8.0 | 8.0 | 0.895 |
| 0 | 2048 | 8 | 8 | random | bounded_merge | 32 | 4 | 299.44 | 1806 | 15769 | 15 | 252384 | 94 | 512 | 8.0 | 14.0 | 0.751 |
| 1 | 2048 | 8 | 8 | random | bounded_merge | 32 | 4 | 298.82 | 1823 | 15765 | 15 | 252304 | 108 | 511 | 8.0 | 14.0 | 0.757 |
| 2 | 2048 | 8 | 8 | random | bounded_merge | 32 | 4 | 302.19 | 1808 | 15796 | 15 | 252832 | 107 | 511 | 8.0 | 14.0 | 0.746 |
| 3 | 2048 | 8 | 8 | random | bounded_merge | 32 | 4 | 309.50 | 1799 | 15771 | 15 | 252384 | 100 | 512 | 8.0 | 14.0 | 0.741 |
| 4 | 2048 | 8 | 8 | random | bounded_merge | 32 | 4 | 302.06 | 1792 | 15790 | 15 | 252768 | 96 | 512 | 8.0 | 14.0 | 0.739 |
| 0 | 2048 | 8 | 8 | random | bounded_merge | 64 | 4 | 289.54 | 2590 | 15284 | 8 | 244864 | 1026 | 512 | 7.0 | 8.0 | 0.900 |
| 1 | 2048 | 8 | 8 | random | bounded_merge | 64 | 4 | 289.32 | 2587 | 15305 | 8 | 245200 | 996 | 511 | 7.0 | 8.0 | 0.900 |
| 2 | 2048 | 8 | 8 | random | bounded_merge | 64 | 4 | 293.03 | 2645 | 15268 | 8 | 244688 | 1084 | 511 | 7.0 | 8.0 | 0.898 |
| 3 | 2048 | 8 | 8 | random | bounded_merge | 64 | 4 | 299.93 | 2622 | 15286 | 8 | 244960 | 1035 | 512 | 7.0 | 8.0 | 0.899 |
| 4 | 2048 | 8 | 8 | random | bounded_merge | 64 | 4 | 291.71 | 2618 | 15298 | 8 | 245248 | 1051 | 512 | 7.0 | 8.0 | 0.899 |
| 0 | 2048 | 8 | 8 | random | bounded_merge | 128 | 4 | 294.41 | 3530 | 14352 | 8 | 231616 | 1695 | 512 | 5.0 | 7.0 | 0.959 |
| 1 | 2048 | 8 | 8 | random | bounded_merge | 128 | 8 | 602.23 | 3509 | 14391 | 8 | 232224 | 1675 | 511 | 5.0 | 7.0 | 0.959 |
| 2 | 2048 | 8 | 8 | random | bounded_merge | 128 | 4 | 295.38 | 3535 | 14387 | 8 | 232112 | 1684 | 511 | 5.0 | 7.0 | 0.957 |
| 3 | 2048 | 8 | 8 | random | bounded_merge | 128 | 4 | 303.27 | 3536 | 14401 | 8 | 232016 | 1701 | 512 | 5.0 | 7.0 | 0.956 |
| 4 | 2048 | 8 | 8 | random | bounded_merge | 128 | 4 | 296.57 | 3486 | 14434 | 8 | 232960 | 1693 | 512 | 5.0 | 7.0 | 0.957 |
| 0 | 2048 | 8 | 8 | random | q_outer | 16 | 2 | 198.44 | 1024 | 16006 | 16 | 256096 | 0 | 512 | 16.0 | 16.0 | 0.008 |
| 1 | 2048 | 8 | 8 | random | q_outer | 16 | 2 | 200.09 | 1024 | 16006 | 16 | 256096 | 0 | 511 | 16.0 | 16.0 | 0.008 |
| 2 | 2048 | 8 | 8 | random | q_outer | 16 | 2 | 198.84 | 1024 | 16036 | 16 | 256576 | 0 | 511 | 16.0 | 16.0 | 0.008 |
| 3 | 2048 | 8 | 8 | random | q_outer | 16 | 2 | 199.86 | 1024 | 15993 | 16 | 255888 | 0 | 512 | 16.0 | 16.0 | 0.010 |
| 4 | 2048 | 8 | 8 | random | q_outer | 16 | 2 | 200.09 | 1024 | 16042 | 16 | 256672 | 0 | 512 | 16.0 | 16.0 | 0.008 |
| 0 | 2048 | 8 | 8 | random | q_outer | 32 | 4 | 266.96 | 512 | 15769 | 32 | 504608 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 1 | 2048 | 8 | 8 | random | q_outer | 32 | 4 | 268.38 | 512 | 15765 | 32 | 504480 | 0 | 511 | 31.0 | 32.0 | 0.000 |
| 2 | 2048 | 8 | 8 | random | q_outer | 32 | 4 | 267.05 | 512 | 15796 | 32 | 505472 | 0 | 511 | 31.0 | 32.0 | 0.000 |
| 3 | 2048 | 8 | 8 | random | q_outer | 32 | 4 | 266.25 | 512 | 15771 | 32 | 504672 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 4 | 2048 | 8 | 8 | random | q_outer | 32 | 4 | 268.58 | 512 | 15790 | 32 | 505280 | 0 | 512 | 31.0 | 32.0 | 0.000 |
| 0 | 2048 | 8 | 8 | random | q_outer | 64 | 8 | 557.09 | 256 | 15284 | 63 | 978176 | 0 | 512 | 60.0 | 62.0 | 0.000 |
| 1 | 2048 | 8 | 8 | random | q_outer | 64 | 8 | 561.94 | 256 | 15305 | 64 | 979520 | 0 | 511 | 60.0 | 62.0 | 0.000 |
| 2 | 2048 | 8 | 8 | random | q_outer | 64 | 8 | 557.50 | 256 | 15268 | 64 | 977152 | 0 | 511 | 60.0 | 62.0 | 0.000 |
| 3 | 2048 | 8 | 8 | random | q_outer | 64 | 8 | 561.64 | 256 | 15286 | 64 | 978304 | 0 | 512 | 60.0 | 62.0 | 0.000 |
| 4 | 2048 | 8 | 8 | random | q_outer | 64 | 8 | 568.14 | 256 | 15298 | 64 | 979072 | 0 | 512 | 60.0 | 62.0 | 0.000 |
| 0 | 2048 | 8 | 8 | random | q_outer | 128 | 16 | 1523.55 | 128 | 14352 | 120 | 1837056 | 0 | 512 | 113.0 | 116.0 | 0.000 |
| 1 | 2048 | 8 | 8 | random | q_outer | 128 | 16 | 1541.29 | 128 | 14391 | 121 | 1842048 | 0 | 511 | 112.0 | 117.0 | 0.000 |
| 2 | 2048 | 8 | 8 | random | q_outer | 128 | 16 | 1527.52 | 128 | 14387 | 121 | 1841536 | 0 | 511 | 113.0 | 116.0 | 0.000 |
| 3 | 2048 | 8 | 8 | random | q_outer | 128 | 16 | 1546.62 | 128 | 14401 | 122 | 1843328 | 0 | 512 | 113.0 | 117.0 | 0.000 |
| 4 | 2048 | 8 | 8 | random | q_outer | 128 | 16 | 1543.98 | 128 | 14434 | 121 | 1847552 | 0 | 512 | 113.0 | 117.0 | 0.000 |
| 0 | 2048 | 8 | 8 | random | signature | 16 | 2 | 211.67 | 2168 | 16006 | 8 | 256096 | 240 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 8 | 8 | random | signature | 16 | 2 | 210.46 | 2170 | 16006 | 8 | 256096 | 244 | 511 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 8 | 8 | random | signature | 16 | 2 | 209.77 | 2173 | 16036 | 8 | 256576 | 250 | 511 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 8 | 8 | random | signature | 16 | 2 | 210.74 | 2193 | 15993 | 8 | 255888 | 290 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 8 | 8 | random | signature | 16 | 2 | 211.38 | 2170 | 16042 | 8 | 256672 | 244 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 8 | 8 | random | signature | 32 | 4 | 279.46 | 2389 | 15769 | 8 | 252384 | 614 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 8 | 8 | random | signature | 32 | 4 | 281.37 | 2393 | 15765 | 8 | 252304 | 611 | 511 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 8 | 8 | random | signature | 32 | 4 | 281.54 | 2404 | 15796 | 8 | 252832 | 636 | 511 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 8 | 8 | random | signature | 32 | 4 | 282.81 | 2406 | 15771 | 8 | 252384 | 638 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 8 | 8 | random | signature | 32 | 4 | 278.90 | 2403 | 15790 | 8 | 252768 | 647 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 8 | 8 | random | signature | 64 | 4 | 284.02 | 2832 | 15284 | 8 | 244864 | 1184 | 512 | 7.0 | 8.0 | 1.000 |
| 1 | 2048 | 8 | 8 | random | signature | 64 | 4 | 284.52 | 2824 | 15305 | 8 | 245200 | 1156 | 511 | 7.0 | 8.0 | 1.000 |
| 2 | 2048 | 8 | 8 | random | signature | 64 | 4 | 286.81 | 2892 | 15268 | 8 | 244688 | 1239 | 511 | 7.0 | 8.0 | 1.000 |
| 3 | 2048 | 8 | 8 | random | signature | 64 | 4 | 285.93 | 2863 | 15286 | 8 | 244960 | 1188 | 512 | 7.0 | 8.0 | 1.000 |
| 4 | 2048 | 8 | 8 | random | signature | 64 | 4 | 283.98 | 2858 | 15298 | 8 | 245248 | 1219 | 512 | 7.0 | 8.0 | 1.000 |
| 0 | 2048 | 8 | 8 | random | signature | 128 | 4 | 286.81 | 3631 | 14352 | 8 | 231616 | 1728 | 512 | 5.0 | 7.0 | 1.000 |
| 1 | 2048 | 8 | 8 | random | signature | 128 | 8 | 560.79 | 3611 | 14391 | 8 | 232224 | 1707 | 511 | 5.0 | 7.0 | 1.000 |
| 2 | 2048 | 8 | 8 | random | signature | 128 | 4 | 288.21 | 3643 | 14387 | 8 | 232112 | 1719 | 511 | 5.0 | 7.0 | 1.000 |
| 3 | 2048 | 8 | 8 | random | signature | 128 | 4 | 288.48 | 3642 | 14401 | 8 | 232016 | 1743 | 512 | 5.0 | 7.0 | 1.000 |
| 4 | 2048 | 8 | 8 | random | signature | 128 | 4 | 287.32 | 3595 | 14434 | 8 | 232960 | 1729 | 512 | 5.0 | 7.0 | 1.000 |
| 0 | 2048 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 425.56 | 1991 | 32160 | 31 | 514560 | 0 | 510 | 16.0 | 16.0 | 0.970 |
| 1 | 2048 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 420.42 | 1994 | 32197 | 31 | 515152 | 0 | 511 | 16.0 | 16.0 | 0.974 |
| 2 | 2048 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 413.23 | 1980 | 32144 | 31 | 514304 | 0 | 511 | 16.0 | 16.0 | 0.963 |
| 3 | 2048 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 408.13 | 1987 | 32122 | 30 | 513952 | 0 | 510 | 16.0 | 16.0 | 0.974 |
| 4 | 2048 | 8 | 16 | clustered | bounded_merge | 16 | 2 | 404.22 | 1995 | 32172 | 31 | 514752 | 0 | 511 | 16.0 | 16.0 | 0.974 |
| 0 | 2048 | 8 | 16 | clustered | bounded_merge | 32 | 4 | 580.69 | 1890 | 31025 | 31 | 497280 | 31 | 510 | 16.0 | 16.0 | 0.922 |
| 1 | 2048 | 8 | 16 | clustered | bounded_merge | 32 | 4 | 562.85 | 1886 | 31116 | 31 | 498512 | 27 | 511 | 16.0 | 16.0 | 0.915 |
| 2 | 2048 | 8 | 16 | clustered | bounded_merge | 32 | 4 | 558.87 | 1866 | 31073 | 31 | 497360 | 17 | 511 | 16.0 | 16.0 | 0.904 |
| 3 | 2048 | 8 | 16 | clustered | bounded_merge | 32 | 4 | 573.09 | 1896 | 31223 | 31 | 499872 | 18 | 510 | 16.0 | 16.0 | 0.927 |
| 4 | 2048 | 8 | 16 | clustered | bounded_merge | 32 | 4 | 558.36 | 1897 | 31139 | 31 | 498576 | 19 | 511 | 16.0 | 16.0 | 0.928 |
| 0 | 2048 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 551.03 | 2211 | 29123 | 27 | 469600 | 536 | 510 | 16.0 | 16.0 | 0.947 |
| 1 | 2048 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 507.24 | 2227 | 28937 | 16 | 467856 | 567 | 511 | 16.0 | 16.0 | 0.944 |
| 2 | 2048 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 508.67 | 2214 | 29256 | 27 | 471072 | 547 | 511 | 16.0 | 16.0 | 0.943 |
| 3 | 2048 | 8 | 16 | clustered | bounded_merge | 64 | 4 | 534.66 | 2197 | 29278 | 27 | 471584 | 512 | 510 | 16.0 | 16.0 | 0.949 |
| 4 | 2048 | 8 | 16 | clustered | bounded_merge | 64 | 8 | 1095.30 | 2184 | 29167 | 27 | 470080 | 492 | 511 | 16.0 | 16.0 | 0.943 |
| 0 | 2048 | 8 | 16 | clustered | bounded_merge | 128 | 8 | 1013.04 | 2649 | 25642 | 27 | 424304 | 1196 | 510 | 10.0 | 16.0 | 0.970 |
| 1 | 2048 | 8 | 16 | clustered | bounded_merge | 128 | 8 | 1017.33 | 2628 | 25851 | 27 | 428432 | 1151 | 511 | 10.0 | 16.0 | 0.966 |
| 2 | 2048 | 8 | 16 | clustered | bounded_merge | 128 | 8 | 1015.95 | 2627 | 25962 | 27 | 428800 | 1170 | 511 | 10.0 | 16.0 | 0.968 |
| 3 | 2048 | 8 | 16 | clustered | bounded_merge | 128 | 8 | 1016.13 | 2590 | 26030 | 27 | 429456 | 1116 | 510 | 11.0 | 16.0 | 0.966 |
| 4 | 2048 | 8 | 16 | clustered | bounded_merge | 128 | 8 | 1030.25 | 2573 | 26035 | 27 | 429344 | 1091 | 511 | 11.0 | 16.0 | 0.963 |
| 0 | 2048 | 8 | 16 | clustered | q_outer | 16 | 2 | 387.26 | 1024 | 32160 | 32 | 514560 | 0 | 510 | 32.0 | 32.0 | 0.014 |
| 1 | 2048 | 8 | 16 | clustered | q_outer | 16 | 2 | 388.46 | 1024 | 32197 | 32 | 515152 | 0 | 511 | 32.0 | 32.0 | 0.014 |
| 2 | 2048 | 8 | 16 | clustered | q_outer | 16 | 2 | 388.22 | 1024 | 32144 | 32 | 514304 | 0 | 511 | 32.0 | 32.0 | 0.015 |
| 3 | 2048 | 8 | 16 | clustered | q_outer | 16 | 2 | 388.73 | 1024 | 32122 | 32 | 513952 | 0 | 510 | 32.0 | 32.0 | 0.017 |
| 4 | 2048 | 8 | 16 | clustered | q_outer | 16 | 2 | 389.11 | 1024 | 32172 | 32 | 514752 | 0 | 511 | 32.0 | 32.0 | 0.013 |
| 0 | 2048 | 8 | 16 | clustered | q_outer | 32 | 4 | 524.68 | 512 | 31025 | 64 | 992800 | 0 | 510 | 64.0 | 64.0 | 0.000 |
| 1 | 2048 | 8 | 16 | clustered | q_outer | 32 | 4 | 525.78 | 512 | 31116 | 64 | 995712 | 0 | 511 | 64.0 | 64.0 | 0.000 |
| 2 | 2048 | 8 | 16 | clustered | q_outer | 32 | 4 | 524.33 | 512 | 31073 | 64 | 994336 | 0 | 511 | 64.0 | 64.0 | 0.000 |
| 3 | 2048 | 8 | 16 | clustered | q_outer | 32 | 4 | 521.06 | 512 | 31223 | 64 | 999136 | 0 | 510 | 64.0 | 64.0 | 0.000 |
| 4 | 2048 | 8 | 16 | clustered | q_outer | 32 | 4 | 526.01 | 512 | 31139 | 64 | 996448 | 0 | 511 | 64.0 | 64.0 | 0.000 |
| 0 | 2048 | 8 | 16 | clustered | q_outer | 64 | 8 | 1118.33 | 256 | 29123 | 128 | 1863872 | 0 | 510 | 115.0 | 127.0 | 0.000 |
| 1 | 2048 | 8 | 16 | clustered | q_outer | 64 | 8 | 1114.84 | 256 | 28937 | 128 | 1851968 | 0 | 511 | 114.0 | 128.0 | 0.000 |
| 2 | 2048 | 8 | 16 | clustered | q_outer | 64 | 8 | 1116.41 | 256 | 29256 | 128 | 1872384 | 0 | 511 | 116.0 | 128.0 | 0.000 |
| 3 | 2048 | 8 | 16 | clustered | q_outer | 64 | 8 | 1127.15 | 256 | 29278 | 128 | 1873792 | 0 | 510 | 115.0 | 128.0 | 0.000 |
| 4 | 2048 | 8 | 16 | clustered | q_outer | 64 | 8 | 1126.48 | 256 | 29167 | 128 | 1866688 | 0 | 511 | 115.0 | 128.0 | 0.000 |
| 0 | 2048 | 8 | 16 | clustered | q_outer | 128 | 16 | 3034.89 | 128 | 25642 | 238 | 3282176 | 0 | 510 | 200.0 | 225.3 | 0.000 |
| 1 | 2048 | 8 | 16 | clustered | q_outer | 128 | 16 | 3145.63 | 128 | 25851 | 240 | 3308928 | 0 | 511 | 201.0 | 221.0 | 0.000 |
| 2 | 2048 | 8 | 16 | clustered | q_outer | 128 | 16 | 3099.64 | 128 | 25962 | 245 | 3323136 | 0 | 511 | 205.0 | 225.6 | 0.000 |
| 3 | 2048 | 8 | 16 | clustered | q_outer | 128 | 16 | 3044.28 | 128 | 26030 | 245 | 3331840 | 0 | 510 | 204.5 | 225.3 | 0.000 |
| 4 | 2048 | 8 | 16 | clustered | q_outer | 128 | 16 | 3103.20 | 128 | 26035 | 249 | 3332480 | 0 | 511 | 203.5 | 224.6 | 0.000 |
| 0 | 2048 | 8 | 16 | clustered | signature | 16 | 2 | 390.62 | 2102 | 32160 | 16 | 514560 | 110 | 510 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 8 | 16 | clustered | signature | 16 | 2 | 390.19 | 2096 | 32197 | 16 | 515152 | 100 | 511 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 8 | 16 | clustered | signature | 16 | 2 | 391.74 | 2107 | 32144 | 16 | 514304 | 122 | 511 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 8 | 16 | clustered | signature | 16 | 2 | 392.09 | 2105 | 32122 | 16 | 513952 | 116 | 510 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 8 | 16 | clustered | signature | 16 | 2 | 391.83 | 2096 | 32172 | 16 | 514752 | 98 | 511 | 16.0 | 16.0 | 1.000 |
| 0 | 2048 | 8 | 16 | clustered | signature | 32 | 4 | 507.67 | 2212 | 31025 | 16 | 497280 | 337 | 510 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 8 | 16 | clustered | signature | 32 | 4 | 509.73 | 2218 | 31116 | 16 | 498512 | 339 | 511 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 8 | 16 | clustered | signature | 32 | 4 | 508.34 | 2223 | 31073 | 16 | 497360 | 359 | 511 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 8 | 16 | clustered | signature | 32 | 4 | 512.84 | 2205 | 31223 | 16 | 499872 | 315 | 510 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 8 | 16 | clustered | signature | 32 | 4 | 510.31 | 2196 | 31139 | 16 | 498576 | 302 | 511 | 16.0 | 16.0 | 1.000 |
| 0 | 2048 | 8 | 16 | clustered | signature | 64 | 4 | 487.31 | 2422 | 29123 | 16 | 469600 | 722 | 510 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 8 | 16 | clustered | signature | 64 | 4 | 485.55 | 2439 | 28937 | 16 | 467856 | 749 | 511 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 8 | 16 | clustered | signature | 64 | 4 | 487.71 | 2424 | 29256 | 16 | 471072 | 727 | 511 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 8 | 16 | clustered | signature | 64 | 4 | 491.23 | 2397 | 29278 | 16 | 471584 | 685 | 510 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 8 | 16 | clustered | signature | 64 | 8 | 983.31 | 2396 | 29167 | 16 | 470080 | 678 | 511 | 16.0 | 16.0 | 1.000 |
| 0 | 2048 | 8 | 16 | clustered | signature | 128 | 8 | 897.09 | 2749 | 25642 | 16 | 424304 | 1259 | 510 | 9.0 | 16.0 | 1.000 |
| 1 | 2048 | 8 | 16 | clustered | signature | 128 | 8 | 903.80 | 2727 | 25851 | 16 | 428432 | 1211 | 511 | 9.0 | 16.0 | 1.000 |
| 2 | 2048 | 8 | 16 | clustered | signature | 128 | 8 | 901.95 | 2732 | 25962 | 16 | 428800 | 1231 | 511 | 9.0 | 16.0 | 1.000 |
| 3 | 2048 | 8 | 16 | clustered | signature | 128 | 8 | 903.71 | 2698 | 26030 | 16 | 429456 | 1187 | 510 | 10.0 | 16.0 | 1.000 |
| 4 | 2048 | 8 | 16 | clustered | signature | 128 | 8 | 904.29 | 2679 | 26035 | 16 | 429344 | 1163 | 511 | 10.0 | 16.0 | 1.000 |
| 0 | 2048 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 279.63 | 1024 | 24331 | 24 | 389296 | 0 | 512 | 24.0 | 24.0 | 0.337 |
| 1 | 2048 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 281.24 | 1024 | 24332 | 24 | 389312 | 0 | 512 | 24.0 | 24.0 | 0.337 |
| 2 | 2048 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 280.30 | 1024 | 24331 | 24 | 389296 | 0 | 512 | 24.0 | 24.0 | 0.337 |
| 3 | 2048 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 280.54 | 1024 | 24335 | 24 | 389360 | 0 | 512 | 24.0 | 24.0 | 0.337 |
| 4 | 2048 | 8 | 16 | common_private | bounded_merge | 16 | 2 | 280.31 | 1024 | 24272 | 24 | 388352 | 0 | 512 | 24.0 | 24.0 | 0.335 |
| 0 | 2048 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 353.08 | 2560 | 20235 | 8 | 389296 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 355.48 | 2560 | 20236 | 8 | 389312 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 353.36 | 2560 | 20235 | 8 | 389296 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 353.63 | 2560 | 20239 | 8 | 389360 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 8 | 16 | common_private | bounded_merge | 32 | 4 | 353.17 | 2560 | 20208 | 8 | 388352 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 657.97 | 2304 | 18187 | 8 | 389296 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 661.99 | 2304 | 18188 | 8 | 389312 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 657.19 | 2304 | 18187 | 8 | 389296 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 658.28 | 2304 | 18191 | 8 | 389360 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 8 | 16 | common_private | bounded_merge | 64 | 8 | 657.87 | 2304 | 18176 | 8 | 388352 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 1752.60 | 2176 | 17163 | 8 | 389296 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 1761.86 | 2176 | 17164 | 8 | 389312 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 1750.43 | 2176 | 17163 | 8 | 389296 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 1752.36 | 2176 | 17167 | 8 | 389360 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 8 | 16 | common_private | bounded_merge | 128 | 16 | 1747.29 | 2176 | 17160 | 8 | 388352 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 8 | 16 | common_private | q_outer | 16 | 2 | 289.25 | 1024 | 24331 | 24 | 389296 | 0 | 512 | 24.0 | 24.0 | 0.337 |
| 1 | 2048 | 8 | 16 | common_private | q_outer | 16 | 2 | 289.51 | 1024 | 24332 | 24 | 389312 | 0 | 512 | 24.0 | 24.0 | 0.337 |
| 2 | 2048 | 8 | 16 | common_private | q_outer | 16 | 2 | 289.41 | 1024 | 24331 | 24 | 389296 | 0 | 512 | 24.0 | 24.0 | 0.337 |
| 3 | 2048 | 8 | 16 | common_private | q_outer | 16 | 2 | 289.83 | 1024 | 24335 | 24 | 389360 | 0 | 512 | 24.0 | 24.0 | 0.337 |
| 4 | 2048 | 8 | 16 | common_private | q_outer | 16 | 2 | 289.47 | 1024 | 24272 | 24 | 388352 | 0 | 512 | 24.0 | 24.0 | 0.335 |
| 0 | 2048 | 8 | 16 | common_private | q_outer | 32 | 4 | 330.84 | 512 | 20235 | 40 | 647520 | 0 | 512 | 40.0 | 40.0 | 0.202 |
| 1 | 2048 | 8 | 16 | common_private | q_outer | 32 | 4 | 334.70 | 512 | 20236 | 40 | 647552 | 0 | 512 | 40.0 | 40.0 | 0.202 |
| 2 | 2048 | 8 | 16 | common_private | q_outer | 32 | 4 | 334.99 | 512 | 20235 | 40 | 647520 | 0 | 512 | 40.0 | 40.0 | 0.202 |
| 3 | 2048 | 8 | 16 | common_private | q_outer | 32 | 4 | 334.56 | 512 | 20239 | 40 | 647648 | 0 | 512 | 40.0 | 40.0 | 0.202 |
| 4 | 2048 | 8 | 16 | common_private | q_outer | 32 | 4 | 331.07 | 512 | 20208 | 40 | 646656 | 0 | 512 | 40.0 | 40.0 | 0.201 |
| 0 | 2048 | 8 | 16 | common_private | q_outer | 64 | 8 | 642.70 | 256 | 18187 | 72 | 1163968 | 0 | 512 | 71.0 | 72.0 | 0.113 |
| 1 | 2048 | 8 | 16 | common_private | q_outer | 64 | 8 | 650.40 | 256 | 18188 | 72 | 1164032 | 0 | 512 | 71.0 | 72.0 | 0.113 |
| 2 | 2048 | 8 | 16 | common_private | q_outer | 64 | 8 | 651.18 | 256 | 18187 | 72 | 1163968 | 0 | 512 | 71.0 | 72.0 | 0.113 |
| 3 | 2048 | 8 | 16 | common_private | q_outer | 64 | 8 | 643.10 | 256 | 18191 | 72 | 1164224 | 0 | 512 | 71.0 | 72.0 | 0.113 |
| 4 | 2048 | 8 | 16 | common_private | q_outer | 64 | 8 | 642.58 | 256 | 18176 | 72 | 1163264 | 0 | 512 | 71.0 | 72.0 | 0.112 |
| 0 | 2048 | 8 | 16 | common_private | q_outer | 128 | 16 | 1744.88 | 128 | 17163 | 136 | 2196864 | 0 | 512 | 134.0 | 136.0 | 0.060 |
| 1 | 2048 | 8 | 16 | common_private | q_outer | 128 | 16 | 1748.90 | 128 | 17164 | 136 | 2196992 | 0 | 512 | 134.0 | 136.0 | 0.060 |
| 2 | 2048 | 8 | 16 | common_private | q_outer | 128 | 16 | 1747.10 | 128 | 17163 | 136 | 2196864 | 0 | 512 | 134.0 | 136.0 | 0.060 |
| 3 | 2048 | 8 | 16 | common_private | q_outer | 128 | 16 | 1737.23 | 128 | 17167 | 136 | 2197376 | 0 | 512 | 134.0 | 136.0 | 0.060 |
| 4 | 2048 | 8 | 16 | common_private | q_outer | 128 | 16 | 1761.45 | 128 | 17160 | 136 | 2196480 | 0 | 512 | 134.0 | 136.0 | 0.059 |
| 0 | 2048 | 8 | 16 | common_private | signature | 16 | 2 | 323.33 | 3072 | 24331 | 8 | 389296 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 8 | 16 | common_private | signature | 16 | 2 | 325.44 | 3072 | 24332 | 8 | 389312 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 8 | 16 | common_private | signature | 16 | 2 | 323.29 | 3072 | 24331 | 8 | 389296 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 8 | 16 | common_private | signature | 16 | 2 | 321.69 | 3072 | 24335 | 8 | 389360 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 8 | 16 | common_private | signature | 16 | 2 | 324.54 | 3072 | 24272 | 8 | 388352 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 8 | 16 | common_private | signature | 32 | 4 | 353.49 | 2560 | 20235 | 8 | 389296 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 8 | 16 | common_private | signature | 32 | 4 | 357.67 | 2560 | 20236 | 8 | 389312 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 8 | 16 | common_private | signature | 32 | 4 | 353.36 | 2560 | 20235 | 8 | 389296 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 8 | 16 | common_private | signature | 32 | 4 | 353.41 | 2560 | 20239 | 8 | 389360 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 8 | 16 | common_private | signature | 32 | 4 | 353.50 | 2560 | 20208 | 8 | 388352 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 8 | 16 | common_private | signature | 64 | 8 | 657.74 | 2304 | 18187 | 8 | 389296 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 8 | 16 | common_private | signature | 64 | 8 | 666.33 | 2304 | 18188 | 8 | 389312 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 8 | 16 | common_private | signature | 64 | 8 | 657.95 | 2304 | 18187 | 8 | 389296 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 8 | 16 | common_private | signature | 64 | 8 | 658.26 | 2304 | 18191 | 8 | 389360 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 8 | 16 | common_private | signature | 64 | 8 | 658.16 | 2304 | 18176 | 8 | 388352 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 8 | 16 | common_private | signature | 128 | 16 | 1752.58 | 2176 | 17163 | 8 | 389296 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 1 | 2048 | 8 | 16 | common_private | signature | 128 | 16 | 1766.55 | 2176 | 17164 | 8 | 389312 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 2 | 2048 | 8 | 16 | common_private | signature | 128 | 16 | 1750.98 | 2176 | 17163 | 8 | 389296 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 3 | 2048 | 8 | 16 | common_private | signature | 128 | 16 | 1752.36 | 2176 | 17167 | 8 | 389360 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 4 | 2048 | 8 | 16 | common_private | signature | 128 | 16 | 1747.43 | 2176 | 17160 | 8 | 388352 | 2048 | 512 | 8.0 | 8.0 | 1.000 |
| 0 | 2048 | 8 | 16 | random | bounded_merge | 16 | 2 | 406.83 | 1645 | 31770 | 31 | 508320 | 0 | 512 | 16.0 | 31.0 | 0.632 |
| 1 | 2048 | 8 | 16 | random | bounded_merge | 16 | 2 | 409.00 | 1659 | 31807 | 31 | 508912 | 0 | 512 | 16.0 | 31.0 | 0.644 |
| 2 | 2048 | 8 | 16 | random | bounded_merge | 16 | 2 | 408.48 | 1649 | 31821 | 31 | 509136 | 0 | 512 | 16.0 | 31.0 | 0.635 |
| 3 | 2048 | 8 | 16 | random | bounded_merge | 16 | 2 | 408.72 | 1629 | 31757 | 31 | 508112 | 0 | 512 | 16.0 | 31.0 | 0.616 |
| 4 | 2048 | 8 | 16 | random | bounded_merge | 16 | 2 | 408.73 | 1629 | 31746 | 31 | 507936 | 0 | 512 | 16.0 | 31.0 | 0.617 |
| 0 | 2048 | 8 | 16 | random | bounded_merge | 32 | 4 | 564.73 | 2091 | 30783 | 31 | 493152 | 840 | 512 | 15.0 | 29.0 | 0.483 |
| 1 | 2048 | 8 | 16 | random | bounded_merge | 32 | 4 | 566.34 | 2063 | 30855 | 31 | 494128 | 839 | 512 | 15.0 | 30.0 | 0.479 |
| 2 | 2048 | 8 | 16 | random | bounded_merge | 32 | 4 | 560.56 | 2105 | 30827 | 31 | 493552 | 878 | 512 | 15.0 | 29.0 | 0.476 |
| 3 | 2048 | 8 | 16 | random | bounded_merge | 32 | 4 | 580.45 | 2070 | 30840 | 31 | 493856 | 850 | 512 | 15.0 | 30.0 | 0.476 |
| 4 | 2048 | 8 | 16 | random | bounded_merge | 32 | 4 | 563.14 | 2143 | 30772 | 31 | 492960 | 882 | 512 | 15.0 | 29.0 | 0.503 |
| 0 | 2048 | 8 | 16 | random | bounded_merge | 64 | 4 | 555.46 | 4417 | 28927 | 27 | 466176 | 1953 | 512 | 2.0 | 14.0 | 0.893 |
| 1 | 2048 | 8 | 16 | random | bounded_merge | 64 | 4 | 554.29 | 4438 | 28953 | 16 | 466320 | 1974 | 512 | 2.0 | 14.0 | 0.893 |
| 2 | 2048 | 8 | 16 | random | bounded_merge | 64 | 4 | 546.45 | 4448 | 28981 | 16 | 466768 | 1946 | 512 | 2.0 | 14.0 | 0.892 |
| 3 | 2048 | 8 | 16 | random | bounded_merge | 64 | 4 | 554.56 | 4398 | 29020 | 16 | 466944 | 1944 | 512 | 2.0 | 14.0 | 0.892 |
| 4 | 2048 | 8 | 16 | random | bounded_merge | 64 | 4 | 543.89 | 4423 | 28910 | 16 | 466000 | 1958 | 512 | 2.0 | 14.0 | 0.893 |
| 0 | 2048 | 8 | 16 | random | bounded_merge | 128 | 8 | 1035.59 | 7081 | 25699 | 16 | 424144 | 2046 | 512 | 1.0 | 11.0 | 1.000 |
| 1 | 2048 | 8 | 16 | random | bounded_merge | 128 | 8 | 1048.99 | 7079 | 25686 | 15 | 424112 | 2047 | 512 | 1.0 | 11.0 | 1.000 |
| 2 | 2048 | 8 | 16 | random | bounded_merge | 128 | 8 | 1035.87 | 7112 | 25685 | 15 | 424688 | 2048 | 512 | 1.0 | 11.0 | 1.000 |
| 3 | 2048 | 8 | 16 | random | bounded_merge | 128 | 8 | 1052.70 | 7121 | 25679 | 16 | 423520 | 2046 | 512 | 1.0 | 11.0 | 1.000 |
| 4 | 2048 | 8 | 16 | random | bounded_merge | 128 | 8 | 1042.13 | 7022 | 25730 | 15 | 425360 | 2048 | 512 | 1.0 | 11.0 | 1.000 |
| 0 | 2048 | 8 | 16 | random | q_outer | 16 | 2 | 371.76 | 1024 | 31770 | 32 | 508320 | 0 | 512 | 31.0 | 32.0 | 0.015 |
| 1 | 2048 | 8 | 16 | random | q_outer | 16 | 2 | 375.17 | 1024 | 31807 | 32 | 508912 | 0 | 512 | 31.0 | 32.0 | 0.015 |
| 2 | 2048 | 8 | 16 | random | q_outer | 16 | 2 | 372.21 | 1024 | 31821 | 32 | 509136 | 0 | 512 | 31.0 | 32.0 | 0.016 |
| 3 | 2048 | 8 | 16 | random | q_outer | 16 | 2 | 374.41 | 1024 | 31757 | 32 | 508112 | 0 | 512 | 31.0 | 32.0 | 0.016 |
| 4 | 2048 | 8 | 16 | random | q_outer | 16 | 2 | 374.51 | 1024 | 31746 | 32 | 507936 | 0 | 512 | 31.0 | 32.0 | 0.017 |
| 0 | 2048 | 8 | 16 | random | q_outer | 32 | 4 | 506.68 | 512 | 30783 | 64 | 985056 | 0 | 512 | 60.0 | 62.9 | 0.000 |
| 1 | 2048 | 8 | 16 | random | q_outer | 32 | 4 | 511.04 | 512 | 30855 | 64 | 987360 | 0 | 512 | 60.0 | 63.0 | 0.000 |
| 2 | 2048 | 8 | 16 | random | q_outer | 32 | 4 | 506.57 | 512 | 30827 | 64 | 986464 | 0 | 512 | 60.0 | 62.0 | 0.000 |
| 3 | 2048 | 8 | 16 | random | q_outer | 32 | 4 | 502.49 | 512 | 30840 | 64 | 986880 | 0 | 512 | 60.0 | 62.0 | 0.000 |
| 4 | 2048 | 8 | 16 | random | q_outer | 32 | 4 | 510.72 | 512 | 30772 | 64 | 984704 | 0 | 512 | 60.0 | 63.0 | 0.000 |
| 0 | 2048 | 8 | 16 | random | q_outer | 64 | 8 | 1033.88 | 256 | 28927 | 121 | 1851328 | 0 | 512 | 114.0 | 118.0 | 0.000 |
| 1 | 2048 | 8 | 16 | random | q_outer | 64 | 8 | 1042.35 | 256 | 28953 | 121 | 1852992 | 0 | 512 | 113.0 | 117.0 | 0.000 |
| 2 | 2048 | 8 | 16 | random | q_outer | 64 | 8 | 1038.26 | 256 | 28981 | 122 | 1854784 | 0 | 512 | 113.0 | 118.0 | 0.000 |
| 3 | 2048 | 8 | 16 | random | q_outer | 64 | 8 | 1035.57 | 256 | 29020 | 120 | 1857280 | 0 | 512 | 114.0 | 117.0 | 0.000 |
| 4 | 2048 | 8 | 16 | random | q_outer | 64 | 8 | 1051.42 | 256 | 28910 | 122 | 1850240 | 0 | 512 | 113.0 | 117.0 | 0.000 |
| 0 | 2048 | 8 | 16 | random | q_outer | 128 | 16 | 2680.30 | 128 | 25699 | 214 | 3289472 | 0 | 512 | 201.0 | 208.0 | 0.000 |
| 1 | 2048 | 8 | 16 | random | q_outer | 128 | 16 | 2682.21 | 128 | 25686 | 214 | 3287808 | 0 | 512 | 201.0 | 208.3 | 0.000 |
| 2 | 2048 | 8 | 16 | random | q_outer | 128 | 16 | 2712.93 | 128 | 25685 | 217 | 3287680 | 0 | 512 | 201.0 | 207.3 | 0.000 |
| 3 | 2048 | 8 | 16 | random | q_outer | 128 | 16 | 2678.82 | 128 | 25679 | 215 | 3286912 | 0 | 512 | 200.0 | 208.0 | 0.000 |
| 4 | 2048 | 8 | 16 | random | q_outer | 128 | 16 | 2679.43 | 128 | 25730 | 216 | 3293440 | 0 | 512 | 201.0 | 207.3 | 0.000 |
| 0 | 2048 | 8 | 16 | random | signature | 16 | 2 | 383.61 | 2450 | 31770 | 16 | 508320 | 804 | 512 | 15.0 | 16.0 | 1.000 |
| 1 | 2048 | 8 | 16 | random | signature | 16 | 2 | 384.71 | 2437 | 31807 | 16 | 508912 | 778 | 512 | 15.0 | 16.0 | 1.000 |
| 2 | 2048 | 8 | 16 | random | signature | 16 | 2 | 383.41 | 2447 | 31821 | 16 | 509136 | 798 | 512 | 15.0 | 16.0 | 1.000 |
| 3 | 2048 | 8 | 16 | random | signature | 16 | 2 | 385.11 | 2467 | 31757 | 16 | 508112 | 838 | 512 | 15.0 | 16.0 | 1.000 |
| 4 | 2048 | 8 | 16 | random | signature | 16 | 2 | 384.88 | 2467 | 31746 | 16 | 507936 | 838 | 512 | 15.0 | 16.0 | 1.000 |
| 0 | 2048 | 8 | 16 | random | signature | 32 | 4 | 520.62 | 3224 | 30783 | 16 | 493152 | 1579 | 512 | 13.0 | 16.0 | 1.000 |
| 1 | 2048 | 8 | 16 | random | signature | 32 | 4 | 521.12 | 3201 | 30855 | 16 | 494128 | 1581 | 512 | 14.0 | 16.0 | 1.000 |
| 2 | 2048 | 8 | 16 | random | signature | 32 | 4 | 519.78 | 3254 | 30827 | 16 | 493552 | 1613 | 512 | 13.0 | 16.0 | 1.000 |
| 3 | 2048 | 8 | 16 | random | signature | 32 | 4 | 521.88 | 3214 | 30840 | 16 | 493856 | 1574 | 512 | 13.0 | 16.0 | 1.000 |
| 4 | 2048 | 8 | 16 | random | signature | 32 | 4 | 518.82 | 3238 | 30772 | 16 | 492960 | 1582 | 512 | 13.0 | 16.0 | 1.000 |
| 0 | 2048 | 8 | 16 | random | signature | 64 | 4 | 527.42 | 4673 | 28927 | 16 | 466176 | 1982 | 512 | 2.0 | 14.0 | 1.000 |
| 1 | 2048 | 8 | 16 | random | signature | 64 | 4 | 525.55 | 4694 | 28953 | 16 | 466320 | 2003 | 512 | 2.0 | 14.0 | 1.000 |
| 2 | 2048 | 8 | 16 | random | signature | 64 | 4 | 520.25 | 4704 | 28981 | 16 | 466768 | 1986 | 512 | 2.0 | 14.0 | 1.000 |
| 3 | 2048 | 8 | 16 | random | signature | 64 | 4 | 521.42 | 4654 | 29020 | 16 | 466944 | 1980 | 512 | 2.0 | 14.0 | 1.000 |
| 4 | 2048 | 8 | 16 | random | signature | 64 | 4 | 521.83 | 4679 | 28910 | 16 | 466000 | 1994 | 512 | 2.0 | 14.0 | 1.000 |
| 0 | 2048 | 8 | 16 | random | signature | 128 | 8 | 1019.81 | 7081 | 25699 | 16 | 424144 | 2046 | 512 | 1.0 | 11.0 | 1.000 |
| 1 | 2048 | 8 | 16 | random | signature | 128 | 8 | 1026.00 | 7079 | 25686 | 15 | 424112 | 2047 | 512 | 1.0 | 11.0 | 1.000 |
| 2 | 2048 | 8 | 16 | random | signature | 128 | 8 | 1010.52 | 7112 | 25685 | 15 | 424688 | 2048 | 512 | 1.0 | 11.0 | 1.000 |
| 3 | 2048 | 8 | 16 | random | signature | 128 | 8 | 1021.76 | 7121 | 25679 | 16 | 423520 | 2046 | 512 | 1.0 | 11.0 | 1.000 |
| 4 | 2048 | 8 | 16 | random | signature | 128 | 8 | 1015.38 | 7022 | 25730 | 15 | 425360 | 2048 | 512 | 1.0 | 11.0 | 1.000 |
| 0 | 2048 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 795.55 | 1931 | 63469 | 63 | 1015504 | 0 | 510 | 32.0 | 32.0 | 0.942 |
| 1 | 2048 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 797.73 | 1916 | 63447 | 63 | 1015152 | 0 | 511 | 32.0 | 32.0 | 0.929 |
| 2 | 2048 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 800.03 | 1907 | 63269 | 63 | 1012304 | 0 | 511 | 32.0 | 32.0 | 0.925 |
| 3 | 2048 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 833.33 | 1924 | 63393 | 63 | 1014288 | 0 | 510 | 32.0 | 32.0 | 0.940 |
| 4 | 2048 | 8 | 32 | clustered | bounded_merge | 16 | 2 | 801.13 | 1932 | 63533 | 63 | 1016528 | 0 | 511 | 32.0 | 32.0 | 0.941 |
| 0 | 2048 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 1027.00 | 1772 | 59221 | 63 | 953616 | 114 | 510 | 32.0 | 42.0 | 0.840 |
| 1 | 2048 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 1036.42 | 1768 | 59377 | 63 | 954976 | 120 | 511 | 32.0 | 43.0 | 0.822 |
| 2 | 2048 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 1051.84 | 1737 | 59168 | 63 | 949424 | 89 | 511 | 32.0 | 44.0 | 0.815 |
| 3 | 2048 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 1076.55 | 1762 | 59630 | 63 | 957904 | 92 | 510 | 32.0 | 43.0 | 0.832 |
| 4 | 2048 | 8 | 32 | clustered | bounded_merge | 32 | 4 | 1045.26 | 1779 | 59588 | 63 | 957376 | 99 | 511 | 32.0 | 41.0 | 0.842 |
| 0 | 2048 | 8 | 32 | clustered | bounded_merge | 64 | 8 | 1949.44 | 2517 | 52087 | 54 | 859072 | 1087 | 510 | 23.0 | 32.0 | 0.931 |
| 1 | 2048 | 8 | 32 | clustered | bounded_merge | 64 | 8 | 1944.95 | 2523 | 51913 | 32 | 859104 | 1068 | 511 | 24.0 | 32.0 | 0.938 |
| 2 | 2048 | 8 | 32 | clustered | bounded_merge | 64 | 8 | 1937.13 | 2523 | 52201 | 57 | 859824 | 1088 | 511 | 23.0 | 32.0 | 0.931 |
| 3 | 2048 | 8 | 32 | clustered | bounded_merge | 64 | 4 | 925.69 | 2487 | 52683 | 49 | 866144 | 1040 | 510 | 24.0 | 32.0 | 0.934 |
| 4 | 2048 | 8 | 32 | clustered | bounded_merge | 64 | 8 | 1984.41 | 2479 | 52419 | 54 | 860848 | 1029 | 511 | 24.0 | 32.0 | 0.929 |
| 0 | 2048 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 1564.30 | 3109 | 41003 | 54 | 746496 | 1731 | 510 | 11.0 | 31.0 | 0.959 |
| 1 | 2048 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 1581.51 | 3107 | 41371 | 57 | 751568 | 1692 | 511 | 11.0 | 32.0 | 0.960 |
| 2 | 2048 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 1568.63 | 3107 | 41239 | 54 | 752080 | 1721 | 511 | 11.0 | 32.0 | 0.960 |
| 3 | 2048 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 1631.99 | 3094 | 41867 | 57 | 755200 | 1707 | 510 | 11.0 | 32.0 | 0.955 |
| 4 | 2048 | 8 | 32 | clustered | bounded_merge | 128 | 8 | 1661.50 | 3047 | 42088 | 57 | 757472 | 1665 | 511 | 11.0 | 32.0 | 0.958 |
| 0 | 2048 | 8 | 32 | clustered | q_outer | 16 | 2 | 752.40 | 1024 | 63469 | 64 | 1015504 | 0 | 510 | 64.0 | 64.0 | 0.030 |
| 1 | 2048 | 8 | 32 | clustered | q_outer | 16 | 2 | 758.28 | 1024 | 63447 | 64 | 1015152 | 0 | 511 | 64.0 | 64.0 | 0.031 |
| 2 | 2048 | 8 | 32 | clustered | q_outer | 16 | 2 | 752.65 | 1024 | 63269 | 64 | 1012304 | 0 | 511 | 64.0 | 64.0 | 0.034 |
| 3 | 2048 | 8 | 32 | clustered | q_outer | 16 | 2 | 757.44 | 1024 | 63393 | 64 | 1014288 | 0 | 510 | 64.0 | 64.0 | 0.032 |
| 4 | 2048 | 8 | 32 | clustered | q_outer | 16 | 2 | 759.86 | 1024 | 63533 | 64 | 1016528 | 0 | 511 | 64.0 | 64.0 | 0.029 |
| 0 | 2048 | 8 | 32 | clustered | q_outer | 32 | 4 | 1028.15 | 512 | 59221 | 128 | 1895072 | 0 | 510 | 122.0 | 128.0 | 0.000 |
| 1 | 2048 | 8 | 32 | clustered | q_outer | 32 | 4 | 1028.87 | 512 | 59377 | 128 | 1900064 | 0 | 511 | 122.0 | 128.0 | 0.000 |
| 2 | 2048 | 8 | 32 | clustered | q_outer | 32 | 4 | 1026.50 | 512 | 59168 | 128 | 1893376 | 0 | 511 | 120.0 | 128.0 | 0.000 |
| 3 | 2048 | 8 | 32 | clustered | q_outer | 32 | 4 | 1041.48 | 512 | 59630 | 128 | 1908160 | 0 | 510 | 122.0 | 128.0 | 0.000 |
| 4 | 2048 | 8 | 32 | clustered | q_outer | 32 | 4 | 1032.74 | 512 | 59588 | 128 | 1906816 | 0 | 511 | 122.0 | 128.0 | 0.000 |
| 0 | 2048 | 8 | 32 | clustered | q_outer | 64 | 8 | 2009.54 | 256 | 52087 | 256 | 3333568 | 0 | 510 | 206.0 | 232.0 | 0.000 |
| 1 | 2048 | 8 | 32 | clustered | q_outer | 64 | 8 | 1998.38 | 256 | 51913 | 256 | 3322432 | 0 | 511 | 205.0 | 231.0 | 0.000 |
| 2 | 2048 | 8 | 32 | clustered | q_outer | 64 | 8 | 2069.95 | 256 | 52201 | 256 | 3340864 | 0 | 511 | 204.0 | 236.5 | 0.000 |
| 3 | 2048 | 8 | 32 | clustered | q_outer | 64 | 8 | 2206.91 | 256 | 52683 | 253 | 3371712 | 0 | 510 | 208.0 | 234.5 | 0.000 |
| 4 | 2048 | 8 | 32 | clustered | q_outer | 64 | 8 | 1990.14 | 256 | 52419 | 256 | 3354816 | 0 | 511 | 207.5 | 232.0 | 0.000 |
| 0 | 2048 | 8 | 32 | clustered | q_outer | 128 | 16 | 4792.61 | 128 | 41003 | 381 | 5248384 | 0 | 510 | 320.0 | 363.9 | 0.000 |
| 1 | 2048 | 8 | 32 | clustered | q_outer | 128 | 16 | 4786.61 | 128 | 41371 | 384 | 5295488 | 0 | 511 | 322.0 | 358.0 | 0.000 |
| 2 | 2048 | 8 | 32 | clustered | q_outer | 128 | 16 | 5026.12 | 128 | 41239 | 412 | 5278592 | 0 | 511 | 324.0 | 360.3 | 0.000 |
| 3 | 2048 | 8 | 32 | clustered | q_outer | 128 | 16 | 4977.77 | 128 | 41867 | 396 | 5358976 | 0 | 510 | 329.5 | 369.2 | 0.000 |
| 4 | 2048 | 8 | 32 | clustered | q_outer | 128 | 16 | 4938.70 | 128 | 42088 | 396 | 5387264 | 0 | 511 | 335.5 | 367.3 | 0.000 |
| 0 | 2048 | 8 | 32 | clustered | signature | 16 | 2 | 747.52 | 2160 | 63469 | 32 | 1015504 | 226 | 510 | 32.0 | 32.0 | 1.000 |
| 1 | 2048 | 8 | 32 | clustered | signature | 16 | 2 | 746.33 | 2172 | 63447 | 32 | 1015152 | 253 | 511 | 32.0 | 32.0 | 1.000 |
| 2 | 2048 | 8 | 32 | clustered | signature | 16 | 2 | 748.75 | 2183 | 63269 | 32 | 1012304 | 274 | 511 | 32.0 | 32.0 | 1.000 |
| 3 | 2048 | 8 | 32 | clustered | signature | 16 | 2 | 749.49 | 2164 | 63393 | 32 | 1014288 | 238 | 510 | 32.0 | 32.0 | 1.000 |
| 4 | 2048 | 8 | 32 | clustered | signature | 16 | 2 | 750.70 | 2156 | 63533 | 32 | 1016528 | 220 | 511 | 32.0 | 32.0 | 1.000 |
| 0 | 2048 | 8 | 32 | clustered | signature | 32 | 4 | 939.75 | 2388 | 59221 | 32 | 953616 | 658 | 510 | 32.0 | 32.0 | 1.000 |
| 1 | 2048 | 8 | 32 | clustered | signature | 32 | 4 | 943.22 | 2418 | 59377 | 32 | 954976 | 697 | 511 | 32.0 | 32.0 | 1.000 |
| 2 | 2048 | 8 | 32 | clustered | signature | 32 | 4 | 939.42 | 2416 | 59168 | 32 | 949424 | 714 | 511 | 32.0 | 32.0 | 1.000 |
| 3 | 2048 | 8 | 32 | clustered | signature | 32 | 4 | 968.15 | 2387 | 59630 | 32 | 957904 | 656 | 510 | 32.0 | 32.0 | 1.000 |
| 4 | 2048 | 8 | 32 | clustered | signature | 32 | 4 | 948.08 | 2379 | 59588 | 32 | 957376 | 633 | 511 | 32.0 | 32.0 | 1.000 |
| 0 | 2048 | 8 | 32 | clustered | signature | 64 | 8 | 1723.85 | 2765 | 52087 | 32 | 859072 | 1250 | 510 | 19.0 | 32.0 | 1.000 |
| 1 | 2048 | 8 | 32 | clustered | signature | 64 | 8 | 1790.63 | 2772 | 51913 | 32 | 859104 | 1237 | 511 | 19.0 | 32.0 | 1.000 |
| 2 | 2048 | 8 | 32 | clustered | signature | 64 | 8 | 1788.13 | 2771 | 52201 | 32 | 859824 | 1254 | 511 | 19.0 | 32.0 | 1.000 |
| 3 | 2048 | 8 | 32 | clustered | signature | 64 | 4 | 893.48 | 2735 | 52683 | 32 | 866144 | 1200 | 510 | 20.0 | 32.0 | 1.000 |
| 4 | 2048 | 8 | 32 | clustered | signature | 64 | 8 | 1794.41 | 2730 | 52419 | 32 | 860848 | 1215 | 511 | 19.0 | 32.0 | 1.000 |
| 0 | 2048 | 8 | 32 | clustered | signature | 128 | 8 | 1418.43 | 3231 | 41003 | 32 | 746496 | 1775 | 510 | 10.0 | 29.0 | 1.000 |
| 1 | 2048 | 8 | 32 | clustered | signature | 128 | 8 | 1455.39 | 3226 | 41371 | 32 | 751568 | 1745 | 511 | 10.0 | 30.0 | 1.000 |
| 2 | 2048 | 8 | 32 | clustered | signature | 128 | 8 | 1442.32 | 3229 | 41239 | 32 | 752064 | 1764 | 511 | 10.0 | 30.0 | 1.000 |
| 3 | 2048 | 8 | 32 | clustered | signature | 128 | 8 | 1497.02 | 3215 | 41867 | 32 | 755200 | 1757 | 510 | 11.0 | 30.0 | 1.000 |
| 4 | 2048 | 8 | 32 | clustered | signature | 128 | 8 | 1482.59 | 3171 | 42088 | 32 | 757472 | 1720 | 511 | 11.0 | 31.0 | 1.000 |
| 0 | 2048 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 538.49 | 1024 | 48659 | 48 | 778544 | 0 | 512 | 48.0 | 48.0 | 0.337 |
| 1 | 2048 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 539.46 | 1024 | 48657 | 48 | 778512 | 0 | 512 | 48.0 | 48.0 | 0.337 |
| 2 | 2048 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 534.13 | 1024 | 48118 | 48 | 769888 | 0 | 512 | 47.0 | 48.0 | 0.329 |
| 3 | 2048 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 539.38 | 1024 | 48658 | 48 | 778528 | 0 | 512 | 48.0 | 48.0 | 0.337 |
| 4 | 2048 | 8 | 32 | common_private | bounded_merge | 16 | 2 | 539.92 | 1024 | 48655 | 48 | 778480 | 0 | 512 | 48.0 | 48.0 | 0.337 |
| 0 | 2048 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 638.63 | 2560 | 40467 | 16 | 778544 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 639.28 | 2560 | 40465 | 16 | 778512 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 636.19 | 2560 | 40214 | 16 | 769888 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 639.67 | 2560 | 40466 | 16 | 778528 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 8 | 32 | common_private | bounded_merge | 32 | 4 | 665.49 | 2553 | 40463 | 27 | 778480 | 2048 | 512 | 16.0 | 16.0 | 0.995 |
| 0 | 2048 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 1224.91 | 2304 | 36371 | 16 | 778544 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 1230.47 | 2304 | 36369 | 16 | 778512 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 1225.22 | 2304 | 36262 | 16 | 769888 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 1225.60 | 2304 | 36370 | 16 | 778528 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 8 | 32 | common_private | bounded_merge | 64 | 8 | 1332.69 | 2297 | 36367 | 27 | 778480 | 2048 | 512 | 16.0 | 16.0 | 0.995 |
| 0 | 2048 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 3240.02 | 2176 | 34323 | 16 | 778544 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 3241.53 | 2176 | 34321 | 16 | 778512 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 3243.76 | 2176 | 34286 | 16 | 769888 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 3245.12 | 2176 | 34322 | 16 | 778528 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 8 | 32 | common_private | bounded_merge | 128 | 16 | 3589.87 | 2174 | 34319 | 27 | 778480 | 2048 | 512 | 16.0 | 16.0 | 0.998 |
| 0 | 2048 | 8 | 32 | common_private | q_outer | 16 | 2 | 554.62 | 1024 | 48659 | 48 | 778544 | 0 | 512 | 48.0 | 48.0 | 0.337 |
| 1 | 2048 | 8 | 32 | common_private | q_outer | 16 | 2 | 554.17 | 1024 | 48657 | 48 | 778512 | 0 | 512 | 48.0 | 48.0 | 0.337 |
| 2 | 2048 | 8 | 32 | common_private | q_outer | 16 | 2 | 549.37 | 1024 | 48118 | 48 | 769888 | 0 | 512 | 47.0 | 48.0 | 0.329 |
| 3 | 2048 | 8 | 32 | common_private | q_outer | 16 | 2 | 557.09 | 1024 | 48658 | 48 | 778528 | 0 | 512 | 48.0 | 48.0 | 0.337 |
| 4 | 2048 | 8 | 32 | common_private | q_outer | 16 | 2 | 557.14 | 1024 | 48655 | 48 | 778480 | 0 | 512 | 48.0 | 48.0 | 0.337 |
| 0 | 2048 | 8 | 32 | common_private | q_outer | 32 | 4 | 643.24 | 512 | 40467 | 80 | 1294944 | 0 | 512 | 79.0 | 80.0 | 0.202 |
| 1 | 2048 | 8 | 32 | common_private | q_outer | 32 | 4 | 643.06 | 512 | 40465 | 80 | 1294880 | 0 | 512 | 79.0 | 80.0 | 0.202 |
| 2 | 2048 | 8 | 32 | common_private | q_outer | 32 | 4 | 639.83 | 512 | 40214 | 80 | 1286848 | 0 | 512 | 79.0 | 80.0 | 0.197 |
| 3 | 2048 | 8 | 32 | common_private | q_outer | 32 | 4 | 653.45 | 512 | 40466 | 80 | 1294912 | 0 | 512 | 79.0 | 80.0 | 0.202 |
| 4 | 2048 | 8 | 32 | common_private | q_outer | 32 | 4 | 654.24 | 512 | 40463 | 80 | 1294816 | 0 | 512 | 79.0 | 80.0 | 0.202 |
| 0 | 2048 | 8 | 32 | common_private | q_outer | 64 | 8 | 1257.39 | 256 | 36371 | 144 | 2327744 | 0 | 512 | 143.0 | 144.0 | 0.113 |
| 1 | 2048 | 8 | 32 | common_private | q_outer | 64 | 8 | 1254.03 | 256 | 36369 | 144 | 2327616 | 0 | 512 | 142.0 | 144.0 | 0.113 |
| 2 | 2048 | 8 | 32 | common_private | q_outer | 64 | 8 | 1257.55 | 256 | 36262 | 144 | 2320768 | 0 | 512 | 142.0 | 144.0 | 0.109 |
| 3 | 2048 | 8 | 32 | common_private | q_outer | 64 | 8 | 1258.73 | 256 | 36370 | 144 | 2327680 | 0 | 512 | 142.0 | 144.0 | 0.113 |
| 4 | 2048 | 8 | 32 | common_private | q_outer | 64 | 8 | 1262.32 | 256 | 36367 | 144 | 2327488 | 0 | 512 | 143.0 | 144.0 | 0.113 |
| 0 | 2048 | 8 | 32 | common_private | q_outer | 128 | 16 | 3432.18 | 128 | 34323 | 272 | 4393344 | 0 | 512 | 268.0 | 271.0 | 0.060 |
| 1 | 2048 | 8 | 32 | common_private | q_outer | 128 | 16 | 3435.60 | 128 | 34321 | 272 | 4393088 | 0 | 512 | 268.0 | 272.0 | 0.060 |
| 2 | 2048 | 8 | 32 | common_private | q_outer | 128 | 16 | 3433.78 | 128 | 34286 | 272 | 4388608 | 0 | 512 | 268.0 | 271.0 | 0.058 |
| 3 | 2048 | 8 | 32 | common_private | q_outer | 128 | 16 | 3407.72 | 128 | 34322 | 272 | 4393216 | 0 | 512 | 268.0 | 272.0 | 0.060 |
| 4 | 2048 | 8 | 32 | common_private | q_outer | 128 | 16 | 3423.19 | 128 | 34319 | 272 | 4392832 | 0 | 512 | 268.0 | 271.0 | 0.060 |
| 0 | 2048 | 8 | 32 | common_private | signature | 16 | 2 | 583.13 | 3072 | 48659 | 16 | 778544 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 8 | 32 | common_private | signature | 16 | 2 | 590.86 | 3072 | 48657 | 16 | 778512 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 8 | 32 | common_private | signature | 16 | 2 | 576.26 | 3072 | 48118 | 16 | 769888 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 8 | 32 | common_private | signature | 16 | 2 | 582.39 | 3072 | 48658 | 16 | 778528 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 8 | 32 | common_private | signature | 16 | 2 | 583.23 | 3072 | 48655 | 16 | 778480 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 2048 | 8 | 32 | common_private | signature | 32 | 4 | 638.20 | 2560 | 40467 | 16 | 778544 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 8 | 32 | common_private | signature | 32 | 4 | 639.65 | 2560 | 40465 | 16 | 778512 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 8 | 32 | common_private | signature | 32 | 4 | 635.86 | 2560 | 40214 | 16 | 769888 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 8 | 32 | common_private | signature | 32 | 4 | 639.13 | 2560 | 40466 | 16 | 778528 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 8 | 32 | common_private | signature | 32 | 4 | 639.08 | 2560 | 40463 | 16 | 778480 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 2048 | 8 | 32 | common_private | signature | 64 | 8 | 1225.17 | 2304 | 36371 | 16 | 778544 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 8 | 32 | common_private | signature | 64 | 8 | 1229.38 | 2304 | 36369 | 16 | 778512 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 8 | 32 | common_private | signature | 64 | 8 | 1224.85 | 2304 | 36262 | 16 | 769888 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 8 | 32 | common_private | signature | 64 | 8 | 1225.76 | 2304 | 36370 | 16 | 778528 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 8 | 32 | common_private | signature | 64 | 8 | 1226.48 | 2304 | 36367 | 16 | 778480 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 2048 | 8 | 32 | common_private | signature | 128 | 16 | 3240.65 | 2176 | 34323 | 16 | 778544 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 1 | 2048 | 8 | 32 | common_private | signature | 128 | 16 | 3241.49 | 2176 | 34321 | 16 | 778512 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 2 | 2048 | 8 | 32 | common_private | signature | 128 | 16 | 3244.04 | 2176 | 34286 | 16 | 769888 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 3 | 2048 | 8 | 32 | common_private | signature | 128 | 16 | 3246.30 | 2176 | 34322 | 16 | 778528 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 4 | 2048 | 8 | 32 | common_private | signature | 128 | 16 | 3246.41 | 2176 | 34319 | 16 | 778480 | 2048 | 512 | 16.0 | 16.0 | 1.000 |
| 0 | 2048 | 8 | 32 | random | bounded_merge | 16 | 2 | 763.62 | 1148 | 62494 | 63 | 999904 | 0 | 512 | 60.0 | 62.3 | 0.159 |
| 1 | 2048 | 8 | 32 | random | bounded_merge | 16 | 2 | 767.04 | 1133 | 62561 | 63 | 1000976 | 0 | 512 | 61.0 | 63.0 | 0.143 |
| 2 | 2048 | 8 | 32 | random | bounded_merge | 16 | 2 | 765.90 | 1140 | 62648 | 63 | 1002368 | 0 | 512 | 61.0 | 63.0 | 0.148 |
| 3 | 2048 | 8 | 32 | random | bounded_merge | 16 | 2 | 764.75 | 1139 | 62508 | 63 | 1000128 | 0 | 512 | 60.0 | 63.0 | 0.148 |
| 4 | 2048 | 8 | 32 | random | bounded_merge | 16 | 2 | 768.25 | 1170 | 62534 | 63 | 1000544 | 0 | 512 | 60.0 | 62.0 | 0.179 |
| 0 | 2048 | 8 | 32 | random | bounded_merge | 32 | 4 | 996.08 | 4278 | 58732 | 60 | 943504 | 2007 | 512 | 4.0 | 28.0 | 0.738 |
| 1 | 2048 | 8 | 32 | random | bounded_merge | 32 | 4 | 1001.08 | 4323 | 58702 | 57 | 943088 | 2024 | 512 | 4.0 | 28.0 | 0.745 |
| 2 | 2048 | 8 | 32 | random | bounded_merge | 32 | 4 | 1007.99 | 4270 | 58818 | 57 | 944944 | 2020 | 512 | 4.0 | 28.0 | 0.736 |
| 3 | 2048 | 8 | 32 | random | bounded_merge | 32 | 4 | 1090.49 | 4317 | 58668 | 59 | 943216 | 2010 | 512 | 4.0 | 28.0 | 0.740 |
| 4 | 2048 | 8 | 32 | random | bounded_merge | 32 | 4 | 1024.57 | 4220 | 58789 | 63 | 944480 | 1999 | 512 | 4.0 | 28.0 | 0.731 |
| 0 | 2048 | 8 | 32 | random | bounded_merge | 64 | 8 | 2110.59 | 8706 | 52077 | 30 | 855744 | 2048 | 512 | 2.0 | 21.0 | 0.973 |
| 1 | 2048 | 8 | 32 | random | bounded_merge | 64 | 8 | 2096.65 | 8768 | 52062 | 28 | 855984 | 2048 | 512 | 2.0 | 21.0 | 0.974 |
| 2 | 2048 | 8 | 32 | random | bounded_merge | 64 | 8 | 2094.98 | 8781 | 52037 | 30 | 855376 | 2048 | 512 | 2.0 | 21.0 | 0.976 |
| 3 | 2048 | 8 | 32 | random | bounded_merge | 64 | 8 | 2089.50 | 8800 | 51936 | 28 | 854640 | 2048 | 512 | 2.0 | 21.0 | 0.977 |
| 4 | 2048 | 8 | 32 | random | bounded_merge | 64 | 8 | 2091.39 | 8708 | 52079 | 28 | 855344 | 2048 | 512 | 2.0 | 21.0 | 0.974 |
| 0 | 2048 | 8 | 32 | random | bounded_merge | 128 | 8 | 1781.96 | 15378 | 41567 | 22 | 746288 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 2048 | 8 | 32 | random | bounded_merge | 128 | 8 | 1787.09 | 15390 | 41589 | 21 | 746592 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 2048 | 8 | 32 | random | bounded_merge | 128 | 8 | 1778.43 | 15380 | 41550 | 20 | 747104 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 2048 | 8 | 32 | random | bounded_merge | 128 | 8 | 1761.34 | 15388 | 41486 | 22 | 747632 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 2048 | 8 | 32 | random | bounded_merge | 128 | 8 | 1788.11 | 15333 | 41588 | 21 | 747440 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 0 | 2048 | 8 | 32 | random | q_outer | 16 | 2 | 710.61 | 1024 | 62494 | 64 | 999904 | 0 | 512 | 61.0 | 63.0 | 0.034 |
| 1 | 2048 | 8 | 32 | random | q_outer | 16 | 2 | 711.69 | 1024 | 62561 | 64 | 1000976 | 0 | 512 | 61.0 | 63.0 | 0.033 |
| 2 | 2048 | 8 | 32 | random | q_outer | 16 | 2 | 712.41 | 1024 | 62648 | 64 | 1002368 | 0 | 512 | 61.0 | 63.0 | 0.031 |
| 3 | 2048 | 8 | 32 | random | q_outer | 16 | 2 | 716.02 | 1024 | 62508 | 64 | 1000128 | 0 | 512 | 61.0 | 63.0 | 0.033 |
| 4 | 2048 | 8 | 32 | random | q_outer | 16 | 2 | 715.96 | 1024 | 62534 | 64 | 1000544 | 0 | 512 | 61.0 | 63.0 | 0.032 |
| 0 | 2048 | 8 | 32 | random | q_outer | 32 | 4 | 952.06 | 512 | 58732 | 124 | 1879424 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 1 | 2048 | 8 | 32 | random | q_outer | 32 | 4 | 945.89 | 512 | 58702 | 122 | 1878464 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 2 | 2048 | 8 | 32 | random | q_outer | 32 | 4 | 952.23 | 512 | 58818 | 122 | 1882176 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 3 | 2048 | 8 | 32 | random | q_outer | 32 | 4 | 947.55 | 512 | 58668 | 123 | 1877376 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 4 | 2048 | 8 | 32 | random | q_outer | 32 | 4 | 949.98 | 512 | 58789 | 124 | 1881248 | 0 | 512 | 115.0 | 119.0 | 0.000 |
| 0 | 2048 | 8 | 32 | random | q_outer | 64 | 8 | 1828.51 | 256 | 52077 | 217 | 3332928 | 0 | 512 | 203.5 | 210.0 | 0.000 |
| 1 | 2048 | 8 | 32 | random | q_outer | 64 | 8 | 1823.31 | 256 | 52062 | 217 | 3331968 | 0 | 512 | 203.0 | 211.0 | 0.000 |
| 2 | 2048 | 8 | 32 | random | q_outer | 64 | 8 | 1848.02 | 256 | 52037 | 217 | 3330368 | 0 | 512 | 203.0 | 211.0 | 0.000 |
| 3 | 2048 | 8 | 32 | random | q_outer | 64 | 8 | 1838.75 | 256 | 51936 | 218 | 3323904 | 0 | 512 | 203.0 | 210.0 | 0.000 |
| 4 | 2048 | 8 | 32 | random | q_outer | 64 | 8 | 1839.69 | 256 | 52079 | 219 | 3333056 | 0 | 512 | 203.0 | 210.0 | 0.000 |
| 0 | 2048 | 8 | 32 | random | q_outer | 128 | 16 | 4377.51 | 128 | 41567 | 354 | 5320576 | 0 | 512 | 325.0 | 334.0 | 0.000 |
| 1 | 2048 | 8 | 32 | random | q_outer | 128 | 16 | 4228.31 | 128 | 41589 | 347 | 5323392 | 0 | 512 | 324.0 | 335.0 | 0.000 |
| 2 | 2048 | 8 | 32 | random | q_outer | 128 | 16 | 4198.44 | 128 | 41550 | 346 | 5318400 | 0 | 512 | 325.0 | 334.0 | 0.000 |
| 3 | 2048 | 8 | 32 | random | q_outer | 128 | 16 | 4276.58 | 128 | 41486 | 351 | 5310208 | 0 | 512 | 323.0 | 336.0 | 0.000 |
| 4 | 2048 | 8 | 32 | random | q_outer | 128 | 16 | 4219.20 | 128 | 41588 | 343 | 5323264 | 0 | 512 | 324.0 | 336.0 | 0.000 |
| 0 | 2048 | 8 | 32 | random | signature | 16 | 2 | 733.72 | 2948 | 62494 | 32 | 999904 | 1800 | 512 | 29.0 | 31.0 | 1.000 |
| 1 | 2048 | 8 | 32 | random | signature | 16 | 2 | 733.55 | 2963 | 62561 | 32 | 1000976 | 1830 | 512 | 29.0 | 31.0 | 1.000 |
| 2 | 2048 | 8 | 32 | random | signature | 16 | 2 | 733.15 | 2956 | 62648 | 32 | 1002368 | 1816 | 512 | 29.0 | 31.0 | 1.000 |
| 3 | 2048 | 8 | 32 | random | signature | 16 | 2 | 729.73 | 2957 | 62508 | 32 | 1000128 | 1818 | 512 | 29.0 | 31.0 | 1.000 |
| 4 | 2048 | 8 | 32 | random | signature | 16 | 2 | 733.12 | 2926 | 62534 | 32 | 1000544 | 1756 | 512 | 29.0 | 31.0 | 1.000 |
| 0 | 2048 | 8 | 32 | random | signature | 32 | 4 | 969.36 | 4865 | 58732 | 32 | 943504 | 2044 | 512 | 3.0 | 28.0 | 1.000 |
| 1 | 2048 | 8 | 32 | random | signature | 32 | 4 | 966.63 | 4899 | 58702 | 32 | 943088 | 2047 | 512 | 3.0 | 27.0 | 1.000 |
| 2 | 2048 | 8 | 32 | random | signature | 32 | 4 | 966.03 | 4863 | 58818 | 31 | 944944 | 2047 | 512 | 3.0 | 28.0 | 1.000 |
| 3 | 2048 | 8 | 32 | random | signature | 32 | 4 | 967.44 | 4905 | 58668 | 32 | 943216 | 2043 | 512 | 3.0 | 27.0 | 1.000 |
| 4 | 2048 | 8 | 32 | random | signature | 32 | 4 | 964.70 | 4825 | 58789 | 32 | 944480 | 2041 | 512 | 3.0 | 28.0 | 1.000 |
| 0 | 2048 | 8 | 32 | random | signature | 64 | 8 | 1885.39 | 8773 | 52077 | 30 | 855744 | 2048 | 512 | 2.0 | 21.0 | 1.000 |
| 1 | 2048 | 8 | 32 | random | signature | 64 | 8 | 1878.34 | 8831 | 52062 | 28 | 855984 | 2048 | 512 | 2.0 | 21.0 | 1.000 |
| 2 | 2048 | 8 | 32 | random | signature | 64 | 8 | 1879.02 | 8839 | 52037 | 29 | 855376 | 2048 | 512 | 2.0 | 21.0 | 1.000 |
| 3 | 2048 | 8 | 32 | random | signature | 64 | 8 | 1879.33 | 8858 | 51936 | 28 | 854640 | 2048 | 512 | 2.0 | 21.0 | 1.000 |
| 4 | 2048 | 8 | 32 | random | signature | 64 | 8 | 1876.65 | 8772 | 52079 | 28 | 855344 | 2048 | 512 | 2.0 | 21.0 | 1.000 |
| 0 | 2048 | 8 | 32 | random | signature | 128 | 8 | 1782.07 | 15378 | 41567 | 22 | 746288 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 1 | 2048 | 8 | 32 | random | signature | 128 | 8 | 1785.98 | 15390 | 41589 | 21 | 746592 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 2 | 2048 | 8 | 32 | random | signature | 128 | 8 | 1779.19 | 15380 | 41550 | 20 | 747104 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 3 | 2048 | 8 | 32 | random | signature | 128 | 8 | 1752.60 | 15388 | 41486 | 22 | 747632 | 2048 | 512 | 1.0 | 10.0 | 1.000 |
| 4 | 2048 | 8 | 32 | random | signature | 128 | 8 | 1787.21 | 15333 | 41588 | 21 | 747440 | 2048 | 512 | 1.0 | 10.0 | 1.000 |


## E3 — 교정된 선택기와 oracle (§11 E3)

교정 seed [0, 1, 2, 3, 4] / 평가 seed [101, 102, 103, 104, 105, 106, 107, 108, 109] / 형상 ['128:16:4', '256:16:8', '512:16:4', '512:16:8', '2048:16:8', '512:16:16', '256:16:16'] / 패턴 ['common_private', 'random', 'clustered']. 셀 189개.

§7.7 은 선택 규칙에 `extra_prepare_cost` 를 넣으라고 하면서 regret 은 GPU 시간으로 채점한다. 규칙과 채점이 서로 다른 것을 재므로 두 지표를 모두 남긴다.

| 규칙 | regret 중앙값 | p90 | 최악 | RBC_active |
|---|---:|---:|---:|---:|
| GPU-only 규칙 (§7.7 진단 기준) | 2.45% | 36.32% | 212.64% | 47/189 |
| production 규칙 (extra_prepare 포함) | 3.22% | 174.08% | 215.11% | 0/189 |

기준: 중앙값 3% 이하, 최악 5% 이하. 선택기 CPU 중앙값 799us (probe + 선택된 계획 생성).

### 형상·패턴별

| 형상 | 패턴 | 선택 | oracle 승자 | regret 중앙값 | 예측 q | 예측 s | margin us | extra us | 이유 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 128:16:4 | clustered | q_outer | signature | 196.7% | 128 | 53 | 28 | 64 | 구간 불일치(BM 다름) — 예측 비교 불가 |
| 128:16:4 | common_private | q_outer | signature | 175.8% | 83 | 36 | 18 | 64 | predicted_saving<=extra_prepare+er |
| 128:16:4 | random | q_outer | bounded_merge | 170.1% | 128 | 56 | 28 | 64 | predicted_saving<=extra_prepare+er |
| 2048:16:8 | clustered | q_outer | q_outer | 0.0% | 344 | 342 | 80 | 64 | predicted_saving<=extra_prepare+er |
| 2048:16:8 | common_private | q_outer | bounded_merge | 3.3% | 265 | 310 | 61 | 64 | predicted_saving<=extra_prepare+er |
| 2048:16:8 | random | q_outer | q_outer | 0.0% | 340 | 357 | 79 | 64 | predicted_saving<=extra_prepare+er |
| 256:16:16 | clustered | q_outer | signature | 2.8% | — | — | 0 | 0 | unsupported_features(BM 미교정) |
| 256:16:16 | common_private | q_outer | bounded_merge | 2.5% | — | — | 0 | 0 | unsupported_features(BM 미교정) |
| 256:16:16 | random | q_outer | bounded_merge | 2.6% | — | — | 0 | 0 | unsupported_features(BM 미교정) |
| 256:16:8 | clustered | q_outer | signature | 43.6% | 90 | 69 | 21 | 64 | predicted_saving<=extra_prepare+er |
| 256:16:8 | common_private | q_outer | signature | 13.9% | 70 | 55 | 16 | 64 | predicted_saving<=extra_prepare+er |
| 256:16:8 | random | q_outer | signature | 37.5% | 90 | 72 | 21 | 64 | predicted_saving<=extra_prepare+er |
| 512:16:16 | clustered | q_outer | signature | 3.0% | — | — | 0 | 0 | unsupported_features(BM 미교정) |
| 512:16:16 | common_private | q_outer | bounded_merge | 2.6% | — | — | 0 | 0 | unsupported_features(BM 미교정) |
| 512:16:16 | random | q_outer | signature | 2.2% | — | — | 0 | 0 | unsupported_features(BM 미교정) |
| 512:16:4 | clustered | q_outer | bounded_merge | 36.4% | 175 | 108 | 38 | 64 | predicted_saving<=extra_prepare+er |
| 512:16:4 | common_private | q_outer | signature | 9.7% | 114 | 82 | 25 | 64 | predicted_saving<=extra_prepare+er |
| 512:16:4 | random | q_outer | bounded_merge | 22.5% | 175 | 118 | 38 | 64 | predicted_saving<=extra_prepare+er |
| 512:16:8 | clustered | q_outer | q_outer | 0.0% | 126 | 109 | 29 | 64 | predicted_saving<=extra_prepare+er |
| 512:16:8 | common_private | q_outer | bounded_merge | 2.6% | 98 | 92 | 23 | 64 | predicted_saving<=extra_prepare+er |
| 512:16:8 | random | q_outer | q_outer | 0.0% | 126 | 113 | 29 | 64 | predicted_saving<=extra_prepare+er |

### 전량 원시 셀

| seed | 형상 | 패턴 | 선택 | M | 선택 us | oracle | M | oracle us | regret | best Q us | vs Q | 선택기 CPU us | GPU-only 선택 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 128:16:4 | clustered | q_outer | 16 | 149.5 | bounded_merge | M32 | 52.8 | 183.4% | 149.5 | 0.0% | 471 | signature |
| 102 | 128:16:4 | clustered | q_outer | 16 | 146.4 | signature | M16 | 50.4 | 190.5% | 146.4 | 0.0% | 744 | q_outer |
| 103 | 128:16:4 | clustered | q_outer | 16 | 149.1 | signature | M32 | 49.7 | 199.7% | 149.1 | 0.0% | 518 | q_outer |
| 104 | 128:16:4 | clustered | q_outer | 16 | 145.7 | signature | M16 | 49.4 | 194.7% | 145.7 | 0.0% | 458 | q_outer |
| 105 | 128:16:4 | clustered | q_outer | 16 | 148.2 | bounded_merge | M32 | 52.6 | 181.4% | 148.2 | 0.0% | 482 | signature |
| 106 | 128:16:4 | clustered | q_outer | 16 | 147.2 | signature | M32 | 49.6 | 196.7% | 147.2 | 0.0% | 520 | q_outer |
| 107 | 128:16:4 | clustered | q_outer | 16 | 152.0 | bounded_merge | M32 | 48.2 | 215.1% | 152.0 | 0.0% | 534 | signature |
| 108 | 128:16:4 | clustered | q_outer | 16 | 149.2 | signature | M16 | 47.7 | 212.6% | 149.2 | 0.0% | 628 | q_outer |
| 109 | 128:16:4 | clustered | q_outer | 16 | 149.7 | signature | M16 | 49.4 | 203.0% | 149.7 | 0.0% | 565 | q_outer |
| 101 | 128:16:4 | common_private | q_outer | 16 | 96.8 | signature | M16 | 35.1 | 175.7% | 96.8 | 0.0% | 2851 | signature |
| 102 | 128:16:4 | common_private | q_outer | 16 | 96.9 | signature | M16 | 34.2 | 183.4% | 96.9 | 0.0% | 503 | signature |
| 103 | 128:16:4 | common_private | q_outer | 16 | 97.0 | signature | M16 | 33.4 | 190.2% | 97.0 | 0.0% | 549 | signature |
| 104 | 128:16:4 | common_private | q_outer | 16 | 98.2 | signature | M16 | 36.0 | 173.0% | 98.2 | 0.0% | 504 | signature |
| 105 | 128:16:4 | common_private | q_outer | 16 | 98.5 | signature | M16 | 36.0 | 174.0% | 98.5 | 0.0% | 528 | signature |
| 106 | 128:16:4 | common_private | q_outer | 16 | 98.1 | signature | M16 | 34.0 | 188.3% | 98.1 | 0.0% | 634 | signature |
| 107 | 128:16:4 | common_private | q_outer | 16 | 98.8 | signature | M16 | 35.8 | 175.8% | 98.8 | 0.0% | 540 | signature |
| 108 | 128:16:4 | common_private | q_outer | 16 | 97.0 | signature | M16 | 34.4 | 182.0% | 97.0 | 0.0% | 560 | signature |
| 109 | 128:16:4 | common_private | q_outer | 16 | 95.0 | signature | M16 | 34.6 | 174.2% | 95.0 | 0.0% | 535 | signature |
| 101 | 128:16:4 | random | q_outer | 16 | 142.9 | bounded_merge | M32 | 54.5 | 162.3% | 142.9 | 0.0% | 808 | signature |
| 102 | 128:16:4 | random | q_outer | 16 | 147.6 | bounded_merge | M32 | 54.6 | 170.1% | 147.6 | 0.0% | 501 | signature |
| 103 | 128:16:4 | random | q_outer | 16 | 144.1 | bounded_merge | M32 | 53.5 | 169.1% | 144.1 | 0.0% | 533 | signature |
| 104 | 128:16:4 | random | q_outer | 16 | 144.9 | bounded_merge | M32 | 52.6 | 175.3% | 144.9 | 0.0% | 524 | signature |
| 105 | 128:16:4 | random | q_outer | 16 | 145.6 | bounded_merge | M32 | 55.6 | 162.0% | 145.6 | 0.0% | 507 | signature |
| 106 | 128:16:4 | random | q_outer | 16 | 144.2 | bounded_merge | M32 | 52.1 | 176.8% | 144.2 | 0.0% | 505 | signature |
| 107 | 128:16:4 | random | q_outer | 16 | 146.8 | bounded_merge | M32 | 53.0 | 177.2% | 146.8 | 0.0% | 545 | signature |
| 108 | 128:16:4 | random | q_outer | 16 | 146.8 | bounded_merge | M32 | 54.2 | 170.8% | 146.8 | 0.0% | 533 | signature |
| 109 | 128:16:4 | random | q_outer | 16 | 146.0 | bounded_merge | M32 | 54.1 | 170.1% | 146.0 | 0.0% | 673 | signature |
| 101 | 2048:16:8 | clustered | q_outer | 16 | 387.5 | q_outer | M16 | 387.5 | 0.0% | 387.5 | 0.0% | 3399 | q_outer |
| 102 | 2048:16:8 | clustered | q_outer | 16 | 386.2 | q_outer | M16 | 386.2 | 0.0% | 386.2 | 0.0% | 2259 | q_outer |
| 103 | 2048:16:8 | clustered | q_outer | 16 | 388.2 | q_outer | M16 | 388.2 | 0.0% | 388.2 | 0.0% | 1969 | q_outer |
| 104 | 2048:16:8 | clustered | q_outer | 16 | 390.8 | signature | M16 | 390.3 | 0.1% | 390.8 | 0.0% | 2080 | q_outer |
| 105 | 2048:16:8 | clustered | q_outer | 16 | 387.6 | q_outer | M16 | 387.6 | 0.0% | 387.6 | 0.0% | 1974 | q_outer |
| 106 | 2048:16:8 | clustered | q_outer | 16 | 391.3 | q_outer | M16 | 391.3 | 0.0% | 391.3 | 0.0% | 1966 | q_outer |
| 107 | 2048:16:8 | clustered | q_outer | 16 | 391.0 | q_outer | M16 | 391.0 | 0.0% | 391.0 | 0.0% | 1960 | q_outer |
| 108 | 2048:16:8 | clustered | q_outer | 16 | 390.7 | q_outer | M16 | 390.7 | 0.0% | 390.7 | 0.0% | 2678 | q_outer |
| 109 | 2048:16:8 | clustered | q_outer | 16 | 387.0 | q_outer | M16 | 387.0 | 0.0% | 387.0 | 0.0% | 2307 | q_outer |
| 101 | 2048:16:8 | common_private | q_outer | 16 | 288.7 | bounded_merge | M16 | 279.4 | 3.3% | 288.7 | 0.0% | 2546 | q_outer |
| 102 | 2048:16:8 | common_private | q_outer | 16 | 288.6 | bounded_merge | M16 | 279.0 | 3.5% | 288.6 | 0.0% | 3280 | q_outer |
| 103 | 2048:16:8 | common_private | q_outer | 16 | 288.2 | bounded_merge | M16 | 278.9 | 3.4% | 288.2 | 0.0% | 2268 | q_outer |
| 104 | 2048:16:8 | common_private | q_outer | 16 | 289.0 | bounded_merge | M16 | 279.6 | 3.4% | 289.0 | 0.0% | 2223 | q_outer |
| 105 | 2048:16:8 | common_private | q_outer | 16 | 282.7 | bounded_merge | M16 | 274.0 | 3.2% | 282.7 | 0.0% | 2120 | q_outer |
| 106 | 2048:16:8 | common_private | q_outer | 16 | 288.6 | bounded_merge | M16 | 279.2 | 3.3% | 288.6 | 0.0% | 2147 | q_outer |
| 107 | 2048:16:8 | common_private | q_outer | 16 | 289.5 | bounded_merge | M16 | 280.4 | 3.2% | 289.5 | 0.0% | 1964 | q_outer |
| 108 | 2048:16:8 | common_private | q_outer | 16 | 288.5 | bounded_merge | M16 | 279.4 | 3.2% | 288.5 | 0.0% | 2341 | q_outer |
| 109 | 2048:16:8 | common_private | q_outer | 16 | 289.0 | bounded_merge | M16 | 279.5 | 3.4% | 289.0 | 0.0% | 2038 | q_outer |
| 101 | 2048:16:8 | random | q_outer | 16 | 375.0 | q_outer | M16 | 375.0 | 0.0% | 375.0 | 0.0% | 2712 | q_outer |
| 102 | 2048:16:8 | random | q_outer | 16 | 373.0 | q_outer | M16 | 373.0 | 0.0% | 373.0 | 0.0% | 2250 | q_outer |
| 103 | 2048:16:8 | random | q_outer | 16 | 368.8 | q_outer | M16 | 368.8 | 0.0% | 368.8 | 0.0% | 2236 | q_outer |
| 104 | 2048:16:8 | random | q_outer | 16 | 372.1 | q_outer | M16 | 372.1 | 0.0% | 372.1 | 0.0% | 2329 | q_outer |
| 105 | 2048:16:8 | random | q_outer | 16 | 375.2 | q_outer | M16 | 375.2 | 0.0% | 375.2 | 0.0% | 2173 | q_outer |
| 106 | 2048:16:8 | random | q_outer | 16 | 374.4 | q_outer | M16 | 374.4 | 0.0% | 374.4 | 0.0% | 2153 | q_outer |
| 107 | 2048:16:8 | random | q_outer | 16 | 375.2 | q_outer | M16 | 375.2 | 0.0% | 375.2 | 0.0% | 2147 | q_outer |
| 108 | 2048:16:8 | random | q_outer | 16 | 372.6 | q_outer | M16 | 372.6 | 0.0% | 372.6 | 0.0% | 2156 | q_outer |
| 109 | 2048:16:8 | random | q_outer | 16 | 375.6 | q_outer | M16 | 375.6 | 0.0% | 375.6 | 0.0% | 2148 | q_outer |
| 101 | 256:16:16 | clustered | q_outer | 16 | 58.1 | bounded_merge | M16 | 56.5 | 2.8% | 58.1 | 0.0% | 634 | q_outer |
| 102 | 256:16:16 | clustered | q_outer | 16 | 57.6 | bounded_merge | M16 | 55.9 | 3.0% | 57.6 | 0.0% | 646 | q_outer |
| 103 | 256:16:16 | clustered | q_outer | 16 | 57.8 | signature | M16 | 56.6 | 2.3% | 57.8 | 0.0% | 600 | q_outer |
| 104 | 256:16:16 | clustered | q_outer | 16 | 57.7 | bounded_merge | M16 | 56.4 | 2.3% | 57.7 | 0.0% | 656 | q_outer |
| 105 | 256:16:16 | clustered | q_outer | 16 | 57.4 | bounded_merge | M16 | 55.9 | 2.8% | 57.4 | 0.0% | 590 | q_outer |
| 106 | 256:16:16 | clustered | q_outer | 16 | 57.6 | signature | M16 | 56.4 | 2.2% | 57.6 | 0.0% | 608 | q_outer |
| 107 | 256:16:16 | clustered | q_outer | 16 | 58.2 | signature | M16 | 56.6 | 2.9% | 58.2 | 0.0% | 583 | q_outer |
| 108 | 256:16:16 | clustered | q_outer | 16 | 58.0 | signature | M16 | 56.3 | 3.0% | 58.0 | 0.0% | 585 | q_outer |
| 109 | 256:16:16 | clustered | q_outer | 16 | 57.2 | signature | M16 | 56.2 | 1.9% | 57.2 | 0.0% | 636 | q_outer |
| 101 | 256:16:16 | common_private | q_outer | 16 | 53.9 | signature | M16 | 52.7 | 2.3% | 53.9 | 0.0% | 640 | q_outer |
| 102 | 256:16:16 | common_private | q_outer | 16 | 53.2 | bounded_merge | M16 | 51.8 | 2.8% | 53.2 | 0.0% | 621 | q_outer |
| 103 | 256:16:16 | common_private | q_outer | 16 | 53.2 | bounded_merge | M16 | 51.8 | 2.6% | 53.2 | 0.0% | 657 | q_outer |
| 104 | 256:16:16 | common_private | q_outer | 16 | 53.5 | bounded_merge | M16 | 52.4 | 2.3% | 53.5 | 0.0% | 631 | q_outer |
| 105 | 256:16:16 | common_private | q_outer | 16 | 54.4 | bounded_merge | M16 | 52.8 | 3.1% | 54.4 | 0.0% | 581 | q_outer |
| 106 | 256:16:16 | common_private | q_outer | 16 | 53.1 | signature | M16 | 52.2 | 1.7% | 53.1 | 0.0% | 609 | q_outer |
| 107 | 256:16:16 | common_private | q_outer | 16 | 53.5 | signature | M16 | 52.2 | 2.5% | 53.5 | 0.0% | 611 | q_outer |
| 108 | 256:16:16 | common_private | q_outer | 16 | 53.1 | bounded_merge | M16 | 51.0 | 4.1% | 53.1 | 0.0% | 607 | q_outer |
| 109 | 256:16:16 | common_private | q_outer | 16 | 53.5 | signature | M16 | 52.2 | 2.4% | 53.5 | 0.0% | 684 | q_outer |
| 101 | 256:16:16 | random | q_outer | 16 | 55.8 | bounded_merge | M16 | 54.5 | 2.5% | 55.8 | 0.0% | 626 | q_outer |
| 102 | 256:16:16 | random | q_outer | 16 | 55.6 | signature | M16 | 54.1 | 2.7% | 55.6 | 0.0% | 626 | q_outer |
| 103 | 256:16:16 | random | q_outer | 16 | 55.7 | bounded_merge | M16 | 54.5 | 2.3% | 55.7 | 0.0% | 627 | q_outer |
| 104 | 256:16:16 | random | q_outer | 16 | 55.4 | signature | M16 | 54.2 | 2.3% | 55.4 | 0.0% | 605 | q_outer |
| 105 | 256:16:16 | random | q_outer | 16 | 55.9 | signature | M16 | 54.4 | 2.8% | 55.9 | 0.0% | 599 | q_outer |
| 106 | 256:16:16 | random | q_outer | 16 | 55.7 | signature | M16 | 54.2 | 2.8% | 55.7 | 0.0% | 587 | q_outer |
| 107 | 256:16:16 | random | q_outer | 16 | 55.8 | bounded_merge | M16 | 54.4 | 2.6% | 55.8 | 0.0% | 601 | q_outer |
| 108 | 256:16:16 | random | q_outer | 16 | 55.3 | bounded_merge | M16 | 54.3 | 1.9% | 55.3 | 0.0% | 615 | q_outer |
| 109 | 256:16:16 | random | q_outer | 16 | 56.2 | bounded_merge | M16 | 54.5 | 3.1% | 56.2 | 0.0% | 633 | q_outer |
| 101 | 256:16:8 | clustered | q_outer | 16 | 87.0 | signature | M16 | 60.1 | 44.7% | 87.0 | 0.0% | 657 | q_outer |
| 102 | 256:16:8 | clustered | q_outer | 16 | 87.2 | signature | M16 | 61.2 | 42.6% | 87.2 | 0.0% | 688 | q_outer |
| 103 | 256:16:8 | clustered | q_outer | 16 | 87.3 | signature | M16 | 60.4 | 44.7% | 87.3 | 0.0% | 629 | q_outer |
| 104 | 256:16:8 | clustered | q_outer | 16 | 87.4 | signature | M32 | 65.1 | 34.3% | 87.4 | 0.0% | 599 | q_outer |
| 105 | 256:16:8 | clustered | q_outer | 16 | 87.3 | signature | M16 | 60.7 | 44.0% | 87.3 | 0.0% | 611 | q_outer |
| 106 | 256:16:8 | clustered | q_outer | 16 | 89.6 | signature | M16 | 66.0 | 35.7% | 89.6 | 0.0% | 739 | q_outer |
| 107 | 256:16:8 | clustered | q_outer | 16 | 87.3 | signature | M16 | 60.8 | 43.6% | 87.3 | 0.0% | 648 | q_outer |
| 108 | 256:16:8 | clustered | q_outer | 16 | 84.8 | signature | M16 | 60.6 | 39.9% | 84.8 | 0.0% | 580 | q_outer |
| 109 | 256:16:8 | clustered | q_outer | 16 | 88.2 | signature | M16 | 60.5 | 45.7% | 88.2 | 0.0% | 583 | q_outer |
| 101 | 256:16:8 | common_private | q_outer | 16 | 69.8 | signature | M16 | 61.3 | 13.9% | 69.8 | 0.0% | 705 | q_outer |
| 102 | 256:16:8 | common_private | q_outer | 16 | 69.8 | signature | M16 | 61.3 | 13.9% | 69.8 | 0.0% | 761 | q_outer |
| 103 | 256:16:8 | common_private | q_outer | 16 | 69.6 | signature | M16 | 60.8 | 14.4% | 69.6 | 0.0% | 726 | q_outer |
| 104 | 256:16:8 | common_private | q_outer | 16 | 71.1 | signature | M16 | 61.2 | 16.2% | 71.1 | 0.0% | 661 | q_outer |
| 105 | 256:16:8 | common_private | q_outer | 16 | 69.5 | signature | M16 | 61.3 | 13.4% | 69.5 | 0.0% | 632 | q_outer |
| 106 | 256:16:8 | common_private | q_outer | 16 | 70.7 | signature | M16 | 61.2 | 15.5% | 70.7 | 0.0% | 770 | q_outer |
| 107 | 256:16:8 | common_private | q_outer | 16 | 69.5 | signature | M16 | 61.4 | 13.2% | 69.5 | 0.0% | 618 | q_outer |
| 108 | 256:16:8 | common_private | q_outer | 16 | 73.0 | signature | M16 | 61.7 | 18.4% | 73.0 | 0.0% | 693 | q_outer |
| 109 | 256:16:8 | common_private | q_outer | 16 | 69.6 | signature | M16 | 61.6 | 13.1% | 69.6 | 0.0% | 627 | q_outer |
| 101 | 256:16:8 | random | q_outer | 16 | 86.3 | signature | M16 | 62.1 | 39.1% | 86.3 | 0.0% | 764 | q_outer |
| 102 | 256:16:8 | random | q_outer | 16 | 86.7 | signature | M16 | 62.6 | 38.4% | 86.7 | 0.0% | 785 | q_outer |
| 103 | 256:16:8 | random | q_outer | 16 | 86.6 | signature | M16 | 63.7 | 36.0% | 86.6 | 0.0% | 699 | q_outer |
| 104 | 256:16:8 | random | q_outer | 16 | 86.6 | signature | M16 | 64.6 | 34.1% | 86.6 | 0.0% | 649 | q_outer |
| 105 | 256:16:8 | random | q_outer | 16 | 86.3 | signature | M16 | 62.4 | 38.2% | 86.3 | 0.0% | 646 | q_outer |
| 106 | 256:16:8 | random | q_outer | 16 | 85.8 | signature | M16 | 62.4 | 37.5% | 85.8 | 0.0% | 721 | q_outer |
| 107 | 256:16:8 | random | q_outer | 16 | 85.9 | signature | M16 | 63.9 | 34.3% | 85.9 | 0.0% | 637 | q_outer |
| 108 | 256:16:8 | random | q_outer | 16 | 87.6 | signature | M16 | 62.8 | 39.6% | 87.6 | 0.0% | 667 | q_outer |
| 109 | 256:16:8 | random | q_outer | 16 | 86.3 | signature | M16 | 65.6 | 31.6% | 86.3 | 0.0% | 676 | q_outer |
| 101 | 512:16:16 | clustered | q_outer | 16 | 105.8 | bounded_merge | M16 | 102.6 | 3.2% | 105.8 | 0.0% | 899 | q_outer |
| 102 | 512:16:16 | clustered | q_outer | 16 | 105.7 | bounded_merge | M16 | 102.6 | 3.0% | 105.7 | 0.0% | 956 | q_outer |
| 103 | 512:16:16 | clustered | q_outer | 16 | 105.9 | signature | M16 | 102.7 | 3.1% | 105.9 | 0.0% | 865 | q_outer |
| 104 | 512:16:16 | clustered | q_outer | 16 | 105.9 | signature | M16 | 102.5 | 3.3% | 105.9 | 0.0% | 837 | q_outer |
| 105 | 512:16:16 | clustered | q_outer | 16 | 105.2 | signature | M16 | 102.8 | 2.3% | 105.2 | 0.0% | 838 | q_outer |
| 106 | 512:16:16 | clustered | q_outer | 16 | 105.8 | signature | M16 | 103.4 | 2.3% | 105.8 | 0.0% | 831 | q_outer |
| 107 | 512:16:16 | clustered | q_outer | 16 | 105.2 | bounded_merge | M16 | 103.0 | 2.1% | 105.2 | 0.0% | 1263 | q_outer |
| 108 | 512:16:16 | clustered | q_outer | 16 | 106.3 | signature | M16 | 103.1 | 3.0% | 106.3 | 0.0% | 859 | q_outer |
| 109 | 512:16:16 | clustered | q_outer | 16 | 105.2 | bounded_merge | M16 | 102.9 | 2.3% | 105.2 | 0.0% | 890 | q_outer |
| 101 | 512:16:16 | common_private | q_outer | 16 | 100.6 | signature | M16 | 97.4 | 3.2% | 100.6 | 0.0% | 902 | q_outer |
| 102 | 512:16:16 | common_private | q_outer | 16 | 100.8 | bounded_merge | M16 | 98.0 | 2.9% | 100.8 | 0.0% | 951 | q_outer |
| 103 | 512:16:16 | common_private | q_outer | 16 | 100.3 | bounded_merge | M16 | 97.7 | 2.6% | 100.3 | 0.0% | 916 | q_outer |
| 104 | 512:16:16 | common_private | q_outer | 16 | 99.9 | bounded_merge | M16 | 97.4 | 2.6% | 99.9 | 0.0% | 894 | q_outer |
| 105 | 512:16:16 | common_private | q_outer | 16 | 99.7 | bounded_merge | M16 | 97.5 | 2.3% | 99.7 | 0.0% | 914 | q_outer |
| 106 | 512:16:16 | common_private | q_outer | 16 | 100.3 | bounded_merge | M16 | 97.3 | 3.1% | 100.3 | 0.0% | 860 | q_outer |
| 107 | 512:16:16 | common_private | q_outer | 16 | 99.0 | bounded_merge | M16 | 97.0 | 2.1% | 99.0 | 0.0% | 883 | q_outer |
| 108 | 512:16:16 | common_private | q_outer | 16 | 100.5 | bounded_merge | M16 | 97.0 | 3.6% | 100.5 | 0.0% | 856 | q_outer |
| 109 | 512:16:16 | common_private | q_outer | 16 | 100.3 | bounded_merge | M16 | 97.9 | 2.5% | 100.3 | 0.0% | 936 | q_outer |
| 101 | 512:16:16 | random | q_outer | 16 | 101.9 | signature | M16 | 99.5 | 2.4% | 101.9 | 0.0% | 950 | q_outer |
| 102 | 512:16:16 | random | q_outer | 16 | 101.1 | signature | M16 | 99.0 | 2.1% | 101.1 | 0.0% | 905 | q_outer |
| 103 | 512:16:16 | random | q_outer | 16 | 102.0 | signature | M16 | 99.0 | 3.0% | 102.0 | 0.0% | 883 | q_outer |
| 104 | 512:16:16 | random | q_outer | 16 | 101.4 | signature | M16 | 99.2 | 2.2% | 101.4 | 0.0% | 860 | q_outer |
| 105 | 512:16:16 | random | q_outer | 16 | 102.4 | signature | M16 | 99.8 | 2.7% | 102.4 | 0.0% | 836 | q_outer |
| 106 | 512:16:16 | random | q_outer | 16 | 101.4 | signature | M16 | 99.6 | 1.9% | 101.4 | 0.0% | 869 | q_outer |
| 107 | 512:16:16 | random | q_outer | 16 | 101.6 | signature | M16 | 100.0 | 1.6% | 101.6 | 0.0% | 825 | q_outer |
| 108 | 512:16:16 | random | q_outer | 16 | 102.5 | signature | M16 | 99.9 | 2.6% | 102.5 | 0.0% | 850 | q_outer |
| 109 | 512:16:16 | random | q_outer | 16 | 101.9 | signature | M16 | 99.9 | 1.9% | 101.9 | 0.0% | 943 | q_outer |
| 101 | 512:16:4 | clustered | q_outer | 16 | 152.7 | bounded_merge | M32 | 106.5 | 43.3% | 152.7 | 0.0% | 810 | signature |
| 102 | 512:16:4 | clustered | q_outer | 16 | 148.6 | signature | M64 | 115.6 | 28.6% | 148.6 | 0.0% | 777 | signature |
| 103 | 512:16:4 | clustered | q_outer | 16 | 150.9 | bounded_merge | M32 | 106.8 | 41.3% | 150.9 | 0.0% | 790 | signature |
| 104 | 512:16:4 | clustered | q_outer | 16 | 148.8 | signature | M64 | 114.6 | 29.9% | 148.8 | 0.0% | 777 | signature |
| 105 | 512:16:4 | clustered | q_outer | 16 | 150.7 | bounded_merge | M32 | 111.5 | 35.2% | 150.7 | 0.0% | 809 | signature |
| 106 | 512:16:4 | clustered | q_outer | 16 | 149.9 | signature | M64 | 118.2 | 26.8% | 149.9 | 0.0% | 792 | signature |
| 107 | 512:16:4 | clustered | q_outer | 16 | 150.9 | bounded_merge | M32 | 110.7 | 36.4% | 150.9 | 0.0% | 781 | signature |
| 108 | 512:16:4 | clustered | q_outer | 16 | 149.2 | bounded_merge | M32 | 107.1 | 39.3% | 149.2 | 0.0% | 784 | q_outer |
| 109 | 512:16:4 | clustered | q_outer | 16 | 150.8 | bounded_merge | M32 | 110.3 | 36.7% | 150.8 | 0.0% | 776 | signature |
| 101 | 512:16:4 | common_private | q_outer | 16 | 98.6 | signature | M16 | 89.7 | 9.9% | 98.6 | 0.0% | 893 | signature |
| 102 | 512:16:4 | common_private | q_outer | 16 | 98.9 | signature | M16 | 90.1 | 9.7% | 98.9 | 0.0% | 873 | signature |
| 103 | 512:16:4 | common_private | q_outer | 16 | 99.6 | signature | M16 | 90.3 | 10.4% | 99.6 | 0.0% | 785 | signature |
| 104 | 512:16:4 | common_private | q_outer | 16 | 98.3 | signature | M16 | 90.1 | 9.1% | 98.3 | 0.0% | 805 | signature |
| 105 | 512:16:4 | common_private | q_outer | 16 | 99.1 | signature | M16 | 90.3 | 9.7% | 99.1 | 0.0% | 735 | signature |
| 106 | 512:16:4 | common_private | q_outer | 16 | 99.0 | signature | M16 | 89.9 | 10.1% | 99.0 | 0.0% | 803 | signature |
| 107 | 512:16:4 | common_private | q_outer | 16 | 98.4 | signature | M16 | 90.5 | 8.8% | 98.4 | 0.0% | 768 | signature |
| 108 | 512:16:4 | common_private | q_outer | 16 | 100.6 | signature | M16 | 90.2 | 11.6% | 100.6 | 0.0% | 725 | signature |
| 109 | 512:16:4 | common_private | q_outer | 16 | 98.5 | signature | M16 | 90.3 | 9.1% | 98.5 | 0.0% | 753 | signature |
| 101 | 512:16:4 | random | q_outer | 16 | 148.9 | bounded_merge | M32 | 118.5 | 25.6% | 148.9 | 0.0% | 970 | signature |
| 102 | 512:16:4 | random | q_outer | 16 | 146.0 | signature | M16 | 121.2 | 20.4% | 146.0 | 0.0% | 929 | signature |
| 103 | 512:16:4 | random | q_outer | 16 | 145.5 | bounded_merge | M32 | 118.1 | 23.1% | 145.5 | 0.0% | 967 | signature |
| 104 | 512:16:4 | random | q_outer | 16 | 146.1 | bounded_merge | M32 | 118.0 | 23.8% | 146.1 | 0.0% | 2340 | signature |
| 105 | 512:16:4 | random | q_outer | 16 | 146.7 | signature | M16 | 119.8 | 22.5% | 146.7 | 0.0% | 915 | signature |
| 106 | 512:16:4 | random | q_outer | 16 | 146.4 | signature | M16 | 121.8 | 20.2% | 146.4 | 0.0% | 920 | signature |
| 107 | 512:16:4 | random | q_outer | 16 | 145.2 | signature | M16 | 121.5 | 19.5% | 145.2 | 0.0% | 851 | signature |
| 108 | 512:16:4 | random | q_outer | 16 | 146.7 | bounded_merge | M32 | 119.4 | 22.8% | 146.7 | 0.0% | 818 | signature |
| 109 | 512:16:4 | random | q_outer | 16 | 147.0 | bounded_merge | M32 | 120.2 | 22.3% | 147.0 | 0.0% | 872 | signature |
| 101 | 512:16:8 | clustered | q_outer | 16 | 104.5 | q_outer | M16 | 104.5 | 0.0% | 104.5 | 0.0% | 868 | q_outer |
| 102 | 512:16:8 | clustered | q_outer | 16 | 104.6 | q_outer | M16 | 104.6 | 0.0% | 104.6 | 0.0% | 868 | q_outer |
| 103 | 512:16:8 | clustered | q_outer | 16 | 106.6 | q_outer | M16 | 106.6 | 0.0% | 106.6 | 0.0% | 832 | q_outer |
| 104 | 512:16:8 | clustered | q_outer | 16 | 105.8 | q_outer | M16 | 105.8 | 0.0% | 105.8 | 0.0% | 870 | q_outer |
| 105 | 512:16:8 | clustered | q_outer | 16 | 105.4 | q_outer | M16 | 105.4 | 0.0% | 105.4 | 0.0% | 775 | q_outer |
| 106 | 512:16:8 | clustered | q_outer | 16 | 106.2 | q_outer | M16 | 106.2 | 0.0% | 106.2 | 0.0% | 976 | q_outer |
| 107 | 512:16:8 | clustered | q_outer | 16 | 105.8 | q_outer | M16 | 105.8 | 0.0% | 105.8 | 0.0% | 820 | q_outer |
| 108 | 512:16:8 | clustered | q_outer | 16 | 104.4 | q_outer | M16 | 104.4 | 0.0% | 104.4 | 0.0% | 799 | q_outer |
| 109 | 512:16:8 | clustered | q_outer | 16 | 105.5 | q_outer | M16 | 105.5 | 0.0% | 105.5 | 0.0% | 782 | q_outer |
| 101 | 512:16:8 | common_private | q_outer | 16 | 78.5 | bounded_merge | M16 | 78.4 | 0.1% | 78.5 | 0.0% | 935 | q_outer |
| 102 | 512:16:8 | common_private | q_outer | 16 | 78.8 | bounded_merge | M16 | 76.8 | 2.5% | 78.8 | 0.0% | 924 | q_outer |
| 103 | 512:16:8 | common_private | q_outer | 16 | 78.4 | bounded_merge | M16 | 75.8 | 3.5% | 78.4 | 0.0% | 911 | q_outer |
| 104 | 512:16:8 | common_private | q_outer | 16 | 79.3 | bounded_merge | M16 | 76.5 | 3.6% | 79.3 | 0.0% | 903 | q_outer |
| 105 | 512:16:8 | common_private | q_outer | 16 | 79.5 | bounded_merge | M16 | 76.3 | 4.2% | 79.5 | 0.0% | 784 | q_outer |
| 106 | 512:16:8 | common_private | q_outer | 16 | 78.1 | bounded_merge | M16 | 76.2 | 2.6% | 78.1 | 0.0% | 873 | q_outer |
| 107 | 512:16:8 | common_private | q_outer | 16 | 78.8 | bounded_merge | M16 | 76.8 | 2.6% | 78.8 | 0.0% | 840 | q_outer |
| 108 | 512:16:8 | common_private | q_outer | 16 | 78.1 | q_outer | M16 | 78.1 | 0.0% | 78.1 | 0.0% | 821 | q_outer |
| 109 | 512:16:8 | common_private | q_outer | 16 | 79.1 | bounded_merge | M16 | 78.9 | 0.3% | 79.1 | 0.0% | 806 | q_outer |
| 101 | 512:16:8 | random | q_outer | 16 | 101.9 | q_outer | M16 | 101.9 | 0.0% | 101.9 | 0.0% | 899 | q_outer |
| 102 | 512:16:8 | random | q_outer | 16 | 101.0 | q_outer | M16 | 101.0 | 0.0% | 101.0 | 0.0% | 935 | q_outer |
| 103 | 512:16:8 | random | q_outer | 16 | 101.1 | q_outer | M16 | 101.1 | 0.0% | 101.1 | 0.0% | 1022 | q_outer |
| 104 | 512:16:8 | random | q_outer | 16 | 101.8 | q_outer | M16 | 101.8 | 0.0% | 101.8 | 0.0% | 909 | q_outer |
| 105 | 512:16:8 | random | q_outer | 16 | 101.9 | q_outer | M16 | 101.9 | 0.0% | 101.9 | 0.0% | 887 | q_outer |
| 106 | 512:16:8 | random | q_outer | 16 | 101.1 | q_outer | M16 | 101.1 | 0.0% | 101.1 | 0.0% | 900 | q_outer |
| 107 | 512:16:8 | random | q_outer | 16 | 102.6 | q_outer | M16 | 102.6 | 0.0% | 102.6 | 0.0% | 892 | q_outer |
| 108 | 512:16:8 | random | q_outer | 16 | 102.2 | q_outer | M16 | 102.2 | 0.0% | 102.2 | 0.0% | 842 | q_outer |
| 109 | 512:16:8 | random | q_outer | 16 | 102.4 | q_outer | M16 | 102.4 | 0.0% | 102.4 | 0.0% | 840 | q_outer |


## E4 — 전체 준비비 포함 비교 (§11 E4) — **주 판정**

형상 ['512:16:4', '256:16:8', '512:16:8'] / 패턴 ['common_private', 'random', 'clustered'] / 평가 seed [101, 102, 103, 104, 105, 106, 107, 108, 109]. 레코드 162개, 오류 0건.

측정 모드: `run_only`(준비된 plan 의 GPU), `cold_init`(capacity 할당·pin), `pool_fresh_host`(새 host mask 계획·pack·H2D·GPU·완료), `pool_fresh_gpu`(GPU-origin mask 의 D2H 포함 — **주 판정**), `exact_plan_cache`(같은 mask guard 후 재사용).

CPU 시간과 GPU 시간을 더해 wall 이라 쓰지 않았다. `wall_us` 는 실제 벽시계다.

### 판정 표 (pool_fresh_gpu, gain = 1 − T/T_q_outer. 음수 = 느려짐)

| 패턴 | 형상 | 후보 | q_outer wall us | 후보 wall us | gain(wall) | q_outer GPU us | 후보 GPU us | gain(GPU만) | 판정 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| clustered | 256:16:8 | signature | 767 | 812 | -5.8% | 86.4 | 60.9 | +29.5% | 개선 없음 |
| clustered | 256:16:8 | bounded_merge | 767 | 820 | -6.8% | 86.4 | 90.9 | -5.2% | 개선 없음 |
| common_private | 256:16:8 | signature | 717 | 781 | -8.8% | 69.7 | 62.3 | +10.6% | 개선 없음 |
| common_private | 256:16:8 | bounded_merge | 717 | 780 | -8.8% | 69.7 | 68.4 | +1.9% | 개선 없음 |
| random | 256:16:8 | signature | 765 | 826 | -8.0% | 86.2 | 63.4 | +26.5% | 개선 없음 |
| random | 256:16:8 | bounded_merge | 765 | 832 | -8.7% | 86.2 | 94.4 | -9.5% | 개선 없음 |
| clustered | 512:16:4 | signature | 959 | 1027 | -7.0% | 151.6 | 118.3 | +22.0% | 개선 없음 |
| clustered | 512:16:4 | bounded_merge | 959 | 1074 | -12.0% | 151.6 | 174.6 | -15.2% | 개선 없음 |
| common_private | 512:16:4 | signature | 862 | 992 | -15.1% | 97.8 | 91.3 | +6.7% | 개선 없음 |
| common_private | 512:16:4 | bounded_merge | 862 | 989 | -14.7% | 97.8 | 96.5 | +1.4% | 개선 없음 |
| random | 512:16:4 | signature | 956 | 1161 | -21.5% | 148.2 | 121.4 | +18.1% | 개선 없음 |
| random | 512:16:4 | bounded_merge | 956 | 1141 | -19.4% | 148.2 | 166.5 | -12.4% | 개선 없음 |
| clustered | 512:16:8 | signature | 933 | 1041 | -11.5% | 105.0 | 111.3 | -6.0% | 개선 없음 |
| clustered | 512:16:8 | bounded_merge | 933 | 1053 | -12.8% | 105.0 | 136.7 | -30.2% | 개선 없음 |
| common_private | 512:16:8 | signature | 892 | 1024 | -14.9% | 78.6 | 91.7 | -16.8% | 개선 없음 |
| common_private | 512:16:8 | bounded_merge | 892 | 941 | -5.5% | 78.6 | 77.2 | +1.8% | 개선 없음 |
| random | 512:16:8 | signature | 953 | 1093 | -14.7% | 101.6 | 110.9 | -9.1% | 개선 없음 |
| random | 512:16:8 | bounded_merge | 953 | 1085 | -13.8% | 101.6 | 141.7 | -39.5% | 개선 없음 |

GPU 시간만 보면 signature 가 이기는 조합이 있으나, 계획 CPU 비용이 그보다 커서 전체 시간에서는 모두 뒤집힌다.

### 비용 분해 (pool_fresh_gpu, 중앙값)

| 패턴 | 형상 | 후보 | cpu plan+pack us | DMA us | GPU us | wall us | desc live B | H2D B | H2D 회수 | D2H B | tasks | partial | hot pin/alloc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| common_private | 512:16:4 | q_outer | 405 | 42 | 214 | 862 | 25704 | 66568 | 2 | 32720 | 128 | 0 | 0/0 |
| common_private | 512:16:4 | signature | 528 | 34 | 205 | 992 | 42088 | 42088 | 1 | 32720 | 640 | 1024 | 0/0 |
| common_private | 512:16:4 | bounded_merge | 522 | 47 | 212 | 989 | 25704 | 66568 | 2 | 32720 | 128 | 0 | 0/0 |
| random | 512:16:4 | q_outer | 474 | 42 | 252 | 956 | 36280 | 98296 | 2 | 32680 | 128 | 0 | 0/0 |
| random | 512:16:4 | signature | 675 | 33 | 241 | 1161 | 54856 | 54856 | 1 | 32680 | 837 | 1076 | 0/0 |
| random | 512:16:4 | bounded_merge | 594 | 49 | 291 | 1141 | 36696 | 98712 | 2 | 32680 | 180 | 0 | 0/0 |
| clustered | 512:16:4 | q_outer | 445 | 46 | 265 | 959 | 36592 | 99224 | 2 | 32764 | 128 | 0 | 0/0 |
| clustered | 512:16:4 | signature | 563 | 31 | 229 | 1027 | 42032 | 42032 | 1 | 32764 | 553 | 170 | 0/0 |
| clustered | 512:16:4 | bounded_merge | 555 | 44 | 293 | 1074 | 39216 | 101848 | 2 | 32764 | 456 | 0 | 0/0 |
| common_private | 256:16:8 | q_outer | 316 | 46 | 170 | 717 | 15504 | 40064 | 2 | 16376 | 128 | 0 | 0/0 |
| common_private | 256:16:8 | signature | 391 | 34 | 167 | 781 | 23696 | 23696 | 1 | 16376 | 384 | 512 | 0/0 |
| common_private | 256:16:8 | bounded_merge | 361 | 48 | 175 | 780 | 15504 | 40064 | 2 | 16376 | 128 | 0 | 0/0 |
| random | 256:16:8 | q_outer | 343 | 44 | 186 | 765 | 19232 | 51248 | 2 | 16368 | 128 | 0 | 0/0 |
| random | 256:16:8 | signature | 413 | 34 | 168 | 826 | 23728 | 23728 | 1 | 16368 | 318 | 248 | 0/0 |
| random | 256:16:8 | bounded_merge | 386 | 45 | 196 | 832 | 19760 | 51776 | 2 | 16368 | 194 | 0 | 0/0 |
| clustered | 256:16:8 | q_outer | 339 | 45 | 191 | 767 | 19320 | 51512 | 2 | 16384 | 128 | 0 | 0/0 |
| clustered | 256:16:8 | signature | 410 | 34 | 169 | 812 | 20792 | 20792 | 1 | 16384 | 264 | 32 | 0/0 |
| clustered | 256:16:8 | bounded_merge | 379 | 46 | 195 | 820 | 20280 | 52472 | 2 | 16384 | 248 | 0 | 0/0 |
| common_private | 512:16:8 | q_outer | 463 | 42 | 189 | 892 | 30824 | 79880 | 2 | 32720 | 256 | 0 | 0/0 |
| common_private | 512:16:8 | signature | 580 | 32 | 200 | 1024 | 47208 | 47208 | 1 | 32720 | 768 | 1024 | 0/0 |
| common_private | 512:16:8 | bounded_merge | 511 | 43 | 187 | 941 | 30824 | 79880 | 2 | 32720 | 256 | 0 | 0/0 |
| random | 512:16:8 | q_outer | 507 | 43 | 212 | 953 | 38312 | 102344 | 2 | 32680 | 256 | 0 | 0/0 |
| random | 512:16:8 | signature | 612 | 32 | 230 | 1093 | 46904 | 46904 | 1 | 32680 | 629 | 468 | 0/0 |
| random | 512:16:8 | bounded_merge | 579 | 45 | 258 | 1085 | 39416 | 103448 | 2 | 32680 | 395 | 0 | 0/0 |
| clustered | 512:16:8 | q_outer | 489 | 43 | 211 | 933 | 38448 | 102752 | 2 | 32764 | 256 | 0 | 0/0 |
| clustered | 512:16:8 | signature | 573 | 32 | 217 | 1041 | 41440 | 41440 | 1 | 32764 | 529 | 68 | 0/0 |
| clustered | 512:16:8 | bounded_merge | 556 | 50 | 247 | 1053 | 40352 | 104656 | 2 | 32764 | 495 | 0 | 0/0 |

`hot pin/alloc` 은 hot path 의 host pin 호출 수 / device allocation 요청 수다. §8.7 의 목표는 0/0 이고 달성했다.

### 측정 모드별 전량

| 형상 | 패턴 | 후보 | 모드 | mask 출발 | wall us | GPU us | cpu plan+pack us | main us | merge us | cache | numeric | exact-edge |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| 256:16:8 | clustered | bounded_merge | cold_init | n/a | 896.1 | — | — | — | — | N | — | — |
| 256:16:8 | clustered | bounded_merge | exact_plan_cache | host | 288.8 | — | 39.9 | — | — | Y | — | — |
| 256:16:8 | clustered | bounded_merge | pool_fresh_gpu | gpu | 819.5 | 194.8 | 379.1 | — | — | N | pass | — |
| 256:16:8 | clustered | bounded_merge | pool_fresh_host | host | 764.2 | 187.1 | 372.9 | — | — | N | — | — |
| 256:16:8 | clustered | bounded_merge | pool_fresh_host_same_mask | host | 742.6 | — | — | — | — | N | — | — |
| 256:16:8 | clustered | bounded_merge | run_only | host | — | 90.9 | — | 90.9 | 0.0 | N | pass | pass |
| 256:16:8 | clustered | q_outer | cold_init | n/a | 1494.7 | — | — | — | — | N | — | — |
| 256:16:8 | clustered | q_outer | exact_plan_cache | host | 281.3 | — | 39.8 | — | — | Y | — | — |
| 256:16:8 | clustered | q_outer | pool_fresh_gpu | gpu | 767.0 | 190.6 | 339.2 | — | — | N | pass | — |
| 256:16:8 | clustered | q_outer | pool_fresh_host | host | 719.8 | 184.9 | 337.6 | — | — | N | — | — |
| 256:16:8 | clustered | q_outer | pool_fresh_host_same_mask | host | 722.8 | — | — | — | — | N | — | — |
| 256:16:8 | clustered | q_outer | run_only | host | — | 86.4 | — | 86.4 | 0.0 | N | pass | pass |
| 256:16:8 | clustered | signature | cold_init | n/a | 870.1 | — | — | — | — | N | — | — |
| 256:16:8 | clustered | signature | exact_plan_cache | host | 257.3 | — | 39.7 | — | — | Y | — | — |
| 256:16:8 | clustered | signature | pool_fresh_gpu | gpu | 811.7 | 169.0 | 410.2 | — | — | N | pass | — |
| 256:16:8 | clustered | signature | pool_fresh_host | host | 924.7 | 186.5 | 398.1 | — | — | N | — | — |
| 256:16:8 | clustered | signature | pool_fresh_host_same_mask | host | 721.2 | — | — | — | — | N | — | — |
| 256:16:8 | clustered | signature | run_only | host | — | 60.9 | — | 57.6 | 3.6 | N | pass | pass |
| 256:16:8 | common_private | bounded_merge | cold_init | n/a | 846.0 | — | — | — | — | N | — | — |
| 256:16:8 | common_private | bounded_merge | exact_plan_cache | host | 265.0 | — | 40.0 | — | — | Y | — | — |
| 256:16:8 | common_private | bounded_merge | pool_fresh_gpu | gpu | 780.1 | 175.1 | 360.9 | — | — | N | pass | — |
| 256:16:8 | common_private | bounded_merge | pool_fresh_host | host | 731.5 | 170.6 | 364.0 | — | — | N | — | — |
| 256:16:8 | common_private | bounded_merge | pool_fresh_host_same_mask | host | 710.4 | — | — | — | — | N | — | — |
| 256:16:8 | common_private | bounded_merge | run_only | host | — | 68.4 | — | 68.5 | 0.0 | N | pass | pass |
| 256:16:8 | common_private | q_outer | cold_init | n/a | 1417.9 | — | — | — | — | N | — | — |
| 256:16:8 | common_private | q_outer | exact_plan_cache | host | 270.7 | — | 39.8 | — | — | Y | — | — |
| 256:16:8 | common_private | q_outer | pool_fresh_gpu | gpu | 717.3 | 170.1 | 315.8 | — | — | N | pass | — |
| 256:16:8 | common_private | q_outer | pool_fresh_host | host | 683.5 | 169.2 | 320.1 | — | — | N | — | — |
| 256:16:8 | common_private | q_outer | pool_fresh_host_same_mask | host | 690.5 | — | — | — | — | N | — | — |
| 256:16:8 | common_private | q_outer | run_only | host | — | 69.7 | — | 69.7 | 0.0 | N | pass | pass |
| 256:16:8 | common_private | signature | cold_init | n/a | 1972.6 | — | — | — | — | N | — | — |
| 256:16:8 | common_private | signature | exact_plan_cache | host | 259.8 | — | 39.9 | — | — | Y | — | — |
| 256:16:8 | common_private | signature | pool_fresh_gpu | gpu | 780.5 | 166.9 | 390.8 | — | — | N | pass | — |
| 256:16:8 | common_private | signature | pool_fresh_host | host | 739.6 | 165.3 | 383.9 | — | — | N | — | — |
| 256:16:8 | common_private | signature | pool_fresh_host_same_mask | host | 717.7 | — | — | — | — | N | — | — |
| 256:16:8 | common_private | signature | run_only | host | — | 62.3 | — | 57.8 | 4.3 | N | pass | pass |
| 256:16:8 | random | bounded_merge | cold_init | n/a | 877.5 | — | — | — | — | N | — | — |
| 256:16:8 | random | bounded_merge | exact_plan_cache | host | 293.3 | — | 39.9 | — | — | Y | — | — |
| 256:16:8 | random | bounded_merge | pool_fresh_gpu | gpu | 831.6 | 195.8 | 386.4 | — | — | N | pass | — |
| 256:16:8 | random | bounded_merge | pool_fresh_host | host | 782.2 | 199.6 | 385.4 | — | — | N | — | — |
| 256:16:8 | random | bounded_merge | pool_fresh_host_same_mask | host | 759.9 | — | — | — | — | N | — | — |
| 256:16:8 | random | bounded_merge | run_only | host | — | 94.4 | — | 94.5 | 0.0 | N | pass | pass |
| 256:16:8 | random | q_outer | cold_init | n/a | 1063.6 | — | — | — | — | N | — | — |
| 256:16:8 | random | q_outer | exact_plan_cache | host | 281.5 | — | 39.5 | — | — | Y | — | — |
| 256:16:8 | random | q_outer | pool_fresh_gpu | gpu | 765.1 | 185.8 | 343.0 | — | — | N | pass | — |
| 256:16:8 | random | q_outer | pool_fresh_host | host | 729.6 | 191.8 | 349.3 | — | — | N | — | — |
| 256:16:8 | random | q_outer | pool_fresh_host_same_mask | host | 738.0 | — | — | — | — | N | — | — |
| 256:16:8 | random | q_outer | run_only | host | — | 86.2 | — | 86.2 | 0.0 | N | pass | pass |
| 256:16:8 | random | signature | cold_init | n/a | 877.8 | — | — | — | — | N | — | — |
| 256:16:8 | random | signature | exact_plan_cache | host | 266.6 | — | 40.2 | — | — | Y | — | — |
| 256:16:8 | random | signature | pool_fresh_gpu | gpu | 826.2 | 167.8 | 413.5 | — | — | N | pass | — |
| 256:16:8 | random | signature | pool_fresh_host | host | 806.2 | 174.1 | 425.8 | — | — | N | — | — |
| 256:16:8 | random | signature | pool_fresh_host_same_mask | host | 740.1 | — | — | — | — | N | — | — |
| 256:16:8 | random | signature | run_only | host | — | 63.4 | — | 60.1 | 3.7 | N | pass | pass |
| 512:16:4 | clustered | bounded_merge | cold_init | n/a | 798.2 | — | — | — | — | N | — | — |
| 512:16:4 | clustered | bounded_merge | exact_plan_cache | host | 412.8 | — | 76.6 | — | — | Y | — | — |
| 512:16:4 | clustered | bounded_merge | pool_fresh_gpu | gpu | 1073.9 | 292.6 | 555.0 | — | — | N | pass | — |
| 512:16:4 | clustered | bounded_merge | pool_fresh_host | host | 1044.0 | 287.6 | 554.2 | — | — | N | — | — |
| 512:16:4 | clustered | bounded_merge | pool_fresh_host_same_mask | host | 1007.1 | — | — | — | — | N | — | — |
| 512:16:4 | clustered | bounded_merge | run_only | host | — | 174.6 | — | 174.8 | 0.0 | N | pass | pass |
| 512:16:4 | clustered | q_outer | cold_init | n/a | 1464.8 | — | — | — | — | N | — | — |
| 512:16:4 | clustered | q_outer | exact_plan_cache | host | 388.3 | — | 76.7 | — | — | Y | — | — |
| 512:16:4 | clustered | q_outer | pool_fresh_gpu | gpu | 959.2 | 265.2 | 445.1 | — | — | N | pass | — |
| 512:16:4 | clustered | q_outer | pool_fresh_host | host | 902.6 | 252.4 | 453.3 | — | — | N | — | — |
| 512:16:4 | clustered | q_outer | pool_fresh_host_same_mask | host | 888.2 | — | — | — | — | N | — | — |
| 512:16:4 | clustered | q_outer | run_only | host | — | 151.6 | — | 151.0 | 0.0 | N | pass | pass |
| 512:16:4 | clustered | signature | cold_init | n/a | 785.8 | — | — | — | — | N | — | — |
| 512:16:4 | clustered | signature | exact_plan_cache | host | 355.8 | — | 76.6 | — | — | Y | — | — |
| 512:16:4 | clustered | signature | pool_fresh_gpu | gpu | 1026.7 | 228.7 | 563.1 | — | — | N | pass | — |
| 512:16:4 | clustered | signature | pool_fresh_host | host | 1017.5 | 230.4 | 573.3 | — | — | N | — | — |
| 512:16:4 | clustered | signature | pool_fresh_host_same_mask | host | 981.5 | — | — | — | — | N | — | — |
| 512:16:4 | clustered | signature | run_only | host | — | 118.3 | — | 113.8 | 3.6 | N | pass | pass |
| 512:16:4 | common_private | bounded_merge | cold_init | n/a | 814.1 | — | — | — | — | N | — | — |
| 512:16:4 | common_private | bounded_merge | exact_plan_cache | host | 337.7 | — | 76.9 | — | — | Y | — | — |
| 512:16:4 | common_private | bounded_merge | pool_fresh_gpu | gpu | 989.1 | 211.7 | 521.7 | — | — | N | pass | — |
| 512:16:4 | common_private | bounded_merge | pool_fresh_host | host | 943.9 | 220.2 | 509.5 | — | — | N | — | — |
| 512:16:4 | common_private | bounded_merge | pool_fresh_host_same_mask | host | 926.8 | — | — | — | — | N | — | — |
| 512:16:4 | common_private | bounded_merge | run_only | host | — | 96.5 | — | 96.5 | 0.0 | N | pass | pass |
| 512:16:4 | common_private | q_outer | cold_init | n/a | 17334.0 | — | — | — | — | N | — | — |
| 512:16:4 | common_private | q_outer | exact_plan_cache | host | 332.6 | — | 76.3 | — | — | Y | — | — |
| 512:16:4 | common_private | q_outer | pool_fresh_gpu | gpu | 861.9 | 213.5 | 405.2 | — | — | N | pass | — |
| 512:16:4 | common_private | q_outer | pool_fresh_host | host | 811.8 | 207.4 | 400.7 | — | — | N | — | — |
| 512:16:4 | common_private | q_outer | pool_fresh_host_same_mask | host | 778.2 | — | — | — | — | N | — | — |
| 512:16:4 | common_private | q_outer | run_only | host | — | 97.8 | — | 97.8 | 0.0 | N | pass | pass |
| 512:16:4 | common_private | signature | cold_init | n/a | 2072.7 | — | — | — | — | N | — | — |
| 512:16:4 | common_private | signature | exact_plan_cache | host | 330.6 | — | 77.1 | — | — | Y | — | — |
| 512:16:4 | common_private | signature | pool_fresh_gpu | gpu | 992.2 | 205.3 | 527.6 | — | — | N | pass | — |
| 512:16:4 | common_private | signature | pool_fresh_host | host | 915.9 | 198.5 | 520.8 | — | — | N | — | — |
| 512:16:4 | common_private | signature | pool_fresh_host_same_mask | host | 904.2 | — | — | — | — | N | — | — |
| 512:16:4 | common_private | signature | run_only | host | — | 91.3 | — | 86.5 | 4.3 | N | pass | pass |
| 512:16:4 | random | bounded_merge | cold_init | n/a | 768.0 | — | — | — | — | N | — | — |
| 512:16:4 | random | bounded_merge | exact_plan_cache | host | 408.6 | — | 76.3 | — | — | Y | — | — |
| 512:16:4 | random | bounded_merge | pool_fresh_gpu | gpu | 1141.2 | 291.2 | 593.8 | — | — | N | pass | — |
| 512:16:4 | random | bounded_merge | pool_fresh_host | host | 1128.3 | 292.8 | 608.8 | — | — | N | — | — |
| 512:16:4 | random | bounded_merge | pool_fresh_host_same_mask | host | 1042.6 | — | — | — | — | N | — | — |
| 512:16:4 | random | bounded_merge | run_only | host | — | 166.5 | — | 166.5 | 0.0 | N | pass | pass |
| 512:16:4 | random | q_outer | cold_init | n/a | 1442.5 | — | — | — | — | N | — | — |
| 512:16:4 | random | q_outer | exact_plan_cache | host | 390.9 | — | 76.4 | — | — | Y | — | — |
| 512:16:4 | random | q_outer | pool_fresh_gpu | gpu | 955.6 | 252.4 | 474.4 | — | — | N | pass | — |
| 512:16:4 | random | q_outer | pool_fresh_host | host | 925.6 | 257.8 | 468.1 | — | — | N | — | — |
| 512:16:4 | random | q_outer | pool_fresh_host_same_mask | host | 916.0 | — | — | — | — | N | — | — |
| 512:16:4 | random | q_outer | run_only | host | — | 148.2 | — | 148.0 | 0.0 | N | pass | pass |
| 512:16:4 | random | signature | cold_init | n/a | 778.9 | — | — | — | — | N | — | — |
| 512:16:4 | random | signature | exact_plan_cache | host | 353.8 | — | 76.6 | — | — | Y | — | — |
| 512:16:4 | random | signature | pool_fresh_gpu | gpu | 1161.0 | 240.9 | 675.3 | — | — | N | pass | — |
| 512:16:4 | random | signature | pool_fresh_host | host | 1282.8 | 257.6 | 699.8 | — | — | N | — | — |
| 512:16:4 | random | signature | pool_fresh_host_same_mask | host | 1099.6 | — | — | — | — | N | — | — |
| 512:16:4 | random | signature | run_only | host | — | 121.4 | — | 115.9 | 4.4 | N | pass | pass |
| 512:16:8 | clustered | bounded_merge | cold_init | n/a | 908.0 | — | — | — | — | N | — | — |
| 512:16:8 | clustered | bounded_merge | exact_plan_cache | host | 370.3 | — | 76.3 | — | — | Y | — | — |
| 512:16:8 | clustered | bounded_merge | pool_fresh_gpu | gpu | 1053.1 | 247.2 | 556.5 | — | — | N | pass | — |
| 512:16:8 | clustered | bounded_merge | pool_fresh_host | host | 1000.0 | 251.1 | 541.1 | — | — | N | — | — |
| 512:16:8 | clustered | bounded_merge | pool_fresh_host_same_mask | host | 981.8 | — | — | — | — | N | — | — |
| 512:16:8 | clustered | bounded_merge | run_only | host | — | 136.7 | — | 136.5 | 0.0 | N | pass | pass |
| 512:16:8 | clustered | q_outer | cold_init | n/a | 1272.0 | — | — | — | — | N | — | — |
| 512:16:8 | clustered | q_outer | exact_plan_cache | host | 337.8 | — | 76.3 | — | — | Y | — | — |
| 512:16:8 | clustered | q_outer | pool_fresh_gpu | gpu | 933.3 | 211.0 | 488.6 | — | — | N | pass | — |
| 512:16:8 | clustered | q_outer | pool_fresh_host | host | 904.2 | 216.8 | 483.4 | — | — | N | — | — |
| 512:16:8 | clustered | q_outer | pool_fresh_host_same_mask | host | 885.8 | — | — | — | — | N | — | — |
| 512:16:8 | clustered | q_outer | run_only | host | — | 105.0 | — | 105.1 | 0.0 | N | pass | pass |
| 512:16:8 | clustered | signature | cold_init | n/a | 893.5 | — | — | — | — | N | — | — |
| 512:16:8 | clustered | signature | exact_plan_cache | host | 348.8 | — | 77.5 | — | — | Y | — | — |
| 512:16:8 | clustered | signature | pool_fresh_gpu | gpu | 1040.9 | 217.4 | 572.6 | — | — | N | pass | — |
| 512:16:8 | clustered | signature | pool_fresh_host | host | 981.5 | 216.0 | 559.1 | — | — | N | — | — |
| 512:16:8 | clustered | signature | pool_fresh_host_same_mask | host | 946.7 | — | — | — | — | N | — | — |
| 512:16:8 | clustered | signature | run_only | host | — | 111.3 | — | 107.8 | 3.7 | N | pass | pass |
| 512:16:8 | common_private | bounded_merge | cold_init | n/a | 765.2 | — | — | — | — | N | — | — |
| 512:16:8 | common_private | bounded_merge | exact_plan_cache | host | 312.7 | — | 76.0 | — | — | Y | — | — |
| 512:16:8 | common_private | bounded_merge | pool_fresh_gpu | gpu | 940.7 | 187.4 | 510.8 | — | — | N | pass | — |
| 512:16:8 | common_private | bounded_merge | pool_fresh_host | host | 898.0 | 186.3 | 509.5 | — | — | N | — | — |
| 512:16:8 | common_private | bounded_merge | pool_fresh_host_same_mask | host | 882.2 | — | — | — | — | N | — | — |
| 512:16:8 | common_private | bounded_merge | run_only | host | — | 77.2 | — | 77.2 | 0.0 | N | pass | pass |
| 512:16:8 | common_private | q_outer | cold_init | n/a | 1806.9 | — | — | — | — | N | — | — |
| 512:16:8 | common_private | q_outer | exact_plan_cache | host | 313.5 | — | 76.2 | — | — | Y | — | — |
| 512:16:8 | common_private | q_outer | pool_fresh_gpu | gpu | 891.6 | 189.2 | 463.5 | — | — | N | pass | — |
| 512:16:8 | common_private | q_outer | pool_fresh_host | host | 830.0 | 185.2 | 453.3 | — | — | N | — | — |
| 512:16:8 | common_private | q_outer | pool_fresh_host_same_mask | host | 821.2 | — | — | — | — | N | — | — |
| 512:16:8 | common_private | q_outer | run_only | host | — | 78.6 | — | 78.4 | 0.0 | N | pass | pass |
| 512:16:8 | common_private | signature | cold_init | n/a | 804.1 | — | — | — | — | N | — | — |
| 512:16:8 | common_private | signature | exact_plan_cache | host | 328.7 | — | 76.3 | — | — | Y | — | — |
| 512:16:8 | common_private | signature | pool_fresh_gpu | gpu | 1024.1 | 199.7 | 580.5 | — | — | N | pass | — |
| 512:16:8 | common_private | signature | pool_fresh_host | host | 961.9 | 198.4 | 567.9 | — | — | N | — | — |
| 512:16:8 | common_private | signature | pool_fresh_host_same_mask | host | 949.4 | — | — | — | — | N | — | — |
| 512:16:8 | common_private | signature | run_only | host | — | 91.7 | — | 85.8 | 5.7 | N | pass | pass |
| 512:16:8 | random | bounded_merge | cold_init | n/a | 1010.0 | — | — | — | — | N | — | — |
| 512:16:8 | random | bounded_merge | exact_plan_cache | host | 379.7 | — | 76.4 | — | — | Y | — | — |
| 512:16:8 | random | bounded_merge | pool_fresh_gpu | gpu | 1085.0 | 257.9 | 579.3 | — | — | N | pass | — |
| 512:16:8 | random | bounded_merge | pool_fresh_host | host | 1038.2 | 255.6 | 570.9 | — | — | N | — | — |
| 512:16:8 | random | bounded_merge | pool_fresh_host_same_mask | host | 1005.0 | — | — | — | — | N | — | — |
| 512:16:8 | random | bounded_merge | run_only | host | — | 141.7 | — | 141.7 | 0.0 | N | pass | pass |
| 512:16:8 | random | q_outer | cold_init | n/a | 1269.3 | — | — | — | — | N | — | — |
| 512:16:8 | random | q_outer | exact_plan_cache | host | 332.9 | — | 76.3 | — | — | Y | — | — |
| 512:16:8 | random | q_outer | pool_fresh_gpu | gpu | 953.1 | 212.0 | 506.7 | — | — | N | pass | — |
| 512:16:8 | random | q_outer | pool_fresh_host | host | 895.3 | 208.6 | 490.6 | — | — | N | — | — |
| 512:16:8 | random | q_outer | pool_fresh_host_same_mask | host | 878.8 | — | — | — | — | N | — | — |
| 512:16:8 | random | q_outer | run_only | host | — | 101.6 | — | 101.6 | 0.0 | N | pass | pass |
| 512:16:8 | random | signature | cold_init | n/a | 895.6 | — | — | — | — | N | — | — |
| 512:16:8 | random | signature | exact_plan_cache | host | 347.1 | — | 77.0 | — | — | Y | — | — |
| 512:16:8 | random | signature | pool_fresh_gpu | gpu | 1093.5 | 230.2 | 612.3 | — | — | N | pass | — |
| 512:16:8 | random | signature | pool_fresh_host | host | 1036.8 | 231.8 | 604.7 | — | — | N | — | — |
| 512:16:8 | random | signature | pool_fresh_host_same_mask | host | 1004.7 | — | — | — | — | N | — | — |
| 512:16:8 | random | signature | run_only | host | — | 110.9 | — | 106.4 | 4.0 | N | pass | pass |

### 플래너 최적화 전후 (§8.6 적용)

계획 CPU 비용은 줄었지만 **공통 개선**이라 기준선도 함께 빨라져 상대 회귀 폭은 줄지 않았다.

|  | 조합 수 | Q대비 중앙값 | 최선 | 개선 조합 |
|---|---:|---:|---:|---:|
| 최적화 전 | 18 | +9.6% | +2.4% | 0 |
| 최적화 후 | 18 | +11.7% | +5.5% | 0 |

### 원본 JSON 스키마 (§14.2)

`raw/runtime/e4_full_cost.json` 의 각 레코드는 §14.2 필드를 그대로 갖는다. 측정하지 않은 값은 `null` 과 사유(`notes`)를 유지하고 0 으로 바꾸지 않았다. 레코드 예 (첫 `pool_fresh_gpu`):

```json
{
 "version": "rbc-v02",
 "source_commit": "075f7d7113197c2da78827d124ddc5f4920c1c7a",
 "gpu_uuid": "43a78ae4-f89f-ca72-2ff2-27e8a8a27f74",
 "mask_origin": "gpu",
 "timing_mode": "pool_fresh_gpu",
 "input_hash": "845d36759bad0568535fe9ead0cae54d",
 "mask_hash": "83b354ada2ea03798409a72273795916;2e46cedecadbb6093bd57f84aa45ba18;ee0e60469996ac255d6935fe461f14da;...",
 "q_position_hash": "33fc5709bb104f1fe887379839c2b9f6",
 "selected_policy": "q_outer",
 "plan_hash_before_upload": null,
 "plan_hash_executed": null,
 "split_policy": "none",
 "actual_tasks": 128,
 "kernel_buckets": 1,
 "direct_rows": 512,
 "multi_owner_rows": 0,
 "partial_slots": 0,
 "kv_unique_blocks": null,
 "kv_block_visits": null,
 "descriptor_live_bytes": 25704,
 "descriptor_capacity_bytes": 545740,
 "host_pin_calls_hotpath": 0,
 "device_allocation_requests_hotpath": 0,
 "d2h_bytes": 32720,
 "h2d_bytes": 66568,
 "h2d_copy_count": 2,
 "plan_cache_hit": false,
 "cpu_plan_pack_us": [
  448.38798930868506,
  464.7240275517106,
  405.2349831908941,
  "...(9개)"
 ],
 "dma_activity_us": [
  52.22399905323982,
  52.06400156021118,
  40.41599854826927,
  "...(9개)"
 ],
 "main_us": [],
 "merge_us": [],
 "gpu_total_us": [
  226.1440008878708,
  224.89599883556366,
  208.19200575351715,
  "...(9개)"
 ],
 "wall_us": [
  981.177028734237,
  980.95100838691,
  853.9860136806965,
  "...(9개)"
 ],
 "exact_edge_passed": null,
 "numeric_passed": true,
 "notes": [
  "mask indices 의 D2H 를 계측 안에 포함했다",
  "CSR offset 은 host 에 있다는 계약 (indexer 계산은 제외)",
  "main/merge 분리는 run_only 에서만 (여기서는 단계 분리 측정이 벽시계를 왜곡하므로 넣지 않았다)"
 ],
 "shape": "512:16:4",
 "pattern": "common_private"
}
```


## 계획 descriptor 덤프 (§4.4)

### common_private (Nq512/G8/D128/BK128, edges 8180)

| 후보 | max_m | split | BM | tasks | slots | partial | direct | multi | empty | sum_k | max_k | uniq | padded | plan ms | 예측 us | 정확 분할 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| q_outer | 16 | none | 2 | 256 | 512 | 0 | 512 | 0 | 0 | 6132 | 24 | 512 | 98112 | 1.17 | 98 | pass |
| q_outer | 16 | one_wave | 2 | 256 | 512 | 0 | 512 | 0 | 0 | 6132 | 24 | 512 | 98112 | 0.39 | 98 | pass |
| q_outer | 16 | two_waves | 2 | 264 | 528 | 32 | 496 | 16 | 0 | 6132 | 24 | 512 | 98112 | 0.43 | 98 | pass |
| q_outer | 32 | none | 4 | 128 | 512 | 0 | 512 | 0 | 0 | 5108 | 40 | 512 | 163456 | 0.47 | 160 | pass |
| q_outer | 32 | one_wave | 4 | 132 | 528 | 32 | 496 | 16 | 0 | 5108 | 40 | 512 | 163456 | 0.37 | 161 | pass |
| q_outer | 32 | two_waves | 4 | 264 | 1056 | 1056 | 0 | 512 | 0 | 5108 | 20 | 512 | 163456 | 0.84 | 126 | pass |
| q_outer | 64 | none | 8 | 64 | 512 | 0 | 512 | 0 | 0 | 4596 | 72 | 512 | 294144 | 0.41 | 384 | pass |
| q_outer | 64 | one_wave | 8 | 132 | 1056 | 1056 | 0 | 512 | 0 | 4596 | 36 | 512 | 294144 | 0.52 | 272 | pass |
| q_outer | 64 | two_waves | 8 | 264 | 2112 | 2112 | 0 | 512 | 0 | 4596 | 18 | 512 | 294144 | 0.91 | 212 | pass |
| q_outer | 128 | none | 16 | 32 | 512 | 0 | 512 | 0 | 0 | 4340 | 136 | 512 | 555520 | 0.38 | 1734 | pass |
| q_outer | 128 | one_wave | 16 | 132 | 2112 | 2112 | 0 | 512 | 0 | 4340 | 34 | 512 | 555520 | 0.59 | 755 | pass |
| q_outer | 128 | two_waves | 16 | 264 | 4224 | 4224 | 0 | 512 | 0 | 4340 | 17 | 512 | 555520 | 0.98 | 557 | pass |
| signature | 16 | none | 2 | 768 | 1024 | 1024 | 0 | 512 | 0 | 6132 | 8 | 512 | 98112 | 0.58 | 92 | pass |
| signature | 16 | one_wave | 2 | 768 | 1024 | 1024 | 0 | 512 | 0 | 6132 | 8 | 512 | 98112 | 0.52 | 92 | pass |
| signature | 16 | two_waves | 2 | 768 | 1024 | 1024 | 0 | 512 | 0 | 6132 | 8 | 512 | 98112 | 0.51 | 92 | pass |
| signature | 32 | none | 4 | 640 | 1024 | 1024 | 0 | 512 | 0 | 5108 | 8 | 512 | 98112 | 0.59 | 108 | pass |
| signature | 32 | one_wave | 4 | 640 | 1024 | 1024 | 0 | 512 | 0 | 5108 | 8 | 512 | 98112 | 0.48 | 108 | pass |
| signature | 32 | two_waves | 4 | 640 | 1024 | 1024 | 0 | 512 | 0 | 5108 | 8 | 512 | 98112 | 0.49 | 108 | pass |
| signature | 64 | none | 8 | 576 | 1024 | 1024 | 0 | 512 | 0 | 4596 | 8 | 512 | 98112 | 0.54 | 190 | pass |
| signature | 64 | one_wave | 8 | 576 | 1024 | 1024 | 0 | 512 | 0 | 4596 | 8 | 512 | 98112 | 0.46 | 190 | pass |
| signature | 64 | two_waves | 8 | 576 | 1024 | 1024 | 0 | 512 | 0 | 4596 | 8 | 512 | 98112 | 0.45 | 190 | pass |
| signature | 128 | none | 16 | 544 | 1024 | 1024 | 0 | 512 | 0 | 4340 | 8 | 512 | 98112 | 0.54 | 462 | pass |
| signature | 128 | one_wave | 16 | 544 | 1024 | 1024 | 0 | 512 | 0 | 4340 | 8 | 512 | 98112 | 0.46 | 462 | pass |
| signature | 128 | two_waves | 16 | 544 | 1024 | 1024 | 0 | 512 | 0 | 4340 | 8 | 512 | 98112 | 0.45 | 462 | pass |
| bounded_merge | 16 | none | 2 | 256 | 512 | 0 | 512 | 0 | 0 | 6132 | 24 | 512 | 98112 | 0.61 | 98 | pass |
| bounded_merge | 16 | one_wave | 2 | 256 | 512 | 0 | 512 | 0 | 0 | 6132 | 24 | 512 | 98112 | 0.52 | 98 | pass |
| bounded_merge | 16 | two_waves | 2 | 264 | 528 | 32 | 496 | 16 | 0 | 6132 | 24 | 512 | 98112 | 0.55 | 98 | pass |
| bounded_merge | 32 | none | 4 | 640 | 1024 | 1024 | 0 | 512 | 0 | 5108 | 8 | 512 | 98112 | 0.53 | 108 | pass |
| bounded_merge | 32 | one_wave | 4 | 640 | 1024 | 1024 | 0 | 512 | 0 | 5108 | 8 | 512 | 98112 | 0.49 | 108 | pass |
| bounded_merge | 32 | two_waves | 4 | 640 | 1024 | 1024 | 0 | 512 | 0 | 5108 | 8 | 512 | 98112 | 0.48 | 108 | pass |
| bounded_merge | 64 | none | 8 | 576 | 1024 | 1024 | 0 | 512 | 0 | 4596 | 8 | 512 | 98112 | 0.51 | 190 | pass |
| bounded_merge | 64 | one_wave | 8 | 576 | 1024 | 1024 | 0 | 512 | 0 | 4596 | 8 | 512 | 98112 | 0.48 | 190 | pass |
| bounded_merge | 64 | two_waves | 8 | 576 | 1024 | 1024 | 0 | 512 | 0 | 4596 | 8 | 512 | 98112 | 0.48 | 190 | pass |
| bounded_merge | 128 | none | 16 | 544 | 1024 | 1024 | 0 | 512 | 0 | 4340 | 8 | 512 | 98112 | 0.50 | 462 | pass |
| bounded_merge | 128 | one_wave | 16 | 544 | 1024 | 1024 | 0 | 512 | 0 | 4340 | 8 | 512 | 98112 | 0.46 | 462 | pass |
| bounded_merge | 128 | two_waves | 16 | 544 | 1024 | 1024 | 0 | 512 | 0 | 4340 | 8 | 512 | 98112 | 0.46 | 462 | pass |

### random (Nq512/G8/D128/BK128, edges 8168)

| 후보 | max_m | split | BM | tasks | slots | partial | direct | multi | empty | sum_k | max_k | uniq | padded | plan ms | 예측 us | 정확 분할 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| q_outer | 16 | none | 2 | 256 | 512 | 0 | 512 | 0 | 0 | 8036 | 32 | 512 | 128576 | 1.37 | 126 | pass |
| q_outer | 16 | one_wave | 2 | 256 | 512 | 0 | 512 | 0 | 0 | 8036 | 32 | 512 | 128576 | 0.44 | 126 | pass |
| q_outer | 16 | two_waves | 2 | 264 | 528 | 32 | 496 | 16 | 0 | 8036 | 32 | 512 | 128576 | 0.48 | 126 | pass |
| q_outer | 32 | none | 4 | 128 | 512 | 0 | 512 | 0 | 0 | 7798 | 64 | 512 | 249536 | 0.65 | 247 | pass |
| q_outer | 32 | one_wave | 4 | 132 | 528 | 32 | 496 | 16 | 0 | 7798 | 63 | 512 | 249536 | 0.44 | 245 | pass |
| q_outer | 32 | two_waves | 4 | 264 | 1056 | 1056 | 0 | 512 | 0 | 7798 | 32 | 512 | 249536 | 0.95 | 185 | pass |
| q_outer | 64 | none | 8 | 64 | 512 | 0 | 512 | 0 | 0 | 7354 | 120 | 512 | 470656 | 0.47 | 626 | pass |
| q_outer | 64 | one_wave | 8 | 132 | 1056 | 1056 | 0 | 512 | 0 | 7354 | 60 | 512 | 470656 | 0.60 | 426 | pass |
| q_outer | 64 | two_waves | 8 | 264 | 2112 | 2112 | 0 | 512 | 0 | 7354 | 30 | 512 | 470656 | 1.01 | 323 | pass |
| q_outer | 128 | none | 16 | 32 | 512 | 0 | 512 | 0 | 0 | 6541 | 215 | 512 | 837248 | 0.45 | 2657 | pass |
| q_outer | 128 | one_wave | 16 | 132 | 2112 | 2112 | 0 | 512 | 0 | 6541 | 54 | 512 | 837248 | 0.67 | 1041 | pass |
| q_outer | 128 | two_waves | 16 | 264 | 4224 | 4224 | 0 | 512 | 0 | 6541 | 27 | 512 | 837248 | 1.05 | 734 | pass |
| signature | 16 | none | 2 | 616 | 720 | 416 | 304 | 208 | 0 | 8036 | 16 | 512 | 128576 | 0.63 | 113 | pass |
| signature | 16 | one_wave | 2 | 616 | 720 | 416 | 304 | 208 | 0 | 8036 | 16 | 512 | 128576 | 0.58 | 113 | pass |
| signature | 16 | two_waves | 2 | 616 | 720 | 416 | 304 | 208 | 0 | 8036 | 16 | 512 | 128576 | 0.58 | 113 | pass |
| signature | 32 | none | 4 | 807 | 1110 | 997 | 113 | 399 | 0 | 7798 | 16 | 512 | 124896 | 0.70 | 160 | pass |
| signature | 32 | one_wave | 4 | 807 | 1110 | 997 | 113 | 399 | 0 | 7798 | 16 | 512 | 124896 | 0.60 | 160 | pass |
| signature | 32 | two_waves | 4 | 807 | 1110 | 997 | 113 | 399 | 0 | 7798 | 16 | 512 | 124896 | 0.59 | 160 | pass |
| signature | 64 | none | 4 | 1163 | 1866 | 1853 | 13 | 499 | 0 | 7354 | 16 | 512 | 118448 | 0.88 | 165 | pass |
| signature | 64 | one_wave | 4 | 1163 | 1866 | 1853 | 13 | 499 | 0 | 7354 | 16 | 512 | 118448 | 0.67 | 165 | pass |
| signature | 64 | two_waves | 4 | 1163 | 1866 | 1853 | 13 | 499 | 0 | 7354 | 16 | 512 | 118448 | 0.69 | 165 | pass |
| signature | 128 | none | 8 | 1780 | 3266 | 3265 | 1 | 511 | 0 | 6541 | 16 | 512 | 107904 | 0.96 | 310 | pass |
| signature | 128 | one_wave | 8 | 1780 | 3266 | 3265 | 1 | 511 | 0 | 6541 | 16 | 512 | 107904 | 0.78 | 310 | pass |
| signature | 128 | two_waves | 8 | 1780 | 3266 | 3265 | 1 | 511 | 0 | 6541 | 16 | 512 | 107904 | 0.78 | 310 | pass |
| bounded_merge | 16 | none | 2 | 408 | 512 | 0 | 512 | 0 | 0 | 8036 | 31 | 512 | 128576 | 0.68 | 130 | pass |
| bounded_merge | 16 | one_wave | 2 | 408 | 512 | 0 | 512 | 0 | 0 | 8036 | 31 | 512 | 128576 | 0.57 | 130 | pass |
| bounded_merge | 16 | two_waves | 2 | 408 | 512 | 0 | 512 | 0 | 0 | 8036 | 31 | 512 | 128576 | 0.56 | 130 | pass |
| bounded_merge | 32 | none | 4 | 526 | 829 | 534 | 295 | 217 | 0 | 7798 | 31 | 512 | 124896 | 0.71 | 185 | pass |
| bounded_merge | 32 | one_wave | 4 | 526 | 829 | 534 | 295 | 217 | 0 | 7798 | 31 | 512 | 124896 | 0.67 | 185 | pass |
| bounded_merge | 32 | two_waves | 4 | 526 | 829 | 534 | 295 | 217 | 0 | 7798 | 31 | 512 | 124896 | 0.65 | 185 | pass |
| bounded_merge | 64 | none | 4 | 1099 | 1802 | 1778 | 24 | 488 | 0 | 7354 | 16 | 512 | 118448 | 0.80 | 164 | pass |
| bounded_merge | 64 | one_wave | 4 | 1099 | 1802 | 1778 | 24 | 488 | 0 | 7354 | 16 | 512 | 118448 | 0.76 | 164 | pass |
| bounded_merge | 64 | two_waves | 4 | 1099 | 1802 | 1778 | 24 | 488 | 0 | 7354 | 16 | 512 | 118448 | 0.78 | 164 | pass |
| bounded_merge | 128 | none | 8 | 1780 | 3266 | 3265 | 1 | 511 | 0 | 6541 | 16 | 512 | 107904 | 0.85 | 310 | pass |
| bounded_merge | 128 | one_wave | 8 | 1780 | 3266 | 3265 | 1 | 511 | 0 | 6541 | 16 | 512 | 107904 | 0.79 | 310 | pass |
| bounded_merge | 128 | two_waves | 8 | 1780 | 3266 | 3265 | 1 | 511 | 0 | 6541 | 16 | 512 | 107904 | 0.78 | 310 | pass |

### clustered (Nq512/G8/D128/BK128, edges 8192)

| 후보 | max_m | split | BM | tasks | slots | partial | direct | multi | empty | sum_k | max_k | uniq | padded | plan ms | 예측 us | 정확 분할 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| q_outer | 16 | none | 2 | 256 | 512 | 0 | 512 | 0 | 0 | 8052 | 32 | 509 | 128832 | 1.23 | 126 | pass |
| q_outer | 16 | one_wave | 2 | 256 | 512 | 0 | 512 | 0 | 0 | 8052 | 32 | 509 | 128832 | 0.42 | 126 | pass |
| q_outer | 16 | two_waves | 2 | 264 | 528 | 32 | 496 | 16 | 0 | 8052 | 32 | 509 | 128832 | 0.44 | 126 | pass |
| q_outer | 32 | none | 4 | 128 | 512 | 0 | 512 | 0 | 0 | 7796 | 64 | 509 | 249472 | 0.66 | 247 | pass |
| q_outer | 32 | one_wave | 4 | 132 | 528 | 32 | 496 | 16 | 0 | 7796 | 64 | 509 | 249472 | 0.39 | 247 | pass |
| q_outer | 32 | two_waves | 4 | 264 | 1056 | 1056 | 0 | 512 | 0 | 7796 | 32 | 509 | 249472 | 0.85 | 185 | pass |
| q_outer | 64 | none | 8 | 64 | 512 | 0 | 512 | 0 | 0 | 7322 | 128 | 509 | 468608 | 0.47 | 656 | pass |
| q_outer | 64 | one_wave | 8 | 132 | 1056 | 1056 | 0 | 512 | 0 | 7322 | 64 | 509 | 468608 | 0.57 | 441 | pass |
| q_outer | 64 | two_waves | 8 | 264 | 2112 | 2112 | 0 | 512 | 0 | 7322 | 32 | 509 | 468608 | 0.93 | 329 | pass |
| q_outer | 128 | none | 16 | 32 | 512 | 0 | 512 | 0 | 0 | 6587 | 244 | 509 | 843136 | 0.39 | 2858 | pass |
| q_outer | 128 | one_wave | 16 | 132 | 2112 | 2112 | 0 | 512 | 0 | 6587 | 59 | 509 | 843136 | 0.60 | 1090 | pass |
| q_outer | 128 | two_waves | 16 | 264 | 4224 | 4224 | 0 | 512 | 0 | 6587 | 30 | 509 | 843136 | 0.96 | 767 | pass |
| signature | 16 | none | 2 | 529 | 546 | 68 | 478 | 34 | 0 | 8052 | 16 | 509 | 128832 | 0.53 | 108 | pass |
| signature | 16 | one_wave | 2 | 529 | 546 | 68 | 478 | 34 | 0 | 8052 | 16 | 509 | 128832 | 0.50 | 108 | pass |
| signature | 16 | two_waves | 2 | 529 | 546 | 68 | 478 | 34 | 0 | 8052 | 16 | 509 | 128832 | 0.51 | 108 | pass |
| signature | 32 | none | 4 | 563 | 618 | 204 | 414 | 98 | 0 | 7796 | 16 | 509 | 124944 | 0.57 | 149 | pass |
| signature | 32 | one_wave | 4 | 563 | 618 | 204 | 414 | 98 | 0 | 7796 | 16 | 509 | 124944 | 0.54 | 149 | pass |
| signature | 32 | two_waves | 4 | 563 | 618 | 204 | 414 | 98 | 0 | 7796 | 16 | 509 | 124944 | 0.50 | 149 | pass |
| signature | 64 | none | 4 | 609 | 731 | 404 | 327 | 185 | 0 | 7322 | 16 | 509 | 118096 | 0.56 | 146 | pass |
| signature | 64 | one_wave | 4 | 609 | 731 | 404 | 327 | 185 | 0 | 7322 | 16 | 509 | 118096 | 0.51 | 146 | pass |
| signature | 64 | two_waves | 4 | 609 | 731 | 404 | 327 | 185 | 0 | 7322 | 16 | 509 | 118096 | 0.49 | 146 | pass |
| signature | 128 | none | 8 | 681 | 936 | 719 | 217 | 295 | 0 | 6587 | 16 | 509 | 109440 | 0.60 | 264 | pass |
| signature | 128 | one_wave | 8 | 681 | 936 | 719 | 217 | 295 | 0 | 6587 | 16 | 509 | 109440 | 0.50 | 264 | pass |
| signature | 128 | two_waves | 8 | 681 | 936 | 719 | 217 | 295 | 0 | 6587 | 16 | 509 | 109440 | 0.51 | 264 | pass |
| bounded_merge | 16 | none | 2 | 495 | 512 | 0 | 512 | 0 | 0 | 8052 | 31 | 509 | 128832 | 0.57 | 133 | pass |
| bounded_merge | 16 | one_wave | 2 | 495 | 512 | 0 | 512 | 0 | 0 | 8052 | 31 | 509 | 128832 | 0.51 | 133 | pass |
| bounded_merge | 16 | two_waves | 2 | 495 | 512 | 0 | 512 | 0 | 0 | 8052 | 31 | 509 | 128832 | 0.53 | 133 | pass |
| bounded_merge | 32 | none | 4 | 469 | 524 | 22 | 502 | 10 | 0 | 7796 | 31 | 509 | 124944 | 0.56 | 180 | pass |
| bounded_merge | 32 | one_wave | 4 | 469 | 524 | 22 | 502 | 10 | 0 | 7796 | 31 | 509 | 124944 | 0.51 | 180 | pass |
| bounded_merge | 32 | two_waves | 4 | 469 | 524 | 22 | 502 | 10 | 0 | 7796 | 31 | 509 | 124944 | 0.50 | 180 | pass |
| bounded_merge | 64 | none | 4 | 560 | 682 | 314 | 368 | 144 | 0 | 7322 | 16 | 509 | 118096 | 0.57 | 144 | pass |
| bounded_merge | 64 | one_wave | 4 | 560 | 682 | 314 | 368 | 144 | 0 | 7322 | 16 | 509 | 118096 | 0.53 | 144 | pass |
| bounded_merge | 64 | two_waves | 4 | 560 | 682 | 314 | 368 | 144 | 0 | 7322 | 16 | 509 | 118096 | 0.54 | 144 | pass |
| bounded_merge | 128 | none | 8 | 654 | 910 | 674 | 236 | 276 | 0 | 6587 | 19 | 509 | 109440 | 0.60 | 269 | pass |
| bounded_merge | 128 | one_wave | 8 | 654 | 910 | 674 | 236 | 276 | 0 | 6587 | 19 | 509 | 109440 | 0.58 | 269 | pass |
| bounded_merge | 128 | two_waves | 8 | 654 | 910 | 674 | 236 | 276 | 0 | 6587 | 19 | 509 | 109440 | 0.56 | 269 | pass |


## 작업 중 발견·수정한 결함

측정을 신뢰하려면 계측 자체의 결함을 먼저 드러내야 한다. 이번 작업에서 찾아 고친 것들이다.

| 증상 | 원인 | 조치 |
|---|---|---|
| 선택기가 최악의 계획 선택 (regret 415%) | 회귀가 음수 시간을 예측했고 `max(0, ·)` 클리핑 때문에 0us 로 보여 '가장 빠른 후보'가 됨 | 비음수 최소제곱(NNLS)으로 교체. 음수·범위 밖 예측은 `inf` 로 배제 |
| 선택기가 189셀 전부에서 무력화 (예측 전부 inf) | `refit` 이 raw 에서 신규 특징을 복원하지 못해 조용히 0 을 채웠고, 그 결과 관측 범위가 0 이 되어 모든 입력이 '범위 초과'로 판정 | raw 에 특징 저장, `refit` 은 필드 누락 시 명시적 실패. 특징값 기반 외삽 거부 제거 |
| max_m 선택 오류 (regret 217%) | (BM,G) 구간별 독립 회귀의 예측값을 구간을 넘어 직접 비교. 각 구간이 자기 절편을 가지므로 교정이 보정하지 않은 비교 | max_m 은 교정 lookup 으로 결정. 후보 간 BM 구간이 다르면 비교를 거부하고 Q-outer |
| E4 의 CPU plan+pack 4.9ms | `slots.py` 의 FULL membership 판정이 task 수만큼 파이썬 루프 (signature 816 task) | task 별 기대 마스크를 KV 길이만큼 펼쳐 한 번에 비교. 4,888 → 847us |
| E4 의 signature 첫 측정 160ms | 그 후보가 쓰는 커널 variant(FULL=False + merge)의 JIT 가 첫 측정에 섞임 | 후보별 warmup 분리, `jit_first_us`·`warmup_us` 를 별도 항목으로 기록 |
| 플래너 재작성 후 크래시 | OpenMP worker 가 마스터의 `thread_local` 스크래치를 보지 못함 | 마스터 인스턴스를 포인터로 전달 |
| FINAL_RESULT 의 P1 효과 65% | 생성기가 max_m 을 전부 섞은 중앙값을 비교 | 정책별 최적 max_m 기준으로 집계 수정 (실측과 일치하는 1.8%) |
| 게이트 46/81 vs 45/81 불일치 | 생성기가 seed 를 섞어 최소를 취해 운 좋은 seed 하나가 결과를 정함 | (정책, max_m)별 중앙값을 먼저 낸 뒤 최적 선택 |


## 판정과 해석

### 세 단계 판정 (§1.3)

| 판정 | 결과 | 근거 |
|---|---|---|
| 구현 개선 | **통과** | P2 direct output 이 동일 정책의 v0.1 대비 GPU 시간 −5~8%. 플래너 최적화로 계획 CPU −20~29% |
| RBC GPU 기여 | **부분 통과** | §13.1 게이트 46/81 영역 (중앙값 −35.7%, 최대 −69.5%). 다만 승자는 signature_only 36 / bounded_merge 10 으로, RBC 고유 병합이 signature 까지 이긴 것은 10개 영역 |
| CPU–GPU 전체 개선 | **미달** | E4 의 9개 입력 전부 회귀. 목표는 완료시간 10% 감소 |

### 비용이 남은 지점 (§16 요구)

**1. GPU 이득의 원인이 설계의 주장과 다르다.** 게이트 통과 영역 대부분에서 KV 방문 비가 1.00 이다. 즉 RBC 가 내세운 KV 재읽기 감소가 아니라 Q-outer union 직사각형의 padded dot 낭비가 사라진 것이 이득의 원인이다. E1 이 이를 직접 보여준다 — max_m 을 16→128 로 키우면 KV 방문은 −29% 줄지만 GPU 시간은 76.9→1,751.7us(+2,180%)로 악화한다. 이득원과 비용이 반대 방향이다.

**2. GPU 이득이 CPU 계획비보다 작다.** Nq512 에서 CPU plan+pack 이 405~528us 인데 §8.7 목표는 50us 이고, GPU 이득의 절대 크기는 20~75us 다. §8.6 을 적용해 계획비를 20~29% 줄였지만 이는 **공통 개선**이라 기준선도 함께 빨라졌다.

**3. 남은 격차는 알고리즘 고유 비용이다.** signature 는 같은 입력에서 task 를 816개, q_outer 는 128개 만든다. task 수에 비례하는 비용(슬롯 생성, kmask 채우기, descriptor 길이)은 할당을 없애도 남는다. 구현 최적화로 제거할 성질이 아니다.

### 해석 제한 (§15)

- 합성 mask 연산자 수준의 결과다. 모델 tok/s·서비스 SLO·신규성 주장으로 확장하지 않는다.
- Q-outer fallback·pinned pool·일반 커널 개선은 RBC 분해의 성과가 아니다. E2 표에서 '공통 Q-outer 도 얻은 효과'와 'RBC−Q'를 분리했다.
- KV 합집합 바이트로 HBM 비용을 계산하지 않았다. `kv_block_visits` 는 발행량 특징이며 실측 HBM 바이트가 아니다.
- P6/E5(pipeline·실제 capture 확대)는 §13.1 규정에 따라 진행하지 않았다.
- NCU/NSYS 프로파일은 수집하지 않았다. 시간 모델은 §7.5 대로 profiling 없이 교정했다.


## 산출물과 재현

```bash
python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage reproduce
python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage plan-audit
python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage kernel
python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage calibrate
python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage online
python tools/run_v02_campaign.py --config configs/rbc_v02.yaml --stage report
```

`RBC_Attn_v02_result/MANIFEST.sha256` — 파일 56개.

<details><summary>전체 해시 목록</summary>

```
a50651348be9c0ce315b11c34ebd66ae382665a11eb86e43252ce35fb8a4945a  FINAL_RESULT.md
6edb0caaa6a78eb6020cb3db552a53d976bf515f3ce1049cc8ecb7a767f3b336  calibration/calibration_raw.json
0767692e270d6960c6cfa713de104e1cd4f32da439071b2b73d81dcc9be5061a  calibration/e3_selector.json
47d6af105d52100605a617f6d8b81474485a51f23fe4b7d1fd0f027a32530efe  configs/plan_cost_table.json
108424b7dddb9438a58a364a0b2de312fc7260a8832e91d3994fda071a370e22  configs/rbc_v02.yaml
11fcbe803c3c6d6b4862026828d25cfd6975cbd1c92d73fd074ed4dcad2a89af  manifest/base_commit.txt
8f097c1bbb8cc9bbe712602f500043f25dc976aca979fb3e47fafd44eb7eab17  manifest/gpus.txt
104ebe6ed482f71ce1e22a05960229a2b20a1b0c4be66ca35bce0e087af9c549  manifest/ncu_metrics.txt
2d9d5b790a56200c6f276274de256d1ef993da377405dd41fba702d892ba88f0  manifest/ncu_version.txt
2d006cfc938a957e6bd2055a01c8aedaa00e391369fd1672ec8962b6d2d8b982  manifest/nsys_version.txt
c9c2536961b47052032d99522bdf59acfb6b0d0cb18ec77b96f27018ab40a68f  manifest/numa.txt
57d8639181f7f505e376df974268eb1d38f3964bd0f88bc318efa7f0ce8f2fd1  manifest/pip_freeze.txt
61d478dc55988c20c1b541a8a5d5907a2bae00e0b2377b2fab2e574b3a8aa59f  manifest/source_map.json
892636df475958c8245171eaa4ab4369f49fbcef7d10a83dfcac5c370edf000c  manifest/topology.txt
66eb307c9593f7a832cfd812a4af75d0ae0248a300b271b9392426e6786cf4d8  raw/kernel/e0_baseline.json
b6a5cd43cdac699c798ec8e728a838988208677379367ce26b7de5d30996523e  raw/kernel/e1_finalization.json
b9a552f5bb37825a75bdcf9894573dd58c9e07970600d65b07e38c5c16e7f408  raw/kernel/e2_ablation.json
c70a57058cfbbc9ee524091e6da6b01cc20da3e7f0df3fdbda0339636f216d68  raw/kernel/gate_region_sweep.json
88f2b1fbdf2b78ad995efd4456eb598eda7f21a5bb3de7309347f3c66eca398e  raw/kernel/gate_region_sweep_small.json
5772c40db1a478f2680121e16020742f956b4853b031925dc3d3bf5c399e4fe3  raw/runtime/e4_full_cost.json
21f251d2e4a34fd9488bbbde07975253d03fa4f1ba6acb1e2c236018842d345d  raw/runtime/e4_full_cost_before_planner_opt.json
acb03e8c58c097f402f48aa392af9e60c14c576b86e61dcecc58a144601c6a16  src/RESULT.md
af0aab65fe0bdc578faffeff3f3c8ffb24b5f73c3e5b8cf6448348c1224c6e66  src/bench.py
31e1c4fbf48b788e8ac73203e5045b5b6809b35052699e2415c7198f7dd65b9e  src/build.sh
df7df0aa0c28dff42d35e298eec6d21bba37fb1aaf81bc6399f020cd9e9f1063  src/cpp/planner.cpp
5df3dd9b7cfe29fdd84ad0a669c6126684ecc5b66bad785204f01f06db8e80a6  src/profile_one.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  src/rbc/__init__.py
c21b175e41b196682f94aeedcf969611eaf4db376352cb7005ef6621868e0ff8  src/rbc/data.py
a85407eeb62675221cde4583055f4d9731cddaed8b06fbb9753dde7307e22b39  src/rbc/kernels.py
e78223cd78f4ddc8cbe53723b75c4a46db6f380a52932cb4934d94eddb350fc9  src/rbc/native.py
93799b9ee2db44c3f88211f82c9a4e391b8faa244c232b1a20838afc0f821850  src/rbc/planner.py
32c404ea182ba726757ef198e26a3fa41fed1e4eb2b537c407a233c98a8fe0a0  src/rbc/runtime.py
a2a7f2f97928b81267685fd0c1214f6d67460642d1fd4c2c7396f22b5d4bf527  src/rbc/select.py
dfba7552332f5b2f4ca434ba06bf45efaec3d8091f02ae32e25df459ac956e79  src/rbc/slots.py
4a719b6fd52462d70daaba3839a8f48395edef40fa0c5d52d00334d1f533ea55  src/run_gpu.sh
17450c193430b3fb74ccdd43a4ca05d218c4fcd5f63216a87cf86ef25d7f29d5  src/run_host_tests.sh
c250b424c2066f40c1fa2ba9a98ef4ec78da776979b48bba6f99a69ae9a20224  src/run_matrix.sh
216bb9f3e020f14f012a30ce359a6cdeedeb438a08084317155b60c10036ad24  src/stream_bench.py
28f80a95f4715ddc3e9ff98a2bfd39a081724b4669e6409e6a8f8dfd929bbc52  tests/test_gpu.py
43e5deb8a86821c3c6ed65ef46bbd05f6cd9d07517c79194c679492622a719a8  tests/test_host.py
5ea20b96521d6cf6a9b4a0e0d623ecb8bd9138ab134ce0a65d08e1a03d3b7560  tests/test_native_metadata.py
1456b8397c205c17e274757fd29e851ddac5798ce53d43b738f5fdd5f29351ac  tests/test_v02_output_ownership.py
2f73b434002e5247a38f0bc1daff3368c04d067157177ae789789b0f2e5577ad  tests/test_v02_plan_finalization.py
6683e88be5fe4827f7f68d75249e00ddf4332301a3ab720154fbddd7136b6b60  tests/test_v02_runtime_lifetime.py
7d041775a25aa7fa849cd51be15c7b7c1390f8c1eab6af7e7e0ae5961e6fd354  tools/calibrate_plan_cost.py
a5b45744237c85e4d93b7e60e24d1fd0a778fdc413ae3d5f51f2bf8e8005230a  tools/compare_v02.py
f8e63aa95b1e16a5167063c2a6752e6c32040922e2cacaf5f269f1d29b3ceff2  tools/dump_plan.py
b8c8e6b15d3d7a9b80507018081c08020fd95c6d3c490d477ab0c4399ca2fca9  tools/e0_baseline.py
99b252be86d40bd69dc9c5ba040b619cf1a32171846692dae1a6631e8de3b861  tools/e1_finalization.py
47f6b832b30d0291ba4f38923d5d80c661f09fafd1f0def582596930abfe86b1  tools/e2_ablation.py
c26071668abcb9712ffd5d72304e1ba17228c82bcfbf195f5ca8dc888c491a2f  tools/e4_full_cost.py
d1dd9018cc3220c7c9ca96d6710363bfae3a64ae36fcc05d5fe106b4a69514e3  tools/gate_region_sweep.py
9c9fb5ae118c7644a3e1f580e55ae168e4ca9018af8f36fda3e0ee58ca22020d  tools/make_final_result.py
df5e78a87c800fb64f961699af05494ead7af473933ed494d3aaaa2b145e5ae6  tools/pack_result.py
f4a849b92a136faad437452243c045046a18bc68dbb11e9e24e2193f97eeb30d  tools/refit_plan_cost.py
93700157f783ac9667a44d3bbb0b050d650219af5d8d2f083423c3bb55cb55b7  tools/run_v02_campaign.py
```

</details>

### 원본 파일

| 경로 | 크기 |
|---|---:|
| results/v02/kernel/e0_baseline.json | 54 KB |
| results/v02/kernel/e1_finalization.json | 47 KB |
| results/v02/kernel/e2_ablation.json | 186 KB |
| results/v02/kernel/gate_region_sweep.json | 147 KB |
| results/v02/kernel/gate_region_sweep_small.json | 1335 KB |
| results/v02/manifest/base_commit.txt | 0 KB |
| results/v02/manifest/gpus.txt | 0 KB |
| results/v02/manifest/ncu_metrics.txt | 976 KB |
| results/v02/manifest/ncu_version.txt | 0 KB |
| results/v02/manifest/nsys_version.txt | 0 KB |
| results/v02/manifest/numa.txt | 1 KB |
| results/v02/manifest/pip_freeze.txt | 7 KB |
| results/v02/manifest/source_map.json | 2 KB |
| results/v02/manifest/topology.txt | 2 KB |
| results/v02/plan/dump_clustered.json | 18 KB |
| results/v02/plan/dump_common_private.json | 18 KB |
| results/v02/plan/dump_random.json | 18 KB |
| results/v02/runtime/e4_full_cost.json | 289 KB |
| results/v02/runtime/e4_full_cost_before_planner_opt.json | 289 KB |
| results/v02/select/calibration_raw.json | 1874 KB |
| results/v02/select/e3_selector.json | 486 KB |

