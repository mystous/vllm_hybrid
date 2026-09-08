# IDE_034 2차 — 측정 워크로드(sonnet 512/128 C32) 자체의 decode 패스만으로 hot set 재구성 (2026-09-09 01:34~02:00)
트레이스: decode 패스 512개 (≤64 tok). 현행(prompt) hotmap 의 decode 커버리지 H=96 96.0% / E[D_c](B32) 7.8 → 새 hotmap 97.9% / 4.7. 층당 교체 ≈10.6개.

| 항목 | prompt hotmap (기준, 같은 계측 빌드) | decode-only hotmap (2차) | 변화 |
|---|---|---|---|
| decode 층당 활성 cold expert 중앙값 / 평균 | 6 / 6.5 | **4 / 4.3** | −33% |
| 커널 total 중앙값 (numa0) | 500µs | **352µs** | −30% |
| C32 TPOT | 43.5 | **38.0** | −12.5% |
| C32 TTFT | 1304 | **2560** | +96% |
| C32 tok/s | 580~600 | 554 | −7% |
| C64 tok/s / TPOT | 760~786 / 66.7~69.8 | 762.8 / **57.0** | TPOT −15% |
| GSM40 | 97.5% | **97.5%** | 무손실 |
| wrapper (numa_job − 커널, merge) | 20 / 21µs | 4 / 17µs | — |

**판정**: 사전 등록 기준 중 D_c ≤4.5 **통과**, GSM40 **통과**, C32 ≥640 **미달**. decode 측 기전은 예측대로 실현 (D_c −33% → 커널 −30% → TPOT −12.5%) 되었으나 **prefill 이 느려져** (TTFT ×2: prefill 은 cold (token,expert) 쌍 수에 비례하는 AMX 연산이고, decode 최적 hot set 은 prefill 커버리지를 낮춤) 입력 512/출력 128 워크로드의 처리량은 −7%.
**함의**: 최적 hot set 이 **phase 에 따라 다르다** (prefill: 쌍 수 최소화, decode: 배치당 distinct cold expert 최소화). GPU 상주 집합은 하나이므로 워크로드의 prefill:decode 작업 비율로 가중한 선택이 필요 → 3차: score(e) = α·p_prefill(e) + (1−α)·p_decode(e), α ∈ {0.25, 0.5, 0.75} 스윕 (`…_ide034_mixed_hotmap/`). 모델 (M2 정책) 에 "phase-가중 hot set" 항으로 편입.
