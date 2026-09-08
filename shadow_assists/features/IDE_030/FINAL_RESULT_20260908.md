# IDE_030 최종 결과 (2026-09-08) — CPU 구간 정량 분해 → hot-96 + deferred: C=32 490.9 tok/s (+53.5%)

전일 확정 구성 (hot-80, 319.9~327 tok/s) 을 기준으로, 교차 실행 공사에서 얻은 측정 도구 (CPU 층 호출 마이크로벤치, 라우팅 트레이스 재분석, torch profiler 분해) 로 CPU 구간의 실체를 정량화하고, 그 결과가 가리키는 두 손잡이 (cold expert 바이트 감축, deferred 겹침) 를 적용했다.

## 1. 최종 표 (violet-h100-016, Qwen3-Coder-480B-A35B FP8, TP=4 H100×4 + Xeon 8480+×2 AMXINT4 96스레드, turbo OFF 2.0GHz, sonnet 512/128)

| 구성 | C=16 | C=32 | C=64 | GSM8K |
|---|---|---|---|---|
| GPU-only 기준 (8-27, R1 대체 480B GPU-only OOM → 참고 56.3) | — | 56.3 | — | — |
| hot-80 (전일 최종) | 251.5 | 319.9~327 | 343.1 | 97.5% (40) |
| **hot-96** | — | **458.9** | — | 95.0% (40) |
| **hot-96 + deferred 4 (최종)** | **358.6** | **490.9** (프로파일 중 462.1) | **408.2** | **97.0% (100)** |

- 최종 구성 플래그 (hot-80 대비 변경분): `--kt-num-gpu-experts 96 --mem-fraction-static 0.92 --max-total-tokens 24576 --cuda-graph-max-bs 64 --kt-max-deferred-experts-per-token 4`
- greedy 4문항 정상 (Paris / fibonacci / 삼원색 / 17×23 풀이 시작). GSM8K 100문항 97.0%.
- C=64 가 C=32 보다 낮은 이유: max-total-tokens 24576 에 64×640≈41K 토큰이 안 들어가 KV 압박. hot expert 수 ↔ KV 용량 (동시성 상한) 의 교환 관계. hot-104 는 가중치 후 여유 2.95 GB 로 불가 → **TP=4·80GB 에서 hot-96 이 메모리 상한**.
- 결과 디렉토리: `eval/results/20260908_091623_ide030_final_hot96_def4/` (bench_c16/32/64, bench_c32_noprof, gsm100.json, 4 rank 트레이스, trace_decomp_TP0.txt), `20260908_090*_ide030_hot96_mf92_c32`, `…_hot96_mf92_def4_c32`, `…_hot80_def2/def4_c32`, `20260908_090741_ide030_gate_hot96` (GSM40).

## 2. CPU 구간의 실체 (측정)

### 2.1 480B AMXINT4 층 1회 호출 마이크로벤치 (`eval/results/20260908_085300_ide030_mb480_cpu_layer/`)
| T(토큰) | n_cold=1 | 2 | 4 | 7 | 12 | 24 | 48 |
|---|---|---|---|---|---|---|---|
| 32 | 8735µs | 5227 | 4858 | 4849 | 4529 | 4823 | 5297 |
| 16 | 4357 | 2632 | 2509 | 2535 | 2406 | 2783 | 3727 |
| 1 | 1567 | 812 | 435 | 454 | 440 | 443 | 467 |

- T=1 (쌍 8개): n_cold≥4 에서 ~440µs 평평 = 8 expert × 23.6MB = 189MB / 0.44ms ≈ **430 GB/s** — 이 기계의 DDR 실측 천장 (torch 96스레드 읽기 ~390 GB/s, 복사 ~250 GB/s) 과 같은 수준. **cold expert 스트리밍은 이미 대역폭 한계.**
- T=32, n_cold=7: 같은 165MB 인데 4.85ms → 토큰×expert 쌍당 ~19µs 의 vec_mul (qlen ≤ 4·E/k=80 이면 AMX 타일 대신 벡터 곱) 계산 한계. 서빙 (expert 당 행 ≈1.2) 에선 무의미.
- 모델: expert 당 ≈ 55µs 스트리밍 + 19µs × (행−1). 서빙 층당 ≈ 6.7 expert → ~450µs → 62층 ≈ 28~31ms.

### 2.2 라우팅 트레이스 (8-29 `expert_distribution_recorder`, sonnet 18,776 토큰)
- hot-80 커버리지 평균 97.0% (min 85.7, max 98.9). bs=32 에서 층당 기대 distinct cold expert **6.7** (bs=16: 3.6, bs=64: 12.1). 스텝당 cold ≈ 9.8 GB.
- 시사점: 배치 2분할은 층당 CPU 호출 2회 (3.6+3.6 expert, 고정비 2배) → **교차 실행은 구조적 손해** (순차 모드 실측 −20% 가 이것). cold 바이트 감축 (hot expert 확대) 이 가장 큰 손잡이.

### 2.3 torch profiler 분해 (C=32 정상 상태, hot-96+def4, TP0, `trace_decomp_TP0.txt`)
- decode 스텝 39.0ms = GPU 커널 27.7ms (fused_moe_kernel 17.4 — hot expert 96개 GPU 연산, fp8 GEMM 2.9, AR fusion 1.3, attention ~1.2) + 복사 1.9ms (H2D 19µs×60, D2H 3µs×181) + **CPU 대기 유휴 9.9ms**. 스텝 사이 호스트 갭 0~0.7ms (호스트는 병목 아님).
- hot-80 시절 "GPU 26 / CPU 대기 40" 에서 **GPU 27.7 / CPU 대기 9.9** 로 균형이 뒤집힘. 남은 CPU 노출은 스텝의 25%.

## 3. 교차 실행 (interleaved dual micro-batch) 의 종결 근거
- 동시 재생: 격리 수리 5건 + 신호식 대기 C++ (비블로킹 sync) + 층 사다리 (rank-로컬 memop 격자) + AR fusion 비활성까지 적용해도 두 통신그룹 (custom_all_reduce_v2, symmetric memory) 의 collective 단계에서 정지 — 미해결.
- 그러나 §2 의 측정으로 **상한 자체가 낮음** 이 확정: 분할은 층당 CPU 호출 2배. 단일 통신그룹 + 완전 순서 (층당 4간선) 로 교착을 피하면 겹침이 소거됨. 따라서 우선순위에서 제외.
- 부산물 (보존·upstream 후보): 신호식 대기 C++ (`cpuinfer.h`: `submit_signal_with_cuda_stream` / `wait_signal_on_stream` / `write_flag_on_stream`, graph host 노드 인자 영속성 요구), kt×sglang 다중-graph 격리 수리 5건, 계측 (`[kt-signal]` 오류 검사). 상세: `interleave_design.md`.

## 4. 남은 손잡이 (호스트 측)
- CPU 대기 9.9ms/스텝: 바이트 한계 (DDR 천장) — 더 줄이려면 cold 바이트 감축뿐인데 GPU 메모리 상한. KV 를 CPU DRAM 으로 (IDE_025 DRAM KV tier) 옮겨 GPU 메모리를 hot expert 에 더 배정하는 조합이 다음 후보 (hot-112 = +11.8 GB/GPU 필요).
- C=64 KV 압박: 같은 조합 (DRAM KV) 으로 해소 가능성.
- GPU 측 fused_moe_kernel 17.4ms (E=96 튜닝표 부재) 는 GPU 손잡이라 본 과제 범위 밖 — 정보로만 기록.
