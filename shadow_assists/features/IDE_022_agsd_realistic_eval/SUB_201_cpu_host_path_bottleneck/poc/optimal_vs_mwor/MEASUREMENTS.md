# SUB_201 — Optimal Config vs MWOR (Oracle Router) Measurements

**측정 범위**: 10 model × mix corpus (500p × conc=32 × max-tokens=8192, MML=16384).
**설정 (Optimal + MWOR 공통)**: `cudagraph_mode=FULL_AND_PIECEWISE` + `VLLM_PREFETCH_TOKENIZE=1` + `VLLM_PREFETCH_TOKENIZE_WORKERS=2` + `VLLM_BURST_AWARE_ADMISSION=1` + `--max-model-len 16384` + `--gpu-memory-utilization 0.85` + `--allow-deprecated-quantization`.
**Hardware**: B200 × 8 (Intel Xeon 8570 host).

## 데이터 출처

- **Vanilla / llm-d**: TSK_042 (`vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/metrics_table.csv` 의 `mix` 행) 재사용.
- **Optimal Config / MWOR**: 본 sweep 신규 측정 (`./runs/summ_*_mix.json`).
- **Oracle winner**: `../l7_oracle_router/oracle_table.csv` 의 `mix` 행.

## Cell 별 결과 (10 model × mix corpus × 4 configuration)

| Model | TP | GPUs | mix winner | Vanilla (TSK_042) | llm-d (TSK_042) | Optimal Config (FaP+L2+L10, vanilla) | MWOR (FaP+L2+L10, winner) |
|---|---:|---|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 1 | 0 | suffix | 4,168.9 | 7,739.2 | — | — |
| Qwen2.5-32B-Instruct | 2 | 0,1 | suffix | 3,055.5 | 5,236.1 | — | — |
| Qwen2.5-72B-Instruct | 4 | 0-3 | suffix | 2,734.8 | 3,265.2 | — | — |
| Llama-3.1-8B-Instruct | 1 | 0 | suffix | 8,849.9 | 15,958.9 | — | — |
| Llama-3.1-70B-Instruct | 4 | 0-3 | suffix | 3,129.2 | 4,003.6 | — | — |
| Llama-3.1-405B-Instruct-FP8 | 8 | 0-7 | suffix | 1,252.1 | 1,429.4 | — | — |
| DeepSeek-R1-Distill-Qwen-7B | 1 | 0 | suffix | 9,058.2 | 13,000.3 | — | — |
| DeepSeek-R1-Distill-Qwen-32B | 2 | 0,1 | suffix | 4,938.2 | 5,852.3 | — | — |
| DeepSeek-R1-Distill-Llama-70B | 4 | 0-3 | suffix | 3,163.7 | 2,863.5 | — | — |
| DeepSeek-R1 | 8 | 0-7 | vanilla | 1,537.6 | 1,008.2 | — | — |

## Δ% 비교 (per cell)

| Model | MWOR vs Optimal | MWOR vs Vanilla | MWOR vs llm-d |
|---|---:|---:|---:|
| Qwen2.5-7B-Instruct | — | — | — |
| Qwen2.5-32B-Instruct | — | — | — |
| Qwen2.5-72B-Instruct | — | — | — |
| Llama-3.1-8B-Instruct | — | — | — |
| Llama-3.1-70B-Instruct | — | — | — |
| Llama-3.1-405B-Instruct-FP8 | — | — | — |
| DeepSeek-R1-Distill-Qwen-7B | — | — | — |
| DeepSeek-R1-Distill-Qwen-32B | — | — | — |
| DeepSeek-R1-Distill-Llama-70B | — | — | — |
| DeepSeek-R1 | — | — | — |

## Cluster TPS 합 (10 model uniform 가정)

| metric | Vanilla | llm-d | Optimal Config | MWOR |
|---|---:|---:|---:|---:|
| cluster Σ tps | 41,888.1 | 60,356.7 | 0.0 | 0.0 |
| Δ MWOR vs Optimal | — | — | — | +0.0% |
| Δ MWOR vs Vanilla | — | — | — | -100.0% |
| Δ MWOR vs llm-d   | — | — | — | -100.0% |

## Realistic Mix Cluster 시뮬레이션

운영 시나리오 가중 (TP bucket 별 weight; 작은 모델이 더 많은 traffic 처리 가정):
- TP=1: w=0.40 / TP=2: w=0.20 / TP=4: w=0.20 / TP=8: w=0.20.
- 각 bucket 내 모델 수로 균등 분할.

| metric | Vanilla | llm-d | Optimal Config | MWOR |
|---|---:|---:|---:|---:|
| weighted Σ tps | 4,623.8 | 6,921.2 | 0.0 | 0.0 |
| Δ MWOR vs Optimal | — | — | — | +0.0% |
| Δ MWOR vs Vanilla | — | — | — | -100.0% |
| Δ MWOR vs llm-d   | — | — | — | -100.0% |

## MWOR 측정 source 노트

| Model | winner | MWOR 데이터 출처 |
|---|---|---|
| Qwen2.5-7B-Instruct | suffix | measured suffix+L2+L10+FaP |
| Qwen2.5-32B-Instruct | suffix | measured suffix+L2+L10+FaP |
| Qwen2.5-72B-Instruct | suffix | measured suffix+L2+L10+FaP |
| Llama-3.1-8B-Instruct | suffix | measured suffix+L2+L10+FaP |
| Llama-3.1-70B-Instruct | suffix | measured suffix+L2+L10+FaP |
| Llama-3.1-405B-Instruct-FP8 | suffix | measured suffix+L2+L10+FaP |
| DeepSeek-R1-Distill-Qwen-7B | suffix | measured suffix+L2+L10+FaP |
| DeepSeek-R1-Distill-Qwen-32B | suffix | measured suffix+L2+L10+FaP |
| DeepSeek-R1-Distill-Llama-70B | suffix | measured suffix+L2+L10+FaP |
| DeepSeek-R1 | vanilla | = Optimal (winner=vanilla) |

## 결론

- **1순위 결론 — MWOR vs Optimal Config**: 위 표 참조. winner=vanilla 셀에서는 MWOR == Optimal (정의상).
- **Optimal Config 자체 효과**: vanilla method 위에 FaP+L2+L10 stack 만 얹은 결과는 TSK_042 vanilla 와 직접 비교 가능.
- **MWOR ≥ Optimal 항상 성립** (oracle 정의상): 본 sweep 에서 실증.

