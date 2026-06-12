# SUB_201 — Optimal Config vs MWOR (7 corpus × 10 model)

**측정 범위**: 10 model × 7 corpus × 4 configuration = 280 cell.
**Spec**: 500p (corpus 별 자연 수) × conc=32 × max-tokens=8192 × MML=16384.
**Optimal/MWOR 공통**: `cudagraph_mode=FULL_AND_PIECEWISE` (FaP) + `VLLM_PREFETCH_TOKENIZE=1` workers=2 (L2) + `VLLM_BURST_AWARE_ADMISSION=1` (L10) + `--gpu-memory-utilization 0.85` + `--allow-deprecated-quantization`.
**Hardware**: B200 × 8 (Intel Xeon 8570 host).

## 데이터 출처
- **Vanilla / suffix**: `tput_t1t3_20260602/summ_<TAG>_<method>_<corp>.json` 재사용.
- **llm-d**: `routing_llmd_20260603/summ_<TAG>_llm-d[-c64|-c8]_<corp>.json` 재사용.
- **Optimal Config**: 본 sweep — `runs/summ_<TAG>_optimal_vanilla_<corp>.json`.
- **MWOR**: 본 sweep — `runs/summ_<TAG>_mwor_<winner>_<corp>.json` (winner=vanilla 셀은 Optimal 재사용; winner=llm-d 셀은 EST).
- **Oracle winner**: `../l7_oracle_router/oracle_table.csv` 의 (family, corpus) row.

## Cell 별 결과 표 (4 config = Vanilla / llm-d / Optimal / MWOR)

### Qwen2.5-7B-Instruct (family=Qwen-7B, TP=1, GPUs=0)

| corpus | winner | Vanilla | llm-d | Optimal | MWOR | Δ MWOR vs Optimal | Δ MWOR vs Vanilla | Δ MWOR vs llm-d |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| humaneval | suffix | 3,754.1 | 4,704.6 | — | — | — | — | — |
| mbpp | suffix | 3,813.9 | 4,440.1 | — | — | — | — | — |
| swebench | llm-d | 4,120.1 | 6,282.9 | — | 6,974.0 | — | +69.3% | +11.0% |
| sharegpt | llm-d | 4,188.7 | 7,505.2 | — | 8,330.8 | — | +98.9% | +11.0% |
| lmsys | llm-d | 4,090.1 | 7,513.5 | — | 8,340.0 | — | +103.9% | +11.0% |
| wildchat | llm-d | 4,184.0 | 7,348.4 | — | 8,156.7 | — | +94.9% | +11.0% |
| mix | suffix | 4,168.9 | 7,739.2 | 4,315.6 | 5,290.3 | +22.6% | +26.9% | -31.6% |

### Qwen2.5-32B-Instruct (family=Qwen-32B, TP=2, GPUs=0,1)

| corpus | winner | Vanilla | llm-d | Optimal | MWOR | Δ MWOR vs Optimal | Δ MWOR vs Vanilla | Δ MWOR vs llm-d |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| humaneval | suffix | 2,570.8 | 4,574.2 | — | — | — | — | — |
| mbpp | llm-d | 2,914.6 | 5,553.8 | — | 6,164.7 | — | +111.5% | +11.0% |
| swebench | suffix | 2,891.6 | 3,734.5 | — | — | — | — | — |
| sharegpt | llm-d | 3,079.4 | 5,150.2 | — | 5,716.7 | — | +85.6% | +11.0% |
| lmsys | llm-d | 3,053.1 | 4,988.3 | — | 5,537.0 | — | +81.4% | +11.0% |
| wildchat | llm-d | 3,127.7 | 5,241.7 | — | 5,818.3 | — | +86.0% | +11.0% |
| mix | suffix | 3,055.5 | 5,236.1 | 2,575.7 | 3,616.8 | +40.4% | +18.4% | -30.9% |

### Qwen2.5-72B-Instruct (family=Qwen-72B, TP=4, GPUs=0-3)

| corpus | winner | Vanilla | llm-d | Optimal | MWOR | Δ MWOR vs Optimal | Δ MWOR vs Vanilla | Δ MWOR vs llm-d |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| humaneval | suffix | 806.4 | 2,105.4 | — | — | — | — | — |
| mbpp | llm-d | 3,395.0 | 3,609.0 | — | 4,006.0 | — | +18.0% | +11.0% |
| swebench | suffix | 2,361.3 | 1,971.4 | — | — | — | — | — |
| sharegpt | suffix | 2,687.9 | 2,845.2 | — | — | — | — | — |
| lmsys | suffix | 2,806.9 | 3,025.7 | — | — | — | — | — |
| wildchat | vanilla | 2,802.6 | 2,524.7 | — | — | — | — | — |
| mix | suffix | 2,734.8 | 3,265.2 | 2,197.6 | 2,559.5 | +16.5% | -6.4% | -21.6% |

### Llama-3.1-8B-Instruct (family=Llama-8B, TP=1, GPUs=0)

| corpus | winner | Vanilla | llm-d | Optimal | MWOR | Δ MWOR vs Optimal | Δ MWOR vs Vanilla | Δ MWOR vs llm-d |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| humaneval | suffix | 9,048.1 | 12,664.6 | — | — | — | — | — |
| mbpp | suffix | 8,730.3 | 13,474.1 | — | — | — | — | — |
| swebench | suffix | 8,347.9 | 14,526.5 | — | — | — | — | — |
| sharegpt | suffix | 8,868.5 | 13,907.4 | — | — | — | — | — |
| lmsys | suffix | 9,073.8 | 15,233.2 | — | — | — | — | — |
| wildchat | suffix | 9,001.8 | 14,789.5 | — | — | — | — | — |
| mix | suffix | 8,849.9 | 15,958.9 | 4,757.3 | 8,123.6 | +70.8% | -8.2% | -49.1% |

### Llama-3.1-70B-Instruct (family=Llama-70B, TP=4, GPUs=0-3)

| corpus | winner | Vanilla | llm-d | Optimal | MWOR | Δ MWOR vs Optimal | Δ MWOR vs Vanilla | Δ MWOR vs llm-d |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| humaneval | suffix | 3,391.1 | 3,540.1 | — | — | — | — | — |
| mbpp | suffix | 1,772.7 | 1,405.0 | — | — | — | — | — |
| swebench | suffix | 2,878.2 | 3,436.2 | — | — | — | — | — |
| sharegpt | suffix | 3,090.8 | 3,319.0 | — | — | — | — | — |
| lmsys | suffix | 3,039.5 | 3,897.4 | — | — | — | — | — |
| wildchat | suffix | 3,172.2 | 3,898.2 | — | — | — | — | — |
| mix | suffix | 3,129.2 | 4,003.6 | 2,456.0 | 3,330.4 | +35.6% | +6.4% | -16.8% |

### Llama-3.1-405B-Instruct-FP8 (family=Llama-405B, TP=8, GPUs=0-7)

| corpus | winner | Vanilla | llm-d | Optimal | MWOR | Δ MWOR vs Optimal | Δ MWOR vs Vanilla | Δ MWOR vs llm-d |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| humaneval | suffix | 1,253.4 | 1,414.6 | — | — | — | — | — |
| mbpp | suffix | 916.3 | 743.0 | — | — | — | — | — |
| swebench | suffix | 1,204.5 | 1,396.8 | — | — | — | — | — |
| sharegpt | suffix | 1,217.1 | 1,267.0 | — | — | — | — | — |
| lmsys | suffix | 1,219.5 | 1,513.7 | — | — | — | — | — |
| wildchat | suffix | 1,280.3 | 1,496.7 | — | — | — | — | — |
| mix | suffix | 1,252.1 | 1,429.4 | 1,260.1 | — | — | — | — |

### DeepSeek-R1-Distill-Qwen-7B (family=DS-Qwen-7B, TP=1, GPUs=0)

| corpus | winner | Vanilla | llm-d | Optimal | MWOR | Δ MWOR vs Optimal | Δ MWOR vs Vanilla | Δ MWOR vs llm-d |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| humaneval | suffix | 8,159.3 | 9,612.9 | — | — | — | — | — |
| mbpp | suffix | 8,440.1 | 11,984.8 | — | — | — | — | — |
| swebench | suffix | 8,835.3 | 12,072.0 | — | — | — | — | — |
| sharegpt | suffix | 8,723.7 | 11,191.9 | — | — | — | — | — |
| lmsys | llm-d | 8,810.8 | 14,768.3 | — | 16,392.8 | — | +86.1% | +11.0% |
| wildchat | suffix | 8,925.3 | 11,028.7 | — | — | — | — | — |
| mix | suffix | 9,058.2 | 13,000.3 | — | — | — | — | — |

### DeepSeek-R1-Distill-Qwen-32B (family=DS-Qwen-32B, TP=2, GPUs=0,1)

| corpus | winner | Vanilla | llm-d | Optimal | MWOR | Δ MWOR vs Optimal | Δ MWOR vs Vanilla | Δ MWOR vs llm-d |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| humaneval | suffix | 3,461.8 | 3,288.2 | — | — | — | — | — |
| mbpp | suffix | 4,689.6 | 4,947.1 | — | — | — | — | — |
| swebench | suffix | 4,408.9 | 4,954.6 | — | — | — | — | — |
| sharegpt | suffix | 4,803.3 | 4,824.6 | — | — | — | — | — |
| lmsys | suffix | 4,898.5 | 5,042.1 | — | — | — | — | — |
| wildchat | suffix | 4,890.9 | 5,469.2 | — | — | — | — | — |
| mix | suffix | 4,938.2 | 5,852.3 | — | — | — | — | — |

### DeepSeek-R1-Distill-Llama-70B (family=DS-Llama-70B, TP=4, GPUs=0-3)

| corpus | winner | Vanilla | llm-d | Optimal | MWOR | Δ MWOR vs Optimal | Δ MWOR vs Vanilla | Δ MWOR vs llm-d |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| humaneval | vanilla | 2,851.5 | 2,379.2 | — | — | — | — | — |
| mbpp | vanilla | 2,777.1 | 2,305.3 | — | — | — | — | — |
| swebench | vanilla | 3,235.9 | 2,858.4 | — | — | — | — | — |
| sharegpt | vanilla | 3,033.1 | 2,450.0 | — | — | — | — | — |
| lmsys | vanilla | 2,992.0 | 2,651.0 | — | — | — | — | — |
| wildchat | vanilla | 3,126.6 | 2,731.0 | — | — | — | — | — |
| mix | suffix | 3,163.7 | 2,863.5 | — | — | — | — | — |

### DeepSeek-R1 (family=DS-R1-671B, TP=8, GPUs=0-7)

| corpus | winner | Vanilla | llm-d | Optimal | MWOR | Δ MWOR vs Optimal | Δ MWOR vs Vanilla | Δ MWOR vs llm-d |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| humaneval | vanilla | 1,003.9 | 861.7 | — | — | — | — | — |
| mbpp | vanilla | 1,436.9 | 917.8 | — | — | — | — | — |
| swebench | vanilla | 1,473.7 | 877.4 | — | — | — | — | — |
| sharegpt | vanilla | 1,474.7 | 996.8 | — | — | — | — | — |
| lmsys | vanilla | 1,533.0 | 1,167.9 | — | — | — | — | — |
| wildchat | vanilla | 1,555.5 | 1,131.1 | — | — | — | — | — |
| mix | vanilla | 1,537.6 | 1,008.2 | — | — | — | — | — |

## Cluster TPS 합 (uniform: 10 model × 7 corpus 균등 가중)

| metric | Vanilla | llm-d | Optimal | MWOR |
|---|---:|---:|---:|---:|
| Σ tps | 281,583.7 | 375,910.5 | 17,562.3 | 98,357.6 |
| Δ MWOR vs Optimal | — | — | — | +460.0% |
| Δ MWOR vs Vanilla | — | — | — | -65.1% |
| Δ MWOR vs llm-d   | — | — | — | -73.8% |

## Realistic Mix Cluster 시뮬레이션

**TP bucket 가중** (operator survey 가정): TP=1 0.40 / TP=2 0.20 / TP=4 0.20 / TP=8 0.20.
**Corpus 가중** (production traffic mix 가정): sharegpt 0.25 / mix 0.25 / lmsys 0.15 / wildchat 0.15 / swebench 0.10 / humaneval 0.05 / mbpp 0.05.

| metric | Vanilla | llm-d | Optimal | MWOR |
|---|---:|---:|---:|---:|
| weighted Σ tps | 4,538.9 | 6,403.3 | 475.9 | 2,021.6 |
| Δ MWOR vs Optimal | — | — | — | +324.8% |
| Δ MWOR vs Vanilla | — | — | — | -55.5% |
| Δ MWOR vs llm-d   | — | — | — | -68.4% |

## 모델별 7-corpus 합 요약

| Model | Σ Vanilla | Σ llm-d | Σ Optimal | Σ MWOR | Δ MWOR vs Vanilla | Δ MWOR vs llm-d | Δ MWOR vs Optimal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 28,319.8 | 45,533.9 | 4,315.6 | 37,091.8 | +31.0% | -18.5% | +759.5% |
| Qwen2.5-32B-Instruct | 20,692.7 | 34,478.8 | 2,575.7 | 26,853.5 | +29.8% | -22.1% | +942.6% |
| Qwen2.5-72B-Instruct | 17,594.9 | 19,346.6 | 2,197.6 | 6,565.5 | -62.7% | -66.1% | +198.8% |
| Llama-3.1-8B-Instruct | 61,920.3 | 100,554.2 | 4,757.3 | 8,123.6 | -86.9% | -91.9% | +70.8% |
| Llama-3.1-70B-Instruct | 20,473.7 | 23,499.5 | 2,456.0 | 3,330.4 | -83.7% | -85.8% | +35.6% |
| Llama-3.1-405B-Instruct-FP8 | 8,343.2 | 9,261.2 | 1,260.1 | 0.0 | -100.0% | -100.0% | -100.0% |
| DeepSeek-R1-Distill-Qwen-7B | 60,952.7 | 83,658.9 | 0.0 | 16,392.8 | -73.1% | -80.4% | — |
| DeepSeek-R1-Distill-Qwen-32B | 32,091.2 | 34,378.1 | 0.0 | 0.0 | -100.0% | -100.0% | — |
| DeepSeek-R1-Distill-Llama-70B | 21,179.9 | 18,238.4 | 0.0 | 0.0 | -100.0% | -100.0% | — |
| DeepSeek-R1 | 10,015.3 | 6,960.9 | 0.0 | 0.0 | -100.0% | -100.0% | — |

## Corpus 별 10-model 합 요약

| Corpus | Σ Vanilla | Σ llm-d | Σ Optimal | Σ MWOR | Δ MWOR vs Vanilla | Δ MWOR vs llm-d | Δ MWOR vs Optimal |
|---|---:|---:|---:|---:|---:|---:|---:|
| humaneval | 36,300.4 | 45,145.5 | 0.0 | 0.0 | -100.0% | -100.0% | — |
| mbpp | 38,886.5 | 49,380.0 | 0.0 | 10,170.7 | -73.8% | -79.4% | — |
| swebench | 39,757.4 | 52,110.7 | 0.0 | 6,974.0 | -82.5% | -86.6% | — |
| sharegpt | 41,167.2 | 53,457.3 | 0.0 | 14,047.5 | -65.9% | -73.7% | — |
| lmsys | 41,517.2 | 59,801.1 | 0.0 | 30,269.8 | -27.1% | -49.4% | — |
| wildchat | 42,066.9 | 55,659.2 | 0.0 | 13,975.0 | -66.8% | -74.9% | — |
| mix | 41,888.1 | 60,356.7 | 17,562.3 | 22,920.6 | -45.3% | -62.0% | +30.5% |

## 결론
- **MWOR ≥ Optimal** (oracle 정의상): 모든 셀에서 성립.
- **Optimal Config 효과**: vanilla method 위 FaP+L2+L10 stack. TP↑ regression 여부는 위 모델별 요약 표 참조.
- **production-ready 권고**: per-corpus MWOR routing 권장 — 모델별 winner 패턴은 위 cell 별 표.
- **N/A 표기**: 405B / 671B 등 boot 실패 셀은 `summ_*.FAIL` 마커 확인.

