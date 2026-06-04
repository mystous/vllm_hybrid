# TSK_042 (B) 측정 결과 — 실 trace, conc=32, max_tokens=8192, stream, TP=8(7B Qwen만 4)

모델 10 × method 3 × 조건 7. 셀 91개.

## 처리량 (output_tps) — 조건별 model×method

### sharegpt
| model | llm-d | llm-d-c64 | llm-d-c8 | best | suffix vs van |
|---|---|---|---|---|---|
| DeepSeek-R1 | 997 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Llama-70B | 2,450 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Qwen-32B | 4,825 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Qwen-7B | 11,192 | — | — | llm-d | — |
| Llama-3.1-405B-Instruct-FP8 | 1,267 | — | — | llm-d | — |
| Llama-3.1-70B-Instruct | 3,319 | — | — | llm-d | — |
| Llama-3.1-8B-Instruct | 13,907 | — | — | llm-d | — |
| Qwen2.5-32B-Instruct | 5,150 | — | 1,773 | llm-d | — |
| Qwen2.5-72B-Instruct | 2,845 | — | — | llm-d | — |
| Qwen2.5-7B-Instruct | 7,505 | 9,729 | 3,040 | llm-d-c64 | — |

### swebench
| model | llm-d | llm-d-c64 | llm-d-c8 | best | suffix vs van |
|---|---|---|---|---|---|
| DeepSeek-R1 | 877 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Llama-70B | 2,858 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Qwen-32B | 4,955 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Qwen-7B | 12,072 | — | — | llm-d | — |
| Llama-3.1-405B-Instruct-FP8 | 1,397 | — | — | llm-d | — |
| Llama-3.1-70B-Instruct | 3,436 | — | — | llm-d | — |
| Llama-3.1-8B-Instruct | 14,526 | — | — | llm-d | — |
| Qwen2.5-32B-Instruct | 3,734 | — | 1,707 | llm-d | — |
| Qwen2.5-72B-Instruct | 1,971 | — | — | llm-d | — |
| Qwen2.5-7B-Instruct | 6,283 | 7,533 | 2,830 | llm-d-c64 | — |

### humaneval
| model | llm-d | llm-d-c64 | llm-d-c8 | best | suffix vs van |
|---|---|---|---|---|---|
| DeepSeek-R1 | 862 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Llama-70B | 2,379 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Qwen-32B | 3,288 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Qwen-7B | 9,613 | — | — | llm-d | — |
| Llama-3.1-405B-Instruct-FP8 | 1,415 | — | — | llm-d | — |
| Llama-3.1-70B-Instruct | 3,540 | — | — | llm-d | — |
| Llama-3.1-8B-Instruct | 12,665 | — | — | llm-d | — |
| Qwen2.5-32B-Instruct | 4,574 | — | 1,853 | llm-d | — |
| Qwen2.5-72B-Instruct | 2,105 | — | — | llm-d | — |
| Qwen2.5-7B-Instruct | 4,705 | 7,047 | 2,737 | llm-d-c64 | — |

### mbpp
| model | llm-d | llm-d-c64 | llm-d-c8 | best | suffix vs van |
|---|---|---|---|---|---|
| DeepSeek-R1 | 918 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Llama-70B | 2,305 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Qwen-32B | 4,947 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Qwen-7B | 11,985 | — | — | llm-d | — |
| Llama-3.1-405B-Instruct-FP8 | 743 | — | — | llm-d | — |
| Llama-3.1-70B-Instruct | 1,405 | — | — | llm-d | — |
| Llama-3.1-8B-Instruct | 13,474 | — | — | llm-d | — |
| Qwen2.5-32B-Instruct | 5,554 | — | 1,920 | llm-d | — |
| Qwen2.5-72B-Instruct | 3,609 | — | — | llm-d | — |
| Qwen2.5-7B-Instruct | 4,440 | 5,859 | 2,529 | llm-d-c64 | — |

### wildchat
| model | llm-d | llm-d-c64 | llm-d-c8 | best | suffix vs van |
|---|---|---|---|---|---|
| DeepSeek-R1 | 1,131 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Llama-70B | 2,731 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Qwen-32B | 5,469 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Qwen-7B | 11,029 | — | — | llm-d | — |
| Llama-3.1-405B-Instruct-FP8 | 1,497 | — | — | llm-d | — |
| Llama-3.1-70B-Instruct | 3,898 | — | — | llm-d | — |
| Llama-3.1-8B-Instruct | 14,790 | — | — | llm-d | — |
| Qwen2.5-32B-Instruct | 5,242 | — | 1,814 | llm-d | — |
| Qwen2.5-72B-Instruct | 2,525 | — | — | llm-d | — |
| Qwen2.5-7B-Instruct | 7,348 | 9,476 | 3,169 | llm-d-c64 | — |

### lmsys
| model | llm-d | llm-d-c64 | llm-d-c8 | best | suffix vs van |
|---|---|---|---|---|---|
| DeepSeek-R1 | 1,168 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Llama-70B | 2,651 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Qwen-32B | 5,042 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Qwen-7B | 14,768 | — | — | llm-d | — |
| Llama-3.1-405B-Instruct-FP8 | 1,514 | — | — | llm-d | — |
| Llama-3.1-70B-Instruct | 3,897 | — | — | llm-d | — |
| Llama-3.1-8B-Instruct | 15,233 | — | — | llm-d | — |
| Qwen2.5-32B-Instruct | 4,988 | — | 1,876 | llm-d | — |
| Qwen2.5-72B-Instruct | 3,026 | — | — | llm-d | — |
| Qwen2.5-7B-Instruct | 7,514 | 9,068 | 3,147 | llm-d-c64 | — |

### mix
| model | llm-d | llm-d-c64 | llm-d-c8 | best | suffix vs van |
|---|---|---|---|---|---|
| DeepSeek-R1 | 1,008 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Llama-70B | 2,864 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Qwen-32B | 5,852 | — | — | llm-d | — |
| DeepSeek-R1-Distill-Qwen-7B | 13,000 | — | — | llm-d | — |
| Llama-3.1-405B-Instruct-FP8 | 1,429 | — | — | llm-d | — |
| Llama-3.1-70B-Instruct | 4,004 | — | — | llm-d | — |
| Llama-3.1-8B-Instruct | 15,959 | — | — | llm-d | — |
| Qwen2.5-32B-Instruct | 5,236 | — | 1,472 | llm-d | — |
| Qwen2.5-72B-Instruct | 3,265 | — | — | llm-d | — |
| Qwen2.5-7B-Instruct | 7,739 | 10,346 | 3,267 | llm-d-c64 | — |

## condition-level oracle (model × condition → best method) — regret 입력
| model | sharegpt | swebench | humaneval | mbpp | wildchat | lmsys | mix |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1 | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d |
| DeepSeek-R1-Distill-Llama-70B | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d |
| Llama-3.1-405B-Instruct-FP8 | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d |
| Llama-3.1-70B-Instruct | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d |
| Llama-3.1-8B-Instruct | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d |
| Qwen2.5-32B-Instruct | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d |
| Qwen2.5-72B-Instruct | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d | llm-d |
| Qwen2.5-7B-Instruct | llm-d-c64 | llm-d-c64 | llm-d-c64 | llm-d-c64 | llm-d-c64 | llm-d-c64 | llm-d-c64 |

## 지연 TTFT (mix, p50/p99 ms)
| model | llm-d | llm-d-c64 | llm-d-c8 |
|---|---|---|---|
| DeepSeek-R1 | 93.2/255.5 | — | — |
| DeepSeek-R1-Distill-Llama-70B | 40.3/124.6 | — | — |
| DeepSeek-R1-Distill-Qwen-32B | 29.8/78.9 | — | — |
| DeepSeek-R1-Distill-Qwen-7B | 21.6/56.2 | — | — |
| Llama-3.1-405B-Instruct-FP8 | 97.0/458.7 | — | — |
| Llama-3.1-70B-Instruct | 43.8/148.4 | — | — |
| Llama-3.1-8B-Instruct | 22.4/76.3 | — | — |
| Qwen2.5-32B-Instruct | 32.2/93.4 | — | 26.2/49.0 |
| Qwen2.5-72B-Instruct | 41.8/140.7 | — | — |
| Qwen2.5-7B-Instruct | 25.5/52.4 | 52.8/120.7 | 19.9/29.1 |

## 지연 TPOT (mix, p50/p99 ms)
| model | llm-d | llm-d-c64 | llm-d-c8 |
|---|---|---|---|
| DeepSeek-R1 | 21.3/80.2 | — | — |
| DeepSeek-R1-Distill-Llama-70B | 10.2/15.7 | — | — |
| DeepSeek-R1-Distill-Qwen-32B | 6.5/11.0 | — | — |
| DeepSeek-R1-Distill-Qwen-7B | 3.2/5.9 | — | — |
| Llama-3.1-405B-Instruct-FP8 | 27.5/47.2 | — | — |
| Llama-3.1-70B-Instruct | 9.5/16.7 | — | — |
| Llama-3.1-8B-Instruct | 1.1/4.0 | — | — |
| Qwen2.5-32B-Instruct | 6.1/11.7 | — | 6.0/10.6 |
| Qwen2.5-72B-Instruct | 9.9/16.6 | — | — |
| Qwen2.5-7B-Instruct | 4.6/11.6 | 4.1/14.6 | 3.5/5.7 |

## accept α (mix, accepted/draft) — spec method
| model | llm-d | llm-d-c64 | llm-d-c8 |
|---|---|---|---|
| DeepSeek-R1 | 0.464 | — | — |
| DeepSeek-R1-Distill-Llama-70B | 0.393 | — | — |
| DeepSeek-R1-Distill-Qwen-32B | 0.670 | — | — |
| DeepSeek-R1-Distill-Qwen-7B | 0.711 | — | — |
| Llama-3.1-405B-Instruct-FP8 | 0.746 | — | — |
| Llama-3.1-70B-Instruct | 0.825 | — | — |
| Llama-3.1-8B-Instruct | 0.885 | — | — |
| Qwen2.5-32B-Instruct | 0.769 | — | — |
| Qwen2.5-72B-Instruct | 0.677 | — | — |
| Qwen2.5-7B-Instruct | 0.750 | 0.871 | 0.756 |

## util (mix, gpu%/cpu%, gpu_mem GiB)
| model | llm-d | llm-d-c64 | llm-d-c8 |
|---|---|---|---|
| DeepSeek-R1 | 92.2/5.1 (1386G) | — | — |
| DeepSeek-R1-Distill-Llama-70B | 90.5/5.6 (1202G) | — | — |
| DeepSeek-R1-Distill-Qwen-32B | 87.3/5.6 (1202G) | — | — |
| DeepSeek-R1-Distill-Qwen-7B | 81.0/3.9 (600G) | — | — |
| Llama-3.1-405B-Instruct-FP8 | 86.2/4.9 (1205G) | — | — |
| Llama-3.1-70B-Instruct | 83.7/5.3 (1202G) | — | — |
| Llama-3.1-8B-Instruct | 79.5/5.2 (1200G) | — | — |
| Qwen2.5-32B-Instruct | 79.7/4.9 (1202G) | — | 84.8/4.4 (1197G) |
| Qwen2.5-72B-Instruct | 88.3/5.1 (1203G) | — | — |
| Qwen2.5-7B-Instruct | 61.0/3.2 (600G) | 55.0/3.4 (600G) | 67.9/3.1 (600G) |

## raw 데이터
- per-request 전체 로그: [`per_request_raw.jsonl`](per_request_raw.jsonl) (corpus·ok·wall_ms·ttft_ms·tpot_ms·completion/prompt_tokens·model·method·condition)
- long-format 메트릭: [`metrics_table.parquet`](metrics_table.parquet) / [`metrics_table.csv`](metrics_table.csv)
- 백엔드/측정 로그: `_logs/`

## 셀 상세 문서 (셀별 메트릭 + raw 분포)
| model | method | condition | tps | 상세 | summ |
|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | llm-d | humaneval | 2,379 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__llm-d__humaneval.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_llm-d_humaneval.json) |
| DeepSeek-R1-Distill-Llama-70B | llm-d | lmsys | 2,651 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__llm-d__lmsys.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_llm-d_lmsys.json) |
| DeepSeek-R1-Distill-Llama-70B | llm-d | mbpp | 2,305 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__llm-d__mbpp.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_llm-d_mbpp.json) |
| DeepSeek-R1-Distill-Llama-70B | llm-d | mix | 2,864 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__llm-d__mix.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_llm-d_mix.json) |
| DeepSeek-R1-Distill-Llama-70B | llm-d | sharegpt | 2,450 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__llm-d__sharegpt.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_llm-d_sharegpt.json) |
| DeepSeek-R1-Distill-Llama-70B | llm-d | swebench | 2,858 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__llm-d__swebench.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_llm-d_swebench.json) |
| DeepSeek-R1-Distill-Llama-70B | llm-d | wildchat | 2,731 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__llm-d__wildchat.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_llm-d_wildchat.json) |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | humaneval | 3,288 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__llm-d__humaneval.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_humaneval.json) |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | lmsys | 5,042 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__llm-d__lmsys.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_lmsys.json) |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | mbpp | 4,947 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__llm-d__mbpp.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_mbpp.json) |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | mix | 5,852 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__llm-d__mix.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_mix.json) |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | sharegpt | 4,825 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__llm-d__sharegpt.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_sharegpt.json) |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | swebench | 4,955 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__llm-d__swebench.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_swebench.json) |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | wildchat | 5,469 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__llm-d__wildchat.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_wildchat.json) |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | humaneval | 9,613 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__llm-d__humaneval.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_humaneval.json) |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | lmsys | 14,768 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__llm-d__lmsys.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_lmsys.json) |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | mbpp | 11,985 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__llm-d__mbpp.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_mbpp.json) |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | mix | 13,000 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__llm-d__mix.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_mix.json) |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | sharegpt | 11,192 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__llm-d__sharegpt.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_sharegpt.json) |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | swebench | 12,072 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__llm-d__swebench.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_swebench.json) |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | wildchat | 11,029 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__llm-d__wildchat.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_wildchat.json) |
| DeepSeek-R1 | llm-d | humaneval | 862 | [상세](cells/cell_DeepSeek-R1__llm-d__humaneval.md) | [json](summ_DeepSeek-R1_llm-d_humaneval.json) |
| DeepSeek-R1 | llm-d | lmsys | 1,168 | [상세](cells/cell_DeepSeek-R1__llm-d__lmsys.md) | [json](summ_DeepSeek-R1_llm-d_lmsys.json) |
| DeepSeek-R1 | llm-d | mbpp | 918 | [상세](cells/cell_DeepSeek-R1__llm-d__mbpp.md) | [json](summ_DeepSeek-R1_llm-d_mbpp.json) |
| DeepSeek-R1 | llm-d | mix | 1,008 | [상세](cells/cell_DeepSeek-R1__llm-d__mix.md) | [json](summ_DeepSeek-R1_llm-d_mix.json) |
| DeepSeek-R1 | llm-d | sharegpt | 997 | [상세](cells/cell_DeepSeek-R1__llm-d__sharegpt.md) | [json](summ_DeepSeek-R1_llm-d_sharegpt.json) |
| DeepSeek-R1 | llm-d | swebench | 877 | [상세](cells/cell_DeepSeek-R1__llm-d__swebench.md) | [json](summ_DeepSeek-R1_llm-d_swebench.json) |
| DeepSeek-R1 | llm-d | wildchat | 1,131 | [상세](cells/cell_DeepSeek-R1__llm-d__wildchat.md) | [json](summ_DeepSeek-R1_llm-d_wildchat.json) |
| Llama-3.1-405B-Instruct-FP8 | llm-d | humaneval | 1,415 | [상세](cells/cell_Llama-3.1-405B-Instruct-FP8__llm-d__humaneval.md) | [json](summ_Llama-3.1-405B-Instruct-FP8_llm-d_humaneval.json) |
| Llama-3.1-405B-Instruct-FP8 | llm-d | lmsys | 1,514 | [상세](cells/cell_Llama-3.1-405B-Instruct-FP8__llm-d__lmsys.md) | [json](summ_Llama-3.1-405B-Instruct-FP8_llm-d_lmsys.json) |
| Llama-3.1-405B-Instruct-FP8 | llm-d | mbpp | 743 | [상세](cells/cell_Llama-3.1-405B-Instruct-FP8__llm-d__mbpp.md) | [json](summ_Llama-3.1-405B-Instruct-FP8_llm-d_mbpp.json) |
| Llama-3.1-405B-Instruct-FP8 | llm-d | mix | 1,429 | [상세](cells/cell_Llama-3.1-405B-Instruct-FP8__llm-d__mix.md) | [json](summ_Llama-3.1-405B-Instruct-FP8_llm-d_mix.json) |
| Llama-3.1-405B-Instruct-FP8 | llm-d | sharegpt | 1,267 | [상세](cells/cell_Llama-3.1-405B-Instruct-FP8__llm-d__sharegpt.md) | [json](summ_Llama-3.1-405B-Instruct-FP8_llm-d_sharegpt.json) |
| Llama-3.1-405B-Instruct-FP8 | llm-d | swebench | 1,397 | [상세](cells/cell_Llama-3.1-405B-Instruct-FP8__llm-d__swebench.md) | [json](summ_Llama-3.1-405B-Instruct-FP8_llm-d_swebench.json) |
| Llama-3.1-405B-Instruct-FP8 | llm-d | wildchat | 1,497 | [상세](cells/cell_Llama-3.1-405B-Instruct-FP8__llm-d__wildchat.md) | [json](summ_Llama-3.1-405B-Instruct-FP8_llm-d_wildchat.json) |
| Llama-3.1-70B-Instruct | llm-d | humaneval | 3,540 | [상세](cells/cell_Llama-3.1-70B-Instruct__llm-d__humaneval.md) | [json](summ_Llama-3.1-70B-Instruct_llm-d_humaneval.json) |
| Llama-3.1-70B-Instruct | llm-d | lmsys | 3,897 | [상세](cells/cell_Llama-3.1-70B-Instruct__llm-d__lmsys.md) | [json](summ_Llama-3.1-70B-Instruct_llm-d_lmsys.json) |
| Llama-3.1-70B-Instruct | llm-d | mbpp | 1,405 | [상세](cells/cell_Llama-3.1-70B-Instruct__llm-d__mbpp.md) | [json](summ_Llama-3.1-70B-Instruct_llm-d_mbpp.json) |
| Llama-3.1-70B-Instruct | llm-d | mix | 4,004 | [상세](cells/cell_Llama-3.1-70B-Instruct__llm-d__mix.md) | [json](summ_Llama-3.1-70B-Instruct_llm-d_mix.json) |
| Llama-3.1-70B-Instruct | llm-d | sharegpt | 3,319 | [상세](cells/cell_Llama-3.1-70B-Instruct__llm-d__sharegpt.md) | [json](summ_Llama-3.1-70B-Instruct_llm-d_sharegpt.json) |
| Llama-3.1-70B-Instruct | llm-d | swebench | 3,436 | [상세](cells/cell_Llama-3.1-70B-Instruct__llm-d__swebench.md) | [json](summ_Llama-3.1-70B-Instruct_llm-d_swebench.json) |
| Llama-3.1-70B-Instruct | llm-d | wildchat | 3,898 | [상세](cells/cell_Llama-3.1-70B-Instruct__llm-d__wildchat.md) | [json](summ_Llama-3.1-70B-Instruct_llm-d_wildchat.json) |
| Llama-3.1-8B-Instruct | llm-d | humaneval | 12,665 | [상세](cells/cell_Llama-3.1-8B-Instruct__llm-d__humaneval.md) | [json](summ_Llama-3.1-8B-Instruct_llm-d_humaneval.json) |
| Llama-3.1-8B-Instruct | llm-d | lmsys | 15,233 | [상세](cells/cell_Llama-3.1-8B-Instruct__llm-d__lmsys.md) | [json](summ_Llama-3.1-8B-Instruct_llm-d_lmsys.json) |
| Llama-3.1-8B-Instruct | llm-d | mbpp | 13,474 | [상세](cells/cell_Llama-3.1-8B-Instruct__llm-d__mbpp.md) | [json](summ_Llama-3.1-8B-Instruct_llm-d_mbpp.json) |
| Llama-3.1-8B-Instruct | llm-d | mix | 15,959 | [상세](cells/cell_Llama-3.1-8B-Instruct__llm-d__mix.md) | [json](summ_Llama-3.1-8B-Instruct_llm-d_mix.json) |
| Llama-3.1-8B-Instruct | llm-d | sharegpt | 13,907 | [상세](cells/cell_Llama-3.1-8B-Instruct__llm-d__sharegpt.md) | [json](summ_Llama-3.1-8B-Instruct_llm-d_sharegpt.json) |
| Llama-3.1-8B-Instruct | llm-d | swebench | 14,526 | [상세](cells/cell_Llama-3.1-8B-Instruct__llm-d__swebench.md) | [json](summ_Llama-3.1-8B-Instruct_llm-d_swebench.json) |
| Llama-3.1-8B-Instruct | llm-d | wildchat | 14,790 | [상세](cells/cell_Llama-3.1-8B-Instruct__llm-d__wildchat.md) | [json](summ_Llama-3.1-8B-Instruct_llm-d_wildchat.json) |
| Qwen2.5-32B-Instruct | llm-d-c8 | humaneval | 1,853 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d-c8__humaneval.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d-c8_humaneval.json) |
| Qwen2.5-32B-Instruct | llm-d-c8 | lmsys | 1,876 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d-c8__lmsys.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d-c8_lmsys.json) |
| Qwen2.5-32B-Instruct | llm-d-c8 | mbpp | 1,920 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d-c8__mbpp.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d-c8_mbpp.json) |
| Qwen2.5-32B-Instruct | llm-d-c8 | mix | 1,472 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d-c8__mix.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d-c8_mix.json) |
| Qwen2.5-32B-Instruct | llm-d-c8 | sharegpt | 1,773 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d-c8__sharegpt.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d-c8_sharegpt.json) |
| Qwen2.5-32B-Instruct | llm-d-c8 | swebench | 1,707 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d-c8__swebench.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d-c8_swebench.json) |
| Qwen2.5-32B-Instruct | llm-d-c8 | wildchat | 1,814 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d-c8__wildchat.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d-c8_wildchat.json) |
| Qwen2.5-32B-Instruct | llm-d | humaneval | 4,574 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d__humaneval.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d_humaneval.json) |
| Qwen2.5-32B-Instruct | llm-d | lmsys | 4,988 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d__lmsys.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d_lmsys.json) |
| Qwen2.5-32B-Instruct | llm-d | mbpp | 5,554 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d__mbpp.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d_mbpp.json) |
| Qwen2.5-32B-Instruct | llm-d | mix | 5,236 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d__mix.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d_mix.json) |
| Qwen2.5-32B-Instruct | llm-d | sharegpt | 5,150 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d__sharegpt.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d_sharegpt.json) |
| Qwen2.5-32B-Instruct | llm-d | swebench | 3,734 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d__swebench.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d_swebench.json) |
| Qwen2.5-32B-Instruct | llm-d | wildchat | 5,242 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d__wildchat.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d_wildchat.json) |
| Qwen2.5-72B-Instruct | llm-d | humaneval | 2,105 | [상세](cells/cell_Qwen2.5-72B-Instruct__llm-d__humaneval.md) | [json](summ_Qwen2.5-72B-Instruct_llm-d_humaneval.json) |
| Qwen2.5-72B-Instruct | llm-d | lmsys | 3,026 | [상세](cells/cell_Qwen2.5-72B-Instruct__llm-d__lmsys.md) | [json](summ_Qwen2.5-72B-Instruct_llm-d_lmsys.json) |
| Qwen2.5-72B-Instruct | llm-d | mbpp | 3,609 | [상세](cells/cell_Qwen2.5-72B-Instruct__llm-d__mbpp.md) | [json](summ_Qwen2.5-72B-Instruct_llm-d_mbpp.json) |
| Qwen2.5-72B-Instruct | llm-d | mix | 3,265 | [상세](cells/cell_Qwen2.5-72B-Instruct__llm-d__mix.md) | [json](summ_Qwen2.5-72B-Instruct_llm-d_mix.json) |
| Qwen2.5-72B-Instruct | llm-d | sharegpt | 2,845 | [상세](cells/cell_Qwen2.5-72B-Instruct__llm-d__sharegpt.md) | [json](summ_Qwen2.5-72B-Instruct_llm-d_sharegpt.json) |
| Qwen2.5-72B-Instruct | llm-d | swebench | 1,971 | [상세](cells/cell_Qwen2.5-72B-Instruct__llm-d__swebench.md) | [json](summ_Qwen2.5-72B-Instruct_llm-d_swebench.json) |
| Qwen2.5-72B-Instruct | llm-d | wildchat | 2,525 | [상세](cells/cell_Qwen2.5-72B-Instruct__llm-d__wildchat.md) | [json](summ_Qwen2.5-72B-Instruct_llm-d_wildchat.json) |
| Qwen2.5-7B-Instruct | llm-d-c64 | humaneval | 7,047 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d-c64__humaneval.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d-c64_humaneval.json) |
| Qwen2.5-7B-Instruct | llm-d-c64 | lmsys | 9,068 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d-c64__lmsys.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d-c64_lmsys.json) |
| Qwen2.5-7B-Instruct | llm-d-c64 | mbpp | 5,859 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d-c64__mbpp.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d-c64_mbpp.json) |
| Qwen2.5-7B-Instruct | llm-d-c64 | mix | 10,346 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d-c64__mix.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d-c64_mix.json) |
| Qwen2.5-7B-Instruct | llm-d-c64 | sharegpt | 9,729 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d-c64__sharegpt.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d-c64_sharegpt.json) |
| Qwen2.5-7B-Instruct | llm-d-c64 | swebench | 7,533 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d-c64__swebench.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d-c64_swebench.json) |
| Qwen2.5-7B-Instruct | llm-d-c64 | wildchat | 9,476 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d-c64__wildchat.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d-c64_wildchat.json) |
| Qwen2.5-7B-Instruct | llm-d-c8 | humaneval | 2,737 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d-c8__humaneval.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d-c8_humaneval.json) |
| Qwen2.5-7B-Instruct | llm-d-c8 | lmsys | 3,147 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d-c8__lmsys.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d-c8_lmsys.json) |
| Qwen2.5-7B-Instruct | llm-d-c8 | mbpp | 2,529 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d-c8__mbpp.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d-c8_mbpp.json) |
| Qwen2.5-7B-Instruct | llm-d-c8 | mix | 3,267 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d-c8__mix.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d-c8_mix.json) |
| Qwen2.5-7B-Instruct | llm-d-c8 | sharegpt | 3,040 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d-c8__sharegpt.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d-c8_sharegpt.json) |
| Qwen2.5-7B-Instruct | llm-d-c8 | swebench | 2,830 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d-c8__swebench.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d-c8_swebench.json) |
| Qwen2.5-7B-Instruct | llm-d-c8 | wildchat | 3,169 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d-c8__wildchat.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d-c8_wildchat.json) |
| Qwen2.5-7B-Instruct | llm-d | humaneval | 4,705 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d__humaneval.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d_humaneval.json) |
| Qwen2.5-7B-Instruct | llm-d | lmsys | 7,514 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d__lmsys.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d_lmsys.json) |
| Qwen2.5-7B-Instruct | llm-d | mbpp | 4,440 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d__mbpp.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d_mbpp.json) |
| Qwen2.5-7B-Instruct | llm-d | mix | 7,739 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d__mix.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d_mix.json) |
| Qwen2.5-7B-Instruct | llm-d | sharegpt | 7,505 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d__sharegpt.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d_sharegpt.json) |
| Qwen2.5-7B-Instruct | llm-d | swebench | 6,283 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d__swebench.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d_swebench.json) |
| Qwen2.5-7B-Instruct | llm-d | wildchat | 7,348 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d__wildchat.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d_wildchat.json) |

