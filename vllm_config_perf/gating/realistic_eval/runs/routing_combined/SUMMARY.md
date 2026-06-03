# TSK_042 (B) 측정 결과 — 실 trace, conc=32, max_tokens=8192, stream, TP=8(7B Qwen만 4)

모델 8 × method 4 × 조건 7. 셀 173개.

## 처리량 (output_tps) — 조건별 model×method

### sharegpt
| model | llm-d | ngram | suffix | vanilla | best | suffix vs van |
|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | 2,450 | — | 2,660 | 3,033 | vanilla | -12% |
| DeepSeek-R1-Distill-Qwen-32B | 4,825 | — | 4,996 | 4,803 | suffix | +4% |
| DeepSeek-R1-Distill-Qwen-7B | 11,192 | — | 11,961 | 8,724 | suffix | +37% |
| Llama-3.1-70B-Instruct | 3,319 | — | 4,864 | 3,091 | suffix | +57% |
| Llama-3.1-8B-Instruct | 13,907 | — | 19,054 | 8,868 | suffix | +115% |
| Qwen2.5-32B-Instruct | 5,150 | — | 4,662 | 3,079 | llm-d | +51% |
| Qwen2.5-72B-Instruct | 2,845 | — | 3,219 | 2,688 | suffix | +20% |
| Qwen2.5-7B-Instruct | 7,505 | 359 | 6,167 | 4,189 | llm-d | +47% |

### swebench
| model | llm-d | ngram | suffix | vanilla | best | suffix vs van |
|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | 2,858 | — | 2,739 | 3,236 | vanilla | -15% |
| DeepSeek-R1-Distill-Qwen-32B | 4,955 | — | 5,241 | 4,409 | suffix | +19% |
| DeepSeek-R1-Distill-Qwen-7B | 12,072 | — | 15,422 | 8,835 | suffix | +75% |
| Llama-3.1-70B-Instruct | 3,436 | — | 6,026 | 2,878 | suffix | +109% |
| Llama-3.1-8B-Instruct | 14,526 | — | 21,353 | 8,348 | suffix | +156% |
| Qwen2.5-32B-Instruct | 3,734 | — | 5,002 | 2,892 | suffix | +73% |
| Qwen2.5-72B-Instruct | 1,971 | — | 2,647 | 2,361 | suffix | +12% |
| Qwen2.5-7B-Instruct | 6,283 | 364 | 5,416 | 4,120 | llm-d | +31% |

### humaneval
| model | llm-d | ngram | suffix | vanilla | best | suffix vs van |
|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | 2,379 | — | 2,788 | 2,852 | vanilla | -2% |
| DeepSeek-R1-Distill-Qwen-32B | 3,288 | — | 3,771 | 3,462 | suffix | +9% |
| DeepSeek-R1-Distill-Qwen-7B | 9,613 | — | 11,459 | 8,159 | suffix | +40% |
| Llama-3.1-70B-Instruct | 3,540 | — | 4,728 | 3,391 | suffix | +39% |
| Llama-3.1-8B-Instruct | 12,665 | — | 15,126 | 9,048 | suffix | +67% |
| Qwen2.5-32B-Instruct | 4,574 | — | 4,859 | 2,571 | suffix | +89% |
| Qwen2.5-72B-Instruct | 2,105 | — | 2,489 | 806 | suffix | +209% |
| Qwen2.5-7B-Instruct | 4,705 | 370 | 5,213 | 3,754 | suffix | +39% |

### mbpp
| model | llm-d | ngram | suffix | vanilla | best | suffix vs van |
|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | 2,305 | — | 2,426 | 2,777 | vanilla | -13% |
| DeepSeek-R1-Distill-Qwen-32B | 4,947 | — | 5,690 | 4,690 | suffix | +21% |
| DeepSeek-R1-Distill-Qwen-7B | 11,985 | — | 12,398 | 8,440 | suffix | +47% |
| Llama-3.1-70B-Instruct | 1,405 | — | 3,266 | 1,773 | suffix | +84% |
| Llama-3.1-8B-Instruct | 13,474 | — | 17,825 | 8,730 | suffix | +104% |
| Qwen2.5-32B-Instruct | 5,554 | — | 5,138 | 2,915 | llm-d | +76% |
| Qwen2.5-72B-Instruct | 3,609 | — | 3,234 | 3,395 | llm-d | -5% |
| Qwen2.5-7B-Instruct | 4,440 | 378 | 5,506 | 3,814 | suffix | +44% |

### wildchat
| model | llm-d | ngram | suffix | vanilla | best | suffix vs van |
|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | 2,731 | — | 2,658 | 3,127 | vanilla | -15% |
| DeepSeek-R1-Distill-Qwen-32B | 5,469 | — | 5,729 | 4,891 | suffix | +17% |
| DeepSeek-R1-Distill-Qwen-7B | 11,029 | — | 11,717 | 8,925 | suffix | +31% |
| Llama-3.1-70B-Instruct | 3,898 | — | 5,261 | 3,172 | suffix | +66% |
| Llama-3.1-8B-Instruct | 14,790 | — | 19,856 | 9,002 | suffix | +121% |
| Qwen2.5-32B-Instruct | 5,242 | — | 4,884 | 3,128 | llm-d | +56% |
| Qwen2.5-72B-Instruct | 2,525 | — | 2,621 | 2,803 | vanilla | -6% |
| Qwen2.5-7B-Instruct | 7,348 | 353 | 6,285 | 4,184 | llm-d | +50% |

### lmsys
| model | llm-d | ngram | suffix | vanilla | best | suffix vs van |
|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | 2,651 | — | 2,848 | 2,992 | vanilla | -5% |
| DeepSeek-R1-Distill-Qwen-32B | 5,042 | — | 5,356 | 4,898 | suffix | +9% |
| DeepSeek-R1-Distill-Qwen-7B | 14,768 | — | 11,360 | 8,811 | llm-d | +29% |
| Llama-3.1-70B-Instruct | 3,897 | — | 3,958 | 3,040 | suffix | +30% |
| Llama-3.1-8B-Instruct | 15,233 | — | 19,862 | 9,074 | suffix | +119% |
| Qwen2.5-32B-Instruct | 4,988 | — | 4,478 | 3,053 | llm-d | +47% |
| Qwen2.5-72B-Instruct | 3,026 | — | 3,429 | 2,807 | suffix | +22% |
| Qwen2.5-7B-Instruct | 7,514 | — | 5,956 | 4,090 | llm-d | +46% |

### mix
| model | llm-d | ngram | suffix | vanilla | best | suffix vs van |
|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | 2,864 | — | 6,127 | 3,164 | suffix | +94% |
| DeepSeek-R1-Distill-Qwen-32B | 5,852 | — | 9,056 | 4,938 | suffix | +83% |
| DeepSeek-R1-Distill-Qwen-7B | 13,000 | — | 24,458 | 9,058 | suffix | +170% |
| Llama-3.1-70B-Instruct | 4,004 | — | 10,400 | 3,129 | suffix | +232% |
| Llama-3.1-8B-Instruct | 15,959 | — | 27,851 | 8,850 | suffix | +215% |
| Qwen2.5-32B-Instruct | 5,236 | — | 6,597 | 3,056 | suffix | +116% |
| Qwen2.5-72B-Instruct | 3,265 | — | 5,268 | 2,735 | suffix | +93% |
| Qwen2.5-7B-Instruct | 7,739 | — | 7,803 | 4,169 | suffix | +87% |

## condition-level oracle (model × condition → best method) — regret 입력
| model | sharegpt | swebench | humaneval | mbpp | wildchat | lmsys | mix |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | vanilla | vanilla | vanilla | vanilla | vanilla | vanilla | suffix |
| DeepSeek-R1-Distill-Qwen-32B | suffix | suffix | suffix | suffix | suffix | suffix | suffix |
| DeepSeek-R1-Distill-Qwen-7B | suffix | suffix | suffix | suffix | suffix | llm-d | suffix |
| Llama-3.1-70B-Instruct | suffix | suffix | suffix | suffix | suffix | suffix | suffix |
| Llama-3.1-8B-Instruct | suffix | suffix | suffix | suffix | suffix | suffix | suffix |
| Qwen2.5-32B-Instruct | llm-d | suffix | suffix | llm-d | llm-d | llm-d | suffix |
| Qwen2.5-72B-Instruct | suffix | suffix | suffix | llm-d | vanilla | suffix | suffix |
| Qwen2.5-7B-Instruct | llm-d | llm-d | suffix | suffix | llm-d | llm-d | suffix |

## 지연 TTFT (mix, p50/p99 ms)
| model | llm-d | ngram | suffix | vanilla |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | 40.3/124.6 | — | 50.1/130.6 | 28.4/128.0 |
| DeepSeek-R1-Distill-Qwen-32B | 29.8/78.9 | — | 37.0/98.8 | 24.1/119.4 |
| DeepSeek-R1-Distill-Qwen-7B | 21.6/56.2 | — | 22.3/56.3 | 16.8/46.4 |
| Llama-3.1-70B-Instruct | 43.8/148.4 | — | 56.3/118.6 | 28.4/114.0 |
| Llama-3.1-8B-Instruct | 22.4/76.3 | — | 24.7/65.4 | 22.8/59.9 |
| Qwen2.5-32B-Instruct | 32.2/93.4 | — | 65.4/96.8 | 29.6/76.2 |
| Qwen2.5-72B-Instruct | 41.8/140.7 | — | 57.4/83.6 | 31.2/70.2 |
| Qwen2.5-7B-Instruct | 25.5/52.4 | — | 69.3/85.6 | 26.2/46.2 |

## 지연 TPOT (mix, p50/p99 ms)
| model | llm-d | ngram | suffix | vanilla |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | 10.2/15.7 | — | 2.1/16.0 | 9.0/9.2 |
| DeepSeek-R1-Distill-Qwen-32B | 6.5/11.0 | — | 1.6/12.6 | 5.9/6.1 |
| DeepSeek-R1-Distill-Qwen-7B | 3.2/5.9 | — | 0.9/7.2 | 3.3/3.3 |
| Llama-3.1-70B-Instruct | 9.5/16.7 | — | 2.5/16.8 | 9.2/9.6 |
| Llama-3.1-8B-Instruct | 1.1/4.0 | — | 1.0/4.4 | 3.5/3.5 |
| Qwen2.5-32B-Instruct | 6.1/11.7 | — | 3.1/22.6 | 9.3/10.4 |
| Qwen2.5-72B-Instruct | 9.9/16.6 | — | 2.6/18.2 | 9.3/10.6 |
| Qwen2.5-7B-Instruct | 4.6/11.6 | — | 3.1/23.2 | 6.9/8.7 |

## accept α (mix, accepted/draft) — spec method
| model | llm-d | ngram | suffix |
|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | 0.393 | — | 0.786 |
| DeepSeek-R1-Distill-Qwen-32B | 0.670 | — | 0.801 |
| DeepSeek-R1-Distill-Qwen-7B | 0.711 | — | 0.876 |
| Llama-3.1-70B-Instruct | 0.825 | — | 0.915 |
| Llama-3.1-8B-Instruct | 0.885 | — | 0.933 |
| Qwen2.5-32B-Instruct | 0.769 | — | 0.857 |
| Qwen2.5-72B-Instruct | 0.677 | — | 0.852 |
| Qwen2.5-7B-Instruct | 0.750 | — | 0.881 |

## util (mix, gpu%/cpu%, gpu_mem GiB)
| model | llm-d | ngram | suffix | vanilla |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | 90.5/5.6 (1202G) | — | 85.0/4.4 (1239G) | 98.3/4.9 (1239G) |
| DeepSeek-R1-Distill-Qwen-32B | 87.3/5.6 (1202G) | — | 79.9/4.4 (1238G) | 97.9/4.9 (1238G) |
| DeepSeek-R1-Distill-Qwen-7B | 81.0/3.9 (600G) | — | 63.8/2.6 (618G) | 91.8/2.8 (618G) |
| Llama-3.1-70B-Instruct | 83.7/5.3 (1202G) | — | 83.4/4.4 (1239G) | 98.5/4.8 (1239G) |
| Llama-3.1-8B-Instruct | 79.5/5.2 (1200G) | — | 62.8/4.4 (1236G) | 94.9/4.6 (1236G) |
| Qwen2.5-32B-Instruct | 79.7/4.9 (1202G) | — | 64.8/4.3 (1238G) | 94.3/4.4 (1238G) |
| Qwen2.5-72B-Instruct | 88.3/5.1 (1203G) | — | 82.5/4.3 (1239G) | 97.6/4.5 (1239G) |
| Qwen2.5-7B-Instruct | 61.0/3.2 (600G) | — | 26.5/2.5 (618G) | 82.5/2.7 (618G) |

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
| DeepSeek-R1-Distill-Llama-70B | suffix | humaneval | 2,788 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__suffix__humaneval.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_suffix_humaneval.json) |
| DeepSeek-R1-Distill-Llama-70B | suffix | lmsys | 2,848 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__suffix__lmsys.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_suffix_lmsys.json) |
| DeepSeek-R1-Distill-Llama-70B | suffix | mbpp | 2,426 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__suffix__mbpp.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_suffix_mbpp.json) |
| DeepSeek-R1-Distill-Llama-70B | suffix | mix | 6,127 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__suffix__mix.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_suffix_mix.json) |
| DeepSeek-R1-Distill-Llama-70B | suffix | sharegpt | 2,660 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__suffix__sharegpt.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_suffix_sharegpt.json) |
| DeepSeek-R1-Distill-Llama-70B | suffix | swebench | 2,739 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__suffix__swebench.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_suffix_swebench.json) |
| DeepSeek-R1-Distill-Llama-70B | suffix | wildchat | 2,658 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__suffix__wildchat.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_suffix_wildchat.json) |
| DeepSeek-R1-Distill-Llama-70B | vanilla | humaneval | 2,852 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__vanilla__humaneval.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_vanilla_humaneval.json) |
| DeepSeek-R1-Distill-Llama-70B | vanilla | lmsys | 2,992 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__vanilla__lmsys.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_vanilla_lmsys.json) |
| DeepSeek-R1-Distill-Llama-70B | vanilla | mbpp | 2,777 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__vanilla__mbpp.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_vanilla_mbpp.json) |
| DeepSeek-R1-Distill-Llama-70B | vanilla | mix | 3,164 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__vanilla__mix.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_vanilla_mix.json) |
| DeepSeek-R1-Distill-Llama-70B | vanilla | sharegpt | 3,033 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__vanilla__sharegpt.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_vanilla_sharegpt.json) |
| DeepSeek-R1-Distill-Llama-70B | vanilla | swebench | 3,236 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__vanilla__swebench.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_vanilla_swebench.json) |
| DeepSeek-R1-Distill-Llama-70B | vanilla | wildchat | 3,127 | [상세](cells/cell_DeepSeek-R1-Distill-Llama-70B__vanilla__wildchat.md) | [json](summ_DeepSeek-R1-Distill-Llama-70B_vanilla_wildchat.json) |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | humaneval | 3,288 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__llm-d__humaneval.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_humaneval.json) |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | lmsys | 5,042 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__llm-d__lmsys.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_lmsys.json) |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | mbpp | 4,947 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__llm-d__mbpp.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_mbpp.json) |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | mix | 5,852 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__llm-d__mix.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_mix.json) |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | sharegpt | 4,825 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__llm-d__sharegpt.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_sharegpt.json) |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | swebench | 4,955 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__llm-d__swebench.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_swebench.json) |
| DeepSeek-R1-Distill-Qwen-32B | llm-d | wildchat | 5,469 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__llm-d__wildchat.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_wildchat.json) |
| DeepSeek-R1-Distill-Qwen-32B | suffix | humaneval | 3,771 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__suffix__humaneval.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_suffix_humaneval.json) |
| DeepSeek-R1-Distill-Qwen-32B | suffix | lmsys | 5,356 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__suffix__lmsys.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_suffix_lmsys.json) |
| DeepSeek-R1-Distill-Qwen-32B | suffix | mbpp | 5,690 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__suffix__mbpp.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_suffix_mbpp.json) |
| DeepSeek-R1-Distill-Qwen-32B | suffix | mix | 9,056 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__suffix__mix.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_suffix_mix.json) |
| DeepSeek-R1-Distill-Qwen-32B | suffix | sharegpt | 4,996 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__suffix__sharegpt.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_suffix_sharegpt.json) |
| DeepSeek-R1-Distill-Qwen-32B | suffix | swebench | 5,241 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__suffix__swebench.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_suffix_swebench.json) |
| DeepSeek-R1-Distill-Qwen-32B | suffix | wildchat | 5,729 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__suffix__wildchat.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_suffix_wildchat.json) |
| DeepSeek-R1-Distill-Qwen-32B | vanilla | humaneval | 3,462 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__vanilla__humaneval.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_humaneval.json) |
| DeepSeek-R1-Distill-Qwen-32B | vanilla | lmsys | 4,898 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__vanilla__lmsys.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_lmsys.json) |
| DeepSeek-R1-Distill-Qwen-32B | vanilla | mbpp | 4,690 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__vanilla__mbpp.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_mbpp.json) |
| DeepSeek-R1-Distill-Qwen-32B | vanilla | mix | 4,938 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__vanilla__mix.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_mix.json) |
| DeepSeek-R1-Distill-Qwen-32B | vanilla | sharegpt | 4,803 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__vanilla__sharegpt.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_sharegpt.json) |
| DeepSeek-R1-Distill-Qwen-32B | vanilla | swebench | 4,409 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__vanilla__swebench.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_swebench.json) |
| DeepSeek-R1-Distill-Qwen-32B | vanilla | wildchat | 4,891 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-32B__vanilla__wildchat.md) | [json](summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_wildchat.json) |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | humaneval | 9,613 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__llm-d__humaneval.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_humaneval.json) |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | lmsys | 14,768 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__llm-d__lmsys.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_lmsys.json) |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | mbpp | 11,985 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__llm-d__mbpp.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_mbpp.json) |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | mix | 13,000 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__llm-d__mix.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_mix.json) |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | sharegpt | 11,192 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__llm-d__sharegpt.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_sharegpt.json) |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | swebench | 12,072 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__llm-d__swebench.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_swebench.json) |
| DeepSeek-R1-Distill-Qwen-7B | llm-d | wildchat | 11,029 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__llm-d__wildchat.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_wildchat.json) |
| DeepSeek-R1-Distill-Qwen-7B | suffix | humaneval | 11,459 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__suffix__humaneval.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_suffix_humaneval.json) |
| DeepSeek-R1-Distill-Qwen-7B | suffix | lmsys | 11,360 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__suffix__lmsys.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_suffix_lmsys.json) |
| DeepSeek-R1-Distill-Qwen-7B | suffix | mbpp | 12,398 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__suffix__mbpp.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_suffix_mbpp.json) |
| DeepSeek-R1-Distill-Qwen-7B | suffix | mix | 24,458 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__suffix__mix.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_suffix_mix.json) |
| DeepSeek-R1-Distill-Qwen-7B | suffix | sharegpt | 11,961 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__suffix__sharegpt.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_suffix_sharegpt.json) |
| DeepSeek-R1-Distill-Qwen-7B | suffix | swebench | 15,422 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__suffix__swebench.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_suffix_swebench.json) |
| DeepSeek-R1-Distill-Qwen-7B | suffix | wildchat | 11,717 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__suffix__wildchat.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_suffix_wildchat.json) |
| DeepSeek-R1-Distill-Qwen-7B | vanilla | humaneval | 8,159 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__vanilla__humaneval.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_humaneval.json) |
| DeepSeek-R1-Distill-Qwen-7B | vanilla | lmsys | 8,811 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__vanilla__lmsys.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_lmsys.json) |
| DeepSeek-R1-Distill-Qwen-7B | vanilla | mbpp | 8,440 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__vanilla__mbpp.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_mbpp.json) |
| DeepSeek-R1-Distill-Qwen-7B | vanilla | mix | 9,058 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__vanilla__mix.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_mix.json) |
| DeepSeek-R1-Distill-Qwen-7B | vanilla | sharegpt | 8,724 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__vanilla__sharegpt.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_sharegpt.json) |
| DeepSeek-R1-Distill-Qwen-7B | vanilla | swebench | 8,835 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__vanilla__swebench.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_swebench.json) |
| DeepSeek-R1-Distill-Qwen-7B | vanilla | wildchat | 8,925 | [상세](cells/cell_DeepSeek-R1-Distill-Qwen-7B__vanilla__wildchat.md) | [json](summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_wildchat.json) |
| Llama-3.1-70B-Instruct | llm-d | humaneval | 3,540 | [상세](cells/cell_Llama-3.1-70B-Instruct__llm-d__humaneval.md) | [json](summ_Llama-3.1-70B-Instruct_llm-d_humaneval.json) |
| Llama-3.1-70B-Instruct | llm-d | lmsys | 3,897 | [상세](cells/cell_Llama-3.1-70B-Instruct__llm-d__lmsys.md) | [json](summ_Llama-3.1-70B-Instruct_llm-d_lmsys.json) |
| Llama-3.1-70B-Instruct | llm-d | mbpp | 1,405 | [상세](cells/cell_Llama-3.1-70B-Instruct__llm-d__mbpp.md) | [json](summ_Llama-3.1-70B-Instruct_llm-d_mbpp.json) |
| Llama-3.1-70B-Instruct | llm-d | mix | 4,004 | [상세](cells/cell_Llama-3.1-70B-Instruct__llm-d__mix.md) | [json](summ_Llama-3.1-70B-Instruct_llm-d_mix.json) |
| Llama-3.1-70B-Instruct | llm-d | sharegpt | 3,319 | [상세](cells/cell_Llama-3.1-70B-Instruct__llm-d__sharegpt.md) | [json](summ_Llama-3.1-70B-Instruct_llm-d_sharegpt.json) |
| Llama-3.1-70B-Instruct | llm-d | swebench | 3,436 | [상세](cells/cell_Llama-3.1-70B-Instruct__llm-d__swebench.md) | [json](summ_Llama-3.1-70B-Instruct_llm-d_swebench.json) |
| Llama-3.1-70B-Instruct | llm-d | wildchat | 3,898 | [상세](cells/cell_Llama-3.1-70B-Instruct__llm-d__wildchat.md) | [json](summ_Llama-3.1-70B-Instruct_llm-d_wildchat.json) |
| Llama-3.1-70B-Instruct | suffix | humaneval | 4,728 | [상세](cells/cell_Llama-3.1-70B-Instruct__suffix__humaneval.md) | [json](summ_Llama-3.1-70B-Instruct_suffix_humaneval.json) |
| Llama-3.1-70B-Instruct | suffix | lmsys | 3,958 | [상세](cells/cell_Llama-3.1-70B-Instruct__suffix__lmsys.md) | [json](summ_Llama-3.1-70B-Instruct_suffix_lmsys.json) |
| Llama-3.1-70B-Instruct | suffix | mbpp | 3,266 | [상세](cells/cell_Llama-3.1-70B-Instruct__suffix__mbpp.md) | [json](summ_Llama-3.1-70B-Instruct_suffix_mbpp.json) |
| Llama-3.1-70B-Instruct | suffix | mix | 10,400 | [상세](cells/cell_Llama-3.1-70B-Instruct__suffix__mix.md) | [json](summ_Llama-3.1-70B-Instruct_suffix_mix.json) |
| Llama-3.1-70B-Instruct | suffix | sharegpt | 4,864 | [상세](cells/cell_Llama-3.1-70B-Instruct__suffix__sharegpt.md) | [json](summ_Llama-3.1-70B-Instruct_suffix_sharegpt.json) |
| Llama-3.1-70B-Instruct | suffix | swebench | 6,026 | [상세](cells/cell_Llama-3.1-70B-Instruct__suffix__swebench.md) | [json](summ_Llama-3.1-70B-Instruct_suffix_swebench.json) |
| Llama-3.1-70B-Instruct | suffix | wildchat | 5,261 | [상세](cells/cell_Llama-3.1-70B-Instruct__suffix__wildchat.md) | [json](summ_Llama-3.1-70B-Instruct_suffix_wildchat.json) |
| Llama-3.1-70B-Instruct | vanilla | humaneval | 3,391 | [상세](cells/cell_Llama-3.1-70B-Instruct__vanilla__humaneval.md) | [json](summ_Llama-3.1-70B-Instruct_vanilla_humaneval.json) |
| Llama-3.1-70B-Instruct | vanilla | lmsys | 3,040 | [상세](cells/cell_Llama-3.1-70B-Instruct__vanilla__lmsys.md) | [json](summ_Llama-3.1-70B-Instruct_vanilla_lmsys.json) |
| Llama-3.1-70B-Instruct | vanilla | mbpp | 1,773 | [상세](cells/cell_Llama-3.1-70B-Instruct__vanilla__mbpp.md) | [json](summ_Llama-3.1-70B-Instruct_vanilla_mbpp.json) |
| Llama-3.1-70B-Instruct | vanilla | mix | 3,129 | [상세](cells/cell_Llama-3.1-70B-Instruct__vanilla__mix.md) | [json](summ_Llama-3.1-70B-Instruct_vanilla_mix.json) |
| Llama-3.1-70B-Instruct | vanilla | sharegpt | 3,091 | [상세](cells/cell_Llama-3.1-70B-Instruct__vanilla__sharegpt.md) | [json](summ_Llama-3.1-70B-Instruct_vanilla_sharegpt.json) |
| Llama-3.1-70B-Instruct | vanilla | swebench | 2,878 | [상세](cells/cell_Llama-3.1-70B-Instruct__vanilla__swebench.md) | [json](summ_Llama-3.1-70B-Instruct_vanilla_swebench.json) |
| Llama-3.1-70B-Instruct | vanilla | wildchat | 3,172 | [상세](cells/cell_Llama-3.1-70B-Instruct__vanilla__wildchat.md) | [json](summ_Llama-3.1-70B-Instruct_vanilla_wildchat.json) |
| Llama-3.1-8B-Instruct | llm-d | humaneval | 12,665 | [상세](cells/cell_Llama-3.1-8B-Instruct__llm-d__humaneval.md) | [json](summ_Llama-3.1-8B-Instruct_llm-d_humaneval.json) |
| Llama-3.1-8B-Instruct | llm-d | lmsys | 15,233 | [상세](cells/cell_Llama-3.1-8B-Instruct__llm-d__lmsys.md) | [json](summ_Llama-3.1-8B-Instruct_llm-d_lmsys.json) |
| Llama-3.1-8B-Instruct | llm-d | mbpp | 13,474 | [상세](cells/cell_Llama-3.1-8B-Instruct__llm-d__mbpp.md) | [json](summ_Llama-3.1-8B-Instruct_llm-d_mbpp.json) |
| Llama-3.1-8B-Instruct | llm-d | mix | 15,959 | [상세](cells/cell_Llama-3.1-8B-Instruct__llm-d__mix.md) | [json](summ_Llama-3.1-8B-Instruct_llm-d_mix.json) |
| Llama-3.1-8B-Instruct | llm-d | sharegpt | 13,907 | [상세](cells/cell_Llama-3.1-8B-Instruct__llm-d__sharegpt.md) | [json](summ_Llama-3.1-8B-Instruct_llm-d_sharegpt.json) |
| Llama-3.1-8B-Instruct | llm-d | swebench | 14,526 | [상세](cells/cell_Llama-3.1-8B-Instruct__llm-d__swebench.md) | [json](summ_Llama-3.1-8B-Instruct_llm-d_swebench.json) |
| Llama-3.1-8B-Instruct | llm-d | wildchat | 14,790 | [상세](cells/cell_Llama-3.1-8B-Instruct__llm-d__wildchat.md) | [json](summ_Llama-3.1-8B-Instruct_llm-d_wildchat.json) |
| Llama-3.1-8B-Instruct | suffix | humaneval | 15,126 | [상세](cells/cell_Llama-3.1-8B-Instruct__suffix__humaneval.md) | [json](summ_Llama-3.1-8B-Instruct_suffix_humaneval.json) |
| Llama-3.1-8B-Instruct | suffix | lmsys | 19,862 | [상세](cells/cell_Llama-3.1-8B-Instruct__suffix__lmsys.md) | [json](summ_Llama-3.1-8B-Instruct_suffix_lmsys.json) |
| Llama-3.1-8B-Instruct | suffix | mbpp | 17,825 | [상세](cells/cell_Llama-3.1-8B-Instruct__suffix__mbpp.md) | [json](summ_Llama-3.1-8B-Instruct_suffix_mbpp.json) |
| Llama-3.1-8B-Instruct | suffix | mix | 27,851 | [상세](cells/cell_Llama-3.1-8B-Instruct__suffix__mix.md) | [json](summ_Llama-3.1-8B-Instruct_suffix_mix.json) |
| Llama-3.1-8B-Instruct | suffix | sharegpt | 19,054 | [상세](cells/cell_Llama-3.1-8B-Instruct__suffix__sharegpt.md) | [json](summ_Llama-3.1-8B-Instruct_suffix_sharegpt.json) |
| Llama-3.1-8B-Instruct | suffix | swebench | 21,353 | [상세](cells/cell_Llama-3.1-8B-Instruct__suffix__swebench.md) | [json](summ_Llama-3.1-8B-Instruct_suffix_swebench.json) |
| Llama-3.1-8B-Instruct | suffix | wildchat | 19,856 | [상세](cells/cell_Llama-3.1-8B-Instruct__suffix__wildchat.md) | [json](summ_Llama-3.1-8B-Instruct_suffix_wildchat.json) |
| Llama-3.1-8B-Instruct | vanilla | humaneval | 9,048 | [상세](cells/cell_Llama-3.1-8B-Instruct__vanilla__humaneval.md) | [json](summ_Llama-3.1-8B-Instruct_vanilla_humaneval.json) |
| Llama-3.1-8B-Instruct | vanilla | lmsys | 9,074 | [상세](cells/cell_Llama-3.1-8B-Instruct__vanilla__lmsys.md) | [json](summ_Llama-3.1-8B-Instruct_vanilla_lmsys.json) |
| Llama-3.1-8B-Instruct | vanilla | mbpp | 8,730 | [상세](cells/cell_Llama-3.1-8B-Instruct__vanilla__mbpp.md) | [json](summ_Llama-3.1-8B-Instruct_vanilla_mbpp.json) |
| Llama-3.1-8B-Instruct | vanilla | mix | 8,850 | [상세](cells/cell_Llama-3.1-8B-Instruct__vanilla__mix.md) | [json](summ_Llama-3.1-8B-Instruct_vanilla_mix.json) |
| Llama-3.1-8B-Instruct | vanilla | sharegpt | 8,868 | [상세](cells/cell_Llama-3.1-8B-Instruct__vanilla__sharegpt.md) | [json](summ_Llama-3.1-8B-Instruct_vanilla_sharegpt.json) |
| Llama-3.1-8B-Instruct | vanilla | swebench | 8,348 | [상세](cells/cell_Llama-3.1-8B-Instruct__vanilla__swebench.md) | [json](summ_Llama-3.1-8B-Instruct_vanilla_swebench.json) |
| Llama-3.1-8B-Instruct | vanilla | wildchat | 9,002 | [상세](cells/cell_Llama-3.1-8B-Instruct__vanilla__wildchat.md) | [json](summ_Llama-3.1-8B-Instruct_vanilla_wildchat.json) |
| Qwen2.5-32B-Instruct | llm-d | humaneval | 4,574 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d__humaneval.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d_humaneval.json) |
| Qwen2.5-32B-Instruct | llm-d | lmsys | 4,988 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d__lmsys.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d_lmsys.json) |
| Qwen2.5-32B-Instruct | llm-d | mbpp | 5,554 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d__mbpp.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d_mbpp.json) |
| Qwen2.5-32B-Instruct | llm-d | mix | 5,236 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d__mix.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d_mix.json) |
| Qwen2.5-32B-Instruct | llm-d | sharegpt | 5,150 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d__sharegpt.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d_sharegpt.json) |
| Qwen2.5-32B-Instruct | llm-d | swebench | 3,734 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d__swebench.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d_swebench.json) |
| Qwen2.5-32B-Instruct | llm-d | wildchat | 5,242 | [상세](cells/cell_Qwen2.5-32B-Instruct__llm-d__wildchat.md) | [json](summ_Qwen2.5-32B-Instruct_llm-d_wildchat.json) |
| Qwen2.5-32B-Instruct | suffix | humaneval | 4,859 | [상세](cells/cell_Qwen2.5-32B-Instruct__suffix__humaneval.md) | [json](summ_Qwen2.5-32B-Instruct_suffix_humaneval.json) |
| Qwen2.5-32B-Instruct | suffix | lmsys | 4,478 | [상세](cells/cell_Qwen2.5-32B-Instruct__suffix__lmsys.md) | [json](summ_Qwen2.5-32B-Instruct_suffix_lmsys.json) |
| Qwen2.5-32B-Instruct | suffix | mbpp | 5,138 | [상세](cells/cell_Qwen2.5-32B-Instruct__suffix__mbpp.md) | [json](summ_Qwen2.5-32B-Instruct_suffix_mbpp.json) |
| Qwen2.5-32B-Instruct | suffix | mix | 6,597 | [상세](cells/cell_Qwen2.5-32B-Instruct__suffix__mix.md) | [json](summ_Qwen2.5-32B-Instruct_suffix_mix.json) |
| Qwen2.5-32B-Instruct | suffix | sharegpt | 4,662 | [상세](cells/cell_Qwen2.5-32B-Instruct__suffix__sharegpt.md) | [json](summ_Qwen2.5-32B-Instruct_suffix_sharegpt.json) |
| Qwen2.5-32B-Instruct | suffix | swebench | 5,002 | [상세](cells/cell_Qwen2.5-32B-Instruct__suffix__swebench.md) | [json](summ_Qwen2.5-32B-Instruct_suffix_swebench.json) |
| Qwen2.5-32B-Instruct | suffix | wildchat | 4,884 | [상세](cells/cell_Qwen2.5-32B-Instruct__suffix__wildchat.md) | [json](summ_Qwen2.5-32B-Instruct_suffix_wildchat.json) |
| Qwen2.5-32B-Instruct | vanilla | humaneval | 2,571 | [상세](cells/cell_Qwen2.5-32B-Instruct__vanilla__humaneval.md) | [json](summ_Qwen2.5-32B-Instruct_vanilla_humaneval.json) |
| Qwen2.5-32B-Instruct | vanilla | lmsys | 3,053 | [상세](cells/cell_Qwen2.5-32B-Instruct__vanilla__lmsys.md) | [json](summ_Qwen2.5-32B-Instruct_vanilla_lmsys.json) |
| Qwen2.5-32B-Instruct | vanilla | mbpp | 2,915 | [상세](cells/cell_Qwen2.5-32B-Instruct__vanilla__mbpp.md) | [json](summ_Qwen2.5-32B-Instruct_vanilla_mbpp.json) |
| Qwen2.5-32B-Instruct | vanilla | mix | 3,056 | [상세](cells/cell_Qwen2.5-32B-Instruct__vanilla__mix.md) | [json](summ_Qwen2.5-32B-Instruct_vanilla_mix.json) |
| Qwen2.5-32B-Instruct | vanilla | sharegpt | 3,079 | [상세](cells/cell_Qwen2.5-32B-Instruct__vanilla__sharegpt.md) | [json](summ_Qwen2.5-32B-Instruct_vanilla_sharegpt.json) |
| Qwen2.5-32B-Instruct | vanilla | swebench | 2,892 | [상세](cells/cell_Qwen2.5-32B-Instruct__vanilla__swebench.md) | [json](summ_Qwen2.5-32B-Instruct_vanilla_swebench.json) |
| Qwen2.5-32B-Instruct | vanilla | wildchat | 3,128 | [상세](cells/cell_Qwen2.5-32B-Instruct__vanilla__wildchat.md) | [json](summ_Qwen2.5-32B-Instruct_vanilla_wildchat.json) |
| Qwen2.5-72B-Instruct | llm-d | humaneval | 2,105 | [상세](cells/cell_Qwen2.5-72B-Instruct__llm-d__humaneval.md) | [json](summ_Qwen2.5-72B-Instruct_llm-d_humaneval.json) |
| Qwen2.5-72B-Instruct | llm-d | lmsys | 3,026 | [상세](cells/cell_Qwen2.5-72B-Instruct__llm-d__lmsys.md) | [json](summ_Qwen2.5-72B-Instruct_llm-d_lmsys.json) |
| Qwen2.5-72B-Instruct | llm-d | mbpp | 3,609 | [상세](cells/cell_Qwen2.5-72B-Instruct__llm-d__mbpp.md) | [json](summ_Qwen2.5-72B-Instruct_llm-d_mbpp.json) |
| Qwen2.5-72B-Instruct | llm-d | mix | 3,265 | [상세](cells/cell_Qwen2.5-72B-Instruct__llm-d__mix.md) | [json](summ_Qwen2.5-72B-Instruct_llm-d_mix.json) |
| Qwen2.5-72B-Instruct | llm-d | sharegpt | 2,845 | [상세](cells/cell_Qwen2.5-72B-Instruct__llm-d__sharegpt.md) | [json](summ_Qwen2.5-72B-Instruct_llm-d_sharegpt.json) |
| Qwen2.5-72B-Instruct | llm-d | swebench | 1,971 | [상세](cells/cell_Qwen2.5-72B-Instruct__llm-d__swebench.md) | [json](summ_Qwen2.5-72B-Instruct_llm-d_swebench.json) |
| Qwen2.5-72B-Instruct | llm-d | wildchat | 2,525 | [상세](cells/cell_Qwen2.5-72B-Instruct__llm-d__wildchat.md) | [json](summ_Qwen2.5-72B-Instruct_llm-d_wildchat.json) |
| Qwen2.5-72B-Instruct | suffix | humaneval | 2,489 | [상세](cells/cell_Qwen2.5-72B-Instruct__suffix__humaneval.md) | [json](summ_Qwen2.5-72B-Instruct_suffix_humaneval.json) |
| Qwen2.5-72B-Instruct | suffix | lmsys | 3,429 | [상세](cells/cell_Qwen2.5-72B-Instruct__suffix__lmsys.md) | [json](summ_Qwen2.5-72B-Instruct_suffix_lmsys.json) |
| Qwen2.5-72B-Instruct | suffix | mbpp | 3,234 | [상세](cells/cell_Qwen2.5-72B-Instruct__suffix__mbpp.md) | [json](summ_Qwen2.5-72B-Instruct_suffix_mbpp.json) |
| Qwen2.5-72B-Instruct | suffix | mix | 5,268 | [상세](cells/cell_Qwen2.5-72B-Instruct__suffix__mix.md) | [json](summ_Qwen2.5-72B-Instruct_suffix_mix.json) |
| Qwen2.5-72B-Instruct | suffix | sharegpt | 3,219 | [상세](cells/cell_Qwen2.5-72B-Instruct__suffix__sharegpt.md) | [json](summ_Qwen2.5-72B-Instruct_suffix_sharegpt.json) |
| Qwen2.5-72B-Instruct | suffix | swebench | 2,647 | [상세](cells/cell_Qwen2.5-72B-Instruct__suffix__swebench.md) | [json](summ_Qwen2.5-72B-Instruct_suffix_swebench.json) |
| Qwen2.5-72B-Instruct | suffix | wildchat | 2,621 | [상세](cells/cell_Qwen2.5-72B-Instruct__suffix__wildchat.md) | [json](summ_Qwen2.5-72B-Instruct_suffix_wildchat.json) |
| Qwen2.5-72B-Instruct | vanilla | humaneval | 806 | [상세](cells/cell_Qwen2.5-72B-Instruct__vanilla__humaneval.md) | [json](summ_Qwen2.5-72B-Instruct_vanilla_humaneval.json) |
| Qwen2.5-72B-Instruct | vanilla | lmsys | 2,807 | [상세](cells/cell_Qwen2.5-72B-Instruct__vanilla__lmsys.md) | [json](summ_Qwen2.5-72B-Instruct_vanilla_lmsys.json) |
| Qwen2.5-72B-Instruct | vanilla | mbpp | 3,395 | [상세](cells/cell_Qwen2.5-72B-Instruct__vanilla__mbpp.md) | [json](summ_Qwen2.5-72B-Instruct_vanilla_mbpp.json) |
| Qwen2.5-72B-Instruct | vanilla | mix | 2,735 | [상세](cells/cell_Qwen2.5-72B-Instruct__vanilla__mix.md) | [json](summ_Qwen2.5-72B-Instruct_vanilla_mix.json) |
| Qwen2.5-72B-Instruct | vanilla | sharegpt | 2,688 | [상세](cells/cell_Qwen2.5-72B-Instruct__vanilla__sharegpt.md) | [json](summ_Qwen2.5-72B-Instruct_vanilla_sharegpt.json) |
| Qwen2.5-72B-Instruct | vanilla | swebench | 2,361 | [상세](cells/cell_Qwen2.5-72B-Instruct__vanilla__swebench.md) | [json](summ_Qwen2.5-72B-Instruct_vanilla_swebench.json) |
| Qwen2.5-72B-Instruct | vanilla | wildchat | 2,803 | [상세](cells/cell_Qwen2.5-72B-Instruct__vanilla__wildchat.md) | [json](summ_Qwen2.5-72B-Instruct_vanilla_wildchat.json) |
| Qwen2.5-7B-Instruct | llm-d | humaneval | 4,705 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d__humaneval.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d_humaneval.json) |
| Qwen2.5-7B-Instruct | llm-d | lmsys | 7,514 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d__lmsys.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d_lmsys.json) |
| Qwen2.5-7B-Instruct | llm-d | mbpp | 4,440 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d__mbpp.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d_mbpp.json) |
| Qwen2.5-7B-Instruct | llm-d | mix | 7,739 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d__mix.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d_mix.json) |
| Qwen2.5-7B-Instruct | llm-d | sharegpt | 7,505 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d__sharegpt.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d_sharegpt.json) |
| Qwen2.5-7B-Instruct | llm-d | swebench | 6,283 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d__swebench.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d_swebench.json) |
| Qwen2.5-7B-Instruct | llm-d | wildchat | 7,348 | [상세](cells/cell_Qwen2.5-7B-Instruct__llm-d__wildchat.md) | [json](summ_Qwen2.5-7B-Instruct_llm-d_wildchat.json) |
| Qwen2.5-7B-Instruct | ngram | humaneval | 370 | [상세](cells/cell_Qwen2.5-7B-Instruct__ngram__humaneval.md) | [json](summ_Qwen2.5-7B-Instruct_ngram_humaneval.json) |
| Qwen2.5-7B-Instruct | ngram | mbpp | 378 | [상세](cells/cell_Qwen2.5-7B-Instruct__ngram__mbpp.md) | [json](summ_Qwen2.5-7B-Instruct_ngram_mbpp.json) |
| Qwen2.5-7B-Instruct | ngram | sharegpt | 359 | [상세](cells/cell_Qwen2.5-7B-Instruct__ngram__sharegpt.md) | [json](summ_Qwen2.5-7B-Instruct_ngram_sharegpt.json) |
| Qwen2.5-7B-Instruct | ngram | swebench | 364 | [상세](cells/cell_Qwen2.5-7B-Instruct__ngram__swebench.md) | [json](summ_Qwen2.5-7B-Instruct_ngram_swebench.json) |
| Qwen2.5-7B-Instruct | ngram | wildchat | 353 | [상세](cells/cell_Qwen2.5-7B-Instruct__ngram__wildchat.md) | [json](summ_Qwen2.5-7B-Instruct_ngram_wildchat.json) |
| Qwen2.5-7B-Instruct | suffix | humaneval | 5,213 | [상세](cells/cell_Qwen2.5-7B-Instruct__suffix__humaneval.md) | [json](summ_Qwen2.5-7B-Instruct_suffix_humaneval.json) |
| Qwen2.5-7B-Instruct | suffix | lmsys | 5,956 | [상세](cells/cell_Qwen2.5-7B-Instruct__suffix__lmsys.md) | [json](summ_Qwen2.5-7B-Instruct_suffix_lmsys.json) |
| Qwen2.5-7B-Instruct | suffix | mbpp | 5,506 | [상세](cells/cell_Qwen2.5-7B-Instruct__suffix__mbpp.md) | [json](summ_Qwen2.5-7B-Instruct_suffix_mbpp.json) |
| Qwen2.5-7B-Instruct | suffix | mix | 7,803 | [상세](cells/cell_Qwen2.5-7B-Instruct__suffix__mix.md) | [json](summ_Qwen2.5-7B-Instruct_suffix_mix.json) |
| Qwen2.5-7B-Instruct | suffix | sharegpt | 6,167 | [상세](cells/cell_Qwen2.5-7B-Instruct__suffix__sharegpt.md) | [json](summ_Qwen2.5-7B-Instruct_suffix_sharegpt.json) |
| Qwen2.5-7B-Instruct | suffix | swebench | 5,416 | [상세](cells/cell_Qwen2.5-7B-Instruct__suffix__swebench.md) | [json](summ_Qwen2.5-7B-Instruct_suffix_swebench.json) |
| Qwen2.5-7B-Instruct | suffix | wildchat | 6,285 | [상세](cells/cell_Qwen2.5-7B-Instruct__suffix__wildchat.md) | [json](summ_Qwen2.5-7B-Instruct_suffix_wildchat.json) |
| Qwen2.5-7B-Instruct | vanilla | humaneval | 3,754 | [상세](cells/cell_Qwen2.5-7B-Instruct__vanilla__humaneval.md) | [json](summ_Qwen2.5-7B-Instruct_vanilla_humaneval.json) |
| Qwen2.5-7B-Instruct | vanilla | lmsys | 4,090 | [상세](cells/cell_Qwen2.5-7B-Instruct__vanilla__lmsys.md) | [json](summ_Qwen2.5-7B-Instruct_vanilla_lmsys.json) |
| Qwen2.5-7B-Instruct | vanilla | mbpp | 3,814 | [상세](cells/cell_Qwen2.5-7B-Instruct__vanilla__mbpp.md) | [json](summ_Qwen2.5-7B-Instruct_vanilla_mbpp.json) |
| Qwen2.5-7B-Instruct | vanilla | mix | 4,169 | [상세](cells/cell_Qwen2.5-7B-Instruct__vanilla__mix.md) | [json](summ_Qwen2.5-7B-Instruct_vanilla_mix.json) |
| Qwen2.5-7B-Instruct | vanilla | sharegpt | 4,189 | [상세](cells/cell_Qwen2.5-7B-Instruct__vanilla__sharegpt.md) | [json](summ_Qwen2.5-7B-Instruct_vanilla_sharegpt.json) |
| Qwen2.5-7B-Instruct | vanilla | swebench | 4,120 | [상세](cells/cell_Qwen2.5-7B-Instruct__vanilla__swebench.md) | [json](summ_Qwen2.5-7B-Instruct_vanilla_swebench.json) |
| Qwen2.5-7B-Instruct | vanilla | wildchat | 4,184 | [상세](cells/cell_Qwen2.5-7B-Instruct__vanilla__wildchat.md) | [json](summ_Qwen2.5-7B-Instruct_vanilla_wildchat.json) |

