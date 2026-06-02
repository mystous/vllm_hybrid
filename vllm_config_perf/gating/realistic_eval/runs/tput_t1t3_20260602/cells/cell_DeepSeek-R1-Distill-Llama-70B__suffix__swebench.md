# DeepSeek-R1-Distill-Llama-70B × suffix × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2739.1 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 356.5 |
| total_completion_tokens | 976438 |
| TTFT p50/p99 ms | 52.9/340.8 |
| TPOT p50/p99 ms | 12.9/16.4 |
| accept α (acc/draft) | 0.3226 (483286.0/1497888.0) |
| GPU util / mem MiB | 86.1 / 1268784 |
| CPU util | 4.4 |
| reqtps_avg | 93.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 741 | 9590.85 | 37.12 | 1.67 |
| p50 | 2430 | 30043.26 | 52.89 | 12.94 |
| p99 | 8192 | 81353.2 | 340.79 | 16.39 |
| max | 8192 | 102148.61 | 342.67 | 17.03 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_suffix_swebench.json`](../summ_DeepSeek-R1-Distill-Llama-70B_suffix_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="suffix" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
