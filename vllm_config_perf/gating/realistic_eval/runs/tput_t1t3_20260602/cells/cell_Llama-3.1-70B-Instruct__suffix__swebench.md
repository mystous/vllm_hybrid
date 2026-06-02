# Llama-3.1-70B-Instruct × suffix × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 6025.8 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 105.6 |
| total_completion_tokens | 636051 |
| TTFT p50/p99 ms | 60.1/294.8 |
| TPOT p50/p99 ms | 14.0/20.4 |
| accept α (acc/draft) | 0.814 (528673.0/649505.0) |
| GPU util / mem MiB | 85.3 / 1268784 |
| CPU util | 4.3 |
| reqtps_avg | 149.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 87.24 | 35.68 | 1.69 |
| p50 | 323 | 4740.46 | 60.08 | 13.98 |
| p99 | 8192 | 60645.57 | 294.82 | 20.44 |
| max | 8192 | 75625.8 | 326.84 | 27.03 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_suffix_swebench.json`](../summ_Llama-3.1-70B-Instruct_suffix_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="suffix" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
