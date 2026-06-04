# Qwen2.5-7B-Instruct × llm-d-c8 × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2829.8 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 207.9 |
| total_completion_tokens | 588321 |
| TTFT p50/p99 ms | 20.1/41.9 |
| TPOT p50/p99 ms | 3.6/5.8 |
| accept α (acc/draft) | 0.6321 (282335.0/446695.0) |
| GPU util / mem MiB | 70.7 / 614764 |
| CPU util | 3.1 |
| reqtps_avg | 349.9 |
| concurrency / max_tokens / stream | 8 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 32 | 116.48 | 14.21 | 0.6 |
| p50 | 658 | 2556.01 | 20.08 | 3.61 |
| p99 | 8192 | 33082.77 | 41.9 | 5.78 |
| max | 8192 | 44715.08 | 56.08 | 8.16 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d-c8_swebench.json`](../summ_Qwen2.5-7B-Instruct_llm-d-c8_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d-c8" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
