# Qwen2.5-32B-Instruct × llm-d × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4574.2 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 73.7 |
| total_completion_tokens | 337325 |
| TTFT p50/p99 ms | 32.8/81.5 |
| TPOT p50/p99 ms | 5.3/8.1 |
| accept α (acc/draft) | 0.783 (166663.0/212842.0) |
| GPU util / mem MiB | 77.1 / 1230928 |
| CPU util | 4.9 |
| reqtps_avg | 314.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=328)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 14 | 105.05 | 22.22 | 1.1 |
| p50 | 308 | 2138.52 | 32.36 | 6.67 |
| p99 | 8192 | 66377.91 | 81.4 | 12.82 |
| max | 8192 | 66383.85 | 82.82 | 14.24 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_llm-d_humaneval.json`](../summ_Qwen2.5-32B-Instruct_llm-d_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="llm-d" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
