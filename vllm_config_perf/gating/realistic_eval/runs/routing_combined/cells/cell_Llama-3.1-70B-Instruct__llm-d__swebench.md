# Llama-3.1-70B-Instruct × llm-d × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3436.2 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 190.6 |
| total_completion_tokens | 654905 |
| TTFT p50/p99 ms | 53.7/298.2 |
| TPOT p50/p99 ms | 10.8/18.7 |
| accept α (acc/draft) | 0.8312 (298804.0/359497.0) |
| GPU util / mem MiB | 79.3 / 1231056 |
| CPU util | 5.0 |
| reqtps_avg | 136.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 64.1 | 31.19 | 1.42 |
| p50 | 326 | 4026.68 | 53.71 | 10.8 |
| p99 | 8192 | 88704.41 | 298.17 | 18.67 |
| max | 8192 | 88738.54 | 298.65 | 22.39 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_llm-d_swebench.json`](../summ_Llama-3.1-70B-Instruct_llm-d_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="llm-d" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
