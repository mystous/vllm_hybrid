[table] 60 rows (60 ok) → /workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/oracle_smoke/oracle_table.parquet

## method spread (prompt-level (max−min)/max) — kill-gate 1차
| model | corpus | n | mean spread | <5% 비율 |
|---|---|---:|---:|---:|
| Qwen2.5-32B-Instruct | sharegpt | 20 | 16.8% | 0% |

→ overall mean spread 16.8% / <5% 0% → **OK (method 우열 존재)**
  (method 차이가 충분해야 분류기/regret 이 의미. <5% 면 AGSD 가치 약함)
