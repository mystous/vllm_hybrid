# TSK_042 10-model LHC Path 1 Validation — Final

Generated: aggregate_all.py

## Δ% Matrix (LHC Path 1 vs vanilla — paired by sweep)

Cell format: `Δ%±CI95 (n)`. `—` = no data.


| Model | sharegpt | swebench | humaneval | mbpp | wildchat | lmsys | mix |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | — | — | — | — | — | — | — |
| Qwen2.5-7B-Instruct | — | — | — | — | — | — | — |
| DeepSeek-R1-Distill-Qwen-7B | — | — | — | — | — | — | — |
| Qwen2.5-32B-Instruct | — | — | — | — | — | — | — |
| DeepSeek-R1-Distill-Qwen-32B | — | — | — | — | — | — | — |
| Llama-3.1-70B-Instruct | — | — | — | — | — | — | — |
| DeepSeek-R1-Distill-Llama-70B | — | — | — | — | — | — | — |
| Qwen2.5-72B-Instruct | — | — | — | — | — | — | — |
| Llama-3.1-405B-Instruct-FP8 | — | — | — | — | — | — | — |
| DeepSeek-R1 | — | — | — | — | — | — | — |

## Positive Cells (Δ% ≥ +5%, CI95 LB > 0)

- (none yet)


## Notes

- Llama-3.1-8B-Instruct is run by sister agent (#72); results consumed from `lhc_phase4/tsk042_validation/`.

- 70B+ models use 1 sweep (no CI). 405B / 671B sanity-only (1 corpus × 1 sweep).

- Same harness as TSK_042 (concurrency=32, max_tokens=8192, streaming).

