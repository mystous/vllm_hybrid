# TSK_042 10-model LHC Path 1 Unified Validation — Final

Generated: aggregate_all.py (single sequential agent)

## Δ% Matrix (LHC Path 1 vs vanilla — paired by sweep)

Cell format: `Δ%±CI95 (n)`. `—` = no data.


| Model | sharegpt | swebench | humaneval | mbpp | wildchat | lmsys | mix |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | +0.5±3.3 (3) | +1.0±0.5 (3) | -0.3±1.1 (3) | +0.5±2.0 (3) | +0.0±2.0 (3) | -0.3±0.7 (3) | +0.7±1.7 (3) |
| Qwen2.5-7B-Instruct | +0.1±3.8 (3) | -0.1±7.5 (3) | +0.2±6.8 (3) | -2.6±1.5 (3) | -1.4±6.3 (3) | -0.2±2.6 (3) | -0.4±3.2 (3) |
| DeepSeek-R1-Distill-Qwen-7B | -0.6±1.3 (3) | -1.0±1.7 (3) | +0.6±2.4 (3) | +0.1±0.6 (3) | -0.6±1.7 (3) | -0.5±2.0 (3) | -0.5±0.5 (3) |
| Llama-3.1-70B-Instruct | -0.4±5.8 (2) | -1.1±64.3 (2) | -2.4±38.0 (2) | -4.0±39.2 (2) | +0.1±1.3 (2) | -0.7±5.7 (2) | +0.0±6.9 (2) |
| Qwen2.5-32B-Instruct | -0.2±1.7 (3) | -4.4±4.9 (3) | -0.4±5.8 (3) | -2.7±13.6 (3) | +0.6±1.7 (3) | -0.2±3.2 (3) | -0.5±3.1 (3) |
| DeepSeek-R1-Distill-Qwen-32B | — | — | — | — | — | — | — |
| DeepSeek-R1-Distill-Llama-70B | — | — | — | — | — | — | — |
| Qwen2.5-72B-Instruct | — | — | — | — | — | — | — |
| Llama-3.1-405B-Instruct-FP8 | — | — | — | — | — | — | — |
| DeepSeek-R1 | — | — | — | — | — | — | — |

## Positive Cells (Δ% ≥ +5%, CI95 LB > 0)

- (none yet)


## Notes

- All 10 models run in a single sequential agent (no concurrent agent contention).

- Sweep schedule per user spec: 8B/7B=3, 32B=3, 70B/72B=2, 405B=1, R1=1 (mix only).

- Same harness as TSK_042 (concurrency=32, max_tokens=8192, streaming).

- LHC Path 1: VLLM_LHC_AMX_C3_PREFIX=1 + libamx_c3.so.

