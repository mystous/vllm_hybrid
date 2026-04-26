# TST_003 e2e accuracy run

- timestamp:           2026-04-26 13:57:54
- baseline env:        eval/envs/vllm_original.env
- split_on env:        eval/envs/ide006_cold_kv_split_on.env
- model:               Qwen/Qwen2.5-7B-Instruct
- tensor_parallel:     1
- max_model_len:       4096
- max_tokens:          None
- logprobs_k:          10
- num_prompts:         4
- baseline duration:   42.1s
- split_on duration:   36.5s

## Tolerance
- MAX_DIVERGING_TOKENS: 10
- ATOL_LOGPROB:         0.5
- RTOL_PPL:             0.1

## Verdict
- D-i  (token divergence):  FAIL (worst = 255 tokens)
- D-ii (logprob / PPL):     PASS (worst max_abs=0.1262, worst ppl_rel=0.0013)
- overall:                  FAIL

## Files
- baseline.json     — per-prompt outputs from baseline (default forward)
- split_on.json     — per-prompt outputs from feature-on (TSK_002 Phase 4c)
- comparison.json   — full per-prompt + aggregate D-i / D-ii verdict
