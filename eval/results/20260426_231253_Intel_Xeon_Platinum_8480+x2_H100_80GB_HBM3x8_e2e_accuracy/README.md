# TST_003 e2e accuracy run

- timestamp:           2026-04-26 23:17:42
- baseline env:        eval/envs/vllm_original_long_ctx.env
- split_on env:        eval/envs/ide006_cold_kv_split_on_long_ctx.env
- model:               meta-llama/Llama-3.3-70B-Instruct
- tensor_parallel:     8
- max_model_len:       16384
- max_tokens:          64
- logprobs_k:          10
- num_prompts:         8
- baseline duration:   109.2s
- split_on duration:   169.1s

## Tolerance
- MAX_DIVERGING_TOKENS: 10
- ATOL_LOGPROB:         0.5
- RTOL_PPL:             0.1

## Verdict
- D-i  (token divergence):  PASS (worst = 0 tokens)
- D-ii (logprob / PPL):     PASS (worst max_abs=0.0512, worst ppl_rel=0.0014)
- overall:                  PASS

## Files
- baseline.json     — per-prompt outputs from baseline (default forward)
- split_on.json     — per-prompt outputs from feature-on (TSK_002 Phase 4c)
- comparison.json   — full per-prompt + aggregate D-i / D-ii verdict
