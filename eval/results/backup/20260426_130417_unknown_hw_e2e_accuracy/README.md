# TST_003 e2e accuracy run — Qwen/Qwen2.5-7B-Instruct

- timestamp:           2026-04-26 13:05:24
- model:               Qwen/Qwen2.5-7B-Instruct
- tensor_parallel:     1
- gpu_memory_util:     0.85
- max_model_len:       4096
- max_tokens:          32
- logprobs_k:          10
- cpu_bytes_to_use:    1073741824
- num_prompts:         4
- baseline duration:   36.4s
- split_on duration:   30.6s

## Tolerance
- MAX_DIVERGING_TOKENS: 10
- ATOL_LOGPROB:         0.5
- RTOL_PPL:             0.1

## Verdict
- D-i  (token divergence):  PASS (worst = 0 tokens)
- D-ii (logprob / PPL):     PASS (worst max_abs=0.0848, worst ppl_rel=0.0087)
- overall:                  PASS

## Files
- baseline.json     — per-prompt outputs from baseline (default forward)
- split_on.json     — per-prompt outputs from feature-on (TSK_002 Phase 4c)
- comparison.json   — full per-prompt + aggregate D-i / D-ii verdict
