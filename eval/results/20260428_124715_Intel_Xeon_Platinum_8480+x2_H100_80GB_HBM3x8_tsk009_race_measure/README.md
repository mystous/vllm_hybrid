# 20260428_124715_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_tsk009_race_measure

- timestamp: 20260428_124715
- hw_tag:    Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    dea755ffbd28632a1c8fc07bc34b3de8ae09f3bc
- python:    /workspace/vllm_dev_prj/bin/python
- vllm:      0.1.dev15917+g0a6396b45
- race_mode: measure

## scope
- baseline: envs/vllm_original_long_ctx.env (split off)
- split_on: envs/ide006_cold_kv_split_on_long_ctx.env (split on, race=measure)
- max-prompts: 20
- max-tokens:  16
- logprobs:    1

## exit code
- e2e RC: 1

## race histogram
- race log lines: 2400
- cpu wins:       0
0
- gpu wins:       2400

## comparison.json (jq)
```
{
  "verdict_d_i": false,
  "verdict_d_ii": false,
  "verdict_overall": false,
  "worst_diverging_tokens": 14,
  "worst_max_abs_logprob": 3.6273874044443346,
  "worst_ppl_relative_diff": 0.2385873912511068,
  "suspicious_no_cold_path": false,
  "tolerances": {
    "max_diverging_tokens": 10,
    "atol_logprob": 0.5,
    "rtol_ppl": 0.1
  },
  "per_prompt": [
    {
      "prompt_index": 0,
      "len_baseline": 16,
      "len_split_on": 16,
      "n_diverging_tokens": 0,
      "max_abs_logprob": 1.656158566476961,
      "ppl_relative_diff": 0.11018722791415599,
      "d_i_pass": true,
      "d_ii_pass": false
    },
    {
      "prompt_index": 1,
      "len_baseline": 16,
      "len_split_on": 16,
      "n_diverging_tokens": 5,
      "max_abs_logprob": 2.4727292060854325,
      "ppl_relative_diff": 0.1498175238120875,
      "d_i_pass": true,
      "d_ii_pass": false
    },
    {
      "prompt_index": 2,
      "len_baseline": 16,
      "len_split_on": 16,
      "n_diverging_tokens": 0,
      "max_abs_logprob": 2.092904925402763,
```
