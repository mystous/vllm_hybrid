# 20260428_041131_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_quick_tst003

- timestamp: 20260428_041131
- hw_tag:    Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    bf525c24d9ad3fb85a721654245356957de14669
- python:    /workspace/vllm_dev_prj/bin/python
- vllm:      0.1.dev15917+g0a6396b45

## scope
- baseline: vllm_original_long_ctx.env (Llama-3.3-70B + TP=8, split off)
- split_on: ide006_cold_kv_split_on_long_ctx.env
- max-prompts: 30
- max-tokens:  16
- logprobs:    1  (D-ii 측정 필수)

## 통과 기준
- e2e RC = 0
- comparison.json 의 D-ii (logprob max abs / PPL rel) 가 tolerance 내
- prompt2 의 발산 여부 정량 확인

## exit code
- e2e RC: 1

## comparison.json (jq)
```
{
  "verdict_d_i": false,
  "verdict_d_ii": false,
  "verdict_overall": false,
  "worst_diverging_tokens": 14,
  "worst_max_abs_logprob": 3.429074535961263,
  "worst_ppl_relative_diff": 0.2298565973934289,
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
      "max_abs_logprob": 2.1457271941471845e-06,
      "ppl_relative_diff": 1.0430923706829207e-07,
      "d_i_pass": true,
      "d_ii_pass": true
    },
    {
      "prompt_index": 1,
      "len_baseline": 16,
      "len_split_on": 16,
      "n_diverging_tokens": 0,
      "max_abs_logprob": 9.77407762547955e-06,
      "ppl_relative_diff": 4.991229577976741e-07,
      "d_i_pass": true,
      "d_ii_pass": true
    },
    {
      "prompt_index": 2,
      "len_baseline": 16,
      "len_split_on": 16,
      "n_diverging_tokens": 0,
      "max_abs_logprob": 3.5237520933151245e-05,
      "ppl_relative_diff": 6.303182830510092e-07,
      "d_i_pass": true,
      "d_ii_pass": true
    },
    {
      "prompt_index": 3,
      "len_baseline": 16,
      "len_split_on": 16,
      "n_diverging_tokens": 0,
      "max_abs_logprob": 0.0002183942124247551,
      "ppl_relative_diff": 1.3820303901344663e-05,
      "d_i_pass": true,
      "d_ii_pass": true
    },
    {
      "prompt_index": 4,
      "len_baseline": 16,
      "len_split_on": 16,
      "n_diverging_tokens": 1,
      "max_abs_logprob": 3.113578796400361,
```
