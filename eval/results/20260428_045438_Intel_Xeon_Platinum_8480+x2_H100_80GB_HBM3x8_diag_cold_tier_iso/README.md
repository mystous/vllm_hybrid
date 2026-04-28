# 20260428_045438_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_diag_cold_tier_iso

- timestamp: 20260428_045438
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    ec6cd7de226557c0ca3cbc36d70dabcf26b046b3
- python:    /workspace/vllm_dev_prj/bin/python
- vllm:      0.1.dev15917+g0a6396b45

## scope
- baseline: vllm_original_long_ctx.env       (cold-tier 비활성)
- split_on: ide006_cold_tier_only_long_ctx.env  (cold-tier 활성, IDE_006 비활성)
- max-prompts: 30
- max-tokens:  16
- logprobs:    1

## 진단 결정 분기
- worst_lp ≈ 3.43 → cold-tier 자체가 발산 source → TSK_012 진행
- worst_lp ≪ 1   → cold-tier 는 OK → TSK_012 design 재검토

## exit code
- e2e RC: 0

## comparison.json (jq)
```
{
  "verdict_d_i": true,
  "verdict_d_ii": true,
  "verdict_overall": true,
  "worst_diverging_tokens": 0,
  "worst_max_abs_logprob": 1.1920883480343036e-07,
  "worst_ppl_relative_diff": 7.4505521074038186e-09,
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
      "max_abs_logprob": 0.0,
      "ppl_relative_diff": 0.0,
      "d_i_pass": true,
      "d_ii_pass": true
    },
    {
      "prompt_index": 1,
      "len_baseline": 16,
      "len_split_on": 16,
      "n_diverging_tokens": 0,
      "max_abs_logprob": 0.0,
```
